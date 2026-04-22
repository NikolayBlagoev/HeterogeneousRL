from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sys import argv
import torch.distributed as dist
import torch
import time
from transformers import GenerationConfig
import math
from coms import setup_comms, gather, pad_tensor, unpad_tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_rollouts

from grpo import sequences_log_probs, Experience,grpo_loss, grpo_train_loop, advantage_compute, GRPOConfig
from datasets import load_dataset
from interp_config import process_config
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=128, lora_alpha=256, lora_dropout=0.0)
seed = 42
ds_seed = 42

device_index = int(argv[1])
method = argv[2] # nis, vis or tis, with f-* for filtering (e.g. f-vis is filtered vis)
scenario = argv[3]
out_dir = argv[4]



system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags with nothing outside said tags, i.e., <think> reasoning process here </think><answer> answer here </answer>\n
"""

scenario = process_config(scenario,ds_seed,device_index=device_index)

lr = 1e-6

kl_weight = 0
comm_style = scenario["comm_style"]
group_size = scenario["group_size"]
world_size = scenario["world_size"]
batch_size = scenario["batch_size"]

if comm_style != "alone" and world_size > 1:
    setup_comms(device_index,world_size)


if comm_style == "horizontal":
    group_size = group_size // world_size
grpo_config = GRPOConfig(num_generations=group_size)

model_name = scenario["model_name"]

reward_func = scenario["reward_func"]

device = f"cuda:{device_index}"

if "PEFT" in model_name:
    model = AutoModelForCausalLM.from_pretrained(model_name[5:], device_map=device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name[5:])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    lr = 1e-4
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

model.generation_config.max_new_tokens = None
    

train_dataset = scenario["dl_benign"]
train_kwargs = scenario["train_kwargs"]
val_ds = scenario["val_loader"]
val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)



prompt_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=False
)
data_interp = scenario["data_interp"]
replay_buffer = []
global_counter = 0
print(tokenizer.pad_token_id)
for k, prompt_batch in enumerate(prompt_loader):
    if k == 51:
        break
    rollout_returns = []
    rollout_indv = []
    replay_buffer.clear()
    questions, solutions, answers = data_interp(prompt_batch)
    generation_times = 0
    comm_times = 0
    # if k <= 10:
    #     comm_style = "horizontal"
    #     group_size = scenario["group_size"] // world_size
    # else:
    #     comm_style = "vertical"
    #     group_size = scenario["group_size"]
    with (torch.no_grad()):
        idx = -1
        for q, s, a in zip(questions, solutions, answers):
            idx += 1
            if comm_style == "vertical" and idx % world_size != device_index:
                continue
            gen_start = time.time()
            for _ in range(1):
                sequence_ids, completion_mask, start_seq, completions = generate_rollouts(model=model, tokenizer=tokenizer, question=q, sys_prompt=system_prompt, num_rollouts=group_size)
                returns, _, _ = reward_func(completions,a)
                if torch.count_nonzero(returns).item() != 0:
                    break


            if len(replay_buffer) == 0:
                print(f"{completions[0]}")
                # print(f"{completions[1]}")
            # print(prom)
            

        
            attention_mask = sequence_ids != tokenizer.pad_token_id
            
            seq_log_probs, _ = sequences_log_probs(
                        model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                        start_seq=start_seq, batch_size=3
            )

            rollout_indv.append(returns)
            returns = returns.to(device)
            global_start_seq = torch.tensor([start_seq],device = seq_log_probs.device)
            
            if comm_style != "alone" and world_size > 1:
                seq_log_probs = gather(seq_log_probs,device_index,world_size)
                attention_mask = gather(attention_mask,device_index,world_size)
                returns = gather(returns,device_index,world_size)
                global_start_seq = gather(global_start_seq,device_index,world_size)
                sequence_ids = gather(sequence_ids,device_index,world_size)
            

            if comm_style == "alone" or world_size == 1:
                sequence_ids, max_l = unpad_tensor(sequence_ids,tokenizer.pad_token_id)
                attention_mask = attention_mask[:,:max_l]
                
                rollout_returns.append(returns.to("cpu"))
                advantages = advantage_compute(returns)

                completion_mask = attention_mask[:,start_seq:]
                seq_log_probs = seq_log_probs[:,:max_l - start_seq]
                ref_log_probs = None
                exp = Experience(sequence_ids=sequence_ids,advantages=advantages,attention_mask=attention_mask,ref_log_probs=ref_log_probs,
                                action_mask=completion_mask,start_ids=0, logits_to_keep=start_seq,gen_log_probs=seq_log_probs)
                replay_buffer.append(exp.to("cpu"))
            elif comm_style == "vertical":
                for loc_rank in range(world_size):
                    loc_sequence_ids = sequence_ids[loc_rank]
                    loc_attention_mask = attention_mask[loc_rank]
                    
                    loc_sequence_ids, max_l = unpad_tensor(loc_sequence_ids,tokenizer.pad_token_id)
                    loc_attention_mask = loc_attention_mask[:,:max_l]
                    start_seq = global_start_seq[loc_rank].item()
                    
                    loc_returns = returns[loc_rank]
                    rollout_returns.append(loc_returns.to("cpu"))
                    advantages = advantage_compute(loc_returns)
                    loc_seq_log_probs = seq_log_probs[loc_rank]

                    completion_mask = loc_attention_mask[:,start_seq:]
                    loc_seq_log_probs = loc_seq_log_probs[:,:max_l - start_seq]
                    ref_log_probs = None
                    exp = Experience(sequence_ids=loc_sequence_ids,advantages=advantages,attention_mask=loc_attention_mask,ref_log_probs=ref_log_probs,
                                action_mask=completion_mask,start_ids=0, logits_to_keep=start_seq,gen_log_probs=loc_seq_log_probs)
                    replay_buffer.append(exp.to("cpu"))
            elif comm_style == "horizontal":   
                sequence_ids = torch.cat(sequence_ids, dim = 0)
                attention_mask = torch.cat(attention_mask, dim = 0)
                
                sequence_ids, max_l = unpad_tensor(sequence_ids,tokenizer.pad_token_id)
                attention_mask = attention_mask[:,:max_l]
                
                
                returns = torch.cat(returns, dim = 0)
                rollout_returns.append(returns.to("cpu"))
                advantages = advantage_compute(returns)
                seq_log_probs = torch.cat(seq_log_probs, dim = 0)

                completion_mask = attention_mask[:,start_seq:]
                seq_log_probs = seq_log_probs[:,:max_l - start_seq]
                ref_log_probs = None
                exp = Experience(sequence_ids=sequence_ids,advantages=advantages,attention_mask=attention_mask,ref_log_probs=ref_log_probs,
                            action_mask=completion_mask,start_ids=0, logits_to_keep=start_seq,gen_log_probs=seq_log_probs)
                replay_buffer.append(exp.to("cpu"))
            print(len(replay_buffer))

    print(f"generation time of step {k}: {generation_times:.4f}")
    print(f"communication time of step {k}: {comm_times:.4f}")
    torch.cuda.empty_cache()

    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"idividual returns of step {k}: {episode_reward:.4f}")
    if k % 5 == 0:
        model.eval()
        val_dl = iter(val_loader)
        rewards = 0
        for _ in range(32):
            val_batch = next(val_dl)
            q,s,a = data_interp(val_batch)
            chat_prompts = []
            for question in q:
                chat_messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
                
                chat_prompts.append(tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                ))
            model_inputs = tokenizer(
                chat_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_attention_mask=True,
                add_special_tokens=False
            ).to(model.device)
            generation_config = GenerationConfig(
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature = 1.0,
                top_p = 1.0,
                top_k = 50,
            )

            start_seq = model_inputs["input_ids"].shape[1]
            model_inputs["input_ids"] = model_inputs["input_ids"]
            model_inputs["attention_mask"] = model_inputs["attention_mask"]
            with (torch.no_grad()):
                completion_ids = model.generate(**model_inputs,generation_config = generation_config)
                completion_ids = completion_ids[:, start_seq :]
                completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                rewards += reward_func(completions,a)[0].mean().item() / 32
        print(f"Validation of step {k}: {rewards:.4f}")


    torch.cuda.empty_cache()
    update_start = time.time()
    if True:
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss_hist, kl_hist, entropy_hist = grpo_train_loop(model, optimizer, replay_buffer, grpo_config,method = method, step = k, **train_kwargs)
        print(f"Loss at step {k}", loss_hist[0])
        print(f"KL at step {k}", kl_hist[0])
        print(f"ENTROPY at step {k}", entropy_hist[0])
    print(f"update time of step {k}: {time.time() - update_start}")
    # post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)
model.save_pretrained(out_dir)
print("MODEL SAVED")
dist.barrier()