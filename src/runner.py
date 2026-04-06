from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sys import argv
import torch.distributed as dist
import torch
import time
import math
from coms import setup_comms, gather, pad_tensor, unpad_tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_rollouts

from grpo import sequences_log_probs, Experience,grpo_loss, grpo_train_loop, advantage_compute, GRPOConfig
from datasets import load_dataset
from interp_config import process_config

seed = 42
ds_seed = 42

device_index = int(argv[1])
scenario = argv[2]
out_dir = argv[3]


system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags with nothing outside said tags, i.e., <think> reasoning process here </think><answer> answer here </answer>\n
"""

scenario = process_config(scenario,ds_seed,device_index=device_index)

lr = 5e-6
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
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device,attn_implementation="kernels-community/flash-attn3", dtype=torch.bfloat16)


optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = scenario["dl_benign"]
train_kwargs = scenario["train_kwargs"]
val_ds = scenario["val_loader"]
val_loader = DataLoader(
    val_ds,
    batch_size=20,
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

    with (torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)):
        idx = -1
        for q, s, a in zip(questions, solutions, answers):
            idx += 1
            if comm_style == "vertical" and idx % world_size != device_index:
                continue

            gen_start = time.time()
            for _ in range(1):
                prompt_ids, prompt_mask, completion_ids, completion_mask = generate_rollouts(model=model, tokenizer=tokenizer, question=q, sys_prompt=system_prompt, num_rollouts=group_size)
                completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                returns, _, _ = reward_func(completions,a)
                if torch.count_nonzero(returns).item() != 0:
                    break


            if len(replay_buffer) == 0:
                print(f"{completions[0]}")
                # print(f"{completions[1]}")
            # print(prom)
            
            if comm_style == "horizontal":
                completion_ids = pad_tensor(completion_ids,tokenizer.pad_token_id,1024)
                completion_mask = pad_tensor(completion_mask,0,1024)
            sequence_ids = torch.cat((prompt_ids,completion_ids),dim=1)
            attention_mask = torch.cat((prompt_mask, completion_mask), dim = 1)
            
            seq_log_probs, _ = sequences_log_probs(
                        model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                        logits_to_keep=completion_ids.shape[1]
            )
            gen_end = time.time()
            generation_times += gen_end - gen_start
            rollout_indv.append(returns)
            returns = returns.to(device)
            comm_start = time.time()
            
            if comm_style != "alone" and world_size > 1:
                seq_log_probs = gather(seq_log_probs,device_index,world_size)
                attention_mask = gather(attention_mask,device_index,world_size)
                completion_ids = gather(completion_ids,device_index,world_size)
                returns = gather(returns,device_index,world_size)
                completion_mask = gather(completion_mask,device_index,world_size)
                sequence_ids = gather(sequence_ids,device_index,world_size)
            
            comm_end = time.time()
            comm_times += comm_end - comm_start
            if comm_style == "alone" or world_size == 1:
                advantages = advantage_compute(returns)
                rollout_returns.append(returns.to("cpu"))
                exp = Experience(sequence_ids=sequence_ids,advantages=advantages,attention_mask=attention_mask,action_mask=completion_mask,start_ids=0, logits_to_keep=completion_ids.shape[1],gen_log_probs=seq_log_probs)
                replay_buffer.append(exp.to("cpu"))
            elif comm_style == "vertical":
                for loc_rank in range(world_size):
                    # print()
                    advantages = advantage_compute(returns[loc_rank])
                    rollout_returns.append(returns[loc_rank].to("cpu"))
                    exp = Experience(sequence_ids=sequence_ids[loc_rank],advantages=advantages,attention_mask=attention_mask[loc_rank],action_mask=completion_mask[loc_rank],start_ids=0, logits_to_keep=completion_ids[loc_rank].shape[1],gen_log_probs=seq_log_probs[loc_rank])
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
                completion_ids = torch.cat(completion_ids, dim = 0)
                completion_mask = torch.cat(completion_mask, dim = 0)
                completion_ids, max_l = unpad_tensor(completion_ids,tokenizer.pad_token_id)
                completion_mask = completion_mask[:,:max_l]
                seq_log_probs = seq_log_probs[:,:max_l]
                
                exp = Experience(sequence_ids=sequence_ids,advantages=advantages,attention_mask=attention_mask,action_mask=completion_mask,start_ids=0, logits_to_keep=completion_ids.shape[1],gen_log_probs=seq_log_probs)
                replay_buffer.append(exp.to("cpu"))
            print(len(replay_buffer))

    print(f"generation time of step {k}: {generation_times:.4f}")
    print(f"communication time of step {k}: {comm_times:.4f}")
    torch.cuda.empty_cache()

    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"idividual returns of step {k}: {episode_reward:.4f}")



    torch.cuda.empty_cache()
    update_start = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss_hist, kl_hist, entropy_hist = grpo_train_loop(model, optimizer, replay_buffer, grpo_config,**train_kwargs)
        print(f"Loss at step {k}", loss_hist[0])
        print(f"KL at step {k}", kl_hist[0])
        print(f"ENTROPY at step {k}", entropy_hist[0])
    print(f"update time of step {k}: {time.time() - update_start}")
    # post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)
model.save_pretrained(out_dir)