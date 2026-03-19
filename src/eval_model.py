from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import GenerationConfig
from torch.utils.data import DataLoader
from sys import argv
import torch.distributed as dist
import torch
from reward import reward_answer_binary
from datasets import load_dataset
base_model = "Qwen/Qwen2.5-1.5B"

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags with nothing outside said tags, i.e., <think> reasoning process here </think><answer> answer here </answer>\n
"""
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("1B_code", device_map="cuda:0",attn_implementation="kernels-community/flash-attn3", dtype=torch.bfloat16)

val_loader = load_dataset("openai/gsm8k","main", split="test",streaming = True, trust_remote_code=True)
val_loader = val_loader.shuffle(buffer_size=5_000, seed=22)
val_loader = DataLoader(
    val_loader,
    batch_size=16,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)

def extract_gsm8k(prompt_batch):
    return prompt_batch["question"], list(map(lambda el: el.split("####")[0], prompt_batch["answer"])), list(map(lambda el: el.split(" ")[-1], prompt_batch["answer"]))


model1 = model
model = AutoModelForCausalLM.from_pretrained("1B_base", device_map="cuda:0",attn_implementation="kernels-community/flash-attn3", dtype=torch.bfloat16)




model2 = model





generation_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            top_k = 50,
        )


def one_only(s1,s2,choice = 0):
    if choice == 0:
        s1 = torch.softmax(s1, dim=-1)
        v1,m1 = torch.topk(s1, k=1, dim=-1)
        return m1
    else:
        s2 = torch.softmax(s2, dim=-1)
        v2,m2 = torch.topk(s2, k=1, dim=-1)
        return m2

def max_decision(s1,s2):
    s1 = torch.softmax(s1, dim=-1)
    s2 = torch.softmax(s2, dim=-1)
    
    v1,m1 = torch.topk(s1, k=1, dim=-1)
    v2,m2 = torch.topk(s2, k=1, dim=-1)
    return m1 * (v1 > v2).long() + m2 * (v2 >= v1).long()

def log_max(s1,s2):
    s1 = torch.log_softmax(s1, dim=-1)
    s2 = torch.log_softmax(s2, dim=-1)
    
    v1,m1 = torch.topk(s1, k=1, dim=-1)
    v2,m2 = torch.topk(s2, k=1, dim=-1)
    return m1 * (v1 > v2).long() + m2 * (v2 >= v1).long()

def log_mix(s1,s2,alpha=0.5):
    s1 = torch.log_softmax(s1, dim=-1)
    s2 = torch.log_softmax(s2, dim=-1)
    mix = torch.log(s1.exp() * alpha + s2.exp()* (1-alpha))
    v1,m1 = torch.topk(mix, k=1, dim=-1)
   
    return m1

def kl(s1,s2, alpha = 0.5):
    s1 = torch.log_softmax(s1, dim=-1)
    s2 = torch.log_softmax(s2, dim=-1)
    mix = torch.log(s1.exp() * alpha + s2.exp()* (1-alpha))
    s1 = s1.exp()
    kl1 = (s1.exp() * (s1  - mix))
    kl2 = (s2.exp() * (s2 - mix))
    kl1 = kl1.sum(-1)
    kl2 = kl2.sum(-1)
    v1,m1 = torch.topk(s1, k=1, dim=-1)
    v2,m2 = torch.topk(s2, k=1, dim=-1)
    return m1 * (kl1 < kl2).unsqueeze(-1).long() + m2 * (kl2 <= kl1).unsqueeze(-1).long()


    
val_loader = iter(val_loader)
sum_rewards = 0
cannot_solve = []
for mb in range(4):
    val_batch = next(val_loader)
    q,s,a = extract_gsm8k(val_batch)
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
    ).to(model2.device)
    start_seq = model_inputs["input_ids"].shape[1]
    sequence_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    storage = []
    unfinished_sequences = torch.ones((sequence_ids.shape[0],1), dtype=torch.long, device=sequence_ids.device)
    past1, past2 = None, None
    next_token = sequence_ids
    with torch.no_grad():
        while True:
            if sequence_ids.shape[1] == 2048:
                break
            out1 = model1(next_token,attention_mask = attention_mask, past_key_values=past1)
            s1 = out1.logits[:,-1]
            out2 = model2(next_token,attention_mask = attention_mask, past_key_values=past2)
            s2 = out2.logits[:,-1]
            past1 = out1.past_key_values
            past2 = out2.past_key_values

            # next_token = kl(s1,s2,0.5)
            next_token = one_only(s1,s2,0)
            next_token = next_token*unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            sequence_ids =  torch.cat([sequence_ids, next_token], dim=-1)
            
            attention_mask =  torch.cat([attention_mask, unfinished_sequences], dim=-1)
            unfinished_sequences = unfinished_sequences & (next_token != tokenizer.eos_token_id)
            
            if unfinished_sequences.sum().item() == 0:
                break
    
    completion_ids = sequence_ids[:, start_seq :]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    rewards = reward_answer_binary(completions,a)[0]
    print(completions[0])
    print(a[0])
    print(q[0])
    sum_rewards += rewards.sum().item()
    cannot_solve += [elm + mb*16 for elm in (rewards.flatten() == 0).nonzero()]
    # exit()
    print(sum_rewards)
    print(cannot_solve)

print(sum_rewards)
print(cannot_solve)