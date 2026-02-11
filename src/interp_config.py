import json
import re
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
import random
from reward import reward_answer_binary
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, IterableDataset



def extract_gsm8k(prompt_batch):
    return prompt_batch["question"], list(map(lambda el: el.split("####")[0], prompt_batch["answer"])), list(map(lambda el: el.split(" ")[-1], prompt_batch["answer"]))

def process_config(config, ds_seed, device_index = 0):
    tmp = {}
    with open(config,"r") as fd:
        tmp = json.load(fd)
    
    model_name = tmp["model_name"][device_index]
    world_size = len(tmp["model_name"])
    group_size = int(tmp["group_size"])
    batch_size = int(tmp["batch_size"])
    comm_style = tmp["comm_style"]
    # task_dataset = tmp["task"]
    skip = 0
    
    data_interp = extract_gsm8k
    dl = load_dataset("openai/gsm8k","main", split="train",streaming = True, trust_remote_code=True)
    val_loader = load_dataset("openai/gsm8k","main", split="test",streaming = True, trust_remote_code=True)
    reward_func = reward_answer_binary

    dl = dl.shuffle(buffer_size=5_000, seed=ds_seed)
    val_loader = val_loader.shuffle(buffer_size=5_000, seed=22)
    


    return {
        "model_name": model_name,
        "dl_benign": dl,
        "val_loader": val_loader,
        "group_size": group_size,
        "batch_size": batch_size,
        "reward_func": reward_func,
        "data_interp": data_interp,
        "comm_style": comm_style,
        "world_size": world_size
    }
