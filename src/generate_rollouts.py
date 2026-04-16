import torch
from transformers import GenerationConfig
import torch.nn.functional as F
from typing import List

@torch.no_grad()
def generate_rollouts(model, tokenizer,  question : str, sys_prompt: str = None, num_rollouts = 6, generation_config : GenerationConfig = None, is_conversational = True):

    model.eval()
    chat_messages = []
    if generation_config == None:
        generation_config = GenerationConfig(
            max_length=760,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature = 1.0,
            top_p = 1.0,
            top_k = 50,
        )

    if is_conversational:
        if sys_prompt != None:
            chat_messages.append({
                "role": "system",
                "content": sys_prompt,
            })
        chat_messages.append({
                "role": "user",
                "content": question,
            })
        model_inputs = tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    return_dict=True,
                    return_attention_mask=True
                ).to(model.device)
    else:
        model_inputs = tokenizer(
            [question],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        ).to(model.device)
    
    # TODO: Add prefix caching
    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollouts, 1)
    start_seq = model_inputs["input_ids"].shape[1]


    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)

    completion_ids = sequence_ids[:, start_seq :]
    action_mask = (completion_ids != tokenizer.pad_token_id).long()

    return model_inputs["input_ids"], model_inputs["attention_mask"], completion_ids, action_mask