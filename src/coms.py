import torch.distributed as dist
import torch.nn.functional as F
import torch
import os

def pad_tensor(tensor: torch.Tensor, pad_token, pad_to = 2048):
    return F.pad(seq_log_probs, (0,pad_to - tensor.shape[1]), "constant", pad_token)

def unpad_tensor(tensor: torch.Tensor, pad_token: int):
    max_el = 0
    for el in range(tensor.shape[0]):
        t = tensor.shape[1] - 1
        while t > 0:
            if tensor[el][t] != pad_token:
                max_el = max(max_el,t+1)
                break
            t -= 1
    tensor = tensor[:,:max_el]
    
    return tensor, max_el


def gather(out: torch.Tensor, rank, world_size, group = None):
    tensor_shapes = len(out.shape)
    _t1 = [torch.zeros(tensor_shapes,dtype = torch.long, device = out.device) for _ in range(world_size)]
    dist.all_gather(_t1,torch.tensor(list(out.shape), device = out.device), group=group)
    _t2 = [torch.zeros(_t1[id].tolist(), dtype = out.dtype, device = out.device) for id in range(world_size)]
    for elm in _t2:
        print(elm.shape,elm.dtype,elm.device)
    print(out.shape,out.dtype,out.device)
    dist.all_gather(_t2,out, group=group)
    return _t2


def setup_comms(rank, world_size, backend = "nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)