from typing import Any, Iterator, Optional,List
import itertools
import numpy as np
import torch
import torch.nn.functional as F
@torch.no_grad()
def pass_at_k(n: int,c: List[int],k:int):
    def estimator(n:int,c:int,k:int):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    n = itertools.repeat(n,len(c))
    return np.array([estimator(n,c,k) for n,c in zip(n,c)])
@torch.no_grad()
def compute_entropy_from_logits(logits, chunk_size: int = 128) -> torch.Tensor:
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)