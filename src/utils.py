from typing import Any, Iterator, Optional,List
import itertools
import numpy as np
def pass_at_k(n: int,c: List[int],k:int):
    def estimator(n:int,c:int,k:int):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    n = itertools.repeat(n,len(c))
    return np.array([estimator(n,c,k) for n,c in zip(n,c)])