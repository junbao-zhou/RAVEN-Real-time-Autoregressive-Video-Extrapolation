"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import contextlib
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from project.utils import comm
from project.utils.dataclass import Dataclass


@dataclass
class RandomState(Dataclass):
    """ Random number generators for different libraries. """
    seed: int = field(default=None)
    python_generator: random.Random = field(default=None)
    numpy_generator: np.random.Generator = field(default=None)
    torch_generator: torch.Generator = field(default=None)
    torch_cuda_generator: torch.Generator = field(default=None)

    def __post_init__(self) -> None:
        if self.seed is not None:
            if self.python_generator is None:
                self.python_generator = random.Random(self.seed)
            if self.numpy_generator is None:
                self.numpy_generator = np.random.default_rng(self.seed)
            if self.torch_generator is None:
                self.torch_generator = torch.Generator().manual_seed(self.seed)
            if self.torch_cuda_generator is None and torch.cuda.is_available():
                self.torch_cuda_generator = torch.Generator(device=comm.get_device()).manual_seed(self.seed)
        super().__post_init__()

    def fork(self, index: int) -> "RandomState":
        assert self.seed is not None, "Cannot fork RandomState without seed"
        assert index >= 0, f"RandomState fork index must be non-negative, got {index}"
        seed = self.seed
        for _ in range(index + 1):
            seed = yield_seed(seed)
        return RandomState(seed=seed)


@contextlib.contextmanager
def local_seed(seed: Optional[int]):
    """
    Create a local context with seed is set, but exit back to the original random state.
    If seed is None, do nothing.
    """
    if seed is not None:
        random_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        torch_cuda_state = torch.cuda.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state(torch_cuda_state)
    else:
        yield


def yield_seed(seed, a=1103515245, c=12345, m=2**31):
    """
    Yield a random number from a given seed.
    """
    return (a * seed + c) % m


def combine_seed(*args) -> int:
    """Deterministic seed combination without collision."""
    h = hashlib.sha256("_".join(str(a) for a in args).encode()).hexdigest()
    return int(h, 16) % (2**63)
