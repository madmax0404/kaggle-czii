from __future__ import annotations
import os, random, time
import numpy as np
import torch

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0; self.cnt = 0
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)