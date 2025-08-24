from __future__ import annotations
import torch

@torch.no_grad()
def fbeta_score(pred_logits: torch.Tensor, target: torch.Tensor, beta: float=4.0, num_classes:int=2):
    # pred_logits: (N,C,D,H,W), target: (N,D,H,W) int
    pred = pred_logits.argmax(dim=1)
    fbetas = []
    for c in range(1, num_classes):  # ignore background class 0
        tp = ((pred==c) & (target==c)).sum().float()
        fp = ((pred==c) & (target!=c)).sum().float()
        fn = ((pred!=c) & (target==c)).sum().float()
        beta2 = beta*beta
        denom = (1+beta2)*tp + beta2*fn + fp + 1e-8
        f = ((1+beta2)*tp) / denom
        fbetas.append(f.item())
    return sum(fbetas)/max(1, len(fbetas))