"""Loss / accuracy helpers shared by the SAE trainers."""
import torch
import torch.nn.functional as F


def nmse(rec, tgt):
    """Scale-normalised MSE: ||rec - tgt||^2 / ||tgt||^2 (batch mean)."""
    return F.mse_loss(rec, tgt) / (tgt.pow(2).mean() + 1e-8)


def hier_target(child_act, mask):
    """child_act: [B, n_child] in [0,1]; mask: [n_parent, n_child] 0/1.
    target_parent = min(1, sum over children) -> [B, n_parent]."""
    return (child_act @ mask.T).clamp(max=1.0)


def align_loss(logits, target_idx, kind="ce"):
    """CE (scale-free) or MSE to a one-hot target (anchors slots to 0/1)."""
    if kind == "ce":
        return F.cross_entropy(logits, target_idx)
    tgt = torch.zeros_like(logits)
    tgt[torch.arange(logits.shape[0], device=logits.device), target_idx] = 1.0
    return F.mse_loss(logits, tgt)


def value_loss_and_acc(value_logits, target_multihot, vmask, K):
    """value_logits: list[P] of [B, V_p]; target_multihot: [B, P, Vmax] (0/1).
    BCE over each property's vocabulary; accuracy = 'top-K contains every true value',
    averaged over (sample, property)."""
    P = len(value_logits)
    dev = value_logits[0].device
    total_bce = torch.zeros((), device=dev)
    hit = torch.zeros((), device=dev)
    for p in range(P):
        Vp = value_logits[p].shape[1]
        tgt = target_multihot[:, p, :Vp]
        total_bce = total_bce + F.binary_cross_entropy_with_logits(value_logits[p], tgt, reduction="mean")
        kk = min(K, Vp)
        topk = value_logits[p].topk(kk, dim=-1).indices
        pred = torch.zeros_like(tgt).scatter_(1, topk, 1.0)
        ok = ((tgt * (1.0 - pred)).sum(-1) == 0).float()
        hit = hit + ok.mean()
    return total_bce / P, hit / P
