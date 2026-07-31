import torch

def cross_entropy_loss(logits, targets):
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (lse - target_logits).mean()