
def cross_entropy_loss(logits, targets):
    # Get max logits
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    lse = shifted.exp().sum(dim=-1).log()
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss_per_position = m.squeeze(-1) + lse - target_logits


    return loss_per_position.mean()