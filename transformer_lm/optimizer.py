from typing import Callable, Optional
import torch
import math



# AdamW optimizer
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01):
        defaults = {
            "lr" : lr,
            "betas" : betas,
            "eps" : eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)


    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["betas"][0]
            b2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                g = p.grad.data
                m = state.get("m")
                v = state.get("v")
                t = state.get("t",0)

                t += 1 
                state["m"] = (b1*m) + ((1-b1) * g)
                state["v"] = (b2*v) + ((1-b2) * torch.square(g))
                m = state["m"]
                v = state["v"]

                alpha_t = lr * math.sqrt(1-(b2**t)) / (1 - b1**t)

                p.data -= lr * weight_decay * p.data
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["t"] = t
        return loss       



def gradient_clipping(params, m, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    scale = torch.clamp(m / (norm + eps), max=1.0)
    for g in grads:
        g.mul_(scale)



# Cosine learning rate scheduling
def get_lr_cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_iters, cosine_cycle_iters):

    # Warmup
    if it < warmup_iters:
        return max_lr * it/warmup_iters
    
    # cosine annealing
    if warmup_iters <= it <= cosine_cycle_iters:
        return min_lr + (1 + math.cos(math.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)))/2 * (max_lr-min_lr)
    
    return min_lr