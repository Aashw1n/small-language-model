import numpy as np
import argparse
import wandb
import torch

from transformer_lm.data import get_batch, save_checkpoint
from transformer_lm.loss import cross_entropy_loss
from transformer_lm.model import TransformerLM
from transformer_lm.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping





torch.set_float32_matmul_precision("high")

def parse_args():
    p = argparse.ArgumentParser()
    # model config
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--train-path", type=str, required=True)
    p.add_argument("--val-path", type=str, required=True)
    p.add_argument("--ckpt-path", type=str, default="checkpoint.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    return p.parse_args()


def train(args):
    wandb.init(
        project="small-language-model",
        config= vars(args),
        name = f"lr{args.lr}-bs{args.batch_size}"
    )
    device = args.device
    model = TransformerLM(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.theta).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r")

    for step in range(args.total_steps):
        # batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # forward + loss
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # clip grads (after backward, before step)
        gradient_clipping(model.parameters(), args.max_grad_norm)

        # cosine lr -> inject into optimizer
        lr = get_lr_cosine_schedule(
            step, args.lr, args.lr * 0.1, args.warmup_steps, args.total_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # step
        optimizer.step()

        # logging
        if step % args.log_every == 0:
            wandb.log({"train_loss": loss.item(), "lr": lr}, step=step)
            print(f"step {step}  loss {loss.item():.4f}  lr {lr:.2e}")

        # validation
        if step % args.val_every == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, device)
                val_loss = cross_entropy_loss(model(vx), vy).item()
            model.train()
            wandb.log({"val_loss": val_loss}, step=step)
            print(f"step {step}  val_loss {val_loss:.4f}")

        # checkpoint
        if step % args.ckpt_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.ckpt_path)

    save_checkpoint(model, optimizer, args.total_steps, args.ckpt_path)
    if device == "cuda":
        print(f"peak alloc: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    wandb.finish()



def main():
    args = parse_args()
    train(args)

if __name__ =="__main__":
    main()