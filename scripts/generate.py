import argparse
import torch

from transformer_lm.model import TransformerLM
from transformer_lm.tokenizer.tokenizer import Tokenizer


def top_p_filter(probs, p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # keep everything up to and including the token that crosses p
    keep = (cumsum - sorted_probs) < p
    sorted_probs[~keep] = 0.0
    sorted_probs /= sorted_probs.sum()
    out = torch.zeros_like(probs)
    out.scatter_(0, sorted_idx, sorted_probs)
    return out


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, temperature, top_p, context_length, device):
    eot_id = tok.byte_to_id["<|endoftext|>".encode("utf-8")]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        window = ids[:, -context_length:]
        logits = model(window)[0, -1]

        if temperature <= 0:                      # greedy
            next_id = torch.argmax(logits)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                probs = top_p_filter(probs, top_p)
            next_id = torch.multinomial(probs, 1).squeeze()

        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == eot_id:
            break

    return tok.decode(ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="cuda")
    # must match training config
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--theta", type=float, default=10000.0)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.vocab, args.merges, ["<|endoftext|>"])

    model = TransformerLM(
        args.vocab_size, args.context_length, args.num_layers,
        args.d_model, args.num_heads, args.d_ff, args.theta,
    ).to(args.device)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from iteration {ckpt['iteration']}\n")

    for i in range(args.num_samples):
        text = generate(
            model, tok, args.prompt,
            args.max_new_tokens, args.temperature, args.top_p,
            args.context_length, args.device,
        )
        print(f"--- sample {i+1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()