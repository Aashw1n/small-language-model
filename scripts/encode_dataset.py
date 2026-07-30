import argparse
import numpy as np

from transformer_lm.tokenizer.tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    args = p.parse_args()

    tok = Tokenizer.from_file(args.vocab, args.merges, args.special_tokens)

    ids = []
    with open(args.input, encoding="utf-8") as f:
        for token_id in tok.encode_iterable(f):
            ids.append(token_id)

    arr = np.array(ids, dtype=np.uint16)
    np.save(args.output, arr)
    print(f"encoded {len(arr)} tokens -> {args.output}.npy")


if __name__ == "__main__":
    main()