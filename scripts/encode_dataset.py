import argparse
import numpy as np
from multiprocessing import Pool

from transformer_lm.tokenizer.tokenizer import Tokenizer
from transformer_lm.tokenizer.train_tokenizer import find_chunk_boundaries


def encode_chunk(input_path, start, end, vocab_path, merges_path, special_tokens):
    tok = Tokenizer.from_file(vocab_path, merges_path, special_tokens)
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.array(tok.encode(text), dtype=np.uint16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--num-workers", type=int, default=30)
    p.add_argument("--num-chunks", type=int, default=300)
    args = p.parse_args()

    boundaries = find_chunk_boundaries(args.input, args.num_chunks, "<|endoftext|>")
    tasks = [
        (args.input, s, e, args.vocab, args.merges, args.special_tokens)
        for s, e in boundaries
    ]

    with Pool(processes=args.num_workers) as pool:
        chunks = pool.starmap(encode_chunk, tasks)

    arr = np.concatenate(chunks)
    np.save(args.output, arr)
    print(f"encoded {len(arr)} tokens -> {args.output}.npy")


if __name__ == "__main__":
    main()