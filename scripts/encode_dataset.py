import argparse
import time
import numpy as np
from multiprocessing import Pool

from transformer_lm.tokenizer.tokenizer import Tokenizer
from transformer_lm.tokenizer.train_tokenizer import find_chunk_boundaries


def encode_chunk(task):
    input_path, start, end, vocab_path, merges_path, special_tokens = task
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

    print(f"finding chunk boundaries in {args.input}...", flush=True)
    boundaries = find_chunk_boundaries(args.input, args.num_chunks, "<|endoftext|>")
    total = len(boundaries)
    print(f"{total} chunks, {args.num_workers} workers", flush=True)

    tasks = [
        (args.input, s, e, args.vocab, args.merges, args.special_tokens)
        for s, e in boundaries
    ]

    chunks = []
    tokens_so_far = 0
    t0 = time.time()

    with Pool(processes=args.num_workers) as pool:
        for i, arr in enumerate(pool.imap(encode_chunk, tasks), start=1):
            chunks.append(arr)
            tokens_so_far += len(arr)
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"  chunk {i}/{total}  ({100*i/total:.1f}%)  "
                f"{tokens_so_far:,} tokens  "
                f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m",
                flush=True,
            )

    print("concatenating...", flush=True)
    out = np.concatenate(chunks)
    np.save(args.output, out)
    print(f"done. encoded {len(out):,} tokens -> {args.output}.npy "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()