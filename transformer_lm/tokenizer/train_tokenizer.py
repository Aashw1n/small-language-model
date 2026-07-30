import argparse
from collections import defaultdict, Counter
import json
import regex as re
import os
from pathlib import Path
from multiprocessing import Pool

from transformer_lm.tokenizer.tokenizer import gpt2_bytes_to_unicode


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
INPUT_PATH = "tests/fixtures/tinystories_sample_5M.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]



def find_chunk_boundaries(input_path: str, desired_num_chunks: int, split_special_token: str):
    with open(input_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)


        chunk_size = file_size // desired_num_chunks

        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        mini_chunk_size = 4096 # our lookahead. 

        for b in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[b]
            f.seek(initial_position)
            while True:
                mini_chunk = f.read(mini_chunk_size) # Look ahead by 4096
                if mini_chunk == b"":
                    chunk_boundaries[b] = file_size
                    break
                found_at = mini_chunk.find(split_special_token.encode("utf-8"))
                if found_at != -1:
                    chunk_boundaries[b] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        unique_boundaries = sorted(set(chunk_boundaries))
        boundary_pairs = [(boundary1, boundary2) for boundary1, boundary2 in zip(unique_boundaries, unique_boundaries[1:])]
        return boundary_pairs



chunk_pattern = re.compile(PAT) 

def pretokenize(input_path: str | Path, chunk_start: int, chunk_end: int, special_tokens: list[str]):
    counts = defaultdict(int)
    split_pattern = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None
    with open(input_path, "rb") as f:
        f.seek(chunk_start)
        segment = f.read(chunk_end - chunk_start).decode("utf-8", errors = "ignore")
        pieces = split_pattern.split(segment) if split_pattern else [segment]
        for p in pieces:
            for match in chunk_pattern.finditer(p):
                token = match.group()
                counts[tuple(token.encode("utf-8"))] += 1

    return counts


def train_tokenizer(input_path: str | Path, vocab_size: int, special_tokens: list[str]): 
    vocab: dict[int, bytes] = { x:bytes([x]) for x in range(256) }
    merges: list[tuple[bytes, bytes]] = [] # The index for the list indicates the actual replaced index in vocab.

    num_merges = vocab_size - 256 - len(special_tokens)

    # Get max counts across the text corpus

    # prepare inputs
    chunk_boundaries = find_chunk_boundaries(input_path, 48, "<|endoftext|>")
    inputs = [(input_path, start, end, special_tokens) for start,end in chunk_boundaries]
    results = []

    # Get a word frequency map from each worker
    with Pool(processes = 30) as pool:
        results = pool.starmap(pretokenize, inputs)

    # Compile in to a single frequency map, representing byte array frequencies across training corpus. 
    freq_map = defaultdict(int)
    for result in results:
        for token, c in result.items():
            freq_map[token] += c


    # Working data structures for performing merges

    words: list[list] = [] # Mutable form of the freq map
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)                 # Map pairs to the index in the words list
    counts: dict[tuple[int, int], int] = defaultdict(int)                             # Global pair counts dict

    # Populate pair_to_word and counts

    for idx, (byte_tuple, cts) in enumerate(freq_map.items()):
        words.append([list(byte_tuple), cts])
        for byte_pair in zip(byte_tuple, byte_tuple[1:]):
            counts[byte_pair] += cts
            pair_to_words[byte_pair].add(idx)

    m = 0
    while m < num_merges:
        if not counts:
            break

        pair = max(counts, key = lambda  p: (counts[p], vocab[p[0]], vocab[p[1]])) # Lexicographically largest occurrence pair
        indices = pair_to_words.get(pair)
        indices_copy = indices.copy()
        new_index = 256 + m

        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))


        # Perform the merge
        for index in indices_copy:
            original_list = words[index][0]
            new_list = []

            w = 0
            while w < len(original_list):
                if w < len(original_list) - 1 and original_list[w] == pair[0] and original_list[w+1] == pair[1]:
                    new_list.append(new_index)
                    w += 2
                else:
                    new_list.append(original_list[w])
                    w += 1

            old = Counter([(o1, o2) for o1,o2 in zip(original_list, original_list[1:])])
            new = Counter([(n1, n2) for n1,n2 in zip(new_list, new_list[1:])])
            cnt = words[index][1]

            for pr in set(old) | set(new):
                delta = new[pr] - old[pr] # Count the change in pair frequency pre and post merge
                if delta:
                    counts[pr] += cnt * delta
                    if new[pr] and not old[pr]:
                        pair_to_words[pr].add(index)
                    if old[pr] and not new[pr]:
                        pair_to_words[pr].discard(index)
            
            words[index][0] = new_list

        del counts[pair]
        del pair_to_words[pair]


    
        m += 1

    # Add in special tokens.
    for tok in special_tokens:
        vocab[256+m] = tok.encode("utf-8")
        m += 1


    return vocab, merges



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--vocab-out", required=True)
    p.add_argument("--merges-out", required=True)
    args = p.parse_args()

    vocab, merges = train_tokenizer(args.input, args.vocab_size, SPECIAL_TOKENS)

    byte_encoder = gpt2_bytes_to_unicode()
    to_str = lambda b: "".join(byte_encoder[x] for x in b)

    with open(args.vocab_out, "w", encoding="utf-8") as f:
        json.dump({to_str(tok): idx for idx, tok in vocab.items()}, f, ensure_ascii=False)

    with open(args.merges_out, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{to_str(a)} {to_str(b)}\n")

    print(f"vocab size {len(vocab)}, merges {len(merges)}")