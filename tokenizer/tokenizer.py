from pathlib import Path
import regex as re
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator


def gpt2_bytes_to_unicode() -> dict[int, str]:
        bs = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))
        cs = bs[:]
        n = 0
        for i in range(256):
            if i not in bs:
                bs.append(i)
                cs.append(256+n)
                n += 1

        return {b:chr(c) for b,c in zip(bs,cs)}
    
def gpt2_unicode_to_bytes() -> dict[str,int]:
        return {v:k for k,v in gpt2_bytes_to_unicode().items()}


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in self.vocab.items()} # for encoding
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}


    
    # Load vocab from file. 
    @classmethod
    def from_file(cls, vocab_filepath: str | Path, merges_filepath: str| Path, special_tokens: list[str] | None = None) -> "Tokenizer":
        # populate vocab
        byte_decoder = gpt2_unicode_to_bytes()
        to_bytes = lambda s: bytes(byte_decoder[ch] for ch in s)

        with open(vocab_filepath, encoding = "utf-8") as f:
            raw = json.load(f)
            vocab = {k:to_bytes(v) for v,k in raw.items()}

        # Populate merges
        merges = []
        with open(merges_filepath, encoding = "utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                a , b = line.split(" ")
                merges.append((to_bytes(a), to_bytes(b)))

        return cls(vocab, merges, special_tokens)
        

    def encode(self, string: str) -> list[int]:
        # Segment on special tokens first.
        if self.special_tokens:
            specials = sorted(self.special_tokens, key = len, reverse=True)
            segment_pattern = re.compile("(" + "|".join(re.escape(t) for t in specials) + ")")
            segments = re.split(segment_pattern, string)
        else:
            segments = [string]

        result = []
        chunk_pattern = re.compile(PAT)

        for segment in segments:
            if segment in self.special_tokens:
                result.append(self.byte_to_id[segment.encode("utf-8")])

            else:
                
                for match in chunk_pattern.finditer(segment):
                    token = match.group()
                    byte_list = [bytes([x]) for x in token.encode("utf-8")]
                    new_list = byte_list.copy()

                    # Apply merges
                    while True:
                        # If we don't have enough for a pair, we break off
                        if len(new_list) < 2:
                            break

                        # Get all pairs
                        pairs = [(i1,i2) for i1,i2 in zip(new_list, new_list[1:])]

                        # Get all ranks in the merge list. 
                        rank = {p: self.merge_rank[p] for p in pairs if p in self.merge_rank}
                        if not rank:
                            break
                        min_pair = min(rank, key=rank.get)

                        # We apply merge now
                        merge_applied_list = []
                        i = 0
                        while i < len(new_list):
                            if i < len(new_list) - 1 and new_list[i] == min_pair[0] and new_list[i+1] == min_pair[1]:
                                merge_applied_list.append(min_pair[0] + min_pair[1]) # Concatenate/merge
                                i += 2
                            else:
                                merge_applied_list.append(new_list[i])
                                i += 1

                        new_list = merge_applied_list

                    for tok in new_list:
                        result.append(self.byte_to_id[tok])


        return result 

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)



    def decode(self, byte_array: list[int]) -> str:
        decoded: bytes = b""
        for i in byte_array:
            decoded += self.vocab[i]
        return decoded.decode("utf-8", errors="replace")



if __name__ == "__main__":
     tokenizer = Tokenizer.from_file( "tests/fixtures/train-bpe-reference-vocab.json","tests/fixtures/train-bpe-reference-merges.txt")
     print(tokenizer.encode("hello my friend"))
     print(tokenizer.decode([259, 76, 491, 486, 377, 73, 69, 269]))