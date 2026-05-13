"""Tiny BPE (byte-pair encoding) tokenizer.

Standard greedy BPE: start with byte-level vocabulary, repeatedly merge
the most frequent adjacent pair into a new token until target vocab size.

This is the same algorithm GPT-2 and friends use, just minimal-Python.
For the physics corpus a 256-token vocab compresses each line by ~4× vs
char-level, letting the LM see longer effective context.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class BPETokenizer:
    """A toy BPE tokenizer with greedy encode."""

    vocab: list[bytes]                       # token id -> bytes
    merges: list[tuple[bytes, bytes]]        # ordered list of merge rules

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int = 256) -> "BPETokenizer":
        data = text.encode("utf-8")
        # Start with each byte as its own token: tokens are list[int] for now.
        tokens = list(data)
        # Per-byte vocab: id i -> bytes([i])
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        merges: list[tuple[int, int]] = []

        while len(vocab) < vocab_size:
            # Count adjacent pairs.
            pairs: Counter[tuple[int, int]] = Counter()
            for a, b in zip(tokens, tokens[1:]):
                pairs[(a, b)] += 1
            if not pairs:
                break
            best_pair, best_count = pairs.most_common(1)[0]
            if best_count < 2:
                break
            new_id = len(vocab)
            vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
            merges.append(best_pair)

            # Replace every occurrence of best_pair with new_id, in-place.
            out: list[int] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out

        # Convert merges to bytes-pair form for portability.
        return cls(
            vocab=vocab,
            merges=[(vocab[a], vocab[b]) for (a, b) in merges],
        )

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        tokens: list[bytes] = [bytes([b]) for b in data]

        for left, right in self.merges:
            merged = left + right
            out: list[bytes] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == left and tokens[i + 1] == right:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out

        token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        return [token_to_id[tok] for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
