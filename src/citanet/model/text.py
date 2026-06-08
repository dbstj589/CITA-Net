"""Text encoding (§4.2, offline default).

`learned`: a token vocabulary built from training labels, learned token
embeddings, mean-pooled per label. Fully deterministic and offline -- no
network. `sbert` is an optional path that auto-falls-back to `learned` if the
model cannot be loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ..data.text_utils import tokenize

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Token <-> id mapping built deterministically from a corpus of labels."""

    def __init__(self, tokens: list[str]):
        self.itos = [PAD, UNK] + tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_tokens: int) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:max_tokens]
        if not ids:
            ids = [1]            # <unk> for empty labels
        return ids

    @classmethod
    def build(cls, labels: list[str]) -> "Vocab":
        vocab: set[str] = set()
        for lab in labels:
            vocab.update(tokenize(lab))
        return cls(sorted(vocab))

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        v = cls.__new__(cls)
        v.itos = d["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class LearnedTextEncoder(nn.Module):
    """Mean-pooled learned token embeddings -> d_model."""

    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(len(vocab), d_model, padding_idx=0)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # token_ids: (N, L), mask: (N, L) in {0,1}
        emb = self.embed(token_ids)                       # (N, L, d)
        m = mask.unsqueeze(-1).float()
        summed = (emb * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.proj(pooled)
