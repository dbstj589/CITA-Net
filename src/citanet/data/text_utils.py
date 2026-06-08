"""Offline tokenisation + a lightweight bag-of-tokens cosine.

Used by (a) blocking, as a cheap pre-filter semantic similarity that does not
depend on any trained weights, and (b) the learned text encoder, to build its
vocabulary. Tokenisation keeps Korean/alphanumeric tokens and preserves
internal hyphens so military designators like ``T-72`` stay intact.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Split on whitespace; then trim leading/trailing punctuation but keep internal
# hyphens/dots (e.g. "T-72", "MO-120-RT", "2S1").
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣][0-9A-Za-z가-힣\-\.]*")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def token_counts(text: str) -> Counter:
    return Counter(tokenize(text))


def text_cosine(a: str, b: str) -> float:
    """Cosine similarity between two strings' bag-of-tokens vectors in [0, 1]."""
    ca, cb = token_counts(a), token_counts(b)
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
