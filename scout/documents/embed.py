"""Deterministic hash embeddings for offline sample runs (no API keys)."""

from __future__ import annotations

import hashlib
import json

import numpy as np

_DEFAULT_DIM = 64


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    dim = dimensions or _DEFAULT_DIM
    return _hash_embed(text, dim)


def _hash_embed(text: str, dimensions: int) -> list[float]:
    seed = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
    vec = rng.standard_normal(dimensions).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


def json_to_vector(blob: str) -> np.ndarray:
    return np.array(json.loads(blob), dtype=np.float32)
