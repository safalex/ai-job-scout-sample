"""
Hybrid retrieval — teaches why *semantic similarity ≠ evidence*.

Embeddings match broad phrases ("platform", "distributed systems").
Keyword overlap catches exact skills ("Ceph", "LiteLLM", "Kafka Connect").
We fuse both scores so specialist terms can win over vague semantic neighbors.
"""

from __future__ import annotations

import re
from collections import Counter

from scout.rag.models import RankedChunk

# Fusion weights (local-first; no external reranker API required).
VECTOR_WEIGHT = 0.55
KEYWORD_WEIGHT = 0.45

_TOKEN = re.compile(r"[a-z0-9+#.]{2,}", re.I)

# Phrases that must match literally for keyword boost (specialist evidence).
_EXACT_PHRASE_BOOST = 0.25


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def keyword_score(query: str, document: str) -> float:
    """
    Lightweight BM25-style keyword score (no Elasticsearch required).

    Returns 0.0–1.0 normalized score based on query term coverage in the document.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = tokenize(document)
    if not d_tokens:
        return 0.0
    df = Counter(d_tokens)
    doc_len = len(d_tokens)
    scores: list[float] = []
    for qt in set(q_tokens):
        tf = df.get(qt, 0)
        if tf == 0:
            continue
        # Saturating TF with length normalization
        num = (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_len / 120))
        scores.append(num)
    base = sum(scores) / max(len(set(q_tokens)), 1)
    base = min(1.0, base / 3.0)

    q_lower = query.lower()
    d_lower = document.lower()
    for phrase in re.findall(r'"([^"]+)"|\'([^\']+)\'', query):
        p = (phrase[0] or phrase[1]).strip().lower()
        if p and p in d_lower:
            base = min(1.0, base + _EXACT_PHRASE_BOOST)
    # Multi-word skill phrases from query (3+ char tokens joined)
    words = [t for t in q_tokens if len(t) > 2]
    if len(words) >= 2:
        bigram = f"{words[0]} {words[1]}"
        if bigram in d_lower:
            base = min(1.0, base + _EXACT_PHRASE_BOOST * 0.5)
    return base


def fuse_scores(vector_score: float, kw_score: float) -> float:
    """Combine dense and sparse signals."""
    v = max(0.0, min(1.0, vector_score))
    k = max(0.0, min(1.0, kw_score))
    return VECTOR_WEIGHT * v + KEYWORD_WEIGHT * k


def rank_hybrid(
    query: str,
    candidates: list[RankedChunk],
) -> list[RankedChunk]:
    """Re-score candidate chunks with keyword layer + fusion."""
    out: list[RankedChunk] = []
    for c in candidates:
        kw = keyword_score(query, c.content)
        hybrid = fuse_scores(c.vector_score, kw)
        out.append(
            c.model_copy(
                update={
                    "keyword_score": kw,
                    "hybrid_score": hybrid,
                    "relevance": hybrid,
                }
            )
        )
    out.sort(key=lambda x: x.hybrid_score, reverse=True)
    return out
