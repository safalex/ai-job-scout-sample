"""
In-memory hybrid retrieval for the public sample.

Loads fictional CV chunks from fixtures/cv_chunks.json — no database or API keys.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from scout.documents.embed import embed_text
from scout.domain.models import EvidenceChunk
from scout.rag.hybrid_search import rank_hybrid
from scout.rag.models import ChunkMetadata, RankedChunk

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "cv_chunks.json"


@lru_cache(maxsize=1)
def _load_index() -> list[RankedChunk]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    rows: list[RankedChunk] = []
    for item in data.get("chunks", []):
        meta_raw = item.get("metadata") or {}
        rows.append(
            RankedChunk(
                chunk_id=int(item["chunk_id"]),
                document_id=int(item["document_id"]),
                content=str(item["content"]),
                path=str(item["path"]),
                metadata=ChunkMetadata.model_validate(meta_raw),
            )
        )
    return rows


def hybrid_retrieve(
    query: str,
    *,
    top_k: int = 8,
    doc_type: str | None = None,
    path_contains: str | None = None,
) -> list[RankedChunk]:
    candidates = _load_index()
    if path_contains:
        candidates = [c for c in candidates if path_contains in c.path]
    if not candidates:
        return []

    query_vec = np.array(embed_text(query), dtype=np.float32)
    scored: list[RankedChunk] = []
    for c in candidates:
        vec = np.array(embed_text(c.content), dtype=np.float32)
        n = min(vec.shape[0], query_vec.shape[0])
        sim = float(
            np.dot(vec[:n], query_vec[:n])
            / (np.linalg.norm(vec[:n]) * np.linalg.norm(query_vec[:n]) + 1e-9)
        )
        scored.append(c.model_copy(update={"vector_score": sim, "relevance": sim}))

    fused = rank_hybrid(query, scored)
    return fused[:top_k]


def to_evidence_chunks(ranked: list[RankedChunk]) -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            content=c.content,
            path=c.path,
            score=c.hybrid_score or c.vector_score,
        )
        for c in ranked
    ]
