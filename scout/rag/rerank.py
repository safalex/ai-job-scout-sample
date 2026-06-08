"""
Reranking — second pass after hybrid retrieval.

Teaches *reranking*: top vector hits may be broadly similar but wrong requirement.
We rescore with requirement-aware rules + evidence_strength patterns.
"""

from __future__ import annotations

import re

from scout.config.profile_policy import ProfilePolicy
from scout.cv_engine import evidence_strength
from scout.rag.models import PrimaryRequirement, RankedChunk

# Penalize chunks that only echo broad JD terms without specialist tokens.
_BROAD_ONLY = re.compile(
    r"\b(platform engineer|distributed systems|backend systems|data integration)\b",
    re.I,
)


def rerank_for_requirement(
    req: PrimaryRequirement,
    chunks: list[RankedChunk],
    profile: dict,
    *,
    profile_summary: str = "",
) -> list[RankedChunk]:
    """
    Rerank retrieved chunks for one requirement.

    Output relevance 0–1 and evidence_strength hint used by evidence.classify_coverage.
    """
    policy = ProfilePolicy.from_profile(profile)
    req_blob = f"{req.name} {req.evidence_needed}"
    out: list[RankedChunk] = []

    for c in chunks:
        blob = c.content
        if profile_summary:
            blob = f"{profile_summary[:500]} {blob}"

        level = evidence_strength.classify_requirement_evidence(
            req.name,
            blob,
            policy,
            specialist_domain=req.specialist_domain,
        )
        strength_bonus = {"strong": 0.35, "adjacent": 0.08, "weak": 0.02, "missing": -0.15}[level]

        hybrid = c.hybrid_score or c.vector_score
        relevance = min(1.0, max(0.0, hybrid + strength_bonus))

        # Specialist: demote broad-only chunks
        if req.importance == "specialist" or req.specialist_domain:
            req_tokens = set(re.findall(r"[a-z0-9]{4,}", req.name.lower()))
            chunk_tokens = set(re.findall(r"[a-z0-9]{4,}", c.content.lower()))
            if _BROAD_ONLY.search(c.content) and not (req_tokens & chunk_tokens):
                relevance *= 0.45

        out.append(c.model_copy(update={"relevance": relevance}))

    out.sort(key=lambda x: x.relevance, reverse=True)
    return out
