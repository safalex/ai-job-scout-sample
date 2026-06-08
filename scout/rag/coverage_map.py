"""Shared coverage row helpers for graph nodes and legacy RAG scoring."""

from __future__ import annotations

from scout.llm.schemas import RequirementEvidence
from scout.rag.evidence import coverage_to_requirement_evidence
from scout.rag.models import RequirementCoverageItem


def rows_to_requirement_evidence(rows: list[dict]) -> list[RequirementEvidence]:
    """Normalize graph state evidence_coverage dicts to RequirementEvidence."""
    out: list[RequirementEvidence] = []
    for row in rows:
        try:
            out.append(RequirementEvidence.model_validate(row))
        except Exception:
            continue
    return out


def coverage_items_to_evidence_dicts(items: list[RequirementCoverageItem]) -> list[dict]:
    """Pipeline report rows → graph state / adjustments JSON."""
    return [coverage_to_requirement_evidence(item).model_dump() for item in items]


def summarize_coverage_counts(rows: list[dict]) -> str:
    """Human-readable summary for graph trace (classify_evidence_coverage_node)."""
    counts: dict[str, int] = {}
    for row in rows:
        cov = row.get("coverage", "missing")
        key = str(cov).replace("_evidence", "")
        counts[key] = counts.get(key, 0) + 1
    return " / ".join(f"{v} {k}" for k, v in sorted(counts.items()))
