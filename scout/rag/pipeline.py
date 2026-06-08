"""
Requirement-level RAG pipeline entry — per-requirement retrieve, rerank, coverage.

Flow:
  JD → structured requirements → queries per requirement → hybrid retrieve → rerank → coverage
"""

from __future__ import annotations

from scout.domain.enums import DocType
from scout.domain.models import EvidenceChunk, JobRecord
from scout.llm.schemas import JDRequirementsExtraction, RequirementEvidence
from scout.rag.evidence import (
    build_coverage_report,
    classify_chunk_coverage,
    coverage_to_requirement_evidence,
)
from scout.rag.models import RankedChunk, RequirementCoverageReport
from scout.rag.requirements import build_structured_requirements, expand_queries_for_requirement
from scout.rag.rerank import rerank_for_requirement
from scout.rag.retrieval import hybrid_retrieve, to_evidence_chunks


def retrieve_for_requirements_v2(
    job: JobRecord,
    extraction: JDRequirementsExtraction,
    profile: dict,
    *,
    per_requirement_top_k: int = 3,
    max_requirements: int = 10,
    max_total_chunks: int = 14,
    role_centre: str | None = None,
) -> tuple[list[EvidenceChunk], list[RequirementEvidence], RequirementCoverageReport]:
    """Per-requirement hybrid RAG with rerank and evidence coverage."""
    pipeline = profile.get("pipeline") or {}
    per_requirement_top_k = int(
        pipeline.get("requirement_rag_per_req_top_k", per_requirement_top_k)
    )
    max_requirements = int(pipeline.get("requirement_rag_max_requirements", max_requirements))
    max_total_chunks = int(pipeline.get("job_evidence_top_k", max_total_chunks))

    structured = build_structured_requirements(
        job, extraction, profile, role_centre=role_centre
    )
    reqs = structured.primary_requirements[:max_requirements]
    if not reqs:
        reqs = structured.secondary_requirements[:max_requirements]

    profile_summary = (profile.get("candidate") or {}).get("cv_skills_summary", "")
    merged: dict[int, RankedChunk] = {}
    coverage_items = []

    for req in reqs:
        qset = expand_queries_for_requirement(
            req,
            profile,
            job_title=job.title,
            job_description=job.description_text,
        )
        per_req_chunks: list[RankedChunk] = []
        for query in qset.queries:
            hits = hybrid_retrieve(
                query,
                top_k=per_requirement_top_k,
                doc_type=DocType.CV.value,
            )
            per_req_chunks.extend(hits)

        # Dedupe by chunk_id, keep best relevance
        deduped: dict[int, RankedChunk] = {}
        for c in per_req_chunks:
            prev = deduped.get(c.chunk_id)
            if prev is None or c.relevance > prev.relevance:
                deduped[c.chunk_id] = c
        ranked = rerank_for_requirement(
            req, list(deduped.values()), profile, profile_summary=profile_summary
        )
        ranked = ranked[:per_requirement_top_k]

        item = classify_chunk_coverage(req, ranked, profile, profile_summary=profile_summary)
        coverage_items.append(item)
        for c in ranked:
            prev = merged.get(c.chunk_id)
            if prev is None or c.relevance > prev.relevance:
                merged[c.chunk_id] = c

    report = build_coverage_report(coverage_items, role_centre=structured.role_centre)
    legacy_rows = [coverage_to_requirement_evidence(i) for i in coverage_items]
    evidence = to_evidence_chunks(
        sorted(merged.values(), key=lambda x: x.relevance, reverse=True)[:max_total_chunks]
    )
    return evidence, legacy_rows, report


def retrieve_job_evidence(
    job: JobRecord,
    profile: dict,
    *,
    extraction: JDRequirementsExtraction | None = None,
    role_centre: str | None = None,
) -> tuple[list[EvidenceChunk], list[RequirementEvidence], JDRequirementsExtraction | None]:
    """Unified entry used by runner/analyzer/rescore (3-tuple for backward compatibility)."""
    pipeline = profile.get("pipeline") or {}
    use_rag = pipeline.get("requirement_rag_enabled", True)

    if not use_rag or extraction is None:
        from scout.rag.retrieval import retrieve as hybrid_retrieve_flat

        top_k = int(pipeline.get("job_evidence_top_k", 8))
        query = f"{job.title} {job.company} {job.description_text[:500]}"
        chunks = hybrid_retrieve_flat(query, top_k=top_k, doc_type=DocType.CV.value)
        return chunks, [], extraction

    evidence, coverage, _report = retrieve_for_requirements_v2(
        job, extraction, profile, role_centre=role_centre
    )
    return evidence, coverage, extraction
