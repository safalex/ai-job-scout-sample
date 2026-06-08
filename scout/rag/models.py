"""RAG domain models — requirement-level evidence, not whole-document similarity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class PrimaryRequirement(BaseModel):
    """One structured requirement extracted from a JD (or role-centre gate)."""

    id: str
    name: str
    type: str = "skill"  # skill | stack | delivery | domain | leadership
    importance: str = "primary"  # primary | secondary | hard | specialist
    evidence_needed: str = ""
    specialist_domain: str | None = None


class RequirementQuerySet(BaseModel):
    """Retrieval queries for a single requirement (query expansion)."""

    requirement_id: str
    queries: list[str] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    """Chunk-level metadata for grounded citations (stored as JSON on DocumentChunk)."""

    stable_chunk_id: str
    source_doc: str = ""
    section: str = ""
    role: str = ""
    company: str = ""
    date_range: str = ""
    chunk_type: str = "paragraph"  # experience_bullet | experience_block | skills | header | paragraph
    skills: list[str] = Field(default_factory=list)
    project: str | None = None


class RankedChunk(BaseModel):
    """Chunk after hybrid retrieval + rerank."""

    chunk_id: int
    document_id: int
    content: str
    path: str
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    relevance: float = 0.0
    metadata: ChunkMetadata | None = None


class EvidenceChunkRef(BaseModel):
    """Citation surfaced in UI / exports."""

    chunk_id: int
    source: str = ""
    text_excerpt: str = ""
    reason: str = ""
    used_as_evidence: bool = True


class RequirementCoverageItem(BaseModel):
    """Coverage for one requirement — maps to RequirementEvidence for cache compat."""

    requirement: str
    requirement_id: str = ""
    importance: str = "primary"
    coverage: str  # strong | adjacent | weak | missing (also legacy *_evidence suffix)
    evidence_chunks: list[EvidenceChunkRef] = Field(default_factory=list)
    inspected_chunks: list[EvidenceChunkRef] = Field(
        default_factory=list,
        description="Chunks retrieved and reviewed but rejected (missing coverage only).",
    )
    chunk_ids: list[int] = Field(default_factory=list)
    note: str = ""
    reason: str = ""


class StructuredRequirements(BaseModel):
    """Full JD requirement decomposition for per-requirement RAG."""

    role_centre: str = ""
    primary_requirements: list[PrimaryRequirement] = Field(default_factory=list)
    secondary_requirements: list[PrimaryRequirement] = Field(default_factory=list)


class RequirementCoverageReport(BaseModel):
    """All requirements for one job analyse pass."""

    role_centre: str = ""
    requirement_coverage: list[RequirementCoverageItem] = Field(default_factory=list)
    direct_evidence: list[str] = Field(default_factory=list)
    adjacent_evidence: list[str] = Field(default_factory=list)
    missing_or_risk: list[str] = Field(default_factory=list)

    def to_legacy_rows(self) -> list[dict[str, Any]]:
        """Serialize for analysis_cache / RequirementEvidence compatibility."""
        from scout.rag.evidence import coverage_to_requirement_evidence

        return [coverage_to_requirement_evidence(item).model_dump() for item in self.requirement_coverage]


@dataclass
class RagEvalCase:
    """Golden case for rag-eval."""

    case: str
    title: str
    jd_excerpt: str
    role_centre: str = ""
    expected_requirements: list[str] = field(default_factory=list)
    expected_strong_themes: list[str] = field(default_factory=list)
    expected_missing: list[str] = field(default_factory=list)
    expected_adjacent: list[str] = field(default_factory=list)
    expected_decision_support: str = ""
