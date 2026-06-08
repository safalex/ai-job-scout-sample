from __future__ import annotations

from pydantic import BaseModel, Field


class JDRequirementsExtraction(BaseModel):
    """Structured JD parse for targeted RAG and gap checks."""

    role_family: str = ""
    primary_stack: list[str] = Field(default_factory=list)
    must_have_requirements: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    location_restrictions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    summary: str = ""


class EvidenceChunkRef(BaseModel):
    chunk_id: int = 0
    source: str = ""
    text_excerpt: str = ""
    reason: str = ""
    used_as_evidence: bool = True


class RequirementEvidence(BaseModel):
    requirement: str
    coverage: str
    chunk_ids: list[int] = Field(default_factory=list)
    note: str = ""
    requirement_id: str = ""
    importance: str = ""
    evidence_chunks: list[EvidenceChunkRef] = Field(default_factory=list)
    reason: str = ""
