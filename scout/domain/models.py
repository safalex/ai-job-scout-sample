from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JobRecord:
    id: int | None
    fingerprint: str
    company: str
    title: str
    location: str
    remote_type: str
    source: str
    external_id: str
    url: str
    description_text: str
    salary_text: str
    status: str


@dataclass
class EvidenceChunk:
    chunk_id: int
    document_id: int
    content: str
    path: str
    score: float = 0.0
