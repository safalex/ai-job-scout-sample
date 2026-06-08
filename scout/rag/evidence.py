"""
Evidence classification and coverage reports.

Teaches *evidence grounding*:
- strong = directly proves requirement
- adjacent = transferable, NOT sufficient for specialist must-haves
- weak / missing = do not invent evidence
"""

from __future__ import annotations

import re

from scout.config.profile_policy import ProfilePolicy
from scout.cv_engine import evidence_strength
from scout.llm.schemas import RequirementEvidence
from scout.rag.models import (
    EvidenceChunkRef,
    PrimaryRequirement,
    RankedChunk,
    RequirementCoverageItem,
    RequirementCoverageReport,
)

_COVERAGE_TO_LEGACY = {
    "strong": "strong_evidence",
    "adjacent": "adjacent_evidence",
    "weak": "weak_evidence",
    "missing": "missing_evidence",
}

_LEGACY_TO_COVERAGE = {v: k for k, v in _COVERAGE_TO_LEGACY.items()}


def _excerpt(text: str, limit: int = 220) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _chunks_are_skills_only(chunks: list[RankedChunk]) -> bool:
    """Skills-section or keyword-list chunks are weak proof of delivery depth."""
    if not chunks:
        return False
    checked = chunks[:3]
    skill_like = 0
    for c in checked:
        ctype = (c.metadata.chunk_type if c.metadata else "") or ""
        section = (c.metadata.section if c.metadata else "") or ""
        head = c.content[:120].lower()
        if ctype in {"skills", "header"}:
            skill_like += 1
            continue
        if re.search(r"^(technical skills|skills|competencies|technologies)\b", head, re.I):
            skill_like += 1
            continue
        if section and re.search(r"\bskills?\b", section, re.I) and len(c.content) < 400:
            skill_like += 1
    return skill_like == len(checked)


def _downgrade_skills_only_strong(level: str, chunks: list[RankedChunk]) -> str:
    if level == "strong" and _chunks_are_skills_only(chunks):
        return "weak"
    return level


def _chunk_ref(
    chunk: RankedChunk,
    *,
    used_as_evidence: bool,
    rejection_detail: str = "",
) -> EvidenceChunkRef:
    rel = chunk.relevance
    if used_as_evidence:
        reason = f"Hybrid relevance {rel:.2f}"
    else:
        detail = rejection_detail or "does not prove this requirement"
        reason = f"Retrieved but insufficient — hybrid relevance {rel:.2f}; {detail}"
    return EvidenceChunkRef(
        chunk_id=chunk.chunk_id,
        source=_source_label(chunk),
        text_excerpt=_excerpt(chunk.content),
        reason=reason,
        used_as_evidence=used_as_evidence,
    )


def _source_label(chunk: RankedChunk) -> str:
    meta = chunk.metadata
    if meta and (meta.company or meta.role or meta.project):
        parts = [p for p in (meta.company, meta.role, meta.project) if p]
        return " / ".join(parts)
    path = chunk.path.replace("\\", "/")
    if "/" in path:
        return path.rsplit("/", 1)[-1].replace(".pdf", "")
    return path


def classify_chunk_coverage(
    req: PrimaryRequirement,
    chunks: list[RankedChunk],
    profile: dict,
    *,
    profile_summary: str = "",
) -> RequirementCoverageItem:
    """Classify one requirement from reranked chunks — no hallucination if empty."""
    policy = ProfilePolicy.from_profile(profile)
    if not chunks:
        return RequirementCoverageItem(
            requirement=req.name,
            requirement_id=req.id,
            importance=req.importance,
            coverage="missing",
            note="No CV chunks retrieved for this requirement.",
            reason="No CV chunks retrieved for this requirement.",
        )

    blob = " ".join(c.content for c in chunks[:5])
    if profile_summary:
        blob = f"{profile_summary[:800]} {blob}"

    level = evidence_strength.classify_requirement_evidence(
        req.name,
        blob,
        policy,
        specialist_domain=req.specialist_domain,
    )

    # Token overlap when profile has no tier (role-agnostic; grounded in retrieved chunks).
    has_experience_bullet = any(
        c.metadata and c.metadata.chunk_type == "experience_bullet" for c in chunks[:3]
    )
    if level in ("missing", "weak", "adjacent") and chunks:
        req_l = req.name.lower()
        req_tokens = [
            t
            for t in re.findall(r"[a-z0-9+#./-]{2,}", req_l)
            if len(t) > 2 or t in {"ai", "js", "go", "ml"}
        ]
        for short in re.findall(r"\b(ai|js|k8s|go|ml)\b", req_l):
            if short not in req_tokens:
                req_tokens.append(short)
        # Node.js / TypeScript → node, typescript
        if "node" in req_l or "node.js" in req_l:
            req_tokens.append("node")
        if "typescript" in req_l:
            req_tokens.append("typescript")
        hits = sum(1 for t in req_tokens if t in blob.lower())
        ratio = hits / max(len(req_tokens), 1)
        if ratio >= 0.55 or (len(req_tokens) <= 2 and hits == len(req_tokens)):
            level = "strong" if has_experience_bullet else "weak"
        elif ratio >= 0.25:
            level = "weak" if not has_experience_bullet else "adjacent"

    # "Experience with X" — certifications/education alone are not production proof.
    if level == "strong" and re.search(r"^experience with\b", req.name, re.I):
        if re.search(
            r"\b(certification|bootcamp|dissertation|harvardx|cs50|magento cert|"
            r"rubik|mindstorms|education\s*&\s*cert)\b",
            blob,
            re.I,
        ) and not re.search(
            r"\b(production|owned|operated|maintained|shipped|deployed|"
            r"years?.{0,24}experience)\b",
            blob,
            re.I,
        ):
            level = "weak"

    # Specialist must-haves: adjacent cannot satisfy — broad platform talk is not Ceph/JVM proof.
    if req.importance == "specialist" and level == "adjacent":
        level = "missing"

    level = _downgrade_skills_only_strong(level, chunks)

    reason = ""
    if level == "strong":
        reason = f"Direct evidence for {req.name}."
    elif level == "adjacent":
        reason = "Adjacent evidence — transferable but not exact; specialist requirements need direct proof."
    elif level == "weak":
        reason = "Weak or partial match — does not fully prove the requirement."
    else:
        reason = "Retrieved related chunks but none qualify as direct evidence."

    rejection_detail = ""
    if level == "missing" and req.importance == "specialist":
        rejection_detail = "specialist requirement needs direct proof, not adjacent platform language"
    elif level == "missing":
        rejection_detail = "chunk text does not substantiate this requirement"

    chunk_ids = [c.chunk_id for c in chunks[:3]]
    if level == "missing":
        inspected = [
            _chunk_ref(c, used_as_evidence=False, rejection_detail=rejection_detail)
            for c in chunks[:3]
        ]
        return RequirementCoverageItem(
            requirement=req.name,
            requirement_id=req.id,
            importance=req.importance,
            coverage=level,
            evidence_chunks=[],
            inspected_chunks=inspected,
            chunk_ids=chunk_ids,
            note=reason,
            reason=reason,
        )

    evidence = [_chunk_ref(c, used_as_evidence=True) for c in chunks[:3]]
    return RequirementCoverageItem(
        requirement=req.name,
        requirement_id=req.id,
        importance=req.importance,
        coverage=level,
        evidence_chunks=evidence,
        inspected_chunks=[],
        chunk_ids=chunk_ids,
        note=reason,
        reason=reason,
    )


def build_coverage_report(
    items: list[RequirementCoverageItem],
    *,
    role_centre: str = "",
) -> RequirementCoverageReport:
    """Aggregate direct / adjacent / missing lists for UI and portfolio alignment."""
    direct: list[str] = []
    adjacent: list[str] = []
    missing: list[str] = []
    for row in items:
        label = f"{row.requirement} — {row.coverage.title()}"
        if row.coverage == "strong":
            direct.append(row.requirement)
        elif row.coverage == "adjacent":
            adjacent.append(row.requirement)
        elif row.coverage in ("weak", "missing"):
            missing.append(row.requirement)
    return RequirementCoverageReport(
        role_centre=role_centre,
        requirement_coverage=items,
        direct_evidence=direct,
        adjacent_evidence=adjacent,
        missing_or_risk=missing,
    )


def coverage_to_requirement_evidence(item: RequirementCoverageItem) -> RequirementEvidence:
    """Map to legacy cache / LLM schema with grounded citations."""
    from scout.llm.schemas import EvidenceChunkRef

    legacy = _COVERAGE_TO_LEGACY.get(item.coverage, "missing_evidence")
    source_refs = item.evidence_chunks or item.inspected_chunks
    refs = [
        EvidenceChunkRef(
            chunk_id=r.chunk_id,
            source=r.source,
            text_excerpt=r.text_excerpt,
            reason=r.reason,
            used_as_evidence=r.used_as_evidence,
        )
        for r in source_refs
    ]
    return RequirementEvidence(
        requirement=item.requirement,
        requirement_id=item.requirement_id,
        importance=item.importance,
        coverage=legacy,
        chunk_ids=item.chunk_ids,
        evidence_chunks=refs,
        note=item.reason or item.note,
        reason=item.reason or item.note,
    )


