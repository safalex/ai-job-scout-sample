"""
Requirement extraction and query expansion.

Teaches *requirement decomposition* and *query expansion*:
- Split JD into primary/secondary/specialist requirements (not one blob query).
- Generate 2–3 search queries per requirement using synonyms + profile vocabulary.
"""

from __future__ import annotations

import re
from typing import Any

from scout.config.profile_policy import ProfilePolicy
from scout.domain.models import JobRecord
from scout.sample.role_centres import centre_for, classify_role_centre
from scout.llm.schemas import JDRequirementsExtraction
from scout.rag.models import PrimaryRequirement, RequirementQuerySet, StructuredRequirements

# Synonym expansion for retrieval (exact terms beat vague embeddings).
_QUERY_SYNONYMS: dict[str, list[str]] = {
    "llm": ["litellm", "model routing", "structured outputs", "rag", "agentic"],
    "llmops": ["model routing", "traceability", "monitoring", "evaluation", "retry"],
    "langgraph": ["agent orchestration", "workflow graph", "state machine"],
    "ceph": ["distributed block storage", "object storage cluster"],
    "spark": ["apache spark", "flink", "kafka connect", "beam"],
    "jvm": ["java", "garbage collection", "jdbc driver", "concurrency"],
    "devex": ["developer experience", "github actions", "internal platform", "release platform"],
    "platform": ["internal developer platform", "developer tooling", "ci/cd"],
    "github": ["github actions", "workflow automation", "ci/cd"],
    "actions": ["github actions", "workflow triggers", "ci/cd"],
    "ci/cd": ["continuous integration", "deployment pipeline", "github actions"],
    "release": ["release workflow", "release platform", "environment provisioning"],
    "tooling": ["developer tooling", "internal platform", "ci/cd"],
}

# JD themes → structured requirements (eval cases + DevEx postings).
_JD_THEME_SPECS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"github\s*actions?", re.I), "GitHub Actions", "GitHub Actions workflow automation"),
    (re.compile(r"\bci/?cd\b", re.I), "CI/CD pipelines", "CI/CD and deployment automation"),
    (
        re.compile(r"\brelease\s+(workflow|platform|engineering)\b", re.I),
        "release workflows",
        "Release workflow and environment provisioning",
    ),
    (
        re.compile(r"\binternal\s+(developer\s+)?platform\b", re.I),
        "internal developer platform",
        "Internal developer platform ownership",
    ),
    (
        re.compile(r"\b(discovery workshops|customer[- ]facing|client[- ]facing)\b", re.I),
        "customer-facing discovery",
        "Stakeholder workshops and client-facing technical leadership",
    ),
    (
        re.compile(r"\b(forward deployed|ai solution lifecycle|prototype to production)\b", re.I),
        "AI solution delivery lifecycle",
        "Prototype-to-production AI solution delivery for external clients",
    ),
]

_BROAD_TERMS = frozenset(
    {
        "platform engineering",
        "distributed systems",
        "backend systems",
        "data integration",
        "ai",
        "machine learning",
    }
)


def _req_id(index: int) -> str:
    return f"req_{index:03d}"


def _importance_from_text(req: str, *, specialist: bool) -> str:
    if specialist:
        return "specialist"
    if re.search(r"\b(must|required|mandatory|essential)\b", req, re.I):
        return "hard"
    return "primary"


def build_structured_requirements(
    job: JobRecord,
    extraction: JDRequirementsExtraction | None,
    profile: dict,
    *,
    role_centre: str | None = None,
) -> StructuredRequirements:
    """
    Merge LLM JD extraction + role-centre primary requirements into structured rows.

    Location/eligibility stay outside RAG (handled by scout.filtering).
    """
    if role_centre:
        centre = role_centre
        rc = centre_for(role_centre)
    else:
        rc = classify_role_centre(job)
        centre = rc.role_centre
    primary: list[PrimaryRequirement] = []
    secondary: list[PrimaryRequirement] = []
    idx = 1

    def add_req(
        name: str,
        *,
        req_type: str = "skill",
        importance: str = "primary",
        evidence: str = "",
        specialist_domain: str | None = None,
    ) -> None:
        nonlocal idx
        rid = _req_id(idx)
        idx += 1
        row = PrimaryRequirement(
            id=rid,
            name=name.strip(),
            type=req_type,
            importance=importance,
            evidence_needed=evidence or f"Direct evidence for: {name}",
            specialist_domain=specialist_domain,
        )
        if importance in ("primary", "hard", "specialist"):
            primary.append(row)
        else:
            secondary.append(row)

    for label in rc.primary_requirements or []:
        imp = "specialist" if rc.specialist_domain_detected else "primary"
        add_req(
            label,
            importance=imp,
            specialist_domain=rc.specialist_domain if rc.specialist_domain_detected else None,
            evidence=label,
        )

    if extraction:
        for req in extraction.must_have_requirements or []:
            if any(p.name.lower() == req.lower() for p in primary):
                continue
            spec = rc.specialist_domain if rc.specialist_domain_detected else None
            imp = _importance_from_text(req, specialist=bool(spec))
            add_req(req, importance=imp, specialist_domain=spec)
        for stack in extraction.primary_stack or []:
            add_req(
                f"Experience with {stack}",
                req_type="stack",
                importance="primary",
            )
        for nice in extraction.nice_to_have or []:
            add_req(nice, importance="secondary")

    if not primary and extraction and extraction.primary_stack:
        for stack in extraction.primary_stack[:6]:
            add_req(f"Experience with {stack}", req_type="stack")

    jd_blob = f"{job.title} {job.description_text[:12000]}"
    for pattern, name, evidence_hint in _JD_THEME_SPECS:
        if not pattern.search(jd_blob):
            continue
        if any(p.name.lower() == name.lower() for p in primary + secondary):
            continue
        add_req(name, importance="primary", evidence=evidence_hint)

    if not primary:
        add_req(
            job.title,
            req_type="delivery",
            importance="primary",
            evidence="Role-relevant production experience",
        )

    return StructuredRequirements(
        role_centre=centre,
        primary_requirements=primary[:12],
        secondary_requirements=secondary[:6],
    )


def expand_queries_for_requirement(
    req: PrimaryRequirement,
    profile: dict,
    *,
    job_title: str = "",
    job_description: str = "",
) -> RequirementQuerySet:
    """
    Query expansion: exact JD terms + synonyms + profile vocabulary.

    Multiple queries per requirement improve recall without relying on one embedding.
    """
    policy = ProfilePolicy.from_profile(profile)
    name_l = req.name.lower()
    queries: list[str] = []

    # 1) Literal requirement + title context
    queries.append(f"{job_title} {req.name} {req.evidence_needed}".strip())

    # 2) Synonym expansion
    extra_terms: list[str] = []
    for key, syns in _QUERY_SYNONYMS.items():
        if key in name_l:
            extra_terms.extend(syns[:4])
    for term in re.findall(r"[a-z0-9+#.]{3,}", name_l):
        if term in _QUERY_SYNONYMS:
            extra_terms.extend(_QUERY_SYNONYMS[term][:3])

    # 3) Profile strongest skills that overlap requirement tokens
    cv_summary = (profile.get("candidate") or {}).get("cv_skills_summary", "")
    for skill in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{2,}", cv_summary):
        if skill.lower() in name_l or any(skill.lower() in s for s in extra_terms):
            extra_terms.append(skill)

    if extra_terms:
        queries.append(f"{req.name} {' '.join(list(dict.fromkeys(extra_terms))[:8])}")

    # 3b) Pull literal JD phrases into queries (e.g. GitHub Actions in excerpt but not in req name)
    if job_description:
        jd_l = job_description.lower()
        for key, syns in _QUERY_SYNONYMS.items():
            if key in jd_l or any(s in jd_l for s in syns[:2]):
                extra_terms.extend(syns[:3])
        for pattern, name, _ in _JD_THEME_SPECS:
            if pattern.search(job_description) and name.lower() not in name_l:
                extra_terms.append(name)

    # 4) Specialist: force exact phrase quotes for keyword layer
    if req.importance == "specialist" or req.specialist_domain:
        tokens = [t for t in re.findall(r"[a-z0-9]{3,}", name_l) if t not in _BROAD_TERMS]
        if tokens:
            queries.append(" ".join(f'"{t}"' if len(t) > 4 else t for t in tokens[:5]))

    deduped = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:4]
    return RequirementQuerySet(requirement_id=req.id, queries=deduped or [req.name])
