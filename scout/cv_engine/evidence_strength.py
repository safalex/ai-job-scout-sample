"""Classify CV evidence strength per requirement — profile-first, role-agnostic."""

from __future__ import annotations

import re
from typing import Literal

from scout.config.profile_policy import ProfilePolicy
from scout.llm.schemas import RequirementEvidence

EvidenceLevel = Literal["strong", "adjacent", "weak", "missing"]

# Literal proof for specialist / hard must-haves only (Ceph, JVM connectors, etc.).
# Not used for role-centre primaries (leadership, platform, stack) — those use profile tiers
# + token overlap in scout.rag.evidence.classify_chunk_coverage.
SPECIALIST_STRONG_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "ceph": [re.compile(r"\bceph\b", re.I)],
    "block storage": [re.compile(r"\b(block storage|distributed block)\b", re.I)],
    "object storage": [re.compile(r"\bobject storage\b", re.I)],
    "filesystem internals": [
        re.compile(r"\b(zfs|btrfs|ext4|erofs|bcachefs|filesystem internals?)\b", re.I)
    ],
    "network storage cluster design": [
        re.compile(r"\b(network storage|storage cluster)\b", re.I)
    ],
    "storage cluster operations": [re.compile(r"\bstorage cluster\b", re.I)],
    "stateful workload migration": [
        re.compile(r"\b(live migration|stateful workloads?)\b", re.I)
    ],
    "production ceph clusters": [re.compile(r"\bceph\b", re.I)],
    "distributed block storage": [
        re.compile(r"\b(distributed block|block storage infrastructure)\b", re.I)
    ],
    "llm training platform": [
        re.compile(r"\b(training platform|llm training|distributed training)\b", re.I)
    ],
    "distributed training": [re.compile(r"\bdistributed training\b", re.I)],
    "gpu scheduling": [re.compile(r"\b(gpu scheduling|cuda|nccl)\b", re.I)],
    "ml infrastructure operations": [
        re.compile(r"\b(ml infrastructure|training at scale)\b", re.I)
    ],
    "kubernetes production ownership": [
        re.compile(r"\b(kubernetes|k8s).{0,40}(production|owner|operate)\b", re.I)
    ],
    "terraform": [re.compile(r"\bterraform\b", re.I)],
    "on-call": [re.compile(r"\b(on[- ]call|pagerduty)\b", re.I)],
    "incident response": [re.compile(r"\b(incident response|incident management)\b", re.I)],
    "payment integrations": [
        re.compile(r"\b(payment|stripe|adyen|checkout|reconciliation|webhook)\b", re.I)
    ],
    "external apis": [re.compile(r"\b(api integration|external api|partner api)\b", re.I)],
    "github actions": [
        re.compile(r"\b(github actions|github[- ]based|github automation)\b", re.I),
    ],
    "ci/cd": [
        re.compile(
            r"\b(ci/?cd|continuous integration|deployment pipeline|release pipeline)\b",
            re.I,
        ),
    ],
    "ci/cd pipelines": [
        re.compile(r"\b(ci/?cd|continuous integration|deployment pipeline)\b", re.I),
    ],
    "internal platform": [
        re.compile(r"\b(internal developer platform|internal platform)\b", re.I),
    ],
    "internal developer tooling": [
        re.compile(
            r"\b(internal developer platform|developer tooling|internal platform)\b",
            re.I,
        ),
    ],
    "release": [
        re.compile(r"\b(release workflow|release platform|release coordination)\b", re.I),
    ],
    "release workflows": [
        re.compile(r"\b(release workflow|release platform|release coordination)\b", re.I),
    ],
    "backend apis around ai outputs": [
        re.compile(r"\b(ai api|inference api|llm api)\b", re.I)
    ],
    "spark/flink/kafka connect/beam internals": [
        re.compile(r"\b(apache spark|apache flink|kafka connect|apache beam)\b", re.I),
        re.compile(r"\b(spark|flink).{0,30}\b(connector|internals?)\b", re.I),
    ],
    "big-data connector/sink/source development": [
        re.compile(r"\b(kafka connect|connector).{0,40}\b(sink|source)\b", re.I),
        re.compile(r"\b(big data|spark|flink).{0,40}\bconnector\b", re.I),
    ],
    "java/jvm proficiency": [
        re.compile(r"\b(java|jvm|kotlin).{0,40}\b(production|owner|primary)\b", re.I),
    ],
    "java/jvm production ownership": [
        re.compile(r"\b(java|jvm).{0,40}\b(production|owner|primary)\b", re.I),
    ],
    "jvm memory/gc tuning": [
        re.compile(r"\b(gc tuning|garbage collection|jvm memory|java memory)\b", re.I),
    ],
    "java concurrency": [
        re.compile(r"\b(java concurrency|concurrent java|java threads)\b", re.I),
    ],
    "jdbc/driver-level engineering": [
        re.compile(r"\b(jdbc driver|jdbc).{0,30}\b(driver|connector)\b", re.I),
    ],
}

# Back-compat alias (docs/tests may reference the old name).
STRONG_PATTERNS = SPECIALIST_STRONG_PATTERNS

# Adjacent evidence must NOT satisfy specialist must-haves
ADJACENT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "storage_infrastructure": [
        re.compile(
            r"\b(distributed systems|microservices|platform engineer|backend|"
            r"internal developer platform|kubernetes|cloud)\b",
            re.I,
        ),
    ],
    "ml_infrastructure": [
        re.compile(
            r"\b(ai workflow|applied ai|llm workflow|backend api|agentic|litellm)\b",
            re.I,
        ),
    ],
    "llm_training_platform": [
        re.compile(r"\b(ai workflow|applied ai|llm workflow|backend api)\b", re.I),
    ],
    "devops_sre_primary": [
        re.compile(r"\b(docker|ci/?cd|deploy)\b", re.I),
    ],
    "payments_integrations": [
        re.compile(r"\b(rest api|graphql|integration|event[- ]driven)\b", re.I),
    ],
    "applied_ai_backend": [
        re.compile(r"\b(backend api|python|fastapi|workflow)\b", re.I),
    ],
    "jvm_data_infrastructure": [
        re.compile(
            r"\b(backend|platform|data integration|etl|postgresql|mysql|api|"
            r"event[- ]driven|python|php|integration)\b",
            re.I,
        ),
    ],
    "big_data_connectors": [
        re.compile(
            r"\b(backend|data integration|etl|sql|api|integration|postgresql)\b",
            re.I,
        ),
    ],
}

DEFAULT_ADJACENT = [
    re.compile(
        r"\b(backend|platform|distributed|api|microservice|software engineer)\b",
        re.I,
    ),
]


def _norm_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.lower().strip())


def _label_matches_specialist_key(label: str, pat_key: str) -> bool:
    """Word-boundary match — avoid 'platform' in label matching 'platform engineering'."""
    lab = _norm_label(label)
    key = _norm_label(pat_key)
    if lab == key:
        return True
    return bool(re.search(rf"\b{re.escape(key)}\b", lab))


def _requirement_relates_to_profile_skill(requirement: str, skill_entry: str) -> bool:
    req = _norm_label(requirement)
    ent = skill_entry.lower().strip()
    if not ent:
        return False
    if ent in req or re.search(rf"\b{re.escape(ent)}\b", req):
        return True
    if ent == "litellm" and re.search(r"\b(llm|litellm|rag|agentic)\b", req):
        return True
    return False


def _profile_strong_skills(label: str, cv_blob: str, policy: ProfilePolicy) -> bool:
    """Role-agnostic: profile skill_confidence tiers for this candidate."""
    key = _norm_label(label)
    exp = re.match(r"experience with (.+)", key)
    if exp:
        skill = exp.group(1).strip()
        tier = policy.classify_skill(skill, cv_text=cv_blob)
        if tier in ("strongest", "strong_recent"):
            return True
    for token in re.findall(r"[a-z0-9+#.]{2,}", key):
        tier = policy.classify_skill(token, cv_text=cv_blob)
        if tier in ("strongest", "strong_recent"):
            return True
    tier = policy.classify_skill(label, cv_text=cv_blob)
    if tier in ("strongest", "strong_recent"):
        return True
    conf = policy.skill_confidence
    for tier_name in ("strongest", "strong_recent"):
        for entry in conf.get(tier_name) or []:
            if not _requirement_relates_to_profile_skill(key, str(entry)):
                continue
            if policy.classify_skill(str(entry), cv_text=cv_blob) in (
                "strongest",
                "strong_recent",
            ):
                return True
    return False


def _specialist_literal_strong(
    label: str,
    cv_blob: str,
    *,
    specialist_domain: str | None,
) -> bool:
    """Literal JD/CV terms for niche domains — only when a specialist lane is active."""
    if not specialist_domain:
        return False
    key = _norm_label(label)
    for pat_key, patterns in SPECIALIST_STRONG_PATTERNS.items():
        if not _label_matches_specialist_key(key, pat_key):
            continue
        if any(p.search(cv_blob) for p in patterns):
            return True
    return False


def _strong_for_label(
    label: str,
    cv_blob: str,
    policy: ProfilePolicy,
    *,
    specialist_domain: str | None = None,
) -> bool:
    if _profile_strong_skills(label, cv_blob, policy):
        return True
    if _specialist_literal_strong(label, cv_blob, specialist_domain=specialist_domain):
        return True
    return False


def _adjacent_for_domain(domain: str, cv_blob: str) -> bool:
    patterns = ADJACENT_PATTERNS.get(domain, DEFAULT_ADJACENT)
    return any(p.search(cv_blob) for p in patterns)


def classify_requirement_evidence(
    requirement: str,
    cv_blob: str,
    policy: ProfilePolicy,
    *,
    specialist_domain: str | None = None,
) -> EvidenceLevel:
    if _strong_for_label(requirement, cv_blob, policy, specialist_domain=specialist_domain):
        return "strong"
    tier = policy.classify_skill(requirement, cv_text=cv_blob)
    if tier == "stretch":
        return "weak"
    if tier in ("avoid_overclaiming", "missing_or_avoid_overclaiming"):
        return "missing"
    if specialist_domain and _adjacent_for_domain(specialist_domain, cv_blob):
        return "adjacent"
    if any(p.search(cv_blob) for p in DEFAULT_ADJACENT):
        return "adjacent"
    return "missing"


def evaluate_required_evidence(
    required_labels: list[str],
    cv_blob: str,
    policy: ProfilePolicy,
    *,
    specialist_domain: str | None = None,
) -> list[RequirementEvidence]:
    rows: list[RequirementEvidence] = []
    for label in required_labels:
        level = classify_requirement_evidence(
            label, cv_blob, policy, specialist_domain=specialist_domain
        )
        coverage = {
            "strong": "strong_evidence",
            "adjacent": "adjacent_evidence",
            "weak": "weak_evidence",
            "missing": "missing_evidence",
        }[level]
        note = ""
        if level == "adjacent":
            note = "Adjacent — does not satisfy specialist must-have"
        rows.append(RequirementEvidence(requirement=label, coverage=coverage, note=note))
    return rows


def direct_evidence_satisfied(coverage: list[RequirementEvidence]) -> bool:
    """True when every required item has strong evidence (adjacent does not count)."""
    if not coverage:
        return True
    return all(r.coverage == "strong_evidence" for r in coverage)


def primary_direct_evidence_missing(coverage: list[RequirementEvidence]) -> bool:
    """True when specialist must-haves are not all strong (adjacent does not count)."""
    if not coverage:
        return False
    return not direct_evidence_satisfied(coverage)
