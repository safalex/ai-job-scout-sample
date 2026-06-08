"""Minimal role-centre primaries for the public sample (no full classifier)."""

from __future__ import annotations

from dataclasses import dataclass

_CENTRE_PRIMARIES: dict[str, list[str]] = {
    "storage_infrastructure": [
        "Ceph production clusters",
        "distributed block storage",
        "filesystem internals",
        "network storage cluster design",
    ],
    "applied_ai_backend": [
        "AI engineering leadership",
        "LLM workflow development",
        "model routing traceability",
        "production reliability",
    ],
    "jvm_data_infrastructure": [
        "JVM performance engineering",
        "Kafka Connect integrations",
        "Spark data pipelines",
    ],
}

_SPECIALIST_CENTRES = frozenset(
    {
        "storage_infrastructure",
        "ml_infrastructure",
        "jvm_data_infrastructure",
        "llm_training_platform",
    }
)


@dataclass(frozen=True)
class RoleCentreClassification:
    role_centre: str
    specialist_domain_detected: bool
    specialist_domain: str | None
    primary_requirements: list[str]


def classify_role_centre(_job) -> RoleCentreClassification:
    """Sample stub — use explicit role_centre in fixtures instead."""
    return RoleCentreClassification(
        role_centre="other",
        specialist_domain_detected=False,
        specialist_domain=None,
        primary_requirements=[],
    )


def centre_for(role_centre: str) -> RoleCentreClassification:
    primaries = _CENTRE_PRIMARIES.get(role_centre, [])
    specialist = role_centre in _SPECIALIST_CENTRES
    return RoleCentreClassification(
        role_centre=role_centre,
        specialist_domain_detected=specialist,
        specialist_domain=role_centre if specialist else None,
        primary_requirements=list(primaries),
    )
