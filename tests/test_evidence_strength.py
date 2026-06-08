"""Unit tests for evidence strength classification."""

from scout.config.profile_policy import ProfilePolicy
from scout.cv_engine.evidence_strength import (
    classify_requirement_evidence,
    direct_evidence_satisfied,
    evaluate_required_evidence,
    primary_direct_evidence_missing,
)
from scout.llm.schemas import RequirementEvidence


def _demo_policy() -> ProfilePolicy:
    return ProfilePolicy.from_profile(
        {
            "profile_name": "demo",
            "candidate": {
                "cv_skills_summary": (
                    "Senior backend engineer. Python, PostgreSQL, Docker, "
                    "LiteLLM, event-driven workers."
                ),
            },
            "skill_confidence": {
                "strongest": ["Python", "PostgreSQL"],
                "missing_or_avoid_overclaiming": ["Ceph", "block storage"],
            },
        }
    )


def test_ceph_missing_for_demo_cv():
    policy = _demo_policy()
    blob = policy.raw["candidate"]["cv_skills_summary"].lower()
    assert classify_requirement_evidence("Ceph", blob, policy) == "missing"


def test_adjacent_does_not_satisfy_must_haves():
    policy = _demo_policy()
    blob = policy.raw["candidate"]["cv_skills_summary"].lower()
    rows = evaluate_required_evidence(
        ["Ceph", "block storage"],
        blob,
        policy,
        specialist_domain="storage_infrastructure",
    )
    assert not direct_evidence_satisfied(rows)
    assert primary_direct_evidence_missing(rows)


def test_strong_when_cv_has_specialist_term():
    policy = ProfilePolicy.from_profile(
        {
            "candidate": {"cv_skills_summary": "Production Ceph clusters and ZFS operations."},
            "skill_confidence": {"strongest": ["Ceph", "ZFS"]},
        }
    )
    blob = "production ceph clusters and zfs operations"
    rows = evaluate_required_evidence(["Ceph"], blob, policy)
    assert rows[0].coverage == "strong_evidence"
    assert direct_evidence_satisfied(rows)


def test_avoid_tier_counts_as_missing_not_adjacent():
    policy = _demo_policy()
    blob = "backend platform microservices kubernetes"
    level = classify_requirement_evidence(
        "Ceph",
        blob,
        policy,
        specialist_domain="storage_infrastructure",
    )
    assert level == "missing"
    rows = [
        RequirementEvidence(requirement="Ceph", coverage="adjacent_evidence", note="x"),
    ]
    assert primary_direct_evidence_missing(rows)


def test_platform_engineering_not_strong_without_profile_or_specialist_domain():
    policy = ProfilePolicy.from_profile(
        {
            "candidate": {"cv_skills_summary": "backend platform microservices"},
            "skill_confidence": {"strongest": ["Python"]},
        }
    )
    blob = policy.raw["candidate"]["cv_skills_summary"].lower()
    assert (
        classify_requirement_evidence(
            "backend/platform around AI outputs",
            blob,
            policy,
            specialist_domain=None,
        )
        != "strong"
    )


def test_generic_backend_counts_as_adjacent_for_storage_domain():
    policy = ProfilePolicy.from_profile(
        {
            "candidate": {"cv_skills_summary": "backend platform microservices kubernetes"},
            "skill_confidence": {"strongest": ["Python"]},
        }
    )
    blob = policy.raw["candidate"]["cv_skills_summary"].lower()
    level = classify_requirement_evidence(
        "RareWidget",
        blob,
        policy,
        specialist_domain="storage_infrastructure",
    )
    assert level == "adjacent"
