"""End-to-end tests for requirement-level RAG and coverage classification."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scout.domain.models import JobRecord
from scout.llm.schemas import JDRequirementsExtraction
from scout.rag.evidence import classify_chunk_coverage
from scout.rag.models import PrimaryRequirement, RankedChunk
from scout.rag.pipeline import retrieve_for_requirements_v2

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def _load_profile() -> dict:
    with (FIXTURES / "profile_demo.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_storage_job_does_not_strong_match_ceph():
    profile = _load_profile()
    job = JobRecord(
        id=1,
        fingerprint="demo-storage",
        company="Acme Cloud",
        title="Staff Platform Engineer — Storage",
        location="Remote",
        remote_type="remote",
        source="demo",
        external_id="1",
        url="",
        description_text=(
            "Own distributed block storage and Ceph clusters. "
            "Filesystem internals and petabyte-scale stateful workloads."
        ),
        salary_text="",
        status="new",
    )
    extraction = JDRequirementsExtraction(
        must_have_requirements=["Ceph production clusters", "distributed block storage"],
        primary_stack=["Python"],
    )

    _evidence, legacy, report = retrieve_for_requirements_v2(
        job,
        extraction,
        profile,
        role_centre="storage_infrastructure",
    )

    ceph_rows = [r for r in report.requirement_coverage if "ceph" in r.requirement.lower()]
    assert ceph_rows, "expected a Ceph requirement row"
    assert ceph_rows[0].coverage == "missing"
    assert ceph_rows[0].evidence_chunks == []
    assert len(ceph_rows[0].inspected_chunks) >= 1
    assert all(not c.used_as_evidence for c in ceph_rows[0].inspected_chunks)
    assert report.direct_evidence == [] or "Ceph" not in " ".join(report.direct_evidence)


def test_ecommerce_cv_not_strong_on_ai_platform_requirement():
    profile = {
        "candidate": {"cv_skills_summary": "Senior Ecommerce / Backend Platform Engineer"},
        "skill_confidence": {"strongest": ["PHP", "Laravel"]},
    }
    chunks = [
        RankedChunk(
            chunk_id=1,
            document_id=1,
            content=(
                "Senior Ecommerce / Backend Platform Engineer building integrations "
                "and Magento storefronts"
            ),
            path="fixtures/cv/demo_ecommerce_engineer.pdf",
            hybrid_score=0.8,
        )
    ]
    req = PrimaryRequirement(
        id="req_004",
        name="backend/platform around AI outputs",
        importance="primary",
        specialist_domain="applied_ai_backend",
    )
    item = classify_chunk_coverage(req, chunks, profile)
    assert item.coverage in ("adjacent", "missing", "weak")


def test_llm_workflows_strong_when_profile_lists_litellm():
    from scout.config.profile_policy import ProfilePolicy
    from scout.cv_engine.evidence_strength import classify_requirement_evidence

    profile = {
        "candidate": {"cv_skills_summary": "LiteLLM RAG agentic workflows"},
        "skill_confidence": {"strongest": ["LiteLLM", "RAG"]},
    }
    policy = ProfilePolicy.from_profile(profile)
    level = classify_requirement_evidence(
        "LLM workflows",
        "built litellm routing for internal tools",
        policy,
        specialist_domain="applied_ai_backend",
    )
    assert level == "strong"


def test_eval_case_expectations():
    case = json.loads((ROOT / "evals" / "rag_cases" / "acme_storage.json").read_text())
    profile = _load_profile()
    job = JobRecord(
        id=1,
        fingerprint="eval",
        company="Acme Cloud",
        title=case["title"],
        location="Remote",
        remote_type="remote",
        source="demo",
        external_id="1",
        url="",
        description_text=case["jd_excerpt"],
        salary_text="",
        status="new",
    )
    extraction = JDRequirementsExtraction(
        must_have_requirements=case["expected_missing"][:2],
    )
    _evidence, _legacy, report = retrieve_for_requirements_v2(
        job,
        extraction,
        profile,
        role_centre=case["role_centre"],
    )
    missing_hits = sum(
        1
        for r in report.requirement_coverage
        if r.coverage in ("missing", "weak")
        and any(m.lower() in r.requirement.lower() for m in case["expected_missing"])
    )
    assert missing_hits >= 1
