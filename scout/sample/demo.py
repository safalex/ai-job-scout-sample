#!/usr/bin/env python3
"""Run requirement-level RAG on a fictional job and print coverage (no API keys)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scout.domain.models import JobRecord
from scout.llm.schemas import JDRequirementsExtraction
from scout.rag.models import RequirementCoverageItem
from scout.rag.pipeline import retrieve_for_requirements_v2


def _format_chunk_line(row: RequirementCoverageItem) -> str | None:
    """CLI label: evidence/citations vs inspected-but-rejected chunks."""
    if row.coverage == "missing":
        chunks = row.inspected_chunks
        if not chunks:
            return None
        refs = ", ".join(f"chunk#{c.chunk_id}" for c in chunks[:3])
        return f"  inspected chunks (retrieved but insufficient): {refs}"
    chunks = row.evidence_chunks
    if not chunks:
        return None
    refs = ", ".join(f"chunk#{c.chunk_id}" for c in chunks[:3])
    label = "evidence" if row.coverage == "strong" else "citations"
    return f"  {label}: {refs}"


def _load_profile() -> dict:
    with (ROOT / "fixtures" / "profile_demo.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _storage_job() -> JobRecord:
    return JobRecord(
        id=1,
        fingerprint="demo-storage-001",
        company="Acme Cloud",
        title="Staff Platform Engineer — Storage",
        location="Remote",
        remote_type="remote",
        source="demo",
        external_id="1",
        url="",
        description_text=(
            "Own distributed block storage and Ceph clusters. "
            "Filesystem internals, network storage cluster design, "
            "petabyte-scale stateful workloads. "
            "Platform engineering and distributed systems background required."
        ),
        salary_text="",
        status="new",
    )


def main() -> int:
    profile = _load_profile()
    job = _storage_job()
    extraction = JDRequirementsExtraction(
        must_have_requirements=["Ceph production clusters", "distributed block storage"],
        primary_stack=["Python"],
        nice_to_have=["Kubernetes"],
    )

    _evidence, _legacy, report = retrieve_for_requirements_v2(
        job,
        extraction,
        profile,
        role_centre="storage_infrastructure",
    )

    print("=== Requirement-level RAG demo ===")
    print(f"Job: {job.title} @ {job.company}")
    print(f"Role centre: {report.role_centre}")
    print()
    for row in report.requirement_coverage:
        print(f"- {row.requirement}")
        print(f"  coverage: {row.coverage}")
        print(f"  reason: {row.note}")
        chunk_line = _format_chunk_line(row)
        if chunk_line:
            print(chunk_line)
        print()

    print("Summary buckets:")
    print("  direct:", report.direct_evidence)
    print("  adjacent:", report.adjacent_evidence)
    print("  missing/risk:", report.missing_or_risk)
    print()
    print(json.dumps(report.model_dump(), indent=2)[:1200], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
