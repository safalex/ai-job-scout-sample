"""CV chunk type weighting — skills sections must not prove delivery depth."""

from scout.rag.evidence import _chunks_are_skills_only, _downgrade_skills_only_strong
from scout.rag.models import ChunkMetadata, RankedChunk


def _chunk(content: str, *, chunk_type: str = "paragraph", section: str = "") -> RankedChunk:
    return RankedChunk(
        chunk_id=1,
        document_id=1,
        content=content,
        path="fixtures/cv/demo.pdf",
        metadata=ChunkMetadata(
            stable_chunk_id="c1",
            chunk_type=chunk_type,
            section=section,
        ),
    )


def test_skills_only_chunks_downgraded_from_strong():
    chunks = [
        _chunk(
            "Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, LLM, RAG",
            chunk_type="skills",
            section="Skills",
        ),
        _chunk(
            "TypeScript, Node.js, AWS, Azure",
            chunk_type="skills",
            section="Technical skills",
        ),
    ]
    assert _chunks_are_skills_only(chunks) is True
    assert _downgrade_skills_only_strong("strong", chunks) == "weak"


def test_experience_bullets_not_downgraded():
    chunks = [
        _chunk(
            "Built RAG retrieval pipeline for production AI platform serving external clients.",
            chunk_type="experience_bullet",
            section="Work Experience",
        ),
    ]
    assert _chunks_are_skills_only(chunks) is False
    assert _downgrade_skills_only_strong("strong", chunks) == "strong"
