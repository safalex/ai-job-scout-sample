# Why are you proud of this code?

I'm proud of this slice because it solves a real product problem I kept hitting: **convincing-looking job matches that fall apart on specialist requirements**.

Most matching systems stop at similarity — embed the JD, embed the CV, take the top chunks, maybe ask an LLM “is this a good fit?” That pattern is fast but brittle. It happily connects “platform engineer” evidence to “Ceph cluster operator” roles because both mention infrastructure at a high level.

This code makes the **requirement** the unit of work, not the document. Each must-have gets its own retrieval pass, rerank, and a **four-level evidence verdict** (strong / adjacent / weak / missing). Adjacent is explicit: transferable, but **not enough** to claim a specialist skill. That distinction is enforced in tests — for example, a fictional platform CV must not strong-match Ceph, and skills-list chunks are downgraded so keyword stuffing cannot fake depth.

I'm also proud of the engineering trade-offs baked in:

- **Deterministic gates before expensive AI** — hard filters and profile policy run first in the full pipeline; this sample focuses on the RAG layer but keeps the same philosophy.
- **Profile-first classification** — skill confidence tiers and “avoid overclaiming” lists prevent the system from flattering me on storage or niche domains I don't claim.
- **Structured outputs** — Pydantic models for requirements, coverage rows, and legacy-compatible evidence enums so downstream ranking and UI stay consistent.
- **Testability without production data** — in-memory fixtures and eval JSON cases let me regression-test false-positive scenarios without shipping real CVs or scraped jobs.

This is **not** a claim of massive production scale. It's a personal project I run for my own job search, extracted into a small public repo. The sample is roughly a few thousand lines with 12 tests passing on a laptop in under three seconds. What I would improve next — live LLM requirement extraction, a proper cross-encoder reranker, and a broader eval harness — is listed honestly in the README.

What matters to me is that the design **reduces harm**: fewer false “you should apply” nudges, clearer reasons when evidence is only adjacent, and code a reviewer can read without wading through SaaS boilerplate.
