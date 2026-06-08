#!/usr/bin/env bash
# Publish gist: technical writing + product screenshots (no source dump).
# Code lives in the public GitHub repo — link from GIST_README.md.
#
# Usage:
#   ./scripts/publish-application-gist.sh              # create new public gist
#   ./scripts/publish-application-gist.sh GIST_ID      # update existing gist
#
# Optional env:
#   CODE_REPO_URL=https://github.com/you/ai-job-scout-sample
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GIST_ID="${1:-}"
CODE_REPO_URL="${CODE_REPO_URL:-REPO_URL}"
DESC="AI Job Scout — requirement-level RAG (writing + product screenshots)"

python3 - "$ROOT" "$GIST_ID" "$DESC" "$CODE_REPO_URL" <<'PY'
import base64
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
gist_id = sys.argv[2]
desc = sys.argv[3]
repo_url = sys.argv[4]

writing = (root / "docs/REQUIREMENT_LEVEL_RAG_FOR_JOB_FIT.md").read_text(encoding="utf-8")
screenshots = [
    ("01-application-profile-wizard.png", "Layer 1 profile wizard (local app)"),
    ("02-job-analysis-skip-with-evidence.png", "Job analyse: skip decision with CV citations"),
]

readme = f"""# AI Job Scout — gist overview

This gist is a **supporting bundle** for job applications: technical writing plus UI screenshots of the local product. **Source code** is in the public repo (not duplicated here — gists are poor for multi-file code trees).

## Code sample (clone & run)

**{repo_url}**

```bash
pip install -e ".[dev]"
pytest -q
python scout/sample/demo.py
```

## Contents of this gist

| File | Description |
|------|-------------|
| `REQUIREMENT_LEVEL_RAG_FOR_JOB_FIT.md` | Technical writing: requirement-level evidence design |
| `01-application-profile-wizard.png` | Profile onboarding — structured candidate evidence |
| `02-job-analysis-skip-with-evidence.png` | Analyse view — skip with grounded citations |

## Screenshots

### Application profile (Layer 1 wizard)

![Application profile wizard](01-application-profile-wizard.png)

Structured profile inputs: CV variants, skills summary, target roles, geo/eligibility, compensation policy — used by deterministic gates before RAG.

### Job analysis — skip with evidence

![Job analysis with evidence citations](02-job-analysis-skip-with-evidence.png)

Example **skip** outcome for an unrelated role (SRE/on-call + HealthTech vs platform CV). Shows match assessment, risks/gaps, and per-citation evidence — the product withholding an apply nudge when specialist proof is missing.

---

See also: `{repo_url}` README and `examples/sample_output.txt`.
"""

files: dict[str, dict] = {
    "GIST_README.md": {"content": readme},
    "REQUIREMENT_LEVEL_RAG_FOR_JOB_FIT.md": {"content": writing},
}

shot_dir = root / "docs/screenshots"
for fname, _ in screenshots:
    raw = (shot_dir / fname).read_bytes()
    # Gist stores binary as base64 in content with encoding field
    files[fname] = {
        "content": base64.b64encode(raw).decode("ascii"),
        "encoding": "base64",
    }


def gist_file_names(gid: str) -> list[str]:
    out = subprocess.run(["gh", "api", f"gists/{gid}"], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    return list(data.get("files", {}).keys())


def patch_batches(gid: str, payload_files: dict, batch_size: int = 6) -> None:
    items = list(payload_files.items())
    for i in range(0, len(items), batch_size):
        body = {"files": dict(items[i : i + batch_size])}
        if i == 0:
            body["description"] = desc
        subprocess.run(
            ["gh", "api", f"gists/{gid}", "-X", "PATCH", "--input", "-"],
            input=json.dumps(body),
            text=True,
            check=True,
            capture_output=True,
        )


if gist_id:
    while True:
        existing = gist_file_names(gist_id)
        if not existing:
            break
        batch = {n: None for n in existing[:15]}
        out = subprocess.run(
            ["gh", "api", f"gists/{gist_id}", "-X", "PATCH", "--input", "-"],
            input=json.dumps({"files": batch}),
            text=True,
            capture_output=True,
        )
        if out.returncode != 0:
            print(out.stderr or out.stdout, file=sys.stderr)
            sys.exit(out.returncode)
    patch_batches(gist_id, files)
    r = subprocess.run(["gh", "api", f"gists/{gist_id}"], capture_output=True, text=True, check=True)
    print(json.loads(r.stdout)["html_url"])
else:
    body = {"description": desc, "public": True, "files": files}
    out = subprocess.run(
        ["gh", "api", "gists", "-X", "POST", "--input", "-"],
        input=json.dumps(body),
        text=True,
        capture_output=True,
        check=True,
    )
    print(json.loads(out.stdout)["html_url"])
PY
