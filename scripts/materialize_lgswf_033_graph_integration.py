#!/usr/bin/env python3
"""Copy LGSWF-033 integration files from payloads, not the merge-target tree."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PAIRS = (
    (
        "scripts/lgswf_payloads/semantic_work_graph_integration.py",
        "ipfs_accelerate_py/agent_supervisor/planning/semantic_work_graph_integration.py",
    ),
    (
        "scripts/lgswf_payloads/test_agent_supervisor_semantic_work_graph_acceptance.py",
        "test/api/test_agent_supervisor_semantic_work_graph_acceptance.py",
    ),
)


def apply_033(dest: Path) -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    copied = []
    for src_rel, dst_rel in PAIRS:
        source = src_root / src_rel
        target = dest / dst_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(dst_rel)
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *copied],
        cwd=dest,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "copied": copied,
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(apply_033(Path.cwd()), indent=2, sort_keys=True))
