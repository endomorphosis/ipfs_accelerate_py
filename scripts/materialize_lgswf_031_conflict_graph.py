#!/usr/bin/env python3
"""Apply the LGSWF-031 conflict-graph extension without dirtying the merge target."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MARKER = "LGSWF_CONFLICT_SCOPES"
EXPORTS = (
    '    "LGSWF_CONFLICT_SCOPES",\n'
    '    "SemanticConflictError",\n'
    '    "admit_conflict_free_frontier",\n'
    '    "evaluate_semantic_conflict",\n'
)


def apply_031(dest: Path) -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    payload = (src_root / "scripts/lgswf_payloads/031_conflict_graph_extension.py").read_text(
        encoding="utf-8"
    )
    test_src = src_root / "scripts/lgswf_payloads/test_agent_supervisor_semantic_conflict_graph.py"
    module = dest / "ipfs_accelerate_py/agent_supervisor/core/conflict_graph.py"
    test_dst = dest / "test/api/test_agent_supervisor_semantic_conflict_graph.py"
    text = module.read_text(encoding="utf-8")
    if MARKER not in text:
        if "\n__all__ = [\n" not in text:
            raise RuntimeError("conflict_graph.py is missing __all__ for extension insert")
        text = text.replace("\n__all__ = [\n", "\n" + payload.rstrip() + "\n\n\n__all__ = [\n", 1)
        text = text.replace(
            '    "update_conflict_weights",\n]',
            '    "update_conflict_weights",\n' + EXPORTS + "]",
            1,
        )
        module.write_text(text, encoding="utf-8")
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.write_text(test_src.read_text(encoding="utf-8"), encoding="utf-8")
    outputs = [
        "ipfs_accelerate_py/agent_supervisor/core/conflict_graph.py",
        "test/api/test_agent_supervisor_semantic_conflict_graph.py",
    ]
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *outputs],
        cwd=dest,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "applied": MARKER in module.read_text(encoding="utf-8"),
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
    }


if __name__ == "__main__":
    print(json.dumps(apply_031(Path.cwd()), indent=2, sort_keys=True))
