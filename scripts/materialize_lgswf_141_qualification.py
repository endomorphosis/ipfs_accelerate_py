#!/usr/bin/env python3
"""Stage the independent LGSWF-141 qualification review artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

dest = Path.cwd()
outs = [
    "docs/architecture/LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_QUALIFICATION.md",
    "data/agent_supervisor/logic_governed_semantic_work_fabric/release/qualification-decision.json",
]
for rel in outs:
    path = dest / rel
    if not path.is_file():
        raise SystemExit(f"missing review artifact: {rel}")
decision = json.loads((dest / outs[1]).read_text(encoding="utf-8"))
if decision.get("schema") != "lgswf/qualification-decision@1":
    raise SystemExit("decision schema rejected")
add = subprocess.run(
    ["git", "--literal-pathspecs", "add", "--force", "--", *outs],
    cwd=dest,
    text=True,
    capture_output=True,
)
print(
    json.dumps(
        {
            "staged": add.returncode == 0,
            "level": decision.get("level"),
            "continuous_operation": decision.get("continuous_operation"),
        }
    )
)

