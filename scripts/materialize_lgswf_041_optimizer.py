#!/usr/bin/env python3
from pathlib import Path
import json, subprocess

PAIRS = (
    ("scripts/lgswf_payloads/frontier_optimizer.py",
     "ipfs_accelerate_py/agent_supervisor/planning/frontier_optimizer.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_frontier_optimizer.py",
     "test/api/test_agent_supervisor_frontier_optimizer.py"),
)

def main() -> None:
    src = Path(__file__).resolve().parents[1]
    dest = Path.cwd()
    copied = []
    for a, b in PAIRS:
        target = dest / b
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((src / a).read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(b)
    add = subprocess.run(["git", "--literal-pathspecs", "add", "--force", "--", *copied], cwd=dest, text=True, capture_output=True)
    print(json.dumps({"copied": copied, "staged": add.returncode == 0, "stage_stderr": (add.stderr or "")[-500:]}, indent=2))

if __name__ == "__main__":
    main()
