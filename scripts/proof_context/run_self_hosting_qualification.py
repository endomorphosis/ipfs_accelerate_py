#!/usr/bin/env python3
"""Run or validate self-hosting harness evidence; never computes qualification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The runner is intentionally usable before PCCE-052 packages a console entry
# point.  It still imports the same public package surface used after install.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from ipfs_accelerate_py.agent_supervisor.self_hosting import (
    ExperimentPlan,
    SelfHostingQualificationHarness,
    is_evidence_envelope,
)
from ipfs_accelerate_py.agent_supervisor.self_hosting.harness import canonical_evidence_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", help="frozen experiment plan JSON")
    parser.add_argument("--repository", help="explicit Git repository for live/simulated plans")
    parser.add_argument("--output", help="write canonical evidence JSON")
    parser.add_argument("--check", help="validate an existing evidence file or directory containing manifest.json")
    args = parser.parse_args(argv)
    if args.check:
        target = Path(args.check)
        if target.is_dir(): target = target / "manifest.json"
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            valid = is_evidence_envelope(data)
        except (OSError, json.JSONDecodeError): valid = False
        return 0 if valid else 2
    if not args.plan:
        parser.error("--plan is required unless --check is used")
    plan = ExperimentPlan.from_mapping(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    if plan.evidence_kind != "replayed" and not args.repository:
        parser.error("--repository is required for live and simulated plans")
    evidence = SelfHostingQualificationHarness(plan, args.repository).run()
    rendered = canonical_evidence_json(evidence) + "\n"
    if args.output: Path(args.output).write_text(rendered, encoding="utf-8")
    else: sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
