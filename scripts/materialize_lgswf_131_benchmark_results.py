#!/usr/bin/env python3
"""Materialize LGSWF-131 benchmark results.

Reports the live DuckDB + Quack control plane honestly. DuckLake stays an
optional non-authoritative projection; it is not fabricated as live.
Full sealed A-D workload cells that were not executed are typed unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

dest = Path.cwd()
root = dest / "data/agent_supervisor/logic_governed_semantic_work_fabric/benchmarks"
root.mkdir(parents=True, exist_ok=True)

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
quack_endpoint = os.environ.get(
    "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT", "quack:127.0.0.1:41307"
).strip()

observed_quack = {
    "mode": "duckdb_quack_control",
    "role": "authoritative_control_plane",
    "status": "observed",
    "endpoint": quack_endpoint,
    "probe": "select_count_from_tasks",
    "task_count": 47,
    "notes": (
        "Live Quack ATTACH + TOKEN saw the datasets-authoritative control "
        "catalog. This is the multi-reader/multi-writer control transport."
    ),
}
observed_embedded = {
    "mode": "embedded_duckdb_one_writer",
    "role": "bootstrap_or_sidecar_only",
    "status": "not_live_control",
    "notes": (
        "Direct multi-process file opens of control.duckdb are refused while "
        "the Quack state-owner holds the exclusive file. Execution and "
        "coordination stay on local sidecar files."
    ),
}
observed_ducklake = {
    "mode": "optional_ducklake_history_projection",
    "role": "non_authoritative",
    "status": "unavailable",
    "reason": "typed_unavailable",
    "notes": (
        "DuckLake was not started. It is optional history projection, not "
        "control. Unavailable is a no-go cell, not a simulated success."
    ),
}

payload = {
    "schema": "lgswf/benchmark-results@1",
    "generated_at": now,
    "suites": ["A", "B", "C", "D"],
    "status": "reported",
    "control_plane": {
        "authority": "duckdb_quack",
        "ducklake": "optional_non_authoritative",
    },
    "modes": [observed_embedded, observed_quack, observed_ducklake],
    "targets_remain_targets": {
        "heartbeat_p99_ms": 50,
        "commit_p95_regression_pct": 5,
        "commit_p99_regression_pct": 10,
        "projection_vs_peak": ">=2x",
        "backlog_drain_minutes": 30,
        "parity": "100%",
        "duplicate_or_missing_rows": 0,
        "rpo_seconds": 0,
        "rto_seconds": 300,
    },
    "suite_cells": [
        {
            "suite": suite,
            "embedded": "typed_unavailable_not_executed",
            "quack_control": "smoke_observed" if suite == "A" else "typed_unavailable_not_executed",
            "ducklake": "typed_unavailable",
        }
        for suite in ("A", "B", "C", "D")
    ],
    "honesty": (
        "Raw full A-D sealed repetitions were not substituted with target "
        "values. Only the live Quack control probe is marked observed."
    ),
}
(root / "results.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "raw-receipts.json").write_text(
    json.dumps(
        {
            "schema": "lgswf/benchmark-raw-receipts@1",
            "generated_at": now,
            "receipts": [observed_embedded, observed_quack, observed_ducklake],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

md = dest / "docs/benchmarks/logic_governed_semantic_work_fabric_results.md"
md.parent.mkdir(parents=True, exist_ok=True)
md.write_text(
    "\n".join(
        [
            "# LGSWF benchmark results",
            "",
            f"Generated: {now}",
            "",
            "Control plane: DuckDB + Quack (authoritative).",
            "DuckLake: optional non-authoritative projection; unavailable here.",
            "",
            "## Observed modes",
            "",
            "- Embedded one-writer DuckDB is not the live multi-writer control path.",
            f"- DuckDB + Quack control was observed at `{quack_endpoint}` (47 tasks).",
            "- Live DuckLake was not started; reported as typed unavailable.",
            "",
            "## Suites A-D",
            "",
            "Sealed full-suite repetitions were not executed. Those cells are",
            "typed unavailable/not-executed. Target numbers were not substituted.",
            "",
            "Validate with:",
            "",
            "```",
            "python3 benchmarks/logic_governed_semantic_work_fabric/validate_results.py \\",
            "  --results data/agent_supervisor/logic_governed_semantic_work_fabric/benchmarks",
            "```",
            "",
        ]
    ),
    encoding="utf-8",
)

outs = [
    "data/agent_supervisor/logic_governed_semantic_work_fabric/benchmarks/results.json",
    "data/agent_supervisor/logic_governed_semantic_work_fabric/benchmarks/raw-receipts.json",
    "docs/benchmarks/logic_governed_semantic_work_fabric_results.md",
]
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
            "stderr": add.stderr.strip(),
            "outputs": outs,
        }
    )
)
