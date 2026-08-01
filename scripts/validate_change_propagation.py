#!/usr/bin/env python3
"""CLI for proof-gated change-propagation operations, metrics, flags, and rollback.

RPR-046 / RPR-G220.  Validates exact sources, capabilities, graph/index
coverage, proof reconstruction, transaction health, and benchmark floors.
Exposes doctor / status / replay plus shadow-default rollout policy inspection.

This script never grants mutation, completion, merge, or process authority.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_rollout import (  # noqa: E402
    BENCHMARK_STAGES,
    METRICS_INTERFACE,
    ROLLOUT_POLICY_INTERFACE,
    VALIDATOR_INTERFACE,
    ChangePropagationMetrics,
    ChangePropagationRolloutError,
    ChangePropagationRolloutPolicy,
    CheckStatus,
    RolloutMode,
    _plain,
    check_benchmark_floors,
    check_capability_health,
    check_exact_source_bindings,
    check_feature_flags,
    check_graph_index_coverage,
    check_guide_boundaries,
    check_plan_objective_task_dag,
    check_proof_reconstruction,
    check_rollback_gates,
    check_supervisor_process_state,
    check_transaction_health,
    collect_metrics,
    content_identity,
    default_rollout_policy,
    doctor,
    replay_decision_receipt,
    repository_root,
    run_all_checks,
    status,
)


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> None:
    """Atomically write a coordinate checkpoint when the env dir is configured."""

    raw = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", "").strip()
    if not raw:
        return
    directory = Path(raw)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}.json"
        data = json.dumps(_plain(payload), sort_keys=True, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    except OSError:
        return


def _print_json(payload: Mapping[str, Any]) -> None:
    json.dump(_plain(payload), sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/).",
    )
    common.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON.",
    )
    common.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the adversarial benchmark floor check.",
    )
    common.add_argument(
        "--skip-capabilities",
        action="store_true",
        help="Skip the capability health probe.",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Validate proof-gated change-propagation operations, metrics, "
            "feature flags, and rollback gates (RPR-046)."
        ),
        parents=[common],
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Run the full validation suite (default when no subcommand).",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser(
        "doctor",
        help="Run control-plane, coverage, transaction, and health checks.",
        parents=[common],
    )
    sub.add_parser(
        "status",
        help="Show mode, bindings, coverage, and supervisor state.",
        parents=[common],
    )
    replay_p = sub.add_parser(
        "replay",
        help="Replay a sealed plan/completion/benchmark receipt.",
        parents=[common],
    )
    replay_p.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="Path to a JSON receipt to replay.",
    )
    sub.add_parser(
        "check-dag",
        help="Check plan/objective/task DAG only.",
        parents=[common],
    )
    sub.add_parser(
        "check-bindings",
        help="Check exact source bindings only.",
        parents=[common],
    )
    sub.add_parser(
        "check-capabilities",
        help="Probe capability health only.",
        parents=[common],
    )
    sub.add_parser(
        "check-graph-index",
        help="Check graph/index coverage only.",
        parents=[common],
    )
    sub.add_parser(
        "check-proof-reconstruction",
        help="Check proof reconstruction surfaces only.",
        parents=[common],
    )
    sub.add_parser(
        "check-transaction",
        help="Check transaction health only.",
        parents=[common],
    )
    sub.add_parser(
        "check-supervisor",
        help="Inspect supervisor/process state only.",
        parents=[common],
    )
    sub.add_parser(
        "check-benchmark-floors",
        help="Run benchmark safety floors only.",
        parents=[common],
    )
    sub.add_parser(
        "check-flags",
        help="Validate feature-flag defaults.",
        parents=[common],
    )
    sub.add_parser(
        "check-rollback",
        help="Validate rollback gates.",
        parents=[common],
    )
    sub.add_parser(
        "check-guide",
        help="Validate operator guide boundaries.",
        parents=[common],
    )
    sub.add_parser(
        "metrics",
        help="Emit operator metrics from the benchmark.",
        parents=[common],
    )
    sub.add_parser(
        "policy",
        help="Emit the default (shadow) rollout policy.",
        parents=[common],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = (args.repo_root or repository_root()).resolve()
    command = args.command
    if args.check_all or command is None:
        command = "check-all"

    try:
        if command in {"check-all", "doctor"}:
            report = run_all_checks(
                root,
                run_benchmark=not args.skip_benchmark,
                probe_capabilities=not args.skip_capabilities,
            )
            report["command"] = "doctor" if command == "doctor" else "check-all"
            write_checkpoint("rpr-046-validation-report", report)
            if args.json:
                _print_json(report)
            else:
                status_word = "healthy" if report.get("valid") else "unhealthy"
                print(
                    f"{VALIDATOR_INTERFACE} command={report.get('command')} "
                    f"valid={report.get('valid')} status={status_word} "
                    f"default_mode={report.get('default_mode')} "
                    f"failed={report.get('failed')} "
                    f"report_id={report.get('report_id')}"
                )
            return 0 if report.get("valid") else 1

        if command == "status":
            report = status(root)
            write_checkpoint("rpr-046-status", report)
            if args.json:
                _print_json(report)
            else:
                print(
                    f"{VALIDATOR_INTERFACE} mode={report['mode']} "
                    f"valid={report['valid']} "
                    f"master={report['supervisor']['evidence'].get('master_status')} "
                    f"report_id={report['report_id']}"
                )
            return 0 if report.get("valid") else 1

        if command == "replay":
            receipt_path: Path = args.receipt
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            report = replay_decision_receipt(receipt)
            if args.json:
                _print_json(report)
            else:
                print(
                    f"{VALIDATOR_INTERFACE} replay valid={report['valid']} "
                    f"auto={report['automated_mutation_authorized']} "
                    f"errors={report['errors']} "
                    f"report_id={report['report_id']}"
                )
            return 0 if report.get("valid") else 1

        check_map = {
            "check-dag": lambda: check_plan_objective_task_dag(root),
            "check-bindings": lambda: check_exact_source_bindings(root),
            "check-capabilities": lambda: check_capability_health(
                root, probe=not args.skip_capabilities
            ),
            "check-graph-index": lambda: check_graph_index_coverage(root),
            "check-proof-reconstruction": lambda: check_proof_reconstruction(root),
            "check-transaction": lambda: check_transaction_health(root),
            "check-supervisor": lambda: check_supervisor_process_state(root),
            "check-benchmark-floors": lambda: check_benchmark_floors(
                root, run=not args.skip_benchmark
            ),
            "check-flags": check_feature_flags,
            "check-rollback": check_rollback_gates,
            "check-guide": lambda: check_guide_boundaries(root),
        }
        if command in check_map:
            result = check_map[command]()
            payload = result.to_dict()
            if args.json:
                _print_json(payload)
            else:
                print(f"{result.name}: {result.status.value} — {result.detail}")
            return 0 if result.ok else 1

        if command == "metrics":
            metrics = collect_metrics(run_benchmark=not args.skip_benchmark)
            payload = metrics.to_dict()
            write_checkpoint("rpr-046-metrics", payload)
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"{METRICS_INTERFACE} cases={metrics.case_count} "
                    f"analytical={metrics.analytical_coverage} "
                    f"model={metrics.model_rate} "
                    f"tokens={metrics.tokens} context_bytes={metrics.context_bytes} "
                    f"fixed_point_iters={metrics.fixed_point_iterations_total} "
                    f"stages={list(BENCHMARK_STAGES)} "
                    f"floors_ok={metrics.floors_hold()} "
                    f"metrics_id={metrics.metrics_id}"
                )
            return 0 if metrics.floors_hold() else 1

        if command == "policy":
            policy = default_rollout_policy()
            payload = policy.to_dict()
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"{ROLLOUT_POLICY_INTERFACE} mode={policy.mode_value} "
                    f"mutation_authorized={policy.mutation_authorized} "
                    f"policy_binding_id={policy.policy_binding_id}"
                )
            return 0

        parser.error(f"unknown command: {command}")
        return 2
    except ChangePropagationRolloutError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
