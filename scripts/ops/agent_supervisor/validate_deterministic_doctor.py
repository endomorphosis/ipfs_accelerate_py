#!/usr/bin/env python3
"""Operator validation for deterministic-doctor rollout controls (LPR-041).

Validates immutable report-only defaults, hard-off model/network flags,
hard-on safety gates, resource limits, manual monotonic promotion, rollback
gates, lifecycle doctor read-only/idempotent behaviour, and optional-provider
absence that does not block report-only startup.

This script never grants mutation, completion, merge, or process authority.
Cold import and ``--help`` start no process and import no optional providers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_rollout import (  # noqa: E402
    ROLLOUT_POLICY_INTERFACE,
    VALIDATOR_INTERFACE,
    DeterministicDoctorRolloutError,
    _plain,
    check_artifacts_present,
    check_config_defaults,
    check_feature_flags,
    check_guide_boundaries,
    check_lifecycle_doctor_readonly,
    check_limits,
    check_optional_provider_absence,
    check_promotion_monotonicity,
    check_related_surfaces,
    check_rollback_gates,
    default_rollout_policy,
    doctor,
    repository_root,
    run_all_checks,
    status,
    write_checkpoint,
)


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

    parser = argparse.ArgumentParser(
        description=(
            "Validate deterministic-doctor rollout controls, defaults, "
            "gates, promotion, rollback, and lifecycle safety (LPR-041)."
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
        help="Run control-plane doctor checks (read-only).",
        parents=[common],
    )
    sub.add_parser(
        "status",
        help="Show mode, kill switch, limits, and lifecycle state.",
        parents=[common],
    )
    sub.add_parser("check-config", help="Validate immutable config defaults.", parents=[common])
    sub.add_parser("check-flags", help="Validate feature-flag defaults.", parents=[common])
    sub.add_parser("check-limits", help="Validate resource limits.", parents=[common])
    sub.add_parser("check-promotion", help="Validate manual monotonic promotion.", parents=[common])
    sub.add_parser("check-rollback", help="Validate rollback gates.", parents=[common])
    sub.add_parser(
        "check-lifecycle",
        help="Validate ordinary lifecycle doctor remains read-only/idempotent.",
        parents=[common],
    )
    sub.add_parser(
        "check-providers",
        help="Validate optional provider absence does not block report-only startup.",
        parents=[common],
    )
    sub.add_parser("check-guide", help="Validate operator guide boundaries.", parents=[common])
    sub.add_parser("check-artifacts", help="Validate declared artifacts exist.", parents=[common])
    sub.add_parser("check-related", help="Validate related doctor surfaces.", parents=[common])
    sub.add_parser("policy", help="Emit the default (report-only) rollout policy.", parents=[common])
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
            report = doctor(root) if command == "doctor" else run_all_checks(root)
            if command == "check-all":
                report = dict(report)
                report["command"] = "check-all"
            write_checkpoint("lpr-041-validation-report", report)
            if args.json:
                _print_json(report)
            else:
                status_word = "healthy" if report.get("valid") else "unhealthy"
                print(
                    f"{VALIDATOR_INTERFACE} command={report.get('command', command)} "
                    f"valid={report.get('valid')} status={status_word} "
                    f"default_mode={report.get('default_mode')} "
                    f"failed={report.get('failed')} "
                    f"report_id={report.get('report_id')}"
                )
            return 0 if report.get("valid") else 1

        if command == "status":
            report = status(root)
            write_checkpoint("lpr-041-status", report)
            if args.json:
                _print_json(report)
            else:
                print(
                    f"{VALIDATOR_INTERFACE} mode={report['mode']} "
                    f"effective_mode={report['effective_mode']} "
                    f"kill_switch={report['kill_switch_engaged']} "
                    f"valid={report['valid']} "
                    f"report_id={report['report_id']}"
                )
            return 0 if report.get("valid") else 1

        check_map = {
            "check-config": lambda: check_config_defaults(root),
            "check-flags": check_feature_flags,
            "check-limits": check_limits,
            "check-promotion": check_promotion_monotonicity,
            "check-rollback": check_rollback_gates,
            "check-lifecycle": lambda: check_lifecycle_doctor_readonly(root),
            "check-providers": check_optional_provider_absence,
            "check-guide": lambda: check_guide_boundaries(root),
            "check-artifacts": lambda: check_artifacts_present(root),
            "check-related": lambda: check_related_surfaces(root),
        }
        if command in check_map:
            result = check_map[command]()
            payload = result.to_dict()
            if args.json:
                _print_json(payload)
            else:
                print(f"{result.name}: {result.status.value} — {result.detail}")
            return 0 if result.ok else 1

        if command == "policy":
            policy = default_rollout_policy()
            payload = policy.to_dict()
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"{ROLLOUT_POLICY_INTERFACE} mode={policy.mode_value} "
                    f"mutation_authorized={policy.mutation_authorized} "
                    f"narrow_auto={policy.narrow_autonomous_mutation_enabled} "
                    f"policy_binding_id={policy.policy_binding_id}"
                )
            return 0

        parser.error(f"unknown command: {command}")
        return 2
    except DeterministicDoctorRolloutError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
