#!/usr/bin/env python3
"""Thin ops facade for DuckDB/Quack control-plane stall diagnostics (DQP-032).

Parses closed subcommands and delegates to
:class:`~ipfs_accelerate_py.agent_supervisor.rescue.database_watchdog.DatabaseWatchdog`.

Cold import and ``--help`` start no process, open no database, and load no
optional providers. The doctor **exposes evidence** and **abstains** when
server ownership is unknown. It never signals a raw PID or deletes a lock
based on file age alone.

Exit codes
----------
* 0 — healthy / report-only success
* 1 — failure
* 2 — usage error
* 3 — abstain (ownership unknown)
* 4 — actionable diagnoses present
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_ABSTAIN = 3
EXIT_ACTIONABLE = 4

_SUBCOMMANDS = (
    "diagnose",
    "doctor",
    "repair",
    "status",
    "authority",
    "classifications",
)

_FORBIDDEN_ARGV_MARKERS = (
    "--token",
    "--auth-token",
    "--password",
    "--secret",
    "--api-key",
    "--apikey",
    "--authorization",
    "--bearer",
    "--credential",
    "--private-key",
    "--cookie",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _reject_forbidden_argv(argv: Sequence[str]) -> None:
    lowered = [str(item).strip().lower() for item in argv]
    for item in lowered:
        name = item.split("=", 1)[0]
        if name in _FORBIDDEN_ARGV_MARKERS:
            raise SystemExit(
                f"refusing argv credential flag {name!r}; doctor is body-free"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duckdb_quack_doctor",
        description=(
            "Database-derived stall diagnostics and safe fenced repair for the "
            "DuckDB/Quack control plane. Exposes evidence; abstains when "
            "ownership is unknown; never acts on file age alone."
        ),
    )
    # Positional command (not argparse subparsers) so shared options remain
    # valid both before and after the command name, matching ops CLI usage
    # such as ``doctor --database X --observation-json Y``.
    parser.add_argument(
        "command",
        choices=_SUBCOMMANDS,
        help=(
            "Operation: "
            + ", ".join(_SUBCOMMANDS)
            + ". classifications/authority are cold metadata surfaces; "
            "diagnose/doctor expose evidence and abstain when ownership is "
            "unknown; repair requires exact fence/process-birth/generation."
        ),
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Path to watchdog DuckDB store (created if missing)",
    )
    parser.add_argument(
        "--observation-json",
        default=None,
        help="Path to a WatchdogObservation@1 JSON document",
    )
    parser.add_argument(
        "--diagnosis-json",
        default=None,
        help="Path to a StallDiagnosis@1 JSON document (repair)",
    )
    parser.add_argument(
        "--idempotency-key",
        default="",
        help="Optional idempotency key for repair",
    )
    parser.add_argument(
        "--expected-fence-epoch",
        type=int,
        default=None,
        help="Expected fence epoch for repair",
    )
    parser.add_argument(
        "--expected-fencing-token",
        type=int,
        default=None,
        help="Expected fencing token for repair",
    )
    parser.add_argument(
        "--expected-generation",
        type=int,
        default=None,
        help="Expected store generation for repair",
    )
    parser.add_argument(
        "--expected-process-birth-id",
        default="",
        help="Expected process-birth id for repair",
    )
    parser.add_argument(
        "--propose-repairs",
        action="store_true",
        help="When diagnosing, also decide fenced repair commands",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not persist diagnoses to the watchdog store",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _emit(payload: Mapping[str, Any], *, pretty: bool) -> None:
    if pretty:
        text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    else:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(text + "\n")


def _default_database(args: argparse.Namespace) -> Path:
    if args.database:
        return Path(args.database)
    return Path.cwd() / "state" / "watchdog" / "database_watchdog.duckdb"


def _require_observation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.observation_json:
        raise SystemExit("--observation-json is required for this command")
    return _load_json(args.observation_json)


def _system_exit_code(exc: SystemExit, *, default: int = EXIT_FAILURE) -> int:
    code = exc.code
    if code is None:
        return EXIT_SUCCESS
    if isinstance(code, int):
        return code
    # SystemExit("message") — treat as failure and surface the message.
    sys.stderr.write(f"{code}\n")
    return default


def run(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_forbidden_argv(argv)
    except SystemExit as exc:
        return _system_exit_code(exc, default=EXIT_USAGE)

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = _system_exit_code(exc, default=EXIT_USAGE)
        return code if code in {EXIT_SUCCESS, EXIT_USAGE} else EXIT_USAGE

    _ensure_repo_path()

    try:
        from ipfs_accelerate_py.agent_supervisor.rescue.database_watchdog import (
            CLASSIFICATION_VALUES,
            CommandStatus,
            DatabaseWatchdog,
            DoctorDisposition,
            OwnershipState,
            StallDiagnosis,
            WatchdogEvidence,
            WatchdogObservation,
            duckdb_available,
            open_database_watchdog,
        )
    except Exception as exc:  # pragma: no cover - import surface
        sys.stderr.write(f"failed to import database_watchdog: {exc}\n")
        return EXIT_FAILURE

    if args.command == "classifications":
        _emit(
            {
                "interface": "DatabaseWatchdog@1",
                "classifications": list(CLASSIFICATION_VALUES),
                "policy": {
                    "file_age_alone_action": False,
                    "raw_pid_signal": False,
                    "lock_deletion": False,
                    "ownership_unknown": "abstain",
                },
            },
            pretty=bool(args.pretty),
        )
        return EXIT_SUCCESS

    if not duckdb_available():
        sys.stderr.write("DuckDB is required for duckdb_quack_doctor\n")
        return EXIT_FAILURE

    db_path = _default_database(args)

    try:
        if args.command == "authority":
            with open_database_watchdog(db_path) as watchdog:
                _emit(
                    {
                        "interface": DatabaseWatchdog.INTERFACE,
                        "schema": DatabaseWatchdog.SCHEMA,
                        "authority": watchdog.authority_policy(),
                        "database": str(db_path),
                    },
                    pretty=bool(args.pretty),
                )
            return EXIT_SUCCESS

        if args.command == "status":
            with open_database_watchdog(db_path) as watchdog:
                diagnoses = watchdog.list_diagnoses(limit=32)
                _emit(
                    {
                        "interface": DatabaseWatchdog.INTERFACE,
                        "database": str(db_path),
                        "diagnosis_count": len(diagnoses),
                        "actionable_count": sum(1 for d in diagnoses if d.actionable),
                        "diagnoses": [d.to_dict() for d in diagnoses],
                        "authority": watchdog.authority_policy(),
                    },
                    pretty=bool(args.pretty),
                )
            return EXIT_SUCCESS

        if args.command in {"diagnose", "doctor"}:
            raw = _require_observation(args)
            observation = WatchdogObservation.from_mapping(raw)
            with open_database_watchdog(db_path) as watchdog:
                report = watchdog.doctor(
                    observation,
                    persist=not bool(args.no_persist),
                    propose_repairs=bool(args.propose_repairs),
                )
                _emit(report.to_dict(), pretty=bool(args.pretty))
            if report.disposition is DoctorDisposition.ABSTAIN:
                return EXIT_ABSTAIN
            if report.disposition is DoctorDisposition.ACTIONABLE:
                return EXIT_ACTIONABLE
            return EXIT_SUCCESS

        if args.command == "repair":
            if not args.diagnosis_json and not args.observation_json:
                sys.stderr.write(
                    "repair requires --diagnosis-json or --observation-json\n"
                )
                return EXIT_USAGE
            with open_database_watchdog(db_path) as watchdog:
                if args.diagnosis_json:
                    if (
                        args.expected_fence_epoch is None
                        or args.expected_fencing_token is None
                        or args.expected_generation is None
                    ):
                        sys.stderr.write(
                            "repair with --diagnosis-json requires "
                            "--expected-fence-epoch, --expected-fencing-token, "
                            "and --expected-generation\n"
                        )
                        return EXIT_USAGE
                    diag_raw = _load_json(args.diagnosis_json)
                    # Accept either a full StallDiagnosis or a nested form.
                    if "classification" not in diag_raw and "diagnosis" in diag_raw:
                        diag_raw = dict(diag_raw["diagnosis"])

                    diagnosis = StallDiagnosis(
                        diagnosis_id=str(
                            diag_raw.get("diagnosis_id") or "diagnosis:cli"
                        ),
                        classification=str(diag_raw.get("classification") or ""),
                        severity=str(diag_raw.get("severity") or "error"),
                        actionable=bool(diag_raw.get("actionable", True)),
                        observed_at_ms=int(diag_raw.get("observed_at_ms") or 0),
                        reason=str(diag_raw.get("reason") or ""),
                        subject_kind=str(diag_raw.get("subject_kind") or ""),
                        subject_id=str(diag_raw.get("subject_id") or ""),
                        task_cid=str(diag_raw.get("task_cid") or ""),
                        session_id=str(diag_raw.get("session_id") or ""),
                        worktree_id=str(diag_raw.get("worktree_id") or ""),
                        lease_id=str(diag_raw.get("lease_id") or ""),
                        evidence=tuple(
                            WatchdogEvidence.from_mapping(item)
                            for item in (diag_raw.get("evidence") or [])
                            if isinstance(item, Mapping)
                        ),
                        body=dict(diag_raw.get("body") or {}),
                    )
                    ownership = OwnershipState.ABSENT
                    fence_epoch = int(args.expected_fence_epoch)
                    fencing_token = int(args.expected_fencing_token)
                    generation = int(args.expected_generation)
                    birth_id = str(args.expected_process_birth_id or "")
                else:
                    raw = _require_observation(args)
                    observation = WatchdogObservation.from_mapping(raw)
                    report = watchdog.doctor(
                        observation,
                        persist=not bool(args.no_persist),
                        propose_repairs=False,
                    )
                    if report.disposition is DoctorDisposition.ABSTAIN:
                        _emit(report.to_dict(), pretty=bool(args.pretty))
                        return EXIT_ABSTAIN
                    actionable = [d for d in report.diagnoses if d.actionable]
                    if not actionable:
                        _emit(
                            {
                                "status": "no_actionable_diagnoses",
                                "report": report.to_dict(),
                            },
                            pretty=bool(args.pretty),
                        )
                        return EXIT_SUCCESS
                    diagnosis = actionable[0]
                    ownership = observation.ownership
                    fence_epoch = int(
                        args.expected_fence_epoch
                        if args.expected_fence_epoch is not None
                        else observation.fence_epoch
                    )
                    fencing_token = int(
                        args.expected_fencing_token
                        if args.expected_fencing_token is not None
                        else observation.fencing_token
                    )
                    generation = int(
                        args.expected_generation
                        if args.expected_generation is not None
                        else observation.generation
                    )
                    birth_id = str(
                        args.expected_process_birth_id
                        or observation.server_process_birth_id
                    )

                command = watchdog.decide_and_apply(
                    diagnosis,
                    expected_fence_epoch=fence_epoch,
                    expected_fencing_token=fencing_token,
                    expected_generation=generation,
                    expected_process_birth_id=birth_id,
                    idempotency_key=str(args.idempotency_key or ""),
                    ownership=ownership,
                )
                _emit(command.to_dict(), pretty=bool(args.pretty))
                if command.status is CommandStatus.ABSTAINED:
                    return EXIT_ABSTAIN
                if command.status is CommandStatus.REJECTED:
                    return EXIT_FAILURE
                return EXIT_SUCCESS

        sys.stderr.write(f"unknown command: {args.command}\n")
        return EXIT_USAGE
    except SystemExit as exc:
        return _system_exit_code(exc)
    except Exception as exc:
        sys.stderr.write(f"duckdb_quack_doctor failed: {exc}\n")
        return EXIT_FAILURE


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
