#!/usr/bin/env python3
"""Operator facade for DuckDB/Quack control-plane rollout cutover (DQP-038).

Delegates staged canary, default cutover, and rollback decisions to
:class:`~ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout.DatabaseRollout`
(``DatabaseRolloutPolicy@1`` / ``DatabaseCutoverReceipt@1``).

Also exposes exact health / backup / restore / upgrade procedure recipes that
operators must follow. Cold import and ``--help`` start no process, open no
database, and load no optional providers. Auth tokens are never accepted on
argv.

Exit codes
----------
* 0 — success / report-only
* 1 — failure / denied promotion
* 2 — usage error
* 3 — abstain (insufficient evidence / kill switch holds)
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

_SUBCOMMANDS = (
    "status",
    "stages",
    "promote",
    "rollback",
    "kill-switch",
    "kill-switch-clear",
    "default-program",
    "release-gate",
    "health",
    "backup",
    "restore",
    "upgrade",
    "recipe",
    "limitations",
)

_FORBIDDEN_ARGV_MARKERS = (
    "--token",
    "--auth-token",
    "--quack-token",
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

# Exact operator procedures (also documented in the guide).
HEALTH_PROCEDURE: Mapping[str, Any] = {
    "name": "health",
    "steps": [
        "Confirm state-owner process birth identity via quack_state_server status/ready.",
        "Query database readiness (store generation, schema fingerprint, server id).",
        "Run duckdb_quack_doctor diagnose; abstain when ownership is unknown.",
        "Verify loopback bind (127.0.0.1 / ::1) unless a reviewed remote policy exists.",
        "Confirm backup age is within DEFAULT_MAX_BACKUP_AGE_SECONDS (30 days).",
        "Refuse file-age PID signalling; use fenced reclaim only.",
    ],
    "commands": [
        "python scripts/ops/agent_supervisor/quack_state_server.py --database PATH status",
        "python scripts/ops/agent_supervisor/quack_state_server.py --database PATH ready",
        "python scripts/ops/agent_supervisor/duckdb_quack_doctor.py diagnose --database PATH",
        "python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py health",
    ],
}

BACKUP_PROCEDURE: Mapping[str, Any] = {
    "name": "backup",
    "steps": [
        "Acquire exclusive maintenance lease or stop the state-owner cleanly.",
        "Refuse direct-file copy while ownership is live or unknown.",
        "Invoke ControlPlaneBackup@1 to create a verified consistent snapshot.",
        "Independently verify digest, schema version, and authority roots.",
        "Update retention manifest; prune only verified excess snapshots.",
        "Record backup age for the release gate.",
    ],
    "module": "ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup",
    "interface": "ControlPlaneBackup@1",
    "max_backup_age_seconds": 30 * 24 * 3600,
}

RESTORE_PROCEDURE: Mapping[str, Any] = {
    "name": "restore",
    "steps": [
        "Stop or fence the live state-owner; never restore under unknown ownership.",
        "Select a verified backup.manifest.json with matching schema fingerprint.",
        "Invoke RestoreReceipt@1 rehearsal first when validating a new profile.",
        "Apply restore; rotate store generation so pre-rotation writers fail closed.",
        "Invalidate stale clients/leases; require re-attach with new generation.",
        "Run health procedure; confirm event/task/lease roots match the snapshot.",
    ],
    "module": "ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup",
    "interface": "RestoreReceipt@1",
}

UPGRADE_PROCEDURE: Mapping[str, Any] = {
    "name": "upgrade",
    "steps": [
        "Pin identical DuckDB and Quack extension fingerprints on server and clients.",
        "Backup and restore-rehearse on the current pinned 1.5.x profile first.",
        "Apply schema migrations under exclusive maintenance; prove fresh==upgraded.",
        "Re-run chaos, canary, shadow, and churn gates on the exact tree/profile.",
        "Promote one stage at a time via this control-plane facade; never jump.",
        "Only after canary + valid release gate, cut over default for new programs.",
    ],
    "pinned_profile": "duckdb-1.5.x-quack-pinned",
    "future_profile_requires": "separately tested restore rehearsal + full release gate",
}

BETA_LIMITATIONS: tuple[str, ...] = (
    "quack_is_beta_in_duckdb_1_5_x",
    "protocol_names_and_defaults_may_change_before_duckdb_2_0",
    "server_and_clients_must_use_identical_pinned_build",
    "default_authorization_callback_permits_every_authenticated_query",
    "no_server_push_clients_must_poll",
    "one_quack_server_is_one_failure_domain",
    "loopback_bind_required_unless_separately_reviewed",
    "unsigned_or_community_extension_path_is_not_attested_integrity",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _reject_forbidden_argv(argv: Sequence[str]) -> str | None:
    lowered = [str(item).strip().lower() for item in argv]
    for item in lowered:
        name = item.split("=", 1)[0]
        if name in _FORBIDDEN_ARGV_MARKERS:
            return (
                f"refusing argv credential flag {name!r}; use secret handles only"
            )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duckdb_quack_control_plane",
        description=(
            "Staged canary, default cutover, rollback, and operator procedures "
            "for the DuckDB/Quack control plane (DatabaseRolloutPolicy@1)."
        ),
    )
    parser.add_argument(
        "command",
        choices=_SUBCOMMANDS,
        help="Operation: " + ", ".join(_SUBCOMMANDS),
    )
    parser.add_argument(
        "--state-json",
        default=None,
        help="Path to a persisted rollout status JSON (optional)",
    )
    parser.add_argument(
        "--evidence-json",
        default=None,
        help="Path to release-gate evidence map JSON",
    )
    parser.add_argument(
        "--target-stage",
        default=None,
        help="Target stage for promote/rollback",
    )
    parser.add_argument(
        "--repository-id",
        default="repository:local",
        help="Repository identity bound into receipts",
    )
    parser.add_argument(
        "--tree-id",
        default="tree:local",
        help="Current tree identity bound into receipts",
    )
    parser.add_argument(
        "--store-id",
        default="control.duckdb",
        help="Logical store identity",
    )
    parser.add_argument(
        "--store-generation",
        type=int,
        default=1,
        help="Store generation bound into receipts",
    )
    parser.add_argument(
        "--schema-revision",
        default="control-plane-schema@1",
        help="Schema revision bound into receipts",
    )
    parser.add_argument(
        "--quack-profile",
        default="duckdb-1.5.x-quack-pinned",
        help="Pinned Quack compatibility profile",
    )
    parser.add_argument(
        "--program-id",
        default="",
        help="Program id for default-program registration",
    )
    parser.add_argument(
        "--allow-default",
        action="store_true",
        help="Admit the full ladder including default cutover",
    )
    parser.add_argument(
        "--server-available",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the Quack state-owner is available",
    )
    parser.add_argument(
        "--backup-age-seconds",
        type=int,
        default=0,
        help="Observed backup age for the release gate",
    )
    parser.add_argument(
        "--remote-bind-requested",
        action="store_true",
        help="Mark a non-loopback bind request (denied unless policy allows)",
    )
    parser.add_argument(
        "--beta-waiver-recorded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether beta limitation waiver is recorded",
    )
    parser.add_argument(
        "--partial-rollout",
        action="store_true",
        help="Mark partial multi-program rollout (denies promotion)",
    )
    parser.add_argument(
        "--delete-history",
        action="store_true",
        help="Attempt history deletion on rollback (always refused)",
    )
    parser.add_argument(
        "--accept-legacy-dual-writes",
        action="store_true",
        help="Attempt permanent dual-write acceptance (always refused)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on stdout",
    )
    return parser


def _load_json(path: str | None) -> Any:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_rollout(args: argparse.Namespace) -> Any:
    _ensure_repo_path()
    from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
        DatabaseRollout,
        DatabaseRolloutBinding,
        DatabaseRolloutPolicy,
        DatabaseRolloutStage,
        RolloutDisposition,
    )

    policy = (
        DatabaseRolloutPolicy.with_default_cutover()
        if args.allow_default
        else DatabaseRolloutPolicy.default()
    )
    binding = DatabaseRolloutBinding(
        repository_id=args.repository_id,
        tree_id=args.tree_id,
        store_id=args.store_id,
        store_generation=args.store_generation,
        schema_revision=args.schema_revision,
        quack_profile=args.quack_profile,
        program_id=args.program_id or "",
    )
    rollout = DatabaseRollout(policy, binding=binding)

    # Restore prior stage from state snapshot when provided.
    state = _load_json(args.state_json)
    if isinstance(state, Mapping):
        stage = state.get("stage")
        if stage:
            # Walk forward without evidence for off→assist only when restoring
            # hermetic operator state; canary/default still need gates later.
            target = DatabaseRolloutStage(str(stage))
            for step in DatabaseRolloutStage:
                if step.rank == 0:
                    continue
                if step.rank > target.rank:
                    break
                if step.rank <= DatabaseRolloutStage.ASSIST.rank:
                    rollout.promote(step)
                else:
                    break
            # If saved stage is canary/default, operator must re-promote with evidence.
            if target.rank > DatabaseRolloutStage.ASSIST.rank and rollout.stage != target:
                # Leave at assist; status will show partial restore.
                pass
        if state.get("kill_switch_engaged"):
            rollout.engage_kill_switch()
        if state.get("canary_completed"):
            rollout._canary_completed = True  # noqa: SLF001 — restore hermetic flag

    return rollout, RolloutDisposition


def _emit(payload: Mapping[str, Any], *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        for key, value in payload.items():
            if isinstance(value, (dict, list, tuple)):
                print(f"{key}: {json.dumps(value, sort_keys=True, default=str)}")
            else:
                print(f"{key}: {value}")
    return EXIT_SUCCESS


def _recipe() -> dict[str, Any]:
    return {
        "interface": "DatabaseRolloutPolicy@1",
        "receipt_interface": "DatabaseCutoverReceipt@1",
        "commands": list(_SUBCOMMANDS),
        "ladder": ["off", "observe", "shadow", "assist", "canary", "default"],
        "beta_limitations": list(BETA_LIMITATIONS),
        "single_failure_domain": True,
        "loopback_required_by_default": True,
        "procedures": {
            "health": dict(HEALTH_PROCEDURE),
            "backup": dict(BACKUP_PROCEDURE),
            "restore": dict(RESTORE_PROCEDURE),
            "upgrade": dict(UPGRADE_PROCEDURE),
        },
        "acceptance": {
            "new_programs_default_to_quack_only_under_valid_release_gate": True,
            "rollback_switches_route_without_deleting_history": True,
            "rollback_never_accepts_legacy_dual_writes": True,
        },
    }


def run(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    forbidden = _reject_forbidden_argv(argv)
    if forbidden is not None:
        print(f"error: {forbidden}", file=sys.stderr)
        return EXIT_USAGE
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
        return code if code != 0 else EXIT_SUCCESS

    command = args.command

    if command == "recipe":
        return _emit(_recipe(), as_json=args.json or True)

    if command == "limitations":
        return _emit(
            {
                "beta_limitations": list(BETA_LIMITATIONS),
                "single_failure_domain": (
                    "one_quack_server_is_one_failure_domain"
                ),
                "loopback": (
                    "loopback_bind_required_unless_separately_reviewed"
                ),
                "note": (
                    "Quack is beta; one supervised server with restore is "
                    "resilient but not highly available."
                ),
            },
            as_json=args.json or True,
        )

    if command == "health":
        return _emit(dict(HEALTH_PROCEDURE), as_json=args.json or True)

    if command == "backup":
        return _emit(dict(BACKUP_PROCEDURE), as_json=args.json or True)

    if command == "restore":
        return _emit(dict(RESTORE_PROCEDURE), as_json=args.json or True)

    if command == "upgrade":
        return _emit(dict(UPGRADE_PROCEDURE), as_json=args.json or True)

    if command == "stages":
        _ensure_repo_path()
        from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
            authority_mode_for_stage,
            closed_rollout_stages,
            dual_observation_allowed,
        )

        stages = []
        for name in closed_rollout_stages():
            stages.append(
                {
                    "stage": name,
                    "authority_mode": authority_mode_for_stage(name).value,
                    "dual_observation": dual_observation_allowed(name),
                }
            )
        return _emit({"stages": stages}, as_json=args.json or True)

    rollout, RolloutDisposition = _build_rollout(args)
    evidence = _load_json(args.evidence_json)

    if command == "status":
        return _emit(rollout.status(), as_json=args.json or True)

    if command == "kill-switch":
        receipt = rollout.engage_kill_switch()
        payload = receipt.to_dict()
        payload["status"] = rollout.status()
        _emit(payload, as_json=args.json or True)
        return EXIT_SUCCESS

    if command == "kill-switch-clear":
        rollout.clear_kill_switch()
        return _emit(rollout.status(), as_json=args.json or True)

    if command == "default-program":
        if not args.program_id:
            print("error: --program-id is required for default-program", file=sys.stderr)
            return EXIT_USAGE
        record = rollout.register_new_program(args.program_id)
        _emit(record, as_json=args.json or True)
        return EXIT_SUCCESS if record.get("defaults_to_quack") or True else EXIT_SUCCESS

    if command == "release-gate":
        gate = rollout.evaluate_release_gate(
            evidence,
            server_available=args.server_available,
            backup_age_seconds=args.backup_age_seconds,
            remote_bind_requested=args.remote_bind_requested,
            beta_waiver_recorded=args.beta_waiver_recorded,
            dual_write_accepted=args.accept_legacy_dual_writes,
            history_deletion_requested=args.delete_history,
            partial_rollout=args.partial_rollout,
            target_stage=args.target_stage or "canary",
        )
        _emit(gate.to_dict(), as_json=args.json or True)
        if gate.valid:
            return EXIT_SUCCESS
        if "kill_switch" in gate.reason_codes:
            return EXIT_ABSTAIN
        return EXIT_FAILURE

    if command == "promote":
        if not args.target_stage:
            print("error: --target-stage is required for promote", file=sys.stderr)
            return EXIT_USAGE
        receipt = rollout.promote(
            args.target_stage,
            evidence=evidence,
            server_available=args.server_available,
            backup_age_seconds=args.backup_age_seconds,
            remote_bind_requested=args.remote_bind_requested,
            beta_waiver_recorded=args.beta_waiver_recorded,
            dual_write_accepted=args.accept_legacy_dual_writes,
            history_deletion_requested=args.delete_history,
            partial_rollout=args.partial_rollout,
        )
        _emit(receipt.to_dict(), as_json=args.json or True)
        if receipt.disposition is RolloutDisposition.PROMOTE:
            return EXIT_SUCCESS
        if receipt.disposition is RolloutDisposition.HOLD:
            return EXIT_ABSTAIN
        return EXIT_FAILURE

    if command == "rollback":
        receipt = rollout.rollback(
            args.target_stage,
            delete_history=args.delete_history,
            accept_legacy_dual_writes=args.accept_legacy_dual_writes,
        )
        _emit(receipt.to_dict(), as_json=args.json or True)
        # Rollback always preserves history and refuses dual writes.
        if not receipt.history_preserved or receipt.legacy_dual_write_accepted:
            return EXIT_FAILURE
        return EXIT_SUCCESS

    print(f"error: unhandled command {command!r}", file=sys.stderr)
    return EXIT_USAGE


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
