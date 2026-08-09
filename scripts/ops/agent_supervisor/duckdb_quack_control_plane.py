#!/usr/bin/env python3
"""Operator CLI for DuckDB/Quack control-plane staged rollout (DQP-038).

Subcommands:

* ``status`` — show current stage / authority binding
* ``promote`` — attempt a staged transition under evidence gates
* ``rollback`` — kill-switch route change (history preserved)
* ``stages`` — list closed stage vocabulary and transitions
* ``guide-check`` — verify operator guide exists and states beta limits

Cold import and ``--help`` start no process and open no database.

Exit codes: 0 success, 1 failure/denied, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duckdb_quack_control_plane",
        description="DuckDB/Quack control-plane staged cutover operator tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="List closed rollout stages and transitions")
    sub.add_parser("status", help="Show default controller status")

    promote = sub.add_parser("promote", help="Attempt staged promotion")
    promote.add_argument(
        "--to",
        required=True,
        choices=("observe", "shadow", "assist", "canary", "default"),
        help="Target stage",
    )
    promote.add_argument(
        "--from-stage",
        default="off",
        choices=("off", "observe", "shadow", "assist", "canary", "default", "rollback"),
        help="Starting stage (default: off)",
    )
    promote.add_argument(
        "--json-evidence",
        default="",
        help="Optional path to EvidenceBundle JSON; hermetic pass used if omitted",
    )
    promote.add_argument(
        "--walk",
        action="store_true",
        help="Walk the full path from --from-stage toward --to",
    )

    rollback = sub.add_parser("rollback", help="Engage kill-switch rollback")
    rollback.add_argument(
        "--from-stage",
        default="canary",
        choices=("off", "observe", "shadow", "assist", "canary", "default", "rollback"),
    )

    guide = sub.add_parser("guide-check", help="Verify operator guide presence")
    guide.add_argument(
        "--path",
        default="",
        help="Override guide path (default: docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md)",
    )

    return parser


def _load_evidence(path: str):
    from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
        EvidenceBundle,
        EvidenceItem,
        hermetic_passing_evidence,
    )

    if not path:
        return hermetic_passing_evidence()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = tuple(
        EvidenceItem(
            root=str(item["root"]),
            identity=str(item["identity"]),
            age_seconds=int(item.get("age_seconds", 0)),
            passed=bool(item.get("passed", False)),
            synthetic=bool(item.get("synthetic", False)),
            skipped=bool(item.get("skipped", False)),
            tree_id=str(item.get("tree_id") or ""),
            schema_checksum=str(item.get("schema_checksum") or ""),
            profile_id=str(item.get("profile_id") or ""),
        )
        for item in payload.get("items") or ()
    )
    return EvidenceBundle(
        items=items,
        tree_id=str(payload["tree_id"]),
        schema_checksum=str(payload["schema_checksum"]),
        store_generation=int(payload.get("store_generation", 1)),
        quack_profile=str(payload["quack_profile"]),
        server_available=bool(payload.get("server_available", True)),
        remote_endpoint=bool(payload.get("remote_endpoint", False)),
        beta_waiver=bool(payload.get("beta_waiver", False)),
        backup_age_seconds=int(payload.get("backup_age_seconds", 0)),
    )


def _cmd_stages() -> int:
    from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
        STAGE_AUTHORITY,
        DatabaseRollout,
        RolloutStage,
    )

    controller = DatabaseRollout()
    payload = {
        "stages": [item.value for item in RolloutStage],
        "authority": dict(STAGE_AUTHORITY),
        "transitions": {
            stage.value: sorted(t.value for t in controller.allowed_transitions())
            for stage in RolloutStage
            for controller in [DatabaseRollout(initial_stage=stage)]
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return EXIT_SUCCESS


def _cmd_status(from_stage: str = "off") -> int:
    from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
        DatabaseRollout,
        parse_stage,
    )

    controller = DatabaseRollout(initial_stage=parse_stage(from_stage))
    print(
        json.dumps(
            {
                "stage": controller.stage.value,
                "authority_mode": controller.authority_mode,
                "allowed": sorted(t.value for t in controller.allowed_transitions()),
                "kill_switch": controller.policy.kill_switch_engaged,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_SUCCESS


def _cmd_promote(args: argparse.Namespace) -> int:
    from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
        DatabaseRollout,
        RolloutStage,
        parse_stage,
    )

    evidence = _load_evidence(args.json_evidence)
    controller = DatabaseRollout(initial_stage=parse_stage(args.from_stage))
    target = parse_stage(args.to)
    receipts: list[dict[str, Any]] = []

    if args.walk:
        order = [
            RolloutStage.OBSERVE,
            RolloutStage.SHADOW,
            RolloutStage.ASSIST,
            RolloutStage.CANARY,
            RolloutStage.DEFAULT,
        ]
        # Start walking from the stage after current.
        started = False
        for stage in order:
            if stage == controller.stage:
                started = True
                continue
            if controller.stage is RolloutStage.OFF and stage is RolloutStage.OBSERVE:
                started = True
            if not started and stage != target:
                # Jump-start from off toward target path.
                if controller.stage is RolloutStage.OFF:
                    started = True
                else:
                    continue
            receipt = controller.transition(stage, evidence)
            receipts.append(receipt.to_dict())
            if not receipt.promoted:
                print(json.dumps({"receipts": receipts}, indent=2, sort_keys=True))
                return EXIT_FAILURE
            if stage == target:
                break
    else:
        receipt = controller.transition(target, evidence)
        receipts.append(receipt.to_dict())
        print(json.dumps({"receipts": receipts}, indent=2, sort_keys=True))
        return EXIT_SUCCESS if receipt.promoted else EXIT_FAILURE

    print(json.dumps({"receipts": receipts}, indent=2, sort_keys=True))
    return EXIT_SUCCESS


def _cmd_rollback(args: argparse.Namespace) -> int:
    from ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout import (
        DatabaseRollout,
        hermetic_passing_evidence,
        parse_stage,
    )

    controller = DatabaseRollout(initial_stage=parse_stage(args.from_stage))
    receipt = controller.transition(
        "rollback", hermetic_passing_evidence(), force_rollback=True
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return EXIT_SUCCESS if receipt.verdict.value == "rolled_back" else EXIT_FAILURE


def _cmd_guide_check(args: argparse.Namespace) -> int:
    root = _repo_root()
    path = Path(args.path) if args.path else (
        root / "docs" / "guides" / "AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md"
    )
    if not path.is_file():
        print(json.dumps({"ok": False, "error": f"missing guide: {path}"}))
        return EXIT_FAILURE
    text = path.read_text(encoding="utf-8")
    required_phrases = (
        "beta",
        "single-failure-domain",
        "loopback",
        "backup",
        "restore",
        "rollback",
        "health",
        "upgrade",
    )
    missing = [phrase for phrase in required_phrases if phrase.lower() not in text.lower()]
    ok = not missing
    print(
        json.dumps(
            {"ok": ok, "path": str(path), "missing_phrases": missing},
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_SUCCESS if ok else EXIT_FAILURE


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else EXIT_USAGE
        return code if code in {EXIT_SUCCESS, EXIT_USAGE} else EXIT_USAGE

    _ensure_repo_path()
    try:
        if args.command == "stages":
            return _cmd_stages()
        if args.command == "status":
            return _cmd_status()
        if args.command == "promote":
            return _cmd_promote(args)
        if args.command == "rollback":
            return _cmd_rollback(args)
        if args.command == "guide-check":
            return _cmd_guide_check(args)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
