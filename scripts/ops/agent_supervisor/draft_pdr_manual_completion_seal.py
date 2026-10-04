#!/usr/bin/env python3
"""Draft (and optionally apply) operator manual-completion seals for PDR gates.

Threat model / authority policy
-------------------------------
Manual completion seals are **operator-owned**.  Models, candidates, and
automatic controllers must not claim ``interactive_user`` authority.

This script therefore:

* always builds a **draft** package with exact artifact digests and a pinned
  ``receipt_id`` under ``state/.../operator_review/``;
* only writes a production seal, updates the scheduler pin, and/or marks the
  board task completed when ``--operator-ack`` is supplied by a human.

Default mode is draft-only.

Examples
--------
Draft PDR-060 and PDR-072 packages (no authority side effects)::

    python scripts/ops/agent_supervisor/draft_pdr_manual_completion_seal.py draft

Apply a seal after human review (writes seal + scheduler pin only)::

    python scripts/ops/agent_supervisor/draft_pdr_manual_completion_seal.py apply \\
        --task PDR-060 --operator-ack

Apply seal and mark the board complete::

    python scripts/ops/agent_supervisor/draft_pdr_manual_completion_seal.py apply \\
        --task PDR-060 --operator-ack --mark-complete
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Allow invocation without requiring an editable install.
_REPO_CANDIDATE = Path(__file__).resolve().parents[3]
if str(_REPO_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_CANDIDATE))

from ipfs_accelerate_py.agent_supervisor.control.manual_completion_seal import (
    ManualCompletionSealError,
    verify_manual_completion_seal,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    load_supervisor_scheduler_config,
)

BOARD_NAMESPACE = "agent-supervisor-proof-directed-planner-doctor-v1"
DEFAULT_SCHEDULER = (
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)
DEFAULT_TODO = (
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md"
)
DEFAULT_DRAFT_DIR = (
    "state/pdr-operator-review/manual_completion_seals"
)

# Closed seal specs for currently unblocked operator gates.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "PDR-060": {
        "receipt_path": "config/agent_supervisor_planner_doctor_attestation.seal.json",
        "schema": (
            "ipfs_accelerate_py.agent_supervisor.planner_doctor.attestation_seal@1"
        ),
        "interface": "PlannerDoctorAttestationSeal@1",
        "policy_revision": "1",
        "artifact_paths": {
            "attestation_module": (
                "ipfs_accelerate_py/agent_supervisor/proof/"
                "planner_doctor_attestation.py"
            ),
            "zkp_threat_model": (
                "docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md"
            ),
            "attestation_test": (
                "test/api/test_agent_supervisor_planner_doctor_attestation.py"
            ),
        },
        "grant_type": "attestation_activation",
        "grant_action": "activate_attestation_contract",
        "grant_claims": {},
        "reviewed_base_claims": {},
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_planner_doctor_attestation.py "
            "test/api/test_agent_supervisor_program_analysis_zkp.py -q"
        ),
    },
    "PDR-072": {
        "receipt_path": (
            "config/agent_supervisor_planner_doctor_quality_oracle.seal.json"
        ),
        "schema": (
            "ipfs_accelerate_py.agent_supervisor.planner_doctor."
            "quality_oracle_seal@1"
        ),
        "interface": "PlannerDoctorQualityOracleSeal@1",
        "policy_revision": "1",
        "artifact_paths": {
            "quality_oracle": (
                "ipfs_accelerate_py/agent_supervisor/validation/"
                "planner_doctor_quality_oracle.py"
            ),
            "oracle_manifest": (
                "test/fixtures/agent_supervisor/planner_doctor_holdout/"
                "oracle.manifest.json"
            ),
            "quality_oracle_test": (
                "test/api/test_agent_supervisor_planner_doctor_quality_oracle.py"
            ),
        },
        "grant_type": "oracle_activation",
        "grant_action": "activate_quality_oracle",
        "grant_claims": {
            "automatic_promotion": False,
            "oracle_activation": True,
        },
        "reviewed_base_claims": {
            "oracle_handle": "opaque:operator-cas/planner-doctor-quality-oracle@1",
            "operator_seal_handle": (
                "opaque:operator-seal/planner-doctor-quality-oracle-pdr-072@1"
            ),
            "corpus_authority": "operator-owned-holdout-judge",
        },
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_planner_doctor_quality_oracle.py -q"
        ),
    },
}


def _repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _receipt_identity(receipt: Mapping[str, Any]) -> str:
    body = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_id"
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _artifact_entry(root: Path, role: str, relative: str) -> dict[str, Any]:
    payload = (root / relative).read_bytes()
    return {
        "role": role,
        "path": relative,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def build_receipt(
    root: Path,
    task_id: str,
    *,
    commit: str | None = None,
    tree: str | None = None,
) -> dict[str, Any]:
    if task_id not in TASK_SPECS:
        raise SystemExit(f"unsupported task_id: {task_id}")
    spec = TASK_SPECS[task_id]
    commit = commit or _git(root, "rev-parse", "HEAD")
    tree = tree or _git(root, "rev-parse", "HEAD^{tree}")
    artifacts = [
        _artifact_entry(root, role, relative)
        for role, relative in spec["artifact_paths"].items()
    ]
    receipt: dict[str, Any] = {
        "schema": spec["schema"],
        "interface": spec["interface"],
        "receipt_version": "1",
        "task_id": task_id,
        "board_namespace": BOARD_NAMESPACE,
        "decision": "sealed",
        "policy_revision": str(spec["policy_revision"]),
        "reviewed_base": {
            "commit": commit,
            "tree": tree,
            "git_object_format": "sha1",
            "relation_to_activation_head": "equal_or_ancestor",
            **dict(spec.get("reviewed_base_claims") or {}),
        },
        "artifacts": artifacts,
        "operator": {
            "identity": "interactive_user",
            "authority_basis": "interactive_user_delegation",
            "candidate": False,
            "model": False,
            "automatic_controller": False,
        },
        "grant": {
            "type": spec["grant_type"],
            "allowed_actions": [spec["grant_action"]],
            **dict(spec.get("grant_claims") or {}),
            "board_namespace": BOARD_NAMESPACE,
            "policy_revision": str(spec["policy_revision"]),
            "delegable": False,
            "mutation_authority": False,
            "completion_authority": False,
            "promotion_authority": False,
            "task_status_authority": False,
            "protected_anchor_write_authority": False,
        },
    }
    receipt["receipt_id"] = _receipt_identity(receipt)
    return receipt


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def draft_package(root: Path, task_id: str, draft_dir: Path) -> dict[str, Any]:
    spec = TASK_SPECS[task_id]
    receipt = build_receipt(root, task_id)
    package = {
        "schema": "ipfs_accelerate_py.agent_supervisor.pdr.operator_seal_draft@1",
        "task_id": task_id,
        "board_namespace": BOARD_NAMESPACE,
        "receipt_path": spec["receipt_path"],
        "expected_receipt_id": receipt["receipt_id"],
        "validation": spec["validation"],
        "artifact_paths": spec["artifact_paths"],
        "reviewed_base": receipt["reviewed_base"],
        "operator_checklist": [
            "Confirm acceptance criteria in the taskboard for this task.",
            f"Run validation: {spec['validation']}",
            "Confirm you are a human interactive operator (not a model/controller).",
            "Apply with: draft_pdr_manual_completion_seal.py apply "
            f"--task {task_id} --operator-ack",
            "Optionally mark complete with --mark-complete after apply.",
            "Restart the PDR supervisor so authority epoch reloads.",
        ],
        "threat_model_reminder": (
            "Models and automatic controllers must not issue this seal. "
            "decision=sealed requires interactive_user_delegation."
        ),
        "receipt": receipt,
    }
    out = draft_dir / f"{task_id}.draft.json"
    _write_json(out, package)
    package["draft_path"] = str(out.relative_to(root))
    return package


def _update_scheduler_pin(
    root: Path,
    scheduler_rel: str,
    task_id: str,
    receipt_id: str,
) -> None:
    path = root / scheduler_rel
    payload = json.loads(path.read_text(encoding="utf-8"))
    seals = payload.setdefault("manual_completion_seals", {})
    if task_id not in seals:
        raise SystemExit(
            f"{task_id} missing from scheduler manual_completion_seals; "
            "wire the seal slot before applying"
        )
    seals[task_id]["expected_receipt_id"] = receipt_id
    # Keep artifact / grant metadata aligned with the closed script specs.
    spec = TASK_SPECS[task_id]
    seals[task_id]["receipt_path"] = spec["receipt_path"]
    seals[task_id]["schema"] = spec["schema"]
    seals[task_id]["interface"] = spec["interface"]
    seals[task_id]["policy_revision"] = spec["policy_revision"]
    seals[task_id]["artifact_paths"] = dict(spec["artifact_paths"])
    seals[task_id]["grant_type"] = spec["grant_type"]
    seals[task_id]["grant_action"] = spec["grant_action"]
    seals[task_id]["grant_claims"] = dict(spec.get("grant_claims") or {})
    seals[task_id]["reviewed_base_claims"] = dict(
        spec.get("reviewed_base_claims") or {}
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mark_todo_complete(root: Path, todo_rel: str, task_id: str) -> None:
    path = root / todo_rel
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(## {re.escape(task_id)}\b.*?\n- Status: )pending(\n)",
        re.S,
    )
    updated, count = pattern.subn(r"\1completed\2", text, count=1)
    if count != 1:
        # Allow already-completed boards.
        if re.search(
            rf"## {re.escape(task_id)}\b.*?\n- Status: completed\n",
            text,
            re.S,
        ):
            return
        raise SystemExit(
            f"could not mark {task_id} completed in {todo_rel} "
            "(expected Status: pending)"
        )
    path.write_text(updated, encoding="utf-8")


def apply_seal(
    root: Path,
    task_id: str,
    *,
    scheduler_rel: str,
    todo_rel: str,
    mark_complete: bool,
    draft_dir: Path,
) -> dict[str, Any]:
    package = draft_package(root, task_id, draft_dir)
    receipt = package["receipt"]
    spec = TASK_SPECS[task_id]
    seal_path = root / spec["receipt_path"]
    _write_json(seal_path, receipt)
    _update_scheduler_pin(
        root,
        scheduler_rel,
        task_id,
        str(receipt["receipt_id"]),
    )
    try:
        verified = verify_manual_completion_seal(
            str(spec["receipt_path"]),
            repo_root=root,
            task_id=task_id,
            board_namespace=BOARD_NAMESPACE,
            schema=str(spec["schema"]),
            interface=str(spec["interface"]),
            policy_revision=str(spec["policy_revision"]),
            expected_receipt_id=str(receipt["receipt_id"]),
            artifact_paths=dict(spec["artifact_paths"]),
            grant_type=str(spec["grant_type"]),
            grant_action=str(spec["grant_action"]),
            reviewed_base_claims=dict(spec.get("reviewed_base_claims") or {}),
            grant_claims=dict(spec.get("grant_claims") or {}),
        )
    except ManualCompletionSealError as exc:
        raise SystemExit(f"seal verification failed: {exc}") from exc

    if mark_complete:
        _mark_todo_complete(root, todo_rel, task_id)
        # Load scheduler only after complete so activation path is exercised.
        profile = load_supervisor_scheduler_config(
            root / scheduler_rel,
            repo_root=root,
        )
        activated = set(profile.get("activated_protected_task_ids") or ())
        if task_id not in activated:
            raise SystemExit(
                f"{task_id} seal wrote but was not activated by scheduler load"
            )
        package["activated"] = True
        package["manual_completion_authority_required_task_ids"] = list(
            profile.get("manual_completion_authority_required_task_ids") or ()
        )
    else:
        package["activated"] = False

    package["verified_receipt_id"] = verified["receipt_id"]
    package["seal_path"] = spec["receipt_path"]
    package["applied"] = True
    package["marked_complete"] = mark_complete
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draft or apply operator manual-completion seals for PDR-060/PDR-072"
        )
    )
    parser.add_argument(
        "command",
        choices=("draft", "apply", "status"),
        help="draft packages, apply with operator ack, or show pin status",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=sorted(TASK_SPECS),
        help="Task id (repeatable). Default: all supported tasks.",
    )
    parser.add_argument(
        "--operator-ack",
        action="store_true",
        help="Required human acknowledgement for apply/mark-complete",
    )
    parser.add_argument(
        "--mark-complete",
        action="store_true",
        help="After apply, mark Status: completed on the taskboard",
    )
    parser.add_argument(
        "--scheduler-config",
        default=DEFAULT_SCHEDULER,
        help="Repository-relative scheduler config path",
    )
    parser.add_argument(
        "--todo-path",
        default=DEFAULT_TODO,
        help="Repository-relative taskboard path",
    )
    parser.add_argument(
        "--draft-dir",
        default=DEFAULT_DRAFT_DIR,
        help="Repository-relative draft output directory",
    )
    parser.add_argument(
        "--update-pins",
        action="store_true",
        help=(
            "With draft: rewrite scheduler expected_receipt_id pins from the "
            "current tree without writing production seals"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    tasks = list(args.tasks) if args.tasks else sorted(TASK_SPECS)
    draft_dir = root / args.draft_dir

    if args.command == "status":
        scheduler = json.loads(
            (root / args.scheduler_config).read_text(encoding="utf-8")
        )
        seals = scheduler.get("manual_completion_seals") or {}
        todo = (root / args.todo_path).read_text(encoding="utf-8")
        report: dict[str, Any] = {"tasks": {}}
        for task_id in tasks:
            spec = TASK_SPECS[task_id]
            seal_path = root / spec["receipt_path"]
            status_match = re.search(
                rf"## {re.escape(task_id)}\b.*?\n- Status: (\S+)",
                todo,
                re.S,
            )
            pin = (seals.get(task_id) or {}).get("expected_receipt_id")
            draft_receipt = build_receipt(root, task_id)
            report["tasks"][task_id] = {
                "board_status": status_match.group(1) if status_match else None,
                "seal_exists": seal_path.is_file(),
                "scheduler_pin": pin,
                "current_receipt_id": draft_receipt["receipt_id"],
                "pin_matches_current_tree": pin == draft_receipt["receipt_id"],
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "draft":
        packages = [
            draft_package(root, task_id, draft_dir) for task_id in tasks
        ]
        if args.update_pins:
            for item in packages:
                _update_scheduler_pin(
                    root,
                    args.scheduler_config,
                    item["task_id"],
                    str(item["expected_receipt_id"]),
                )
        print(
            json.dumps(
                {
                    "command": "draft",
                    "count": len(packages),
                    "pins_updated": bool(args.update_pins),
                    "packages": [
                        {
                            "task_id": item["task_id"],
                            "draft_path": item["draft_path"],
                            "expected_receipt_id": item["expected_receipt_id"],
                            "validation": item["validation"],
                        }
                        for item in packages
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    # apply
    if not args.operator_ack:
        raise SystemExit(
            "apply requires --operator-ack from a human interactive operator"
        )
    if args.mark_complete and not args.operator_ack:
        raise SystemExit("--mark-complete requires --operator-ack")

    results = []
    for task_id in tasks:
        results.append(
            apply_seal(
                root,
                task_id,
                scheduler_rel=args.scheduler_config,
                todo_rel=args.todo_path,
                mark_complete=bool(args.mark_complete),
                draft_dir=draft_dir,
            )
        )
    print(
        json.dumps(
            {
                "command": "apply",
                "operator_ack": True,
                "results": [
                    {
                        "task_id": item["task_id"],
                        "seal_path": item["seal_path"],
                        "verified_receipt_id": item["verified_receipt_id"],
                        "marked_complete": item["marked_complete"],
                        "activated": item.get("activated"),
                        "draft_path": item["draft_path"],
                    }
                    for item in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
