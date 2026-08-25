#!/usr/bin/env python3
"""Bootstrap and operate the VRIF DuckDB + Quack control plane.

The authority split is deliberately narrow:

* ``DatabaseTaskSource@1`` over DuckDB is transactional task/goal authority.
* one fenced loopback Quack process exclusively owns the DuckDB file while
  supervisors are running;
* DuckLake is an optional, rebuildable history projection and is never read by
  readiness, completion, promotion, or release gates.

The Markdown plan, objectives, and task board are immutable bootstrap inputs.
This operator never mutates their status fields and never publishes the raw
Quack authentication token.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG: Final = Path("config/agent_supervisor_residual_intelligence_scheduler.json")
RUNTIME_RELATIVE: Final = Path("data/agent_supervisor/residual_intelligence_foundry")
BOOTSTRAP_RECEIPT_NAME: Final = "bootstrap-materialization.json"
DUCKLAKE_RECEIPT_NAME: Final = "ducklake-history-projection.json"
OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-foundry-operator@1"
)
POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-foundry-population@1"
)
BOOTSTRAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-foundry-bootstrap@1"
)
DUCKLAKE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-ducklake-projection@1"
)
OWNER_RESTART_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-restart-admission@1"
)
OWNER_RESTART_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-restart-receipt@1"
)
OWNER_DATABASE_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-database-verification@1"
)
SUPERVISOR_LAUNCH_ACK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-supervisor-launch-ack@1"
)
DATABASE_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/database-task-source@1"
)
OWNER_RESTART_ALLOWED_SOURCE_FIELDS: Final = frozenset(
    {
        "accelerator_required_ancestor",
        "accelerator_planning_revision",
        "accelerator_planning_tree",
    }
)
OWNER_RESTART_CONFIG_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-config-transition@1"
)
VRIF_RUNTIME_HARDENING_CONFIG_IDENTITY: Final = (
    "sha256:db657932421d3f40dcde15da782a9d887ad738dde90cb84066773ee2dcc79bde"
)
VRIF_RUNTIME_HARDENING_PROTECTED_INSERTIONS: Final = (
    (
        "ipfs_accelerate_py/agent_supervisor/runtime/process_security.py",
        ("ipfs_accelerate_py/agent_supervisor/runtime/vrif_runtime_settlement.py",),
    ),
    (
        "test/api/residual_intelligence/test_board.py",
        (
            "test/api/residual_intelligence/test_goal_authority.py",
            "test/api/residual_intelligence/test_runtime_settlement.py",
        ),
    ),
)
GOAL_RE: Final = re.compile(r"^## (VRIF-G\d{3}) (.+)$", re.MULTILINE)
QUACK_ENDPOINT_RE: Final = re.compile(
    r"^quack:(?://)?(127(?:\.\d{1,3}){3}|localhost):(\d{1,5})$",
    re.IGNORECASE,
)
READY_STATUSES: Final = (
    "proposed",
    "admitted",
    "pending",
    "ready",
    "todo",
    "queued",
    "retrying",
)
COMPLETED_STATUSES: Final = ("completed", "skipped", "complete", "done")
ACTIVE_STATUSES: Final = ("claimed", "in_progress", "running")
TERMINAL_STATUSES: Final = (
    *COMPLETED_STATUSES,
    "cancelled",
    "failed",
    "quarantined",
    "rejected",
)
VRIF_ROOT_COMPLETION_POLICY_FIELDS: Final = (
    "all_task_dependencies_terminal_required",
    "goal_completion_contracts_required",
    "current_tree_required",
    "active_mutating_claims_empty_required",
    "merge_queue_settled_required",
    "blocking_obligations_empty_required",
    "required_receipts_and_seals_verify",
    "non_success_terminals_never_report_success",
    "ducklake_outage_cannot_block_core_completion",
    "final_report_required",
)
VRIF_RELEASE_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-release-report@2"
)
OWNER_COMMAND_ENVELOPE_MAX_BYTES: Final = 1_048_576
TYPED_DEFERRAL_PROVIDER_CANARY_MAX_BYTES: Final = 8 * 1024 * 1024
TYPED_DEFERRAL_PROVIDER_CANARY_TIMEOUT_SECONDS: Final = 600
VRIF_RUNTIME_SETTLEMENT_LOCK_TIMEOUT_SECONDS: Final[float] = 1.0
TYPED_DEFERRAL_RECOVERY_PRODUCTION_PATHS: Final = frozenset(
    {
        "ipfs_accelerate_py/agent_implementation_route.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        (
            "ipfs_accelerate_py/agent_supervisor/task_sources/"
            "database_task_source.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_daemon.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/validation/"
            "project_dependency_preflight.py"
        ),
    }
)


class OperatorError(RuntimeError):
    """Fail-closed VRIF operator error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _vrif_frozen_benchmark_contract(
    *,
    task_families: Sequence[str],
    source_commit: str,
    source_tree: str,
    split_root: str,
    base_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Construct the one closed, content-addressed no-training benchmark.

    The program has no admitted training rows, model, or learned tokenizer.  A
    truthful frozen benchmark therefore seals the full family/partition/fault
    schedule and an all-abstain, not-run paired result; it does not invent case
    payloads or learned capability.  The returned cases are provider-computable
    from admitted Git identities and can be independently reconstructed by the
    owner.
    """

    expected_base_keys = {
        "repository_states",
        "objective_revisions",
        "operation_catalog",
        "provider_policy",
        "tokenizer",
        "model_versions",
        "validation_policy",
    }
    if (
        set(base_bindings) != expected_base_keys
        or not task_families
        or len(set(task_families)) != len(task_families)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or not split_root
    ):
        raise OperatorError("VRIF frozen benchmark inputs are not exact")
    partitions = ["training", "development", "held_out", "adversarial"]
    case_kinds = ["boundary", "negative", "cross_repository", "unknown_ood"]
    schedule_entries: list[dict[str, Any]] = []
    for family in task_families:
        for partition, kind in zip(partitions, case_kinds, strict=True):
            group_body = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "residual-benchmark-lineage-group@1"
                ),
                "family": family,
                "partition": partition,
                "kind": kind,
                "source_tree": source_tree,
                "split_root": split_root,
            }
            schedule_entries.append(
                {
                    "family": family,
                    "partition": partition,
                    "kind": kind,
                    "hidden_test": partition in {"held_out", "adversarial"},
                    "group_id": _identity(group_body),
                }
            )
    fault_schedule: dict[str, Any] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "residual-benchmark-fault-schedule@1"
        ),
        "source_tree": source_tree,
        "split_root": split_root,
        "entries": schedule_entries,
    }
    fault_schedule["schedule_id"] = _identity(fault_schedule)
    bindings = dict(base_bindings)
    bindings["fault_schedule"] = str(fault_schedule["schedule_id"])
    binding_set_id = _identity(bindings)

    cases: list[dict[str, Any]] = []
    for scheduled in schedule_entries:
        input_contract = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "residual-benchmark-unavailable-input@1"
            ),
            "family": scheduled["family"],
            "partition": scheduled["partition"],
            "kind": scheduled["kind"],
            "group_id": scheduled["group_id"],
            "source_tree": source_tree,
            "disposition": "payload_unavailable_training_unavailable",
        }
        case_body: dict[str, Any] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "residual-frozen-benchmark-case@2"
            ),
            **scheduled,
            "input_identity": _identity(input_contract),
            "input_disposition": "payload_unavailable_training_unavailable",
            "expected_outcome": "CAPABILITY_UNAVAILABLE",
        }
        case_body["case_id"] = _identity(
            {**case_body, "freeze_binding_set_id": binding_set_id}
        )
        cases.append(case_body)
    case_root = _identity(cases)
    denominators = {family: len(partitions) for family in task_families}
    scores = {
        "accept": 0,
        "abstain": len(cases),
        "total": len(cases),
        "denominators_by_family": denominators,
    }
    source = {"commit": source_commit, "tree": source_tree}
    paired_baseline: dict[str, Any] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "residual-paired-benchmark-baseline@2"
        ),
        "prior_source": source,
        "evaluated_source": source,
        "comparison_disposition": (
            "identical_no_candidate_training_unavailable"
        ),
        "case_payload_disposition": "payload_unavailable_training_unavailable",
        "evaluation_disposition": "all_abstain_not_run",
        "case_count": len(cases),
        "case_root": case_root,
        "binding_set_id": binding_set_id,
        "before": scores,
        "after": scores,
        "candidate_only": True,
        "training_performed": False,
    }
    paired_baseline["paired_baseline_id"] = _identity(paired_baseline)
    benchmark_freeze: dict[str, Any] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/residual-benchmark-freeze@2"
        ),
        "state": "frozen",
        "source": source,
        "case_payload_disposition": "payload_unavailable_training_unavailable",
        "evaluation_disposition": "all_abstain_not_run",
        "bindings": bindings,
        "binding_set_id": binding_set_id,
        "fault_schedule": fault_schedule,
        "case_count": len(cases),
        "case_root": case_root,
        "paired_baseline": paired_baseline,
    }
    benchmark_freeze["freeze_id"] = _identity(benchmark_freeze)
    return {
        "partitions": partitions,
        "case_kinds": case_kinds,
        "cases": cases,
        "scores": scores,
        "fault_schedule": fault_schedule,
        "bindings": bindings,
        "binding_set_id": binding_set_id,
        "paired_baseline": paired_baseline,
        "benchmark_freeze": benchmark_freeze,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"JSON root must be an object: {path}")
    return value


def _safe_path(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise OperatorError(f"{field} must be a safe repository-relative path")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OperatorError(f"{field} escapes repository") from exc
    return resolved


def _git(*arguments: str, check: bool = True, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if check and completed.returncode != 0:
        error = completed.stderr or completed.stdout
        if isinstance(error, bytes):
            error = error.decode("utf-8", errors="replace")
        raise OperatorError(f"git {' '.join(arguments)} failed: {str(error).strip()}")
    return completed.stdout


def _assert_clean_current_tree(config: Mapping[str, Any]) -> tuple[str, str]:
    status_output = str(_git("status", "--porcelain=v1", "--untracked-files=all")).strip()
    if status_output:
        raise OperatorError(
            "refusing to materialize from a dirty worktree; commit the exact "
            "plan, board, configuration, validator, and operator first"
        )
    head = str(_git("rev-parse", "HEAD")).strip()
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip()
    branch = str(_git("branch", "--show-current")).strip()
    required_branch = str(config.get("merge_target_branch") or "").strip()
    if required_branch and branch != required_branch:
        raise OperatorError(
            f"execution branch {branch!r} differs from configured branch {required_branch!r}"
        )
    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ancestor = str(binding.get("accelerator_required_ancestor") or "").strip()
    if ancestor:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise OperatorError("configured accelerator base is not an ancestor")
    return head, tree


def _tracked_bytes(path: Path, *, head: str) -> bytes:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError(f"authority input escapes repository: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise OperatorError(f"authority input is not a regular file: {relative}")
    working = path.read_bytes()
    recorded = _git("show", f"{head}:{relative}", binary=True)
    if not isinstance(recorded, bytes) or working != recorded:
        raise OperatorError(f"authority input differs from current HEAD: {relative}")
    return working


def _git_commit_tree(commit: Any, *, field: str) -> str:
    revision = str(commit or "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise OperatorError(f"{field} must be an exact Git commit")
    try:
        object_type = str(_git("cat-file", "-t", revision)).strip()
        tree = str(_git("rev-parse", f"{revision}^{{tree}}")).strip()
    except OperatorError as exc:
        raise OperatorError(f"{field} is not available in the repository") from exc
    if object_type != "commit" or re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise OperatorError(f"{field} is not an exact Git commit")
    return tree


def _git_is_ancestor(ancestor: str, descendant: str, *, field: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    if completed.returncode == 1:
        raise OperatorError(f"{field} is not monotonic")
    raise OperatorError(f"cannot verify {field}")


def _git_blob_at(*, head: str, path: Path, field: str) -> bytes:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError(f"{field} escapes repository") from exc
    try:
        value = _git("show", f"{head}:{relative}", binary=True)
    except OperatorError as exc:
        raise OperatorError(f"{field} is absent from the bootstrap source") from exc
    if not isinstance(value, bytes):
        raise OperatorError(f"{field} could not be read as bytes")
    return value


def _json_mapping_bytes(value: bytes, *, field: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded_object: dict[str, Any] = {}
        for key, item in pairs:
            if key in decoded_object:
                raise ValueError("duplicate JSON object key")
            decoded_object[key] = item
        return decoded_object

    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OperatorError(f"{field} must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise OperatorError(f"{field} must be a JSON object")
    return decoded


def _restart_source_binding(
    config: Mapping[str, Any],
    *,
    label: str,
    source_head: str,
) -> dict[str, str]:
    raw = config.get("source_binding")
    if not isinstance(raw, Mapping):
        raise OperatorError(f"{label} source_binding must be an object")
    values = {
        field: str(raw.get(field) or "").strip()
        for field in OWNER_RESTART_ALLOWED_SOURCE_FIELDS
    }
    planning_revision = values["accelerator_planning_revision"]
    required_ancestor = values["accelerator_required_ancestor"]
    planning_tree = values["accelerator_planning_tree"]
    if required_ancestor != planning_revision:
        raise OperatorError(
            f"{label} planning revision and required ancestor must be exact"
        )
    observed_tree = _git_commit_tree(
        planning_revision,
        field=f"{label}.source_binding.accelerator_planning_revision",
    )
    if planning_tree != observed_tree:
        raise OperatorError(f"{label} planning tree does not match its commit")
    _git_is_ancestor(
        planning_revision,
        source_head,
        field=f"{label} planning revision ancestry",
    )
    return values


def _restart_static_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_bytes(config))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperatorError(f"{label} config is not canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise OperatorError(f"{label} config must be an object")
    source_binding = normalized.get("source_binding")
    if not isinstance(source_binding, dict):
        raise OperatorError(f"{label} source_binding must be an object")
    for field in OWNER_RESTART_ALLOWED_SOURCE_FIELDS:
        source_binding.pop(field, None)
    return normalized


def _exact_vrif_runtime_hardening_transition(
    bootstrap_static: Mapping[str, Any],
    current_static: Mapping[str, Any],
    *,
    planning_revision: str,
    planning_tree: str,
    config_path: Path,
) -> dict[str, Any]:
    """Admit the one frozen runtime-settlement hardening migration.

    The bootstrap configuration remains the immutable default.  This narrow
    exception admits only the content-pinned settlement block and the exact
    three protection insertions that make its implementation and tests
    immutable to workers.  The resulting static config must also have existed
    at the explicitly advanced planning revision, so a later descendant cannot
    smuggle an unsealed static change into an owner restart.
    """

    if "runtime_settlement" in bootstrap_static:
        raise OperatorError(
            "current config changes fields outside the admitted source-binding lineage"
        )
    runtime_settlement = current_static.get("runtime_settlement")
    if (
        not isinstance(runtime_settlement, Mapping)
        or _identity(runtime_settlement) != VRIF_RUNTIME_HARDENING_CONFIG_IDENTITY
    ):
        raise OperatorError(
            "current config changes fields outside the admitted source-binding lineage"
        )

    bootstrap_paths_raw = bootstrap_static.get("protected_paths")
    current_paths_raw = current_static.get("protected_paths")
    if (
        not isinstance(bootstrap_paths_raw, list)
        or not isinstance(current_paths_raw, list)
        or any(type(item) is not str or not item for item in bootstrap_paths_raw)
        or any(type(item) is not str or not item for item in current_paths_raw)
        or len(set(bootstrap_paths_raw)) != len(bootstrap_paths_raw)
        or len(set(current_paths_raw)) != len(current_paths_raw)
    ):
        raise OperatorError(
            "current config changes fields outside the admitted source-binding lineage"
        )
    expected_paths = list(bootstrap_paths_raw)
    for anchor, additions in VRIF_RUNTIME_HARDENING_PROTECTED_INSERTIONS:
        if expected_paths.count(anchor) != 1 or any(
            addition in expected_paths for addition in additions
        ):
            raise OperatorError(
                "current config changes fields outside the admitted source-binding lineage"
            )
        offset = expected_paths.index(anchor) + 1
        expected_paths[offset:offset] = list(additions)

    expected_current = dict(bootstrap_static)
    expected_current["protected_paths"] = expected_paths
    expected_current["runtime_settlement"] = dict(runtime_settlement)
    if _canonical_bytes(expected_current) != _canonical_bytes(current_static):
        raise OperatorError(
            "current config changes fields outside the admitted source-binding lineage"
        )

    planned_config = _json_mapping_bytes(
        _git_blob_at(
            head=planning_revision,
            path=config_path,
            field="current planning config",
        ),
        field="current planning config",
    )
    planned_static = _restart_static_config(planned_config, label="current planning")
    if _canonical_bytes(planned_static) != _canonical_bytes(current_static):
        raise OperatorError(
            "runtime hardening config is not sealed by the planning revision"
        )

    transition: dict[str, Any] = {
        "schema": OWNER_RESTART_CONFIG_TRANSITION_SCHEMA,
        "mode": "exact_runtime_settlement_hardening",
        "bootstrap_static_config_identity": _identity(bootstrap_static),
        "current_static_config_identity": _identity(current_static),
        "planning_revision": planning_revision,
        "planning_tree": planning_tree,
        "planning_config_identity": _identity(planned_config),
        "runtime_settlement_identity": VRIF_RUNTIME_HARDENING_CONFIG_IDENTITY,
        "protected_path_insertions": [
            addition
            for _anchor, additions in VRIF_RUNTIME_HARDENING_PROTECTED_INSERTIONS
            for addition in additions
        ],
    }
    transition["transition_id"] = _identity(transition)
    return transition


def _owner_restart_admission(
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Admit only the sealed bootstrap tree or a bounded descendant restart."""

    current_head, current_tree = _assert_clean_current_tree(config)
    bootstrap = _json_object(paths["bootstrap_receipt"])
    if bootstrap.get("schema") != BOOTSTRAP_SCHEMA:
        raise OperatorError("owner restart bootstrap schema is not admitted")
    bootstrap_receipt_id = str(bootstrap.get("bootstrap_receipt_id") or "").strip()
    bootstrap_body = dict(bootstrap)
    bootstrap_body.pop("bootstrap_receipt_id", None)
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", bootstrap_receipt_id) is None
        or _identity(bootstrap_body) != bootstrap_receipt_id
    ):
        raise OperatorError("owner restart bootstrap receipt identity is invalid")

    bootstrap_head = str(bootstrap.get("source_head") or "").strip()
    bootstrap_tree = str(bootstrap.get("repository_tree_id") or "").strip()
    observed_bootstrap_tree = _git_commit_tree(
        bootstrap_head,
        field="bootstrap source_head",
    )
    if bootstrap_tree != observed_bootstrap_tree:
        raise OperatorError("bootstrap source tree does not match its commit")
    _git_is_ancestor(
        bootstrap_head,
        current_head,
        field="bootstrap-to-current source ancestry",
    )

    plan_root_cid = str(bootstrap.get("plan_root_cid") or "").strip()
    database_receipt = bootstrap.get("database_task_source_receipt")
    if not isinstance(database_receipt, Mapping):
        raise OperatorError("bootstrap database task-source receipt is absent")
    if str(database_receipt.get("schema") or "") != DATABASE_TASK_SOURCE_SCHEMA:
        raise OperatorError("bootstrap database task-source schema is not admitted")
    if (
        str(database_receipt.get("repository_tree_id") or "") != bootstrap_tree
        or str(database_receipt.get("plan_root_cid") or "") != plan_root_cid
    ):
        raise OperatorError("bootstrap database authority roots are inconsistent")
    task_cids_raw = database_receipt.get("task_cids")
    if (
        not isinstance(task_cids_raw, Sequence)
        or isinstance(task_cids_raw, (str, bytes, bytearray))
    ):
        raise OperatorError("bootstrap database task identities are absent")
    task_cids = tuple(str(item or "").strip() for item in task_cids_raw)
    if not task_cids or any(not item for item in task_cids):
        raise OperatorError("bootstrap database task identities are invalid")
    if len(set(task_cids)) != len(task_cids):
        raise OperatorError("bootstrap database task identities are not unique")
    try:
        task_count = int(database_receipt.get("task_count"))
        goal_count = int(database_receipt.get("goal_count"))
        plan_count = int(database_receipt.get("plan_count"))
    except (TypeError, ValueError) as exc:
        raise OperatorError("bootstrap database authority counts are invalid") from exc
    if task_count != len(task_cids) or goal_count < 1 or plan_count < 1:
        raise OperatorError("bootstrap database authority counts are inconsistent")

    source_identities = bootstrap.get("source_identities")
    if not isinstance(source_identities, Mapping):
        raise OperatorError("bootstrap source identities are absent")
    source_paths = {
        "config": board.config_path,
        "taskboard": board.path(board.taskboard_path),
        "objectives": board.path(board.objectives_path),
        "plan": board.path(board.plan_path),
        "validator": board.path(board.validator_path),
    }
    if set(source_identities) != set(source_paths):
        raise OperatorError("bootstrap source identity key set is not exact")
    bootstrap_sources: dict[str, bytes] = {}
    current_sources: dict[str, bytes] = {}
    for name, path in source_paths.items():
        expected = str(source_identities.get(name) or "").strip()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
            raise OperatorError(f"bootstrap {name} source identity is invalid")
        bootstrap_bytes = _git_blob_at(
            head=bootstrap_head,
            path=path,
            field=f"bootstrap {name}",
        )
        if _identity(bootstrap_bytes) != expected:
            raise OperatorError(f"bootstrap {name} bytes differ from their seal")
        current_bytes = _tracked_bytes(path, head=current_head)
        if name != "config" and _identity(current_bytes) != expected:
            raise OperatorError(f"current {name} bytes differ from bootstrap")
        bootstrap_sources[name] = bootstrap_bytes
        current_sources[name] = current_bytes

    bootstrap_config = _json_mapping_bytes(
        bootstrap_sources["config"],
        field="bootstrap config",
    )
    current_config = _json_mapping_bytes(
        current_sources["config"],
        field="current config",
    )
    if _canonical_bytes(current_config) != _canonical_bytes(config):
        raise OperatorError("loaded config differs from tracked current config")
    bootstrap_binding = _restart_source_binding(
        bootstrap_config,
        label="bootstrap",
        source_head=bootstrap_head,
    )
    current_binding = _restart_source_binding(
        current_config,
        label="current",
        source_head=current_head,
    )
    _git_is_ancestor(
        bootstrap_binding["accelerator_planning_revision"],
        current_binding["accelerator_planning_revision"],
        field="planning revision lineage",
    )
    bootstrap_static = _restart_static_config(
        bootstrap_config,
        label="bootstrap",
    )
    current_static = _restart_static_config(current_config, label="current")
    if _canonical_bytes(bootstrap_static) == _canonical_bytes(current_static):
        config_transition: dict[str, Any] = {
            "schema": OWNER_RESTART_CONFIG_TRANSITION_SCHEMA,
            "mode": "exact_static_config",
            "bootstrap_static_config_identity": _identity(bootstrap_static),
            "current_static_config_identity": _identity(current_static),
        }
        config_transition["transition_id"] = _identity(config_transition)
    else:
        if (
            current_binding["accelerator_planning_revision"]
            == bootstrap_binding["accelerator_planning_revision"]
        ):
            raise OperatorError(
                "runtime hardening config requires an advanced planning revision"
            )
        config_transition = _exact_vrif_runtime_hardening_transition(
            bootstrap_static,
            current_static,
            planning_revision=current_binding["accelerator_planning_revision"],
            planning_tree=current_binding["accelerator_planning_tree"],
            config_path=board.config_path,
        )

    mode = (
        "exact_bootstrap"
        if current_head == bootstrap_head and current_tree == bootstrap_tree
        else "verified_descendant"
    )
    admission: dict[str, Any] = {
        "schema": OWNER_RESTART_ADMISSION_SCHEMA,
        "mode": mode,
        "bootstrap_receipt_id": bootstrap_receipt_id,
        "bootstrap_source_head": bootstrap_head,
        "bootstrap_source_tree": bootstrap_tree,
        "current_source_head": current_head,
        "current_source_tree": current_tree,
        "plan_root_cid": plan_root_cid,
        "bootstrap_config_identity": _identity(bootstrap_sources["config"]),
        "current_config_identity": _identity(current_sources["config"]),
        "static_config_identity": _identity(bootstrap_static),
        "config_transition": config_transition,
        "source_identities": {
            name: str(source_identities[name]) for name in sorted(source_paths)
        },
        "planning_lineage": {
            "bootstrap_revision": bootstrap_binding[
                "accelerator_planning_revision"
            ],
            "bootstrap_tree": bootstrap_binding["accelerator_planning_tree"],
            "current_revision": current_binding["accelerator_planning_revision"],
            "current_tree": current_binding["accelerator_planning_tree"],
        },
        "authority_config_identity": _identity(
            {
                "database_program": current_config.get("database_program"),
                "runtime_paths": current_config.get("runtime_paths"),
            }
        ),
        "database_authority": {
            "receipt_identity": _identity(database_receipt),
            "schema": DATABASE_TASK_SOURCE_SCHEMA,
            "repository_tree_id": bootstrap_tree,
            "source_head": bootstrap_head,
            "plan_root_cid": plan_root_cid,
            "projection_cid": str(database_receipt.get("projection_cid") or ""),
            "task_cids": sorted(task_cids),
            "task_count": task_count,
            "goal_count": goal_count,
            "plan_count": plan_count,
        },
    }
    admission["admission_id"] = _identity(admission)
    return admission


def _owner_restart_prior_status(path: Path) -> dict[str, Any]:
    """Admit only an absent, stopped, or provably dead prior owner."""

    try:
        observed = path.lstat()
    except FileNotFoundError:
        return {
            "state": "absent",
            "status_identity": "",
            "server_id": "",
            "database_uuid": "",
            "store_id": "",
            "schema_revision": 0,
            "schema_fingerprint": "",
            "generation": 0,
            "fence_epoch": 0,
            "process_birth_id": "",
        }
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise OperatorError("prior state-owner status is not a private regular file")
    payload = _json_object(path)
    if (
        payload.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/quack-state-server@1"
        or payload.get("interface") != "QuackStateServer@1"
    ):
        raise OperatorError("prior state-owner status contract is not admitted")
    lifecycle = str(payload.get("lifecycle") or "").strip()
    liveness = _owner_liveness(payload)
    if lifecycle == "ready" and liveness in {"alive", "unknown"}:
        raise OperatorError(
            f"prior ready state owner has {liveness} process-birth liveness"
        )
    if lifecycle != "stopped" and liveness != "dead":
        raise OperatorError("prior state-owner status is neither stopped nor dead")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        if lifecycle != "stopped":
            raise OperatorError("prior dead state owner has no exact identity")
        identity = {}
    try:
        generation = int(identity.get("generation") or 0)
        fence_epoch = int(identity.get("fence_epoch") or 0)
        schema_revision = int(identity.get("schema_revision") or 0)
    except (TypeError, ValueError) as exc:
        raise OperatorError("prior state-owner identity counters are invalid") from exc
    if identity and (
        not str(identity.get("server_id") or "").strip()
        or not str(identity.get("database_uuid") or "").strip()
        or not str(identity.get("store_id") or "").strip()
        or generation < 1
        or fence_epoch < 1
        or schema_revision < 1
    ):
        raise OperatorError("prior state-owner identity is incomplete")
    return {
        "state": "stopped" if lifecycle == "stopped" else "dead",
        "lifecycle": lifecycle,
        "liveness": liveness,
        "status_identity": _identity(payload),
        "server_id": str(identity.get("server_id") or ""),
        "database_uuid": str(identity.get("database_uuid") or ""),
        "store_id": str(identity.get("store_id") or ""),
        "schema_revision": schema_revision,
        "schema_fingerprint": str(identity.get("schema_fingerprint") or ""),
        "generation": generation,
        "fence_epoch": fence_epoch,
        "process_birth_id": str(identity.get("process_birth_id") or ""),
        "identity": dict(identity),
    }


def _rows(connection: Any, sql: str, parameters: Sequence[Any] = ()) -> list[Any]:
    try:
        return list(connection.execute(sql, list(parameters)).fetchall())
    except Exception as exc:
        raise OperatorError("cannot verify bound state-owner database authority") from exc


def _row_item(row: Any, index: int, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    try:
        return row[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise OperatorError("bound database returned a malformed authority row") from exc


def _owner_database_verification(
    connection: Any,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce the sealed immutable population through the owner connection."""

    authority = admission.get("database_authority")
    if not isinstance(authority, Mapping):
        raise OperatorError("owner restart admission has no database authority")
    expected_tasks_raw = authority.get("task_cids")
    if not isinstance(expected_tasks_raw, Sequence) or isinstance(
        expected_tasks_raw, (str, bytes, bytearray)
    ):
        raise OperatorError("owner restart task authority is malformed")
    expected_tasks = sorted(str(item) for item in expected_tasks_raw)
    task_rows = _rows(
        connection,
        "SELECT task_cid, identity_json FROM tasks ORDER BY task_cid",
    )
    actual_tasks: list[str] = []
    expected_tree = str(authority.get("repository_tree_id") or "")
    for row in task_rows:
        task_cid = str(_row_item(row, 0, "task_cid") or "")
        identity_json = str(_row_item(row, 1, "identity_json") or "")
        try:
            task_identity = json.loads(identity_json)
        except json.JSONDecodeError as exc:
            raise OperatorError("bound database task identity is not JSON") from exc
        if (
            not isinstance(task_identity, Mapping)
            or str(task_identity.get("task_cid") or "") != task_cid
            or str(task_identity.get("repository_tree_id") or "") != expected_tree
        ):
            raise OperatorError("bound database task identity differs from bootstrap")
        actual_tasks.append(task_cid)
    if actual_tasks != expected_tasks or len(actual_tasks) != int(
        authority.get("task_count") or 0
    ):
        raise OperatorError("bound database task population differs from bootstrap")

    goal_rows = _rows(connection, "SELECT goal_cid FROM goals ORDER BY goal_cid")
    if len(goal_rows) != int(authority.get("goal_count") or 0):
        raise OperatorError("bound database goal population differs from bootstrap")
    plan_rows = _rows(connection, "SELECT plan_cid, body_json FROM plans ORDER BY plan_cid")
    if len(plan_rows) != int(authority.get("plan_count") or 0):
        raise OperatorError("bound database plan population differs from bootstrap")
    expected_plan = str(authority.get("plan_root_cid") or "")
    plan_body: Mapping[str, Any] | None = None
    for row in plan_rows:
        if str(_row_item(row, 0, "plan_cid") or "") != expected_plan:
            continue
        raw_body = str(_row_item(row, 1, "body_json") or "")
        try:
            decoded = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise OperatorError("bound database plan body is not JSON") from exc
        if not isinstance(decoded, Mapping):
            raise OperatorError("bound database plan body is not an object")
        plan_body = decoded
        break
    if plan_body is None:
        raise OperatorError("bound database plan root differs from bootstrap")
    if (
        str(plan_body.get("repository_tree_id") or "") != expected_tree
        or str(plan_body.get("source_head") or "")
        != str(authority.get("source_head") or "")
    ):
        raise OperatorError("bound database plan lineage differs from bootstrap")
    projection = {
        "schema": OWNER_DATABASE_VERIFICATION_SCHEMA,
        "bootstrap_database_receipt_identity": str(
            authority.get("receipt_identity") or ""
        ),
        "repository_tree_id": expected_tree,
        "source_head": str(authority.get("source_head") or ""),
        "plan_root_cid": expected_plan,
        "task_cids": actual_tasks,
        "task_count": len(actual_tasks),
        "goal_count": len(goal_rows),
        "plan_count": len(plan_rows),
    }
    projection["verification_id"] = _identity(projection)
    return projection


def _owner_restart_receipt(
    admission: Mapping[str, Any],
    identity: Any,
    *,
    expected_store_id: str,
    prior_owner: Mapping[str, Any],
    database_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an admitted source transition to the newly live server identity."""

    store_id = str(getattr(identity, "store_id", "") or "")
    database_uuid = str(getattr(identity, "database_uuid", "") or "")
    try:
        generation = int(getattr(identity, "generation", 0) or 0)
        fence_epoch = int(getattr(identity, "fence_epoch", 0) or 0)
        schema_revision = int(getattr(identity, "schema_revision", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise OperatorError("new state-owner identity counters are invalid") from exc
    if (
        store_id != expected_store_id
        or not str(getattr(identity, "server_id", "") or "")
        or not database_uuid
        or generation < 1
        or fence_epoch < 1
        or schema_revision < 1
        or not str(getattr(identity, "schema_fingerprint", "") or "")
        or not str(getattr(identity, "process_birth_id", "") or "")
    ):
        raise OperatorError("new state-owner database identity is invalid")
    verification_id = str(database_verification.get("verification_id") or "")
    verification_body = dict(database_verification)
    verification_body.pop("verification_id", None)
    if (
        database_verification.get("schema") != OWNER_DATABASE_VERIFICATION_SCHEMA
        or re.fullmatch(r"sha256:[0-9a-f]{64}", verification_id) is None
        or _identity(verification_body) != verification_id
        or str(database_verification.get("plan_root_cid") or "")
        != str(admission.get("plan_root_cid") or "")
        or str(database_verification.get("repository_tree_id") or "")
        != str(admission.get("bootstrap_source_tree") or "")
    ):
        raise OperatorError("bound database verification is invalid")
    prior_generation = int(prior_owner.get("generation") or 0)
    prior_fence = int(prior_owner.get("fence_epoch") or 0)
    prior_server_id = str(prior_owner.get("server_id") or "")
    if prior_generation:
        if generation <= prior_generation or fence_epoch <= prior_fence:
            raise OperatorError("new state-owner generation does not advance prior owner")
        if prior_server_id == str(getattr(identity, "server_id", "") or ""):
            raise OperatorError("new state-owner server identity was reused")
        if str(prior_owner.get("store_id") or "") != store_id:
            raise OperatorError("new state-owner store differs from prior owner")
        if str(prior_owner.get("database_uuid") or "") != database_uuid:
            raise OperatorError("new state-owner database differs from prior owner")
        if int(prior_owner.get("schema_revision") or 0) != schema_revision:
            raise OperatorError("new state-owner schema differs from prior owner")
        if str(prior_owner.get("schema_fingerprint") or "") != str(
            getattr(identity, "schema_fingerprint", "") or ""
        ):
            raise OperatorError("new state-owner schema fingerprint differs from prior owner")
    elif str(admission.get("mode") or "") == "verified_descendant" and generation <= 1:
        raise OperatorError("descendant owner restart did not advance store generation")
    receipt: dict[str, Any] = {
        "schema": OWNER_RESTART_RECEIPT_SCHEMA,
        "admission_id": str(admission.get("admission_id") or ""),
        "mode": str(admission.get("mode") or ""),
        "bootstrap_receipt_id": str(admission.get("bootstrap_receipt_id") or ""),
        "bootstrap_source_head": str(
            admission.get("bootstrap_source_head") or ""
        ),
        "bootstrap_source_tree": str(
            admission.get("bootstrap_source_tree") or ""
        ),
        "current_source_head": str(admission.get("current_source_head") or ""),
        "current_source_tree": str(admission.get("current_source_tree") or ""),
        "plan_root_cid": str(admission.get("plan_root_cid") or ""),
        "authority_config_identity": str(
            admission.get("authority_config_identity") or ""
        ),
        "prior_state_owner": dict(prior_owner),
        "database_verification": dict(database_verification),
        "state_owner": {
            "server_id": str(getattr(identity, "server_id", "") or ""),
            "store_id": store_id,
            "database_uuid": database_uuid,
            "schema_revision": schema_revision,
            "schema_fingerprint": str(
                getattr(identity, "schema_fingerprint", "") or ""
            ),
            "generation": generation,
            "fence_epoch": fence_epoch,
            "process_birth_id": str(
                getattr(identity, "process_birth_id", "") or ""
            ),
        },
    }
    receipt["receipt_id"] = _identity(receipt)
    return receipt


def _source_forest(config: Mapping[str, Any], *, head: str) -> dict[str, Any]:
    """Verify configured sibling gitlinks without granting write authority."""

    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    nested: list[dict[str, str]] = []
    configured_repositories = (
        (
            "ipfs_datasets",
            ("ipfs_datasets_submodule_path", "datasets_submodule_path"),
            ("ipfs_datasets_planning_revision", "datasets_planning_revision"),
        ),
        (
            "ipfs_kit",
            ("ipfs_kit_submodule_path", "kit_submodule_path"),
            ("ipfs_kit_planning_revision", "kit_planning_revision"),
        ),
        (
            "mcp_plus_plus",
            ("mcp_plus_plus_submodule_path",),
            ("mcp_plus_plus_planning_revision",),
        ),
    )
    for prefix, path_fields, revision_fields in configured_repositories:
        raw_path = next(
            (binding.get(field) for field in path_fields if binding.get(field)),
            None,
        )
        raw_revision = next(
            (binding.get(field) for field in revision_fields if binding.get(field)),
            None,
        )
        if raw_path in (None, "") and raw_revision in (None, ""):
            continue
        if raw_path in (None, "") or raw_revision in (None, ""):
            raise OperatorError(f"{prefix} source binding is incomplete")
        nested_path = _safe_path(
            ROOT,
            raw_path,
            field=f"source_binding.{prefix}_submodule_path",
        )
        if not nested_path.is_dir():
            raise OperatorError(f"{prefix} submodule is not initialized")
        nested_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if nested_status.returncode != 0 or nested_status.stdout.strip():
            raise OperatorError(f"{prefix} nested worktree is not clean")
        nested_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        nested_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        revision = nested_head.stdout.strip()
        tree = nested_tree.stdout.strip()
        if (
            nested_head.returncode != 0
            or nested_tree.returncode != 0
            or revision != str(raw_revision)
            or not tree
        ):
            raise OperatorError(f"{prefix} nested revision differs from its seal")
        relative = nested_path.relative_to(ROOT).as_posix()
        tree_row = str(_git("ls-tree", head, "--", relative)).strip().split()
        if (
            len(tree_row) < 3
            or tree_row[0] != "160000"
            or tree_row[1] != "commit"
            or tree_row[2] != revision
        ):
            raise OperatorError(f"{prefix} gitlink differs from its nested HEAD")
        nested.append(
            {
                "repository": prefix,
                "path": relative,
                "head": revision,
                "tree": tree,
                "access": "read_only_contract_audit",
            }
        )
    result: dict[str, Any] = {
        "source_head": head,
        "nested_repositories": nested,
        "cross_repository_writes": False,
    }
    result["source_forest_root"] = _identity(result)
    return result


def _load_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
    )

    board = load_configured_board(config_path, repo_root=ROOT)
    payload = dict(board.payload)
    if board.task_prefix.removeprefix("## ") != "VRIF-":
        raise OperatorError("VRIF operator requires task_prefix='VRIF-'")
    if board.board_namespace != "agent-supervisor-verified-residual-intelligence-foundry-v1":
        raise OperatorError("scheduler board_namespace is not the VRIF v1 namespace")
    program = board.resolved_database_program()
    if program.authority_mode != "quack" or program.task_source_kind != "duckdb":
        raise OperatorError("VRIF requires DuckDB task authority served through Quack")
    if program.failover_policy != "fail_closed":
        raise OperatorError("VRIF Quack authority must fail closed")
    if QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint) is None:
        raise OperatorError("VRIF Quack endpoint must be a bounded loopback URI")
    projection = payload.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    if int(projection.get("task_count") or -1) != 33:
        raise OperatorError("VRIF v1 requires exactly 33 configured tasks")
    if int(projection.get("goal_count") or -1) != 9:
        raise OperatorError("VRIF v1 requires exactly 9 configured goals")
    return board, payload


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _metadata_value(value: Any) -> str:
    """Accept plain PCAR fields and the board validator's bold field form."""

    text = str(value or "").strip()
    if text.startswith("**"):
        text = text[2:].lstrip()
    return text


def _goal_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        normalize_metadata_key,
    )

    matches = list(GOAL_RE.finditer(text))
    result: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[match.end() : end].splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", 1)
            normalized = normalize_metadata_key(key)
            if normalized in fields:
                raise OperatorError(
                    f"{match.group(1)} contains duplicate metadata field {normalized}"
                )
            fields[normalized] = _metadata_value(value)
        result.append((match.group(1), match.group(2).strip(), fields))
    return result


def _population(board: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
        split_validation_commands,
    )

    head, tree = _assert_clean_current_tree(config)
    source_forest = _source_forest(config, head=head)
    sources = {
        "config": _tracked_bytes(board.config_path, head=head),
        "taskboard": _tracked_bytes(board.path(board.taskboard_path), head=head),
        "objectives": _tracked_bytes(board.path(board.objectives_path), head=head),
        "plan": _tracked_bytes(board.path(board.plan_path), head=head),
        "validator": _tracked_bytes(board.path(board.validator_path), head=head),
    }
    plan_root = content_identity(
        {
            "schema": "vrif-plan-root@1",
            "source_head": head,
            "repository_tree_id": tree,
            "sources": {name: _identity(value) for name, value in sorted(sources.items())},
        }
    )

    objective_text = sources["objectives"].decode("utf-8")
    parsed_goals = _goal_blocks(objective_text)
    if not parsed_goals or parsed_goals[0][0] != "VRIF-G000":
        raise OperatorError("objectives must begin with root VRIF-G000")
    if len({item[0] for item in parsed_goals}) != len(parsed_goals):
        raise OperatorError("objectives contain duplicate goal IDs")
    goal_cids = {
        goal_id: content_identity(
            {
                "goal_id": goal_id,
                "title": title,
                "metadata": fields,
                "plan_root_cid": plan_root,
            }
        )
        for goal_id, title, fields in parsed_goals
    }
    goals: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    observed_goals: set[str] = set()
    for ordinal, (goal_id, title, fields) in enumerate(parsed_goals, start=1):
        parent = str(fields.get("parent") or "").strip()
        if parent and parent not in observed_goals:
            raise OperatorError(f"{goal_id} parent must precede it: {parent}")
        dependencies = _split_csv(fields.get("depends_on"))
        unknown = [item for item in dependencies if item not in goal_cids]
        if unknown:
            raise OperatorError(f"{goal_id} has unknown goal dependencies: {unknown}")
        goal = {
            "goal_cid": goal_cids[goal_id],
            "goal_id": goal_id,
            "goal_alias": goal_id,
            "title": title,
            "ordinal": ordinal,
            "status": str(fields.get("status") or "open").lower(),
            "objective_id": "objective:vrif-root" if goal_id == "VRIF-G000" else "",
            "objective_alias": "VRIF-G000",
            "priority": str(fields.get("priority") or "P0"),
            "body": dict(fields),
        }
        if parent:
            goal["parent_goal_cid"] = goal_cids[parent]
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in dependencies:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
        goals.append(goal)
        observed_goals.add(goal_id)

    task_text = sources["taskboard"].decode("utf-8")
    parsed_tasks = parse_todo_blocks(task_text, task_header_prefix="## VRIF-")
    parsed_tasks = [
        (
            task_id,
            title,
            source_line,
            {key: _metadata_value(value) for key, value in fields.items()},
        )
        for task_id, title, source_line, fields in parsed_tasks
    ]
    if not parsed_tasks:
        raise OperatorError("task board contains no VRIF tasks")
    task_ids = [item[0] for item in parsed_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise OperatorError("task board contains duplicate VRIF task IDs")
    expected_task_ids = [f"VRIF-{ordinal:03d}" for ordinal in range(33)]
    if task_ids != expected_task_ids:
        raise OperatorError("task board must contain ordered VRIF-000 through VRIF-032")
    task_cids = {
        task_id: content_identity(
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "metadata": fields,
                "plan_root_cid": plan_root,
                "repository_tree_id": tree,
            }
        )
        for task_id, title, source_line, fields in parsed_tasks
    }
    tasks: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    for ordinal, (task_id, title, source_line, fields) in enumerate(parsed_tasks, start=1):
        dependencies = _split_csv(fields.get("depends_on"))
        unknown = [item for item in dependencies if item not in task_cids]
        if unknown:
            raise OperatorError(f"{task_id} has unknown dependencies: {unknown}")
        future = [item for item in dependencies if item not in observed_tasks]
        if future:
            raise OperatorError(
                f"{task_id} dependencies must precede it for atomic ingestion: {future}"
            )
        goal_id = str(
            fields.get("subgoal_id") or fields.get("goal_id") or fields.get("goal") or "VRIF-G000"
        ).strip()
        if goal_id not in goal_cids:
            raise OperatorError(f"{task_id} refers to unknown goal {goal_id}")
        output_paths = _split_csv(fields.get("outputs") or fields.get("predicted_files"))
        task = dict(fields)
        task.update(
            {
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "title": title,
                "source_line": source_line,
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "plan_cid": plan_root,
                "objective_id": "objective:vrif-root",
                "ordinal": ordinal,
                "status": str(fields.get("status") or "todo").lower(),
                "priority": str(fields.get("priority") or "P1"),
                "dependencies": [task_cids[item] for item in dependencies],
                "depends_on": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"task_cid": task_cids[task_id], "path": path}
                        ),
                    }
                    for path in output_paths
                ],
                "acceptance": [
                    str(fields.get("acceptance") or fields.get("acceptance_subset") or "")
                ],
                "validations": list(split_validation_commands(str(fields.get("validation") or ""))),
                "accepted_plan_root_cid": plan_root,
                "base_revision": head,
                "base_repository_tree_id": tree,
                "owning_repository": "ipfs_accelerate_py",
            }
        )
        tasks.append(task)
        observed_tasks.add(task_id)

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    expected_tasks = projection.get("task_count")
    expected_goals = projection.get("goal_count")
    expected_dependencies = projection.get("task_dependency_count")
    if expected_tasks is not None and int(expected_tasks) != len(tasks):
        raise OperatorError("task count differs from configured initial projection")
    if expected_goals is not None and int(expected_goals) != len(goals):
        raise OperatorError("goal count differs from configured initial projection")
    dependency_count = sum(len(_split_csv(item[3].get("depends_on"))) for item in parsed_tasks)
    if expected_dependencies is not None and int(expected_dependencies) != dependency_count:
        raise OperatorError("task dependency count differs from configured initial projection")
    return {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "source_identities": {name: _identity(value) for name, value in sorted(sources.items())},
        "source_forest": source_forest,
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "VRIF-PLAN-V1",
                "goal_cid": goal_cids["VRIF-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _vrif_goal_completion_authority_spec(
    board: Any,
    config: Mapping[str, Any],
    admission: Mapping[str, Any],
    connection: Any,
) -> dict[str, Any]:
    """Reconstruct the exact immutable VRIF goal/task graph for the owner.

    Goal CIDs are reproduced from the sealed bootstrap objective bytes and
    plan root.  Task CIDs must be members of the sealed database receipt, and
    aliases/goal ownership must reproduce the static ``task_groups`` config.
    No live Markdown status is consulted.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA,
        GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
        split_validation_commands,
    )

    authority = admission.get("database_authority")
    if not isinstance(authority, Mapping):
        raise OperatorError("goal authority has no sealed database binding")
    bootstrap_head = str(admission.get("bootstrap_source_head") or "")
    plan_root_cid = str(admission.get("plan_root_cid") or "")
    objective_bytes = _git_blob_at(
        head=bootstrap_head,
        path=board.path(board.objectives_path),
        field="bootstrap goal authority objectives",
    )
    try:
        objective_text = objective_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperatorError("bootstrap goal authority objectives are not UTF-8") from exc
    parsed_goals = _goal_blocks(objective_text)
    if len(parsed_goals) != 9:
        raise OperatorError("VRIF goal completion authority requires exactly 9 goals")
    expected_aliases = {
        "VRIF-G000",
        "VRIF-G010",
        "VRIF-G011",
        "VRIF-G020",
        "VRIF-G021",
        "VRIF-G030",
        "VRIF-G031",
        "VRIF-G040",
        "VRIF-G041",
    }
    aliases = [item[0] for item in parsed_goals]
    if len(set(aliases)) != 9 or set(aliases) != expected_aliases:
        raise OperatorError("VRIF goal aliases are not the exact configured population")
    goal_cids = {
        goal_alias: content_identity(
            {
                "goal_id": goal_alias,
                "title": title,
                "metadata": fields,
                "plan_root_cid": plan_root_cid,
            }
        )
        for goal_alias, title, fields in parsed_goals
    }

    raw_hierarchy = config.get("goal_hierarchy")
    if not isinstance(raw_hierarchy, Mapping):
        raise OperatorError("VRIF goal hierarchy is absent")
    configured_hierarchy: dict[str, list[str]] = {}
    for parent, children in raw_hierarchy.items():
        if not isinstance(children, Sequence) or isinstance(
            children, (str, bytes, bytearray)
        ):
            raise OperatorError("VRIF goal hierarchy children must be a sequence")
        configured_hierarchy[str(parent)] = [str(child) for child in children]
    observed_hierarchy: dict[str, list[str]] = {}
    goals: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for ordinal, (goal_alias, _title, fields) in enumerate(parsed_goals, start=1):
        parent_alias = str(fields.get("parent") or "").strip()
        if goal_alias == "VRIF-G000":
            if parent_alias:
                raise OperatorError("VRIF root goal unexpectedly has a parent")
        elif parent_alias not in goal_cids:
            raise OperatorError(f"{goal_alias} has an unknown parent")
        parent_cid = goal_cids.get(parent_alias, "")
        goals.append(
            {
                "goal_cid": goal_cids[goal_alias],
                "goal_alias": goal_alias,
                "parent_goal_cid": parent_cid,
                "ordinal": ordinal,
            }
        )
        if parent_alias:
            observed_hierarchy.setdefault(parent_alias, []).append(goal_alias)
            edges.append(
                {
                    "parent_goal_cid": parent_cid,
                    "child_goal_cid": goal_cids[goal_alias],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency_alias in _split_csv(fields.get("depends_on")):
            if dependency_alias not in goal_cids:
                raise OperatorError(f"{goal_alias} has an unknown goal dependency")
            edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency_alias],
                    "child_goal_cid": goal_cids[goal_alias],
                    "edge_kind": "goal_dependency",
                }
            )
    if configured_hierarchy != observed_hierarchy:
        raise OperatorError("configured VRIF goal hierarchy differs from sealed objectives")

    raw_groups = config.get("task_groups")
    if not isinstance(raw_groups, Mapping):
        raise OperatorError("VRIF task groups are absent")
    expected_leaf_aliases = {"VRIF-G011", "VRIF-G021", "VRIF-G031", "VRIF-G041"}
    if set(str(key) for key in raw_groups) != expected_leaf_aliases:
        raise OperatorError("VRIF task groups do not name the exact four leaf goals")
    expected_task_goal: dict[str, str] = {}
    for goal_alias, members in raw_groups.items():
        if not isinstance(members, Sequence) or isinstance(
            members, (str, bytes, bytearray)
        ):
            raise OperatorError("VRIF task group members must be a sequence")
        for member in members:
            task_alias = str(member)
            if task_alias in expected_task_goal:
                raise OperatorError("VRIF task groups contain a duplicate task")
            expected_task_goal[task_alias] = str(goal_alias)
    expected_task_aliases = {f"VRIF-{ordinal:03d}" for ordinal in range(33)}
    if set(expected_task_goal) != expected_task_aliases:
        raise OperatorError("VRIF task groups must cover exactly VRIF-000 through VRIF-032")

    bootstrap_tree = str(authority.get("repository_tree_id") or "")
    if re.fullmatch(r"[0-9a-f]{40}", bootstrap_tree) is None:
        raise OperatorError("VRIF sealed task authority has no bootstrap tree")
    taskboard_bytes = _git_blob_at(
        head=bootstrap_head,
        path=board.path(board.taskboard_path),
        field="bootstrap goal authority taskboard",
    )
    try:
        taskboard_text = taskboard_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperatorError("bootstrap goal authority taskboard is not UTF-8") from exc
    parsed_tasks = parse_todo_blocks(
        taskboard_text, task_header_prefix="## VRIF-"
    )
    parsed_tasks = [
        (
            task_alias,
            title,
            source_line,
            {key: _metadata_value(value) for key, value in fields.items()},
        )
        for task_alias, title, source_line, fields in parsed_tasks
    ]
    if [item[0] for item in parsed_tasks] != [
        f"VRIF-{ordinal:03d}" for ordinal in range(33)
    ]:
        raise OperatorError(
            "VRIF sealed taskboard is not exact ordered VRIF-000 through VRIF-032"
        )
    expected_task_cids = {
        task_alias: content_identity(
            {
                "task_id": task_alias,
                "title": title,
                "source_line": source_line,
                "metadata": fields,
                "plan_root_cid": plan_root_cid,
                "repository_tree_id": bootstrap_tree,
            }
        )
        for task_alias, title, source_line, fields in parsed_tasks
    }
    parsed_task_fields = {item[0]: item[3] for item in parsed_tasks}

    task_rows = _rows(
        connection,
        "SELECT task_cid, task_alias, goal_cid FROM tasks ORDER BY task_alias, task_cid",
    )
    sealed_task_cids_raw = authority.get("task_cids")
    if not isinstance(sealed_task_cids_raw, Sequence) or isinstance(
        sealed_task_cids_raw, (str, bytes, bytearray)
    ):
        raise OperatorError("VRIF sealed task identities are absent")
    sealed_task_cids = sorted(str(item) for item in sealed_task_cids_raw)
    tasks: list[dict[str, str]] = []
    for row in task_rows:
        task_cid = str(_row_item(row, 0, "task_cid") or "")
        task_alias = str(_row_item(row, 1, "task_alias") or "")
        goal_cid = str(_row_item(row, 2, "goal_cid") or "")
        expected_goal_alias = expected_task_goal.get(task_alias)
        if (
            expected_goal_alias is None
            or goal_cid != goal_cids[expected_goal_alias]
            or task_cid != expected_task_cids.get(task_alias)
        ):
            raise OperatorError("VRIF database task-to-goal ownership is not exact")
        tasks.append(
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "goal_cid": goal_cid,
            }
        )
    if (
        len(tasks) != 33
        or {item["task_alias"] for item in tasks} != expected_task_aliases
        or sorted(item["task_cid"] for item in tasks) != sealed_task_cids
    ):
        raise OperatorError("VRIF goal authority task population differs from its seal")

    task_dependencies: list[dict[str, str]] = []
    for task_alias, _title, _source_line, fields in parsed_tasks:
        for dependency_alias in _split_csv(fields.get("depends_on")):
            if dependency_alias not in expected_task_cids:
                raise OperatorError(
                    f"{task_alias} has an unknown sealed task dependency"
                )
            task_dependencies.append(
                {
                    "task_cid": expected_task_cids[task_alias],
                    "dependency_task_cid": expected_task_cids[dependency_alias],
                    "kind": "depends_on",
                }
            )
    task_dependencies.sort(
        key=lambda item: (
            item["task_cid"],
            item["dependency_task_cid"],
            item["kind"],
        )
    )
    expected_dependency_count = int(
        (config.get("initial_projection") or {}).get("task_dependency_count")
        or 0
    )
    if expected_dependency_count != 111 or len(task_dependencies) != 111:
        raise OperatorError(
            "VRIF sealed task dependency graph must contain exactly 111 edges"
        )

    terminal_task_alias = "VRIF-032"
    terminal_fields = parsed_task_fields[terminal_task_alias]
    producer_task_aliases = ["VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031"]
    producer_output_paths = {
        task_alias: _split_csv(
            parsed_task_fields[task_alias].get("outputs")
            or parsed_task_fields[task_alias].get("predicted_files")
        )
        for task_alias in producer_task_aliases
    }
    producer_validation_commands = {
        task_alias: [
            [str(command)]
            for command in split_validation_commands(
                str(parsed_task_fields[task_alias].get("validation") or "")
            )
        ]
        for task_alias in producer_task_aliases
    }
    flattened_producer_paths = [
        path
        for task_alias in producer_task_aliases
        for path in producer_output_paths[task_alias]
    ]
    if (
        any(not producer_output_paths[task_alias] for task_alias in producer_task_aliases)
        or any(
            not producer_validation_commands[task_alias]
            for task_alias in producer_task_aliases
        )
        or len(flattened_producer_paths) != len(set(flattened_producer_paths))
    ):
        raise OperatorError("VRIF report producer output ownership is not exact")
    terminal_outputs = _split_csv(
        terminal_fields.get("outputs") or terminal_fields.get("predicted_files")
    )
    terminal_symbols = [
        item.strip()
        for item in re.split(r"[,;]", str(terminal_fields.get("predicted_symbols") or ""))
        if item.strip()
    ]
    terminal_acceptance = [
        str(
            terminal_fields.get("acceptance")
            or terminal_fields.get("acceptance_subset")
            or ""
        )
    ]
    terminal_validations = [
        [str(command)]
        for command in split_validation_commands(
            str(terminal_fields.get("validation") or "")
        )
    ]
    required_report_paths = [
        "docs/architecture/residual_intelligence_inventory/final_release_report.json",
        "docs/architecture/residual_intelligence_inventory/final_release_report.md",
    ]
    if (
        any(path not in terminal_outputs for path in required_report_paths)
        or any(path in terminal_outputs for path in flattened_producer_paths)
        or terminal_symbols
        != [
            "ResidualIntelligenceReleaseReport",
            "ResidualGapReport",
            "validate_release_claims",
        ]
        or not terminal_acceptance[0]
        or not terminal_validations
    ):
        raise OperatorError("VRIF-032 sealed report contract is incomplete")
    terminal_report_contract: dict[str, Any] = {
        "schema": GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA,
        "task_cid": expected_task_cids[terminal_task_alias],
        "task_alias": terminal_task_alias,
        "declared_output_paths": terminal_outputs,
        "declared_symbols": terminal_symbols,
        "required_report_paths": required_report_paths,
        "producer_output_paths": producer_output_paths,
        "producer_validation_commands": producer_validation_commands,
        "acceptance_criteria": terminal_acceptance,
        "validation_commands": terminal_validations,
    }
    terminal_report_contract["contract_id"] = content_identity(
        terminal_report_contract
    )

    completion_policy = config.get("completion_policy")
    if not isinstance(completion_policy, Mapping):
        raise OperatorError("VRIF completion policy is absent")
    exact_policy = {
        field: completion_policy.get(field)
        for field in VRIF_ROOT_COMPLETION_POLICY_FIELDS
    }
    if any(value is not True for value in exact_policy.values()):
        raise OperatorError("VRIF root completion policy is not fail-closed")
    if str(completion_policy.get("terminal_task_id") or "") != terminal_task_alias:
        raise OperatorError("VRIF root completion policy terminal task is not exact")
    exact_policy["terminal_task_id"] = terminal_task_alias
    initial_projection = config.get("initial_projection")
    initial_projection = (
        initial_projection if isinstance(initial_projection, Mapping) else {}
    )
    if str(initial_projection.get("root_goal_id") or "") != "VRIF-G000":
        raise OperatorError("VRIF completion authority root is not exact")
    spec: dict[str, Any] = {
        "schema": GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA,
        "board_namespace": str(config.get("board_namespace") or ""),
        "goal_count": 9,
        "task_count": 33,
        "root_goal_cid": goal_cids["VRIF-G000"],
        "root_goal_alias": "VRIF-G000",
        "goals": goals,
        "goal_edges": sorted(
            edges,
            key=lambda item: (
                item["edge_kind"],
                item["parent_goal_cid"],
                item["child_goal_cid"],
            ),
        ),
        "tasks": tasks,
        "task_dependencies": task_dependencies,
        "terminal_report_contract": terminal_report_contract,
        "completion_policy": exact_policy,
        "receipt_backfill_goal_cids": [
            goal_cids["VRIF-G010"],
            goal_cids["VRIF-G011"],
        ],
    }
    spec["authority_spec_id"] = content_identity(spec)
    return spec


def _vrif_portal_completion_binding(
    control_receipt: Mapping[str, Any],
    *,
    task_cid: str,
) -> dict[str, str] | None:
    """Validate the exact production Portal validation/binding envelope."""

    validation = control_receipt.get("validation")
    validation_fields = {
        "outcome",
        "evidence_digest",
        "argv",
        "validator",
        "task_cid",
        "attempt_id",
        "portal_receipt_id",
        "portal_completion_binding",
    }
    replayed_validation_fields = validation_fields | {"replayed"}
    binding_fields = {
        "schema",
        "task_cid",
        "attempt_id",
        "binding_id",
        "portal_receipt_id",
        "evidence_digest",
        "baseline_commit",
        "baseline_tree",
        "implementation_commit",
        "completion_event_id",
        "receipt_id",
    }
    raw_binding = (
        validation.get("portal_completion_binding")
        if isinstance(validation, Mapping)
        else None
    )
    if (
        control_receipt.get("operation") != "database_complete"
        or not isinstance(validation, Mapping)
        or (
            set(validation) != validation_fields
            and not (
                set(validation) == replayed_validation_fields
                and type(validation.get("replayed")) is bool
            )
        )
        or not isinstance(raw_binding, Mapping)
        or set(raw_binding) != binding_fields
    ):
        return None
    binding = {
        field: str(raw_binding.get(field) or "")
        for field in binding_fields - {"receipt_id"}
    }
    receipt_id = str(raw_binding.get("receipt_id") or "")
    evidence_digest = str(control_receipt.get("evidence_digest") or "")
    attempt_id = str(control_receipt.get("attempt_id") or "")
    portal_receipt_id = str(validation.get("portal_receipt_id") or "")
    if (
        validation.get("outcome") != "passed"
        or validation.get("argv") != ["portal-supervisor-gates"]
        or validation.get("validator") != "DatabasePortalExecutionBridge@1"
        or str(validation.get("task_cid") or "") != task_cid
        or str(validation.get("attempt_id") or "") != attempt_id
        or str(validation.get("evidence_digest") or "") != evidence_digest
        or re.fullmatch(r"sha256:[0-9a-f]{64}", evidence_digest) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", portal_receipt_id) is None
        or binding["schema"]
        != "ipfs_accelerate_py/agent-supervisor/database-portal-completion-binding@1"
        or binding["task_cid"] != task_cid
        or binding["attempt_id"] != attempt_id
        or binding["portal_receipt_id"] != portal_receipt_id
        or binding["evidence_digest"] != evidence_digest
        or any(
            re.fullmatch(r"[0-9a-f]{40}", binding[field]) is None
            for field in ("baseline_commit", "baseline_tree", "implementation_commit")
        )
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", binding[field]) is None
            for field in ("binding_id", "completion_event_id")
        )
        or receipt_id != _identity(binding)
    ):
        return None
    binding["receipt_id"] = receipt_id
    return binding


def _vrif_release_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the exact human companion for one closed machine report."""

    sections: list[tuple[str, Any]] = [
        (
            "Lineage",
            {
                "start_tree": report.get("start_tree"),
                "end_tree": report.get("end_tree"),
            },
        ),
        ("Files and Symbols", report.get("files_symbols")),
        ("Corpus Rights and Splits", report.get("corpus_rights_splits")),
        (
            "Architecture, Tokenizer, Checkpoint, and Training",
            report.get("architecture_tokenizer_checkpoint"),
        ),
        ("Expert Dispositions", report.get("expert_dispositions")),
        (
            "Before/After Denominators",
            {"before": report.get("before"), "after": report.get("after")},
        ),
        ("Costs and Break-even", report.get("costs")),
        ("Proof and Validation", report.get("proof_validation")),
        ("Drift", report.get("drift")),
        (
            "Rollback, Blockers, and Eligibility",
            report.get("rollback_blocker_eligibility"),
        ),
        ("Unsupported Gaps", report.get("gaps")),
    ]
    rendered = [
        "# VRIF Final Release Report\n\n",
        "This report is non-authoritative and cannot promote a residual expert.\n\n",
    ]
    for title, payload in sections:
        rendered.extend(
            (
                f"## {title}\n\n",
                "```json\n",
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "\n```\n\n",
            )
        )
    rendered.extend(
        (
            "## Complete Machine Report\n\n",
            "```json\n",
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "\n```\n",
        )
    )
    return "".join(rendered)


def _vrif_terminal_report_evidence(
    specification: Mapping[str, Any],
    admission: Mapping[str, Any],
    connection: Any,
) -> Mapping[str, Any] | None:
    """Return exact current-tree VRIF-032 report evidence, or fail open-safe.

    Missing, bootstrap-fixture, malformed, or unreceipted reports are an
    ordinary incomplete-board condition.  They keep the root gate absent and
    never prevent the Quack owner from starting to finish the remaining work.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA,
    )

    contract = specification.get("terminal_report_contract")
    if not isinstance(contract, Mapping):
        return None
    task_cid = str(contract.get("task_cid") or "")
    task_alias = str(contract.get("task_alias") or "")
    task_rows = _rows(
        connection,
        "SELECT status, revision, body_json FROM tasks WHERE task_cid = ?",
        [task_cid],
    )
    if len(task_rows) != 1:
        return None
    status = str(_row_item(task_rows[0], 0, "status") or "").strip().lower()
    revision = _row_item(task_rows[0], 1, "revision")
    try:
        task_body = json.loads(str(_row_item(task_rows[0], 2, "body_json") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        status not in {"completed", "complete", "done"}
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(task_body, Mapping)
    ):
        return None
    control_receipt = task_body.get("completion_receipt")
    if not isinstance(control_receipt, Mapping):
        return None
    receipt_rows = _rows(
        connection,
        "SELECT receipt_cid, evidence_digest, body_json "
        "FROM completion_receipts WHERE task_cid = ? "
        "ORDER BY completed_at, receipt_cid",
        [task_cid],
    )
    matching_receipts: list[tuple[str, str, Mapping[str, Any]]] = []
    for row in receipt_rows:
        try:
            receipt_body = json.loads(
                str(_row_item(row, 2, "body_json") or "{}")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(receipt_body, Mapping)
            and receipt_body.get("revision") == revision
        ):
            matching_receipts.append(
                (
                    str(_row_item(row, 0, "receipt_cid") or ""),
                    str(_row_item(row, 1, "evidence_digest") or ""),
                    receipt_body,
                )
            )
    if len(matching_receipts) != 1:
        return None
    receipt_cid, observed_evidence_digest, receipt_body = matching_receipts[0]
    evidence_digests = receipt_body.get("evidence_digests")
    validation = control_receipt.get("validation")
    raw_portal_binding = (
        validation.get("portal_completion_binding")
        if isinstance(validation, Mapping)
        else None
    )
    portal_binding_fields = {
        "schema",
        "task_cid",
        "attempt_id",
        "binding_id",
        "portal_receipt_id",
        "evidence_digest",
        "baseline_commit",
        "baseline_tree",
        "implementation_commit",
        "completion_event_id",
        "receipt_id",
    }
    if not isinstance(raw_portal_binding, Mapping) or set(raw_portal_binding) != (
        portal_binding_fields
    ):
        return None
    portal_completion_binding = {
        field: str(raw_portal_binding.get(field) or "")
        for field in portal_binding_fields - {"receipt_id"}
    }
    expected_portal_binding_receipt_id = _identity(portal_completion_binding)
    portal_completion_binding["receipt_id"] = str(
        raw_portal_binding.get("receipt_id") or ""
    )
    expected_evidence_digest = content_identity(
        {
            "task_cid": task_cid,
            "revision": revision,
            "receipt": dict(control_receipt),
            "evidence_digests": list(evidence_digests or ()),
        }
    )
    expected_receipt_cid = content_identity(
        {
            "namespace": "completion-receipt",
            "task_cid": task_cid,
            "revision": revision,
            "evidence_digest": expected_evidence_digest,
        }
    )
    if (
        receipt_body.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1"
        or dict(receipt_body.get("receipt") or {}) != dict(control_receipt)
        or not isinstance(evidence_digests, list)
        or len(evidence_digests) != 1
        or control_receipt.get("operation") != "database_complete"
        or not isinstance(validation, Mapping)
        or validation.get("outcome") != "passed"
        or validation.get("argv") != ["portal-supervisor-gates"]
        or validation.get("validator") != "DatabasePortalExecutionBridge@1"
        or validation.get("task_cid") != task_cid
        or validation.get("attempt_id") != control_receipt.get("attempt_id")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(validation.get("portal_receipt_id") or ""),
        )
        is None
        or validation.get("evidence_digest")
        != control_receipt.get("evidence_digest")
        or portal_completion_binding["schema"]
        != "ipfs_accelerate_py/agent-supervisor/database-portal-completion-binding@1"
        or portal_completion_binding["task_cid"] != task_cid
        or portal_completion_binding["attempt_id"]
        != str(control_receipt.get("attempt_id") or "")
        or portal_completion_binding["portal_receipt_id"]
        != str(validation.get("portal_receipt_id") or "")
        or portal_completion_binding["evidence_digest"]
        != str(control_receipt.get("evidence_digest") or "")
        or any(
            re.fullmatch(r"[0-9a-f]{40}", portal_completion_binding[field])
            is None
            for field in ("baseline_commit", "baseline_tree", "implementation_commit")
        )
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", portal_completion_binding["binding_id"]
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            portal_completion_binding["completion_event_id"],
        )
        is None
        or portal_completion_binding["receipt_id"]
        != expected_portal_binding_receipt_id
        or evidence_digests[0] != control_receipt.get("evidence_digest")
        or observed_evidence_digest != expected_evidence_digest
        or receipt_cid != expected_receipt_cid
    ):
        return None
    if (
        _vrif_portal_completion_binding(control_receipt, task_cid=task_cid)
        != portal_completion_binding
    ):
        return None

    validation_rows = _rows(
        connection,
        "SELECT vr.run_id, vr.attempt_id, vr.status, vr.command_digest, "
        "vr.body_json AS run_body_json, result.result_id, result.outcome, "
        "result.evidence_digest, result.body_json AS result_body_json "
        "FROM validation_runs AS vr "
        "JOIN validation_results AS result ON result.run_id = vr.run_id "
        "WHERE vr.task_cid = ? AND result.task_cid = ? "
        "ORDER BY vr.run_id, result.result_id",
        [task_cid, task_cid],
    )
    validation_lineage: dict[str, str] | None = None
    matching_lineage: list[dict[str, str]] = []
    expected_validation_body = dict(validation)
    expected_run_body = {
        "argv": ["portal-supervisor-gates"],
        **expected_validation_body,
    }
    for row in validation_rows:
        try:
            run_body = json.loads(
                str(_row_item(row, 4, "run_body_json") or "{}")
            )
            result_body = json.loads(
                str(_row_item(row, 8, "result_body_json") or "{}")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        run_id = str(_row_item(row, 0, "run_id") or "")
        result_id = str(_row_item(row, 5, "result_id") or "")
        if (
            str(_row_item(row, 1, "attempt_id") or "")
            != str(control_receipt.get("attempt_id") or "")
            or str(_row_item(row, 2, "status") or "") != "passed"
            or str(_row_item(row, 3, "command_digest") or "")
            != content_identity({"argv": ["portal-supervisor-gates"]})
            or run_body != expected_run_body
            or str(_row_item(row, 6, "outcome") or "") != "passed"
            or str(_row_item(row, 7, "evidence_digest") or "")
            != str(control_receipt.get("evidence_digest") or "")
            or result_body != expected_validation_body
        ):
            continue
        validation_evidence_id = content_identity(
            {
                "task_cid": task_cid,
                "evidence_kind": "validation",
                "digest": str(control_receipt.get("evidence_digest") or ""),
                "run_id": run_id,
            }
        )
        evidence_rows = _rows(
            connection,
            "SELECT evidence_kind, digest, body_json FROM evidence_nodes "
            "WHERE task_cid = ? AND evidence_id = ?",
            [task_cid, validation_evidence_id],
        )
        if len(evidence_rows) != 1:
            continue
        try:
            evidence_body = json.loads(
                str(_row_item(evidence_rows[0], 2, "body_json") or "{}")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            str(_row_item(evidence_rows[0], 0, "evidence_kind") or "")
            != "validation"
            or str(_row_item(evidence_rows[0], 1, "digest") or "")
            != str(control_receipt.get("evidence_digest") or "")
            or evidence_body
            != {
                "run_id": run_id,
                "result_id": result_id,
                "argv": ["portal-supervisor-gates"],
                "outcome": "passed",
            }
        ):
            continue
        matching_lineage.append(
            {
                "validation_run_id": run_id,
                "validation_result_id": result_id,
                "validation_evidence_id": validation_evidence_id,
            }
        )
    if len(matching_lineage) != 1:
        return None
    validation_lineage = matching_lineage[0]

    task_specs = {
        str(item.get("task_cid") or ""): item
        for item in specification.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    producer_task_cids = sorted(
        {
            str(edge.get("dependency_task_cid") or "")
            for edge in specification.get("task_dependencies") or ()
            if isinstance(edge, Mapping)
            and str(edge.get("task_cid") or "") == task_cid
        },
        key=lambda producer_cid: str(
            (task_specs.get(producer_cid) or {}).get("task_alias") or ""
        ),
    )
    if [
        str((task_specs.get(producer_cid) or {}).get("task_alias") or "")
        for producer_cid in producer_task_cids
    ] != ["VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031"]:
        return None
    producer_task_cid_by_alias = {
        str((task_specs.get(producer_cid) or {}).get("task_alias") or ""): producer_cid
        for producer_cid in producer_task_cids
    }
    producer_receipts: dict[str, str] = {}
    producer_portal_bindings: dict[str, dict[str, str]] = {}
    for producer_cid in producer_task_cids:
        producer_spec = task_specs.get(producer_cid) or {}
        producer_alias = str(producer_spec.get("task_alias") or "")
        producer_rows = _rows(
            connection,
            "SELECT status, revision, body_json FROM tasks WHERE task_cid = ?",
            [producer_cid],
        )
        producer_completion_rows = _rows(
            connection,
            "SELECT receipt_cid, evidence_digest, body_json "
            "FROM completion_receipts WHERE task_cid = ? "
            "ORDER BY completed_at, receipt_cid",
            [producer_cid],
        )
        if len(producer_rows) != 1:
            return None
        producer_status = str(
            _row_item(producer_rows[0], 0, "status") or ""
        ).strip().lower()
        producer_revision = _row_item(producer_rows[0], 1, "revision")
        try:
            producer_task_body = json.loads(
                str(_row_item(producer_rows[0], 2, "body_json") or "{}")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        producer_control_receipt = (
            producer_task_body.get("completion_receipt")
            if isinstance(producer_task_body, Mapping)
            else None
        )
        producer_matches: list[tuple[str, str, Mapping[str, Any]]] = []
        for row in producer_completion_rows:
            try:
                producer_receipt_body = json.loads(
                    str(_row_item(row, 2, "body_json") or "{}")
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(producer_receipt_body, Mapping)
                and producer_receipt_body.get("revision") == producer_revision
            ):
                producer_matches.append(
                    (
                        str(_row_item(row, 0, "receipt_cid") or ""),
                        str(_row_item(row, 1, "evidence_digest") or ""),
                        producer_receipt_body,
                    )
                )
        if (
            producer_status not in {"completed", "complete", "done"}
            or isinstance(producer_revision, bool)
            or not isinstance(producer_revision, int)
            or producer_revision < 1
            or not isinstance(producer_control_receipt, Mapping)
            or len(producer_matches) != 1
        ):
            return None
        producer_receipt_cid, producer_observed_digest, producer_receipt_body = (
            producer_matches[0]
        )
        producer_evidence_digests = producer_receipt_body.get("evidence_digests")
        producer_portal_binding = _vrif_portal_completion_binding(
            producer_control_receipt,
            task_cid=producer_cid,
        )
        if (
            not isinstance(producer_evidence_digests, list)
            or len(producer_evidence_digests) != 1
            or producer_evidence_digests[0]
            != producer_control_receipt.get("evidence_digest")
            or producer_portal_binding is None
        ):
            return None
        producer_expected_digest = content_identity(
            {
                "task_cid": producer_cid,
                "revision": producer_revision,
                "receipt": dict(producer_control_receipt),
                "evidence_digests": producer_evidence_digests,
            }
        )
        producer_expected_receipt_cid = content_identity(
            {
                "namespace": "completion-receipt",
                "task_cid": producer_cid,
                "revision": producer_revision,
                "evidence_digest": producer_expected_digest,
            }
        )
        if (
            producer_receipt_body.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1"
            or dict(producer_receipt_body.get("receipt") or {})
            != dict(producer_control_receipt)
            or producer_observed_digest != producer_expected_digest
            or producer_receipt_cid != producer_expected_receipt_cid
        ):
            return None
        producer_receipts[producer_alias] = producer_receipt_cid
        producer_portal_bindings[producer_alias] = producer_portal_binding

    current_head = str(admission.get("current_source_head") or "")
    bootstrap_head = str(admission.get("bootstrap_source_head") or "")
    portal_baseline_commit = portal_completion_binding["baseline_commit"]
    portal_baseline_tree = portal_completion_binding["baseline_tree"]
    implementation_commit = portal_completion_binding["implementation_commit"]
    required_paths = contract.get("required_report_paths")
    if not isinstance(required_paths, Sequence) or isinstance(
        required_paths, (str, bytes, bytearray)
    ) or len(required_paths) != 2:
        return None
    artifacts: list[dict[str, str]] = []
    current_blobs: dict[str, bytes] = {}
    try:
        _git_is_ancestor(
            portal_baseline_commit,
            implementation_commit,
            field="VRIF-032 Portal evaluated-source lineage",
        )
        if _git_commit_tree(
            portal_baseline_commit,
            field="VRIF-032 Portal evaluated source",
        ) != portal_baseline_tree:
            return None
        terminal_changed_paths = [
            line.strip()
            for line in str(
                _git(
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    portal_baseline_commit,
                    implementation_commit,
                )
            ).splitlines()
            if line.strip()
        ]
        declared_terminal_paths = [
            str(path) for path in contract.get("declared_output_paths") or ()
        ]
        if (
            not terminal_changed_paths
            or len(terminal_changed_paths) != len(set(terminal_changed_paths))
            or set(terminal_changed_paths) != set(declared_terminal_paths)
        ):
            return None
        _git_is_ancestor(
            implementation_commit,
            current_head,
            field="VRIF-032 Portal implementation lineage",
        )

        def regular_blob(head: str, path: str, *, field: str) -> bytes:
            tree_row = str(_git("ls-tree", head, "--", path)).strip()
            parts = tree_row.split(maxsplit=3)
            if (
                len(parts) != 4
                or parts[0] not in {"100644", "100755"}
                or parts[1] != "blob"
                or parts[3] != path
            ):
                raise OperatorError(f"{field} is not one exact regular Git blob")
            return _git_blob_at(
                head=head,
                path=ROOT / path,
                field=field,
            )

        producer_artifact_tasks: list[dict[str, Any]] = []
        producer_output_paths = contract.get("producer_output_paths")
        if not isinstance(producer_output_paths, Mapping):
            return None
        for producer_alias in sorted(producer_output_paths):
            raw_paths = producer_output_paths.get(producer_alias)
            producer_binding = producer_portal_bindings.get(str(producer_alias))
            if (
                not isinstance(raw_paths, list)
                or not raw_paths
                or producer_binding is None
            ):
                return None
            _git_is_ancestor(
                producer_binding["baseline_commit"],
                producer_binding["implementation_commit"],
                field=f"{producer_alias} Portal evaluated-source lineage",
            )
            if _git_commit_tree(
                producer_binding["baseline_commit"],
                field=f"{producer_alias} Portal evaluated source",
            ) != producer_binding["baseline_tree"]:
                return None
            _git_is_ancestor(
                producer_binding["implementation_commit"],
                implementation_commit,
                field=f"{producer_alias} to VRIF-032 Portal lineage",
            )
            producer_artifact_rows: list[dict[str, str]] = []
            for path in sorted(str(item) for item in raw_paths):
                producer_blob = regular_blob(
                    producer_binding["implementation_commit"],
                    path,
                    field=f"{producer_alias} Portal producer artifact {path}",
                )
                implementation_blob = regular_blob(
                    implementation_commit,
                    path,
                    field=f"Portal producer artifact {producer_alias}:{path}",
                )
                current_blob = regular_blob(
                    current_head,
                    path,
                    field=f"current producer artifact {producer_alias}:{path}",
                )
                blob_identity = _identity(implementation_blob)
                if (
                    _identity(producer_blob) != blob_identity
                    or _identity(current_blob) != blob_identity
                ):
                    return None
                try:
                    bootstrap_blob = _git_blob_at(
                        head=bootstrap_head,
                        path=ROOT / path,
                        field=f"bootstrap producer artifact {producer_alias}:{path}",
                    )
                except OperatorError:
                    bootstrap_blob = None
                if (
                    isinstance(bootstrap_blob, bytes)
                    and _identity(bootstrap_blob) == blob_identity
                ):
                    return None
                producer_artifact_rows.append(
                    {"path": path, "blob_identity": blob_identity}
                )
            producer_task_bundle: dict[str, Any] = {
                "task_alias": str(producer_alias),
                "artifacts": producer_artifact_rows,
            }
            producer_task_bundle["bundle_id"] = _identity(producer_task_bundle)
            producer_artifact_tasks.append(producer_task_bundle)
        producer_artifacts: dict[str, Any] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "goal-terminal-producer-artifacts@1"
            ),
            "digest_algorithm": "sha256",
            "tasks": producer_artifact_tasks,
        }
        producer_artifacts["bundle_id"] = _identity(producer_artifacts)
        producer_receipt_bindings: list[dict[str, Any]] = []
        for artifact_task in producer_artifact_tasks:
            producer_alias = str(artifact_task["task_alias"])
            receipt_binding: dict[str, Any] = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "goal-terminal-producer-receipt-binding@1"
                ),
                "task_alias": producer_alias,
                "task_cid": producer_task_cid_by_alias[producer_alias],
                "completion_receipt_cid": producer_receipts[producer_alias],
                "portal_completion_binding": producer_portal_bindings[
                    producer_alias
                ],
                "artifact_bundle_id": artifact_task["bundle_id"],
            }
            receipt_binding["binding_id"] = _identity(receipt_binding)
            producer_receipt_bindings.append(receipt_binding)
        benchmark_producer_binding = producer_portal_bindings["VRIF-030"]
        benchmark_binding_paths = {
            "objective_revisions": [
                (
                    "docs/architecture/"
                    "agent_supervisor_residual_intelligence.objectives.md"
                ),
                (
                    "docs/architecture/"
                    "agent_supervisor_residual_intelligence.todo.md"
                ),
            ],
            "provider_policy": [
                "config/agent_supervisor_residual_intelligence_scheduler.json"
            ],
            "operation_catalog": [
                "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
            ],
            "validation_policy": [
                "test/api/residual_intelligence/test_benchmark.py"
            ],
        }
        benchmark_binding_blob_identities: dict[str, str] = {}
        for dimension, paths in benchmark_binding_paths.items():
            path_identities: dict[str, str] = {}
            for path in paths:
                producer_blob = regular_blob(
                    benchmark_producer_binding["implementation_commit"],
                    path,
                    field=f"VRIF-030 frozen {dimension} binding {path}",
                )
                identity = _identity(producer_blob)
                # Cross-wave inputs are historical VRIF-030 bindings.  VRIF-029
                # may legitimately merge a newer operation catalog afterwards;
                # only the VRIF-030-owned validation/artifact population is
                # required to remain byte-identical through VRIF-032/current.
                if dimension == "validation_policy":
                    terminal_blob = regular_blob(
                        implementation_commit,
                        path,
                        field=f"VRIF-032 frozen {dimension} binding {path}",
                    )
                    current_blob = regular_blob(
                        current_head,
                        path,
                        field=f"current frozen {dimension} binding {path}",
                    )
                    if (
                        _identity(terminal_blob) != identity
                        or _identity(current_blob) != identity
                    ):
                        return None
                path_identities[path] = identity
            benchmark_binding_blob_identities[dimension] = (
                _identity(
                    {
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "residual-benchmark-objective-revisions@1"
                        ),
                        "artifacts": path_identities,
                    }
                )
                if dimension == "objective_revisions"
                else next(iter(path_identities.values()))
            )
        declared_output_paths = contract.get("declared_output_paths")
        if not isinstance(declared_output_paths, list):
            return None
        for raw_path in declared_output_paths:
            path = str(raw_path)
            current_blob = regular_blob(
                current_head,
                path,
                field=f"current terminal report {path}",
            )
            implementation_blob = regular_blob(
                implementation_commit,
                path,
                field=f"Portal-validated terminal report {path}",
            )
            current_identity = _identity(current_blob)
            if _identity(implementation_blob) != current_identity:
                return None
            if path not in required_paths:
                continue
            bootstrap_blob = regular_blob(
                bootstrap_head,
                path,
                field=f"bootstrap terminal report {path}",
            )
            bootstrap_identity = _identity(bootstrap_blob)
            if current_identity == bootstrap_identity:
                return None
            current_blobs[path] = current_blob
            artifacts.append(
                {
                    "path": path,
                    "blob_identity": current_identity,
                    "bootstrap_blob_identity": bootstrap_identity,
                }
            )
        anchor_payloads: dict[str, Mapping[str, Any]] = {}
        anchor_blob_identities: dict[str, str] = {}
        for path in (
            "docs/architecture/residual_intelligence_inventory/baseline.json",
            "docs/architecture/residual_intelligence_inventory/"
            "residual_model_call_inventory.json",
            "docs/architecture/residual_intelligence_inventory/pgir_training_gate.json",
            "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_training_admission.json",
            "benchmarks/agent_supervisor/residual_intelligence/"
            "synthetic_split_manifest.json",
            "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
        ):
            current_anchor = _git_blob_at(
                head=current_head,
                path=ROOT / path,
                field=f"current terminal report anchor {path}",
            )
            implementation_anchor = _git_blob_at(
                head=implementation_commit,
                path=ROOT / path,
                field=f"Portal-validated terminal report anchor {path}",
            )
            if _identity(current_anchor) != _identity(implementation_anchor):
                return None
            anchor_blob_identities[path] = _identity(current_anchor)
            anchor_payloads[path] = _json_mapping_bytes(
                current_anchor,
                field=f"terminal report anchor {path}",
            )
        cases_path = "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"
        current_cases = _git_blob_at(
            head=current_head,
            path=ROOT / cases_path,
            field="current terminal report benchmark cases",
        )
        implementation_cases = _git_blob_at(
            head=implementation_commit,
            path=ROOT / cases_path,
            field="Portal-validated terminal report benchmark cases",
        )
        if _identity(current_cases) != _identity(implementation_cases):
            return None
        benchmark_cases = [
            json.loads(line)
            for line in current_cases.decode("utf-8").splitlines()
            if line.strip()
        ]
        release_source_path = (
            "ipfs_accelerate_py/agent_supervisor/residual_intelligence/release.py"
        )
        current_release_source = _git_blob_at(
            head=current_head,
            path=ROOT / release_source_path,
            field="current terminal report symbol source",
        )
        implementation_release_source = _git_blob_at(
            head=implementation_commit,
            path=ROOT / release_source_path,
            field="Portal-validated terminal report symbol source",
        )
        if _identity(current_release_source) != _identity(
            implementation_release_source
        ):
            return None
        release_module = ast.parse(current_release_source.decode("utf-8"))
        observed_release_symbols = {
            node.name
            for node in release_module.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
    except (OperatorError, UnicodeDecodeError, json.JSONDecodeError, SyntaxError):
        return None

    baseline = anchor_payloads[
        "docs/architecture/residual_intelligence_inventory/baseline.json"
    ]
    training_admission = anchor_payloads[
        "benchmarks/agent_supervisor/residual_intelligence/"
        "synthetic_training_admission.json"
    ]
    benchmark_manifest = anchor_payloads[
        "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
    ]
    model_call_inventory = anchor_payloads[
        "docs/architecture/residual_intelligence_inventory/"
        "residual_model_call_inventory.json"
    ]
    training_gate = anchor_payloads[
        "docs/architecture/residual_intelligence_inventory/pgir_training_gate.json"
    ]
    split_manifest = anchor_payloads[
        "benchmarks/agent_supervisor/residual_intelligence/"
        "synthetic_split_manifest.json"
    ]
    baseline_source = baseline.get("source")
    baseline_environment = baseline.get("environment")
    baseline_hardware = (
        baseline_environment.get("hardware")
        if isinstance(baseline_environment, Mapping)
        else None
    )
    admission_body = dict(training_admission)
    admission_id = str(admission_body.pop("admission_id", "") or "")
    task_families = benchmark_manifest.get("task_families")
    split_assignments = split_manifest.get("assignments")
    split_partitions = (
        [str(item.get("partition") or "") for item in split_assignments]
        if isinstance(split_assignments, list)
        and all(isinstance(item, Mapping) for item in split_assignments)
        else []
    )
    pgir = training_gate.get("pgir")
    admission_leakage = training_admission.get("leakage_audit")
    try:
        from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
            ExpertDisposition,
            ResidualTaskFamily,
        )

        expected_task_families = [item.value for item in ResidualTaskFamily]
        allowed_expert_dispositions = {item.value for item in ExpertDisposition}
    except Exception:
        return None
    base_frozen_bindings = {
        "repository_states": _identity(
            {
                "commit": benchmark_producer_binding["baseline_commit"],
                "tree": benchmark_producer_binding["baseline_tree"],
            }
        ),
        "objective_revisions": benchmark_binding_blob_identities[
            "objective_revisions"
        ],
        "operation_catalog": benchmark_binding_blob_identities[
            "operation_catalog"
        ],
        "provider_policy": benchmark_binding_blob_identities["provider_policy"],
        "tokenizer": _identity(
            {
                "admission_id": admission_id,
                "disposition": "no_learned_tokenizer_admitted",
            }
        ),
        "model_versions": _identity(
            {
                "inventory_blob_identity": anchor_blob_identities[
                    "docs/architecture/residual_intelligence_inventory/"
                    "residual_model_call_inventory.json"
                ],
                "disposition": "training_unavailable",
            }
        ),
        "validation_policy": _identity(
            {
                "argv": contract["producer_validation_commands"]["VRIF-030"],
                "test_blob_identity": benchmark_binding_blob_identities[
                    "validation_policy"
                ],
            }
        ),
    }
    try:
        frozen_benchmark = _vrif_frozen_benchmark_contract(
            task_families=expected_task_families,
            source_commit=str(benchmark_producer_binding["baseline_commit"]),
            source_tree=str(benchmark_producer_binding["baseline_tree"]),
            split_root=str(split_manifest.get("split_root") or ""),
            base_bindings=base_frozen_bindings,
        )
    except OperatorError:
        return None
    expected_partitions = frozen_benchmark["partitions"]
    expected_case_kinds = frozen_benchmark["case_kinds"]
    expected_benchmark_cases = frozen_benchmark["cases"]
    expected_benchmark_scores = frozen_benchmark["scores"]
    frozen_bindings = frozen_benchmark["bindings"]
    frozen_binding_set_id = frozen_benchmark["binding_set_id"]
    paired_baseline = frozen_benchmark["paired_baseline"]
    benchmark_freeze = frozen_benchmark["benchmark_freeze"]
    try:
        baseline_tree = _git_commit_tree(
            str(baseline_source.get("commit") or "")
            if isinstance(baseline_source, Mapping)
            else "",
            field="terminal report baseline source",
        )
    except OperatorError:
        return None
    if (
        baseline.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/residual-intelligence-baseline@1"
        or not isinstance(baseline_source, Mapping)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(baseline_source.get("commit") or "")
        )
        is None
        or baseline_tree != str(baseline_source.get("tree") or "")
        or training_admission.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/training-corpus-admission@1"
        or training_admission.get("admission_decision") != "training_unavailable"
        or admission_id != content_identity(admission_body)
        or split_manifest.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/residual-semantic-split-manifest@1"
        or not split_assignments
        or set(split_partitions)
        != {"training", "development", "held_out", "adversarial"}
        or any(
            bool(item.get("hidden_from_training"))
            != (str(item.get("partition") or "") in {"held_out", "adversarial"})
            for item in split_assignments
        )
        or split_manifest.get("split_root") != training_admission.get("split_root")
        or not isinstance(admission_leakage, Mapping)
        or admission_leakage.get("split_root") != split_manifest.get("split_root")
        or admission_leakage.get("hidden_test_bodies_accessed") is not False
        or benchmark_manifest.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/"
        "residual-intelligence-benchmark-manifest@1"
        or set(benchmark_manifest)
        != {
            "schema",
            "program_identifier",
            "status",
            "owner_task",
            "source_revision",
            "partitions",
            "required_case_kinds",
            "task_families",
            "training_admission",
            "weights_committed",
            "large_corpus_committed",
            "promotion_evidence",
            "benchmark_freeze",
        }
        or benchmark_manifest.get("program_identifier")
        != "agent-supervisor-verified-residual-intelligence-foundry-v1"
        or benchmark_manifest.get("status") != "staged_not_qualified"
        or benchmark_manifest.get("owner_task") != "VRIF-030"
        or benchmark_manifest.get("source_revision")
        != benchmark_producer_binding["baseline_commit"]
        or task_families != expected_task_families
        or benchmark_manifest.get("partitions")
        != expected_partitions
        or benchmark_manifest.get("required_case_kinds") != expected_case_kinds
        or benchmark_manifest.get("training_admission") != "training_unavailable"
        or benchmark_manifest.get("weights_committed") is not False
        or benchmark_manifest.get("large_corpus_committed") is not False
        or benchmark_manifest.get("promotion_evidence") is not False
        or benchmark_manifest.get("benchmark_freeze") != benchmark_freeze
        or len(set(task_families)) != 24
        or training_gate.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/"
        "residual-intelligence-training-gate@1"
        or training_gate.get("decision") != "training_unavailable"
        or training_gate.get("training_attempted") is not False
        or training_gate.get("checkpoint_created") is not False
        or training_gate.get("promotion_attempted") is not False
        or not isinstance(pgir, Mapping)
        or pgir.get("decision") != "no_go"
        or pgir.get("training_admitted_rows") != 0
        or pgir.get("candidate_checkpoint") is not None
        or pgir.get("learned_tokenizer_status")
        != "no_learned_tokenizer_admitted"
        or set(contract.get("declared_symbols") or ())
        != {
            "ResidualIntelligenceReleaseReport",
            "ResidualGapReport",
            "validate_release_claims",
        }
        or not set(contract.get("declared_symbols") or ()).issubset(
            observed_release_symbols
        )
        or not isinstance(baseline_hardware, Mapping)
        or baseline_hardware.get("local_cuda_inference_qualified") is not False
        or benchmark_manifest.get("promotion_evidence") is not False
        or model_call_inventory.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/"
        "residual-model-call-surface-inventory@1"
        or model_call_inventory.get("source_tree") != baseline_source.get("tree")
        or model_call_inventory.get("trajectory_observation_count") != 0
        or model_call_inventory.get("training_examples_created") != 0
        or benchmark_cases != expected_benchmark_cases
        or len(benchmark_cases) != 24 * 4
        or len({str(case.get("case_id")) for case in benchmark_cases})
        != len(benchmark_cases)
        or len({str(case.get("input_identity")) for case in benchmark_cases})
        != len(benchmark_cases)
    ):
        return None

    json_paths = [path for path in current_blobs if path.endswith(".json")]
    markdown_paths = [path for path in current_blobs if path.endswith(".md")]
    if len(json_paths) != 1 or len(markdown_paths) != 1:
        return None
    try:
        report = json.loads(current_blobs[json_paths[0]].decode("utf-8"))
        markdown = current_blobs[markdown_paths[0]].decode("utf-8")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    report_fields = {
        "schema",
        "start_tree",
        "end_tree",
        "corpus_admission_id",
        "expert_dispositions",
        "before",
        "after",
        "costs",
        "promotion_eligible",
        "rollback_target",
        "gaps",
        "producer_artifacts",
        "files_symbols",
        "corpus_rights_splits",
        "architecture_tokenizer_checkpoint",
        "proof_validation",
        "drift",
        "rollback_blocker_eligibility",
    }
    gap_fields = {"blockers", "unsupported_claims", "not_run"}
    forbidden_claims = {
        "learned",
        "verified",
        "safe",
        "autonomous",
        "token-efficient",
        "production-ready",
    }
    exact_unsupported_claims = [
        "learned",
        "verified",
        "safe",
        "autonomous",
        "token-efficient",
        "production-ready",
    ]
    exact_not_run = ["gpu_live_qualification", "promotion", "training"]
    exact_blockers = ["training_unavailable"]
    expected_files_symbols = {
        "disposition": "current_tracked_blobs_bound",
        "declared_output_paths": list(contract.get("declared_output_paths") or ()),
        "required_report_paths": list(required_paths),
        "declared_symbols": list(contract.get("declared_symbols") or ()),
        "producer_artifact_bundle_id": producer_artifacts["bundle_id"],
    }
    expected_corpus_rights_splits = {
        "disposition": "training_unavailable",
        "admission_id": admission_id,
        "corpus_root": str(training_admission.get("corpus_root") or ""),
        "source_rights_root": str(
            training_admission.get("source_rights_root") or ""
        ),
        "split_root": str(split_manifest.get("split_root") or ""),
        "partitions": ["training", "development", "held_out", "adversarial"],
        "hidden_test_bodies_accessed": False,
        "privacy_disposition": "public_report_bounded",
    }
    expected_architecture_tokenizer_checkpoint = {
        "disposition": "training_unavailable",
        "architecture": "not_selected",
        "tokenizer": "no_learned_tokenizer_admitted",
        "checkpoint": "not_created",
        "training": "not_attempted",
    }
    expected_proof_validation = {
        "disposition": "owner_receipts_required",
        "validation_commands": [
            list(command) for command in contract.get("validation_commands") or ()
        ],
        "producer_artifact_bundle_id": producer_artifacts["bundle_id"],
        "benchmark_freeze_id": benchmark_freeze["freeze_id"],
        "benchmark_case_root": benchmark_freeze["case_root"],
        "benchmark_binding_set_id": frozen_binding_set_id,
        "paired_baseline_id": paired_baseline["paired_baseline_id"],
        "benchmark_case_payload_disposition": benchmark_freeze[
            "case_payload_disposition"
        ],
        "benchmark_evaluation_disposition": benchmark_freeze[
            "evaluation_disposition"
        ],
        "producer_database_portal_validations": "required",
        "terminal_database_portal_validation": "required",
        "report_authoritative": False,
    }
    expected_drift = {
        "disposition": "not_run_training_unavailable",
        "reference_tree": str(baseline_source.get("tree") or ""),
        "evaluated_tree": portal_baseline_tree,
        "checkpoint_available": False,
        "detectors_run": [],
        "reason_codes": ["no_admitted_checkpoint", "training_unavailable"],
    }
    expected_rollback_blocker_eligibility = {
        "promotion_eligible": False,
        "rollback_target": str(baseline_source.get("commit") or ""),
        "blockers": exact_blockers,
        "not_run": exact_not_run,
        "report_authority": "non_authoritative",
    }
    gaps = report.get("gaps") if isinstance(report, Mapping) else None
    unsupported_claims = (
        gaps.get("unsupported_claims") if isinstance(gaps, Mapping) else None
    )
    required_text_fields = (
        "start_tree",
        "end_tree",
        "corpus_admission_id",
        "rollback_target",
    )
    bounded_text_fields = bool(
        isinstance(report, Mapping)
        and all(
            isinstance(report.get(field), str)
            and 0 < len(str(report.get(field)).strip()) <= 4_096
            for field in required_text_fields
        )
    )
    expert_dispositions = (
        report.get("expert_dispositions") if isinstance(report, Mapping) else None
    )
    count_maps = [
        report.get(field) if isinstance(report, Mapping) else None
        for field in ("before", "after", "costs")
    ]
    valid_expert_dispositions = bool(
        isinstance(expert_dispositions, Mapping)
        and set(expert_dispositions) == set(expected_task_families)
        and all(
            isinstance(value, str) and value in allowed_expert_dispositions
            for key, value in expert_dispositions.items()
        )
    )
    valid_count_maps = bool(
        isinstance(count_maps[0], Mapping)
        and isinstance(count_maps[1], Mapping)
        and dict(count_maps[0]) == expected_benchmark_scores
        and dict(count_maps[1]) == expected_benchmark_scores
        and isinstance(count_maps[2], Mapping)
        and set(count_maps[2]) == {"tokens", "break_even"}
        and dict(count_maps[2]) == {"tokens": 0, "break_even": 0}
    )
    valid_gap_lists = bool(
        isinstance(gaps, Mapping)
        and all(
            isinstance(gaps.get(field), list)
            and len(gaps.get(field)) <= 4_096
            and all(
                isinstance(item, str) and 0 < len(item.strip()) <= 4_096
                for item in gaps.get(field)
            )
            for field in gap_fields
        )
    )
    if (
        not isinstance(report, Mapping)
        or set(report) != report_fields
        or report.get("schema") != VRIF_RELEASE_REPORT_SCHEMA
        or report.get("promotion_eligible") is not False
        or report.get("start_tree") != baseline_source.get("tree")
        or report.get("end_tree") != portal_baseline_tree
        or report.get("rollback_target") != baseline_source.get("commit")
        or report.get("corpus_admission_id") != admission_id
        or set(expert_dispositions) != set(expected_task_families)
        or any(
            value != "CAPABILITY_UNAVAILABLE"
            for value in expert_dispositions.values()
        )
        or report.get("producer_artifacts") != producer_artifacts
        or report.get("files_symbols") != expected_files_symbols
        or report.get("corpus_rights_splits") != expected_corpus_rights_splits
        or report.get("architecture_tokenizer_checkpoint")
        != expected_architecture_tokenizer_checkpoint
        or report.get("proof_validation") != expected_proof_validation
        or report.get("drift") != expected_drift
        or report.get("rollback_blocker_eligibility")
        != expected_rollback_blocker_eligibility
        or not bounded_text_fields
        or not valid_expert_dispositions
        or not valid_count_maps
        or not isinstance(gaps, Mapping)
        or set(gaps) != gap_fields
        or not valid_gap_lists
        or not isinstance(unsupported_claims, list)
        or unsupported_claims != exact_unsupported_claims
        or set(unsupported_claims) != forbidden_claims
        or gaps.get("blockers") != exact_blockers
        or gaps.get("not_run") != exact_not_run
        or markdown != _vrif_release_report_markdown(report)
    ):
        return None

    evidence: dict[str, Any] = {
        "schema": GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA,
        "terminal_report_contract_id": str(contract.get("contract_id") or ""),
        "task_cid": task_cid,
        "task_alias": task_alias,
        "task_revision": int(revision),
        "completion_receipt_cid": receipt_cid,
        "completion_evidence_digest": expected_evidence_digest,
        "control_receipt_id": content_identity(dict(control_receipt)),
        "portal_receipt_id": str(validation.get("portal_receipt_id") or ""),
        "portal_completion_binding": portal_completion_binding,
        "producer_receipts": producer_receipts,
        "producer_artifacts": producer_artifacts,
        "producer_receipt_bindings": producer_receipt_bindings,
        **validation_lineage,
        "report_artifacts": artifacts,
    }
    evidence["evidence_id"] = content_identity(evidence)
    return evidence


def _vrif_root_completion_gate(
    specification: Mapping[str, Any],
    admission: Mapping[str, Any],
    restart_receipt: Mapping[str, Any],
    connection: Any,
    *,
    runtime_settlement_binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        GOAL_ROOT_COMPLETION_GATE_SCHEMA,
    )

    receipt_id = str(restart_receipt.get("receipt_id") or "")
    admission_id = str(admission.get("admission_id") or "")
    if not receipt_id or not admission_id:
        raise OperatorError("VRIF root completion gate lacks owner restart authority")
    if not isinstance(runtime_settlement_binding, Mapping):
        return None
    state_owner = restart_receipt.get("state_owner")
    owner_generation = (
        state_owner.get("generation") if isinstance(state_owner, Mapping) else None
    )
    if (
        isinstance(owner_generation, bool)
        or not isinstance(owner_generation, int)
        or owner_generation < 1
    ):
        raise OperatorError("VRIF root completion gate has no owner generation")
    if runtime_settlement_binding.get("owner_generation") != owner_generation:
        raise OperatorError(
            "VRIF root completion gate runtime settlement generation differs"
        )
    terminal_report_evidence = _vrif_terminal_report_evidence(
        specification,
        admission,
        connection,
    )
    if terminal_report_evidence is None:
        return None

    def completion_gate(predecessor_gate_id: str) -> dict[str, Any]:
        gate: dict[str, Any] = {
            "schema": GOAL_ROOT_COMPLETION_GATE_SCHEMA,
            "authority_spec_id": str(specification.get("authority_spec_id") or ""),
            "source_head": str(admission.get("current_source_head") or ""),
            "repository_tree_id": str(admission.get("current_source_tree") or ""),
            "predecessor_gate_id": predecessor_gate_id,
            "owner_generation": owner_generation,
            "owner_restart_admission_id": admission_id,
            "owner_restart_receipt_id": receipt_id,
            "completion_policy": dict(specification.get("completion_policy") or {}),
            "runtime_settlement_binding": dict(runtime_settlement_binding),
            "terminal_report_evidence": dict(terminal_report_evidence),
        }
        gate["gate_id"] = content_identity(gate)
        return gate

    predecessor_gate_id = ""
    root_rows = _rows(
        connection,
        "SELECT status, body_json FROM goals WHERE goal_cid = ?",
        [str(specification.get("root_goal_cid") or "")],
    )
    if len(root_rows) != 1:
        raise OperatorError("VRIF root completion gate has no exact root row")
    root_status = str(_row_item(root_rows[0], 0, "status") or "").strip().lower()
    if root_status in {"completed", "complete", "done", "verified_complete"}:
        try:
            root_body = json.loads(
                str(_row_item(root_rows[0], 1, "body_json") or "{}")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        stored_receipt = (
            root_body.get("completion_receipt")
            if isinstance(root_body, Mapping)
            else None
        )
        stored_gate = (
            stored_receipt.get("root_completion_gate")
            if isinstance(stored_receipt, Mapping)
            else None
        )
        if not isinstance(stored_gate, Mapping):
            return None
        stored_gate_body = dict(stored_gate)
        stored_gate_id = str(stored_gate_body.pop("gate_id", "") or "")
        stored_generation = stored_gate.get("owner_generation")
        stored_source_head = str(stored_gate.get("source_head") or "")
        if (
            stored_gate.get("authority_spec_id")
            != specification.get("authority_spec_id")
            or stored_gate_id != content_identity(stored_gate_body)
            or isinstance(stored_generation, bool)
            or not isinstance(stored_generation, int)
            or stored_generation < 1
            or re.fullmatch(r"[0-9a-f]{40}", stored_source_head) is None
        ):
            return None
        if owner_generation < stored_generation:
            return None
        if owner_generation == stored_generation:
            same_generation_gate = completion_gate(
                str(stored_gate.get("predecessor_gate_id") or "")
            )
            return (
                same_generation_gate
                if same_generation_gate == dict(stored_gate)
                else None
            )
        try:
            _git_is_ancestor(
                stored_source_head,
                str(admission.get("current_source_head") or ""),
                field="completed root gate source lineage",
            )
        except OperatorError:
            return None
        predecessor_gate_id = stored_gate_id
    return completion_gate(predecessor_gate_id)


def _current_vrif_root_completion_gate(
    config: Mapping[str, Any],
    admission: Mapping[str, Any],
    gate: Mapping[str, Any] | None,
    *,
    runtime_settlement_binding: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Return the gate only while the owner's admitted clean tree is current."""

    if gate is None or not isinstance(runtime_settlement_binding, Mapping):
        return None
    try:
        head, tree = _assert_clean_current_tree(config)
    except OperatorError:
        return None
    if (
        head != str(admission.get("current_source_head") or "")
        or tree != str(admission.get("current_source_tree") or "")
        or head != str(gate.get("source_head") or "")
        or tree != str(gate.get("repository_tree_id") or "")
        or gate.get("runtime_settlement_binding")
        != dict(runtime_settlement_binding)
    ):
        return None
    return gate


def _reconcile_vrif_goal_completion(
    repository: Any,
    specification: Mapping[str, Any],
    *,
    root_completion_gate: Mapping[str, Any] | None,
    root_gate_current_validator: Any = None,
    conflict_retries: int = 3,
) -> Mapping[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepositoryConflictError,
    )

    last_conflict: BaseException | None = None
    for _attempt in range(conflict_retries):
        try:
            return repository.reconcile_goal_completion_authority(
                specification,
                root_completion_gate=root_completion_gate,
                root_gate_current_validator=root_gate_current_validator,
            )
        except IntentRepositoryConflictError as exc:
            last_conflict = exc
    raise OperatorError("VRIF goal completion CAS conflict retry was exhausted") from last_conflict


def _vrif_runtime_target(config: Mapping[str, Any]) -> tuple[str, str]:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )

    target_repository_id = checkout_repository_id(ROOT)
    target_branch = str(config.get("merge_target_branch") or "").strip()
    if (
        re.fullmatch(r"repository:baguqeera[a-z2-7]{52}", target_repository_id)
        is None
        or not target_branch
        or "\x00" in target_branch
    ):
        raise OperatorError("VRIF runtime settlement target identity is invalid")
    return target_repository_id, target_branch


def _vrif_owner_generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OperatorError("VRIF runtime settlement owner generation is invalid")
    return value


def _reconcile_vrif_goal_completion_under_runtime_guard(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    admission: Mapping[str, Any],
    restart_receipt: Mapping[str, Any],
    repository: Any,
    specification: Mapping[str, Any],
    connection: Any,
) -> Mapping[str, Any]:
    """Reconcile goals while retaining every runtime settlement lock.

    Runtime unavailability or a well-formed unsettled receipt only withholds the
    root gate; non-root goals still reconcile.  If a settled guard fails while
    leaving after an admitted root reconciliation, the owner fails closed
    instead of pretending that an already-committed root CAS did not occur.
    """

    from ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement import (
        VRIFRuntimeSettlementError,
        hold_vrif_runtime_settlement,
        vrif_runtime_settlement_binding,
    )

    target_repository_id, target_branch = _vrif_runtime_target(config)
    state_owner = restart_receipt.get("state_owner")
    owner_generation = _vrif_owner_generation(
        state_owner.get("generation") if isinstance(state_owner, Mapping) else None
    )
    reconciliation_completed = False
    admitted_root_gate: Mapping[str, Any] | None = None
    try:
        with hold_vrif_runtime_settlement(
            config_path,
            repository_root=ROOT,
            target_repository_id=target_repository_id,
            target_branch=target_branch,
            owner_generation=owner_generation,
            lock_timeout_seconds=VRIF_RUNTIME_SETTLEMENT_LOCK_TIMEOUT_SECONDS,
        ) as runtime_receipt:
            runtime_binding: Mapping[str, Any] | None = None
            if runtime_receipt.get("settled") is True:
                runtime_binding = vrif_runtime_settlement_binding(
                    runtime_receipt,
                    target_repository_id=target_repository_id,
                    target_branch=target_branch,
                    owner_generation=owner_generation,
                )
            root_gate = (
                _vrif_root_completion_gate(
                    specification,
                    admission,
                    restart_receipt,
                    connection,
                    runtime_settlement_binding=runtime_binding,
                )
                if runtime_binding is not None
                else None
            )
            admitted_root_gate = _current_vrif_root_completion_gate(
                config,
                admission,
                root_gate,
                runtime_settlement_binding=runtime_binding,
            )
            reconciliation = _reconcile_vrif_goal_completion(
                repository,
                specification,
                root_completion_gate=admitted_root_gate,
                root_gate_current_validator=(
                    lambda candidate: _current_vrif_root_completion_gate(
                        config,
                        admission,
                        candidate,
                        runtime_settlement_binding=runtime_binding,
                    )
                    is not None
                ),
            )
            reconciliation_completed = True
            return reconciliation
    except VRIFRuntimeSettlementError as exc:
        if reconciliation_completed and admitted_root_gate is not None:
            raise OperatorError(
                "VRIF runtime settlement changed across the root completion CAS"
            ) from exc
        return _reconcile_vrif_goal_completion(
            repository,
            specification,
            root_completion_gate=None,
            root_gate_current_validator=None,
        )


def _runtime_paths(board: Any) -> dict[str, Path]:
    program = board.resolved_database_program()
    database = _safe_path(ROOT, program.store_id, field="database_program.store_id")
    runtime = board.path(board.runtime_paths["root"])
    try:
        database.relative_to(runtime)
    except ValueError as exc:
        raise OperatorError("DuckDB authority store must be below runtime_paths.root") from exc
    raw_runtime = board.payload.get("runtime_paths")
    raw_runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    evidence = _safe_path(
        ROOT,
        raw_runtime.get("evidence") or runtime.relative_to(ROOT) / "evidence",
        field="runtime_paths.evidence",
    )
    owner = _safe_path(
        ROOT,
        raw_runtime.get("quack_owner") or runtime.relative_to(ROOT) / "quack-owner",
        field="runtime_paths.quack_owner",
    )
    raw_ducklake = board.payload.get("ducklake_projection_program")
    raw_ducklake = raw_ducklake if isinstance(raw_ducklake, Mapping) else {}
    ducklake_catalog = _safe_path(
        ROOT,
        raw_ducklake.get("catalog_path")
        or runtime.relative_to(ROOT) / "ducklake" / "catalog.duckdb",
        field="ducklake_projection_program.catalog_path",
    )
    ducklake_data = _safe_path(
        ROOT,
        raw_ducklake.get("data_path") or runtime.relative_to(ROOT) / "ducklake" / "data",
        field="ducklake_projection_program.data_path",
    )
    for label, path in (
        ("evidence", evidence),
        ("quack_owner", owner),
        ("ducklake_catalog", ducklake_catalog),
        ("ducklake_data", ducklake_data),
    ):
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError(f"{label} must be below runtime_paths.root") from exc
    return {
        "runtime": runtime,
        "database": database,
        "owner": owner,
        "bootstrap_receipt": evidence / "bootstrap" / BOOTSTRAP_RECEIPT_NAME,
        "ducklake_receipt": evidence / "bootstrap" / DUCKLAKE_RECEIPT_NAME,
        "ducklake_catalog": ducklake_catalog,
        "ducklake_data": ducklake_data,
    }


def _ducklake_projection(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one non-authoritative bootstrap observation to DuckLake."""

    projection: dict[str, Any] = {
        "schema": DUCKLAKE_SCHEMA,
        "authoritative": False,
        "scheduler_gate": False,
        "completion_gate": False,
        "status": "unavailable",
        "reason_code": "ducklake_projection_unavailable",
        "source_head": str(population["source_head"]),
        "repository_tree_id": str(population["repository_tree_id"]),
        "plan_root_cid": str(population["plan_root_cid"]),
    }
    try:
        import duckdb

        catalog = paths["ducklake_catalog"]
        data_path = paths["ducklake_data"]
        catalog.parent.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        memory = duckdb.connect(":memory:")
        try:
            memory.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data_path).replace("'", "''")
            memory.execute(
                f"ATTACH 'ducklake:{catalog_sql}' AS vrif_history (DATA_PATH '{data_sql}')"
            )
            memory.execute(
                """
                CREATE TABLE IF NOT EXISTS vrif_history.bootstrap_history (
                    event_id VARCHAR,
                    observed_at_epoch DOUBLE,
                    source_head VARCHAR,
                    repository_tree_id VARCHAR,
                    plan_root_cid VARCHAR,
                    projection_cid VARCHAR,
                    task_count BIGINT,
                    goal_count BIGINT,
                    body_json VARCHAR
                )
                """
            )
            event_id = _identity(
                {
                    "source_head": population["source_head"],
                    "plan_root_cid": population["plan_root_cid"],
                    "projection_cid": control_receipt.get("projection_cid"),
                }
            )
            existing = memory.execute(
                "SELECT COUNT(*) FROM vrif_history.bootstrap_history WHERE event_id = ?",
                [event_id],
            ).fetchone()
            if existing is None or int(existing[0]) == 0:
                memory.execute(
                    """
                    INSERT INTO vrif_history.bootstrap_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        event_id,
                        time.time(),
                        population["source_head"],
                        population["repository_tree_id"],
                        population["plan_root_cid"],
                        str(control_receipt.get("projection_cid") or ""),
                        int(control_receipt.get("task_count") or 0),
                        int(control_receipt.get("goal_count") or 0),
                        json.dumps(
                            {
                                "authority": "DuckDB/DatabaseTaskSource@1",
                                "transport": "QuackStateServer@1",
                                "projection": "DuckLake/non-authoritative",
                            },
                            sort_keys=True,
                        ),
                    ],
                )
            row_count = int(
                memory.execute("SELECT COUNT(*) FROM vrif_history.bootstrap_history").fetchone()[0]
            )
            memory.execute("DETACH vrif_history")
        finally:
            memory.close()
        projection.update(
            {
                "status": "available",
                "reason_code": "",
                "event_id": event_id,
                "row_count": row_count,
                "catalog_path": str(catalog.relative_to(ROOT)),
                "data_path": str(data_path.relative_to(ROOT)),
            }
        )
    except Exception as exc:
        # This projection is optional by contract. Preserve a typed absence and
        # never use it to reject a valid DuckDB materialization.
        projection["error_class"] = type(exc).__name__
    projection["projection_receipt_id"] = _identity(projection)
    _atomic_json(paths["ducklake_receipt"], projection)
    return projection


def _run_bootstrap_validation(*, board: Any, population: Mapping[str, Any]) -> dict[str, Any]:
    """Run the fixed hermetic qualification used for bootstrap completions."""

    commands = (
        (
            sys.executable,
            str(board.path(board.validator_path)),
            "--check-all",
            "--json",
        ),
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test/api/residual_intelligence",
        ),
    )
    observations: list[dict[str, Any]] = []
    for argv in commands:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=False,
            timeout=600,
            check=False,
        )
        observation = {
            "argv": list(argv),
            "returncode": int(completed.returncode),
            "stdout_digest": _identity(completed.stdout),
            "stderr_digest": _identity(completed.stderr),
        }
        observations.append(observation)
        if completed.returncode != 0:
            raise OperatorError("sealed bootstrap validation failed")
    receipt: dict[str, Any] = {
        "schema": "vrif-bootstrap-validation@1",
        "source_head": str(population["source_head"]),
        "repository_tree_id": str(population["repository_tree_id"]),
        "plan_root_cid": str(population["plan_root_cid"]),
        "commands": observations,
        "hermetic": True,
        "training_performed": False,
    }
    receipt["validation_digest"] = _identity(receipt)
    return receipt


def _seal_completed_tasks(
    source: Any,
    *,
    completed_aliases: Sequence[str],
    validation_receipt: Mapping[str, Any],
    population: Mapping[str, Any],
) -> list[str]:
    """Complete bootstrap tasks through evidence-gated CAS, never board status."""

    completion_receipt_cids: list[str] = []
    for alias in completed_aliases:
        task = source.get_task(alias)
        if task is None:
            raise OperatorError(f"bootstrap completion task is absent: {alias}")
        evidence_digest = _identity(
            {
                "task_cid": task.task_cid,
                "task_alias": alias,
                "validation_digest": validation_receipt["validation_digest"],
                "source_head": population["source_head"],
                "repository_tree_id": population["repository_tree_id"],
            }
        )
        source.intent.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed",
            evidence_digest=evidence_digest,
            argv=["sealed-vrif-bootstrap-validation"],
            body={
                "producer": "vrif-bootstrap-operator@1",
                "validation_receipt": dict(validation_receipt),
                "task_alias": alias,
            },
        )
        completion = source.intent.cas_task_status(
            task_cid=task.task_cid,
            expected_revision=task.revision,
            new_status="completed",
            evidence_digests=[evidence_digest],
            receipt={
                "schema": "vrif-bootstrap-completion@1",
                "producer": "vrif-bootstrap-operator@1",
                "candidate_only": False,
                "model_created": False,
                "source_head": population["source_head"],
                "repository_tree_id": population["repository_tree_id"],
                "validation_digest": validation_receipt["validation_digest"],
                "task_alias": alias,
            },
        )
        receipt_cid = str(completion.details.get("completion_receipt_cid") or "")
        if not receipt_cid:
            raise OperatorError(f"completion receipt missing for {alias}")
        completion_receipt_cids.append(receipt_cid)
    return completion_receipt_cids


def materialize(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    population = _population(board, config)
    receipt_path = paths["bootstrap_receipt"]
    if paths["database"].exists() or receipt_path.exists():
        if not paths["database"].is_file() or not receipt_path.is_file():
            raise OperatorError("partial bootstrap state exists; operator review required")
        prior = _json_object(receipt_path)
        exact = all(
            prior.get(key) == population.get(key)
            for key in ("source_head", "repository_tree_id", "plan_root_cid")
        )
        if not exact:
            raise OperatorError(
                "existing DuckDB authority is bound to a different source tree or plan"
            )
        with DatabaseTaskSource(
            paths["database"],
            owner_id="vrif-bootstrap:verify-existing",
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        ) as source:
            snapshot = source.snapshot().to_dict()
            ready_ids = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
            completion_projection = dict(source.intent.completion_evidence_projection())
        if int(snapshot["task_count"]) != len(population["tasks"]):
            raise OperatorError("existing DuckDB task population differs from sealed board")
        if int(snapshot["goal_count"]) != len(population["objectives"]):
            raise OperatorError("existing DuckDB goal population differs from sealed board")
        projection = config.get("initial_projection")
        projection = projection if isinstance(projection, Mapping) else {}
        expected_ready = [str(item) for item in projection.get("ready_task_ids", ())]
        if ready_ids != expected_ready:
            raise OperatorError("existing DuckDB readiness frontier differs from seal")
        expected_completed = {
            population["task_cids_by_alias"][str(alias)]
            for alias in projection.get("completed_task_ids", ())
        }
        receipt_tasks = {
            str(item["task_cid"]) for item in completion_projection["completion_receipts"]
        }
        if receipt_tasks != expected_completed:
            raise OperatorError("existing bootstrap completion receipts are incomplete")
        return {
            "schema": OPERATOR_SCHEMA,
            "command": "materialize",
            "idempotent_replay": True,
            "materialized": True,
            "bootstrap_receipt": prior,
            "snapshot": snapshot,
            "completion_evidence_projection_cid": completion_projection["projection_cid"],
        }

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    completed_aliases = [str(item) for item in projection.get("completed_task_ids", ())]
    if completed_aliases != [f"VRIF-{ordinal:03d}" for ordinal in range(9)]:
        raise OperatorError("bootstrap completion set must be exactly VRIF-000..008")
    validation_receipt = _run_bootstrap_validation(
        board=board,
        population=population,
    )
    ingestion_population = dict(population)
    ingestion_population["tasks"] = [
        {
            **task,
            "status": "todo" if task["task_alias"] in completed_aliases else task["status"],
        }
        for task in population["tasks"]
    ]
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    with DatabaseTaskSource(
        paths["database"],
        owner_id="vrif-bootstrap:single-writer",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    ) as source:
        control_receipt = dict(source.materialize(ingestion_population))
        completion_receipt_cids = _seal_completed_tasks(
            source,
            completed_aliases=completed_aliases,
            validation_receipt=validation_receipt,
            population=population,
        )
        snapshot = source.snapshot().to_dict()
        completion_projection = dict(source.intent.completion_evidence_projection())
        ready_ids = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    if int(snapshot["task_count"]) != len(population["tasks"]):
        raise OperatorError("DuckDB materialization task count is not exact")
    if int(snapshot["goal_count"]) != len(population["objectives"]):
        raise OperatorError("DuckDB materialization goal count is not exact")
    expected_ready_ids = [str(item) for item in projection.get("ready_task_ids", ())]
    if ready_ids != expected_ready_ids:
        raise OperatorError("initial DuckDB readiness frontier differs from the sealed projection")
    ducklake = _ducklake_projection(
        paths=paths,
        population=population,
        control_receipt=control_receipt,
    )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "source_identities": population["source_identities"],
        "source_forest": population["source_forest"],
        "database_task_source_receipt": control_receipt,
        "projection_cid": snapshot["projection_cid"],
        "task_count": snapshot["task_count"],
        "goal_count": snapshot["goal_count"],
        "dependency_count": snapshot["dependency_count"],
        "initial_ready_task_ids": ready_ids,
        "bootstrap_validation": validation_receipt,
        "completion_receipt_cids": completion_receipt_cids,
        "completion_evidence_projection_cid": completion_projection["projection_cid"],
        "authority": {
            "semantic_state": "DuckDB/DatabaseTaskSource@1",
            "state_owner_transport": "QuackStateServer@1",
            "ducklake": "optional_non_authoritative_history_projection",
        },
        "ducklake_projection": ducklake,
    }
    receipt["bootstrap_receipt_id"] = _identity(receipt)
    _atomic_json(receipt_path, receipt)
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "materialize",
        "idempotent_replay": False,
        "materialized": True,
        "bootstrap_receipt": receipt,
        "snapshot": snapshot,
    }


class _LiveQuackTransport:
    """Real loopback Quack transport with an identity-complete live probe."""

    def __init__(self) -> None:
        self._listen_uri = ""

    def start(
        self,
        connection: Any,
        *,
        host: str,
        port: int,
        token: str,
        identity: Any,
    ) -> Mapping[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            listen_uri,
        )

        uri = listen_uri(host, port)
        connection.execute(
            "SELECT * FROM quack_serve(?, token := ?, "
            "allow_other_hostname := false, disable_ssl := true)",
            [uri, token],
        )
        self._listen_uri = uri
        return MappingProxyType(
            {
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": uri,
            }
        )

    def live_query(
        self,
        connection: Any,
        *,
        identity: Any,
        token: str,
    ) -> Mapping[str, Any]:
        del token
        row = connection.execute("SELECT 1").fetchone()
        if row is None:
            raise OperatorError("Quack owner connection failed its live query")
        return MappingProxyType(
            {
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": self._listen_uri,
            }
        )

    def stop(self, connection: Any | None = None) -> None:
        if connection is None:
            return
        try:
            connection.execute("SELECT quack_stop()")
        except Exception:
            pass


def _verify_control_plane(path: Path) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        MigrationRunReport,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        CONTROL_PLANE_MIGRATION_VERSION,
        load_control_plane_catalog,
        verify_installed_schema,
    )

    # VRIF uses the canonical full control-plane schema revision ``1``.  The
    # smaller datasets-authoritative operational profile is deliberately not
    # selected: the generic multi-supervisor rejects that profile for live
    # Quack operation, and the VRIF board needs the full proof/evidence tables.
    verification = verify_installed_schema(path)
    fingerprint = str(verification.get("schema_fingerprint") or "")
    if not fingerprint:
        raise OperatorError("existing full control plane has no schema fingerprint")
    return MigrationRunReport(
        from_version=CONTROL_PLANE_MIGRATION_VERSION,
        to_version=CONTROL_PLANE_MIGRATION_VERSION,
        receipts=(),
        schema_fingerprint=fingerprint,
        catalog_fingerprint=load_control_plane_catalog().fingerprint(),
        changed=False,
    )


def _drop_fragile_task_status_indexes(connection: Any) -> None:
    """Drop ART indexes that fatally fail when ``tasks.status`` is updated.

    DuckDB 1.5 can abort the exclusive owner with
    ``Failed to delete all rows from index`` while CAS-ing ``blocked`` to
    ``retrying``. The VRIF board is 33 rows; status scans without these
    indexes remain exact.
    """

    for name in ("tasks_status_idx", "tasks_goal_idx"):
        try:
            connection.execute(f"DROP INDEX IF EXISTS {name}")
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "schema": OPERATOR_SCHEMA,
                        "event": "task_status_index_drop_failed",
                        "index": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def _owner_connection(path: Path) -> Any:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )

    connection = duckdb.connect(str(path))
    try:
        connection.execute("LOAD quack")
    except BaseException:
        connection.close()
        raise
    return DuckDBConnection.wrap(connection)


def _typed_deferral_repair_context(
    config: Mapping[str, Any],
    *,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
) -> dict[str, Any]:
    """Bind one blocked task generation to the exact clean repair HEAD."""

    source_head = str(task_body.get("base_revision") or "")
    source_tree = str(task_body.get("base_repository_tree_id") or "")
    repair_head_text = str(repair_head or "")
    repair_tree_text = str(repair_tree or "")
    if (
        not str(task_cid or "")
        or isinstance(task_revision, bool)
        or not isinstance(task_revision, int)
        or task_revision < 1
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or re.fullmatch(r"[0-9a-f]{40}", repair_head_text) is None
        or re.fullmatch(r"[0-9a-f]{40}", repair_tree_text) is None
        or repair_head_text == source_head
    ):
        raise OperatorError("typed-deferral repair generation is invalid")
    current_head, current_tree = _assert_clean_current_tree(config)
    if current_head != repair_head_text or current_tree != repair_tree_text:
        raise OperatorError(
            "typed-deferral recovery requires the exact clean current HEAD/tree"
        )
    if _git_commit_tree(source_head, field="typed-deferral source") != source_tree:
        raise OperatorError("typed-deferral source tree does not match its commit")
    if (
        _git_commit_tree(repair_head_text, field="typed-deferral repair")
        != repair_tree_text
    ):
        raise OperatorError("typed-deferral repair tree does not match its commit")
    _git_is_ancestor(
        source_head,
        repair_head_text,
        field="typed-deferral source-to-repair ancestry",
    )
    changed_raw = _git(
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        "--no-renames",
        source_head,
        repair_head_text,
        "--",
        *sorted(TYPED_DEFERRAL_RECOVERY_PRODUCTION_PATHS),
    )
    changed_paths = {
        line.strip()
        for line in str(changed_raw).splitlines()
        if line.strip()
    }
    if not changed_paths or not changed_paths.issubset(
        TYPED_DEFERRAL_RECOVERY_PRODUCTION_PATHS
    ):
        raise OperatorError(
            "typed-deferral repair changes no admitted production recovery path"
        )
    after_head, after_tree = _assert_clean_current_tree(config)
    if (after_head, after_tree) != (current_head, current_tree):
        raise OperatorError("typed-deferral repair generation changed during admission")
    return {
        "task_cid": str(task_cid),
        "task_revision": int(task_revision),
        "source_head": source_head,
        "source_tree": source_tree,
        "repair_head": repair_head_text,
        "repair_tree": repair_tree_text,
        "changed_production_paths": sorted(changed_paths),
    }


def _terminate_provider_canary(process: subprocess.Popen[bytes]) -> None:
    """Boundedly terminate one detached canary process group."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_typed_deferral_provider_canary(
    *, database_program: Any
) -> dict[str, Mapping[str, Any]]:
    """Run the real quota/high route in an inert disposable Git workspace."""

    from ipfs_accelerate_py.agent_implementation_route import (
        resolve_agent_implementation_route,
        valid_agent_implementation_failure_receipt,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner import (
        build_grok_quota_routed_agent_command,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        provider_subprocess_environment,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
        extract_grok_failure_receipts,
        extract_grok_route_outcomes,
        valid_grok_route_outcome,
    )

    route = resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_exhausted",
        fallback_reasoning_effort="high",
    )
    nonce = secrets.token_hex(32)
    provider_env = provider_subprocess_environment(
        os.environ,
        program=database_program,
    )
    grok = shutil.which("grok", path=provider_env.get("PATH")) or ""
    codex = shutil.which("codex", path=provider_env.get("PATH")) or ""
    if not grok or not codex:
        raise OperatorError("typed-deferral canary requires trusted Grok and Codex CLIs")
    git_env = {
        key: value
        for key, value in provider_env.items()
        if not key.startswith("GIT_")
    }
    git_env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    with tempfile.TemporaryDirectory(
        prefix="vrif-provider-route-canary-"
    ) as raw_workspace:
        workspace = Path(raw_workspace).resolve()
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(workspace)],
            env=git_env,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=VRIF Provider Canary",
                "-c",
                "user.email=vrif-canary.invalid",
                "commit",
                "--allow-empty",
                "--no-verify",
                "-q",
                "-m",
                "provider-route canary baseline",
            ],
            env=git_env,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        command = build_grok_quota_routed_agent_command(
            workspace=workspace,
            python_executable=sys.executable,
            grok_bin=grok,
            codex_bin=codex,
            fallback_reasoning_effort="high",
            accepted_runner_path=(
                ROOT
                / "ipfs_accelerate_py"
                / "agent_supervisor"
                / "runtime"
                / "grok_cli_runner.py"
            ),
        )
        command.extend(
            [
                "--grok-failure-receipt-nonce",
                nonce,
                "--agent-implementation-route-json",
                json.dumps(
                    route.as_binding_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
        prompt = (
            "Provider-route recovery canary only. Do not inspect or modify "
            "files and do not invoke tools. Reply with exactly CANARY_OK.\n"
        ).encode("utf-8")
        with tempfile.TemporaryFile() as control:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=git_env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=control,
                    start_new_session=True,
                )
            except OSError as exc:
                raise OperatorError("typed-deferral provider canary did not start") from exc
            try:
                assert process.stdin is not None
                process.stdin.write(prompt)
                process.stdin.close()
                deadline = (
                    time.monotonic()
                    + TYPED_DEFERRAL_PROVIDER_CANARY_TIMEOUT_SECONDS
                )
                while process.poll() is None:
                    if os.fstat(control.fileno()).st_size > (
                        TYPED_DEFERRAL_PROVIDER_CANARY_MAX_BYTES
                    ):
                        raise OperatorError(
                            "typed-deferral provider canary exceeded its log bound"
                        )
                    if time.monotonic() >= deadline:
                        raise OperatorError("typed-deferral provider canary timed out")
                    time.sleep(0.1)
            except BaseException:
                _terminate_provider_canary(process)
                raise
            control.flush()
            if os.fstat(control.fileno()).st_size > (
                TYPED_DEFERRAL_PROVIDER_CANARY_MAX_BYTES
            ):
                raise OperatorError(
                    "typed-deferral provider canary exceeded its log bound"
                )
            control.seek(0)
            control_text = control.read().decode("utf-8", errors="replace")

        receipts = extract_grok_failure_receipts(control_text)
        outcomes = extract_grok_route_outcomes(control_text)
        if len(receipts) != 1 or len(outcomes) != 1:
            raise OperatorError(
                "typed-deferral provider canary returned an ambiguous receipt chain"
            )
        receipt = receipts[0]
        outcome = outcomes[0]
        probe_returncode = receipt.get("probe_returncode")
        observed_now_ms = int(time.time() * 1000)
        valid = bool(
            process.returncode == 0
            and isinstance(probe_returncode, int)
            and not isinstance(probe_returncode, bool)
            and valid_agent_implementation_failure_receipt(
                receipt,
                nonce=nonce,
                model=route.primary_model_id,
                probe_returncode=probe_returncode,
                now_ms=observed_now_ms,
                max_age_ms=60_000,
            )
            and receipt.get("failure_class") == "hard_quota_exhausted"
            and valid_grok_route_outcome(
                outcome,
                receipt=receipt,
                route_plan=route.as_outcome_dict(),
                runner_returncode=process.returncode,
            )
            and outcome.get("decision") == "fallback_succeeded"
            and outcome.get("verifier_status") == "confirmed_quota"
            and outcome.get("fallback_dispatched") is True
            and outcome.get("fallback_returncode") == 0
            and bool(outcome.get("quota_evidence_id"))
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            env=git_env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not valid or status:
            raise OperatorError(
                "typed-deferral provider canary did not produce the exact "
                "fresh quota/high success chain"
            )
        return {
            "quota_probe_receipt": receipt,
            "route_outcome": outcome,
        }


def _owner_typed_deferral_provider_evidence(
    config: Mapping[str, Any],
    *,
    database_program: Any,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
) -> dict[str, Mapping[str, Any]]:
    """Validate Git on both sides of the owner-executed provider canary."""

    before = _typed_deferral_repair_context(
        config,
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
        repair_head=repair_head,
        repair_tree=repair_tree,
    )
    evidence = _run_typed_deferral_provider_canary(
        database_program=database_program,
    )
    after = _typed_deferral_repair_context(
        config,
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
        repair_head=repair_head,
        repair_tree=repair_tree,
    )
    if before != after:
        raise OperatorError("typed-deferral repair context changed during canary")
    return evidence


def _process_owner_commands(
    repository: Any,
    command_dir: Path,
    *,
    token: str,
    expected_store_id: str,
    expected_store_generation: str,
    typed_deferral_provider_evidence_factory: Any = None,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        execute_quack_owner_command,
        quack_owner_command_error_code,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
        quack_owner_command_response,
        validate_quack_owner_command_request,
    )

    command_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(command_dir, 0o700)
    for request in sorted(command_dir.glob("*.request.json")):
        done = request.with_name(request.name.replace(".request.json", ".done.json"))
        payload: Mapping[str, Any] = {}
        expected_request_id = request.name.removesuffix(".request.json")
        try:
            metadata = request.lstat()
            if not stat.S_ISREG(metadata.st_mode) or request.is_symlink():
                raise OperatorError("owner command must be a regular non-symlink file")
            if (
                metadata.st_uid != os.getuid()
                or metadata.st_size > OWNER_COMMAND_ENVELOPE_MAX_BYTES
            ):
                raise OperatorError("owner command file owner or size is invalid")
            try:
                decoded = json.loads(request.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # The client creates a tiny same-filesystem request. A partial
                # read is retried rather than converted into a false failure.
                continue
            if not isinstance(decoded, Mapping):
                raise OperatorError("owner command request must be an object")
            payload = decoded
            command, command_payload = validate_quack_owner_command_request(
                payload,
                token=token,
                expected_request_id=expected_request_id,
                expected_store_id=expected_store_id,
                expected_store_generation=expected_store_generation,
            )
            owner_bindings: dict[str, Any] = {
                "request_id": expected_request_id,
                "store_id": expected_store_id,
                "store_generation": expected_store_generation,
            }
            if command == QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET:
                if not callable(typed_deferral_provider_evidence_factory):
                    raise OperatorError(
                        "typed-deferral recovery provider boundary is unavailable"
                    )
                owner_bindings["typed_deferral_provider_evidence_factory"] = (
                    typed_deferral_provider_evidence_factory
                )
            result = execute_quack_owner_command(
                repository,
                command,
                command_payload,
                **owner_bindings,
            )
            _atomic_json(
                done,
                quack_owner_command_response(payload, token=token, result=result),
            )
        except Exception as exc:
            response_request = (
                payload
                if payload
                else {
                    "request_id": expected_request_id,
                    "command": "invalid",
                    "store_id": expected_store_id,
                    "store_generation": expected_store_generation,
                }
            )
            error_code = quack_owner_command_error_code(exc)
            if error_code == "owner_error":
                print(
                    json.dumps(
                        {
                            "schema": OPERATOR_SCHEMA,
                            "event": "owner_command_error",
                            "command": str(
                                (payload or {}).get("command") or "invalid"
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _atomic_json(
                done,
                quack_owner_command_response(
                    response_request,
                    token=token,
                    error_code=error_code,
                    error_message=(
                        str(exc)
                        if error_code != "owner_error"
                        else (
                            "typed owner command rejected "
                            f"({type(exc).__name__}: {str(exc)[:240]})"
                        )
                    ),
                ),
            )
        try:
            request.unlink()
        except FileNotFoundError:
            pass


def state_owner(config_path: Path) -> int:
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        ServerLifecycle,
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        IntentRepository,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    if not paths["database"].is_file() or not paths["bootstrap_receipt"].is_file():
        raise OperatorError("materialize the sealed VRIF board before starting Quack")
    prior_owner = _owner_restart_prior_status(
        paths["owner"] / "quack-state-server.status.json"
    )
    restart_admission = _owner_restart_admission(board, config, paths)
    program = board.resolved_database_program()
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    if endpoint is None:
        raise OperatorError("configured Quack endpoint is not loopback")
    host = endpoint.group(1)
    port = int(endpoint.group(2))
    if not 1 <= port <= 65535:
        raise OperatorError("configured Quack port is out of range")
    server = build_server(
        database_path=paths["database"],
        state_dir=paths["owner"],
        host=host,
        port=port,
        repository_id="repository:ipfs_accelerate_py",
        store_id=program.store_id,
        secret_handle=program.endpoint_secret_handle,
        # The installed Quack extension is a live-qualified beta build. This
        # permits that real transport; it never permits simulated inference.
        allow_experimental=True,
        migrate=_verify_control_plane,
        connection_factory=_owner_connection,
        transport=_LiveQuackTransport(),
    )
    identity = server.start()
    try:
        ready = server.ready()
        after_head, after_tree = _assert_clean_current_tree(config)
        if (
            after_head != restart_admission["current_source_head"]
            or after_tree != restart_admission["current_source_tree"]
        ):
            raise OperatorError("owner restart source changed during admission")
        owner_connection = getattr(server, "_connection", None)
        if owner_connection is None:
            raise OperatorError("state-owner connection is unavailable")
        _drop_fragile_task_status_indexes(owner_connection)
        database_verification = _owner_database_verification(
            owner_connection,
            restart_admission,
        )
        restart_receipt = _owner_restart_receipt(
            restart_admission,
            identity,
            expected_store_id=program.store_id,
            prior_owner=prior_owner,
            database_verification=database_verification,
        )
        restart_receipt_path = (
            paths["bootstrap_receipt"].parent
            / "owner-restarts"
            / (
                f"{int(identity.generation):020d}-"
                f"{restart_receipt['receipt_id'].removeprefix('sha256:')}.json"
            )
        )
        _atomic_json(restart_receipt_path, restart_receipt)
        owner_token = _read_owner_token(
            _token_path(paths["owner"], program.endpoint_secret_handle)
        )
        # The state owner retains the raw transport/command credential in memory.
        # Same-UID provider processes must not be able to recover it through procfs.
        os.environ["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = owner_token
        harden_state_authority_process()
        owner_repository = IntentRepository(
            paths["database"],
            bound_connection=owner_connection,
            owner_id="vrif-quack-owner",
            session_id=f"vrif-quack-owner-{os.getpid()}",
            install_schema=False,
        )
        goal_authority_spec = _vrif_goal_completion_authority_spec(
            board,
            config,
            restart_admission,
            owner_connection,
        )
        initial_goal_reconciliation = (
            _reconcile_vrif_goal_completion_under_runtime_guard(
                config_path=config_path,
                config=config,
                admission=restart_admission,
                restart_receipt=restart_receipt,
                repository=owner_repository,
                specification=goal_authority_spec,
                connection=owner_connection,
            )
        )
    except BaseException:
        server.stop()
        raise
    print(
        json.dumps(
            {
                "schema": OPERATOR_SCHEMA,
                "command": "state-owner",
                "ready": True,
                "identity": identity.to_dict(),
                "live": ready,
                "owner_restart_receipt": restart_receipt,
                "owner_restart_receipt_path": str(
                    restart_receipt_path.relative_to(ROOT)
                ),
                "owner_command_dir": str(
                    (paths["owner"] / "mutations").relative_to(ROOT)
                ),
                "goal_authority_spec_id": goal_authority_spec[
                    "authority_spec_id"
                ],
                "goal_reconciliation": {
                    "changed": initial_goal_reconciliation["changed"],
                    "changed_goal_ids": initial_goal_reconciliation[
                        "changed_goal_ids"
                    ],
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stopped = {"value": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stopped["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    command_dir = paths["owner"] / "mutations"
    control_path = server.stop_control_path()
    next_goal_reconcile = time.monotonic() + 1.0
    while server.lifecycle is ServerLifecycle.READY and not stopped["value"]:
        if control_path.is_file():
            break
        _process_owner_commands(
            owner_repository,
            command_dir,
            token=owner_token,
            expected_store_id=program.store_id,
            expected_store_generation=program.store_generation,
            typed_deferral_provider_evidence_factory=(
                lambda **context: _owner_typed_deferral_provider_evidence(
                    config,
                    database_program=program,
                    **context,
                )
            ),
        )
        if time.monotonic() >= next_goal_reconcile:
            reconciliation = _reconcile_vrif_goal_completion_under_runtime_guard(
                config_path=config_path,
                config=config,
                admission=restart_admission,
                restart_receipt=restart_receipt,
                repository=owner_repository,
                specification=goal_authority_spec,
                connection=owner_connection,
            )
            if reconciliation["changed"]:
                goal_authority = reconciliation.get("goal_authority")
                goal_authority = (
                    goal_authority if isinstance(goal_authority, Mapping) else {}
                )
                print(
                    json.dumps(
                        {
                            "schema": OPERATOR_SCHEMA,
                            "event": "goal_completion_reconciled",
                            "changed_goal_ids": reconciliation[
                                "changed_goal_ids"
                            ],
                            "all_goals_satisfied": bool(
                                goal_authority.get("all_goals_satisfied")
                            ),
                            "projection_cid": str(
                                goal_authority.get("projection_cid") or ""
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            next_goal_reconcile = time.monotonic() + 1.0
        time.sleep(0.05)
    owner_repository.close()
    result = server.stop()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _unlink_token_vault(path: Path) -> None:
    """Remove one validated token file after trusted processes inherit it."""

    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_nlink != 1
    ):
        raise OperatorError("refusing to unlink an unsafe Quack token vault")
    path.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _launch_with_one_use_owner_token(
    launch: Any,
    launch_args: Sequence[str],
    *,
    token_path: Path,
) -> int:
    """Consume the one-use vault only after the trusted launcher succeeds."""

    result = int(launch(list(launch_args)))
    if result == 0:
        _unlink_token_vault(token_path)
    return result


def launch_supervisor(
    config_path: Path,
    *,
    dry_run: bool = False,
    duration_seconds: float = float("inf"),
) -> int:
    """Launch the configured parallel supervisor without a credential file.

    Preflight runs while the owner token is still recoverable.  For a real
    launch, this process becomes non-dumpable, starts the trusted launcher,
    and unlinks the single validated token file only after that launcher
    returns success.  Failed launches retain the vault.  Provider subprocesses
    use the canonical scrubbed environment.
    """

    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        main as configured_board_main,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    _assert_clean_current_tree(config)
    owner_status_path = paths["owner"] / "quack-state-server.status.json"
    if not owner_status_path.is_file():
        raise OperatorError("Quack state owner has no current status")
    owner_status = _json_object(owner_status_path)
    if (
        str(owner_status.get("lifecycle") or "") != "ready"
        or _owner_liveness(owner_status) != "alive"
    ):
        raise OperatorError("Quack state owner is not live-ready")
    program = board.resolved_database_program()
    token_path = _token_path(paths["owner"], program.endpoint_secret_handle)
    token = _read_owner_token(token_path)
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = token
    # Do not ATTACH here.  A launch-time probe shares Quack's tiny listen
    # backlog with the four lanes and can make the live token fail closed
    # before any daemon claims work.  Owner liveness is already required.
    common = [
        "--repo-root",
        str(ROOT),
        "--config",
        str(config_path),
    ]
    preflight_result = configured_board_main([*common, "preflight"])
    if preflight_result != 0:
        return int(preflight_result)
    launch_args = [*common, "launch", "--implement"]
    if dry_run:
        launch_args.append("--dry-run")
        return int(configured_board_main(launch_args))
    if duration_seconds != float("inf"):
        if duration_seconds <= 0:
            raise OperatorError("supervisor duration must be positive")
        launch_args.extend(["--duration-seconds", str(duration_seconds)])
    harden_state_authority_process()
    return _launch_with_one_use_owner_token(
        configured_board_main,
        launch_args,
        token_path=token_path,
    )


def _owner_liveness(status_payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    identity = status_payload.get("identity")
    if not isinstance(identity, Mapping):
        return "absent"
    birth_payload = identity.get("process_birth")
    if not isinstance(birth_payload, Mapping):
        return "unknown"
    try:
        observed = owner_liveness(ProcessBirthIdentity.from_dict(birth_payload))
    except Exception:
        return "unknown"
    if observed is OwnerLiveness.ALIVE:
        return "alive"
    if observed is OwnerLiveness.DEAD:
        return "dead"
    return "unknown"


def _token_path(owner_dir: Path, secret_handle: str) -> Path:
    safe = secret_handle.replace(":", "_").replace("/", "_")
    return owner_dir / f"{safe}.quack-token"


def _read_owner_token(path: Path) -> str:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise OperatorError("Quack token vault file is not a private regular file")
    token = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,}", token):
        raise OperatorError("Quack token vault material is malformed")
    return token


def _probe_quack_attach(endpoint: str, token: str) -> None:
    """Fail closed if the vault token cannot attach to the live owner."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_transport_connection,
    )

    try:
        connection = open_quack_transport_connection(endpoint, token=token)
    except Exception as exc:
        raise OperatorError(
            "Quack vault token does not authenticate to the live state-owner"
        ) from exc
    try:
        connection.close()
    except Exception:
        pass


def _task_status(connection: Any) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
    ).fetchall()
    counts = {str(row[0]): int(row[1]) for row in rows}
    # The current Quack table transport supports simple scans but can reject a
    # correlated NOT EXISTS plan as unimplemented.  Read the three canonical
    # relations separately and calculate this read-only projection locally;
    # task/dependency/block rows remain authoritative in DuckDB.
    task_rows = connection.execute(
        "SELECT task_cid, task_alias, ordinal, status FROM tasks ORDER BY ordinal, task_alias"
    ).fetchall()
    dependency_rows = connection.execute(
        "SELECT task_cid, dependency_task_cid FROM task_dependencies"
    ).fetchall()
    blocked_rows = connection.execute(
        "SELECT task_cid FROM task_blocks WHERE state = 'active'"
    ).fetchall()
    status_by_cid = {str(row[0]): str(row[3]) for row in task_rows}
    dependencies_by_cid: dict[str, list[str]] = {}
    for row in dependency_rows:
        dependencies_by_cid.setdefault(str(row[0]), []).append(str(row[1]))
    actively_blocked = {str(row[0]) for row in blocked_rows}
    ready_ids = [
        str(row[1])
        for row in task_rows
        if str(row[3]) in READY_STATUSES
        and str(row[0]) not in actively_blocked
        and all(
            status_by_cid.get(dependency) in COMPLETED_STATUSES
            for dependency in dependencies_by_cid.get(str(row[0]), ())
        )
    ][:100]
    active_rows = connection.execute(
        "SELECT task_alias FROM tasks WHERE status IN (?, ?, ?) "
        "ORDER BY ordinal, task_alias LIMIT 100",
        list(ACTIVE_STATUSES),
    ).fetchall()
    return {
        "status_counts": counts,
        "dependency_ready_task_ids": ready_ids,
        "active_task_ids": [str(row[0]) for row in active_rows],
        "blocked_count": int(counts.get("blocked", 0)),
        "terminal_count": sum(counts.get(item, 0) for item in TERMINAL_STATUSES),
        "task_count": sum(counts.values()),
    }


def _quack_status_authority_snapshot(
    *,
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    program: Any,
    runtime_settlement_binding: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read task/spec/goal authority from one Quack MVCC snapshot."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_transport_connection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        goal_authority_projection_on_connection,
    )

    token = _read_owner_token(
        _token_path(paths["owner"], program.endpoint_secret_handle)
    )
    connection = open_quack_transport_connection(
        program.quack_endpoint,
        token=token,
    )
    transaction_open = False
    try:
        connection.execute("BEGIN TRANSACTION")
        transaction_open = True
        task_projection = {
            "available": True,
            "transport": "quack",
            **_task_status(connection),
        }
        admission = _owner_restart_admission(board, config, paths)
        specification = _vrif_goal_completion_authority_spec(
            board,
            config,
            admission,
            connection,
        )
        current_head, current_tree = _assert_clean_current_tree(config)
        projected = goal_authority_projection_on_connection(
            connection,
            specification,
            root_gate_context={
                "current_tree_clean": True,
                "source_head": current_head,
                "repository_tree_id": current_tree,
                "runtime_settlement_binding": (
                    dict(runtime_settlement_binding)
                    if isinstance(runtime_settlement_binding, Mapping)
                    else None
                ),
            },
            transaction_owned_by_caller=True,
        )
        goal_projection = {
            "available": True,
            "transport": "quack",
            **dict(projected),
        }
        connection.execute("COMMIT")
        transaction_open = False
        return task_projection, goal_projection
    except BaseException:
        if transaction_open:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass


def status(config_path: Path) -> dict[str, Any]:
    board, _config = _load_config(config_path)
    paths = _runtime_paths(board)
    program = board.resolved_database_program()
    state_status_path = paths["owner"] / "quack-state-server.status.json"
    owner_status: dict[str, Any] = {}
    if state_status_path.is_file():
        try:
            owner_status = _json_object(state_status_path)
        except OperatorError:
            owner_status = {"lifecycle": "malformed"}
    liveness = _owner_liveness(owner_status)
    lifecycle = str(owner_status.get("lifecycle") or "absent")
    live_ready = lifecycle == "ready" and liveness == "alive"
    task_projection: dict[str, Any] = {
        "available": False,
        "reason_code": "control_plane_unavailable",
    }
    goal_projection: dict[str, Any] = {
        "available": False,
        "reason_code": "control_plane_unavailable",
    }
    runtime_settlement: dict[str, Any] = {
        "available": False,
        "settled": False,
        "reason_code": "state_owner_unavailable",
    }
    try:
        if live_ready:
            try:
                from ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement import (
                    VRIFRuntimeSettlementError,
                    hold_vrif_runtime_settlement,
                    vrif_runtime_settlement_binding,
                )
                target_repository_id, target_branch = _vrif_runtime_target(_config)
                owner_identity = owner_status.get("identity")
                owner_generation = _vrif_owner_generation(
                    owner_identity.get("generation")
                    if isinstance(owner_identity, Mapping)
                    else None
                )
            except Exception as exc:
                runtime_settlement = {
                    "available": False,
                    "settled": False,
                    "reason_code": "runtime_settlement_configuration_invalid",
                    "error_class": type(exc).__name__,
                }
                try:
                    task_projection, goal_projection = (
                        _quack_status_authority_snapshot(
                            board=board,
                            config=_config,
                            paths=paths,
                            program=program,
                            runtime_settlement_binding=None,
                        )
                    )
                except Exception as probe_exc:
                    raise probe_exc
            else:
                try:
                    with hold_vrif_runtime_settlement(
                        config_path,
                        repository_root=ROOT,
                        target_repository_id=target_repository_id,
                        target_branch=target_branch,
                        owner_generation=owner_generation,
                        lock_timeout_seconds=(
                            VRIF_RUNTIME_SETTLEMENT_LOCK_TIMEOUT_SECONDS
                        ),
                    ) as runtime_receipt:
                        runtime_binding: Mapping[str, Any] | None = None
                        if runtime_receipt.get("settled") is True:
                            runtime_binding = vrif_runtime_settlement_binding(
                                runtime_receipt,
                                target_repository_id=target_repository_id,
                                target_branch=target_branch,
                                owner_generation=owner_generation,
                            )
                            runtime_settlement = {
                                "available": True,
                                "settled": True,
                                "reason_code": "settled",
                                "binding": dict(runtime_binding),
                            }
                        else:
                            active_counts = runtime_receipt.get("active_counts")
                            runtime_settlement = {
                                "available": True,
                                "settled": False,
                                "reason_code": "runtime_not_settled",
                                "active_counts": (
                                    dict(active_counts)
                                    if isinstance(active_counts, Mapping)
                                    else {}
                                ),
                            }
                        task_projection, goal_projection = (
                            _quack_status_authority_snapshot(
                                board=board,
                                config=_config,
                                paths=paths,
                                program=program,
                                runtime_settlement_binding=runtime_binding,
                            )
                        )
                except VRIFRuntimeSettlementError as exc:
                    runtime_settlement = {
                        "available": False,
                        "settled": False,
                        "reason_code": "runtime_settlement_unavailable",
                        "error_class": type(exc).__name__,
                    }
                    task_projection, goal_projection = (
                        _quack_status_authority_snapshot(
                            board=board,
                            config=_config,
                            paths=paths,
                            program=program,
                            runtime_settlement_binding=None,
                        )
                    )
        elif paths["database"].is_file():
            task_projection = {
                "available": False,
                "reason_code": "quack_authority_unavailable",
            }
            goal_projection = {
                "available": False,
                "reason_code": "quack_authority_unavailable",
            }
    except Exception as exc:
        probe_reason = (
            "goal_authority_probe_failed"
            if live_ready
            else "control_plane_probe_failed"
        )
        task_projection = {
            "available": False,
            "reason_code": probe_reason,
            "error_class": type(exc).__name__,
        }
        goal_projection = {
            "available": False,
            "reason_code": probe_reason,
            "error_class": type(exc).__name__,
        }
    ducklake: dict[str, Any] = {
        "status": "absent",
        "authoritative": False,
        "scheduler_gate": False,
    }
    if paths["ducklake_receipt"].is_file():
        try:
            observed = _json_object(paths["ducklake_receipt"])
            ducklake = {
                "status": str(observed.get("status") or "unknown"),
                "authoritative": False,
                "scheduler_gate": False,
                "projection_receipt_id": str(observed.get("projection_receipt_id") or ""),
            }
        except OperatorError:
            ducklake["status"] = "malformed"
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "status",
        "materialized": paths["database"].is_file() and paths["bootstrap_receipt"].is_file(),
        "state_owner": {
            "ready": live_ready,
            "lifecycle": lifecycle,
            "liveness": liveness,
            "identity": owner_status.get("identity"),
        },
        "task_authority": task_projection,
        "goal_authority": goal_projection,
        "runtime_settlement": runtime_settlement,
        "ducklake_projection": ducklake,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="repository-relative or absolute configured-board JSON",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "materialize",
        help="seal the committed Markdown bootstrap into DuckDB and DuckLake",
    )
    commands.add_parser(
        "state-owner",
        help="serve the materialized DuckDB authority through fenced loopback Quack",
    )
    status_parser = commands.add_parser(
        "status",
        help="report owner liveness and durable task readiness without exposing tokens",
    )
    status_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit nonzero unless Quack task and goal authority are queryable",
    )
    launch_parser = commands.add_parser(
        "launch-supervisor",
        help="preflight and launch the credential-isolated parallel supervisor",
    )
    launch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the launch without unlinking the state credential or starting workers",
    )
    launch_parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float("inf"),
        help="optional positive supervisor runtime bound",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        if arguments.command == "materialize":
            result = materialize(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if arguments.command == "state-owner":
            return state_owner(config_path)
        if arguments.command == "status":
            result = status(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            if arguments.require_ready and not (
                result["state_owner"]["ready"]
                and result["task_authority"].get("available") is True
                and result["goal_authority"].get("available") is True
            ):
                return 1
            return 0
        if arguments.command == "launch-supervisor":
            return launch_supervisor(
                config_path,
                dry_run=bool(arguments.dry_run),
                duration_seconds=float(arguments.duration_seconds),
            )
        raise OperatorError(f"unsupported command: {arguments.command}")
    except OperatorError as exc:
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        # Third-party transport exception text is not a trusted secret-
        # redaction surface, so unexpected failures publish only their class.
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": "operation failed closed",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
