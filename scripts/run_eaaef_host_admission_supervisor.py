#!/usr/bin/env python3
"""Host-controlled immutable EAAEF evidence supervisor.

The default ``--immutable-observation`` scope publishes source-addressed no-go
evidence and never opens the control/task DuckDB; its EAAEF-182 capability probe
may use only an isolated in-memory connection. ``--immutable-full-observation``
extends one explicitly selected current early observation across EAAEF-180..191
using static public evidence and likewise bypasses the control database, owner
lease, and status. Historical mutable bootstrap selectors remain parseable for
CLI compatibility but fail closed until the signed CASF owner fabric exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY_ROOT = ROOT

if TYPE_CHECKING:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

CURSOR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "generation-cursor.json"
)
STATUS_PATH = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "host-admission-supervisor-status.json"
)
BOARD_PATH = (
    ROOT
    / "docs/architecture/external_agent_autonomous_execution_fabric"
    / "task_board.json"
)
S_AUTO = {f"EAAEF-{number}" for number in range(180, 191)}
A_AUTO = {f"EAAEF-{number:03d}" for number in range(0, 10)}
HOST_AUTO = S_AUTO | A_AUTO | {"EAAEF-191"}
EARLY_FRONTIER = frozenset({"EAAEF-180", "EAAEF-181", "EAAEF-182", "EAAEF-183"})
EARLY_FRONTIER_ORDER = ("EAAEF-180", "EAAEF-181", "EAAEF-182", "EAAEF-183")
HOST_ADMISSION_ORDER = tuple(f"EAAEF-{number}" for number in range(180, 192))
BOOTSTRAP = HOST_AUTO
ADMIT_WAIT_STATUS = {
    "EAAEF-180": "waiting_current_blocker_inventory",
    "EAAEF-181": "waiting_current_runtime_principals",
    "EAAEF-182": "waiting_exact_duckdb_quack_155",
    "EAAEF-183": "waiting_rootless_engine",
    "EAAEF-184": "waiting_signed_provider_authorization",
    "EAAEF-185": "waiting_signed_worker_image",
    "EAAEF-186": "waiting_signed_execution_profile_v2",
    "EAAEF-187": "waiting_signed_worker_network_lanes",
    "EAAEF-188": "waiting_signed_command_fabric",
    "EAAEF-189": "waiting_signed_native_lane_dispatcher",
    "EAAEF-190": "waiting_signed_plan_r2_remote_owner",
    "EAAEF-191": "waiting_signed_admission_bundle",
}
COMPLETION_DECISIONS = {
    "EAAEF-180": frozenset({"inventory"}),
    "EAAEF-181": frozenset({"bound_unadmitted"}),
    **{f"EAAEF-{number}": frozenset({"admitted"}) for number in range(182, 192)},
}
S_PYTEST = {
    "EAAEF-180": "inventory",
    "EAAEF-181": "principals",
    "EAAEF-182": "duckdb_quack",
    "EAAEF-183": "engine_mode",
    "EAAEF-184": "provider_authorization",
    "EAAEF-185": "worker_image",
    "EAAEF-186": "container_profile",
    "EAAEF-187": "worker_network",
    "EAAEF-188": "command_fabric",
    "EAAEF-189": "native_lane",
    "EAAEF-190": "plan_r2",
    "EAAEF-191": "admission_bundle",
}
MAX_PASSES = 24


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded execution scope; an omitted scope is immutable-only."""

    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--early-frontier",
        dest="scope",
        action="store_const",
        const="early_frontier",
        default=argparse.SUPPRESS,
        help=(
            "disabled legacy mutable scope for EAAEF-180 through EAAEF-183; "
            "use --immutable-observation"
        ),
    )
    scope.add_argument(
        "--immutable-observation",
        dest="scope",
        action="store_const",
        const="immutable_observation",
        default=argparse.SUPPRESS,
        help=(
            "publish an immutable EAAEF-180 through EAAEF-183 no-go "
            "observation without opening the control/task DuckDB (default)"
        ),
    )
    scope.add_argument(
        "--immutable-full-observation",
        dest="scope",
        action="store_const",
        const="immutable_full_observation",
        default=argparse.SUPPRESS,
        help=(
            "publish a full immutable EAAEF-180 through EAAEF-191 no-go "
            "observation without opening the control/task DuckDB"
        ),
    )
    scope.add_argument(
        "--full-bootstrap",
        dest="scope",
        action="store_const",
        const="full_bootstrap",
        default=argparse.SUPPRESS,
        help=(
            "disabled legacy mutable scope for the complete S/A bootstrap; "
            "use --immutable-full-observation"
        ),
    )
    parser.add_argument(
        "--early-observation-cid",
        default=argparse.SUPPRESS,
        help=(
            "exact current early-frontier observation CID required by "
            "--immutable-full-observation"
        ),
    )
    args = parser.parse_args(argv)
    parsed_scope = str(getattr(args, "scope", "immutable_observation"))
    early_cid = str(getattr(args, "early_observation_cid", "") or "")
    if parsed_scope == "immutable_full_observation":
        if not _full_sha256_cid(early_cid):
            parser.error(
                "--immutable-full-observation requires a valid "
                "--early-observation-cid"
            )
    elif early_cid:
        parser.error(
            "--early-observation-cid is valid only with "
            "--immutable-full-observation"
        )
    return args


def _ensure_repository_importable() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _receipt_contract() -> tuple[Path, dict[str, str]]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        RECEIPT_DIR,
        RECEIPT_FILES,
    )

    return RECEIPT_DIR, RECEIPT_FILES


def _current_host_admission_identity() -> dict[str, str]:
    """Return the exact source and board identity receipts must bind."""

    revisions: dict[str, str] = {}
    for name, revision in (("source_head", "HEAD"), ("source_tree", "HEAD^{tree}")):
        completed = _run_argv(
            ["git", "rev-parse", "--verify", revision],
            SOURCE_REPOSITORY_ROOT,
            10,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0 or not value:
            raise RuntimeError(
                f"cannot resolve current EAAEF {name}: {completed.stderr.strip()}"
            )
        revisions[name] = value
    board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    board_namespace = str(board.get("board_namespace") or "")
    board_cid = str(board.get("board_cid") or "")
    if not board_namespace or not board_cid:
        raise RuntimeError("canonical EAAEF board identity is unavailable")
    return {
        **revisions,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
    }


def _verify_host_admission_task_receipt(
    *,
    task_id: str,
    receipt_dir: Path,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    """Call the canonical verifier and normalize every failure to a no-go."""

    try:
        from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
            verify_host_admission_task_receipt,
        )

        raw = verify_host_admission_task_receipt(
            task_id=task_id,
            receipt_dir=receipt_dir,
            expected_source_head=expected_identity["source_head"],
            expected_source_tree=expected_identity["source_tree"],
            expected_board_namespace=expected_identity["board_namespace"],
            expected_board_cid=expected_identity["board_cid"],
        )
    except Exception as exc:  # A verifier failure must never admit a dependency.
        return {
            "valid": False,
            "decision": "",
            "blockers": [
                f"host receipt verification failed: {type(exc).__name__}: {exc}"
            ],
        }
    if not isinstance(raw, dict):
        return {
            "valid": False,
            "decision": "",
            "blockers": ["host receipt verifier returned a non-object result"],
        }
    raw_blockers = raw.get("blockers")
    blockers = (
        [str(item) for item in raw_blockers if str(item)]
        if isinstance(raw_blockers, (list, tuple))
        else []
    )
    result: dict[str, Any] = {
        "valid": raw.get("valid") is True,
        "decision": str(raw.get("decision") or ""),
        "blockers": blockers,
    }
    receipt_cid = raw.get("receipt_cid")
    if (
        isinstance(receipt_cid, str)
        and len(receipt_cid) == 71
        and receipt_cid.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in receipt_cid[7:])
    ):
        result["receipt_cid"] = receipt_cid
    elif task_id == "EAAEF-191" and receipt_cid is not None:
        result["valid"] = False
        result["blockers"].append("EAAEF-191 verified receipt CID is invalid")
    return result


def _host_receipt_completion_verdict(
    *,
    task_id: str,
    receipt_dir: Path,
    receipt_path: Path,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    """Return whether one receipt may satisfy its task and dependencies."""

    verification = _verify_host_admission_task_receipt(
        task_id=task_id,
        receipt_dir=receipt_dir,
        expected_identity=expected_identity,
    )
    blockers = list(verification["blockers"])
    receipt_cid = str(verification.get("receipt_cid") or "")
    receipt_present = bool(receipt_cid) if task_id == "EAAEF-191" else receipt_path.is_file()
    if not receipt_present:
        blockers.append(f"{task_id} host receipt is missing")
    decision = str(verification["decision"])
    allowed_decisions = COMPLETION_DECISIONS.get(task_id, frozenset())
    if decision not in allowed_decisions:
        blockers.append(
            f"{task_id} decision {decision or '<missing>'!r} is not "
            "completion-authorizing"
        )
    blockers = list(dict.fromkeys(blockers))
    return {
        "valid": verification["valid"],
        "decision": decision,
        "receipt_cid": receipt_cid,
        "blockers": blockers,
        "completion_allowed": (
            verification["valid"] is True
            and receipt_present
            and decision in allowed_decisions
            and not blockers
        ),
    }


def _host_receipt_evidence_reference(
    *,
    task_id: str,
    receipt_path: Path,
    verdict: dict[str, Any],
    expected_identity: dict[str, str],
) -> tuple[str, str]:
    """Return the reviewed evidence path and digest for one verified receipt."""

    if task_id == "EAAEF-191":
        _ensure_repository_importable()
        from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
            source_addressed_admission_bundle_logical_paths,
        )

        logical_path, _logical_signatures_path = (
            source_addressed_admission_bundle_logical_paths(
                source_head=expected_identity["source_head"],
            )
        )
        return logical_path.as_posix(), str(verdict.get("receipt_cid") or "")
    digest = _cid_bytes(receipt_path.read_bytes()) if receipt_path.is_file() else ""
    return str(receipt_path.relative_to(ROOT)), digest


def _collect_host_admission() -> dict[str, Any]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_early_frontier_and_write,
    )

    return collect_early_frontier_and_write()


def _collect_immutable_observation() -> dict[str, Any]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_early_frontier_and_publish_observation,
    )

    return collect_early_frontier_and_publish_observation()


def _collect_immutable_host_admission_observation(
    early_observation_cid: str,
) -> dict[str, Any]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_host_admission_and_publish_observation,
    )

    return collect_host_admission_and_publish_observation(
        early_frontier_observation_cid=early_observation_cid
    )


def _full_sha256_cid(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _validated_early_frontier_collection(
    collection: object,
    *,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    """Accept only the exact immutable no-go publisher result."""

    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        EARLY_FRONTIER_OBSERVATION_DB_BINDING_BLOCKER,
        EARLY_FRONTIER_OBSERVATION_SCHEMA,
        EARLY_FRONTIER_OBSERVATION_SCOPE,
        source_addressed_early_frontier_observation_logical_path,
    )

    if not isinstance(collection, dict):
        raise RuntimeError("early-frontier collection is not an object")
    expected_fields = {
        "schema",
        "scope",
        "published",
        "created",
        "logical_path",
        "observation_cid",
        "child_receipt_cids",
        "typed_missing_task_ids",
        "direct_completion_eligible_task_ids",
        "casf_owner_transaction_eligible_task_ids",
        "decisions",
        "source_head",
        "source_tree",
        "board_namespace",
        "board_cid",
        "decision",
        "process_started",
        "supervisor_process_started",
        "configured_board_launch",
        "provider_invoked",
        "observation_only",
        "admission_authority",
        "live_admission_allowed",
        "live_launch_allowed",
        "eaaef_191_authority",
        "direct_task_completion_allowed",
        "direct_database_binding_allowed",
        "casf_owner_binding_required",
        "database_binding_blocker",
    }
    if set(collection) != expected_fields:
        raise RuntimeError("early-frontier collection fields differ")
    if (
        collection.get("schema") != EARLY_FRONTIER_OBSERVATION_SCHEMA
        or collection.get("scope") != EARLY_FRONTIER_OBSERVATION_SCOPE
        or collection.get("published") is not True
        or type(collection.get("created")) is not bool
        or collection.get("decision") != "no_go"
        or collection.get("observation_only") is not True
    ):
        raise RuntimeError("early-frontier collection identity differs")
    current_identity = {
        field: str(collection.get(field) or "")
        for field in ("source_head", "source_tree", "board_namespace", "board_cid")
    }
    if current_identity != expected_identity:
        raise RuntimeError("early-frontier collection source or board differs")
    false_fields = (
        "process_started",
        "supervisor_process_started",
        "configured_board_launch",
        "provider_invoked",
        "admission_authority",
        "live_admission_allowed",
        "live_launch_allowed",
        "eaaef_191_authority",
        "direct_task_completion_allowed",
        "direct_database_binding_allowed",
    )
    if any(collection.get(field) is not False for field in false_fields):
        raise RuntimeError("early-frontier collection widened authority")
    if (
        collection.get("casf_owner_binding_required") is not True
        or collection.get("database_binding_blocker")
        != EARLY_FRONTIER_OBSERVATION_DB_BINDING_BLOCKER
    ):
        raise RuntimeError("early-frontier collection CASF-owner boundary differs")
    observation_cid = collection.get("observation_cid")
    if not _full_sha256_cid(observation_cid):
        raise RuntimeError("early-frontier collection observation CID is invalid")
    expected_logical_path = (
        source_addressed_early_frontier_observation_logical_path(
            source_head=expected_identity["source_head"],
            observation_cid=str(observation_cid),
        ).as_posix()
    )
    if collection.get("logical_path") != expected_logical_path:
        raise RuntimeError("early-frontier collection logical path differs")
    decisions = collection.get("decisions")
    child_cids = collection.get("child_receipt_cids")
    if (
        not isinstance(decisions, dict)
        or tuple(decisions) != EARLY_FRONTIER_ORDER
        or not isinstance(child_cids, dict)
        or tuple(child_cids) != EARLY_FRONTIER_ORDER
        or any(not _full_sha256_cid(child_cids[task_id]) for task_id in EARLY_FRONTIER_ORDER)
    ):
        raise RuntimeError("early-frontier collection child bindings differ")
    allowed_decisions = {
        "EAAEF-180": {"inventory"},
        "EAAEF-181": {"bound_unadmitted"},
        "EAAEF-182": {"admitted", "typed_missing"},
        "EAAEF-183": {"admitted", "typed_missing"},
    }
    if any(
        decisions[task_id] not in allowed_decisions[task_id]
        for task_id in EARLY_FRONTIER_ORDER
    ):
        raise RuntimeError("early-frontier collection child decision differs")
    typed_missing = [
        task_id
        for task_id in EARLY_FRONTIER_ORDER
        if decisions[task_id] == "typed_missing"
    ]
    owner_eligible = [
        task_id
        for task_id in EARLY_FRONTIER_ORDER
        if decisions[task_id]
        == {
            "EAAEF-180": "inventory",
            "EAAEF-181": "bound_unadmitted",
            "EAAEF-182": "admitted",
            "EAAEF-183": "admitted",
        }[task_id]
    ]
    if (
        collection.get("typed_missing_task_ids") != typed_missing
        or collection.get("direct_completion_eligible_task_ids") != []
        or collection.get("casf_owner_transaction_eligible_task_ids")
        != owner_eligible
    ):
        raise RuntimeError("early-frontier collection completion boundary differs")
    return collection


def _validated_host_admission_observation_collection(
    collection: object,
    *,
    expected_identity: dict[str, str],
    expected_early_observation_cid: str,
) -> dict[str, Any]:
    """Accept only the exact full S-frontier immutable no-go result."""

    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        HOST_ADMISSION_OBSERVATION_DB_BINDING_BLOCKER,
        HOST_ADMISSION_OBSERVATION_SCHEMA,
        HOST_ADMISSION_OBSERVATION_SCOPE,
        source_addressed_early_frontier_observation_logical_path,
        source_addressed_host_admission_observation_logical_path,
    )

    if not isinstance(collection, dict):
        raise RuntimeError("host-admission observation collection is not an object")
    expected_fields = {
        "schema",
        "scope",
        "published",
        "created",
        "logical_path",
        "observation_cid",
        "early_frontier_observation_cid",
        "early_frontier_observation_logical_path",
        "child_receipt_cids",
        "typed_missing_task_ids",
        "no_go_task_ids",
        "direct_completion_eligible_task_ids",
        "casf_owner_transaction_eligible_task_ids",
        "decisions",
        "source_head",
        "source_tree",
        "board_namespace",
        "board_cid",
        "decision",
        "process_started",
        "supervisor_process_started",
        "configured_board_launch",
        "provider_invoked",
        "staging_receipts_written",
        "control_database_opened",
        "database_state_observed",
        "observation_only",
        "admission_authority",
        "live_admission_allowed",
        "live_launch_allowed",
        "eaaef_191_authority",
        "direct_task_completion_allowed",
        "direct_database_binding_allowed",
        "casf_owner_binding_required",
        "database_binding_blocker",
    }
    if set(collection) != expected_fields:
        raise RuntimeError("host-admission observation collection fields differ")
    if (
        collection.get("schema") != HOST_ADMISSION_OBSERVATION_SCHEMA
        or collection.get("scope") != HOST_ADMISSION_OBSERVATION_SCOPE
        or collection.get("published") is not True
        or type(collection.get("created")) is not bool
        or collection.get("decision") != "no_go"
        or collection.get("observation_only") is not True
    ):
        raise RuntimeError("host-admission observation collection identity differs")
    current_identity = {
        field: str(collection.get(field) or "")
        for field in ("source_head", "source_tree", "board_namespace", "board_cid")
    }
    if current_identity != expected_identity:
        raise RuntimeError("host-admission observation source or board differs")
    false_fields = (
        "process_started",
        "supervisor_process_started",
        "configured_board_launch",
        "provider_invoked",
        "staging_receipts_written",
        "control_database_opened",
        "database_state_observed",
        "admission_authority",
        "live_admission_allowed",
        "live_launch_allowed",
        "eaaef_191_authority",
        "direct_task_completion_allowed",
        "direct_database_binding_allowed",
    )
    if any(collection.get(field) is not False for field in false_fields):
        raise RuntimeError("host-admission observation widened authority")
    if (
        collection.get("early_frontier_observation_cid")
        != expected_early_observation_cid
        or collection.get("early_frontier_observation_logical_path")
        != source_addressed_early_frontier_observation_logical_path(
            source_head=expected_identity["source_head"],
            observation_cid=expected_early_observation_cid,
        ).as_posix()
    ):
        raise RuntimeError("host-admission observation early parent differs")
    if (
        collection.get("casf_owner_binding_required") is not True
        or collection.get("database_binding_blocker")
        != HOST_ADMISSION_OBSERVATION_DB_BINDING_BLOCKER
    ):
        raise RuntimeError("host-admission observation CASF-owner boundary differs")
    observation_cid = collection.get("observation_cid")
    if not _full_sha256_cid(observation_cid):
        raise RuntimeError("host-admission observation CID is invalid")
    expected_logical_path = (
        source_addressed_host_admission_observation_logical_path(
            source_head=expected_identity["source_head"],
            observation_cid=str(observation_cid),
        ).as_posix()
    )
    if collection.get("logical_path") != expected_logical_path:
        raise RuntimeError("host-admission observation logical path differs")
    decisions = collection.get("decisions")
    child_cids = collection.get("child_receipt_cids")
    if (
        not isinstance(decisions, dict)
        or tuple(decisions) != HOST_ADMISSION_ORDER
        or not isinstance(child_cids, dict)
        or tuple(child_cids) != HOST_ADMISSION_ORDER
        or any(
            not _full_sha256_cid(child_cids[task_id])
            for task_id in HOST_ADMISSION_ORDER
        )
    ):
        raise RuntimeError("host-admission observation child bindings differ")
    allowed_decisions = {
        "EAAEF-180": {"inventory"},
        "EAAEF-181": {"bound_unadmitted"},
        "EAAEF-182": {"admitted", "typed_missing"},
        "EAAEF-183": {"admitted", "typed_missing"},
        **{f"EAAEF-{number}": {"typed_missing"} for number in range(184, 191)},
        "EAAEF-191": {"no_go"},
    }
    if any(
        decisions[task_id] not in allowed_decisions[task_id]
        for task_id in HOST_ADMISSION_ORDER
    ):
        raise RuntimeError("host-admission observation child decision differs")
    typed_missing = [
        task_id
        for task_id in HOST_ADMISSION_ORDER
        if decisions[task_id] == "typed_missing"
    ]
    accepted = {
        "EAAEF-180": "inventory",
        "EAAEF-181": "bound_unadmitted",
        "EAAEF-182": "admitted",
        "EAAEF-183": "admitted",
    }
    owner_eligible = [
        task_id
        for task_id in EARLY_FRONTIER_ORDER
        if decisions[task_id] == accepted[task_id]
    ]
    if (
        collection.get("typed_missing_task_ids") != typed_missing
        or collection.get("no_go_task_ids") != ["EAAEF-191"]
        or collection.get("direct_completion_eligible_task_ids") != []
        or collection.get("casf_owner_transaction_eligible_task_ids")
        != owner_eligible
    ):
        raise RuntimeError("host-admission observation completion boundary differs")
    return collection


def _collect_full_host_admission() -> dict[str, Any]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_and_write,
    )

    return collect_and_write()


def _database_task_source_class() -> type[DatabaseTaskSource]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    return DatabaseTaskSource


def _acquire_state_owner_lease(control: Path) -> Any:
    """Acquire the same OS lease/fence used by the live Quack owner.

    The caller must acquire this before collecting/writing host receipts or
    constructing ``DatabaseTaskSource``.  A competing live owner therefore
    fails this offline path before any DuckDB connection can be opened.
    """

    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OWNER_LOCK_SUFFIX,
        OWNER_MARKER_SUFFIX,
        ExclusiveOwnerLease,
    )

    database = Path(control)
    lease = ExclusiveOwnerLease(
        lock_path=database.with_name(f".{database.name}{OWNER_LOCK_SUFFIX}"),
        marker_path=database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}"),
    )
    lease.acquire(
        server_id=f"offline:eaaef-host-admission:{os.getpid()}",
        process_birth=current_process_birth(),
        database_path=database,
        generation=1,
    )
    return lease


def _cid_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _active_control_db() -> Path:
    if CURSOR.is_file():
        cursor = json.loads(CURSOR.read_text(encoding="utf-8"))
        generation = str(cursor.get("active_generation") or "eaaef-run-v7")
    else:
        generation = "eaaef-run-v7"
    number = generation.rsplit("-v", 1)[-1]
    return (
        ROOT
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / f"run-v{number}"
        / "control.duckdb"
    )


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _board_validations(alias: str) -> list[dict[str, object]]:
    board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    for task in board.get("tasks") or ():
        if str(task.get("stable_task_id") or "") == alias:
            return list(task.get("execution_validation") or [])
    return []


def _run_argv(
    argv: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _is_pytest_argv(argv: list[str]) -> bool:
    return "-m" in argv and "pytest" in argv


def _run_pytest_file_isolation(
    argv: list[str],
    cwd: Path,
    timeout: int,
    stdout: str,
) -> dict:
    from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
        pytest_isolation_argv,
        pytest_isolation_files,
    )

    files = pytest_isolation_files(argv=argv, stdout=stdout)
    if not files:
        return {"passed": False, "reason": "no_pytest_files_to_isolate", "results": []}
    started = time.time()
    results: list[dict] = []
    remaining = timeout
    for path in files:
        if remaining <= 1:
            return {
                "passed": False,
                "reason": "isolation_timeout",
                "results": results,
            }
        completed = _run_argv(pytest_isolation_argv(path), cwd, remaining)
        remaining = max(1, timeout - int(time.time() - started))
        results.append(
            {
                "path": path,
                "returncode": completed.returncode,
                "passed": completed.returncode == 0,
                "stdout": completed.stdout[-400:],
                "stderr": completed.stderr[-200:],
            }
        )
        if completed.returncode != 0:
            return {
                "passed": False,
                "reason": "isolated_pytest_file_failed",
                "failed_path": path,
                "results": results,
            }
    return {
        "passed": True,
        "reason": "isolated_pytest_files_passed",
        "results": results,
    }


def _complete_s_task(
    source: DatabaseTaskSource,
    alias: str,
    expected_identity: dict[str, str],
) -> dict:
    receipt_dir, receipt_files = _receipt_contract()
    task = source.get_task(alias)
    if task is None:
        return {"task_id": alias, "status": "missing"}
    receipt_name = receipt_files[alias]
    receipt_path = receipt_dir / receipt_name
    verdict = _host_receipt_completion_verdict(
        task_id=alias,
        receipt_dir=receipt_dir,
        receipt_path=receipt_path,
        expected_identity=expected_identity,
    )
    if verdict["completion_allowed"] is not True:
        return {
            "task_id": alias,
            "status": ADMIT_WAIT_STATUS.get(
                alias, "waiting_current_valid_host_admission_receipt"
            ),
            "decision": verdict["decision"],
            "receipt_valid": verdict["valid"],
            "blockers": verdict["blockers"],
        }
    if task.status == "completed":
        return {
            "task_id": alias,
            "status": "already_completed",
            "decision": verdict["decision"],
            "receipt_valid": True,
        }
    evidence_path, receipt_digest_before = _host_receipt_evidence_reference(
        task_id=alias,
        receipt_path=receipt_path,
        verdict=verdict,
        expected_identity=expected_identity,
    )
    validation = [
        "python3",
        "-m",
        "pytest",
        "-q",
        "test/api/test_eaaef_host_admission_unblocking.py",
        "-k",
        S_PYTEST[alias],
    ]
    completed = _run_argv(validation, ROOT, 180)
    digest = _cid_bytes(
        (completed.stdout + completed.stderr + str(completed.returncode)).encode()
    )
    source.record_validation_result(
        task_cid=task.task_cid,
        outcome="passed" if completed.returncode == 0 else "failed",
        evidence_digest=digest,
        argv=validation,
    )
    if completed.returncode != 0:
        return {
            "task_id": alias,
            "status": "validation_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-400:],
        }
    final_verdict = _host_receipt_completion_verdict(
        task_id=alias,
        receipt_dir=receipt_dir,
        receipt_path=receipt_path,
        expected_identity=expected_identity,
    )
    final_evidence_path, receipt_digest_after = _host_receipt_evidence_reference(
        task_id=alias,
        receipt_path=receipt_path,
        verdict=final_verdict,
        expected_identity=expected_identity,
    )
    if (
        final_verdict["completion_allowed"] is not True
        or final_evidence_path != evidence_path
        or receipt_digest_after != receipt_digest_before
    ):
        blockers = list(final_verdict["blockers"])
        if (
            final_evidence_path != evidence_path
            or receipt_digest_after != receipt_digest_before
        ):
            blockers.append(f"{alias} host receipt changed during validation")
        return {
            "task_id": alias,
            "status": "receipt_changed_or_revoked",
            "decision": final_verdict["decision"],
            "receipt_valid": final_verdict["valid"],
            "blockers": list(dict.fromkeys(blockers)),
        }
    source.record_evidence(
        task_cid=task.task_cid,
        evidence_kind="host_admission_receipt",
        digest=receipt_digest_after,
        body={
            "path": final_evidence_path,
            "source_head": expected_identity["source_head"],
            "source_tree": expected_identity["source_tree"],
            "board_namespace": expected_identity["board_namespace"],
            "board_cid": expected_identity["board_cid"],
            "decision": final_verdict["decision"],
        },
    )
    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {"validation": "passed", "host_controlled": True},
        evidence_digests=[digest, receipt_digest_after],
    )
    return {
        "task_id": alias,
        "status": result.task.status,
        "changed": result.changed,
        "receipt_cid": result.receipt_cid,
    }


def _complete_a_task(source: DatabaseTaskSource, alias: str) -> dict:
    task = source.get_task(alias)
    if task is None:
        return {"task_id": alias, "status": "missing"}
    if task.status == "completed":
        return {"task_id": alias, "status": "already_completed"}
    commands = _board_validations(alias)
    if not commands:
        return {"task_id": alias, "status": "missing_execution_validation"}
    digests: list[str] = []
    isolation: dict | None = None
    for item in commands:
        argv = [str(part) for part in item.get("argv") or ()]
        working = str(item.get("working_directory") or ".")
        if not argv:
            return {"task_id": alias, "status": "empty_execution_validation"}
        cwd = ROOT if working in {".", ""} else ROOT / working
        started = time.time()
        completed = _run_argv(argv, cwd, 1800)
        digest = _cid_bytes(
            (completed.stdout + completed.stderr + str(completed.returncode)).encode()
        )
        digests.append(digest)
        outcome = "passed" if completed.returncode == 0 else "failed"
        isolation = None
        if completed.returncode != 0 and _is_pytest_argv(argv):
            remaining = max(60, 1800 - int(time.time() - started))
            isolation = _run_pytest_file_isolation(
                argv, cwd, remaining, completed.stdout + completed.stderr
            )
            if isolation.get("passed") is True:
                outcome = "passed"
                isolation_digest = _cid_bytes(
                    json.dumps(isolation, sort_keys=True).encode()
                )
                source.record_validation_result(
                    task_cid=task.task_cid,
                    outcome="passed",
                    evidence_digest=isolation_digest,
                    argv=["python3", "-m", "pytest", "-q", "--eaaef-file-isolation"],
                )
                digests.append(isolation_digest)
        source.record_validation_result(
            task_cid=task.task_cid,
            outcome=outcome,
            evidence_digest=digest,
            argv=argv,
        )
        if outcome != "passed":
            payload = {
                "task_id": alias,
                "status": "validation_failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-1200:],
                "stderr": completed.stderr[-800:],
                "argv": argv,
            }
            if isolation is not None:
                payload["pytest_file_isolation"] = isolation
            return payload
    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {
            "validation": "passed",
            "host_controlled": True,
            "duckdb": True,
            **(
                {"pytest_file_isolation": True}
                if isolation and isolation.get("passed") is True
                else {}
            ),
        },
        evidence_digests=digests,
    )
    return {
        "task_id": alias,
        "status": result.task.status,
        "changed": result.changed,
        "receipt_cid": result.receipt_cid,
    }


def _plan_r2_remote_owner_admitted(
    expected_identity: dict[str, str],
) -> bool:
    receipt_dir, receipt_files = _receipt_contract()
    receipt_path = receipt_dir / receipt_files.get(
        "EAAEF-190", "plan_r2_remote_owner.json"
    )
    verdict = _host_receipt_completion_verdict(
        task_id="EAAEF-190",
        receipt_dir=receipt_dir,
        receipt_path=receipt_path,
        expected_identity=expected_identity,
    )
    return verdict["completion_allowed"] is True


def _complete(
    source: DatabaseTaskSource,
    alias: str,
    expected_identity: dict[str, str],
) -> dict:
    _receipt_dir, receipt_files = _receipt_contract()
    if alias in receipt_files:
        return _complete_s_task(source, alias, expected_identity)
    if alias == "EAAEF-009" and not _plan_r2_remote_owner_admitted(expected_identity):
        return {
            "task_id": alias,
            "status": "waiting_signed_plan_r2_remote_owner",
            "plan_r2_admitted": False,
            "held_board_materialized": False,
            "reason": "independently signed Plan-R2 remote-owner capability is absent",
        }
    return _complete_a_task(source, alias)


def _reopen_invalid_host_admission_tasks(
    source: DatabaseTaskSource,
    expected_identity: dict[str, str],
    *,
    task_ids: frozenset[str] | None = None,
) -> list[dict]:
    """Reopen completed S tasks whose receipts cannot satisfy dependencies."""

    receipt_dir, receipt_files = _receipt_contract()
    reopened: list[dict] = []
    selected = set(receipt_files) if task_ids is None else set(task_ids)
    if not selected.issubset(receipt_files):
        raise ValueError("host-admission reopen scope contains an unknown task")
    for alias in sorted(selected):
        task = source.get_task(alias)
        if task is None or task.status != "completed":
            continue
        receipt_path = receipt_dir / receipt_files[alias]
        verdict = _host_receipt_completion_verdict(
            task_id=alias,
            receipt_dir=receipt_dir,
            receipt_path=receipt_path,
            expected_identity=expected_identity,
        )
        if verdict["completion_allowed"] is True:
            continue
        result = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "todo",
            {
                "validation": "reopened_invalid_host_admission_receipt",
                "host_controlled": True,
                "receipt_quarantined": True,
                "previous_decision": verdict["decision"],
                "receipt_valid": verdict["valid"],
                "receipt_blockers": verdict["blockers"],
            },
        )
        reopened.append(
            {
                "task_id": alias,
                "status": result.task.status,
                "changed": result.changed,
                "reason": "receipt_invalid_for_completion",
                "decision": verdict["decision"],
                "receipt_valid": verdict["valid"],
                "receipt_quarantined": True,
                "blockers": verdict["blockers"],
            }
        )
    return reopened


def run_once(
    *,
    scope: str = "immutable_observation",
    early_observation_cid: str | None = None,
) -> dict:
    if scope not in {
        "early_frontier",
        "immutable_observation",
        "immutable_full_observation",
        "full_bootstrap",
    }:
        raise ValueError("EAAEF host-admission scope is invalid")
    selected_early_cid = str(early_observation_cid or "")
    if scope == "immutable_full_observation":
        if not _full_sha256_cid(selected_early_cid):
            raise ValueError(
                "immutable full observation requires an exact early observation CID"
            )
        expected_identity = _current_host_admission_identity()
        collection = _collect_immutable_host_admission_observation(
            selected_early_cid
        )
        collection = _validated_host_admission_observation_collection(
            collection,
            expected_identity=expected_identity,
            expected_early_observation_cid=selected_early_cid,
        )
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "eaaef-host-admission-supervisor@1"
            ),
            "execution_scope": scope,
            "process_started": False,
            "configured_board_launch": False,
            "live_multi_supervisor": False,
            "provider_invoked": False,
            "expected_receipt_identity": expected_identity,
            "collection": collection["decisions"],
            "immutable_full_observation": {
                "logical_path": collection["logical_path"],
                "observation_cid": collection["observation_cid"],
                "early_frontier_observation_cid": collection[
                    "early_frontier_observation_cid"
                ],
                "early_frontier_observation_logical_path": collection[
                    "early_frontier_observation_logical_path"
                ],
                "child_receipt_cids": collection["child_receipt_cids"],
                "typed_missing_task_ids": collection["typed_missing_task_ids"],
                "no_go_task_ids": collection["no_go_task_ids"],
                "direct_completion_eligible_task_ids": [],
                "casf_owner_transaction_eligible_task_ids": collection[
                    "casf_owner_transaction_eligible_task_ids"
                ],
            },
            "control_database_path_resolved": False,
            "control_database_owner_lease_acquired": False,
            "control_database_opened": False,
            "in_memory_duckdb_capability_probe_may_open": False,
            "owner_contention_strategy": (
                "authority_registry_create_once_without_control_plane_lease"
            ),
            "staging_receipts_written": False,
            "database_state_observed": False,
            "direct_database_binding_allowed": False,
            "casf_owner_binding_required": True,
            "database_binding_blocker": collection["database_binding_blocker"],
            "completed": [],
            "ready_before": [],
            "ready_after": [],
            "blocked_held": [],
            "task_count": None,
            "status_counts": {},
            "updated_at": int(time.time()),
        }
    if selected_early_cid:
        raise ValueError(
            "early observation CID is valid only for immutable full observation"
        )
    if scope == "immutable_observation":
        expected_identity = _current_host_admission_identity()
        collection = _validated_early_frontier_collection(
            _collect_immutable_observation(),
            expected_identity=expected_identity,
        )
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "eaaef-host-admission-supervisor@1"
            ),
            "execution_scope": scope,
            "process_started": False,
            "configured_board_launch": False,
            "live_multi_supervisor": False,
            "provider_invoked": False,
            "expected_receipt_identity": expected_identity,
            "collection": collection["decisions"],
            "immutable_observation": {
                "logical_path": collection["logical_path"],
                "observation_cid": collection["observation_cid"],
                "child_receipt_cids": collection["child_receipt_cids"],
                "typed_missing_task_ids": collection["typed_missing_task_ids"],
                "direct_completion_eligible_task_ids": [],
                "casf_owner_transaction_eligible_task_ids": collection[
                    "casf_owner_transaction_eligible_task_ids"
                ],
            },
            "control_database_path_resolved": False,
            "control_database_opened": False,
            "control_database_owner_lease_acquired": False,
            "in_memory_duckdb_capability_probe_may_open": True,
            "owner_contention_strategy": (
                "authority_registry_create_once_without_control_plane_lease"
            ),
            "staging_receipts_written": False,
            "direct_database_binding_allowed": False,
            "casf_owner_binding_required": True,
            "database_binding_blocker": collection["database_binding_blocker"],
            "completed": [],
            "ready_before": [],
            "ready_after": [],
            "blocked_held": [],
            "database_state_observed": False,
            "task_count": None,
            "status_counts": {},
            "updated_at": int(time.time()),
        }
    raise RuntimeError(
        "legacy mutable EAAEF host-admission scope is disabled; publish an "
        "immutable observation and use the signed CASF owner command fabric"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scope = str(getattr(args, "scope", "immutable_observation"))
    if scope == "immutable_full_observation":
        payload = run_once(
            scope=scope,
            early_observation_cid=str(args.early_observation_cid),
        )
    else:
        payload = run_once(scope=scope)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
