#!/usr/bin/python3
"""One-shot VRIF-032 current-tree requalification rearm.

Run only after VRIF-030 has completed, its supervisor has exited, the worktree
is clean, and the Quack state owner has been restarted past the generation-69
owner used by the VRIF-030 post-merge completion recovery.

This script performs exactly one completed -> retrying control CAS.  It never
retries that CAS.  The fresh typed CAS result is then used to remove lane 1's
stale promoted logical completion while the lane coordinator lock is held.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path(
    "/home/barberb/lift_coding/.worktrees/"
    "verified-residual-intelligence-foundry-v1"
)
EXPECTED_SCRIPT_PATH = Path(
    "scripts/ops/agent_supervisor/vrif032_one_shot_rearm.py"
)
CONFIG_PATH = Path("config/agent_supervisor_residual_intelligence_scheduler.json")
VRIF030_CID = "baguqeerasq4tuzk6ruqiwho5altj72sx43wxcgiumngxdtyr4rvbljrsy4zq"
VRIF032_CID = "baguqeeramcmm6fru2dasf34ddyuciz7udrrsxo2zd5idobmgxv7ygqfgnwyq"
VRIF032_ALIAS = "VRIF-032"
VRIF030_COMPLETION_REVISION = 48
VRIF030_ATTEMPT_ID = "attempt:d0facd1b52d640be83ce8d8bb77bab1b"
VRIF030_EVIDENCE_DIGEST = (
    "sha256:15512105e88caffba0ab2fb1b4ca918c9a131ba5c4f7b2d7b6638f97cc934a91"
)
VRIF030_BASELINE_COMMIT = "3cf925ca62b583427c2e16843608b688901f6e6e"
VRIF030_IMPLEMENTATION_COMMIT = "0d4fa2bdcd66bac2e5193e8f6e96679433ac322e"
MIN_OWNER_GENERATION = 70
OWNER_RESTART_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-residual-intelligence-foundry-owner-restart-receipt@1"
)
REARM_OPERATION = "database_declared_outputs_on_head_rearm"
REARM_REASON = "vrif_032_owner_exact_release_contract_recovery"
ACTIVE_COUNT_FIELDS = (
    "active_task_claims",
    "active_resource_claims",
    "active_task_attempts",
    "active_fenced_leases",
    "active_maintenance_leases",
)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=not binary,
    )
    require(completed.returncode == 0, f"git {' '.join(args)} failed")
    return completed.stdout


def clean_git_state(root: Path) -> tuple[str, str]:
    status = str(git(root, "status", "--porcelain=v1", "--untracked-files=all"))
    require(not status.strip(), "worktree is not clean")
    head = str(git(root, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    tree = str(git(root, "rev-parse", "--verify", "HEAD^{tree}")).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "HEAD is invalid")
    require(re.fullmatch(r"[0-9a-f]{40}", tree) is not None, "tree is invalid")
    return head, tree


def safe_regular_bytes(
    path: Path,
    *,
    maximum: int,
    exact_mode: int | None = None,
) -> bytes:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"{path.name} is not regular")
    require(metadata.st_uid == os.getuid(), f"{path.name} has a foreign owner")
    require(metadata.st_nlink == 1, f"{path.name} has multiple links")
    if exact_mode is not None:
        require(
            stat.S_IMODE(metadata.st_mode) == exact_mode,
            f"{path.name} mode is not {exact_mode:o}",
        )
    require(0 < metadata.st_size <= maximum, f"{path.name} size is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    require(bool(nofollow), "platform lacks O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    try:
        before = os.fstat(descriptor)
        require(
            before.st_dev == metadata.st_dev
            and before.st_ino == metadata.st_ino
            and before.st_uid == metadata.st_uid
            and before.st_size == metadata.st_size,
            f"{path.name} changed before read",
        )
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        require(
            len(encoded) == metadata.st_size
            and after.st_dev == before.st_dev
            and after.st_ino == before.st_ino
            and after.st_size == before.st_size,
            f"{path.name} changed during read",
        )
        return encoded
    finally:
        os.close(descriptor)


def safe_private_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(
            safe_regular_bytes(path, maximum=65_536, exact_mode=0o600).decode(
                "utf-8"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path.name} is not valid JSON") from exc
    require(isinstance(loaded, dict), f"{path.name} is not a JSON object")
    return loaded


def canonical_identity(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def owner_restart_receipt(
    runtime_root: Path,
    *,
    identity: Mapping[str, Any],
    head: str,
    tree: str,
) -> dict[str, Any]:
    generation = int(identity.get("generation") or 0)
    receipt_dir = runtime_root / "evidence/bootstrap/owner-restarts"
    candidates = sorted(receipt_dir.glob(f"{generation:020d}-*.json"))
    require(
        len(candidates) == 1,
        "Quack owner generation has no unique restart receipt",
    )
    receipt_path = candidates[0]
    try:
        loaded = json.loads(
            safe_regular_bytes(
                receipt_path,
                maximum=65_536,
                exact_mode=0o600,
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Quack owner restart receipt is invalid JSON") from exc
    require(isinstance(loaded, dict), "Quack owner restart receipt is not an object")
    receipt = dict(loaded)
    receipt_id = str(receipt.get("receipt_id") or "")
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_id", None)
    require(
        receipt.get("schema") == OWNER_RESTART_RECEIPT_SCHEMA
        and re.fullmatch(r"sha256:[0-9a-f]{64}", receipt_id) is not None
        and canonical_identity(receipt_body) == receipt_id
        and receipt_path.name
        == f"{generation:020d}-{receipt_id.removeprefix('sha256:')}.json"
        and receipt.get("mode") == "verified_descendant"
        and receipt.get("current_source_head") == head
        and receipt.get("current_source_tree") == tree
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(receipt.get("admission_id") or "")
        )
        is not None
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("bootstrap_receipt_id") or ""),
        )
        is not None
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("authority_config_identity") or ""),
        )
        is not None,
        "Quack owner restart receipt does not bind the current HEAD/tree",
    )
    state_owner = receipt.get("state_owner")
    require(isinstance(state_owner, Mapping), "restart receipt owner binding is absent")
    owner_fields = (
        "server_id",
        "store_id",
        "database_uuid",
        "schema_revision",
        "schema_fingerprint",
        "generation",
        "fence_epoch",
        "process_birth_id",
    )
    require(
        set(state_owner) == set(owner_fields)
        and all(state_owner.get(field) == identity.get(field) for field in owner_fields),
        "restart receipt differs from the live Quack owner identity",
    )
    return receipt


def current_runtime_pid_markers(runtime_root: Path) -> list[str]:
    state_dir = runtime_root / "state"
    candidates = [
        state_dir / "configured-board-master.pid",
        state_dir / "configured-board-wave.pid",
    ]
    for lane_index in range(4):
        lane_dir = state_dir / f"lane-{lane_index}"
        prefix = f"vrif_lane_{lane_index}"
        candidates.extend(
            [
                lane_dir / f"{prefix}_supervisor.pid",
                lane_dir / f"{prefix}_managed_daemon.pid",
                lane_dir / f"{prefix}_daemon.pid",
            ]
        )
    present: list[str] = []
    for path in candidates:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        present.append(str(path.relative_to(runtime_root)))
    return present


def owner_identity(
    status_path: Path,
    *,
    endpoint: str,
    store_id: str,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.daemon_registry import (
        process_birth_id,
    )

    status_payload = safe_private_json(status_path)
    require(
        status_payload.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/quack-state-server@1"
        and status_payload.get("interface") == "QuackStateServer@1"
        and status_payload.get("lifecycle") == "ready",
        "Quack owner status is not live-ready",
    )
    identity = status_payload.get("identity")
    require(isinstance(identity, Mapping), "Quack owner identity is absent")
    value = dict(identity)
    generation = value.get("generation")
    fence_epoch = value.get("fence_epoch")
    require(
        type(generation) is int
        and generation >= MIN_OWNER_GENERATION
        and fence_epoch == generation,
        "Quack owner has not been freshly restarted past generation 69",
    )
    require(
        value.get("status") == "ready"
        and value.get("listen_uri") == endpoint
        and value.get("store_id") == store_id
        and status_payload.get("store_id") == store_id,
        "Quack owner endpoint/store binding differs",
    )
    birth_raw = value.get("process_birth")
    require(isinstance(birth_raw, Mapping), "Quack owner birth is absent")
    birth = ProcessBirthIdentity.from_dict(birth_raw)
    require(owner_liveness(birth) is OwnerLiveness.ALIVE, "Quack owner is not alive")
    require(
        process_birth_id(birth) == value.get("process_birth_id"),
        "Quack owner process-birth ID differs",
    )
    require(
        int(birth.pid) == int(birth_raw.get("pid") or 0),
        "Quack owner PID binding differs",
    )
    return value


def safe_token(path: Path) -> str:
    try:
        raw = safe_regular_bytes(path, maximum=4_096, exact_mode=0o600)
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Quack token vault encoding is invalid") from exc
    token = decoded.strip()
    require(
        decoded in {token, token + "\n"}
        and re.fullmatch(r"[A-Za-z0-9_-]{8,}", token) is not None,
        "Quack token vault material is malformed",
    )
    return token


def validate_private_dir(path: Path) -> None:
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode), f"{path.name} is not a directory")
    require(metadata.st_uid == os.getuid(), f"{path.name} has a foreign owner")
    require(
        stat.S_IMODE(metadata.st_mode) == 0o700,
        f"{path.name} mode is not 700",
    )


def scoped_vrif_supervisor_pids(root: Path) -> list[int]:
    markers = (
        "multi_supervisor_runner",
        "implementation_supervisor_entry.py",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
    )
    found: list[int] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if str(root) in command and any(marker in command for marker in markers):
            found.append(int(item.name))
    return sorted(found)


def mutation_names(path: Path, suffix: str) -> set[str]:
    result: set[str] = set()
    for item in path.glob(f"*{suffix}"):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
        ):
            result.add(item.name)
    return result


def require_zero_active(projection: Mapping[str, Any]) -> dict[str, int]:
    counts_raw = projection.get("counts")
    require(isinstance(counts_raw, Mapping), "coordination counts are absent")
    counts = {str(key): int(value) for key, value in counts_raw.items()}
    require(
        all(counts.get(field) == 0 for field in ACTIVE_COUNT_FIELDS),
        "lane 1 coordination is not quiescent",
    )
    return counts


def lane1_completion_binding(
    projection: Mapping[str, Any],
    *,
    task_revision: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_zero_active(projection)
    task_rows = [
        dict(item)
        for item in projection.get("tasks", [])
        if isinstance(item, Mapping) and item.get("task_cid") == VRIF032_CID
    ]
    require(len(task_rows) == 1, "lane 1 has no exact VRIF-032 registry row")
    require(
        task_rows[0].get("task_id") == VRIF032_ALIAS
        and task_rows[0].get("ready") is False,
        "lane 1 VRIF-032 registry state is not the expected completed state",
    )
    completion_rows = [
        dict(item)
        for item in projection.get("logical_completions", [])
        if isinstance(item, Mapping) and item.get("task_cid") == VRIF032_CID
    ]
    require(len(completion_rows) == 1, "lane 1 has no unique VRIF-032 completion")
    completion = completion_rows[0]
    require(completion.get("status") == "succeeded", "VRIF-032 completion failed")
    body = completion.get("body")
    require(isinstance(body, Mapping), "VRIF-032 completion body is absent")
    control = body.get("control_completion")
    require(isinstance(control, Mapping), "VRIF-032 control completion is absent")
    control_map = dict(control)
    require(
        set(control_map)
        == {"task_cid", "status", "revision", "receipt_cid", "receipt_digest"}
        and control_map.get("task_cid") == VRIF032_CID
        and control_map.get("status") == "completed"
        and control_map.get("revision") == task_revision
        and isinstance(control_map.get("receipt_cid"), str)
        and str(control_map.get("receipt_cid")).startswith("baguqeera")
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(control_map.get("receipt_digest") or "")
        )
        is not None,
        "VRIF-032 control completion binding is stale or malformed",
    )
    require(
        body.get("attempt_number") == 3
        and control_map.get("receipt_cid")
        == "baguqeerahjxjelpzsuhgzbbq32m4nl67w52g5qek6h7peubgsdmb6iowygwa",
        "VRIF-032 is not the admitted attempt-3 historical completion",
    )
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    require(
        "d694b164cf196c2df48b45b494b3df4fdd3f3e87" in encoded
        and "554be0a183b6c6e22aef64e1b3cdafc5620c3275" in encoded,
        "VRIF-032 historical baseline/implementation binding differs",
    )
    return completion, control_map


def task_receipt(task: Any) -> dict[str, Any]:
    body = getattr(task, "body", None)
    require(isinstance(body, Mapping), "VRIF-032 task body is absent")
    receipt = body.get("completion_receipt")
    require(isinstance(receipt, Mapping) and receipt, "VRIF-032 receipt is absent")
    return dict(receipt)


def receipt_digest(receipt: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
        canonical_json_bytes,
    )

    return "sha256:" + hashlib.sha256(canonical_json_bytes(dict(receipt))).hexdigest()


def vrif030_completion_evidence(task: Any) -> dict[str, str | int]:
    require(
        getattr(task, "task_cid", "") == VRIF030_CID
        and getattr(task, "task_alias", "") == "VRIF-030"
        and getattr(task, "status", "") == "completed"
        and int(getattr(task, "revision", 0) or 0) == VRIF030_COMPLETION_REVISION,
        "VRIF-030 is not the exact corrected completed revision",
    )
    receipt = task_receipt(task)
    validation = receipt.get("validation")
    preparation = receipt.get("coordination_preparation")
    require(
        receipt.get("operation") == "database_complete"
        and receipt.get("attempt_id") == VRIF030_ATTEMPT_ID
        and receipt.get("evidence_digest") == VRIF030_EVIDENCE_DIGEST
        and isinstance(validation, Mapping)
        and isinstance(preparation, Mapping),
        "VRIF-030 corrected completion receipt is absent",
    )
    binding = validation.get("portal_completion_binding")
    require(
        validation.get("validator") == "DatabasePortalExecutionBridge@1"
        and validation.get("argv") == ["portal-supervisor-gates"]
        and validation.get("outcome") == "passed"
        and validation.get("task_cid") == VRIF030_CID
        and validation.get("attempt_id") == VRIF030_ATTEMPT_ID
        and validation.get("evidence_digest") == VRIF030_EVIDENCE_DIGEST
        and isinstance(binding, Mapping),
        "VRIF-030 corrected validation binding differs",
    )
    require(
        binding.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/database-portal-completion-binding@1"
        and binding.get("task_cid") == VRIF030_CID
        and binding.get("attempt_id") == VRIF030_ATTEMPT_ID
        and binding.get("evidence_digest") == VRIF030_EVIDENCE_DIGEST
        and binding.get("baseline_commit") == VRIF030_BASELINE_COMMIT
        and binding.get("implementation_commit") == VRIF030_IMPLEMENTATION_COMMIT,
        "VRIF-030 corrected Portal lineage differs",
    )
    preparation_body = preparation.get("body")
    require(
        preparation.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/task-completion-preparation@1"
        and preparation.get("status") == "prepared"
        and preparation.get("task_cid") == VRIF030_CID
        and preparation.get("attempt_id") == VRIF030_ATTEMPT_ID
        and preparation.get("attempt_number") == 7
        and preparation.get("control_expected_status") == "in_progress"
        and preparation.get("control_expected_revision")
        == VRIF030_COMPLETION_REVISION - 1
        and preparation.get("evidence_digest") == VRIF030_EVIDENCE_DIGEST
        and preparation.get("claim_id") == receipt.get("claim_id")
        and preparation.get("lease_id") == receipt.get("lease_id")
        and preparation.get("fence_epoch") == receipt.get("fence_epoch") == 7
        and preparation.get("fencing_token") == receipt.get("fencing_token") == 7
        and isinstance(preparation_body, Mapping)
        and preparation_body.get("validation") == validation,
        "VRIF-030 corrected completion preparation differs",
    )
    return {
        "revision": VRIF030_COMPLETION_REVISION,
        "attempt_id": VRIF030_ATTEMPT_ID,
        "receipt_digest": receipt_digest(receipt),
        "evidence_digest": VRIF030_EVIDENCE_DIGEST,
        "baseline_commit": VRIF030_BASELINE_COMMIT,
        "implementation_commit": VRIF030_IMPLEMENTATION_COMMIT,
    }


def main() -> int:
    discovered = Path(
        str(git(Path.cwd(), "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    root = discovered
    require(root == EXPECTED_ROOT.resolve(), "refusing to operate on another repository")
    os.chdir(root)
    head, tree = clean_git_state(root)
    script_path = Path(__file__).resolve()
    require(
        script_path == (root / EXPECTED_SCRIPT_PATH).resolve(),
        "rearm helper is not running from its committed canonical path",
    )
    committed_script = git(
        root,
        "show",
        f"{head}:{EXPECTED_SCRIPT_PATH.as_posix()}",
        binary=True,
    )
    require(
        bytes(committed_script)
        == safe_regular_bytes(script_path, maximum=131_072),
        "running rearm helper differs from its current-HEAD blob",
    )

    root_text = str(root)
    sys.path[:] = [root_text, *(entry for entry in sys.path if entry != root_text)]

    config_blob = git(root, "show", f"{head}:{CONFIG_PATH.as_posix()}", binary=True)
    try:
        config = json.loads(bytes(config_blob).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("committed scheduler config is invalid") from exc
    require(isinstance(config, Mapping), "committed scheduler config is not an object")
    program = config.get("database_program")
    require(isinstance(program, Mapping), "database_program is absent")
    endpoint = str(program.get("quack_endpoint") or "")
    store_id = str(program.get("store_id") or "")
    store_generation = str(program.get("store_generation") or "")
    secret_handle = str(program.get("endpoint_secret_handle") or "")
    require(
        program.get("authority_mode") == "quack"
        and endpoint == "quack:127.0.0.1:41327"
        and store_id
        == "data/agent_supervisor/residual_intelligence_foundry/control.duckdb"
        and store_generation == "vrif-v1"
        and secret_handle == "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "committed Quack program binding differs",
    )

    runtime_root = root / "data/agent_supervisor/residual_intelligence_foundry"
    owner_dir = runtime_root / "quack-owner"
    status_path = owner_dir / "quack-state-server.status.json"
    mutation_dir = owner_dir / "mutations"
    token_name = secret_handle.replace(":", "_").replace("/", "_")
    token_path = owner_dir / f"{token_name}.quack-token"
    lane_path = runtime_root / "state/lane-1/quack-lane-control.coordination.duckdb"

    validate_private_dir(mutation_dir)
    lane_metadata = lane_path.lstat()
    require(
        stat.S_ISREG(lane_metadata.st_mode)
        and lane_metadata.st_uid == os.getuid(),
        "lane 1 coordination authority is not a same-owner regular file",
    )
    first_owner = owner_identity(status_path, endpoint=endpoint, store_id=store_id)
    first_restart = owner_restart_receipt(
        runtime_root,
        identity=first_owner,
        head=head,
        tree=tree,
    )
    require(
        not current_runtime_pid_markers(runtime_root),
        "supported supervisor cleanup has not reconciled current PID markers",
    )
    token = safe_token(token_path)
    require(
        not scoped_vrif_supervisor_pids(root),
        "a VRIF supervisor/implementation process is still running",
    )

    os.environ["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = token
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
    )

    require(
        harden_state_authority_process() is True,
        "credential-bearing rearm process was not made non-dumpable",
    )
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] = str(mutation_dir)
    os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_ID"] = store_id
    os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = store_generation

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        open_database_coordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source: DatabaseTaskSource | None = None
    held_coordinator: Any | None = None
    cas: Any | None = None
    phase = "preflight"
    before_done = mutation_names(mutation_dir, ".done.json")
    before_requests = mutation_names(mutation_dir, ".request.json")
    require(not before_requests, "pending Quack owner commands exist before rearm")
    prior_revision = 0
    prior_receipt_digest = ""
    lane_before_root = ""
    try:
        source = DatabaseTaskSource(
            endpoint,
            owner_id=f"vrif-terminal-requalification:{os.getpid()}",
            repository_tree_id=tree,
            install_schema=False,
        )
        board_page = source.list_tasks(limit=100)
        require(not board_page.next_cursor, "VRIF task authority exceeds one page")
        aliases = [task.task_alias for task in board_page.tasks]
        require(
            len(board_page.tasks) == 33
            and aliases == [f"VRIF-{index:03d}" for index in range(33)]
            and all(task.status == "completed" for task in board_page.tasks),
            "pre-requalification board is not exact 33/33 completed",
        )
        prerequisite = source.get_task(VRIF030_CID)
        require(
            prerequisite is not None,
            "VRIF-030 prerequisite is absent",
        )
        prerequisite_evidence = vrif030_completion_evidence(prerequisite)
        stable_prerequisite = source.get_task(VRIF030_CID)
        require(
            stable_prerequisite is not None
            and stable_prerequisite.to_dict() == prerequisite.to_dict(),
            "VRIF-030 corrected completion changed during fresh read",
        )
        task = source.get_task(VRIF032_CID)
        require(
            task is not None
            and task.task_alias == VRIF032_ALIAS
            and task.status == "completed"
            and int(task.revision) == 17,
            "VRIF-032 is not the exact completed terminal task",
        )
        stable_task = source.get_task(VRIF032_CID)
        require(
            stable_task is not None and stable_task.to_dict() == task.to_dict(),
            "VRIF-032 authority changed during fresh read",
        )
        prior_revision = int(task.revision)
        previous_receipt = task_receipt(task)
        prior_receipt_digest = receipt_digest(previous_receipt)
        previous_receipt_bytes = json.dumps(
            previous_receipt,
            sort_keys=True,
            separators=(",", ":"),
        )
        require(
            head not in previous_receipt_bytes and tree not in previous_receipt_bytes,
            "VRIF-032 completion already claims the current HEAD/tree",
        )

        # First open/close checkpoints any old WAL before the control CAS.
        checkpoint = open_database_coordinator(lane_path)
        try:
            checkpoint_projection = checkpoint.coordination_registry_projection()
            _completion, prior_control = lane1_completion_binding(
                checkpoint_projection,
                task_revision=prior_revision,
            )
            lane_before_root = str(checkpoint_projection.get("projection_root") or "")
        finally:
            checkpoint.close()

        # Reopen and hold the lane-1 writer lock across the only control CAS.
        held_coordinator = open_database_coordinator(lane_path)
        held_projection = held_coordinator.coordination_registry_projection()
        _completion, held_control = lane1_completion_binding(
            held_projection,
            task_revision=prior_revision,
        )
        require(
            held_control == prior_control
            and held_projection.get("projection_root") == lane_before_root,
            "lane 1 changed after its checkpoint",
        )

        phase = "final-preconditions"
        require(clean_git_state(root) == (head, tree), "HEAD/tree changed before CAS")
        second_owner = owner_identity(status_path, endpoint=endpoint, store_id=store_id)
        require(second_owner == first_owner, "Quack owner changed before CAS")
        require(
            owner_restart_receipt(
                runtime_root,
                identity=second_owner,
                head=head,
                tree=tree,
            )
            == first_restart,
            "Quack owner restart receipt changed before CAS",
        )
        require(
            not current_runtime_pid_markers(runtime_root),
            "a supervisor PID marker appeared before CAS",
        )
        require(
            mutation_names(mutation_dir, ".request.json") == before_requests,
            "a Quack owner command appeared before CAS",
        )
        require(
            not scoped_vrif_supervisor_pids(root),
            "a VRIF supervisor started before CAS",
        )
        final_task = source.get_task(VRIF032_CID)
        require(
            final_task is not None and final_task.to_dict() == task.to_dict(),
            "VRIF-032 changed before CAS",
        )
        final_prerequisite = source.get_task(VRIF030_CID)
        require(
            final_prerequisite is not None
            and final_prerequisite.to_dict() == prerequisite.to_dict()
            and vrif030_completion_evidence(final_prerequisite)
            == prerequisite_evidence,
            "VRIF-030 corrected completion changed before CAS",
        )

        transition_receipt = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "vrif-terminal-current-tree-requalification-rearm@1"
            ),
            "operation": REARM_OPERATION,
            "reason": REARM_REASON,
            "task_cid": VRIF032_CID,
            "task_alias": VRIF032_ALIAS,
            "expected_status": "completed",
            "expected_revision": prior_revision,
            "new_status": "retrying",
            "source_head": head,
            "repository_tree_id": tree,
            "owner_generation": int(first_owner["generation"]),
            "owner_fence_epoch": int(first_owner["fence_epoch"]),
            "owner_process_birth_id": str(first_owner["process_birth_id"]),
            "owner_restart_admission_id": str(first_restart["admission_id"]),
            "owner_restart_receipt_id": str(first_restart["receipt_id"]),
            "previous_completion_receipt_digest": prior_receipt_digest,
            "previous_control_completion_receipt_cid": str(
                prior_control["receipt_cid"]
            ),
            "previous_control_completion_receipt_digest": str(
                prior_control["receipt_digest"]
            ),
            "previous_control_completion_revision": int(prior_control["revision"]),
            "lane": 1,
            "lane_coordination_projection_root": lane_before_root,
            "prerequisite_task_cid": VRIF030_CID,
            "prerequisite_task_alias": "VRIF-030",
            "prerequisite_status": "completed",
            "prerequisite_revision": prerequisite_evidence["revision"],
            "prerequisite_attempt_id": prerequisite_evidence["attempt_id"],
            "prerequisite_completion_receipt_digest": prerequisite_evidence[
                "receipt_digest"
            ],
            "prerequisite_evidence_digest": prerequisite_evidence[
                "evidence_digest"
            ],
            "prerequisite_baseline_commit": prerequisite_evidence[
                "baseline_commit"
            ],
            "prerequisite_implementation_commit": prerequisite_evidence[
                "implementation_commit"
            ],
            "requalification_required": True,
        }

        # The sole control mutation. Never put this call in a retry loop.
        phase = "control-cas-submitted"
        cas = source.compare_and_set_status(
            VRIF032_CID,
            prior_revision,
            "retrying",
            transition_receipt,
            expected_control_receipt=previous_receipt,
        )
        phase = "control-cas-returned"
        require(
            cas.changed is True
            and cas.previous_status == "completed"
            and cas.revision == prior_revision + 1
            and cas.task.task_cid == VRIF032_CID
            and cas.task.task_alias == VRIF032_ALIAS
            and cas.task.status == "retrying"
            and cas.task.revision == prior_revision + 1
            and task_receipt(cas.task) == transition_receipt
            and bool(cas.receipt_cid),
            "owner returned a malformed VRIF-032 CAS result",
        )

        # Consume that exact typed CAS while the sidecar remains exclusively held.
        phase = "lane-rearm"
        lane_rearm = held_coordinator.rearm_task_from_control(
            VRIF032_CID,
            control_task_observation=cas.to_dict(),
        )
        require(
            lane_rearm.get("replayed") is False
            and lane_rearm.get("previous_control_revision") == prior_revision
            and lane_rearm.get("control_revision") == prior_revision + 1
            and lane_rearm.get("control_status") == "retrying"
            and lane_rearm.get("ready") is True,
            "lane 1 did not accept the exact fresh control rearm",
        )
        lane_after = held_coordinator.coordination_registry_projection()
        after_counts = require_zero_active(lane_after)
        require(
            not any(
                isinstance(item, Mapping) and item.get("task_cid") == VRIF032_CID
                for item in lane_after.get("logical_completions", [])
            )
            and any(
                isinstance(item, Mapping)
                and item.get("task_cid") == VRIF032_CID
                and item.get("task_id") == VRIF032_ALIAS
                and item.get("ready") is True
                for item in lane_after.get("tasks", [])
            )
            and after_counts.get("logical_completions")
            == int(checkpoint_projection["counts"]["logical_completions"]) - 1,
            "lane 1 post-rearm projection is invalid",
        )
        lane_after_root = str(lane_after.get("projection_root") or "")
        phase = "post-verification"

        source.close()
        source = None
        verifier = DatabaseTaskSource(
            endpoint,
            owner_id=f"vrif-terminal-requalification-verify:{os.getpid()}",
            repository_tree_id=tree,
            install_schema=False,
        )
        try:
            post_task = verifier.get_task(VRIF032_CID)
            require(
                post_task is not None
                and post_task.status == "retrying"
                and post_task.revision == prior_revision + 1
                and task_receipt(post_task) == transition_receipt,
                "fresh Quack read does not show the exact VRIF-032 rearm",
            )
        finally:
            verifier.close()
        require(clean_git_state(root) == (head, tree), "HEAD/tree changed after CAS")
        require(
            owner_identity(status_path, endpoint=endpoint, store_id=store_id)
            == first_owner,
            "Quack owner changed after CAS",
        )
        require(
            owner_restart_receipt(
                runtime_root,
                identity=first_owner,
                head=head,
                tree=tree,
            )
            == first_restart,
            "Quack owner restart receipt changed after CAS",
        )
        require(
            not current_runtime_pid_markers(runtime_root),
            "a supervisor PID marker appeared during rearm",
        )
        after_done = mutation_names(mutation_dir, ".done.json")
        after_requests = mutation_names(mutation_dir, ".request.json")
        require(
            after_requests == before_requests,
            "a Quack owner command remains pending after rearm",
        )
        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "vrif-terminal-requalification-rearm-result@1"
                    ),
                    "status": "rearmed",
                    "source_head": head,
                    "repository_tree_id": tree,
                    "owner_generation": int(first_owner["generation"]),
                    "owner_process_birth_id": str(first_owner["process_birth_id"]),
                    "owner_restart_receipt_id": str(first_restart["receipt_id"]),
                    "task_cid": VRIF032_CID,
                    "task_alias": VRIF032_ALIAS,
                    "previous_status": "completed",
                    "previous_revision": prior_revision,
                    "revision": prior_revision + 1,
                    "status_after": "retrying",
                    "previous_completion_receipt_digest": prior_receipt_digest,
                    "control_cas_receipt_cid": str(cas.receipt_cid),
                    "control_event_cursor": int(cas.event_cursor),
                    "lane": 1,
                    "lane_projection_before": lane_before_root,
                    "lane_projection_after": lane_after_root,
                    "new_done_files": sorted(after_done - before_done),
                    "new_request_files": sorted(after_requests - before_requests),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        after_done = mutation_names(mutation_dir, ".done.json")
        after_requests = mutation_names(mutation_dir, ".request.json")
        diagnostic: dict[str, Any] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "vrif-terminal-requalification-rearm-diagnostic@1"
            ),
            "status": "failed_closed",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc)[-1_000:],
            "task_cid": VRIF032_CID,
            "expected_previous_revision": prior_revision,
            "previous_completion_receipt_digest": prior_receipt_digest,
            "new_done_files": sorted(after_done - before_done),
            "new_request_files": sorted(after_requests - before_requests),
            "cas_may_have_landed": phase not in {"preflight", "final-preconditions"},
            "instruction": (
                "Do not rerun the control CAS. Fresh-read Quack authority and inspect "
                "only the named new owner-command artifacts."
            ),
        }
        if cas is not None:
            diagnostic["returned_cas"] = {
                "previous_status": str(cas.previous_status),
                "revision": int(cas.revision),
                "event_cursor": int(cas.event_cursor),
                "changed": bool(cas.changed),
                "receipt_cid": str(cas.receipt_cid),
            }
        print(json.dumps(diagnostic, sort_keys=True))
        raise
    finally:
        if source is not None:
            source.close()
        if held_coordinator is not None:
            held_coordinator.close()


if __name__ == "__main__":
    raise SystemExit(main())
