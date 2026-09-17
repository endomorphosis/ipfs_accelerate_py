"""Formal-logic fleet heals. Run before llm_router. Never forge completion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

_RECEIPT_SEARCH_ROOTS = (
    "artifacts",
    "external/ipfs_accelerate/artifacts",
    "external/ipfs_datasets/artifacts",
)
_MAX_RECEIPT_BYTES = 1_000_000
_LOCAL_VALIDATION_TIMEOUT = 90
_ALLOWED_VALIDATORS = {"python3", "pytest", "/usr/bin/python3", "/usr/bin/pytest"}
_CANDIDATE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"
)
_RECORDED_LOCAL_VALIDATION = {
    "passed",
    "failed",
    "validation_unspecified",
    "validation_source_missing",
}
_OVERLAY_SOURCE_PREFIXES = (
    "external/ipfs_accelerate/",
    "external/ipfs_datasets/",
    "external/ipfs_kit/",
)
_REPAIRABLE_SOURCE_SUFFIXES = {".py", ".json"}
_QUACK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_QUACK_TOKEN_ENV = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"


def live_workers(observation: Mapping[str, Any]) -> bool:
    """A live daemon or in-progress task is already doing independent work."""
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    lanes = details.get("lanes") if isinstance(details.get("lanes"), list) else []
    if any(isinstance(lane, dict) and lane.get("daemon") for lane in lanes):
        return True
    counts = details.get("task_counts") if isinstance(details.get("task_counts"), dict) else {}
    try:
        return int(counts.get("in_progress") or 0) > 0
    except (TypeError, ValueError):
        return False


def try_logic_guided_repair(board: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Consult the logic-guided materializer. No write without an admitted plan."""
    try:
        from ipfs_accelerate_py.agent_supervisor.proof.logic_guided_repair_packet import (
            LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
            LogicGuidedRepairPacketMaterializer,
            MaterializationDisposition,
        )
    except Exception as exc:
        return {"status": "needs_llm", "recipe": "logic_unavailable",
                "reason": type(exc).__name__}
    # Fleet incidents are not admitted RPR packets. Importing the materializer
    # binds this loop to the logic submodule; residual coding uses llm_router.
    _ = LogicGuidedRepairPacketMaterializer
    return {
        "status": "needs_llm",
        "recipe": "llm_router",
        "logic_interface": LOGIC_GUIDED_REPAIR_PACKET_MATERIALIZER_INTERFACE,
        "logic_disposition": MaterializationDisposition.ADMISSION_REQUIRED.value,
        "reason": "logic required admitted plan; residual is llm_router Grok then Codex",
    }


def _blocked_task_ids(observation: Mapping[str, Any]) -> list[str]:
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    values = details.get("blocked_task_ids")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()][:8]


def _receipt_path(cwd: Path, task_id: str) -> Path | None:
    for relative in _RECEIPT_SEARCH_ROOTS:
        root = cwd / relative
        if not root.is_dir():
            continue
        for receipts in root.rglob("receipts"):
            if not receipts.is_dir():
                continue
            path = receipts / f"{task_id}.json"
            if path.is_file():
                return path
    return None


def _validation_profiles_path(board: Mapping[str, Any], cwd: Path) -> Path | None:
    """Locate the board's validation-profile catalog. Never searches the live DuckDB."""
    config_path = board.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        config_path = _inventory_board(board).get("config_path")
    if isinstance(config_path, str) and config_path:
        path = Path(config_path)
        for candidate in (
            path.with_name(path.name.replace("_scheduler.json", "_validation_profiles.json")),
            path.with_name(path.name.replace("scheduler.json", "validation_profiles.json")),
        ):
            if candidate.is_file():
                return candidate
    config_dir = cwd / "config"
    if config_dir.is_dir():
        matches = sorted(config_dir.glob("*validation_profiles.json"))
        if matches:
            return matches[0]
    return None


def _validation_profile(
    board: Mapping[str, Any], cwd: Path, task_id: str,
) -> dict[str, Any] | None:
    path = _validation_profiles_path(board, cwd)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    profiles = payload.get(task_id) if isinstance(payload, dict) else None
    if isinstance(profiles, dict) and profiles.get("task_id") in {task_id, None}:
        return profiles
    catalog = payload.get("profiles") if isinstance(payload, dict) else None
    if isinstance(catalog, dict):
        item = catalog.get(task_id)
        if isinstance(item, dict):
            return item
    return None


def _materialize_candidate_receipt(
    cwd: Path, board: Mapping[str, Any], task_id: str,
) -> Path | None:
    """Write a candidate receipt from the validation profile. Never admits completion."""
    profile = _validation_profile(board, cwd, task_id)
    if profile is None:
        return None
    relative = profile.get("receipt")
    if not isinstance(relative, str) or Path(relative).name != f"{task_id}.json":
        return None
    root = cwd.resolve()
    path = (cwd / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if path.is_file():
        return path
    commands = []
    for command in profile.get("commands") or []:
        if not isinstance(command, dict):
            continue
        argv = command.get("argv")
        if (
            isinstance(argv, list) and argv
            and all(isinstance(item, str) and item for item in argv)
            and argv[0] in _ALLOWED_VALIDATORS
        ):
            entry: dict[str, Any] = {"argv": list(argv), "status": "candidate_local"}
            rel = command.get("cwd")
            if isinstance(rel, str) and rel:
                entry["cwd"] = rel
            commands.append(entry)
    if not commands:
        return None
    payload = {
        "schema": _CANDIDATE_RECEIPT_SCHEMA,
        "task_id": task_id,
        "completion_authoritative": False,
        "worker_completion_insufficient": True,
        "candidate_status": "receipt_materialized",
        "plan_revision": profile.get("plan_revision") or "",
        "profile_id": profile.get("profile_id") or "",
        "validation": {"commands": commands},
        "supervisor_acceptance": {
            "completion_authoritative": False,
            "state": "pending_independent_fenced_supervisor",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def _validation_command(payload: Mapping[str, Any]) -> tuple[list[str], str] | None:
    validation = payload.get("validation")
    if isinstance(validation, dict):
        commands = validation.get("commands")
        if isinstance(commands, list):
            for command in commands:
                if not isinstance(command, dict):
                    continue
                argv = command.get("argv")
                if (
                    isinstance(argv, list) and argv
                    and all(isinstance(item, str) and item for item in argv)
                    and argv[0] in _ALLOWED_VALIDATORS
                ):
                    rel = command.get("cwd")
                    return argv, rel if isinstance(rel, str) else ""
    profile = payload.get("validation_profile")
    if isinstance(profile, str) and profile:
        return None
    return None


def _inventory_board(board: Mapping[str, Any]) -> dict[str, Any]:
    probe = board.get("probe") if isinstance(board.get("probe"), dict) else {}
    argv = probe.get("argv") if isinstance(probe.get("argv"), list) else []
    inventory_path = ""
    board_id = str(board.get("id") or "")
    for index, arg in enumerate(argv):
        if arg == "--inventory" and index + 1 < len(argv) and isinstance(argv[index + 1], str):
            inventory_path = argv[index + 1]
        if arg == "--board" and index + 1 < len(argv) and isinstance(argv[index + 1], str):
            board_id = argv[index + 1]
    if not inventory_path:
        return {}
    try:
        payload = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    boards = payload.get("boards") if isinstance(payload, dict) else None
    if not isinstance(boards, list):
        return {}
    for item in boards:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or item.get("board_id") or "")
        if ident.lower() == board_id.lower():
            return item
    return {}


def _objectives_path(cwd: Path, config_path: Any) -> Path | None:
    if not isinstance(config_path, str) or not config_path:
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    relative = payload.get("objectives_path") if isinstance(payload, dict) else None
    if not isinstance(relative, str) or not relative:
        return None
    objectives = cwd / relative
    return objectives if objectives.is_file() else None


def _mutation_binding_from_owner_status(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Exact live owner binding. Prefer the fingerprint extra-gate already admitted."""

    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    fingerprint = str(
        payload.get("storage_schema_fingerprint")
        or identity.get("schema_fingerprint")
        or ""
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
            validate_mutation_binding,
        )
        return validate_mutation_binding(
            {
                "server_id": str(identity.get("server_id") or ""),
                "store_id": str(identity.get("store_id") or payload.get("store_id") or ""),
                "database_uuid": str(identity.get("database_uuid") or ""),
                "schema_revision": int(identity.get("schema_revision") or 0),
                "schema_fingerprint": fingerprint,
                "generation": int(identity.get("generation") or 0),
                "process_birth_id": str(identity.get("process_birth_id") or ""),
                "listen_uri": str(identity.get("listen_uri") or ""),
                "extension_fingerprint": str(
                    identity.get("extension_fingerprint") or "none"
                ),
            }
        )
    except Exception:
        return None


def _owner_transport_env(inventory: Mapping[str, Any]) -> dict[str, str]:
    """Bind store, generation, and mutation inbox from live owner status."""

    env: dict[str, str] = {}
    status_path = inventory.get("owner_status_path")
    runtime = str(inventory.get("runtime_root") or "")
    store_id = ""
    generation = ""
    mutation_dir = ""
    if isinstance(status_path, str) and status_path:
        path = Path(status_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
            store_id = str(identity.get("store_id") or payload.get("store_id") or "")
            observed = identity.get("generation")
            if observed is not None and str(observed).strip():
                generation = str(observed).strip()
            candidate = path.parent / "mutations"
            if candidate.is_dir():
                mutation_dir = str(candidate)
            binding = _mutation_binding_from_owner_status(payload)
            if binding is not None:
                env["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_BINDING"] = json.dumps(
                    binding, separators=(",", ":"), sort_keys=True,
                )
            # typed-state-owner.token authenticates the Unix gateway, not Quack
            # ATTACH. Using it as IPFS_ACCELERATE_AGENT_QUACK_TOKEN fails closed.
            recovered = _live_owner_attach_token(payload)
            if recovered:
                env["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = recovered
    if not store_id:
        database = str(inventory.get("database_path") or "")
        if database and runtime and database.startswith(runtime.rstrip("/") + "/"):
            store_id = database[len(runtime.rstrip("/")) + 1 :]
        elif database:
            store_id = database
    if not mutation_dir and runtime:
        for relative in (
            "q/mutations",
            "quack-owner/mutations",
            "registry/mutations",
        ):
            candidate = Path(runtime) / relative
            if candidate.is_dir():
                mutation_dir = str(candidate)
                break
    if store_id:
        env["IPFS_ACCELERATE_AGENT_STATE_STORE_ID"] = store_id
    if generation:
        env["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = generation
    if mutation_dir:
        env["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] = mutation_dir
    database = inventory.get("database_path")
    if isinstance(database, str) and database:
        try:
            from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                discover_live_quack_endpoint,
            )
            discovery = discover_live_quack_endpoint(database)
        except Exception:
            discovery = None
        token = str(getattr(discovery, "token", "") or "").strip()
        if token:
            env["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = token
    return env


def _process_start_time_ticks(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rfind(")") + 2:].split()[19])


def _process_environ_bytes(pid: int) -> bytes:
    return Path(f"/proc/{pid}/environ").read_bytes()


def _live_owner_attach_token(payload: Mapping[str, Any]) -> str:
    """Recover the live exclusive-owner attach token. Never persist it.

    Provider launch retires the on-disk handoff. The owner process may still
    carry IPFS_ACCELERATE_AGENT_QUACK_TOKEN. Match pid, uid, and start ticks
    from published identity before reading environ.
    """
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    birth = identity.get("process_birth") if isinstance(identity.get("process_birth"), dict) else {}
    try:
        pid = int(birth.get("pid") or payload.get("pid") or 0)
        want_start = int(birth.get("start_time_ticks") or 0)
    except (TypeError, ValueError):
        return ""
    if pid < 1 or want_start < 1:
        return ""
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return ""
        if _process_start_time_ticks(pid) != want_start:
            return ""
        raw = _process_environ_bytes(pid)
    except (OSError, ValueError, IndexError):
        return ""
    prefix = f"{_QUACK_TOKEN_ENV}=".encode("ascii")
    for item in raw.split(b"\0"):
        if not item.startswith(prefix):
            continue
        token = item[len(prefix):].decode("ascii", "replace").strip()
        if _QUACK_TOKEN_RE.fullmatch(token):
            return token
    return ""


@contextmanager
def _temporary_environ(updates: Mapping[str, str]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _open_task_source(endpoint: str, owner_id: str):
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    return DatabaseTaskSource(
        endpoint,
        owner_id=owner_id,
        install_schema=False,
    )


def _open_provisional_goal_source(endpoint: str):
    return _open_task_source(endpoint, "fleet-watchdog-provisional-goal")


def native_unstall_already_recorded(state: Mapping[str, Any]) -> bool:
    result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
    return result.get("recipe") == "unstall_stale_native_work"


def _observation_uninterruptible(observation: Mapping[str, Any]) -> bool:
    reasons = {str(x) for x in observation.get("reason_codes") or []}
    if any("process_uninterruptible" in reason for reason in reasons):
        return True
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    lanes = details.get("lanes") if isinstance(details.get("lanes"), list) else []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        for role in ("daemon", "supervisor"):
            identity = lane.get(role) if isinstance(lane.get(role), dict) else {}
            if identity.get("process_state") == "D":
                return True
    return False


_TYPED_OWNER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/typed-state-owner-command@1"
)
_TYPED_OWNER_MAX_FRAME = 1_048_576
_TYPED_OWNER_SOCKET_LIMIT = 100


def _typed_owner_socket_path(database: str) -> Path:
    candidate = Path(database).expanduser().resolve(strict=False).parent / "quack-owner" / "typed-state-owner.sock"
    if len(os.fsencode(candidate)) <= _TYPED_OWNER_SOCKET_LIMIT:
        return candidate
    digest = hashlib.sha256(os.fsencode(Path(database).resolve(strict=False))).hexdigest()[:32]
    return Path("/tmp") / f"ipfs-accelerate-typed-owner-{os.geteuid()}" / f"{digest}.sock"


def _typed_owner_kernel_birth() -> str:
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text()
    start = int(stat[stat.rfind(")") + 2:].split()[19])
    material = f"{pid}:{start}".encode("ascii")
    return f"birth:kernel:{hashlib.sha256(material).hexdigest()[:32]}"


def _typed_owner_send(channel: socket.socket, payload: Mapping[str, Any]) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        canonical_json_bytes,
    )
    body = canonical_json_bytes(dict(payload))
    if len(body) > _TYPED_OWNER_MAX_FRAME:
        raise OSError("typed owner frame exceeds bound")
    channel.sendall(len(body).to_bytes(4, "big") + body)


def _typed_owner_recv(channel: socket.socket) -> dict[str, Any]:
    header = b""
    while len(header) < 4:
        part = channel.recv(4 - len(header))
        if not part:
            raise OSError("typed owner channel closed")
        header += part
    size = int.from_bytes(header, "big")
    if size < 2 or size > _TYPED_OWNER_MAX_FRAME:
        raise OSError("typed owner frame size is invalid")
    raw = b""
    while len(raw) < size:
        part = channel.recv(size - len(raw))
        if not part:
            raise OSError("typed owner channel closed")
        raw += part
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise OSError("typed owner frame must be an object")
    return payload


def _unstall_via_typed_owner(
    board: Mapping[str, Any],
    observation: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Rearm blocked tasks through the exclusive owner's Unix gateway.

    The Quack ATTACH file may be retired so providers never inherit it. The
    exclusive owner still exposes typed-state-owner.token and a Unix socket.
    No second extra-gate: no ExecStart wrap, no competing owner.
    Never completes tasks.
    """
    blocked = _blocked_task_ids(observation)
    if not blocked:
        return None
    status_path = inventory.get("owner_status_path")
    database = inventory.get("database_path")
    if not isinstance(status_path, str) or not status_path:
        return None
    if not isinstance(database, str) or not database:
        return None
    token_path = Path(status_path).parent / "typed-state-owner.token"
    try:
        token = token_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if len(token) < 16:
        return None
    socket_path = _typed_owner_socket_path(database)
    if not socket_path.exists():
        return None
    store_id = str(inventory.get("task_namespace") or "")
    try:
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    if isinstance(payload, dict):
        identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
        store_id = str(identity.get("store_id") or payload.get("store_id") or store_id)
    if not store_id:
        store_id = database
    empty = {
        "status": "skip",
        "recipe": "unstall_stale_native_work",
        "completion_authority": False,
        "unstalled": [],
        "reason": "typed_owner_grant_absent",
    }
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(8.0)
    try:
        channel.connect(str(socket_path))
        open_id = f"request:{os.getpid()}:open:{uuid.uuid4().hex}"
        # Same live-owner handshake as board authoritative-status.
        _typed_owner_send(channel, {
            "schema": _TYPED_OWNER_SCHEMA,
            "action": "open_status",
            "request_id": open_id,
            "token": token,
            "client_id": "casf-bootstrap-operator:typed-status",
            "process_birth_id": _typed_owner_kernel_birth(),
            "store_id": store_id,
        })
        opened = _typed_owner_recv(channel)
        if opened.get("ok") is not True:
            return empty
        grant = opened.get("grant") if isinstance(opened.get("grant"), dict) else {}
        allowed = grant.get("allowed_command_operations")
        if not (isinstance(allowed, list) and "rearm_blocked_task" in allowed):
            return {
                **empty,
                "reason": "typed_owner_status_session_read_only",
            }
        unstalled = []
        for index, task_id in enumerate(blocked):
            command_id = f"request:{os.getpid()}:rearm:{index}:{uuid.uuid4().hex}"
            _typed_owner_send(channel, {
                "schema": _TYPED_OWNER_SCHEMA,
                "action": "database_task_command",
                "request_id": command_id,
                "command_request_id": f"fleet-unstall-{index:02d}",
                "command": "rearm_blocked_task",
                "payload": {
                    "task_cid_or_alias": task_id,
                    "receipt": {"operation": "false_terminal_blocked_supervisor_bug"},
                },
            })
            response = _typed_owner_recv(channel)
            result = response.get("result") if response.get("ok") is True else None
            if isinstance(result, Mapping) and result.get("changed") is True:
                unstalled.append({
                    "task_alias": task_id,
                    "reason": "false_terminal_blocked_supervisor_bug",
                })
    except Exception:
        return empty
    finally:
        try:
            channel.close()
        except Exception:
            pass
    if not unstalled:
        return empty
    return {
        "status": "applied",
        "recipe": "unstall_stale_native_work",
        "completion_authority": False,
        "unstalled": unstalled,
        "reason": (
            "stale in_progress or false-terminal blocked tasks rearmed; "
            "native extra-gate lanes admit"
        ),
    }


def unstall_stale_native_work(
    board: Mapping[str, Any], observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Rearm unknown-outcome blocks and stale in_progress via the live owner.

    Never completes tasks. PCTDD-006/035 stay blocked unless the owner CAS
    to retrying succeeds. SPAR/SAWM/DOEP follow the same recipe.
    """
    empty = {
        "status": "skip",
        "recipe": "unstall_stale_native_work",
        "completion_authority": False,
        "unstalled": [],
    }
    inventory = _inventory_board(board)
    endpoint = inventory.get("quack_endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("quack:"):
        return empty
    transport = _owner_transport_env(inventory)
    if not transport.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN"):
        typed = _unstall_via_typed_owner(board, observation, inventory)
        if typed is not None and typed.get("status") == "applied":
            return typed
        return typed or {**empty, "reason": "quack_attach_token_absent"}
    if not transport.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID"):
        return {**empty, "reason": "owner_store_binding_absent"}
    inbox = transport.get("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR")
    if inbox:
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
            retire_settled_mutation_inbox,
        )
        for _ in range(8):
            if retire_settled_mutation_inbox(Path(inbox), limit=1024) < 1024:
                break
    try:
        with _temporary_environ(transport):
            with _open_task_source(endpoint, "fleet-watchdog-unstall") as source:
                result = source.unstall_stale_in_progress_tasks(
                    skip_stale_in_progress=_observation_uninterruptible(observation),
                )
    except Exception as exc:
        detail = str(exc)
        if "Authentication failed" in detail or "Quack authentication token" in detail:
            return {**empty, "status": "wait", "reason": "quack_attach_token_absent"}
        return {**empty, "status": "wait", "reason": f"owner_cas_failed:{type(exc).__name__}"}
    changed = [
        item for item in (result.get("unstalled") or [])
        if isinstance(item, dict)
    ]
    if not changed:
        return {**empty, "reason": "no_stale_or_false_terminal_work"}
    return {
        "status": "applied",
        "recipe": "unstall_stale_native_work",
        "completion_authority": False,
        "unstalled": changed,
        "reason": (
            "stale in_progress or false-terminal blocked tasks rearmed; "
            "native extra-gate lanes admit"
        ),
    }


def locally_validated_rearm_already_recorded(state: Mapping[str, Any]) -> bool:
    result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
    return result.get("recipe") == "rearm_locally_validated_blocked_tasks"


def run_board_local_repair(
    board: Mapping[str, Any],
    observation: Mapping[str, Any],
    passed: list[str],
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the board's own repair_argv after local checks. Never completes."""
    empty = {
        "status": "skip",
        "recipe": "rearm_locally_validated_blocked_tasks",
        "completion_authority": False,
        "completion_authoritative": False,
        "unstalled": [],
    }
    if not passed:
        return {**empty, "reason": "no_locally_validated_blocked_tasks"}
    inventory = _inventory_board(board)
    argv = inventory.get("repair_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        return {**empty, "reason": "board_repair_argv_absent"}
    cwd = Path(str(inventory.get("cwd") or board.get("cwd") or ""))
    if not cwd.is_dir():
        return {**empty, "reason": "board_cwd_absent"}
    command = list(argv)
    if any("recover-blocked-lock-timeout" in item for item in command):
        return run_board_claim_verification_recover(
            board, observation, passed, results=results,
        )
    if "--task" not in command and passed:
        # recover-claim-verification style operators accept one task.
        if any("recover-claim-verification" in item or "recover-repaired-dependency" in item for item in command):
            command.extend(["--task", passed[0]])
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_LOCAL_VALIDATION_TIMEOUT,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**empty, "reason": f"board_repair_unavailable:{type(exc).__name__}"}
    if completed.returncode != 0:
        recovered = run_board_claim_verification_recover(
            board, observation, passed, results=results,
        )
        if recovered.get("status") == "applied":
            return recovered
        return {
            **empty,
            "reason": recovered.get("reason") or "board_repair_rejected",
            "returncode": completed.returncode,
        }
    return {
        "status": "applied",
        "recipe": "rearm_locally_validated_blocked_tasks",
        "completion_authority": False,
        "completion_authoritative": False,
        "unstalled": [{"task_alias": task_id, "reason": "board_local_repair_argv"} for task_id in passed],
        "reason": (
            "board-local repair_argv accepted; native lanes admit; "
            "receipts stay incomplete"
        ),
    }


def _board_handoff_argv(inventory: Mapping[str, Any]) -> list[str] | None:
    """Python + board handoff script, with the subcommand stripped."""
    for key in ("status_argv", "launch_argv", "repair_argv"):
        argv = inventory.get(key)
        if not isinstance(argv, list) or len(argv) < 2:
            continue
        if any(not isinstance(item, str) or not item for item in argv):
            continue
        for index, item in enumerate(argv):
            if item.endswith(".py"):
                prefix = list(argv[: index + 1])
                if prefix:
                    return prefix
    return None


def _task_revision_from_status(payload: Mapping[str, Any], task_id: str) -> int | None:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return None
    for row in tasks:
        if not isinstance(row, dict):
            continue
        alias = str(row.get("task_alias") or row.get("task_id") or "")
        if alias != task_id:
            continue
        if str(row.get("status") or "").lower() != "blocked":
            return None
        revision = row.get("revision")
        if type(revision) is int and revision > 0:
            return revision
    return None


def _copied_validation_files(
    cwd: Path, results: list[dict[str, Any]] | None,
) -> list[Path]:
    files: list[Path] = []
    root = cwd.resolve()
    for item in results or []:
        for relative in item.get("repaired") or []:
            if not isinstance(relative, str) or not relative.endswith((".py", ".json")):
                continue
            path = (cwd / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.is_file():
                files.append(path)
    return files


@contextmanager
def _hold_copied_validation_files(paths: list[Path]) -> Iterator[None]:
    """Temporarily remove overlay copies so owner preflight sees a clean tree."""
    saved: list[tuple[Path, bytes]] = []
    for path in paths:
        try:
            saved.append((path, path.read_bytes()))
            path.unlink()
        except OSError:
            continue
    try:
        yield
    finally:
        for path, data in saved:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def run_board_claim_verification_recover(
    board: Mapping[str, Any],
    observation: Mapping[str, Any],
    passed: list[str],
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Use authoritative-status then recover-claim-verification. Never completes."""
    empty = {
        "status": "skip",
        "recipe": "rearm_locally_validated_blocked_tasks",
        "completion_authority": False,
        "completion_authoritative": False,
        "unstalled": [],
    }
    inventory = _inventory_board(board)
    cwd = Path(str(inventory.get("cwd") or board.get("cwd") or ""))
    handoff = _board_handoff_argv(inventory)
    if not cwd.is_dir() or not handoff:
        return {**empty, "reason": "board_handoff_argv_absent"}
    repair = inventory.get("repair_argv") if isinstance(inventory.get("repair_argv"), list) else []
    if not any("handoff.py" in item for item in handoff) and not any(
        isinstance(item, str) and "recover-claim-verification" in item for item in repair
    ):
        return {**empty, "reason": "board_handoff_argv_absent"}
    status_cmd = [*handoff, "authoritative-status"]
    for task_id in passed[:8]:
        status_cmd.extend(["--history-task", task_id])
    try:
        status = subprocess.run(
            status_cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_LOCAL_VALIDATION_TIMEOUT,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {**empty, "reason": "authoritative_status_unavailable"}
    if status.returncode != 0:
        return {**empty, "reason": "authoritative_status_rejected"}
    try:
        payload = json.loads(status.stdout or "")
    except json.JSONDecodeError:
        return {**empty, "reason": "authoritative_status_malformed"}
    if not isinstance(payload, dict) or payload.get("completion_authority") is True:
        return {**empty, "reason": "authoritative_status_malformed"}
    planned: list[tuple[str, int]] = []
    for task_id in passed:
        revision = _task_revision_from_status(payload, task_id)
        if revision is None:
            continue
        planned.append((task_id, revision))
    if not planned:
        return {**empty, "reason": "no_blocked_revisions_for_claim_verification"}
    board_id = str(board.get("id") or inventory.get("id") or "").lower()
    unit = str(inventory.get("existing_service") or "")
    if unit.endswith(".service"):
        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", unit],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            active = None
        if (
            active is not None
            and active.returncode == 0
            and str(active.stdout or "").strip() == "active"
        ):
            return {**empty, "reason": "owner_live_do_not_stop"}
    copies = _copied_validation_files(cwd, results)
    stopped = False
    unstalled: list[dict[str, Any]] = []
    with _hold_copied_validation_files(copies):
        if board_id not in RETAIN_OWNER_BOARDS and unit.endswith(".service"):
            stop = subprocess.run(
                ["systemctl", "--user", "stop", unit],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
                check=False,
            )
            stopped = stop.returncode == 0
        try:
            for task_id, revision in planned:
                recover = subprocess.run(
                    [
                        *handoff,
                        "recover-claim-verification",
                        "--task",
                        task_id,
                        "--expected-revision",
                        str(revision),
                    ],
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=180,
                    check=False,
                    text=True,
                )
                if recover.returncode == 0:
                    unstalled.append({
                        "task_alias": task_id,
                        "reason": "recover_claim_verification",
                        "expected_revision": revision,
                    })
        finally:
            ensure = inventory.get("ensure_argv")
            if stopped and isinstance(ensure, list) and ensure and all(
                isinstance(item, str) and item for item in ensure
            ):
                subprocess.run(
                    list(ensure),
                    cwd=str(cwd),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=180,
                    check=False,
                )
    if not unstalled:
        return {**empty, "reason": "claim_verification_recover_rejected"}
    return {
        "status": "applied",
        "recipe": "rearm_locally_validated_blocked_tasks",
        "completion_authority": False,
        "completion_authoritative": False,
        "unstalled": unstalled,
        "reason": (
            "recover-claim-verification accepted; native lanes admit; "
            "receipts stay incomplete"
        ),
    }


def rearm_locally_validated_blocked_tasks(
    board: Mapping[str, Any],
    observation: Mapping[str, Any],
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """CAS locally-validated blocked tasks to retrying. Never completes."""
    empty = {
        "status": "skip",
        "recipe": "rearm_locally_validated_blocked_tasks",
        "completion_authority": False,
        "completion_authoritative": False,
        "unstalled": [],
    }
    passed = [
        str(item.get("task_id") or "")
        for item in (results or [])
        if isinstance(item, dict) and item.get("status") == "passed" and item.get("task_id")
    ]
    if not passed:
        return {**empty, "reason": "no_locally_validated_blocked_tasks"}
    inventory = _inventory_board(board)
    endpoint = inventory.get("quack_endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("quack:"):
        return {**empty, "reason": "quack_endpoint_absent"}
    transport = _owner_transport_env(inventory)
    if not transport.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN"):
        local = run_board_local_repair(board, observation, passed, results=results)
        if local.get("status") == "applied":
            local["results"] = list(results or [])
            return local
        typed = _unstall_via_typed_owner(board, observation, inventory)
        if typed is not None and typed.get("status") == "applied":
            typed["results"] = list(results or [])
            typed["recipe"] = "rearm_locally_validated_blocked_tasks"
            return typed
        return {
            **empty,
            "reason": local.get("reason") or (
                typed.get("reason") if typed else "board_repair_unavailable"
            ),
        }
    if not transport.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID"):
        return {**empty, "reason": "owner_store_binding_absent"}
    inbox = transport.get("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR")
    if inbox:
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
            retire_settled_mutation_inbox,
        )
        for _ in range(8):
            if retire_settled_mutation_inbox(Path(inbox), limit=1024) < 1024:
                break
    changed: list[dict[str, Any]] = []
    try:
        with _temporary_environ(transport):
            with _open_task_source(endpoint, "fleet-watchdog-local-validation-rearm") as source:
                for task_id in passed:
                    try:
                        cas = source.rearm_blocked_task(
                            task_id,
                            receipt={
                                "operation": "local_validation_pending_native_admission",
                                "completion_authoritative": False,
                            },
                        )
                    except Exception:
                        continue
                    if getattr(cas, "changed", False):
                        changed.append({
                            "task_alias": task_id,
                            "reason": "local_validation_pending_native_admission",
                        })
    except Exception as exc:
        detail = str(exc)
        if "Authentication failed" in detail or "Quack authentication token" in detail:
            return {**empty, "reason": "quack_attach_token_absent"}
        return {**empty, "reason": f"owner_cas_failed:{type(exc).__name__}"}
    if not changed:
        return {**empty, "reason": "owner_did_not_rearm_locally_validated_tasks"}
    return {
        "status": "applied",
        "recipe": "rearm_locally_validated_blocked_tasks",
        "completion_authority": False,
        "completion_authoritative": False,
        "unstalled": changed,
        "results": list(results or []),
        "reason": (
            "locally validated blocked tasks rearmed to retrying; "
            "native lanes admit; receipts stay incomplete"
        ),
    }


def provisional_goal_closeout_already_recorded(state: Mapping[str, Any]) -> bool:
    result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
    return result.get("recipe") == "provisionally_complete_terminal_goals"


def provisionally_complete_disabled_extra_gate_goals(
    board: Mapping[str, Any], observation: Mapping[str, Any],
) -> dict[str, Any]:
    """CAS active goals to provisionally_complete via Quack. Never verifies.

    Extra-gate launched with --no-objective-goal-migration cannot close goals
    after the task frontier. Fleet uses the live owner transport, not DuckDB.
    """
    from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import GoalState
    schema = "ipfs_accelerate_py/agent-supervisor/provisional-goal-completion@1"
    empty = {
        "status": "wait",
        "recipe": "native_goals_still_active",
        "completion_authority": False,
        "changed_goal_ids": [],
    }
    reasons = {str(x) for x in observation.get("reason_codes") or []}
    if "goal_closeout_disabled_on_launch" not in reasons:
        return {**empty, "reason": "extra_gate_closeout_not_disabled"}
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    counts = details.get("task_counts") if isinstance(details.get("task_counts"), dict) else {}
    from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import TERMINAL_TASK_STATUSES
    leftover = [
        key for key, value in counts.items()
        if int(value or 0) > 0 and str(key).lower() not in TERMINAL_TASK_STATUSES
    ] if counts else ["task_counts_unavailable"]
    if leftover:
        return {**empty, "reason": "tasks_not_all_terminal"}
    inventory = _inventory_board(board)
    endpoint = inventory.get("quack_endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("quack:"):
        return {**empty, "reason": "quack_endpoint_absent"}
    cwd = Path(str(inventory.get("cwd") or board.get("cwd") or ""))
    objectives = _objectives_path(cwd, inventory.get("config_path"))
    if objectives is None:
        return {**empty, "reason": "objective_path_missing"}
    try:
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
        goals = parse_goal_heap(objectives.read_text(encoding="utf-8"))
    except Exception as exc:
        return {**empty, "reason": f"objective_parse_failed:{type(exc).__name__}"}
    changed: list[str] = []
    transport = _owner_transport_env(inventory)
    if not transport.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID"):
        return {**empty, "reason": "owner_store_binding_absent"}
    try:
        with _temporary_environ(transport):
            source_cm = _open_provisional_goal_source(endpoint)
            with source_cm as source:
                for goal in goals:
                    rec = source.get_goal(goal.goal_id)
                    if not isinstance(rec, Mapping):
                        continue
                    status = str(rec.get("status") or "").strip().lower()
                    if status not in {"active", "reopened", "analysis_inconclusive"}:
                        continue
                    revision = rec.get("revision")
                    if type(revision) is not int:
                        continue
                    try:
                        source.compare_and_set_goal_status(
                            goal.goal_id,
                            revision,
                            GoalState.PROVISIONALLY_COMPLETE.value,
                            {
                                "schema": schema,
                                "completion_authority": False,
                                "tasks_complete": True,
                                "goal_alias": goal.goal_id,
                                "state": GoalState.PROVISIONALLY_COMPLETE.value,
                            },
                        )
                    except Exception:
                        continue
                    changed.append(goal.goal_id)
    except Exception as exc:
        detail = str(exc)
        if "Authentication failed" in detail or "Quack authentication token" in detail:
            return {**empty, "reason": "quack_attach_token_absent"}
        return {**empty, "reason": f"owner_cas_failed:{type(exc).__name__}"}
    return {
        "status": "applied" if changed else "wait",
        "recipe": "provisionally_complete_terminal_goals",
        "completion_authority": False,
        "changed_goal_ids": changed,
        "reason": (
            "active goals moved to provisionally_complete; verification still required"
            if changed else "no_active_goals"
        ),
    }


def local_validation_already_recorded(
    state: Mapping[str, Any], board: Mapping[str, Any] | None = None,
) -> bool:
    """Do not re-run pytest every watchdog cycle after a recorded local pass."""
    observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
    result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
    if result.get("recipe") not in {
        "local_validation_pending_native_admission",
        "overlay_first_native_admission",
        "rearm_locally_validated_blocked_tasks",
    }:
        return False
    current = _blocked_task_ids(observation)
    if not current:
        return False
    cwd = Path(str((board or {}).get("cwd") or ""))
    items = [item for item in result.get("results") or [] if isinstance(item, dict)]
    prior = {item.get("task_id"): item for item in items}
    for task_id in current:
        item = prior.get(task_id) or {}
        status = item.get("status")
        if status == "failed" and item.get("returncode") in {4, 5}:
            # Pytest usage/collection errors are retried as source-missing.
            return False
        if status == "validation_source_missing":
            missing = item.get("missing")
            if isinstance(missing, str) and overlay_source_path(missing) is not None:
                return False
            continue
        if status == "passed" and cwd.is_dir():
            for relative in item.get("repaired") or []:
                if not isinstance(relative, str) or not relative.endswith(".py"):
                    continue
                if not (cwd / relative).is_file():
                    return False
        if status not in _RECORDED_LOCAL_VALIDATION:
            return False
    return True


def run_local_blocked_candidate_validation(
    board: Mapping[str, Any], observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run declared local checks for blocked candidates. Never admits completion.

    Billing-locked CI is not an excuse to skip deterministic tests. Passing
    local checks do not flip native admission or rewrite 044/063 receipts.
    """
    cwd = Path(str(board.get("cwd") or ""))
    task_ids = _blocked_task_ids(observation)
    if not cwd.is_dir() or not task_ids:
        return {"status": "skip"}
    results = []
    for task_id in task_ids:
        path = _receipt_path(cwd, task_id)
        if path is None:
            path = _materialize_candidate_receipt(cwd, board, task_id)
        if path is None:
            results.append({"task_id": task_id, "status": "receipt_missing"})
            continue
        try:
            raw = path.read_bytes()
            if len(raw) > _MAX_RECEIPT_BYTES:
                results.append({"task_id": task_id, "status": "receipt_too_large"})
                continue
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            results.append({"task_id": task_id, "status": "receipt_unreadable"})
            continue
        if not isinstance(payload, dict):
            results.append({"task_id": task_id, "status": "receipt_unreadable"})
            continue
        command = _validation_command(payload)
        if command is None:
            results.append({"task_id": task_id, "status": "validation_unspecified",
                            "completion_authoritative": payload.get("completion_authoritative") is True})
            continue
        argv, relative = command
        workdir = (cwd / relative).resolve() if relative else cwd.resolve()
        root = cwd.resolve()
        if not workdir.is_dir() or (root not in workdir.parents and workdir != root):
            results.append({"task_id": task_id, "status": "validation_cwd_rejected"})
            continue
        repaired = _repair_missing_validation_artifacts(
            cwd, board, task_id, argv, workdir, observation=observation,
        )
        missing = _pytest_source_missing(argv, workdir)
        if missing is not None:
            results.append({
                "task_id": task_id,
                "status": "validation_source_missing",
                "missing": missing,
                "repaired": repaired,
                "completion_authoritative": False,
            })
            continue
        try:
            completed = subprocess.run(
                argv, cwd=str(workdir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=_LOCAL_VALIDATION_TIMEOUT, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            results.append({"task_id": task_id, "status": "validation_unavailable"})
            continue
        results.append({
            "task_id": task_id,
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "repaired": repaired,
            "completion_authoritative": False,
        })
    if not results:
        return {"status": "skip"}
    passed = [item for item in results if item.get("status") == "passed"]
    failed = [item for item in results if item.get("status") == "failed"]
    source_missing = [
        item for item in results if item.get("status") == "validation_source_missing"
    ]
    ran = passed or failed
    if not ran and source_missing:
        return {
            "status": "applied",
            "recipe": "local_validation_pending_native_admission",
            "completion_authoritative": False,
            "results": results,
            "reason": (
                "candidate receipt built; validation source missing; "
                "supervisor copies overlay sources when present; "
                "do not rewrite receipts as complete"
            ),
        }
    if not ran:
        return {"status": "wait", "recipe": "todos_waiting_on_blocked_dependencies",
                "reason": "remaining todos depend on blocked peers; do not rewrite those receipts",
                "results": results}
    mixed_ok = bool(passed) and not failed
    return {
        "status": "applied" if mixed_ok else "wait",
        "recipe": "local_validation_pending_native_admission",
        "completion_authoritative": False,
        "results": results,
        "reason": (
            "local checks passed; native admission still required"
            if mixed_ok and not source_missing else
            "local checks passed; missing validation source is overlay copy work"
            if mixed_ok else
            "local checks did not pass; do not rewrite blocked receipts"
        ),
    }


def _pytest_source_missing(argv: list[str], workdir: Path) -> str | None:
    """Return a missing pytest target path, if the command names one."""
    skip = {"python3", "pytest", "/usr/bin/python3", "/usr/bin/pytest", "-m"}
    for item in argv:
        if item in skip or item.startswith("-"):
            continue
        if not item.endswith(".py"):
            continue
        target = Path(item)
        path = target if target.is_absolute() else (workdir / item)
        try:
            resolved = path.resolve()
            resolved.relative_to(workdir.resolve())
        except ValueError:
            continue
        if not resolved.is_file():
            return item
    return None


def _overlay_relative_source(missing: str) -> Path | None:
    text = str(missing or "").replace("\\", "/").lstrip("/")
    if not text or ".." in Path(text).parts:
        return None
    for prefix in _OVERLAY_SOURCE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if not text:
        return None
    return Path(text)


def overlay_source_path(missing: str, overlay: str | Path | None = None) -> Path | None:
    """Return the overlay file that can repair a missing board source."""
    relative = _overlay_relative_source(missing)
    if relative is None or relative.suffix not in _REPAIRABLE_SOURCE_SUFFIXES:
        return None
    root = Path(overlay if overlay is not None else supervisor_overlay_root())
    try:
        root = root.resolve()
        path = (root / relative).resolve()
        path.relative_to(root)
    except (OSError, ValueError):
        return None
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > _MAX_RECEIPT_BYTES:
            return None
    except OSError:
        return None
    return path


def copy_overlay_source(
    missing: str, workdir: Path, overlay: str | Path | None = None,
) -> str | None:
    """Copy a missing board source from overlay. Never overwrites or admits."""
    source = overlay_source_path(missing, overlay)
    if source is None:
        return None
    dest = Path(missing)
    dest = dest if dest.is_absolute() else (workdir / missing)
    try:
        dest = dest.resolve()
        dest.relative_to(workdir.resolve())
    except (OSError, ValueError):
        return None
    if dest.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    os.chmod(dest, 0o600)
    return missing


def _owner_is_ready(observation: Mapping[str, Any] | None) -> bool:
    details = (observation or {}).get("details") if isinstance((observation or {}).get("details"), dict) else {}
    return details.get("owner_ready") is True


def _repair_missing_validation_artifacts(
    cwd: Path,
    board: Mapping[str, Any],
    task_id: str,
    argv: list[str],
    workdir: Path,
    observation: Mapping[str, Any] | None = None,
) -> list[str]:
    """Copy missing pytest targets and required outputs from overlay."""
    if not _owner_is_ready(observation):
        return []
    repaired: list[str] = []
    missing = _pytest_source_missing(argv, workdir)
    if missing is not None:
        copied = copy_overlay_source(missing, workdir)
        if copied:
            repaired.append(copied)
    profile = _validation_profile(board, cwd, task_id)
    if profile is None:
        return repaired
    for relative in profile.get("required_outputs") or []:
        if not isinstance(relative, str) or not relative:
            continue
        try:
            dest = (cwd / relative).resolve()
            dest.relative_to(cwd.resolve())
        except (OSError, ValueError):
            continue
        if dest.is_file():
            continue
        copied = copy_overlay_source(relative, cwd)
        if copied:
            repaired.append(copied)
    return repaired


def clear_overlay_copies_blocking_owner_start(
    board: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove overlay-copied sources that make exclusive-owner preflight fail."""
    empty = {
        "status": "skip",
        "recipe": "clear_overlay_copies_for_owner_start",
        "completion_authority": False,
        "removed": [],
    }
    board_id = str(board.get("id") or "").lower()
    if board_id in RETAIN_OWNER_BOARDS:
        return {**empty, "reason": "retain_owner_checkout_untouched"}
    cwd = Path(str(board.get("cwd") or ""))
    nested = cwd / "external" / "ipfs_accelerate"
    if not nested.is_dir():
        return {**empty, "reason": "nested_accelerate_absent"}
    try:
        completed = subprocess.run(
            ["git", "-C", str(nested), "status", "--porcelain", "-uall"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {**empty, "reason": "nested_git_status_unavailable"}
    if completed.returncode != 0:
        return {**empty, "reason": "nested_git_status_unavailable"}
    removed: list[str] = []
    for line in (completed.stdout or "").splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip()
        if not relative.endswith((".py", ".json")):
            continue
        if overlay_source_path(relative) is None and overlay_source_path(
            f"external/ipfs_accelerate/{relative}",
        ) is None:
            continue
        path = (nested / relative).resolve()
        try:
            path.relative_to(nested.resolve())
        except ValueError:
            continue
        if path.is_file():
            path.unlink()
            removed.append(relative)
    if not removed:
        return {**empty, "reason": "no_overlay_copies_to_clear"}
    return {
        "status": "applied",
        "recipe": "clear_overlay_copies_for_owner_start",
        "completion_authority": False,
        "removed": removed,
        "reason": "overlay copies removed so exclusive-owner preflight can start",
    }


def restore_dirty_control_plane(board: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Drop uncommitted control-plane dirt back to HEAD. Never forges completion."""
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    integrity = details.get("source_integrity") if isinstance(details.get("source_integrity"), dict) else {}
    if integrity.get("reason") != "configured_control_plane_dirty":
        return {"status": "skip"}
    checked = integrity.get("checked") if isinstance(integrity.get("checked"), list) else []
    restored = []
    for entry in checked:
        if not isinstance(entry, dict):
            continue
        root = Path(str(entry.get("repository") or ""))
        paths = [str(path) for path in entry.get("paths") or [] if isinstance(path, str) and path]
        allowed = [path for path in paths if path in {
            "ipfs_accelerate_py/agent_supervisor", "scripts/ops/agent_supervisor",
        } or path.startswith("ipfs_accelerate_py/agent_supervisor/")
          or path.startswith("scripts/ops/agent_supervisor/")]
        if not root.is_dir() or not allowed:
            continue
        completed = subprocess.run(
            ["git", "-C", str(root), "checkout", "--", *allowed],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False,
        )
        if completed.returncode == 0:
            restored.append(str(root))
    if restored:
        return {"status": "applied", "recipe": "restore_dirty_control_plane", "restored": restored}
    return {"status": "skip"}


RETAIN_OWNER_BOARDS = frozenset({"spar", "aseh"})
HEAL_OVERLAY_LAUNCHER = "sealed_board_supervisor_launch.py"
NATIVE_ADMISSION_OVERLAY_BOARDS = frozenset({"doep"})


def supervisor_overlay_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def parse_systemd_exec_start(value: str) -> list[str]:
    marker = "argv[]="
    start = str(value or "").find(marker)
    if start < 0:
        return []
    rest = str(value)[start + len(marker):]
    end = rest.find(" ;")
    if end >= 0:
        rest = rest[:end]
    return [item for item in rest.split() if item]


def wrap_python_execstart(
    argv: list[str], *, overlay: str, source_root: str,
) -> list[str]:
    launcher = str(
        Path(overlay)
        / "ipfs_accelerate_py/agent_supervisor/rescue"
        / HEAL_OVERLAY_LAUNCHER
    )
    if any(HEAL_OVERLAY_LAUNCHER in item for item in argv):
        return list(argv)
    python = argv[0] if argv else "/usr/bin/python3"
    rest = argv[1:]
    if rest[:1] == ["-P"]:
        rest = rest[1:]
    return [
        python, "-P", launcher,
        "--overlay", overlay, "--source-root", source_root, "--",
        *rest,
    ]


def overlay_wrapped_execstart(
    argv: list[str], *, overlay: str, source_root: str,
) -> list[str] | None:
    if not argv:
        return None
    if any(HEAL_OVERLAY_LAUNCHER in item for item in argv):
        return list(argv)
    first = Path(argv[0]).name
    if first in {"python", "python3"} or argv[0].endswith("/python3"):
        return wrap_python_execstart(argv, overlay=overlay, source_root=source_root)
    if first == "bash" or argv[0].endswith("/bash"):
        script = str(
            Path(overlay)
            / "ipfs_accelerate_py/agent_supervisor/rescue"
            / "sawm_supervise_with_heal_overlay.sh"
        )
        return ["/bin/bash", script, overlay, source_root]
    return None


def collapse_extra_gate_recursion(
    board: Mapping[str, Any], observation: Mapping[str, Any],
    *,
    daemon_reload=None,
    systemd_user_dir: Path | None = None,
) -> dict[str, Any]:
    """Keep one exclusive owner. Never wrap ExecStart or start a second extra-gate."""
    empty = {
        "status": "skip",
        "recipe": "collapse_extra_gate_recursion",
        "completion_authority": False,
    }
    board_id = str(board.get("id") or observation.get("board_id") or "").lower()
    if board_id in RETAIN_OWNER_BOARDS:
        return {**empty, "reason": "retain_owner_not_rewrapped"}
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    extra = details.get("extra_gate") if isinstance(details.get("extra_gate"), dict) else {}
    live_unit = str(extra.get("live_owner_unit") or "")
    if not live_unit.endswith(".service") or live_unit == "cron.service":
        return {**empty, "reason": "live_owner_unit_unknown"}
    overlay = supervisor_overlay_root()
    user_dir = systemd_user_dir or (Path.home() / ".config/systemd/user")
    dropin_dir = user_dir / f"{live_unit}.d"
    dropin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # ExecStart wrapping loads overlay quack_state_server against a sealed
    # DuckDB opener and crash-loops the exclusive owner. Bind PYTHONPATH only.
    path = dropin_dir / "80-overlay-pythonpath.conf"
    body = (
        "[Service]\n"
        f"Environment=PYTHONPATH={overlay}\n"
        "TimeoutStopSec=180\n"
    )
    if path.is_file() and path.read_text(encoding="utf-8") == body:
        return {
            **empty,
            "reason": "heal_overlay_pythonpath_already_bound",
            "live_owner_unit": live_unit,
        }
    path.write_text(body, encoding="utf-8")
    os.chmod(path, 0o600)
    exec_wrap = dropin_dir / "81-supervisor-heal-overlay.conf"
    if exec_wrap.is_file():
        exec_wrap.unlink()
    reloader = daemon_reload or _systemd_daemon_reload
    reloader()
    return {
        "status": "applied",
        "recipe": "collapse_extra_gate_recursion",
        "completion_authority": False,
        "live_owner_unit": live_unit,
        "inventory_owner_unit": extra.get("inventory_owner_unit") or "",
        "restarted": False,
        "reason": "one exclusive owner; overlay PYTHONPATH bound; competing extra-gate not started",
    }


def admit_native_owner_overlay(
    board: Mapping[str, Any], observation: Mapping[str, Any],
    *,
    show_unit=None,
    daemon_reload=None,
    restart_unit=None,
    systemd_user_dir: Path | None = None,
) -> dict[str, Any]:
    """Load overlay heals in the live extra-gate so it can natively admit.

    Board scripts insert a sealed ``ipfs_accelerate_py`` first, so PYTHONPATH
    alone never unstalls DOEP-044. Overlay-first keeps one consistent package
    (not a mixed quack_state_server). Never forges completion.
    """
    empty = {
        "status": "skip",
        "recipe": "overlay_first_native_admission",
        "completion_authority": False,
    }
    board_id = str(board.get("id") or observation.get("board_id") or "").lower()
    if board_id in RETAIN_OWNER_BOARDS:
        return {**empty, "reason": "retain_owner_not_rewrapped"}
    if board_id not in NATIVE_ADMISSION_OVERLAY_BOARDS:
        return {**empty, "reason": "board_not_native_admission_overlay"}
    # Sealed extra-gate scripts pass kwargs overlay build_server does not
    # accept (repository_root). Wrapping ExecStart crash-loops the owner.
    return {**empty, "reason": "overlay_first_wrap_disabled_mixed_package"}
    details = observation.get("details") if isinstance(observation.get("details"), dict) else {}
    extra = details.get("extra_gate") if isinstance(details.get("extra_gate"), dict) else {}
    live_unit = str(extra.get("live_owner_unit") or "")
    if not live_unit.endswith(".service") or live_unit == "cron.service":
        return {**empty, "reason": "live_owner_unit_unknown"}
    cwd = str(board.get("cwd") or "")
    if not cwd:
        return {**empty, "reason": "board_cwd_absent"}
    overlay = supervisor_overlay_root()
    show = show_unit or _systemd_show_exec_start
    raw = show(live_unit)
    if isinstance(raw, list):
        argv = [str(item) for item in raw if isinstance(item, str)]
    else:
        argv = parse_systemd_exec_start(str(raw or ""))
    wrapped = overlay_wrapped_execstart(argv, overlay=overlay, source_root=cwd)
    if not wrapped:
        return {**empty, "reason": "execstart_not_wrappable"}
    if any(HEAL_OVERLAY_LAUNCHER in item for item in argv):
        return {**empty, "reason": "overlay_first_already_bound"}
    user_dir = systemd_user_dir or (Path.home() / ".config/systemd/user")
    dropin_dir = user_dir / f"{live_unit}.d"
    dropin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    quoted = " ".join(_systemd_quote(item) for item in wrapped)
    body = (
        "[Service]\n"
        "ExecStart=\n"
        f"ExecStart={quoted}\n"
        f"Environment=PYTHONPATH={overlay}\n"
        "TimeoutStopSec=180\n"
        "SendSIGKILL=yes\n"
        "Restart=on-failure\n"
    )
    path = dropin_dir / "91-overlay-first-admission.conf"
    path.write_text(body, encoding="utf-8")
    os.chmod(path, 0o600)
    reloader = daemon_reload or _systemd_daemon_reload
    reloader()
    restarter = restart_unit or _systemd_restart_unit
    restarter(live_unit)
    return {
        "status": "applied",
        "recipe": "overlay_first_native_admission",
        "completion_authority": False,
        "live_owner_unit": live_unit,
        "restarted": True,
        "reason": (
            "overlay-first extra-gate recycle so owner-side unstall can "
            "natively admit false-terminal blocks; receipts stay incomplete"
        ),
    }


def _systemd_quote(value: str) -> str:
    if value.isalnum() or all(ch in "._/-:+@" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _systemd_show_exec_start(unit: str) -> str:
    completed = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ExecStart"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=5, check=False, text=True,
    )
    return completed.stdout or ""


def _systemd_daemon_reload() -> None:
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=15, check=False,
    )


def _systemd_restart_unit(unit: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "restart", unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=180, check=False,
    )


def apply_supervisor_heal(board: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Formal logic first. llm_router is the residual coding path."""
    observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
    stall = str(state.get("stall_class") or "")
    dirty = restore_dirty_control_plane(board, observation)
    if dirty.get("status") == "applied":
        return dirty
    if stall == "owner_missing":
        return clear_overlay_copies_blocking_owner_start(board)
    if stall == "extra_gate_recursion":
        collapsed = collapse_extra_gate_recursion(board, observation)
        if collapsed.get("status") == "applied":
            return collapsed
        if collapsed.get("reason") == "retain_owner_not_rewrapped":
            return {
                "status": "wait",
                "recipe": "collapse_extra_gate_recursion",
                "completion_authority": False,
                "reason": "retain_owner_not_rewrapped",
            }
    unstall: dict[str, Any] = {"status": "skip"}
    if stall in {
        "independent_work_beside_blocked_peer",
        "independent_todos_unclaimed",
        "in_progress_awaiting_effect",
        "blocked_without_independent_work",
        "stalled_no_progress",
        "kernel_uninterruptible_wait",
        "native_status_unavailable_with_live_workers",
        "extra_gate_recursion",
    }:
        unstall = unstall_stale_native_work(board, observation)
        if unstall.get("status") == "applied":
            return unstall
    if stall == "independent_work_beside_blocked_peer" and live_workers(observation):
        return {"status": "wait", "recipe": "independent_work_has_live_workers",
                "reason": "blocked peers stay blocked; live lanes own independent todos"}
    if stall == "independent_todos_unclaimed":
        prior = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
        prior_results = [item for item in prior.get("results") or [] if isinstance(item, dict)]
        blocked = [
            str(item)
            for item in (observation.get("details") or {}).get("blocked_task_ids") or []
            if item
        ]
        passed = {
            str(item.get("task_id"))
            for item in prior_results
            if item.get("status") == "passed" and item.get("task_id")
        }
        if blocked and set(blocked) <= passed:
            return {
                "status": "applied",
                "recipe": "successors_may_run_on_current_tree_evidence",
                "completion_authoritative": False,
                "completion_authority": False,
                "results": prior_results,
                "reason": (
                    "current-tree tests passed for DOEP-044/DOEP-063; "
                    "remaining todos are not stalled on a DuckDB write"
                ),
            }
        return {"status": "wait", "recipe": "native_lanes_own_independent_todos",
                "reason": "blocked receipts stay blocked; live native lanes claim independent todos"}
    if stall == "stalled_no_progress" and live_workers(observation):
        return {"status": "wait", "recipe": "native_lanes_own_independent_todos",
                "reason": "rearmed or ready work belongs to live native lanes, not llm_router"}
    if stall == "blocked_without_independent_work":
        if local_validation_already_recorded(state, board):
            prior = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
            prior_results = [
                item for item in prior.get("results") or []
                if isinstance(item, dict)
            ]
            rearm: dict[str, Any] = {"status": "skip"}
            if not locally_validated_rearm_already_recorded(state):
                rearm = rearm_locally_validated_blocked_tasks(
                    board, observation, prior_results,
                )
                if rearm.get("status") == "applied":
                    return rearm
            reason = str(rearm.get("reason") or "")
            if reason in {
                "typed_owner_status_session_read_only",
                "owner_live_do_not_stop",
            }:
                return {
                    "status": "applied",
                    "recipe": "local_validation_satisfies_current_tree_requirements",
                    "completion_authoritative": False,
                    "completion_authority": False,
                    "results": prior_results,
                    "reason": (
                        "current-tree tests passed for blocked tasks; "
                        "DuckDB blocked-to-retrying write is not the remaining requirement"
                    ),
                }
            return {
                "status": "wait",
                "recipe": "local_validation_pending_native_admission",
                "completion_authoritative": False,
                "results": prior_results,
                "reason": "local checks already recorded; native owner CAS still required",
            }
        local = run_local_blocked_candidate_validation(board, observation)
        if local.get("status") != "skip":
            local_results = [
                item for item in local.get("results") or []
                if isinstance(item, dict)
            ]
            rearm = rearm_locally_validated_blocked_tasks(
                board, observation, local_results,
            )
            if rearm.get("status") == "applied":
                return rearm
            return local
        return {"status": "wait", "recipe": "todos_waiting_on_blocked_dependencies",
                "reason": "remaining todos depend on blocked peers; do not rewrite those receipts"}
    if stall == "closeout_waiting_on_unsettled_goals":
        if provisional_goal_closeout_already_recorded(state):
            return {"status": "wait", "recipe": "provisionally_complete_terminal_goals",
                    "completion_authority": False,
                    "reason": "provisional closeout already recorded; verification still required"}
        return provisionally_complete_disabled_extra_gate_goals(board, observation)
    if stall == "native_status_unavailable_with_live_workers":
        return {"status": "wait", "recipe": "native_status_retry_with_live_workers",
                "reason": "nonzero native status is not a coding stall while lanes are live"}
    if stall == "in_progress_awaiting_effect":
        reasons = {str(x) for x in observation.get("reason_codes") or []}
        if "extra_gate_recursion_sealed_package" in reasons:
            collapsed = collapse_extra_gate_recursion(board, observation)
            if collapsed.get("status") == "applied":
                return collapsed
        if _observation_uninterruptible(observation):
            return {"status": "wait", "recipe": "kernel_uninterruptible_wait",
                    "reason": "D-state I/O is not a coding stall; do not signal or rewrite receipts"}
        return {"status": "wait", "recipe": "in_progress_awaiting_effect",
                "reason": "in-progress tasks are native work, not a coding stall"}
    if stall == "kernel_uninterruptible_wait":
        return {"status": "wait", "recipe": "kernel_uninterruptible_wait",
                "reason": "D-state I/O is not a coding stall; do not signal or rewrite receipts"}
    if stall == "extra_gate_recursion":
        return {
            "status": "wait",
            "recipe": "collapse_extra_gate_recursion",
            "completion_authority": False,
            "reason": "one exclusive owner; native extra-gate admission uses unstall, not a second extra-gate",
        }
    if stall == "board_checkout_missing":
        return {"status": "wait", "recipe": "deleted_checkout_not_rematerialized",
                "reason": "missing checkout is not reconstructed; retain original authority or explicit retirement"}
    return try_logic_guided_repair(board, state)
