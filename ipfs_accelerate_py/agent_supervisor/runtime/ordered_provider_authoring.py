"""Sealed authority boundary for model-assisted implementation authoring.

This module deliberately does not authorize deterministic repair.  It only
allows an implementation provider to prepare a proposal for an immutable task
that was present in a configured-board bootstrap seal.  Planner, Doctor,
proof, publication, and completion authority remain false on every receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..task_sources.task_identity import (
    canonical_content_cid,
    canonical_task_identity,
)
from ..validation.validation_commands import split_validation_commands

AUTHORING_BOARD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ordered-provider-authoring-board@1"
)
AUTHORING_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ordered-provider-authoring-task@1"
)
AUTHORING_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ordered-provider-authoring-receipt@1"
)
AUTHORING_GATE_INTERFACE: Final[str] = "OrderedProviderAuthoringGate@1"
AUTHORING_LAUNCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ordered-provider-authoring-launch@1"
)
BOOTSTRAP_SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-bootstrap-seal@2"
)

CONFIGURED_BOARD_NAMESPACE_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_NAMESPACE"
)
CONFIGURED_BOARD_CONFIG_PATH_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_CONFIG_PATH"
)
BOOTSTRAP_SEAL_PATH_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_BOOTSTRAP_SEAL_PATH"
)
BOOTSTRAP_SEAL_ID_ENV: Final[str] = "IPFS_ACCELERATE_CONFIGURED_BOARD_BOOTSTRAP_SEAL_ID"
BOOTSTRAP_FOREST_ID_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_BOOTSTRAP_FOREST_ID"
)
BOOTSTRAP_INVENTORY_ID_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_BOOTSTRAP_INVENTORY_ID"
)
BOOTSTRAP_BASELINE_ID_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_BOOTSTRAP_BASELINE_ID"
)
BOOTSTRAP_AUTHORING_BOARD_ID_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_AUTHORING_BOARD_ID"
)
CONFIGURED_BOARD_LAUNCH_HEAD_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_LAUNCH_HEAD"
)
CONFIGURED_BOARD_LAUNCH_TREE_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_LAUNCH_TREE"
)
CONFIGURED_BOARD_LAUNCH_ID_ENV: Final[str] = (
    "IPFS_ACCELERATE_CONFIGURED_BOARD_LAUNCH_ID"
)

PROVIDER_ENV: Final[str] = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
FALLBACK_PROVIDER_ENV: Final[str] = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
)
FALLBACK_TRIGGER_ENV: Final[str] = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
)
GROK_MODEL_ENV: Final[str] = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
GROK_MAX_TURNS_ENV: Final[str] = "IPFS_ACCELERATE_AGENT_GROK_MAX_TURNS"
CODEX_MODEL_ENV: Final[str] = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
CODEX_REASONING_EFFORT_ENV: Final[str] = "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"

ORDERED_PROVIDER_ENV_NAMES: Final[tuple[str, ...]] = (
    CONFIGURED_BOARD_NAMESPACE_ENV,
    CONFIGURED_BOARD_CONFIG_PATH_ENV,
    BOOTSTRAP_SEAL_PATH_ENV,
    BOOTSTRAP_SEAL_ID_ENV,
    BOOTSTRAP_FOREST_ID_ENV,
    BOOTSTRAP_INVENTORY_ID_ENV,
    BOOTSTRAP_BASELINE_ID_ENV,
    BOOTSTRAP_AUTHORING_BOARD_ID_ENV,
    CONFIGURED_BOARD_LAUNCH_HEAD_ENV,
    CONFIGURED_BOARD_LAUNCH_TREE_ENV,
    CONFIGURED_BOARD_LAUNCH_ID_ENV,
    GROK_MAX_TURNS_ENV,
)

_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CID = re.compile(r"^b[a-z2-7]{20,}$")
_CANONICAL_TASK_CID = re.compile(r"^baguqeera[a-z2-7]{52}$")
_MUTABLE_METADATA = frozenset({"status"})
_EXPECTED_PROVIDER_ROLES = ("grok-implement", "codex-review")
_EXPECTED_LLM_CONTEXT_BUDGET_BYTES = 262_144
_EXPECTED_CONTEXT_BUDGET_TOKENS = 16_384
_EXPECTED_ROUTE = {
    "primary_provider_id": "grok_cli",
    "primary_model_id": "grok-4.5",
    "fallback_provider_id": "codex",
    "fallback_model_id": "gpt-5.6-terra",
    "fallback_trigger": "primary_quota_exhausted",
    "fallback_reasoning_effort": "high",
    "fallback_for_other_failures": False,
}
_AUTHORIZED_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "interface",
        "status",
        "reason_code",
        "task_id",
        "canonical_task_cid",
        "authoring_task_id",
        "authoring_board_id",
        "attempt",
        "current_forest_id",
        "current_git_tree_id",
        "output_paths",
        "predicted_files",
        "allowed_paths",
        "authorized_write_paths",
        "validation_commands",
        "llm_context_budget_bytes",
        "context_budget_tokens",
        "workspace_path",
        "provider_command",
        "provider_command_id",
        "launch_authority",
        "provider_route",
        "provider_authorized",
        "provider_hook_count",
        "authority",
        "receipt_id",
    }
)
_LAUNCH_AUTHORITY_KEYS = frozenset(
    {
        "board_namespace",
        "scheduler_config_path",
        "scheduler_config_sha256",
        "launch_id",
        "launch_head",
        "launch_tree",
        "bootstrap_seal_path",
        "seal_id",
        "forest_id",
        "inventory_id",
        "baseline_id",
        "authoring_board_id",
    }
)


class OrderedProviderAuthoringError(ValueError):
    """A fail-closed authoring authority error with a stable reason code."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha_content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OrderedProviderAuthoringError(
                f"duplicate JSON key: {key}",
                reason_code="duplicate_bootstrap_seal_key",
            )
        result[key] = value
    return result


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise OrderedProviderAuthoringError(
            f"{field} is not a string path",
            reason_code="unsafe_authoring_path",
        )
    source = str(value or "")
    raw = source.strip()
    path = PurePosixPath(raw)
    if (
        not raw
        or source != raw
        or "\\" in raw
        or path.as_posix() != raw
        or raw in {".", ".."}
        or raw.startswith("/")
        or "\x00" in raw
        or "://" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(marker in raw for marker in "*?[")
    ):
        raise OrderedProviderAuthoringError(
            f"{field} is not a canonical repository-relative path",
            reason_code="unsafe_authoring_path",
        )
    return path.as_posix()


def _normalized_metadata(
    metadata: Mapping[str, Any],
    *,
    reject_duplicate_normalized_keys: bool = False,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key).strip().lower().replace("_", " ")
        if not key:
            raise OrderedProviderAuthoringError(
                "task metadata contains an empty key",
                reason_code="invalid_authoring_metadata",
            )
        if reject_duplicate_normalized_keys and key in normalized:
            raise OrderedProviderAuthoringError(
                f"task metadata repeats {key!r}",
                reason_code="duplicate_authoring_metadata",
            )
        value = str(raw_value or "").strip()
        if "\x00" in value:
            raise OrderedProviderAuthoringError(
                f"task metadata {key!r} contains NUL",
                reason_code="invalid_authoring_metadata",
            )
        normalized[key] = value
    return normalized


def build_authoring_task_record(
    *,
    task_id: str,
    title: str,
    metadata: Mapping[str, Any],
    board_namespace: str,
) -> dict[str, Any]:
    """Build one status-insensitive, full-metadata task authority record."""

    normalized_id = str(task_id or "").strip()
    normalized_title = " ".join(str(title or "").split())
    namespace = str(board_namespace or "").strip()
    if _TASK_ID.fullmatch(normalized_id) is None:
        raise OrderedProviderAuthoringError(
            "task_id is invalid",
            reason_code="invalid_authoring_task_id",
        )
    if not normalized_title or not namespace:
        raise OrderedProviderAuthoringError(
            "task title and board namespace are required",
            reason_code="invalid_authoring_task_identity",
        )
    normalized = _normalized_metadata(metadata)
    declared_namespace = normalized.get("board namespace", "")
    if declared_namespace != namespace:
        raise OrderedProviderAuthoringError(
            "task board namespace does not match the sealed board",
            reason_code="authoring_board_namespace_mismatch",
        )
    immutable_metadata = {
        key: value
        for key, value in sorted(normalized.items())
        if key not in _MUTABLE_METADATA
    }
    body = {
        "schema": AUTHORING_TASK_SCHEMA,
        "task_id": normalized_id,
        "title": normalized_title,
        "board_namespace": namespace,
        "immutable_metadata": immutable_metadata,
    }
    return {**body, "authoring_task_id": _sha_content_id(body)}


def _parse_authoring_task_records(
    text: str,
    *,
    task_header_prefix: str,
    board_namespace: str,
) -> list[dict[str, Any]]:
    raw_prefix = str(task_header_prefix or "").strip()
    prefix = raw_prefix if raw_prefix.startswith("## ") else f"## {raw_prefix}"
    if not raw_prefix or prefix == "## ":
        raise OrderedProviderAuthoringError(
            "task header prefix is empty",
            reason_code="invalid_authoring_task_prefix",
        )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    current_id = ""
    current_title = ""
    current_metadata: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_id, current_title, current_metadata
        if not current_id:
            return
        if current_id in seen_ids:
            raise OrderedProviderAuthoringError(
                f"duplicate task id: {current_id}",
                reason_code="duplicate_authoring_task",
            )
        seen_ids.add(current_id)
        records.append(
            build_authoring_task_record(
                task_id=current_id,
                title=current_title,
                metadata=current_metadata,
                board_namespace=board_namespace,
            )
        )
        current_id = ""
        current_title = ""
        current_metadata = {}

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            if not line.startswith(prefix):
                continue
            header = line[3:].strip()
            task_parts = header.split(" ", 1)
            current_id = task_parts[0]
            current_title = task_parts[1] if len(task_parts) == 2 else ""
            continue
        if not current_id:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        raw_key, raw_value = stripped[2:].split(":", 1)
        key = raw_key.strip().lower().replace("_", " ")
        if key in current_metadata:
            raise OrderedProviderAuthoringError(
                f"task {current_id} repeats metadata key {key!r}",
                reason_code="duplicate_authoring_metadata",
            )
        current_metadata[key] = raw_value.strip()
    flush()
    return records


def build_authoring_board_projection(
    *,
    taskboard_path: Path | str,
    task_header_prefix: str,
    board_namespace: str,
) -> dict[str, Any]:
    """Build a status-insensitive projection of every sealed task block."""

    try:
        text = Path(taskboard_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OrderedProviderAuthoringError(
            "taskboard is unreadable",
            reason_code="authoring_taskboard_unreadable",
        ) from exc
    normalized_prefix = str(task_header_prefix).strip()
    normalized_prefix = (
        normalized_prefix
        if normalized_prefix.startswith("## ")
        else f"## {normalized_prefix}"
    )
    records = _parse_authoring_task_records(
        text,
        task_header_prefix=normalized_prefix,
        board_namespace=board_namespace,
    )
    task_ids = [record["task_id"] for record in records]
    body = {
        "schema": AUTHORING_BOARD_SCHEMA,
        "board_namespace": str(board_namespace).strip(),
        "task_header_prefix": normalized_prefix,
        "tasks": [
            {
                "task_id": record["task_id"],
                "authoring_task_id": record["authoring_task_id"],
            }
            for record in records
        ],
    }
    if len(task_ids) != len(set(task_ids)):
        raise OrderedProviderAuthoringError(
            "taskboard contains duplicate task ids",
            reason_code="duplicate_authoring_task",
        )
    return {**body, "authoring_board_id": _sha_content_id(body)}


def _content_identity_matches(payload: Mapping[str, Any], field: str) -> bool:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or _SHA256_ID.fullmatch(claimed) is None:
        return False
    body = {key: value for key, value in payload.items() if key != field}
    return _sha_content_id(body) == claimed


def _load_bootstrap_seal(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except OrderedProviderAuthoringError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrderedProviderAuthoringError(
            "bootstrap seal is unreadable",
            reason_code="authoring_bootstrap_seal_unreadable",
        ) from exc
    if not isinstance(payload, dict):
        raise OrderedProviderAuthoringError(
            "bootstrap seal must be an object",
            reason_code="invalid_authoring_bootstrap_seal",
        )
    return payload


def _load_scheduler_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except OrderedProviderAuthoringError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrderedProviderAuthoringError(
            "configured-board scheduler config is unreadable",
            reason_code="authoring_scheduler_config_unreadable",
        ) from exc
    if not isinstance(payload, dict):
        raise OrderedProviderAuthoringError(
            "configured-board scheduler config must be an object",
            reason_code="invalid_authoring_scheduler_config",
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _environment_value(environment: Mapping[str, Any], name: str) -> str:
    return str(environment.get(name) or "").strip()


def _verified_launch_authority(
    *,
    repo_root: Path,
    environment: Mapping[str, Any],
    current_authoring_board: Mapping[str, Any],
    taskboard_path: Path,
    task_header_prefix: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    namespace = _environment_value(environment, CONFIGURED_BOARD_NAMESPACE_ENV)
    scheduler_config_path = _safe_relative(
        _environment_value(environment, CONFIGURED_BOARD_CONFIG_PATH_ENV),
        field="scheduler_config_path",
    )
    seal_path = _safe_relative(
        _environment_value(environment, BOOTSTRAP_SEAL_PATH_ENV),
        field="bootstrap_seal_path",
    )
    expected = {
        "seal_id": _environment_value(environment, BOOTSTRAP_SEAL_ID_ENV),
        "forest_id": _environment_value(environment, BOOTSTRAP_FOREST_ID_ENV),
        "inventory_id": _environment_value(environment, BOOTSTRAP_INVENTORY_ID_ENV),
        "baseline_id": _environment_value(environment, BOOTSTRAP_BASELINE_ID_ENV),
        "authoring_board_id": _environment_value(
            environment, BOOTSTRAP_AUTHORING_BOARD_ID_ENV
        ),
    }
    launch_head = _environment_value(environment, CONFIGURED_BOARD_LAUNCH_HEAD_ENV)
    launch_tree = _environment_value(environment, CONFIGURED_BOARD_LAUNCH_TREE_ENV)
    launch_id = _environment_value(environment, CONFIGURED_BOARD_LAUNCH_ID_ENV)
    if (
        not namespace
        or any(_SHA256_ID.fullmatch(value) is None for value in expected.values())
        or any(
            re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in (launch_head, launch_tree)
        )
        or _SHA256_ID.fullmatch(launch_id) is None
    ):
        raise OrderedProviderAuthoringError(
            "configured-board launch authority is incomplete",
            reason_code="missing_authoring_launch_authority",
        )
    seal_file = (repo_root / seal_path).resolve()
    try:
        seal_file.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise OrderedProviderAuthoringError(
            "bootstrap seal escapes the repository",
            reason_code="unsafe_authoring_seal_path",
        ) from exc
    if seal_file.is_symlink():
        raise OrderedProviderAuthoringError(
            "bootstrap seal must not be a symlink",
            reason_code="unsafe_authoring_seal_path",
        )
    config_file = (repo_root / scheduler_config_path).resolve()
    try:
        config_file.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise OrderedProviderAuthoringError(
            "scheduler config escapes the repository",
            reason_code="unsafe_authoring_scheduler_config_path",
        ) from exc
    if config_file.is_symlink() or not config_file.is_file():
        raise OrderedProviderAuthoringError(
            "scheduler config is missing or unsafe",
            reason_code="unsafe_authoring_scheduler_config_path",
        )
    scheduler_config, scheduler_config_sha256 = _load_scheduler_config(config_file)
    seal = _load_bootstrap_seal(seal_file)
    if seal.get("schema") != BOOTSTRAP_SEAL_SCHEMA:
        raise OrderedProviderAuthoringError(
            "bootstrap seal schema is not authoring-capable",
            reason_code="authoring_bootstrap_seal_schema_mismatch",
        )
    if seal.get("board_namespace") != namespace:
        raise OrderedProviderAuthoringError(
            "bootstrap seal namespace drifted",
            reason_code="authoring_board_namespace_mismatch",
        )
    forest = seal.get("forest")
    inventory = seal.get("inventory")
    baseline = seal.get("baseline")
    sealed_board = seal.get("authoring_board")
    if not all(
        isinstance(value, Mapping)
        for value in (forest, inventory, baseline, sealed_board)
    ):
        raise OrderedProviderAuthoringError(
            "bootstrap seal is missing typed components",
            reason_code="invalid_authoring_bootstrap_seal",
        )
    components = (
        (seal, "seal_id", expected["seal_id"]),
        (forest, "forest_id", expected["forest_id"]),
        (inventory, "inventory_id", expected["inventory_id"]),
        (baseline, "baseline_id", expected["baseline_id"]),
        (
            sealed_board,
            "authoring_board_id",
            expected["authoring_board_id"],
        ),
    )
    for payload, field, expected_value in components:
        if payload.get(field) != expected_value or not _content_identity_matches(
            payload, field
        ):
            raise OrderedProviderAuthoringError(
                f"bootstrap {field} is forged or stale",
                reason_code="forged_authoring_launch_authority",
            )
    source_binding = scheduler_config.get("source_binding")
    provider = scheduler_config.get("provider")
    execution_policy = scheduler_config.get("execution_policy")
    protected_paths = scheduler_config.get("protected_paths")
    configured_taskboard = scheduler_config.get("taskboard_path")
    configured_prefix = str(scheduler_config.get("task_prefix") or "").strip()
    observed_prefix = str(task_header_prefix or "").strip()
    if observed_prefix.startswith("## "):
        observed_prefix = observed_prefix[3:].strip()
    configured_taskboard = _safe_relative(
        configured_taskboard,
        field="scheduler taskboard_path",
    )
    expected_taskboard = (repo_root / configured_taskboard).resolve()
    if (
        scheduler_config.get("board_namespace") != namespace
        or not isinstance(source_binding, Mapping)
        or source_binding.get("bootstrap_seal_path") != seal_path
        or configured_prefix != observed_prefix
        or expected_taskboard != taskboard_path.resolve()
        or not isinstance(protected_paths, list)
        or not all(isinstance(item, str) for item in protected_paths)
        or not {scheduler_config_path, seal_path, configured_taskboard}.issubset(
            set(protected_paths)
        )
    ):
        raise OrderedProviderAuthoringError(
            "scheduler config does not bind this authoring board",
            reason_code="authoring_scheduler_config_mismatch",
        )
    if (
        not isinstance(provider, Mapping)
        or any(
            provider.get(key) != value
            for key, value in _EXPECTED_ROUTE.items()
            if key != "fallback_for_other_failures"
        )
        or provider.get("provider_fallback_for_other_failures") is not False
        or not isinstance(execution_policy, Mapping)
        or execution_policy.get("implementation_authoring_mode") != "ordered_provider"
        or execution_policy.get("implementation_provider_role")
        != "grok-implement, codex-review"
        or execution_policy.get("repair_runtime_mode") != "deterministic_only"
        or execution_policy.get("repair_runtime_model_calls") != 0
        or execution_policy.get("repair_runtime_llm_calls") != 0
        or execution_policy.get("implementation_llm_context_budget_bytes")
        != _EXPECTED_LLM_CONTEXT_BUDGET_BYTES
        or execution_policy.get("implementation_context_budget_tokens")
        != _EXPECTED_CONTEXT_BUDGET_TOKENS
        or execution_policy.get(
            "provider_fallback_allowed_only_for_primary_quota_exhaustion"
        )
        is not True
    ):
        raise OrderedProviderAuthoringError(
            "scheduler execution policy differs from reviewed authoring policy",
            reason_code="authoring_scheduler_policy_mismatch",
        )
    controls = inventory.get("controls")
    if not isinstance(controls, list):
        raise OrderedProviderAuthoringError(
            "bootstrap inventory controls are malformed",
            reason_code="authoring_scheduler_config_unsealed",
        )
    config_records = [
        item
        for item in controls
        if isinstance(item, Mapping) and item.get("path") == scheduler_config_path
    ]
    if (
        len(config_records) != 1
        or config_records[0].get("sha256") != scheduler_config_sha256
    ):
        raise OrderedProviderAuthoringError(
            "scheduler config bytes are not bound by the bootstrap inventory",
            reason_code="authoring_scheduler_config_unsealed",
        )
    launch_body = {
        "schema": AUTHORING_LAUNCH_SCHEMA,
        "board_namespace": namespace,
        "scheduler_config_path": scheduler_config_path,
        "scheduler_config_sha256": scheduler_config_sha256,
        "seal_id": expected["seal_id"],
        "forest_id": expected["forest_id"],
        "inventory_id": expected["inventory_id"],
        "baseline_id": expected["baseline_id"],
        "authoring_board_id": expected["authoring_board_id"],
        "launch_head": launch_head,
        "launch_tree": launch_tree,
    }
    if _sha_content_id(launch_body) != launch_id:
        raise OrderedProviderAuthoringError(
            "configured-board launch binding is forged",
            reason_code="forged_authoring_launch_binding",
        )
    roots = forest.get("roots")
    if (
        not isinstance(roots, list)
        or not roots
        or not all(
            isinstance(item, Mapping) and _content_identity_matches(item, "root_id")
            for item in roots
        )
    ):
        raise OrderedProviderAuthoringError(
            "bootstrap forest has no content-bound roots",
            reason_code="forged_authoring_launch_authority",
        )
    if dict(sealed_board) != dict(current_authoring_board):
        raise OrderedProviderAuthoringError(
            "current taskboard authoring projection differs from the seal",
            reason_code="authoring_board_drift",
        )
    return {
        "board_namespace": namespace,
        "scheduler_config_path": scheduler_config_path,
        "scheduler_config_sha256": scheduler_config_sha256,
        "launch_id": launch_id,
        "launch_head": launch_head,
        "launch_tree": launch_tree,
        "bootstrap_seal_path": seal_path,
        **expected,
    }, seal


def _provider_route(environment: Mapping[str, Any]) -> dict[str, Any]:
    route: dict[str, Any] = {
        "primary_provider_id": _environment_value(environment, PROVIDER_ENV),
        "primary_model_id": _environment_value(environment, GROK_MODEL_ENV),
        "fallback_provider_id": _environment_value(environment, FALLBACK_PROVIDER_ENV),
        "fallback_model_id": _environment_value(environment, CODEX_MODEL_ENV),
        "fallback_trigger": _environment_value(environment, FALLBACK_TRIGGER_ENV),
        "fallback_reasoning_effort": _environment_value(
            environment, CODEX_REASONING_EFFORT_ENV
        ),
        "fallback_for_other_failures": False,
    }
    if any(route[key] != value for key, value in _EXPECTED_ROUTE.items()):
        raise OrderedProviderAuthoringError(
            "ordered provider route differs from reviewed Grok/Terra policy",
            reason_code="authoring_provider_route_mismatch",
        )
    return route


def _single_command_flag(values: Sequence[str], flag: str) -> str:
    indexes = [index for index, value in enumerate(values) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(values):
        raise OrderedProviderAuthoringError(
            f"ordered provider command has no unique {flag}",
            reason_code="authoring_provider_command_mismatch",
        )
    return values[indexes[0] + 1]


def validate_ordered_provider_command(
    command: Sequence[Any],
    *,
    workspace_path: Path | str,
) -> str:
    """Return the digest of one exact Grok-first, quota-only Terra argv."""

    if isinstance(command, (str, bytes, bytearray)):
        raise OrderedProviderAuthoringError(
            "ordered provider command must be an argv vector",
            reason_code="authoring_provider_command_mismatch",
        )
    if not all(isinstance(item, str) for item in command):
        raise OrderedProviderAuthoringError(
            "ordered provider command contains a non-string argument",
            reason_code="authoring_provider_command_mismatch",
        )
    values = [str(item) for item in command]
    expected_flags = (
        "--workspace",
        "--model",
        "--max-turns",
        "--mode",
        "--codex-fallback-reasoning-effort",
        "--codex-fallback-command-json",
        "--grok-bin",
        "--grok-failure-receipt-nonce",
    )
    if (
        len(values) != 3 + 2 * len(expected_flags)
        or values[:3]
        != [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
        or tuple(values[3::2]) != expected_flags
    ):
        raise OrderedProviderAuthoringError(
            "ordered provider command is not the canonical runner argv",
            reason_code="authoring_provider_command_mismatch",
        )
    workspace = Path(workspace_path).resolve()
    if not workspace.is_dir() or _single_command_flag(values, "--workspace") != str(
        workspace
    ):
        raise OrderedProviderAuthoringError(
            "ordered provider workspace is stale",
            reason_code="authoring_provider_command_mismatch",
        )
    if (
        _single_command_flag(values, "--model") != "grok-4.5"
        or _single_command_flag(values, "--mode") != "agent"
        or _single_command_flag(values, "--codex-fallback-reasoning-effort") != "high"
    ):
        raise OrderedProviderAuthoringError(
            "ordered provider model or effort drifted",
            reason_code="authoring_provider_command_mismatch",
        )
    max_turns = _single_command_flag(values, "--max-turns")
    if max_turns != "100000":
        raise OrderedProviderAuthoringError(
            "ordered provider max-turns is invalid",
            reason_code="authoring_provider_command_mismatch",
        )
    nonce = _single_command_flag(values, "--grok-failure-receipt-nonce")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise OrderedProviderAuthoringError(
            "ordered provider invocation nonce is invalid",
            reason_code="authoring_provider_command_mismatch",
        )
    grok = Path(_single_command_flag(values, "--grok-bin"))
    if (
        not grok.is_absolute()
        or grok.name.casefold() not in {"grok", "grok.exe"}
        or not grok.is_file()
        or not os.access(grok, os.X_OK)
        or grok.resolve().is_relative_to(workspace)
    ):
        raise OrderedProviderAuthoringError(
            "ordered provider Grok executable is untrusted",
            reason_code="authoring_provider_command_mismatch",
        )
    try:
        fallback = json.loads(
            _single_command_flag(values, "--codex-fallback-command-json")
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise OrderedProviderAuthoringError(
            "ordered provider fallback argv is malformed",
            reason_code="authoring_provider_command_mismatch",
        ) from exc
    if not isinstance(fallback, list) or not all(
        isinstance(item, str) for item in fallback
    ):
        raise OrderedProviderAuthoringError(
            "ordered provider fallback argv is malformed",
            reason_code="authoring_provider_command_mismatch",
        )
    codex = Path(fallback[0]) if fallback else Path("")
    expected_fallback_tail = [
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]
    if (
        fallback[1:] != expected_fallback_tail
        or not codex.is_absolute()
        or codex.name.casefold() not in {"codex", "codex.exe"}
        or not codex.is_file()
        or not os.access(codex, os.X_OK)
        or codex.resolve().is_relative_to(workspace)
    ):
        raise OrderedProviderAuthoringError(
            "ordered provider fallback is not quota-only Terra/high",
            reason_code="authoring_provider_command_mismatch",
        )
    return _sha_content_id({"argv": values})


def _expected_canonical_task_cid(
    *,
    task_id: str,
    title: str,
    metadata: Mapping[str, str],
    outputs: Sequence[str],
    board_namespace: str,
) -> str:
    identity_outputs = list(outputs)
    evidence_outputs = [
        _safe_relative(item.strip(), field="evidence output")
        for item in metadata.get("evidence outputs", "").split(",")
        if item.strip()
    ]
    if evidence_outputs:
        missing = {
            _safe_relative(item.strip(), field="missing evidence")
            for item in metadata.get("missing evidence", "").split(",")
            if item.strip()
        }
        subset = {
            _safe_relative(item.strip(), field="evidence subset")
            for item in metadata.get("evidence subset", "").split(",")
            if item.strip()
        }
        if set(evidence_outputs).issubset(missing) and set(evidence_outputs).issubset(
            subset
        ):
            identity_outputs.extend(evidence_outputs)
    return canonical_task_identity(
        {
            "task_id": task_id,
            "title": title,
            "outputs": list(dict.fromkeys(identity_outputs)),
            "acceptance": metadata.get("acceptance", ""),
            "metadata": dict(metadata),
        },
        board_namespace=board_namespace,
    ).canonical_task_cid


def _authorize_task_metadata(
    metadata: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    int,
    int,
]:
    normalized = _normalized_metadata(metadata)
    roles = tuple(
        value.strip().lower()
        for value in re.split(r"[,;]", normalized.get("provider role", ""))
        if value.strip()
    )
    if normalized.get("implementation mode", "").lower() != "ordered_provider":
        raise OrderedProviderAuthoringError(
            "task is not an ordered-provider authoring task",
            reason_code="authoring_mode_mismatch",
        )
    if roles != _EXPECTED_PROVIDER_ROLES:
        raise OrderedProviderAuthoringError(
            "task provider roles are not the reviewed ordered pair",
            reason_code="authoring_provider_roles_mismatch",
        )
    if normalized.get("runtime model calls") != "0":
        raise OrderedProviderAuthoringError(
            "authored runtime must declare zero model calls",
            reason_code="authoring_runtime_model_calls_nonzero",
        )
    if normalized.get("symbolic first", "").lower() != "true":
        raise OrderedProviderAuthoringError(
            "task is not symbolic-first",
            reason_code="authoring_symbolic_first_required",
        )
    if normalized.get("deterministic repair route") or normalized.get(
        "execution route"
    ):
        raise OrderedProviderAuthoringError(
            "repair execution routes cannot use authoring authority",
            reason_code="repair_route_forbidden_from_authoring",
        )
    if normalized.get("canonical task key") or normalized.get("canonical task cid"):
        raise OrderedProviderAuthoringError(
            "caller-supplied canonical task identities cannot authorize authoring",
            reason_code="explicit_task_identity_override_forbidden",
        )
    outputs = tuple(
        _safe_relative(value.strip(), field="task output")
        for value in normalized.get("outputs", "").split(",")
        if value.strip()
    )
    predicted_files = tuple(
        _safe_relative(value.strip(), field="predicted file")
        for value in normalized.get("predicted files", "").split(",")
        if value.strip()
    )
    allowed_paths = tuple(
        _safe_relative(value.strip(), field="allowed path")
        for value in normalized.get("allowed paths", "").split(",")
        if value.strip()
    )
    validations = tuple(split_validation_commands(normalized.get("validation", "")))
    if (
        not outputs
        or len(outputs) != len(set(outputs))
        or not predicted_files
        or len(predicted_files) != len(set(predicted_files))
        or len(allowed_paths) != len(set(allowed_paths))
        or not validations
    ):
        raise OrderedProviderAuthoringError(
            "task outputs and validation must be exact and nonempty",
            reason_code="incomplete_authoring_task_contract",
        )
    if any("\x00" in value or "\n" in value for value in validations):
        raise OrderedProviderAuthoringError(
            "task validation contains unsafe bytes",
            reason_code="invalid_authoring_validation",
        )
    try:
        llm_context_budget_bytes = int(normalized.get("llm context budget bytes", ""))
        context_budget_tokens = int(normalized.get("context budget tokens", ""))
    except ValueError as exc:
        raise OrderedProviderAuthoringError(
            "authoring context budgets are missing or malformed",
            reason_code="authoring_context_budget_mismatch",
        ) from exc
    if (
        llm_context_budget_bytes != _EXPECTED_LLM_CONTEXT_BUDGET_BYTES
        or context_budget_tokens != _EXPECTED_CONTEXT_BUDGET_TOKENS
    ):
        raise OrderedProviderAuthoringError(
            "authoring context budgets differ from reviewed limits",
            reason_code="authoring_context_budget_mismatch",
        )
    authorized_write_paths = tuple(
        dict.fromkeys([*outputs, *predicted_files, *allowed_paths])
    )
    return (
        normalized,
        outputs,
        predicted_files,
        allowed_paths,
        authorized_write_paths,
        validations,
        llm_context_budget_bytes,
        context_budget_tokens,
    )


def evaluate_ordered_provider_authoring(
    *,
    repo_root: Path | str,
    taskboard_path: Path | str,
    task_header_prefix: str,
    task_id: str,
    title: str,
    metadata: Mapping[str, Any],
    canonical_task_cid: str,
    current_forest_id: str,
    current_git_tree_id: str,
    workspace_path: Path | str,
    provider_command: Sequence[Any],
    runtime_write_scope: Sequence[Any],
    isolated_worktree: bool,
    attempt: int,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a content-addressed authorize/reject authoring decision."""

    normalized_task_id = str(task_id or "").strip()
    normalized_task_cid = str(canonical_task_cid or "").strip()
    try:
        if _TASK_ID.fullmatch(normalized_task_id) is None:
            raise OrderedProviderAuthoringError(
                "task id is invalid",
                reason_code="invalid_authoring_task_id",
            )
        if _CANONICAL_TASK_CID.fullmatch(normalized_task_cid) is None:
            raise OrderedProviderAuthoringError(
                "canonical task CID is missing or malformed",
                reason_code="invalid_authoring_task_cid",
            )
        if isinstance(attempt, bool) or int(attempt) < 1:
            raise OrderedProviderAuthoringError(
                "attempt must be positive",
                reason_code="invalid_authoring_attempt",
            )
        if _SHA256_ID.fullmatch(str(current_forest_id or "").strip()) is None:
            raise OrderedProviderAuthoringError(
                "current forest identity is malformed",
                reason_code="missing_authoring_forest",
            )
        if (
            re.fullmatch(r"[0-9a-f]{40}", str(current_git_tree_id or "").strip())
            is None
        ):
            raise OrderedProviderAuthoringError(
                "current Git tree identity is malformed",
                reason_code="missing_authoring_forest",
            )
        root = Path(repo_root).resolve()
        raw_board_file = Path(taskboard_path)
        raw_board_file = (
            raw_board_file if raw_board_file.is_absolute() else root / raw_board_file
        )
        if raw_board_file.is_symlink():
            raise OrderedProviderAuthoringError(
                "taskboard is missing or unsafe",
                reason_code="unsafe_authoring_taskboard_path",
            )
        board_file = raw_board_file.resolve()
        try:
            board_file.relative_to(root)
        except ValueError as exc:
            raise OrderedProviderAuthoringError(
                "taskboard escapes the repository",
                reason_code="unsafe_authoring_taskboard_path",
            ) from exc
        if not board_file.is_file():
            raise OrderedProviderAuthoringError(
                "taskboard is missing or unsafe",
                reason_code="unsafe_authoring_taskboard_path",
            )
        workspace = Path(workspace_path).resolve()
        if isolated_worktree is not True or workspace == root:
            raise OrderedProviderAuthoringError(
                "ordered-provider authoring requires an isolated worktree",
                reason_code="authoring_isolated_worktree_required",
            )
        provider_command_id = validate_ordered_provider_command(
            provider_command,
            workspace_path=workspace,
        )
        namespace = _environment_value(environment, CONFIGURED_BOARD_NAMESPACE_ENV)
        current_board = build_authoring_board_projection(
            taskboard_path=board_file,
            task_header_prefix=task_header_prefix,
            board_namespace=namespace,
        )
        launch_authority, _seal = _verified_launch_authority(
            repo_root=Path(repo_root).resolve(),
            environment=environment,
            current_authoring_board=current_board,
            taskboard_path=board_file,
            task_header_prefix=task_header_prefix,
        )
        task_record = build_authoring_task_record(
            task_id=normalized_task_id,
            title=title,
            metadata=metadata,
            board_namespace=namespace,
        )
        sealed_tasks = {
            str(item.get("task_id") or ""): str(item.get("authoring_task_id") or "")
            for item in current_board.get("tasks", ())
            if isinstance(item, Mapping)
        }
        if sealed_tasks.get(normalized_task_id) != task_record["authoring_task_id"]:
            raise OrderedProviderAuthoringError(
                "task metadata is not in the sealed authoring projection",
                reason_code="unsealed_authoring_task",
            )
        (
            normalized,
            outputs,
            predicted_files,
            allowed_paths,
            authorized_write_paths,
            validations,
            llm_context_budget_bytes,
            context_budget_tokens,
        ) = _authorize_task_metadata(metadata)
        normalized_runtime_scope = tuple(
            _safe_relative(item, field="runtime write scope")
            for item in runtime_write_scope
        )
        if len(normalized_runtime_scope) != len(set(normalized_runtime_scope)) or set(
            normalized_runtime_scope
        ) != set(authorized_write_paths):
            raise OrderedProviderAuthoringError(
                "runtime write scope differs from the sealed task contract",
                reason_code="authoring_write_scope_mismatch",
            )
        if normalized_task_cid != _expected_canonical_task_cid(
            task_id=normalized_task_id,
            title=title,
            metadata=normalized,
            outputs=outputs,
            board_namespace=namespace,
        ):
            raise OrderedProviderAuthoringError(
                "canonical task CID does not match current task semantics",
                reason_code="authoring_task_cid_mismatch",
            )
        route = _provider_route(environment)
        body: dict[str, Any] = {
            "schema": AUTHORING_RECEIPT_SCHEMA,
            "interface": AUTHORING_GATE_INTERFACE,
            "status": "authorized",
            "reason_code": "sealed_ordered_provider_authoring",
            "task_id": normalized_task_id,
            "canonical_task_cid": normalized_task_cid,
            "authoring_task_id": task_record["authoring_task_id"],
            "authoring_board_id": current_board["authoring_board_id"],
            "attempt": int(attempt),
            "current_forest_id": str(current_forest_id).strip(),
            "current_git_tree_id": str(current_git_tree_id).strip(),
            "output_paths": list(outputs),
            "predicted_files": list(predicted_files),
            "allowed_paths": list(allowed_paths),
            "authorized_write_paths": list(authorized_write_paths),
            "validation_commands": list(validations),
            "llm_context_budget_bytes": llm_context_budget_bytes,
            "context_budget_tokens": context_budget_tokens,
            "workspace_path": str(workspace),
            "provider_command": [str(item) for item in provider_command],
            "provider_command_id": provider_command_id,
            "launch_authority": launch_authority,
            "provider_route": route,
            "provider_authorized": True,
            "provider_hook_count": 0,
            "authority": {
                "proposal_only": True,
                "deterministic_repair": False,
                "runtime_repair": False,
                "planner": False,
                "doctor": False,
                "proof": False,
                "publication": False,
                "completion": False,
            },
        }
    except OrderedProviderAuthoringError as exc:
        body = {
            "schema": AUTHORING_RECEIPT_SCHEMA,
            "interface": AUTHORING_GATE_INTERFACE,
            "status": "rejected",
            "reason_code": exc.reason_code,
            "task_id": normalized_task_id,
            "canonical_task_cid": normalized_task_cid,
            "attempt": int(attempt) if isinstance(attempt, int) else 0,
            "provider_authorized": False,
            "provider_hook_count": 0,
            "authority": {
                "proposal_only": False,
                "deterministic_repair": False,
                "runtime_repair": False,
                "planner": False,
                "doctor": False,
                "proof": False,
                "publication": False,
                "completion": False,
            },
        }
    return {**body, "receipt_id": canonical_content_cid(body)}


def authoring_provider_invocation_authorized(receipt: Mapping[str, Any]) -> bool:
    """Recompute the receipt and require its exact non-repair authority shape."""

    if set(receipt) != _AUTHORIZED_RECEIPT_KEYS:
        return False
    payload = dict(receipt)
    receipt_id = payload.pop("receipt_id", "")
    authority = payload.get("authority")
    launch_authority = payload.get("launch_authority")
    provider_route = payload.get("provider_route")
    if (
        not isinstance(receipt_id, str)
        or canonical_content_cid(payload) != receipt_id
        or payload.get("schema") != AUTHORING_RECEIPT_SCHEMA
        or payload.get("interface") != AUTHORING_GATE_INTERFACE
        or payload.get("status") != "authorized"
        or payload.get("reason_code") != "sealed_ordered_provider_authoring"
        or payload.get("provider_authorized") is not True
        or type(payload.get("provider_hook_count")) is not int
        or payload.get("provider_hook_count") != 0
        or type(payload.get("attempt")) is not int
        or int(payload["attempt"]) < 1
        or not isinstance(payload.get("task_id"), str)
        or _TASK_ID.fullmatch(str(payload.get("task_id") or "")) is None
        or not isinstance(payload.get("canonical_task_cid"), str)
        or _CANONICAL_TASK_CID.fullmatch(str(payload.get("canonical_task_cid") or ""))
        is None
        or not isinstance(payload.get("authoring_task_id"), str)
        or _SHA256_ID.fullmatch(str(payload.get("authoring_task_id") or "")) is None
        or not isinstance(payload.get("authoring_board_id"), str)
        or _SHA256_ID.fullmatch(str(payload.get("authoring_board_id") or "")) is None
        or not isinstance(payload.get("current_forest_id"), str)
        or _SHA256_ID.fullmatch(str(payload.get("current_forest_id") or "")) is None
        or not isinstance(payload.get("current_git_tree_id"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(payload.get("current_git_tree_id") or ""))
        is None
        or payload.get("llm_context_budget_bytes") != _EXPECTED_LLM_CONTEXT_BUDGET_BYTES
        or payload.get("context_budget_tokens") != _EXPECTED_CONTEXT_BUDGET_TOKENS
        or not isinstance(launch_authority, Mapping)
        or set(launch_authority) != _LAUNCH_AUTHORITY_KEYS
        or not all(isinstance(value, str) for value in launch_authority.values())
        or not str(launch_authority.get("board_namespace") or "").strip()
        or not isinstance(provider_route, Mapping)
        or set(provider_route) != set(_EXPECTED_ROUTE)
        or any(
            provider_route.get(key) is not value
            if isinstance(value, bool)
            else provider_route.get(key) != value
            for key, value in _EXPECTED_ROUTE.items()
        )
        or not isinstance(authority, Mapping)
        or set(authority)
        != {
            "proposal_only",
            "deterministic_repair",
            "runtime_repair",
            "planner",
            "doctor",
            "proof",
            "publication",
            "completion",
        }
        or authority.get("proposal_only") is not True
        or any(
            authority.get(field) is not False
            for field in (
                "deterministic_repair",
                "runtime_repair",
                "planner",
                "doctor",
                "proof",
                "publication",
                "completion",
            )
        )
    ):
        return False
    if not all(
        _SHA256_ID.fullmatch(str(launch_authority.get(field) or ""))
        for field in (
            "launch_id",
            "seal_id",
            "forest_id",
            "inventory_id",
            "baseline_id",
            "authoring_board_id",
        )
    ):
        return False
    try:
        _safe_relative(
            launch_authority.get("scheduler_config_path"),
            field="scheduler config path",
        )
        _safe_relative(
            launch_authority.get("bootstrap_seal_path"),
            field="bootstrap seal path",
        )
        if (
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(launch_authority.get("scheduler_config_sha256") or ""),
            )
            is None
        ):
            return False
        if any(
            re.fullmatch(
                r"[0-9a-f]{40}",
                str(launch_authority.get(field) or ""),
            )
            is None
            for field in ("launch_head", "launch_tree")
        ):
            return False
        launch_body = {
            "schema": AUTHORING_LAUNCH_SCHEMA,
            "board_namespace": launch_authority["board_namespace"],
            "scheduler_config_path": launch_authority["scheduler_config_path"],
            "scheduler_config_sha256": launch_authority["scheduler_config_sha256"],
            "seal_id": launch_authority["seal_id"],
            "forest_id": launch_authority["forest_id"],
            "inventory_id": launch_authority["inventory_id"],
            "baseline_id": launch_authority["baseline_id"],
            "authoring_board_id": launch_authority["authoring_board_id"],
            "launch_head": launch_authority["launch_head"],
            "launch_tree": launch_authority["launch_tree"],
        }
        if _sha_content_id(launch_body) != launch_authority["launch_id"]:
            return False
        path_lists: dict[str, tuple[str, ...]] = {}
        for field in (
            "output_paths",
            "predicted_files",
            "allowed_paths",
            "authorized_write_paths",
        ):
            raw = payload.get(field)
            if not isinstance(raw, list) or not all(
                isinstance(item, str) for item in raw
            ):
                return False
            normalized = tuple(_safe_relative(item, field=field) for item in raw)
            if len(normalized) != len(set(normalized)):
                return False
            path_lists[field] = normalized
        if (
            not path_lists["output_paths"]
            or not path_lists["predicted_files"]
            or tuple(
                dict.fromkeys(
                    [
                        *path_lists["output_paths"],
                        *path_lists["predicted_files"],
                        *path_lists["allowed_paths"],
                    ]
                )
            )
            != path_lists["authorized_write_paths"]
        ):
            return False
        validations = payload.get("validation_commands")
        if (
            not isinstance(validations, list)
            or not validations
            or not all(
                isinstance(item, str)
                and item.strip() == item
                and "\x00" not in item
                and "\n" not in item
                for item in validations
            )
        ):
            return False
        workspace = payload.get("workspace_path")
        if not isinstance(workspace, str) or not Path(workspace).is_absolute():
            return False
        provider_command = payload.get("provider_command")
        if not isinstance(provider_command, list):
            return False
        if validate_ordered_provider_command(
            provider_command,
            workspace_path=workspace,
        ) != payload.get("provider_command_id"):
            return False
    except (OSError, OrderedProviderAuthoringError, TypeError, ValueError):
        return False
    return True


def assert_current_authoring_dispatch_authority(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path | str,
    taskboard_path: Path | str,
    workspace_path: Path | str,
) -> None:
    """Reobserve tracked launch controls and the exact pre-dispatch tree.

    A self-consistent receipt is only a structural claim.  Provider process
    invocation additionally requires the scheduler config, bootstrap seal,
    and taskboard to remain tracked and unmodified, and the isolated task
    worktree to remain at the merge target's current commit and tree.
    """

    if not authoring_provider_invocation_authorized(receipt):
        raise OrderedProviderAuthoringError(
            "authoring receipt is not structurally authorized",
            reason_code="forged_authoring_receipt",
        )
    root = Path(repo_root).resolve()
    workspace = Path(workspace_path).resolve()
    board = Path(taskboard_path)
    board = (board if board.is_absolute() else root / board).resolve()
    launch = receipt["launch_authority"]
    config_relative = _safe_relative(
        launch["scheduler_config_path"],
        field="scheduler_config_path",
    )
    seal_relative = _safe_relative(
        launch["bootstrap_seal_path"],
        field="bootstrap_seal_path",
    )
    config_path = (root / config_relative).resolve()
    scheduler_config, scheduler_sha256 = _load_scheduler_config(config_path)
    if scheduler_sha256 != launch["scheduler_config_sha256"]:
        raise OrderedProviderAuthoringError(
            "scheduler config changed after authoring evaluation",
            reason_code="authoring_scheduler_config_drift",
        )
    configured_board = _safe_relative(
        scheduler_config.get("taskboard_path"),
        field="scheduler taskboard_path",
    )
    if (root / configured_board).resolve() != board:
        raise OrderedProviderAuthoringError(
            "scheduler taskboard changed after authoring evaluation",
            reason_code="authoring_taskboard_drift",
        )

    def git(cwd: Path, *args: str) -> str:
        try:
            result = subprocess.run(
                ("git", *args),
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OrderedProviderAuthoringError(
                "Git state cannot be observed before provider dispatch",
                reason_code="authoring_git_observation_failed",
            ) from exc
        if result.returncode != 0:
            raise OrderedProviderAuthoringError(
                "Git state is not valid before provider dispatch",
                reason_code="authoring_git_observation_failed",
            )
        return result.stdout.strip()

    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise OrderedProviderAuthoringError(
            "authoring repository root is not exact",
            reason_code="authoring_git_root_mismatch",
        )
    if Path(git(workspace, "rev-parse", "--show-toplevel")).resolve() != workspace:
        raise OrderedProviderAuthoringError(
            "authoring worktree root is not exact",
            reason_code="authoring_git_root_mismatch",
        )
    controls = (config_relative, seal_relative, configured_board)
    git(root, "ls-files", "--error-unmatch", "--", *controls)
    if git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *controls,
    ):
        raise OrderedProviderAuthoringError(
            "authoring launch controls are locally modified",
            reason_code="authoring_launch_controls_dirty",
        )
    source_binding = scheduler_config.get("source_binding")
    expected_branch = (
        str(source_binding.get("accelerator_required_branch") or "").strip()
        if isinstance(source_binding, Mapping)
        else ""
    )
    if not expected_branch or git(root, "branch", "--show-current") != expected_branch:
        raise OrderedProviderAuthoringError(
            "authoring merge-target branch drifted",
            reason_code="authoring_branch_mismatch",
        )
    root_head = git(root, "rev-parse", "HEAD^{commit}")
    workspace_head = git(workspace, "rev-parse", "HEAD^{commit}")
    root_tree = git(root, "rev-parse", "HEAD^{tree}")
    workspace_tree = git(workspace, "rev-parse", "HEAD^{tree}")
    # The configured merge-target branch is the append-only trust root after
    # launch.  Parallel supervisor merges may advance it, but a rewritten or
    # unrelated history cannot replay the launch binding.
    git(root, "merge-base", "--is-ancestor", launch["launch_head"], root_head)
    if (
        root_head != workspace_head
        or root_tree != workspace_tree
        or workspace_tree != receipt["current_git_tree_id"]
        or str(workspace) != receipt["workspace_path"]
        or (root_head == launch["launch_head"] and root_tree != launch["launch_tree"])
    ):
        raise OrderedProviderAuthoringError(
            "authoring worktree no longer matches the current merge target",
            reason_code="authoring_worktree_binding_drift",
        )
    if git(workspace, "status", "--porcelain=v1", "--untracked-files=all"):
        raise OrderedProviderAuthoringError(
            "authoring worktree is not a clean provider baseline",
            reason_code="authoring_worktree_dirty",
        )
    configured_roots = scheduler_config.get("worktree_submodule_paths")
    if (
        not isinstance(configured_roots, list)
        or not configured_roots
        or not all(isinstance(item, str) for item in configured_roots)
    ):
        raise OrderedProviderAuthoringError(
            "scheduler configured roots are missing",
            reason_code="authoring_configured_roots_malformed",
        )
    normalized_roots = tuple(
        _safe_relative(item, field="worktree_submodule_path")
        for item in configured_roots
    )
    if len(normalized_roots) != len(set(normalized_roots)):
        raise OrderedProviderAuthoringError(
            "scheduler configured roots are duplicated",
            reason_code="authoring_configured_roots_malformed",
        )
    for relative in normalized_roots:
        tree_row = git(workspace, "ls-tree", "HEAD", "--", relative)
        match = re.fullmatch(
            rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}",
            tree_row,
        )
        target = (workspace / relative).resolve()
        if match is None or not target.is_dir():
            raise OrderedProviderAuthoringError(
                "configured authoring root is absent or not a Gitlink",
                reason_code="authoring_configured_root_mismatch",
            )
        if Path(git(target, "rev-parse", "--show-toplevel")).resolve() != target:
            raise OrderedProviderAuthoringError(
                "configured authoring root is not an exact worktree",
                reason_code="authoring_configured_root_mismatch",
            )
        if git(target, "rev-parse", "HEAD^{commit}") != match.group(1) or git(
            target, "status", "--porcelain=v1", "--untracked-files=all"
        ):
            raise OrderedProviderAuthoringError(
                "configured authoring root is stale or dirty",
                reason_code="authoring_configured_root_mismatch",
            )


__all__ = (
    "AUTHORING_BOARD_SCHEMA",
    "AUTHORING_GATE_INTERFACE",
    "AUTHORING_LAUNCH_SCHEMA",
    "AUTHORING_RECEIPT_SCHEMA",
    "AUTHORING_TASK_SCHEMA",
    "BOOTSTRAP_AUTHORING_BOARD_ID_ENV",
    "BOOTSTRAP_BASELINE_ID_ENV",
    "BOOTSTRAP_FOREST_ID_ENV",
    "BOOTSTRAP_INVENTORY_ID_ENV",
    "BOOTSTRAP_SEAL_ID_ENV",
    "BOOTSTRAP_SEAL_PATH_ENV",
    "CONFIGURED_BOARD_CONFIG_PATH_ENV",
    "CONFIGURED_BOARD_LAUNCH_HEAD_ENV",
    "CONFIGURED_BOARD_LAUNCH_ID_ENV",
    "CONFIGURED_BOARD_LAUNCH_TREE_ENV",
    "CONFIGURED_BOARD_NAMESPACE_ENV",
    "GROK_MAX_TURNS_ENV",
    "ORDERED_PROVIDER_ENV_NAMES",
    "OrderedProviderAuthoringError",
    "assert_current_authoring_dispatch_authority",
    "authoring_provider_invocation_authorized",
    "build_authoring_board_projection",
    "build_authoring_task_record",
    "evaluate_ordered_provider_authoring",
)
