"""Fail-closed task authority, phase, dependency, and completion receipts.

The provider may propose bytes only for ``Provider effects``.  Files declared
as ``Supervisor outputs`` remain part of the immutable task DAG contract, but
are materialized only by trusted supervisor code after the implementation and
acceptance gates have passed.  This module deliberately contains no process
execution or Git mutation; it only builds and verifies bounded content-bound
contracts exchanged by those lifecycle phases.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)
GOVERNED_VALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/governed-validation-receipt@1"
)
GOVERNED_EXECUTION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/governed-execution-plan@1"
)
GOVERNED_PHASE_COMMAND_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/governed-phase-command-set@1"
)
SUPERVISOR_TASK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-task-receipt@1"
)
PRE_CHANGE_POLICIES: Final[frozenset[str]] = frozenset(
    {"require-pass", "record-baseline"}
)
GOVERNED_VALIDATION_PHASES: Final[frozenset[str]] = frozenset(
    {"pre_change", "post_change", "acceptance"}
)
EXECUTOR_KINDS: Final[frozenset[str]] = frozenset(
    {"evidence_job", "patch_job", "execution_job", "gate_job"}
)
GOVERNED_REPOSITORY_ROOTS: Final[dict[str, str]] = {
    "control": ".",
    "endomorphosis/ipfs_datasets_py": "external/ipfs_datasets",
    "endomorphosis/ipfs_kit_py": "external/ipfs_kit",
    "endomorphosis/ipfs_accelerate_py": "external/ipfs_accelerate",
    "endomorphosis/Mcp-Plus-Plus": "Mcp-Plus-Plus",
}
GOVERNED_COMMAND_ENVIRONMENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "CI",
        "IPFS_ACCEL_SKIP_CORE",
        "IPFS_KIT_DISABLE",
        "LANG",
        "LC_ALL",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "TZ",
    }
)
GOVERNED_COMMAND_EXECUTABLES: Final[frozenset[str]] = frozenset(
    {
        "python",
        "python3",
    }
)
_SHELL_CONTROL_ARGUMENTS: Final[frozenset[str]] = frozenset(
    {"&&", "||", ";", "|", "&", "<", ">", "(", ")"}
)
_COMMAND_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
)

_MAX_ITEMS = 256
_MAX_DEPTH = 8
_MAX_TEXT = 4096
_SENSITIVE_KEYS = frozenset(
    {
        "output",
        "outputs",
        "stdout",
        "stderr",
        "prompt",
        "prompts",
        "model_output",
        "provider_response",
        "raw_context",
        "raw_output",
        "raw_response",
        "source_body",
        "source_code",
    }
)


def _normalized_metadata(task: Any) -> dict[str, str]:
    metadata = getattr(task, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("task metadata must be a mapping")
    return {
        str(key).strip().lower().replace("_", " ").replace("-", " "): str(
            value or ""
        ).strip()
        for key, value in metadata.items()
    }


def _canonical_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or raw != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(character in raw for character in "*?[]{}")
        or any(ord(character) < 32 for character in raw)
        or ".git" in path.parts
    ):
        raise ValueError(f"task path is unsafe: {value!r}")
    return raw


def _split_paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[,;\n]+", str(value or ""))
    paths = tuple(_canonical_path(item) for item in raw if str(item).strip())
    if len(paths) > 64:
        raise ValueError("task path declarations exceed the governed bound")
    if len(paths) != len(set(paths)):
        raise ValueError("task path declarations must be unique")
    return paths


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _parse_canonical_json(raw: str, *, field: str) -> Any:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be canonical JSON") from exc
    try:
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON values") from exc
    if canonical != raw:
        raise ValueError(f"{field} must use compact canonical JSON serialization")
    return value


def _canonical_relative_path(value: Any, *, allow_dot: bool) -> str:
    raw = str(value or "")
    if raw == "." and allow_dot:
        return raw
    if raw != raw.strip() or "\\" in raw:
        raise ValueError("governed repository/CWD path is not canonical")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.casefold() == ".git" for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("governed repository/CWD path is unsafe")
    return raw


@dataclass(frozen=True, slots=True)
class GovernedPhaseCommand:
    """One shell-free command bound to an exact governed repository root."""

    command_id: str
    repository: str
    repository_root: str
    cwd: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    timeout_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(self.env),
            "id": self.command_id,
            "repository": self.repository,
            "repository_root": self.repository_root,
            "timeout_seconds": self.timeout_seconds,
        }


def _governed_phase_command(value: Mapping[str, Any]) -> GovernedPhaseCommand:
    expected_keys = {
        "argv",
        "cwd",
        "env",
        "id",
        "repository",
        "repository_root",
        "timeout_seconds",
    }
    if set(value) != expected_keys:
        raise ValueError("governed phase command shape is invalid")
    command_id = str(value.get("id") or "")
    if _COMMAND_ID_RE.fullmatch(command_id) is None:
        raise ValueError("governed command id is invalid")
    repository = str(value.get("repository") or "")
    if repository not in GOVERNED_REPOSITORY_ROOTS:
        raise ValueError("governed command repository is not allowlisted")
    repository_root = _canonical_relative_path(
        value.get("repository_root"),
        allow_dot=True,
    )
    if repository_root != GOVERNED_REPOSITORY_ROOTS[repository]:
        raise ValueError("governed command repository/root binding is invalid")
    cwd = _canonical_relative_path(value.get("cwd"), allow_dot=True)
    raw_argv = value.get("argv")
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or len(raw_argv) > 256
        or not all(isinstance(item, str) and item for item in raw_argv)
    ):
        raise ValueError("governed command argv must be a bounded string array")
    argv = tuple(raw_argv)
    if (
        argv[0] not in GOVERNED_COMMAND_EXECUTABLES
        or argv[0] in {"bash", "sh"}
        or any(
            item in _SHELL_CONTROL_ARGUMENTS
            or "\x00" in item
            or any(ord(character) < 32 for character in item)
            or len(item.encode("utf-8")) > 4096
            for item in argv
        )
    ):
        raise ValueError("governed command argv violates process policy")
    raw_env = value.get("env")
    if not isinstance(raw_env, Mapping) or len(raw_env) > 32:
        raise ValueError("governed command env must be a bounded object")
    environment: list[tuple[str, str]] = []
    for key in sorted(raw_env):
        item = raw_env[key]
        if (
            not isinstance(key, str)
            or key not in GOVERNED_COMMAND_ENVIRONMENT_KEYS
            or not isinstance(item, str)
            or len(item.encode("utf-8")) > 4096
            or "\x00" in item
            or any(ord(character) < 32 for character in item)
        ):
            raise ValueError("governed command env violates process policy")
        if key == "PYTHONPATH":
            entries = item.split(":")
            if not entries or any(
                not entry
                or (
                    entry != "."
                    and _canonical_relative_path(entry, allow_dot=True) != entry
                )
                for entry in entries
            ):
                raise ValueError("governed PYTHONPATH must stay repository-relative")
        environment.append((key, item))
    timeout = value.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise ValueError("governed command timeout must be 1..3600 seconds")
    return GovernedPhaseCommand(
        command_id=command_id,
        repository=repository,
        repository_root=repository_root,
        cwd=cwd,
        argv=argv,
        env=tuple(environment),
        timeout_seconds=timeout,
    )


def _supervisor_only_path(path: str) -> bool:
    lowered = path.casefold()
    return bool(
        "/receipts/" in f"/{lowered}"
        or lowered.endswith("/task-receipt.json")
        or lowered.endswith("/control-receipt.json")
        or lowered.endswith("/completion-receipt.json")
    )


@dataclass(frozen=True, slots=True)
class TaskAuthorityPartition:
    """Exact immutable partition between provider and supervisor writes."""

    outputs: tuple[str, ...]
    provider_effects: tuple[str, ...]
    supervisor_outputs: tuple[str, ...]
    explicit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": list(self.outputs),
            "provider_effects": list(self.provider_effects),
            "supervisor_outputs": list(self.supervisor_outputs),
            "explicit": self.explicit,
        }


def task_authority_partition(task: Any) -> TaskAuthorityPartition:
    """Return the exact partition, rejecting widening and receipt authorship."""

    outputs = _split_paths(getattr(task, "outputs", ()) or ())
    if not outputs:
        raise ValueError("task outputs must be explicit")
    metadata = _normalized_metadata(task)
    provider_declared = "provider effects" in metadata
    supervisor_declared = "supervisor outputs" in metadata
    explicit = provider_declared or supervisor_declared
    if explicit:
        # Once either authority field is used, both declarations are required.
        # An empty supervisor field is valid and explicitly means no trusted
        # generated output.
        if not (provider_declared and supervisor_declared):
            raise ValueError(
                "Provider effects and Supervisor outputs must be declared together"
            )
        provider = _split_paths(metadata.get("provider effects", ""))
        supervisor = _split_paths(metadata.get("supervisor outputs", ""))
    else:
        # Compatibility is limited to ordinary product files.  A legacy task
        # that puts a receipt in Outputs must be amended instead of granting a
        # model permission to author supervisor evidence.
        if any(_supervisor_only_path(path) for path in outputs):
            raise ValueError(
                "receipt/control outputs require an explicit supervisor partition"
            )
        provider = outputs
        supervisor = ()
    if set(provider) & set(supervisor):
        raise ValueError("provider and supervisor authority must be disjoint")
    if set((*provider, *supervisor)) != set(outputs) or len(
        (*provider, *supervisor)
    ) != len(outputs):
        raise ValueError("provider/supervisor authority must exactly partition Outputs")
    if any(_supervisor_only_path(path) for path in provider):
        raise ValueError("provider effects may not include supervisor receipt paths")
    return TaskAuthorityPartition(outputs, provider, supervisor, explicit)


def task_phase_commands(task: Any, phase: str) -> tuple[dict[str, Any], ...]:
    """Parse one canonical shell-free phase command set from task metadata."""

    metadata = _normalized_metadata(task)
    key = {
        "pre_change": "pre change validation",
        "post_change": "post change validation",
        "acceptance": "acceptance validation",
    }.get(str(phase or "").strip())
    if key is None:
        raise ValueError("unsupported governed validation phase")
    raw = metadata.get(key, "")
    if not raw:
        return ()
    value = _parse_canonical_json(raw, field=key)
    if not isinstance(value, Mapping) or set(value) != {"commands", "schema"}:
        raise ValueError(f"{key} command-set shape is invalid")
    if value.get("schema") != GOVERNED_PHASE_COMMAND_SET_SCHEMA:
        raise ValueError(f"{key} command-set schema is invalid")
    raw_commands = value.get("commands")
    if (
        not isinstance(raw_commands, list)
        or not raw_commands
        or len(raw_commands) > 64
        or not all(isinstance(item, Mapping) for item in raw_commands)
    ):
        raise ValueError(f"{key} commands must be a nonempty bounded array")
    commands = tuple(_governed_phase_command(item) for item in raw_commands)
    ids = [item.command_id for item in commands]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{key} command ids must be unique")
    return tuple(command.to_dict() for command in commands)


def task_pre_change_policy(task: Any) -> str:
    metadata = _normalized_metadata(task)
    policy = metadata.get("pre change validation policy", "").strip().lower()
    commands = task_phase_commands(task, "pre_change")
    if not commands:
        return "not-required"
    if policy not in PRE_CHANGE_POLICIES:
        raise ValueError("pre-change validation policy must be explicit")
    return policy


def task_executor_kind(task: Any) -> str:
    """Return the closed execution class; legacy product edits are patches."""

    metadata = _normalized_metadata(task)
    explicit = "executor kind" in metadata
    value = metadata.get("executor kind", "patch_job").strip().lower()
    if value not in EXECUTOR_KINDS:
        raise ValueError("Executor kind must use the closed governed taxonomy")
    if explicit and not value:
        raise ValueError("Executor kind must not be empty")
    return value


def task_expected_baseline_failure(task: Any) -> dict[str, Any] | None:
    """Return an explicit known-red signature for record-baseline policy."""

    metadata = _normalized_metadata(task)
    raw = metadata.get("expected baseline failure", "").strip()
    if not raw:
        return None
    value = _parse_canonical_json(raw, field="expected baseline failure")
    if not isinstance(value, Mapping):
        raise ValueError("expected baseline failure must be an object")
    allowed = {
        "exit_class",
        "failed_test_ids",
        "normalized_output_cid",
        "returncodes",
    }
    if set(value) != allowed:
        raise ValueError("expected baseline failure shape is invalid")
    exit_class = str(value.get("exit_class") or "")
    normalized_output_cid = str(value.get("normalized_output_cid") or "")
    test_ids = value.get("failed_test_ids", [])
    returncodes = value.get("returncodes")
    if exit_class not in {"test_failure", "static_failure", "proof_failure"}:
        raise ValueError("expected baseline failure exit_class is invalid")
    if not re.fullmatch(r"b[a-z2-7]{20,120}", normalized_output_cid):
        raise ValueError("expected baseline failure output CID is invalid")
    if not isinstance(test_ids, list) or not all(
        isinstance(item, str) and item for item in test_ids
    ):
        raise ValueError("expected baseline failed_test_ids must be strings")
    if (
        not isinstance(returncodes, Mapping)
        or not returncodes
        or not all(
            isinstance(key, str)
            and _COMMAND_ID_RE.fullmatch(key) is not None
            and isinstance(item, int)
            and not isinstance(item, bool)
            and item != 0
            for key, item in returncodes.items()
        )
    ):
        raise ValueError("expected baseline returncodes must be exact nonzero integers")
    return {
        "exit_class": exit_class,
        "failed_test_ids": sorted(set(test_ids)),
        "normalized_output_cid": normalized_output_cid,
        "returncodes": {key: returncodes[key] for key in sorted(returncodes)},
    }


def task_dependency_completion_receipts(task: Any) -> dict[str, dict[str, str]]:
    """Return explicit active-generation selectors for predecessor receipts."""

    metadata = _normalized_metadata(task)
    raw = metadata.get("dependency completion receipts", "")
    if not raw:
        return {}
    value = _parse_canonical_json(raw, field="dependency completion receipts")
    if not isinstance(value, Mapping) or len(value) > 256:
        raise ValueError("dependency completion receipts must be an object")
    result: dict[str, dict[str, str]] = {}
    for dependency in sorted(value):
        selector = value[dependency]
        if (
            not isinstance(dependency, str)
            or not dependency
            or not isinstance(selector, Mapping)
            or set(selector)
            != {
                "completion_generation",
                "path",
                "producer_task_cid",
                "schema",
            }
        ):
            raise ValueError("dependency completion receipt selector is invalid")
        path = _canonical_path(selector.get("path"))
        schema = str(selector.get("schema") or "")
        generation = str(selector.get("completion_generation") or "")
        producer_task_cid = str(selector.get("producer_task_cid") or "")
        if (
            schema != SUPERVISOR_TASK_RECEIPT_SCHEMA
            or not generation
            or _COMMAND_ID_RE.fullmatch(generation) is None
            or not producer_task_cid
        ):
            raise ValueError("dependency completion receipt authority is invalid")
        result[dependency] = {
            "completion_generation": generation,
            "path": path,
            "producer_task_cid": producer_task_cid,
            "schema": schema,
        }
    return result


def validation_command_specs(
    commands: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Revalidate and detach declared command specs for a receipt."""

    result = [_governed_phase_command(item).to_dict() for item in commands]
    ids = [item["id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("governed command ids must be unique")
    return result


def _sensitive_key(value: str) -> bool:
    normalized = value.strip().casefold().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith("_output")


def bounded_receipt_projection(value: Any, *, depth: int = 0) -> Any:
    """Remove raw provider/command prose and bound receipt diagnostics."""

    if depth >= _MAX_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, bool)):
        return bounded_receipt_projection(enum_value, depth=depth + 1)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ContractValidationError("receipt keys must be strings")
        result: dict[str, Any] = {}
        for key in sorted(value)[:_MAX_ITEMS]:
            if _sensitive_key(key):
                continue
            result[key[:_MAX_TEXT]] = bounded_receipt_projection(
                value[key], depth=depth + 1
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            bounded_receipt_projection(item, depth=depth + 1)
            for item in value[:_MAX_ITEMS]
        ]
    raise ContractValidationError(
        f"unsupported receipt value: {type(value).__name__}"
    )


def _failed_test_ids(result: Mapping[str, Any]) -> list[str]:
    candidates: list[Any] = []
    root_candidates = result.get("failed_tests", [])
    if isinstance(root_candidates, Sequence) and not isinstance(
        root_candidates, (str, bytes, bytearray)
    ):
        candidates.extend(root_candidates)
    command_results = result.get("results", [])
    if isinstance(command_results, Sequence) and not isinstance(
        command_results, (str, bytes, bytearray)
    ):
        for command_result in command_results:
            if not isinstance(command_result, Mapping):
                continue
            values = command_result.get("failed_test_ids", [])
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray)
            ):
                candidates.extend(values)
    return sorted({str(item) for item in candidates if str(item)})[:64]


def _validation_failure_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    raw_results = result.get("results", [])
    command_results = (
        list(raw_results)
        if isinstance(raw_results, Sequence)
        and not isinstance(raw_results, (str, bytes, bytearray))
        else []
    )
    returncodes: dict[str, int] = {}
    output_bindings: list[dict[str, str]] = []
    timed_out = False
    unavailable = False
    for item in command_results:
        if not isinstance(item, Mapping):
            continue
        command_id = str(item.get("id") or "")
        raw_returncode = item.get("returncode", 1)
        returncode = (
            raw_returncode
            if isinstance(raw_returncode, int)
            and not isinstance(raw_returncode, bool)
            else 1
        )
        if command_id and returncode != 0:
            returncodes[command_id] = returncode
        normalized_output_cid = str(item.get("normalized_output_cid") or "")
        if command_id and re.fullmatch(r"b[a-z2-7]{20,120}", normalized_output_cid):
            output_bindings.append(
                {"id": command_id, "normalized_output_cid": normalized_output_cid}
            )
        timed_out = timed_out or item.get("timed_out") is True
        unavailable = unavailable or item.get("infrastructure_failure") is True
    if unavailable:
        exit_class = "infrastructure_failure"
    elif timed_out:
        exit_class = "timeout"
    elif returncodes:
        declared = str(result.get("failure_class") or "test_failure")
        exit_class = (
            declared
            if declared in {"test_failure", "static_failure", "proof_failure"}
            else "test_failure"
        )
    elif result.get("attempted") is True:
        exit_class = "success"
    else:
        exit_class = "not_attempted"
    return {
        "exit_class": exit_class,
        "failed_test_ids": _failed_test_ids(result),
        "normalized_output_cid": content_identity(output_bindings),
        "returncodes": {key: returncodes[key] for key in sorted(returncodes)},
    }


def build_governed_execution_plan(
    *,
    task_contract: Mapping[str, Any],
    baseline_commit: str,
    baseline_forest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind immutable task authority to one exact five-root baseline forest."""

    contract = dict(task_contract)
    task_contract_cid = content_identity(contract)
    roots = bounded_receipt_projection([dict(item) for item in baseline_forest])
    if not isinstance(roots, list):
        raise ContractValidationError("baseline forest must be a list")
    declared_roots = [str(item.get("repository_root") or "") for item in roots]
    expected_roots = sorted(GOVERNED_REPOSITORY_ROOTS.values())
    if sorted(declared_roots) != expected_roots or len(set(declared_roots)) != len(
        declared_roots
    ):
        raise ValueError("execution plan must bind every governed repository root")
    for item in roots:
        root = str(item.get("repository_root") or "")
        repository = str(item.get("repository") or "")
        if GOVERNED_REPOSITORY_ROOTS.get(repository) != root:
            raise ValueError("execution plan repository forest is not canonical")
        if item.get("present") is True:
            if not all(
                str(item.get(key) or "")
                for key in ("commit", "repository_id", "tree_id")
            ):
                raise ValueError("present execution-plan root lacks Git identity")
        elif item.get("present") is not False:
            raise ValueError("execution-plan root presence is invalid")
    baseline = str(baseline_commit or "").strip()
    if not baseline:
        raise ValueError("execution plan baseline commit is required")
    payload = {
        "schema": GOVERNED_EXECUTION_PLAN_SCHEMA,
        "task_id": str(contract.get("task_id") or ""),
        "task_cid": str(contract.get("canonical_task_cid") or ""),
        "task_intent_cid": str(contract.get("task_intent_cid") or ""),
        "task_contract_cid": task_contract_cid,
        "baseline_commit": baseline,
        "baseline_forest": roots,
        "baseline_forest_cid": content_identity(roots),
        "authority_partition": bounded_receipt_projection(
            dict(contract.get("authority_partition") or {})
        ),
        "pre_change_validation": bounded_receipt_projection(
            list(contract.get("pre_change_validation") or [])
        ),
        "post_change_validation": bounded_receipt_projection(
            list(contract.get("post_change_validation") or [])
        ),
        "acceptance_validation": bounded_receipt_projection(
            list(contract.get("acceptance_validation") or [])
        ),
        "dependency_completion_receipts": bounded_receipt_projection(
            list(contract.get("dependency_completion_receipts") or [])
        ),
    }
    if not all(
        str(payload.get(key) or "")
        for key in ("task_id", "task_cid", "task_intent_cid")
    ):
        raise ValueError("execution plan task binding is incomplete")
    payload["execution_plan_cid"] = content_identity(payload)
    return payload


def verify_governed_execution_plan(
    value: Mapping[str, Any] | None,
    *,
    task_contract_cid: str = "",
    baseline_commit: str = "",
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return False, ("governed_execution_plan_missing",)
    reasons: list[str] = []
    if value.get("schema") != GOVERNED_EXECUTION_PLAN_SCHEMA:
        reasons.append("governed_execution_plan_schema_invalid")
    if task_contract_cid and value.get("task_contract_cid") != task_contract_cid:
        reasons.append("governed_execution_plan_task_contract_mismatch")
    if baseline_commit and value.get("baseline_commit") != baseline_commit:
        reasons.append("governed_execution_plan_baseline_mismatch")
    forest = value.get("baseline_forest")
    if not isinstance(forest, list) or value.get("baseline_forest_cid") != content_identity(
        forest if isinstance(forest, list) else []
    ):
        reasons.append("governed_execution_plan_forest_invalid")
    try:
        unsigned = dict(value)
        plan_cid = unsigned.pop("execution_plan_cid", None)
        if plan_cid != content_identity(unsigned):
            reasons.append("governed_execution_plan_cid_mismatch")
    except (ContractValidationError, TypeError, ValueError):
        reasons.append("governed_execution_plan_not_canonical")
    return not reasons, tuple(dict.fromkeys(reasons))


def build_governed_validation_receipt(
    *,
    phase: str,
    task_id: str,
    task_cid: str,
    task_contract_cid: str,
    execution_plan_cid: str,
    target_commit: str,
    repository_tree_id: str,
    repository_id: str,
    commands: Sequence[Mapping[str, Any]],
    validation_result: Mapping[str, Any],
    policy: str,
    expected_baseline_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a typed phase receipt with supervisor-derived admission."""

    phase = str(phase or "").strip()
    if phase not in GOVERNED_VALIDATION_PHASES:
        raise ValueError("unsupported governed validation phase")
    task_id = str(task_id or "").strip()
    task_cid = str(task_cid or "").strip()
    task_contract_cid = str(task_contract_cid or "").strip()
    execution_plan_cid = str(execution_plan_cid or "").strip()
    target_commit = str(target_commit or "").strip()
    repository_tree_id = str(repository_tree_id or "").strip()
    repository_id = str(repository_id or "").strip()
    if not all(
        (
            task_id,
            task_cid,
            task_contract_cid,
            execution_plan_cid,
            target_commit,
            repository_tree_id,
            repository_id,
        )
    ):
        raise ValueError("phase receipt requires task and Git bindings")
    specs = validation_command_specs(tuple(commands))
    if phase in {"post_change", "acceptance"}:
        policy = "require-pass" if specs else "not-required"
    elif policy not in {*PRE_CHANGE_POLICIES, "not-required"}:
        raise ValueError("unsupported pre-change validation policy")
    if bool(specs) == (policy == "not-required"):
        raise ValueError("validation policy/command declaration mismatch")

    projection = bounded_receipt_projection(dict(validation_result))
    if not isinstance(projection, dict):
        raise ContractValidationError("validation result projection must be an object")
    result_cid = content_identity(projection)
    attempted = validation_result.get("attempted") is True
    raw_returncode = validation_result.get("returncode", 1)
    returncode = (
        raw_returncode
        if isinstance(raw_returncode, int) and not isinstance(raw_returncode, bool)
        else 1
    )
    validated_commit = str(
        validation_result.get("validated_commit") or target_commit
    ).strip()
    stale = bool(
        validation_result.get("stale")
        or validation_result.get("validation_stale")
        or validated_commit != target_commit
    )
    reason = str(validation_result.get("reason") or "").strip()
    unavailable = bool(
        validation_result.get("infrastructure_failure")
        or validation_result.get("unavailable")
        or validation_result.get("timed_out")
        or any(
            token in reason.casefold()
            for token in (
                "unavailable",
                "workspace_missing",
                "scheduler_unavailable",
                "execution_exception",
                "timeout",
            )
        )
    )
    commands_passed = bool(
        specs
        and attempted
        and validation_result.get("passed") is True
        and returncode == 0
        and not stale
        and not unavailable
    )
    expected = (
        bounded_receipt_projection(dict(expected_baseline_failure))
        if expected_baseline_failure is not None
        else None
    )
    failure_signature = _validation_failure_signature(validation_result)
    failure_signature_cid = content_identity(failure_signature)
    expected_match = bool(
        phase == "pre_change"
        and policy == "record-baseline"
        and isinstance(expected, Mapping)
        and attempted
        and not stale
        and not unavailable
        and failure_signature == dict(expected)
    )
    forest_before = bounded_receipt_projection(
        list(validation_result.get("forest_before") or [])
    )
    forest_after = bounded_receipt_projection(
        list(validation_result.get("forest_after") or [])
    )
    forest_stable = bool(forest_before and forest_before == forest_after)
    admitted = bool(forest_stable and (commands_passed or expected_match))
    payload: dict[str, Any] = {
        "schema": GOVERNED_VALIDATION_RECEIPT_SCHEMA,
        "phase": phase,
        "task_id": task_id,
        "task_cid": task_cid,
        "task_contract_cid": task_contract_cid,
        "execution_plan_cid": execution_plan_cid,
        "target_commit": target_commit,
        "validated_commit": validated_commit,
        "repository_tree_id": repository_tree_id,
        "repository_id": repository_id,
        "policy": policy,
        "command_specs": specs,
        "command_specs_cid": content_identity(specs),
        "attempted": attempted,
        "commands_passed": commands_passed,
        "expected_baseline_failure": expected,
        "expected_baseline_failure_matched": expected_match,
        "baseline_failure_signature": failure_signature,
        "baseline_failure_signature_cid": failure_signature_cid,
        "admitted": admitted,
        "returncode": returncode,
        "stale": stale,
        "unavailable": unavailable,
        "result_cid": result_cid,
        "result": projection,
        "forest_before": forest_before,
        "forest_after": forest_after,
    }
    payload["forest_before_cid"] = content_identity(payload["forest_before"])
    payload["forest_after_cid"] = content_identity(payload["forest_after"])
    payload["receipt_cid"] = content_identity(payload)
    return payload


def verify_governed_validation_receipt(
    value: Mapping[str, Any] | None,
    *,
    phase: str = "",
    task_id: str = "",
    task_cid: str = "",
    task_contract_cid: str = "",
    execution_plan_cid: str = "",
    target_commit: str = "",
    repository_tree_id: str = "",
    repository_id: str = "",
    forest_before: Sequence[Mapping[str, Any]] | None = None,
    forest_after: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return False, ("governed_validation_receipt_missing",)
    reasons: list[str] = []
    if value.get("schema") != GOVERNED_VALIDATION_RECEIPT_SCHEMA:
        reasons.append("governed_validation_schema_invalid")
    for field, expected in (
        ("phase", phase),
        ("task_id", task_id),
        ("task_cid", task_cid),
        ("task_contract_cid", task_contract_cid),
        ("execution_plan_cid", execution_plan_cid),
        ("target_commit", target_commit),
        ("repository_tree_id", repository_tree_id),
        ("repository_id", repository_id),
    ):
        actual = value.get(field)
        if not isinstance(actual, str) or not actual:
            reasons.append(f"governed_validation_{field}_invalid")
        elif expected and actual != expected:
            reasons.append(f"governed_validation_{field}_mismatch")
    result = value.get("result")
    specs = value.get("command_specs")
    receipt_forest_before = value.get("forest_before")
    receipt_forest_after = value.get("forest_after")
    try:
        if not isinstance(result, Mapping) or dict(result) != bounded_receipt_projection(result):
            raise ValueError
        if value.get("result_cid") != content_identity(result):
            reasons.append("governed_validation_result_cid_mismatch")
        if not isinstance(specs, list) or value.get("command_specs_cid") != content_identity(specs):
            reasons.append("governed_validation_command_specs_cid_mismatch")
        if (
            not isinstance(receipt_forest_before, list)
            or not isinstance(receipt_forest_after, list)
            or value.get("forest_before_cid")
            != content_identity(receipt_forest_before)
            or value.get("forest_after_cid")
            != content_identity(receipt_forest_after)
        ):
            reasons.append("governed_validation_forest_binding_invalid")
        if forest_before is not None and receipt_forest_before != [
            dict(item) for item in forest_before
        ]:
            reasons.append("governed_validation_forest_before_mismatch")
        if forest_after is not None and receipt_forest_after != [
            dict(item) for item in forest_after
        ]:
            reasons.append("governed_validation_forest_after_mismatch")
        if value.get("admitted") is True and (
            not receipt_forest_before
            or receipt_forest_before != receipt_forest_after
        ):
            reasons.append("governed_validation_forest_not_stable")
        unsigned = dict(value)
        receipt_cid = unsigned.pop("receipt_cid", None)
        if receipt_cid != content_identity(unsigned):
            reasons.append("governed_validation_receipt_cid_mismatch")
        commands = [dict(item) for item in specs if isinstance(item, Mapping)]
        rebuilt = build_governed_validation_receipt(
            phase=str(value.get("phase") or ""),
            task_id=str(value.get("task_id") or ""),
            task_cid=str(value.get("task_cid") or ""),
            task_contract_cid=str(value.get("task_contract_cid") or ""),
            execution_plan_cid=str(value.get("execution_plan_cid") or ""),
            target_commit=str(value.get("target_commit") or ""),
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            repository_id=str(value.get("repository_id") or ""),
            commands=commands,
            validation_result=result,
            policy=str(value.get("policy") or ""),
            expected_baseline_failure=(
                value.get("expected_baseline_failure")
                if isinstance(value.get("expected_baseline_failure"), Mapping)
                else None
            ),
        )
        if dict(value) != rebuilt:
            reasons.append("governed_validation_derived_fields_mismatch")
    except (ContractValidationError, TypeError, ValueError):
        reasons.append("governed_validation_receipt_not_canonical")
    if value.get("admitted") is True and (
        value.get("stale") is not False or value.get("unavailable") is not False
    ):
        reasons.append("governed_validation_invalid_admission")
    return not reasons, tuple(dict.fromkeys(reasons))


def build_supervisor_task_receipt(
    *,
    task_contract: Mapping[str, Any],
    task_contract_cid: str,
    completion_generation: str,
    baseline_commit: str,
    baseline_tree_id: str,
    candidate_commit: str,
    candidate_tree_id: str,
    integration_commit: str,
    integration_tree_id: str,
    effect_identities: Sequence[Mapping[str, Any]],
    dependency_evidence: Sequence[Mapping[str, Any]],
    pre_change_receipt: Mapping[str, Any] | None,
    post_change_receipt: Mapping[str, Any],
    acceptance_receipt: Mapping[str, Any] | None,
    provider_evidence: Mapping[str, Any],
    completion_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the trusted output without a self-referential containing commit."""

    exact_contract = dict(task_contract)
    exact_contract_cid = content_identity(exact_contract)
    if str(task_contract_cid or "") != exact_contract_cid:
        raise ValueError("supervisor receipt task contract CID is invalid")
    contract = bounded_receipt_projection(exact_contract)
    if not isinstance(contract, dict):
        raise ContractValidationError("task contract must be an object")
    partition = contract.get("authority_partition")
    if not isinstance(partition, Mapping):
        raise ValueError("task contract authority partition is missing")
    payload: dict[str, Any] = {
        "schema": SUPERVISOR_TASK_RECEIPT_SCHEMA,
        "completion_generation": str(completion_generation or ""),
        "task_id": str(contract.get("task_id") or ""),
        "task_cid": str(contract.get("canonical_task_cid") or ""),
        "task_contract_cid": exact_contract_cid,
        "baseline_commit": str(baseline_commit or ""),
        "baseline_tree_id": str(baseline_tree_id or ""),
        "candidate_commit": str(candidate_commit or ""),
        "candidate_tree_id": str(candidate_tree_id or ""),
        "integration_commit": str(integration_commit or ""),
        "integration_tree_id": str(integration_tree_id or ""),
        "provider_effects": list(partition.get("provider_effects") or []),
        "supervisor_outputs": list(partition.get("supervisor_outputs") or []),
        "effect_identities": bounded_receipt_projection(list(effect_identities)),
        "dependency_evidence": bounded_receipt_projection(list(dependency_evidence)),
        "pre_change_receipt_cid": str(
            (pre_change_receipt or {}).get("receipt_cid") or ""
        ),
        "post_change_receipt_cid": str(
            post_change_receipt.get("validation_receipt_id")
            or post_change_receipt.get("receipt_cid")
            or ""
        ),
        "acceptance_receipt_cid": str(
            (acceptance_receipt or {}).get("receipt_cid") or ""
        ),
        "provider_evidence": bounded_receipt_projection(dict(provider_evidence)),
        "completion_gate": bounded_receipt_projection(dict(completion_gate)),
        "completion_authority": "supervisor_post_validation_gate",
        "completion_authoritative": bool(
            completion_gate.get("completion_authoritative")
            or completion_gate.get("admitted")
        ),
    }
    required = (
        payload["completion_generation"],
        payload["task_id"],
        payload["task_cid"],
        payload["task_contract_cid"],
        payload["baseline_commit"],
        payload["baseline_tree_id"],
        payload["candidate_commit"],
        payload["candidate_tree_id"],
        payload["integration_commit"],
        payload["integration_tree_id"],
        payload["post_change_receipt_cid"],
    )
    if not all(required) or not payload["completion_authoritative"]:
        raise ValueError("supervisor task receipt authority binding is incomplete")
    payload["receipt_cid"] = content_identity(payload)
    return payload


def verify_supervisor_task_receipt(
    value: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    task_cid: str = "",
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return False, ("supervisor_task_receipt_missing",)
    reasons: list[str] = []
    if value.get("schema") != SUPERVISOR_TASK_RECEIPT_SCHEMA:
        reasons.append("supervisor_task_receipt_schema_invalid")
    if task_id and value.get("task_id") != task_id:
        reasons.append("supervisor_task_receipt_task_mismatch")
    if task_cid and value.get("task_cid") != task_cid:
        reasons.append("supervisor_task_receipt_task_cid_mismatch")
    for field in (
        "completion_generation",
        "task_id",
        "task_cid",
        "task_contract_cid",
        "baseline_commit",
        "baseline_tree_id",
        "candidate_commit",
        "candidate_tree_id",
        "integration_commit",
        "integration_tree_id",
        "post_change_receipt_cid",
        "receipt_cid",
    ):
        if not isinstance(value.get(field), str) or not value.get(field):
            reasons.append(f"supervisor_task_receipt_{field}_invalid")
    if value.get("completion_authoritative") is not True:
        reasons.append("supervisor_task_receipt_not_authoritative")
    try:
        unsigned = dict(value)
        receipt_cid = unsigned.pop("receipt_cid", None)
        if receipt_cid != content_identity(unsigned):
            reasons.append("supervisor_task_receipt_cid_mismatch")
    except (ContractValidationError, TypeError, ValueError):
        reasons.append("supervisor_task_receipt_not_canonical")
    return not reasons, tuple(dict.fromkeys(reasons))


__all__ = [
    "GOVERNED_COMMAND_ENVIRONMENT_KEYS",
    "GOVERNED_EXECUTION_PLAN_SCHEMA",
    "GOVERNED_PHASE_COMMAND_SET_SCHEMA",
    "GOVERNED_REPOSITORY_ROOTS",
    "GOVERNED_VALIDATION_RECEIPT_SCHEMA",
    "EXECUTOR_KINDS",
    "PRE_CHANGE_POLICIES",
    "SUPERVISOR_TASK_RECEIPT_SCHEMA",
    "TaskAuthorityPartition",
    "GovernedPhaseCommand",
    "bounded_receipt_projection",
    "build_governed_validation_receipt",
    "build_governed_execution_plan",
    "build_supervisor_task_receipt",
    "task_authority_partition",
    "task_expected_baseline_failure",
    "task_executor_kind",
    "task_dependency_completion_receipts",
    "task_phase_commands",
    "task_pre_change_policy",
    "validation_command_specs",
    "verify_governed_validation_receipt",
    "verify_governed_execution_plan",
    "verify_supervisor_task_receipt",
]
