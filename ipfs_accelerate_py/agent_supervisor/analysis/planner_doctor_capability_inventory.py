"""Provider-free capability inventory for the proof-directed Planner/Doctor.

The inventory is an observation boundary, not an authority boundary.  It
answers four deliberately separate questions:

* which interfaces and artifacts are present at an explicitly selected
  audited Git revision;
* whether the normal/default construction path actually wires them;
* which task, goal, and configuration state is present in the live checkout;
* which optional tools a caller-supplied *metadata-only* probe reports.

No target module is imported or executed.  Source files are inspected with
Python's AST or bounded byte searches and only body-free digests and compact
observations are retained.  Optional probes are injected, must attest that
they performed no network or process work, and remain non-authoritative.

The canonical identity helpers below intentionally match
``proof.formal_verification_contracts`` (CIDv1, DAG-JSON, sha2-256) without
importing that package.  This keeps this first-wave inventory independently
cold-importable while preserving the repository-wide wire identity.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

PLANNER_DOCTOR_CAPABILITY_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-capability-inventory@1"
)
PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-repository-revision@1"
)
PLANNER_DOCTOR_GITLINK_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-gitlink-binding@1"
)
PLANNER_DOCTOR_CAPABILITY_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-capability-record@1"
)
PLANNER_DOCTOR_CAPABILITY_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-capability-evidence@1"
)
PLANNER_DOCTOR_ARTIFACT_STATUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-artifact-status@1"
)
PLANNER_DOCTOR_CONTROL_STATUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-control-status@1"
)
PLANNER_DOCTOR_CONFIGURATION_STATUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-configuration-status@1"
)
PLANNER_DOCTOR_TOOL_HEALTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "planner-doctor-tool-health@1"
)

PDR_AUDITED_BASELINE_COMMIT: Final[str] = (
    "f25e5719cb738a50fb96bac4bea3f66ebca9800b"
)
DEFAULT_TASKBOARD_PATH: Final[str] = (
    "docs/architecture/"
    "agent_supervisor_proof_directed_planner_doctor.todo.md"
)
DEFAULT_OBJECTIVES_PATH: Final[str] = (
    "docs/architecture/"
    "agent_supervisor_proof_directed_planner_doctor.objectives.md"
)
DEFAULT_SCHEDULER_PATH: Final[str] = (
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)

INVENTORY_IS_PROOF_EVIDENCE: Final[bool] = False
INVENTORY_IS_COMPLETION_EVIDENCE: Final[bool] = False
INVENTORY_AUTHORIZES_MUTATION: Final[bool] = False
PACKAGE_PRESENCE_IS_CAPABILITY: Final[bool] = False

_MAX_TEXT_CHARS: Final[int] = 4 * 1024 * 1024
_MAX_RECORDS: Final[int] = 16_384
_MAX_GITLINK_DEPTH: Final[int] = 8
_GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}$")
_CID = re.compile(r"^b[a-z2-7]{20,}$")
_REASON = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,255}$")
_TASK_HEADING = re.compile(r"^## (?P<id>PDR-\d{3})\b")
_GOAL_HEADING = re.compile(r"^## (?P<id>PDR-G\d{3})\b")
_STATUS_LINE = re.compile(r"^- Status:\s*(?P<status>[A-Za-z0-9_-]+)\s*$")

_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
    }
)

DEFAULT_OPTIONAL_TOOL_IDS: Final[tuple[str, ...]] = (
    "cvc5",
    "ipfs_datasets_py.knowledge_graphs",
    "ipfs_datasets_py.program_analysis",
    "ipfs_datasets_py.vector_index",
    "lean",
    "program_analysis_zkp",
    "z3",
)

_AUDITED_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/__init__.py",
    "ipfs_accelerate_py/hf_space_inference.py",
    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py",
    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_plan_admission.py",
    "ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
    "ipfs_accelerate_py/agent_supervisor/control/control_cli.py",
    "ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_service.py",
    "ipfs_accelerate_py/agent_supervisor/analysis/doctor_repository_diagnostics.py",
    "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py",
    "ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transaction.py",
    "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_fixed_point.py",
    "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_benchmark.py",
    "ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_v2_benchmark.py",
)

_AUDITED_TEST_PATHS: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_prompt_workflow.py",
    "test/api/test_agent_supervisor_prompt_plan_admission.py",
    "test/api/test_agent_supervisor_deterministic_doctor_service.py",
    "test/api/test_agent_supervisor_deterministic_doctor_transaction.py",
    "test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py",
    "test/api/test_agent_supervisor_deterministic_doctor_benchmark.py",
    "test/api/test_agent_supervisor_supervisor_v2_benchmark.py",
    "test/api/test_agent_supervisor_doctor_cold_import.py",
)

_FUTURE_INTERFACE_PATHS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/analysis/doctor_contract_adapters.py",
    "ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py",
    "ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py",
    "ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py",
    "ipfs_accelerate_py/agent_supervisor/validation/"
    "deterministic_doctor_live_fixed_point.py",
    "ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_live_benchmark.py",
)

_CONFIG_SETTING_PATHS: Final[tuple[str, ...]] = (
    "schema",
    "doctor.default_mode",
    "doctor.enabled_at_bootstrap",
    "doctor.mutation_authorized",
    "doctor.allow_llm",
    "doctor.allow_network",
    "planner.default_mode",
    "derived_refill.enabled_at_bootstrap",
    "derived_refill.enabled_after_task",
    "benchmark.live_evidence_required",
    "benchmark.synthetic_evidence_may_promote",
    "benchmark.skipped_checks_may_promote",
    "benchmark.concurrency_sweep",
    "rollout.initial_mode",
    "rollout.automatic_enabled",
)


class PlannerDoctorInventoryError(ValueError):
    """Base class for stable inventory failures."""


class PlannerDoctorInventoryValidationError(PlannerDoctorInventoryError):
    """A contract input is malformed or outside the bounded schema."""


class PlannerDoctorInventoryIntegrityError(PlannerDoctorInventoryError):
    """A claimed content identity or live observation does not replay."""


class PlannerDoctorInventoryGitError(PlannerDoctorInventoryError):
    """An exact Git identity could not be observed."""


class CapabilityAvailability(str, Enum):
    """Whether implementation material exists at the audited revision."""

    SHIPPED = "shipped"
    PARTIAL = "partial"
    MISSING = "missing"
    UNKNOWN = "unknown"


class DefaultWiringState(str, Enum):
    """Whether the ordinary construction path activates a capability."""

    WIRED = "wired"
    UNWIRED = "unwired"
    DISABLED = "disabled"
    SYNTHETIC_ONLY = "synthetic_only"
    INCOMPATIBLE = "incompatible"
    UNSAFE_STUB = "unsafe_stub"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ArtifactKind(str, Enum):
    SOURCE = "source"
    TEST = "test"
    CONFIG = "config"
    TASKBOARD = "taskboard"
    OBJECTIVES = "objectives"


class ArtifactState(str, Enum):
    PRESENT = "present"
    MISSING = "missing"


class ToolHealthState(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_PROBED = "not_probed"
    UNKNOWN = "unknown"


class MetadataToolProbe(Protocol):
    """Injected provider-free discovery boundary."""

    def probe(self) -> Mapping[str, Any] | ToolHealthObservation:
        """Return one metadata-only health observation."""


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise PlannerDoctorInventoryValidationError(
            "canonical inventory values cannot contain floats"
        )
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PlannerDoctorInventoryValidationError(
                "canonical inventory object keys must be strings"
            )
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise PlannerDoctorInventoryValidationError(
        f"unsupported canonical inventory value: {type(value).__name__}"
    )


def canonical_inventory_json_bytes(value: Any) -> bytes:
    """Return deterministic DAG-JSON-compatible bytes."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _base32_cid(prefix: bytes, digest: bytes) -> str:
    return "b" + base64.b32encode(prefix + digest).decode("ascii").rstrip("=").lower()


def inventory_content_id(value: Any) -> str:
    """Return CIDv1 DAG-JSON/sha2-256, matching formal-verification contracts."""

    digest = hashlib.sha256(canonical_inventory_json_bytes(value)).digest()
    return _base32_cid(b"\x01\xa9\x02\x12\x20", digest)


def raw_blob_content_id(value: bytes) -> str:
    """Return CIDv1 raw/sha2-256 without retaining file bodies."""

    if not isinstance(value, bytes):
        raise PlannerDoctorInventoryValidationError("raw blob value must be bytes")
    return _base32_cid(b"\x01\x55\x12\x20", hashlib.sha256(value).digest())


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlannerDoctorInventoryValidationError(f"{field_name} must be an object")
    return value


def _strict_fields(
    value: Mapping[str, Any],
    allowed: Iterable[str],
    field_name: str,
) -> None:
    if set(value).difference(allowed):
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} contains unsupported fields"
        )


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    maximum: int = 512,
) -> str:
    if not isinstance(value, str):
        raise PlannerDoctorInventoryValidationError(f"{field_name} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise PlannerDoctorInventoryValidationError(f"{field_name} is required")
    if len(normalized) > maximum:
        raise PlannerDoctorInventoryValidationError(f"{field_name} is too long")
    return normalized


def _identifier(value: Any, field_name: str) -> str:
    normalized = _text(value, field_name, maximum=256)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a stable identifier"
        )
    return normalized


def _reason(value: Any, field_name: str = "reason_code") -> str:
    normalized = _text(value, field_name, maximum=256).lower()
    if _REASON.fullmatch(normalized) is None:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a stable reason code"
        )
    return normalized


def _reasons(values: Any, field_name: str = "reason_codes") -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw: Sequence[Any] = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray)
    ):
        raw = values
    else:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a sequence"
        )
    result = tuple(sorted({_reason(item, field_name) for item in raw}))
    if len(result) > _MAX_RECORDS:
        raise PlannerDoctorInventoryValidationError(f"{field_name} is too large")
    return result


def _strings(
    values: Any,
    field_name: str,
    *,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw: Sequence[Any] = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray)
    ):
        raw = values
    else:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a sequence"
        )
    normalizer = _identifier if identifiers else _text
    result = tuple(sorted({normalizer(item, field_name) for item in raw}))
    if len(result) > _MAX_RECORDS:
        raise PlannerDoctorInventoryValidationError(f"{field_name} is too large")
    return result


def _relative_path(value: Any, field_name: str = "path") -> str:
    normalized = _text(value, field_name, maximum=1024).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith("./") or ".." in path.parts:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a normalized repository-relative path"
        )
    canonical = path.as_posix()
    if canonical in {"", "."} or canonical != normalized:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a normalized repository-relative path"
        )
    return canonical


def _git_oid(value: Any, field_name: str, *, required: bool = True) -> str:
    normalized = _text(value, field_name, required=required, maximum=64)
    if normalized and _GIT_OID.fullmatch(normalized) is None:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a Git object id"
        )
    return normalized


def _cid(value: Any, field_name: str, *, required: bool = True) -> str:
    normalized = _text(value, field_name, required=required, maximum=256)
    if normalized and _CID.fullmatch(normalized) is None:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} must be a canonical CID"
        )
    return normalized


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PlannerDoctorInventoryValidationError(f"{field_name} must be boolean")
    return value


def _integer(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_RECORDS,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerDoctorInventoryValidationError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} is outside the supported range"
        )
    return value


def _enum(value: Any, kind: type[Enum], field_name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(value))
    except (TypeError, ValueError) as exc:
        raise PlannerDoctorInventoryValidationError(
            f"{field_name} has an unsupported value"
        ) from exc


def _reject_private_keys(value: Mapping[str, Any], field_name: str) -> None:
    for raw_key, item in value.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if any(
            key == marker
            or key.endswith("_" + marker)
            or marker in key
            for marker in _PRIVATE_FIELD_MARKERS
        ):
            raise PlannerDoctorInventoryValidationError(
                f"{field_name} contains private material"
            )
        if isinstance(item, Mapping):
            _reject_private_keys(item, field_name)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                if isinstance(nested, Mapping):
                    _reject_private_keys(nested, field_name)


@dataclass(frozen=True)
class GitlinkBinding:
    """One known Gitlink location, including opaque recursive frontiers."""

    path: str
    commit: str
    tree: str = ""
    parent_path: str = ""
    depth: int = 0
    resolved: bool = False
    reason_codes: tuple[str, ...] = ()
    schema: str = PLANNER_DOCTOR_GITLINK_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(self, "commit", _git_oid(self.commit, "commit"))
        object.__setattr__(
            self, "tree", _git_oid(self.tree, "tree", required=False)
        )
        parent = (
            _relative_path(self.parent_path, "parent_path")
            if self.parent_path
            else ""
        )
        object.__setattr__(self, "parent_path", parent)
        object.__setattr__(
            self,
            "depth",
            _integer(self.depth, "depth", maximum=_MAX_GITLINK_DEPTH),
        )
        object.__setattr__(self, "resolved", _boolean(self.resolved, "resolved"))
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        if self.depth == 0 and parent:
            raise PlannerDoctorInventoryValidationError(
                "top-level gitlink cannot have parent_path"
            )
        if self.depth > 0 and not parent:
            raise PlannerDoctorInventoryValidationError(
                "nested gitlink requires parent_path"
            )
        if self.resolved and not self.tree:
            raise PlannerDoctorInventoryValidationError(
                "resolved gitlink requires tree"
            )
        if self.schema != PLANNER_DOCTOR_GITLINK_BINDING_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported gitlink binding schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "commit": self.commit,
            "tree": self.tree,
            "parent_path": self.parent_path,
            "depth": self.depth,
            "resolved": self.resolved,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GitlinkBinding:
        payload = _mapping(value, "gitlink")
        _strict_fields(
            payload,
            {
                "schema",
                "path",
                "commit",
                "tree",
                "parent_path",
                "depth",
                "resolved",
                "reason_codes",
            },
            "gitlink",
        )
        return cls(
            schema=payload.get("schema", PLANNER_DOCTOR_GITLINK_BINDING_SCHEMA),
            path=payload.get("path", ""),
            commit=payload.get("commit", ""),
            tree=payload.get("tree", ""),
            parent_path=payload.get("parent_path", ""),
            depth=payload.get("depth", 0),
            resolved=payload.get("resolved", False),
            reason_codes=tuple(payload.get("reason_codes", ())),
        )


@dataclass(frozen=True)
class RepositoryRevision:
    """Exact commit/tree/overlay/Gitlink observation for one repository state."""

    label: str
    commit: str
    tree: str
    dirty: bool
    dirty_overlay_cid: str
    status_cid: str
    gitlinks: tuple[GitlinkBinding, ...] = ()
    gitlink_closure_complete: bool = True
    reason_codes: tuple[str, ...] = ()
    schema: str = PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label", maximum=256))
        object.__setattr__(self, "commit", _git_oid(self.commit, "commit"))
        object.__setattr__(self, "tree", _git_oid(self.tree, "tree"))
        object.__setattr__(self, "dirty", _boolean(self.dirty, "dirty"))
        object.__setattr__(
            self,
            "dirty_overlay_cid",
            _cid(self.dirty_overlay_cid, "dirty_overlay_cid"),
        )
        object.__setattr__(self, "status_cid", _cid(self.status_cid, "status_cid"))
        links = tuple(
            item if isinstance(item, GitlinkBinding) else GitlinkBinding.from_dict(item)
            for item in self.gitlinks
        )
        links = tuple(sorted(links, key=lambda item: (item.depth, item.path)))
        if len({item.path for item in links}) != len(links):
            raise PlannerDoctorInventoryValidationError("duplicate gitlink path")
        object.__setattr__(self, "gitlinks", links)
        object.__setattr__(
            self,
            "gitlink_closure_complete",
            _boolean(self.gitlink_closure_complete, "gitlink_closure_complete"),
        )
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        if self.schema != PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported repository revision schema"
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "label": self.label,
            "commit": self.commit,
            "tree": self.tree,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "status_cid": self.status_cid,
            "gitlinks": [item.to_dict() for item in self.gitlinks],
            "gitlink_closure_complete": self.gitlink_closure_complete,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def revision_id(self) -> str:
        return inventory_content_id(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "revision_id": self.revision_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RepositoryRevision:
        payload = _mapping(value, "repository_revision")
        _strict_fields(
            payload,
            {
                "schema",
                "label",
                "commit",
                "tree",
                "dirty",
                "dirty_overlay_cid",
                "status_cid",
                "gitlinks",
                "gitlink_closure_complete",
                "reason_codes",
                "revision_id",
            },
            "repository_revision",
        )
        result = cls(
            schema=payload.get(
                "schema", PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA
            ),
            label=payload.get("label", ""),
            commit=payload.get("commit", ""),
            tree=payload.get("tree", ""),
            dirty=payload.get("dirty", False),
            dirty_overlay_cid=payload.get("dirty_overlay_cid", ""),
            status_cid=payload.get("status_cid", ""),
            gitlinks=tuple(
                GitlinkBinding.from_dict(item)
                for item in payload.get("gitlinks", ())
            ),
            gitlink_closure_complete=payload.get(
                "gitlink_closure_complete", True
            ),
            reason_codes=tuple(payload.get("reason_codes", ())),
        )
        claimed = payload.get("revision_id")
        if claimed is not None and _cid(claimed, "revision_id") != result.revision_id:
            raise PlannerDoctorInventoryIntegrityError(
                "repository revision identity mismatch"
            )
        return result


@dataclass(frozen=True, order=True)
class StatusEntry:
    item_id: str
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _identifier(self.item_id, "item_id"))
        object.__setattr__(
            self, "status", _identifier(self.status.lower(), "status")
        )

    def to_dict(self) -> dict[str, str]:
        return {"item_id": self.item_id, "status": self.status}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StatusEntry:
        payload = _mapping(value, "status_entry")
        _strict_fields(payload, {"item_id", "status"}, "status_entry")
        return cls(item_id=payload.get("item_id", ""), status=payload.get("status", ""))


@dataclass(frozen=True)
class ControlStatusSnapshot:
    """Content-addressed PDR task and goal status projection."""

    taskboard_path: str
    taskboard_blob_cid: str
    objective_path: str
    objective_blob_cid: str
    tasks: tuple[StatusEntry, ...]
    goals: tuple[StatusEntry, ...]
    schema: str = PLANNER_DOCTOR_CONTROL_STATUS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "taskboard_path", _relative_path(self.taskboard_path)
        )
        object.__setattr__(
            self,
            "taskboard_blob_cid",
            _cid(self.taskboard_blob_cid, "taskboard_blob_cid"),
        )
        object.__setattr__(
            self, "objective_path", _relative_path(self.objective_path)
        )
        object.__setattr__(
            self,
            "objective_blob_cid",
            _cid(self.objective_blob_cid, "objective_blob_cid"),
        )
        tasks = tuple(
            item if isinstance(item, StatusEntry) else StatusEntry.from_dict(item)
            for item in self.tasks
        )
        goals = tuple(
            item if isinstance(item, StatusEntry) else StatusEntry.from_dict(item)
            for item in self.goals
        )
        tasks = tuple(sorted(tasks))
        goals = tuple(sorted(goals))
        if not tasks or len({item.item_id for item in tasks}) != len(tasks):
            raise PlannerDoctorInventoryValidationError(
                "task status projection is empty or contains duplicates"
            )
        if not goals or len({item.item_id for item in goals}) != len(goals):
            raise PlannerDoctorInventoryValidationError(
                "goal status projection is empty or contains duplicates"
            )
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "goals", goals)
        if self.schema != PLANNER_DOCTOR_CONTROL_STATUS_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported control status schema"
            )

    @property
    def snapshot_id(self) -> str:
        return inventory_content_id(self._identity_payload())

    @property
    def completed_task_count(self) -> int:
        return sum(item.status == "completed" for item in self.tasks)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "taskboard_path": self.taskboard_path,
            "taskboard_blob_cid": self.taskboard_blob_cid,
            "objective_path": self.objective_path,
            "objective_blob_cid": self.objective_blob_cid,
            "tasks": [item.to_dict() for item in self.tasks],
            "goals": [item.to_dict() for item in self.goals],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "snapshot_id": self.snapshot_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ControlStatusSnapshot:
        payload = _mapping(value, "control_status")
        _strict_fields(
            payload,
            {
                "schema",
                "taskboard_path",
                "taskboard_blob_cid",
                "objective_path",
                "objective_blob_cid",
                "tasks",
                "goals",
                "snapshot_id",
            },
            "control_status",
        )
        result = cls(
            schema=payload.get("schema", PLANNER_DOCTOR_CONTROL_STATUS_SCHEMA),
            taskboard_path=payload.get("taskboard_path", ""),
            taskboard_blob_cid=payload.get("taskboard_blob_cid", ""),
            objective_path=payload.get("objective_path", ""),
            objective_blob_cid=payload.get("objective_blob_cid", ""),
            tasks=tuple(
                StatusEntry.from_dict(item) for item in payload.get("tasks", ())
            ),
            goals=tuple(
                StatusEntry.from_dict(item) for item in payload.get("goals", ())
            ),
        )
        claimed = payload.get("snapshot_id")
        if claimed is not None and _cid(claimed, "snapshot_id") != result.snapshot_id:
            raise PlannerDoctorInventoryIntegrityError(
                "control status identity mismatch"
            )
        return result


@dataclass(frozen=True, order=True)
class ConfigurationSetting:
    path: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _identifier(self.path, "setting.path"))
        canonical = _canonical_value(self.value)
        if isinstance(canonical, dict):
            _reject_private_keys(canonical, "setting.value")
        object.__setattr__(self, "value", canonical)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConfigurationSetting:
        payload = _mapping(value, "configuration_setting")
        _strict_fields(payload, {"path", "value"}, "configuration_setting")
        return cls(path=payload.get("path", ""), value=payload.get("value"))


@dataclass(frozen=True)
class ConfigurationStatus:
    path: str
    present: bool
    blob_cid: str = ""
    config_schema: str = ""
    settings: tuple[ConfigurationSetting, ...] = ()
    reason_codes: tuple[str, ...] = ()
    schema: str = PLANNER_DOCTOR_CONFIGURATION_STATUS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(self, "present", _boolean(self.present, "present"))
        object.__setattr__(
            self, "blob_cid", _cid(self.blob_cid, "blob_cid", required=False)
        )
        object.__setattr__(
            self,
            "config_schema",
            _text(
                self.config_schema,
                "config_schema",
                required=False,
                maximum=512,
            ),
        )
        settings = tuple(
            item
            if isinstance(item, ConfigurationSetting)
            else ConfigurationSetting.from_dict(item)
            for item in self.settings
        )
        settings = tuple(sorted(settings))
        if len({item.path for item in settings}) != len(settings):
            raise PlannerDoctorInventoryValidationError(
                "configuration settings contain duplicate paths"
            )
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        if self.present and not self.blob_cid:
            raise PlannerDoctorInventoryValidationError(
                "present configuration requires blob_cid"
            )
        if not self.present and (self.blob_cid or self.settings):
            raise PlannerDoctorInventoryValidationError(
                "missing configuration cannot contain material"
            )
        if self.schema != PLANNER_DOCTOR_CONFIGURATION_STATUS_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported configuration status schema"
            )

    def setting(self, path: str, default: Any = None) -> Any:
        for item in self.settings:
            if item.path == path:
                return item.value
        return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "present": self.present,
            "blob_cid": self.blob_cid,
            "config_schema": self.config_schema,
            "settings": [item.to_dict() for item in self.settings],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConfigurationStatus:
        payload = _mapping(value, "configuration_status")
        _strict_fields(
            payload,
            {
                "schema",
                "path",
                "present",
                "blob_cid",
                "config_schema",
                "settings",
                "reason_codes",
            },
            "configuration_status",
        )
        return cls(
            schema=payload.get(
                "schema", PLANNER_DOCTOR_CONFIGURATION_STATUS_SCHEMA
            ),
            path=payload.get("path", ""),
            present=payload.get("present", False),
            blob_cid=payload.get("blob_cid", ""),
            config_schema=payload.get("config_schema", ""),
            settings=tuple(
                ConfigurationSetting.from_dict(item)
                for item in payload.get("settings", ())
            ),
            reason_codes=tuple(payload.get("reason_codes", ())),
        )


@dataclass(frozen=True)
class ArtifactStatus:
    path: str
    kind: ArtifactKind
    state: ArtifactState
    source: str
    blob_cid: str = ""
    reason_codes: tuple[str, ...] = ()
    schema: str = PLANNER_DOCTOR_ARTIFACT_STATUS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(self, "kind", _enum(self.kind, ArtifactKind, "kind"))
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        object.__setattr__(
            self, "blob_cid", _cid(self.blob_cid, "blob_cid", required=False)
        )
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        if self.state is ArtifactState.PRESENT and not self.blob_cid:
            raise PlannerDoctorInventoryValidationError(
                "present artifact requires blob_cid"
            )
        if self.state is ArtifactState.MISSING and self.blob_cid:
            raise PlannerDoctorInventoryValidationError(
                "missing artifact cannot contain blob_cid"
            )
        if self.schema != PLANNER_DOCTOR_ARTIFACT_STATUS_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported artifact status schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "kind": self.kind.value,
            "state": self.state.value,
            "source": self.source,
            "blob_cid": self.blob_cid,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactStatus:
        payload = _mapping(value, "artifact_status")
        _strict_fields(
            payload,
            {
                "schema",
                "path",
                "kind",
                "state",
                "source",
                "blob_cid",
                "reason_codes",
            },
            "artifact_status",
        )
        return cls(
            schema=payload.get("schema", PLANNER_DOCTOR_ARTIFACT_STATUS_SCHEMA),
            path=payload.get("path", ""),
            kind=payload.get("kind", ""),
            state=payload.get("state", ""),
            source=payload.get("source", ""),
            blob_cid=payload.get("blob_cid", ""),
            reason_codes=tuple(payload.get("reason_codes", ())),
        )


@dataclass(frozen=True)
class CapabilityEvidence:
    path: str
    observation: str
    symbol: str = ""
    blob_cid: str = ""
    schema: str = PLANNER_DOCTOR_CAPABILITY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(
            self,
            "observation",
            _identifier(self.observation, "observation"),
        )
        object.__setattr__(
            self,
            "symbol",
            _text(self.symbol, "symbol", required=False, maximum=256),
        )
        object.__setattr__(
            self, "blob_cid", _cid(self.blob_cid, "blob_cid", required=False)
        )
        if self.schema != PLANNER_DOCTOR_CAPABILITY_EVIDENCE_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported capability evidence schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "observation": self.observation,
            "symbol": self.symbol,
            "blob_cid": self.blob_cid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityEvidence:
        payload = _mapping(value, "capability_evidence")
        _strict_fields(
            payload,
            {"schema", "path", "observation", "symbol", "blob_cid"},
            "capability_evidence",
        )
        return cls(
            schema=payload.get(
                "schema", PLANNER_DOCTOR_CAPABILITY_EVIDENCE_SCHEMA
            ),
            path=payload.get("path", ""),
            observation=payload.get("observation", ""),
            symbol=payload.get("symbol", ""),
            blob_cid=payload.get("blob_cid", ""),
        )


@dataclass(frozen=True)
class CapabilityRecord:
    capability_id: str
    interface_ids: tuple[str, ...]
    availability: CapabilityAvailability
    default_wiring: DefaultWiringState
    evidence: tuple[CapabilityEvidence, ...]
    test_paths: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    optional: bool = False
    schema: str = PLANNER_DOCTOR_CAPABILITY_RECORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capability_id", _identifier(self.capability_id, "capability_id")
        )
        object.__setattr__(
            self,
            "interface_ids",
            _strings(self.interface_ids, "interface_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "availability",
            _enum(self.availability, CapabilityAvailability, "availability"),
        )
        object.__setattr__(
            self,
            "default_wiring",
            _enum(self.default_wiring, DefaultWiringState, "default_wiring"),
        )
        evidence = tuple(
            item
            if isinstance(item, CapabilityEvidence)
            else CapabilityEvidence.from_dict(item)
            for item in self.evidence
        )
        evidence = tuple(
            sorted(
                evidence,
                key=lambda item: (item.path, item.symbol, item.observation),
            )
        )
        if not evidence:
            raise PlannerDoctorInventoryValidationError(
                "capability record requires body-free evidence"
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "test_paths",
            tuple(sorted({_relative_path(item, "test_path") for item in self.test_paths})),
        )
        object.__setattr__(
            self,
            "config_paths",
            tuple(
                sorted({_relative_path(item, "config_path") for item in self.config_paths})
            ),
        )
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        object.__setattr__(self, "optional", _boolean(self.optional, "optional"))
        if self.schema != PLANNER_DOCTOR_CAPABILITY_RECORD_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported capability record schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capability_id": self.capability_id,
            "interface_ids": list(self.interface_ids),
            "availability": self.availability.value,
            "default_wiring": self.default_wiring.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "test_paths": list(self.test_paths),
            "config_paths": list(self.config_paths),
            "reason_codes": list(self.reason_codes),
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityRecord:
        payload = _mapping(value, "capability_record")
        _strict_fields(
            payload,
            {
                "schema",
                "capability_id",
                "interface_ids",
                "availability",
                "default_wiring",
                "evidence",
                "test_paths",
                "config_paths",
                "reason_codes",
                "optional",
            },
            "capability_record",
        )
        return cls(
            schema=payload.get("schema", PLANNER_DOCTOR_CAPABILITY_RECORD_SCHEMA),
            capability_id=payload.get("capability_id", ""),
            interface_ids=tuple(payload.get("interface_ids", ())),
            availability=payload.get("availability", ""),
            default_wiring=payload.get("default_wiring", ""),
            evidence=tuple(
                CapabilityEvidence.from_dict(item)
                for item in payload.get("evidence", ())
            ),
            test_paths=tuple(payload.get("test_paths", ())),
            config_paths=tuple(payload.get("config_paths", ())),
            reason_codes=tuple(payload.get("reason_codes", ())),
            optional=payload.get("optional", False),
        )


@dataclass(frozen=True)
class ToolHealthObservation:
    tool_id: str
    probe_id: str
    health: ToolHealthState
    capability_ids: tuple[str, ...] = ()
    version: str = ""
    reason_codes: tuple[str, ...] = ()
    metadata_only: bool = True
    network_used: bool = False
    process_started: bool = False
    schema: str = PLANNER_DOCTOR_TOOL_HEALTH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _identifier(self.tool_id, "tool_id"))
        object.__setattr__(self, "probe_id", _identifier(self.probe_id, "probe_id"))
        object.__setattr__(
            self, "health", _enum(self.health, ToolHealthState, "health")
        )
        object.__setattr__(
            self,
            "capability_ids",
            _strings(self.capability_ids, "capability_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, "version", required=False, maximum=256),
        )
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        object.__setattr__(
            self, "metadata_only", _boolean(self.metadata_only, "metadata_only")
        )
        object.__setattr__(
            self, "network_used", _boolean(self.network_used, "network_used")
        )
        object.__setattr__(
            self, "process_started", _boolean(self.process_started, "process_started")
        )
        if not self.metadata_only or self.network_used or self.process_started:
            raise PlannerDoctorInventoryValidationError(
                "tool health probes must be metadata-only and side-effect free"
            )
        if self.schema != PLANNER_DOCTOR_TOOL_HEALTH_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported tool health schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tool_id": self.tool_id,
            "probe_id": self.probe_id,
            "health": self.health.value,
            "capability_ids": list(self.capability_ids),
            "version": self.version,
            "reason_codes": list(self.reason_codes),
            "metadata_only": self.metadata_only,
            "network_used": self.network_used,
            "process_started": self.process_started,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolHealthObservation:
        payload = _mapping(value, "tool_health")
        _strict_fields(
            payload,
            {
                "schema",
                "tool_id",
                "probe_id",
                "health",
                "capability_ids",
                "version",
                "reason_codes",
                "metadata_only",
                "network_used",
                "process_started",
            },
            "tool_health",
        )
        return cls(
            schema=payload.get("schema", PLANNER_DOCTOR_TOOL_HEALTH_SCHEMA),
            tool_id=payload.get("tool_id", ""),
            probe_id=payload.get("probe_id", ""),
            health=payload.get("health", ""),
            capability_ids=tuple(payload.get("capability_ids", ())),
            version=payload.get("version", ""),
            reason_codes=tuple(payload.get("reason_codes", ())),
            metadata_only=payload.get("metadata_only", True),
            network_used=payload.get("network_used", False),
            process_started=payload.get("process_started", False),
        )


@dataclass(frozen=True)
class PlannerDoctorCapabilityInventory:
    """Immutable shipped-vs-wired inventory over audited and live roots."""

    audited_baseline: RepositoryRevision
    current_checkout: RepositoryRevision
    control_status: ControlStatusSnapshot
    capabilities: tuple[CapabilityRecord, ...]
    artifacts: tuple[ArtifactStatus, ...]
    configurations: tuple[ConfigurationStatus, ...]
    tool_health: tuple[ToolHealthObservation, ...]
    schema: str = PLANNER_DOCTOR_CAPABILITY_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        audited = (
            self.audited_baseline
            if isinstance(self.audited_baseline, RepositoryRevision)
            else RepositoryRevision.from_dict(self.audited_baseline)
        )
        current = (
            self.current_checkout
            if isinstance(self.current_checkout, RepositoryRevision)
            else RepositoryRevision.from_dict(self.current_checkout)
        )
        control = (
            self.control_status
            if isinstance(self.control_status, ControlStatusSnapshot)
            else ControlStatusSnapshot.from_dict(self.control_status)
        )
        capabilities = tuple(
            item
            if isinstance(item, CapabilityRecord)
            else CapabilityRecord.from_dict(item)
            for item in self.capabilities
        )
        artifacts = tuple(
            item
            if isinstance(item, ArtifactStatus)
            else ArtifactStatus.from_dict(item)
            for item in self.artifacts
        )
        configurations = tuple(
            item
            if isinstance(item, ConfigurationStatus)
            else ConfigurationStatus.from_dict(item)
            for item in self.configurations
        )
        tools = tuple(
            item
            if isinstance(item, ToolHealthObservation)
            else ToolHealthObservation.from_dict(item)
            for item in self.tool_health
        )
        capabilities = tuple(sorted(capabilities, key=lambda item: item.capability_id))
        artifacts = tuple(
            sorted(artifacts, key=lambda item: (item.source, item.kind.value, item.path))
        )
        configurations = tuple(sorted(configurations, key=lambda item: item.path))
        tools = tuple(sorted(tools, key=lambda item: (item.tool_id, item.probe_id)))
        if len({item.capability_id for item in capabilities}) != len(capabilities):
            raise PlannerDoctorInventoryValidationError("duplicate capability id")
        if len({(item.source, item.path) for item in artifacts}) != len(artifacts):
            raise PlannerDoctorInventoryValidationError("duplicate artifact observation")
        if len({item.path for item in configurations}) != len(configurations):
            raise PlannerDoctorInventoryValidationError(
                "duplicate configuration observation"
            )
        if len({item.tool_id for item in tools}) != len(tools):
            raise PlannerDoctorInventoryValidationError(
                "duplicate tool health observation"
            )
        object.__setattr__(self, "audited_baseline", audited)
        object.__setattr__(self, "current_checkout", current)
        object.__setattr__(self, "control_status", control)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "tool_health", tools)
        if self.schema != PLANNER_DOCTOR_CAPABILITY_INVENTORY_SCHEMA:
            raise PlannerDoctorInventoryValidationError(
                "unsupported planner/doctor inventory schema"
            )

    def capability(self, capability_id: str) -> CapabilityRecord:
        normalized = _identifier(capability_id, "capability_id")
        for item in self.capabilities:
            if item.capability_id == normalized:
                return item
        raise KeyError(normalized)

    @property
    def gap_capability_ids(self) -> tuple[str, ...]:
        return tuple(
            item.capability_id
            for item in self.capabilities
            if item.availability is not CapabilityAvailability.SHIPPED
            or item.default_wiring
            not in {DefaultWiringState.WIRED, DefaultWiringState.NOT_APPLICABLE}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority": {
                "inventory_is_proof_evidence": INVENTORY_IS_PROOF_EVIDENCE,
                "inventory_is_completion_evidence": (
                    INVENTORY_IS_COMPLETION_EVIDENCE
                ),
                "inventory_authorizes_mutation": INVENTORY_AUTHORIZES_MUTATION,
                "package_presence_is_capability": PACKAGE_PRESENCE_IS_CAPABILITY,
                "tool_probe_authority": "metadata_observation_only",
            },
            "audited_baseline": self.audited_baseline.to_dict(),
            "current_checkout": self.current_checkout.to_dict(),
            "control_status": self.control_status.to_dict(),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "configurations": [item.to_dict() for item in self.configurations],
            "tool_health": [item.to_dict() for item in self.tool_health],
            "gap_capability_ids": list(self.gap_capability_ids),
        }

    @property
    def content_id(self) -> str:
        return inventory_content_id(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    def to_json(self) -> str:
        return canonical_inventory_json_bytes(self.to_record()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlannerDoctorCapabilityInventory:
        payload = _mapping(value, "planner_doctor_capability_inventory")
        _strict_fields(
            payload,
            {
                "schema",
                "authority",
                "audited_baseline",
                "current_checkout",
                "control_status",
                "capabilities",
                "artifacts",
                "configurations",
                "tool_health",
                "gap_capability_ids",
                "content_id",
            },
            "planner_doctor_capability_inventory",
        )
        authority = _mapping(payload.get("authority", {}), "authority")
        expected_authority = {
            "inventory_is_proof_evidence": False,
            "inventory_is_completion_evidence": False,
            "inventory_authorizes_mutation": False,
            "package_presence_is_capability": False,
            "tool_probe_authority": "metadata_observation_only",
        }
        if dict(authority) != expected_authority:
            raise PlannerDoctorInventoryValidationError(
                "inventory authority boundary mismatch"
            )
        result = cls(
            schema=payload.get(
                "schema", PLANNER_DOCTOR_CAPABILITY_INVENTORY_SCHEMA
            ),
            audited_baseline=RepositoryRevision.from_dict(
                payload.get("audited_baseline", {})
            ),
            current_checkout=RepositoryRevision.from_dict(
                payload.get("current_checkout", {})
            ),
            control_status=ControlStatusSnapshot.from_dict(
                payload.get("control_status", {})
            ),
            capabilities=tuple(
                CapabilityRecord.from_dict(item)
                for item in payload.get("capabilities", ())
            ),
            artifacts=tuple(
                ArtifactStatus.from_dict(item)
                for item in payload.get("artifacts", ())
            ),
            configurations=tuple(
                ConfigurationStatus.from_dict(item)
                for item in payload.get("configurations", ())
            ),
            tool_health=tuple(
                ToolHealthObservation.from_dict(item)
                for item in payload.get("tool_health", ())
            ),
        )
        claimed_gaps = tuple(payload.get("gap_capability_ids", ()))
        if claimed_gaps != result.gap_capability_ids:
            raise PlannerDoctorInventoryIntegrityError(
                "gap capability projection mismatch"
            )
        claimed = payload.get("content_id")
        if claimed is not None and _cid(claimed, "content_id") != result.content_id:
            raise PlannerDoctorInventoryIntegrityError(
                "planner/doctor inventory content identity mismatch"
            )
        return result


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git(
    root: Path,
    *arguments: str,
    required: bool = True,
) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if required:
            raise PlannerDoctorInventoryGitError("git_command_unavailable") from exc
        return None
    if result.returncode != 0:
        if required:
            raise PlannerDoctorInventoryGitError("git_observation_failed")
        return None
    return bytes(result.stdout)


def _git_text(root: Path, *arguments: str, required: bool = True) -> str:
    value = _git(root, *arguments, required=required)
    if value is None:
        return ""
    return value.decode("utf-8", "surrogateescape").strip()


def _resolve_commit(root: Path, ref: str) -> str:
    candidate = _text(ref, "audited_ref", maximum=256)
    if candidate.startswith("-") or "\x00" in candidate or any(
        character.isspace() for character in candidate
    ):
        raise PlannerDoctorInventoryValidationError("audited_ref is unsafe")
    commit = _git_text(
        root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{candidate}^{{commit}}",
    )
    return _git_oid(commit, "audited_commit")


def _tree_for_commit(root: Path, commit: str) -> str:
    return _git_oid(
        _git_text(root, "rev-parse", "--verify", f"{commit}^{{tree}}"),
        "tree",
    )


def _gitlink_rows(root: Path, commit: str) -> tuple[tuple[str, str], ...]:
    raw = _git(root, "ls-tree", "-r", "-z", commit)
    assert raw is not None
    rows: list[tuple[str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            metadata, raw_path = item.split(b"\t", 1)
            mode, object_type, oid = metadata.decode("ascii").split(" ", 2)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PlannerDoctorInventoryGitError("git_tree_record_malformed") from exc
        if mode != "160000" or object_type != "commit":
            continue
        path = raw_path.decode("utf-8", "surrogateescape")
        rows.append((_relative_path(path, "gitlink.path"), _git_oid(oid, "gitlink.commit")))
    return tuple(sorted(rows))


def _is_exact_git_root(path: Path) -> bool:
    top = _git_text(path, "rev-parse", "--show-toplevel", required=False)
    if not top:
        return False
    try:
        return Path(top).resolve(strict=True) == path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _gitlink_closure(
    root: Path,
    commit: str,
    *,
    maximum_depth: int = _MAX_GITLINK_DEPTH,
) -> tuple[tuple[GitlinkBinding, ...], bool, tuple[str, ...]]:
    root = root.resolve(strict=True)
    records: list[GitlinkBinding] = []
    reasons: list[str] = []
    complete = True
    visited: set[tuple[str, str]] = set()

    def walk(
        repository: Path,
        parent_commit: str,
        *,
        prefix: str = "",
        parent_path: str = "",
        depth: int = 0,
    ) -> None:
        nonlocal complete
        if depth > maximum_depth:
            complete = False
            reasons.append("recursive_gitlink_depth_exceeded")
            return
        key = (str(repository.resolve(strict=False)), parent_commit)
        if key in visited:
            complete = False
            reasons.append("recursive_gitlink_cycle")
            return
        visited.add(key)
        try:
            rows = _gitlink_rows(repository, parent_commit)
        except PlannerDoctorInventoryError:
            complete = False
            reasons.append("recursive_gitlink_map_unavailable")
            return
        for relative, recorded_commit in rows:
            full_path = f"{prefix}/{relative}" if prefix else relative
            candidate = repository / relative
            child_reasons: list[str] = []
            child_tree = ""
            resolved = False
            try:
                resolved_candidate = candidate.resolve(strict=False)
                resolved_candidate.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                child_reasons.append("gitlink_checkout_outside_repository")
            else:
                if _is_exact_git_root(candidate):
                    child_tree = _git_text(
                        candidate,
                        "rev-parse",
                        "--verify",
                        f"{recorded_commit}^{{tree}}",
                        required=False,
                    )
                    if child_tree and _GIT_OID.fullmatch(child_tree):
                        resolved = True
                    else:
                        child_tree = ""
                        child_reasons.append("gitlink_commit_unavailable")
                else:
                    child_reasons.append("gitlink_checkout_unavailable")
            if not resolved:
                complete = False
                reasons.extend(child_reasons)
            record = GitlinkBinding(
                path=full_path,
                commit=recorded_commit,
                tree=child_tree,
                parent_path=parent_path,
                depth=depth,
                resolved=resolved,
                reason_codes=tuple(child_reasons),
            )
            records.append(record)
            if resolved:
                walk(
                    candidate,
                    recorded_commit,
                    prefix=full_path,
                    parent_path=full_path,
                    depth=depth + 1,
                )

    walk(root, commit)
    return (
        tuple(sorted(records, key=lambda item: (item.depth, item.path))),
        complete,
        tuple(sorted(set(reasons))),
    )


def _untracked_blob_records(root: Path, status: bytes) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        raw_path = entry[3:].decode("utf-8", "surrogateescape")
        path = _relative_path(raw_path, "untracked.path")
        candidate = root / path
        try:
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                body = os.readlink(candidate).encode("utf-8", "surrogateescape")
                kind = "symlink"
            elif stat.S_ISREG(mode):
                body = candidate.read_bytes()
                kind = "file"
            else:
                body = b""
                kind = "unsupported"
        except OSError:
            body = b""
            kind = "unavailable"
        records.append(
            {"path": path, "kind": kind, "blob_cid": raw_blob_content_id(body)}
        )
    return tuple(sorted(records, key=lambda item: item["path"]))


def _overlay_observation(root: Path) -> tuple[bool, str, str]:
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    staged = _git(
        root,
        "diff",
        "--cached",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
    )
    unstaged = _git(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
    )
    assert status is not None and staged is not None and unstaged is not None
    status_cid = raw_blob_content_id(status)
    payload = {
        "schema": PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA + "/dirty-overlay",
        "status_cid": status_cid,
        "staged_diff_cid": raw_blob_content_id(staged),
        "unstaged_diff_cid": raw_blob_content_id(unstaged),
        "untracked": list(_untracked_blob_records(root, status)),
    }
    return bool(status), inventory_content_id(payload), status_cid


def _observe_revision(
    root: Path,
    *,
    label: str,
    commit: str,
    include_dirty_overlay: bool,
) -> RepositoryRevision:
    tree = _tree_for_commit(root, commit)
    links, complete, reasons = _gitlink_closure(root, commit)
    if include_dirty_overlay:
        first_overlay = _overlay_observation(root)
        second_overlay = _overlay_observation(root)
        if first_overlay != second_overlay:
            raise PlannerDoctorInventoryIntegrityError(
                "repository changed during dirty-overlay observation"
            )
        dirty, overlay_cid, status_cid = first_overlay
    else:
        empty_status = raw_blob_content_id(b"")
        dirty = False
        status_cid = empty_status
        overlay_cid = inventory_content_id(
            {
                "schema": (
                    PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA + "/dirty-overlay"
                ),
                "status_cid": empty_status,
                "staged_diff_cid": empty_status,
                "unstaged_diff_cid": empty_status,
                "untracked": [],
            }
        )
    return RepositoryRevision(
        label=label,
        commit=commit,
        tree=tree,
        dirty=dirty,
        dirty_overlay_cid=overlay_cid,
        status_cid=status_cid,
        gitlinks=links,
        gitlink_closure_complete=complete,
        reason_codes=reasons,
    )


def _stable_live_bytes(root: Path, relative: str) -> bytes:
    path = root / _relative_path(relative)
    try:
        first_mode = path.lstat().st_mode
    except OSError as exc:
        raise PlannerDoctorInventoryValidationError(
            f"required control artifact is unavailable: {relative}"
        ) from exc
    if not stat.S_ISREG(first_mode):
        raise PlannerDoctorInventoryValidationError(
            f"required control artifact is not a regular file: {relative}"
        )
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise PlannerDoctorInventoryIntegrityError(
            "control artifact changed during observation"
        )
    return first


def _parse_statuses(data: bytes, *, goal: bool) -> tuple[StatusEntry, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlannerDoctorInventoryValidationError(
            "control artifact must be UTF-8"
        ) from exc
    heading = _GOAL_HEADING if goal else _TASK_HEADING
    current = ""
    records: list[StatusEntry] = []
    for line in text.splitlines():
        match = heading.match(line)
        if match is not None:
            current = match.group("id")
            continue
        if line.startswith("## "):
            current = ""
            continue
        status = _STATUS_LINE.match(line)
        if current and status is not None:
            records.append(StatusEntry(current, status.group("status").lower()))
            current = ""
    if not records:
        raise PlannerDoctorInventoryValidationError(
            "control artifact contains no status records"
        )
    if len({item.item_id for item in records}) != len(records):
        raise PlannerDoctorInventoryValidationError(
            "control artifact contains duplicate status records"
        )
    return tuple(sorted(records))


def _control_status(
    root: Path,
    *,
    taskboard_path: str,
    objectives_path: str,
) -> ControlStatusSnapshot:
    task_bytes = _stable_live_bytes(root, taskboard_path)
    objective_bytes = _stable_live_bytes(root, objectives_path)
    return ControlStatusSnapshot(
        taskboard_path=taskboard_path,
        taskboard_blob_cid=raw_blob_content_id(task_bytes),
        objective_path=objectives_path,
        objective_blob_cid=raw_blob_content_id(objective_bytes),
        tasks=_parse_statuses(task_bytes, goal=False),
        goals=_parse_statuses(objective_bytes, goal=True),
    )


def _nested_config_value(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _configuration_status(root: Path, relative: str) -> ConfigurationStatus:
    path = root / _relative_path(relative)
    if not path.exists():
        return ConfigurationStatus(
            path=relative,
            present=False,
            reason_codes=("configuration_missing",),
        )
    data = _stable_live_bytes(root, relative)
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlannerDoctorInventoryValidationError(
            "configuration is not canonical JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PlannerDoctorInventoryValidationError(
            "configuration root must be an object"
        )
    _reject_private_keys(payload, "configuration")
    settings: list[ConfigurationSetting] = []
    for dotted in _CONFIG_SETTING_PATHS:
        value = _nested_config_value(payload, dotted)
        if value is not None:
            settings.append(ConfigurationSetting(dotted, value))
    return ConfigurationStatus(
        path=relative,
        present=True,
        blob_cid=raw_blob_content_id(data),
        config_schema=str(payload.get("schema") or ""),
        settings=tuple(settings),
    )


class _AuditedRefReader:
    def __init__(self, root: Path, commit: str) -> None:
        self.root = root
        self.commit = commit
        self._bytes: dict[str, bytes | None] = {}

    def read(self, relative: str) -> bytes | None:
        path = _relative_path(relative)
        if path not in self._bytes:
            self._bytes[path] = _git(
                self.root,
                "cat-file",
                "blob",
                f"{self.commit}:{path}",
                required=False,
            )
        return self._bytes[path]

    def text(self, relative: str) -> str:
        value = self.read(relative)
        if value is None:
            return ""
        if len(value) > _MAX_TEXT_CHARS:
            return ""
        return value.decode("utf-8", "replace")

    def blob_cid(self, relative: str) -> str:
        value = self.read(relative)
        return raw_blob_content_id(value) if value is not None else ""

    def present(self, relative: str) -> bool:
        return self.read(relative) is not None


def _function_default(
    text: str,
    *,
    class_name: str,
    function_name: str,
    parameter_name: str,
) -> str:
    if not text:
        return "source_missing"
    try:
        module = ast.parse(text)
    except SyntaxError:
        return "syntax_error"
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name != function_name:
                continue
            positional = [*child.args.posonlyargs, *child.args.args]
            defaults: dict[str, ast.expr | None] = {
                item.arg: None for item in positional
            }
            offset = len(positional) - len(child.args.defaults)
            for index, default in enumerate(child.args.defaults):
                defaults[positional[offset + index].arg] = default
            for item, default in zip(
                child.args.kwonlyargs, child.args.kw_defaults, strict=True
            ):
                defaults[item.arg] = default
            if parameter_name not in defaults:
                return "parameter_missing"
            default = defaults[parameter_name]
            if default is None:
                return "required_parameter"
            if isinstance(default, ast.Constant) and default.value is None:
                return "default_none"
            return "default_non_none"
        return "function_missing"
    return "class_missing"


def _symbol_present(text: str, symbol: str) -> bool:
    if not text:
        return False
    try:
        module = ast.parse(text)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
        for node in ast.walk(module)
    )


def _evidence(
    reader: _AuditedRefReader,
    path: str,
    observation: str,
    *,
    symbol: str = "",
) -> CapabilityEvidence:
    return CapabilityEvidence(
        path=path,
        observation=observation,
        symbol=symbol,
        blob_cid=reader.blob_cid(path),
    )


def _prompt_default_capability(
    reader: _AuditedRefReader,
    *,
    capability_id: str,
    parameter_name: str,
    reason_code: str,
) -> CapabilityRecord:
    path = "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py"
    default = _function_default(
        reader.text(path),
        class_name="PromptSupervisorService",
        function_name="__init__",
        parameter_name=parameter_name,
    )
    if default == "default_none":
        availability = CapabilityAvailability.SHIPPED
        wiring = DefaultWiringState.UNWIRED
        reasons = (reason_code,)
    elif default == "default_non_none":
        availability = CapabilityAvailability.SHIPPED
        wiring = DefaultWiringState.WIRED
        reasons = ()
    elif default in {"source_missing", "class_missing", "function_missing"}:
        availability = CapabilityAvailability.MISSING
        wiring = DefaultWiringState.UNWIRED
        reasons = ("prompt_supervisor_service_missing",)
    else:
        availability = CapabilityAvailability.PARTIAL
        wiring = DefaultWiringState.UNKNOWN
        reasons = ("prompt_default_unresolved",)
    return CapabilityRecord(
        capability_id=capability_id,
        interface_ids=("PromptSupervisorService@1",),
        availability=availability,
        default_wiring=wiring,
        evidence=(
            _evidence(
                reader,
                path,
                default,
                symbol=f"PromptSupervisorService.__init__.{parameter_name}",
            ),
        ),
        test_paths=("test/api/test_agent_supervisor_prompt_workflow.py",),
        reason_codes=reasons,
    )


def _path_capability(
    reader: _AuditedRefReader,
    *,
    capability_id: str,
    interface_ids: Sequence[str],
    path: str,
    symbol: str,
    missing_reason: str,
    test_paths: Sequence[str],
) -> CapabilityRecord:
    text = reader.text(path)
    present = reader.present(path) and _symbol_present(text, symbol)
    return CapabilityRecord(
        capability_id=capability_id,
        interface_ids=tuple(interface_ids),
        availability=(
            CapabilityAvailability.SHIPPED
            if present
            else CapabilityAvailability.MISSING
        ),
        default_wiring=(
            DefaultWiringState.UNKNOWN if present else DefaultWiringState.UNWIRED
        ),
        evidence=(
            _evidence(
                reader,
                path,
                "symbol_present" if present else "path_or_symbol_missing",
                symbol=symbol,
            ),
        ),
        test_paths=tuple(test_paths),
        reason_codes=() if present else (missing_reason,),
    )


def _configuration_by_path(
    configurations: Sequence[ConfigurationStatus],
    path: str,
) -> ConfigurationStatus | None:
    return next((item for item in configurations if item.path == path), None)


def _capability_records(
    reader: _AuditedRefReader,
    configurations: Sequence[ConfigurationStatus],
) -> tuple[CapabilityRecord, ...]:
    records: list[CapabilityRecord] = [
        _prompt_default_capability(
            reader,
            capability_id="prompt.repository_analysis",
            parameter_name="optional_analysis",
            reason_code="default_optional_analysis_unset",
        ),
        _prompt_default_capability(
            reader,
            capability_id="prompt.independent_plan_admission",
            parameter_name="admission_request_factory",
            reason_code="default_admission_request_factory_unset",
        ),
        _path_capability(
            reader,
            capability_id="planner.create_steer_revision",
            interface_ids=("PlanRevision@1", "PlanDelta@1"),
            path=(
                "ipfs_accelerate_py/agent_supervisor/planning/"
                "plan_revision_contracts.py"
            ),
            symbol="PlanRevision",
            missing_reason="create_steer_revision_contracts_missing",
            test_paths=(
                "test/api/test_agent_supervisor_plan_revision_contracts.py",
            ),
        ),
        _path_capability(
            reader,
            capability_id="planner.parallel_execution_plan",
            interface_ids=("ParallelExecutionPlan@1",),
            path=(
                "ipfs_accelerate_py/agent_supervisor/planning/"
                "parallel_plan_compiler.py"
            ),
            symbol="ParallelPlanCompiler",
            missing_reason="parallel_plan_compiler_missing",
            test_paths=(
                "test/api/test_agent_supervisor_parallel_plan_compiler.py",
            ),
        ),
    ]

    doctor_path = (
        "ipfs_accelerate_py/agent_supervisor/control/"
        "deterministic_doctor_service.py"
    )
    backends_default = _function_default(
        reader.text(doctor_path),
        class_name="DeterministicDoctorService",
        function_name="__init__",
        parameter_name="backends",
    )
    records.append(
        CapabilityRecord(
            capability_id="doctor.production_stage_backends",
            interface_ids=("DeterministicDoctorService@1",),
            availability=(
                CapabilityAvailability.SHIPPED
                if reader.present(doctor_path)
                else CapabilityAvailability.MISSING
            ),
            default_wiring=(
                DefaultWiringState.UNWIRED
                if backends_default == "default_none"
                else DefaultWiringState.UNKNOWN
            ),
            evidence=(
                _evidence(
                    reader,
                    doctor_path,
                    backends_default,
                    symbol="DeterministicDoctorService.__init__.backends",
                ),
            ),
            test_paths=("test/api/test_agent_supervisor_deterministic_doctor_service.py",),
            reason_codes=(
                ("default_doctor_stage_backends_empty",)
                if backends_default == "default_none"
                else ("doctor_stage_backend_default_unresolved",)
            ),
        )
    )

    repository_snapshot = (
        "ipfs_accelerate_py/agent_supervisor/analysis/"
        "doctor_repository_diagnostics.py"
    )
    deterministic_snapshot = (
        "ipfs_accelerate_py/agent_supervisor/analysis/"
        "deterministic_doctor_contracts.py"
    )
    adapter_path = (
        "ipfs_accelerate_py/agent_supervisor/analysis/doctor_contract_adapters.py"
    )
    adapter_present = reader.present(adapter_path)
    schemas_differ = (
        "doctor-evidence-snapshot@1" in reader.text(repository_snapshot)
        and "deterministic-doctor/evidence-snapshot@1"
        in reader.text(deterministic_snapshot)
    )
    records.append(
        CapabilityRecord(
            capability_id="doctor.snapshot_contract_bridge",
            interface_ids=("RepositoryReasoningSnapshot@1",),
            availability=(
                CapabilityAvailability.SHIPPED
                if adapter_present
                else CapabilityAvailability.PARTIAL
            ),
            default_wiring=(
                DefaultWiringState.WIRED
                if adapter_present
                else DefaultWiringState.INCOMPATIBLE
            ),
            evidence=(
                _evidence(
                    reader,
                    repository_snapshot,
                    "repository_snapshot_schema_present",
                    symbol="DoctorEvidenceSnapshot",
                ),
                _evidence(
                    reader,
                    deterministic_snapshot,
                    "deterministic_snapshot_schema_present",
                    symbol="DoctorEvidenceSnapshot",
                ),
                _evidence(
                    reader,
                    adapter_path,
                    "adapter_present" if adapter_present else "adapter_missing",
                ),
            ),
            test_paths=(
                "test/api/test_agent_supervisor_doctor_contract_adapters.py",
            ),
            reason_codes=(
                ()
                if adapter_present
                else (
                    "doctor_snapshot_adapter_missing",
                    "doctor_snapshot_schemas_incompatible"
                    if schemas_differ
                    else "doctor_snapshot_compatibility_unknown",
                )
            ),
        )
    )

    proof_path = (
        "ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py"
    )
    proof_present = reader.present(proof_path)
    records.append(
        CapabilityRecord(
            capability_id="doctor.pinned_proof_authority",
            interface_ids=("DeterministicDoctorHammer@1",),
            availability=(
                CapabilityAvailability.SHIPPED
                if proof_present
                else CapabilityAvailability.PARTIAL
            ),
            # File presence cannot prove that the normal Doctor path invokes a
            # pinned solver and independently reconstructs its theorem.
            default_wiring=DefaultWiringState.UNWIRED,
            evidence=(
                _evidence(
                    reader,
                    proof_path,
                    "pinned_hammer_present" if proof_present else "pinned_hammer_missing",
                ),
            ),
            test_paths=(
                "test/api/test_agent_supervisor_deterministic_doctor_proof_authority.py",
            ),
            reason_codes=(
                ("pinned_doctor_hammer_not_default_wired",)
                if proof_present
                else ("pinned_doctor_hammer_missing",)
            ),
        )
    )

    transaction_path = (
        "ipfs_accelerate_py/agent_supervisor/planning/"
        "deterministic_doctor_transaction.py"
    )
    transaction_text = reader.text(transaction_path)
    live_worktree_path = (
        "ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py"
    )
    live_transaction = reader.present(live_worktree_path)
    static_stub = (
        "def _default_static_applicator" in transaction_text
        and "def _default_restore" in transaction_text
        and "return True" in transaction_text
    )
    records.append(
        CapabilityRecord(
            capability_id="doctor.live_transaction",
            interface_ids=("DeterministicDoctorTransaction@1",),
            availability=(
                CapabilityAvailability.SHIPPED
                if live_transaction
                else CapabilityAvailability.PARTIAL
            ),
            default_wiring=(
                DefaultWiringState.WIRED
                if live_transaction and not static_stub
                else DefaultWiringState.UNSAFE_STUB
            ),
            evidence=(
                _evidence(
                    reader,
                    transaction_path,
                    "static_apply_restore_stub" if static_stub else "static_stub_not_found",
                    symbol="_default_static_applicator",
                ),
                _evidence(
                    reader,
                    live_worktree_path,
                    "live_adapter_present"
                    if live_transaction
                    else "live_adapter_missing",
                ),
            ),
            test_paths=(
                "test/api/test_agent_supervisor_deterministic_doctor_transaction.py",
            ),
            reason_codes=(
                ()
                if live_transaction and not static_stub
                else (
                    "default_transaction_does_not_change_bytes",
                    "default_restore_does_not_verify_bytes",
                )
            ),
        )
    )

    fixed_path = (
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "deterministic_doctor_fixed_point.py"
    )
    live_fixed_path = (
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "deterministic_doctor_live_fixed_point.py"
    )
    fixed_text = reader.text(fixed_path)
    live_fixed = reader.present(live_fixed_path)
    fixed_stub = "def _default_restore" in fixed_text and "return True" in fixed_text
    records.append(
        CapabilityRecord(
            capability_id="doctor.live_fixed_point",
            interface_ids=("DeterministicDoctorLiveFixedPoint@1",),
            availability=(
                CapabilityAvailability.SHIPPED
                if live_fixed
                else CapabilityAvailability.PARTIAL
            ),
            default_wiring=(
                DefaultWiringState.WIRED
                if live_fixed and not fixed_stub
                else DefaultWiringState.UNSAFE_STUB
            ),
            evidence=(
                _evidence(
                    reader,
                    fixed_path,
                    "restore_stub_present" if fixed_stub else "restore_stub_not_found",
                    symbol="_default_restore",
                ),
                _evidence(
                    reader,
                    live_fixed_path,
                    "live_fixed_point_present"
                    if live_fixed
                    else "live_fixed_point_missing",
                ),
            ),
            test_paths=(
                "test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py",
            ),
            reason_codes=(
                ()
                if live_fixed and not fixed_stub
                else (
                    "live_fixed_point_runner_missing",
                    "fixed_point_restore_not_independently_verified",
                )
            ),
        )
    )

    live_benchmark_path = (
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "planner_doctor_live_benchmark.py"
    )
    live_benchmark = reader.present(live_benchmark_path)
    synthetic_paths = (
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "deterministic_doctor_benchmark.py",
        "ipfs_accelerate_py/agent_supervisor/self_improvement/"
        "supervisor_v2_benchmark.py",
    )
    records.append(
        CapabilityRecord(
            capability_id="benchmark.live_paired_runner",
            interface_ids=("PlannerDoctorLiveBenchmark@1",),
            availability=(
                CapabilityAvailability.SHIPPED
                if live_benchmark
                else CapabilityAvailability.PARTIAL
            ),
            default_wiring=(
                DefaultWiringState.WIRED
                if live_benchmark
                else DefaultWiringState.SYNTHETIC_ONLY
            ),
            evidence=tuple(
                [
                    _evidence(
                        reader,
                        live_benchmark_path,
                        "live_runner_present"
                        if live_benchmark
                        else "live_runner_missing",
                    )
                ]
                + [
                    _evidence(reader, path, "fixture_benchmark_present")
                    for path in synthetic_paths
                ]
            ),
            test_paths=(
                "test/api/test_agent_supervisor_deterministic_doctor_benchmark.py",
                "test/api/test_agent_supervisor_supervisor_v2_benchmark.py",
            ),
            reason_codes=() if live_benchmark else ("live_paired_benchmark_missing",),
        )
    )

    scheduler = _configuration_by_path(configurations, DEFAULT_SCHEDULER_PATH)
    refill_enabled = (
        scheduler.setting("derived_refill.enabled_at_bootstrap")
        if scheduler is not None and scheduler.present
        else None
    )
    records.append(
        CapabilityRecord(
            capability_id="self_improvement.derived_refill",
            interface_ids=("PlannerDoctorDerivedRefill@1",),
            availability=CapabilityAvailability.PARTIAL,
            default_wiring=(
                DefaultWiringState.WIRED
                if refill_enabled is True
                else DefaultWiringState.DISABLED
                if refill_enabled is False
                else DefaultWiringState.UNKNOWN
            ),
            evidence=(
                CapabilityEvidence(
                    path=DEFAULT_SCHEDULER_PATH,
                    observation=(
                        "refill_enabled"
                        if refill_enabled is True
                        else "refill_disabled"
                        if refill_enabled is False
                        else "refill_status_unknown"
                    ),
                    symbol="derived_refill.enabled_at_bootstrap",
                    blob_cid=scheduler.blob_cid if scheduler is not None else "",
                ),
            ),
            config_paths=(DEFAULT_SCHEDULER_PATH,),
            reason_codes=(
                ()
                if refill_enabled is True
                else ("derived_refill_disabled",)
                if refill_enabled is False
                else ("derived_refill_status_unknown",)
            ),
        )
    )

    root_init = "ipfs_accelerate_py/__init__.py"
    hf_inference = "ipfs_accelerate_py/hf_space_inference.py"
    eager_hf = "from .hf_space_inference import" in reader.text(root_init)
    requests_reachable = "import requests" in reader.text(hf_inference)
    cold_regression = eager_hf and requests_reachable
    records.append(
        CapabilityRecord(
            capability_id="runtime.cold_import_hygiene",
            interface_ids=("PlannerDoctorColdImport@1",),
            availability=CapabilityAvailability.PARTIAL,
            default_wiring=(
                DefaultWiringState.INCOMPATIBLE
                if cold_regression
                else DefaultWiringState.WIRED
            ),
            evidence=(
                _evidence(
                    reader,
                    root_init,
                    "eager_hf_space_import" if eager_hf else "lazy_hf_space_import",
                ),
                _evidence(
                    reader,
                    hf_inference,
                    "requests_import_reachable"
                    if requests_reachable
                    else "requests_import_not_found",
                ),
            ),
            test_paths=("test/api/test_agent_supervisor_doctor_cold_import.py",),
            reason_codes=(
                ("cold_import_network_client_reachable",)
                if cold_regression
                else ()
            ),
        )
    )
    return tuple(sorted(records, key=lambda item: item.capability_id))


def _artifact_status_at_ref(
    reader: _AuditedRefReader,
    path: str,
    kind: ArtifactKind,
) -> ArtifactStatus:
    present = reader.present(path)
    return ArtifactStatus(
        path=path,
        kind=kind,
        state=ArtifactState.PRESENT if present else ArtifactState.MISSING,
        source="audited_baseline",
        blob_cid=reader.blob_cid(path),
        reason_codes=() if present else ("artifact_missing",),
    )


def _artifact_status_live(
    root: Path,
    path: str,
    kind: ArtifactKind,
) -> ArtifactStatus:
    candidate = root / _relative_path(path)
    if not candidate.exists():
        return ArtifactStatus(
            path=path,
            kind=kind,
            state=ArtifactState.MISSING,
            source="current_checkout",
            reason_codes=("artifact_missing",),
        )
    data = _stable_live_bytes(root, path)
    return ArtifactStatus(
        path=path,
        kind=kind,
        state=ArtifactState.PRESENT,
        source="current_checkout",
        blob_cid=raw_blob_content_id(data),
    )


def _tool_observations(
    probes: Sequence[
        MetadataToolProbe
        | Callable[[], Mapping[str, Any] | ToolHealthObservation]
        | Mapping[str, Any]
        | ToolHealthObservation
    ],
    optional_tool_ids: Sequence[str],
) -> tuple[ToolHealthObservation, ...]:
    by_tool: dict[str, ToolHealthObservation] = {}
    for raw_probe in probes:
        if isinstance(raw_probe, ToolHealthObservation):
            observation = raw_probe
        elif isinstance(raw_probe, Mapping):
            observation = ToolHealthObservation.from_dict(raw_probe)
        else:
            operation = getattr(raw_probe, "probe", None)
            if not callable(operation):
                operation = raw_probe
            if not callable(operation):
                raise PlannerDoctorInventoryValidationError(
                    "tool probe must be a mapping, callable, or implement probe"
                )
            raw = operation()
            observation = (
                raw
                if isinstance(raw, ToolHealthObservation)
                else ToolHealthObservation.from_dict(_mapping(raw, "tool_probe_result"))
            )
        if observation.tool_id in by_tool:
            raise PlannerDoctorInventoryValidationError(
                "duplicate injected tool probe"
            )
        by_tool[observation.tool_id] = observation
    for tool_id in _strings(optional_tool_ids, "optional_tool_ids", identifiers=True):
        by_tool.setdefault(
            tool_id,
            ToolHealthObservation(
                tool_id=tool_id,
                probe_id="inventory.not_probed",
                health=ToolHealthState.NOT_PROBED,
                reason_codes=("metadata_probe_not_supplied",),
            ),
        )
    return tuple(sorted(by_tool.values(), key=lambda item: item.tool_id))


def build_planner_doctor_capability_inventory(
    repository_root: str | os.PathLike[str],
    *,
    audited_ref: str,
    taskboard_path: str = DEFAULT_TASKBOARD_PATH,
    objectives_path: str = DEFAULT_OBJECTIVES_PATH,
    config_paths: Sequence[str] = (DEFAULT_SCHEDULER_PATH,),
    tool_probes: Sequence[
        MetadataToolProbe
        | Callable[[], Mapping[str, Any] | ToolHealthObservation]
        | Mapping[str, Any]
        | ToolHealthObservation
    ] = (),
    optional_tool_ids: Sequence[str] = DEFAULT_OPTIONAL_TOOL_IDS,
) -> PlannerDoctorCapabilityInventory:
    """Build a stable inventory over an explicit baseline and live checkout.

    The audited ref is never inferred from a branch name or taskboard.  The
    live checkout is observed twice; concurrent HEAD, dirty-overlay, or
    Gitlink changes fail closed rather than yielding a mixed snapshot.
    """

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise PlannerDoctorInventoryValidationError(
            "repository_root must be a directory"
        )
    top = _git_text(root, "rev-parse", "--show-toplevel")
    try:
        if Path(top).resolve(strict=True) != root:
            raise PlannerDoctorInventoryValidationError(
                "repository_root must be the Git toplevel"
            )
    except (OSError, RuntimeError) as exc:
        raise PlannerDoctorInventoryValidationError(
            "repository_root is not resolvable"
        ) from exc

    audited_commit = _resolve_commit(root, audited_ref)
    audited = _observe_revision(
        root,
        label=f"audited:{audited_ref}",
        commit=audited_commit,
        include_dirty_overlay=False,
    )
    current_commit = _resolve_commit(root, "HEAD")
    current = _observe_revision(
        root,
        label="current:HEAD+overlay",
        commit=current_commit,
        include_dirty_overlay=True,
    )
    control = _control_status(
        root,
        taskboard_path=_relative_path(taskboard_path, "taskboard_path"),
        objectives_path=_relative_path(objectives_path, "objectives_path"),
    )
    configurations = tuple(
        _configuration_status(root, _relative_path(path, "config_path"))
        for path in config_paths
    )

    reader = _AuditedRefReader(root, audited_commit)
    capabilities = _capability_records(reader, configurations)
    artifacts = tuple(
        [
            *(
                _artifact_status_at_ref(reader, path, ArtifactKind.SOURCE)
                for path in (*_AUDITED_SOURCE_PATHS, *_FUTURE_INTERFACE_PATHS)
            ),
            *(
                _artifact_status_at_ref(reader, path, ArtifactKind.TEST)
                for path in _AUDITED_TEST_PATHS
            ),
            _artifact_status_live(root, taskboard_path, ArtifactKind.TASKBOARD),
            _artifact_status_live(root, objectives_path, ArtifactKind.OBJECTIVES),
            *(
                _artifact_status_live(root, path, ArtifactKind.CONFIG)
                for path in config_paths
            ),
        ]
    )
    tools = _tool_observations(tool_probes, optional_tool_ids)

    current_replay = _observe_revision(
        root,
        label="current:HEAD+overlay",
        commit=_resolve_commit(root, "HEAD"),
        include_dirty_overlay=True,
    )
    if current_replay.revision_id != current.revision_id:
        raise PlannerDoctorInventoryIntegrityError(
            "repository changed during capability inventory"
        )

    return PlannerDoctorCapabilityInventory(
        audited_baseline=audited,
        current_checkout=current,
        control_status=control,
        capabilities=capabilities,
        artifacts=artifacts,
        configurations=configurations,
        tool_health=tools,
    )


# Taskboard wording uses "inventory" as the operation; retain a concise alias.
inventory_planner_doctor_capabilities = build_planner_doctor_capability_inventory


def replay_planner_doctor_capability_inventory(
    value: Mapping[str, Any] | str,
) -> PlannerDoctorCapabilityInventory:
    """Strictly reload an inventory record and verify every claimed identity."""

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PlannerDoctorInventoryValidationError(
                "inventory JSON is malformed"
            ) from exc
        payload = _mapping(decoded, "inventory")
    else:
        payload = _mapping(value, "inventory")
    if "content_id" not in payload:
        raise PlannerDoctorInventoryIntegrityError(
            "replayed inventory requires content_id"
        )
    return PlannerDoctorCapabilityInventory.from_dict(payload)


def discover_planner_doctor_inventory_schemas() -> tuple[str, ...]:
    """Provider-free schema discovery for control-plane negotiation."""

    return (
        PLANNER_DOCTOR_ARTIFACT_STATUS_SCHEMA,
        PLANNER_DOCTOR_CAPABILITY_EVIDENCE_SCHEMA,
        PLANNER_DOCTOR_CAPABILITY_INVENTORY_SCHEMA,
        PLANNER_DOCTOR_CAPABILITY_RECORD_SCHEMA,
        PLANNER_DOCTOR_CONFIGURATION_STATUS_SCHEMA,
        PLANNER_DOCTOR_CONTROL_STATUS_SCHEMA,
        PLANNER_DOCTOR_GITLINK_BINDING_SCHEMA,
        PLANNER_DOCTOR_REPOSITORY_REVISION_SCHEMA,
        PLANNER_DOCTOR_TOOL_HEALTH_SCHEMA,
    )


__all__ = [
    "ArtifactKind",
    "ArtifactState",
    "ArtifactStatus",
    "CapabilityAvailability",
    "CapabilityEvidence",
    "CapabilityRecord",
    "ConfigurationSetting",
    "ConfigurationStatus",
    "ControlStatusSnapshot",
    "DEFAULT_OBJECTIVES_PATH",
    "DEFAULT_OPTIONAL_TOOL_IDS",
    "DEFAULT_SCHEDULER_PATH",
    "DEFAULT_TASKBOARD_PATH",
    "DefaultWiringState",
    "GitlinkBinding",
    "INVENTORY_AUTHORIZES_MUTATION",
    "INVENTORY_IS_COMPLETION_EVIDENCE",
    "INVENTORY_IS_PROOF_EVIDENCE",
    "MetadataToolProbe",
    "PACKAGE_PRESENCE_IS_CAPABILITY",
    "PDR_AUDITED_BASELINE_COMMIT",
    "PLANNER_DOCTOR_CAPABILITY_INVENTORY_SCHEMA",
    "PlannerDoctorCapabilityInventory",
    "PlannerDoctorInventoryError",
    "PlannerDoctorInventoryGitError",
    "PlannerDoctorInventoryIntegrityError",
    "PlannerDoctorInventoryValidationError",
    "RepositoryRevision",
    "StatusEntry",
    "ToolHealthObservation",
    "ToolHealthState",
    "build_planner_doctor_capability_inventory",
    "canonical_inventory_json_bytes",
    "discover_planner_doctor_inventory_schemas",
    "inventory_content_id",
    "inventory_planner_doctor_capabilities",
    "raw_blob_content_id",
    "replay_planner_doctor_capability_inventory",
]
