"""Fail-closed CASF promotion, rollback, and quarantine recommendations.

The gate consumes the real CASF-030/032--041 wire artifacts. It deliberately
does not accept a caller-authored ``passed`` bit, origin label, or receipt ID as
qualification evidence. Each artifact is bounded and canonicalized, then
decoded by its owning contract (or reconstructed as its exact owning type when
the legacy owner has no wire decoder).

The current artifact population is non-promoting: CASF-030/032/033 lack an
accepted-producer, state-owner, and full qualification-identity provenance
envelope; CASF-035 and CASF-036 reports remain non-authoritative without that
same accepted provenance; CASF-037 always requires upstream reverification;
and CASF-038--041 explicitly record non-promotion or unavailable/not-run live
capability. The truthful current disposition is therefore
``quarantine_required``.

No function in this module applies promotion, rollback, or quarantine. Only a
future registered typed state-owner policy operation may do so after independent
reverification.

Interface: ``FederationPromotionGate@1``
Evidence: ``casf/promotion-evidence-bundle@1``
Decision: ``casf/promotion-decision@1``
"""

# Python 3.8 remains supported by the package.
# ruff: noqa: UP042

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.prover_matrix_registry import ProverState
from ..self_improvement.supervisor_state_model import (
    ModelCheckBounds,
    ModelCheckerTool,
)
from ..task_sources.control_plane_contracts import content_identity
from .chaos import CASF_CHAOS_REPORT_SCHEMA, ChaosReport
from .cli import federation_cli_discovery_manifest
from .contracts import (
    FederationAuthorityError,
    FederationBoundsError,
    FederationContractError,
    FederationSecretError,
    UnknownNormativeFieldError,
)
from .control_service import (
    FEDERATION_CONTROL_AUDIT_SCHEMA,
    FederationControlAuditReceipt,
)
from .drift_monitor import DRIFT_REPORT_SCHEMA, DriftReport, validate_current_drift_report
from .ducklake_projection import ProjectionReceipt, ProjectionRecoveryReceipt
from .fixed_point import FixedPointReceipt
from .formal import (
    CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA,
    CASF_FORMAL_IDENTITY_SCHEMA,
    CASF_FORMAL_SUITE_SCHEMA,
    ExternalCheckStatus,
    ExternalModelCheckReceipt,
    ExternalModelInvariant,
    FederationFormalError,
    FederationFormalIdentity,
    FederationFormalProperty,
    HermeticCheckStatus,
    build_federation_formal_suite,
    check_federation_formal_suite,
)

FEDERATION_PROMOTION_GATE_INTERFACE: Final[str] = "FederationPromotionGate@1"
QUALIFICATION_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/qualification-identity@1"
)
PROMOTION_EVIDENCE_BUNDLE_SCHEMA: Final[str] = "casf/promotion-evidence-bundle@1"
ARTIFACT_ASSESSMENT_SCHEMA: Final[str] = "casf/promotion-artifact-assessment@1"
PROMOTION_DECISION_SCHEMA: Final[str] = "casf/promotion-decision@1"
ROLLBACK_DECISION_SCHEMA: Final[str] = "casf/rollback-decision@1"
QUARANTINE_DECISION_SCHEMA: Final[str] = "casf/quarantine-decision@1"
DECISION_VALIDATION_SCHEMA: Final[str] = "casf/promotion-decision-validation@1"
CASF_CONTROL_PARITY_REPORT_SCHEMA: Final[str] = "casf/control-parity@1"
CASF_FORMAL_MODEL_REPORT_SCHEMA: Final[str] = "casf/formal-model-report@1"
_CONTROL_MCP_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-discovery@1"
)
_CONTROL_MCP_INTERFACE: Final[str] = "FederationControlMCP@1"
_CONTROL_MCP_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-result@1"
)
_CONTROL_MCP_ERROR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-error@1"
)

_FIXED_POINT_SCHEMA: Final[str] = FixedPointReceipt.SCHEMA
_DUCKLAKE_PROJECTION_SCHEMA: Final[str] = ProjectionReceipt.SCHEMA
_DUCKLAKE_RECOVERY_SCHEMA: Final[str] = ProjectionRecoveryReceipt.SCHEMA
_IDLE_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-benchmark-manifest@2"
)
_PARALLEL_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/parallel-benchmark-manifest@1"
)
_LOAD_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/load-benchmark-manifest@1"
)
_TOKEN_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/token-benchmark-manifest@1"
)
_BENCHMARK_RESULT_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "CASF-038": "casf/idle-benchmark@1",
        "CASF-039": "casf/parallel-benchmark@1",
        "CASF-040": "casf/load-benchmark@1",
        "CASF-041": "casf/token-benchmark@1",
    }
)
_BENCHMARK_MANIFEST_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "CASF-038": _IDLE_MANIFEST_SCHEMA,
        "CASF-039": _PARALLEL_MANIFEST_SCHEMA,
        "CASF-040": _LOAD_MANIFEST_SCHEMA,
        "CASF-041": _TOKEN_MANIFEST_SCHEMA,
    }
)
_BENCHMARK_MANIFEST_FILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "CASF-038": "idle_manifest.json",
        "CASF-039": "parallel_manifest.json",
        "CASF-040": "load_manifest.json",
        "CASF-041": "token_manifest.json",
    }
)
_BENCHMARK_MANIFEST_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "CASF-038": "4a9d80d5905202433b145b1b5b0ff809e7137962f3fccedc725699f0701a3bd0",
        "CASF-039": "2d2c506f6fb2a68074449b50e09ac0748f373ea357642176d4e27fc7c1c7bae6",
        "CASF-040": "0585e25bb98036edb1941477db43780d46ebb736a2f826b90d3ce6e12539b357",
        "CASF-041": "34cbda396f72dc5bdee7eda5ec6639bdfbf7574e9da132bea01536c5d0cd924e",
    }
)

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_OID = re.compile(r"[0-9a-f]{40}")
_CONTENT_REF = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})")
_SECRET = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|\b(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{8,}"
    r"|\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
_SECRET_KEY_ATOMS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "jwt",
        "passwd",
        "passphrase",
        "password",
        "secret",
    }
)
_SECRET_KEY_COMPOUNDS: Final[frozenset[str]] = frozenset(
    {
        "accesskey",
        "accesskeyid",
        "accesstoken",
        "apikey",
        "apitoken",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "clienttoken",
        "connectionstring",
        "credential",
        "credentials",
        "idtoken",
        "password",
        "passwd",
        "passphrase",
        "privatekey",
        "proxyauthorization",
        "refreshtoken",
        "secretaccesskey",
        "secretkey",
        "sessiontoken",
        "setcookie",
        "signingkey",
        "signingsecret",
        "webhooksecret",
    }
)
_SECRET_KEY_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("access", "key"),
        ("access", "token"),
        ("api", "key"),
        ("api", "token"),
        ("auth", "header"),
        ("auth", "token"),
        ("authorization", "header"),
        ("bearer", "token"),
        ("client", "secret"),
        ("client", "token"),
        ("connection", "string"),
        ("id", "token"),
        ("private", "key"),
        ("proxy", "authorization"),
        ("refresh", "token"),
        ("secret", "access"),
        ("secret", "key"),
        ("session", "token"),
        ("set", "cookie"),
        ("signing", "key"),
        ("signing", "secret"),
        ("webhook", "secret"),
    }
)
_KNOWN_PUBLIC_HANDLE_KEYS: Final[frozenset[str]] = frozenset({"authorization_id"})
_PUBLIC_SECRET_REFERENCE_SUFFIXES: Final[tuple[str, ...]] = (
    "_digest",
    "_fingerprint",
    "_ref",
    "_refs",
)

MAX_CAPABILITIES: Final[int] = 128
MAX_ARTIFACT_BYTES: Final[int] = 2 * 1024 * 1024
MAX_BUNDLE_BYTES: Final[int] = 8 * 1024 * 1024
MAX_JSON_DEPTH: Final[int] = 32
MAX_JSON_NODES: Final[int] = 200_000
MAX_JSON_CONTAINER_ITEMS: Final[int] = 65_536
MAX_JSON_TEXT_BYTES: Final[int] = 128 * 1024
MAX_BENCHMARK_MANIFEST_BYTES: Final[int] = 64 * 1024
MAX_FORMAL_EXTERNAL_RECEIPTS: Final[int] = 12
MAX_FORMAL_REASON_BYTES: Final[int] = 4 * 1024

_FORMAL_BOUND_LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "max_steps": 64,
        "max_retries": 16,
        "max_fence": 1_024,
        "max_tasks": 32,
        "max_agents": 16,
        "max_states": 32,
        "max_transitions": 64,
        "max_evidence_ids": 128,
    }
)

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_BENCHMARK_ROOT: Final[Path] = (
    _REPOSITORY_ROOT / "benchmarks/agent_supervisor/causal_event_federation"
)


class PromotionGateError(FederationContractError):
    """Promotion evidence is malformed, stale, oversized, or unsafe."""


class StaleQualificationEvidenceError(PromotionGateError):
    """A decision is not bound to the exact current qualification identity."""


class MissingQualificationCapabilityError(PromotionGateError):
    """A caller required permission from a decision that remains blocked."""


class GateProfile(str, Enum):
    """DuckLake is an independent profile and never qualifies the core path."""

    DUCKDB_QUACK = "duckdb_quack"
    DUCKLAKE = "ducklake"


class DecisionKind(str, Enum):
    PROMOTION = "promotion"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"


class DecisionStatus(str, Enum):
    PERMITTED = "permitted"
    BLOCKED = "blocked"


class DecisionDisposition(str, Enum):
    PROMOTION_RECOMMENDED = "promotion_recommended"
    ROLLBACK_RECOMMENDED = "rollback_recommended"
    QUARANTINE_REQUIRED = "quarantine_required"


class ArtifactStatus(str, Enum):
    PASSED = "passed"
    MISSING = "missing"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    NONAUTHORITATIVE = "nonauthoritative"


class EvidenceSlot(str, Enum):
    FIXED_POINT = "fixed_point_receipt"
    DUCKLAKE_PROJECTION = "ducklake_projection_receipt"
    DUCKLAKE_RECOVERY = "ducklake_recovery_receipt"
    DRIFT = "drift_report"
    CONTROL_AUDIT = "control_audit_receipt"
    CONTROL_PARITY = "control_parity_report"
    FORMAL = "formal_report"
    ADVERSARIAL = "adversarial_report"
    IDLE = "idle_benchmark"
    PARALLEL = "parallel_benchmark"
    LOAD = "load_benchmark"
    TOKEN = "token_benchmark"


_SLOT_TASK: Final[Mapping[EvidenceSlot, str]] = MappingProxyType(
    {
        EvidenceSlot.FIXED_POINT: "CASF-030",
        EvidenceSlot.DUCKLAKE_PROJECTION: "CASF-032",
        EvidenceSlot.DUCKLAKE_RECOVERY: "CASF-032",
        EvidenceSlot.DRIFT: "CASF-033",
        EvidenceSlot.CONTROL_AUDIT: "CASF-034",
        EvidenceSlot.CONTROL_PARITY: "CASF-035",
        EvidenceSlot.FORMAL: "CASF-036",
        EvidenceSlot.ADVERSARIAL: "CASF-037",
        EvidenceSlot.IDLE: "CASF-038",
        EvidenceSlot.PARALLEL: "CASF-039",
        EvidenceSlot.LOAD: "CASF-040",
        EvidenceSlot.TOKEN: "CASF-041",
    }
)

_CORE_REQUIRED_SLOTS: Final[tuple[EvidenceSlot, ...]] = (
    EvidenceSlot.FIXED_POINT,
    EvidenceSlot.DRIFT,
    EvidenceSlot.CONTROL_AUDIT,
    EvidenceSlot.CONTROL_PARITY,
    EvidenceSlot.FORMAL,
    EvidenceSlot.ADVERSARIAL,
    EvidenceSlot.IDLE,
    EvidenceSlot.PARALLEL,
    EvidenceSlot.LOAD,
    EvidenceSlot.TOKEN,
)
_DUCKLAKE_REQUIRED_SLOTS: Final[tuple[EvidenceSlot, ...]] = (
    EvidenceSlot.DUCKLAKE_PROJECTION,
    EvidenceSlot.DUCKLAKE_RECOVERY,
)


def _token(value: Any, name: str) -> str:
    if type(value) is not str or value != value.strip() or not value:
        raise PromotionGateError(f"{name} must be nonempty exact text")
    if _SECRET.search(value):
        raise FederationSecretError(f"{name} contains credential-shaped material")
    if _TOKEN.fullmatch(value) is None:
        raise PromotionGateError(f"{name} is not a compact identity")
    return value


def _oid(value: Any, name: str) -> str:
    value = _token(value, name)
    if _OID.fullmatch(value) is None:
        raise PromotionGateError(f"{name} must be a lowercase 40-hex Git object id")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PromotionGateError(f"{name} must be an exact integer >= {minimum}")
    return value


def _content_ref(value: Any, name: str) -> str:
    value = _token(value, name)
    if _CONTENT_REF.fullmatch(value) is None:
        raise PromotionGateError(f"{name} must be content addressed")
    return value


def _exact_list(value: Any, name: str, *, maximum: int) -> list[Any]:
    if type(value) is not list:
        raise PromotionGateError(f"{name} must be an exact JSON array")
    if len(value) > maximum:
        raise FederationBoundsError(f"{name} exceeds its array bound")
    return value


def _closed_dict(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise PromotionGateError(f"{label} must be an exact object")
    _validate_json_value(value, label, depth=0, ancestors=frozenset(), counter=[0])
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise UnknownNormativeFieldError(
            f"{label} has unknown fields: {sorted(str(item) for item in unknown)!r}"
        )
    if missing:
        raise PromotionGateError(
            f"{label} is missing fields: {sorted(str(item) for item in missing)!r}"
        )
    return value


def _exact_wire(value: Any, expected: Any, label: str) -> None:
    """Require exact JSON types and byte-canonical equality to trusted wire data."""

    _validate_json_value(value, label, depth=0, ancestors=frozenset(), counter=[0])
    try:
        actual_bytes = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        expected_bytes = json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromotionGateError(f"{label} is not canonical JSON") from exc
    if actual_bytes != expected_bytes:
        raise PromotionGateError(f"{label} does not match canonical producer wire data")


def _exact_boolean(value: Any, expected: bool, name: str) -> None:
    if type(value) is not bool or value is not expected:
        raise PromotionGateError(f"{name} must be exact {str(expected).lower()}")


def _bounded_report_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PromotionGateError(f"{name} must be nonempty exact text")
    if len(value.encode("utf-8")) > MAX_FORMAL_REASON_BYTES:
        raise FederationBoundsError(f"{name} exceeds its text bound")
    return value


def _canonical_tokens(value: Any, name: str, *, maximum: int) -> tuple[str, ...]:
    if type(value) not in (tuple, list):
        raise PromotionGateError(f"{name} must be an exact array")
    if len(value) > maximum:
        raise FederationBoundsError(f"{name} exceeds its bound")
    result = tuple(_token(item, f"{name}[{index}]") for index, item in enumerate(value))
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise PromotionGateError(f"{name} must be sorted and unique")
    return result


def _normalized_json_key(key: str) -> str:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _is_public_secret_reference(key: str, value: Any) -> bool:
    """Allow secret-material *references*, never inline secret-bearing fields."""

    normalized = _normalized_json_key(key)
    if not normalized.endswith(_PUBLIC_SECRET_REFERENCE_SUFFIXES):
        return False
    if type(value) is str:
        return _CONTENT_REF.fullmatch(value) is not None
    if type(value) is list and value:
        return all(type(item) is str and _CONTENT_REF.fullmatch(item) is not None for item in value)
    return False


def _secret_shaped_key(key: str, value: Any) -> bool:
    normalized = _normalized_json_key(key)
    if (
        normalized in _KNOWN_PUBLIC_HANDLE_KEYS
        and type(value) is str
        and _TOKEN.fullmatch(value) is not None
        and _SECRET.search(value) is None
    ):
        return False
    atoms = tuple(item for item in normalized.split("_") if item)
    compact = "".join(atoms)
    adjacent_pairs = {(atoms[index], atoms[index + 1]) for index in range(len(atoms) - 1)}
    secret_shaped = bool(
        _SECRET_KEY_ATOMS.intersection(atoms)
        or _SECRET_KEY_PAIRS.intersection(adjacent_pairs)
        or any(compound in compact for compound in _SECRET_KEY_COMPOUNDS)
    )
    if normalized in {"key", "token"}:
        secret_shaped = True
    return secret_shaped and not _is_public_secret_reference(key, value)


def _validate_json_value(
    value: Any,
    name: str,
    *,
    depth: int,
    ancestors: frozenset[int],
    counter: list[int],
) -> None:
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise FederationBoundsError(f"{name} exceeds the JSON node bound")
    if depth > MAX_JSON_DEPTH:
        raise FederationBoundsError(f"{name} exceeds the JSON depth bound")
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_JSON_TEXT_BYTES:
            raise FederationBoundsError(f"{name} contains oversized text")
        if _SECRET.search(value):
            raise FederationSecretError(f"{name} contains credential-shaped material")
        return
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise PromotionGateError(f"{name} contains a non-finite number")
        return
    if type(value) not in (dict, list):
        raise PromotionGateError(f"{name} contains a non-JSON exact type")
    marker = id(value)
    if marker in ancestors:
        raise PromotionGateError(f"{name} contains a cycle")
    if len(value) > MAX_JSON_CONTAINER_ITEMS:
        raise FederationBoundsError(f"{name} contains an oversized container")
    descendants = ancestors | {marker}
    children = value.items() if type(value) is dict else enumerate(value)
    for key, child in children:
        if type(value) is dict:
            if type(key) is not str:
                raise PromotionGateError(f"{name} contains a non-text object key")
            if len(key.encode("utf-8")) > 512:
                raise FederationBoundsError(f"{name} contains an oversized object key")
            if _SECRET.search(key) or _secret_shaped_key(key, child):
                raise FederationSecretError(f"{name} contains an unsafe object key")
            child_name = f"{name}.{key}"
        else:
            child_name = f"{name}[{key}]"
        _validate_json_value(
            child,
            child_name,
            depth=depth + 1,
            ancestors=descendants,
            counter=counter,
        )


def _canonical_artifact(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise PromotionGateError(f"{name} must be an exact JSON object or null")
    _validate_json_value(value, name, depth=0, ancestors=frozenset(), counter=[0])
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PromotionGateError(f"{name} is not canonical JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_ARTIFACT_BYTES:
        raise FederationBoundsError(f"{name} exceeds its byte bound")
    return encoded


def _json_content_ref(value: Any, name: str, *, maximum: int) -> str:
    """Hash bounded canonical JSON that may contain finite benchmark floats."""

    _validate_json_value(value, name, depth=0, ancestors=frozenset(), counter=[0])
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromotionGateError(f"{name} is not canonical JSON") from exc
    if len(encoded) > maximum:
        raise FederationBoundsError(f"{name} exceeds its content-addressing bound")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _artifact_dict(value: str | None, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if type(decoded) is not dict:
        raise PromotionGateError(f"{name} did not decode to an object")
    return decoded


@dataclass(frozen=True)
class QualificationIdentity:
    """Exact current tree, policy, capability, assignment, lease, and fence."""

    tenant_id: str
    federation_id: str
    repository_id: str
    revision: str
    tree_id: str
    schema_id: str
    generation_id: str
    control_plane_generation: int
    policy_id: str
    policy_revision: int
    capability_ids: tuple[str, ...]
    task_id: str
    attempt_id: str
    lease_id: str
    fencing_epoch: int
    assignment_revision: int
    worktree_id: str
    world_snapshot_ref: str
    event_watermark: int
    schema: str = QUALIFICATION_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != QUALIFICATION_IDENTITY_SCHEMA:
            raise PromotionGateError("unsupported qualification identity schema")
        for name in (
            "tenant_id",
            "federation_id",
            "repository_id",
            "schema_id",
            "generation_id",
            "policy_id",
            "task_id",
            "attempt_id",
            "lease_id",
            "worktree_id",
            "world_snapshot_ref",
        ):
            object.__setattr__(self, name, _token(getattr(self, name), name))
        if self.task_id != "CASF-042":
            raise PromotionGateError("task_id must be the exact CASF-042 identity")
        object.__setattr__(self, "revision", _oid(self.revision, "revision"))
        object.__setattr__(self, "tree_id", _oid(self.tree_id, "tree_id"))
        if type(self.capability_ids) is not tuple:
            raise PromotionGateError("capability_ids must be an exact tuple")
        object.__setattr__(
            self,
            "capability_ids",
            _canonical_tokens(self.capability_ids, "capability_ids", maximum=MAX_CAPABILITIES),
        )
        if not self.capability_ids:
            raise PromotionGateError("capability_ids must not be empty")
        _integer(self.control_plane_generation, "control_plane_generation", minimum=1)
        _integer(self.policy_revision, "policy_revision", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _integer(self.assignment_revision, "assignment_revision", minimum=1)
        _integer(self.event_watermark, "event_watermark")
        object.__setattr__(
            self,
            "world_snapshot_ref",
            _content_ref(self.world_snapshot_ref, "world_snapshot_ref"),
        )

    @property
    def identity_id(self) -> str:
        return "qualification:" + content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "tenant_id": self.tenant_id,
            "federation_id": self.federation_id,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "tree_id": self.tree_id,
            "schema_id": self.schema_id,
            "generation_id": self.generation_id,
            "control_plane_generation": self.control_plane_generation,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capability_ids": list(self.capability_ids),
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "assignment_revision": self.assignment_revision,
            "worktree_id": self.worktree_id,
            "world_snapshot_ref": self.world_snapshot_ref,
            "event_watermark": self.event_watermark,
        }
        if include_identity:
            value["identity_id"] = self.identity_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualificationIdentity:
        fields = frozenset(
            {
                "schema",
                "tenant_id",
                "federation_id",
                "repository_id",
                "revision",
                "tree_id",
                "schema_id",
                "generation_id",
                "control_plane_generation",
                "policy_id",
                "policy_revision",
                "capability_ids",
                "task_id",
                "attempt_id",
                "lease_id",
                "fencing_epoch",
                "assignment_revision",
                "worktree_id",
                "world_snapshot_ref",
                "event_watermark",
                "identity_id",
            }
        )
        data = _closed_dict(value, fields, "identity")
        try:
            capability_ids = _exact_list(
                data["capability_ids"], "identity.capability_ids", maximum=MAX_CAPABILITIES
            )
            result = cls(
                tenant_id=data["tenant_id"],
                federation_id=data["federation_id"],
                repository_id=data["repository_id"],
                revision=data["revision"],
                tree_id=data["tree_id"],
                schema_id=data["schema_id"],
                generation_id=data["generation_id"],
                control_plane_generation=data["control_plane_generation"],
                policy_id=data["policy_id"],
                policy_revision=data["policy_revision"],
                capability_ids=tuple(capability_ids),
                task_id=data["task_id"],
                attempt_id=data["attempt_id"],
                lease_id=data["lease_id"],
                fencing_epoch=data["fencing_epoch"],
                assignment_revision=data["assignment_revision"],
                worktree_id=data["worktree_id"],
                world_snapshot_ref=data["world_snapshot_ref"],
                event_watermark=data["event_watermark"],
                schema=data["schema"],
            )
        except PromotionGateError:
            raise
        except (KeyError, TypeError) as exc:
            raise PromotionGateError("qualification identity is malformed") from exc
        if data["identity_id"] != result.identity_id:
            raise PromotionGateError("qualification identity content identity mismatches")
        return result


@dataclass(frozen=True, init=False)
class QualificationEvidenceBundle:
    """Canonical copies of the only artifact slots CASF-042 understands."""

    identity_id: str
    fixed_point_receipt: str | None
    ducklake_projection_receipt: str | None
    ducklake_recovery_receipt: str | None
    drift_report: str | None
    control_audit_receipt: str | None
    control_parity_report: str | None
    formal_report: str | None
    adversarial_report: str | None
    idle_benchmark: str | None
    parallel_benchmark: str | None
    load_benchmark: str | None
    token_benchmark: str | None
    schema: str

    def __init__(
        self,
        *,
        identity_id: str,
        fixed_point_receipt: dict[str, Any] | None = None,
        ducklake_projection_receipt: dict[str, Any] | None = None,
        ducklake_recovery_receipt: dict[str, Any] | None = None,
        drift_report: dict[str, Any] | None = None,
        control_audit_receipt: dict[str, Any] | None = None,
        control_parity_report: dict[str, Any] | None = None,
        formal_report: dict[str, Any] | None = None,
        adversarial_report: dict[str, Any] | None = None,
        idle_benchmark: dict[str, Any] | None = None,
        parallel_benchmark: dict[str, Any] | None = None,
        load_benchmark: dict[str, Any] | None = None,
        token_benchmark: dict[str, Any] | None = None,
        schema: str = PROMOTION_EVIDENCE_BUNDLE_SCHEMA,
    ) -> None:
        if type(schema) is not str or schema != PROMOTION_EVIDENCE_BUNDLE_SCHEMA:
            raise PromotionGateError("unsupported promotion evidence bundle schema")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "identity_id", _token(identity_id, "bundle.identity_id"))
        supplied = {
            EvidenceSlot.FIXED_POINT: fixed_point_receipt,
            EvidenceSlot.DUCKLAKE_PROJECTION: ducklake_projection_receipt,
            EvidenceSlot.DUCKLAKE_RECOVERY: ducklake_recovery_receipt,
            EvidenceSlot.DRIFT: drift_report,
            EvidenceSlot.CONTROL_AUDIT: control_audit_receipt,
            EvidenceSlot.CONTROL_PARITY: control_parity_report,
            EvidenceSlot.FORMAL: formal_report,
            EvidenceSlot.ADVERSARIAL: adversarial_report,
            EvidenceSlot.IDLE: idle_benchmark,
            EvidenceSlot.PARALLEL: parallel_benchmark,
            EvidenceSlot.LOAD: load_benchmark,
            EvidenceSlot.TOKEN: token_benchmark,
        }
        total = 0
        for slot, artifact in supplied.items():
            encoded = _canonical_artifact(artifact, slot.value)
            if encoded is not None:
                total += len(encoded.encode("utf-8"))
            object.__setattr__(self, slot.value, encoded)
        if total > MAX_BUNDLE_BYTES:
            raise FederationBoundsError("promotion evidence bundle exceeds its byte bound")

    @property
    def bundle_id(self) -> str:
        return "promotion-evidence:" + _json_content_ref(
            self.to_dict(include_identity=False),
            "promotion evidence bundle identity",
            maximum=MAX_BUNDLE_BYTES * 2,
        )

    def artifact(self, slot: EvidenceSlot) -> dict[str, Any] | None:
        if type(slot) is not EvidenceSlot:
            raise PromotionGateError("artifact slot must be a closed exact value")
        return _artifact_dict(getattr(self, slot.value), slot.value)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "identity_id": self.identity_id,
            **{slot.value: self.artifact(slot) for slot in EvidenceSlot},
        }
        if include_identity:
            value["bundle_id"] = self.bundle_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualificationEvidenceBundle:
        fields = frozenset(
            {"schema", "identity_id", "bundle_id", *(slot.value for slot in EvidenceSlot)}
        )
        data = _closed_dict(value, fields, "bundle")
        result = cls(
            identity_id=data["identity_id"],
            **{slot.value: data[slot.value] for slot in EvidenceSlot},
            schema=data["schema"],
        )
        if data["bundle_id"] != result.bundle_id:
            raise PromotionGateError("promotion evidence bundle content identity mismatches")
        return result


@dataclass(frozen=True)
class ArtifactAssessment:
    """Derived result of one owning-schema decoder; never caller admission."""

    slot: EvidenceSlot
    task_id: str
    schema_id: str
    artifact_id: str
    status: ArtifactStatus
    blockers: tuple[str, ...]
    authoritative: bool = False
    schema: str = ARTIFACT_ASSESSMENT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != ARTIFACT_ASSESSMENT_SCHEMA:
            raise PromotionGateError("unsupported artifact assessment schema")
        if (
            type(self.slot) is not EvidenceSlot
            or type(self.task_id) is not str
            or _SLOT_TASK[self.slot] != self.task_id
        ):
            raise PromotionGateError("assessment task does not match its closed slot")
        if type(self.status) is not ArtifactStatus:
            raise PromotionGateError("assessment status is not closed")
        if type(self.schema_id) is not str or type(self.artifact_id) is not str:
            raise PromotionGateError("assessment identifiers must be exact text")
        if self.schema_id:
            _token(self.schema_id, "assessment.schema_id")
        if self.artifact_id and _CONTENT_REF.fullmatch(self.artifact_id) is None:
            raise PromotionGateError("assessment artifact_id is not content addressed")
        if type(self.blockers) is not tuple:
            raise PromotionGateError("assessment blockers must be an exact tuple")
        blockers = _canonical_tokens(self.blockers, "assessment.blockers", maximum=32)
        object.__setattr__(self, "blockers", blockers)
        if (self.status is ArtifactStatus.PASSED) != (not blockers):
            raise PromotionGateError("assessment status disagrees with blockers")
        if self.authoritative is not False:
            raise FederationAuthorityError("artifact assessment cannot create authority")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "slot": self.slot.value,
            "task_id": self.task_id,
            "schema_id": self.schema_id,
            "artifact_id": self.artifact_id,
            "status": self.status.value,
            "blockers": list(self.blockers),
            "authoritative": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactAssessment:
        fields = frozenset(
            {
                "schema",
                "slot",
                "task_id",
                "schema_id",
                "artifact_id",
                "status",
                "blockers",
                "authoritative",
            }
        )
        data = _closed_dict(value, fields, "assessment")
        if data["authoritative"] is not False:
            raise FederationAuthorityError("assessment has unsafe authority")
        try:
            blockers = _exact_list(data["blockers"], "assessment.blockers", maximum=32)
            return cls(
                slot=EvidenceSlot(data["slot"]),
                task_id=data["task_id"],
                schema_id=data["schema_id"],
                artifact_id=data["artifact_id"],
                status=ArtifactStatus(data["status"]),
                blockers=tuple(blockers),
                schema=data["schema"],
            )
        except PromotionGateError:
            raise
        except (TypeError, ValueError) as exc:
            raise PromotionGateError("artifact assessment is malformed") from exc


def _artifact_identity(payload: dict[str, Any]) -> str:
    return _json_content_ref(
        payload, "promotion evidence artifact identity", maximum=MAX_ARTIFACT_BYTES
    )


def _assessment(
    slot: EvidenceSlot,
    payload: dict[str, Any] | None,
    status: ArtifactStatus,
    *blockers: str,
) -> ArtifactAssessment:
    schema_id = payload.get("schema", "") if payload is not None else ""
    return ArtifactAssessment(
        slot=slot,
        task_id=_SLOT_TASK[slot],
        schema_id=schema_id if type(schema_id) is str else "",
        artifact_id=_artifact_identity(payload) if payload is not None else "",
        status=status,
        blockers=tuple(sorted(blockers)),
    )


def _missing_provenance(task_id: str) -> tuple[str, ...]:
    slug = task_id.lower().replace("-", "_")
    return (
        f"missing:{slug}_accepted_producer_provenance",
        f"missing:{slug}_full_qualification_identity_binding",
        f"missing:{slug}_state_owner_provenance",
    )


def _schema(payload: dict[str, Any], expected: str, label: str) -> None:
    if payload.get("schema") != expected:
        raise PromotionGateError(f"{label} schema is unsupported")


def _assess_fixed_point(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.FIXED_POINT
    if payload is None:
        return _assessment(slot, None, ArtifactStatus.MISSING, "missing:casf_030_fixed_point")
    try:
        fields = frozenset(
            {
                "schema",
                "world_snapshot_ref",
                "event_watermark",
                "outstanding_required_work",
                "fencing_epoch",
                "outcome",
                "evidence_refs",
                "receipt_id",
            }
        )
        data = _closed_dict(payload, fields, "CASF-030 fixed-point receipt")
        _schema(data, _FIXED_POINT_SCHEMA, "CASF-030 fixed-point receipt")
        evidence_refs = _exact_list(
            data["evidence_refs"], "CASF-030 evidence_refs", maximum=MAX_CAPABILITIES
        )
        if evidence_refs != data["evidence_refs"]:
            raise PromotionGateError("CASF-030 evidence references are not canonical")
        receipt = FixedPointReceipt.from_dict(data)
        if (
            receipt.world_snapshot_ref != identity.world_snapshot_ref
            or receipt.event_watermark != identity.event_watermark
            or receipt.fencing_epoch != identity.fencing_epoch
        ):
            return _assessment(
                slot,
                payload,
                ArtifactStatus.BLOCKED,
                "stale:casf_030_fixed_point_identity",
            )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            *_missing_provenance("CASF-030"),
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(slot, payload, ArtifactStatus.INVALID, "invalid:casf_030_fixed_point")


def _decode_projection(payload: dict[str, Any]) -> ProjectionReceipt:
    fields = frozenset(
        {
            "schema",
            "status",
            "source_root",
            "tree_id",
            "from_watermark",
            "to_watermark",
            "source_checksum",
            "cursor_watermark",
            "partition_ids",
            "authoritative",
            "receipt_id",
        }
    )
    data = _closed_dict(payload, fields, "CASF-032 projection receipt")
    _schema(data, _DUCKLAKE_PROJECTION_SCHEMA, "CASF-032 projection receipt")
    partition_ids = _exact_list(
        data["partition_ids"], "CASF-032 partition_ids", maximum=MAX_JSON_CONTAINER_ITEMS
    )
    receipt = ProjectionReceipt(
        status=data["status"],
        source_root=data["source_root"],
        tree_id=data["tree_id"],
        from_watermark=data["from_watermark"],
        to_watermark=data["to_watermark"],
        source_checksum=data["source_checksum"],
        cursor_watermark=data["cursor_watermark"],
        partition_ids=tuple(partition_ids),
        authoritative=data["authoritative"],
    )
    if data["receipt_id"] != receipt.cid:
        raise PromotionGateError("CASF-032 projection receipt identity mismatches")
    return receipt


def _assess_projection(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.DUCKLAKE_PROJECTION
    if payload is None:
        return _assessment(
            slot, None, ArtifactStatus.MISSING, "missing:casf_032_ducklake_projection"
        )
    try:
        receipt = _decode_projection(payload)
        if receipt.tree_id != identity.tree_id:
            return _assessment(
                slot, payload, ArtifactStatus.BLOCKED, "stale:casf_032_projection_tree"
            )
        if receipt.status != "current":
            return _assessment(
                slot,
                payload,
                ArtifactStatus.UNAVAILABLE,
                f"unavailable:casf_032_projection_{receipt.status}",
            )
        if (
            receipt.cursor_watermark != receipt.to_watermark
            or not receipt.partition_ids
            or not receipt.source_checksum
        ):
            return _assessment(
                slot,
                payload,
                ArtifactStatus.BLOCKED,
                "blocked:casf_032_projection_incomplete",
            )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            *_missing_provenance("CASF-032"),
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(
            slot, payload, ArtifactStatus.INVALID, "invalid:casf_032_ducklake_projection"
        )


def _assess_recovery(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.DUCKLAKE_RECOVERY
    if payload is None:
        return _assessment(slot, None, ArtifactStatus.MISSING, "missing:casf_032_ducklake_recovery")
    try:
        fields = frozenset(
            {
                "schema",
                "status",
                "tenant_id",
                "schema_revision",
                "recovered_from_watermark",
                "recovered_to_watermark",
                "preserved_partition_ids",
                "recovered_partition_ids",
                "rewritten",
                "authoritative",
                "receipt_id",
            }
        )
        data = _closed_dict(payload, fields, "CASF-032 recovery receipt")
        _schema(data, _DUCKLAKE_RECOVERY_SCHEMA, "CASF-032 recovery receipt")
        preserved_partition_ids = _exact_list(
            data["preserved_partition_ids"],
            "CASF-032 preserved_partition_ids",
            maximum=MAX_JSON_CONTAINER_ITEMS,
        )
        recovered_partition_ids = _exact_list(
            data["recovered_partition_ids"],
            "CASF-032 recovered_partition_ids",
            maximum=MAX_JSON_CONTAINER_ITEMS,
        )
        receipt = ProjectionRecoveryReceipt(
            status=data["status"],
            tenant_id=data["tenant_id"],
            schema_revision=data["schema_revision"],
            recovered_from_watermark=data["recovered_from_watermark"],
            recovered_to_watermark=data["recovered_to_watermark"],
            preserved_partition_ids=tuple(preserved_partition_ids),
            recovered_partition_ids=tuple(recovered_partition_ids),
            rewritten=data["rewritten"],
            authoritative=data["authoritative"],
        )
        if data["receipt_id"] != receipt.cid:
            raise PromotionGateError("CASF-032 recovery receipt identity mismatches")
        if receipt.tenant_id != identity.tenant_id:
            return _assessment(
                slot, payload, ArtifactStatus.BLOCKED, "stale:casf_032_recovery_tenant"
            )
        if receipt.status != "current":
            return _assessment(
                slot,
                payload,
                ArtifactStatus.UNAVAILABLE,
                f"unavailable:casf_032_recovery_{receipt.status}",
            )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            *_missing_provenance("CASF-032"),
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(
            slot, payload, ArtifactStatus.INVALID, "invalid:casf_032_ducklake_recovery"
        )


def _assess_drift(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.DRIFT
    if payload is None:
        return _assessment(slot, None, ArtifactStatus.MISSING, "missing:casf_033_drift_report")
    try:
        _schema(payload, DRIFT_REPORT_SCHEMA, "CASF-033 drift report")
        report = DriftReport.from_dict(payload)
        if report.to_dict() != payload:
            raise PromotionGateError("CASF-033 drift report is not exact canonical wire data")
        validate_current_drift_report(
            report,
            current_repository_tree_id=identity.tree_id,
            current_control_plane_generation=identity.control_plane_generation,
            require_drift_free=True,
        )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            *_missing_provenance("CASF-033"),
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(slot, payload, ArtifactStatus.INVALID, "invalid:casf_033_drift_report")


def _assess_control(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.CONTROL_AUDIT
    if payload is None:
        return _assessment(
            slot,
            None,
            ArtifactStatus.MISSING,
            "missing:casf_034_typed_state_owner_audit",
        )
    try:
        _schema(payload, FEDERATION_CONTROL_AUDIT_SCHEMA, "CASF-034 control audit")
        receipt = FederationControlAuditReceipt.from_dict(payload)
        if receipt.to_dict() != payload:
            raise PromotionGateError("CASF-034 control audit is not exact canonical wire data")
        if (
            receipt.control_plane_generation != identity.control_plane_generation
            or receipt.fencing_epoch != identity.fencing_epoch
        ):
            return _assessment(
                slot, payload, ArtifactStatus.BLOCKED, "stale:casf_034_control_audit"
            )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            "blocked:casf_034_current_state_owner_capability_unattested",
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(slot, payload, ArtifactStatus.INVALID, "invalid:casf_034_control_audit")


def _decode_control_parity_report(payload: dict[str, Any]) -> tuple[str, str]:
    fields = frozenset(
        {
            "schema",
            "source_revision",
            "source_tree",
            "task_id",
            "bounded",
            "authority_created",
            "cli_manifest",
            "mcp_manifest",
            "report_id",
        }
    )
    data = _closed_dict(payload, fields, "CASF-035 control parity report")
    _schema(data, CASF_CONTROL_PARITY_REPORT_SCHEMA, "CASF-035 control parity report")
    source_revision = _oid(data["source_revision"], "CASF-035 source_revision")
    source_tree = _oid(data["source_tree"], "CASF-035 source_tree")
    if _token(data["task_id"], "CASF-035 task_id") != "CASF-035":
        raise PromotionGateError("CASF-035 report has another task identity")
    _exact_boolean(data["bounded"], True, "CASF-035 bounded")
    _exact_boolean(data["authority_created"], False, "CASF-035 authority_created")

    # These are cold, deterministic producer manifests.  Exact canonical-byte
    # comparison rejects bool/int equality tricks and any invented field while
    # preserving the intentionally different CLI and MCP envelopes.
    expected_cli = federation_cli_discovery_manifest()
    expected_mcp = {
        "schema": _CONTROL_MCP_DISCOVERY_SCHEMA,
        "interface": _CONTROL_MCP_INTERFACE,
        "category": "agent_supervisor",
        "dispatch": expected_cli["dispatch"],
        "tools": {
            "federation_" + operation.removeprefix("federation."): operation
            for operation in sorted(expected_cli["commands"].values())
        },
        "request_schemas": expected_cli["request_schemas"],
        "result_schema": _CONTROL_MCP_RESULT_SCHEMA,
        "error_schema": _CONTROL_MCP_ERROR_SCHEMA,
        "shell_out": False,
        "embedded_fallback": False,
        "create_via_trigger_gateway": expected_cli["create_via_trigger_gateway"],
        "max_canonical_bytes": expected_cli["max_canonical_bytes"],
    }
    cli_manifest = _closed_dict(
        data["cli_manifest"], frozenset(expected_cli), "CASF-035 CLI manifest"
    )
    mcp_manifest = _closed_dict(
        data["mcp_manifest"], frozenset(expected_mcp), "CASF-035 MCP manifest"
    )
    _exact_wire(cli_manifest, expected_cli, "CASF-035 CLI manifest")
    _exact_wire(mcp_manifest, expected_mcp, "CASF-035 MCP manifest")

    cli_operations = tuple(sorted(cli_manifest["commands"].values()))
    mcp_operations = tuple(sorted(mcp_manifest["tools"].values()))
    if cli_operations != mcp_operations:
        raise PromotionGateError("CASF-035 operation catalogs are not parity-equivalent")
    for name in (
        "request_schemas",
        "dispatch",
        "shell_out",
        "embedded_fallback",
        "create_via_trigger_gateway",
        "max_canonical_bytes",
    ):
        _exact_wire(
            cli_manifest[name],
            mcp_manifest[name],
            f"CASF-035 shared {name}",
        )

    report_id = _content_ref(data["report_id"], "CASF-035 report_id")
    body = {name: value for name, value in data.items() if name != "report_id"}
    if report_id != content_identity(body):
        raise PromotionGateError("CASF-035 report content identity mismatches")
    return source_revision, source_tree


def _assess_control_parity(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.CONTROL_PARITY
    if payload is None:
        return _assessment(
            slot,
            None,
            ArtifactStatus.MISSING,
            "missing:casf_035_control_parity_report",
        )
    try:
        source_revision, source_tree = _decode_control_parity_report(payload)
        if source_revision != identity.revision or source_tree != identity.tree_id:
            return _assessment(
                slot,
                payload,
                ArtifactStatus.BLOCKED,
                "stale:casf_035_control_parity_identity",
            )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            *_missing_provenance("CASF-035"),
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(
            slot,
            payload,
            ArtifactStatus.INVALID,
            "invalid:casf_035_control_parity_report",
        )


def _decode_formal_bounds(value: Any) -> ModelCheckBounds:
    fields = frozenset({"schema", *_FORMAL_BOUND_LIMITS})
    data = _closed_dict(value, fields, "CASF-036 formal bounds")
    for name, maximum in _FORMAL_BOUND_LIMITS.items():
        minimum = 0 if name == "max_retries" else 1
        if _integer(data[name], f"CASF-036 bounds.{name}", minimum=minimum) > maximum:
            raise FederationBoundsError(f"CASF-036 bounds.{name} exceeds its bound")
    bounds = ModelCheckBounds.from_dict(data)
    _exact_wire(data, bounds.to_dict(), "CASF-036 formal bounds")
    return bounds


def _decode_external_formal_receipts(
    value: Any,
    *,
    suite: Any,
    hermetic_receipts: tuple[Any, ...],
) -> tuple[ExternalModelCheckReceipt, ...]:
    payloads = _exact_list(
        value,
        "CASF-036 external receipts",
        maximum=MAX_FORMAL_EXTERNAL_RECEIPTS,
    )
    if payloads and len(payloads) != MAX_FORMAL_EXTERNAL_RECEIPTS:
        raise PromotionGateError("CASF-036 external receipt matrix is incomplete")
    fields = frozenset(
        {
            "schema",
            "scenario_id",
            "scenario_property",
            "property",
            "property_scope",
            "external_model_satisfies_casf_property_alone",
            "tool",
            "status",
            "ran",
            "bounded",
            "unbounded_proof",
            "authority_created",
            "matrix_snapshot_id",
            "matrix_entry_state",
            "generated_model_identity",
            "model_check_receipt_id",
            "paired_hermetic_receipt_id",
            "paired_hermetic_status",
            "casf_property_satisfied_by_pair",
            "reason",
            "receipt_id",
        }
    )
    hermetic_by_property = {item.property: item for item in hermetic_receipts}
    results: list[ExternalModelCheckReceipt] = []
    observed: list[tuple[FederationFormalProperty, ModelCheckerTool]] = []
    closed_matrix_states = frozenset(item.value for item in ProverState)
    for index, payload in enumerate(payloads):
        data = _closed_dict(payload, fields, f"CASF-036 external receipt[{index}]")
        _schema(data, CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA, "CASF-036 external receipt")
        _exact_boolean(data["bounded"], True, "CASF-036 external bounded")
        _exact_boolean(data["unbounded_proof"], False, "CASF-036 external unbounded_proof")
        _exact_boolean(
            data["authority_created"], False, "CASF-036 external authority_created"
        )
        _exact_boolean(
            data["external_model_satisfies_casf_property_alone"],
            False,
            "CASF-036 external standalone satisfaction",
        )
        if data["property_scope"] != "generic_supervisor_state_model":
            raise PromotionGateError("CASF-036 external property scope is unsupported")
        for name in (
            "scenario_id",
            "matrix_snapshot_id",
            "generated_model_identity",
            "receipt_id",
        ):
            _content_ref(data[name], f"CASF-036 external {name}")
        for name in ("model_check_receipt_id", "paired_hermetic_receipt_id"):
            if data[name]:
                _content_ref(data[name], f"CASF-036 external {name}")
            elif type(data[name]) is not str:
                raise PromotionGateError(f"CASF-036 external {name} must be exact text")
        matrix_entry_state = _token(
            data["matrix_entry_state"], "CASF-036 external matrix_entry_state"
        )
        if matrix_entry_state not in closed_matrix_states:
            raise PromotionGateError("CASF-036 external matrix state is not closed")
        _bounded_report_text(data["reason"], "CASF-036 external reason")

        scenario_property = FederationFormalProperty(data["scenario_property"])
        tool = ModelCheckerTool(data["tool"])
        paired_status = (
            None
            if data["paired_hermetic_status"] is None
            else HermeticCheckStatus(data["paired_hermetic_status"])
        )
        receipt = ExternalModelCheckReceipt(
            scenario_id=data["scenario_id"],
            scenario_property=scenario_property,
            property=ExternalModelInvariant(data["property"]),
            tool=tool,
            status=ExternalCheckStatus(data["status"]),
            ran=data["ran"],
            matrix_snapshot_id=data["matrix_snapshot_id"],
            matrix_entry_state=matrix_entry_state,
            generated_model_identity=data["generated_model_identity"],
            model_check_receipt_id=data["model_check_receipt_id"],
            paired_hermetic_receipt_id=data["paired_hermetic_receipt_id"],
            paired_hermetic_status=paired_status,
            casf_property_satisfied_by_pair=data["casf_property_satisfied_by_pair"],
            reason=data["reason"],
            schema=data["schema"],
        )
        _exact_wire(data, receipt.to_dict(), f"CASF-036 external receipt[{index}]")
        scenario = suite.scenario(scenario_property)
        if (
            receipt.scenario_id != scenario.scenario_id
            or receipt.generated_model_identity
            != scenario.generated_model.artifact_identity
        ):
            raise PromotionGateError("CASF-036 external receipt binds another scenario")
        paired = hermetic_by_property[scenario_property]
        if receipt.paired_hermetic_receipt_id and (
            receipt.paired_hermetic_receipt_id != paired.receipt_id
            or receipt.paired_hermetic_status is not paired.status
        ):
            raise PromotionGateError("CASF-036 external receipt has a stale hermetic pair")
        results.append(receipt)
        observed.append((scenario_property, tool))

    expected_order = [
        (scenario.property, tool)
        for scenario in suite.scenarios
        for tool in (ModelCheckerTool.TLC, ModelCheckerTool.APALACHE)
    ]
    if payloads and observed != expected_order:
        raise PromotionGateError("CASF-036 external receipt matrix is not canonical")
    return tuple(results)


def _decode_formal_model_report(
    payload: dict[str, Any],
) -> tuple[FederationFormalIdentity, tuple[Any, ...], tuple[ExternalModelCheckReceipt, ...]]:
    fields = frozenset(
        {
            "schema",
            "bounded",
            "unbounded_proof",
            "authority_created",
            "identity",
            "suite",
            "hermetic_receipts",
            "external_receipts",
            "report_id",
        }
    )
    data = _closed_dict(payload, fields, "CASF-036 formal model report")
    _schema(data, CASF_FORMAL_MODEL_REPORT_SCHEMA, "CASF-036 formal model report")
    _exact_boolean(data["bounded"], True, "CASF-036 bounded")
    _exact_boolean(data["unbounded_proof"], False, "CASF-036 unbounded_proof")
    _exact_boolean(data["authority_created"], False, "CASF-036 authority_created")

    identity_fields = frozenset(
        {
            "schema",
            "source_revision",
            "source_tree",
            "state_schema",
            "generation_id",
            "policy_id",
            "policy_revision",
            "capability_ids",
            "federation_id",
            "supervisor_ids",
            "task_id",
            "attempt_id",
            "lease_id",
            "fencing_epoch",
            "assignment_revision",
            "worktree_id",
            "identity",
        }
    )
    identity_data = _closed_dict(data["identity"], identity_fields, "CASF-036 identity")
    _schema(identity_data, CASF_FORMAL_IDENTITY_SCHEMA, "CASF-036 identity")
    formal_identity = FederationFormalIdentity.from_dict(identity_data)
    _exact_wire(identity_data, formal_identity.to_dict(), "CASF-036 identity")

    suite_fields = frozenset(
        {
            "schema",
            "bounded",
            "unbounded_proof",
            "authority_created",
            "identity",
            "bounds",
            "scenarios",
            "suite_id",
        }
    )
    suite_data = _closed_dict(data["suite"], suite_fields, "CASF-036 formal suite")
    _schema(suite_data, CASF_FORMAL_SUITE_SCHEMA, "CASF-036 formal suite")
    _exact_boolean(suite_data["bounded"], True, "CASF-036 suite bounded")
    _exact_boolean(suite_data["unbounded_proof"], False, "CASF-036 suite unbounded_proof")
    _exact_boolean(
        suite_data["authority_created"], False, "CASF-036 suite authority_created"
    )
    _exact_wire(suite_data["identity"], identity_data, "CASF-036 suite identity")
    bounds = _decode_formal_bounds(suite_data["bounds"])
    suite = build_federation_formal_suite(formal_identity, bounds=bounds)
    _exact_wire(suite_data, suite.to_dict(), "CASF-036 formal suite")

    supplied_hermetic = _exact_list(
        data["hermetic_receipts"],
        "CASF-036 hermetic receipts",
        maximum=len(FederationFormalProperty),
    )
    trusted_hermetic = check_federation_formal_suite(suite)
    _exact_wire(
        supplied_hermetic,
        [item.to_dict() for item in trusted_hermetic],
        "CASF-036 hermetic receipts",
    )
    external = _decode_external_formal_receipts(
        data["external_receipts"],
        suite=suite,
        hermetic_receipts=trusted_hermetic,
    )
    report_id = _content_ref(data["report_id"], "CASF-036 report_id")
    body = {name: value for name, value in data.items() if name != "report_id"}
    if report_id != content_identity(body):
        raise PromotionGateError("CASF-036 report content identity mismatches")
    return formal_identity, trusted_hermetic, external


def _assess_formal(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.FORMAL
    if payload is None:
        return _assessment(
            slot,
            None,
            ArtifactStatus.MISSING,
            "missing:casf_036_formal_model_report",
        )
    try:
        formal_identity, hermetic, external = _decode_formal_model_report(payload)
        if (
            formal_identity.source_revision != identity.revision
            or formal_identity.source_tree != identity.tree_id
            or formal_identity.state_schema != identity.schema_id
            or formal_identity.generation_id != identity.generation_id
            or formal_identity.federation_id != identity.federation_id
            or formal_identity.policy_id != identity.policy_id
            or formal_identity.policy_revision != identity.policy_revision
            or formal_identity.capability_ids != identity.capability_ids
            or formal_identity.fencing_epoch != identity.fencing_epoch
            or formal_identity.assignment_revision != identity.assignment_revision
        ):
            return _assessment(
                slot,
                payload,
                ArtifactStatus.BLOCKED,
                "stale:casf_036_formal_identity",
            )
        if not all(item.status is HermeticCheckStatus.PASSED for item in hermetic):
            return _assessment(
                slot,
                payload,
                ArtifactStatus.BLOCKED,
                "blocked:casf_036_hermetic_checks_incomplete",
            )
        if any(item.status is ExternalCheckStatus.COUNTEREXAMPLE for item in external):
            return _assessment(
                slot,
                payload,
                ArtifactStatus.BLOCKED,
                "blocked:casf_036_external_counterexample",
            )
        return _assessment(
            slot,
            payload,
            ArtifactStatus.NONAUTHORITATIVE,
            *_missing_provenance("CASF-036"),
        )
    except (KeyError, TypeError, ValueError, FederationFormalError, FederationContractError):
        return _assessment(
            slot,
            payload,
            ArtifactStatus.INVALID,
            "invalid:casf_036_formal_model_report",
        )


def _assess_adversarial(
    identity: QualificationIdentity, payload: dict[str, Any] | None
) -> ArtifactAssessment:
    slot = EvidenceSlot.ADVERSARIAL
    if payload is None:
        return _assessment(
            slot, None, ArtifactStatus.MISSING, "missing:casf_037_adversarial_report"
        )
    try:
        _schema(payload, CASF_CHAOS_REPORT_SCHEMA, "CASF-037 adversarial report")
        report = ChaosReport.from_dict(payload)
        if report.to_dict() != payload:
            raise PromotionGateError("CASF-037 report is not exact canonical wire data")
        source = report.suite.identity
        if (
            source.source_revision != identity.revision
            or source.source_tree != identity.tree_id
            or source.state_schema != identity.schema_id
            or source.generation_id != identity.generation_id
            or source.federation_id != identity.federation_id
            or source.policy_id != identity.policy_id
            or source.policy_revision != identity.policy_revision
            or source.capability_ids != identity.capability_ids
        ):
            return _assessment(
                slot, payload, ArtifactStatus.BLOCKED, "stale:casf_037_adversarial_identity"
            )
        if (
            report.qualified
            or report.promotion_eligible
            or payload.get("local_qualification_available") is not False
            or payload.get("upstream_reverification_required") is not True
        ):
            raise PromotionGateError("CASF-037 report overstates local qualification")
        return _assessment(
            slot,
            payload,
            ArtifactStatus.BLOCKED,
            "blocked:casf_037_local_qualification_unavailable",
        )
    except (KeyError, TypeError, ValueError, FederationContractError):
        return _assessment(
            slot, payload, ArtifactStatus.INVALID, "invalid:casf_037_adversarial_report"
        )


def _load_pinned_benchmark_manifest(task_id: str) -> dict[str, Any]:
    """Read one fixed data artifact without following links or executing code."""

    filename = _BENCHMARK_MANIFEST_FILES[task_id]
    if Path(filename).name != filename:
        raise PromotionGateError(f"{task_id} manifest path escapes its closed root")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if type(no_follow) is not int or type(directory_flag) is not int:
        raise PromotionGateError("platform cannot enforce no-follow manifest reads")
    common_flags = no_follow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    root_descriptor = os.open(_BENCHMARK_ROOT, os.O_RDONLY | directory_flag | common_flags)
    try:
        descriptor = os.open(filename, os.O_RDONLY | common_flags, dir_fd=root_descriptor)
    finally:
        os.close(root_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PromotionGateError(f"{task_id} manifest is not a regular file")
        if metadata.st_size < 1 or metadata.st_size > MAX_BENCHMARK_MANIFEST_BYTES:
            raise FederationBoundsError(f"{task_id} manifest exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = MAX_BENCHMARK_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_BENCHMARK_MANIFEST_BYTES:
        raise FederationBoundsError(f"{task_id} manifest exceeds its byte bound")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _BENCHMARK_MANIFEST_SHA256[task_id]:
        raise PromotionGateError(f"{task_id} manifest does not match its pinned hash")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionGateError(f"{task_id} manifest is not exact JSON") from exc
    if type(decoded) is not dict:
        raise PromotionGateError(f"{task_id} manifest is not an exact object")
    _canonical_artifact(decoded, f"{task_id} pinned manifest")
    return decoded


def _validate_pinned_benchmark_manifest(task_id: str, payload: dict[str, Any]) -> None:
    canonical = _load_pinned_benchmark_manifest(task_id)
    if payload != canonical:
        raise PromotionGateError(f"{task_id} manifest differs from its pinned artifact")
    expected_state = "specification_only" if task_id == "CASF-038" else "capability_unavailable"
    if (
        payload.get("schema") != _BENCHMARK_MANIFEST_SCHEMAS[task_id]
        or payload.get("objective_id") != task_id
        or payload.get("frozen") is not True
        or payload.get("state") != expected_state
        or payload.get("authoritative") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise PromotionGateError(f"{task_id} manifest overstates its current authority")
    live_key = "typed_quack_live" if task_id == "CASF-038" else "live_capability"
    live = payload.get(live_key)
    if type(live) is not dict or (
        live.get("availability") != "unavailable"
        or live.get("execution_status") != "not_run"
        or live.get("metrics_omitted") is not True
        or ("ran" in live and live.get("ran") is not False)
        or ("qualified" in live and live.get("qualified") is not False)
    ):
        raise PromotionGateError(f"{task_id} current unavailable outcome changed")


def _assess_benchmark(
    slot: EvidenceSlot,
    payload: dict[str, Any] | None,
) -> ArtifactAssessment:
    task_id = _SLOT_TASK[slot]
    suffix = task_id.lower().replace("-", "_")
    if payload is None:
        return _assessment(slot, None, ArtifactStatus.MISSING, f"missing:{suffix}_benchmark")
    try:
        schema = payload.get("schema")
        if schema == _BENCHMARK_MANIFEST_SCHEMAS[task_id]:
            _validate_pinned_benchmark_manifest(task_id, payload)
            return _assessment(
                slot,
                payload,
                ArtifactStatus.UNAVAILABLE,
                f"unavailable:{suffix}_live_not_run",
            )
        if schema == _BENCHMARK_RESULT_SCHEMAS[task_id]:
            return _assessment(
                slot,
                payload,
                ArtifactStatus.INVALID,
                f"unsupported:{suffix}_pure_result_decoder_unavailable",
            )
        raise PromotionGateError(f"{task_id} artifact schema is unsupported")
    except (KeyError, TypeError, ValueError, OSError, FederationContractError):
        return _assessment(slot, payload, ArtifactStatus.INVALID, f"invalid:{suffix}_benchmark")


def required_slots(profile: GateProfile) -> tuple[EvidenceSlot, ...]:
    if type(profile) is not GateProfile:
        raise PromotionGateError("profile must be a closed exact enum value")
    return _CORE_REQUIRED_SLOTS + (
        _DUCKLAKE_REQUIRED_SLOTS if profile is GateProfile.DUCKLAKE else ()
    )


def _assess_bundle(
    identity: QualificationIdentity,
    profile: GateProfile,
    bundle: QualificationEvidenceBundle,
) -> tuple[ArtifactAssessment, ...]:
    if type(identity) is not QualificationIdentity:
        raise PromotionGateError("evaluation requires an exact qualification identity")
    if type(bundle) is not QualificationEvidenceBundle:
        raise PromotionGateError("evaluation requires an exact evidence bundle")
    if bundle.identity_id != identity.identity_id:
        raise StaleQualificationEvidenceError("evidence bundle binds another identity")
    if type(profile) is not GateProfile:
        raise PromotionGateError("profile must be a closed exact enum value")

    payloads = {slot: bundle.artifact(slot) for slot in EvidenceSlot}
    assessments = {
        EvidenceSlot.FIXED_POINT: _assess_fixed_point(identity, payloads[EvidenceSlot.FIXED_POINT]),
        EvidenceSlot.DUCKLAKE_PROJECTION: _assess_projection(
            identity, payloads[EvidenceSlot.DUCKLAKE_PROJECTION]
        ),
        EvidenceSlot.DUCKLAKE_RECOVERY: _assess_recovery(
            identity, payloads[EvidenceSlot.DUCKLAKE_RECOVERY]
        ),
        EvidenceSlot.DRIFT: _assess_drift(identity, payloads[EvidenceSlot.DRIFT]),
        EvidenceSlot.CONTROL_AUDIT: _assess_control(identity, payloads[EvidenceSlot.CONTROL_AUDIT]),
        EvidenceSlot.CONTROL_PARITY: _assess_control_parity(
            identity, payloads[EvidenceSlot.CONTROL_PARITY]
        ),
        EvidenceSlot.FORMAL: _assess_formal(identity, payloads[EvidenceSlot.FORMAL]),
        EvidenceSlot.ADVERSARIAL: _assess_adversarial(identity, payloads[EvidenceSlot.ADVERSARIAL]),
        EvidenceSlot.IDLE: _assess_benchmark(EvidenceSlot.IDLE, payloads[EvidenceSlot.IDLE]),
        EvidenceSlot.PARALLEL: _assess_benchmark(
            EvidenceSlot.PARALLEL, payloads[EvidenceSlot.PARALLEL]
        ),
        EvidenceSlot.LOAD: _assess_benchmark(EvidenceSlot.LOAD, payloads[EvidenceSlot.LOAD]),
        EvidenceSlot.TOKEN: _assess_benchmark(EvidenceSlot.TOKEN, payloads[EvidenceSlot.TOKEN]),
    }
    return tuple(assessments[slot] for slot in EvidenceSlot)


def _blockers_for(
    profile: GateProfile, assessments: tuple[ArtifactAssessment, ...]
) -> tuple[str, ...]:
    required = set(required_slots(profile))
    return tuple(
        sorted(
            {
                blocker
                for assessment in assessments
                if assessment.slot in required
                for blocker in assessment.blockers
            }
        )
    )


def _decision_schema(kind: DecisionKind) -> str:
    return {
        DecisionKind.PROMOTION: PROMOTION_DECISION_SCHEMA,
        DecisionKind.ROLLBACK: ROLLBACK_DECISION_SCHEMA,
        DecisionKind.QUARANTINE: QUARANTINE_DECISION_SCHEMA,
    }[kind]


def _decision_prefix(kind: DecisionKind) -> str:
    return {
        DecisionKind.PROMOTION: "promotion-decision",
        DecisionKind.ROLLBACK: "rollback-decision",
        DecisionKind.QUARANTINE: "quarantine-decision",
    }[kind]


def _validate_rollback_target(
    active: QualificationIdentity, predecessor: QualificationIdentity
) -> None:
    if (
        active.tenant_id != predecessor.tenant_id
        or active.federation_id != predecessor.federation_id
        or active.repository_id != predecessor.repository_id
    ):
        raise PromotionGateError("rollback target crosses its tenant/federation/repository")
    if active.revision == predecessor.revision or active.tree_id == predecessor.tree_id:
        raise PromotionGateError("rollback target must be a distinct predecessor")
    if active.control_plane_generation == predecessor.control_plane_generation:
        raise PromotionGateError("rollback target must restore a predecessor generation")
    if active.fencing_epoch == predecessor.fencing_epoch:
        raise PromotionGateError("rollback target must use a distinct fenced authority")
    if active.lease_id == predecessor.lease_id:
        raise PromotionGateError("rollback target must use a distinct lease")


@dataclass(frozen=True)
class GateDecision:
    """Deterministic non-authoritative recommendation with its source bundle."""

    kind: DecisionKind
    identity: QualificationIdentity
    profile: GateProfile
    bundle: QualificationEvidenceBundle
    assessments: tuple[ArtifactAssessment, ...]
    blockers: tuple[str, ...]
    status: DecisionStatus
    disposition: DecisionDisposition
    rollback_target: QualificationIdentity | None = None
    schema: str = PROMOTION_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not DecisionKind
            or type(self.schema) is not str
            or self.schema != _decision_schema(self.kind)
        ):
            raise PromotionGateError("decision kind and schema disagree")
        if type(self.identity) is not QualificationIdentity:
            raise PromotionGateError("decision requires an exact identity")
        if type(self.profile) is not GateProfile or type(self.status) is not DecisionStatus:
            raise PromotionGateError("decision profile/status is not closed")
        if type(self.disposition) is not DecisionDisposition:
            raise PromotionGateError("decision disposition is not closed")
        if type(self.assessments) is not tuple or type(self.blockers) is not tuple:
            raise PromotionGateError("decision assessments/blockers must be exact tuples")
        expected_assessments = _assess_bundle(self.identity, self.profile, self.bundle)
        if self.assessments != expected_assessments:
            raise PromotionGateError("decision assessments were not derived from artifacts")
        expected_blockers = _blockers_for(self.profile, expected_assessments)
        if self.kind is DecisionKind.ROLLBACK:
            if type(self.rollback_target) is not QualificationIdentity:
                raise PromotionGateError("rollback decision requires an exact predecessor")
            _validate_rollback_target(self.identity, self.rollback_target)
            expected_blockers = tuple(
                sorted((*expected_blockers, "missing:rollback_state_owner_predecessor_receipt"))
            )
        elif self.rollback_target is not None:
            raise PromotionGateError("only rollback may name a predecessor")
        if self.blockers != expected_blockers:
            raise PromotionGateError("decision blockers disagree with decoded artifacts")
        expected_status = (
            DecisionStatus.PERMITTED if not expected_blockers else DecisionStatus.BLOCKED
        )
        if self.status is not expected_status:
            raise PromotionGateError("decision status disagrees with blockers")
        expected_disposition = (
            {
                DecisionKind.PROMOTION: DecisionDisposition.PROMOTION_RECOMMENDED,
                DecisionKind.ROLLBACK: DecisionDisposition.ROLLBACK_RECOMMENDED,
                DecisionKind.QUARANTINE: DecisionDisposition.QUARANTINE_REQUIRED,
            }[self.kind]
            if not expected_blockers
            else DecisionDisposition.QUARANTINE_REQUIRED
        )
        if self.disposition is not expected_disposition:
            raise PromotionGateError("decision disposition disagrees with blockers")

    @property
    def permitted(self) -> bool:
        return self.status is DecisionStatus.PERMITTED

    @property
    def decision_id(self) -> str:
        return (
            _decision_prefix(self.kind)
            + ":"
            + _json_content_ref(
                self.to_dict(include_identity=False),
                "promotion gate decision identity",
                maximum=MAX_BUNDLE_BYTES * 2,
            )
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        promotion_eligible = bool(self.permitted and self.kind is DecisionKind.PROMOTION)
        value: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind.value,
            "identity": self.identity.to_dict(),
            "profile": self.profile.value,
            "bundle": self.bundle.to_dict(),
            "assessments": [item.to_dict() for item in self.assessments],
            "blockers": list(self.blockers),
            "status": self.status.value,
            "disposition": self.disposition.value,
            "promotion_eligible": promotion_eligible,
            "release_eligible": promotion_eligible,
            "promotion_applied": False,
            "quarantine_required": self.disposition is DecisionDisposition.QUARANTINE_REQUIRED,
            "quarantine_applied": False,
            "rollback_applied": False,
            "production_state_changed": False,
            "authoritative_state_changed": False,
            "authority_created": False,
            "completion_created": False,
            "upstream_reverification_required": True,
        }
        if self.rollback_target is not None:
            value["rollback_target"] = self.rollback_target.to_dict()
        if include_identity:
            value["decision_id"] = self.decision_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateDecision:
        if type(value) is not dict:
            raise PromotionGateError("gate decision must be an exact object")
        base = {
            "schema",
            "kind",
            "identity",
            "profile",
            "bundle",
            "assessments",
            "blockers",
            "status",
            "disposition",
            "promotion_eligible",
            "release_eligible",
            "promotion_applied",
            "quarantine_required",
            "quarantine_applied",
            "rollback_applied",
            "production_state_changed",
            "authoritative_state_changed",
            "authority_created",
            "completion_created",
            "upstream_reverification_required",
            "decision_id",
        }
        fields = frozenset(base | ({"rollback_target"} if "rollback_target" in value else set()))
        data = _closed_dict(value, fields, "gate decision")
        try:
            assessments = _exact_list(
                data["assessments"],
                "gate decision assessments",
                maximum=len(EvidenceSlot),
            )
            blockers = _exact_list(
                data["blockers"], "gate decision blockers", maximum=MAX_CAPABILITIES
            )
            kind = DecisionKind(data["kind"])
            result = cls(
                kind=kind,
                identity=QualificationIdentity.from_dict(data["identity"]),
                profile=GateProfile(data["profile"]),
                bundle=QualificationEvidenceBundle.from_dict(data["bundle"]),
                assessments=tuple(ArtifactAssessment.from_dict(item) for item in assessments),
                blockers=tuple(blockers),
                status=DecisionStatus(data["status"]),
                disposition=DecisionDisposition(data["disposition"]),
                rollback_target=(
                    QualificationIdentity.from_dict(data["rollback_target"])
                    if "rollback_target" in data
                    else None
                ),
                schema=data["schema"],
            )
        except PromotionGateError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionGateError("gate decision is malformed") from exc
        expected = result.to_dict()
        if data != expected:
            unsafe = {
                "promotion_applied",
                "quarantine_applied",
                "rollback_applied",
                "production_state_changed",
                "authoritative_state_changed",
                "authority_created",
                "completion_created",
            }
            if any(data.get(name) is not False for name in unsafe):
                raise FederationAuthorityError("decision claims an applied or authoritative effect")
            raise PromotionGateError("gate decision fields or content identity mismatch")
        return result


def _decision(
    kind: DecisionKind,
    identity: QualificationIdentity,
    profile: GateProfile,
    bundle: QualificationEvidenceBundle,
    *,
    rollback_target: QualificationIdentity | None = None,
) -> GateDecision:
    assessments = _assess_bundle(identity, profile, bundle)
    blockers = _blockers_for(profile, assessments)
    if kind is DecisionKind.ROLLBACK:
        blockers = tuple(sorted((*blockers, "missing:rollback_state_owner_predecessor_receipt")))
    status = DecisionStatus.PERMITTED if not blockers else DecisionStatus.BLOCKED
    disposition = (
        {
            DecisionKind.PROMOTION: DecisionDisposition.PROMOTION_RECOMMENDED,
            DecisionKind.ROLLBACK: DecisionDisposition.ROLLBACK_RECOMMENDED,
            DecisionKind.QUARANTINE: DecisionDisposition.QUARANTINE_REQUIRED,
        }[kind]
        if not blockers
        else DecisionDisposition.QUARANTINE_REQUIRED
    )
    return GateDecision(
        kind=kind,
        identity=identity,
        profile=profile,
        bundle=bundle,
        assessments=assessments,
        blockers=blockers,
        status=status,
        disposition=disposition,
        rollback_target=rollback_target,
        schema=_decision_schema(kind),
    )


class FederationPromotionGate:
    """Recommendation facade; no method can apply a state transition."""

    INTERFACE: ClassVar[str] = FEDERATION_PROMOTION_GATE_INTERFACE

    @staticmethod
    def promote(
        identity: QualificationIdentity,
        profile: GateProfile,
        bundle: QualificationEvidenceBundle,
    ) -> GateDecision:
        return _decision(DecisionKind.PROMOTION, identity, profile, bundle)

    @staticmethod
    def rollback(
        active: QualificationIdentity,
        predecessor: QualificationIdentity,
        profile: GateProfile,
        bundle: QualificationEvidenceBundle,
    ) -> GateDecision:
        if (
            type(active) is not QualificationIdentity
            or type(predecessor) is not QualificationIdentity
        ):
            raise PromotionGateError("rollback requires exact active and predecessor identities")
        _validate_rollback_target(active, predecessor)
        return _decision(
            DecisionKind.ROLLBACK,
            active,
            profile,
            bundle,
            rollback_target=predecessor,
        )

    @staticmethod
    def quarantine(
        identity: QualificationIdentity,
        profile: GateProfile,
        bundle: QualificationEvidenceBundle,
    ) -> GateDecision:
        decision = _decision(DecisionKind.QUARANTINE, identity, profile, bundle)
        if not decision.blockers:
            raise PromotionGateError("quarantine requires a decoded qualification blocker")
        return decision


def evaluate_promotion(
    identity: QualificationIdentity,
    profile: GateProfile,
    bundle: QualificationEvidenceBundle,
) -> GateDecision:
    """Evaluate a closed real-artifact bundle; generic passed claims are invalid."""

    return FederationPromotionGate.promote(identity, profile, bundle)


def validate_current_decision(
    decision: GateDecision,
    *,
    current_identity: QualificationIdentity,
    require_permitted: bool = False,
) -> Mapping[str, Any]:
    """Re-decode artifacts and bind the recommendation to the current identity."""

    if type(decision) is not GateDecision:
        raise StaleQualificationEvidenceError("decision must be an exact gate decision")
    if type(current_identity) is not QualificationIdentity or current_identity != decision.identity:
        raise StaleQualificationEvidenceError("decision is stale for the current exact identity")
    reconstructed = _decision(
        decision.kind,
        decision.identity,
        decision.profile,
        decision.bundle,
        rollback_target=decision.rollback_target,
    )
    if reconstructed != decision or reconstructed.decision_id != decision.decision_id:
        raise StaleQualificationEvidenceError("decision content identity is invalid")
    if require_permitted and not decision.permitted:
        raise MissingQualificationCapabilityError("decision remains blocked")
    wire = decision.to_dict()
    return MappingProxyType(
        {
            "schema": DECISION_VALIDATION_SCHEMA,
            "decision_id": decision.decision_id,
            "current_identity_bound": True,
            "permitted": decision.permitted,
            "promotion_eligible": wire["promotion_eligible"],
            "release_eligible": wire["release_eligible"],
            "quarantine_required": wire["quarantine_required"],
            "promotion_applied": False,
            "quarantine_applied": False,
            "rollback_applied": False,
            "production_state_changed": False,
            "authoritative_state_changed": False,
            "authority_created": False,
            "completion_created": False,
            "upstream_reverification_required": True,
        }
    )


__all__ = [
    "ARTIFACT_ASSESSMENT_SCHEMA",
    "ArtifactAssessment",
    "ArtifactStatus",
    "CASF_CONTROL_PARITY_REPORT_SCHEMA",
    "CASF_FORMAL_MODEL_REPORT_SCHEMA",
    "DECISION_VALIDATION_SCHEMA",
    "DecisionDisposition",
    "DecisionKind",
    "DecisionStatus",
    "EvidenceSlot",
    "FEDERATION_PROMOTION_GATE_INTERFACE",
    "FederationPromotionGate",
    "GateDecision",
    "GateProfile",
    "MAX_ARTIFACT_BYTES",
    "MAX_BUNDLE_BYTES",
    "MissingQualificationCapabilityError",
    "PROMOTION_DECISION_SCHEMA",
    "PROMOTION_EVIDENCE_BUNDLE_SCHEMA",
    "PromotionGateError",
    "QUALIFICATION_IDENTITY_SCHEMA",
    "QUARANTINE_DECISION_SCHEMA",
    "QualificationEvidenceBundle",
    "QualificationIdentity",
    "ROLLBACK_DECISION_SCHEMA",
    "StaleQualificationEvidenceError",
    "evaluate_promotion",
    "required_slots",
    "validate_current_decision",
]
