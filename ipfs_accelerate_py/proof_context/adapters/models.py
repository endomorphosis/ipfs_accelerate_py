"""Exact v0.1 MCP++ wire-record bindings for the coding-agent adapter (PCCE-030).

These records are closed projections of the frozen task-specification,
coding-agent-invocation, patch-proposal, context-pack, and model-route-decision
schemas. They do not introduce a second wire schema or canonicalizer: emitted
bytes are RFC 8785 JCS for the admitted JSON subset (sorted keys, compact
separators, integers and strings only). Importing this module performs no I/O,
network, process, or filesystem mutation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.compatibility import (
    CompatibilityError,
    reject_pseudo_cid,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    MalformedError,
    PseudoCidError,
    SchemaMismatchError,
    UnknownFieldError,
)

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
CONTRACT_VERSION: Final[str] = "0.1"
CONTRACT_SCHEMA_PREFIX: Final[str] = "pcce/proof-context/v0.1/"

TASK_SPECIFICATION_SCHEMA: Final[str] = "pcce/proof-context/v0.1/task-specification"
CODING_AGENT_INVOCATION_SCHEMA: Final[str] = (
    "pcce/proof-context/v0.1/coding-agent-invocation"
)
PATCH_PROPOSAL_SCHEMA: Final[str] = "pcce/proof-context/v0.1/patch-proposal"
CONTEXT_PACK_SCHEMA: Final[str] = "pcce/proof-context/v0.1/context-pack"
MODEL_ROUTE_DECISION_SCHEMA: Final[str] = "pcce/proof-context/v0.1/model-route-decision"

WIRE_SCHEMAS: Final[tuple[str, ...]] = (
    TASK_SPECIFICATION_SCHEMA,
    CODING_AGENT_INVOCATION_SCHEMA,
    PATCH_PROPOSAL_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    MODEL_ROUTE_DECISION_SCHEMA,
)

PCCE_006_CONTENT_ID: Final[str] = (
    "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
)
TASK_SPECIFICATION_SCHEMA_DIGEST: Final[str] = (
    "sha256:4bec8e259e9634a7d60d09b956cb5f71c02b2c9c12d5e449db80d353d990f91d"
)
CODING_AGENT_INVOCATION_SCHEMA_DIGEST: Final[str] = (
    "sha256:c414a30750310968b814c549956b3576b779491df5451687778d8fbf61275149"
)
PATCH_PROPOSAL_SCHEMA_DIGEST: Final[str] = (
    "sha256:6789efd5f75d35861df24ae1b466adeddb35d13b3ef389a751acf3cbb8bdb241"
)
CONTEXT_PACK_SCHEMA_DIGEST: Final[str] = (
    "sha256:0a4966fccb7cba741c2519a05b5a40bd6bbc815f623f1b41c909ad987fd919e6"
)
MODEL_ROUTE_DECISION_SCHEMA_DIGEST: Final[str] = (
    "sha256:7d6eae1885d269c2aaee6fab9918d8ce060b0cf6a0b3bfba4c50627113837f92"
)

PROVENANCES: Final[tuple[str, ...]] = ("live", "replayed", "simulated")
SUFFICIENCIES: Final[tuple[str, ...]] = (
    "sufficient",
    "insufficient",
    "opaque",
    "stale",
    "unavailable",
)

MAX_ID_LENGTH: Final[int] = 128
MAX_MODEL_LENGTH: Final[int] = 256
MAX_REVISION_LENGTH: Final[int] = 256
MAX_TIER_LENGTH: Final[int] = 64
MAX_PATH_LENGTH: Final[int] = 1024
MAX_DECLARED_FILES: Final[int] = 1024
MAX_OWNED_PATHS: Final[int] = 1024
MAX_CAPSULE_CIDS: Final[int] = 4096
MAX_SAFE_INTEGER: Final[int] = 9007199254740991
MAX_PATCH_BYTES: Final[int] = 2_000_000
MAX_PROVIDER_OUTPUT_BYTES: Final[int] = 2_500_000
MAX_FILE_BYTES: Final[int] = 1_048_576
MAX_LOG_BYTES: Final[int] = 65_536
CID_MIN_LENGTH: Final[int] = 59
CID_MAX_LENGTH: Final[int] = 128

CID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{58,}$")

APPROVAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accepted",
        "approved",
        "self_approved",
        "adapter_approved",
        "published",
        "canonical_branch",
        "mutation_authority",
    }
)
HIDDEN_EVALUATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "evaluation_data",
        "hidden_prompt",
        "gold_labels",
        "credentials",
        "secret",
        "api_key",
    }
)
FORBIDDEN_WIRE_FIELDS: Final[frozenset[str]] = APPROVAL_FIELDS | HIDDEN_EVALUATION_FIELDS

TASK_SPECIFICATION_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "repository_state_cid",
        "objective_id",
        "owned_paths",
        "provenance",
    }
)
TASK_SPECIFICATION_OPTIONAL: Final[frozenset[str]] = frozenset(
    {"route_cid", "declared_files"}
)
INVOCATION_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "invocation_cid",
        "task_id",
        "repository_state_cid",
        "provider",
        "model",
        "revision",
        "tier",
        "provenance",
    }
)
INVOCATION_OPTIONAL: Final[frozenset[str]] = frozenset(
    {
        "route_cid",
        "token_count",
        "cached_token_count",
        "latency_ms",
        "cost_micros",
        "response_artifact_cid",
    }
)
INVOCATION_USAGE_FIELDS: Final[tuple[str, ...]] = (
    "token_count",
    "cached_token_count",
    "latency_ms",
    "cost_micros",
)
PATCH_PROPOSAL_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "proposal_cid",
        "task_id",
        "repository_state_cid",
        "declared_files",
        "provenance",
    }
)
PATCH_PROPOSAL_OPTIONAL: Final[frozenset[str]] = frozenset(
    {"invocation_cid", "patch_cid"}
)
CONTEXT_PACK_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "pack_cid",
        "repository_state_cid",
        "sufficiency",
        "provenance",
    }
)
CONTEXT_PACK_OPTIONAL: Final[frozenset[str]] = frozenset({"task_id", "capsule_cids"})
MODEL_ROUTE_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "decision_cid",
        "task_id",
        "tier",
        "provider",
        "model",
        "provenance",
    }
)
MODEL_ROUTE_OPTIONAL: Final[frozenset[str]] = frozenset(
    {"repository_state_cid", "revision"}
)


def wire_canonical_utf8(value: Any) -> str:
    """RFC 8785 JCS bytes for the admitted JSON subset used by v0.1 records."""

    return _canonicalize(value)


def wire_canonical_bytes(value: Any) -> bytes:
    return wire_canonical_utf8(value).encode("utf-8")


def _canonicalize(value: Any) -> str:
    if isinstance(value, float):
        raise MalformedError("NaN/Infinity and floats are not admitted on v0.1 wire records")
    if value is None or isinstance(value, (bool, int, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        parts = []
        for key in sorted(str(item) for item in value):
            raw_key = key if key in value else _lookup_key(value, key)
            parts.append(
                json.dumps(str(key), ensure_ascii=False, separators=(",", ":"))
                + ":"
                + _canonicalize(value[raw_key])
            )
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonicalize(item) for item in value) + "]"
    raise MalformedError(
        f"unsupported wire canonicalization type {type(value).__name__}"
    )


def _lookup_key(payload: Mapping[str, Any], key: str) -> Any:
    for item in payload:
        if str(item) == key:
            return item
    raise MalformedError(f"canonical key {key!r} is missing")


def admit_cid(value: Any, *, field: str = "cid") -> str:
    if not isinstance(value, str) or not value:
        raise PseudoCidError(f"{field} is not a CIDv1 base32 identity")
    try:
        reject_pseudo_cid(value)
    except CompatibilityError as exc:
        raise PseudoCidError(f"{field} is not a CIDv1 base32 identity") from exc
    if (
        not CID_PATTERN.fullmatch(value)
        or len(value) < CID_MIN_LENGTH
        or len(value) > CID_MAX_LENGTH
    ):
        raise PseudoCidError(f"{field} is not a CIDv1 base32 identity")
    return value


def admit_provenance(value: Any) -> str:
    if not isinstance(value, str) or value not in PROVENANCES:
        raise UnknownFieldError(f"unknown provenance {value!r}")
    return value


def admit_sufficiency(value: Any) -> str:
    if not isinstance(value, str) or value not in SUFFICIENCIES:
        raise UnknownFieldError(f"unknown sufficiency {value!r}")
    return value


def admit_bounded_text(
    value: Any,
    *,
    field: str,
    min_length: int = 1,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise MalformedError(f"{field} must be a string")
    if "\x00" in value:
        raise MalformedError(f"{field} contains a NUL byte")
    if len(value) < min_length or len(value) > max_length:
        raise MalformedError(f"{field} length is outside the frozen bound")
    return value


def admit_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedError(f"{field} must be an integer")
    if value < 0 or value > MAX_SAFE_INTEGER:
        raise MalformedError(f"{field} is outside the frozen integer bound")
    return value


def admit_relative_path(value: Any, *, field: str = "path") -> str:
    text = admit_bounded_text(value, field=field, max_length=MAX_PATH_LENGTH)
    if text.startswith("/") or text.startswith("\\") or ":" in text[:2]:
        raise BoundaryViolationError(f"{field} must be repository-relative")
    parts = text.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BoundaryViolationError(f"{field} escapes declared repository scope")
    return text


def admit_path_list(
    value: Any,
    *,
    field: str,
    min_items: int,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MalformedError(f"{field} must be an array of paths")
    if len(value) < min_items or len(value) > max_items:
        raise MalformedError(f"{field} count is outside the frozen bound")
    admitted = tuple(
        admit_relative_path(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    return admitted


def admit_cid_list(
    value: Any,
    *,
    field: str,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MalformedError(f"{field} must be an array of CIDs")
    if len(value) > max_items:
        raise MalformedError(f"{field} count is outside the frozen bound")
    return tuple(
        admit_cid(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )


def admit_bounded_bytes(
    value: Any,
    *,
    field: str,
    max_bytes: int,
) -> bytes:
    if value is None:
        return b""
    if isinstance(value, str):
        data = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        raise MalformedError(f"{field} must be bytes")
    if len(data) > max_bytes:
        raise BoundaryViolationError(f"{field} exceeds the frozen byte bound")
    return data


def admit_bounded_patch(value: Any) -> bytes:
    return admit_bounded_bytes(value, field="patch", max_bytes=MAX_PATCH_BYTES)


def admit_bounded_log(value: Any) -> bytes:
    return admit_bounded_bytes(value, field="log", max_bytes=MAX_LOG_BYTES)


def _as_mapping(payload: Any, *, what: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MalformedError(f"{what} must be a mapping")
    return payload


def _reject_unknown(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    what: str,
) -> None:
    extra = set(payload) - allowed
    if not extra:
        return
    name = sorted(extra)[0]
    if name in FORBIDDEN_WIRE_FIELDS:
        raise BoundaryViolationError(
            f"{what} cannot carry approval, credentials, or hidden evaluation field {name!r}"
        )
    raise UnknownFieldError(f"unknown {what} field {name!r}")


def _require(
    payload: Mapping[str, Any],
    required: frozenset[str],
    *,
    what: str,
) -> None:
    missing = sorted(name for name in required if payload.get(name) is None)
    if missing:
        raise MalformedError(f"{what} is missing required field {missing[0]!r}")


def _admit_schema(payload: Mapping[str, Any], expected: str, *, what: str) -> None:
    marker = payload.get("schema")
    if marker != expected:
        raise SchemaMismatchError(f"{what} schema {marker!r} is not {expected}")


def _omit_none(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    admitted: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            admitted[key] = list(value)
        else:
            admitted[key] = value
    return MappingProxyType(admitted)


def declared_files_are_in_scope(
    declared_files: Sequence[str],
    owned_paths: Sequence[str],
    extra_allow: Sequence[str] | None = None,
) -> bool:
    allowed = set(owned_paths)
    if extra_allow is not None:
        allowed &= set(extra_allow)
    return all(path in allowed for path in declared_files)


def assert_declared_scope(
    declared_files: Sequence[str],
    owned_paths: Sequence[str],
    extra_allow: Sequence[str] | None = None,
) -> None:
    if not declared_files_are_in_scope(declared_files, owned_paths, extra_allow):
        raise BoundaryViolationError("proposal declared files escape owned paths")


@dataclass(frozen=True)
class TaskSpecification:
    """Frozen MCP++ TaskSpecification@0.1 wire record."""

    schema: str
    task_id: str
    objective_id: str
    repository_state_cid: str
    owned_paths: tuple[str, ...]
    provenance: str
    route_cid: str | None = None
    declared_files: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema != TASK_SPECIFICATION_SCHEMA:
            raise SchemaMismatchError(
                f"task specification schema {self.schema!r} is not {TASK_SPECIFICATION_SCHEMA}"
            )
        object.__setattr__(
            self, "task_id", admit_bounded_text(self.task_id, field="task_id", max_length=MAX_ID_LENGTH)
        )
        object.__setattr__(
            self,
            "objective_id",
            admit_bounded_text(
                self.objective_id, field="objective_id", max_length=MAX_ID_LENGTH
            ),
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            admit_cid(self.repository_state_cid, field="repository_state_cid"),
        )
        object.__setattr__(
            self,
            "owned_paths",
            admit_path_list(
                self.owned_paths,
                field="owned_paths",
                min_items=1,
                max_items=MAX_OWNED_PATHS,
            ),
        )
        object.__setattr__(self, "provenance", admit_provenance(self.provenance))
        if self.route_cid is not None:
            object.__setattr__(self, "route_cid", admit_cid(self.route_cid, field="route_cid"))
        if self.declared_files is not None:
            object.__setattr__(
                self,
                "declared_files",
                admit_path_list(
                    self.declared_files,
                    field="declared_files",
                    min_items=0,
                    max_items=MAX_DECLARED_FILES,
                ),
            )
            assert_declared_scope(self.declared_files, self.owned_paths)

    def to_mapping(self) -> Mapping[str, Any]:
        return _omit_none(
            {
                "schema": self.schema,
                "task_id": self.task_id,
                "objective_id": self.objective_id,
                "repository_state_cid": self.repository_state_cid,
                "owned_paths": self.owned_paths,
                "declared_files": self.declared_files,
                "route_cid": self.route_cid,
                "provenance": self.provenance,
            }
        )

    def to_canonical_utf8(self) -> str:
        return wire_canonical_utf8(dict(self.to_mapping()))

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_utf8().encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TaskSpecification:
        data = _as_mapping(payload, what="task specification")
        _reject_unknown(
            data,
            allowed=TASK_SPECIFICATION_REQUIRED | TASK_SPECIFICATION_OPTIONAL,
            what="task specification",
        )
        _require(data, TASK_SPECIFICATION_REQUIRED, what="task specification")
        _admit_schema(data, TASK_SPECIFICATION_SCHEMA, what="task specification")
        declared = data.get("declared_files")
        return cls(
            schema=str(data["schema"]),
            task_id=str(data["task_id"]),
            objective_id=str(data["objective_id"]),
            repository_state_cid=str(data["repository_state_cid"]),
            owned_paths=tuple(data["owned_paths"]),
            provenance=str(data["provenance"]),
            route_cid=str(data["route_cid"]) if data.get("route_cid") is not None else None,
            declared_files=tuple(declared) if declared is not None else None,
        )


@dataclass(frozen=True)
class CodingAgentInvocation:
    """Frozen MCP++ CodingAgentInvocation@0.1 wire record."""

    schema: str
    invocation_cid: str
    task_id: str
    repository_state_cid: str
    provider: str
    model: str
    revision: str
    tier: str
    provenance: str
    route_cid: str | None = None
    token_count: int | None = None
    cached_token_count: int | None = None
    latency_ms: int | None = None
    cost_micros: int | None = None
    response_artifact_cid: str | None = None

    def __post_init__(self) -> None:
        if self.schema != CODING_AGENT_INVOCATION_SCHEMA:
            raise SchemaMismatchError(
                f"invocation schema {self.schema!r} is not {CODING_AGENT_INVOCATION_SCHEMA}"
            )
        object.__setattr__(
            self,
            "invocation_cid",
            admit_cid(self.invocation_cid, field="invocation_cid"),
        )
        object.__setattr__(
            self, "task_id", admit_bounded_text(self.task_id, field="task_id", max_length=MAX_ID_LENGTH)
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            admit_cid(self.repository_state_cid, field="repository_state_cid"),
        )
        object.__setattr__(
            self,
            "provider",
            admit_bounded_text(self.provider, field="provider", max_length=MAX_ID_LENGTH),
        )
        object.__setattr__(
            self,
            "model",
            admit_bounded_text(self.model, field="model", max_length=MAX_MODEL_LENGTH),
        )
        object.__setattr__(
            self,
            "revision",
            admit_bounded_text(self.revision, field="revision", max_length=MAX_REVISION_LENGTH),
        )
        object.__setattr__(
            self, "tier", admit_bounded_text(self.tier, field="tier", max_length=MAX_TIER_LENGTH)
        )
        object.__setattr__(self, "provenance", admit_provenance(self.provenance))
        if self.route_cid is not None:
            object.__setattr__(self, "route_cid", admit_cid(self.route_cid, field="route_cid"))
        if self.token_count is not None:
            object.__setattr__(
                self, "token_count", admit_non_negative_int(self.token_count, field="token_count")
            )
        if self.cached_token_count is not None:
            object.__setattr__(
                self,
                "cached_token_count",
                admit_non_negative_int(self.cached_token_count, field="cached_token_count"),
            )
        if self.latency_ms is not None:
            object.__setattr__(
                self, "latency_ms", admit_non_negative_int(self.latency_ms, field="latency_ms")
            )
        if self.cost_micros is not None:
            object.__setattr__(
                self, "cost_micros", admit_non_negative_int(self.cost_micros, field="cost_micros")
            )
        if self.response_artifact_cid is not None:
            object.__setattr__(
                self,
                "response_artifact_cid",
                admit_cid(self.response_artifact_cid, field="response_artifact_cid"),
            )

    def usage_is_explicit(self) -> bool:
        return all(getattr(self, name) is not None for name in INVOCATION_USAGE_FIELDS)

    def has_live_evidence(self) -> bool:
        return self.response_artifact_cid is not None and self.usage_is_explicit()

    def to_mapping(self) -> Mapping[str, Any]:
        return _omit_none(
            {
                "schema": self.schema,
                "invocation_cid": self.invocation_cid,
                "task_id": self.task_id,
                "repository_state_cid": self.repository_state_cid,
                "route_cid": self.route_cid,
                "provider": self.provider,
                "model": self.model,
                "revision": self.revision,
                "tier": self.tier,
                "token_count": self.token_count,
                "cached_token_count": self.cached_token_count,
                "latency_ms": self.latency_ms,
                "cost_micros": self.cost_micros,
                "response_artifact_cid": self.response_artifact_cid,
                "provenance": self.provenance,
            }
        )

    def to_canonical_utf8(self) -> str:
        return wire_canonical_utf8(dict(self.to_mapping()))

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_utf8().encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CodingAgentInvocation:
        data = _as_mapping(payload, what="coding-agent invocation")
        _reject_unknown(
            data,
            allowed=INVOCATION_REQUIRED | INVOCATION_OPTIONAL,
            what="coding-agent invocation",
        )
        _require(data, INVOCATION_REQUIRED, what="coding-agent invocation")
        _admit_schema(data, CODING_AGENT_INVOCATION_SCHEMA, what="coding-agent invocation")
        return cls(
            schema=str(data["schema"]),
            invocation_cid=str(data["invocation_cid"]),
            task_id=str(data["task_id"]),
            repository_state_cid=str(data["repository_state_cid"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            revision=str(data["revision"]),
            tier=str(data["tier"]),
            provenance=str(data["provenance"]),
            route_cid=str(data["route_cid"]) if data.get("route_cid") is not None else None,
            token_count=data.get("token_count"),
            cached_token_count=data.get("cached_token_count"),
            latency_ms=data.get("latency_ms"),
            cost_micros=data.get("cost_micros"),
            response_artifact_cid=(
                str(data["response_artifact_cid"])
                if data.get("response_artifact_cid") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PatchProposal:
    """Frozen MCP++ PatchProposal@0.1 wire record. Never self-approves."""

    schema: str
    proposal_cid: str
    task_id: str
    repository_state_cid: str
    declared_files: tuple[str, ...]
    provenance: str
    invocation_cid: str | None = None
    patch_cid: str | None = None

    def __post_init__(self) -> None:
        if self.schema != PATCH_PROPOSAL_SCHEMA:
            raise SchemaMismatchError(
                f"patch proposal schema {self.schema!r} is not {PATCH_PROPOSAL_SCHEMA}"
            )
        object.__setattr__(
            self, "proposal_cid", admit_cid(self.proposal_cid, field="proposal_cid")
        )
        object.__setattr__(
            self, "task_id", admit_bounded_text(self.task_id, field="task_id", max_length=MAX_ID_LENGTH)
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            admit_cid(self.repository_state_cid, field="repository_state_cid"),
        )
        object.__setattr__(
            self,
            "declared_files",
            admit_path_list(
                self.declared_files,
                field="declared_files",
                min_items=1,
                max_items=MAX_DECLARED_FILES,
            ),
        )
        object.__setattr__(self, "provenance", admit_provenance(self.provenance))
        if self.invocation_cid is not None:
            object.__setattr__(
                self, "invocation_cid", admit_cid(self.invocation_cid, field="invocation_cid")
            )
        if self.patch_cid is not None:
            object.__setattr__(self, "patch_cid", admit_cid(self.patch_cid, field="patch_cid"))

    @property
    def accepted(self) -> bool:
        return False

    @accepted.setter
    def accepted(self, value: bool) -> None:
        if value:
            raise BoundaryViolationError("a patch proposal cannot approve itself")

    @property
    def approved(self) -> bool:
        return False

    @approved.setter
    def approved(self, value: bool) -> None:
        if value:
            raise BoundaryViolationError("a patch proposal cannot approve itself")

    def to_mapping(self) -> Mapping[str, Any]:
        return _omit_none(
            {
                "schema": self.schema,
                "proposal_cid": self.proposal_cid,
                "task_id": self.task_id,
                "repository_state_cid": self.repository_state_cid,
                "invocation_cid": self.invocation_cid,
                "patch_cid": self.patch_cid,
                "declared_files": self.declared_files,
                "provenance": self.provenance,
            }
        )

    def to_canonical_utf8(self) -> str:
        return wire_canonical_utf8(dict(self.to_mapping()))

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_utf8().encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PatchProposal:
        data = _as_mapping(payload, what="patch proposal")
        _reject_unknown(
            data,
            allowed=PATCH_PROPOSAL_REQUIRED | PATCH_PROPOSAL_OPTIONAL,
            what="patch proposal",
        )
        _require(data, PATCH_PROPOSAL_REQUIRED, what="patch proposal")
        _admit_schema(data, PATCH_PROPOSAL_SCHEMA, what="patch proposal")
        return cls(
            schema=str(data["schema"]),
            proposal_cid=str(data["proposal_cid"]),
            task_id=str(data["task_id"]),
            repository_state_cid=str(data["repository_state_cid"]),
            declared_files=tuple(data["declared_files"]),
            provenance=str(data["provenance"]),
            invocation_cid=(
                str(data["invocation_cid"]) if data.get("invocation_cid") is not None else None
            ),
            patch_cid=str(data["patch_cid"]) if data.get("patch_cid") is not None else None,
        )


@dataclass(frozen=True)
class ContextPack:
    """Frozen MCP++ ContextPack@0.1 wire record. Datasets remains the builder."""

    schema: str
    pack_cid: str
    repository_state_cid: str
    sufficiency: str
    provenance: str
    task_id: str | None = None
    capsule_cids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema != CONTEXT_PACK_SCHEMA:
            raise SchemaMismatchError(
                f"context pack schema {self.schema!r} is not {CONTEXT_PACK_SCHEMA}"
            )
        object.__setattr__(self, "pack_cid", admit_cid(self.pack_cid, field="pack_cid"))
        object.__setattr__(
            self,
            "repository_state_cid",
            admit_cid(self.repository_state_cid, field="repository_state_cid"),
        )
        object.__setattr__(self, "sufficiency", admit_sufficiency(self.sufficiency))
        object.__setattr__(self, "provenance", admit_provenance(self.provenance))
        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                admit_bounded_text(self.task_id, field="task_id", max_length=MAX_ID_LENGTH),
            )
        if self.capsule_cids is not None:
            object.__setattr__(
                self,
                "capsule_cids",
                admit_cid_list(
                    self.capsule_cids, field="capsule_cids", max_items=MAX_CAPSULE_CIDS
                ),
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return _omit_none(
            {
                "schema": self.schema,
                "pack_cid": self.pack_cid,
                "repository_state_cid": self.repository_state_cid,
                "task_id": self.task_id,
                "sufficiency": self.sufficiency,
                "capsule_cids": self.capsule_cids,
                "provenance": self.provenance,
            }
        )

    def to_canonical_utf8(self) -> str:
        return wire_canonical_utf8(dict(self.to_mapping()))

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_utf8().encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ContextPack:
        data = _as_mapping(payload, what="context pack")
        _reject_unknown(
            data,
            allowed=CONTEXT_PACK_REQUIRED | CONTEXT_PACK_OPTIONAL,
            what="context pack",
        )
        _require(data, CONTEXT_PACK_REQUIRED, what="context pack")
        _admit_schema(data, CONTEXT_PACK_SCHEMA, what="context pack")
        capsules = data.get("capsule_cids")
        return cls(
            schema=str(data["schema"]),
            pack_cid=str(data["pack_cid"]),
            repository_state_cid=str(data["repository_state_cid"]),
            sufficiency=str(data["sufficiency"]),
            provenance=str(data["provenance"]),
            task_id=str(data["task_id"]) if data.get("task_id") is not None else None,
            capsule_cids=tuple(capsules) if capsules is not None else None,
        )


@dataclass(frozen=True)
class ModelRouteDecision:
    """Frozen MCP++ ModelRouteDecision@0.1 wire record. No credentials."""

    schema: str
    decision_cid: str
    task_id: str
    tier: str
    provider: str
    model: str
    provenance: str
    repository_state_cid: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.schema != MODEL_ROUTE_DECISION_SCHEMA:
            raise SchemaMismatchError(
                f"model route schema {self.schema!r} is not {MODEL_ROUTE_DECISION_SCHEMA}"
            )
        object.__setattr__(
            self, "decision_cid", admit_cid(self.decision_cid, field="decision_cid")
        )
        object.__setattr__(
            self, "task_id", admit_bounded_text(self.task_id, field="task_id", max_length=MAX_ID_LENGTH)
        )
        object.__setattr__(
            self, "tier", admit_bounded_text(self.tier, field="tier", max_length=MAX_TIER_LENGTH)
        )
        object.__setattr__(
            self,
            "provider",
            admit_bounded_text(self.provider, field="provider", max_length=MAX_ID_LENGTH),
        )
        object.__setattr__(
            self,
            "model",
            admit_bounded_text(self.model, field="model", max_length=MAX_MODEL_LENGTH),
        )
        object.__setattr__(self, "provenance", admit_provenance(self.provenance))
        if self.repository_state_cid is not None:
            object.__setattr__(
                self,
                "repository_state_cid",
                admit_cid(self.repository_state_cid, field="repository_state_cid"),
            )
        if self.revision is not None:
            object.__setattr__(
                self,
                "revision",
                admit_bounded_text(self.revision, field="revision", max_length=MAX_REVISION_LENGTH),
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return _omit_none(
            {
                "schema": self.schema,
                "decision_cid": self.decision_cid,
                "task_id": self.task_id,
                "repository_state_cid": self.repository_state_cid,
                "tier": self.tier,
                "provider": self.provider,
                "model": self.model,
                "revision": self.revision,
                "provenance": self.provenance,
            }
        )

    def to_canonical_utf8(self) -> str:
        return wire_canonical_utf8(dict(self.to_mapping()))

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_utf8().encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ModelRouteDecision:
        data = _as_mapping(payload, what="model route decision")
        _reject_unknown(
            data,
            allowed=MODEL_ROUTE_REQUIRED | MODEL_ROUTE_OPTIONAL,
            what="model route decision",
        )
        _require(data, MODEL_ROUTE_REQUIRED, what="model route decision")
        _admit_schema(data, MODEL_ROUTE_DECISION_SCHEMA, what="model route decision")
        return cls(
            schema=str(data["schema"]),
            decision_cid=str(data["decision_cid"]),
            task_id=str(data["task_id"]),
            tier=str(data["tier"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            provenance=str(data["provenance"]),
            repository_state_cid=(
                str(data["repository_state_cid"])
                if data.get("repository_state_cid") is not None
                else None
            ),
            revision=str(data["revision"]) if data.get("revision") is not None else None,
        )


WIRE_RECORD_TYPES: Final[Mapping[str, type]] = MappingProxyType(
    {
        TASK_SPECIFICATION_SCHEMA: TaskSpecification,
        CODING_AGENT_INVOCATION_SCHEMA: CodingAgentInvocation,
        PATCH_PROPOSAL_SCHEMA: PatchProposal,
        CONTEXT_PACK_SCHEMA: ContextPack,
        MODEL_ROUTE_DECISION_SCHEMA: ModelRouteDecision,
    }
)


def parse_wire_record(payload: Mapping[str, Any]) -> Any:
    data = _as_mapping(payload, what="wire record")
    marker = data.get("schema")
    record_type = WIRE_RECORD_TYPES.get(marker) if isinstance(marker, str) else None
    if record_type is None:
        raise SchemaMismatchError(f"wire record schema {marker!r} is not a frozen adapter schema")
    return record_type.from_mapping(data)


__all__ = [
    "APPROVAL_FIELDS",
    "CID_PATTERN",
    "CODING_AGENT_INVOCATION_SCHEMA",
    "CODING_AGENT_INVOCATION_SCHEMA_DIGEST",
    "CONTEXT_PACK_SCHEMA",
    "CONTEXT_PACK_SCHEMA_DIGEST",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "CodingAgentInvocation",
    "ContextPack",
    "FORBIDDEN_WIRE_FIELDS",
    "HIDDEN_EVALUATION_FIELDS",
    "INVOCATION_USAGE_FIELDS",
    "MAX_CAPSULE_CIDS",
    "MAX_DECLARED_FILES",
    "MAX_FILE_BYTES",
    "MAX_LOG_BYTES",
    "MAX_OWNED_PATHS",
    "MAX_PATCH_BYTES",
    "MAX_PATH_LENGTH",
    "MAX_PROVIDER_OUTPUT_BYTES",
    "MAX_SAFE_INTEGER",
    "MODEL_ROUTE_DECISION_SCHEMA",
    "MODEL_ROUTE_DECISION_SCHEMA_DIGEST",
    "ModelRouteDecision",
    "PCCE_006_CONTENT_ID",
    "PATCH_PROPOSAL_SCHEMA",
    "PATCH_PROPOSAL_SCHEMA_DIGEST",
    "PROVENANCES",
    "PatchProposal",
    "SCHEMA",
    "SUFFICIENCIES",
    "TASK_SPECIFICATION_SCHEMA",
    "TASK_SPECIFICATION_SCHEMA_DIGEST",
    "TaskSpecification",
    "WIRE_RECORD_TYPES",
    "WIRE_SCHEMAS",
    "admit_bounded_log",
    "admit_bounded_patch",
    "admit_bounded_bytes",
    "admit_bounded_text",
    "admit_cid",
    "admit_non_negative_int",
    "admit_path_list",
    "admit_provenance",
    "admit_relative_path",
    "admit_sufficiency",
    "assert_declared_scope",
    "declared_files_are_in_scope",
    "parse_wire_record",
    "wire_canonical_bytes",
    "wire_canonical_utf8",
]
