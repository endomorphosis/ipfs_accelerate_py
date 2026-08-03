"""Minimal, contract-directed edit packets for MCP contract findings.

The packet produced here is the narrow hand-off from ``ContractFinding@1`` to
an implementation provider.  It deliberately contains an obligation-first
projection rather than a repository review prompt:

* the complete task/contract/obligation core is non-optional;
* repository, source, AST, receipt, and proof bodies are forbidden;
* large artifacts are represented by content-addressed expansion handles;
* finding-derived text is explicitly labeled as untrusted data;
* exact read/write paths and current snapshot identity are checked eagerly;
* the provider-visible input may not exceed 8,192 tokens; and
* an unchanged retry contains only a parent binding and ``proof_delta``.

This module composes the existing :class:`CodeEditPacket` and
:class:`ContextCapsule` contracts.  It does not fetch expansion handles or
grant a model semantic authority.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..analysis.contract_mismatch_analyzer import (
    ContractFinding,
    FindingLifecycle,
    MismatchState,
)
from ..analysis.contract_repair_contracts import (
    DecisionDisposition,
    RepairStrategy,
    RepairTargetDecision,
)
from ..context.context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextReference,
    ContextTier,
)
from ..planning.repair_target_admission import AdmissionResult
from .code_edit_packet import (
    CodeEditPacket as BaseCodeEditPacket,
    build_code_edit_packet,
)
from .formal_verification_contracts import canonical_json_bytes, content_identity


MCP_CONTRACT_EDIT_PACKET_INTERFACE: Final = "CodeEditPacket@1"
MCP_CONTRACT_EDIT_PACKET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-edit-packet@1"
)
MCP_CONTRACT_EDIT_RETRY_INTERFACE: Final = "CodeEditPacketProofDelta@1"
MCP_CONTRACT_EDIT_RETRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-edit-proof-delta@1"
)
MCP_CONTRACT_EDIT_PACKET_VERSION: Final = "1"
MCP_CONTRACT_EDIT_PACKET_CONTRACT_VERSION: Final = 1
# Explicit, opt-in @2 decision path.  Default materialization remains @1 so
# legacy callers keep exact affected_paths equality semantics.
MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION: Final = 2
WRITE_PATH_AUTHORITY_AFFECTED_PATHS: Final = "finding_affected_paths"
WRITE_PATH_AUTHORITY_TARGET_DECISION: Final = "repair_target_decision"

UNTRUSTED_DATA_LABEL: Final = "untrusted_repository_data"
DATA_NOT_INSTRUCTIONS: Final = "data_not_instructions"
MAX_PACKET_INPUT_TOKENS: Final = 8_192
FIXTURE_MEDIAN_TARGET_TOKENS: Final = 2_048
DEFAULT_PACKET_MAX_SERIALIZED_BYTES: Final = 256 * 1024
MAX_INLINE_DATA_BYTES: Final = 32 * 1024
MAX_COMMANDS: Final = 64
MAX_PATHS: Final = 1_024
MAX_DEPENDENCIES: Final = 1_024


class ContractEditPacketReason(str, Enum):
    """Stable fail-closed reasons for packet admission."""

    STALE_FINDING = "stale_finding"
    MISSING_MANDATORY_DEPENDENCY = "missing_mandatory_dependency"
    FORBIDDEN_BODY = "forbidden_body"
    PATH_SCOPE_MISMATCH = "path_scope_mismatch"
    REQUIRED_CORE_MISSING = "required_core_missing"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    DECISION_REQUIRED = "decision_required"
    DECISION_NOT_ADMITTED = "decision_not_admitted"
    DECISION_SCOPE_MISMATCH = "decision_scope_mismatch"
    MALFORMED = "malformed"


class ContractEditPacketError(ValueError):
    """A contract edit packet failed deterministic admission."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ContractEditPacketReason | str = ContractEditPacketReason.MALFORMED,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


# Exact normalized keys which indicate an embedded artifact rather than a
# compact counterexample/slice or a content-addressed handle.
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "ast_nodes",
        "file_content",
        "file_contents",
        "full_receipt",
        "gold_ir",
        "gold_ir_body",
        "kernel_proof_body",
        "lean_source",
        "private_witness",
        "proof",
        "proof_body",
        "proof_text",
        "receipt_body",
        "repository_body",
        "repository_content",
        "repository_corpus",
        "solver_trace",
        "source_body",
        "source_code",
        "source_text",
        "witness",
    }
)


def _key(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _reject_embedded_bodies(value: Any, *, location: str = "packet") -> None:
    """Reject artifact bodies recursively while allowing IDs and handles."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            if not isinstance(raw_name, str):
                raise ContractEditPacketError(
                    f"{location} keys must be strings",
                    reason_code=ContractEditPacketReason.MALFORMED,
                )
            name = _key(raw_name)
            if name in _FORBIDDEN_BODY_KEYS or name.endswith("_body"):
                raise ContractEditPacketError(
                    f"{location} must not embed {raw_name!r}; use an expansion handle",
                    reason_code=ContractEditPacketReason.FORBIDDEN_BODY,
                )
            _reject_embedded_bodies(item, location=f"{location}.{raw_name}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            _reject_embedded_bodies(item, location=f"{location}[{index}]")


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum_bytes: int = 16_384,
    single_line: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ContractEditPacketError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise ContractEditPacketError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise ContractEditPacketError(
            f"{name} is required",
            reason_code=ContractEditPacketReason.REQUIRED_CORE_MISSING,
        )
    if single_line and ("\n" in value or "\r" in value):
        raise ContractEditPacketError(f"{name} must be one line")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ContractEditPacketError(f"{name} exceeds its byte bound")
    return value


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_DEPENDENCIES,
) -> tuple[str, ...]:
    if values is None:
        source: Iterable[Any] = ()
    elif isinstance(values, str):
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractEditPacketError(f"{name} must be a sequence of strings")
    result: set[str] = set()
    for item in source:
        result.add(
            _text(
                item,
                name,
                maximum_bytes=4_096,
                single_line=True,
            )
        )
        if len(result) > maximum:
            raise ContractEditPacketError(f"{name} exceeds its item bound")
    if required and not result:
        raise ContractEditPacketError(
            f"{name} must not be empty",
            reason_code=ContractEditPacketReason.REQUIRED_CORE_MISSING,
        )
    return tuple(sorted(result))


def _path(value: Any, name: str) -> str:
    raw = _text(value, name, maximum_bytes=4_096, single_line=True)
    if "\\" in raw:
        raise ContractEditPacketError(f"{name} must use POSIX separators")
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or raw.startswith("./")
        or ".." in candidate.parts
        or candidate.as_posix() in {"", "."}
        or any(character in raw for character in "*?[]{}")
    ):
        raise ContractEditPacketError(
            f"{name} must be one exact repository-relative path",
            reason_code=ContractEditPacketReason.PATH_SCOPE_MISMATCH,
        )
    return candidate.as_posix()


def _paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        raise ContractEditPacketError(f"{name} must be a sequence of paths")
    result = tuple(sorted({_path(item, name) for item in values}))
    if required and not result:
        raise ContractEditPacketError(
            f"{name} must not be empty",
            reason_code=ContractEditPacketReason.REQUIRED_CORE_MISSING,
        )
    if len(result) > MAX_PATHS:
        raise ContractEditPacketError(f"{name} exceeds its item bound")
    return result


def _commands(values: Any, name: str) -> tuple[str, ...]:
    result = _ids(values, name, required=True, maximum=MAX_COMMANDS)
    for command in result:
        if "\n" in command or "\r" in command:
            raise ContractEditPacketError(f"{name} entries must be one line")
    return result


def _plain_json(value: Any, *, name: str, depth: int = 0) -> Any:
    if depth > 12:
        raise ContractEditPacketError(f"{name} exceeds its nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ContractEditPacketError(f"{name} must not contain floats")
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise ContractEditPacketError(
                f"{name} objects require at most 1024 string keys"
            )
        return {
            key: _plain_json(value[key], name=name, depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        if len(value) > 2_048:
            raise ContractEditPacketError(f"{name} sequence is oversized")
        return [_plain_json(item, name=name, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain_json(to_dict(), name=name, depth=depth + 1)
    raise ContractEditPacketError(
        f"{name} contains unsupported value {type(value).__name__}"
    )


def _bounded_data(value: Any, *, name: str) -> Any:
    result = _plain_json(value, name=name)
    _reject_embedded_bodies(result, location=name)
    if len(canonical_json_bytes(result)) > MAX_INLINE_DATA_BYTES:
        raise ContractEditPacketError(
            f"{name} is too large to inline; use an expansion handle",
            reason_code=ContractEditPacketReason.FORBIDDEN_BODY,
        )
    return result


def _labeled_data(value: Any, *, source: str) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "data_label": UNTRUSTED_DATA_LABEL,
            "data_source": source,
            "instruction_authority": False,
            "treat_as": DATA_NOT_INSTRUCTIONS,
            "value": _bounded_data(value, name=source),
        }
    )


def _measure_tokens(
    value: Any,
    tokenizer: Callable[[str], Any] | None = None,
) -> int:
    """Measure provider input with a conservative deterministic fallback."""

    encoded = canonical_json_bytes(value)
    if tokenizer is None:
        # Four UTF-8 bytes per token is deliberately more conservative than
        # the repository-wide compact estimator used for terse identifiers.
        return max(1, (len(encoded) + 3) // 4)
    measured = tokenizer(encoded.decode("utf-8"))
    if isinstance(measured, bool):
        raise ContractEditPacketError("tokenizer returned a boolean")
    if isinstance(measured, int):
        count = measured
    else:
        try:
            count = len(measured)
        except TypeError as exc:
            raise ContractEditPacketError(
                "tokenizer must return an integer or sized token sequence"
            ) from exc
    if count < 0:
        raise ContractEditPacketError("tokenizer returned a negative count")
    return max(1, count)


@dataclass(frozen=True, slots=True)
class ExpansionHandle:
    """Content-addressed pointer to an artifact which is never embedded."""

    handle_id: str
    kind: str
    content_id: str
    byte_count: int = 0
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        for name in ("handle_id", "kind", "content_id", "media_type"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    maximum_bytes=4_096,
                    single_line=True,
                ),
            )
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise ContractEditPacketError("byte_count must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "kind": self.kind,
            "content_id": self.content_id,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
            "body_embedded": False,
        }

    @classmethod
    def from_value(
        cls, value: "ExpansionHandle | Mapping[str, Any] | str", *, index: int = 0
    ) -> "ExpansionHandle":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(
                handle_id=f"cas:{index:04d}",
                kind="cas_artifact",
                content_id=value,
            )
        if not isinstance(value, Mapping):
            raise ContractEditPacketError("expansion handle is malformed")
        _reject_embedded_bodies(value, location="expansion_handle")
        return cls(
            handle_id=str(
                value.get("handle_id")
                or value.get("reference_id")
                or f"handle:{index:04d}"
            ),
            kind=str(value.get("kind") or "cas_artifact"),
            content_id=str(
                value.get("content_id")
                or value.get("cid")
                or value.get("referenced_content_id")
                or ""
            ),
            byte_count=value.get("byte_count", 0),
            media_type=str(
                value.get("media_type") or "application/octet-stream"
            ),
        )


def _coerce_handles(
    values: Sequence[ExpansionHandle | Mapping[str, Any] | str],
) -> tuple[ExpansionHandle, ...]:
    by_id: dict[str, ExpansionHandle] = {}
    for index, raw in enumerate(values):
        item = ExpansionHandle.from_value(raw, index=index)
        previous = by_id.get(item.handle_id)
        if previous is not None and previous != item:
            raise ContractEditPacketError(
                f"conflicting expansion handle {item.handle_id!r}"
            )
        by_id[item.handle_id] = item
    return tuple(by_id[key] for key in sorted(by_id))


def _context_reference(
    handle: ExpansionHandle,
    *,
    repository_id: str,
    tree_id: str,
) -> ContextReference:
    return ContextReference(
        reference_id=handle.handle_id,
        kind=handle.kind,
        tier=ContextTier.EXPANSION,
        referenced_content_id=handle.content_id,
        repository_id=repository_id,
        tree_id=tree_id,
        summary="Content-addressed artifact; expand only when specifically required.",
        byte_count=handle.byte_count,
        token_count=0,
        metadata={
            "body_embedded": False,
            "data_label": UNTRUSTED_DATA_LABEL,
            "instruction_authority": False,
            "treat_as": DATA_NOT_INSTRUCTIONS,
        },
    )


def _finding(value: ContractFinding | Mapping[str, Any]) -> ContractFinding:
    if isinstance(value, ContractFinding):
        return value
    if not isinstance(value, Mapping):
        raise ContractEditPacketError("finding must be ContractFinding@1")
    try:
        return ContractFinding.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ContractEditPacketError(
            f"finding is invalid: {exc}",
            reason_code=ContractEditPacketReason.MALFORMED,
        ) from exc


def _assert_finding_current(
    finding: ContractFinding,
    *,
    current_snapshot_id: str,
    expected_finding_record_id: str = "",
) -> None:
    current = _text(current_snapshot_id, "current_snapshot_id")
    expected_record = _text(
        expected_finding_record_id,
        "expected_finding_record_id",
        required=False,
    )
    stale_lifecycle = finding.lifecycle in {
        FindingLifecycle.STALE,
        FindingLifecycle.RESOLVED,
    }
    stale_state = finding.state in {
        MismatchState.STALE,
        MismatchState.UNSUPPORTED,
        MismatchState.NOT_MEASURED,
    }
    if (
        finding.snapshot_id != current
        or stale_lifecycle
        or stale_state
        or (expected_record and finding.record_id != expected_record)
    ):
        raise ContractEditPacketError(
            "counterexample finding is stale or not implementation-ready",
            reason_code=ContractEditPacketReason.STALE_FINDING,
        )


@dataclass(frozen=True, slots=True)
class McpContractEditPacket:
    """One current, bounded, content-addressed contract repair packet."""

    base_packet: BaseCodeEditPacket
    context_capsule: ContextCapsule
    finding_id: str
    finding_record_id: str
    counterexample_id: str
    dependency_ids: tuple[str, ...]
    mandatory_dependency_ids: tuple[str, ...]
    expansion_handles: tuple[ExpansionHandle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_packet, BaseCodeEditPacket):
            raise ContractEditPacketError("base_packet must be CodeEditPacket@1")
        if not isinstance(self.context_capsule, ContextCapsule):
            raise ContractEditPacketError("context_capsule must be ContextCapsule")
        for name in ("finding_id", "finding_record_id", "counterexample_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "dependency_ids",
            _ids(self.dependency_ids, "dependency_ids"),
        )
        object.__setattr__(
            self,
            "mandatory_dependency_ids",
            _ids(self.mandatory_dependency_ids, "mandatory_dependency_ids"),
        )
        handles = _coerce_handles(self.expansion_handles)
        object.__setattr__(self, "expansion_handles", handles)

        missing = set(self.mandatory_dependency_ids).difference(self.dependency_ids)
        if missing:
            raise ContractEditPacketError(
                "mandatory dependencies omitted: " + ", ".join(sorted(missing)),
                reason_code=ContractEditPacketReason.MISSING_MANDATORY_DEPENDENCY,
            )
        if self.base_packet.repository_tree_id != self.context_capsule.tree_id:
            raise ContractEditPacketError("packet and context snapshot bindings differ")
        if self.base_packet.task_id != self.context_capsule.objective_id:
            raise ContractEditPacketError("packet and context task bindings differ")
        if tuple(self.base_packet.predicted_files) != self.write_paths:
            raise ContractEditPacketError("base packet write paths are not exact")
        context_handle_ids = {
            item.reference_id for item in self.context_capsule.expansion_references
        }
        if context_handle_ids != {item.handle_id for item in handles}:
            raise ContractEditPacketError("context expansion manifest is incomplete")
        if self.input_tokens > MAX_PACKET_INPUT_TOKENS:
            raise ContractEditPacketError(
                "packet exceeds the absolute 8,192-token input limit",
                reason_code=ContractEditPacketReason.TOKEN_BUDGET_EXCEEDED,
            )
        _reject_embedded_bodies(self.to_dict(include_id=False))

    @property
    def interface(self) -> str:
        return MCP_CONTRACT_EDIT_PACKET_INTERFACE

    @property
    def packet_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    @property
    def content_id(self) -> str:
        return self.packet_id

    @property
    def task_id(self) -> str:
        return self.base_packet.task_id

    @property
    def snapshot_id(self) -> str:
        return self.base_packet.repository_tree_id

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return self.base_packet.property_ids

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return self.base_packet.obligation_ids

    @property
    def affected_symbols(self) -> tuple[str, ...]:
        raw = self.context_capsule.goal.get("affected_symbols", ())
        return tuple(raw)

    @property
    def read_paths(self) -> tuple[str, ...]:
        return tuple(self.context_capsule.scope.get("read_paths", ()))

    @property
    def write_paths(self) -> tuple[str, ...]:
        return tuple(self.context_capsule.scope.get("write_paths", ()))

    @property
    def input_tokens(self) -> int:
        return self.context_capsule.input_tokens

    @property
    def token_count(self) -> int:
        return self.input_tokens

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return self.context_capsule.provider_input_payload

    @property
    def required_core(self) -> Mapping[str, Any]:
        return self.context_capsule.invariant_core

    @property
    def required_core_truncated(self) -> bool:
        """Mandatory goal/authority/scope/acceptance fields are never omitted."""

        return False

    def assert_current(
        self,
        current_snapshot_id: str,
        *,
        finding_record_id: str = "",
    ) -> None:
        current = _text(current_snapshot_id, "current_snapshot_id")
        record = _text(
            finding_record_id, "finding_record_id", required=False
        )
        if current != self.snapshot_id or (
            record and record != self.finding_record_id
        ):
            raise ContractEditPacketError(
                "edit packet is stale",
                reason_code=ContractEditPacketReason.STALE_FINDING,
            )

    def retry(
        self,
        proof_delta: Mapping[str, Any],
        *,
        current_snapshot_id: str | None = None,
        finding_record_id: str = "",
        tokenizer: Callable[[str], Any] | None = None,
    ) -> "ContractEditRetryPacket":
        return build_contract_edit_retry(
            self,
            proof_delta=proof_delta,
            current_snapshot_id=current_snapshot_id or self.snapshot_id,
            finding_record_id=finding_record_id,
            tokenizer=tokenizer,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": MCP_CONTRACT_EDIT_PACKET_SCHEMA,
            "interface": MCP_CONTRACT_EDIT_PACKET_INTERFACE,
            "version": MCP_CONTRACT_EDIT_PACKET_VERSION,
            "contract_version": MCP_CONTRACT_EDIT_PACKET_CONTRACT_VERSION,
            "base_packet": self.base_packet.to_record(),
            "context_capsule": self.context_capsule.to_record(),
            "finding_id": self.finding_id,
            "finding_record_id": self.finding_record_id,
            "counterexample_id": self.counterexample_id,
            "dependency_ids": list(self.dependency_ids),
            "mandatory_dependency_ids": list(self.mandatory_dependency_ids),
            "expansion_handles": [
                item.to_dict() for item in self.expansion_handles
            ],
            "input_tokens": self.input_tokens,
            "max_input_tokens": MAX_PACKET_INPUT_TOKENS,
        }
        if include_id:
            payload["packet_id"] = content_identity(payload)
            payload["content_id"] = payload["packet_id"]
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McpContractEditPacket":
        if not isinstance(value, Mapping):
            raise ContractEditPacketError("packet must be an object")
        if value.get("schema") not in (None, MCP_CONTRACT_EDIT_PACKET_SCHEMA):
            raise ContractEditPacketError("unsupported contract edit packet schema")
        if value.get("interface") not in (
            None,
            MCP_CONTRACT_EDIT_PACKET_INTERFACE,
        ):
            raise ContractEditPacketError("unsupported contract edit packet interface")
        _reject_embedded_bodies(value)
        base = value.get("base_packet")
        context = value.get("context_capsule")
        if not isinstance(base, Mapping) or not isinstance(context, Mapping):
            raise ContractEditPacketError("packet core contracts are required")
        result = cls(
            base_packet=BaseCodeEditPacket.from_dict(base),
            context_capsule=ContextCapsule.from_dict(context),
            finding_id=str(value.get("finding_id") or ""),
            finding_record_id=str(value.get("finding_record_id") or ""),
            counterexample_id=str(value.get("counterexample_id") or ""),
            dependency_ids=tuple(value.get("dependency_ids") or ()),
            mandatory_dependency_ids=tuple(
                value.get("mandatory_dependency_ids") or ()
            ),
            expansion_handles=tuple(
                ExpansionHandle.from_value(item, index=index)
                for index, item in enumerate(value.get("expansion_handles") or ())
            ),
        )
        claimed = value.get("packet_id") or value.get("content_id")
        if claimed not in (None, "", result.packet_id):
            raise ContractEditPacketError("packet content identity mismatch")
        if value.get("input_tokens") not in (None, result.input_tokens):
            raise ContractEditPacketError("packet token accounting mismatch")
        if value.get("max_input_tokens") not in (
            None,
            MAX_PACKET_INPUT_TOKENS,
        ):
            raise ContractEditPacketError("packet token ceiling mismatch")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "McpContractEditPacket":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractEditPacketError("packet JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ContractEditPacketError("packet JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class ContractEditRetryPacket:
    """Parent-bound provider input containing only a labeled proof delta."""

    parent_packet_id: str
    snapshot_id: str
    task_id: str
    finding_record_id: str
    proof_delta: Mapping[str, Any]
    input_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "parent_packet_id",
            "snapshot_id",
            "task_id",
            "finding_record_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        delta = _bounded_data(self.proof_delta, name="proof_delta")
        object.__setattr__(
            self, "proof_delta", _labeled_data(delta, source="proof_delta")
        )
        if (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or self.input_tokens <= 0
            or self.input_tokens > MAX_PACKET_INPUT_TOKENS
        ):
            raise ContractEditPacketError(
                "retry input exceeds the packet token limit",
                reason_code=ContractEditPacketReason.TOKEN_BUDGET_EXCEEDED,
            )

    @property
    def proof_delta_only(self) -> bool:
        return True

    @property
    def packet_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    @property
    def content_id(self) -> str:
        return self.packet_id

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": MCP_CONTRACT_EDIT_RETRY_INTERFACE,
                "parent_packet_id": self.parent_packet_id,
                "snapshot_id": self.snapshot_id,
                "task_id": self.task_id,
                "proof_delta": self.proof_delta,
            }
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": MCP_CONTRACT_EDIT_RETRY_SCHEMA,
            "interface": MCP_CONTRACT_EDIT_RETRY_INTERFACE,
            "version": MCP_CONTRACT_EDIT_PACKET_VERSION,
            "parent_packet_id": self.parent_packet_id,
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "finding_record_id": self.finding_record_id,
            "proof_delta": dict(self.proof_delta),
            "proof_delta_only": True,
            "input_tokens": self.input_tokens,
            "max_input_tokens": MAX_PACKET_INPUT_TOKENS,
        }
        if include_id:
            payload["packet_id"] = content_identity(payload)
            payload["content_id"] = payload["packet_id"]
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractEditRetryPacket":
        if value.get("schema") not in (None, MCP_CONTRACT_EDIT_RETRY_SCHEMA):
            raise ContractEditPacketError("unsupported retry packet schema")
        if value.get("proof_delta_only") not in (None, True):
            raise ContractEditPacketError("retry packet must be proof_delta-only")
        raw_delta = value.get("proof_delta")
        if not isinstance(raw_delta, Mapping):
            raise ContractEditPacketError("proof_delta is required")
        if raw_delta.get("data_label") == UNTRUSTED_DATA_LABEL:
            raw_delta = raw_delta.get("value")
        if not isinstance(raw_delta, Mapping):
            raise ContractEditPacketError("labeled proof_delta value is malformed")
        result = cls(
            parent_packet_id=str(value.get("parent_packet_id") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            task_id=str(value.get("task_id") or ""),
            finding_record_id=str(value.get("finding_record_id") or ""),
            proof_delta=raw_delta,
            input_tokens=value.get("input_tokens", 0),
        )
        claimed = value.get("packet_id") or value.get("content_id")
        if claimed not in (None, "", result.packet_id):
            raise ContractEditPacketError("retry packet identity mismatch")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "ContractEditRetryPacket":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractEditPacketError("retry packet JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ContractEditPacketError(
                "retry packet JSON must contain an object"
            )
        return cls.from_dict(payload)


def _require_admitted_decision(
    *,
    target_decision: RepairTargetDecision | None,
    admission: AdmissionResult | None,
) -> RepairTargetDecision:
    """Return the single admitted decision that may set @2 write authority."""

    decision: RepairTargetDecision | None = None
    if admission is not None:
        if not isinstance(admission, AdmissionResult):
            raise ContractEditPacketError(
                "admission must be AdmissionResult",
                reason_code=ContractEditPacketReason.DECISION_REQUIRED,
            )
        decision = admission.decision
    if target_decision is not None:
        if not isinstance(target_decision, RepairTargetDecision):
            raise ContractEditPacketError(
                "target_decision must be RepairTargetDecision",
                reason_code=ContractEditPacketReason.DECISION_REQUIRED,
            )
        if decision is not None and decision.content_id != target_decision.content_id:
            raise ContractEditPacketError(
                "admission and target_decision identities differ",
                reason_code=ContractEditPacketReason.DECISION_SCOPE_MISMATCH,
            )
        decision = target_decision
    if decision is None:
        raise ContractEditPacketError(
            "packet_version 2 requires an admitted RepairTargetDecision",
            reason_code=ContractEditPacketReason.DECISION_REQUIRED,
        )
    if (
        decision.disposition is not DecisionDisposition.ADMITTED
        or decision.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}
        or not decision.permitted_write_paths
    ):
        raise ContractEditPacketError(
            "only a current admitted non-abstaining decision may set write paths",
            reason_code=ContractEditPacketReason.DECISION_NOT_ADMITTED,
        )
    return decision


def materialize_contract_edit_packet(
    finding: ContractFinding | Mapping[str, Any],
    *,
    current_snapshot_id: str,
    task_id: str,
    expected_postcondition: str | Mapping[str, Any],
    validation_commands: Sequence[str],
    reproof_commands: Sequence[str],
    repository_id: str = "repository:ipfs-accelerate",
    read_paths: Sequence[str] | None = None,
    write_paths: Sequence[str] | None = None,
    compact_slice: Mapping[str, Any] | None = None,
    expansion_handles: Sequence[
        ExpansionHandle | Mapping[str, Any] | str
    ] = (),
    dependency_ids: Sequence[str] = (),
    mandatory_dependency_ids: Sequence[str] = (),
    expected_finding_record_id: str = "",
    policy_id: str = "policy:contract-edit-packet",
    policy_revision: str = "1",
    caller: str = "agent-supervisor:contract-edit-materializer",
    max_input_tokens: int = MAX_PACKET_INPUT_TOKENS,
    tokenizer: Callable[[str], Any] | None = None,
    packet_version: int = 1,
    target_decision: RepairTargetDecision | None = None,
    admission: AdmissionResult | None = None,
) -> McpContractEditPacket:
    """Materialize one current finding into a minimal implementation packet.

    ``packet_version=1`` (default) preserves legacy semantics: write paths must
    equal ``finding.affected_paths`` exactly.  ``packet_version=2`` is the
    explicit proof-gated cutover: write paths are taken only from an admitted
    ``RepairTargetDecision`` (or ``AdmissionResult``) and may diverge from the
    finding's historical affected paths (e.g. rename-to-moved-file).
    """

    item = _finding(finding)
    _assert_finding_current(
        item,
        current_snapshot_id=current_snapshot_id,
        expected_finding_record_id=expected_finding_record_id,
    )
    selected_task = _text(task_id, "task_id", single_line=True)
    selected_repository = _text(
        repository_id, "repository_id", single_line=True
    )
    selected_policy = _text(policy_id, "policy_id", single_line=True)
    selected_policy_revision = _text(
        policy_revision, "policy_revision", single_line=True
    )
    selected_caller = _text(caller, "caller", single_line=True)
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or max_input_tokens <= 0
        or max_input_tokens > MAX_PACKET_INPUT_TOKENS
    ):
        raise ContractEditPacketError(
            "max_input_tokens must be between 1 and 8,192",
            reason_code=ContractEditPacketReason.TOKEN_BUDGET_EXCEEDED,
        )
    if isinstance(packet_version, bool) or not isinstance(packet_version, int):
        raise ContractEditPacketError(
            "packet_version must be 1 or 2",
            reason_code=ContractEditPacketReason.MALFORMED,
        )
    if packet_version not in (1, MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION):
        raise ContractEditPacketError(
            "packet_version must be 1 or 2",
            reason_code=ContractEditPacketReason.MALFORMED,
        )
    # Binding a decision always selects the @2 write-authority path; callers
    # cannot half-upgrade by passing a decision under version 1.
    if target_decision is not None or admission is not None:
        packet_version = MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION

    affected_paths = tuple(item.affected_paths)
    decision: RepairTargetDecision | None = None
    write_authority = WRITE_PATH_AUTHORITY_AFFECTED_PATHS
    decision_id = ""
    if packet_version == MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION:
        decision = _require_admitted_decision(
            target_decision=target_decision, admission=admission
        )
        write_authority = WRITE_PATH_AUTHORITY_TARGET_DECISION
        decision_id = decision.content_id
        selected_write = _paths(decision.permitted_write_paths, "write_paths")
        if write_paths is not None:
            requested_write = _paths(write_paths, "write_paths")
            if requested_write != selected_write:
                raise ContractEditPacketError(
                    "write_paths must exactly equal the admitted decision allowlist",
                    reason_code=ContractEditPacketReason.DECISION_SCOPE_MISMATCH,
                )
        decision_reads = _paths(decision.permitted_read_paths, "read_paths")
        if read_paths is None:
            # Decision read authority is exact; diagnostic finding paths may be
            # added only as additional reads when the caller supplies them.
            selected_read = decision_reads
        else:
            selected_read = _paths(read_paths, "read_paths")
            if not set(decision_reads).issubset(selected_read):
                raise ContractEditPacketError(
                    "read_paths must include every decision read path",
                    reason_code=ContractEditPacketReason.DECISION_SCOPE_MISMATCH,
                )
            # Write scope remains decision-only; extra reads cannot mint writes.
            if not set(selected_write).issubset(selected_read):
                raise ContractEditPacketError(
                    "read_paths must include every decision write path",
                    reason_code=ContractEditPacketReason.DECISION_SCOPE_MISMATCH,
                )
    else:
        selected_write = _paths(
            affected_paths if write_paths is None else write_paths,
            "write_paths",
        )
        if selected_write != affected_paths:
            raise ContractEditPacketError(
                "write_paths must exactly match the finding's affected paths",
                reason_code=ContractEditPacketReason.PATH_SCOPE_MISMATCH,
            )
        selected_read = _paths(
            affected_paths if read_paths is None else read_paths,
            "read_paths",
        )
        if not set(affected_paths).issubset(selected_read):
            raise ContractEditPacketError(
                "read_paths must include every affected path",
                reason_code=ContractEditPacketReason.PATH_SCOPE_MISMATCH,
            )

    validations = _commands(validation_commands, "validation_commands")
    reproof = _commands(reproof_commands, "reproof_commands")
    obligations = _ids(
        item.reproduction.obligation_ids,
        "obligation_ids",
        required=True,
    )
    dependencies = _ids(dependency_ids, "dependency_ids")
    mandatory = _ids(
        mandatory_dependency_ids, "mandatory_dependency_ids"
    )
    missing = set(mandatory).difference(dependencies)
    if missing:
        raise ContractEditPacketError(
            "mandatory dependencies omitted: " + ", ".join(sorted(missing)),
            reason_code=ContractEditPacketReason.MISSING_MANDATORY_DEPENDENCY,
        )

    default_slice: Mapping[str, Any] = {
        "affected_paths": list(affected_paths),
        "affected_symbols": list(item.affected_symbols),
        "impact_truncated": item.impact_truncated,
    }
    if decision is not None:
        default_slice = {
            **default_slice,
            "decision_write_paths": list(selected_write),
            "decision_read_paths": list(decision.permitted_read_paths),
            "selected_candidate_id": decision.selected_candidate_id,
            "strategy": decision.strategy.value,
        }
    selected_slice = _labeled_data(
        compact_slice if compact_slice is not None else default_slice,
        source="bounded_contract_slice",
    )
    counterexample = _labeled_data(
        item.counterexample,
        source="contract_counterexample",
    )
    postcondition = _bounded_data(
        expected_postcondition,
        name="expected_postcondition",
    )
    if postcondition in ("", {}, []):
        raise ContractEditPacketError(
            "expected_postcondition is required",
            reason_code=ContractEditPacketReason.REQUIRED_CORE_MISSING,
        )

    cas_handles = tuple(
        ExpansionHandle(
            handle_id=f"finding-cas:{index:04d}",
            kind="finding_artifact",
            content_id=content_id,
        )
        for index, content_id in enumerate(item.reproduction.cas_handles)
    )
    handles = _coerce_handles((*cas_handles, *expansion_handles))

    goal: dict[str, Any] = {
        "task_id": selected_task,
        "finding_id": item.finding_id,
        "finding_record_id": item.record_id,
        "contract_ids": [item.contract_id],
        "obligation_ids": list(obligations),
        "affected_symbols": list(item.affected_symbols),
        "bounded_contract_slice": selected_slice,
        "counterexample_id": item.counterexample_id,
        "counterexample": counterexample,
        "failed_premise_ids": sorted(
            {
                premise_id
                for evidence in item.evidence
                for premise_id in evidence.premise_ids
            }
        ),
        "reason_codes": sorted(
            {
                reason
                for evidence in item.evidence
                for reason in evidence.reason_codes
            }
        ),
        "packet_version": packet_version,
    }
    if decision_id:
        goal["decision_id"] = decision_id
        goal["selected_strategy"] = decision.strategy.value if decision else ""
        goal["selected_candidate_id"] = (
            decision.selected_candidate_id if decision else ""
        )
    authority = {
        "provider_semantic_authority": False,
        "artifact_bodies_embedded": False,
        "expansion_requires_handle": True,
        "untrusted_data_label": UNTRUSTED_DATA_LABEL,
        "untrusted_text_treatment": DATA_NOT_INSTRUCTIONS,
        "write_path_authority": write_authority,
        "packet_version": packet_version,
    }
    if decision_id:
        authority["decision_id"] = decision_id
    scope = {
        "read_paths": list(selected_read),
        "write_paths": list(selected_write),
        "path_allowlists_exact": True,
        "dependency_ids": list(dependencies),
        "mandatory_dependency_ids": list(mandatory),
        "write_path_authority": write_authority,
    }
    if decision_id:
        scope["decision_id"] = decision_id
    acceptance = {
        "expected_postcondition": postcondition,
        "validation_commands": list(validations),
        "reproof_commands": list(reproof),
        "current_snapshot_required": item.snapshot_id,
    }
    references = tuple(
        _context_reference(
            handle,
            repository_id=selected_repository,
            tree_id=item.snapshot_id,
        )
        for handle in handles
    )
    provider_payload = {
        "contract_version": 1,
        "repository_id": selected_repository,
        "tree_id": item.snapshot_id,
        "objective_id": selected_task,
        "objective_revision": item.finding_id,
        "policy_id": selected_policy,
        "policy_revision": selected_policy_revision,
        "caller": selected_caller,
        "stage": "implementation",
        "goal": goal,
        "authority": authority,
        "scope": scope,
        "acceptance": acceptance,
        "evidence": (),
    }
    input_tokens = _measure_tokens(provider_payload, tokenizer)
    if input_tokens > max_input_tokens:
        raise ContractEditPacketError(
            "required packet core exceeds its input-token budget",
            reason_code=ContractEditPacketReason.TOKEN_BUDGET_EXCEEDED,
        )

    budget = ContextBudget(
        max_input_tokens=max_input_tokens,
        # ContextBudget applies this structural bound independently while
        # freezing each invariant core field; it is not merely the expansion
        # reference count.  The packet's own validators retain tighter
        # per-field and per-list bounds.
        max_items=max(1_024, len(references)),
        max_item_bytes=16_384,
        max_serialized_bytes=DEFAULT_PACKET_MAX_SERIALIZED_BYTES,
        max_depth=12,
        max_text_bytes=16_384,
    )
    context = ContextCapsule(
        repository_id=selected_repository,
        tree_id=item.snapshot_id,
        objective_id=selected_task,
        objective_revision=item.finding_id,
        policy_id=selected_policy,
        policy_revision=selected_policy_revision,
        caller=selected_caller,
        stage="implementation",
        budget=budget,
        goal=goal,
        authority=authority,
        scope=scope,
        acceptance=acceptance,
        evidence=(),
        expansion_references=references,
        input_tokens=input_tokens,
        truncated=bool(references),
        omissions=tuple(
            f"{reference.reference_id}:token_budget"
            for reference in references
        ),
    )
    # Measure the contract's exact provider projection, not just the pre-build
    # structurally equivalent object.  Rebuild when canonical record fields
    # make the conservative estimate larger.
    exact_tokens = _measure_tokens(context.provider_input_payload, tokenizer)
    if exact_tokens > max_input_tokens:
        raise ContractEditPacketError(
            "required packet core exceeds its input-token budget",
            reason_code=ContractEditPacketReason.TOKEN_BUDGET_EXCEEDED,
        )
    if exact_tokens != context.input_tokens:
        context = ContextCapsule(
            repository_id=context.repository_id,
            tree_id=context.tree_id,
            objective_id=context.objective_id,
            objective_revision=context.objective_revision,
            policy_id=context.policy_id,
            policy_revision=context.policy_revision,
            caller=context.caller,
            stage=context.stage,
            budget=context.budget,
            goal=context.goal,
            authority=context.authority,
            scope=context.scope,
            acceptance=context.acceptance,
            evidence=context.evidence,
            expansion_references=context.expansion_references,
            input_tokens=exact_tokens,
            truncated=context.truncated,
            omissions=context.omissions,
        )

    base = build_code_edit_packet(
        repository_tree_id=item.snapshot_id,
        repository_id=selected_repository,
        task_id=selected_task,
        claim_ids=tuple(evidence.claim_id for evidence in item.evidence),
        obligation_ids=obligations,
        invalidation_reasons=tuple(
            reason
            for evidence in item.evidence
            for reason in evidence.reason_codes
        ),
        predicted_files=selected_write,
        acceptance_ids=(
            content_identity(postcondition),
            *(content_identity(command) for command in validations),
            *(content_identity(command) for command in reproof),
        ),
        property_ids=(item.contract_id,),
        residual_ref_ids=tuple(handle.content_id for handle in handles),
        metadata={
            "contract_finding_id": item.finding_id,
            "contract_finding_record_id": item.record_id,
            "counterexample_id": item.counterexample_id,
            "required_core_non_truncatable": True,
            "artifact_bodies_embedded": False,
            "path_allowlists_exact": True,
        },
    )
    return McpContractEditPacket(
        base_packet=base,
        context_capsule=context,
        finding_id=item.finding_id,
        finding_record_id=item.record_id,
        counterexample_id=item.counterexample_id,
        dependency_ids=dependencies,
        mandatory_dependency_ids=mandatory,
        expansion_handles=handles,
    )


def build_contract_edit_retry(
    parent: McpContractEditPacket,
    *,
    proof_delta: Mapping[str, Any],
    current_snapshot_id: str,
    finding_record_id: str = "",
    tokenizer: Callable[[str], Any] | None = None,
) -> ContractEditRetryPacket:
    """Build a compact unchanged retry; stable packet core is not repeated."""

    if not isinstance(parent, McpContractEditPacket):
        raise ContractEditPacketError("parent must be a contract edit packet")
    parent.assert_current(
        current_snapshot_id,
        finding_record_id=finding_record_id,
    )
    if not isinstance(proof_delta, Mapping):
        raise ContractEditPacketError("proof_delta must be an object")
    delta = _bounded_data(proof_delta, name="proof_delta")
    labeled = _labeled_data(delta, source="proof_delta")
    provider_payload = {
        "interface": MCP_CONTRACT_EDIT_RETRY_INTERFACE,
        "parent_packet_id": parent.packet_id,
        "snapshot_id": parent.snapshot_id,
        "task_id": parent.task_id,
        "proof_delta": labeled,
    }
    tokens = _measure_tokens(provider_payload, tokenizer)
    return ContractEditRetryPacket(
        parent_packet_id=parent.packet_id,
        snapshot_id=parent.snapshot_id,
        task_id=parent.task_id,
        finding_record_id=parent.finding_record_id,
        proof_delta=delta,
        input_tokens=tokens,
    )


def packet_token_median(
    packets: Sequence[McpContractEditPacket],
) -> int:
    """Return the deterministic integer median used by compact fixtures."""

    if not packets:
        raise ContractEditPacketError("at least one packet is required")
    values = sorted(packet.input_tokens for packet in packets)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) // 2


# Compatibility spellings for callers which use either MCP- or generic packet
# vocabulary.
CodeEditPacket = McpContractEditPacket
ContractEditPacket = McpContractEditPacket
ContractDirectedEditPacket = McpContractEditPacket
McpContractEditPacketError = ContractEditPacketError
materialize_mcp_contract_edit_packet = materialize_contract_edit_packet
build_mcp_contract_edit_packet = materialize_contract_edit_packet
materialize_edit_packet = materialize_contract_edit_packet
materialize_retry_packet = build_contract_edit_retry
build_proof_delta_retry = build_contract_edit_retry


__all__ = [
    "BaseCodeEditPacket",
    "CodeEditPacket",
    "ContractDirectedEditPacket",
    "ContractEditPacket",
    "ContractEditPacketError",
    "ContractEditPacketReason",
    "ContractEditRetryPacket",
    "DATA_NOT_INSTRUCTIONS",
    "DEFAULT_PACKET_MAX_SERIALIZED_BYTES",
    "ExpansionHandle",
    "FIXTURE_MEDIAN_TARGET_TOKENS",
    "MAX_PACKET_INPUT_TOKENS",
    "MCP_CONTRACT_EDIT_PACKET_CONTRACT_VERSION",
    "MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION",
    "MCP_CONTRACT_EDIT_PACKET_INTERFACE",
    "MCP_CONTRACT_EDIT_PACKET_SCHEMA",
    "MCP_CONTRACT_EDIT_PACKET_VERSION",
    "MCP_CONTRACT_EDIT_RETRY_INTERFACE",
    "MCP_CONTRACT_EDIT_RETRY_SCHEMA",
    "McpContractEditPacket",
    "McpContractEditPacketError",
    "UNTRUSTED_DATA_LABEL",
    "WRITE_PATH_AUTHORITY_AFFECTED_PATHS",
    "WRITE_PATH_AUTHORITY_TARGET_DECISION",
    "build_contract_edit_retry",
    "build_mcp_contract_edit_packet",
    "build_proof_delta_retry",
    "materialize_contract_edit_packet",
    "materialize_edit_packet",
    "materialize_mcp_contract_edit_packet",
    "materialize_retry_packet",
    "packet_token_median",
]
