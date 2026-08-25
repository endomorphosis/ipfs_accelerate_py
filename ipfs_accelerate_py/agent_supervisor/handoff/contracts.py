"""Transport-neutral External Agent Handoff contract family (EAAEF-010).

These records are the shared serialization boundary for Python, CLI, and MCP
handoff surfaces.  They are immutable, DAG-JSON compatible, content addressed,
and strictly versioned at major ``@1``.  Unknown schema names, unknown major
versions, floats, private material, and hidden chain-of-thought are rejected.

Raw export bytes, the ordered normalized event stream, the session, the
objective, context artifacts, the repository, and patches keep distinct
identities.  Imported history is provenance and never authority: imported tool
calls are not executed, imported success claims are not trusted, and only a
locally reverified or independently admitted receipt may satisfy a completion
gate.  Public receipts carry content-addressed references, not transcript
bodies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeAlias, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


HANDOFF_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = HANDOFF_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = HANDOFF_CONTRACT_VERSION

EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE: Final[str] = "ExternalAgentHandoffRequest@1"
EXTERNAL_AGENT_SESSION_INTERFACE: Final[str] = "ExternalAgentSession@1"
CONVERSATION_EVENT_INTERFACE: Final[str] = "ConversationEvent@1"
TOOL_INVOCATION_EVENT_INTERFACE: Final[str] = "ToolInvocationEvent@1"
TOOL_RESULT_EVENT_INTERFACE: Final[str] = "ToolResultEvent@1"
PATCH_EVENT_INTERFACE: Final[str] = "PatchEvent@1"
APPROVAL_EVENT_INTERFACE: Final[str] = "ApprovalEvent@1"
AGENT_CHECKPOINT_INTERFACE: Final[str] = "AgentCheckpoint@1"
AGENT_CONTEXT_ARTIFACT_INTERFACE: Final[str] = "AgentContextArtifact@1"
HANDOFF_NORMALIZATION_REPORT_INTERFACE: Final[str] = "HandoffNormalizationReport@1"
HANDOFF_ADMISSION_RECEIPT_INTERFACE: Final[str] = "HandoffAdmissionReceipt@1"
HANDOFF_BOUNDS_INTERFACE: Final[str] = "HandoffBounds@1"
HANDOFF_PROVENANCE_INTERFACE: Final[str] = "HandoffProvenance@1"
ENCRYPTED_EXPORT_REFERENCE_INTERFACE: Final[str] = "EncryptedExportReference@1"

EXTERNAL_AGENT_HANDOFF_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-handoff-request@1"
)
EXTERNAL_AGENT_SESSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-session@1"
)
CONVERSATION_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/conversation-event@1"
)
TOOL_INVOCATION_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tool-invocation-event@1"
)
TOOL_RESULT_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tool-result-event@1"
)
PATCH_EVENT_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/patch-event@1"
APPROVAL_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/approval-event@1"
)
AGENT_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/agent-checkpoint@1"
)
AGENT_CONTEXT_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/agent-context-artifact@1"
)
HANDOFF_NORMALIZATION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/handoff-normalization-report@1"
)
HANDOFF_ADMISSION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/handoff-admission-receipt@1"
)
HANDOFF_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/handoff-bounds@1"
)
HANDOFF_PROVENANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/handoff-provenance@1"
)
ENCRYPTED_EXPORT_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/encrypted-export-reference@1"
)
NORMALIZED_STREAM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/handoff-normalized-stream@1"
)

HANDOFF_CONTRACT_FAMILY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "request": EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE,
        "session": EXTERNAL_AGENT_SESSION_INTERFACE,
        "conversation_event": CONVERSATION_EVENT_INTERFACE,
        "tool_invocation_event": TOOL_INVOCATION_EVENT_INTERFACE,
        "tool_result_event": TOOL_RESULT_EVENT_INTERFACE,
        "patch_event": PATCH_EVENT_INTERFACE,
        "approval_event": APPROVAL_EVENT_INTERFACE,
        "checkpoint": AGENT_CHECKPOINT_INTERFACE,
        "context_artifact": AGENT_CONTEXT_ARTIFACT_INTERFACE,
        "normalization_report": HANDOFF_NORMALIZATION_REPORT_INTERFACE,
        "admission_receipt": HANDOFF_ADMISSION_RECEIPT_INTERFACE,
    }
)

ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_RECORD_BYTES: Final[int] = 1_048_576
ABSOLUTE_MAX_EVENTS: Final[int] = 4_096
ABSOLUTE_MAX_CHECKPOINTS: Final[int] = 256
ABSOLUTE_MAX_CONTEXT_ARTIFACTS: Final[int] = 512
ABSOLUTE_MAX_PATHS: Final[int] = 512
ABSOLUTE_MAX_DEPTH: Final[int] = 16
ABSOLUTE_MAX_ITEMS: Final[int] = 4_096
ABSOLUTE_MAX_UNKNOWN_FIELDS: Final[int] = 32
ABSOLUTE_MAX_UNKNOWN_FIELD_BYTES: Final[int] = 8_192
ABSOLUTE_MAX_ID_BYTES: Final[int] = 256
ABSOLUTE_MAX_REASON_BYTES: Final[int] = 256

DEFAULT_MAX_EVENTS: Final[int] = 1_024
DEFAULT_MAX_CHECKPOINTS: Final[int] = 64
DEFAULT_MAX_CONTEXT_ARTIFACTS: Final[int] = 128
DEFAULT_MAX_PATHS: Final[int] = 128
DEFAULT_MAX_TEXT_BYTES: Final[int] = 16_384
DEFAULT_MAX_RECORD_BYTES: Final[int] = 65_536
DEFAULT_MAX_SERIALIZED_BYTES: Final[int] = 262_144
DEFAULT_MAX_DEPTH: Final[int] = 8
DEFAULT_MAX_UNKNOWN_FIELDS: Final[int] = 16
DEFAULT_MAX_UNKNOWN_FIELD_BYTES: Final[int] = 2_048
DEFAULT_MAX_ID_BYTES: Final[int] = 128

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_CIDV1_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_HIDDEN_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "hidden_chain_of_thought",
        "hidden_cot",
        "hidden_reasoning",
        "hidden_thoughts",
        "internal_monologue",
        "model_thoughts",
        "private_reasoning",
        "private_thinking",
        "scratchpad",
        "thinking",
        "thinking_blocks",
        "thinking_private",
        "thinking_text",
    }
)
_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "transcript_body",
        "witness",
    }
)
_TRANSCRIPT_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "full_transcript",
        "raw_bytes",
        "raw_export",
        "raw_transcript",
        "transcript",
        "transcript_body",
        "transcript_text",
    }
)
TEnum = TypeVar("TEnum", bound=Enum)


class HandoffContractError(ContractValidationError):
    """Malformed or unsafe external-agent handoff contract."""


class HandoffBoundsError(HandoffContractError):
    """A handoff value exceeded a declared resource bound."""


class HandoffIdentityError(HandoffContractError):
    """A claimed content identity did not match its canonical payload."""


class HandoffVersionError(HandoffContractError):
    """Unsupported handoff schema name or contract version."""


class HandoffTrustError(HandoffContractError):
    """Imported history attempted to grant authority or completion."""


class SourceFamily(str, Enum):
    """Closed, transport-neutral source families admitted by @1 adapters."""

    CODEX = "codex"
    CLAUDE_CODE = "claude_code"
    GEMINI_CLI = "gemini_cli"
    GENERIC_MCP = "generic_mcp"
    GENERIC_JSON = "generic_json"
    GENERIC_JSONL = "generic_jsonl"


class TrustClass(str, Enum):
    """Trust assigned to imported or locally checked handoff material."""

    IMPORTED_UNVERIFIED = "imported_unverified"
    IMPORTED_EXPORTABLE = "imported_exportable"
    LOCALLY_REVERIFIED = "locally_reverified"
    INDEPENDENTLY_ADMITTED = "independently_admitted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"

    @property
    def may_satisfy_completion(self) -> bool:
        return self in {
            TrustClass.LOCALLY_REVERIFIED,
            TrustClass.INDEPENDENTLY_ADMITTED,
        }

    @property
    def imported(self) -> bool:
        return self in {
            TrustClass.IMPORTED_UNVERIFIED,
            TrustClass.IMPORTED_EXPORTABLE,
        }


class EventKind(str, Enum):
    """Closed event discriminator for the normalized stream."""

    CONVERSATION = "conversation"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_RESULT = "tool_result"
    PATCH = "patch"
    APPROVAL = "approval"


class ConversationRole(str, Enum):
    """Exportable conversation roles.  Hidden model roles are not represented."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"


class PatchKind(str, Enum):
    """Content-addressed patch representations admitted on the wire."""

    UNIFIED_DIFF = "unified_diff"
    GIT_COMMIT = "git_commit"
    OVERLAY_REFERENCE = "overlay_reference"


class ApprovalKind(str, Enum):
    """How an approval event was produced."""

    HUMAN = "human"
    IMPORTED_CLAIM = "imported_claim"
    POLICY = "policy"


class ApprovalDecision(str, Enum):
    """Closed approval vocabulary.  Imported claims never grant effects."""

    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


class HandoffMode(str, Enum):
    """Caller-requested handoff operation.  Not an authority grant."""

    PREVIEW = "preview"
    ATTACH = "attach"
    CONTINUE = "continue"
    IMPORT_ONLY = "import_only"


class AdmissionVerdict(str, Enum):
    """Closed admission outcomes.  Preview is not mutation admission."""

    ADMITTED = "admitted"
    PREVIEW_ONLY = "preview_only"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"

    @property
    def admits_session(self) -> bool:
        return self in {AdmissionVerdict.ADMITTED, AdmissionVerdict.PREVIEW_ONLY}


class RetentionClass(str, Enum):
    """Retention bound carried with encrypted raw material and projections."""

    EPHEMERAL = "ephemeral"
    SESSION = "session"
    RETAINED = "retained"
    LEGAL_HOLD = "legal_hold"


class DisclosureClass(str, Enum):
    """Disclosure bound.  Public receipts use public_projection or redacted."""

    PUBLIC_PROJECTION = "public_projection"
    ENCRYPTED_RAW = "encrypted_raw"
    REDACTED = "redacted"
    LOCAL_ONLY = "local_only"


class ContextArtifactKind(str, Enum):
    """Kinds of context artifacts referenced from a session."""

    PROMPT = "prompt"
    OBJECTIVE = "objective"
    FILE_REFERENCE = "file_reference"
    CAPSULE = "capsule"
    POLICY = "policy"
    OTHER = "other"


class EncryptionAlgorithm(str, Enum):
    """Closed encryption algorithms for raw-export ciphertext references."""

    AES_256_GCM = "aes-256-gcm"


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _enum(value: Any, enum_type: type[TEnum], name: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise HandoffContractError(f"{name} must be one of: {allowed}") from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise HandoffContractError(f"{name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise HandoffContractError(f"{name} is required")
    if "\x00" in result:
        raise HandoffContractError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise HandoffBoundsError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise HandoffContractError(f"{name} must be a boolean")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HandoffContractError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise HandoffContractError(f"{name} must be at least 1")
    return result


def _major_version(name: str) -> int | None:
    if not isinstance(name, str) or "@" not in name:
        return None
    suffix = name.rsplit("@", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _require_versioned_name(name: str, expected: str, field_name: str) -> None:
    if name != expected:
        supplied_major = _major_version(name)
        expected_major = _major_version(expected) or HANDOFF_CONTRACT_VERSION
        if supplied_major is not None and supplied_major != expected_major:
            raise HandoffVersionError(
                f"unsupported {field_name} {name!r}; rebuild with {expected}"
            )
        raise HandoffVersionError(
            f"unsupported {field_name} {name!r}; expected {expected}"
        )


def _schema_and_version(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_interface: str,
    *,
    artifact_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise HandoffContractError(f"{artifact_name} payload must be an object")
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        _require_versioned_name(str(schema), expected_schema, "schema")
    interface = payload.get("interface")
    if interface not in (None, "", expected_interface):
        _require_versioned_name(str(interface), expected_interface, "interface")
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", HANDOFF_CONTRACT_VERSION):
            raise HandoffVersionError(
                f"unsupported {artifact_name} contract version; rebuild with "
                f"{expected_interface}"
            )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], *, artifact_name: str
) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise HandoffContractError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )


def _claimed_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise HandoffIdentityError(
                f"{artifact_name} content identity does not match payload"
            )


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    max_items: int = ABSOLUTE_MAX_EVENTS,
    max_bytes: int = ABSOLUTE_MAX_ID_BYTES,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise HandoffContractError(f"{name} must be a sequence of strings")
    else:
        items = values
    if len(items) > max_items:
        raise HandoffBoundsError(f"{name} exceeds its item-count limit")
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _content_ref(item, name, max_bytes=max_bytes)
        if text in seen:
            raise HandoffContractError(f"{name} must not contain duplicate identities")
        seen.add(text)
        result.append(text)
    if required and not result:
        raise HandoffContractError(f"{name} must not be empty")
    return tuple(result)


def _relative_paths(
    values: Any, name: str, *, max_items: int = ABSOLUTE_MAX_PATHS
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise HandoffContractError(f"{name} must be a sequence of paths")
    else:
        items = values
    if len(items) > max_items:
        raise HandoffBoundsError(f"{name} exceeds its item-count limit")
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item, name, max_bytes=DEFAULT_MAX_ID_BYTES * 4).replace("\\", "/")
        candidate = PurePosixPath(text)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or (candidate.parts and candidate.parts[0].endswith(":"))
        ):
            raise HandoffContractError(f"{name} must be repository-relative")
        normalized = candidate.as_posix().removeprefix("./")
        if normalized in ("", "."):
            raise HandoffContractError(f"{name} must not be empty")
        if normalized in seen:
            raise HandoffContractError(f"{name} must not contain duplicate paths")
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _digest_sha256(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, max_bytes=80)
    if not text:
        return ""
    if _HEX64_RE.fullmatch(text):
        return f"sha256:{text}"
    if _SHA256_RE.fullmatch(text):
        return text
    raise HandoffContractError(f"{name} must be a sha256 hex digest")


def _content_ref(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_ID_BYTES,
) -> str:
    text = _text(value, name, required=required, max_bytes=max_bytes)
    if not text:
        return ""
    if _SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text):
        return text
    raise HandoffContractError(f"{name} must be a sha256 or CIDv1 identity")


def _key_is_forbidden(key: str) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _HIDDEN_CHAIN_OF_THOUGHT_KEYS:
        return "hidden_chain_of_thought"
    if normalized in _TRANSCRIPT_BODY_KEYS:
        return "transcript_body"
    if any(
        normalized == marker or normalized.endswith("_" + marker) or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    ):
        return "private_material"
    return None


def _reject_forbidden_keys(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            reason = _key_is_forbidden(str(raw_key))
            if reason == "hidden_chain_of_thought":
                raise HandoffContractError(
                    f"{name} must not represent hidden chain-of-thought"
                )
            if reason == "transcript_body":
                raise HandoffContractError(
                    f"{name} must not embed transcript bodies; use content-addressed references"
                )
            if reason == "private_material":
                raise HandoffContractError(
                    f"{name} must not contain private material"
                )
            _reject_forbidden_keys(item, name=name)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _reject_forbidden_keys(item, name=name)


def _freeze_bounded(
    value: Any,
    *,
    name: str,
    max_depth: int,
    max_items: int,
    max_text_bytes: int,
) -> Any:
    seen = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise HandoffBoundsError(f"{name} exceeds its item-count limit")
        if depth > max_depth:
            raise HandoffBoundsError(f"{name} exceeds its nesting-depth limit")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise HandoffContractError(f"{name} cannot contain floats")
        if isinstance(item, str):
            return _text(item, name, required=False, max_bytes=max_text_bytes)
        if isinstance(item, Enum):
            return visit(item.value, depth)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise HandoffContractError(f"{name} object keys must be strings")
            frozen: dict[str, Any] = {}
            for key in sorted(item):
                normalized_key = _text(key, f"{name} key", max_bytes=max_text_bytes)
                reason = _key_is_forbidden(normalized_key)
                if reason == "hidden_chain_of_thought":
                    raise HandoffContractError(
                        f"{name} must not represent hidden chain-of-thought"
                    )
                if reason == "transcript_body":
                    raise HandoffContractError(
                        f"{name} must not embed transcript bodies; use content-addressed references"
                    )
                if reason == "private_material":
                    raise HandoffContractError(
                        f"{name} must not contain private material"
                    )
                frozen[normalized_key] = visit(item[key], depth + 1)
            return MappingProxyType(frozen)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1) for member in item)
        raise HandoffContractError(
            f"{name} contains unsupported value type {type(item).__name__}"
        )

    return visit(value, 0)


def _residual_fields(
    value: Any,
    *,
    bounds: "HandoffBounds",
) -> Mapping[str, Any]:
    if value is None:
        frozen = MappingProxyType({})
    elif not isinstance(value, Mapping):
        raise HandoffContractError("residual_fields must be an object")
    else:
        if len(value) > bounds.max_unknown_fields:
            raise HandoffBoundsError("residual_fields exceeds its field-count limit")
        frozen = _freeze_bounded(
            value,
            name="residual_fields",
            max_depth=bounds.max_depth,
            max_items=ABSOLUTE_MAX_ITEMS,
            max_text_bytes=bounds.max_text_bytes,
        )
        if not isinstance(frozen, Mapping):
            raise HandoffContractError("residual_fields must be an object")
    encoded = canonical_json_bytes(dict(frozen))
    if len(encoded) > bounds.max_unknown_field_bytes:
        raise HandoffBoundsError("residual_fields exceeds its byte bound")
    return frozen


def _distinct_identities(pairs: Sequence[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, identity in pairs:
        if not identity:
            continue
        previous = seen.get(identity)
        if previous is not None and previous != name:
            raise HandoffIdentityError(
                f"{name} identity must be distinct from {previous}"
            )
        seen[identity] = name


def normalized_stream_identity(event_content_ids: Sequence[str]) -> str:
    """Return the content identity of one ordered normalized event stream."""

    return content_identity(
        {
            "schema": NORMALIZED_STREAM_SCHEMA,
            "contract_version": HANDOFF_CONTRACT_VERSION,
            "event_content_ids": list(event_content_ids),
        }
    )


def _envelope(interface: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface": interface,
        "contract_version": HANDOFF_CONTRACT_VERSION,
        **dict(body),
    }


def _require_record_bound(
    record: CanonicalContract,
    *,
    artifact_name: str,
    bounds: HandoffBounds | None = None,
    serialized: bool = False,
) -> None:
    size = len(record.canonical_bytes())
    if size > ABSOLUTE_MAX_RECORD_BYTES:
        raise HandoffBoundsError(
            f"{artifact_name} exceeds the absolute record bound of "
            f"{ABSOLUTE_MAX_RECORD_BYTES} bytes"
        )
    if bounds is None:
        return
    limit = bounds.max_serialized_bytes if serialized else bounds.max_record_bytes
    limit_name = "max_serialized_bytes" if serialized else "max_record_bytes"
    if size > limit:
        raise HandoffBoundsError(f"{artifact_name} exceeds {limit_name}")


class _HandoffCanonicalContract(CanonicalContract):
    """Canonical mixin that preserves handoff error types on decode."""

    INTERFACE: ClassVar[str] = ""

    @property
    def schema_version(self) -> int:
        return HANDOFF_CONTRACT_VERSION

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @classmethod
    def from_json(cls, payload: str) -> "_HandoffCanonicalContract":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HandoffContractError("handoff contract JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise HandoffContractError("handoff contract JSON must contain an object")
        decoder = getattr(cls, "from_dict", None)
        if decoder is None:
            raise HandoffContractError(f"{cls.__name__} does not support from_dict")
        return decoder(value)


_COMMON_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "schema_version",
        "content_id",
        "cid",
        "identity",
        "canonical_id",
    }
)


@dataclass(frozen=True)
class HandoffBounds(_HandoffCanonicalContract):
    """Absolute and default count/byte/depth limits for one handoff record."""

    SCHEMA: ClassVar[str] = HANDOFF_BOUNDS_SCHEMA
    INTERFACE: ClassVar[str] = HANDOFF_BOUNDS_INTERFACE

    max_events: int = DEFAULT_MAX_EVENTS
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS
    max_context_artifacts: int = DEFAULT_MAX_CONTEXT_ARTIFACTS
    max_paths: int = DEFAULT_MAX_PATHS
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES
    max_serialized_bytes: int = DEFAULT_MAX_SERIALIZED_BYTES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_unknown_fields: int = DEFAULT_MAX_UNKNOWN_FIELDS
    max_unknown_field_bytes: int = DEFAULT_MAX_UNKNOWN_FIELD_BYTES
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", _positive_int(self.max_events, "max_events"))
        object.__setattr__(
            self, "max_checkpoints", _positive_int(self.max_checkpoints, "max_checkpoints")
        )
        object.__setattr__(
            self,
            "max_context_artifacts",
            _positive_int(self.max_context_artifacts, "max_context_artifacts"),
        )
        object.__setattr__(self, "max_paths", _positive_int(self.max_paths, "max_paths"))
        object.__setattr__(
            self, "max_text_bytes", _positive_int(self.max_text_bytes, "max_text_bytes")
        )
        object.__setattr__(
            self, "max_record_bytes", _positive_int(self.max_record_bytes, "max_record_bytes")
        )
        object.__setattr__(
            self,
            "max_serialized_bytes",
            _positive_int(self.max_serialized_bytes, "max_serialized_bytes"),
        )
        object.__setattr__(self, "max_depth", _positive_int(self.max_depth, "max_depth"))
        object.__setattr__(
            self,
            "max_unknown_fields",
            _positive_int(self.max_unknown_fields, "max_unknown_fields"),
        )
        object.__setattr__(
            self,
            "max_unknown_field_bytes",
            _positive_int(self.max_unknown_field_bytes, "max_unknown_field_bytes"),
        )
        object.__setattr__(
            self, "max_id_bytes", _positive_int(self.max_id_bytes, "max_id_bytes")
        )
        if self.max_events > ABSOLUTE_MAX_EVENTS:
            raise HandoffBoundsError("max_events exceeds the absolute limit")
        if self.max_checkpoints > ABSOLUTE_MAX_CHECKPOINTS:
            raise HandoffBoundsError("max_checkpoints exceeds the absolute limit")
        if self.max_context_artifacts > ABSOLUTE_MAX_CONTEXT_ARTIFACTS:
            raise HandoffBoundsError("max_context_artifacts exceeds the absolute limit")
        if self.max_paths > ABSOLUTE_MAX_PATHS:
            raise HandoffBoundsError("max_paths exceeds the absolute limit")
        if self.max_text_bytes > ABSOLUTE_MAX_TEXT_BYTES:
            raise HandoffBoundsError("max_text_bytes exceeds the absolute limit")
        if self.max_record_bytes > ABSOLUTE_MAX_RECORD_BYTES:
            raise HandoffBoundsError("max_record_bytes exceeds the absolute limit")
        if self.max_serialized_bytes > ABSOLUTE_MAX_RECORD_BYTES:
            raise HandoffBoundsError("max_serialized_bytes exceeds the absolute limit")
        if self.max_depth > ABSOLUTE_MAX_DEPTH:
            raise HandoffBoundsError("max_depth exceeds the absolute limit")
        if self.max_unknown_fields > ABSOLUTE_MAX_UNKNOWN_FIELDS:
            raise HandoffBoundsError("max_unknown_fields exceeds the absolute limit")
        if self.max_unknown_field_bytes > ABSOLUTE_MAX_UNKNOWN_FIELD_BYTES:
            raise HandoffBoundsError("max_unknown_field_bytes exceeds the absolute limit")
        if self.max_id_bytes > ABSOLUTE_MAX_ID_BYTES:
            raise HandoffBoundsError("max_id_bytes exceeds the absolute limit")
        if self.max_text_bytes > self.max_record_bytes:
            raise HandoffBoundsError("max_text_bytes cannot exceed max_record_bytes")
        if self.max_record_bytes > self.max_serialized_bytes:
            raise HandoffBoundsError("max_record_bytes cannot exceed max_serialized_bytes")
        if self.max_unknown_field_bytes > self.max_record_bytes:
            raise HandoffBoundsError(
                "max_unknown_field_bytes cannot exceed max_record_bytes"
            )

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "max_events": self.max_events,
                "max_checkpoints": self.max_checkpoints,
                "max_context_artifacts": self.max_context_artifacts,
                "max_paths": self.max_paths,
                "max_text_bytes": self.max_text_bytes,
                "max_record_bytes": self.max_record_bytes,
                "max_serialized_bytes": self.max_serialized_bytes,
                "max_depth": self.max_depth,
                "max_unknown_fields": self.max_unknown_fields,
                "max_unknown_field_bytes": self.max_unknown_field_bytes,
                "max_id_bytes": self.max_id_bytes,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "HandoffBounds":
        if payload is None:
            return cls()
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="handoff bounds"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "max_events",
                    "max_checkpoints",
                    "max_context_artifacts",
                    "max_paths",
                    "max_text_bytes",
                    "max_record_bytes",
                    "max_serialized_bytes",
                    "max_depth",
                    "max_unknown_fields",
                    "max_unknown_field_bytes",
                    "max_id_bytes",
                }
            ),
            artifact_name="handoff bounds",
        )
        defaults = cls()
        result = cls(
            max_events=payload.get("max_events", defaults.max_events),
            max_checkpoints=payload.get("max_checkpoints", defaults.max_checkpoints),
            max_context_artifacts=payload.get(
                "max_context_artifacts", defaults.max_context_artifacts
            ),
            max_paths=payload.get("max_paths", defaults.max_paths),
            max_text_bytes=payload.get("max_text_bytes", defaults.max_text_bytes),
            max_record_bytes=payload.get("max_record_bytes", defaults.max_record_bytes),
            max_serialized_bytes=payload.get(
                "max_serialized_bytes", defaults.max_serialized_bytes
            ),
            max_depth=payload.get("max_depth", defaults.max_depth),
            max_unknown_fields=payload.get(
                "max_unknown_fields", defaults.max_unknown_fields
            ),
            max_unknown_field_bytes=payload.get(
                "max_unknown_field_bytes", defaults.max_unknown_field_bytes
            ),
            max_id_bytes=payload.get("max_id_bytes", defaults.max_id_bytes),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id"),
            artifact_name="handoff bounds",
        )
        return result


def _coerce_bounds(value: Any) -> HandoffBounds:
    if value is None:
        return HandoffBounds()
    if isinstance(value, HandoffBounds):
        return value
    if isinstance(value, Mapping):
        return HandoffBounds.from_dict(value)
    raise HandoffContractError("bounds must be a HandoffBounds object")


@dataclass(frozen=True)
class HandoffProvenance(_HandoffCanonicalContract):
    """Provenance and trust classification for one imported or local record."""

    SCHEMA: ClassVar[str] = HANDOFF_PROVENANCE_SCHEMA
    INTERFACE: ClassVar[str] = HANDOFF_PROVENANCE_INTERFACE

    source_family: SourceFamily
    source_export_version: str
    adapter_id: str = ""
    captured_at_ms: int = 0
    origin_uri: str = ""
    trust_class: TrustClass = TrustClass.IMPORTED_UNVERIFIED
    exportable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_family",
            _enum(self.source_family, SourceFamily, "source_family"),
        )
        object.__setattr__(
            self,
            "source_export_version",
            _text(
                self.source_export_version,
                "source_export_version",
                max_bytes=DEFAULT_MAX_ID_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "adapter_id",
            _text(self.adapter_id, "adapter_id", required=False, max_bytes=DEFAULT_MAX_ID_BYTES),
        )
        object.__setattr__(
            self, "captured_at_ms", _nonnegative_int(self.captured_at_ms, "captured_at_ms")
        )
        object.__setattr__(
            self,
            "origin_uri",
            _text(self.origin_uri, "origin_uri", required=False, max_bytes=DEFAULT_MAX_TEXT_BYTES),
        )
        object.__setattr__(
            self, "trust_class", _enum(self.trust_class, TrustClass, "trust_class")
        )
        object.__setattr__(self, "exportable", _bool(self.exportable, "exportable"))
        if self.trust_class is TrustClass.REJECTED and self.exportable:
            raise HandoffContractError("rejected provenance cannot be marked exportable")

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "source_family": self.source_family.value,
                "source_export_version": self.source_export_version,
                "adapter_id": self.adapter_id,
                "captured_at_ms": self.captured_at_ms,
                "origin_uri": self.origin_uri,
                "trust_class": self.trust_class.value,
                "exportable": self.exportable,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HandoffProvenance":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="handoff provenance"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "source_family",
                    "source_export_version",
                    "adapter_id",
                    "captured_at_ms",
                    "origin_uri",
                    "trust_class",
                    "exportable",
                }
            ),
            artifact_name="handoff provenance",
        )
        result = cls(
            source_family=payload.get("source_family"),
            source_export_version=payload.get("source_export_version", ""),
            adapter_id=payload.get("adapter_id", ""),
            captured_at_ms=payload.get("captured_at_ms", 0),
            origin_uri=payload.get("origin_uri", ""),
            trust_class=payload.get("trust_class", TrustClass.IMPORTED_UNVERIFIED),
            exportable=payload.get("exportable", True),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id"),
            artifact_name="handoff provenance",
        )
        return result


def _coerce_provenance(value: Any) -> HandoffProvenance:
    if isinstance(value, HandoffProvenance):
        return value
    if isinstance(value, Mapping):
        return HandoffProvenance.from_dict(value)
    raise HandoffContractError("provenance must be a HandoffProvenance object")


@dataclass(frozen=True)
class EncryptedExportReference(_HandoffCanonicalContract):
    """Content-addressed pointer to encrypted raw export bytes.

    Public records never embed transcript bodies.  Exact exported bytes are
    stored behind a ciphertext identity and an optional key-envelope identity.
    """

    SCHEMA: ClassVar[str] = ENCRYPTED_EXPORT_REFERENCE_SCHEMA
    INTERFACE: ClassVar[str] = ENCRYPTED_EXPORT_REFERENCE_INTERFACE

    ciphertext_cid: str
    digest_sha256: str
    byte_count: int
    media_type: str = "application/octet-stream"
    key_envelope_cid: str = ""
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    disclosure_class: DisclosureClass = DisclosureClass.ENCRYPTED_RAW
    retention_class: RetentionClass = RetentionClass.SESSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ciphertext_cid", _content_ref(self.ciphertext_cid, "ciphertext_cid")
        )
        object.__setattr__(
            self, "digest_sha256", _digest_sha256(self.digest_sha256, "digest_sha256")
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        object.__setattr__(
            self,
            "media_type",
            _text(self.media_type, "media_type", max_bytes=DEFAULT_MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "key_envelope_cid",
            _content_ref(self.key_envelope_cid, "key_envelope_cid", required=False),
        )
        object.__setattr__(
            self,
            "encryption_algorithm",
            _enum(self.encryption_algorithm, EncryptionAlgorithm, "encryption_algorithm"),
        )
        object.__setattr__(
            self,
            "disclosure_class",
            _enum(self.disclosure_class, DisclosureClass, "disclosure_class"),
        )
        object.__setattr__(
            self,
            "retention_class",
            _enum(self.retention_class, RetentionClass, "retention_class"),
        )
        if self.disclosure_class is not DisclosureClass.ENCRYPTED_RAW:
            raise HandoffContractError(
                "encrypted export references must use encrypted_raw disclosure"
            )
        _distinct_identities(
            (
                ("ciphertext", self.ciphertext_cid),
                ("digest", self.digest_sha256),
                ("key_envelope", self.key_envelope_cid),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="encrypted export reference")
        _require_record_bound(self, artifact_name="encrypted export reference")

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "ciphertext_cid": self.ciphertext_cid,
                "digest_sha256": self.digest_sha256,
                "byte_count": self.byte_count,
                "media_type": self.media_type,
                "key_envelope_cid": self.key_envelope_cid,
                "encryption_algorithm": self.encryption_algorithm.value,
                "disclosure_class": self.disclosure_class.value,
                "retention_class": self.retention_class.value,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EncryptedExportReference":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="encrypted export reference",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "ciphertext_cid",
                    "digest_sha256",
                    "byte_count",
                    "media_type",
                    "key_envelope_cid",
                    "encryption_algorithm",
                    "disclosure_class",
                    "retention_class",
                }
            ),
            artifact_name="encrypted export reference",
        )
        result = cls(
            ciphertext_cid=payload.get("ciphertext_cid", ""),
            digest_sha256=payload.get("digest_sha256", ""),
            byte_count=payload.get("byte_count", 0),
            media_type=payload.get("media_type", "application/octet-stream"),
            key_envelope_cid=payload.get("key_envelope_cid", ""),
            encryption_algorithm=payload.get(
                "encryption_algorithm", EncryptionAlgorithm.AES_256_GCM
            ),
            disclosure_class=payload.get(
                "disclosure_class", DisclosureClass.ENCRYPTED_RAW
            ),
            retention_class=payload.get("retention_class", RetentionClass.SESSION),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id"),
            artifact_name="encrypted export reference",
        )
        return result


def _coerce_export_ref(value: Any) -> EncryptedExportReference:
    if isinstance(value, EncryptedExportReference):
        return value
    if isinstance(value, Mapping):
        return EncryptedExportReference.from_dict(value)
    raise HandoffContractError(
        "raw_export_ref must be an EncryptedExportReference object"
    )


def _event_common_init(
    record: Any,
    *,
    sequence: Any,
    provenance: Any,
    residual_fields: Any,
    bounds: Any,
    created_at_ms: Any,
) -> None:
    object.__setattr__(record, "sequence", _nonnegative_int(sequence, "sequence"))
    object.__setattr__(record, "provenance", _coerce_provenance(provenance))
    object.__setattr__(record, "bounds", _coerce_bounds(bounds))
    object.__setattr__(
        record, "created_at_ms", _nonnegative_int(created_at_ms, "created_at_ms")
    )
    object.__setattr__(
        record, "residual_fields", _residual_fields(residual_fields, bounds=record.bounds)
    )
    if record.sequence >= record.bounds.max_events:
        raise HandoffBoundsError("event sequence exceeds max_events")


def _event_identity_fields() -> frozenset[str]:
    return _COMMON_WIRE_FIELDS.union(
        {
            "kind",
            "sequence",
            "provenance",
            "residual_fields",
            "bounds",
            "created_at_ms",
            "event_id",
        }
    )


@dataclass(frozen=True)
class ConversationEvent(_HandoffCanonicalContract):
    """One exportable conversation turn.  Hidden chain-of-thought is rejected."""

    SCHEMA: ClassVar[str] = CONVERSATION_EVENT_SCHEMA
    INTERFACE: ClassVar[str] = CONVERSATION_EVENT_INTERFACE

    sequence: int
    role: ConversationRole
    provenance: HandoffProvenance
    text: str = ""
    reasoning_summary: str = ""
    residual_fields: Mapping[str, Any] = MappingProxyType({})
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        _event_common_init(
            self,
            sequence=self.sequence,
            provenance=self.provenance,
            residual_fields=self.residual_fields,
            bounds=self.bounds,
            created_at_ms=self.created_at_ms,
        )
        object.__setattr__(self, "role", _enum(self.role, ConversationRole, "role"))
        object.__setattr__(
            self,
            "text",
            _text(self.text, "text", required=False, max_bytes=self.bounds.max_text_bytes),
        )
        object.__setattr__(
            self,
            "reasoning_summary",
            _text(
                self.reasoning_summary,
                "reasoning_summary",
                required=False,
                max_bytes=self.bounds.max_text_bytes,
            ),
        )
        if not self.text and not self.reasoning_summary:
            raise HandoffContractError(
                "conversation event text or reasoning_summary is required"
            )
        _reject_forbidden_keys(self.to_dict(), name="conversation event")
        _require_record_bound(
            self, artifact_name="conversation event", bounds=self.bounds
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.CONVERSATION

    @property
    def event_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "kind": self.kind.value,
                "sequence": self.sequence,
                "role": self.role.value,
                "text": self.text,
                "reasoning_summary": self.reasoning_summary,
                "residual_fields": dict(self.residual_fields),
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConversationEvent":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="conversation event"
        )
        _reject_unknown(
            payload,
            _event_identity_fields().union({"role", "text", "reasoning_summary"}),
            artifact_name="conversation event",
        )
        kind = payload.get("kind")
        if kind not in (None, "", EventKind.CONVERSATION.value, EventKind.CONVERSATION):
            raise HandoffContractError("conversation event kind must be conversation")
        result = cls(
            sequence=payload.get("sequence", 0),
            role=payload.get("role"),
            text=payload.get("text", ""),
            reasoning_summary=payload.get("reasoning_summary", ""),
            residual_fields=payload.get("residual_fields", {}),
            provenance=payload.get("provenance"),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "event_id"),
            artifact_name="conversation event",
        )
        return result


@dataclass(frozen=True)
class ToolInvocationEvent(_HandoffCanonicalContract):
    """Recorded imported tool call.  Imported calls are never executed."""

    SCHEMA: ClassVar[str] = TOOL_INVOCATION_EVENT_SCHEMA
    INTERFACE: ClassVar[str] = TOOL_INVOCATION_EVENT_INTERFACE

    sequence: int
    tool_name: str
    provenance: HandoffProvenance
    arguments: Mapping[str, Any] = MappingProxyType({})
    residual_fields: Mapping[str, Any] = MappingProxyType({})
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0
    executed: bool = False

    def __post_init__(self) -> None:
        _event_common_init(
            self,
            sequence=self.sequence,
            provenance=self.provenance,
            residual_fields=self.residual_fields,
            bounds=self.bounds,
            created_at_ms=self.created_at_ms,
        )
        object.__setattr__(
            self,
            "tool_name",
            _text(self.tool_name, "tool_name", max_bytes=self.bounds.max_id_bytes),
        )
        if self.arguments is None:
            arguments: Any = {}
        else:
            arguments = self.arguments
        if not isinstance(arguments, Mapping):
            raise HandoffContractError("arguments must be an object")
        object.__setattr__(
            self,
            "arguments",
            _freeze_bounded(
                arguments,
                name="arguments",
                max_depth=self.bounds.max_depth,
                max_items=ABSOLUTE_MAX_ITEMS,
                max_text_bytes=self.bounds.max_text_bytes,
            ),
        )
        if self.executed:
            raise HandoffTrustError("imported tool invocations must not be executed")
        object.__setattr__(self, "executed", False)
        _reject_forbidden_keys(self.to_dict(), name="tool invocation event")
        _require_record_bound(
            self, artifact_name="tool invocation event", bounds=self.bounds
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_INVOCATION

    @property
    def event_id(self) -> str:
        return self.content_id

    @property
    def arguments_content_id(self) -> str:
        return content_identity(dict(self.arguments))

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "kind": self.kind.value,
                "sequence": self.sequence,
                "tool_name": self.tool_name,
                "arguments": dict(self.arguments),
                "executed": False,
                "residual_fields": dict(self.residual_fields),
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolInvocationEvent":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="tool invocation event"
        )
        _reject_unknown(
            payload,
            _event_identity_fields().union({"tool_name", "arguments", "executed"}),
            artifact_name="tool invocation event",
        )
        kind = payload.get("kind")
        if kind not in (
            None,
            "",
            EventKind.TOOL_INVOCATION.value,
            EventKind.TOOL_INVOCATION,
        ):
            raise HandoffContractError(
                "tool invocation event kind must be tool_invocation"
            )
        if payload.get("executed") not in (None, False):
            raise HandoffTrustError("imported tool invocations must not be executed")
        result = cls(
            sequence=payload.get("sequence", 0),
            tool_name=payload.get("tool_name", ""),
            arguments=payload.get("arguments", {}),
            residual_fields=payload.get("residual_fields", {}),
            provenance=payload.get("provenance"),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
            executed=False,
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "event_id"),
            artifact_name="tool invocation event",
        )
        return result


@dataclass(frozen=True)
class ToolResultEvent(_HandoffCanonicalContract):
    """Recorded imported tool result.  Success claims are never trusted."""

    SCHEMA: ClassVar[str] = TOOL_RESULT_EVENT_SCHEMA
    INTERFACE: ClassVar[str] = TOOL_RESULT_EVENT_INTERFACE

    sequence: int
    tool_name: str
    invocation_event_id: str
    result_content_id: str
    provenance: HandoffProvenance
    result_excerpt: str = ""
    claimed_success: bool = False
    residual_fields: Mapping[str, Any] = MappingProxyType({})
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0
    trusted_success: bool = False

    def __post_init__(self) -> None:
        _event_common_init(
            self,
            sequence=self.sequence,
            provenance=self.provenance,
            residual_fields=self.residual_fields,
            bounds=self.bounds,
            created_at_ms=self.created_at_ms,
        )
        object.__setattr__(
            self,
            "tool_name",
            _text(self.tool_name, "tool_name", max_bytes=self.bounds.max_id_bytes),
        )
        object.__setattr__(
            self,
            "invocation_event_id",
            _content_ref(self.invocation_event_id, "invocation_event_id"),
        )
        object.__setattr__(
            self,
            "result_content_id",
            _content_ref(self.result_content_id, "result_content_id"),
        )
        object.__setattr__(
            self,
            "result_excerpt",
            _text(
                self.result_excerpt,
                "result_excerpt",
                required=False,
                max_bytes=self.bounds.max_text_bytes,
            ),
        )
        object.__setattr__(
            self, "claimed_success", _bool(self.claimed_success, "claimed_success")
        )
        if self.trusted_success:
            raise HandoffTrustError("imported tool success claims are not trusted")
        object.__setattr__(self, "trusted_success", False)
        if self.result_content_id == self.invocation_event_id:
            raise HandoffIdentityError(
                "result_content_id must be distinct from invocation_event_id"
            )
        _reject_forbidden_keys(self.to_dict(), name="tool result event")
        _require_record_bound(
            self, artifact_name="tool result event", bounds=self.bounds
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_RESULT

    @property
    def event_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "kind": self.kind.value,
                "sequence": self.sequence,
                "tool_name": self.tool_name,
                "invocation_event_id": self.invocation_event_id,
                "result_content_id": self.result_content_id,
                "result_excerpt": self.result_excerpt,
                "claimed_success": self.claimed_success,
                "trusted_success": False,
                "residual_fields": dict(self.residual_fields),
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolResultEvent":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="tool result event"
        )
        _reject_unknown(
            payload,
            _event_identity_fields().union(
                {
                    "tool_name",
                    "invocation_event_id",
                    "result_content_id",
                    "result_excerpt",
                    "claimed_success",
                    "trusted_success",
                }
            ),
            artifact_name="tool result event",
        )
        kind = payload.get("kind")
        if kind not in (None, "", EventKind.TOOL_RESULT.value, EventKind.TOOL_RESULT):
            raise HandoffContractError("tool result event kind must be tool_result")
        if payload.get("trusted_success") not in (None, False):
            raise HandoffTrustError("imported tool success claims are not trusted")
        result = cls(
            sequence=payload.get("sequence", 0),
            tool_name=payload.get("tool_name", ""),
            invocation_event_id=payload.get("invocation_event_id", ""),
            result_content_id=payload.get("result_content_id", ""),
            result_excerpt=payload.get("result_excerpt", ""),
            claimed_success=payload.get("claimed_success", False),
            residual_fields=payload.get("residual_fields", {}),
            provenance=payload.get("provenance"),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
            trusted_success=False,
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "event_id"),
            artifact_name="tool result event",
        )
        return result


@dataclass(frozen=True)
class PatchEvent(_HandoffCanonicalContract):
    """Content-addressed patch nomination.  Claimed application is untrusted."""

    SCHEMA: ClassVar[str] = PATCH_EVENT_SCHEMA
    INTERFACE: ClassVar[str] = PATCH_EVENT_INTERFACE

    sequence: int
    patch_kind: PatchKind
    patch_content_id: str
    provenance: HandoffProvenance
    paths: tuple[str, ...] = ()
    claimed_applied: bool = False
    residual_fields: Mapping[str, Any] = MappingProxyType({})
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0
    applied: bool = False

    def __post_init__(self) -> None:
        _event_common_init(
            self,
            sequence=self.sequence,
            provenance=self.provenance,
            residual_fields=self.residual_fields,
            bounds=self.bounds,
            created_at_ms=self.created_at_ms,
        )
        object.__setattr__(
            self, "patch_kind", _enum(self.patch_kind, PatchKind, "patch_kind")
        )
        object.__setattr__(
            self, "patch_content_id", _content_ref(self.patch_content_id, "patch_content_id")
        )
        object.__setattr__(
            self, "paths", _relative_paths(self.paths, "paths", max_items=self.bounds.max_paths)
        )
        object.__setattr__(
            self, "claimed_applied", _bool(self.claimed_applied, "claimed_applied")
        )
        if self.applied:
            raise HandoffTrustError("imported patches are not marked applied")
        object.__setattr__(self, "applied", False)
        _reject_forbidden_keys(self.to_dict(), name="patch event")
        _require_record_bound(self, artifact_name="patch event", bounds=self.bounds)

    @property
    def kind(self) -> EventKind:
        return EventKind.PATCH

    @property
    def event_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "kind": self.kind.value,
                "sequence": self.sequence,
                "patch_kind": self.patch_kind.value,
                "patch_content_id": self.patch_content_id,
                "paths": list(self.paths),
                "claimed_applied": self.claimed_applied,
                "applied": False,
                "residual_fields": dict(self.residual_fields),
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchEvent":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="patch event"
        )
        _reject_unknown(
            payload,
            _event_identity_fields().union(
                {
                    "patch_kind",
                    "patch_content_id",
                    "paths",
                    "claimed_applied",
                    "applied",
                }
            ),
            artifact_name="patch event",
        )
        kind = payload.get("kind")
        if kind not in (None, "", EventKind.PATCH.value, EventKind.PATCH):
            raise HandoffContractError("patch event kind must be patch")
        if payload.get("applied") not in (None, False):
            raise HandoffTrustError("imported patches are not marked applied")
        result = cls(
            sequence=payload.get("sequence", 0),
            patch_kind=payload.get("patch_kind"),
            patch_content_id=payload.get("patch_content_id", ""),
            paths=payload.get("paths", ()),
            claimed_applied=payload.get("claimed_applied", False),
            residual_fields=payload.get("residual_fields", {}),
            provenance=payload.get("provenance"),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
            applied=False,
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "event_id"),
            artifact_name="patch event",
        )
        return result


@dataclass(frozen=True)
class ApprovalEvent(_HandoffCanonicalContract):
    """Recorded approval.  Imported claims never grant effects."""

    SCHEMA: ClassVar[str] = APPROVAL_EVENT_SCHEMA
    INTERFACE: ClassVar[str] = APPROVAL_EVENT_INTERFACE

    sequence: int
    approval_kind: ApprovalKind
    decision: ApprovalDecision
    subject_content_id: str
    provenance: HandoffProvenance
    authority_binding_id: str = ""
    residual_fields: Mapping[str, Any] = MappingProxyType({})
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0
    grants_effects: bool = False

    def __post_init__(self) -> None:
        _event_common_init(
            self,
            sequence=self.sequence,
            provenance=self.provenance,
            residual_fields=self.residual_fields,
            bounds=self.bounds,
            created_at_ms=self.created_at_ms,
        )
        object.__setattr__(
            self,
            "approval_kind",
            _enum(self.approval_kind, ApprovalKind, "approval_kind"),
        )
        object.__setattr__(
            self, "decision", _enum(self.decision, ApprovalDecision, "decision")
        )
        object.__setattr__(
            self,
            "subject_content_id",
            _content_ref(self.subject_content_id, "subject_content_id"),
        )
        object.__setattr__(
            self,
            "authority_binding_id",
            _text(
                self.authority_binding_id,
                "authority_binding_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        imported = (
            self.approval_kind is ApprovalKind.IMPORTED_CLAIM
            or self.provenance.trust_class.imported
        )
        if imported and self.grants_effects:
            raise HandoffTrustError("imported approval claims cannot grant effects")
        if imported:
            object.__setattr__(self, "grants_effects", False)
        else:
            object.__setattr__(
                self, "grants_effects", _bool(self.grants_effects, "grants_effects")
            )
        if (
            self.approval_kind is not ApprovalKind.IMPORTED_CLAIM
            and not imported
            and self.grants_effects
            and not self.authority_binding_id
        ):
            raise HandoffTrustError(
                "effect-granting approvals require an authority binding identity"
            )
        _reject_forbidden_keys(self.to_dict(), name="approval event")
        _require_record_bound(self, artifact_name="approval event", bounds=self.bounds)

    @property
    def kind(self) -> EventKind:
        return EventKind.APPROVAL

    @property
    def event_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "kind": self.kind.value,
                "sequence": self.sequence,
                "approval_kind": self.approval_kind.value,
                "decision": self.decision.value,
                "subject_content_id": self.subject_content_id,
                "authority_binding_id": self.authority_binding_id,
                "grants_effects": self.grants_effects,
                "residual_fields": dict(self.residual_fields),
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ApprovalEvent":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="approval event"
        )
        _reject_unknown(
            payload,
            _event_identity_fields().union(
                {
                    "approval_kind",
                    "decision",
                    "subject_content_id",
                    "authority_binding_id",
                    "grants_effects",
                }
            ),
            artifact_name="approval event",
        )
        kind = payload.get("kind")
        if kind not in (None, "", EventKind.APPROVAL.value, EventKind.APPROVAL):
            raise HandoffContractError("approval event kind must be approval")
        result = cls(
            sequence=payload.get("sequence", 0),
            approval_kind=payload.get("approval_kind"),
            decision=payload.get("decision"),
            subject_content_id=payload.get("subject_content_id", ""),
            authority_binding_id=payload.get("authority_binding_id", ""),
            residual_fields=payload.get("residual_fields", {}),
            provenance=payload.get("provenance"),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
            grants_effects=payload.get("grants_effects", False),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "event_id"),
            artifact_name="approval event",
        )
        return result


HandoffEvent: TypeAlias = (
    ConversationEvent
    | ToolInvocationEvent
    | ToolResultEvent
    | PatchEvent
    | ApprovalEvent
)

_EVENT_DECODERS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        CONVERSATION_EVENT_SCHEMA: ConversationEvent.from_dict,
        CONVERSATION_EVENT_INTERFACE: ConversationEvent.from_dict,
        EventKind.CONVERSATION.value: ConversationEvent.from_dict,
        TOOL_INVOCATION_EVENT_SCHEMA: ToolInvocationEvent.from_dict,
        TOOL_INVOCATION_EVENT_INTERFACE: ToolInvocationEvent.from_dict,
        EventKind.TOOL_INVOCATION.value: ToolInvocationEvent.from_dict,
        TOOL_RESULT_EVENT_SCHEMA: ToolResultEvent.from_dict,
        TOOL_RESULT_EVENT_INTERFACE: ToolResultEvent.from_dict,
        EventKind.TOOL_RESULT.value: ToolResultEvent.from_dict,
        PATCH_EVENT_SCHEMA: PatchEvent.from_dict,
        PATCH_EVENT_INTERFACE: PatchEvent.from_dict,
        EventKind.PATCH.value: PatchEvent.from_dict,
        APPROVAL_EVENT_SCHEMA: ApprovalEvent.from_dict,
        APPROVAL_EVENT_INTERFACE: ApprovalEvent.from_dict,
        EventKind.APPROVAL.value: ApprovalEvent.from_dict,
    }
)


def decode_handoff_event(payload: Mapping[str, Any] | HandoffEvent) -> HandoffEvent:
    """Decode one strictly versioned normalized handoff event."""

    if isinstance(
        payload,
        (
            ConversationEvent,
            ToolInvocationEvent,
            ToolResultEvent,
            PatchEvent,
            ApprovalEvent,
        ),
    ):
        return payload
    if not isinstance(payload, Mapping):
        raise HandoffContractError("handoff event payload must be an object")
    for key in (payload.get("schema"), payload.get("interface"), payload.get("kind")):
        decoder = _EVENT_DECODERS.get(str(key) if key is not None else "")
        if decoder is not None:
            return decoder(payload)
    raise HandoffVersionError("unsupported handoff event schema")


def validate_event_sequence(events: Sequence[HandoffEvent]) -> tuple[str, ...]:
    """Require strictly increasing sequences and return ordered event identities."""

    if len(events) > ABSOLUTE_MAX_EVENTS:
        raise HandoffBoundsError("event sequence exceeds the absolute event limit")
    identities: list[str] = []
    seen: set[str] = set()
    previous = -1
    for event in events:
        if event.sequence <= previous:
            raise HandoffContractError("event sequence must be strictly increasing")
        previous = event.sequence
        identity = event.event_id
        if identity in seen:
            raise HandoffIdentityError("event sequence must not contain duplicate identities")
        seen.add(identity)
        identities.append(identity)
    return tuple(identities)


@dataclass(frozen=True)
class AgentCheckpoint(_HandoffCanonicalContract):
    """Restart-safe checkpoint over a prefix of the normalized stream."""

    SCHEMA: ClassVar[str] = AGENT_CHECKPOINT_SCHEMA
    INTERFACE: ClassVar[str] = AGENT_CHECKPOINT_INTERFACE

    session_id: str
    sequence: int
    event_content_ids: tuple[str, ...]
    normalized_stream_id: str
    provenance: HandoffProvenance
    repository_id: str = ""
    tree_id: str = ""
    restart_safe: bool = True
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self,
            "session_id",
            _content_ref(self.session_id, "session_id"),
        )
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "event_content_ids",
            _ids(
                self.event_content_ids,
                "event_content_ids",
                max_items=self.bounds.max_events,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(self, "provenance", _coerce_provenance(self.provenance))
        object.__setattr__(
            self,
            "repository_id",
            _text(
                self.repository_id,
                "repository_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "tree_id",
            _text(self.tree_id, "tree_id", required=False, max_bytes=self.bounds.max_id_bytes),
        )
        object.__setattr__(
            self, "restart_safe", _bool(self.restart_safe, "restart_safe")
        )
        if not self.restart_safe:
            raise HandoffContractError("agent checkpoints must be restart-safe")
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        if self.sequence >= self.bounds.max_checkpoints:
            raise HandoffBoundsError("checkpoint sequence exceeds max_checkpoints")
        expected_stream = normalized_stream_identity(self.event_content_ids)
        supplied = _text(
            self.normalized_stream_id,
            "normalized_stream_id",
            required=False,
            max_bytes=ABSOLUTE_MAX_ID_BYTES,
        )
        if supplied and supplied != expected_stream:
            raise HandoffIdentityError(
                "checkpoint normalized_stream_id does not match event prefix"
            )
        object.__setattr__(self, "normalized_stream_id", expected_stream)
        _distinct_identities(
            (
                ("session", self.session_id),
                ("normalized_stream", self.normalized_stream_id),
                ("repository", self.repository_id),
                ("tree", self.tree_id),
            )
        )
        _require_record_bound(
            self, artifact_name="agent checkpoint", bounds=self.bounds
        )

    @property
    def checkpoint_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "session_id": self.session_id,
                "sequence": self.sequence,
                "event_content_ids": list(self.event_content_ids),
                "normalized_stream_id": self.normalized_stream_id,
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "restart_safe": True,
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentCheckpoint":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="agent checkpoint"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "session_id",
                    "sequence",
                    "event_content_ids",
                    "normalized_stream_id",
                    "repository_id",
                    "tree_id",
                    "restart_safe",
                    "provenance",
                    "bounds",
                    "created_at_ms",
                    "checkpoint_id",
                }
            ),
            artifact_name="agent checkpoint",
        )
        if payload.get("restart_safe") not in (None, True):
            raise HandoffContractError("agent checkpoints must be restart-safe")
        result = cls(
            session_id=payload.get("session_id", ""),
            sequence=payload.get("sequence", 0),
            event_content_ids=payload.get("event_content_ids", ()),
            normalized_stream_id=payload.get("normalized_stream_id", ""),
            provenance=payload.get("provenance"),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            restart_safe=True,
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "checkpoint_id"),
            artifact_name="agent checkpoint",
        )
        return result


@dataclass(frozen=True)
class AgentContextArtifact(_HandoffCanonicalContract):
    """Content-addressed context artifact.  Bodies stay out of the record."""

    SCHEMA: ClassVar[str] = AGENT_CONTEXT_ARTIFACT_SCHEMA
    INTERFACE: ClassVar[str] = AGENT_CONTEXT_ARTIFACT_INTERFACE

    kind: ContextArtifactKind
    artifact_content_id: str
    provenance: HandoffProvenance
    media_type: str = "application/json"
    byte_count: int = 0
    summary: str = ""
    disclosure_class: DisclosureClass = DisclosureClass.PUBLIC_PROJECTION
    retention_class: RetentionClass = RetentionClass.SESSION
    bounds: HandoffBounds = HandoffBounds()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "kind", _enum(self.kind, ContextArtifactKind, "kind")
        )
        object.__setattr__(
            self,
            "artifact_content_id",
            _content_ref(self.artifact_content_id, "artifact_content_id"),
        )
        object.__setattr__(self, "provenance", _coerce_provenance(self.provenance))
        object.__setattr__(
            self,
            "media_type",
            _text(self.media_type, "media_type", max_bytes=self.bounds.max_id_bytes),
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                "summary",
                required=False,
                max_bytes=self.bounds.max_text_bytes,
            ),
        )
        object.__setattr__(
            self,
            "disclosure_class",
            _enum(self.disclosure_class, DisclosureClass, "disclosure_class"),
        )
        object.__setattr__(
            self,
            "retention_class",
            _enum(self.retention_class, RetentionClass, "retention_class"),
        )
        if self.disclosure_class is DisclosureClass.ENCRYPTED_RAW:
            raise HandoffContractError(
                "context artifacts must not embed encrypted raw export bodies"
            )
        _reject_forbidden_keys(self.to_dict(), name="agent context artifact")
        _require_record_bound(
            self, artifact_name="agent context artifact", bounds=self.bounds
        )

    @property
    def artifact_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "kind": self.kind.value,
                "artifact_content_id": self.artifact_content_id,
                "media_type": self.media_type,
                "byte_count": self.byte_count,
                "summary": self.summary,
                "disclosure_class": self.disclosure_class.value,
                "retention_class": self.retention_class.value,
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentContextArtifact":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="agent context artifact"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "kind",
                    "artifact_content_id",
                    "media_type",
                    "byte_count",
                    "summary",
                    "disclosure_class",
                    "retention_class",
                    "provenance",
                    "bounds",
                    "artifact_id",
                }
            ),
            artifact_name="agent context artifact",
        )
        result = cls(
            kind=payload.get("kind"),
            artifact_content_id=payload.get("artifact_content_id", ""),
            provenance=payload.get("provenance"),
            media_type=payload.get("media_type", "application/json"),
            byte_count=payload.get("byte_count", 0),
            summary=payload.get("summary", ""),
            disclosure_class=payload.get(
                "disclosure_class", DisclosureClass.PUBLIC_PROJECTION
            ),
            retention_class=payload.get("retention_class", RetentionClass.SESSION),
            bounds=payload.get("bounds"),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "artifact_id"),
            artifact_name="agent context artifact",
        )
        return result


@dataclass(frozen=True)
class ExternalAgentSession(_HandoffCanonicalContract):
    """Content-addressed session binding distinct raw and normalized identities."""

    SCHEMA: ClassVar[str] = EXTERNAL_AGENT_SESSION_SCHEMA
    INTERFACE: ClassVar[str] = EXTERNAL_AGENT_SESSION_INTERFACE

    source_family: SourceFamily
    raw_export_id: str
    event_content_ids: tuple[str, ...]
    provenance: HandoffProvenance
    objective_id: str = ""
    context_id: str = ""
    repository_id: str = ""
    checkpoint_ids: tuple[str, ...] = ()
    context_artifact_ids: tuple[str, ...] = ()
    patch_ids: tuple[str, ...] = ()
    normalized_stream_id: str = ""
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "source_family", _enum(self.source_family, SourceFamily, "source_family")
        )
        object.__setattr__(
            self, "raw_export_id", _content_ref(self.raw_export_id, "raw_export_id")
        )
        object.__setattr__(
            self,
            "event_content_ids",
            _ids(
                self.event_content_ids,
                "event_content_ids",
                max_items=self.bounds.max_events,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(self, "provenance", _coerce_provenance(self.provenance))
        if self.provenance.source_family is not self.source_family:
            raise HandoffContractError("session source_family must match provenance")
        object.__setattr__(
            self,
            "objective_id",
            _text(
                self.objective_id,
                "objective_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "context_id",
            _text(
                self.context_id,
                "context_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "repository_id",
            _text(
                self.repository_id,
                "repository_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_ids",
            _ids(
                self.checkpoint_ids,
                "checkpoint_ids",
                max_items=self.bounds.max_checkpoints,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "context_artifact_ids",
            _ids(
                self.context_artifact_ids,
                "context_artifact_ids",
                max_items=self.bounds.max_context_artifacts,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "patch_ids",
            _ids(
                self.patch_ids,
                "patch_ids",
                max_items=self.bounds.max_paths,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        expected_stream = normalized_stream_identity(self.event_content_ids)
        supplied = _text(
            self.normalized_stream_id,
            "normalized_stream_id",
            required=False,
            max_bytes=ABSOLUTE_MAX_ID_BYTES,
        )
        if supplied and supplied != expected_stream:
            raise HandoffIdentityError(
                "session normalized_stream_id does not match event_content_ids"
            )
        object.__setattr__(self, "normalized_stream_id", expected_stream)
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _distinct_identities(
            (
                ("raw_export", self.raw_export_id),
                ("normalized_stream", self.normalized_stream_id),
                ("objective", self.objective_id),
                ("context", self.context_id),
                ("repository", self.repository_id),
            )
        )
        _require_record_bound(
            self,
            artifact_name="external agent session",
            bounds=self.bounds,
            serialized=True,
        )

    @property
    def session_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "source_family": self.source_family.value,
                "raw_export_id": self.raw_export_id,
                "normalized_stream_id": self.normalized_stream_id,
                "event_content_ids": list(self.event_content_ids),
                "objective_id": self.objective_id,
                "context_id": self.context_id,
                "repository_id": self.repository_id,
                "checkpoint_ids": list(self.checkpoint_ids),
                "context_artifact_ids": list(self.context_artifact_ids),
                "patch_ids": list(self.patch_ids),
                "provenance": self.provenance.to_dict(),
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalAgentSession":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="external agent session"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "source_family",
                    "raw_export_id",
                    "normalized_stream_id",
                    "event_content_ids",
                    "objective_id",
                    "context_id",
                    "repository_id",
                    "checkpoint_ids",
                    "context_artifact_ids",
                    "patch_ids",
                    "provenance",
                    "bounds",
                    "created_at_ms",
                    "session_id",
                }
            ),
            artifact_name="external agent session",
        )
        result = cls(
            source_family=payload.get("source_family"),
            raw_export_id=payload.get("raw_export_id", ""),
            event_content_ids=payload.get("event_content_ids", ()),
            provenance=payload.get("provenance"),
            objective_id=payload.get("objective_id", ""),
            context_id=payload.get("context_id", ""),
            repository_id=payload.get("repository_id", ""),
            checkpoint_ids=payload.get("checkpoint_ids", ()),
            context_artifact_ids=payload.get("context_artifact_ids", ()),
            patch_ids=payload.get("patch_ids", ()),
            normalized_stream_id=payload.get("normalized_stream_id", ""),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "session_id"),
            artifact_name="external agent session",
        )
        return result


@dataclass(frozen=True)
class ExternalAgentHandoffRequest(_HandoffCanonicalContract):
    """Transport-neutral handoff request.  Provider selection is not imported."""

    SCHEMA: ClassVar[str] = EXTERNAL_AGENT_HANDOFF_REQUEST_SCHEMA
    INTERFACE: ClassVar[str] = EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE

    source_family: SourceFamily
    source_export_version: str
    raw_export_ref: EncryptedExportReference
    session_id: str
    caller_principal_id: str
    idempotency_key: str
    mode: HandoffMode = HandoffMode.PREVIEW
    disclosure_class: DisclosureClass = DisclosureClass.PUBLIC_PROJECTION
    retention_class: RetentionClass = RetentionClass.SESSION
    objective_id: str = ""
    context_id: str = ""
    repository_id: str = ""
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "source_family", _enum(self.source_family, SourceFamily, "source_family")
        )
        object.__setattr__(
            self,
            "source_export_version",
            _text(
                self.source_export_version,
                "source_export_version",
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(self, "raw_export_ref", _coerce_export_ref(self.raw_export_ref))
        object.__setattr__(
            self, "session_id", _content_ref(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "caller_principal_id",
            _text(
                self.caller_principal_id,
                "caller_principal_id",
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _text(
                self.idempotency_key,
                "idempotency_key",
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(self, "mode", _enum(self.mode, HandoffMode, "mode"))
        object.__setattr__(
            self,
            "disclosure_class",
            _enum(self.disclosure_class, DisclosureClass, "disclosure_class"),
        )
        object.__setattr__(
            self,
            "retention_class",
            _enum(self.retention_class, RetentionClass, "retention_class"),
        )
        object.__setattr__(
            self,
            "objective_id",
            _text(
                self.objective_id,
                "objective_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "context_id",
            _text(
                self.context_id,
                "context_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "repository_id",
            _text(
                self.repository_id,
                "repository_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        if self.disclosure_class is DisclosureClass.ENCRYPTED_RAW:
            raise HandoffContractError(
                "handoff requests must not place encrypted raw exports in the public projection"
            )
        _distinct_identities(
            (
                ("raw_export", self.raw_export_ref.content_id),
                ("ciphertext", self.raw_export_ref.ciphertext_cid),
                ("session", self.session_id),
                ("objective", self.objective_id),
                ("context", self.context_id),
                ("repository", self.repository_id),
                ("caller", self.caller_principal_id),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="external agent handoff request")
        _require_record_bound(
            self,
            artifact_name="external agent handoff request",
            bounds=self.bounds,
            serialized=True,
        )

    @property
    def request_id(self) -> str:
        return self.content_id

    @property
    def raw_export_id(self) -> str:
        """Content identity of the encrypted export reference, not the ciphertext."""

        return self.raw_export_ref.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "source_family": self.source_family.value,
                "source_export_version": self.source_export_version,
                "raw_export_ref": self.raw_export_ref.to_dict(),
                "session_id": self.session_id,
                "caller_principal_id": self.caller_principal_id,
                "idempotency_key": self.idempotency_key,
                "mode": self.mode.value,
                "disclosure_class": self.disclosure_class.value,
                "retention_class": self.retention_class.value,
                "objective_id": self.objective_id,
                "context_id": self.context_id,
                "repository_id": self.repository_id,
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalAgentHandoffRequest":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="external agent handoff request",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "source_family",
                    "source_export_version",
                    "raw_export_ref",
                    "session_id",
                    "caller_principal_id",
                    "idempotency_key",
                    "mode",
                    "disclosure_class",
                    "retention_class",
                    "objective_id",
                    "context_id",
                    "repository_id",
                    "bounds",
                    "created_at_ms",
                    "request_id",
                    "provider_id",
                    "provider_route",
                }
            ),
            artifact_name="external agent handoff request",
        )
        if payload.get("provider_id") not in (None, "") or payload.get(
            "provider_route"
        ) not in (None, ""):
            raise HandoffTrustError(
                "imported history cannot select a provider on a handoff request"
            )
        result = cls(
            source_family=payload.get("source_family"),
            source_export_version=payload.get("source_export_version", ""),
            raw_export_ref=payload.get("raw_export_ref"),
            session_id=payload.get("session_id", ""),
            caller_principal_id=payload.get("caller_principal_id", ""),
            idempotency_key=payload.get("idempotency_key", ""),
            mode=payload.get("mode", HandoffMode.PREVIEW),
            disclosure_class=payload.get(
                "disclosure_class", DisclosureClass.PUBLIC_PROJECTION
            ),
            retention_class=payload.get("retention_class", RetentionClass.SESSION),
            objective_id=payload.get("objective_id", ""),
            context_id=payload.get("context_id", ""),
            repository_id=payload.get("repository_id", ""),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "request_id"),
            artifact_name="external agent handoff request",
        )
        return result


@dataclass(frozen=True)
class HandoffNormalizationReport(_HandoffCanonicalContract):
    """Bounded normalization receipt.  Public form has no transcript bodies."""

    SCHEMA: ClassVar[str] = HANDOFF_NORMALIZATION_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = HANDOFF_NORMALIZATION_REPORT_INTERFACE

    request_id: str
    session_id: str
    source_family: SourceFamily
    raw_export_id: str
    accepted_event_ids: tuple[str, ...]
    rejected_event_count: int = 0
    truncated: bool = False
    unknown_fields_retained: int = 0
    hidden_chain_of_thought_rejected: int = 0
    imported_success_claims_untrusted: int = 0
    imported_invocations_not_executed: bool = True
    normalized_stream_id: str = ""
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "request_id", _content_ref(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "session_id", _content_ref(self.session_id, "session_id")
        )
        object.__setattr__(
            self, "source_family", _enum(self.source_family, SourceFamily, "source_family")
        )
        object.__setattr__(
            self, "raw_export_id", _content_ref(self.raw_export_id, "raw_export_id")
        )
        object.__setattr__(
            self,
            "accepted_event_ids",
            _ids(
                self.accepted_event_ids,
                "accepted_event_ids",
                max_items=self.bounds.max_events,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "rejected_event_count",
            _nonnegative_int(self.rejected_event_count, "rejected_event_count"),
        )
        object.__setattr__(self, "truncated", _bool(self.truncated, "truncated"))
        object.__setattr__(
            self,
            "unknown_fields_retained",
            _nonnegative_int(self.unknown_fields_retained, "unknown_fields_retained"),
        )
        if self.unknown_fields_retained > self.bounds.max_unknown_fields * max(
            1, len(self.accepted_event_ids)
        ):
            raise HandoffBoundsError("unknown_fields_retained exceeds the bound")
        object.__setattr__(
            self,
            "hidden_chain_of_thought_rejected",
            _nonnegative_int(
                self.hidden_chain_of_thought_rejected,
                "hidden_chain_of_thought_rejected",
            ),
        )
        object.__setattr__(
            self,
            "imported_success_claims_untrusted",
            _nonnegative_int(
                self.imported_success_claims_untrusted,
                "imported_success_claims_untrusted",
            ),
        )
        if not self.imported_invocations_not_executed:
            raise HandoffTrustError("normalization must not execute imported calls")
        object.__setattr__(self, "imported_invocations_not_executed", True)
        expected_stream = normalized_stream_identity(self.accepted_event_ids)
        supplied = _text(
            self.normalized_stream_id,
            "normalized_stream_id",
            required=False,
            max_bytes=ABSOLUTE_MAX_ID_BYTES,
        )
        if supplied and supplied != expected_stream:
            raise HandoffIdentityError(
                "normalization normalized_stream_id does not match accepted_event_ids"
            )
        object.__setattr__(self, "normalized_stream_id", expected_stream)
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _distinct_identities(
            (
                ("request", self.request_id),
                ("session", self.session_id),
                ("raw_export", self.raw_export_id),
                ("normalized_stream", self.normalized_stream_id),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="handoff normalization report")
        _require_record_bound(
            self,
            artifact_name="handoff normalization report",
            bounds=self.bounds,
            serialized=True,
        )

    @property
    def report_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "request_id": self.request_id,
                "session_id": self.session_id,
                "source_family": self.source_family.value,
                "raw_export_id": self.raw_export_id,
                "normalized_stream_id": self.normalized_stream_id,
                "accepted_event_ids": list(self.accepted_event_ids),
                "rejected_event_count": self.rejected_event_count,
                "truncated": self.truncated,
                "unknown_fields_retained": self.unknown_fields_retained,
                "hidden_chain_of_thought_rejected": self.hidden_chain_of_thought_rejected,
                "imported_success_claims_untrusted": self.imported_success_claims_untrusted,
                "imported_invocations_not_executed": True,
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HandoffNormalizationReport":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="handoff normalization report",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "request_id",
                    "session_id",
                    "source_family",
                    "raw_export_id",
                    "normalized_stream_id",
                    "accepted_event_ids",
                    "rejected_event_count",
                    "truncated",
                    "unknown_fields_retained",
                    "hidden_chain_of_thought_rejected",
                    "imported_success_claims_untrusted",
                    "imported_invocations_not_executed",
                    "bounds",
                    "created_at_ms",
                    "report_id",
                    "transcript",
                    "transcript_body",
                }
            ),
            artifact_name="handoff normalization report",
        )
        if payload.get("transcript") not in (None, "") or payload.get(
            "transcript_body"
        ) not in (None, ""):
            raise HandoffContractError(
                "normalization reports must not embed transcript bodies"
            )
        if payload.get("imported_invocations_not_executed") not in (None, True):
            raise HandoffTrustError("normalization must not execute imported calls")
        result = cls(
            request_id=payload.get("request_id", ""),
            session_id=payload.get("session_id", ""),
            source_family=payload.get("source_family"),
            raw_export_id=payload.get("raw_export_id", ""),
            accepted_event_ids=payload.get("accepted_event_ids", ()),
            rejected_event_count=payload.get("rejected_event_count", 0),
            truncated=payload.get("truncated", False),
            unknown_fields_retained=payload.get("unknown_fields_retained", 0),
            hidden_chain_of_thought_rejected=payload.get(
                "hidden_chain_of_thought_rejected", 0
            ),
            imported_success_claims_untrusted=payload.get(
                "imported_success_claims_untrusted", 0
            ),
            imported_invocations_not_executed=True,
            normalized_stream_id=payload.get("normalized_stream_id", ""),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "report_id"),
            artifact_name="handoff normalization report",
        )
        return result


@dataclass(frozen=True)
class HandoffAdmissionReceipt(_HandoffCanonicalContract):
    """Admission receipt.  Only reverified or admitted trust may complete."""

    SCHEMA: ClassVar[str] = HANDOFF_ADMISSION_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = HANDOFF_ADMISSION_RECEIPT_INTERFACE

    request_id: str
    session_id: str
    verdict: AdmissionVerdict
    trust_class: TrustClass
    raw_export_id: str
    normalized_stream_id: str
    reason_code: str
    policy_id: str
    objective_id: str = ""
    context_id: str = ""
    repository_id: str = ""
    patch_ids: tuple[str, ...] = ()
    bounds: HandoffBounds = HandoffBounds()
    created_at_ms: int = 0
    completion_eligible: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _coerce_bounds(self.bounds))
        object.__setattr__(
            self, "request_id", _content_ref(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "session_id", _content_ref(self.session_id, "session_id")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, AdmissionVerdict, "verdict")
        )
        object.__setattr__(
            self, "trust_class", _enum(self.trust_class, TrustClass, "trust_class")
        )
        object.__setattr__(
            self, "raw_export_id", _content_ref(self.raw_export_id, "raw_export_id")
        )
        object.__setattr__(
            self,
            "normalized_stream_id",
            _content_ref(self.normalized_stream_id, "normalized_stream_id"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", max_bytes=ABSOLUTE_MAX_REASON_BYTES),
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id, "policy_id", max_bytes=self.bounds.max_id_bytes),
        )
        object.__setattr__(
            self,
            "objective_id",
            _text(
                self.objective_id,
                "objective_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "context_id",
            _text(
                self.context_id,
                "context_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "repository_id",
            _text(
                self.repository_id,
                "repository_id",
                required=False,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self,
            "patch_ids",
            _ids(
                self.patch_ids,
                "patch_ids",
                max_items=self.bounds.max_paths,
                max_bytes=self.bounds.max_id_bytes,
            ),
        )
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        eligible = (
            self.verdict is AdmissionVerdict.ADMITTED
            and self.trust_class.may_satisfy_completion
        )
        claimed = self.completion_eligible
        if claimed and not eligible:
            raise HandoffTrustError(
                "only locally reverified or independently admitted receipts may satisfy completion"
            )
        object.__setattr__(self, "completion_eligible", eligible)
        if self.verdict is AdmissionVerdict.ADMITTED and self.trust_class in {
            TrustClass.REJECTED,
            TrustClass.QUARANTINED,
        }:
            raise HandoffTrustError("rejected or quarantined material cannot be admitted")
        _distinct_identities(
            (
                ("request", self.request_id),
                ("session", self.session_id),
                ("raw_export", self.raw_export_id),
                ("normalized_stream", self.normalized_stream_id),
                ("objective", self.objective_id),
                ("context", self.context_id),
                ("repository", self.repository_id),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="handoff admission receipt")
        _require_record_bound(
            self,
            artifact_name="handoff admission receipt",
            bounds=self.bounds,
            serialized=True,
        )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "request_id": self.request_id,
                "session_id": self.session_id,
                "verdict": self.verdict.value,
                "trust_class": self.trust_class.value,
                "raw_export_id": self.raw_export_id,
                "normalized_stream_id": self.normalized_stream_id,
                "objective_id": self.objective_id,
                "context_id": self.context_id,
                "repository_id": self.repository_id,
                "patch_ids": list(self.patch_ids),
                "reason_code": self.reason_code,
                "policy_id": self.policy_id,
                "completion_eligible": self.completion_eligible,
                "bounds": self.bounds.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HandoffAdmissionReceipt":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="handoff admission receipt",
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "request_id",
                    "session_id",
                    "verdict",
                    "trust_class",
                    "raw_export_id",
                    "normalized_stream_id",
                    "objective_id",
                    "context_id",
                    "repository_id",
                    "patch_ids",
                    "reason_code",
                    "policy_id",
                    "completion_eligible",
                    "bounds",
                    "created_at_ms",
                    "receipt_id",
                }
            ),
            artifact_name="handoff admission receipt",
        )
        result = cls(
            request_id=payload.get("request_id", ""),
            session_id=payload.get("session_id", ""),
            verdict=payload.get("verdict"),
            trust_class=payload.get("trust_class"),
            raw_export_id=payload.get("raw_export_id", ""),
            normalized_stream_id=payload.get("normalized_stream_id", ""),
            reason_code=payload.get("reason_code", ""),
            policy_id=payload.get("policy_id", ""),
            objective_id=payload.get("objective_id", ""),
            context_id=payload.get("context_id", ""),
            repository_id=payload.get("repository_id", ""),
            patch_ids=payload.get("patch_ids", ()),
            bounds=payload.get("bounds"),
            created_at_ms=payload.get("created_at_ms", 0),
            completion_eligible=payload.get("completion_eligible", False),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "receipt_id"),
            artifact_name="handoff admission receipt",
        )
        return result


_RECORD_DECODERS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        HANDOFF_BOUNDS_SCHEMA: HandoffBounds.from_dict,
        HANDOFF_BOUNDS_INTERFACE: HandoffBounds.from_dict,
        HANDOFF_PROVENANCE_SCHEMA: HandoffProvenance.from_dict,
        HANDOFF_PROVENANCE_INTERFACE: HandoffProvenance.from_dict,
        ENCRYPTED_EXPORT_REFERENCE_SCHEMA: EncryptedExportReference.from_dict,
        ENCRYPTED_EXPORT_REFERENCE_INTERFACE: EncryptedExportReference.from_dict,
        CONVERSATION_EVENT_SCHEMA: ConversationEvent.from_dict,
        CONVERSATION_EVENT_INTERFACE: ConversationEvent.from_dict,
        TOOL_INVOCATION_EVENT_SCHEMA: ToolInvocationEvent.from_dict,
        TOOL_INVOCATION_EVENT_INTERFACE: ToolInvocationEvent.from_dict,
        TOOL_RESULT_EVENT_SCHEMA: ToolResultEvent.from_dict,
        TOOL_RESULT_EVENT_INTERFACE: ToolResultEvent.from_dict,
        PATCH_EVENT_SCHEMA: PatchEvent.from_dict,
        PATCH_EVENT_INTERFACE: PatchEvent.from_dict,
        APPROVAL_EVENT_SCHEMA: ApprovalEvent.from_dict,
        APPROVAL_EVENT_INTERFACE: ApprovalEvent.from_dict,
        AGENT_CHECKPOINT_SCHEMA: AgentCheckpoint.from_dict,
        AGENT_CHECKPOINT_INTERFACE: AgentCheckpoint.from_dict,
        AGENT_CONTEXT_ARTIFACT_SCHEMA: AgentContextArtifact.from_dict,
        AGENT_CONTEXT_ARTIFACT_INTERFACE: AgentContextArtifact.from_dict,
        EXTERNAL_AGENT_SESSION_SCHEMA: ExternalAgentSession.from_dict,
        EXTERNAL_AGENT_SESSION_INTERFACE: ExternalAgentSession.from_dict,
        EXTERNAL_AGENT_HANDOFF_REQUEST_SCHEMA: ExternalAgentHandoffRequest.from_dict,
        EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE: ExternalAgentHandoffRequest.from_dict,
        HANDOFF_NORMALIZATION_REPORT_SCHEMA: HandoffNormalizationReport.from_dict,
        HANDOFF_NORMALIZATION_REPORT_INTERFACE: HandoffNormalizationReport.from_dict,
        HANDOFF_ADMISSION_RECEIPT_SCHEMA: HandoffAdmissionReceipt.from_dict,
        HANDOFF_ADMISSION_RECEIPT_INTERFACE: HandoffAdmissionReceipt.from_dict,
    }
)

HandoffRecord: TypeAlias = (
    HandoffBounds
    | HandoffProvenance
    | EncryptedExportReference
    | HandoffEvent
    | AgentCheckpoint
    | AgentContextArtifact
    | ExternalAgentSession
    | ExternalAgentHandoffRequest
    | HandoffNormalizationReport
    | HandoffAdmissionReceipt
)


def decode_handoff_contract(payload: Mapping[str, Any] | HandoffRecord) -> HandoffRecord:
    """Decode any strictly versioned handoff family record."""

    if isinstance(payload, CanonicalContract):
        return payload  # type: ignore[return-value]
    if not isinstance(payload, Mapping):
        raise HandoffContractError("handoff contract payload must be an object")
    for key in (payload.get("schema"), payload.get("interface")):
        decoder = _RECORD_DECODERS.get(str(key) if key is not None else "")
        if decoder is not None:
            return decoder(payload)
    raise HandoffVersionError("unsupported handoff contract schema")


def canonical_handoff_json_bytes(value: Any) -> bytes:
    """Encode one handoff value as canonical DAG-JSON UTF-8 bytes."""

    if isinstance(value, CanonicalContract):
        return value.canonical_bytes()
    return canonical_json_bytes(value)


__all__ = (
    "ABSOLUTE_MAX_EVENTS",
    "ABSOLUTE_MAX_RECORD_BYTES",
    "ABSOLUTE_MAX_TEXT_BYTES",
    "AGENT_CHECKPOINT_INTERFACE",
    "AGENT_CHECKPOINT_SCHEMA",
    "AGENT_CONTEXT_ARTIFACT_INTERFACE",
    "AGENT_CONTEXT_ARTIFACT_SCHEMA",
    "APPROVAL_EVENT_INTERFACE",
    "APPROVAL_EVENT_SCHEMA",
    "CONVERSATION_EVENT_INTERFACE",
    "CONVERSATION_EVENT_SCHEMA",
    "CONTRACT_VERSION",
    "ENCRYPTED_EXPORT_REFERENCE_INTERFACE",
    "ENCRYPTED_EXPORT_REFERENCE_SCHEMA",
    "EXTERNAL_AGENT_HANDOFF_REQUEST_INTERFACE",
    "EXTERNAL_AGENT_HANDOFF_REQUEST_SCHEMA",
    "EXTERNAL_AGENT_SESSION_INTERFACE",
    "EXTERNAL_AGENT_SESSION_SCHEMA",
    "HANDOFF_ADMISSION_RECEIPT_INTERFACE",
    "HANDOFF_ADMISSION_RECEIPT_SCHEMA",
    "HANDOFF_BOUNDS_INTERFACE",
    "HANDOFF_BOUNDS_SCHEMA",
    "HANDOFF_CONTRACT_FAMILY",
    "HANDOFF_CONTRACT_VERSION",
    "HANDOFF_NORMALIZATION_REPORT_INTERFACE",
    "HANDOFF_NORMALIZATION_REPORT_SCHEMA",
    "HANDOFF_PROVENANCE_INTERFACE",
    "HANDOFF_PROVENANCE_SCHEMA",
    "NORMALIZED_STREAM_SCHEMA",
    "PATCH_EVENT_INTERFACE",
    "PATCH_EVENT_SCHEMA",
    "SCHEMA_VERSION",
    "TOOL_INVOCATION_EVENT_INTERFACE",
    "TOOL_INVOCATION_EVENT_SCHEMA",
    "TOOL_RESULT_EVENT_INTERFACE",
    "TOOL_RESULT_EVENT_SCHEMA",
    "AdmissionVerdict",
    "AgentCheckpoint",
    "AgentContextArtifact",
    "ApprovalDecision",
    "ApprovalEvent",
    "ApprovalKind",
    "ContextArtifactKind",
    "ConversationEvent",
    "ConversationRole",
    "DisclosureClass",
    "EncryptedExportReference",
    "EncryptionAlgorithm",
    "EventKind",
    "ExternalAgentHandoffRequest",
    "ExternalAgentSession",
    "HandoffAdmissionReceipt",
    "HandoffBounds",
    "HandoffBoundsError",
    "HandoffContractError",
    "HandoffEvent",
    "HandoffIdentityError",
    "HandoffMode",
    "HandoffNormalizationReport",
    "HandoffProvenance",
    "HandoffRecord",
    "HandoffTrustError",
    "HandoffVersionError",
    "PatchEvent",
    "PatchKind",
    "RetentionClass",
    "SourceFamily",
    "ToolInvocationEvent",
    "ToolResultEvent",
    "TrustClass",
    "canonical_handoff_json_bytes",
    "decode_handoff_contract",
    "decode_handoff_event",
    "normalized_stream_identity",
    "validate_event_sequence",
)
