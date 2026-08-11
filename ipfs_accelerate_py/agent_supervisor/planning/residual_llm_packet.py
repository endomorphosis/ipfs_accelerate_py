"""Sealed residual LLM packet and bounds (WPD-002 / ResidualLlmPacket@1).

``ResidualLlmPacket@1`` is the single model-facing residual packet type admitted
for provider invocation under disposition ``residual_llm_authorized``.

Normative rules (aligned with :class:`CodexRepairPacket`; redaction is not
weakened):

* Exact write paths, obligation identifiers, a counterexample capsule, and
  validation commands are required and fail closed when missing.
* Secrets and unbounded source/AST/proof/witness dumps are rejected; only
  redacted public projections and content-addressed references may appear.
* Packet identity is content-addressed over the canonical sealed payload.
* Size and token bounds are enforced before the packet may be handed to a
  provider. Model output remains nomination-only (no write / semantic /
  completion authority).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

RESIDUAL_LLM_PACKET_INTERFACE: Final[str] = "ResidualLlmPacket@1"
RESIDUAL_LLM_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-llm-packet@1"
)
RESIDUAL_LLM_PACKET_VERSION: Final[int] = 1
RESIDUAL_LLM_PACKET_EVIDENCE: Final[str] = "wpd/residual-llm-packet@1"
RESIDUAL_LLM_PACKET_LIMITS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-llm-packet-limits@1"
)

PRODUCER_ID: Final[str] = "residual-llm-packet@1"

# Bounds intentionally no larger than CodexRepairPacket / ReplanLimits defaults
# (max_prompt_bytes, max_prompt_tokens, max_capsule_bytes).
DEFAULT_MAX_PACKET_BYTES: Final[int] = 24_576
DEFAULT_MAX_PACKET_TOKENS: Final[int] = 6_144
DEFAULT_MAX_CAPSULE_BYTES: Final[int] = 16_384
DEFAULT_MAX_WRITE_PATHS: Final[int] = 64
DEFAULT_MAX_OBLIGATIONS: Final[int] = 128
DEFAULT_MAX_VALIDATION_COMMANDS: Final[int] = 64
DEFAULT_MAX_ACCEPTANCE_IDS: Final[int] = 64
DEFAULT_MAX_TEXT_CHARS: Final[int] = 512
DEFAULT_MAX_PATH_CHARS: Final[int] = 1_024
DEFAULT_MAX_COMMAND_CHARS: Final[int] = 1_024
BYTES_PER_TOKEN: Final[int] = 3  # matches CodexRepairPacket estimate

REQUIRED_CORE_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "repository_id",
    "tree_id",
    "write_paths",
    "obligation_ids",
    "counterexample_capsule",
    "validation_commands",
)

RESIDUAL_LLM_PACKET_INVARIANTS: Final[tuple[str, ...]] = (
    "exact write paths are required and path-escape free",
    "obligation identifiers are required and non-empty",
    "counterexample capsule is required and body-free",
    "validation commands are required and non-empty",
    "secrets and unbounded source dumps are rejected",
    "identity is content-addressed over the sealed payload",
    "model output remains nomination-only",
)

# Keys that must never appear as embedded full bodies in a residual packet.
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "ast_nodes",
        "body",
        "code",
        "contents",
        "content",
        "file_content",
        "file_contents",
        "file_text",
        "full_ast",
        "full_graph",
        "full_proof",
        "full_receipt",
        "full_source",
        "full_trace",
        "gold_ir",
        "gold_ir_body",
        "graph",
        "graph_body",
        "hidden_witness",
        "kernel_proof_body",
        "lean_source",
        "private_inputs",
        "private_witness",
        "proof_body",
        "proof_text",
        "proof_transcript",
        "prover_output",
        "raw_ast",
        "raw_output",
        "receipt_body",
        "repository_body",
        "repository_dump",
        "secret",
        "secrets",
        "snippet",
        "solver_trace",
        "source",
        "source_body",
        "source_code",
        "source_excerpt",
        "source_text",
        "transcript",
        "witness",
    }
)

_SECRET_KEY_RE = re.compile(
    r"(?:^|[_\-.])(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|credential|authorization|cookie|"
    r"private[_-]?key|private[_-]?premise|private[_-]?input|"
    r"hidden[_-]?witness|private[_-]?witness)(?:$|[_\-.])",
    re.IGNORECASE,
)

_SECRET_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "api_key",
    "access_token",
    "private_key",
    "-----begin",
    "sk-",
    "password=",
    "authorization: bearer",
)

assert RESIDUAL_LLM_PACKET_INTERFACE == "ResidualLlmPacket@1"
assert RESIDUAL_LLM_PACKET_EVIDENCE == "wpd/residual-llm-packet@1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ResidualLlmPacketReason(str, Enum):
    """Stable, machine-readable residual packet rejection reasons."""

    MALFORMED = "malformed"
    MISSING_WRITE_PATHS = "missing_write_paths"
    MISSING_OBLIGATIONS = "missing_obligations"
    MISSING_VALIDATION_COMMANDS = "missing_validation_commands"
    MISSING_COUNTEREXAMPLE_CAPSULE = "missing_counterexample_capsule"
    PATH_NOT_EXACT = "path_not_exact"
    FORBIDDEN_BODY = "forbidden_body"
    SECRET_MATERIAL = "secret_material"
    OVER_BUDGET = "over_budget"
    AUTHORITY_CLAIM = "authority_claim"
    UNBOUNDED_SOURCE_DUMP = "unbounded_source_dump"


class ResidualLlmPacketError(ContractValidationError):
    """Raised when a residual LLM packet is malformed or unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ResidualLlmPacketReason | str | None = None,
    ) -> None:
        super().__init__(message)
        if isinstance(reason_code, ResidualLlmPacketReason):
            self.reason_code = reason_code.value
        elif reason_code is None:
            self.reason_code = ResidualLlmPacketReason.MALFORMED.value
        else:
            self.reason_code = str(reason_code)


class ResidualLlmPacketBudgetError(ResidualLlmPacketError):
    """Raised when the sealed packet exceeds configured size/token bounds."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ResidualLlmPacketError(
            f"{name} must be a string",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    else:
        text = value.strip()
    if required and not text:
        raise ResidualLlmPacketError(
            f"{name} is required",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    if len(text) > limit:
        raise ResidualLlmPacketError(
            f"{name} exceeds text bound",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return text


def _optional_text(
    value: Any,
    name: str,
    *,
    limit: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    return _text(value, name, required=False, limit=limit)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ResidualLlmPacketError(
            f"{name} must be a boolean",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return value


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResidualLlmPacketError(
            f"{name} must be an integer >= {minimum}",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return value


def _exact_path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=DEFAULT_MAX_PATH_CHARS).replace(
        "\\", "/"
    )
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or raw in {".", ""}
        or raw.startswith("./")
        or any(char in raw for char in "*?[]{}")
        or "//" in raw
        or raw.endswith("/")
    ):
        raise ResidualLlmPacketError(
            f"{name} must be an exact repository-relative path",
            reason_code=ResidualLlmPacketReason.PATH_NOT_EXACT,
        )
    if raw != candidate.as_posix():
        raise ResidualLlmPacketError(
            f"{name} must be a normalized repository-relative path",
            reason_code=ResidualLlmPacketReason.PATH_NOT_EXACT,
        )
    return raw


def _exact_paths(
    values: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = DEFAULT_MAX_WRITE_PATHS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualLlmPacketError(
            f"{name} must be a sequence of exact paths",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ResidualLlmPacketError(
            f"{name} must not be empty",
            reason_code=ResidualLlmPacketReason.MISSING_WRITE_PATHS,
        )
    if len(ordered) > limit:
        raise ResidualLlmPacketError(
            f"{name} exceeds path bound",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return tuple(ordered)


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = DEFAULT_MAX_OBLIGATIONS,
    empty_reason: ResidualLlmPacketReason = ResidualLlmPacketReason.MALFORMED,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualLlmPacketError(
            f"{name} must be a sequence of identifiers",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name, required=True, limit=DEFAULT_MAX_TEXT_CHARS)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if required and not ordered:
        raise ResidualLlmPacketError(
            f"{name} must not be empty",
            reason_code=empty_reason,
        )
    if len(ordered) > limit:
        raise ResidualLlmPacketError(
            f"{name} exceeds collection bound",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return tuple(ordered)


def _commands(
    values: Any,
    name: str = "validation_commands",
    *,
    required: bool = True,
    limit: int = DEFAULT_MAX_VALIDATION_COMMANDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualLlmPacketError(
            f"{name} must be a sequence of commands",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name, required=True, limit=DEFAULT_MAX_COMMAND_CHARS)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if required and not ordered:
        raise ResidualLlmPacketError(
            f"{name} must not be empty",
            reason_code=ResidualLlmPacketReason.MISSING_VALIDATION_COMMANDS,
        )
    if len(ordered) > limit:
        raise ResidualLlmPacketError(
            f"{name} exceeds command bound",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return tuple(ordered)


def _normalize_key(key: Any) -> str:
    return str(key).casefold().replace("-", "_")


def _walk_forbidden(value: Any, *, path: str = "") -> list[tuple[str, str]]:
    """Return (reason, path) pairs for forbidden body keys and secret markers."""

    findings: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            norm = _normalize_key(key_s)
            child = f"{path}.{key_s}" if path else key_s
            if norm in _FORBIDDEN_BODY_KEYS:
                findings.append(
                    (ResidualLlmPacketReason.FORBIDDEN_BODY.value, child)
                )
            if _SECRET_KEY_RE.search(norm):
                findings.append(
                    (ResidualLlmPacketReason.SECRET_MATERIAL.value, child)
                )
            findings.extend(_walk_forbidden(item, path=child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            findings.extend(_walk_forbidden(item, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(marker in lowered for marker in _SECRET_VALUE_MARKERS):
            findings.append(
                (
                    ResidualLlmPacketReason.SECRET_MATERIAL.value,
                    path or "text",
                )
            )
        # Unbounded dumps are rejected when large free-form bodies sneak in.
        if len(value) > DEFAULT_MAX_TEXT_CHARS * 8 and path:
            leaf = path.rsplit(".", 1)[-1]
            if _normalize_key(leaf) in _FORBIDDEN_BODY_KEYS | {
                "dump",
                "payload",
                "blob",
                "raw",
            }:
                findings.append(
                    (
                        ResidualLlmPacketReason.UNBOUNDED_SOURCE_DUMP.value,
                        path,
                    )
                )
    return findings


def _reject_forbidden_payload(payload: Any, *, where: str) -> None:
    findings = _walk_forbidden(payload)
    if not findings:
        return
    reason, location = findings[0]
    raise ResidualLlmPacketError(
        f"{where} contains forbidden material at {location}",
        reason_code=reason,
    )


def _capsule_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        raise ResidualLlmPacketError(
            "counterexample_capsule is required",
            reason_code=ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        )
    if hasattr(value, "to_dict") and callable(value.to_dict):
        mapped = value.to_dict()
    elif isinstance(value, Mapping):
        mapped = value
    else:
        raise ResidualLlmPacketError(
            "counterexample_capsule must be a mapping or capsule contract",
            reason_code=ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        )
    if not isinstance(mapped, Mapping) or not mapped:
        raise ResidualLlmPacketError(
            "counterexample_capsule must be a non-empty mapping",
            reason_code=ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        )
    _reject_forbidden_payload(mapped, where="counterexample_capsule")
    # Capsule must describe at least one counterexample or target binding.
    has_cex = bool(mapped.get("counterexamples")) or bool(
        mapped.get("counterexample_ids")
    )
    has_targets = bool(mapped.get("target_ids"))
    if not has_cex and not has_targets:
        raise ResidualLlmPacketError(
            "counterexample_capsule must bind counterexamples or target_ids",
            reason_code=ResidualLlmPacketReason.MISSING_COUNTEREXAMPLE_CAPSULE,
        )
    # Return a plain dict for stable canonicalization.
    return dict(mapped)


def _authority_roots(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResidualLlmPacketError(
            "authority_roots must be a mapping",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    roots: dict[str, str] = {}
    for key, item in value.items():
        key_s = _text(str(key), "authority_roots key", required=True)
        roots[key_s] = _text(item, f"authority_roots[{key_s}]", required=True)
    _reject_forbidden_payload(roots, where="authority_roots")
    return dict(sorted(roots.items()))


def estimate_tokens(byte_size: int) -> int:
    """Conservative deterministic token estimate (no tokenizer dependency)."""

    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
        raise ResidualLlmPacketError(
            "byte_size must be a non-negative integer",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return (byte_size + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualLlmPacketLimits:
    """Hard bounds for residual LLM packet sealing."""

    max_bytes: int = DEFAULT_MAX_PACKET_BYTES
    max_tokens: int = DEFAULT_MAX_PACKET_TOKENS
    max_capsule_bytes: int = DEFAULT_MAX_CAPSULE_BYTES
    max_write_paths: int = DEFAULT_MAX_WRITE_PATHS
    max_obligations: int = DEFAULT_MAX_OBLIGATIONS
    max_validation_commands: int = DEFAULT_MAX_VALIDATION_COMMANDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_bytes",
            _positive_int(self.max_bytes, "max_bytes", minimum=1024),
        )
        object.__setattr__(
            self,
            "max_tokens",
            _positive_int(self.max_tokens, "max_tokens", minimum=256),
        )
        object.__setattr__(
            self,
            "max_capsule_bytes",
            _positive_int(self.max_capsule_bytes, "max_capsule_bytes", minimum=1024),
        )
        object.__setattr__(
            self,
            "max_write_paths",
            _positive_int(self.max_write_paths, "max_write_paths", minimum=1),
        )
        object.__setattr__(
            self,
            "max_obligations",
            _positive_int(self.max_obligations, "max_obligations", minimum=1),
        )
        object.__setattr__(
            self,
            "max_validation_commands",
            _positive_int(
                self.max_validation_commands,
                "max_validation_commands",
                minimum=1,
            ),
        )
        if self.max_capsule_bytes > self.max_bytes:
            raise ResidualLlmPacketError(
                "max_capsule_bytes cannot exceed max_bytes",
                reason_code=ResidualLlmPacketReason.MALFORMED,
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "schema": RESIDUAL_LLM_PACKET_LIMITS_SCHEMA,
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
            "max_capsule_bytes": self.max_capsule_bytes,
            "max_write_paths": self.max_write_paths,
            "max_obligations": self.max_obligations,
            "max_validation_commands": self.max_validation_commands,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | None
    ) -> "ResidualLlmPacketLimits":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ResidualLlmPacketError(
                "limits must be a mapping",
                reason_code=ResidualLlmPacketReason.MALFORMED,
            )
        return cls(
            max_bytes=payload.get("max_bytes", DEFAULT_MAX_PACKET_BYTES),
            max_tokens=payload.get("max_tokens", DEFAULT_MAX_PACKET_TOKENS),
            max_capsule_bytes=payload.get(
                "max_capsule_bytes", DEFAULT_MAX_CAPSULE_BYTES
            ),
            max_write_paths=payload.get("max_write_paths", DEFAULT_MAX_WRITE_PATHS),
            max_obligations=payload.get("max_obligations", DEFAULT_MAX_OBLIGATIONS),
            max_validation_commands=payload.get(
                "max_validation_commands", DEFAULT_MAX_VALIDATION_COMMANDS
            ),
        )


# ---------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualLlmPacket(CanonicalContract):
    """Sealed residual packet admitted for residual_llm_authorized providers.

    Identity is content-addressed: :attr:`packet_id` equals the content
    identity of the canonical sealed payload (excluding the stored packet_id
    itself so identity remains self-consistent).
    """

    SCHEMA: ClassVar[str] = RESIDUAL_LLM_PACKET_SCHEMA

    task_id: str
    repository_id: str
    tree_id: str
    write_paths: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    counterexample_capsule: Mapping[str, Any]
    validation_commands: tuple[str, ...]
    policy_id: str = ""
    policy_revision: str = ""
    forest_id: str = ""
    acceptance_ids: tuple[str, ...] = ()
    authority_roots: Mapping[str, str] | None = None
    codex_packet_ref: str = ""
    transition_ref: str = ""
    max_bytes: int = DEFAULT_MAX_PACKET_BYTES
    max_tokens: int = DEFAULT_MAX_PACKET_TOKENS
    max_capsule_bytes: int = DEFAULT_MAX_CAPSULE_BYTES
    semantic_authority: bool = False
    write_authority: bool = False
    completion_authority: bool = False
    nomination_only: bool = True
    producer_id: str = PRODUCER_ID
    packet_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "policy_id", _optional_text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _optional_text(self.policy_revision, "policy_revision"),
        )
        object.__setattr__(
            self, "forest_id", _optional_text(self.forest_id, "forest_id")
        )
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(self.write_paths, "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(
                self.obligation_ids,
                "obligation_ids",
                required=True,
                empty_reason=ResidualLlmPacketReason.MISSING_OBLIGATIONS,
            ),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _commands(self.validation_commands, required=True),
        )
        object.__setattr__(
            self,
            "acceptance_ids",
            _ids(
                self.acceptance_ids,
                "acceptance_ids",
                required=False,
                limit=DEFAULT_MAX_ACCEPTANCE_IDS,
            ),
        )
        capsule = _capsule_mapping(self.counterexample_capsule)
        object.__setattr__(self, "counterexample_capsule", capsule)
        roots = _authority_roots(self.authority_roots)
        object.__setattr__(self, "authority_roots", roots)
        object.__setattr__(
            self,
            "codex_packet_ref",
            _optional_text(self.codex_packet_ref, "codex_packet_ref"),
        )
        object.__setattr__(
            self,
            "transition_ref",
            _optional_text(self.transition_ref, "transition_ref"),
        )
        max_bytes = _positive_int(self.max_bytes, "max_bytes", minimum=1024)
        max_tokens = _positive_int(self.max_tokens, "max_tokens", minimum=256)
        max_capsule_bytes = _positive_int(
            self.max_capsule_bytes, "max_capsule_bytes", minimum=1024
        )
        # Capsule budget cannot exceed the sealed packet budget. Clamp rather
        # than reject so callers that only tighten max_bytes remain valid.
        if max_capsule_bytes > max_bytes:
            max_capsule_bytes = max_bytes
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "max_capsule_bytes", max_capsule_bytes)
        object.__setattr__(
            self, "producer_id", _text(self.producer_id or PRODUCER_ID, "producer_id")
        )

        # Authority hard-zeros: residual packets never grant model authority.
        if self.nomination_only is not True:
            raise ResidualLlmPacketError(
                "residual packet must remain nomination_only",
                reason_code=ResidualLlmPacketReason.AUTHORITY_CLAIM,
            )
        for name in (
            "semantic_authority",
            "write_authority",
            "completion_authority",
        ):
            if getattr(self, name) is not False:
                raise ResidualLlmPacketError(
                    f"residual packet must hard-zero {name}",
                    reason_code=ResidualLlmPacketReason.AUTHORITY_CLAIM,
                )
            object.__setattr__(self, name, False)
        object.__setattr__(self, "nomination_only", True)

        # Reject secrets / body dumps anywhere in the sealed surface.
        _reject_forbidden_payload(self._identity_payload(), where="residual packet")

        # Capsule byte bound (CodexRepairPacket-aligned; fail closed).
        capsule_bytes = len(canonical_json(capsule).encode("utf-8"))
        if capsule_bytes > self.max_capsule_bytes:
            raise ResidualLlmPacketBudgetError(
                "counterexample capsule exceeds max_capsule_bytes",
                reason_code=ResidualLlmPacketReason.OVER_BUDGET,
            )

        computed = content_identity(self._identity_payload())
        claimed = _optional_text(self.packet_id, "packet_id")
        if claimed and claimed != computed:
            raise ResidualLlmPacketError(
                "packet_id does not match content identity",
                reason_code=ResidualLlmPacketReason.MALFORMED,
            )
        object.__setattr__(self, "packet_id", computed)

        # Bounds are checked after identity seal so packet_id bytes count.
        if self.byte_size > self.max_bytes:
            raise ResidualLlmPacketBudgetError(
                "residual LLM packet exceeds max_bytes",
                reason_code=ResidualLlmPacketReason.OVER_BUDGET,
            )
        if self.estimated_tokens > self.max_tokens:
            raise ResidualLlmPacketBudgetError(
                "residual LLM packet exceeds max_tokens",
                reason_code=ResidualLlmPacketReason.OVER_BUDGET,
            )

    def _identity_payload(self) -> dict[str, Any]:
        """Payload used for content addressing (excludes stored packet_id)."""

        return {
            "schema": self.SCHEMA,
            "packet_version": RESIDUAL_LLM_PACKET_VERSION,
            "interface": RESIDUAL_LLM_PACKET_INTERFACE,
            "evidence": RESIDUAL_LLM_PACKET_EVIDENCE,
            "task_id": self.task_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "forest_id": self.forest_id,
            "write_paths": list(self.write_paths),
            "obligation_ids": list(self.obligation_ids),
            "counterexample_capsule": dict(self.counterexample_capsule),
            "validation_commands": list(self.validation_commands),
            "acceptance_ids": list(self.acceptance_ids),
            "authority_roots": dict(self.authority_roots),
            "codex_packet_ref": self.codex_packet_ref,
            "transition_ref": self.transition_ref,
            "limits": {
                "max_bytes": self.max_bytes,
                "max_tokens": self.max_tokens,
                "max_capsule_bytes": self.max_capsule_bytes,
            },
            "semantic_authority": False,
            "write_authority": False,
            "completion_authority": False,
            "nomination_only": True,
            "producer_id": self.producer_id,
            "contains_source_body": False,
            "contains_secrets": False,
        }

    def _payload(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["packet_id"] = self.packet_id
        return payload

    @property
    def content_id(self) -> str:
        """Content identity of the sealed residual packet.

        Equals :attr:`packet_id` (identity over the payload excluding the
        stored packet_id field itself).
        """

        return self.packet_id

    @property
    def byte_size(self) -> int:
        # Measure the sealed surface including packet_id once identity is set;
        # during construction packet_id may still be empty — identity payload
        # size is a strict lower bound and is re-checked after identity seal.
        body = self._identity_payload()
        if self.packet_id:
            body = {**body, "packet_id": self.packet_id}
        return len(canonical_json(body).encode("utf-8"))

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.byte_size)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualLlmPacket":
        if not isinstance(payload, Mapping):
            raise ResidualLlmPacketError(
                "residual packet must be a mapping",
                reason_code=ResidualLlmPacketReason.MALFORMED,
            )
        schema = payload.get("schema")
        if schema not in {None, RESIDUAL_LLM_PACKET_SCHEMA}:
            raise ResidualLlmPacketError(
                "unsupported residual LLM packet schema",
                reason_code=ResidualLlmPacketReason.MALFORMED,
            )
        limits = payload.get("limits") if isinstance(payload.get("limits"), Mapping) else {}
        return cls(
            task_id=payload.get("task_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            write_paths=tuple(payload.get("write_paths") or ()),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            counterexample_capsule=payload.get("counterexample_capsule") or {},
            validation_commands=tuple(payload.get("validation_commands") or ()),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            forest_id=payload.get("forest_id", ""),
            acceptance_ids=tuple(payload.get("acceptance_ids") or ()),
            authority_roots=payload.get("authority_roots") or {},
            codex_packet_ref=payload.get("codex_packet_ref", ""),
            transition_ref=payload.get("transition_ref", ""),
            max_bytes=limits.get("max_bytes", payload.get("max_bytes", DEFAULT_MAX_PACKET_BYTES)),
            max_tokens=limits.get(
                "max_tokens", payload.get("max_tokens", DEFAULT_MAX_PACKET_TOKENS)
            ),
            max_capsule_bytes=limits.get(
                "max_capsule_bytes",
                payload.get("max_capsule_bytes", DEFAULT_MAX_CAPSULE_BYTES),
            ),
            semantic_authority=bool(payload.get("semantic_authority", False)),
            write_authority=bool(payload.get("write_authority", False)),
            completion_authority=bool(payload.get("completion_authority", False)),
            nomination_only=payload.get("nomination_only", True),
            producer_id=payload.get("producer_id", PRODUCER_ID),
            packet_id=payload.get("packet_id", ""),
        )


def seal_residual_llm_packet(
    *,
    task_id: str,
    repository_id: str,
    tree_id: str,
    write_paths: Sequence[str],
    obligation_ids: Sequence[str],
    counterexample_capsule: Any,
    validation_commands: Sequence[str],
    policy_id: str = "",
    policy_revision: str = "",
    forest_id: str = "",
    acceptance_ids: Sequence[str] = (),
    authority_roots: Mapping[str, str] | None = None,
    codex_packet: Any = None,
    codex_packet_ref: str = "",
    transition_ref: str = "",
    limits: ResidualLlmPacketLimits | Mapping[str, Any] | None = None,
) -> ResidualLlmPacket:
    """Seal a residual LLM packet for provider invocation.

    Optional ``codex_packet`` (a :class:`CodexRepairPacket` or compatible
    object) supplies the counterexample capsule and transition reference when
    those fields are not provided explicitly. Codex packet redaction is never
    relaxed.
    """

    if limits is None:
        active_limits = ResidualLlmPacketLimits()
    elif isinstance(limits, ResidualLlmPacketLimits):
        active_limits = limits
    elif isinstance(limits, Mapping):
        active_limits = ResidualLlmPacketLimits.from_dict(limits)
    else:
        raise ResidualLlmPacketError(
            "limits must be ResidualLlmPacketLimits or a mapping",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )

    capsule = counterexample_capsule
    resolved_codex_ref = codex_packet_ref
    resolved_transition_ref = transition_ref

    if codex_packet is not None:
        if capsule is None and hasattr(codex_packet, "counterexample_capsule"):
            capsule = codex_packet.counterexample_capsule
        if not resolved_transition_ref and hasattr(codex_packet, "transition"):
            transition = codex_packet.transition
            if hasattr(transition, "content_id"):
                resolved_transition_ref = str(transition.content_id)
            elif hasattr(transition, "to_dict"):
                resolved_transition_ref = content_identity(transition.to_dict())
        if not resolved_codex_ref:
            if hasattr(codex_packet, "to_dict"):
                resolved_codex_ref = content_identity(codex_packet.to_dict())
            else:
                resolved_codex_ref = content_identity({"codex_packet": str(codex_packet)})
        # Align max bounds with Codex packet when tighter.
        if hasattr(codex_packet, "max_bytes"):
            codex_max = int(codex_packet.max_bytes)
            if codex_max < active_limits.max_bytes:
                codex_tokens = int(
                    getattr(codex_packet, "max_tokens", active_limits.max_tokens)
                )
                active_limits = ResidualLlmPacketLimits(
                    max_bytes=codex_max,
                    max_tokens=min(active_limits.max_tokens, codex_tokens),
                    max_capsule_bytes=min(
                        active_limits.max_capsule_bytes, codex_max
                    ),
                    max_write_paths=active_limits.max_write_paths,
                    max_obligations=active_limits.max_obligations,
                    max_validation_commands=active_limits.max_validation_commands,
                )

    write_paths = _exact_paths(
        write_paths,
        "write_paths",
        required=True,
        limit=active_limits.max_write_paths,
    )
    obligation_ids = _ids(
        obligation_ids,
        "obligation_ids",
        required=True,
        limit=active_limits.max_obligations,
        empty_reason=ResidualLlmPacketReason.MISSING_OBLIGATIONS,
    )
    validation_commands = _commands(
        validation_commands,
        required=True,
        limit=active_limits.max_validation_commands,
    )
    capsule_map = _capsule_mapping(capsule)
    capsule_bytes = len(canonical_json(capsule_map).encode("utf-8"))
    if capsule_bytes > active_limits.max_capsule_bytes:
        raise ResidualLlmPacketBudgetError(
            "counterexample capsule exceeds max_capsule_bytes",
            reason_code=ResidualLlmPacketReason.OVER_BUDGET,
        )

    return ResidualLlmPacket(
        task_id=task_id,
        repository_id=repository_id,
        tree_id=tree_id,
        write_paths=write_paths,
        obligation_ids=obligation_ids,
        counterexample_capsule=capsule_map,
        validation_commands=validation_commands,
        policy_id=policy_id,
        policy_revision=policy_revision,
        forest_id=forest_id,
        acceptance_ids=tuple(acceptance_ids or ()),
        authority_roots=authority_roots or {},
        codex_packet_ref=resolved_codex_ref,
        transition_ref=resolved_transition_ref,
        max_bytes=active_limits.max_bytes,
        max_tokens=active_limits.max_tokens,
        max_capsule_bytes=active_limits.max_capsule_bytes,
    )


def residual_llm_packet_from_codex(
    codex_packet: Any,
    *,
    task_id: str,
    repository_id: str,
    tree_id: str,
    write_paths: Sequence[str],
    obligation_ids: Sequence[str],
    validation_commands: Sequence[str],
    policy_id: str = "",
    policy_revision: str = "",
    forest_id: str = "",
    acceptance_ids: Sequence[str] = (),
    authority_roots: Mapping[str, str] | None = None,
    limits: ResidualLlmPacketLimits | Mapping[str, Any] | None = None,
) -> ResidualLlmPacket:
    """Project a CodexRepairPacket into a sealed ResidualLlmPacket@1."""

    if codex_packet is None:
        raise ResidualLlmPacketError(
            "codex_packet is required",
            reason_code=ResidualLlmPacketReason.MALFORMED,
        )
    return seal_residual_llm_packet(
        task_id=task_id,
        repository_id=repository_id,
        tree_id=tree_id,
        write_paths=write_paths,
        obligation_ids=obligation_ids,
        counterexample_capsule=None,
        validation_commands=validation_commands,
        policy_id=policy_id,
        policy_revision=policy_revision,
        forest_id=forest_id,
        acceptance_ids=acceptance_ids,
        authority_roots=authority_roots,
        codex_packet=codex_packet,
        limits=limits,
    )


def packet_satisfies_residual_llm_contract(packet: ResidualLlmPacket | Mapping[str, Any]) -> bool:
    """Return True when ``packet`` is a valid sealed ResidualLlmPacket@1."""

    try:
        if isinstance(packet, ResidualLlmPacket):
            # Re-seal from dict to re-run validation (identity must hold).
            ResidualLlmPacket.from_dict(packet.to_dict())
        else:
            ResidualLlmPacket.from_dict(packet)
    except ResidualLlmPacketError:
        return False
    return True


__all__ = (
    "BYTES_PER_TOKEN",
    "DEFAULT_MAX_CAPSULE_BYTES",
    "DEFAULT_MAX_PACKET_BYTES",
    "DEFAULT_MAX_PACKET_TOKENS",
    "DEFAULT_MAX_OBLIGATIONS",
    "DEFAULT_MAX_VALIDATION_COMMANDS",
    "DEFAULT_MAX_WRITE_PATHS",
    "PRODUCER_ID",
    "REQUIRED_CORE_FIELDS",
    "RESIDUAL_LLM_PACKET_EVIDENCE",
    "RESIDUAL_LLM_PACKET_INTERFACE",
    "RESIDUAL_LLM_PACKET_INVARIANTS",
    "RESIDUAL_LLM_PACKET_LIMITS_SCHEMA",
    "RESIDUAL_LLM_PACKET_SCHEMA",
    "RESIDUAL_LLM_PACKET_VERSION",
    "ResidualLlmPacket",
    "ResidualLlmPacketBudgetError",
    "ResidualLlmPacketError",
    "ResidualLlmPacketLimits",
    "ResidualLlmPacketReason",
    "estimate_tokens",
    "packet_satisfies_residual_llm_contract",
    "residual_llm_packet_from_codex",
    "seal_residual_llm_packet",
)
