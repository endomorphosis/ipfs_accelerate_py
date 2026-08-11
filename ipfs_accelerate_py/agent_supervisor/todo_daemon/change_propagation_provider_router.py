"""Bounded ``llm_router`` fallback for admitted unresolved propagation steps.

RPR-041 / ``ChangePropagationProviderRouter@1``

Analytical repair is attempted first (RPR-037).  This module escalates *only*
when a plan-bound multi-edit packet (RPR-040) already carries a
behavior-complete model-required step **and** a supported analytical
non-success reason.  Routing reuses the existing proposal / independent-review
/ writer-lease boundary in
:mod:`contract_packet_provider_router` and the canonical accelerator
``llm_router`` adapter — never a direct datasets or model call.

Authority rules (fail-closed):

* Analytical steps never invoke a provider.
* Escalation requires a closed analytical non-success reason that is both
  supported for model work and paired with behavior-complete admitted
  semantics (value sources, behavior clauses, exact paths).
* Prompts are redacted, body-free, and bounded (time / token / context / tool /
  path).  Provider, model, and config identities are frozen on the envelope.
* The model may propose a patch within the write lease only.  It cannot choose
  a value source, behavior, owner, dependency, consumer set, plan order, or
  path.
* Proposed diffs are untrusted until deterministic scope parsing, supervisor
  review, admission, and post-edit proof.  Timeout, unavailable, refusal,
  malformed, or scope-escape outcomes create no write.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .contract_packet_provider_router import (
    MAX_PROVIDER_PROMPT_BYTES,
    MAX_PROVIDER_PROMPT_TOKENS,
    MAX_PROVIDER_RESPONSE_BYTES,
    MAX_PROVIDER_TIMEOUT_SECONDS,
    AdmissionCallable,
    AdmissionDecision,
    ImplementationRoutingResult,
    ProviderBounds,
    ProviderCallable,
    ProviderExecutionReceipt,
    ProviderProposal,
    ProviderQuotaLatch,
    ProviderReason,
    ProviderRole,
    ProviderRoutingError,
    RouteStatus,
    TokenCounter,
    WriterCallable,
    build_provider_execution_receipt,
    redact_provider_data,
    route_contract_packet,
)
from .contract_packet_provider_router import (
    _canonical_bytes as _provider_canonical_bytes,
)
from .contract_packet_provider_router import (
    _default_token_count as _provider_default_token_count,
)
from .contract_packet_provider_router import (
    _sha256 as _provider_sha256,
)
from ..proof.change_propagation_edit_packet import (
    CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
    ChangePropagationEditPacket,
    PropagationEditStep,
    PropagationEditStepKind,
)


# ---------------------------------------------------------------------------
# Schema / bounds
# ---------------------------------------------------------------------------

CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE: Final[str] = (
    "ChangePropagationProviderRouter@1"
)
PROPAGATION_PROVIDER_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-provider-envelope@1"
)
PROPAGATION_PROPOSAL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-proposal-receipt@1"
)
PROPAGATION_PROVIDER_ROUTE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-provider-route@1"
)
WRITER_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-writer-lease@1"
)

PRODUCER_ID: Final[str] = "change-propagation-provider-router@1"
CONTRACT_VERSION: Final[int] = 1

MAX_ENVELOPE_PROMPT_BYTES: Final[int] = MAX_PROVIDER_PROMPT_BYTES
MAX_ENVELOPE_PROMPT_TOKENS: Final[int] = MAX_PROVIDER_PROMPT_TOKENS
MAX_ENVELOPE_RESPONSE_BYTES: Final[int] = MAX_PROVIDER_RESPONSE_BYTES
MAX_ENVELOPE_TIMEOUT_SECONDS: Final[float] = MAX_PROVIDER_TIMEOUT_SECONDS
MAX_TOOL_NAMES: Final[int] = 16
MAX_PATHS: Final[int] = 1_024
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_IDS: Final[int] = 1_024

# Closed set of tools the model may be told about.  Empty by default: the
# proposal boundary is patch-only and tool use is not a model choice.
DEFAULT_ALLOWED_TOOLS: Final[tuple[str, ...]] = ()

# Choices the model is explicitly forbidden from making (prompt authority).
MODEL_FORBIDDEN_CHOICES: Final[tuple[str, ...]] = (
    "value_source",
    "behavior",
    "owner",
    "dependency",
    "consumer_set",
    "plan_order",
    "path",
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class AnalyticalNonSuccessReason(str, Enum):
    """Closed analytical dispositions that may escalate to a bounded model.

    Only reasons that leave semantics already admitted (behavior-complete,
    unique value mappings, exact paths) and fail solely on syntax or bounded
    implementation availability are eligible.  Ambiguity, unknown semantics,
    scope escape, and invented behavior remain fail-closed and never escalate.
    """

    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    UNSUPPORTED_KIND = "unsupported_kind"
    MISSING_DETERMINISTIC_RENDER = "missing_deterministic_render"
    COMPLEX_IMPLEMENTATION_REQUIRED = "complex_implementation_required"
    BEHAVIOR_IMPLEMENTATION_GAP = "behavior_implementation_gap"


# Reasons that must never open a model path (semantic / authority failures).
_BLOCKED_ANALYTICAL_REASONS: Final[frozenset[str]] = frozenset(
    {
        "ambiguous",
        "ambiguous_overload",
        "unknown",
        "unknown_semantics",
        "unsupported_semantics",
        "missing_behavior",
        "missing_proof",
        "non_total_mapping",
        "new_dependency",
        "scope_escape",
        "invented_behavior",
        "no_code_authority",
        "root_mismatch",
        "path_not_authorized",
        "stale_span",
        "dynamic_splat",
        "expression_mismatch",
        "alternatives",
        "abstained",
        "rejected",
        "timeout",
        "resource_exhausted",
    }
)

_SUPPORTED_ANALYTICAL_REASONS: Final[frozenset[str]] = frozenset(
    item.value for item in AnalyticalNonSuccessReason
)


class PropagationProviderReason(str, Enum):
    """Stable machine-readable route dispositions."""

    ROUTED = "bounded_propagation_provider_route"
    ANALYTICAL_ONLY = "analytical_step_never_invokes_provider"
    ANALYTICAL_SUCCESS = "analytical_success_no_model"
    ANALYTICAL_REASON_MISSING = "analytical_non_success_reason_required"
    ANALYTICAL_REASON_UNSUPPORTED = "analytical_non_success_reason_unsupported"
    ANALYTICAL_REASON_BLOCKED = "analytical_non_success_reason_blocked"
    BEHAVIOR_INCOMPLETE = "behavior_incomplete"
    VALUE_SOURCE_INCOMPLETE = "value_source_incomplete"
    STEP_NOT_MODEL_REQUIRED = "step_not_model_required"
    STEP_NOT_FOUND = "step_not_found"
    STEP_ORDER_VIOLATION = "step_order_violation"
    PACKET_MALFORMED = "packet_malformed"
    PACKET_STALE = "packet_stale"
    ROOT_MISMATCH = "root_mismatch"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    MODEL_IDENTITY_MISMATCH = "model_identity_mismatch"
    CONFIG_IDENTITY_MISMATCH = "config_identity_mismatch"
    PROMPT_TOO_LARGE = "provider_prompt_too_large"
    PROMPT_TOKEN_BUDGET = "provider_prompt_token_budget_exceeded"
    BROAD_CONTEXT_FORBIDDEN = "broad_repository_context_forbidden"
    SCOPE_ESCAPE = "proposal_scope_escape"
    MALFORMED_PROPOSAL = "proposal_malformed"
    PATH_LEASE_MISMATCH = "writer_lease_path_mismatch"
    WRITER_LEASE_REQUIRED = "writer_lease_required"
    WRITER_LEASE_EXPIRED = "writer_lease_expired"
    WRITER_NOT_CONFIGURED = "writer_not_configured"
    WRITE_FAILED = "admitted_write_failed"
    ADMISSION_REQUIRED = "proposal_admission_required"
    PROPOSAL_REJECTED = "proposal_rejected"
    PROPOSAL_UNTRUSTED = "proposal_untrusted_until_admission"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_REFUSAL = "provider_refusal"
    PROVIDER_FAILURE = "provider_failure"
    PROVIDER_RESPONSE_MALFORMED = "provider_response_malformed"
    PROVIDER_RESPONSE_TOO_LARGE = "provider_response_too_large"
    PROVIDER_AUTHORITY_CLAIM = "provider_authority_claim"
    NO_WRITE = "no_write"
    DELEGATED = "delegated_to_contract_packet_provider_router"


class PropagationRouteStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FALLBACK = "fallback"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class PropagationProviderRoutingError(ValueError):
    """Typed fail-closed boundary error for change-propagation provider routes."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: PropagationProviderReason | ProviderReason | str,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    try:
        return _provider_canonical_bytes(value)
    except ProviderRoutingError as exc:
        raise PropagationProviderRoutingError(
            str(exc),
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        ) from exc


def _sha256(value: bytes) -> str:
    return _provider_sha256(value)


def _default_token_count(value: bytes) -> int:
    return _provider_default_token_count(value)


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            raise PropagationProviderRoutingError(
                f"{name} is required",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        return ""
    if not isinstance(value, str):
        raise PropagationProviderRoutingError(
            f"{name} must be a string",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    text = value.strip()
    if required and not text:
        raise PropagationProviderRoutingError(
            f"{name} is required",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    encoded = text.encode("utf-8")
    if len(encoded) > limit:
        raise PropagationProviderRoutingError(
            f"{name} exceeds its UTF-8 byte bound",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True)
    if any(ch.isspace() for ch in text):
        raise PropagationProviderRoutingError(
            f"{name} must not contain whitespace",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    return text


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, required=True, limit=MAX_PATH_BYTES)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or text.startswith("./"):
        raise PropagationProviderRoutingError(
            f"{name} must be a safe repository-relative path",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    if text.endswith("/") or "\\" in text or "\x00" in text:
        raise PropagationProviderRoutingError(
            f"{name} is not a concrete file path",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    return text


def _paths(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        if required:
            raise PropagationProviderRoutingError(
                f"{name} is required",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PropagationProviderRoutingError(
            f"{name} must be a sequence of paths",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    if len(values) > MAX_PATHS:
        raise PropagationProviderRoutingError(
            f"{name} exceeds the path bound",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    result = tuple(_path(item, f"{name}[{index}]") for index, item in enumerate(values))
    if len(set(result)) != len(result):
        raise PropagationProviderRoutingError(
            f"{name} must not contain duplicates",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    return result


def _ids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values is None:
        if required:
            raise PropagationProviderRoutingError(
                f"{name} is required",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PropagationProviderRoutingError(
            f"{name} must be a sequence of identifiers",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    if len(values) > MAX_IDS:
        raise PropagationProviderRoutingError(
            f"{name} exceeds the identifier bound",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    return tuple(_identifier(item, f"{name}[{index}]") for index, item in enumerate(values))


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise PropagationProviderRoutingError(
                f"{name} is not a supported {enum_cls.__name__}",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            ) from exc
    raise PropagationProviderRoutingError(
        f"{name} must be a {enum_cls.__name__}",
        reason_code=PropagationProviderReason.PACKET_MALFORMED,
    )


def normalize_analytical_non_success_reason(
    value: Any,
) -> AnalyticalNonSuccessReason:
    """Parse and admit only supported analytical non-success reasons."""

    if value is None or value == "":
        raise PropagationProviderRoutingError(
            "analytical non-success reason is required for model escalation",
            reason_code=PropagationProviderReason.ANALYTICAL_REASON_MISSING,
        )
    if isinstance(value, AnalyticalNonSuccessReason):
        return value
    text = str(getattr(value, "value", value)).strip().casefold().replace("-", "_")
    if text in _BLOCKED_ANALYTICAL_REASONS:
        raise PropagationProviderRoutingError(
            f"analytical reason {text!r} is blocked from model escalation",
            reason_code=PropagationProviderReason.ANALYTICAL_REASON_BLOCKED,
        )
    if text not in _SUPPORTED_ANALYTICAL_REASONS:
        raise PropagationProviderRoutingError(
            f"analytical non-success reason {text!r} is not supported for llm_router",
            reason_code=PropagationProviderReason.ANALYTICAL_REASON_UNSUPPORTED,
        )
    return AnalyticalNonSuccessReason(text)


# ---------------------------------------------------------------------------
# Deterministic proposal scope parsing
# ---------------------------------------------------------------------------

_GIT_A_PREFIX: Final[re.Pattern[str]] = re.compile(r"^--- a/(.+)$")
_GIT_B_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_GIT: Final[re.Pattern[str]] = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def parse_proposal_paths(proposal: Mapping[str, Any] | Any) -> tuple[str, ...]:
    """Deterministically extract concrete paths from an untrusted proposal.

    Prefers explicit ``declared_paths`` when present, then falls back to
    unified-diff headers.  Never trusts free-form model narration.
    """

    payload: Mapping[str, Any]
    if isinstance(proposal, Mapping):
        payload = proposal
    else:
        nested = getattr(proposal, "payload", None)
        if isinstance(nested, Mapping):
            payload = nested
        elif isinstance(proposal, str):
            payload = {"patch": proposal}
        else:
            raise PropagationProviderRoutingError(
                "proposal must be a mapping or provider proposal",
                reason_code=PropagationProviderReason.MALFORMED_PROPOSAL,
            )

    # Provider responses often nest the concrete proposal under "proposal".
    inner = payload.get("proposal")
    if isinstance(inner, Mapping):
        # Prefer nested declared_paths/patch when the outer envelope has none.
        if (
            "declared_paths" not in payload
            and "patch" not in payload
            and "diff" not in payload
            and "unified_diff" not in payload
        ):
            payload = inner
        elif "declared_paths" not in payload and "declared_paths" in inner:
            merged = dict(payload)
            merged["declared_paths"] = inner["declared_paths"]
            if "patch" not in merged and "patch" in inner:
                merged["patch"] = inner["patch"]
            payload = merged
        elif "patch" not in payload and "patch" in inner:
            merged = dict(payload)
            merged["patch"] = inner["patch"]
            payload = merged

    declared = payload.get("declared_paths")
    if declared is not None:
        if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
            raise PropagationProviderRoutingError(
                "declared_paths must be a sequence of repository paths",
                reason_code=PropagationProviderReason.MALFORMED_PROPOSAL,
            )
        paths = []
        for index, item in enumerate(declared):
            try:
                paths.append(_path(item, f"declared_paths[{index}]"))
            except PropagationProviderRoutingError as exc:
                raise PropagationProviderRoutingError(
                    f"declared path escapes repository safety: {item!r}",
                    reason_code=PropagationProviderReason.SCOPE_ESCAPE,
                ) from exc
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        ordered: list[str] = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                ordered.append(path)
        return tuple(ordered)

    patch = payload.get("patch")
    if patch is None:
        patch = payload.get("diff") or payload.get("unified_diff") or ""
    if not isinstance(patch, str):
        raise PropagationProviderRoutingError(
            "proposal patch must be a string when declared_paths is absent",
            reason_code=PropagationProviderReason.MALFORMED_PROPOSAL,
        )
    if not patch.strip():
        raise PropagationProviderRoutingError(
            "proposal must declare paths or include a non-empty patch",
            reason_code=PropagationProviderReason.MALFORMED_PROPOSAL,
        )

    found: list[str] = []
    seen_paths: set[str] = set()
    for line in patch.splitlines():
        match = _DIFF_GIT.match(line)
        if match:
            for group in match.groups():
                if group and group != "/dev/null" and group not in seen_paths:
                    try:
                        normalized = _path(group, "patch_path")
                    except PropagationProviderRoutingError as exc:
                        raise PropagationProviderRoutingError(
                            f"patch path escapes repository safety: {group!r}",
                            reason_code=PropagationProviderReason.SCOPE_ESCAPE,
                        ) from exc
                    seen_paths.add(normalized)
                    found.append(normalized)
            continue
        for pattern in (_GIT_A_PREFIX, _GIT_B_PREFIX):
            match = pattern.match(line)
            if match:
                group = match.group(1)
                if group and group != "/dev/null" and group not in seen_paths:
                    try:
                        normalized = _path(group, "patch_path")
                    except PropagationProviderRoutingError as exc:
                        raise PropagationProviderRoutingError(
                            f"patch path escapes repository safety: {group!r}",
                            reason_code=PropagationProviderReason.SCOPE_ESCAPE,
                        ) from exc
                    seen_paths.add(normalized)
                    found.append(normalized)
    if not found:
        raise PropagationProviderRoutingError(
            "could not deterministically parse any concrete path from the proposal",
            reason_code=PropagationProviderReason.MALFORMED_PROPOSAL,
        )
    return tuple(found)


def assert_proposal_within_lease(
    proposal: Mapping[str, Any] | Any,
    *,
    allowed_write_paths: Sequence[str],
) -> tuple[str, ...]:
    """Reject any proposal path outside the exact write lease."""

    allowed = set(_paths(allowed_write_paths, "allowed_write_paths", required=True))
    actual = parse_proposal_paths(proposal)
    escaped = [path for path in actual if path not in allowed]
    if escaped:
        raise PropagationProviderRoutingError(
            f"proposal paths escape the writer lease: {escaped!r}",
            reason_code=PropagationProviderReason.SCOPE_ESCAPE,
        )
    return actual


# ---------------------------------------------------------------------------
# Writer lease
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriterLease:
    """Exact path-bound write authority for one admitted proposal application.

    The model never issues a lease.  Supervisor code binds identity, paths,
    plan/step ids, and optional deadline before any write may run.
    """

    lease_id: str
    permitted_write_paths: tuple[str, ...]
    packet_id: str
    plan_id: str
    step_id: str
    tree_id: str = ""
    provider_id: str = ""
    model_id: str = ""
    config_id: str = ""
    expires_at: str = ""
    schema: str = WRITER_LEASE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id"))
        object.__setattr__(
            self,
            "permitted_write_paths",
            _paths(self.permitted_write_paths, "permitted_write_paths", required=True),
        )
        object.__setattr__(self, "packet_id", _identifier(self.packet_id, "packet_id"))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, "tree_id", required=False)
        )
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, "provider_id", required=False)
        )
        object.__setattr__(
            self, "model_id", _text(self.model_id, "model_id", required=False)
        )
        object.__setattr__(
            self, "config_id", _text(self.config_id, "config_id", required=False)
        )
        object.__setattr__(
            self, "expires_at", _text(self.expires_at, "expires_at", required=False)
        )
        if self.schema != WRITER_LEASE_SCHEMA:
            raise PropagationProviderRoutingError(
                "unsupported writer lease schema",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )

    def contains(self, path: str) -> bool:
        return _path(path, "path") in set(self.permitted_write_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lease_id": self.lease_id,
            "permitted_write_paths": list(self.permitted_write_paths),
            "packet_id": self.packet_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "tree_id": self.tree_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "expires_at": self.expires_at,
            "model_issues_lease": False,
        }


# ---------------------------------------------------------------------------
# Provider / model / config identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderModelConfigIdentity:
    """Frozen provider, model, and config identity for one escalation."""

    provider_id: str
    model_id: str
    config_id: str
    router_backend: str = "llm_router"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _identifier(self.provider_id, "provider_id")
        )
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "config_id", _identifier(self.config_id, "config_id"))
        object.__setattr__(
            self,
            "router_backend",
            _identifier(self.router_backend, "router_backend"),
        )
        if self.router_backend != "llm_router":
            raise PropagationProviderRoutingError(
                "only the canonical llm_router backend is permitted",
                reason_code=PropagationProviderReason.PROVIDER_IDENTITY_MISMATCH,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "router_backend": self.router_backend,
        }


@dataclass(frozen=True, slots=True)
class PropagationProviderBounds:
    """Time / token / context / tool / path bounds for one escalation."""

    max_prompt_tokens: int = MAX_ENVELOPE_PROMPT_TOKENS
    max_prompt_bytes: int = MAX_ENVELOPE_PROMPT_BYTES
    max_response_bytes: int = MAX_ENVELOPE_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    max_context_paths: int = MAX_PATHS
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS

    def __post_init__(self) -> None:
        for name in (
            "max_prompt_tokens",
            "max_prompt_bytes",
            "max_response_bytes",
            "max_context_paths",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        timeout = self.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
            or float(timeout) > MAX_ENVELOPE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"timeout_seconds must be in (0, {MAX_ENVELOPE_TIMEOUT_SECONDS:g}]"
            )
        object.__setattr__(self, "timeout_seconds", float(timeout))
        tools = self.allowed_tools
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise ValueError("allowed_tools must be a sequence of tool names")
        if len(tools) > MAX_TOOL_NAMES:
            raise ValueError("allowed_tools exceeds the tool bound")
        object.__setattr__(
            self,
            "allowed_tools",
            tuple(_identifier(item, f"allowed_tools[{i}]") for i, item in enumerate(tools)),
        )

    def to_provider_bounds(self) -> ProviderBounds:
        return ProviderBounds(
            max_prompt_tokens=self.max_prompt_tokens,
            max_prompt_bytes=self.max_prompt_bytes,
            max_response_bytes=self.max_response_bytes,
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_response_bytes": self.max_response_bytes,
            "timeout_seconds": self.timeout_seconds,
            "max_context_paths": self.max_context_paths,
            "allowed_tools": list(self.allowed_tools),
        }


# ---------------------------------------------------------------------------
# Envelope and receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PropagationProviderEnvelope:
    """Bounded redacted prompt for one admitted unresolved step.

    Contains exact contract delta identity, chosen value mappings, behavior
    clauses, counterexamples, paths, postconditions, validations, analytical
    non-success reason, and frozen provider/model/config identity.  Never
    embeds source, proof bodies, secrets, or alternatives.
    """

    packet_id: str
    plan_id: str
    plan_content_id: str
    step_id: str
    task_id: str
    snapshot_id: str
    analytical_non_success_reason: AnalyticalNonSuccessReason
    delta_id: str
    change_set_id: str
    obligation_ids: tuple[str, ...]
    selected_value_sources: tuple[Mapping[str, Any], ...]
    required_behavior_ids: tuple[str, ...]
    counterexample_refs: tuple[str, ...]
    proof_refs: tuple[str, ...]
    read_paths: tuple[str, ...]
    write_paths: tuple[str, ...]
    before_hashes: tuple[Mapping[str, Any], ...]
    postcondition_refs: tuple[str, ...]
    fixed_point_obligation_ref: str
    validation_commands: tuple[str, ...]
    unsupported_limits: tuple[str, ...]
    identity: ProviderModelConfigIdentity
    bounds: PropagationProviderBounds
    roots: Mapping[str, Any]
    dependency_step_ids: tuple[str, ...] = ()
    scc_group_id: str = ""
    precondition_refs: tuple[str, ...] = ()
    schema: str = PROPAGATION_PROVIDER_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _identifier(self.packet_id, "packet_id"))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "plan_content_id", _identifier(self.plan_content_id, "plan_content_id")
        )
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "analytical_non_success_reason",
            normalize_analytical_non_success_reason(self.analytical_non_success_reason),
        )
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self, "change_set_id", _identifier(self.change_set_id, "change_set_id")
        )
        object.__setattr__(
            self, "obligation_ids", _ids(self.obligation_ids, "obligation_ids", required=True)
        )
        if not isinstance(self.selected_value_sources, Sequence):
            raise PropagationProviderRoutingError(
                "selected_value_sources must be a sequence",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        value_sources: list[Mapping[str, Any]] = []
        for item in self.selected_value_sources:
            if not isinstance(item, Mapping):
                raise PropagationProviderRoutingError(
                    "selected_value_sources entries must be mappings",
                    reason_code=PropagationProviderReason.PACKET_MALFORMED,
                )
            value_sources.append(MappingProxyType(dict(item)))
        object.__setattr__(self, "selected_value_sources", tuple(value_sources))
        object.__setattr__(
            self,
            "required_behavior_ids",
            _ids(self.required_behavior_ids, "required_behavior_ids", required=True),
        )
        object.__setattr__(
            self,
            "counterexample_refs",
            _ids(self.counterexample_refs, "counterexample_refs"),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths"))
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths", required=True)
        )
        if not isinstance(self.before_hashes, Sequence):
            raise PropagationProviderRoutingError(
                "before_hashes must be a sequence",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        hashes: list[Mapping[str, Any]] = []
        for item in self.before_hashes:
            if not isinstance(item, Mapping):
                raise PropagationProviderRoutingError(
                    "before_hashes entries must be mappings",
                    reason_code=PropagationProviderReason.PACKET_MALFORMED,
                )
            hashes.append(MappingProxyType(dict(item)))
        object.__setattr__(self, "before_hashes", tuple(hashes))
        object.__setattr__(
            self,
            "postcondition_refs",
            _ids(self.postcondition_refs, "postcondition_refs"),
        )
        object.__setattr__(
            self,
            "fixed_point_obligation_ref",
            _text(
                self.fixed_point_obligation_ref,
                "fixed_point_obligation_ref",
                required=False,
            ),
        )
        if self.validation_commands is None:
            object.__setattr__(self, "validation_commands", ())
        elif not isinstance(self.validation_commands, Sequence) or isinstance(
            self.validation_commands, (str, bytes)
        ):
            raise PropagationProviderRoutingError(
                "validation_commands must be a sequence of command strings",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        else:
            commands: list[str] = []
            for index, item in enumerate(self.validation_commands):
                commands.append(
                    _text(item, f"validation_commands[{index}]", required=True)
                )
            object.__setattr__(self, "validation_commands", tuple(commands))
        object.__setattr__(
            self,
            "unsupported_limits",
            _ids(self.unsupported_limits, "unsupported_limits"),
        )
        if not isinstance(self.identity, ProviderModelConfigIdentity):
            raise PropagationProviderRoutingError(
                "identity must be ProviderModelConfigIdentity",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        if not isinstance(self.bounds, PropagationProviderBounds):
            raise PropagationProviderRoutingError(
                "bounds must be PropagationProviderBounds",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        if not isinstance(self.roots, Mapping):
            raise PropagationProviderRoutingError(
                "roots must be a mapping",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        object.__setattr__(self, "roots", MappingProxyType(dict(self.roots)))
        object.__setattr__(
            self,
            "dependency_step_ids",
            _ids(self.dependency_step_ids, "dependency_step_ids"),
        )
        object.__setattr__(
            self, "scc_group_id", _text(self.scc_group_id, "scc_group_id", required=False)
        )
        object.__setattr__(
            self,
            "precondition_refs",
            _ids(self.precondition_refs, "precondition_refs"),
        )
        if self.schema != PROPAGATION_PROVIDER_ENVELOPE_SCHEMA:
            raise PropagationProviderRoutingError(
                "unsupported provider envelope schema",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        # Context path bound: read + write paths must fit the configured ceiling.
        context_paths = set(self.read_paths) | set(self.write_paths)
        if len(context_paths) > self.bounds.max_context_paths:
            raise PropagationProviderRoutingError(
                "envelope path context exceeds max_context_paths",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )

    @property
    def content_id(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "producer_id": PRODUCER_ID,
            "packet_id": self.packet_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "analytical_non_success_reason": self.analytical_non_success_reason.value,
            "contract_delta": {
                "delta_id": self.delta_id,
                "change_set_id": self.change_set_id,
                "plan_id": self.plan_id,
                "plan_content_id": self.plan_content_id,
            },
            "obligation_ids": list(self.obligation_ids),
            "selected_value_sources": [dict(item) for item in self.selected_value_sources],
            "required_behavior_ids": list(self.required_behavior_ids),
            "counterexample_refs": list(self.counterexample_refs),
            "proof_refs": list(self.proof_refs),
            "scope": {
                "read_paths": list(self.read_paths),
                "write_paths": list(self.write_paths),
            },
            "before_hashes": [dict(item) for item in self.before_hashes],
            "postcondition_refs": list(self.postcondition_refs),
            "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
            "validation_commands": list(self.validation_commands),
            "unsupported_limits": list(self.unsupported_limits),
            "dependency_step_ids": list(self.dependency_step_ids),
            "scc_group_id": self.scc_group_id,
            "precondition_refs": list(self.precondition_refs),
            "identity": self.identity.to_dict(),
            "bounds": self.bounds.to_dict(),
            "roots": dict(self.roots),
            "authority": {
                "provider_output_tier": "proposal",
                "repository_write_allowed": False,
                "proof_authoritative": False,
                "completion_authoritative": False,
                "model_may_choose": [],
                "model_must_not_choose": list(MODEL_FORBIDDEN_CHOICES),
            },
            "body_embedded": False,
            "secrets_embedded": False,
            "alternatives_embedded": False,
        }

    def provider_input_payload(self) -> Mapping[str, Any]:
        """Compact packet-shaped payload for :class:`ImplementationProviderRouter`."""

        payload = {
            "goal": {
                "packet_interface": CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
                "plan_id": self.plan_id,
                "plan_content_id": self.plan_content_id,
                "step_id": self.step_id,
                "delta_id": self.delta_id,
                "change_set_id": self.change_set_id,
                "obligation_ids": list(self.obligation_ids),
                "required_behavior_ids": list(self.required_behavior_ids),
                "selected_value_sources": [
                    dict(item) for item in self.selected_value_sources
                ],
                "counterexample_refs": list(self.counterexample_refs),
                "proof_refs": list(self.proof_refs),
                "analytical_non_success_reason": (
                    self.analytical_non_success_reason.value
                ),
                "postcondition_refs": list(self.postcondition_refs),
                "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
                "unsupported_limits": list(self.unsupported_limits),
                "dependency_step_ids": list(self.dependency_step_ids),
                "scc_group_id": self.scc_group_id,
            },
            "authority": {
                "provider_semantic_authority": False,
                "proof_authoritative": False,
                "completion_authoritative": False,
                "model_must_not_choose": list(MODEL_FORBIDDEN_CHOICES),
            },
            "scope": {
                "read_paths": list(self.read_paths),
                "write_paths": list(self.write_paths),
            },
            "acceptance": {
                "validation_commands": list(self.validation_commands),
                "postcondition_refs": list(self.postcondition_refs),
                "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
            },
            "identity": self.identity.to_dict(),
            "before_hashes": [dict(item) for item in self.before_hashes],
            "roots": {
                key: self.roots[key]
                for key in (
                    "repository_id",
                    "candidate_tree_id",
                    "graph_id",
                    "index_id",
                    "model_id",
                    "config_id",
                    "policy_id",
                )
                if key in self.roots
            },
        }
        return MappingProxyType(redact_provider_data(payload))


@dataclass(frozen=True, slots=True)
class PropagationProposalReceipt:
    """Receipt for one bounded propagation provider proposal route.

    Proposal content is never completion- or proof-authoritative.  Write
    outcomes are recorded only when a supervisor writer ran under a valid
    lease after admission.
    """

    receipt_id: str
    status: PropagationRouteStatus
    reason_code: str
    packet_id: str
    plan_id: str
    step_id: str
    analytical_non_success_reason: str
    envelope_digest: str
    provider_identity: Mapping[str, Any]
    bounds: Mapping[str, Any]
    proposal_digest: str = ""
    proposal_admitted: bool = False
    proposal_paths: tuple[str, ...] = ()
    scope_parsed: bool = False
    write_performed: bool = False
    writer_lease_id: str = ""
    provider_execution_receipt_id: str = ""
    review_presence: str = ""
    attempts: tuple[Mapping[str, Any], ...] = ()
    schema: str = PROPAGATION_PROPOSAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, PropagationRouteStatus, "status")
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", required=True)
        )
        object.__setattr__(self, "packet_id", _text(self.packet_id, "packet_id", required=False))
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id", required=False))
        object.__setattr__(self, "step_id", _text(self.step_id, "step_id", required=False))
        object.__setattr__(
            self,
            "analytical_non_success_reason",
            _text(
                self.analytical_non_success_reason,
                "analytical_non_success_reason",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "envelope_digest",
            _text(self.envelope_digest, "envelope_digest", required=False),
        )
        if not isinstance(self.provider_identity, Mapping):
            raise PropagationProviderRoutingError(
                "provider_identity must be a mapping",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        object.__setattr__(
            self, "provider_identity", MappingProxyType(dict(self.provider_identity))
        )
        if not isinstance(self.bounds, Mapping):
            raise PropagationProviderRoutingError(
                "bounds must be a mapping",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )
        object.__setattr__(self, "bounds", MappingProxyType(dict(self.bounds)))
        object.__setattr__(
            self,
            "proposal_paths",
            tuple(str(item) for item in self.proposal_paths),
        )
        object.__setattr__(
            self,
            "writer_lease_id",
            self.writer_lease_id if self.write_performed else "",
        )
        if self.schema != PROPAGATION_PROPOSAL_RECEIPT_SCHEMA:
            raise PropagationProviderRoutingError(
                "unsupported proposal receipt schema",
                reason_code=PropagationProviderReason.PACKET_MALFORMED,
            )

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def proposal_trusted(self) -> bool:
        """True only after scope parse + admission (still never completion authority)."""

        return bool(self.proposal_admitted and self.scope_parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE,
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "packet_id": self.packet_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "analytical_non_success_reason": self.analytical_non_success_reason,
            "envelope_digest": self.envelope_digest,
            "provider_identity": dict(self.provider_identity),
            "bounds": dict(self.bounds),
            "proposal_digest": self.proposal_digest,
            "proposal_admitted": self.proposal_admitted,
            "proposal_paths": list(self.proposal_paths),
            "scope_parsed": self.scope_parsed,
            "proposal_trusted": self.proposal_trusted,
            "write_performed": self.write_performed,
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "provider_execution_receipt_id": self.provider_execution_receipt_id,
            "review_presence": self.review_presence,
            "attempts": [dict(item) for item in self.attempts],
            "proof_authoritative": False,
            "completion_authoritative": False,
        }


@dataclass(frozen=True, slots=True)
class PropagationProviderRouteResult:
    """Outcome of one change-propagation provider route attempt."""

    status: PropagationRouteStatus
    reason_code: str
    envelope: PropagationProviderEnvelope | None = None
    receipt: PropagationProposalReceipt | None = None
    selected_proposal: ProviderProposal | None = None
    implementation_route: ImplementationRoutingResult | None = None
    provider_execution_receipt: ProviderExecutionReceipt | None = None
    write_performed: bool = False
    writer_lease_id: str = ""
    proposal_paths: tuple[str, ...] = ()

    @property
    def admitted(self) -> bool:
        return bool(
            self.receipt is not None
            and self.receipt.proposal_admitted
            and self.selected_proposal is not None
            and self.selected_proposal.admitted
        )

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_PROVIDER_ROUTE_SCHEMA,
            "interface": CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "envelope": self.envelope.to_dict() if self.envelope is not None else None,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "write_performed": self.write_performed,
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "proposal_paths": list(self.proposal_paths),
            "proof_authoritative": False,
            "completion_authoritative": False,
            "provider_execution_receipt_id": (
                self.provider_execution_receipt.receipt_id
                if self.provider_execution_receipt is not None
                else ""
            ),
            "implementation_status": (
                self.implementation_route.status.value
                if self.implementation_route is not None
                else ""
            ),
        }


# ---------------------------------------------------------------------------
# Packet adapter for ImplementationProviderRouter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EnvelopePacketAdapter:
    """Thin adapter so the existing contract provider router can host us."""

    envelope: PropagationProviderEnvelope
    implementable: bool = True

    @property
    def packet_id(self) -> str:
        return self.envelope.packet_id

    @property
    def snapshot_id(self) -> str:
        return self.envelope.snapshot_id

    @property
    def task_id(self) -> str:
        return self.envelope.task_id

    def assert_current(self, current_snapshot_id: str) -> None:
        if current_snapshot_id != self.envelope.snapshot_id:
            raise ValueError("stale propagation packet snapshot")

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return self.envelope.provider_input_payload()


# ---------------------------------------------------------------------------
# Envelope materialization
# ---------------------------------------------------------------------------


def _step_by_id(
    packet: ChangePropagationEditPacket, step_id: str
) -> PropagationEditStep:
    for step in packet.steps:
        if step.step_id == step_id:
            return step
    raise PropagationProviderRoutingError(
        f"step {step_id!r} is not present on the edit packet",
        reason_code=PropagationProviderReason.STEP_NOT_FOUND,
    )


def _selected_values_for_step(
    packet: ChangePropagationEditPacket, step: PropagationEditStep
) -> tuple[Mapping[str, Any], ...]:
    # Prefer step-local selected sources; fall back to packet-level sources
    # filtered by obligation consumer when step-local is empty.
    if step.selected_value_sources:
        return tuple(item.to_dict() for item in step.selected_value_sources)
    obligation_set = set(step.obligation_ids)
    matched: list[Mapping[str, Any]] = []
    for item in packet.selected_value_sources:
        # consumer_id may appear as consumer:X while obligation is obligation:consumer:X
        consumer = item.consumer_id
        if any(consumer in oid or oid.endswith(consumer) for oid in obligation_set):
            matched.append(item.to_dict())
        elif not obligation_set:
            matched.append(item.to_dict())
    if matched:
        return tuple(matched)
    # Behavior-complete model steps with placement-only value bindings may still
    # carry packet-level unique sources; expose only unique packet sources that
    # do not broaden alternatives (packet already forbids alternatives).
    return tuple(item.to_dict() for item in packet.selected_value_sources)


def _before_hashes_for_step(
    packet: ChangePropagationEditPacket, step: PropagationEditStep
) -> tuple[Mapping[str, Any], ...]:
    if step.before_hashes:
        return tuple(item.to_dict() for item in step.before_hashes)
    paths = set(step.read_paths) | set(step.write_paths)
    return tuple(
        item.to_dict() for item in packet.before_hashes if item.path in paths
    )


def build_propagation_provider_envelope(
    packet: ChangePropagationEditPacket,
    *,
    step_id: str,
    analytical_non_success_reason: AnalyticalNonSuccessReason | str,
    identity: ProviderModelConfigIdentity,
    bounds: PropagationProviderBounds | None = None,
    task_id: str = "",
    snapshot_id: str = "",
) -> PropagationProviderEnvelope:
    """Materialize a bounded redacted prompt for one model-required step.

    Raises :class:`PropagationProviderRoutingError` when the step is analytical,
    behavior-incomplete, missing value authority, or the analytical reason is
    unsupported / blocked.
    """

    if not isinstance(packet, ChangePropagationEditPacket):
        raise PropagationProviderRoutingError(
            "packet must be a ChangePropagationEditPacket",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )
    reason = normalize_analytical_non_success_reason(analytical_non_success_reason)
    step = _step_by_id(packet, step_id)

    if step.kind is PropagationEditStepKind.ANALYTICAL:
        raise PropagationProviderRoutingError(
            "analytical steps never invoke llm_router",
            reason_code=PropagationProviderReason.ANALYTICAL_ONLY,
        )
    if step.kind is not PropagationEditStepKind.MODEL_REQUIRED:
        raise PropagationProviderRoutingError(
            f"step kind {step.kind.value!r} is not model-required",
            reason_code=PropagationProviderReason.STEP_NOT_MODEL_REQUIRED,
        )
    if step_id not in packet.model_required_step_ids:
        raise PropagationProviderRoutingError(
            "step is not in the packet model-required partition",
            reason_code=PropagationProviderReason.STEP_NOT_MODEL_REQUIRED,
        )
    if not step.required_behavior_ids and not packet.required_behavior_ids:
        raise PropagationProviderRoutingError(
            "model escalation requires behavior-complete required behavior bindings",
            reason_code=PropagationProviderReason.BEHAVIOR_INCOMPLETE,
        )
    if not step.write_paths:
        raise PropagationProviderRoutingError(
            "model-required steps require exact write path authority",
            reason_code=PropagationProviderReason.PACKET_MALFORMED,
        )

    selected = _selected_values_for_step(packet, step)
    # Value sources are required when the packet admits any; empty is allowed
    # only when the step is pure behavior/placement with no missing inputs.
    if packet.selected_value_sources and not selected and step.obligation_ids:
        # Soft: still require at least packet-level unique sources if any exist.
        selected = tuple(item.to_dict() for item in packet.selected_value_sources)

    behavior_ids = step.required_behavior_ids or packet.required_behavior_ids
    if not behavior_ids:
        raise PropagationProviderRoutingError(
            "behavior-complete bindings are required",
            reason_code=PropagationProviderReason.BEHAVIOR_INCOMPLETE,
        )

    # Root / identity binding: packet model/config roots must agree when set.
    roots = packet.roots.to_dict() if hasattr(packet.roots, "to_dict") else dict(packet.roots)  # type: ignore[arg-type]
    if roots.get("model_id") and roots["model_id"] != identity.model_id:
        # Packet root model_id is graph/model root, not llm model — only fail
        # when identity explicitly claims a config root conflict.
        pass
    if roots.get("config_id") and identity.config_id and roots["config_id"] != identity.config_id:
        # Config root on the packet is toolchain/policy config; identity.config_id
        # is the provider config.  They are distinct namespaces — do not hard-fail.
        pass

    resolved_snapshot = snapshot_id or str(
        roots.get("candidate_tree_id") or roots.get("base_tree_id") or packet.plan_content_id
    )
    resolved_task = task_id or f"propagation:{packet.plan_id}:{step.step_id}"

    bounds = bounds or PropagationProviderBounds()
    envelope = PropagationProviderEnvelope(
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        plan_content_id=packet.plan_content_id,
        step_id=step.step_id,
        task_id=resolved_task,
        snapshot_id=resolved_snapshot,
        analytical_non_success_reason=reason,
        delta_id=packet.delta_id,
        change_set_id=packet.change_set_id,
        obligation_ids=step.obligation_ids,
        selected_value_sources=selected,
        required_behavior_ids=behavior_ids,
        counterexample_refs=step.counterexample_refs or packet.counterexample_refs,
        proof_refs=step.proof_refs or packet.proof_refs,
        read_paths=step.read_paths or packet.permitted_read_paths,
        write_paths=step.write_paths,
        before_hashes=_before_hashes_for_step(packet, step),
        postcondition_refs=step.postcondition_refs or packet.per_edit_postcondition_refs,
        fixed_point_obligation_ref=packet.fixed_point_obligation_ref,
        validation_commands=tuple(packet.validation_commands),
        unsupported_limits=step.unsupported_limits or packet.unsupported_limits,
        identity=identity,
        bounds=bounds,
        roots=roots,
        dependency_step_ids=step.dependency_step_ids,
        scc_group_id=step.scc_group_id,
        precondition_refs=step.precondition_refs,
    )
    # Bound the serialized envelope itself.
    encoded = _canonical_bytes(envelope.to_dict())
    if len(encoded) > bounds.max_prompt_bytes:
        raise PropagationProviderRoutingError(
            "propagation provider envelope exceeds prompt byte bound",
            reason_code=PropagationProviderReason.PROMPT_TOO_LARGE,
        )
    tokens = _default_token_count(encoded)
    if tokens > bounds.max_prompt_tokens:
        raise PropagationProviderRoutingError(
            "propagation provider envelope exceeds prompt token bound",
            reason_code=PropagationProviderReason.PROMPT_TOKEN_BUDGET,
        )
    return envelope


# ---------------------------------------------------------------------------
# Canonical llm_router provider adapters
# ---------------------------------------------------------------------------


def make_llm_router_provider(
    *,
    identity: ProviderModelConfigIdentity,
    bounds: PropagationProviderBounds,
    repo_root: str | None = None,
    generate: Callable[[str, Mapping[str, Any]], str] | None = None,
) -> ProviderCallable:
    """Build a :class:`ProviderCallable` that uses canonical ``llm_router``.

    When ``generate`` is supplied (tests / injected adapters), it is used.
    Otherwise the typed child-process adapter :func:`call_llm_router` is loaded
    lazily so cold import stays free of network and process I/O.
    """

    def _provider(request: Any) -> Mapping[str, Any]:
        prompt_obj = request.to_dict() if hasattr(request, "to_dict") else dict(request)
        # Freeze identity into the call configuration; model cannot override.
        prompt_text = json.dumps(
            prompt_obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if len(prompt_text.encode("utf-8")) > bounds.max_prompt_bytes:
            raise ProviderRoutingError(
                "llm_router prompt exceeds byte bound",
                reason_code=ProviderReason.PROMPT_TOO_LARGE,
            )
        if generate is not None:
            text = generate(
                prompt_text,
                {
                    "provider_id": identity.provider_id,
                    "model_id": identity.model_id,
                    "config_id": identity.config_id,
                    "timeout_seconds": bounds.timeout_seconds,
                },
            )
        else:
            from pathlib import Path

            from .llm import LlmRouterInvocation, call_llm_router

            if not repo_root:
                raise ProviderRoutingError(
                    "repo_root is required for canonical llm_router invocation",
                    reason_code=ProviderReason.PROVIDER_FAILURE,
                )
            invocation = LlmRouterInvocation(
                repo_root=Path(repo_root),
                model_name=identity.model_id,
                provider=identity.provider_id,
                allow_local_fallback=False,
                timeout_seconds=int(bounds.timeout_seconds),
                max_new_tokens=min(2048, max(64, bounds.max_response_bytes // 4)),
                max_prompt_chars=bounds.max_prompt_bytes,
                temperature=0.0,
                backend_default="llm_router",
                env_prefix="CHANGE_PROPAGATION_LLM",
                prompt_file_prefix="change-propagation-llm-prompt-",
                result_file_prefix="change-propagation-llm-result-",
                envelope_file_prefix="change-propagation-llm-envelope-",
            )
            try:
                text = call_llm_router(prompt_text, invocation)
            except TimeoutError as exc:
                raise ProviderRoutingError(
                    "llm_router timed out",
                    reason_code=ProviderReason.PROVIDER_TIMEOUT,
                ) from exc
            except Exception as exc:
                message = str(exc).casefold()
                if "timeout" in message:
                    raise ProviderRoutingError(
                        "llm_router timed out",
                        reason_code=ProviderReason.PROVIDER_TIMEOUT,
                    ) from exc
                if "unavailable" in message or "not configured" in message:
                    raise ProviderRoutingError(
                        "llm_router unavailable",
                        reason_code=ProviderReason.GROK_UNAVAILABLE,
                    ) from exc
                if "refus" in message:
                    raise ProviderRoutingError(
                        "llm_router refused the request",
                        reason_code=ProviderReason.PROVIDER_FAILURE,
                    ) from exc
                raise ProviderRoutingError(
                    f"llm_router failed: {type(exc).__name__}",
                    reason_code=ProviderReason.PROVIDER_FAILURE,
                ) from exc

        if not isinstance(text, str) or not text.strip():
            raise ProviderRoutingError(
                "llm_router returned an empty response",
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
            )
        if len(text.encode("utf-8")) > bounds.max_response_bytes:
            raise ProviderRoutingError(
                "llm_router response exceeds byte bound",
                reason_code=ProviderReason.PROVIDER_RESPONSE_TOO_LARGE,
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Treat raw text as a unified diff patch proposal.
            return {"proposal": {"patch": text, "source": "llm_router"}}
        if not isinstance(parsed, Mapping):
            raise ProviderRoutingError(
                "llm_router response must be a JSON object or patch text",
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
            )
        # Normalize nested proposal shapes.
        if "proposal" in parsed and isinstance(parsed["proposal"], Mapping):
            return dict(parsed)
        if "patch" in parsed or "declared_paths" in parsed:
            return {"proposal": dict(parsed), "source": "llm_router"}
        return {"proposal": dict(parsed), "source": "llm_router"}

    return _provider


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class ChangePropagationProviderRouter:
    """Route only admitted unresolved (model-required) steps through llm_router.

    Delegates the sequential Grok proposal → admission → Codex review path to
    :class:`ImplementationProviderRouter` / :func:`route_contract_packet`, and
    adds change-propagation gates: analytical non-success, behavior
    completeness, redacted envelope construction, path lease, and deterministic
    proposal scope parsing.
    """

    identity: ProviderModelConfigIdentity
    bounds: PropagationProviderBounds = field(default_factory=PropagationProviderBounds)
    grok_provider: ProviderCallable | None = None
    codex_provider: ProviderCallable | None = None
    deterministic_provider: ProviderCallable | None = None
    admission_gate: AdmissionCallable | None = None
    writer: WriterCallable | None = None
    grok_quota: ProviderQuotaLatch = field(default_factory=ProviderQuotaLatch)
    codex_quota: ProviderQuotaLatch = field(default_factory=ProviderQuotaLatch)
    token_counter: TokenCounter = _default_token_count
    repo_root: str | None = None
    llm_generate: Callable[[str, Mapping[str, Any]], str] | None = None
    _writer_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ProviderModelConfigIdentity):
            raise TypeError("identity must be ProviderModelConfigIdentity")
        if not isinstance(self.bounds, PropagationProviderBounds):
            if isinstance(self.bounds, Mapping):
                self.bounds = PropagationProviderBounds(**dict(self.bounds))
            else:
                raise TypeError("bounds must be PropagationProviderBounds or a mapping")
        for name in ("grok_quota", "codex_quota"):
            value = getattr(self, name)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, name, ProviderQuotaLatch(remaining_calls=value))
            elif not isinstance(value, ProviderQuotaLatch):
                raise TypeError(f"{name} must be ProviderQuotaLatch or an integer")

    def build_envelope(
        self,
        packet: ChangePropagationEditPacket,
        *,
        step_id: str,
        analytical_non_success_reason: AnalyticalNonSuccessReason | str,
        task_id: str = "",
        snapshot_id: str = "",
    ) -> PropagationProviderEnvelope:
        return build_propagation_provider_envelope(
            packet,
            step_id=step_id,
            analytical_non_success_reason=analytical_non_success_reason,
            identity=self.identity,
            bounds=self.bounds,
            task_id=task_id,
            snapshot_id=snapshot_id,
        )

    def _resolve_implement_provider(self) -> ProviderCallable | None:
        if self.grok_provider is not None:
            return self.grok_provider
        if self.llm_generate is not None or self.repo_root is not None:
            return make_llm_router_provider(
                identity=self.identity,
                bounds=self.bounds,
                repo_root=self.repo_root,
                generate=self.llm_generate,
            )
        return None

    def _scope_admission_gate(
        self,
        *,
        write_paths: Sequence[str],
        outer_gate: AdmissionCallable | None,
    ) -> AdmissionCallable:
        """Compose deterministic scope parsing with the supervisor admission gate.

        The proposal remains untrusted until both scope parse and outer admission
        succeed.  Scope escape rejects before any write path is considered.
        """

        allowed = tuple(write_paths)

        def gate(proposal: ProviderProposal, role: Any = None) -> Any:
            # Path/scope checks apply to implementation proposals only.  Independent
            # review payloads (approve/reject/repair metadata) are not patches.
            role_value = ""
            if role is not None:
                role_value = str(getattr(role, "value", role) or "")
            if not role_value:
                prop_role = getattr(proposal, "role", None)
                role_value = str(getattr(prop_role, "value", prop_role) or "")
            is_review = role_value == ProviderRole.CODEX_REVIEW.value
            if not is_review:
                # Also treat pure review envelopes without a patch as non-path payloads.
                payload = getattr(proposal, "payload", None)
                looks_like_review = (
                    isinstance(payload, Mapping)
                    and "decision" in payload
                    and "patch" not in payload
                    and "declared_paths" not in payload
                    and not (
                        isinstance(payload.get("proposal"), Mapping)
                        and (
                            "patch" in payload["proposal"]
                            or "declared_paths" in payload["proposal"]
                        )
                    )
                )
                if not looks_like_review:
                    try:
                        assert_proposal_within_lease(
                            proposal, allowed_write_paths=allowed
                        )
                    except PropagationProviderRoutingError as exc:
                        return AdmissionDecision(
                            False,
                            exc.reason_code
                            or PropagationProviderReason.SCOPE_ESCAPE.value,
                        )
            if outer_gate is None:
                return AdmissionDecision(
                    False, PropagationProviderReason.ADMISSION_REQUIRED.value
                )
            try:
                return (
                    outer_gate(proposal, role)
                    if role is not None
                    else outer_gate(proposal)
                )
            except TypeError:
                return outer_gate(proposal)

        return gate

    def _lease_bound_writer(
        self,
        *,
        lease: WriterLease | None,
        write_paths: Sequence[str],
    ) -> WriterCallable | None:
        if self.writer is None:
            return None
        allowed = tuple(write_paths)
        outer = self.writer

        def wrapped(proposal: ProviderProposal, lease_id: str) -> Any:
            if lease is None:
                raise PropagationProviderRoutingError(
                    "writer lease is required for apply",
                    reason_code=PropagationProviderReason.WRITER_LEASE_REQUIRED,
                )
            if lease_id != lease.lease_id:
                raise PropagationProviderRoutingError(
                    "writer lease id mismatch",
                    reason_code=PropagationProviderReason.WRITER_LEASE_REQUIRED,
                )
            if set(lease.permitted_write_paths) != set(allowed) and not set(
                lease.permitted_write_paths
            ).issuperset(set(allowed)):
                # Lease may be a subset of packet writes only if it still covers
                # the step write paths exactly.
                if set(lease.permitted_write_paths) != set(allowed):
                    # Require exact match to step write authority.
                    if set(lease.permitted_write_paths) != set(allowed):
                        pass
            if set(lease.permitted_write_paths) != set(allowed):
                raise PropagationProviderRoutingError(
                    "writer lease paths must equal step write authority",
                    reason_code=PropagationProviderReason.PATH_LEASE_MISMATCH,
                )
            actual = assert_proposal_within_lease(
                proposal, allowed_write_paths=lease.permitted_write_paths
            )
            if not actual:
                raise PropagationProviderRoutingError(
                    "admitted proposal has no concrete paths",
                    reason_code=PropagationProviderReason.MALFORMED_PROPOSAL,
                )
            with self._writer_lock:
                return outer(proposal, lease_id)

        return wrapped

    def _receipt(
        self,
        *,
        status: PropagationRouteStatus,
        reason_code: str,
        envelope: PropagationProviderEnvelope | None,
        implementation_route: ImplementationRoutingResult | None = None,
        proposal_paths: Sequence[str] = (),
        scope_parsed: bool = False,
        write_performed: bool = False,
        writer_lease_id: str = "",
        provider_execution_receipt: ProviderExecutionReceipt | None = None,
    ) -> PropagationProposalReceipt:
        selected = (
            implementation_route.selected_proposal
            if implementation_route is not None
            else None
        )
        admitted = bool(selected is not None and selected.admitted)
        body = {
            "status": status.value,
            "reason_code": reason_code,
            "packet_id": envelope.packet_id if envelope else "",
            "plan_id": envelope.plan_id if envelope else "",
            "step_id": envelope.step_id if envelope else "",
            "envelope_digest": envelope.content_id if envelope else "",
            "proposal_digest": selected.response_digest if selected else "",
            "proposal_admitted": admitted,
            "proposal_paths": list(proposal_paths),
            "scope_parsed": scope_parsed,
            "write_performed": write_performed,
            "writer_lease_id": writer_lease_id if write_performed else "",
        }
        receipt_id = _sha256(_canonical_bytes(body))
        return PropagationProposalReceipt(
            receipt_id=receipt_id,
            status=status,
            reason_code=reason_code,
            packet_id=envelope.packet_id if envelope else "",
            plan_id=envelope.plan_id if envelope else "",
            step_id=envelope.step_id if envelope else "",
            analytical_non_success_reason=(
                envelope.analytical_non_success_reason.value if envelope else ""
            ),
            envelope_digest=envelope.content_id if envelope else "",
            provider_identity=self.identity.to_dict(),
            bounds=self.bounds.to_dict(),
            proposal_digest=selected.response_digest if selected else "",
            proposal_admitted=admitted,
            proposal_paths=tuple(proposal_paths),
            scope_parsed=scope_parsed,
            write_performed=write_performed,
            writer_lease_id=writer_lease_id if write_performed else "",
            provider_execution_receipt_id=(
                provider_execution_receipt.receipt_id
                if provider_execution_receipt is not None
                else ""
            ),
            review_presence=(
                implementation_route.review_presence
                if implementation_route is not None
                else ""
            ),
            attempts=tuple(
                item.to_dict()
                for item in (
                    implementation_route.attempts if implementation_route is not None else ()
                )
            ),
        )

    def _result(
        self,
        *,
        status: PropagationRouteStatus,
        reason_code: str,
        envelope: PropagationProviderEnvelope | None = None,
        implementation_route: ImplementationRoutingResult | None = None,
        proposal_paths: Sequence[str] = (),
        scope_parsed: bool = False,
        write_performed: bool = False,
        writer_lease_id: str = "",
    ) -> PropagationProviderRouteResult:
        provider_receipt = None
        if implementation_route is not None:
            try:
                provider_receipt = build_provider_execution_receipt(implementation_route)
            except Exception:
                provider_receipt = None
        # Fail-closed: never report write without admission + scope parse.
        if write_performed and not (scope_parsed and implementation_route is not None):
            write_performed = False
            writer_lease_id = ""
        if write_performed and implementation_route is not None:
            if not implementation_route.write_performed:
                write_performed = False
                writer_lease_id = ""
        receipt = self._receipt(
            status=status,
            reason_code=reason_code,
            envelope=envelope,
            implementation_route=implementation_route,
            proposal_paths=proposal_paths,
            scope_parsed=scope_parsed,
            write_performed=write_performed,
            writer_lease_id=writer_lease_id,
            provider_execution_receipt=provider_receipt,
        )
        selected = (
            implementation_route.selected_proposal
            if implementation_route is not None
            else None
        )
        return PropagationProviderRouteResult(
            status=status,
            reason_code=reason_code,
            envelope=envelope,
            receipt=receipt,
            selected_proposal=selected,
            implementation_route=implementation_route,
            provider_execution_receipt=provider_receipt,
            write_performed=write_performed,
            writer_lease_id=writer_lease_id if write_performed else "",
            proposal_paths=tuple(proposal_paths),
        )

    def route_step(
        self,
        packet: ChangePropagationEditPacket,
        *,
        step_id: str,
        analytical_non_success_reason: AnalyticalNonSuccessReason | str,
        current_snapshot_id: str = "",
        task_id: str = "",
        apply: bool = False,
        writer_lease: WriterLease | None = None,
        writer_lease_id: str = "",
        local_only: bool = False,
    ) -> PropagationProviderRouteResult:
        """Route one admitted unresolved step through the bounded provider path.

        ``apply=False`` is the default.  A write requires an admitted proposal,
        deterministic scope parse, configured writer, and a matching
        :class:`WriterLease`.
        """

        try:
            envelope = self.build_envelope(
                packet,
                step_id=step_id,
                analytical_non_success_reason=analytical_non_success_reason,
                task_id=task_id,
                snapshot_id=current_snapshot_id,
            )
        except PropagationProviderRoutingError as exc:
            status = (
                PropagationRouteStatus.SKIPPED
                if exc.reason_code
                == PropagationProviderReason.ANALYTICAL_ONLY.value
                else PropagationRouteStatus.REJECTED
            )
            return self._result(status=status, reason_code=exc.reason_code)

        snapshot = current_snapshot_id or envelope.snapshot_id
        if snapshot != envelope.snapshot_id:
            return self._result(
                status=PropagationRouteStatus.REJECTED,
                reason_code=PropagationProviderReason.PACKET_STALE.value,
                envelope=envelope,
            )

        # Resolve lease before any provider call when apply is requested so
        # identity mismatches fail closed with no write and no model call when
        # clearly misconfigured... Actually apply can still need a model first.
        # Only validate lease shape early when provided.
        lease = writer_lease
        if apply and lease is None and writer_lease_id:
            # Build a lease from step write paths when only an id is supplied.
            lease = WriterLease(
                lease_id=writer_lease_id,
                permitted_write_paths=envelope.write_paths,
                packet_id=envelope.packet_id,
                plan_id=envelope.plan_id,
                step_id=envelope.step_id,
                tree_id=envelope.snapshot_id,
                provider_id=self.identity.provider_id,
                model_id=self.identity.model_id,
                config_id=self.identity.config_id,
            )
        if apply and lease is not None:
            if lease.step_id != envelope.step_id or lease.packet_id != envelope.packet_id:
                return self._result(
                    status=PropagationRouteStatus.REJECTED,
                    reason_code=PropagationProviderReason.PATH_LEASE_MISMATCH.value,
                    envelope=envelope,
                )
            if set(lease.permitted_write_paths) != set(envelope.write_paths):
                return self._result(
                    status=PropagationRouteStatus.REJECTED,
                    reason_code=PropagationProviderReason.PATH_LEASE_MISMATCH.value,
                    envelope=envelope,
                )

        implement = self._resolve_implement_provider()
        composed_gate = self._scope_admission_gate(
            write_paths=envelope.write_paths,
            outer_gate=self.admission_gate,
        )
        lease_writer = self._lease_bound_writer(
            lease=lease if apply else None,
            write_paths=envelope.write_paths,
        )

        adapter = _EnvelopePacketAdapter(envelope=envelope)
        try:
            implementation_route = route_contract_packet(
                adapter,
                current_snapshot_id=snapshot,
                grok_provider=implement,
                codex_provider=self.codex_provider,
                deterministic_provider=self.deterministic_provider,
                admission_gate=composed_gate,
                writer=lease_writer,
                apply=apply and lease is not None,
                writer_lease_id=lease.lease_id if lease is not None else "",
                local_only=local_only,
                bounds=self.bounds.to_provider_bounds(),
                grok_quota=self.grok_quota,
                codex_quota=self.codex_quota,
            )
        except ProviderRoutingError as exc:
            mapped = self._map_provider_reason(exc.reason_code)
            return self._result(
                status=PropagationRouteStatus.REJECTED,
                reason_code=mapped,
                envelope=envelope,
            )
        except Exception as exc:
            message = str(exc).casefold()
            if "timeout" in message:
                reason = PropagationProviderReason.PROVIDER_TIMEOUT.value
            elif "unavailable" in message:
                reason = PropagationProviderReason.PROVIDER_UNAVAILABLE.value
            elif "refus" in message:
                reason = PropagationProviderReason.PROVIDER_REFUSAL.value
            else:
                reason = PropagationProviderReason.PROVIDER_FAILURE.value
            return self._result(
                status=PropagationRouteStatus.REJECTED,
                reason_code=reason,
                envelope=envelope,
            )

        return self._finish_route(
            envelope=envelope,
            implementation_route=implementation_route,
            apply=apply,
            lease=lease,
        )

    def _map_provider_reason(self, reason_code: str) -> str:
        mapping = {
            ProviderReason.PROVIDER_TIMEOUT.value: (
                PropagationProviderReason.PROVIDER_TIMEOUT.value
            ),
            ProviderReason.PROVIDER_FAILURE.value: (
                PropagationProviderReason.PROVIDER_FAILURE.value
            ),
            ProviderReason.PROVIDER_RESPONSE_MALFORMED.value: (
                PropagationProviderReason.PROVIDER_RESPONSE_MALFORMED.value
            ),
            ProviderReason.PROVIDER_RESPONSE_TOO_LARGE.value: (
                PropagationProviderReason.PROVIDER_RESPONSE_TOO_LARGE.value
            ),
            ProviderReason.PROVIDER_AUTHORITY_CLAIM.value: (
                PropagationProviderReason.PROVIDER_AUTHORITY_CLAIM.value
            ),
            ProviderReason.PROMPT_TOO_LARGE.value: (
                PropagationProviderReason.PROMPT_TOO_LARGE.value
            ),
            ProviderReason.PROMPT_TOKEN_BUDGET.value: (
                PropagationProviderReason.PROMPT_TOKEN_BUDGET.value
            ),
            ProviderReason.BROAD_CONTEXT_FORBIDDEN.value: (
                PropagationProviderReason.BROAD_CONTEXT_FORBIDDEN.value
            ),
            ProviderReason.PACKET_STALE.value: (
                PropagationProviderReason.PACKET_STALE.value
            ),
            ProviderReason.GROK_UNAVAILABLE.value: (
                PropagationProviderReason.PROVIDER_UNAVAILABLE.value
            ),
            ProviderReason.WRITER_LEASE_REQUIRED.value: (
                PropagationProviderReason.WRITER_LEASE_REQUIRED.value
            ),
            ProviderReason.WRITE_FAILED.value: (
                PropagationProviderReason.WRITE_FAILED.value
            ),
            ProviderReason.PROPOSAL_REJECTED.value: (
                PropagationProviderReason.PROPOSAL_REJECTED.value
            ),
            ProviderReason.ADMISSION_REQUIRED.value: (
                PropagationProviderReason.ADMISSION_REQUIRED.value
            ),
        }
        return mapping.get(str(reason_code), str(reason_code))

    def _finish_route(
        self,
        *,
        envelope: PropagationProviderEnvelope,
        implementation_route: ImplementationRoutingResult,
        apply: bool,
        lease: WriterLease | None,
    ) -> PropagationProviderRouteResult:
        status_map = {
            RouteStatus.SUCCEEDED: PropagationRouteStatus.SUCCEEDED,
            RouteStatus.FALLBACK: PropagationRouteStatus.FALLBACK,
            RouteStatus.DEFERRED: PropagationRouteStatus.DEFERRED,
            RouteStatus.REJECTED: PropagationRouteStatus.REJECTED,
        }
        status = status_map.get(
            implementation_route.status, PropagationRouteStatus.REJECTED
        )
        reason = self._map_provider_reason(implementation_route.reason_code)

        proposal_paths: tuple[str, ...] = ()
        scope_parsed = False
        selected = implementation_route.selected_proposal
        if selected is not None and selected.admitted:
            try:
                proposal_paths = assert_proposal_within_lease(
                    selected, allowed_write_paths=envelope.write_paths
                )
                scope_parsed = True
            except PropagationProviderRoutingError as exc:
                # Admitted by outer gate but scope re-check failed: fail closed,
                # no write (even if the inner router wrote — we treat that as
                # write_failed by not reporting write_performed here if scope fails).
                return self._result(
                    status=PropagationRouteStatus.REJECTED,
                    reason_code=exc.reason_code,
                    envelope=envelope,
                    implementation_route=implementation_route,
                    proposal_paths=(),
                    scope_parsed=False,
                    write_performed=False,
                    writer_lease_id="",
                )
        elif selected is not None and not selected.admitted:
            # Untrusted proposal: attempt scope parse only for diagnostics.
            try:
                proposal_paths = parse_proposal_paths(selected)
            except PropagationProviderRoutingError:
                proposal_paths = ()
            scope_parsed = False

        write_performed = bool(
            implementation_route.write_performed
            and scope_parsed
            and selected is not None
            and selected.admitted
            and apply
            and lease is not None
        )
        if apply and not write_performed and implementation_route.write_performed:
            # Inner wrote but we refuse to honor it without scope — should not
            # happen with lease-bound writer; still fail closed on the receipt.
            write_performed = False

        if apply and not write_performed and status is PropagationRouteStatus.SUCCEEDED:
            # Successful review without write is still success if apply was
            # requested but lease missing — reclassify.
            if lease is None:
                status = PropagationRouteStatus.REJECTED
                reason = PropagationProviderReason.WRITER_LEASE_REQUIRED.value
            elif not scope_parsed:
                status = PropagationRouteStatus.REJECTED
                reason = PropagationProviderReason.PROPOSAL_UNTRUSTED.value

        return self._result(
            status=status,
            reason_code=reason
            if reason
            else PropagationProviderReason.DELEGATED.value,
            envelope=envelope,
            implementation_route=implementation_route,
            proposal_paths=proposal_paths,
            scope_parsed=scope_parsed,
            write_performed=write_performed,
            writer_lease_id=lease.lease_id if write_performed and lease else "",
        )

    def route_model_required_steps(
        self,
        packet: ChangePropagationEditPacket,
        *,
        analytical_non_success_by_step: Mapping[str, AnalyticalNonSuccessReason | str],
        current_snapshot_id: str = "",
        task_id: str = "",
        apply: bool = False,
        writer_leases: Mapping[str, WriterLease] | None = None,
        local_only: bool = False,
    ) -> tuple[PropagationProviderRouteResult, ...]:
        """Route every model-required step that has an analytical non-success.

        Analytical steps are never invoked.  Steps without a reason entry are
        rejected with ``analytical_non_success_reason_required``.  Order follows
        the packet's plan-bound ``step_order`` / model-required partition.
        """

        leases = writer_leases or {}
        results: list[PropagationProviderRouteResult] = []
        # Preserve plan order: walk step_order, only model-required members.
        ordered = [
            sid
            for sid in packet.step_order
            if sid in set(packet.model_required_step_ids)
        ]
        if not ordered:
            ordered = list(packet.model_required_step_ids)
        for step_id in ordered:
            reason = analytical_non_success_by_step.get(step_id)
            if reason is None:
                results.append(
                    self._result(
                        status=PropagationRouteStatus.REJECTED,
                        reason_code=(
                            PropagationProviderReason.ANALYTICAL_REASON_MISSING.value
                        ),
                    )
                )
                continue
            results.append(
                self.route_step(
                    packet,
                    step_id=step_id,
                    analytical_non_success_reason=reason,
                    current_snapshot_id=current_snapshot_id,
                    task_id=task_id or f"propagation:{packet.plan_id}:{step_id}",
                    apply=apply,
                    writer_lease=leases.get(step_id),
                    local_only=local_only,
                )
            )
        return tuple(results)


def route_change_propagation_step(
    packet: ChangePropagationEditPacket,
    *,
    step_id: str,
    analytical_non_success_reason: AnalyticalNonSuccessReason | str,
    identity: ProviderModelConfigIdentity,
    bounds: PropagationProviderBounds | Mapping[str, Any] | None = None,
    current_snapshot_id: str = "",
    task_id: str = "",
    grok_provider: ProviderCallable | None = None,
    codex_provider: ProviderCallable | None = None,
    deterministic_provider: ProviderCallable | None = None,
    admission_gate: AdmissionCallable | None = None,
    writer: WriterCallable | None = None,
    apply: bool = False,
    writer_lease: WriterLease | None = None,
    writer_lease_id: str = "",
    local_only: bool = False,
    repo_root: str | None = None,
    llm_generate: Callable[[str, Mapping[str, Any]], str] | None = None,
) -> PropagationProviderRouteResult:
    """Functional facade for one bounded change-propagation provider route."""

    router = ChangePropagationProviderRouter(
        identity=identity,
        bounds=(
            bounds
            if isinstance(bounds, PropagationProviderBounds)
            else PropagationProviderBounds(**dict(bounds or {}))
        ),
        grok_provider=grok_provider,
        codex_provider=codex_provider,
        deterministic_provider=deterministic_provider,
        admission_gate=admission_gate,
        writer=writer,
        repo_root=repo_root,
        llm_generate=llm_generate,
    )
    return router.route_step(
        packet,
        step_id=step_id,
        analytical_non_success_reason=analytical_non_success_reason,
        current_snapshot_id=current_snapshot_id,
        task_id=task_id,
        apply=apply,
        writer_lease=writer_lease,
        writer_lease_id=writer_lease_id,
        local_only=local_only,
    )


__all__ = [
    "AnalyticalNonSuccessReason",
    "CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE",
    "CONTRACT_VERSION",
    "ChangePropagationProviderRouter",
    "DEFAULT_ALLOWED_TOOLS",
    "MAX_ENVELOPE_PROMPT_BYTES",
    "MAX_ENVELOPE_PROMPT_TOKENS",
    "MAX_ENVELOPE_RESPONSE_BYTES",
    "MAX_ENVELOPE_TIMEOUT_SECONDS",
    "MODEL_FORBIDDEN_CHOICES",
    "PROPAGATION_PROPOSAL_RECEIPT_SCHEMA",
    "PROPAGATION_PROVIDER_ENVELOPE_SCHEMA",
    "PROPAGATION_PROVIDER_ROUTE_SCHEMA",
    "PRODUCER_ID",
    "PropagationProviderBounds",
    "PropagationProviderEnvelope",
    "PropagationProviderReason",
    "PropagationProviderRouteResult",
    "PropagationProviderRoutingError",
    "PropagationProposalReceipt",
    "PropagationRouteStatus",
    "ProviderModelConfigIdentity",
    "WRITER_LEASE_SCHEMA",
    "WriterLease",
    "assert_proposal_within_lease",
    "build_propagation_provider_envelope",
    "make_llm_router_provider",
    "normalize_analytical_non_success_reason",
    "parse_proposal_paths",
    "route_change_propagation_step",
]
