"""Optional, fail-closed proof-provider execution boundary.

The supervisor owns this module and depends only on the standard library and
its local proof contracts.  A logic package may be discovered through an
entry point or an explicit ``module:attribute`` reference, but is not imported
until its provider is actually invoked.

Protocol version 1 uses one JSON object per subprocess.  Requests and responses
are correlated by an unpredictable request id and bind the operation, resource
budget, deadline, and network policy.  Provider output is untrusted: malformed,
oversized, mismatched, or version-incompatible output is an explicit failure
and never becomes proof evidence merely because the provider says it succeeded.
"""

from __future__ import annotations

import errno
import importlib
import importlib.metadata
import json
import math
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, IO, Protocol, runtime_checkable

from .formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from .formal_verification_contracts import (
    ResourceBudget,
    canonical_json,
    content_identity,
)


PROOF_PROVIDER_PROTOCOL_VERSION = 1
PROOF_PROVIDER_SUPPORTED_PROTOCOL_VERSIONS = (PROOF_PROVIDER_PROTOCOL_VERSION,)
PROOF_PROVIDER_REQUEST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-provider-request@1"
PROOF_PROVIDER_RESPONSE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-provider-response@1"
PROOF_PROVIDER_ENTRY_POINT_GROUP = "ipfs_accelerate_py.proof_providers"
PROOF_PROVIDER_ENVIRONMENT = "IPFS_ACCELERATE_PROOF_PROVIDER"

DEFAULT_PROVIDER_TIMEOUT_SECONDS = 30.0
DEFAULT_PROVIDER_MAX_REQUEST_BYTES = 1 * 1024 * 1024
DEFAULT_PROVIDER_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_PROVIDER_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_PROVIDER_CPU_TIME_SECONDS = 30
DEFAULT_PROVIDER_MAX_PROCESSES = 16


class ProviderFailureCode(str, Enum):
    """Stable failure vocabulary used by both execution boundaries."""

    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK_DENIED = "network_denied"
    MALFORMED_REQUEST = "malformed_request"
    MALFORMED_RESPONSE = "malformed_response"
    PROTOCOL_ERROR = "protocol_error"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class ProviderFailure:
    """Serializable failure detail with bounded, non-authoritative metadata."""

    code: ProviderFailureCode | str
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            code = ProviderFailureCode(str(getattr(self.code, "value", self.code)))
        except ValueError as exc:
            raise ValueError("unknown proof-provider failure code") from exc
        if not isinstance(self.message, str):
            raise ValueError("proof-provider failure message must be a string")
        message = self.message.strip()
        if not message:
            raise ValueError("proof-provider failure message must not be empty")
        if not isinstance(self.retryable, bool):
            raise ValueError("proof-provider failure retryable must be a boolean")
        if not isinstance(self.details, Mapping):
            raise ValueError("proof-provider failure details must be an object")
        details = _json_object(self.details, field_name="failure details")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message[:4096])
        object.__setattr__(self, "details", details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderFailure":
        if not isinstance(payload, Mapping):
            raise ValueError("proof-provider error must be an object")
        return cls(
            code=payload.get("code", ""),
            message=payload.get("message", ""),
            retryable=payload.get("retryable", False),
            details=payload.get("details", {}),
        )


class ProofProviderError(RuntimeError):
    """Exception a provider may raise to return a typed expected failure."""

    def __init__(
        self,
        code: ProviderFailureCode | str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.failure = ProviderFailure(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )
        super().__init__(self.failure.message)

    @property
    def code(self) -> ProviderFailureCode:
        return self.failure.code


class ProviderInvocationError(ProofProviderError):
    """Raised by :meth:`ProviderResponse.require_result` for failed calls."""


class NetworkAccessDenied(ProofProviderError):
    """Provider-facing spelling for a denied network request."""

    def __init__(self, message: str = "provider network access is denied") -> None:
        super().__init__(ProviderFailureCode.NETWORK_DENIED, message)


class CancellationToken:
    """Thread-safe cooperative cancellation shared with in-process providers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self.cancelled

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ProofProviderError(
                ProviderFailureCode.CANCELLED,
                "proof-provider request was cancelled",
            )


def _json_value(value: Any, *, field_name: str) -> Any:
    """Round-trip through strict JSON so provider objects cannot escape."""

    def validate(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            raise ValueError(f"{field_name} cannot contain floating-point values")
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError(f"{field_name} object keys must be strings")
            for nested in item.values():
                validate(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                validate(nested)
            return
        raise ValueError(f"{field_name} contains unsupported value {type(item).__name__}")

    try:
        validate(value)
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must contain strict JSON values") from exc


def _json_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    normalized = _json_value(dict(value), field_name=field_name)
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be an object")
    return normalized


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {value}")
        ),
    )


def _resource_budget(value: ResourceBudget | Mapping[str, Any] | None) -> ResourceBudget:
    if value is None:
        return ResourceBudget()
    if isinstance(value, ResourceBudget):
        return value
    if isinstance(value, Mapping):
        return ResourceBudget.from_dict(value)
    raise ValueError("resource_budget must be a ResourceBudget or object")


PROOF_ATTEMPT_TRACE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-attempt-trace@1"
LEANSTRAL_PROOF_ATTEMPT_TRACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-proof-attempt-trace@1"
)
PROOF_ATTEMPT_TRACE_CONTRACT_VERSION = 1
J2_PROOF_ATTEMPT_FIELDS: Final = (
    "obligation",
    "state",
    "premises",
    "proposals",
    "model_tool_versions",
    "parse_outcome",
    "elaboration_outcome",
    "prover_outcome",
    "kernel_outcome",
    "errors",
    "counterexamples",
    "timeout",
    "resources",
)
_PROOF_ATTEMPT_TRACE_SCHEMAS = frozenset(
    {
        PROOF_ATTEMPT_TRACE_SCHEMA,
        LEANSTRAL_PROOF_ATTEMPT_TRACE_SCHEMA,
    }
)
_STAGE_OUTCOME_STATUSES = frozenset(
    {
        "not_attempted",
        "parsed",
        "parse_failed",
        "elaborated",
        "elaboration_failed",
        "rejected",
        "timed_out",
        "error",
        "counterexample",
        "accepted",
    }
)
_DEFINITIVE_FALSEHOOD_STATUSES = frozenset({"rejected", "counterexample"})
_IDENTITY_EXCLUDED_TRACE_KEYS = frozenset(
    {
        "attempt_id",
        "content_id",
        "request_id",
        "captured_at_unix_ms",
    }
)


class AttemptTraceFailureKind(str, Enum):
    """Closed admission failures for content-addressed proof-attempt traces."""

    MALFORMED = "malformed"
    STALE = "stale"
    WRONG_STATEMENT = "wrong_statement"
    REPLAYED = "replayed"


class ProofAttemptTraceAdmissionError(ValueError):
    """Fail-closed rejection of a proof-attempt trace."""

    def __init__(self, kind: AttemptTraceFailureKind | str, message: str) -> None:
        try:
            resolved = (
                kind
                if isinstance(kind, AttemptTraceFailureKind)
                else AttemptTraceFailureKind(str(kind))
            )
        except ValueError as exc:
            raise ValueError("unknown proof-attempt trace failure kind") from exc
        if not isinstance(message, str) or not message.strip():
            raise ValueError("proof-attempt trace failure message must not be empty")
        self.kind = resolved
        self.message = message.strip()
        super().__init__(self.message)


@dataclass(frozen=True)
class ProofAttemptTraceExpectation:
    """Current supervisor bindings a persisted attempt must still match."""

    request_id: str = ""
    obligation_id: str = ""
    theorem_id: str = ""
    statement_digest: str = ""
    canonical_source_digest: str = ""
    theorem_equivalence_key: str = ""
    freeze_root_cid: str = ""
    campaign_input_root_cid: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "obligation_id",
            "theorem_id",
            "statement_digest",
            "canonical_source_digest",
            "theorem_equivalence_key",
            "freeze_root_cid",
            "campaign_input_root_cid",
        ):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, "")
                continue
            if not isinstance(value, str):
                raise ValueError(f"proof-attempt expectation {name} must be a string")
            object.__setattr__(self, name, value.strip())


class ProofAttemptTraceStore:
    """In-memory ledger that makes replay of a sealed attempt identity fail."""

    def __init__(self) -> None:
        self._attempt_ids: set[str] = set()

    def seen(self, attempt_id: str) -> bool:
        return str(attempt_id or "") in self._attempt_ids

    def remember(self, attempt_id: str) -> None:
        normalized = str(attempt_id or "").strip()
        if normalized:
            self._attempt_ids.add(normalized)

    def __contains__(self, attempt_id: object) -> bool:
        return self.seen(str(attempt_id or ""))


def _trace_text(value: Any, *, field_name: str, required: bool = False) -> str:
    if value is None:
        if required:
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.MALFORMED,
                f"proof-attempt trace {field_name} is required",
            )
        return ""
    if not isinstance(value, str):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name} must be a string",
        )
    text = value.strip()
    if required and not text:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name} must not be empty",
        )
    return text


def _trace_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name} must be an object",
        )
    try:
        return _json_object(value, field_name=f"proof-attempt trace {field_name}")
    except ValueError as exc:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            str(exc),
        ) from exc


def _trace_array(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name} must be an array",
        )
    try:
        return _json_value(list(value), field_name=f"proof-attempt trace {field_name}")
    except ValueError as exc:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            str(exc),
        ) from exc


def _stage_outcome(value: Any, *, field_name: str) -> dict[str, Any]:
    payload = _trace_object(value, field_name=field_name)
    status = _trace_text(payload.get("status"), field_name=f"{field_name}.status", required=True)
    if status not in _STAGE_OUTCOME_STATUSES:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name}.status is not an admitted outcome",
        )
    reason_codes = _trace_array(payload.get("reason_codes", ()), field_name=f"{field_name}.reason_codes")
    if any(not isinstance(item, str) or not item.strip() for item in reason_codes):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name}.reason_codes must be nonempty strings",
        )
    extra = sorted(set(payload) - {"status", "reason_codes"})
    if extra:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            f"proof-attempt trace {field_name} has unknown fields: {', '.join(extra)}",
        )
    return {
        "status": status,
        "reason_codes": [str(item).strip() for item in reason_codes],
    }


def proof_attempt_trace_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the J2 identity surface; request correlation is excluded."""

    identity = {
        key: payload[key]
        for key in J2_PROOF_ATTEMPT_FIELDS
        if key in payload
    }
    timeout = identity.get("timeout")
    if isinstance(timeout, Mapping):
        # Elapsed wall time is observational and must not fork equivalent attempts.
        identity["timeout"] = {
            key: value for key, value in timeout.items() if key != "elapsed_ms"
        }
    identity["schema"] = payload.get("schema", PROOF_ATTEMPT_TRACE_SCHEMA)
    identity["contract_version"] = payload.get(
        "contract_version", PROOF_ATTEMPT_TRACE_CONTRACT_VERSION
    )
    identity["authoritative"] = False
    identity["candidate_authority"] = False
    identity["verified"] = False
    identity["proof_success"] = False
    return _json_object(identity, field_name="proof-attempt identity")


def compute_proof_attempt_id(payload: Mapping[str, Any]) -> str:
    """Content-address the J2 binding surface of one attempt."""

    return "pgir-attempt-" + content_identity(proof_attempt_trace_identity_payload(payload))


def expectation_from_provider_request(
    request: "ProviderRequest",
) -> ProofAttemptTraceExpectation:
    """Derive current statement/freeze bindings from a provider request."""

    payload = request.payload if isinstance(request.payload, Mapping) else {}
    raw_theorem = payload.get("fixed_theorem")
    if not isinstance(raw_theorem, Mapping):
        raw_theorem = payload.get("theorem_identity")
    if not isinstance(raw_theorem, Mapping):
        raw_theorem = payload.get("theorem")
    theorem = raw_theorem if isinstance(raw_theorem, Mapping) else {}
    obligation_id = payload.get("obligation_id", theorem.get("obligation_id", ""))
    if not obligation_id:
        raw_ids = payload.get("obligation_ids")
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            if raw_ids:
                obligation_id = raw_ids[0]
        elif isinstance(raw_ids, str):
            obligation_id = raw_ids
    return ProofAttemptTraceExpectation(
        request_id=request.request_id,
        obligation_id=str(obligation_id or ""),
        theorem_id=str(payload.get("theorem_id") or theorem.get("theorem_id") or ""),
        statement_digest=str(
            payload.get("statement_digest")
            or theorem.get("identity_digest")
            or theorem.get("statement_digest")
            or ""
        ),
        canonical_source_digest=str(
            payload.get("canonical_source_digest")
            or payload.get("source_digest")
            or theorem.get("canonical_source_digest")
            or ""
        ),
        theorem_equivalence_key=str(
            payload.get("theorem_equivalence_key")
            or theorem.get("equivalence_key")
            or ""
        ),
        freeze_root_cid=str(payload.get("freeze_root_cid") or ""),
        campaign_input_root_cid=str(payload.get("campaign_input_root_cid") or ""),
    )


def build_proof_attempt_trace(
    *,
    obligation: Mapping[str, Any],
    state: Mapping[str, Any],
    premises: Mapping[str, Any],
    proposals: Mapping[str, Any],
    model_tool_versions: Mapping[str, Any],
    parse_outcome: Mapping[str, Any],
    elaboration_outcome: Mapping[str, Any],
    prover_outcome: Mapping[str, Any],
    kernel_outcome: Mapping[str, Any],
    errors: Sequence[Any] = (),
    counterexamples: Sequence[Any] = (),
    timeout: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    request_id: str = "",
    schema: str = PROOF_ATTEMPT_TRACE_SCHEMA,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a strict J2 proof-attempt trace with a content-addressed identity."""

    payload = {
        "schema": schema,
        "contract_version": PROOF_ATTEMPT_TRACE_CONTRACT_VERSION,
        "obligation": dict(obligation),
        "state": dict(state),
        "premises": dict(premises),
        "proposals": dict(proposals),
        "model_tool_versions": dict(model_tool_versions),
        "parse_outcome": dict(parse_outcome),
        "elaboration_outcome": dict(elaboration_outcome),
        "prover_outcome": dict(prover_outcome),
        "kernel_outcome": dict(kernel_outcome),
        "errors": list(errors),
        "counterexamples": list(counterexamples),
        "timeout": dict(timeout or {}),
        "resources": dict(resources or {}),
        "request_id": request_id,
        "authoritative": False,
        "candidate_authority": False,
        "verified": False,
        "proof_success": False,
        "timeout_is_falsehood": False,
    }
    if bindings:
        payload["bindings"] = dict(bindings)
    return admit_proof_attempt_trace(payload)


def admit_proof_attempt_trace(
    payload: Mapping[str, Any],
    *,
    expected: ProofAttemptTraceExpectation | Mapping[str, Any] | None = None,
    store: ProofAttemptTraceStore | None = None,
    remember: bool = True,
) -> dict[str, Any]:
    """Admit one content-addressed attempt. Candidate authority stays false."""

    if not isinstance(payload, Mapping):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "proof-attempt trace must be an object",
        )
    try:
        raw = _json_object(payload, field_name="proof-attempt trace")
    except ValueError as exc:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            str(exc),
        ) from exc

    schema = _trace_text(raw.get("schema"), field_name="schema", required=True)
    if schema not in _PROOF_ATTEMPT_TRACE_SCHEMAS:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "unsupported proof-attempt trace schema",
        )
    contract_version = raw.get("contract_version", PROOF_ATTEMPT_TRACE_CONTRACT_VERSION)
    if contract_version != PROOF_ATTEMPT_TRACE_CONTRACT_VERSION:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "unsupported proof-attempt trace contract version",
        )

    missing = [name for name in J2_PROOF_ATTEMPT_FIELDS if name not in raw]
    if missing:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "proof-attempt trace is missing J2 fields: " + ", ".join(missing),
        )

    for name in (
        "authoritative",
        "candidate_authority",
        "verified",
        "proof_success",
        "timeout_is_falsehood",
    ):
        if raw.get(name, False) is not False:
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.MALFORMED,
                f"proof-attempt trace cannot claim {name}",
            )

    obligation = _trace_object(raw.get("obligation"), field_name="obligation")
    state = _trace_object(raw.get("state"), field_name="state")
    premises = _trace_object(raw.get("premises"), field_name="premises")
    proposals = _trace_object(raw.get("proposals"), field_name="proposals")
    model_tool_versions = _trace_object(
        raw.get("model_tool_versions"), field_name="model_tool_versions"
    )
    parse_outcome = _stage_outcome(raw.get("parse_outcome"), field_name="parse_outcome")
    elaboration_outcome = _stage_outcome(
        raw.get("elaboration_outcome"), field_name="elaboration_outcome"
    )
    prover_outcome = _stage_outcome(raw.get("prover_outcome"), field_name="prover_outcome")
    kernel_outcome = _stage_outcome(raw.get("kernel_outcome"), field_name="kernel_outcome")
    errors = _trace_array(raw.get("errors"), field_name="errors")
    counterexamples = _trace_array(raw.get("counterexamples"), field_name="counterexamples")
    timeout = _trace_object(raw.get("timeout"), field_name="timeout")
    resources = _trace_object(raw.get("resources"), field_name="resources")
    request_id = _trace_text(raw.get("request_id", ""), field_name="request_id")
    bindings = raw.get("bindings", {})
    if bindings in (None, ""):
        bindings = {}
    bindings = _trace_object(bindings, field_name="bindings")

    if timeout.get("is_falsehood", False) is not False:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "timeout cannot be recorded as falsehood",
        )
    timed_out = timeout.get("timed_out", False)
    if not isinstance(timed_out, bool):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "proof-attempt trace timeout.timed_out must be a boolean",
        )
    if timed_out:
        for name, outcome in (
            ("parse_outcome", parse_outcome),
            ("elaboration_outcome", elaboration_outcome),
            ("prover_outcome", prover_outcome),
            ("kernel_outcome", kernel_outcome),
        ):
            if outcome["status"] in _DEFINITIVE_FALSEHOOD_STATUSES:
                raise ProofAttemptTraceAdmissionError(
                    AttemptTraceFailureKind.MALFORMED,
                    f"timeout cannot be recorded as {name} falsehood",
                )

    obligation_id = _trace_text(
        obligation.get("obligation_id"),
        field_name="obligation.obligation_id",
    )
    theorem_id = _trace_text(obligation.get("theorem_id"), field_name="obligation.theorem_id")
    statement_digest = _trace_text(
        obligation.get("statement_digest"),
        field_name="obligation.statement_digest",
    )
    source_digest = _trace_text(
        obligation.get("canonical_source_digest"),
        field_name="obligation.canonical_source_digest",
    )
    equivalence_key = _trace_text(
        bindings.get("theorem_equivalence_key") or state.get("theorem_equivalence_key"),
        field_name="bindings.theorem_equivalence_key",
    )
    freeze_root_cid = _trace_text(
        bindings.get("freeze_root_cid"),
        field_name="bindings.freeze_root_cid",
    )
    campaign_input_root_cid = _trace_text(
        bindings.get("campaign_input_root_cid"),
        field_name="bindings.campaign_input_root_cid",
    )

    admitted = {
        "schema": schema,
        "contract_version": PROOF_ATTEMPT_TRACE_CONTRACT_VERSION,
        "obligation": obligation,
        "state": state,
        "premises": premises,
        "proposals": proposals,
        "model_tool_versions": model_tool_versions,
        "parse_outcome": parse_outcome,
        "elaboration_outcome": elaboration_outcome,
        "prover_outcome": prover_outcome,
        "kernel_outcome": kernel_outcome,
        "errors": errors,
        "counterexamples": counterexamples,
        "timeout": {
            **timeout,
            "is_falsehood": False,
            "timed_out": timed_out,
        },
        "resources": resources,
        "bindings": bindings,
        "request_id": request_id,
        "authoritative": False,
        "candidate_authority": False,
        "verified": False,
        "proof_success": False,
        "timeout_is_falsehood": False,
    }
    extra = sorted(
        set(raw)
        - set(admitted)
        - _IDENTITY_EXCLUDED_TRACE_KEYS
        - {"timeout_is_falsehood"}
    )
    if extra:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "proof-attempt trace has unknown fields: " + ", ".join(extra),
        )

    attempt_id = compute_proof_attempt_id(admitted)
    claimed_id = raw.get("attempt_id") or raw.get("content_id")
    if claimed_id and _trace_text(claimed_id, field_name="attempt_id") != attempt_id:
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "proof-attempt trace identity does not match J2 bindings",
        )
    admitted["attempt_id"] = attempt_id
    admitted["content_id"] = attempt_id

    expected_binding = expected
    if isinstance(expected, Mapping):
        expected_binding = ProofAttemptTraceExpectation(**dict(expected))
    if expected_binding is not None and not isinstance(
        expected_binding, ProofAttemptTraceExpectation
    ):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.MALFORMED,
            "proof-attempt expectation is invalid",
        )
    if expected_binding is not None:
        if expected_binding.request_id and request_id != expected_binding.request_id:
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.REPLAYED,
                "proof-attempt trace request_id does not match the current request",
            )
        if expected_binding.obligation_id and obligation_id != expected_binding.obligation_id:
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.WRONG_STATEMENT,
                "proof-attempt trace obligation does not match the current statement",
            )
        if expected_binding.theorem_id and theorem_id != expected_binding.theorem_id:
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.WRONG_STATEMENT,
                "proof-attempt trace theorem does not match the current statement",
            )
        if (
            expected_binding.statement_digest
            and statement_digest
            and statement_digest != expected_binding.statement_digest
        ):
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.WRONG_STATEMENT,
                "proof-attempt trace statement digest does not match the current statement",
            )
        if (
            expected_binding.theorem_equivalence_key
            and equivalence_key
            and equivalence_key != expected_binding.theorem_equivalence_key
        ):
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.WRONG_STATEMENT,
                "proof-attempt trace theorem equivalence key does not match",
            )
        if (
            expected_binding.canonical_source_digest
            and source_digest
            and source_digest != expected_binding.canonical_source_digest
        ):
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.STALE,
                "proof-attempt trace source digest is stale",
            )
        if (
            expected_binding.freeze_root_cid
            and freeze_root_cid
            and freeze_root_cid != expected_binding.freeze_root_cid
        ):
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.STALE,
                "proof-attempt trace freeze root is stale",
            )
        if (
            expected_binding.campaign_input_root_cid
            and campaign_input_root_cid
            and campaign_input_root_cid != expected_binding.campaign_input_root_cid
        ):
            raise ProofAttemptTraceAdmissionError(
                AttemptTraceFailureKind.STALE,
                "proof-attempt trace campaign input root is stale",
            )

    if store is not None and store.seen(attempt_id):
        raise ProofAttemptTraceAdmissionError(
            AttemptTraceFailureKind.REPLAYED,
            "proof-attempt trace has already been sealed",
        )
    if store is not None and remember:
        store.remember(attempt_id)
    return admitted


def _admit_embedded_attempt_trace(
    container: Mapping[str, Any],
    *,
    request: "ProviderRequest",
    store: ProofAttemptTraceStore | None,
) -> dict[str, Any] | None:
    raw = container.get("proof_attempt_trace")
    if raw is None:
        return None
    try:
        return admit_proof_attempt_trace(
            raw,
            expected=expectation_from_provider_request(request),
            store=store,
        )
    except ProofAttemptTraceAdmissionError as exc:
        raise ValueError(
            f"proof-attempt trace {exc.kind.value}: {exc.message}"
        ) from exc


@dataclass(frozen=True)
class ProviderRequest:
    """Strict versioned request envelope used for all six operations."""

    operation: ProofProviderOperation | str
    payload: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    network_allowed: bool = False
    deadline_unix_ms: int | None = None
    protocol_version: int = PROOF_PROVIDER_PROTOCOL_VERSION
    schema_version: str = PROOF_PROVIDER_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        try:
            operation = ProofProviderOperation(
                str(getattr(self.operation, "value", self.operation))
            )
        except ValueError as exc:
            raise ValueError("unsupported proof-provider operation") from exc
        if not isinstance(self.request_id, str):
            raise ValueError("proof-provider request_id must be a string")
        request_id = self.request_id.strip()
        if not request_id or len(request_id) > 128:
            raise ValueError("proof-provider request_id must contain 1-128 characters")
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version not in PROOF_PROVIDER_SUPPORTED_PROTOCOL_VERSIONS
        ):
            raise ValueError("unsupported proof-provider protocol version")
        if self.schema_version != PROOF_PROVIDER_REQUEST_SCHEMA:
            raise ValueError("unsupported proof-provider request schema")
        if not isinstance(self.network_allowed, bool):
            raise ValueError("network_allowed must be a boolean")
        if self.deadline_unix_ms is not None and (
            isinstance(self.deadline_unix_ms, bool)
            or not isinstance(self.deadline_unix_ms, int)
            or self.deadline_unix_ms < 0
        ):
            raise ValueError("deadline_unix_ms must be a non-negative integer or null")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self, "payload", _json_object(self.payload, field_name="request payload")
        )
        object.__setattr__(self, "resource_budget", _resource_budget(self.resource_budget))

    @property
    def expired(self) -> bool:
        return (
            self.deadline_unix_ms is not None and int(time.time() * 1000) >= self.deadline_unix_ms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "payload": dict(self.payload),
            "resource_budget": self.resource_budget.to_dict(),
            "network_allowed": self.network_allowed,
            "deadline_unix_ms": self.deadline_unix_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("proof-provider request must be an object")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            protocol_version=payload.get("protocol_version", 0),
            request_id=payload.get("request_id", ""),
            operation=payload.get("operation", ""),
            payload=payload.get("payload", {}),
            resource_budget=_resource_budget(payload.get("resource_budget", {})),
            network_allowed=payload.get("network_allowed", False),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
        )


@dataclass(frozen=True)
class ProviderResponse:
    """Correlated provider result; failures never carry a result payload."""

    request_id: str
    operation: ProofProviderOperation | str
    ok: bool
    result: Mapping[str, Any] | None = None
    error: ProviderFailure | Mapping[str, Any] | None = None
    provider_id: str = ""
    provider_version: str = ""
    duration_ms: int = 0
    protocol_version: int = PROOF_PROVIDER_PROTOCOL_VERSION
    schema_version: str = PROOF_PROVIDER_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        try:
            operation = ProofProviderOperation(
                str(getattr(self.operation, "value", self.operation))
            )
        except ValueError as exc:
            raise ValueError("unsupported proof-provider response operation") from exc
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version not in PROOF_PROVIDER_SUPPORTED_PROTOCOL_VERSIONS
        ):
            raise ValueError("unsupported proof-provider response protocol version")
        if self.schema_version != PROOF_PROVIDER_RESPONSE_SCHEMA:
            raise ValueError("unsupported proof-provider response schema")
        if not isinstance(self.ok, bool):
            raise ValueError("proof-provider response ok must be a boolean")
        if not isinstance(self.request_id, str):
            raise ValueError("proof-provider response request_id must be a string")
        request_id = self.request_id.strip()
        if not request_id or len(request_id) > 128:
            raise ValueError("proof-provider response request_id is invalid")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise ValueError("proof-provider duration_ms must be an integer")
        if self.duration_ms < 0:
            raise ValueError("proof-provider duration_ms must be non-negative")
        if not isinstance(self.provider_id, str) or not isinstance(self.provider_version, str):
            raise ValueError("proof-provider identity fields must be strings")
        provider_id = self.provider_id.strip()
        provider_version = self.provider_version.strip()

        result = (
            None if self.result is None else _json_object(self.result, field_name="response result")
        )
        if self.error is None:
            error = None
        elif isinstance(self.error, ProviderFailure):
            error = self.error
        else:
            error = ProviderFailure.from_dict(self.error)
        if self.ok and (result is None or error is not None):
            raise ValueError("successful proof-provider response needs only a result")
        if not self.ok and (error is None or result is not None):
            raise ValueError("failed proof-provider response needs only an error")

        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "provider_version", provider_version)

    @classmethod
    def success(
        cls,
        request: ProviderRequest,
        result: Mapping[str, Any],
        *,
        provider_id: str = "",
        provider_version: str = "",
        duration_ms: int = 0,
    ) -> "ProviderResponse":
        return cls(
            request_id=request.request_id,
            operation=request.operation,
            ok=True,
            result=result,
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=duration_ms,
        )

    @classmethod
    def failure(
        cls,
        request: ProviderRequest,
        code: ProviderFailureCode | str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        provider_id: str = "",
        provider_version: str = "",
        duration_ms: int = 0,
    ) -> "ProviderResponse":
        return cls(
            request_id=request.request_id,
            operation=request.operation,
            ok=False,
            error=ProviderFailure(
                code=code,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=duration_ms,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderResponse":
        if not isinstance(payload, Mapping):
            raise ValueError("proof-provider response must be an object")
        if "ok" not in payload:
            raise ValueError("proof-provider response is missing ok")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            protocol_version=payload.get("protocol_version", 0),
            request_id=payload.get("request_id", ""),
            operation=payload.get("operation", ""),
            ok=payload.get("ok"),
            result=payload.get("result"),
            error=payload.get("error"),
            provider_id=payload.get("provider_id", ""),
            provider_version=payload.get("provider_version", ""),
            duration_ms=payload.get("duration_ms", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "ok": self.ok,
            "result": None if self.result is None else dict(self.result),
            "error": None if self.error is None else self.error.to_dict(),
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "duration_ms": self.duration_ms,
        }

    def require_result(self) -> Mapping[str, Any]:
        if self.ok:
            assert self.result is not None
            return self.result
        assert self.error is not None
        raise ProviderInvocationError(
            self.error.code,
            self.error.message,
            retryable=self.error.retryable,
            details=self.error.details,
        )


@runtime_checkable
class ProofProvider(Protocol):
    """Structural protocol implemented by optional proof providers."""

    provider_id: str
    provider_version: str
    protocol_version: int

    def capability(self, request: ProviderRequest) -> Mapping[str, Any] | ProviderResponse: ...

    def translate(self, request: ProviderRequest) -> Mapping[str, Any] | ProviderResponse: ...

    def prove(self, request: ProviderRequest) -> Mapping[str, Any] | ProviderResponse: ...

    def reconstruct(self, request: ProviderRequest) -> Mapping[str, Any] | ProviderResponse: ...

    def verify(self, request: ProviderRequest) -> Mapping[str, Any] | ProviderResponse: ...

    def attest(self, request: ProviderRequest) -> Mapping[str, Any] | ProviderResponse: ...


@dataclass(frozen=True)
class ProviderInvocationConfig:
    """Host-enforced limits for one provider adapter."""

    timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS
    max_request_bytes: int = DEFAULT_PROVIDER_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_PROVIDER_MAX_RESPONSE_BYTES
    memory_bytes: int = DEFAULT_PROVIDER_MEMORY_BYTES
    cpu_time_seconds: int = DEFAULT_PROVIDER_CPU_TIME_SECONDS
    max_processes: int = DEFAULT_PROVIDER_MAX_PROCESSES
    allow_network: bool = False
    inherit_environment: tuple[str, ...] = (
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SYSTEMROOT",
        "WINDIR",
    )
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("provider timeout_seconds must be finite and positive")
        for name in (
            "max_request_bytes",
            "max_response_bytes",
            "memory_bytes",
            "cpu_time_seconds",
            "max_processes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"provider {name} must be a positive integer")
        if not isinstance(self.allow_network, bool):
            raise ValueError("allow_network must be a boolean")
        inherited = tuple(str(name).strip() for name in self.inherit_environment)
        if any(not name for name in inherited):
            raise ValueError("inherit_environment names must not be empty")
        if not isinstance(self.environment, Mapping):
            raise ValueError("provider environment must be a mapping")
        environment = {str(key): str(value) for key, value in self.environment.items() if str(key)}
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "inherit_environment", inherited)
        object.__setattr__(self, "environment", environment)


def _request_timeout(request: ProviderRequest, config: ProviderInvocationConfig) -> float:
    limits = [config.timeout_seconds]
    if request.resource_budget.wall_time_ms:
        limits.append(request.resource_budget.wall_time_ms / 1000.0)
    if request.deadline_unix_ms is not None:
        limits.append(max(0.0, request.deadline_unix_ms / 1000.0 - time.time()))
    return min(limits)


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _exception_failure(
    exc: BaseException,
) -> tuple[ProviderFailureCode, str, bool]:
    if isinstance(exc, ProofProviderError):
        return exc.failure.code, exc.failure.message, exc.failure.retryable
    if isinstance(exc, (MemoryError, RecursionError)):
        return (
            ProviderFailureCode.RESOURCE_EXHAUSTED,
            f"proof provider exhausted a process resource ({type(exc).__name__})",
            False,
        )
    if isinstance(exc, OSError) and exc.errno in {
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
        errno.ECONNREFUSED,
    }:
        return (
            ProviderFailureCode.NETWORK_DENIED,
            f"proof provider network operation failed ({type(exc).__name__})",
            False,
        )
    return (
        ProviderFailureCode.PROVIDER_ERROR,
        f"proof provider raised {type(exc).__name__}: {str(exc)[:512]}",
        False,
    )


def _exception_details(exc: BaseException) -> Mapping[str, Any]:
    """Keep typed provider failure metadata across every execution boundary."""

    return exc.failure.details if isinstance(exc, ProofProviderError) else {}


def _with_admitted_attempt_trace(
    request: ProviderRequest,
    payload: Mapping[str, Any],
    *,
    store: ProofAttemptTraceStore | None,
) -> dict[str, Any]:
    admitted = _admit_embedded_attempt_trace(payload, request=request, store=store)
    if admitted is None:
        return dict(payload)
    rewritten = dict(payload)
    rewritten["proof_attempt_trace"] = admitted
    return rewritten


def _normalize_provider_result(
    request: ProviderRequest,
    raw_result: Any,
    *,
    provider_id: str,
    provider_version: str,
    duration_ms: int,
    attempt_trace_store: ProofAttemptTraceStore | None = None,
) -> ProviderResponse:
    if isinstance(raw_result, ProviderResponse):
        if (
            raw_result.request_id != request.request_id
            or raw_result.operation is not request.operation
        ):
            raise ValueError("provider returned a response for another request")
        result = raw_result.result
        error = raw_result.error
        if result is not None:
            result = _with_admitted_attempt_trace(
                request, result, store=attempt_trace_store
            )
        if error is not None:
            details = dict(error.details)
            admitted = _admit_embedded_attempt_trace(
                details, request=request, store=attempt_trace_store
            )
            if admitted is not None:
                details["proof_attempt_trace"] = admitted
                error = ProviderFailure(
                    code=error.code,
                    message=error.message,
                    retryable=error.retryable,
                    details=details,
                )
        return ProviderResponse(
            request_id=raw_result.request_id,
            operation=raw_result.operation,
            ok=raw_result.ok,
            result=result,
            error=error,
            provider_id=raw_result.provider_id or provider_id,
            provider_version=raw_result.provider_version or provider_version,
            duration_ms=duration_ms,
            protocol_version=raw_result.protocol_version,
            schema_version=raw_result.schema_version,
        )
    if not isinstance(raw_result, Mapping):
        raise ValueError("provider operation must return an object or ProviderResponse")
    return ProviderResponse.success(
        request,
        _with_admitted_attempt_trace(
            request,
            _json_object(raw_result, field_name="provider result"),
            store=attempt_trace_store,
        ),
        provider_id=provider_id,
        provider_version=provider_version,
        duration_ms=duration_ms,
    )


def dispatch_provider_request(
    provider: Any,
    request: ProviderRequest,
    *,
    attempt_trace_store: ProofAttemptTraceStore | None = None,
) -> ProviderResponse:
    """Invoke one provider method synchronously and normalize all failures."""

    started = time.monotonic()
    provider_id = str(getattr(provider, "provider_id", "")).strip()
    provider_version = str(getattr(provider, "provider_version", "")).strip()
    protocol_version = getattr(provider, "protocol_version", PROOF_PROVIDER_PROTOCOL_VERSION)
    if protocol_version != request.protocol_version:
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.PROTOCOL_ERROR,
            "provider does not support the requested protocol version",
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=_duration_ms(started),
        )
    if request.expired:
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.TIMED_OUT,
            "proof-provider request deadline has expired",
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=_duration_ms(started),
        )
    method = getattr(provider, request.operation.value, None)
    if not callable(method):
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.UNSUPPORTED,
            f"provider does not support {request.operation.value!r}",
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=_duration_ms(started),
        )
    try:
        raw_result = method(request)
    except BaseException as exc:
        code, message, retryable = _exception_failure(exc)
        details = dict(_exception_details(exc))
        try:
            admitted = _admit_embedded_attempt_trace(
                details, request=request, store=attempt_trace_store
            )
        except ValueError as admit_exc:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.MALFORMED_RESPONSE,
                f"provider returned a malformed response: {str(admit_exc)[:512]}",
                provider_id=provider_id,
                provider_version=provider_version,
                duration_ms=_duration_ms(started),
            )
        if admitted is not None:
            details["proof_attempt_trace"] = admitted
        return ProviderResponse.failure(
            request,
            code,
            message,
            retryable=retryable,
            details=details,
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=_duration_ms(started),
        )
    try:
        return _normalize_provider_result(
            request,
            raw_result,
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=_duration_ms(started),
            attempt_trace_store=attempt_trace_store,
        )
    except (TypeError, ValueError) as exc:
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.MALFORMED_RESPONSE,
            f"provider returned a malformed response: {str(exc)[:512]}",
            provider_id=provider_id,
            provider_version=provider_version,
            duration_ms=_duration_ms(started),
        )


class ProviderClient:
    """Shared convenience operations for provider execution adapters."""

    def invoke(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError

    def call(
        self,
        operation: ProofProviderOperation | str,
        payload: Mapping[str, Any] | None = None,
        *,
        resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
        network_allowed: bool = False,
        deadline_unix_ms: int | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ProviderResponse:
        return self.invoke(
            ProviderRequest(
                operation=operation,
                payload=payload or {},
                resource_budget=_resource_budget(resource_budget),
                network_allowed=network_allowed,
                deadline_unix_ms=deadline_unix_ms,
            ),
            cancellation=cancellation,
        )

    def capability(
        self, payload: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> ProviderResponse:
        return self.call(ProofProviderOperation.CAPABILITY, payload, **kwargs)

    def translate(self, payload: Mapping[str, Any], **kwargs: Any) -> ProviderResponse:
        return self.call(ProofProviderOperation.TRANSLATE, payload, **kwargs)

    def prove(self, payload: Mapping[str, Any], **kwargs: Any) -> ProviderResponse:
        return self.call(ProofProviderOperation.PROVE, payload, **kwargs)

    def reconstruct(self, payload: Mapping[str, Any], **kwargs: Any) -> ProviderResponse:
        return self.call(ProofProviderOperation.RECONSTRUCT, payload, **kwargs)

    def verify(self, payload: Mapping[str, Any], **kwargs: Any) -> ProviderResponse:
        return self.call(ProofProviderOperation.VERIFY, payload, **kwargs)

    def attest(self, payload: Mapping[str, Any], **kwargs: Any) -> ProviderResponse:
        return self.call(ProofProviderOperation.ATTEST, payload, **kwargs)


class InProcessProofProvider(ProviderClient):
    """Adapter for a provider object, resolved only on its first invocation.

    Calls run on daemon threads.  A timed-out provider cannot be forcibly
    stopped in-process, so callers needing strong CPU/memory/process isolation
    should select :class:`SubprocessProofProvider`.
    """

    def __init__(
        self,
        provider: Any | Callable[[], Any],
        *,
        config: ProviderInvocationConfig | None = None,
        lazy: bool = False,
        expected_capability: ProofProviderCapability | None = None,
    ) -> None:
        self._provider_or_loader = provider
        self._provider: Any | None = None if lazy else _materialize_provider(provider)
        self._config = config or ProviderInvocationConfig()
        self._expected_capability = expected_capability
        self._lock = threading.Lock()

    def _resolve(self) -> Any:
        if self._provider is not None:
            return self._provider
        with self._lock:
            if self._provider is None:
                loader = self._provider_or_loader
                if not callable(loader):
                    raise TypeError("lazy proof-provider loader is not callable")
                self._provider = _materialize_provider(loader())
        return self._provider

    def invoke(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ProviderResponse:
        started = time.monotonic()
        if cancellation is not None and cancellation.cancelled:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.CANCELLED,
                "proof-provider request was cancelled before invocation",
            )
        if request.expired or _request_timeout(request, self._config) <= 0:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.TIMED_OUT,
                "proof-provider request deadline has expired",
            )
        capability = self._expected_capability
        if capability is not None:
            if not capability.supports(
                request.operation,
                protocol_version=request.protocol_version,
                isolation=ProofProviderIsolation.IN_PROCESS,
            ):
                return ProviderResponse.failure(
                    request,
                    ProviderFailureCode.UNSUPPORTED,
                    "provider capability does not support this in-process operation",
                )
            if capability.network_access_required and not (
                self._config.allow_network and request.network_allowed
            ):
                return ProviderResponse.failure(
                    request,
                    ProviderFailureCode.NETWORK_DENIED,
                    "provider requires network access but the request policy denies it",
                )

        results: queue.Queue[ProviderResponse] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                provider = self._resolve()
                response = dispatch_provider_request(provider, request)
            except BaseException as exc:
                code, message, retryable = _exception_failure(exc)
                response = ProviderResponse.failure(
                    request,
                    code,
                    message,
                    retryable=retryable,
                    details=_exception_details(exc),
                )
            try:
                results.put_nowait(response)
            except queue.Full:
                pass

        worker = threading.Thread(
            target=run,
            name=f"proof-provider-{request.operation.value}-{request.request_id[:8]}",
            daemon=True,
        )
        worker.start()
        timeout = _request_timeout(request, self._config)
        end = time.monotonic() + timeout
        while True:
            if cancellation is not None and cancellation.cancelled:
                return ProviderResponse.failure(
                    request,
                    ProviderFailureCode.CANCELLED,
                    "proof-provider request was cancelled",
                    duration_ms=_duration_ms(started),
                )
            remaining = end - time.monotonic()
            if remaining <= 0:
                return ProviderResponse.failure(
                    request,
                    ProviderFailureCode.TIMED_OUT,
                    "in-process proof provider exceeded its wall-time limit",
                    retryable=True,
                    duration_ms=_duration_ms(started),
                )
            try:
                response = results.get(timeout=min(remaining, 0.05))
                encoded = canonical_json(response.to_dict()).encode("utf-8")
                if len(encoded) > self._config.max_response_bytes:
                    return ProviderResponse.failure(
                        request,
                        ProviderFailureCode.RESOURCE_EXHAUSTED,
                        "in-process proof-provider response exceeded the output limit",
                        duration_ms=_duration_ms(started),
                    )
                return response
            except queue.Empty:
                continue


def _minimum_positive(default: int, requested: int) -> int:
    return min(default, requested) if requested > 0 else default


def _resource_preexec(
    request: ProviderRequest,
    config: ProviderInvocationConfig,
) -> Callable[[], None] | None:
    if os.name != "posix":
        return None

    memory = _minimum_positive(config.memory_bytes, request.resource_budget.memory_bytes)
    cpu_ms = request.resource_budget.cpu_time_ms
    requested_cpu_seconds = max(1, math.ceil(cpu_ms / 1000.0)) if cpu_ms else 0
    cpu_seconds = _minimum_positive(config.cpu_time_seconds, requested_cpu_seconds)
    processes = _minimum_positive(config.max_processes, request.resource_budget.max_processes)
    file_bytes = _minimum_positive(
        config.max_response_bytes,
        request.resource_budget.disk_bytes or request.resource_budget.max_output_bytes,
    )

    def apply_limits() -> None:
        import resource

        for resource_name, limit in (
            (resource.RLIMIT_AS, memory),
            (resource.RLIMIT_CPU, cpu_seconds),
            (resource.RLIMIT_FSIZE, file_bytes),
        ):
            try:
                resource.setrlimit(resource_name, (limit, limit))
            except (OSError, ValueError):
                # Some platforms expose but do not permit a particular limit.
                # Other enforced bounds remain active and the request declares
                # every limit to the provider.
                pass
        if hasattr(resource, "RLIMIT_NPROC"):
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
            except (OSError, ValueError):
                pass

    return apply_limits


def _subprocess_environment(
    request: ProviderRequest,
    config: ProviderInvocationConfig,
) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in config.inherit_environment if name in os.environ
    }
    environment.update(config.environment)
    environment.update(
        {
            "IPFS_ACCELERATE_PROOF_PROVIDER_PROTOCOL": str(request.protocol_version),
            "IPFS_ACCELERATE_PROOF_PROVIDER_NETWORK_ALLOWED": (
                "1" if config.allow_network and request.network_allowed else "0"
            ),
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    if not (config.allow_network and request.network_allowed):
        for name in tuple(environment):
            if name.lower() in {
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "ftp_proxy",
            }:
                environment.pop(name, None)
    return environment


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


class SubprocessProofProvider(ProviderClient):
    """Invoke a provider command through a bounded one-request JSON protocol."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        config: ProviderInvocationConfig | None = None,
        expected_capability: ProofProviderCapability | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError("proof-provider command must be a non-empty argument sequence")
        normalized = tuple(str(argument) for argument in command)
        if any(not argument for argument in normalized):
            raise ValueError("proof-provider command arguments must not be empty")
        self.command = normalized
        self.config = config or ProviderInvocationConfig()
        self.expected_capability = expected_capability
        self.cwd = None if cwd is None else os.fspath(cwd)

    def _preflight(self, request: ProviderRequest) -> ProviderResponse | None:
        capability = self.expected_capability
        if capability is None:
            return None
        if not capability.supports(
            request.operation,
            protocol_version=request.protocol_version,
            isolation=ProofProviderIsolation.SUBPROCESS,
        ):
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.UNSUPPORTED,
                "provider capability does not support this subprocess operation",
                provider_id=capability.provider_id,
                provider_version=capability.provider_version,
            )
        if capability.network_access_required and not (
            self.config.allow_network and request.network_allowed
        ):
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.NETWORK_DENIED,
                "provider requires network access but subprocess policy denies it",
                provider_id=capability.provider_id,
                provider_version=capability.provider_version,
            )
        return None

    def invoke(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ProviderResponse:
        started = time.monotonic()
        if cancellation is not None and cancellation.cancelled:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.CANCELLED,
                "proof-provider request was cancelled before subprocess start",
            )
        preflight = self._preflight(request)
        if preflight is not None:
            return preflight
        timeout = _request_timeout(request, self.config)
        if request.expired or timeout <= 0:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.TIMED_OUT,
                "proof-provider request deadline has expired",
            )
        request_bytes = canonical_json(request.to_dict()).encode("utf-8") + b"\n"
        request_limit = _minimum_positive(
            self.config.max_request_bytes,
            request.resource_budget.max_output_bytes,
        )
        if len(request_bytes) > request_limit:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "proof-provider request exceeded the configured input limit",
                details={"limit_bytes": request_limit},
            )

        output_limit = _minimum_positive(
            self.config.max_response_bytes,
            request.resource_budget.max_output_bytes,
        )
        process: subprocess.Popen[Any] | None = None
        try:
            with (
                tempfile.TemporaryFile(mode="w+b") as stdout_file,
                tempfile.TemporaryFile(mode="w+b") as stderr_file,
            ):
                process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd=self.cwd,
                    env=_subprocess_environment(request, self.config),
                    shell=False,
                    close_fds=True,
                    start_new_session=(os.name == "posix"),
                    preexec_fn=_resource_preexec(request, self.config),
                )
                assert process.stdin is not None
                process.stdin.write(request_bytes)
                process.stdin.close()
                end = time.monotonic() + timeout
                while process.poll() is None:
                    if cancellation is not None and cancellation.cancelled:
                        _terminate_process(process)
                        return ProviderResponse.failure(
                            request,
                            ProviderFailureCode.CANCELLED,
                            "proof-provider subprocess was cancelled",
                            duration_ms=_duration_ms(started),
                        )
                    if time.monotonic() >= end:
                        _terminate_process(process)
                        return ProviderResponse.failure(
                            request,
                            ProviderFailureCode.TIMED_OUT,
                            "proof-provider subprocess exceeded its wall-time limit",
                            retryable=True,
                            duration_ms=_duration_ms(started),
                        )
                    time.sleep(min(0.02, max(0.001, end - time.monotonic())))

                stdout_file.seek(0)
                response_bytes = stdout_file.read(output_limit + 1)
                stderr_file.seek(0)
                stderr_bytes = stderr_file.read(min(output_limit, 16 * 1024) + 1)
                if len(response_bytes) > output_limit or len(stderr_bytes) > 16 * 1024:
                    return ProviderResponse.failure(
                        request,
                        ProviderFailureCode.RESOURCE_EXHAUSTED,
                        "proof-provider subprocess exceeded its output limit",
                        details={"limit_bytes": output_limit},
                        duration_ms=_duration_ms(started),
                    )
                if process.returncode != 0:
                    resource_signals = {
                        -getattr(signal, name)
                        for name in ("SIGXCPU", "SIGXFSZ", "SIGKILL", "SIGSEGV")
                        if hasattr(signal, name)
                    }
                    stderr_text = stderr_bytes.decode("utf-8", "replace")[:2048]
                    file_limit_hit = (
                        len(response_bytes) >= output_limit
                        or "File too large" in stderr_text
                        or f"Errno {errno.EFBIG}" in stderr_text
                    )
                    code = (
                        ProviderFailureCode.RESOURCE_EXHAUSTED
                        if process.returncode in resource_signals or file_limit_hit
                        else ProviderFailureCode.PROVIDER_ERROR
                    )
                    return ProviderResponse.failure(
                        request,
                        code,
                        "proof-provider subprocess exited unsuccessfully",
                        details={
                            "returncode": process.returncode,
                            "stderr": stderr_text,
                        },
                        duration_ms=_duration_ms(started),
                    )
        except FileNotFoundError:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.UNAVAILABLE,
                "proof-provider executable was not found",
                duration_ms=_duration_ms(started),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if process is not None:
                _terminate_process(process)
            code, message, retryable = _exception_failure(exc)
            return ProviderResponse.failure(
                request,
                code,
                message,
                retryable=retryable,
                details=_exception_details(exc),
                duration_ms=_duration_ms(started),
            )

        try:
            text = response_bytes.decode("utf-8")
            if not text.strip():
                raise ValueError("proof-provider returned an empty response")
            decoder = json.JSONDecoder(
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {value}")
                ),
            )
            decoded, end_index = decoder.raw_decode(text)
            if text[end_index:].strip():
                raise ValueError("proof-provider returned more than one JSON value")
            if not isinstance(decoded, Mapping):
                raise ValueError("proof-provider response must be an object")
            response = ProviderResponse.from_dict(decoded)
            if (
                response.request_id != request.request_id
                or response.operation is not request.operation
            ):
                raise ValueError("proof-provider response correlation does not match request")
            return ProviderResponse(
                request_id=response.request_id,
                operation=response.operation,
                ok=response.ok,
                result=response.result,
                error=response.error,
                provider_id=response.provider_id,
                provider_version=response.provider_version,
                duration_ms=_duration_ms(started),
                protocol_version=response.protocol_version,
                schema_version=response.schema_version,
            )
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.MALFORMED_RESPONSE,
                f"proof-provider returned a malformed response: {str(exc)[:512]}",
                duration_ms=_duration_ms(started),
            )


ProviderLoader = Callable[[], Any]


def _materialize_provider(value: Any) -> Any:
    """Normalize the common entry-point forms: instance, class, or factory."""

    if isinstance(value, type):
        value = value()
    elif callable(value) and not any(
        callable(getattr(value, operation.value, None)) for operation in ProofProviderOperation
    ):
        value = value()
    if value is None:
        raise TypeError("proof-provider loader returned null")
    return value


def load_provider_reference(reference: str) -> Any:
    """Load ``module:attribute`` only when an in-process provider is needed."""

    module_name, separator, attribute_path = str(reference).strip().partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("proof-provider reference must use module:attribute syntax")
    module = importlib.import_module(module_name)
    value: Any = module
    for component in attribute_path.split("."):
        if not component:
            raise ValueError("proof-provider attribute path is malformed")
        value = getattr(value, component)
    return _materialize_provider(value)


@dataclass(frozen=True)
class ProviderRegistration:
    """A discovered provider loader that has not imported provider code."""

    provider_id: str
    loader: ProviderLoader
    source: str

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id).strip()
        source = str(self.source).strip()
        if not provider_id or not source:
            raise ValueError("provider registration id and source are required")
        if not callable(self.loader):
            raise ValueError("provider registration loader must be callable")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "source", source)

    def client(self, *, config: ProviderInvocationConfig | None = None) -> InProcessProofProvider:
        return InProcessProofProvider(self.loader, config=config, lazy=True)


class ProofProviderRegistry:
    """Thread-safe registry with lazy environment and entry-point discovery."""

    def __init__(
        self,
        *,
        entry_point_group: str = PROOF_PROVIDER_ENTRY_POINT_GROUP,
        environ: Mapping[str, str] | None = None,
        entry_points: Callable[[], Any] | None = None,
    ) -> None:
        self.entry_point_group = str(entry_point_group)
        self._environ = os.environ if environ is None else environ
        self._entry_points = entry_points or importlib.metadata.entry_points
        self._registrations: dict[str, ProviderRegistration] = {}
        self._discovered = False
        self._lock = threading.RLock()

    def register(
        self,
        provider_id: str,
        provider: Any | ProviderLoader | str,
        *,
        source: str = "explicit",
        replace: bool = False,
    ) -> ProviderRegistration:
        if isinstance(provider, str):

            def loader(reference: str = provider) -> Any:
                return load_provider_reference(reference)

            registration_source = source if source != "explicit" else provider
        elif callable(provider) and not any(
            callable(getattr(provider, operation.value, None))
            for operation in ProofProviderOperation
        ):
            loader = provider
            registration_source = source
        else:

            def loader(value: Any = provider) -> Any:
                return value

            registration_source = source
        registration = ProviderRegistration(
            provider_id=provider_id,
            loader=loader,
            source=registration_source,
        )
        with self._lock:
            if registration.provider_id in self._registrations and not replace:
                raise ValueError(
                    f"proof provider {registration.provider_id!r} is already registered"
                )
            self._registrations[registration.provider_id] = registration
        return registration

    def _select_entry_points(self) -> tuple[Any, ...]:
        discovered = self._entry_points()
        if hasattr(discovered, "select"):
            return tuple(discovered.select(group=self.entry_point_group))
        if isinstance(discovered, Mapping):
            return tuple(discovered.get(self.entry_point_group, ()))
        return tuple(
            item for item in discovered if getattr(item, "group", None) == self.entry_point_group
        )

    def discover(self, *, force_refresh: bool = False) -> tuple[ProviderRegistration, ...]:
        """Return registrations without loading an entry-point provider."""

        with self._lock:
            if self._discovered and not force_refresh:
                return tuple(self._registrations[key] for key in sorted(self._registrations))
            environment_reference = str(self._environ.get(PROOF_PROVIDER_ENVIRONMENT, "")).strip()
            if environment_reference:
                provider_id, separator, reference = environment_reference.partition("=")
                if not separator:
                    reference = provider_id
                    provider_id = "environment"
                if provider_id.strip() and reference.strip():
                    self.register(
                        provider_id.strip(),
                        reference.strip(),
                        source=f"environment:{PROOF_PROVIDER_ENVIRONMENT}",
                        replace=True,
                    )
            try:
                entry_points = self._select_entry_points()
            except Exception:
                entry_points = ()
            for entry_point in entry_points:
                provider_id = str(getattr(entry_point, "name", "")).strip()
                if not provider_id:
                    continue
                self.register(
                    provider_id,
                    lambda item=entry_point: item.load(),
                    source=f"entry-point:{self.entry_point_group}",
                    replace=True,
                )
            self._discovered = True
            return tuple(self._registrations[key] for key in sorted(self._registrations))

    def get(self, provider_id: str) -> ProviderRegistration | None:
        self.discover()
        with self._lock:
            return self._registrations.get(str(provider_id))

    def client(
        self,
        provider_id: str,
        *,
        config: ProviderInvocationConfig | None = None,
    ) -> InProcessProofProvider | None:
        registration = self.get(provider_id)
        return None if registration is None else registration.client(config=config)

    def clear(self) -> None:
        with self._lock:
            self._registrations.clear()
            self._discovered = False


_DEFAULT_PROVIDER_REGISTRY = ProofProviderRegistry()


def register_proof_provider(
    provider_id: str,
    provider: Any | ProviderLoader | str,
    *,
    source: str = "explicit",
    replace: bool = False,
) -> ProviderRegistration:
    return _DEFAULT_PROVIDER_REGISTRY.register(
        provider_id, provider, source=source, replace=replace
    )


def discover_proof_providers(*, force_refresh: bool = False) -> tuple[ProviderRegistration, ...]:
    """Discover optional providers without importing their implementation."""

    return _DEFAULT_PROVIDER_REGISTRY.discover(force_refresh=force_refresh)


def get_proof_provider(
    provider_id: str,
    *,
    config: ProviderInvocationConfig | None = None,
) -> InProcessProofProvider | None:
    """Return a lazy client, or ``None`` when the provider is not installed."""

    return _DEFAULT_PROVIDER_REGISTRY.client(provider_id, config=config)


def clear_proof_provider_registry() -> None:
    _DEFAULT_PROVIDER_REGISTRY.clear()


def serve_provider_json(
    provider: Any,
    *,
    input_stream: IO[bytes] | None = None,
    output_stream: IO[bytes] | None = None,
    max_request_bytes: int = DEFAULT_PROVIDER_MAX_REQUEST_BYTES,
) -> int:
    """Serve one protocol request for a subprocess provider entry point.

    The helper always attempts to emit a versioned response.  If no valid
    request id and operation can be recovered from malformed input, it exits
    non-zero instead, which the parent maps to a closed provider failure.
    """

    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    source = input_stream if input_stream is not None else getattr(os.sys.stdin, "buffer")
    sink = output_stream if output_stream is not None else getattr(os.sys.stdout, "buffer")
    raw = source.read(max_request_bytes + 1)
    if len(raw) > max_request_bytes:
        return 2
    try:
        decoded = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        return 2
    try:
        request = ProviderRequest.from_dict(decoded)
        response = dispatch_provider_request(provider, request)
    except (ValueError, TypeError) as exc:
        if not isinstance(decoded, Mapping):
            return 2
        request_id = decoded.get("request_id")
        operation = decoded.get("operation")
        if not isinstance(request_id, str) or not request_id:
            return 2
        try:
            correlated_request = ProviderRequest(
                request_id=request_id,
                operation=operation,
            )
        except (ValueError, TypeError):
            return 2
        response = ProviderResponse.failure(
            correlated_request,
            ProviderFailureCode.MALFORMED_REQUEST,
            f"proof-provider request is malformed: {str(exc)[:512]}",
        )
    encoded = canonical_json(response.to_dict()).encode("utf-8") + b"\n"
    sink.write(encoded)
    sink.flush()
    return 0


# Compatibility spellings for consumers that use the transport terminology.
ProviderError = ProviderFailure
ProofProviderRequest = ProviderRequest
ProofProviderResponse = ProviderResponse
InProcessProvider = InProcessProofProvider
SubprocessProvider = SubprocessProofProvider
run_provider_stdio = serve_provider_json


def project_provider_request_to_canonical(
    request: "ProviderRequest",
    *,
    cancellation: "CancellationToken | None" = None,
) -> Any:
    """Convert a supervisor provider request to the datasets logic-provider wire type."""

    from ..canonical_logic_adapter import project_provider_request

    return project_provider_request(request, cancellation=cancellation)


def project_resource_budget_to_canonical(
    budget: ResourceBudget | Mapping[str, Any],
) -> dict[str, Any]:
    """Project a supervisor resource budget onto the canonical provider budget shape."""

    from ..canonical_logic_adapter import project_resource_budget

    return project_resource_budget(budget)


__all__ = [
    "DEFAULT_PROVIDER_CPU_TIME_SECONDS",
    "DEFAULT_PROVIDER_MAX_PROCESSES",
    "DEFAULT_PROVIDER_MAX_REQUEST_BYTES",
    "DEFAULT_PROVIDER_MAX_RESPONSE_BYTES",
    "DEFAULT_PROVIDER_MEMORY_BYTES",
    "DEFAULT_PROVIDER_TIMEOUT_SECONDS",
    "PROOF_PROVIDER_ENTRY_POINT_GROUP",
    "PROOF_PROVIDER_ENVIRONMENT",
    "PROOF_PROVIDER_PROTOCOL_VERSION",
    "PROOF_PROVIDER_REQUEST_SCHEMA",
    "PROOF_PROVIDER_RESPONSE_SCHEMA",
    "PROOF_PROVIDER_SUPPORTED_PROTOCOL_VERSIONS",
    "J2_PROOF_ATTEMPT_FIELDS",
    "LEANSTRAL_PROOF_ATTEMPT_TRACE_SCHEMA",
    "PROOF_ATTEMPT_TRACE_CONTRACT_VERSION",
    "PROOF_ATTEMPT_TRACE_SCHEMA",
    "AttemptTraceFailureKind",
    "CancellationToken",
    "ProofAttemptTraceAdmissionError",
    "ProofAttemptTraceExpectation",
    "ProofAttemptTraceStore",
    "admit_proof_attempt_trace",
    "build_proof_attempt_trace",
    "compute_proof_attempt_id",
    "expectation_from_provider_request",
    "InProcessProofProvider",
    "InProcessProvider",
    "NetworkAccessDenied",
    "ProofProvider",
    "ProofProviderError",
    "ProofProviderRegistry",
    "ProofProviderRequest",
    "ProofProviderResponse",
    "ProviderClient",
    "ProviderError",
    "ProviderFailure",
    "ProviderFailureCode",
    "ProviderInvocationConfig",
    "ProviderInvocationError",
    "ProviderRegistration",
    "ProviderRequest",
    "ProviderResponse",
    "SubprocessProofProvider",
    "SubprocessProvider",
    "clear_proof_provider_registry",
    "discover_proof_providers",
    "dispatch_provider_request",
    "get_proof_provider",
    "load_provider_reference",
    "project_provider_request_to_canonical",
    "project_resource_budget_to_canonical",
    "register_proof_provider",
    "run_provider_stdio",
    "serve_provider_json",
]
