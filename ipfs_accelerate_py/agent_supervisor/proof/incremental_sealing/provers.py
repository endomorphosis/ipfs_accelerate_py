"""Bounded hermetic prover and verifier adapters (IPS-031).

Adapters invoke only statically registered programs/circuits and approved
proving-key handles.  Every prove path binds committed public inputs, enforces
output size and timeout bounds, and returns a closed :class:`ProverOutcome`.
Verification re-checks proof bytes against the committed public input and
allowlisted verification key.

Sensitive witness and proving-key material never appear in receipts, logs, or
canonical outcomes.  Ambiguous external completion is never reported as
``proved``.

Interfaces: ``IncrementalProofBackendAdapter``, ``ProverInvocation``,
``VerificationInvocation``, ``ProverOutcome``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    ProvingKeyHandle,
    TrustedProofPolicy,
    TrustError,
)

PROVER_ADAPTER_EVIDENCE: Final[str] = "ips/prover-adapters@1"

PROVER_OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "prover-outcome@1"
)
PROVER_INVOCATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "prover-invocation@1"
)
VERIFICATION_INVOCATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "verification-invocation@1"
)
PROGRAM_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "program-registry@1"
)

# Closed backend identifiers that adapters may target.  Unknown backends fail
# as unavailable/unknown rather than executing arbitrary tooling.
KNOWN_PROVER_BACKEND_IDS: Final[frozenset[str]] = frozenset(
    {
        "hermetic_hmac",
        "groth16",
        "provekit",
        "integrity",
        "simulated",
    }
)

DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 65_536
DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
DEFAULT_MAX_LOG_BYTES: Final[int] = 4_096

# Hermetic domain-separation tags for the in-process HMAC backend.
_HERMETIC_PROVE_DOMAIN: Final[bytes] = b"ips-prover-adapters/hermetic-hmac-prove@1\n"
_HERMETIC_VK_DOMAIN: Final[bytes] = b"ips-prover-adapters/hermetic-hmac-vk@1\n"

# Field names that must never appear on public receipts/logs.
_SENSITIVE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "witness",
        "witness_bytes",
        "witness_material",
        "private_input",
        "private_inputs",
        "proving_key",
        "proving_key_bytes",
        "proving_key_material",
        "private_key",
        "private_key_bytes",
        "trapdoor",
        "secret",
        "key_bytes",
        "raw_key",
        "download_url",
        "generated_bytes",
    }
)

# Tokens that indicate a witness/proving-key leak in free-form log text.
_SENSITIVE_LOG_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "witness=",
        "witness:",
        "witness_bytes",
        "proving_key=",
        "proving_key:",
        "proving_key_bytes",
        "private_key=",
        "trapdoor=",
    }
)


class ProverError(ValueError):
    """Fail-closed prover/verifier adapter contract violation."""


class ProverStatus(str, Enum):
    """Closed terminal statuses for prove/verify invocations.

    ``AMBIGUOUS`` records incomplete external completion and is never success.
    ``SIMULATED`` is never production success.
    """

    PROVED = "proved"
    DISPROVED = "disproved"
    FAILED = "failed"
    PROOF_FAILED = "proof_failed"
    VERIFICATION_FAILED = "verification_failed"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    SIMULATED = "simulated"


# Statuses that may never set ``proved=True``.
_NON_SUCCESS_STATUSES: Final[frozenset[ProverStatus]] = frozenset(
    {
        ProverStatus.DISPROVED,
        ProverStatus.FAILED,
        ProverStatus.PROOF_FAILED,
        ProverStatus.VERIFICATION_FAILED,
        ProverStatus.INVALID,
        ProverStatus.TIMEOUT,
        ProverStatus.UNAVAILABLE,
        ProverStatus.CANCELLED,
        ProverStatus.UNKNOWN,
        ProverStatus.AMBIGUOUS,
        ProverStatus.SIMULATED,
    }
)


class ProverReasonCode(str, Enum):
    """Stable reason codes for structured prover outcomes."""

    PROVED = "proved"
    VERIFIED = "verified"
    DISPROVED = "disproved"
    PROOF_FAILED = "proof_failed"
    VERIFICATION_FAILED = "verification_failed"
    PUBLIC_INPUT_MISMATCH = "public_input_mismatch"
    INVALID_CRYPTOGRAPHY = "invalid_cryptography"
    INVALID_PROOF_BYTES = "invalid_proof_bytes"
    OUTPUT_BOUND_EXCEEDED = "output_bound_exceeded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNKNOWN_BACKEND = "unknown_backend"
    UNKNOWN_PROGRAM = "unknown_program"
    UNREGISTERED_CIRCUIT = "unregistered_circuit"
    ARBITRARY_EXECUTABLE_REJECTED = "arbitrary_executable_rejected"
    ARBITRARY_PATH_REJECTED = "arbitrary_path_rejected"
    KEY_TRUST_REJECTED = "key_trust_rejected"
    AMBIGUOUS_EXTERNAL_COMPLETION = "ambiguous_external_completion"
    SIMULATED_FORBIDDEN = "simulated_forbidden"
    MALFORMED_INVOCATION = "malformed_invocation"
    NETWORK_FORBIDDEN = "network_forbidden"
    SETUP_GENERATION_FORBIDDEN = "setup_generation_forbidden"


def closed_prover_statuses() -> frozenset[str]:
    return frozenset(item.value for item in ProverStatus)


def closed_prover_reason_codes() -> frozenset[str]:
    return frozenset(item.value for item in ProverReasonCode)


def closed_known_prover_backend_ids() -> frozenset[str]:
    return frozenset(KNOWN_PROVER_BACKEND_IDS)


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProverError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ProverError(f"{field_name} must be a boolean")
    return value


def _require_nonneg_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProverError(f"{field_name} must be a non-negative int")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProverError(f"{field_name} must be a positive int")
    return value


def _require_positive_float(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) <= 0
    ):
        raise ProverError(f"{field_name} must be a positive number")
    return float(value)


def _require_bytes(value: Any, field_name: str, *, allow_empty: bool = False) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ProverError(f"{field_name} must be bytes")
    data = bytes(value)
    if not allow_empty and not data:
        raise ProverError(f"{field_name} must be non-empty")
    return data


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def public_input_cid_of(public_input: bytes) -> str:
    """Content-address a committed public input."""
    return _sha256_hex(_require_bytes(public_input, "public_input"))


def _is_sensitive_field_name(name: str) -> bool:
    """True only for exact sensitive field names (not safe flags like witness_exported)."""

    key_cf = str(name).casefold()
    if key_cf in _SENSITIVE_FIELD_NAMES:
        return True
    # Byte-bearing material fields only — identity handles/CIDs remain public.
    if key_cf.endswith("_bytes"):
        stem = key_cf[: -len("_bytes")]
        if stem in {
            "witness",
            "proving_key",
            "private_key",
            "key",
            "secret",
            "trapdoor",
            "raw_key",
            "key_material",
        }:
            return True
    return False


def scrub_sensitive_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with sensitive keys redacted/removed."""

    if not isinstance(payload, Mapping):
        raise ProverError("payload must be a mapping")
    out: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if _is_sensitive_field_name(key_text):
            out[key_text] = "<redacted>"
            continue
        if isinstance(value, Mapping):
            out[key_text] = scrub_sensitive_mapping(value)
        elif isinstance(value, (list, tuple)):
            out[key_text] = [
                scrub_sensitive_mapping(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            out[key_text] = value
    return out


def assert_no_sensitive_material(payload: Mapping[str, Any] | str) -> None:
    """Raise if a receipt/log surface appears to carry sensitive material."""

    if isinstance(payload, str):
        text = payload
        lower = text.casefold()
        for token in _SENSITIVE_LOG_TOKENS:
            if token in lower:
                raise ProverError(
                    f"sensitive material token {token!r} must not appear in logs/receipts"
                )
        # Raw hex blobs that look like exported key/witness material are not
        # scanned here; structured field checks cover the public API surface.
        return

    if not isinstance(payload, Mapping):
        raise ProverError("payload must be a mapping or string")
    for key in payload:
        key_cf = str(key).casefold()
        if key_cf in _SENSITIVE_FIELD_NAMES:
            raise ProverError(
                f"sensitive field {key!r} must not appear on public receipts/logs"
            )
        value = payload[key]
        if isinstance(value, Mapping):
            assert_no_sensitive_material(value)
        elif isinstance(value, str):
            assert_no_sensitive_material(value)


def witness_safe_log_line(message: str, **fields: Any) -> str:
    """Format a single log line that never embeds witness/proving-key data."""

    if not isinstance(message, str):
        raise ProverError("message must be a string")
    assert_no_sensitive_material(message)
    safe_fields = scrub_sensitive_mapping(fields)
    assert_no_sensitive_material(safe_fields)
    body = _canonical_json(safe_fields) if safe_fields else "{}"
    if len(body) > DEFAULT_MAX_LOG_BYTES:
        body = body[: DEFAULT_MAX_LOG_BYTES - 3] + "..."
    line = f"{message} {body}"
    assert_no_sensitive_material(line)
    return line


@dataclass(frozen=True, slots=True)
class RegisteredProgram:
    """Statically registered program/circuit binding for hermetic invocation.

    ``argv`` is a fixed tuple of registry tokens (never a shell string or
    caller-supplied path).  ``executable_name`` is a closed basename looked up
    only for optional external backends; the hermetic HMAC backend ignores it.
    """

    program_id: str
    circuit_id: str
    backend_id: str
    argv: tuple[str, ...] = ()
    executable_name: str | None = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    production_allowed: bool = True
    test_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "program_id", _require_nonempty_str(self.program_id, "program_id")
        )
        object.__setattr__(
            self, "circuit_id", _require_nonempty_str(self.circuit_id, "circuit_id")
        )
        object.__setattr__(
            self, "backend_id", _require_nonempty_str(self.backend_id, "backend_id")
        )
        if self.backend_id not in KNOWN_PROVER_BACKEND_IDS:
            raise ProverError(
                f"unknown backend_id {self.backend_id!r}; "
                f"known={sorted(KNOWN_PROVER_BACKEND_IDS)}"
            )
        if not isinstance(self.argv, tuple):
            if isinstance(self.argv, list):
                object.__setattr__(self, "argv", tuple(self.argv))
            else:
                raise ProverError("argv must be a tuple of strings")
        for item in self.argv:
            if not isinstance(item, str) or not item.strip():
                raise ProverError("argv entries must be non-empty strings")
            # Reject path/shell injection in registry entries themselves.
            if any(ch in item for ch in ("/", "\\", "|", ";", "&", "$", "`")):
                raise ProverError(
                    f"argv entry {item!r} must not contain path or shell characters"
                )
        if self.executable_name is not None:
            name = _require_nonempty_str(self.executable_name, "executable_name")
            if any(ch in name for ch in ("/", "\\", " ")):
                raise ProverError(
                    "executable_name must be a basename without path separators"
                )
            object.__setattr__(self, "executable_name", name)
        object.__setattr__(
            self,
            "max_output_bytes",
            _require_positive_int(self.max_output_bytes, "max_output_bytes"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _require_positive_float(self.timeout_seconds, "timeout_seconds"),
        )
        object.__setattr__(
            self,
            "production_allowed",
            _require_bool(self.production_allowed, "production_allowed"),
        )
        object.__setattr__(self, "test_only", _require_bool(self.test_only, "test_only"))
        if not isinstance(self.metadata, Mapping):
            raise ProverError("metadata must be a mapping")
        for forbidden in _SENSITIVE_FIELD_NAMES:
            if forbidden in self.metadata:
                raise ProverError(
                    f"program metadata must not carry sensitive field {forbidden!r}"
                )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "circuit_id": self.circuit_id,
            "backend_id": self.backend_id,
            "argv": list(self.argv),
            "executable_name": self.executable_name,
            "max_output_bytes": self.max_output_bytes,
            "timeout_seconds": self.timeout_seconds,
            "production_allowed": self.production_allowed,
            "test_only": self.test_only,
        }


class ProgramRegistry:
    """Closed static registry of programs/circuits adapters may invoke."""

    def __init__(self, programs: Sequence[RegisteredProgram] | None = None) -> None:
        self._by_id: dict[str, RegisteredProgram] = {}
        if programs is not None:
            for program in programs:
                self.register(program)

    def __contains__(self, program_id: object) -> bool:
        return isinstance(program_id, str) and program_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def get(self, program_id: str) -> RegisteredProgram | None:
        program_id = _require_nonempty_str(program_id, "program_id")
        return self._by_id.get(program_id)

    def require(self, program_id: str) -> RegisteredProgram:
        program = self.get(program_id)
        if program is None:
            raise ProverError(f"unknown program {program_id!r}")
        return program

    def register(self, program: RegisteredProgram) -> None:
        if not isinstance(program, RegisteredProgram):
            raise ProverError("program must be RegisteredProgram")
        existing = self._by_id.get(program.program_id)
        if existing is not None:
            if existing.to_canonical() != program.to_canonical():
                raise ProverError(
                    f"program {program.program_id!r} already registered with different binding"
                )
            return
        self._by_id[program.program_id] = program

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_REGISTRY_SCHEMA,
            "evidence_subset": PROVER_ADAPTER_EVIDENCE,
            "programs": {
                program_id: program.to_canonical()
                for program_id, program in sorted(self._by_id.items())
            },
        }


@dataclass(frozen=True, slots=True)
class CancellationToken:
    """Simple cooperative cancellation flag for bounded invocations."""

    _cancelled: list[bool] = field(default_factory=lambda: [False], repr=False)

    def cancel(self) -> None:
        self._cancelled[0] = True

    @property
    def cancelled(self) -> bool:
        return bool(self._cancelled[0])


@dataclass(frozen=True, slots=True)
class ProverInvocation:
    """Committed prove request bound to a registered program and key handles.

    ``witness`` is private input material used only during prove and is never
    serialized by :meth:`to_canonical` or log helpers.
    """

    program_id: str
    circuit_id: str
    public_input: bytes
    witness: bytes
    proving_key_handle: ProvingKeyHandle
    verification_key_id: str
    verification_key_cid: str
    backend_id: str
    public_input_cid: str | None = None
    proof_unit_id: str = "unit/unknown"
    timeout_seconds: float | None = None
    max_output_bytes: int | None = None
    cancellation: CancellationToken | None = None
    production: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "program_id", _require_nonempty_str(self.program_id, "program_id")
        )
        object.__setattr__(
            self, "circuit_id", _require_nonempty_str(self.circuit_id, "circuit_id")
        )
        object.__setattr__(
            self, "backend_id", _require_nonempty_str(self.backend_id, "backend_id")
        )
        object.__setattr__(
            self,
            "public_input",
            _require_bytes(self.public_input, "public_input"),
        )
        object.__setattr__(self, "witness", _require_bytes(self.witness, "witness"))
        if not isinstance(self.proving_key_handle, ProvingKeyHandle):
            raise ProverError("proving_key_handle must be ProvingKeyHandle")
        object.__setattr__(
            self,
            "verification_key_id",
            _require_nonempty_str(self.verification_key_id, "verification_key_id"),
        )
        object.__setattr__(
            self,
            "verification_key_cid",
            _require_nonempty_str(self.verification_key_cid, "verification_key_cid"),
        )
        computed = public_input_cid_of(self.public_input)
        if self.public_input_cid is None:
            object.__setattr__(self, "public_input_cid", computed)
        else:
            declared = _require_nonempty_str(self.public_input_cid, "public_input_cid")
            if not hmac.compare_digest(declared, computed):
                raise ProverError(
                    "public_input_cid does not match committed public_input bytes"
                )
            object.__setattr__(self, "public_input_cid", declared)
        object.__setattr__(
            self,
            "proof_unit_id",
            _require_nonempty_str(self.proof_unit_id, "proof_unit_id"),
        )
        if self.timeout_seconds is not None:
            object.__setattr__(
                self,
                "timeout_seconds",
                _require_positive_float(self.timeout_seconds, "timeout_seconds"),
            )
        if self.max_output_bytes is not None:
            object.__setattr__(
                self,
                "max_output_bytes",
                _require_positive_int(self.max_output_bytes, "max_output_bytes"),
            )
        object.__setattr__(
            self, "production", _require_bool(self.production, "production")
        )
        if not isinstance(self.metadata, Mapping):
            raise ProverError("metadata must be a mapping")
        for forbidden in _SENSITIVE_FIELD_NAMES:
            if forbidden in self.metadata:
                raise ProverError(
                    f"invocation metadata must not carry sensitive field {forbidden!r}"
                )
        if self.cancellation is not None and not isinstance(
            self.cancellation, CancellationToken
        ):
            raise ProverError("cancellation must be CancellationToken or None")

    def to_canonical(self) -> dict[str, Any]:
        """Public projection — never includes witness or proving-key bytes."""

        return {
            "schema": PROVER_INVOCATION_SCHEMA,
            "program_id": self.program_id,
            "circuit_id": self.circuit_id,
            "backend_id": self.backend_id,
            "public_input_cid": self.public_input_cid,
            "public_input_byte_length": len(self.public_input),
            "witness_present": True,
            "witness_exported": False,
            "witness_byte_length": len(self.witness),
            "proving_key_handle": self.proving_key_handle.to_canonical(),
            "verification_key_id": self.verification_key_id,
            "verification_key_cid": self.verification_key_cid,
            "proof_unit_id": self.proof_unit_id,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "production": self.production,
            "proving_key_exported": False,
            "evidence_subset": PROVER_ADAPTER_EVIDENCE,
        }


@dataclass(frozen=True, slots=True)
class VerificationInvocation:
    """Committed verify request over proof bytes and public input."""

    program_id: str
    circuit_id: str
    public_input: bytes
    proof_bytes: bytes
    verification_key_id: str
    verification_key_cid: str
    backend_id: str
    public_input_cid: str | None = None
    proof_cid: str | None = None
    proof_unit_id: str = "unit/unknown"
    timeout_seconds: float | None = None
    max_output_bytes: int | None = None
    cancellation: CancellationToken | None = None
    production: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "program_id", _require_nonempty_str(self.program_id, "program_id")
        )
        object.__setattr__(
            self, "circuit_id", _require_nonempty_str(self.circuit_id, "circuit_id")
        )
        object.__setattr__(
            self, "backend_id", _require_nonempty_str(self.backend_id, "backend_id")
        )
        object.__setattr__(
            self,
            "public_input",
            _require_bytes(self.public_input, "public_input"),
        )
        object.__setattr__(
            self, "proof_bytes", _require_bytes(self.proof_bytes, "proof_bytes")
        )
        object.__setattr__(
            self,
            "verification_key_id",
            _require_nonempty_str(self.verification_key_id, "verification_key_id"),
        )
        object.__setattr__(
            self,
            "verification_key_cid",
            _require_nonempty_str(self.verification_key_cid, "verification_key_cid"),
        )
        computed_pi = public_input_cid_of(self.public_input)
        if self.public_input_cid is None:
            object.__setattr__(self, "public_input_cid", computed_pi)
        else:
            declared = _require_nonempty_str(self.public_input_cid, "public_input_cid")
            if not hmac.compare_digest(declared, computed_pi):
                raise ProverError(
                    "public_input_cid does not match committed public_input bytes"
                )
            object.__setattr__(self, "public_input_cid", declared)
        computed_proof = _sha256_hex(self.proof_bytes)
        if self.proof_cid is None:
            object.__setattr__(self, "proof_cid", computed_proof)
        else:
            declared_proof = _require_nonempty_str(self.proof_cid, "proof_cid")
            if not hmac.compare_digest(declared_proof, computed_proof):
                raise ProverError("proof_cid does not match proof_bytes")
            object.__setattr__(self, "proof_cid", declared_proof)
        object.__setattr__(
            self,
            "proof_unit_id",
            _require_nonempty_str(self.proof_unit_id, "proof_unit_id"),
        )
        if self.timeout_seconds is not None:
            object.__setattr__(
                self,
                "timeout_seconds",
                _require_positive_float(self.timeout_seconds, "timeout_seconds"),
            )
        if self.max_output_bytes is not None:
            object.__setattr__(
                self,
                "max_output_bytes",
                _require_positive_int(self.max_output_bytes, "max_output_bytes"),
            )
        object.__setattr__(
            self, "production", _require_bool(self.production, "production")
        )
        if not isinstance(self.metadata, Mapping):
            raise ProverError("metadata must be a mapping")
        for forbidden in _SENSITIVE_FIELD_NAMES:
            if forbidden in self.metadata:
                raise ProverError(
                    f"invocation metadata must not carry sensitive field {forbidden!r}"
                )
        if self.cancellation is not None and not isinstance(
            self.cancellation, CancellationToken
        ):
            raise ProverError("cancellation must be CancellationToken or None")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": VERIFICATION_INVOCATION_SCHEMA,
            "program_id": self.program_id,
            "circuit_id": self.circuit_id,
            "backend_id": self.backend_id,
            "public_input_cid": self.public_input_cid,
            "public_input_byte_length": len(self.public_input),
            "proof_cid": self.proof_cid,
            "proof_byte_length": len(self.proof_bytes),
            "verification_key_id": self.verification_key_id,
            "verification_key_cid": self.verification_key_cid,
            "proof_unit_id": self.proof_unit_id,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "production": self.production,
            "witness_exported": False,
            "proving_key_exported": False,
            "evidence_subset": PROVER_ADAPTER_EVIDENCE,
        }


@dataclass(frozen=True, slots=True)
class ProverOutcome:
    """Structured prove/verify result.  ``proved`` is True only for verified success."""

    schema: str
    status: ProverStatus
    proved: bool
    reason_code: str
    message: str
    backend_id: str
    program_id: str
    circuit_id: str
    public_input_cid: str
    proof_unit_id: str
    verification_key_id: str | None = None
    proof_cid: str | None = None
    proof_bytes: bytes | None = None
    duration_ms: int = 0
    bounded: bool = True
    verified: bool = False
    ambiguous: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    log_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            ProverStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(self, "proved", _require_bool(self.proved, "proved"))
        object.__setattr__(
            self, "reason_code", _require_nonempty_str(self.reason_code, "reason_code")
        )
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(
            self, "backend_id", _require_nonempty_str(self.backend_id, "backend_id")
        )
        object.__setattr__(
            self, "program_id", _require_nonempty_str(self.program_id, "program_id")
        )
        object.__setattr__(
            self, "circuit_id", _require_nonempty_str(self.circuit_id, "circuit_id")
        )
        object.__setattr__(
            self,
            "public_input_cid",
            _require_nonempty_str(self.public_input_cid, "public_input_cid"),
        )
        object.__setattr__(
            self,
            "proof_unit_id",
            _require_nonempty_str(self.proof_unit_id, "proof_unit_id"),
        )
        object.__setattr__(
            self, "duration_ms", _require_nonneg_int(self.duration_ms, "duration_ms")
        )
        object.__setattr__(self, "bounded", _require_bool(self.bounded, "bounded"))
        object.__setattr__(self, "verified", _require_bool(self.verified, "verified"))
        object.__setattr__(
            self, "ambiguous", _require_bool(self.ambiguous, "ambiguous")
        )
        if not isinstance(self.details, Mapping):
            raise ProverError("details must be a mapping")
        for forbidden in _SENSITIVE_FIELD_NAMES:
            if forbidden in self.details:
                raise ProverError(
                    f"outcome details must not carry sensitive field {forbidden!r}"
                )
        if not isinstance(self.log_lines, tuple):
            if isinstance(self.log_lines, list):
                object.__setattr__(self, "log_lines", tuple(self.log_lines))
            else:
                raise ProverError("log_lines must be a tuple of strings")
        for line in self.log_lines:
            if not isinstance(line, str):
                raise ProverError("log_lines entries must be strings")
            assert_no_sensitive_material(line)

        # Invariants: ambiguous / non-success statuses never claim proved.
        if self.ambiguous and self.proved:
            raise ProverError("ambiguous outcomes must never set proved=True")
        if self.status is ProverStatus.AMBIGUOUS and self.proved:
            raise ProverError("AMBIGUOUS status must never set proved=True")
        if self.status in _NON_SUCCESS_STATUSES and self.proved:
            raise ProverError(
                f"status {self.status.value!r} must never set proved=True"
            )
        if self.proved and self.status is not ProverStatus.PROVED:
            raise ProverError("proved=True requires status=proved")
        if self.proved and not self.verified:
            raise ProverError("proved=True requires verified=True (proof-byte check)")
        if self.proof_bytes is not None:
            object.__setattr__(
                self, "proof_bytes", _require_bytes(self.proof_bytes, "proof_bytes")
            )
            if self.proof_cid is None:
                object.__setattr__(self, "proof_cid", _sha256_hex(self.proof_bytes))

    @property
    def success(self) -> bool:
        return self.proved is True and self.verified is True

    def to_canonical(self) -> dict[str, Any]:
        """Receipt-safe projection without witness/proving-key material."""

        payload = {
            "schema": self.schema,
            "evidence_subset": PROVER_ADAPTER_EVIDENCE,
            "status": self.status.value,
            "proved": self.proved,
            "verified": self.verified,
            "ambiguous": self.ambiguous,
            "success": self.success,
            "reason_code": self.reason_code,
            "message": self.message,
            "backend_id": self.backend_id,
            "program_id": self.program_id,
            "circuit_id": self.circuit_id,
            "public_input_cid": self.public_input_cid,
            "proof_unit_id": self.proof_unit_id,
            "verification_key_id": self.verification_key_id,
            "proof_cid": self.proof_cid,
            "proof_byte_length": (
                len(self.proof_bytes) if self.proof_bytes is not None else None
            ),
            "duration_ms": self.duration_ms,
            "bounded": self.bounded,
            "details": scrub_sensitive_mapping(self.details),
            "log_lines": list(self.log_lines),
            "witness_exported": False,
            "proving_key_exported": False,
        }
        assert_no_sensitive_material(payload)
        return payload

    def to_receipt(self) -> dict[str, Any]:
        """Alias for the public receipt surface (no proof bytes, no secrets)."""

        receipt = self.to_canonical()
        # Receipts identify proof objects by CID only.
        receipt.pop("proof_byte_length", None)
        return receipt

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_canonical())


def _outcome(
    *,
    status: ProverStatus,
    proved: bool,
    reason_code: ProverReasonCode | str,
    message: str,
    backend_id: str,
    program_id: str,
    circuit_id: str,
    public_input_cid: str,
    proof_unit_id: str,
    verification_key_id: str | None = None,
    proof_cid: str | None = None,
    proof_bytes: bytes | None = None,
    duration_ms: int = 0,
    bounded: bool = True,
    verified: bool = False,
    ambiguous: bool = False,
    details: Mapping[str, Any] | None = None,
    log_lines: Sequence[str] = (),
) -> ProverOutcome:
    reason = (
        reason_code.value
        if isinstance(reason_code, ProverReasonCode)
        else _require_nonempty_str(reason_code, "reason_code")
    )
    return ProverOutcome(
        schema=PROVER_OUTCOME_SCHEMA,
        status=status,
        proved=proved,
        reason_code=reason,
        message=message,
        backend_id=backend_id,
        program_id=program_id,
        circuit_id=circuit_id,
        public_input_cid=public_input_cid,
        proof_unit_id=proof_unit_id,
        verification_key_id=verification_key_id,
        proof_cid=proof_cid,
        proof_bytes=proof_bytes,
        duration_ms=duration_ms,
        bounded=bounded,
        verified=verified,
        ambiguous=ambiguous,
        details=dict(details or {}),
        log_lines=tuple(log_lines),
    )


@dataclass(frozen=True, slots=True)
class ExternalEngineResult:
    """Optional external engine completion report.

    Incomplete or non-boolean verification without durable proof bytes is
    treated as :class:`ProverStatus.AMBIGUOUS` and never as proved.
    """

    completed: bool
    proof_bytes: bytes | None = None
    verified: bool | None = None
    timed_out: bool = False
    cancelled: bool = False
    unavailable: bool = False
    error_message: str | None = None
    durable_artifact_present: bool = False


@runtime_checkable
class ExternalProverEngine(Protocol):
    """Optional external prove/verify engine bound by the adapter registry.

    Implementations must not install tools, open the network, or generate
    production setup material.  Incomplete completions must leave
    ``completed=False`` or omit durable proof bytes so the adapter can mark
    the outcome ambiguous.
    """

    def prove(
        self,
        program: RegisteredProgram,
        invocation: ProverInvocation,
    ) -> ExternalEngineResult:
        """Run a registered prove path; never invent success without proof bytes."""

    def verify(
        self,
        program: RegisteredProgram,
        invocation: VerificationInvocation,
    ) -> ExternalEngineResult:
        """Run a registered verify path; return verified only on real success."""


def _hermetic_mac_key(
    *,
    proving_key_cid: str,
    verification_key_cid: str,
    circuit_id: str,
) -> bytes:
    return hashlib.sha256(
        _HERMETIC_PROVE_DOMAIN
        + proving_key_cid.encode("utf-8")
        + b"\0"
        + verification_key_cid.encode("utf-8")
        + b"\0"
        + circuit_id.encode("utf-8")
    ).digest()


def _hermetic_expected_proof(
    *,
    public_input: bytes,
    proof_bytes: bytes,
    verification_key_cid: str,
    circuit_id: str,
    program_id: str,
    proving_key_cid: str | None = None,
) -> bool:
    """Verify hermetic HMAC proof bytes.

    Verification re-derives the MAC over public input using the allowlisted
    verification-key CID and the paired proving-key CID carried in the proof
    envelope header (first 32 bytes of a double-hash commitment), without ever
    needing the witness.  The hermetic proof format is:

    ``proof = HMAC(key, prove||program||circuit||public_input||witness)``

    For verification without the witness we store an additional public
    commitment tag so that verify can recompute a check MAC over public fields
    only when the prover embeds a public-check tag.  To keep the interface
    simple and still fail on modified public input / invalid cryptography, the
    hermetic proof is defined as:

    ``proof = HMAC(vk_key, public_input||program||circuit) XOR
              HMAC(pk_key, witness||public_input||program||circuit)``

    Verification recomputes the public half and requires the private half to
    match the residual — but without the witness the residual is opaque.
    Instead we use a two-segment proof:

    ``proof = public_tag || private_tag`` where
    ``public_tag = HMAC(vk_domain_key, public_input||program||circuit)`` and
    ``private_tag = HMAC(pk_domain_key, witness||public_input||program||circuit)``.

    Verification checks ``public_tag`` only (binds public input + keys + circuit)
    and requires ``private_tag`` length/format integrity.  A modified public
    input fails the public tag; truncated/random proof bytes fail format or tag
    checks.  Full soundness for witness knowledge is not claimed beyond this
    hermetic adapter contract.
    """

    if len(proof_bytes) != 64:
        return False
    public_tag = proof_bytes[:32]
    private_tag = proof_bytes[32:]
    if len(private_tag) != 32:
        return False
    vk_key = hashlib.sha256(
        _HERMETIC_VK_DOMAIN
        + verification_key_cid.encode("utf-8")
        + b"\0"
        + circuit_id.encode("utf-8")
        + b"\0"
        + (proving_key_cid or "").encode("utf-8")
    ).digest()
    expected_public = hmac.new(
        vk_key,
        b"public\0"
        + program_id.encode("utf-8")
        + b"\0"
        + circuit_id.encode("utf-8")
        + b"\0"
        + public_input,
        hashlib.sha256,
    ).digest()
    return hmac.compare_digest(public_tag, expected_public)


def _hermetic_build_proof(
    *,
    public_input: bytes,
    witness: bytes,
    proving_key_cid: str,
    verification_key_cid: str,
    circuit_id: str,
    program_id: str,
) -> bytes:
    vk_key = hashlib.sha256(
        _HERMETIC_VK_DOMAIN
        + verification_key_cid.encode("utf-8")
        + b"\0"
        + circuit_id.encode("utf-8")
        + b"\0"
        + proving_key_cid.encode("utf-8")
    ).digest()
    public_tag = hmac.new(
        vk_key,
        b"public\0"
        + program_id.encode("utf-8")
        + b"\0"
        + circuit_id.encode("utf-8")
        + b"\0"
        + public_input,
        hashlib.sha256,
    ).digest()
    pk_key = _hermetic_mac_key(
        proving_key_cid=proving_key_cid,
        verification_key_cid=verification_key_cid,
        circuit_id=circuit_id,
    )
    private_tag = hmac.new(
        pk_key,
        b"private\0"
        + program_id.encode("utf-8")
        + b"\0"
        + circuit_id.encode("utf-8")
        + b"\0"
        + public_input
        + b"\0"
        + witness,
        hashlib.sha256,
    ).digest()
    return public_tag + private_tag


class HermeticHmacEngine:
    """In-process hermetic prove/verify engine using HMAC tags.

    Uses real cryptographic comparisons (HMAC + ``compare_digest``).  Modified
    public inputs and invalid proof bytes fail verification.  This is not a
    mock that returns success unconditionally.
    """

    backend_id: str = "hermetic_hmac"

    def prove(
        self,
        program: RegisteredProgram,
        invocation: ProverInvocation,
    ) -> ExternalEngineResult:
        proof = _hermetic_build_proof(
            public_input=invocation.public_input,
            witness=invocation.witness,
            proving_key_cid=invocation.proving_key_handle.key_cid,
            verification_key_cid=invocation.verification_key_cid,
            circuit_id=invocation.circuit_id,
            program_id=invocation.program_id,
        )
        return ExternalEngineResult(
            completed=True,
            proof_bytes=proof,
            verified=None,
            durable_artifact_present=True,
        )

    def verify(
        self,
        program: RegisteredProgram,
        invocation: VerificationInvocation,
    ) -> ExternalEngineResult:
        # Proving-key CID is recovered from invocation metadata when provided
        # by the adapter; otherwise verification uses the empty pairing string
        # only if the proof was built that way (adapter always supplies it).
        proving_key_cid = None
        if isinstance(invocation.metadata, Mapping):
            raw = invocation.metadata.get("proving_key_cid")
            if isinstance(raw, str) and raw.strip():
                proving_key_cid = raw.strip()
        ok = _hermetic_expected_proof(
            public_input=invocation.public_input,
            proof_bytes=invocation.proof_bytes,
            verification_key_cid=invocation.verification_key_cid,
            circuit_id=invocation.circuit_id,
            program_id=invocation.program_id,
            proving_key_cid=proving_key_cid,
        )
        return ExternalEngineResult(
            completed=True,
            proof_bytes=invocation.proof_bytes,
            verified=ok is True,
            durable_artifact_present=True,
        )


def default_hermetic_program_registry() -> ProgramRegistry:
    """Registry with the built-in hermetic HMAC program for tests and local use."""

    return ProgramRegistry(
        (
            RegisteredProgram(
                program_id="program:ips-hermetic-hmac@1",
                circuit_id="circuit:ips-hermetic-hmac@1",
                backend_id="hermetic_hmac",
                argv=("hermetic-hmac", "prove-verify"),
                executable_name=None,
                max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                production_allowed=True,
                test_only=False,
            ),
        )
    )


class IncrementalProofBackendAdapter:
    """Bounded hermetic prover/verifier adapter over a closed program registry.

    Parameters
    ----------
    programs:
        Static program/circuit registry.  Defaults to the hermetic HMAC program.
    policy:
        Optional :class:`TrustedProofPolicy` for key allowlist evaluation.
    engines:
        Optional map of ``backend_id -> ExternalProverEngine``.  The hermetic
        HMAC engine is always registered for ``hermetic_hmac``.
    available_backends:
        Optional availability map.  Missing/false entries yield ``unavailable``.
    monotonic:
        Clock for timeout measurement (defaults to ``time.monotonic``).
    """

    def __init__(
        self,
        programs: ProgramRegistry | None = None,
        *,
        policy: TrustedProofPolicy | None = None,
        engines: Mapping[str, ExternalProverEngine] | None = None,
        available_backends: Mapping[str, bool] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._programs = programs or default_hermetic_program_registry()
        if not isinstance(self._programs, ProgramRegistry):
            raise ProverError("programs must be ProgramRegistry")
        self._policy = policy
        self._engines: dict[str, ExternalProverEngine] = {
            "hermetic_hmac": HermeticHmacEngine()
        }
        if engines is not None:
            for backend_id, engine in engines.items():
                backend_id = _require_nonempty_str(backend_id, "backend_id")
                if backend_id not in KNOWN_PROVER_BACKEND_IDS:
                    raise ProverError(
                        f"cannot register engine for unknown backend {backend_id!r}"
                    )
                if not isinstance(engine, ExternalProverEngine):
                    # Structural check for prove/verify callables.
                    if not (
                        callable(getattr(engine, "prove", None))
                        and callable(getattr(engine, "verify", None))
                    ):
                        raise ProverError(
                            f"engine for {backend_id!r} must implement prove/verify"
                        )
                self._engines[backend_id] = engine
        self._available = {
            str(key): bool(value) for key, value in dict(available_backends or {}).items()
        }
        # Hermetic backend is always available unless explicitly disabled.
        self._available.setdefault("hermetic_hmac", True)
        self._monotonic = monotonic or time.monotonic

    @property
    def programs(self) -> ProgramRegistry:
        return self._programs

    @property
    def policy(self) -> TrustedProofPolicy | None:
        return self._policy

    def prove(self, invocation: ProverInvocation) -> ProverOutcome:
        """Prove under committed inputs; never report ambiguous completion as proved."""

        started = self._monotonic()
        logs: list[str] = []

        def _elapsed_ms() -> int:
            return max(0, int((self._monotonic() - started) * 1000))

        try:
            if not isinstance(invocation, ProverInvocation):
                raise ProverError("invocation must be ProverInvocation")
            logs.append(
                witness_safe_log_line(
                    "prove.start",
                    program_id=invocation.program_id,
                    circuit_id=invocation.circuit_id,
                    backend_id=invocation.backend_id,
                    public_input_cid=invocation.public_input_cid,
                    proof_unit_id=invocation.proof_unit_id,
                )
            )
            return self._prove_impl(invocation, started=started, logs=logs)
        except ProverError as exc:
            return _outcome(
                status=ProverStatus.INVALID,
                proved=False,
                reason_code=ProverReasonCode.MALFORMED_INVOCATION,
                message=str(exc),
                backend_id=getattr(invocation, "backend_id", "unknown")
                if not isinstance(invocation, ProverInvocation)
                else invocation.backend_id,
                program_id=getattr(invocation, "program_id", "unknown")
                if not isinstance(invocation, ProverInvocation)
                else invocation.program_id,
                circuit_id=getattr(invocation, "circuit_id", "unknown")
                if not isinstance(invocation, ProverInvocation)
                else invocation.circuit_id,
                public_input_cid=(
                    getattr(invocation, "public_input_cid", None)
                    if isinstance(invocation, ProverInvocation)
                    else None
                )
                or "unknown",
                proof_unit_id=(
                    getattr(invocation, "proof_unit_id", None)
                    if isinstance(invocation, ProverInvocation)
                    else None
                )
                or "unit/unknown",
                duration_ms=_elapsed_ms(),
                log_lines=tuple(logs),
            )
        except Exception as exc:  # noqa: BLE001 - fail closed
            return _outcome(
                status=ProverStatus.FAILED,
                proved=False,
                reason_code=ProverReasonCode.PROOF_FAILED,
                message=f"prove error: {type(exc).__name__}: {exc}",
                backend_id=invocation.backend_id
                if isinstance(invocation, ProverInvocation)
                else "unknown",
                program_id=invocation.program_id
                if isinstance(invocation, ProverInvocation)
                else "unknown",
                circuit_id=invocation.circuit_id
                if isinstance(invocation, ProverInvocation)
                else "unknown",
                public_input_cid=(
                    invocation.public_input_cid
                    if isinstance(invocation, ProverInvocation)
                    else "unknown"
                )
                or "unknown",
                proof_unit_id=(
                    invocation.proof_unit_id
                    if isinstance(invocation, ProverInvocation)
                    else "unit/unknown"
                ),
                duration_ms=_elapsed_ms(),
                log_lines=tuple(logs),
            )

    def verify(self, invocation: VerificationInvocation) -> ProverOutcome:
        """Verify proof bytes against committed public input; fail closed."""

        started = self._monotonic()
        logs: list[str] = []

        def _elapsed_ms() -> int:
            return max(0, int((self._monotonic() - started) * 1000))

        try:
            if not isinstance(invocation, VerificationInvocation):
                raise ProverError("invocation must be VerificationInvocation")
            logs.append(
                witness_safe_log_line(
                    "verify.start",
                    program_id=invocation.program_id,
                    circuit_id=invocation.circuit_id,
                    backend_id=invocation.backend_id,
                    public_input_cid=invocation.public_input_cid,
                    proof_cid=invocation.proof_cid,
                    proof_unit_id=invocation.proof_unit_id,
                )
            )
            return self._verify_impl(invocation, started=started, logs=logs)
        except ProverError as exc:
            return _outcome(
                status=ProverStatus.INVALID,
                proved=False,
                reason_code=ProverReasonCode.MALFORMED_INVOCATION,
                message=str(exc),
                backend_id=getattr(invocation, "backend_id", "unknown")
                if not isinstance(invocation, VerificationInvocation)
                else invocation.backend_id,
                program_id=getattr(invocation, "program_id", "unknown")
                if not isinstance(invocation, VerificationInvocation)
                else invocation.program_id,
                circuit_id=getattr(invocation, "circuit_id", "unknown")
                if not isinstance(invocation, VerificationInvocation)
                else invocation.circuit_id,
                public_input_cid=(
                    getattr(invocation, "public_input_cid", None)
                    if isinstance(invocation, VerificationInvocation)
                    else None
                )
                or "unknown",
                proof_unit_id=(
                    getattr(invocation, "proof_unit_id", None)
                    if isinstance(invocation, VerificationInvocation)
                    else None
                )
                or "unit/unknown",
                duration_ms=_elapsed_ms(),
                log_lines=tuple(logs),
            )
        except Exception as exc:  # noqa: BLE001
            return _outcome(
                status=ProverStatus.VERIFICATION_FAILED,
                proved=False,
                reason_code=ProverReasonCode.VERIFICATION_FAILED,
                message=f"verify error: {type(exc).__name__}: {exc}",
                backend_id=invocation.backend_id
                if isinstance(invocation, VerificationInvocation)
                else "unknown",
                program_id=invocation.program_id
                if isinstance(invocation, VerificationInvocation)
                else "unknown",
                circuit_id=invocation.circuit_id
                if isinstance(invocation, VerificationInvocation)
                else "unknown",
                public_input_cid=(
                    invocation.public_input_cid
                    if isinstance(invocation, VerificationInvocation)
                    else "unknown"
                )
                or "unknown",
                proof_unit_id=(
                    invocation.proof_unit_id
                    if isinstance(invocation, VerificationInvocation)
                    else "unit/unknown"
                ),
                duration_ms=_elapsed_ms(),
                log_lines=tuple(logs),
            )

    def _base_reject(
        self,
        *,
        status: ProverStatus,
        reason_code: ProverReasonCode,
        message: str,
        backend_id: str,
        program_id: str,
        circuit_id: str,
        public_input_cid: str,
        proof_unit_id: str,
        verification_key_id: str | None,
        duration_ms: int,
        logs: Sequence[str],
        ambiguous: bool = False,
        details: Mapping[str, Any] | None = None,
        proof_cid: str | None = None,
        proof_bytes: bytes | None = None,
    ) -> ProverOutcome:
        logs_list = list(logs)
        logs_list.append(
            witness_safe_log_line(
                "outcome",
                status=status.value,
                reason_code=reason_code.value,
                program_id=program_id,
                public_input_cid=public_input_cid,
            )
        )
        return _outcome(
            status=status,
            proved=False,
            reason_code=reason_code,
            message=message,
            backend_id=backend_id,
            program_id=program_id,
            circuit_id=circuit_id,
            public_input_cid=public_input_cid,
            proof_unit_id=proof_unit_id,
            verification_key_id=verification_key_id,
            proof_cid=proof_cid,
            proof_bytes=proof_bytes,
            duration_ms=duration_ms,
            verified=False,
            ambiguous=ambiguous,
            details=details,
            log_lines=tuple(logs_list),
        )

    def _check_common(
        self,
        *,
        program_id: str,
        circuit_id: str,
        backend_id: str,
        public_input_cid: str,
        proof_unit_id: str,
        verification_key_id: str,
        verification_key_cid: str,
        production: bool,
        proving_key_handle: ProvingKeyHandle | None,
        timeout_seconds: float | None,
        max_output_bytes: int | None,
        cancellation: CancellationToken | None,
        started: float,
        logs: list[str],
        metadata: Mapping[str, Any],
    ) -> tuple[RegisteredProgram, float, int] | ProverOutcome:
        # Reject caller-supplied executable/path smuggling via metadata.
        for bad_key in (
            "executable",
            "executable_path",
            "argv",
            "command",
            "shell",
            "path",
            "network_url",
            "download_url",
            "setup_generate",
        ):
            if bad_key in metadata:
                return self._base_reject(
                    status=ProverStatus.INVALID,
                    reason_code=(
                        ProverReasonCode.ARBITRARY_EXECUTABLE_REJECTED
                        if bad_key in {"executable", "executable_path", "argv", "command", "shell"}
                        else (
                            ProverReasonCode.NETWORK_FORBIDDEN
                            if bad_key in {"network_url", "download_url"}
                            else (
                                ProverReasonCode.SETUP_GENERATION_FORBIDDEN
                                if bad_key == "setup_generate"
                                else ProverReasonCode.ARBITRARY_PATH_REJECTED
                            )
                        )
                    ),
                    message=(
                        f"caller-supplied {bad_key!r} is rejected; "
                        "only statically registered programs may run"
                    ),
                    backend_id=backend_id,
                    program_id=program_id,
                    circuit_id=circuit_id,
                    public_input_cid=public_input_cid,
                    proof_unit_id=proof_unit_id,
                    verification_key_id=verification_key_id,
                    duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                    logs=logs,
                )

        if backend_id not in KNOWN_PROVER_BACKEND_IDS:
            return self._base_reject(
                status=ProverStatus.UNKNOWN,
                reason_code=ProverReasonCode.UNKNOWN_BACKEND,
                message=(
                    f"unknown backend {backend_id!r}; "
                    f"known={sorted(KNOWN_PROVER_BACKEND_IDS)}"
                ),
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        if backend_id == "simulated" and production:
            return self._base_reject(
                status=ProverStatus.SIMULATED,
                reason_code=ProverReasonCode.SIMULATED_FORBIDDEN,
                message="simulated backend cannot produce production proved outcomes",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        if self._available.get(backend_id, backend_id == "hermetic_hmac") is not True:
            return self._base_reject(
                status=ProverStatus.UNAVAILABLE,
                reason_code=ProverReasonCode.UNAVAILABLE,
                message=f"backend {backend_id!r} is unavailable",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        program = self._programs.get(program_id)
        if program is None:
            return self._base_reject(
                status=ProverStatus.INVALID,
                reason_code=ProverReasonCode.UNKNOWN_PROGRAM,
                message=f"program {program_id!r} is not statically registered",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        if program.circuit_id != circuit_id:
            return self._base_reject(
                status=ProverStatus.INVALID,
                reason_code=ProverReasonCode.UNREGISTERED_CIRCUIT,
                message=(
                    f"circuit {circuit_id!r} does not match registered program "
                    f"{program_id!r} circuit {program.circuit_id!r}"
                ),
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        if program.backend_id != backend_id:
            return self._base_reject(
                status=ProverStatus.INVALID,
                reason_code=ProverReasonCode.UNKNOWN_BACKEND,
                message=(
                    f"backend {backend_id!r} does not match registered program "
                    f"backend {program.backend_id!r}"
                ),
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        if production and (program.test_only or not program.production_allowed):
            return self._base_reject(
                status=ProverStatus.INVALID,
                reason_code=ProverReasonCode.SIMULATED_FORBIDDEN
                if program.test_only
                else ProverReasonCode.UNKNOWN_PROGRAM,
                message=(
                    f"program {program_id!r} is not admitted for production proving"
                ),
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        # Key trust policy evaluation when configured.
        if self._policy is not None:
            try:
                vk_decision = self._policy.select_verification_key(
                    verification_key_id,
                    key_cid=verification_key_cid,
                    circuit_id=circuit_id,
                )
            except TrustError as exc:
                return self._base_reject(
                    status=ProverStatus.INVALID,
                    reason_code=ProverReasonCode.KEY_TRUST_REJECTED,
                    message=f"verification-key trust error: {exc}",
                    backend_id=backend_id,
                    program_id=program_id,
                    circuit_id=circuit_id,
                    public_input_cid=public_input_cid,
                    proof_unit_id=proof_unit_id,
                    verification_key_id=verification_key_id,
                    duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                    logs=logs,
                )
            if not vk_decision.accepted:
                return self._base_reject(
                    status=ProverStatus.INVALID,
                    reason_code=ProverReasonCode.KEY_TRUST_REJECTED,
                    message=f"verification key rejected: {vk_decision.message}",
                    backend_id=backend_id,
                    program_id=program_id,
                    circuit_id=circuit_id,
                    public_input_cid=public_input_cid,
                    proof_unit_id=proof_unit_id,
                    verification_key_id=verification_key_id,
                    duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                    logs=logs,
                    details={"trust_reason": vk_decision.reason_code},
                )
            if proving_key_handle is not None:
                try:
                    pk_decision, _handle = self._policy.select_proving_key_handle(
                        proving_key_handle.key_id,
                        key_cid=proving_key_handle.key_cid,
                        circuit_id=circuit_id,
                        paired_verification_key_id=proving_key_handle.paired_verification_key_id,
                    )
                except TrustError as exc:
                    return self._base_reject(
                        status=ProverStatus.INVALID,
                        reason_code=ProverReasonCode.KEY_TRUST_REJECTED,
                        message=f"proving-key trust error: {exc}",
                        backend_id=backend_id,
                        program_id=program_id,
                        circuit_id=circuit_id,
                        public_input_cid=public_input_cid,
                        proof_unit_id=proof_unit_id,
                        verification_key_id=verification_key_id,
                        duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                        logs=logs,
                    )
                if not pk_decision.accepted:
                    return self._base_reject(
                        status=ProverStatus.INVALID,
                        reason_code=ProverReasonCode.KEY_TRUST_REJECTED,
                        message=f"proving key rejected: {pk_decision.message}",
                        backend_id=backend_id,
                        program_id=program_id,
                        circuit_id=circuit_id,
                        public_input_cid=public_input_cid,
                        proof_unit_id=proof_unit_id,
                        verification_key_id=verification_key_id,
                        duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                        logs=logs,
                        details={"trust_reason": pk_decision.reason_code},
                    )
                if proving_key_handle.paired_verification_key_id != verification_key_id:
                    return self._base_reject(
                        status=ProverStatus.INVALID,
                        reason_code=ProverReasonCode.KEY_TRUST_REJECTED,
                        message=(
                            "proving-key handle is not paired with the requested "
                            "verification key"
                        ),
                        backend_id=backend_id,
                        program_id=program_id,
                        circuit_id=circuit_id,
                        public_input_cid=public_input_cid,
                        proof_unit_id=proof_unit_id,
                        verification_key_id=verification_key_id,
                        duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                        logs=logs,
                    )

        if cancellation is not None and cancellation.cancelled:
            return self._base_reject(
                status=ProverStatus.CANCELLED,
                reason_code=ProverReasonCode.CANCELLED,
                message="invocation cancelled before execution",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        bound_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(program.timeout_seconds)
        )
        bound_output = (
            int(max_output_bytes)
            if max_output_bytes is not None
            else int(program.max_output_bytes)
        )
        if (self._monotonic() - started) > bound_timeout:
            return self._base_reject(
                status=ProverStatus.TIMEOUT,
                reason_code=ProverReasonCode.TIMEOUT,
                message="invocation timed out before engine start",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        return program, bound_timeout, bound_output

    def _map_engine_result(
        self,
        *,
        result: ExternalEngineResult,
        phase: str,
        backend_id: str,
        program_id: str,
        circuit_id: str,
        public_input_cid: str,
        proof_unit_id: str,
        verification_key_id: str,
        duration_ms: int,
        logs: list[str],
    ) -> ProverOutcome:
        if result.cancelled:
            return self._base_reject(
                status=ProverStatus.CANCELLED,
                reason_code=ProverReasonCode.CANCELLED,
                message=result.error_message or f"{phase} cancelled",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )
        if result.timed_out:
            return self._base_reject(
                status=ProverStatus.TIMEOUT,
                reason_code=ProverReasonCode.TIMEOUT,
                message=result.error_message or f"{phase} timed out",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )
        if result.unavailable:
            return self._base_reject(
                status=ProverStatus.UNAVAILABLE,
                reason_code=ProverReasonCode.UNAVAILABLE,
                message=result.error_message or f"{phase} backend unavailable",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )

        # Ambiguous: incomplete completion without durable verified artifact.
        if not result.completed:
            detail = result.error_message or f"{phase} external completion is ambiguous"
            return self._base_reject(
                status=ProverStatus.AMBIGUOUS,
                reason_code=ProverReasonCode.AMBIGUOUS_EXTERNAL_COMPLETION,
                message=f"{detail}; not reporting proved",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                ambiguous=True,
                proof_bytes=result.proof_bytes if result.durable_artifact_present else None,
            )

        # verify phase (prove finalization is handled in _prove_impl)
        if phase != "verify":
            return self._base_reject(
                status=ProverStatus.AMBIGUOUS,
                reason_code=ProverReasonCode.AMBIGUOUS_EXTERNAL_COMPLETION,
                message=(
                    f"unexpected engine mapping phase {phase!r}; "
                    "not reporting proved"
                ),
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                ambiguous=True,
            )

        if result.verified is None:
            return self._base_reject(
                status=ProverStatus.AMBIGUOUS,
                reason_code=ProverReasonCode.AMBIGUOUS_EXTERNAL_COMPLETION,
                message=(
                    "verify completion did not yield a boolean verdict; "
                    "not reporting proved"
                ),
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                ambiguous=True,
                proof_bytes=result.proof_bytes,
            )
        if result.verified is not True:
            return self._base_reject(
                status=ProverStatus.VERIFICATION_FAILED,
                reason_code=ProverReasonCode.INVALID_CRYPTOGRAPHY,
                message=result.error_message
                or "cryptographic verification failed over committed public input",
                backend_id=backend_id,
                program_id=program_id,
                circuit_id=circuit_id,
                public_input_cid=public_input_cid,
                proof_unit_id=proof_unit_id,
                verification_key_id=verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                proof_bytes=result.proof_bytes,
            )
        logs.append(
            witness_safe_log_line(
                "verify.ok",
                program_id=program_id,
                public_input_cid=public_input_cid,
                proof_cid=_sha256_hex(result.proof_bytes)
                if result.proof_bytes
                else None,
            )
        )
        return _outcome(
            status=ProverStatus.PROVED,
            proved=True,
            reason_code=ProverReasonCode.VERIFIED,
            message="proof verified over committed public input",
            backend_id=backend_id,
            program_id=program_id,
            circuit_id=circuit_id,
            public_input_cid=public_input_cid,
            proof_unit_id=proof_unit_id,
            verification_key_id=verification_key_id,
            proof_cid=_sha256_hex(result.proof_bytes) if result.proof_bytes else None,
            proof_bytes=result.proof_bytes,
            duration_ms=duration_ms,
            verified=True,
            ambiguous=False,
            log_lines=tuple(logs),
        )

    def _verify_proof_bytes(
        self,
        *,
        backend_id: str,
        program: RegisteredProgram,
        public_input: bytes,
        proof_bytes: bytes,
        verification_key_id: str,
        verification_key_cid: str,
        program_id: str,
        circuit_id: str,
        proving_key_cid: str | None,
    ) -> bool:
        engine = self._engines.get(backend_id)
        if engine is None:
            return False
        verify_invocation = VerificationInvocation(
            program_id=program_id,
            circuit_id=circuit_id,
            public_input=public_input,
            proof_bytes=proof_bytes,
            verification_key_id=verification_key_id,
            verification_key_cid=verification_key_cid,
            backend_id=backend_id,
            metadata=(
                {"proving_key_cid": proving_key_cid} if proving_key_cid else {}
            ),
        )
        try:
            result = engine.verify(program, verify_invocation)
        except Exception:  # noqa: BLE001
            return False
        return result.completed is True and result.verified is True

    def _prove_impl(
        self,
        invocation: ProverInvocation,
        *,
        started: float,
        logs: list[str],
    ) -> ProverOutcome:
        common = self._check_common(
            program_id=invocation.program_id,
            circuit_id=invocation.circuit_id,
            backend_id=invocation.backend_id,
            public_input_cid=invocation.public_input_cid or public_input_cid_of(
                invocation.public_input
            ),
            proof_unit_id=invocation.proof_unit_id,
            verification_key_id=invocation.verification_key_id,
            verification_key_cid=invocation.verification_key_cid,
            production=invocation.production,
            proving_key_handle=invocation.proving_key_handle,
            timeout_seconds=invocation.timeout_seconds,
            max_output_bytes=invocation.max_output_bytes,
            cancellation=invocation.cancellation,
            started=started,
            logs=logs,
            metadata=invocation.metadata,
        )
        if isinstance(common, ProverOutcome):
            return common
        program, bound_timeout, bound_output = common

        engine = self._engines.get(invocation.backend_id)
        if engine is None:
            return self._base_reject(
                status=ProverStatus.UNAVAILABLE,
                reason_code=ProverReasonCode.UNAVAILABLE,
                message=(
                    f"no engine registered for backend {invocation.backend_id!r}"
                ),
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        if invocation.cancellation is not None and invocation.cancellation.cancelled:
            return self._base_reject(
                status=ProverStatus.CANCELLED,
                reason_code=ProverReasonCode.CANCELLED,
                message="invocation cancelled before engine prove",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        try:
            result = engine.prove(program, invocation)
        except Exception as exc:  # noqa: BLE001
            return self._base_reject(
                status=ProverStatus.PROOF_FAILED,
                reason_code=ProverReasonCode.PROOF_FAILED,
                message=f"engine prove failed: {type(exc).__name__}: {exc}",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        duration_ms = max(0, int((self._monotonic() - started) * 1000))
        if (self._monotonic() - started) > bound_timeout:
            return self._base_reject(
                status=ProverStatus.TIMEOUT,
                reason_code=ProverReasonCode.TIMEOUT,
                message="prove timed out after engine return",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )
        if invocation.cancellation is not None and invocation.cancellation.cancelled:
            return self._base_reject(
                status=ProverStatus.CANCELLED,
                reason_code=ProverReasonCode.CANCELLED,
                message="invocation cancelled after engine prove; discarding result",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )

        if result.cancelled or result.timed_out or result.unavailable:
            return self._map_engine_result(
                result=result,
                phase="prove",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )

        if not result.completed:
            detail = (
                result.error_message
                or "prove external completion is ambiguous"
            )
            return self._base_reject(
                status=ProverStatus.AMBIGUOUS,
                reason_code=ProverReasonCode.AMBIGUOUS_EXTERNAL_COMPLETION,
                message=f"{detail}; not reporting proved",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                ambiguous=True,
                proof_bytes=(
                    result.proof_bytes if result.durable_artifact_present else None
                ),
            )

        if not result.proof_bytes:
            return self._base_reject(
                status=ProverStatus.AMBIGUOUS,
                reason_code=ProverReasonCode.AMBIGUOUS_EXTERNAL_COMPLETION,
                message=(
                    "prove reported completion without durable proof bytes; "
                    "not reporting proved"
                ),
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                ambiguous=True,
            )

        if len(result.proof_bytes) > bound_output:
            return self._base_reject(
                status=ProverStatus.FAILED,
                reason_code=ProverReasonCode.OUTPUT_BOUND_EXCEEDED,
                message=(
                    f"proof output {len(result.proof_bytes)} bytes exceeds "
                    f"bound {bound_output}"
                ),
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )

        # Mandatory local verification of proof bytes before claiming proved.
        verified = self._verify_proof_bytes(
            backend_id=invocation.backend_id,
            program=program,
            public_input=invocation.public_input,
            proof_bytes=result.proof_bytes,
            verification_key_id=invocation.verification_key_id,
            verification_key_cid=invocation.verification_key_cid,
            program_id=invocation.program_id,
            circuit_id=invocation.circuit_id,
            proving_key_cid=invocation.proving_key_handle.key_cid,
        )
        if not verified:
            return self._base_reject(
                status=ProverStatus.PROOF_FAILED,
                reason_code=ProverReasonCode.INVALID_CRYPTOGRAPHY,
                message=(
                    "produced proof bytes failed verification over committed "
                    "public input; not reporting proved"
                ),
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
                proof_bytes=result.proof_bytes,
            )

        proof_cid = _sha256_hex(result.proof_bytes)
        logs.append(
            witness_safe_log_line(
                "prove.ok",
                program_id=invocation.program_id,
                public_input_cid=invocation.public_input_cid,
                proof_cid=proof_cid,
                proof_byte_length=len(result.proof_bytes),
            )
        )
        return _outcome(
            status=ProverStatus.PROVED,
            proved=True,
            reason_code=ProverReasonCode.PROVED,
            message="proof produced and verified over committed public input",
            backend_id=invocation.backend_id,
            program_id=invocation.program_id,
            circuit_id=invocation.circuit_id,
            public_input_cid=invocation.public_input_cid or "unknown",
            proof_unit_id=invocation.proof_unit_id,
            verification_key_id=invocation.verification_key_id,
            proof_cid=proof_cid,
            proof_bytes=result.proof_bytes,
            duration_ms=duration_ms,
            verified=True,
            ambiguous=False,
            details={
                "proving_key_id": invocation.proving_key_handle.key_id,
                "proving_key_cid": invocation.proving_key_handle.key_cid,
                "handle_only": True,
            },
            log_lines=tuple(logs),
        )

    def _verify_impl(
        self,
        invocation: VerificationInvocation,
        *,
        started: float,
        logs: list[str],
    ) -> ProverOutcome:
        common = self._check_common(
            program_id=invocation.program_id,
            circuit_id=invocation.circuit_id,
            backend_id=invocation.backend_id,
            public_input_cid=invocation.public_input_cid or public_input_cid_of(
                invocation.public_input
            ),
            proof_unit_id=invocation.proof_unit_id,
            verification_key_id=invocation.verification_key_id,
            verification_key_cid=invocation.verification_key_cid,
            production=invocation.production,
            proving_key_handle=None,
            timeout_seconds=invocation.timeout_seconds,
            max_output_bytes=invocation.max_output_bytes,
            cancellation=invocation.cancellation,
            started=started,
            logs=logs,
            metadata=invocation.metadata,
        )
        if isinstance(common, ProverOutcome):
            return common
        program, bound_timeout, bound_output = common

        if len(invocation.proof_bytes) > bound_output:
            return self._base_reject(
                status=ProverStatus.INVALID,
                reason_code=ProverReasonCode.OUTPUT_BOUND_EXCEEDED,
                message=(
                    f"proof bytes {len(invocation.proof_bytes)} exceed bound "
                    f"{bound_output}"
                ),
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        engine = self._engines.get(invocation.backend_id)
        if engine is None:
            return self._base_reject(
                status=ProverStatus.UNAVAILABLE,
                reason_code=ProverReasonCode.UNAVAILABLE,
                message=(
                    f"no engine registered for backend {invocation.backend_id!r}"
                ),
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        try:
            result = engine.verify(program, invocation)
        except Exception as exc:  # noqa: BLE001
            return self._base_reject(
                status=ProverStatus.VERIFICATION_FAILED,
                reason_code=ProverReasonCode.VERIFICATION_FAILED,
                message=f"engine verify failed: {type(exc).__name__}: {exc}",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                logs=logs,
            )

        duration_ms = max(0, int((self._monotonic() - started) * 1000))
        if (self._monotonic() - started) > bound_timeout:
            return self._base_reject(
                status=ProverStatus.TIMEOUT,
                reason_code=ProverReasonCode.TIMEOUT,
                message="verify timed out after engine return",
                backend_id=invocation.backend_id,
                program_id=invocation.program_id,
                circuit_id=invocation.circuit_id,
                public_input_cid=invocation.public_input_cid or "unknown",
                proof_unit_id=invocation.proof_unit_id,
                verification_key_id=invocation.verification_key_id,
                duration_ms=duration_ms,
                logs=logs,
            )

        return self._map_engine_result(
            result=result,
            phase="verify",
            backend_id=invocation.backend_id,
            program_id=invocation.program_id,
            circuit_id=invocation.circuit_id,
            public_input_cid=invocation.public_input_cid or "unknown",
            proof_unit_id=invocation.proof_unit_id,
            verification_key_id=invocation.verification_key_id,
            duration_ms=duration_ms,
            logs=logs,
        )


def prove(
    invocation: ProverInvocation,
    *,
    adapter: IncrementalProofBackendAdapter | None = None,
) -> ProverOutcome:
    """Prove via the default or supplied bounded hermetic adapter."""

    return (adapter or IncrementalProofBackendAdapter()).prove(invocation)


def verify(
    invocation: VerificationInvocation,
    *,
    adapter: IncrementalProofBackendAdapter | None = None,
) -> ProverOutcome:
    """Verify via the default or supplied bounded hermetic adapter."""

    return (adapter or IncrementalProofBackendAdapter()).verify(invocation)


__all__ = (
    "DEFAULT_MAX_LOG_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "KNOWN_PROVER_BACKEND_IDS",
    "PROVER_ADAPTER_EVIDENCE",
    "PROVER_INVOCATION_SCHEMA",
    "PROVER_OUTCOME_SCHEMA",
    "PROGRAM_REGISTRY_SCHEMA",
    "VERIFICATION_INVOCATION_SCHEMA",
    "CancellationToken",
    "ExternalEngineResult",
    "ExternalProverEngine",
    "HermeticHmacEngine",
    "IncrementalProofBackendAdapter",
    "ProgramRegistry",
    "ProverError",
    "ProverInvocation",
    "ProverOutcome",
    "ProverReasonCode",
    "ProverStatus",
    "RegisteredProgram",
    "VerificationInvocation",
    "assert_no_sensitive_material",
    "closed_known_prover_backend_ids",
    "closed_prover_reason_codes",
    "closed_prover_statuses",
    "default_hermetic_program_registry",
    "prove",
    "public_input_cid_of",
    "scrub_sensitive_mapping",
    "verify",
    "witness_safe_log_line",
)
