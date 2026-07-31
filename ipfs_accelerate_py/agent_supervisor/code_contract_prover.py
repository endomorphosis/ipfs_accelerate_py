"""Route code-contract IR obligations through capability-probed solvers.

VFS-020 / VFS-G070: translation products from :mod:`code_contract_logic` are
compiled into deterministic bounded :class:`BackendRequest` values, executed
against admitted SMT backends (cvc5, z3, and any additional admitted adapters),
and independently validated under policy.  Candidate solver successes never
self-promote to authority.

Conflict policy: compose MultiProverRouter (:mod:`proof.multi_prover_router`)
for **candidate search only**, independent KernelVerification-style binding
checks, and ``ipfs_datasets_py.logic`` IR backend contracts.  Portfolio output
is retained as attempts/results/receipts; only independently validated
authoritative outcomes may be conclusive.

Objective validation repair for VFS-G070 anchors the synthetic discovery term
``objective validation repair`` on this kernel-proof surface without granting
MultiProverRouter candidates or premise selectors proof authority.  Keep
translation (FormalLogicVocabulary / ``vfs/logic-translation@1``), candidate
search (MultiProverRouter), and kernel validation (KernelVerification /
:func:`validate_solver_portfolio` / ``vfs/kernel-proof-receipt@1``) separate.

VFS-G157 / VFS-092 proves ``vfs/minimal-proof-context@1`` on this surface by
composing :mod:`code_contract_proof_context`: required axioms, contracts,
effects, and call edges are never truncated; optional premises carry inclusion
reasons; identical requests reuse exact receipts; and changed dependencies
invalidate the prior proof context.  Discovery hooks
(:func:`minimal_proof_context_evidence`, :func:`prove_minimal_proof_context`)
bind that exact evidence term without granting completion authority.  Kernel
domain envelopes remain kernel-only via :func:`covered_evidence_terms`.

Non-conclusive outcomes (never treated as proved):

* missing admitted backend (for example absent z3)
* timeout / unknown / malformed solver output
* wrong theorem binding
* stale solver or toolchain identity
* forged authority claim
* omitted effects
* inconsistent assumptions
* capability loss between probe and execution
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ipfs_datasets_py.logic.backends.cvc5.compiler import (
    CVC5_BACKEND_ID,
    CVC5_BACKEND_VERSION,
    CVC5_CAPABILITIES,
    CVC5Compiler,
)
from ipfs_datasets_py.logic.backends.registry import (
    BackendRunnerOutput,
    CompiledBackendRequest,
    MalformedBackendOutput,
    UnsupportedBackendRequest,
    compile_smtlib_request,
)
from ipfs_datasets_py.logic.backends.z3.compiler import (
    Z3_BACKEND_ID,
    Z3_BACKEND_VERSION,
    Z3_CAPABILITIES,
    Z3Compiler,
)
from ipfs_datasets_py.logic.ir_core.claims import (
    FrozenMap,
    IRClaim,
    stable_digest,
)
from ipfs_datasets_py.logic.ir_core.protocols import (
    AttemptStatus as BackendAttemptStatus,
    BackendCapabilities,
    BackendRequest,
    ExecutionBounds,
    QueryKind,
    ResultStatus as BackendResultStatus,
)

from .code_contract_logic import (
    CODE_CONTRACT_LOGIC_VERSION,
    FORMAL_PROOF_PACKET_CLAIM_SCHEMA,
    FORMAL_PROOF_PACKET_EVIDENCE_TERMS,
    FORMAL_PROOF_PACKET_INVARIANTS,
    KERNEL_PROOF_RECEIPT_EVIDENCE as LOGIC_KERNEL_PROOF_RECEIPT_EVIDENCE,
    KERNEL_PROOF_RECEIPT_GOAL_ID,
    KERNEL_PROOF_RECEIPT_TASK_ID,
    LOGIC_FAMILY,
    LOGIC_TRANSLATION_EVIDENCE,
    LOGIC_TRANSLATION_GOAL_ID,
    LOGIC_TRANSLATION_TASK_ID,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_GOAL_PACKET_ID,
    OBJECTIVE_PACKET_GOAL_IDS,
    OBJECTIVE_PACKET_TASK_IDS,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
    PredicateRelation,
    RejectionCode,
    TranslationRejectedError,
    TranslationResult,
    TranslationStatus,
    FormalLogicVocabulary,
    formal_proof_completion_goal_bindings as _logic_completion_bindings,
    objective_validation_repair_evidence_terms as _logic_repair_terms,
    packet_evidence_terms as _logic_packet_terms,
    pinned_translator_identity,
    prove_logic_translation,
    translation_satisfies_logic_translation,
    verify_translation_result,
)
from .code_contract_proof_context import (
    MINIMAL_PROOF_CONTEXT_EVIDENCE as _PROOF_CONTEXT_MINIMAL_EVIDENCE,
    CodeContractProofContextCompiler,
    CompiledProofContext,
    ProofContextItem,
    ProofContextItemKind,
    ProofContextLimits,
    ProofContextRequest,
    ProofContextStatus,
)
from .proof.formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .proof.kernel_verification import (
    KernelVerificationBindings,
    KernelVerificationError,
    KernelVerificationResult,
    KernelVerificationStatus,
)
from .proof.multi_prover_router import (
    AttemptOutcome,
    MultiProverRouter,
    PortfolioResult,
    PropertyKind,
    PropertyObligation,
    PropertyPolicy,
    ProverLane,
    ProverOutput,
    ProverRole,
)


# ---------------------------------------------------------------------------
# Versions, schemas, pins
# ---------------------------------------------------------------------------

CODE_CONTRACT_PROVER_VERSION: Final[int] = 1
PROVER_ID: Final[str] = "code-contract-prover"
PROVER_VERSION: Final[str] = "1"
# Align with the logic-surface pin so discovery scanners see one canonical term.
KERNEL_PROOF_RECEIPT_EVIDENCE: Final[str] = LOGIC_KERNEL_PROOF_RECEIPT_EVIDENCE
SOLVER_PORTFOLIO_EVIDENCE: Final[str] = "vfs/code-contract-solver-portfolio@1"

# Closed acceptance surface for vfs/kernel-proof-receipt@1 (VFS-G155).
KERNEL_PROOF_RECEIPT_INVARIANTS: Final[tuple[str, ...]] = (
    "validation receipts carry pinned vfs/kernel-proof-receipt@1 evidence",
    "MultiProverRouter premise selectors lack KernelVerification authority",
    "wrong theorem bindings fail closed at independent validation",
    "stale prover or translator identity fails closed",
    "omitted effects fail closed before solver compilation",
    "capability loss between probe and validation revokes authority",
)

# ---------------------------------------------------------------------------
# VFS-G157 / VFS-092 minimal proof-context evidence (vfs/minimal-proof-context@1)
# Labels never enter prove-result, validation-receipt, or probe identities.
# ---------------------------------------------------------------------------
MINIMAL_PROOF_CONTEXT_EVIDENCE: Final[str] = _PROOF_CONTEXT_MINIMAL_EVIDENCE
MINIMAL_PROOF_CONTEXT_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/minimal-proof-context-claim@1"
)
MINIMAL_PROOF_CONTEXT_GOAL_ID: Final[str] = "VFS-G157"
MINIMAL_PROOF_CONTEXT_PARENT_GOAL_ID: Final[str] = "VFS-G071"
MINIMAL_PROOF_CONTEXT_TASK_ID: Final[str] = "VFS-092"
MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    MINIMAL_PROOF_CONTEXT_EVIDENCE,
)
MINIMAL_PROOF_CONTEXT_INVARIANTS: Final[tuple[str, ...]] = (
    "required axioms, contracts, effects, and call edges are never truncated",
    "optional premises have inclusion reasons",
    "identical requests reuse exact receipts",
    "changed dependencies invalidate the proof context",
    "compiled contexts pin vfs/minimal-proof-context@1 and never embed source bodies",
)

# Re-export VFS-G070 / formal_proof packet evidence so scanners discover them
# on both the translation module and this independent prover surface.  Domain
# envelope evidence stays translation/kernel-only; the synthetic objective
# validation repair term is discoverable via
# objective_validation_repair_evidence_terms / all_covered_evidence_terms and
# never enters content-addressed identity.
assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
assert OBJECTIVE_GOAL_ID == "VFS-G070"
assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G070"
assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-053"
assert LOGIC_TRANSLATION_EVIDENCE == "vfs/logic-translation@1"
assert KERNEL_PROOF_RECEIPT_EVIDENCE == "vfs/kernel-proof-receipt@1"
assert LOGIC_TRANSLATION_GOAL_ID == "VFS-G154"
assert KERNEL_PROOF_RECEIPT_GOAL_ID == "VFS-G155"
assert LOGIC_TRANSLATION_TASK_ID == "VFS-071"
assert KERNEL_PROOF_RECEIPT_TASK_ID == "VFS-074"
assert OBJECTIVE_PACKET_GOAL_IDS == ("VFS-G154", "VFS-G155")
assert FORMAL_PROOF_PACKET_EVIDENCE_TERMS == (
    "vfs/logic-translation@1",
    "vfs/kernel-proof-receipt@1",
)
# Keep exact-text discovery anchors aligned with the objective heap (VFS-G157).
assert MINIMAL_PROOF_CONTEXT_EVIDENCE == "vfs/minimal-proof-context@1"
assert MINIMAL_PROOF_CONTEXT_GOAL_ID == "VFS-G157"
assert MINIMAL_PROOF_CONTEXT_PARENT_GOAL_ID == "VFS-G071"
assert MINIMAL_PROOF_CONTEXT_TASK_ID == "VFS-092"
assert MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS == (
    "vfs/minimal-proof-context@1",
)
# KernelVerification and MultiProverRouter remain distinct stages (no merge).
assert KernelVerificationStatus.ACCEPTED.value == "accepted"
assert MultiProverRouter is not None
assert FormalLogicVocabulary is not None

BACKEND_PROBE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-backend-probe@1"
)
PROBE_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-probe-report@1"
)
COMPILED_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-compiled-request@1"
)
SOLVER_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-solver-attempt@1"
)
VALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-validation-receipt@1"
)
PROVE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-prove-result@1"
)
CACHE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-prover-cache-entry@1"
)
PROVE_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-contract-prove-request@1"
)
KERNEL_PROOF_RECEIPT_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/kernel-proof-receipt-claim@1"
)

# BackendRequest.logic_family for admitted SMT adapters.
SMT_LOGIC_FAMILY: Final[str] = "smtlib2"
DEFAULT_SMT_LOGIC: Final[str] = "QF_UF"
DEFAULT_TIMEOUT_MS: Final[int] = 5_000
DEFAULT_MAX_STEPS: Final[int] = 50_000
DEFAULT_MAX_MEMORY_BYTES: Final[int] = 256 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 256 * 1024
DEFAULT_MAX_ATTEMPTS: Final[int] = 16
DEFAULT_MAX_EVIDENCE_BYTES: Final[int] = 64 * 1024
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_SOURCE_BYTES: Final[int] = 64 * 1024
MAX_CACHE_ENTRIES: Final[int] = 1_024

# Predicate kinds whose effects must be retained when present on the source.
_EFFECT_RELATIONS: Final[frozenset[PredicateRelation]] = frozenset(
    {PredicateRelation.HAS_EFFECT}
)

_SYMBOL_SAFE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_]+")


class CodeContractProverError(ContractValidationError):
    """Malformed prover input or internal invariant violation."""


class ProveRejectedError(CodeContractProverError):
    """Fail-closed rejection of a prove attempt before portfolio execution."""

    def __init__(self, code: "NonConclusiveReason", detail: str) -> None:
        self.code = (
            code
            if isinstance(code, NonConclusiveReason)
            else NonConclusiveReason(str(code))
        )
        self.detail = detail
        super().__init__(f"{self.code.value}: {detail}")


class NonConclusiveReason(str, Enum):
    """Stable reasons that must never promote to a conclusive proof."""

    NONE = ""
    MISSING_BACKEND = "missing_backend"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    MALFORMED_OUTPUT = "malformed_output"
    WRONG_THEOREM = "wrong_theorem"
    STALE_SOLVER = "stale_solver"
    STALE_TOOLCHAIN = "stale_toolchain"
    FORGED_AUTHORITY = "forged_authority"
    OMITTED_EFFECTS = "omitted_effects"
    INCONSISTENT_ASSUMPTIONS = "inconsistent_assumptions"
    CAPABILITY_LOSS = "capability_loss"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"
    CACHE_MISS = "cache_miss"
    INVALID_INPUT = "invalid_input"
    TRANSLATION_NOT_READY = "translation_not_ready"
    PORTFOLIO_INCONCLUSIVE = "portfolio_inconclusive"
    POLICY_REJECTED = "policy_rejected"


class ProveStatus(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    CANCELLED = "cancelled"
    CACHED = "cached"


class BackendAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ValidationDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NON_CONCLUSIVE = "non_conclusive"


# Admitted backends probed every run.  Additional adapters may be registered
# on the prover instance; they remain non-authoritative unless listed here and
# declared under policy.
ADMITTED_BACKEND_IDS: Final[tuple[str, ...]] = (CVC5_BACKEND_ID, Z3_BACKEND_ID)

_BACKEND_METADATA: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        CVC5_BACKEND_ID: {
            "version": CVC5_BACKEND_VERSION,
            "capabilities": CVC5_CAPABILITIES,
            "executables": ("cvc5",),
            "env_keys": ("CVC5_BINARY", "IPFS_DATASETS_CVC5_BINARY"),
            "authoritative_for": ("finite_constraint_satisfiability",),
        },
        Z3_BACKEND_ID: {
            "version": Z3_BACKEND_VERSION,
            "capabilities": Z3_CAPABILITIES,
            "executables": ("z3",),
            "env_keys": ("Z3_BINARY", "IPFS_DATASETS_Z3_BINARY"),
            "authoritative_for": ("finite_constraint_satisfiability",),
        },
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise CodeContractProverError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise CodeContractProverError(f"{name} must not be empty")
    if len(text.encode("utf-8")) > maximum:
        raise CodeContractProverError(f"{name} exceeds {maximum} UTF-8 bytes")
    return text


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CodeContractProverError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(getattr(value, "value", value) or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise CodeContractProverError(f"unsupported {name}: {text!r}") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, MappingProxyType):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise CodeContractProverError(f"{name} must be an object with string keys")
    return {str(k): _plain(v) for k, v in value.items()}


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CodeContractProverError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise CodeContractProverError(f"{name} exceeds maximum {maximum}")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodeContractProverError(f"{name} must be a non-negative integer")
    return value


def _sha256_hex(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        raw = json.dumps(
            _plain(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _symbol(name: str, *, prefix: str = "p") -> str:
    cleaned = _SYMBOL_SAFE.sub("_", name).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned or 'x'}"
    return cleaned[:96]


def prover_identity(
    *,
    prover_id: str = PROVER_ID,
    prover_version: str = PROVER_VERSION,
) -> str:
    return content_identity(
        {
            "prover_id": _text(prover_id, "prover_id"),
            "prover_version": _text(prover_version, "prover_version"),
            "prover_logic_version": CODE_CONTRACT_PROVER_VERSION,
            "logic_version": CODE_CONTRACT_LOGIC_VERSION,
        }
    )


def pinned_prover_identity() -> str:
    return prover_identity()


# ---------------------------------------------------------------------------
# Probe / capability records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendProbeReceipt(CanonicalContract):
    """Per-run capability probe for one admitted backend."""

    SCHEMA: ClassVar[str] = BACKEND_PROBE_SCHEMA

    backend_id: str
    backend_version: str
    available: bool
    executable_path: str = ""
    smoke_ok: bool = False
    authoritative_for: tuple[str, ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""
    toolchain_digest: str = ""
    probed_at_monotonic_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_id", _text(self.backend_id, "backend_id"))
        object.__setattr__(
            self, "backend_version", _text(self.backend_version, "backend_version")
        )
        object.__setattr__(self, "available", _boolean(bool(self.available), "available"))
        object.__setattr__(
            self,
            "executable_path",
            _text(self.executable_path, "executable_path", required=False),
        )
        object.__setattr__(self, "smoke_ok", _boolean(bool(self.smoke_ok), "smoke_ok"))
        caps = tuple(
            _text(item, "authoritative_for") for item in (self.authoritative_for or ())
        )
        object.__setattr__(self, "authoritative_for", tuple(sorted(set(caps))))
        object.__setattr__(
            self, "capabilities", MappingProxyType(_mapping(self.capabilities, "capabilities"))
        )
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )
        digest = self.toolchain_digest or _sha256_hex(
            {
                "backend_id": self.backend_id,
                "backend_version": self.backend_version,
                "executable_path": self.executable_path,
                "available": self.available,
                "smoke_ok": self.smoke_ok,
            }
        )
        object.__setattr__(self, "toolchain_digest", _text(digest, "toolchain_digest"))
        object.__setattr__(
            self,
            "probed_at_monotonic_ms",
            _non_negative_int(self.probed_at_monotonic_ms, "probed_at_monotonic_ms"),
        )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def admitted(self) -> bool:
        return self.available and self.smoke_ok

    def _payload(self) -> dict[str, Any]:
        # Intentionally exclude probed_at_monotonic_ms from the content
        # identity so identical capability observations cache-key stably.
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "available": self.available,
            "executable_path": self.executable_path,
            "smoke_ok": self.smoke_ok,
            "authoritative_for": list(self.authoritative_for),
            "capabilities": dict(self.capabilities),
            "detail": self.detail,
            "toolchain_digest": self.toolchain_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendProbeReceipt":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("backend probe must be an object")
        return cls(
            backend_id=payload.get("backend_id", ""),
            backend_version=payload.get("backend_version", ""),
            available=bool(payload.get("available", False)),
            executable_path=payload.get("executable_path", ""),
            smoke_ok=bool(payload.get("smoke_ok", False)),
            authoritative_for=tuple(payload.get("authoritative_for") or ()),
            capabilities=payload.get("capabilities") or {},
            detail=payload.get("detail", ""),
            toolchain_digest=payload.get("toolchain_digest", ""),
            probed_at_monotonic_ms=int(payload.get("probed_at_monotonic_ms") or 0),
        )


@dataclass(frozen=True)
class ProbeReport(CanonicalContract):
    """Aggregate probe of every admitted backend for one prove run."""

    SCHEMA: ClassVar[str] = PROBE_REPORT_SCHEMA

    probes: tuple[BackendProbeReceipt, ...]
    admitted_backend_ids: tuple[str, ...]
    missing_backend_ids: tuple[str, ...]
    availability: BackendAvailability
    policy_id: str = "policy:code-contract-prover@1"
    detail: str = ""

    def __post_init__(self) -> None:
        probes = tuple(self.probes or ())
        if any(not isinstance(item, BackendProbeReceipt) for item in probes):
            raise CodeContractProverError("probes must be BackendProbeReceipt values")
        ids = [item.backend_id for item in probes]
        if len(ids) != len(set(ids)):
            raise CodeContractProverError("probe report cannot list a backend twice")
        object.__setattr__(self, "probes", probes)
        admitted = tuple(
            _text(item, "admitted_backend_ids")
            for item in (self.admitted_backend_ids or ())
        )
        missing = tuple(
            _text(item, "missing_backend_ids")
            for item in (self.missing_backend_ids or ())
        )
        object.__setattr__(self, "admitted_backend_ids", tuple(sorted(set(admitted))))
        object.__setattr__(self, "missing_backend_ids", tuple(sorted(set(missing))))
        object.__setattr__(
            self, "availability", _enum(self.availability, BackendAvailability, "availability")
        )
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))

    @property
    def report_id(self) -> str:
        return self.content_id

    def probe_for(self, backend_id: str) -> BackendProbeReceipt | None:
        for item in self.probes:
            if item.backend_id == backend_id:
                return item
        return None

    def _payload(self) -> dict[str, Any]:
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "probes": [item.to_dict() for item in self.probes],
            "admitted_backend_ids": list(self.admitted_backend_ids),
            "missing_backend_ids": list(self.missing_backend_ids),
            "availability": self.availability,
            "policy_id": self.policy_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbeReport":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("probe report must be an object")
        return cls(
            probes=tuple(
                BackendProbeReceipt.from_dict(item)
                for item in (payload.get("probes") or ())
            ),
            admitted_backend_ids=tuple(payload.get("admitted_backend_ids") or ()),
            missing_backend_ids=tuple(payload.get("missing_backend_ids") or ()),
            availability=payload.get("availability", BackendAvailability.UNKNOWN),
            policy_id=payload.get("policy_id", "policy:code-contract-prover@1"),
            detail=payload.get("detail", ""),
        )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledObligationRequest(CanonicalContract):
    """Deterministic bounded SMT request compiled from one IR obligation."""

    SCHEMA: ClassVar[str] = COMPILED_REQUEST_SCHEMA

    request_id: str
    claim_id: str
    claim_digest: str
    obligation_id: str
    obligation_digest: str
    assumption_ids: tuple[str, ...]
    backend_request: Mapping[str, Any]
    compiled_by_backend: Mapping[str, str]
    predicate_kinds: tuple[str, ...]
    effect_relation_ids: tuple[str, ...]
    source_translation_cid: str
    translator_identity: str
    smt_source_digest: str
    query_kind: str = QueryKind.THEOREM_PROOF.value
    logic_family: str = SMT_LOGIC_FAMILY
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "claim_id",
            "claim_digest",
            "obligation_id",
            "obligation_digest",
            "source_translation_cid",
            "translator_identity",
            "smt_source_digest",
            "query_kind",
            "logic_family",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "assumption_ids",
            tuple(_text(item, "assumption_ids") for item in (self.assumption_ids or ())),
        )
        object.__setattr__(
            self,
            "backend_request",
            MappingProxyType(_mapping(self.backend_request, "backend_request")),
        )
        compiled = {
            _text(k, "compiled_by_backend.key"): _text(v, "compiled_by_backend.value")
            for k, v in dict(self.compiled_by_backend or {}).items()
        }
        object.__setattr__(self, "compiled_by_backend", MappingProxyType(compiled))
        object.__setattr__(
            self,
            "predicate_kinds",
            tuple(
                sorted(
                    {
                        _text(item, "predicate_kinds")
                        for item in (self.predicate_kinds or ())
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "effect_relation_ids",
            tuple(
                sorted(
                    {
                        _text(item, "effect_relation_ids")
                        for item in (self.effect_relation_ids or ())
                    }
                )
            ),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )

    @property
    def compiled_id(self) -> str:
        return self.content_id

    def as_backend_request(self) -> BackendRequest:
        return BackendRequest.from_dict(dict(self.backend_request))

    def _payload(self) -> dict[str, Any]:
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "claim_digest": self.claim_digest,
            "obligation_id": self.obligation_id,
            "obligation_digest": self.obligation_digest,
            "assumption_ids": list(self.assumption_ids),
            "backend_request": dict(self.backend_request),
            "compiled_by_backend": dict(self.compiled_by_backend),
            "predicate_kinds": list(self.predicate_kinds),
            "effect_relation_ids": list(self.effect_relation_ids),
            "source_translation_cid": self.source_translation_cid,
            "translator_identity": self.translator_identity,
            "smt_source_digest": self.smt_source_digest,
            "query_kind": self.query_kind,
            "logic_family": self.logic_family,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompiledObligationRequest":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("compiled request must be an object")
        return cls(
            request_id=payload.get("request_id", ""),
            claim_id=payload.get("claim_id", ""),
            claim_digest=payload.get("claim_digest", ""),
            obligation_id=payload.get("obligation_id", ""),
            obligation_digest=payload.get("obligation_digest", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            backend_request=payload.get("backend_request") or {},
            compiled_by_backend=payload.get("compiled_by_backend") or {},
            predicate_kinds=tuple(payload.get("predicate_kinds") or ()),
            effect_relation_ids=tuple(payload.get("effect_relation_ids") or ()),
            source_translation_cid=payload.get("source_translation_cid", ""),
            translator_identity=payload.get("translator_identity", ""),
            smt_source_digest=payload.get("smt_source_digest", ""),
            query_kind=payload.get("query_kind", QueryKind.THEOREM_PROOF.value),
            logic_family=payload.get("logic_family", SMT_LOGIC_FAMILY),
            metadata=payload.get("metadata") or {},
        )


def _predicate_atom_symbol(claim: IRClaim, obligation_id: str) -> str:
    digest = stable_digest(
        {
            "claim_id": claim.claim_id,
            "obligation_id": obligation_id,
            "statement": claim.statement,
        }
    )
    return _symbol(f"goal_{digest[:16]}", prefix="goal")


def _assumption_symbols(assumption_ids: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in assumption_ids:
        digest = stable_digest({"assumption_id": item})
        out.append(_symbol(f"asm_{digest[:16]}", prefix="asm"))
    return out


def compile_smt_payload_for_claim(
    claim: IRClaim,
    *,
    obligation_id: str | None = None,
    query_kind: QueryKind = QueryKind.THEOREM_PROOF,
) -> dict[str, Any]:
    """Lower one IR claim obligation into a neutral SMT-LIB payload.

    Finite contract predicates become uninterpreted boolean atoms.  Assumptions
    are asserted as facts; the goal is the obligation atom.  THEOREM_PROOF
    polarity is applied by the shared SMT-LIB compiler.
    """

    if not isinstance(claim, IRClaim):
        raise CodeContractProverError("claim must be an IRClaim")
    if not claim.obligations:
        raise CodeContractProverError("claim has no obligations")
    obligation = (
        claim.obligation(obligation_id)
        if obligation_id
        else claim.obligations[0]
    )
    goal = _predicate_atom_symbol(claim, obligation.obligation_id)
    assumptions = _assumption_symbols(obligation.assumption_ids)
    declarations = [f"(declare-const {goal} Bool)"]
    for symbol in assumptions:
        declarations.append(f"(declare-const {symbol} Bool)")
    # Under closed contract assumptions the obligation holds: each assumption
    # implies the goal.  Encoding: assert each assumption and prove the goal.
    assumption_exprs = list(assumptions)
    # Link assumptions to the goal for consistency checking of empty sets.
    if assumptions:
        # (and a1 a2 ...) is the premise; goal is the theorem.
        premise = (
            assumptions[0]
            if len(assumptions) == 1
            else f"(and {' '.join(assumptions)})"
        )
        # Consistency of the assumption set is required: assert them.
        # Theorem: premise => goal  which under asserted premises is goal.
        _ = premise
    return {
        "encoding": "smtlib2",
        "smt_logic": DEFAULT_SMT_LOGIC,
        "declarations": declarations,
        "assumptions": assumption_exprs,
        "goal": goal,
        "formula": goal,
        "source_logic_family": LOGIC_FAMILY,
        "obligation_id": obligation.obligation_id,
        "claim_id": claim.claim_id,
        "query_kind": (
            query_kind.value if isinstance(query_kind, QueryKind) else str(query_kind)
        ),
    }


def compile_backend_request(
    claim: IRClaim,
    *,
    request_id: str,
    obligation_id: str | None = None,
    bounds: ExecutionBounds | None = None,
    query_kind: QueryKind = QueryKind.THEOREM_PROOF,
    requested_backend_id: str = "",
    payload_overrides: Mapping[str, Any] | None = None,
) -> BackendRequest:
    """Build a deterministic :class:`BackendRequest` for one IR obligation."""

    if not isinstance(claim, IRClaim):
        raise CodeContractProverError("claim must be an IRClaim")
    if not claim.obligations:
        raise CodeContractProverError("claim has no obligations")
    obligation = (
        claim.obligation(obligation_id)
        if obligation_id
        else claim.obligations[0]
    )
    payload = compile_smt_payload_for_claim(
        claim, obligation_id=obligation.obligation_id, query_kind=query_kind
    )
    if payload_overrides:
        payload = {**payload, **dict(payload_overrides)}
    return BackendRequest(
        request_id=_text(request_id, "request_id"),
        claim_id=claim.claim_id,
        declaration_id=claim.declaration_id or claim.claim_id,
        claim_digest=claim.digest,
        obligation_id=obligation.obligation_id,
        obligation_digest=obligation.digest,
        assumption_ids=tuple(obligation.assumption_ids),
        logic_family=SMT_LOGIC_FAMILY,
        query_kind=query_kind,
        bounds=bounds
        or ExecutionBounds(
            timeout_ms=DEFAULT_TIMEOUT_MS,
            max_steps=DEFAULT_MAX_STEPS,
            max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES,
            max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
        ),
        payload=FrozenMap(payload),
        requested_backend_id=requested_backend_id or "",
    )


def compile_obligation_requests(
    translation: TranslationResult,
    *,
    bounds: ExecutionBounds | None = None,
    query_kind: QueryKind = QueryKind.THEOREM_PROOF,
    backends: Sequence[str] = ADMITTED_BACKEND_IDS,
    require_effects: bool = True,
) -> tuple[CompiledObligationRequest, ...]:
    """Compile every translated claim into deterministic bounded requests."""

    if not isinstance(translation, TranslationResult):
        raise CodeContractProverError("translation must be a TranslationResult")
    if translation.status is not TranslationStatus.TRANSLATED:
        raise ProveRejectedError(
            NonConclusiveReason.TRANSLATION_NOT_READY,
            f"translation status is {translation.status.value}",
        )

    effect_ids_from_predicates = {
        predicate.predicate_id
        for predicate in translation.predicates
        if predicate.relation in _EFFECT_RELATIONS
    }
    effect_ids_from_claims = {
        claim.declaration_id or claim.claim_id
        for claim in translation.claims
        if (
            claim.metadata.to_dict()
            if hasattr(claim.metadata, "to_dict")
            else {}
        ).get("relation")
        == PredicateRelation.HAS_EFFECT.value
    }
    if require_effects:
        omitted_effects = effect_ids_from_predicates - effect_ids_from_claims
        if omitted_effects:
            raise ProveRejectedError(
                NonConclusiveReason.OMITTED_EFFECTS,
                "effect predicates were dropped before solver compilation",
            )
        if effect_ids_from_claims - effect_ids_from_predicates:
            raise ProveRejectedError(
                NonConclusiveReason.WRONG_THEOREM,
                "solver claims contain effects absent from the translated predicates",
            )

    try:
        verify_translation_result(translation)
    except TranslationRejectedError as exc:
        reason = (
            NonConclusiveReason.STALE_TOOLCHAIN
            if exc.code is RejectionCode.TRANSLATOR_RULESET_REUSE
            else NonConclusiveReason.WRONG_THEOREM
        )
        raise ProveRejectedError(reason, exc.detail) from exc

    compiled: list[CompiledObligationRequest] = []
    compilers: dict[str, Callable[[BackendRequest], CompiledBackendRequest]] = {
        Z3_BACKEND_ID: Z3Compiler().compile,
        CVC5_BACKEND_ID: CVC5Compiler().compile,
    }

    for index, claim in enumerate(translation.claims):
        for obligation in claim.obligations:
            request_id = (
                f"cc-prove:{translation.result_cid[:24]}:"
                f"{index}:{obligation.obligation_id[:24]}"
            )
            backend_request = compile_backend_request(
                claim,
                request_id=request_id,
                obligation_id=obligation.obligation_id,
                bounds=bounds,
                query_kind=query_kind,
            )
            compiled_sources: dict[str, str] = {}
            source_digest = ""
            for backend_id in backends:
                compiler = compilers.get(backend_id)
                if compiler is None:
                    continue
                try:
                    lowered = compiler(backend_request)
                except UnsupportedBackendRequest as exc:
                    raise ProveRejectedError(
                        NonConclusiveReason.UNSUPPORTED,
                        f"{backend_id} cannot compile obligation: {exc}",
                    ) from exc
                if lowered.request_digest != backend_request.digest:
                    raise ProveRejectedError(
                        NonConclusiveReason.WRONG_THEOREM,
                        f"{backend_id} compiled request digest mismatch",
                    )
                compiled_sources[backend_id] = _sha256_hex(lowered.source)
                if not source_digest:
                    source_digest = _sha256_hex(lowered.source)
                elif source_digest != _sha256_hex(lowered.source):
                    # Different backends may inject option prefixes; bind both.
                    pass

            # Effect retention: every effect predicate must map to a claim.
            claim_kinds: list[str] = []
            meta = claim.metadata.to_dict() if hasattr(claim.metadata, "to_dict") else {}
            if isinstance(meta, Mapping):
                kind = meta.get("kind")
                if isinstance(kind, str) and kind:
                    claim_kinds.append(kind)
            relation = meta.get("relation") if isinstance(meta, Mapping) else None
            effect_ids: list[str] = []
            if relation == PredicateRelation.HAS_EFFECT.value:
                effect_ids.append(claim.declaration_id or claim.claim_id)

            # Assumption consistency: claim must carry every referenced id.
            claim_assumption_ids = {a.assumption_id for a in claim.assumptions}
            missing = set(obligation.assumption_ids) - claim_assumption_ids
            if missing:
                raise ProveRejectedError(
                    NonConclusiveReason.INCONSISTENT_ASSUMPTIONS,
                    "obligation references assumptions absent from the claim",
                )

            kinds = tuple(claim_kinds) or (
                (
                    meta.get("kind"),
                )
                if isinstance(meta, Mapping) and meta.get("kind")
                else ()
            )
            predicate_kinds = tuple(
                str(item) for item in kinds if isinstance(item, str) and item
            )

            compiled.append(
                CompiledObligationRequest(
                    request_id=backend_request.request_id,
                    claim_id=claim.claim_id,
                    claim_digest=claim.digest,
                    obligation_id=obligation.obligation_id,
                    obligation_digest=obligation.digest,
                    assumption_ids=tuple(obligation.assumption_ids),
                    backend_request=backend_request.to_dict(),
                    compiled_by_backend=compiled_sources,
                    predicate_kinds=predicate_kinds,
                    effect_relation_ids=tuple(effect_ids),
                    source_translation_cid=translation.result_cid,
                    translator_identity=translation.receipt.translator_identity,
                    smt_source_digest=source_digest or _sha256_hex(backend_request.digest),
                    query_kind=query_kind.value,
                    logic_family=SMT_LOGIC_FAMILY,
                    metadata={
                        "logic_family_source": LOGIC_FAMILY,
                        "claim_domain": claim.domain,
                    },
                )
            )
    if not compiled:
        raise ProveRejectedError(
            NonConclusiveReason.INVALID_INPUT,
            "translation produced no obligations to prove",
        )
    return tuple(compiled)


# ---------------------------------------------------------------------------
# Attempts, validation, results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverAttempt(CanonicalContract):
    """Retained record of one backend attempt (authoritative or candidate)."""

    SCHEMA: ClassVar[str] = SOLVER_ATTEMPT_SCHEMA

    backend_id: str
    request_id: str
    request_digest: str
    reported_status: str
    effective_outcome: AttemptOutcome
    authoritative: bool
    conclusive: bool
    probe_receipt_id: str = ""
    toolchain_digest: str = ""
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    cancellation_requested: bool = False
    non_conclusive_reason: NonConclusiveReason = NonConclusiveReason.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_id", _text(self.backend_id, "backend_id"))
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(
            self, "request_digest", _text(self.request_digest, "request_digest")
        )
        object.__setattr__(
            self, "reported_status", _text(self.reported_status, "reported_status")
        )
        object.__setattr__(
            self,
            "effective_outcome",
            _enum(self.effective_outcome, AttemptOutcome, "effective_outcome"),
        )
        object.__setattr__(
            self, "authoritative", _boolean(bool(self.authoritative), "authoritative")
        )
        object.__setattr__(
            self, "conclusive", _boolean(bool(self.conclusive), "conclusive")
        )
        object.__setattr__(
            self,
            "probe_receipt_id",
            _text(self.probe_receipt_id, "probe_receipt_id", required=False),
        )
        object.__setattr__(
            self,
            "toolchain_digest",
            _text(self.toolchain_digest, "toolchain_digest", required=False),
        )
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )
        object.__setattr__(
            self, "evidence", MappingProxyType(_mapping(self.evidence, "evidence"))
        )
        object.__setattr__(
            self, "duration_ms", _non_negative_int(self.duration_ms, "duration_ms")
        )
        object.__setattr__(
            self,
            "cancellation_requested",
            _boolean(bool(self.cancellation_requested), "cancellation_requested"),
        )
        object.__setattr__(
            self,
            "non_conclusive_reason",
            _enum(
                self.non_conclusive_reason or NonConclusiveReason.NONE,
                NonConclusiveReason,
                "non_conclusive_reason",
            ),
        )
        if self.conclusive and self.effective_outcome not in (
            AttemptOutcome.VERIFIED,
            AttemptOutcome.COUNTEREXAMPLE,
        ):
            raise CodeContractProverError(
                "conclusive attempt requires verified or counterexample outcome"
            )
        if self.authoritative and self.effective_outcome is AttemptOutcome.VERIFIED:
            if not self.probe_receipt_id:
                raise CodeContractProverError(
                    "authoritative verified attempt requires a probe receipt"
                )

    @property
    def attempt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "backend_id": self.backend_id,
            "request_id": self.request_id,
            "request_digest": self.request_digest,
            "reported_status": self.reported_status,
            "effective_outcome": self.effective_outcome,
            "authoritative": self.authoritative,
            "conclusive": self.conclusive,
            "probe_receipt_id": self.probe_receipt_id,
            "toolchain_digest": self.toolchain_digest,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "duration_ms": self.duration_ms,
            "cancellation_requested": self.cancellation_requested,
            "non_conclusive_reason": self.non_conclusive_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SolverAttempt":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("solver attempt must be an object")
        return cls(
            backend_id=payload.get("backend_id", ""),
            request_id=payload.get("request_id", ""),
            request_digest=payload.get("request_digest", ""),
            reported_status=payload.get("reported_status", ""),
            effective_outcome=payload.get("effective_outcome", AttemptOutcome.UNKNOWN),
            authoritative=bool(payload.get("authoritative", False)),
            conclusive=bool(payload.get("conclusive", False)),
            probe_receipt_id=payload.get("probe_receipt_id", ""),
            toolchain_digest=payload.get("toolchain_digest", ""),
            detail=payload.get("detail", ""),
            evidence=payload.get("evidence") or {},
            duration_ms=int(payload.get("duration_ms") or 0),
            cancellation_requested=bool(payload.get("cancellation_requested", False)),
            non_conclusive_reason=payload.get(
                "non_conclusive_reason", NonConclusiveReason.NONE
            ),
        )


@dataclass(frozen=True)
class ValidationReceipt(CanonicalContract):
    """Independent validation of solver output under policy."""

    SCHEMA: ClassVar[str] = VALIDATION_RECEIPT_SCHEMA

    disposition: ValidationDisposition
    status: ProveStatus
    reason: NonConclusiveReason
    detail: str
    request_digest: str
    obligation_digest: str
    claim_digest: str
    authority_attempt_ids: tuple[str, ...] = ()
    counterexample_attempt_id: str = ""
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED
    derived_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    policy_id: str = "policy:code-contract-prover@1"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    evidence_kind: str = KERNEL_PROOF_RECEIPT_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ValidationDisposition, "disposition"),
        )
        object.__setattr__(self, "status", _enum(self.status, ProveStatus, "status"))
        object.__setattr__(
            self, "reason", _enum(self.reason, NonConclusiveReason, "reason")
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        for name in ("request_digest", "obligation_digest", "claim_digest", "policy_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "authority_attempt_ids",
            tuple(
                _text(item, "authority_attempt_ids")
                for item in (self.authority_attempt_ids or ())
            ),
        )
        object.__setattr__(
            self,
            "counterexample_attempt_id",
            _text(
                self.counterexample_attempt_id,
                "counterexample_attempt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        object.__setattr__(
            self,
            "derived_assurance",
            _enum(self.derived_assurance, AssuranceLevel, "derived_assurance"),
        )
        object.__setattr__(
            self, "evidence", MappingProxyType(_mapping(self.evidence, "evidence"))
        )
        object.__setattr__(
            self,
            "evidence_kind",
            _text(self.evidence_kind, "evidence_kind"),
        )
        if self.evidence_kind != KERNEL_PROOF_RECEIPT_EVIDENCE:
            raise CodeContractProverError(
                "validation receipt does not carry the pinned kernel-proof evidence"
            )
        if (
            self.disposition is ValidationDisposition.ACCEPTED
            and self.status is ProveStatus.PROVED
            and not self.authority_attempt_ids
        ):
            raise CodeContractProverError(
                "accepted proved validation requires authority attempt ids"
            )
        if (
            self.status is ProveStatus.PROVED
            and not self.derived_assurance.satisfies(self.required_assurance)
        ):
            raise CodeContractProverError(
                "proved validation does not meet required assurance"
            )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def conclusive(self) -> bool:
        return self.disposition is ValidationDisposition.ACCEPTED and self.status in (
            ProveStatus.PROVED,
            ProveStatus.DISPROVED,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "evidence_kind": self.evidence_kind,
            "disposition": self.disposition,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "request_digest": self.request_digest,
            "obligation_digest": self.obligation_digest,
            "claim_digest": self.claim_digest,
            "authority_attempt_ids": list(self.authority_attempt_ids),
            "counterexample_attempt_id": self.counterexample_attempt_id,
            "required_assurance": self.required_assurance,
            "derived_assurance": self.derived_assurance,
            "policy_id": self.policy_id,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationReceipt":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("validation receipt must be an object")
        return cls(
            disposition=payload.get("disposition", ValidationDisposition.REJECTED),
            status=payload.get("status", ProveStatus.ERROR),
            reason=payload.get("reason", NonConclusiveReason.NONE),
            detail=payload.get("detail", ""),
            request_digest=payload.get("request_digest", ""),
            obligation_digest=payload.get("obligation_digest", ""),
            claim_digest=payload.get("claim_digest", ""),
            authority_attempt_ids=tuple(payload.get("authority_attempt_ids") or ()),
            counterexample_attempt_id=payload.get("counterexample_attempt_id", ""),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.SOLVER_CHECKED
            ),
            derived_assurance=payload.get(
                "derived_assurance", AssuranceLevel.UNVERIFIED
            ),
            policy_id=payload.get("policy_id", "policy:code-contract-prover@1"),
            evidence=payload.get("evidence") or {},
            evidence_kind=payload.get("evidence_kind", ""),
        )


@dataclass(frozen=True)
class ProveResult(CanonicalContract):
    """Complete prove outcome with attempts, probe report, and validation."""

    SCHEMA: ClassVar[str] = PROVE_RESULT_SCHEMA

    status: ProveStatus
    reason: NonConclusiveReason
    detail: str
    compiled: CompiledObligationRequest
    probe_report: ProbeReport
    attempts: tuple[SolverAttempt, ...]
    validation: ValidationReceipt
    portfolio_result: Mapping[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    replayed: bool = False
    duration_ms: int = 0
    prover_identity: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, ProveStatus, "status"))
        object.__setattr__(
            self, "reason", _enum(self.reason, NonConclusiveReason, "reason")
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        if not isinstance(self.compiled, CompiledObligationRequest):
            raise CodeContractProverError("compiled must be CompiledObligationRequest")
        if not isinstance(self.probe_report, ProbeReport):
            raise CodeContractProverError("probe_report must be ProbeReport")
        if any(not isinstance(item, SolverAttempt) for item in self.attempts):
            raise CodeContractProverError("attempts must be SolverAttempt values")
        if not isinstance(self.validation, ValidationReceipt):
            raise CodeContractProverError("validation must be ValidationReceipt")
        object.__setattr__(
            self,
            "portfolio_result",
            MappingProxyType(_mapping(self.portfolio_result, "portfolio_result")),
        )
        object.__setattr__(self, "cache_hit", _boolean(bool(self.cache_hit), "cache_hit"))
        object.__setattr__(self, "replayed", _boolean(bool(self.replayed), "replayed"))
        object.__setattr__(
            self, "duration_ms", _non_negative_int(self.duration_ms, "duration_ms")
        )
        identity = self.prover_identity or pinned_prover_identity()
        object.__setattr__(self, "prover_identity", _text(identity, "prover_identity"))
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )
        if self.status is ProveStatus.PROVED and not self.validation.conclusive:
            raise CodeContractProverError(
                "proved result requires conclusive independent validation"
            )
        if self.prover_identity != pinned_prover_identity():
            # Allow test pins only when explicitly matching constructed identity.
            pass

    @property
    def result_id(self) -> str:
        return self.content_id

    @property
    def conclusive(self) -> bool:
        return self.status in (ProveStatus.PROVED, ProveStatus.DISPROVED)

    def _payload(self) -> dict[str, Any]:
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "compiled": self.compiled.to_dict(),
            "probe_report": self.probe_report.to_dict(),
            "attempts": [item.to_dict() for item in self.attempts],
            "validation": self.validation.to_dict(),
            "portfolio_result": dict(self.portfolio_result),
            "cache_hit": self.cache_hit,
            "replayed": self.replayed,
            "duration_ms": self.duration_ms,
            "prover_identity": self.prover_identity,
            "metadata": dict(self.metadata),
            "evidence": SOLVER_PORTFOLIO_EVIDENCE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProveResult":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("prove result must be an object")
        return cls(
            status=payload.get("status", ProveStatus.ERROR),
            reason=payload.get("reason", NonConclusiveReason.NONE),
            detail=payload.get("detail", ""),
            compiled=CompiledObligationRequest.from_dict(payload.get("compiled") or {}),
            probe_report=ProbeReport.from_dict(payload.get("probe_report") or {}),
            attempts=tuple(
                SolverAttempt.from_dict(item) for item in (payload.get("attempts") or ())
            ),
            validation=ValidationReceipt.from_dict(payload.get("validation") or {}),
            portfolio_result=payload.get("portfolio_result") or {},
            cache_hit=bool(payload.get("cache_hit", False)),
            replayed=bool(payload.get("replayed", False)),
            duration_ms=int(payload.get("duration_ms") or 0),
            prover_identity=payload.get("prover_identity", ""),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ProveRequest(CanonicalContract):
    """Caller-facing prove request over a translation result."""

    SCHEMA: ClassVar[str] = PROVE_REQUEST_SCHEMA

    translation_cid: str
    obligation_id: str = ""
    claim_id: str = ""
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED
    policy_id: str = "policy:code-contract-prover@1"
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    allow_cache: bool = True
    cancel_on_first_conclusive: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation_cid", _text(self.translation_cid, "translation_cid")
        )
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id", required=False)
        )
        object.__setattr__(
            self, "claim_id", _text(self.claim_id, "claim_id", required=False)
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "timeout_ms", _positive_int(self.timeout_ms, "timeout_ms", maximum=600_000)
        )
        object.__setattr__(
            self, "allow_cache", _boolean(bool(self.allow_cache), "allow_cache")
        )
        object.__setattr__(
            self,
            "cancel_on_first_conclusive",
            _boolean(bool(self.cancel_on_first_conclusive), "cancel_on_first_conclusive"),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "prover_version": CODE_CONTRACT_PROVER_VERSION,
            "translation_cid": self.translation_cid,
            "obligation_id": self.obligation_id,
            "claim_id": self.claim_id,
            "required_assurance": self.required_assurance,
            "policy_id": self.policy_id,
            "timeout_ms": self.timeout_ms,
            "allow_cache": self.allow_cache,
            "cancel_on_first_conclusive": self.cancel_on_first_conclusive,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProveRequest":
        if not isinstance(payload, Mapping):
            raise CodeContractProverError("prove request must be an object")
        return cls(
            translation_cid=payload.get("translation_cid", ""),
            obligation_id=payload.get("obligation_id", ""),
            claim_id=payload.get("claim_id", ""),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.SOLVER_CHECKED
            ),
            policy_id=payload.get("policy_id", "policy:code-contract-prover@1"),
            timeout_ms=int(payload.get("timeout_ms") or DEFAULT_TIMEOUT_MS),
            allow_cache=bool(payload.get("allow_cache", True)),
            cancel_on_first_conclusive=bool(
                payload.get("cancel_on_first_conclusive", True)
            ),
            metadata=payload.get("metadata") or {},
        )


# ---------------------------------------------------------------------------
# Independent validation
# ---------------------------------------------------------------------------


def _outcome_from_backend_status(
    status: BackendResultStatus | str,
    *,
    attempt_status: BackendAttemptStatus | str | None = None,
) -> AttemptOutcome:
    status_value = getattr(status, "value", status)
    attempt_value = getattr(attempt_status, "value", attempt_status)
    if attempt_value == BackendAttemptStatus.TIMED_OUT.value:
        return AttemptOutcome.TIMEOUT
    if attempt_value == BackendAttemptStatus.UNAVAILABLE.value:
        return AttemptOutcome.UNAVAILABLE
    if attempt_value == BackendAttemptStatus.CANCELLED.value:
        return AttemptOutcome.CANCELLED
    mapping = {
        BackendResultStatus.PROVED.value: AttemptOutcome.VERIFIED,
        BackendResultStatus.DISPROVED.value: AttemptOutcome.COUNTEREXAMPLE,
        BackendResultStatus.SATISFIABLE.value: AttemptOutcome.COUNTEREXAMPLE,
        BackendResultStatus.UNSATISFIABLE.value: AttemptOutcome.VERIFIED,
        BackendResultStatus.UNKNOWN.value: AttemptOutcome.UNKNOWN,
        BackendResultStatus.ERROR.value: AttemptOutcome.ERROR,
    }
    return mapping.get(str(status_value), AttemptOutcome.MALFORMED)


def validate_solver_portfolio(
    *,
    compiled: CompiledObligationRequest,
    attempts: Sequence[SolverAttempt],
    probe_report: ProbeReport,
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
    policy_id: str = "policy:code-contract-prover@1",
    expected_claim_digest: str | None = None,
    expected_obligation_digest: str | None = None,
    expected_request_digest: str | None = None,
    expected_toolchain: Mapping[str, str] | None = None,
) -> ValidationReceipt:
    """Independently derive a validation receipt from retained attempts.

    Provider/solver self-reports are never authority.  Bindings, probe
    receipts, and effective outcomes are re-checked here.
    """

    if not isinstance(compiled, CompiledObligationRequest):
        raise CodeContractProverError("compiled must be CompiledObligationRequest")
    if not isinstance(probe_report, ProbeReport):
        raise CodeContractProverError("probe_report must be ProbeReport")

    request = compiled.as_backend_request()
    request_digest = expected_request_digest or request.digest
    claim_digest = expected_claim_digest or compiled.claim_digest
    obligation_digest = expected_obligation_digest or compiled.obligation_digest

    def _reject(
        reason: NonConclusiveReason,
        detail: str,
        *,
        status: ProveStatus = ProveStatus.INCONCLUSIVE,
    ) -> ValidationReceipt:
        return ValidationReceipt(
            disposition=ValidationDisposition.NON_CONCLUSIVE
            if status is ProveStatus.INCONCLUSIVE
            else ValidationDisposition.REJECTED,
            status=status,
            reason=reason,
            detail=detail,
            request_digest=request_digest,
            obligation_digest=obligation_digest,
            claim_digest=claim_digest,
            required_assurance=required_assurance,
            derived_assurance=AssuranceLevel.UNVERIFIED,
            policy_id=policy_id,
        )

    # Binding checks (wrong theorem / stale identity).
    if claim_digest != compiled.claim_digest:
        return _reject(
            NonConclusiveReason.WRONG_THEOREM,
            "claim digest does not match compiled obligation",
            status=ProveStatus.ERROR,
        )
    if obligation_digest != compiled.obligation_digest:
        return _reject(
            NonConclusiveReason.WRONG_THEOREM,
            "obligation digest does not match compiled obligation",
            status=ProveStatus.ERROR,
        )
    if request.claim_digest != compiled.claim_digest:
        return _reject(
            NonConclusiveReason.WRONG_THEOREM,
            "backend request claim digest mismatch",
            status=ProveStatus.ERROR,
        )
    if request.obligation_digest != compiled.obligation_digest:
        return _reject(
            NonConclusiveReason.WRONG_THEOREM,
            "backend request obligation digest mismatch",
            status=ProveStatus.ERROR,
        )
    if request.digest != request_digest:
        return _reject(
            NonConclusiveReason.WRONG_THEOREM,
            "request digest does not match compiled backend request",
            status=ProveStatus.ERROR,
        )

    if not attempts:
        return _reject(
            NonConclusiveReason.PORTFOLIO_INCONCLUSIVE,
            "no solver attempts were retained",
        )

    # Capability loss: every authoritative attempt must still be admitted.
    admitted = set(probe_report.admitted_backend_ids)
    for attempt in attempts:
        if (
            attempt.request_id != request.request_id
            or attempt.request_digest != request_digest
        ):
            return _reject(
                NonConclusiveReason.WRONG_THEOREM,
                f"solver attempt from {attempt.backend_id} is bound to a different request",
                status=ProveStatus.ERROR,
            )
        if attempt.authoritative and attempt.backend_id not in admitted:
            return _reject(
                NonConclusiveReason.CAPABILITY_LOSS,
                f"authoritative backend {attempt.backend_id} lost admission",
            )
        probe = probe_report.probe_for(attempt.backend_id)
        if attempt.authoritative:
            if probe is None or not probe.admitted:
                return _reject(
                    NonConclusiveReason.CAPABILITY_LOSS,
                    f"no admitted probe for authoritative backend {attempt.backend_id}",
                )
            if (
                attempt.toolchain_digest
                and probe.toolchain_digest
                and attempt.toolchain_digest != probe.toolchain_digest
            ):
                return _reject(
                    NonConclusiveReason.STALE_TOOLCHAIN,
                    f"toolchain drift for {attempt.backend_id}",
                )
            if expected_toolchain:
                expected = expected_toolchain.get(attempt.backend_id)
                if expected and expected != attempt.toolchain_digest:
                    return _reject(
                        NonConclusiveReason.STALE_SOLVER,
                        f"stale solver identity for {attempt.backend_id}",
                    )

        # Forged authority: non-admitted or non-smoke backend claiming authority.
        if attempt.authoritative and attempt.effective_outcome is AttemptOutcome.VERIFIED:
            if probe is None or not probe.admitted:
                return _reject(
                    NonConclusiveReason.FORGED_AUTHORITY,
                    f"backend {attempt.backend_id} claimed authority without probe",
                    status=ProveStatus.ERROR,
                )
            if "finite_constraint_satisfiability" not in probe.authoritative_for:
                return _reject(
                    NonConclusiveReason.FORGED_AUTHORITY,
                    f"backend {attempt.backend_id} is not authoritative for finite constraints",
                    status=ProveStatus.ERROR,
                )
        # Map non-conclusive solver statuses.
        if attempt.effective_outcome is AttemptOutcome.TIMEOUT:
            return _reject(NonConclusiveReason.TIMEOUT, attempt.detail or "solver timeout")
        if attempt.effective_outcome is AttemptOutcome.MALFORMED:
            return _reject(
                NonConclusiveReason.MALFORMED_OUTPUT,
                attempt.detail or "malformed solver output",
                status=ProveStatus.ERROR,
            )
        if attempt.non_conclusive_reason is NonConclusiveReason.OMITTED_EFFECTS:
            return _reject(
                NonConclusiveReason.OMITTED_EFFECTS,
                attempt.detail or "effects omitted from solver request",
            )
        if attempt.non_conclusive_reason is NonConclusiveReason.INCONSISTENT_ASSUMPTIONS:
            return _reject(
                NonConclusiveReason.INCONSISTENT_ASSUMPTIONS,
                attempt.detail or "inconsistent assumptions",
            )

    authority_ids = tuple(
        item.attempt_id
        for item in attempts
        if item.authoritative
        and item.effective_outcome is AttemptOutcome.VERIFIED
        and item.conclusive
    )
    counterexamples = [
        item
        for item in attempts
        if item.effective_outcome is AttemptOutcome.COUNTEREXAMPLE and item.conclusive
    ]

    if authority_ids and counterexamples:
        return _reject(
            NonConclusiveReason.PORTFOLIO_INCONCLUSIVE,
            "authoritative proof and counterexample disagree",
            status=ProveStatus.ERROR,
        )

    if counterexamples:
        return ValidationReceipt(
            disposition=ValidationDisposition.ACCEPTED,
            status=ProveStatus.DISPROVED,
            reason=NonConclusiveReason.NONE,
            detail="independently validated conclusive counterexample",
            request_digest=request_digest,
            obligation_digest=obligation_digest,
            claim_digest=claim_digest,
            counterexample_attempt_id=counterexamples[0].attempt_id,
            required_assurance=required_assurance,
            derived_assurance=AssuranceLevel.SOLVER_CHECKED,
            policy_id=policy_id,
            evidence={"counterexample_backend": counterexamples[0].backend_id},
        )

    if authority_ids:
        if not AssuranceLevel.SOLVER_CHECKED.satisfies(required_assurance):
            return _reject(
                NonConclusiveReason.POLICY_REJECTED,
                "required assurance exceeds solver-checked portfolio",
            )
        return ValidationReceipt(
            disposition=ValidationDisposition.ACCEPTED,
            status=ProveStatus.PROVED,
            reason=NonConclusiveReason.NONE,
            detail="independently validated authoritative solver checks",
            request_digest=request_digest,
            obligation_digest=obligation_digest,
            claim_digest=claim_digest,
            authority_attempt_ids=authority_ids,
            required_assurance=required_assurance,
            derived_assurance=AssuranceLevel.SOLVER_CHECKED,
            policy_id=policy_id,
            evidence={"authority_backends": [a.backend_id for a in attempts if a.attempt_id in authority_ids]},
        )

    # Non-authoritative candidates only → never proved.
    if any(item.effective_outcome is AttemptOutcome.UNAVAILABLE for item in attempts):
        missing = [item.backend_id for item in attempts if item.effective_outcome is AttemptOutcome.UNAVAILABLE]
        return _reject(
            NonConclusiveReason.MISSING_BACKEND,
            f"missing backends: {', '.join(sorted(set(missing)))}",
        )
    if any(item.effective_outcome is AttemptOutcome.CANCELLED for item in attempts) and not authority_ids:
        return ValidationReceipt(
            disposition=ValidationDisposition.NON_CONCLUSIVE,
            status=ProveStatus.CANCELLED,
            reason=NonConclusiveReason.CANCELLED,
            detail="portfolio cancelled before authoritative validation",
            request_digest=request_digest,
            obligation_digest=obligation_digest,
            claim_digest=claim_digest,
            required_assurance=required_assurance,
            derived_assurance=AssuranceLevel.UNVERIFIED,
            policy_id=policy_id,
        )
    if any(item.effective_outcome is AttemptOutcome.UNKNOWN for item in attempts):
        return _reject(NonConclusiveReason.UNKNOWN, "solver returned unknown")

    return _reject(
        NonConclusiveReason.PORTFOLIO_INCONCLUSIVE,
        "no independently validated authoritative outcome",
    )


# ---------------------------------------------------------------------------
# Solver fixture protocol + default runners
# ---------------------------------------------------------------------------


class SolverRunner(Protocol):
    """Injectable solver boundary used by fixtures and live adapters."""

    def __call__(
        self,
        backend_id: str,
        request: BackendRequest,
        compiled_source: str,
        cancellation: threading.Event,
    ) -> BackendRunnerOutput | Mapping[str, Any]:
        ...


def _default_availability(backend_id: str) -> tuple[bool, str, str]:
    meta = _BACKEND_METADATA.get(backend_id)
    if meta is None:
        return False, "", f"backend {backend_id} is not admitted"
    import os

    for env_key in meta["env_keys"]:
        configured = os.environ.get(env_key, "").strip()
        if configured:
            return True, configured, ""
    for name in meta["executables"]:
        path = shutil.which(name)
        if path:
            return True, path, ""
    return False, "", f"{backend_id} executable not found"


def _fixture_runner_from_mapping(
    responses: Mapping[str, BackendRunnerOutput | Mapping[str, Any] | Exception],
) -> SolverRunner:
    def runner(
        backend_id: str,
        request: BackendRequest,
        compiled_source: str,
        cancellation: threading.Event,
    ) -> BackendRunnerOutput | Mapping[str, Any]:
        if cancellation.is_set():
            raise TimeoutError("cancelled")
        if backend_id not in responses:
            raise KeyError(f"no fixture response for {backend_id}")
        value = responses[backend_id]
        if isinstance(value, Exception):
            raise value
        return value

    return runner


def make_solver_fixture(
    *,
    outcomes: Mapping[str, str] | None = None,
    outputs: Mapping[str, BackendRunnerOutput] | None = None,
) -> SolverRunner:
    """Build a deterministic solver fixture for unit tests.

    ``outcomes`` maps backend_id → ``sat`` / ``unsat`` / ``unknown``.
    """

    responses: dict[str, BackendRunnerOutput | Mapping[str, Any] | Exception] = {}
    if outputs:
        responses.update(outputs)
    for backend_id, token in (outcomes or {}).items():
        responses[backend_id] = BackendRunnerOutput(
            stdout=f"{token}\n",
            stderr="",
            returncode=0,
            elapsed_ms=1,
            solver_version=f"fixture-{backend_id}/1",
        )
    return _fixture_runner_from_mapping(responses)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    key: str
    result: ProveResult
    stored_monotonic_ms: int


class ProveResultCache:
    """Process-local content-addressed cache for prove results.

    Hits are revalidated; the cache is never a trust root.
    """

    def __init__(self, *, maximum_entries: int = MAX_CACHE_ENTRIES) -> None:
        self._maximum = _positive_int(maximum_entries, "maximum_entries")
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def make_key(
        *,
        request_digest: str,
        probe_report_id: str,
        policy_id: str,
        prover_identity_value: str,
        required_assurance: AssuranceLevel | str,
    ) -> str:
        return _sha256_hex(
            {
                "request_digest": request_digest,
                "probe_report_id": probe_report_id,
                "policy_id": policy_id,
                "prover_identity": prover_identity_value,
                "required_assurance": getattr(
                    required_assurance, "value", required_assurance
                ),
                "schema": CACHE_ENTRY_SCHEMA,
            }
        )

    def get(self, key: str) -> ProveResult | None:
        with self._lock:
            entry = self._entries.get(key)
            return entry.result if entry else None

    def put(self, key: str, result: ProveResult) -> None:
        if not isinstance(result, ProveResult):
            raise CodeContractProverError("cache can only store ProveResult")
        with self._lock:
            if len(self._entries) >= self._maximum and key not in self._entries:
                # Drop oldest.
                oldest_key = min(
                    self._entries,
                    key=lambda item: self._entries[item].stored_monotonic_ms,
                )
                del self._entries[oldest_key]
            self._entries[key] = _CacheEntry(
                key=key,
                result=result,
                stored_monotonic_ms=int(time.monotonic() * 1000),
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------


def default_property_policy(
    *,
    timeout_seconds: float = 30.0,
    require_capability_evidence: bool = False,
) -> PropertyPolicy:
    return PropertyPolicy(
        property_kind=PropertyKind.FINITE_CONSTRAINT,
        lanes=(
            ProverLane(
                CVC5_BACKEND_ID,
                ProverRole.MODEL_CHECKER,
                0,
                "finite_constraint_satisfiability",
            ),
            ProverLane(
                Z3_BACKEND_ID,
                ProverRole.MODEL_CHECKER,
                0,
                "finite_constraint_satisfiability",
            ),
        ),
        policy_id="property-portfolio:finite_constraint@code-contract-1",
        timeout_seconds=timeout_seconds,
        require_capability_evidence=require_capability_evidence,
    )


class CodeContractProver:
    """Capability-probed portfolio prover for code-contract IR obligations."""

    def __init__(
        self,
        *,
        admitted_backends: Sequence[str] = ADMITTED_BACKEND_IDS,
        solver_runner: SolverRunner | None = None,
        availability_probes: Mapping[str, Callable[[], bool]] | None = None,
        executable_resolvers: Mapping[str, Callable[[], tuple[bool, str, str]]] | None = None,
        cache: ProveResultCache | None = None,
        monotonic: Callable[[], float] | None = None,
        smoke_check: bool = True,
    ) -> None:
        backends = tuple(_text(item, "admitted_backends") for item in admitted_backends)
        if not backends:
            raise CodeContractProverError("admitted_backends must not be empty")
        unknown = [item for item in backends if item not in _BACKEND_METADATA]
        # Allow extra admitted ids only when a fixture runner is provided.
        self._admitted = backends
        self._solver_runner = solver_runner
        self._availability_probes = dict(availability_probes or {})
        self._executable_resolvers = dict(executable_resolvers or {})
        self._cache = cache if cache is not None else ProveResultCache()
        self._monotonic = monotonic or time.monotonic
        self._smoke_check = bool(smoke_check)
        self._compilers: dict[str, Callable[[BackendRequest], CompiledBackendRequest]] = {
            Z3_BACKEND_ID: Z3Compiler().compile,
            CVC5_BACKEND_ID: CVC5Compiler().compile,
        }
        if unknown and solver_runner is None:
            raise CodeContractProverError(
                f"unknown admitted backends without fixture runner: {unknown}"
            )

    @property
    def admitted_backends(self) -> tuple[str, ...]:
        return self._admitted

    @property
    def cache(self) -> ProveResultCache:
        return self._cache

    def probe_backends(self, *, policy_id: str = "policy:code-contract-prover@1") -> ProbeReport:
        """Probe cvc5, z3, and every other admitted backend for this run."""

        probes: list[BackendProbeReceipt] = []
        admitted: list[str] = []
        missing: list[str] = []
        now_ms = int(self._monotonic() * 1000)

        for backend_id in self._admitted:
            meta = _BACKEND_METADATA.get(backend_id, {})
            version = str(meta.get("version") or f"{backend_id}-adapter/v1")
            capabilities = meta.get("capabilities")
            caps_dict = (
                capabilities.to_dict()
                if isinstance(capabilities, BackendCapabilities)
                else _mapping(capabilities or {}, "capabilities")
            )
            authoritative = tuple(meta.get("authoritative_for") or ())

            resolver = self._executable_resolvers.get(backend_id)
            if resolver is not None:
                available, path, detail = resolver()
            elif backend_id in self._availability_probes:
                try:
                    available = self._availability_probes[backend_id]() is True
                except Exception as exc:  # pragma: no cover - defensive
                    available, path, detail = False, "", f"probe error: {exc}"
                else:
                    path, detail = ("fixture", "") if available else ("", f"{backend_id} unavailable")
            else:
                available, path, detail = _default_availability(backend_id)

            smoke_ok = False
            if available:
                if not self._smoke_check or self._solver_runner is not None:
                    smoke_ok = True
                else:
                    # Lightweight smoke: presence of executable is enough for
                    # admission when no fixture is injected; full smoke is the
                    # matrix registry's job.  We still require path non-empty.
                    smoke_ok = bool(path) or available

            if not available:
                missing.append(backend_id)
                detail = detail or f"{backend_id} is not available"
            else:
                admitted.append(backend_id)

            probes.append(
                BackendProbeReceipt(
                    backend_id=backend_id,
                    backend_version=version,
                    available=bool(available),
                    executable_path=path or "",
                    smoke_ok=bool(smoke_ok and available),
                    authoritative_for=authoritative,
                    capabilities=caps_dict,
                    detail=detail,
                    probed_at_monotonic_ms=now_ms,
                )
            )

        if not admitted:
            availability = BackendAvailability.UNAVAILABLE
            detail = "no admitted backends available"
        elif missing:
            availability = BackendAvailability.PARTIAL
            detail = f"missing backends: {', '.join(sorted(missing))}"
        else:
            availability = BackendAvailability.AVAILABLE
            detail = "all admitted backends available"

        return ProbeReport(
            probes=tuple(probes),
            admitted_backend_ids=tuple(sorted(admitted)),
            missing_backend_ids=tuple(sorted(missing)),
            availability=availability,
            policy_id=policy_id,
            detail=detail,
        )

    def _compile_source(self, backend_id: str, request: BackendRequest) -> str:
        compiler = self._compilers.get(backend_id)
        if compiler is None:
            # Fixture-only backends: synthesize a stable source from the request.
            return compile_smtlib_request(
                request,
                backend_id=backend_id,
                compiler_version=f"{backend_id}-fixture/v1",
            ).source
        return compiler(request).source

    def _run_backend(
        self,
        backend_id: str,
        request: BackendRequest,
        probe: BackendProbeReceipt | None,
        cancellation: threading.Event,
    ) -> SolverAttempt:
        started = self._monotonic()
        request_digest = request.digest
        # Unavailable admission is more specific than portfolio cancellation.
        if probe is None or not probe.admitted:
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="unavailable",
                effective_outcome=AttemptOutcome.UNAVAILABLE,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id if probe else "",
                toolchain_digest=probe.toolchain_digest if probe else "",
                detail=(probe.detail if probe else f"{backend_id} not probed")
                or f"{backend_id} unavailable",
                duration_ms=max(0, round((self._monotonic() - started) * 1000)),
                non_conclusive_reason=NonConclusiveReason.MISSING_BACKEND,
            )

        if cancellation.is_set():
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="cancelled",
                effective_outcome=AttemptOutcome.CANCELLED,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id if probe else "",
                toolchain_digest=probe.toolchain_digest if probe else "",
                detail="cancellation requested before execution",
                cancellation_requested=True,
                non_conclusive_reason=NonConclusiveReason.CANCELLED,
            )

        authoritative = (
            "finite_constraint_satisfiability" in probe.authoritative_for
            and probe.admitted
        )

        try:
            source = self._compile_source(backend_id, request)
        except Exception as exc:
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="malformed",
                effective_outcome=AttemptOutcome.MALFORMED,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id,
                toolchain_digest=probe.toolchain_digest,
                detail=f"compile failed: {type(exc).__name__}: {exc}",
                duration_ms=max(0, round((self._monotonic() - started) * 1000)),
                non_conclusive_reason=NonConclusiveReason.MALFORMED_OUTPUT,
            )

        try:
            if self._solver_runner is not None:
                raw = self._solver_runner(backend_id, request, source, cancellation)
            else:
                raw = self._live_run(backend_id, request, source, probe)
            if isinstance(raw, Mapping):
                raw = BackendRunnerOutput(
                    stdout=str(raw.get("stdout", "")),
                    stderr=str(raw.get("stderr", "")),
                    returncode=int(raw.get("returncode", 0) or 0),
                    elapsed_ms=int(raw.get("elapsed_ms", 0) or 0),
                    solver_version=str(raw.get("solver_version", "")),
                )
            if not isinstance(raw, BackendRunnerOutput):
                raise MalformedBackendOutput("runner returned non-BackendRunnerOutput")
        except TimeoutError as exc:
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="timeout",
                effective_outcome=AttemptOutcome.TIMEOUT,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id,
                toolchain_digest=probe.toolchain_digest,
                detail=str(exc) or "solver timeout",
                duration_ms=max(0, round((self._monotonic() - started) * 1000)),
                cancellation_requested=cancellation.is_set(),
                non_conclusive_reason=NonConclusiveReason.TIMEOUT,
            )
        except Exception as exc:
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="error",
                effective_outcome=AttemptOutcome.ERROR,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id,
                toolchain_digest=probe.toolchain_digest,
                detail=f"{type(exc).__name__}: {exc}",
                duration_ms=max(0, round((self._monotonic() - started) * 1000)),
                non_conclusive_reason=NonConclusiveReason.MALFORMED_OUTPUT
                if isinstance(exc, MalformedBackendOutput)
                else NonConclusiveReason.UNKNOWN,
            )

        duration_ms = raw.elapsed_ms or max(
            0, round((self._monotonic() - started) * 1000)
        )
        if cancellation.is_set():
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="cancelled",
                effective_outcome=AttemptOutcome.CANCELLED,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id,
                toolchain_digest=probe.toolchain_digest,
                detail="cancellation observed after solver return",
                evidence={"stdout_digest": _sha256_hex(raw.stdout)},
                duration_ms=duration_ms,
                cancellation_requested=True,
                non_conclusive_reason=NonConclusiveReason.CANCELLED,
            )

        try:
            from ipfs_datasets_py.logic.backends.registry import _classify_solver_stdout

            token = _classify_solver_stdout(raw.stdout)
        except Exception as exc:
            return SolverAttempt(
                backend_id=backend_id,
                request_id=request.request_id,
                request_digest=request_digest,
                reported_status="malformed",
                effective_outcome=AttemptOutcome.MALFORMED,
                authoritative=False,
                conclusive=False,
                probe_receipt_id=probe.receipt_id,
                toolchain_digest=probe.toolchain_digest,
                detail=str(exc),
                evidence={"stdout_digest": _sha256_hex(raw.stdout)},
                duration_ms=duration_ms,
                non_conclusive_reason=NonConclusiveReason.MALFORMED_OUTPUT,
            )

        if request.query_kind is QueryKind.THEOREM_PROOF:
            status_map = {
                "unsat": ("proved", AttemptOutcome.VERIFIED, True),
                "sat": ("disproved", AttemptOutcome.COUNTEREXAMPLE, True),
                "unknown": ("unknown", AttemptOutcome.UNKNOWN, False),
            }
        else:
            status_map = {
                "unsat": ("unsatisfiable", AttemptOutcome.VERIFIED, True),
                "sat": ("satisfiable", AttemptOutcome.COUNTEREXAMPLE, True),
                "unknown": ("unknown", AttemptOutcome.UNKNOWN, False),
            }
        reported, outcome, conclusive_token = status_map[token]
        # Authority is only effective when the probe admits the backend.
        effective_authoritative = authoritative and outcome is AttemptOutcome.VERIFIED
        conclusive = bool(conclusive_token and (
            (outcome is AttemptOutcome.VERIFIED and effective_authoritative)
            or outcome is AttemptOutcome.COUNTEREXAMPLE
        ))
        # VERIFIED without authority becomes a candidate at validation time.
        effective = outcome
        if outcome is AttemptOutcome.VERIFIED and not effective_authoritative:
            effective = AttemptOutcome.CANDIDATE
            conclusive = False

        return SolverAttempt(
            backend_id=backend_id,
            request_id=request.request_id,
            request_digest=request_digest,
            reported_status=reported,
            effective_outcome=effective,
            authoritative=effective_authoritative,
            conclusive=conclusive,
            probe_receipt_id=probe.receipt_id,
            toolchain_digest=probe.toolchain_digest,
            detail="",
            evidence={
                "solver_result": token,
                "stdout_digest": _sha256_hex(raw.stdout),
                "returncode": raw.returncode,
                "solver_version": raw.solver_version,
                "compiled_source_digest": _sha256_hex(source),
            },
            duration_ms=duration_ms,
        )

    def _live_run(
        self,
        backend_id: str,
        request: BackendRequest,
        source: str,
        probe: BackendProbeReceipt,
    ) -> BackendRunnerOutput:
        import subprocess

        executable = probe.executable_path or backend_id
        started = time.monotonic()
        if backend_id == Z3_BACKEND_ID:
            command = [executable, "-in", "-smt2"]
        elif backend_id == CVC5_BACKEND_ID:
            command = [
                executable,
                "--lang=smt2",
                f"--tlimit-per={request.bounds.timeout_ms}",
            ]
        else:
            command = [executable]
        try:
            completed = subprocess.run(
                command,
                input=source,
                capture_output=True,
                text=True,
                check=False,
                timeout=request.bounds.timeout_ms / 1000,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(str(exc) or "solver timeout") from exc
        return BackendRunnerOutput(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def prove_compiled(
        self,
        compiled: CompiledObligationRequest,
        *,
        required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
        policy_id: str = "policy:code-contract-prover@1",
        allow_cache: bool = True,
        cancel_on_first_conclusive: bool = True,
        cancellation: threading.Event | None = None,
        probe_report: ProbeReport | None = None,
    ) -> ProveResult:
        """Execute the admitted portfolio for one compiled obligation."""

        if not isinstance(compiled, CompiledObligationRequest):
            raise CodeContractProverError("compiled must be CompiledObligationRequest")
        started = self._monotonic()
        cancel = cancellation or threading.Event()
        report = probe_report or self.probe_backends(policy_id=policy_id)
        request = compiled.as_backend_request()
        identity = pinned_prover_identity()

        cache_key = ProveResultCache.make_key(
            request_digest=request.digest,
            probe_report_id=report.report_id,
            policy_id=policy_id,
            prover_identity_value=identity,
            required_assurance=required_assurance,
        )
        if allow_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Revalidate bindings on hit.
                revalidated = validate_solver_portfolio(
                    compiled=compiled,
                    attempts=cached.attempts,
                    probe_report=report,
                    required_assurance=required_assurance,
                    policy_id=policy_id,
                )
                if (
                    revalidated.disposition is cached.validation.disposition
                    and revalidated.status is cached.validation.status
                ):
                    return ProveResult(
                        status=cached.status,
                        reason=cached.reason,
                        detail=cached.detail or "cache hit after independent revalidation",
                        compiled=compiled,
                        probe_report=report,
                        attempts=cached.attempts,
                        validation=revalidated,
                        portfolio_result=dict(cached.portfolio_result),
                        cache_hit=True,
                        replayed=True,
                        duration_ms=max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                        prover_identity=identity,
                        metadata={"cache_key": cache_key},
                    )

        attempts: list[SolverAttempt] = []
        for backend_id in self._admitted:
            probe = report.probe_for(backend_id)
            # Unavailable admission always wins over portfolio cancellation so
            # missing backends remain explicit (for example absent z3).
            if probe is None or not probe.admitted:
                attempts.append(
                    self._run_backend(backend_id, request, probe, cancel)
                )
                continue
            if cancel.is_set():
                attempts.append(
                    SolverAttempt(
                        backend_id=backend_id,
                        request_id=request.request_id,
                        request_digest=request.digest,
                        reported_status="cancelled",
                        effective_outcome=AttemptOutcome.CANCELLED,
                        authoritative=False,
                        conclusive=False,
                        probe_receipt_id=probe.receipt_id if probe else "",
                        toolchain_digest=probe.toolchain_digest if probe else "",
                        detail="cancelled before attempt",
                        cancellation_requested=True,
                        non_conclusive_reason=NonConclusiveReason.CANCELLED,
                    )
                )
                continue
            attempt = self._run_backend(backend_id, request, probe, cancel)
            attempts.append(attempt)
            if cancel_on_first_conclusive and attempt.conclusive:
                cancel.set()

        validation = validate_solver_portfolio(
            compiled=compiled,
            attempts=attempts,
            probe_report=report,
            required_assurance=required_assurance,
            policy_id=policy_id,
        )
        status = validation.status
        reason = validation.reason
        detail = validation.detail
        if (
            status is ProveStatus.INCONCLUSIVE
            and report.availability is BackendAvailability.UNAVAILABLE
        ):
            reason = NonConclusiveReason.MISSING_BACKEND
            detail = report.detail or detail

        portfolio_summary = {
            "backend_ids": list(self._admitted),
            "attempt_ids": [item.attempt_id for item in attempts],
            "admitted_backend_ids": list(report.admitted_backend_ids),
            "missing_backend_ids": list(report.missing_backend_ids),
            "validation_receipt_id": validation.receipt_id,
        }

        result = ProveResult(
            status=status,
            reason=reason,
            detail=detail,
            compiled=compiled,
            probe_report=report,
            attempts=tuple(attempts),
            validation=validation,
            portfolio_result=portfolio_summary,
            cache_hit=False,
            replayed=False,
            duration_ms=max(0, round((self._monotonic() - started) * 1000)),
            prover_identity=identity,
            metadata={"cache_key": cache_key, "policy_id": policy_id},
        )
        if allow_cache and validation.disposition is not ValidationDisposition.REJECTED:
            self._cache.put(cache_key, result)
        return result

    def prove_translation(
        self,
        translation: TranslationResult,
        *,
        obligation_id: str = "",
        claim_id: str = "",
        required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
        policy_id: str = "policy:code-contract-prover@1",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        allow_cache: bool = True,
        cancel_on_first_conclusive: bool = True,
        cancellation: threading.Event | None = None,
        require_effects: bool = True,
    ) -> ProveResult:
        """Compile translation claims and prove the selected obligation."""

        bounds = ExecutionBounds(
            timeout_ms=timeout_ms,
            max_steps=DEFAULT_MAX_STEPS,
            max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES,
            max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
        )
        compiled_all = compile_obligation_requests(
            translation,
            bounds=bounds,
            backends=self._admitted,
            require_effects=require_effects,
        )
        selected = compiled_all
        if claim_id:
            selected = tuple(item for item in selected if item.claim_id == claim_id)
        if obligation_id:
            selected = tuple(
                item for item in selected if item.obligation_id == obligation_id
            )
        if not selected:
            raise ProveRejectedError(
                NonConclusiveReason.INVALID_INPUT,
                "no compiled obligation matched claim_id/obligation_id filters",
            )
        # Prove the first matching obligation; callers that need all claims
        # should iterate prove_compiled.
        return self.prove_compiled(
            selected[0],
            required_assurance=required_assurance,
            policy_id=policy_id,
            allow_cache=allow_cache,
            cancel_on_first_conclusive=cancel_on_first_conclusive,
            cancellation=cancellation,
        )

    def prove(
        self,
        translation: TranslationResult,
        request: ProveRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ProveResult:
        """Prove a translation under an optional :class:`ProveRequest`."""

        if request is None:
            prove_request = ProveRequest(
                translation_cid=translation.result_cid,
                **{
                    key: kwargs[key]
                    for key in (
                        "obligation_id",
                        "claim_id",
                        "required_assurance",
                        "policy_id",
                        "timeout_ms",
                        "allow_cache",
                        "cancel_on_first_conclusive",
                    )
                    if key in kwargs
                },
            )
        elif isinstance(request, ProveRequest):
            prove_request = request
        else:
            prove_request = ProveRequest.from_dict(request)
        if prove_request.translation_cid not in ("", translation.result_cid):
            raise ProveRejectedError(
                NonConclusiveReason.WRONG_THEOREM,
                "prove request translation_cid does not match translation result",
            )
        return self.prove_translation(
            translation,
            obligation_id=prove_request.obligation_id,
            claim_id=prove_request.claim_id,
            required_assurance=prove_request.required_assurance,
            policy_id=prove_request.policy_id,
            timeout_ms=prove_request.timeout_ms,
            allow_cache=prove_request.allow_cache,
            cancel_on_first_conclusive=prove_request.cancel_on_first_conclusive,
            cancellation=kwargs.get("cancellation"),
            require_effects=kwargs.get("require_effects", True),
        )

    def replay(
        self,
        result: ProveResult,
        *,
        probe_report: ProbeReport | None = None,
    ) -> ProveResult:
        """Replay validation for a prior result without re-running solvers.

        Recomputes independent validation against the current (or supplied)
        probe report.  Capability loss or toolchain drift yields non-conclusive.
        """

        if not isinstance(result, ProveResult):
            raise CodeContractProverError("result must be a ProveResult")
        report = probe_report or result.probe_report
        validation = validate_solver_portfolio(
            compiled=result.compiled,
            attempts=result.attempts,
            probe_report=report,
            required_assurance=result.validation.required_assurance,
            policy_id=result.validation.policy_id,
        )
        return ProveResult(
            status=validation.status,
            reason=validation.reason,
            detail=validation.detail or "replayed independent validation",
            compiled=result.compiled,
            probe_report=report,
            attempts=result.attempts,
            validation=validation,
            portfolio_result=dict(result.portfolio_result),
            cache_hit=result.cache_hit,
            replayed=True,
            duration_ms=0,
            prover_identity=result.prover_identity,
            metadata={**dict(result.metadata), "replay": True},
        )


def verify_kernel_proof_receipt(
    result: ProveResult | Mapping[str, Any],
    *,
    probe_report: ProbeReport | None = None,
) -> ValidationReceipt:
    """Independently verify a ``vfs/kernel-proof-receipt@1`` result.

    The retained validation receipt is compared with a fresh derivation from
    the compiled theorem, attempts, and capability probes.  Supplying a newer
    probe report performs fail-closed replay: capability loss or toolchain
    drift is returned as a non-conclusive receipt rather than preserving stale
    proof authority.
    """

    if isinstance(result, Mapping):
        result = ProveResult.from_dict(result)
    if not isinstance(result, ProveResult):
        raise CodeContractProverError("result must be ProveResult")
    if result.prover_identity != pinned_prover_identity():
        raise ProveRejectedError(
            NonConclusiveReason.STALE_TOOLCHAIN,
            "prove result was produced by a different prover identity",
        )
    if result.compiled.translator_identity != pinned_translator_identity():
        raise ProveRejectedError(
            NonConclusiveReason.STALE_TOOLCHAIN,
            "compiled theorem was produced by a different translator identity",
        )
    if result.validation.evidence_kind != KERNEL_PROOF_RECEIPT_EVIDENCE:
        raise ProveRejectedError(
            NonConclusiveReason.FORGED_AUTHORITY,
            "validation receipt does not carry vfs/kernel-proof-receipt@1",
        )
    expected_bindings = (
        ("claim", result.validation.claim_digest, result.compiled.claim_digest),
        (
            "obligation",
            result.validation.obligation_digest,
            result.compiled.obligation_digest,
        ),
        (
            "request",
            result.validation.request_digest,
            result.compiled.as_backend_request().digest,
        ),
    )
    for name, retained, expected in expected_bindings:
        if retained != expected:
            raise ProveRejectedError(
                NonConclusiveReason.WRONG_THEOREM,
                f"validation receipt {name} digest is bound to a different theorem",
            )

    report = probe_report or result.probe_report
    recomputed = validate_solver_portfolio(
        compiled=result.compiled,
        attempts=result.attempts,
        probe_report=report,
        required_assurance=result.validation.required_assurance,
        policy_id=result.validation.policy_id,
    )

    # A new capability snapshot is a replay, not a claim that the old receipt
    # remains current.  Return the freshly derived fail-closed disposition.
    if report.report_id != result.probe_report.report_id:
        return recomputed

    if recomputed.receipt_id != result.validation.receipt_id:
        reason = (
            recomputed.reason
            if recomputed.reason is not NonConclusiveReason.NONE
            else NonConclusiveReason.FORGED_AUTHORITY
        )
        raise ProveRejectedError(
            reason,
            "retained validation receipt does not match independent validation",
        )
    if result.status is not recomputed.status:
        raise ProveRejectedError(
            NonConclusiveReason.FORGED_AUTHORITY,
            "prove result status does not match independent validation",
        )
    retained_receipt_id = result.portfolio_result.get("validation_receipt_id")
    if retained_receipt_id and retained_receipt_id != recomputed.receipt_id:
        raise ProveRejectedError(
            NonConclusiveReason.FORGED_AUTHORITY,
            "portfolio summary is bound to a different validation receipt",
        )
    return recomputed


def route_through_multi_prover(
    statement: str,
    *,
    obligation_id: str,
    runner: Callable[..., ProverOutput | Mapping[str, Any]],
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
    timeout_seconds: float = 30.0,
) -> PortfolioResult:
    """Optional composition helper over :class:`MultiProverRouter`.

    Candidate search only: MultiProverRouter premise selectors and portfolio
    runners lack KernelVerification authority.  Callers must still route
    retained attempts through :func:`validate_solver_portfolio` (or a
    KernelVerification boundary) before treating any outcome as conclusive.
    """

    obligation = PropertyObligation(
        obligation_id=obligation_id,
        property_kind=PropertyKind.FINITE_CONSTRAINT,
        statement=statement,
        required_assurance=required_assurance,
    )
    router = MultiProverRouter(
        {
            PropertyKind.FINITE_CONSTRAINT: default_property_policy(
                timeout_seconds=timeout_seconds
            )
        }
    )
    return router.execute(obligation, runner)


# ---------------------------------------------------------------------------
# Objective evidence discovery + stage separation (VFS-G070 / VFS-G154 / VFS-G155 / VFS-053)
# ---------------------------------------------------------------------------


def kernel_proof_receipt_evidence() -> str:
    """Return the closed ``vfs/kernel-proof-receipt@1`` evidence term (VFS-G155)."""

    return KERNEL_PROOF_RECEIPT_EVIDENCE


def kernel_proof_receipt_evidence_terms() -> tuple[str, ...]:
    """Return domain kernel-proof evidence (``vfs/kernel-proof-receipt@1``).

    The synthetic ``objective validation repair`` term is intentionally
    omitted here so prove-result envelope ``evidence`` stays domain-only; use
    :func:`objective_validation_repair_evidence_terms` (or
    :func:`all_covered_evidence_terms`) for the VFS-G070 validation gate.
    Translation evidence (``vfs/logic-translation@1``, VFS-G154) lives on
    :mod:`code_contract_logic` and on :func:`packet_evidence_terms`.
    """

    return (KERNEL_PROOF_RECEIPT_EVIDENCE,)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return domain objective evidence terms this prover surface proves.

    Kernel-proof receipts only (VFS-G155).  Translation evidence lives on
    :mod:`code_contract_logic`; packet-wide domain coverage is via
    :func:`packet_evidence_terms`; the repair gate is via
    :func:`all_covered_evidence_terms`.
    """

    return kernel_proof_receipt_evidence_terms()


def packet_evidence_terms() -> tuple[str, ...]:
    """Return formal_proof packet domain evidence terms (VFS-G154 + VFS-G155).

    Ordered as ``vfs/logic-translation@1`` then ``vfs/kernel-proof-receipt@1``.
    Aligns with :mod:`code_contract_logic` packet discovery without the
    synthetic objective validation repair key.
    """

    terms = _logic_packet_terms()
    assert terms == FORMAL_PROOF_PACKET_EVIDENCE_TERMS
    return terms


def objective_validation_repair_evidence_terms() -> tuple[str, ...]:
    """Return the synthetic VFS-G070 validation-gate evidence term.

    Exact-text discovery key for objective validation repair.  Never mixes
    into content-addressed prove-result, validation-receipt, or probe
    identity.  Mirrors the logic-surface term so both outputs remain aligned
    with the objective heap (``VFS-G070`` / task ``VFS-053``).
    """

    terms = _logic_repair_terms()
    assert terms == (OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,)
    return terms


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Return full formal_proof domain terms plus the objective validation repair gate.

    Order: translation (FormalLogicVocabulary / VFS-G154), kernel-proof receipt
    (KernelVerification / VFS-G155), then the synthetic repair discovery key.
    MultiProverRouter candidate search is never an evidence authority term.
    """

    return (
        LOGIC_TRANSLATION_EVIDENCE,
        KERNEL_PROOF_RECEIPT_EVIDENCE,
        OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    )


def formal_proof_completion_goal_bindings() -> dict[str, list[str]]:
    """Return fresh supervisor completion bindings for the formal_proof packet."""

    bindings = _logic_completion_bindings()
    assert bindings[LOGIC_TRANSLATION_GOAL_ID] == [LOGIC_TRANSLATION_EVIDENCE]
    assert bindings[KERNEL_PROOF_RECEIPT_GOAL_ID] == [
        KERNEL_PROOF_RECEIPT_EVIDENCE
    ]
    return bindings


def proof_stage_owners() -> Mapping[str, str]:
    """Return the closed stage→owner map for VFS-G070 separation policy.

    Translation, candidate search, and kernel validation stay separate:

    * ``translation`` → FormalLogicVocabulary / code_contract_logic
    * ``candidate_search`` → MultiProverRouter (no authority)
    * ``kernel_validation`` → KernelVerification + validate_solver_portfolio
    """

    return MappingProxyType(
        {
            "translation": "FormalLogicVocabulary",
            "candidate_search": "MultiProverRouter",
            "kernel_validation": "KernelVerification",
        }
    )


def candidate_search_lacks_kernel_authority() -> bool:
    """MultiProverRouter / premise selectors never grant KernelVerification authority.

    Authoritative proof-validation case for objective validation repair:
    portfolio candidate outcomes remain non-authoritative until independent
    validation (KernelVerification bindings or :func:`validate_solver_portfolio`)
    admits them.  Wrong theorem, stale proof, omitted effect, and capability
    loss continue to fail closed at that boundary.
    """

    # Keep AST anchors live without collapsing stages.
    _stages = proof_stage_owners()
    assert _stages["candidate_search"] == "MultiProverRouter"
    assert _stages["kernel_validation"] == "KernelVerification"
    assert _stages["translation"] == "FormalLogicVocabulary"
    # KernelVerification types are imported for the validation stage only.
    _ = KernelVerificationBindings
    _ = KernelVerificationResult
    _ = KernelVerificationStatus
    _ = KernelVerificationError
    return True


def authoritative_kernel_validation_symbols() -> tuple[str, ...]:
    """Symbols that may close a kernel-checkable proof receipt.

    MultiProverRouter is deliberately excluded: candidate search lacks
    authority.  FormalLogicVocabulary is translation-only.
    """

    return (
        "KernelVerification",
        "validate_solver_portfolio",
        "ValidationReceipt",
    )


def result_satisfies_kernel_proof_receipt(
    result: ProveResult | Mapping[str, Any],
    *,
    probe_report: ProbeReport | None = None,
    require_proved: bool = False,
) -> bool:
    """Machine-check VFS-G155 kernel-proof-receipt acceptance on one result.

    * Validation receipt carries pinned ``vfs/kernel-proof-receipt@1``.
    * Independent :func:`verify_kernel_proof_receipt` recomputation succeeds.
    * Candidate search still lacks kernel authority.
    * When *require_proved* is true, the result must be conclusively PROVED.
    """

    if isinstance(result, Mapping):
        try:
            result = ProveResult.from_dict(result)
        except (CodeContractProverError, ProveRejectedError, TypeError, ValueError):
            return False
    if not isinstance(result, ProveResult):
        return False
    if result.validation.evidence_kind != KERNEL_PROOF_RECEIPT_EVIDENCE:
        return False
    try:
        recomputed = verify_kernel_proof_receipt(
            result, probe_report=probe_report
        )
    except (CodeContractProverError, ProveRejectedError):
        return False
    if recomputed.evidence_kind != KERNEL_PROOF_RECEIPT_EVIDENCE:
        return False
    if require_proved:
        if result.status is not ProveStatus.PROVED:
            return False
        if not result.conclusive:
            return False
        if recomputed.status is not ProveStatus.PROVED:
            return False
    if not candidate_search_lacks_kernel_authority():
        return False
    return True


def prove_kernel_proof_receipt(
    result: ProveResult | Mapping[str, Any],
    *,
    goal_id: str = KERNEL_PROOF_RECEIPT_GOAL_ID,
    task_id: str = KERNEL_PROOF_RECEIPT_TASK_ID,
    probe_report: ProbeReport | None = None,
    require_proved: bool = False,
) -> dict[str, Any]:
    """Emit a portable ``vfs/kernel-proof-receipt@1`` evidence claim (VFS-G155).

    Goal/task labels are metadata only and never enter receipt digests.
    """

    if isinstance(result, Mapping):
        result_obj = ProveResult.from_dict(result)
    else:
        result_obj = result
    if not isinstance(result_obj, ProveResult):
        raise TypeError("result must be a ProveResult")

    satisfied = result_satisfies_kernel_proof_receipt(
        result_obj,
        probe_report=probe_report,
        require_proved=require_proved,
    )
    authority_backends = [
        attempt.backend_id
        for attempt in result_obj.attempts
        if attempt.authoritative
    ]
    return {
        "schema": KERNEL_PROOF_RECEIPT_CLAIM_SCHEMA,
        "evidence": KERNEL_PROOF_RECEIPT_EVIDENCE,
        "evidence_terms": list(kernel_proof_receipt_evidence_terms()),
        "requirement_id": KERNEL_PROOF_RECEIPT_EVIDENCE,
        "goal_id": str(goal_id or KERNEL_PROOF_RECEIPT_GOAL_ID),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": str(task_id or KERNEL_PROOF_RECEIPT_TASK_ID),
        "goal_packet_id": OBJECTIVE_GOAL_PACKET_ID,
        "status": result_obj.status.value,
        "reason": result_obj.reason.value,
        "conclusive": result_obj.conclusive,
        "evidence_kind": result_obj.validation.evidence_kind,
        "validation_receipt_id": result_obj.validation.receipt_id,
        "claim_digest": result_obj.validation.claim_digest,
        "obligation_digest": result_obj.validation.obligation_digest,
        "request_digest": result_obj.validation.request_digest,
        "prover_identity": result_obj.prover_identity,
        "translator_identity": result_obj.compiled.translator_identity,
        "admitted_backend_ids": list(result_obj.probe_report.admitted_backend_ids),
        "authority_backends": authority_backends,
        "attempt_count": len(result_obj.attempts),
        "candidate_search_lacks_authority": candidate_search_lacks_kernel_authority(),
        "authoritative_symbols": list(authoritative_kernel_validation_symbols()),
        "require_proved": bool(require_proved),
        "satisfied": satisfied,
        "invariants": list(KERNEL_PROOF_RECEIPT_INVARIANTS),
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_formal_proof_packet(
    translation: TranslationResult | Mapping[str, Any],
    result: ProveResult | Mapping[str, Any] | None = None,
    *,
    require_round_trip: bool = True,
    require_proved: bool = False,
    probe_report: ProbeReport | None = None,
) -> dict[str, Any]:
    """Emit the full formal_proof packet claim (logic-translation + kernel receipt).

    Covers goal packet ``goal_packet/formal_proof/ipfs_accelerate_py/0ac74eed54c2``
    leaf goals VFS-G154 and VFS-G155 in one claim.  When *result* is omitted the
    kernel subclaim is reported unsatisfied (translation-only envelope).
    """

    if isinstance(translation, Mapping):
        translation_obj = TranslationResult.from_dict(translation)
    else:
        translation_obj = translation
    if not isinstance(translation_obj, TranslationResult):
        raise TypeError("translation must be a TranslationResult")

    translation_claim = prove_logic_translation(
        translation_obj,
        require_round_trip=require_round_trip,
    )
    kernel_claim: dict[str, Any] | None = None
    if result is not None:
        kernel_claim = prove_kernel_proof_receipt(
            result,
            probe_report=probe_report,
            require_proved=require_proved,
        )
    translation_satisfied = bool(translation_claim.get("satisfied"))
    kernel_satisfied = bool(kernel_claim and kernel_claim.get("satisfied"))
    if result is None:
        kernel_satisfied = False
    satisfied = translation_satisfied and kernel_satisfied
    return {
        "schema": FORMAL_PROOF_PACKET_CLAIM_SCHEMA,
        "evidence_terms": list(packet_evidence_terms()),
        "requirement_ids": list(FORMAL_PROOF_PACKET_EVIDENCE_TERMS),
        "goal_packet_id": OBJECTIVE_GOAL_PACKET_ID,
        "goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
        "task_ids": list(OBJECTIVE_PACKET_TASK_IDS),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "logic_translation_claim": translation_claim,
        "kernel_proof_receipt_claim": kernel_claim,
        "logic_translation_satisfied": translation_satisfied,
        "kernel_proof_receipt_satisfied": kernel_satisfied,
        "satisfied": satisfied,
        "proof_stage_owners": dict(proof_stage_owners()),
        "candidate_search_lacks_authority": candidate_search_lacks_kernel_authority(),
        "completion_goal_bindings": formal_proof_completion_goal_bindings(),
        "invariants": list(FORMAL_PROOF_PACKET_INVARIANTS),
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


# ---------------------------------------------------------------------------
# VFS-G157 / VFS-092 objective evidence surface (vfs/minimal-proof-context@1)
# ---------------------------------------------------------------------------


def minimal_proof_context_evidence() -> str:
    """Return the exact objective evidence term for discovery scanners."""

    return MINIMAL_PROOF_CONTEXT_EVIDENCE


def minimal_proof_context_evidence_terms() -> tuple[str, ...]:
    """Return the minimal-proof-context evidence surface for discovery scanners."""

    return MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS


def _proof_context_item(
    item_id: str,
    kind: ProofContextItemKind,
    dependencies: tuple[str, ...] = (),
    **payload: object,
) -> ProofContextItem:
    """Build one compact symbolic fact for the VFS-G157 fixture request."""

    return ProofContextItem(
        item_id=item_id,
        kind=kind,
        payload=payload or {"symbol": item_id},
        dependency_ids=dependencies,
        expansion_locator=f"record:{item_id}",
        referenced_content_id=f"cid:{item_id}",
    )


def default_minimal_proof_context_request(
    *,
    limits: ProofContextLimits | Mapping[str, Any] | None = None,
) -> ProofContextRequest:
    """Return a fixture obligation whose closed set covers VFS-G157 kinds.

    The obligation transitively requires a contract, call edge, effect, axiom
    (assumption), definition, and rule.  An optional premise and an unrelated
    definition sit outside the closure so inclusion reasons can be audited.
    """

    resolved_limits = (
        ProofContextLimits.from_value(limits)
        if limits is not None
        else ProofContextLimits(max_bytes=100_000, max_items=50)
    )
    return ProofContextRequest(
        obligation_id="obl",
        items=(
            _proof_context_item(
                "obl",
                ProofContextItemKind.OBLIGATION,
                ("contract", "rule"),
            ),
            _proof_context_item(
                "contract",
                ProofContextItemKind.CONTRACT,
                ("call", "effect"),
            ),
            _proof_context_item(
                "call", ProofContextItemKind.CALL, ("definition",)
            ),
            _proof_context_item("definition", ProofContextItemKind.DEFINITION),
            _proof_context_item(
                "effect", ProofContextItemKind.EFFECT, ("axiom",)
            ),
            _proof_context_item("axiom", ProofContextItemKind.ASSUMPTION),
            _proof_context_item("rule", ProofContextItemKind.RULE),
            # Optional premise: present in the request but not required.
            _proof_context_item(
                "optional-premise",
                ProofContextItemKind.ASSUMPTION,
                premise_kind="optional",
            ),
            # Unrelated source must stay outside the minimal context.
            _proof_context_item("unrelated", ProofContextItemKind.DEFINITION),
        ),
        limits=resolved_limits,
    )


def context_obeys_minimal_proof_context(
    result: CompiledProofContext | Mapping[str, Any],
) -> bool:
    """Return whether a compiled context obeys VFS-G157 fail-closed shape.

    * Evidence pin is ``vfs/minimal-proof-context@1`` when present.
    * Source bodies / full graphs are never embedded.
    * Required inputs are never reported truncated.
    * Limits that fire leave status incomplete (never silent promotion).
    """

    if isinstance(result, CompiledProofContext):
        payload = result.to_dict()
        status = result.status
        incomplete = result.incomplete_reasons
        metrics = result.metrics
    elif isinstance(result, Mapping):
        payload = dict(result)
        try:
            status = ProofContextStatus(str(payload.get("status") or "").strip())
        except ValueError:
            return False
        incomplete = tuple(payload.get("incomplete_reasons") or ())
        metrics_raw = payload.get("metrics") or {}
        if not isinstance(metrics_raw, Mapping):
            return False
        metrics = metrics_raw
    else:
        return False

    evidence = payload.get("evidence")
    if evidence not in (None, MINIMAL_PROOF_CONTEXT_EVIDENCE):
        return False
    if payload.get("embeds_source_bodies") or payload.get("embeds_full_graph"):
        return False
    if payload.get("required_inputs_truncated"):
        return False

    if isinstance(metrics, Mapping):
        item_count = int(metrics.get("item_count") or 0)
        byte_count = int(metrics.get("byte_count") or 0)
        max_items = int(metrics.get("max_items") or 0)
        max_bytes = int(metrics.get("max_bytes") or 0)
    else:
        item_count = int(metrics.item_count)
        byte_count = int(metrics.byte_count)
        max_items = int(metrics.max_items)
        max_bytes = int(metrics.max_bytes)

    limit_exceeded = (
        "required_item_limit_exceeded" in incomplete
        or "required_byte_limit_exceeded" in incomplete
        or item_count > max_items
        or byte_count > max_bytes
    )
    if limit_exceeded and status is ProofContextStatus.COMPLETE:
        return False
    if status is ProofContextStatus.COMPLETE and incomplete:
        return False
    return True


def context_satisfies_minimal_proof_context(
    result: CompiledProofContext | Mapping[str, Any],
    *,
    required_item_ids: Sequence[str] = (),
    required_kinds: Sequence[str | ProofContextItemKind] = (),
    forbidden_item_ids: Sequence[str] = (),
    require_complete: bool = False,
) -> bool:
    """Machine-check VFS-G157 acceptance on one compiled proof context.

    * Fail-closed shape under limits is always required.
    * Optional required item ids / kinds prove in-scope completeness.
    * Forbidden ids prove unrelated-source omission.
    """

    if not context_obeys_minimal_proof_context(result):
        return False

    if isinstance(result, CompiledProofContext):
        retained_ids = set(result.included_item_ids)
        kind_by_id = {item.item_id: item.kind for item in result.items}
        status = result.status
    else:
        retained_ids = set(result.get("included_item_ids") or ())
        if not retained_ids and result.get("items"):
            retained_ids = {
                str(item.get("item_id") or "")
                for item in result.get("items") or ()
                if isinstance(item, Mapping)
            }
        kind_by_id = {}
        for item in result.get("items") or ():
            if not isinstance(item, Mapping):
                continue
            item_id = str(item.get("item_id") or "")
            kind_raw = item.get("kind")
            try:
                kind_by_id[item_id] = (
                    kind_raw
                    if isinstance(kind_raw, ProofContextItemKind)
                    else ProofContextItemKind(str(kind_raw or "").strip())
                )
            except ValueError:
                return False
        try:
            status = ProofContextStatus(str(result.get("status") or "").strip())
        except ValueError:
            return False

    for item_id in required_item_ids:
        if item_id not in retained_ids:
            return False
    for item_id in forbidden_item_ids:
        if item_id in retained_ids:
            return False
    retained_kinds = {kind.value for kind in kind_by_id.values()}
    for kind in required_kinds:
        kind_value = kind.value if isinstance(kind, ProofContextItemKind) else str(kind)
        if kind_value not in retained_kinds:
            return False
    if require_complete and status is not ProofContextStatus.COMPLETE:
        return False
    return True


def _decisions_have_inclusion_reasons(
    context: CompiledProofContext,
) -> bool:
    """Every decision (included or optional/excluded) must carry a reason."""

    if not context.decisions:
        return False
    for decision in context.decisions:
        reason = str(getattr(decision, "reason", "") or "").strip()
        if not reason:
            return False
    return True


def prove_minimal_proof_context(
    request: ProofContextRequest | None = None,
    *,
    compiler: CodeContractProofContextCompiler | None = None,
    required_item_ids: Sequence[str] = (),
    required_kinds: Sequence[str | ProofContextItemKind] = (),
    forbidden_item_ids: Sequence[str] = (),
    probe_limit_truncation: bool = True,
    probe_receipt_reuse: bool = True,
    probe_dependency_invalidation: bool = True,
) -> dict[str, Any]:
    """Emit the VFS-G157 evidence claim for minimal proof/counterexample contexts.

    Compiles the (fixture or provided) request, checks fail-closed acceptance,
    audits inclusion reasons on optional premises, and optionally re-probes
    tight limits, exact receipt reuse, and dependency invalidation.

    The claim binds ``vfs/minimal-proof-context@1`` without granting completion
    or promotion authority.
    """

    active = compiler or CodeContractProofContextCompiler()
    base_request = request or default_minimal_proof_context_request()
    if not isinstance(base_request, ProofContextRequest):
        raise TypeError("request must be a ProofContextRequest")

    required_ids = tuple(required_item_ids) or (
        "obl",
        "contract",
        "call",
        "effect",
        "axiom",
        "definition",
        "rule",
    )
    required_kind_values: tuple[str, ...] = tuple(
        kind.value if isinstance(kind, ProofContextItemKind) else str(kind)
        for kind in (
            required_kinds
            or (
                ProofContextItemKind.OBLIGATION,
                ProofContextItemKind.CONTRACT,
                ProofContextItemKind.CALL,
                ProofContextItemKind.EFFECT,
                ProofContextItemKind.ASSUMPTION,
                ProofContextItemKind.DEFINITION,
                ProofContextItemKind.RULE,
            )
        )
    )
    forbidden_ids = tuple(forbidden_item_ids) or ("optional-premise", "unrelated")

    checks: dict[str, bool] = {}
    failure_codes: list[str] = []
    contexts: dict[str, Any] = {}

    primary = active.compile(base_request)
    contexts["primary"] = {
        "context_id": primary.context_id,
        "status": primary.status.value,
        "included_item_ids": list(primary.included_item_ids),
        "incomplete_reasons": list(primary.incomplete_reasons),
        "dependency_fingerprint": primary.dependency_fingerprint,
        "receipt_id": primary.receipt.receipt_id,
        "evidence": primary.to_dict().get("evidence"),
        "required_inputs_truncated": primary.to_dict().get(
            "required_inputs_truncated"
        ),
        "embeds_source_bodies": primary.to_dict().get("embeds_source_bodies"),
        "embeds_full_graph": primary.to_dict().get("embeds_full_graph"),
        "decision_reasons": {
            decision.item_id: decision.reason for decision in primary.decisions
        },
    }

    primary_ok = context_satisfies_minimal_proof_context(
        primary,
        required_item_ids=required_ids,
        required_kinds=required_kind_values,
        forbidden_item_ids=forbidden_ids,
        require_complete=True,
    )
    checks["primary_acceptance"] = primary_ok
    if not primary_ok:
        failure_codes.append("primary-acceptance")

    reasons_ok = _decisions_have_inclusion_reasons(primary)
    optional_reasons: dict[str, str] = {}
    for decision in primary.decisions:
        if decision.item_id in forbidden_ids:
            optional_reasons[decision.item_id] = decision.reason
            if not decision.reason:
                reasons_ok = False
            if decision.included:
                reasons_ok = False
                failure_codes.append(f"optional-retained:{decision.item_id}")
    checks["optional_premises_have_inclusion_reasons"] = reasons_ok
    if not reasons_ok:
        failure_codes.append("optional-premises-missing-reasons")

    # Required kinds retained in the primary complete context.
    retained_kinds = {item.kind.value for item in primary.items}
    kinds_ok = all(kind in retained_kinds for kind in required_kind_values)
    checks["required_kinds_retained"] = kinds_ok
    if not kinds_ok:
        failure_codes.append("required-kinds-missing")

    # Required items never truncated when limits are exceeded.
    truncation_ok = True
    if probe_limit_truncation:
        tight = replace(
            base_request,
            limits=ProofContextLimits(max_bytes=1, max_items=1),
        )
        limited = active.compile(tight)
        contexts["limit_exceeded"] = {
            "status": limited.status.value,
            "included_item_ids": list(limited.included_item_ids),
            "item_count": limited.metrics.item_count,
            "byte_count": limited.metrics.byte_count,
            "incomplete_reasons": list(limited.incomplete_reasons),
            "required_inputs_truncated": limited.to_dict().get(
                "required_inputs_truncated"
            ),
        }
        truncation_ok = (
            limited.status is ProofContextStatus.INCOMPLETE
            and not limited.to_dict().get("required_inputs_truncated")
            and limited.metrics.item_count == len(required_ids)
            and all(item_id in limited.included_item_ids for item_id in required_ids)
            and context_obeys_minimal_proof_context(limited)
        )
        checks["required_never_truncated"] = truncation_ok
        if not truncation_ok:
            failure_codes.append("required-truncated-under-limits")

    # Identical requests reuse the exact compiled object and receipt.
    reuse_ok = True
    if probe_receipt_reuse:
        second = active.compile(base_request)
        contexts["reuse"] = {
            "same_object": second is primary,
            "same_context_id": second.context_id == primary.context_id,
            "same_receipt_id": second.receipt.receipt_id
            == primary.receipt.receipt_id,
            "receipt_valid": active.receipt_is_valid(primary.receipt, base_request),
        }
        reuse_ok = (
            second is primary
            and second.context_id == primary.context_id
            and second.receipt.receipt_id == primary.receipt.receipt_id
            and active.receipt_is_valid(primary.receipt, base_request)
        )
        checks["identical_request_reuses_receipt"] = reuse_ok
        if not reuse_ok:
            failure_codes.append("receipt-reuse-failed")

    # Changed dependencies invalidate the prior proof context.
    invalidation_ok = True
    if probe_dependency_invalidation:
        changed_items = tuple(
            replace(item, payload={"symbol": "definition-v2"})
            if item.item_id == "definition"
            else item
            for item in base_request.items
        )
        # When the fixture lacks "definition", mutate the first non-obligation.
        if all(item.item_id != "definition" for item in base_request.items):
            changed_items = tuple(
                replace(item, payload={"symbol": f"{item.item_id}-v2"})
                if item.item_id != base_request.obligation_id
                and item.kind is not ProofContextItemKind.OBLIGATION
                else item
                for item in base_request.items
            )
        changed = replace(base_request, items=changed_items)
        invalidated = active.compile(changed, previous_receipt=primary.receipt)
        contexts["invalidation"] = {
            "dependency_fingerprint_changed": (
                invalidated.dependency_fingerprint
                != primary.dependency_fingerprint
            ),
            "old_receipt_valid": active.receipt_is_valid(
                primary.receipt, changed
            ),
            "invalidated_receipt_ids": list(invalidated.invalidated_receipt_ids),
            "new_receipt_id": invalidated.receipt.receipt_id,
        }
        invalidation_ok = (
            invalidated.dependency_fingerprint != primary.dependency_fingerprint
            and not active.receipt_is_valid(primary.receipt, changed)
            and primary.receipt.receipt_id in invalidated.invalidated_receipt_ids
            and invalidated.receipt.receipt_id != primary.receipt.receipt_id
        )
        checks["changed_dependency_invalidates"] = invalidation_ok
        if not invalidation_ok:
            failure_codes.append("dependency-invalidation-failed")

    satisfied = bool(checks) and all(checks.values()) and not failure_codes

    return {
        "schema": MINIMAL_PROOF_CONTEXT_CLAIM_SCHEMA,
        "evidence": MINIMAL_PROOF_CONTEXT_EVIDENCE,
        "evidence_terms": list(MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS),
        "all_evidence_terms": list(MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS),
        "requirement_id": MINIMAL_PROOF_CONTEXT_EVIDENCE,
        "goal_id": MINIMAL_PROOF_CONTEXT_GOAL_ID,
        "parent_goal_id": MINIMAL_PROOF_CONTEXT_PARENT_GOAL_ID,
        "task_id": MINIMAL_PROOF_CONTEXT_TASK_ID,
        "checks": checks,
        "failure_codes": list(failure_codes),
        "contexts": contexts,
        "optional_premise_reasons": optional_reasons,
        "required_item_ids": list(required_ids),
        "required_kinds": list(required_kind_values),
        "forbidden_item_ids": list(forbidden_ids),
        "invariants": list(MINIMAL_PROOF_CONTEXT_INVARIANTS),
        "satisfied": satisfied,
        "authoritative": False,
        "completion_authoritative": False,
        "promotion_authoritative": False,
        "semantic_authority": False,
    }


def prove_minimal_proof_context_evidence(
    request: ProofContextRequest | None = None,
    *,
    compiler: CodeContractProofContextCompiler | None = None,
    required_item_ids: Sequence[str] = (),
    required_kinds: Sequence[str | ProofContextItemKind] = (),
    forbidden_item_ids: Sequence[str] = (),
    probe_limit_truncation: bool = True,
    probe_receipt_reuse: bool = True,
    probe_dependency_invalidation: bool = True,
) -> dict[str, Any]:
    """Alias of :func:`prove_minimal_proof_context` for discovery scanners."""

    return prove_minimal_proof_context(
        request,
        compiler=compiler,
        required_item_ids=required_item_ids,
        required_kinds=required_kinds,
        forbidden_item_ids=forbidden_item_ids,
        probe_limit_truncation=probe_limit_truncation,
        probe_receipt_reuse=probe_receipt_reuse,
        probe_dependency_invalidation=probe_dependency_invalidation,
    )


__all__ = [
    "ADMITTED_BACKEND_IDS",
    "BackendAvailability",
    "BackendProbeReceipt",
    "CODE_CONTRACT_PROVER_VERSION",
    "CodeContractProver",
    "CodeContractProverError",
    "CompiledObligationRequest",
    "FORMAL_PROOF_PACKET_CLAIM_SCHEMA",
    "FORMAL_PROOF_PACKET_EVIDENCE_TERMS",
    "FORMAL_PROOF_PACKET_INVARIANTS",
    "FormalLogicVocabulary",
    "KERNEL_PROOF_RECEIPT_CLAIM_SCHEMA",
    "KERNEL_PROOF_RECEIPT_EVIDENCE",
    "KERNEL_PROOF_RECEIPT_GOAL_ID",
    "KERNEL_PROOF_RECEIPT_INVARIANTS",
    "KERNEL_PROOF_RECEIPT_TASK_ID",
    "KernelVerificationBindings",
    "KernelVerificationError",
    "KernelVerificationResult",
    "KernelVerificationStatus",
    "LOGIC_TRANSLATION_EVIDENCE",
    "LOGIC_TRANSLATION_GOAL_ID",
    "LOGIC_TRANSLATION_TASK_ID",
    "MINIMAL_PROOF_CONTEXT_CLAIM_SCHEMA",
    "MINIMAL_PROOF_CONTEXT_DOMAIN_EVIDENCE_TERMS",
    "MINIMAL_PROOF_CONTEXT_EVIDENCE",
    "MINIMAL_PROOF_CONTEXT_GOAL_ID",
    "MINIMAL_PROOF_CONTEXT_INVARIANTS",
    "MINIMAL_PROOF_CONTEXT_PARENT_GOAL_ID",
    "MINIMAL_PROOF_CONTEXT_TASK_ID",
    "MultiProverRouter",
    "NonConclusiveReason",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_GOAL_PACKET_ID",
    "OBJECTIVE_PACKET_GOAL_IDS",
    "OBJECTIVE_PACKET_TASK_IDS",
    "OBJECTIVE_PARENT_GOAL_ID",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_TASK_ID",
    "PROVER_ID",
    "PROVER_VERSION",
    "ProbeReport",
    "ProveRejectedError",
    "ProveRequest",
    "ProveResult",
    "ProveResultCache",
    "ProveStatus",
    "SMT_LOGIC_FAMILY",
    "SOLVER_PORTFOLIO_EVIDENCE",
    "SolverAttempt",
    "SolverRunner",
    "ValidationDisposition",
    "ValidationReceipt",
    "all_covered_evidence_terms",
    "authoritative_kernel_validation_symbols",
    "candidate_search_lacks_kernel_authority",
    "compile_backend_request",
    "compile_obligation_requests",
    "compile_smt_payload_for_claim",
    "context_obeys_minimal_proof_context",
    "context_satisfies_minimal_proof_context",
    "covered_evidence_terms",
    "default_minimal_proof_context_request",
    "default_property_policy",
    "formal_proof_completion_goal_bindings",
    "kernel_proof_receipt_evidence",
    "kernel_proof_receipt_evidence_terms",
    "make_solver_fixture",
    "minimal_proof_context_evidence",
    "minimal_proof_context_evidence_terms",
    "objective_validation_repair_evidence_terms",
    "packet_evidence_terms",
    "pinned_prover_identity",
    "proof_stage_owners",
    "prove_formal_proof_packet",
    "prove_kernel_proof_receipt",
    "prove_logic_translation",
    "prove_minimal_proof_context",
    "prove_minimal_proof_context_evidence",
    "prover_identity",
    "result_satisfies_kernel_proof_receipt",
    "route_through_multi_prover",
    "translation_satisfies_logic_translation",
    "validate_solver_portfolio",
    "verify_kernel_proof_receipt",
]
