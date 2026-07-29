"""ZK public inputs, witness policy, capability, and verifier conformance.

This module is the VFS program-assurance zero-knowledge contract surface
(VFS-022 / VFS-G080 and VFS-024 / VFS-G081).  It defines:

* canonical public commitments for repository forest, inventory, contract,
  call slice, assumptions, analyzer/resolver/translator/prover versions,
  supported result, circuit, proving/verifying keys, ceremony, and the
  public-input codec;
* a private witness and redaction policy that never serializes protected
  openings into public artifacts;
* supported deterministic trace transitions that the bounded
  ``program_contract_trace`` circuit may check; and
* production capability probing, ceremony/setup admission, independent
  verifier replay, and fail-closed authority gates (VFS-024).

Trace validity is intentionally narrow.  A verified program-analysis trace
establishes only that committed public inputs open to a witness following the
declared transition rules and that the trace ends in the committed supported
result.  It does **not** prove inventory completeness, translator soundness,
arbitrary runtime semantics, or any theorem beyond that committed result.

Simulated backends, knowledge-graph fail-open fallbacks, placeholder field
encodings, v1 nonzero-only circuits, incompatible TDFOL-only circuits,
unversioned or missing artifacts, and stale capabilities cannot emit
authoritative ZK receipts.  Shadow rollout is the default until every
production probe dimension is production-eligible.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Final, TypeVar

from .program_assurance_contracts import (
    AuthorityKind,
    ClaimLevel,
    ClaimPromotionError,
    ClaimVerdict,
    InconclusiveState,
    SemanticAuthorityError,
    validate_claim_promotion,
)
from .proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    canonical_json_bytes,
    content_identity,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Versioning and schema identities
# ---------------------------------------------------------------------------

PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION

PUBLIC_INPUT_CODEC_ID: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-public-input-codec"
)
PUBLIC_INPUT_CODEC_VERSION: Final[str] = "1"

PROGRAM_ZKP_PUBLIC_INPUTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-public-inputs@1"
)
PROGRAM_ZKP_STATEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-statement@1"
)
PROGRAM_ZKP_TRACE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-trace@1"
)
PROGRAM_ZKP_WITNESS_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-witness-policy@1"
)
PROGRAM_ZKP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-verification-receipt@1"
)
PROGRAM_ZKP_SHADOW_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-shadow-envelope@1"
)
PROGRAM_ZKP_CAPABILITY_CONFORMANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-capability-conformance@1"
)
PROGRAM_ZKP_CAPABILITY_CHECK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-capability-check@1"
)
PROGRAM_ZKP_PROOF_SCHEMA_ID: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-zkp-proof@1"
)
PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE: Final[str] = (
    "vfs/zk-capability-conformance@1"
)

# Production circuit / codec identities for program_contract_trace@v1.
PROGRAM_CONTRACT_TRACE_CIRCUIT_ID: Final[str] = "circuit:program-contract-trace@1"
PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION: Final[int] = 1
PROGRAM_CONTRACT_TRACE_MAX_TRACE_STEPS: Final[int] = 16
PROGRAM_CONTRACT_TRACE_CANONICAL_TRACE_LENGTH: Final[int] = 8
BN254_SCALAR_FIELD_MODULUS: Final[int] = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
FIELD_ENCODING_BN254_SHA256: Final[str] = "bn254_sha256"
# Simulated proof layout magic from educational backends (must never grant authority).
_SIMZKP_MAGIC: Final[bytes] = b"SIMZKP\x00\x01"
_SIMZKP_PROOF_LENGTH: Final[int] = 160

# Normative ordered public-input slots consumed by the codec and (later) circuit.
PUBLIC_COMMITMENT_KEYS: Final[tuple[str, ...]] = (
    "forest_commitment",
    "inventory_commitment",
    "contract_commitment",
    "call_slice_commitment",
    "assumptions_commitment",
    "analyzer_version",
    "resolver_version",
    "translator_version",
    "prover_version",
    "result_commitment",
    "circuit_id",
    "proving_key_id",
    "verifying_key_id",
    "ceremony_id",
    "public_input_codec_id",
    "public_input_codec_version",
)

# Explicit non-claims: what a valid trace must never be promoted into.
TRACE_VALIDITY_DOES_NOT_PROVE: Final[frozenset[str]] = frozenset(
    {
        "inventory_completeness",
        "translator_soundness",
        "arbitrary_runtime_semantics",
        "theorem_beyond_committed_supported_result",
    }
)

TRACE_VALIDITY_SCOPE_STATEMENT: Final[str] = (
    "Trace validity proves only commitment openings and supported deterministic "
    "trace transitions terminating in the committed supported result. It does not "
    "prove inventory completeness, translator soundness, arbitrary runtime "
    "semantics, or a theorem beyond the committed supported result."
)

# Field names that must never appear in public artifacts with live values.
_PRIVATE_WITNESS_MARKERS: Final[frozenset[str]] = frozenset(
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
        "source_text",
        "ast_node",
        "proof_trace",
        "witness",
        "witness_opening",
        "commitment_opening",
        "opening",
    }
)

_SAFE_PUBLIC_REDACTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "private_witness_redacted",
        "witness_redacted",
        "openings_redacted",
    }
)

MAX_TRACE_STEPS: Final[int] = 64
MAX_ASSUMPTION_ITEMS: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 8_192


class ProgramAnalysisZkpError(ContractValidationError):
    """Base error for program-analysis ZK contract violations."""


class ProgramZkpWitnessDisclosureError(ProgramAnalysisZkpError):
    """Raised when private witness material reaches a public boundary."""


class ProgramZkpTraceError(ProgramAnalysisZkpError):
    """Raised when a trace violates supported deterministic transition rules."""


class ProgramZkpTamperError(ProgramAnalysisZkpError):
    """Raised when public inputs or receipts show tampering."""


class ProgramZkpReplayError(ProgramAnalysisZkpError):
    """Raised when a receipt is reused against drifted or mismatched inputs."""


class ProgramZkpVersionError(ProgramAnalysisZkpError):
    """Raised when codec, circuit, or key versions are incompatible."""


class ProgramZkpClaimPromotionError(ProgramAnalysisZkpError, ClaimPromotionError):
    """Raised when a ZK trace claim is illegally promoted to a stronger claim."""


class ProgramZkpCapabilityError(ProgramAnalysisZkpError):
    """Raised when production ZK capability or ceremony probes fail closed."""


class ProgramZkpAuthorityError(ProgramAnalysisZkpError):
    """Raised when a non-production path attempts to claim ZK authority."""


class ProgramZkpBackendMode(str, Enum):
    """Trust class of the program-analysis ZK path."""

    CRYPTOGRAPHIC = "cryptographic"
    SIMULATED = "simulated"
    SHADOW = "shadow"


class ProgramZkpTrust(str, Enum):
    """Authority derived only after independent verification of a real backend."""

    NON_AUTHORITATIVE = "non_authoritative"
    AUTHORITATIVE = "authoritative"


class ProgramZkpVerdict(str, Enum):
    """Independent verifier result for a program-analysis ZK envelope."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    ERROR = "error"
    SKIPPED = "skipped"


class TraceState(str, Enum):
    """Finite control states of a supported program-assurance ZK trace."""

    INIT = "init"
    FOREST_OPENED = "forest_opened"
    INVENTORY_OPENED = "inventory_opened"
    CONTRACT_OPENED = "contract_opened"
    CALL_SLICE_OPENED = "call_slice_opened"
    ASSUMPTIONS_OPENED = "assumptions_opened"
    VERSIONS_BOUND = "versions_bound"
    RESULT_COMMITTED = "result_committed"
    TERMINAL = "terminal"


class TraceTransitionKind(str, Enum):
    """Atomic deterministic transitions admitted by the contract."""

    OPEN_FOREST = "open_forest"
    OPEN_INVENTORY = "open_inventory"
    OPEN_CONTRACT = "open_contract"
    OPEN_CALL_SLICE = "open_call_slice"
    OPEN_ASSUMPTIONS = "open_assumptions"
    BIND_VERSIONS = "bind_versions"
    COMMIT_RESULT = "commit_result"
    TERMINATE = "terminate"


# Ordered, exclusive transition table: each source has exactly one legal next.
SUPPORTED_TRACE_TRANSITIONS: Final[tuple[tuple[TraceState, TraceTransitionKind, TraceState], ...]] = (
    (TraceState.INIT, TraceTransitionKind.OPEN_FOREST, TraceState.FOREST_OPENED),
    (
        TraceState.FOREST_OPENED,
        TraceTransitionKind.OPEN_INVENTORY,
        TraceState.INVENTORY_OPENED,
    ),
    (
        TraceState.INVENTORY_OPENED,
        TraceTransitionKind.OPEN_CONTRACT,
        TraceState.CONTRACT_OPENED,
    ),
    (
        TraceState.CONTRACT_OPENED,
        TraceTransitionKind.OPEN_CALL_SLICE,
        TraceState.CALL_SLICE_OPENED,
    ),
    (
        TraceState.CALL_SLICE_OPENED,
        TraceTransitionKind.OPEN_ASSUMPTIONS,
        TraceState.ASSUMPTIONS_OPENED,
    ),
    (
        TraceState.ASSUMPTIONS_OPENED,
        TraceTransitionKind.BIND_VERSIONS,
        TraceState.VERSIONS_BOUND,
    ),
    (
        TraceState.VERSIONS_BOUND,
        TraceTransitionKind.COMMIT_RESULT,
        TraceState.RESULT_COMMITTED,
    ),
    (
        TraceState.RESULT_COMMITTED,
        TraceTransitionKind.TERMINATE,
        TraceState.TERMINAL,
    ),
)

_TRANSITION_INDEX: Final[
    dict[tuple[TraceState, TraceTransitionKind], TraceState]
] = {
    (source, kind): target for source, kind, target in SUPPORTED_TRACE_TRANSITIONS
}

_CANONICAL_TRANSITION_SEQUENCE: Final[tuple[TraceTransitionKind, ...]] = tuple(
    kind for _, kind, _ in SUPPORTED_TRACE_TRANSITIONS
)

# Claim levels that a verified ZK trace may never be promoted into.
_ILLEGAL_ZK_PROMOTION_TARGETS: Final[frozenset[ClaimLevel]] = frozenset(
    {
        ClaimLevel.MODEL_PROVED,
        ClaimLevel.MODEL_DISPROVED,
        ClaimLevel.RUNTIME_WITNESSED,
        ClaimLevel.OBSERVED_SYNTAX,
        ClaimLevel.RESOLVED_STATIC,
    }
)


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        normalized = ""
    elif not isinstance(value, str):
        raise ProgramAnalysisZkpError("%s must be a string" % field_name)
    else:
        normalized = value.strip()
    if required and not normalized:
        raise ProgramAnalysisZkpError("%s is required" % field_name)
    if "\x00" in normalized:
        raise ProgramAnalysisZkpError("%s must not contain NUL" % field_name)
    if len(normalized.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ProgramAnalysisZkpError(
            "%s exceeds %s UTF-8 bytes" % (field_name, MAX_TEXT_BYTES)
        )
    return normalized


def _enum(value: Any, enum_type: type[T], *, field_name: str) -> T:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)  # type: ignore[attr-defined]
        raise ProgramAnalysisZkpError(
            "%s must be one of: %s" % (field_name, allowed)
        ) from exc


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramAnalysisZkpError("%s must be a boolean" % field_name)
    return value


def commitment_identity(label: str, material: Any) -> str:
    """Return a content identity for a labeled public commitment payload."""

    name = _text(label, field_name="commitment_label")
    return content_identity({"commitment": name, "material": _canonical_value(material)})


def encode_public_input_vector(public_inputs: Mapping[str, str]) -> tuple[str, ...]:
    """Encode public inputs into the canonical ordered codec vector.

    The vector is the sole ordered representation a circuit adapter may use.
    Missing or extra keys fail closed.
    """

    if not isinstance(public_inputs, Mapping):
        raise ProgramAnalysisZkpError("public_inputs must be a mapping")
    keys = tuple(public_inputs.keys())
    if set(keys) != set(PUBLIC_COMMITMENT_KEYS):
        missing = sorted(set(PUBLIC_COMMITMENT_KEYS) - set(keys))
        extra = sorted(set(keys) - set(PUBLIC_COMMITMENT_KEYS))
        raise ProgramAnalysisZkpError(
            "public_inputs keys mismatch; missing=%s extra=%s" % (missing, extra)
        )
    if keys != PUBLIC_COMMITMENT_KEYS and sorted(keys) == sorted(PUBLIC_COMMITMENT_KEYS):
        # Allow dicts that are not insertion-ordered as long as all keys exist;
        # the returned vector is always in canonical order.
        pass
    vector: list[str] = []
    for key in PUBLIC_COMMITMENT_KEYS:
        value = public_inputs[key]
        if not isinstance(value, str) or not value.strip():
            raise ProgramAnalysisZkpError(
                "public input %s must be a non-empty string" % key
            )
        vector.append(value.strip())
    return tuple(vector)


def public_input_vector_digest(public_inputs: Mapping[str, str]) -> str:
    """Content-address the canonical ordered public-input vector."""

    vector = encode_public_input_vector(public_inputs)
    return content_identity(
        {
            "codec_id": PUBLIC_INPUT_CODEC_ID,
            "codec_version": PUBLIC_INPUT_CODEC_VERSION,
            "vector": list(vector),
        }
    )


def supported_transition_table() -> tuple[dict[str, str], ...]:
    """Return the immutable supported transition table as plain records."""

    return tuple(
        {
            "source": source.value,
            "kind": kind.value,
            "target": target.value,
        }
        for source, kind, target in SUPPORTED_TRACE_TRANSITIONS
    )


def next_trace_state(
    current: TraceState | str, kind: TraceTransitionKind | str
) -> TraceState:
    """Return the unique legal successor or raise :class:`ProgramZkpTraceError`."""

    state = _enum(current, TraceState, field_name="current")
    transition = _enum(kind, TraceTransitionKind, field_name="kind")
    target = _TRANSITION_INDEX.get((state, transition))
    if target is None:
        raise ProgramZkpTraceError(
            "unsupported transition %s from state %s" % (transition.value, state.value)
        )
    return target


def canonical_trace_transition_kinds() -> tuple[TraceTransitionKind, ...]:
    """Return the single legal full-trace transition sequence."""

    return _CANONICAL_TRANSITION_SEQUENCE


def trace_validity_does_not_prove(claim: str) -> bool:
    """Return True when ``claim`` is an explicit non-claim of trace validity."""

    return _text(claim, field_name="claim") in TRACE_VALIDITY_DOES_NOT_PROVE


def reject_illegal_zk_claim_promotion(
    source: ClaimLevel | str,
    target: ClaimLevel | str,
) -> None:
    """Reject promoting a ZK trace claim into a stronger semantic claim class.

    Also rejects treating any other claim level as if it were a ZK trace
    attestation, matching the non-hierarchical claim vocabulary.
    """

    source_level = _enum(source, ClaimLevel, field_name="source")
    target_level = _enum(target, ClaimLevel, field_name="target")
    if source_level is ClaimLevel.ZK_TRACE_ATTESTED:
        if target_level is ClaimLevel.ZK_TRACE_ATTESTED:
            return
        if target_level in _ILLEGAL_ZK_PROMOTION_TARGETS:
            raise ProgramZkpClaimPromotionError(
                "zk_trace_attested cannot be promoted to %s; %s"
                % (target_level.value, TRACE_VALIDITY_SCOPE_STATEMENT)
            )
        raise ProgramZkpClaimPromotionError(
            "zk_trace_attested cannot be promoted to %s" % target_level.value
        )
    if target_level is ClaimLevel.ZK_TRACE_ATTESTED and source_level is not target_level:
        raise ProgramZkpClaimPromotionError(
            "%s cannot be promoted to zk_trace_attested" % source_level.value
        )
    # Fall through to the general non-hierarchical claim rule.
    validate_claim_promotion(source_level, target_level)


# ---------------------------------------------------------------------------
# Public inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramZkpPublicInputs(CanonicalContract):
    """Canonical public commitments for one program-analysis ZK statement.

    Every field is a public identity or commitment digest.  No commitment
    openings, source text, ASTs, or private witness material are accepted.
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_PUBLIC_INPUTS_SCHEMA

    forest_commitment: str
    inventory_commitment: str
    contract_commitment: str
    call_slice_commitment: str
    assumptions_commitment: str
    analyzer_version: str
    resolver_version: str
    translator_version: str
    prover_version: str
    result_commitment: str
    circuit_id: str
    proving_key_id: str
    verifying_key_id: str
    ceremony_id: str
    public_input_codec_id: str = PUBLIC_INPUT_CODEC_ID
    public_input_codec_version: str = PUBLIC_INPUT_CODEC_VERSION

    def __post_init__(self) -> None:
        for name in PUBLIC_COMMITMENT_KEYS:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        if self.public_input_codec_id != PUBLIC_INPUT_CODEC_ID:
            raise ProgramZkpVersionError(
                "public_input_codec_id must be %s" % PUBLIC_INPUT_CODEC_ID
            )
        if self.public_input_codec_version != PUBLIC_INPUT_CODEC_VERSION:
            raise ProgramZkpVersionError(
                "public_input_codec_version must be %s" % PUBLIC_INPUT_CODEC_VERSION
            )

    @property
    def public_inputs(self) -> Dict[str, str]:
        """Return the exact circuit-facing public-input map."""

        return {key: getattr(self, key) for key in PUBLIC_COMMITMENT_KEYS}

    @property
    def public_input_vector(self) -> tuple[str, ...]:
        return encode_public_input_vector(self.public_inputs)

    @property
    def public_input_digest(self) -> str:
        return public_input_vector_digest(self.public_inputs)

    @property
    def public_inputs_digest(self) -> str:
        return self.public_input_digest

    @property
    def inputs_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            **self.public_inputs,
            "public_input_digest": self.public_input_digest,
            "trace_validity_scope": TRACE_VALIDITY_SCOPE_STATEMENT,
            "does_not_prove": sorted(TRACE_VALIDITY_DOES_NOT_PROVE),
        }

    def to_public_artifact(self) -> Dict[str, Any]:
        return self.to_dict()

    def with_overrides(self, **overrides: str) -> "ProgramZkpPublicInputs":
        """Return a copy with selected public fields replaced (for adversarial tests)."""

        values = self.public_inputs
        values.update(overrides)
        return ProgramZkpPublicInputs(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpPublicInputs":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("public inputs payload must be a mapping")
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("trace_validity_scope", None)
        data.pop("does_not_prove", None)
        claimed_digest = data.pop("public_input_digest", None) or data.pop(
            "public_inputs_digest", None
        )
        claimed_id = data.pop("content_id", None) or data.pop("inputs_id", None)
        # Reject private markers at load time.
        reject_private_witness_from_public_payload(data)
        try:
            result = cls(**{key: data.get(key, "") for key in PUBLIC_COMMITMENT_KEYS})
        except TypeError as exc:
            raise ProgramAnalysisZkpError(
                "public inputs payload is missing required fields"
            ) from exc
        if claimed_digest and claimed_digest != result.public_input_digest:
            raise ProgramZkpTamperError(
                "forged or stale public-input digest rejected"
            )
        if claimed_id and claimed_id != result.content_id:
            raise ProgramZkpTamperError("forged public-input content identity rejected")
        return result


def build_program_zkp_public_inputs(
    *,
    forest_commitment: str,
    inventory_commitment: str,
    contract_commitment: str,
    call_slice_commitment: str,
    assumptions_commitment: str,
    analyzer_version: str,
    resolver_version: str,
    translator_version: str,
    prover_version: str,
    result_commitment: str,
    circuit_id: str,
    proving_key_id: str,
    verifying_key_id: str,
    ceremony_id: str,
    public_input_codec_id: str = PUBLIC_INPUT_CODEC_ID,
    public_input_codec_version: str = PUBLIC_INPUT_CODEC_VERSION,
) -> ProgramZkpPublicInputs:
    """Construct validated program-analysis ZK public inputs."""

    return ProgramZkpPublicInputs(
        forest_commitment=forest_commitment,
        inventory_commitment=inventory_commitment,
        contract_commitment=contract_commitment,
        call_slice_commitment=call_slice_commitment,
        assumptions_commitment=assumptions_commitment,
        analyzer_version=analyzer_version,
        resolver_version=resolver_version,
        translator_version=translator_version,
        prover_version=prover_version,
        result_commitment=result_commitment,
        circuit_id=circuit_id,
        proving_key_id=proving_key_id,
        verifying_key_id=verifying_key_id,
        ceremony_id=ceremony_id,
        public_input_codec_id=public_input_codec_id,
        public_input_codec_version=public_input_codec_version,
    )


# ---------------------------------------------------------------------------
# Trace semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramZkpTraceStep:
    """One deterministic step in a supported program-analysis ZK trace."""

    index: int
    kind: TraceTransitionKind
    source_state: TraceState
    target_state: TraceState
    binding_commitment: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ProgramAnalysisZkpError("trace step index must be a non-negative integer")
        object.__setattr__(
            self, "kind", _enum(self.kind, TraceTransitionKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "source_state",
            _enum(self.source_state, TraceState, field_name="source_state"),
        )
        object.__setattr__(
            self,
            "target_state",
            _enum(self.target_state, TraceState, field_name="target_state"),
        )
        object.__setattr__(
            self,
            "binding_commitment",
            _text(self.binding_commitment, field_name="binding_commitment", required=False),
        )
        expected = _TRANSITION_INDEX.get((self.source_state, self.kind))
        if expected is None or expected is not self.target_state:
            raise ProgramZkpTraceError(
                "step %s is not a supported transition" % self.index
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "source_state": self.source_state.value,
            "target_state": self.target_state.value,
            "binding_commitment": self.binding_commitment,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpTraceStep":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("trace step must be a mapping")
        return cls(
            index=payload.get("index", -1),
            kind=payload.get("kind", ""),
            source_state=payload.get("source_state", ""),
            target_state=payload.get("target_state", ""),
            binding_commitment=payload.get("binding_commitment", ""),
        )


@dataclass(frozen=True)
class ProgramZkpTrace(CanonicalContract):
    """A complete supported deterministic program-analysis ZK trace.

    Validity requires every step to follow :data:`SUPPORTED_TRACE_TRANSITIONS`
    in order, with no omissions, reordering, or duplicate kinds, and to end in
    :attr:`TraceState.TERMINAL`.
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_TRACE_SCHEMA

    steps: tuple[ProgramZkpTraceStep, ...]
    result_commitment: str
    public_input_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_commitment",
            _text(self.result_commitment, field_name="result_commitment"),
        )
        object.__setattr__(
            self,
            "public_input_digest",
            _text(self.public_input_digest, field_name="public_input_digest"),
        )
        if isinstance(self.steps, ProgramZkpTraceStep):
            raise ProgramAnalysisZkpError("steps must be a sequence of trace steps")
        if not isinstance(self.steps, Sequence) or isinstance(self.steps, (str, bytes)):
            raise ProgramAnalysisZkpError("steps must be a sequence of trace steps")
        if len(self.steps) > MAX_TRACE_STEPS:
            raise ProgramAnalysisZkpError(
                "trace exceeds %s steps" % MAX_TRACE_STEPS
            )
        normalized: list[ProgramZkpTraceStep] = []
        for index, raw in enumerate(self.steps):
            step = (
                raw
                if isinstance(raw, ProgramZkpTraceStep)
                else ProgramZkpTraceStep.from_dict(raw)
            )
            if step.index != index:
                raise ProgramZkpTraceError(
                    "trace step indices must be contiguous starting at 0"
                )
            normalized.append(step)
        object.__setattr__(self, "steps", tuple(normalized))
        self._validate_transition_sequence()

    def _validate_transition_sequence(self) -> None:
        expected_kinds = canonical_trace_transition_kinds()
        if len(self.steps) != len(expected_kinds):
            raise ProgramZkpTraceError(
                "trace must contain exactly the supported transition sequence "
                "(%s steps); got %s" % (len(expected_kinds), len(self.steps))
            )
        state = TraceState.INIT
        for step, expected_kind in zip(self.steps, expected_kinds):
            if step.kind is not expected_kind:
                raise ProgramZkpTraceError(
                    "reordered or substituted transition at index %s: "
                    "expected %s got %s"
                    % (step.index, expected_kind.value, step.kind.value)
                )
            if step.source_state is not state:
                raise ProgramZkpTraceError(
                    "trace state mismatch at index %s" % step.index
                )
            state = next_trace_state(state, step.kind)
            if step.target_state is not state:
                raise ProgramZkpTraceError(
                    "trace target mismatch at index %s" % step.index
                )
        if state is not TraceState.TERMINAL:
            raise ProgramZkpTraceError("trace must terminate in terminal state")
        # COMMIT_RESULT step must bind the same result commitment as the public input.
        commit_step = self.steps[-2]
        if commit_step.kind is not TraceTransitionKind.COMMIT_RESULT:
            raise ProgramZkpTraceError("penultimate step must commit the result")
        if (
            commit_step.binding_commitment
            and commit_step.binding_commitment != self.result_commitment
        ):
            raise ProgramZkpTraceError(
                "committed result does not match public result_commitment"
            )

    @property
    def trace_id(self) -> str:
        return self.content_id

    @property
    def terminal_state(self) -> TraceState:
        return TraceState.TERMINAL

    @property
    def is_complete(self) -> bool:
        return bool(self.steps) and self.steps[-1].target_state is TraceState.TERMINAL

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "steps": [step.to_dict() for step in self.steps],
            "result_commitment": self.result_commitment,
            "public_input_digest": self.public_input_digest,
            "supported_transitions": list(supported_transition_table()),
            "does_not_prove": sorted(TRACE_VALIDITY_DOES_NOT_PROVE),
            "trace_validity_scope": TRACE_VALIDITY_SCOPE_STATEMENT,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpTrace":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("trace payload must be a mapping")
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("supported_transitions", None)
        data.pop("does_not_prove", None)
        data.pop("trace_validity_scope", None)
        claimed_id = data.pop("content_id", None) or data.pop("trace_id", None)
        result = cls(
            steps=tuple(data.get("steps") or ()),
            result_commitment=data.get("result_commitment", ""),
            public_input_digest=data.get("public_input_digest", ""),
        )
        if claimed_id and claimed_id != result.content_id:
            raise ProgramZkpTamperError("forged trace content identity rejected")
        return result


def build_canonical_program_zkp_trace(
    public_inputs: ProgramZkpPublicInputs,
    *,
    binding_commitments: Mapping[str, str] | None = None,
) -> ProgramZkpTrace:
    """Build the unique full supported trace bound to ``public_inputs``."""

    if not isinstance(public_inputs, ProgramZkpPublicInputs):
        raise ProgramAnalysisZkpError("public_inputs must be ProgramZkpPublicInputs")
    bindings = dict(binding_commitments or {})
    default_bindings = {
        TraceTransitionKind.OPEN_FOREST.value: public_inputs.forest_commitment,
        TraceTransitionKind.OPEN_INVENTORY.value: public_inputs.inventory_commitment,
        TraceTransitionKind.OPEN_CONTRACT.value: public_inputs.contract_commitment,
        TraceTransitionKind.OPEN_CALL_SLICE.value: public_inputs.call_slice_commitment,
        TraceTransitionKind.OPEN_ASSUMPTIONS.value: public_inputs.assumptions_commitment,
        TraceTransitionKind.BIND_VERSIONS.value: content_identity(
            {
                "analyzer_version": public_inputs.analyzer_version,
                "resolver_version": public_inputs.resolver_version,
                "translator_version": public_inputs.translator_version,
                "prover_version": public_inputs.prover_version,
            }
        ),
        TraceTransitionKind.COMMIT_RESULT.value: public_inputs.result_commitment,
        TraceTransitionKind.TERMINATE.value: public_inputs.public_input_digest,
    }
    steps: list[ProgramZkpTraceStep] = []
    state = TraceState.INIT
    for index, kind in enumerate(canonical_trace_transition_kinds()):
        target = next_trace_state(state, kind)
        binding = bindings.get(kind.value, default_bindings.get(kind.value, ""))
        steps.append(
            ProgramZkpTraceStep(
                index=index,
                kind=kind,
                source_state=state,
                target_state=target,
                binding_commitment=binding,
            )
        )
        state = target
    return ProgramZkpTrace(
        steps=tuple(steps),
        result_commitment=public_inputs.result_commitment,
        public_input_digest=public_inputs.public_input_digest,
    )


# ---------------------------------------------------------------------------
# Witness policy and private witness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramZkpWitnessPolicy(CanonicalContract):
    """Redaction and admission policy for private program-analysis witnesses.

    Protected openings may include selected source/AST/proof-trace nodes when
    policy requires redaction.  Public artifacts may only carry boolean
    redaction markers, never field names or values.
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_WITNESS_POLICY_SCHEMA

    allow_source_openings: bool = True
    allow_ast_openings: bool = True
    allow_proof_trace_openings: bool = True
    redact_from_public_artifacts: bool = True
    redact_from_logs: bool = True
    redact_from_cache: bool = True
    max_opening_fields: int = 256
    policy_id: str = "program-analysis-zkp-witness-policy@1"

    def __post_init__(self) -> None:
        for name in (
            "allow_source_openings",
            "allow_ast_openings",
            "allow_proof_trace_openings",
            "redact_from_public_artifacts",
            "redact_from_logs",
            "redact_from_cache",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), field_name=name)
            )
        if (
            isinstance(self.max_opening_fields, bool)
            or not isinstance(self.max_opening_fields, int)
            or self.max_opening_fields < 1
            or self.max_opening_fields > 10_000
        ):
            raise ProgramAnalysisZkpError(
                "max_opening_fields must be an integer in [1, 10000]"
            )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        if not self.redact_from_public_artifacts:
            raise ProgramAnalysisZkpError(
                "witness policy must redact private openings from public artifacts"
            )
        if not self.redact_from_logs:
            raise ProgramAnalysisZkpError(
                "witness policy must redact private openings from logs"
            )
        if not self.redact_from_cache:
            raise ProgramAnalysisZkpError(
                "witness policy must redact private openings from cache entries"
            )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "policy_id": self.policy_id,
            "allow_source_openings": self.allow_source_openings,
            "allow_ast_openings": self.allow_ast_openings,
            "allow_proof_trace_openings": self.allow_proof_trace_openings,
            "redact_from_public_artifacts": self.redact_from_public_artifacts,
            "redact_from_logs": self.redact_from_logs,
            "redact_from_cache": self.redact_from_cache,
            "max_opening_fields": self.max_opening_fields,
            "private_field_markers": sorted(_PRIVATE_WITNESS_MARKERS),
            "safe_redaction_keys": sorted(_SAFE_PUBLIC_REDACTION_KEYS),
        }

    def admits_field(self, field_name: str) -> bool:
        name = _text(field_name, field_name="field_name").lower().replace("-", "_")
        if name in {"source_text", "source_opening"} and not self.allow_source_openings:
            return False
        if name in {"ast_node", "ast_opening"} and not self.allow_ast_openings:
            return False
        if (
            name in {"proof_trace", "proof_trace_opening"}
            and not self.allow_proof_trace_openings
        ):
            return False
        return True

    def redacted_public_marker(self) -> Dict[str, bool]:
        return {"private_witness_redacted": True}

    @classmethod
    def default(cls) -> "ProgramZkpWitnessPolicy":
        return cls()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpWitnessPolicy":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("witness policy payload must be a mapping")
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("private_field_markers", None)
        data.pop("safe_redaction_keys", None)
        data.pop("content_id", None)
        return cls(
            allow_source_openings=data.get("allow_source_openings", True),
            allow_ast_openings=data.get("allow_ast_openings", True),
            allow_proof_trace_openings=data.get("allow_proof_trace_openings", True),
            redact_from_public_artifacts=data.get("redact_from_public_artifacts", True),
            redact_from_logs=data.get("redact_from_logs", True),
            redact_from_cache=data.get("redact_from_cache", True),
            max_opening_fields=data.get("max_opening_fields", 256),
            policy_id=data.get("policy_id", "program-analysis-zkp-witness-policy@1"),
        )


class PrivateProgramAnalysisWitness:
    """Non-serializable private openings for program-analysis ZK proving.

    A backend receives values only inside :meth:`use`.  The wrapper has no
    mapping protocol, public value property, JSON method, or pickle
    representation so generic serializers cannot traverse secrets.
    """

    __slots__ = ("__values", "__policy")

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        policy: ProgramZkpWitnessPolicy | None = None,
    ) -> None:
        if not isinstance(values, Mapping):
            raise ProgramAnalysisZkpError("witness values must be a mapping")
        checked_policy = policy if policy is not None else ProgramZkpWitnessPolicy.default()
        if not isinstance(checked_policy, ProgramZkpWitnessPolicy):
            raise ProgramAnalysisZkpError("policy must be ProgramZkpWitnessPolicy")
        if len(values) > checked_policy.max_opening_fields:
            raise ProgramAnalysisZkpError(
                "witness exceeds max_opening_fields=%s"
                % checked_policy.max_opening_fields
            )
        normalized: Dict[str, Any] = {}
        for raw_name, value in values.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ProgramAnalysisZkpError(
                    "witness field names must be non-empty strings"
                )
            name = raw_name.strip()
            if not checked_policy.admits_field(name):
                raise ProgramAnalysisZkpError(
                    "witness field %s is not admitted by policy" % name
                )
            normalized[name] = value
        if not normalized:
            raise ProgramAnalysisZkpError("witness values must not be empty")
        self.__values = dict(normalized)
        self.__policy = checked_policy

    @property
    def policy(self) -> ProgramZkpWitnessPolicy:
        return self.__policy

    def __repr__(self) -> str:
        return "<PrivateProgramAnalysisWitness redacted>"

    __str__ = __repr__

    def __copy__(self) -> "PrivateProgramAnalysisWitness":
        raise ProgramZkpWitnessDisclosureError("private witness cannot be copied")

    def __deepcopy__(self, memo: Any) -> "PrivateProgramAnalysisWitness":
        del memo
        raise ProgramZkpWitnessDisclosureError("private witness cannot be copied")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise ProgramZkpWitnessDisclosureError(
            "private witness cannot be serialized or cached"
        )

    def __getstate__(self) -> Any:
        raise ProgramZkpWitnessDisclosureError(
            "private witness cannot be serialized or cached"
        )

    def to_dict(self) -> Dict[str, Any]:
        raise ProgramZkpWitnessDisclosureError(
            "private witness has no public dictionary representation"
        )

    def use(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        """Invoke a local prover callback with a read-only witness view."""

        if not callable(consumer):
            raise ProgramAnalysisZkpError("witness consumer must be callable")
        return consumer(MappingProxyType(self.__values))

    def redacted(self) -> Dict[str, bool]:
        return self.__policy.redacted_public_marker()


def reject_private_witness_from_public_payload(value: Any) -> None:
    """Reject private witness material from public receipts and statements."""

    if isinstance(value, PrivateProgramAnalysisWitness):
        raise ProgramZkpWitnessDisclosureError(
            "private witness cannot enter a public program-analysis artifact"
        )
    if isinstance(value, ProgramZkpProvingRequest):
        raise ProgramZkpWitnessDisclosureError(
            "proving requests cannot enter public program-analysis artifacts"
        )
    if _public_payload_has_private_witness(value):
        raise ProgramZkpWitnessDisclosureError(
            "private witness markers are rejected from public program-analysis artifacts"
        )


def _public_payload_has_private_witness(value: Any) -> bool:
    if isinstance(value, PrivateProgramAnalysisWitness):
        return True
    if isinstance(value, ProgramZkpProvingRequest):
        return True
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in _SAFE_PUBLIC_REDACTION_KEYS:
                if isinstance(item, bool):
                    continue
                return True
            if any(
                key == marker or key.endswith("_" + marker) or marker in key
                for marker in _PRIVATE_WITNESS_MARKERS
            ):
                if key not in _SAFE_PUBLIC_REDACTION_KEYS:
                    return True
            if _public_payload_has_private_witness(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_public_payload_has_private_witness(item) for item in value)
    return False


def public_program_zkp_artifact(value: Any) -> Any:
    """Project a value into a public form, rejecting private witnesses."""

    if isinstance(value, PrivateProgramAnalysisWitness):
        raise ProgramZkpWitnessDisclosureError(
            "private witness cannot enter a public program-analysis artifact"
        )
    if isinstance(value, ProgramZkpProvingRequest):
        return value.to_public_artifact()
    if isinstance(value, CanonicalContract):
        public = value.to_dict()
        reject_private_witness_from_public_payload(public)
        return public
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.strip().lower().replace("-", "_")
            if lowered in _SAFE_PUBLIC_REDACTION_KEYS and isinstance(item, bool):
                result[key_text] = item
                continue
            if any(
                lowered == marker or marker in lowered
                for marker in _PRIVATE_WITNESS_MARKERS
            ):
                if lowered not in _SAFE_PUBLIC_REDACTION_KEYS:
                    raise ProgramZkpWitnessDisclosureError(
                        "private witness markers are rejected from public artifacts"
                    )
            result[key_text] = public_program_zkp_artifact(item)
        reject_private_witness_from_public_payload(result)
        return result
    if isinstance(value, (list, tuple)):
        return [public_program_zkp_artifact(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    # Fail closed for unknown objects that might wrap secrets.
    if hasattr(value, "to_public_artifact") and callable(value.to_public_artifact):
        public = value.to_public_artifact()
        reject_private_witness_from_public_payload(public)
        return public
    raise ProgramZkpWitnessDisclosureError(
        "value of type %s is not a public program-analysis artifact"
        % type(value).__name__
    )


def public_artifact_contains(artifact: Any, needle: str) -> bool:
    """Return whether a public artifact serialization contains ``needle``."""

    text = needle if isinstance(needle, str) else str(needle)
    try:
        encoded = json.dumps(
            _canonical_value(artifact) if not isinstance(artifact, (str, bytes)) else artifact,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError, ContractValidationError):
        encoded = repr(artifact)
    return text in encoded


# ---------------------------------------------------------------------------
# Statement, proving request, shadow envelope, verification receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramZkpStatement(CanonicalContract):
    """Public statement proven by a program-analysis ZK trace attestation.

    The statement binds public inputs, the supported-trace identity, circuit
    and key identities, and the explicit non-claim scope.  It never carries
    private openings.
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_STATEMENT_SCHEMA

    public_inputs: ProgramZkpPublicInputs
    trace_id: str
    claim_level: ClaimLevel = ClaimLevel.ZK_TRACE_ATTESTED
    backend_mode: ProgramZkpBackendMode = ProgramZkpBackendMode.SHADOW
    semantic_proof: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.public_inputs, ProgramZkpPublicInputs):
            if isinstance(self.public_inputs, Mapping):
                object.__setattr__(
                    self,
                    "public_inputs",
                    ProgramZkpPublicInputs.from_dict(self.public_inputs),
                )
            else:
                raise ProgramAnalysisZkpError(
                    "public_inputs must be ProgramZkpPublicInputs"
                )
        object.__setattr__(
            self, "trace_id", _text(self.trace_id, field_name="trace_id")
        )
        object.__setattr__(
            self,
            "claim_level",
            _enum(self.claim_level, ClaimLevel, field_name="claim_level"),
        )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(self.backend_mode, ProgramZkpBackendMode, field_name="backend_mode"),
        )
        object.__setattr__(
            self,
            "semantic_proof",
            _boolean(self.semantic_proof, field_name="semantic_proof"),
        )
        if self.claim_level is not ClaimLevel.ZK_TRACE_ATTESTED:
            raise ProgramZkpClaimPromotionError(
                "program-analysis ZK statements must use claim_level zk_trace_attested"
            )
        if self.semantic_proof:
            raise SemanticAuthorityError(
                "a program-analysis ZK statement cannot be presented as semantic proof; "
                + TRACE_VALIDITY_SCOPE_STATEMENT
            )

    @property
    def statement_id(self) -> str:
        return self.content_id

    @property
    def public_input_digest(self) -> str:
        return self.public_inputs.public_input_digest

    @property
    def authoritative(self) -> bool:
        # Generation and serialization never assert authority.
        return False

    @property
    def trust(self) -> ProgramZkpTrust:
        return ProgramZkpTrust.NON_AUTHORITATIVE

    @property
    def does_not_prove(self) -> tuple[str, ...]:
        return tuple(sorted(TRACE_VALIDITY_DOES_NOT_PROVE))

    @property
    def required_authority_kind(self) -> AuthorityKind:
        return AuthorityKind.ZK_VERIFIER

    def _payload(self) -> Dict[str, Any]:
        # statement_id is derived; included only via to_public_artifact/to_record
        # so from_dict can detect identity forgery without recursive hashing.
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "public_inputs": self.public_inputs.to_dict(),
            "public_input_digest": self.public_input_digest,
            "trace_id": self.trace_id,
            "claim_level": self.claim_level.value,
            "backend_mode": self.backend_mode.value,
            "semantic_proof": False,
            "authoritative": False,
            "trust": ProgramZkpTrust.NON_AUTHORITATIVE.value,
            "authority_kind": AuthorityKind.ZK_VERIFIER.value,
            "does_not_prove": list(self.does_not_prove),
            "trace_validity_scope": TRACE_VALIDITY_SCOPE_STATEMENT,
            "circuit_id": self.public_inputs.circuit_id,
            "proving_key_id": self.public_inputs.proving_key_id,
            "verifying_key_id": self.public_inputs.verifying_key_id,
            "ceremony_id": self.public_inputs.ceremony_id,
            "public_input_codec_id": self.public_inputs.public_input_codec_id,
            "public_input_codec_version": self.public_inputs.public_input_codec_version,
        }

    def to_public_artifact(self) -> Dict[str, Any]:
        public = {
            **self.to_dict(),
            "statement_id": self.statement_id,
            "content_id": self.content_id,
        }
        reject_private_witness_from_public_payload(public)
        return public

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpStatement":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("statement payload must be a mapping")
        data = dict(payload)
        reject_private_witness_from_public_payload(data)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("does_not_prove", None)
        data.pop("trace_validity_scope", None)
        data.pop("authority_kind", None)
        data.pop("trust", None)
        data.pop("circuit_id", None)
        data.pop("proving_key_id", None)
        data.pop("verifying_key_id", None)
        data.pop("ceremony_id", None)
        data.pop("public_input_codec_id", None)
        data.pop("public_input_codec_version", None)
        claimed_id = data.pop("content_id", None) or data.pop("statement_id", None)
        claimed_digest = data.pop("public_input_digest", None)
        claimed_auth = data.pop("authoritative", None)
        if claimed_auth is True:
            raise ProgramZkpTamperError(
                "statement cannot assert authority; independent verification is required"
            )
        public_inputs = ProgramZkpPublicInputs.from_dict(data.get("public_inputs") or {})
        result = cls(
            public_inputs=public_inputs,
            trace_id=data.get("trace_id", ""),
            claim_level=data.get("claim_level", ClaimLevel.ZK_TRACE_ATTESTED),
            backend_mode=data.get("backend_mode", ProgramZkpBackendMode.SHADOW),
            semantic_proof=data.get("semantic_proof", False),
        )
        if claimed_digest and claimed_digest != result.public_input_digest:
            raise ProgramZkpTamperError("forged statement public-input digest rejected")
        if claimed_id and claimed_id != result.content_id:
            raise ProgramZkpTamperError("forged statement identity rejected")
        return result


@dataclass(frozen=True, repr=False)
class ProgramZkpProvingRequest:
    """Ephemeral proving request; only its public statement is serializable."""

    statement: ProgramZkpStatement
    trace: ProgramZkpTrace
    _witness: PrivateProgramAnalysisWitness = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.statement, ProgramZkpStatement):
            raise ProgramAnalysisZkpError("statement must be ProgramZkpStatement")
        if not isinstance(self.trace, ProgramZkpTrace):
            raise ProgramAnalysisZkpError("trace must be ProgramZkpTrace")
        if not isinstance(self._witness, PrivateProgramAnalysisWitness):
            raise ProgramAnalysisZkpError(
                "_witness must be PrivateProgramAnalysisWitness"
            )
        if self.trace.trace_id != self.statement.trace_id:
            raise ProgramZkpTamperError("trace identity does not match statement")
        if self.trace.public_input_digest != self.statement.public_input_digest:
            raise ProgramZkpTamperError(
                "trace public-input digest does not match statement"
            )
        if self.trace.result_commitment != self.statement.public_inputs.result_commitment:
            raise ProgramZkpTamperError(
                "trace result commitment does not match public inputs"
            )

    def __repr__(self) -> str:
        return (
            "ProgramZkpProvingRequest(statement_id=%r, witness=<redacted>)"
            % self.statement.statement_id
        )

    __str__ = __repr__

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise ProgramZkpWitnessDisclosureError(
            "program-analysis ZK proving requests cannot be serialized or cached"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement.to_public_artifact(),
            "trace_id": self.trace.trace_id,
            "private_witness_redacted": True,
        }

    to_public_artifact = to_dict
    to_context_capsule = to_dict
    to_log_record = to_dict

    def to_cache_record(self) -> Dict[str, Any]:
        raise ProgramZkpWitnessDisclosureError(
            "proving requests containing a witness cannot be cached"
        )

    def use_witness(self, consumer: Callable[[Mapping[str, Any]], T]) -> T:
        return self._witness.use(consumer)


@dataclass(frozen=True)
class ProgramZkpShadowEnvelope(CanonicalContract):
    """Non-authoritative shadow-mode envelope for serialization and workflow tests.

    Shadow and simulated envelopes never grant authority.  Production
    cryptographic verification requires a production-eligible capability
    report and independent verifier path (VFS-024).
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_SHADOW_ENVELOPE_SCHEMA

    statement: ProgramZkpStatement
    backend_mode: ProgramZkpBackendMode
    proof_artifact_id: str
    proof_digest: str
    prover_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.statement, ProgramZkpStatement):
            if isinstance(self.statement, Mapping):
                object.__setattr__(
                    self, "statement", ProgramZkpStatement.from_dict(self.statement)
                )
            else:
                raise ProgramAnalysisZkpError("statement must be ProgramZkpStatement")
        object.__setattr__(
            self,
            "backend_mode",
            _enum(self.backend_mode, ProgramZkpBackendMode, field_name="backend_mode"),
        )
        object.__setattr__(
            self,
            "proof_artifact_id",
            _text(self.proof_artifact_id, field_name="proof_artifact_id"),
        )
        object.__setattr__(
            self,
            "proof_digest",
            _text(self.proof_digest, field_name="proof_digest"),
        )
        object.__setattr__(
            self, "prover_id", _text(self.prover_id, field_name="prover_id", required=False)
        )
        if self.backend_mode is ProgramZkpBackendMode.CRYPTOGRAPHIC:
            # Cryptographic mode alone never grants authority; VFS-024 capability
            # conformance and independent verification are required first.
            pass

    @property
    def envelope_id(self) -> str:
        return self.content_id

    @property
    def simulated(self) -> bool:
        return self.backend_mode in {
            ProgramZkpBackendMode.SIMULATED,
            ProgramZkpBackendMode.SHADOW,
        }

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def trust(self) -> ProgramZkpTrust:
        return ProgramZkpTrust.NON_AUTHORITATIVE

    @property
    def non_authoritative_reason(self) -> str:
        if self.backend_mode is ProgramZkpBackendMode.SIMULATED:
            return "simulated_zkp_is_non_authoritative"
        if self.backend_mode is ProgramZkpBackendMode.SHADOW:
            return "shadow_zkp_requires_independent_verification"
        return "independent_verification_required"

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "statement": self.statement.to_dict(),
            "statement_id": self.statement.statement_id,
            "public_input_digest": self.statement.public_input_digest,
            "backend_mode": self.backend_mode.value,
            "proof_artifact_id": self.proof_artifact_id,
            "proof_digest": self.proof_digest,
            "prover_id": self.prover_id,
            "authoritative": False,
            "trust": ProgramZkpTrust.NON_AUTHORITATIVE.value,
            "non_authoritative_reason": self.non_authoritative_reason,
            "does_not_prove": sorted(TRACE_VALIDITY_DOES_NOT_PROVE),
        }

    def to_public_artifact(self) -> Dict[str, Any]:
        public = self.to_dict()
        reject_private_witness_from_public_payload(public)
        return public

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpShadowEnvelope":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("envelope payload must be a mapping")
        data = dict(payload)
        reject_private_witness_from_public_payload(data)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("statement_id", None)
        data.pop("public_input_digest", None)
        data.pop("trust", None)
        data.pop("non_authoritative_reason", None)
        data.pop("does_not_prove", None)
        claimed_id = data.pop("content_id", None) or data.pop("envelope_id", None)
        claimed_auth = data.pop("authoritative", None)
        if claimed_auth is True:
            raise ProgramZkpTamperError(
                "envelope cannot assert authority; independent verification is required"
            )
        result = cls(
            statement=ProgramZkpStatement.from_dict(data.get("statement") or {}),
            backend_mode=data.get("backend_mode", ProgramZkpBackendMode.SHADOW),
            proof_artifact_id=data.get("proof_artifact_id", ""),
            proof_digest=data.get("proof_digest", ""),
            prover_id=data.get("prover_id", ""),
        )
        if claimed_id and claimed_id != result.content_id:
            raise ProgramZkpTamperError("forged envelope identity rejected")
        return result


@dataclass(frozen=True)
class ProgramZkpVerificationReceipt(CanonicalContract):
    """Independent (or shadow) verification receipt bound to exact public inputs.

    Replay against drifted public inputs, keys, circuit identity, or codec
    version fails closed.  A verified shadow/simulated receipt remains
    non-authoritative for semantic claims.  Cryptographic verification grants
    authority only when bound to a production-eligible capability epoch
    (VFS-024).
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_RECEIPT_SCHEMA

    statement: ProgramZkpStatement
    verdict: ProgramZkpVerdict
    verifier_id: str
    verifying_key_id: str
    circuit_id: str
    public_input_digest: str
    ceremony_id: str
    public_input_codec_version: str
    backend_mode: ProgramZkpBackendMode = ProgramZkpBackendMode.SHADOW
    capability_epoch: str = ""
    capability_production_eligible: bool = False
    proof_schema_id: str = PROGRAM_ZKP_PROOF_SCHEMA_ID
    independent_verifier: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.statement, ProgramZkpStatement):
            if isinstance(self.statement, Mapping):
                object.__setattr__(
                    self, "statement", ProgramZkpStatement.from_dict(self.statement)
                )
            else:
                raise ProgramAnalysisZkpError("statement must be ProgramZkpStatement")
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ProgramZkpVerdict, field_name="verdict")
        )
        for name in (
            "verifier_id",
            "verifying_key_id",
            "circuit_id",
            "public_input_digest",
            "ceremony_id",
            "public_input_codec_version",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "backend_mode",
            _enum(self.backend_mode, ProgramZkpBackendMode, field_name="backend_mode"),
        )
        object.__setattr__(
            self,
            "capability_epoch",
            _text(self.capability_epoch, field_name="capability_epoch", required=False),
        )
        object.__setattr__(
            self,
            "capability_production_eligible",
            _boolean(
                self.capability_production_eligible,
                field_name="capability_production_eligible",
            ),
        )
        object.__setattr__(
            self,
            "proof_schema_id",
            _text(self.proof_schema_id, field_name="proof_schema_id"),
        )
        object.__setattr__(
            self,
            "independent_verifier",
            _boolean(self.independent_verifier, field_name="independent_verifier"),
        )
        if self.proof_schema_id != PROGRAM_ZKP_PROOF_SCHEMA_ID:
            raise ProgramZkpVersionError(
                "proof_schema_id must be %s" % PROGRAM_ZKP_PROOF_SCHEMA_ID
            )
        # Bind receipt pins to the statement's public inputs (tamper resistance).
        pins = self.statement.public_inputs
        if self.verifying_key_id != pins.verifying_key_id:
            raise ProgramZkpTamperError(
                "receipt verifying_key_id does not match statement"
            )
        if self.circuit_id != pins.circuit_id:
            raise ProgramZkpTamperError("receipt circuit_id does not match statement")
        if self.ceremony_id != pins.ceremony_id:
            raise ProgramZkpTamperError("receipt ceremony_id does not match statement")
        if self.public_input_digest != self.statement.public_input_digest:
            raise ProgramZkpTamperError(
                "receipt public_input_digest does not match statement"
            )
        if self.public_input_codec_version != pins.public_input_codec_version:
            raise ProgramZkpVersionError(
                "receipt public-input codec version does not match statement"
            )
        if self.capability_production_eligible and not self.capability_epoch:
            raise ProgramZkpCapabilityError(
                "production-eligible receipts require a capability_epoch binding"
            )
        if self.capability_production_eligible and not self.independent_verifier:
            raise ProgramZkpCapabilityError(
                "production authority requires independent verification"
            )
        if (
            self.capability_production_eligible
            and self.backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC
        ):
            raise ProgramZkpAuthorityError(
                "only cryptographic backends may bind production-eligible capability"
            )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def verified(self) -> bool:
        return self.verdict is ProgramZkpVerdict.VERIFIED

    @property
    def authoritative(self) -> bool:
        # Shadow/simulated verification never grants production authority.
        # Cryptographic + verified alone is insufficient: VFS-024 requires a
        # production-eligible capability epoch and independent verification.
        if self.backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC:
            return False
        if self.verdict is not ProgramZkpVerdict.VERIFIED:
            return False
        if not self.independent_verifier:
            return False
        if not self.capability_production_eligible or not self.capability_epoch:
            return False
        return True

    @property
    def trust(self) -> ProgramZkpTrust:
        if self.authoritative:
            return ProgramZkpTrust.AUTHORITATIVE
        return ProgramZkpTrust.NON_AUTHORITATIVE

    @property
    def claim_level(self) -> ClaimLevel:
        return ClaimLevel.ZK_TRACE_ATTESTED

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "statement": self.statement.to_dict(),
            "statement_id": self.statement.statement_id,
            "verdict": self.verdict.value,
            "verifier_id": self.verifier_id,
            "verifying_key_id": self.verifying_key_id,
            "circuit_id": self.circuit_id,
            "public_input_digest": self.public_input_digest,
            "ceremony_id": self.ceremony_id,
            "public_input_codec_version": self.public_input_codec_version,
            "backend_mode": self.backend_mode.value,
            "capability_epoch": self.capability_epoch,
            "capability_production_eligible": self.capability_production_eligible,
            "proof_schema_id": self.proof_schema_id,
            "independent_verifier": self.independent_verifier,
            "authoritative": self.authoritative,
            "trust": self.trust.value,
            "claim_level": self.claim_level.value,
            "semantic_proof": False,
            "does_not_prove": sorted(TRACE_VALIDITY_DOES_NOT_PROVE),
            "trace_validity_scope": TRACE_VALIDITY_SCOPE_STATEMENT,
        }

    def to_public_artifact(self) -> Dict[str, Any]:
        public = self.to_dict()
        reject_private_witness_from_public_payload(public)
        return public

    def require_replay(
        self,
        *,
        public_inputs: ProgramZkpPublicInputs | Mapping[str, str],
        verifying_key_id: str,
        circuit_id: str,
        ceremony_id: str,
        public_input_codec_version: str = PUBLIC_INPUT_CODEC_VERSION,
        capability_epoch: str | None = None,
    ) -> None:
        """Fail closed when replay inputs drift from the receipt binding."""

        if isinstance(public_inputs, ProgramZkpPublicInputs):
            digest = public_inputs.public_input_digest
            pins = public_inputs
        elif isinstance(public_inputs, Mapping):
            pins = ProgramZkpPublicInputs.from_dict(public_inputs)
            digest = pins.public_input_digest
        else:
            raise ProgramAnalysisZkpError("public_inputs must be ProgramZkpPublicInputs")
        if digest != self.public_input_digest:
            raise ProgramZkpReplayError(
                "replay public-input digest does not match verification receipt"
            )
        if _text(verifying_key_id, field_name="verifying_key_id") != self.verifying_key_id:
            raise ProgramZkpReplayError(
                "replay verifying_key_id does not match verification receipt"
            )
        if _text(circuit_id, field_name="circuit_id") != self.circuit_id:
            raise ProgramZkpReplayError(
                "replay circuit_id does not match verification receipt"
            )
        if _text(ceremony_id, field_name="ceremony_id") != self.ceremony_id:
            raise ProgramZkpReplayError(
                "replay ceremony_id does not match verification receipt"
            )
        if (
            _text(public_input_codec_version, field_name="public_input_codec_version")
            != self.public_input_codec_version
        ):
            raise ProgramZkpVersionError(
                "replay public-input codec version does not match verification receipt"
            )
        if pins.public_input_digest != self.statement.public_input_digest:
            raise ProgramZkpReplayError(
                "replay public inputs do not match the bound statement"
            )
        if capability_epoch is not None:
            expected = self.capability_epoch
            actual = _text(
                capability_epoch, field_name="capability_epoch", required=False
            )
            if expected and actual != expected:
                raise ProgramZkpReplayError(
                    "replay capability_epoch does not match verification receipt"
                )

    def require_capability_epoch(self, capability_epoch: str) -> None:
        """Fail closed when the bound capability epoch has been lost or replaced."""

        expected = _text(capability_epoch, field_name="capability_epoch")
        if not self.capability_epoch:
            raise ProgramZkpCapabilityError(
                "receipt has no capability epoch; cannot validate authority continuity"
            )
        if self.capability_epoch != expected:
            raise ProgramZkpCapabilityError(
                "capability loss invalidates prior authoritative projection"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpVerificationReceipt":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("receipt payload must be a mapping")
        data = dict(payload)
        reject_private_witness_from_public_payload(data)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("statement_id", None)
        data.pop("trust", None)
        data.pop("claim_level", None)
        data.pop("semantic_proof", None)
        data.pop("does_not_prove", None)
        data.pop("trace_validity_scope", None)
        claimed_id = data.pop("content_id", None) or data.pop("receipt_id", None)
        claimed_auth = data.get("authoritative")
        result = cls(
            statement=ProgramZkpStatement.from_dict(data.get("statement") or {}),
            verdict=data.get("verdict", ProgramZkpVerdict.REJECTED),
            verifier_id=data.get("verifier_id", ""),
            verifying_key_id=data.get("verifying_key_id", ""),
            circuit_id=data.get("circuit_id", ""),
            public_input_digest=data.get("public_input_digest", ""),
            ceremony_id=data.get("ceremony_id", ""),
            public_input_codec_version=data.get(
                "public_input_codec_version", PUBLIC_INPUT_CODEC_VERSION
            ),
            backend_mode=data.get("backend_mode", ProgramZkpBackendMode.SHADOW),
            capability_epoch=data.get("capability_epoch", ""),
            capability_production_eligible=data.get(
                "capability_production_eligible", False
            ),
            proof_schema_id=data.get("proof_schema_id", PROGRAM_ZKP_PROOF_SCHEMA_ID),
            independent_verifier=data.get("independent_verifier", False),
        )
        if claimed_auth is True and not result.authoritative:
            raise ProgramZkpTamperError(
                "receipt cannot assert authority for non-cryptographic or unverified paths"
            )
        if claimed_auth is False and result.authoritative:
            # Re-derived authority may differ from serialized flag only when True
            # was forged; False is always safe.
            pass
        if claimed_id and claimed_id != result.content_id:
            raise ProgramZkpTamperError("forged verification receipt identity rejected")
        return result


def prepare_program_analysis_zkp(
    public_inputs: ProgramZkpPublicInputs,
    *,
    witness: PrivateProgramAnalysisWitness | Mapping[str, Any],
    backend_mode: ProgramZkpBackendMode | str = ProgramZkpBackendMode.SHADOW,
    binding_commitments: Mapping[str, str] | None = None,
) -> ProgramZkpProvingRequest:
    """Prepare a proving request with a canonical trace and private witness."""

    if not isinstance(public_inputs, ProgramZkpPublicInputs):
        raise ProgramAnalysisZkpError("public_inputs must be ProgramZkpPublicInputs")
    private = (
        witness
        if isinstance(witness, PrivateProgramAnalysisWitness)
        else PrivateProgramAnalysisWitness(witness)
    )
    trace = build_canonical_program_zkp_trace(
        public_inputs, binding_commitments=binding_commitments
    )
    statement = ProgramZkpStatement(
        public_inputs=public_inputs,
        trace_id=trace.trace_id,
        claim_level=ClaimLevel.ZK_TRACE_ATTESTED,
        backend_mode=_enum(backend_mode, ProgramZkpBackendMode, field_name="backend_mode"),
        semantic_proof=False,
    )
    return ProgramZkpProvingRequest(
        statement=statement,
        trace=trace,
        _witness=private,
    )


def create_program_zkp_shadow_envelope(
    request: ProgramZkpProvingRequest,
    *,
    proof_artifact_id: str,
    proof_digest: str,
    prover_id: str = "",
    backend_mode: ProgramZkpBackendMode | str | None = None,
) -> ProgramZkpShadowEnvelope:
    """Create a non-authoritative shadow envelope from a proving request."""

    if not isinstance(request, ProgramZkpProvingRequest):
        raise ProgramAnalysisZkpError("request must be ProgramZkpProvingRequest")
    mode = (
        request.statement.backend_mode
        if backend_mode is None
        else _enum(backend_mode, ProgramZkpBackendMode, field_name="backend_mode")
    )
    return ProgramZkpShadowEnvelope(
        statement=request.statement,
        backend_mode=mode,
        proof_artifact_id=proof_artifact_id,
        proof_digest=proof_digest,
        prover_id=prover_id,
    )


def record_program_zkp_verification(
    envelope: ProgramZkpShadowEnvelope,
    *,
    verdict: ProgramZkpVerdict | str,
    verifier_id: str,
    capability_epoch: str = "",
    capability_production_eligible: bool = False,
    independent_verifier: bool = False,
) -> ProgramZkpVerificationReceipt:
    """Record an independent (or shadow) verification bound to envelope pins.

    Production authority requires ``capability_production_eligible=True``, a
    non-empty ``capability_epoch``, ``independent_verifier=True``, cryptographic
    backend mode, and a verified verdict.  Shadow/simulated paths remain
    non-authoritative even when the verdict is verified.
    """

    if not isinstance(envelope, ProgramZkpShadowEnvelope):
        raise ProgramAnalysisZkpError("envelope must be ProgramZkpShadowEnvelope")
    pins = envelope.statement.public_inputs
    return ProgramZkpVerificationReceipt(
        statement=envelope.statement,
        verdict=_enum(verdict, ProgramZkpVerdict, field_name="verdict"),
        verifier_id=verifier_id,
        verifying_key_id=pins.verifying_key_id,
        circuit_id=pins.circuit_id,
        public_input_digest=envelope.statement.public_input_digest,
        ceremony_id=pins.ceremony_id,
        public_input_codec_version=pins.public_input_codec_version,
        backend_mode=envelope.backend_mode,
        capability_epoch=capability_epoch,
        capability_production_eligible=capability_production_eligible,
        independent_verifier=independent_verifier,
    )


def assert_trace_non_claims(receipt_or_statement: Any) -> None:
    """Assert the artifact still carries the normative non-claim set."""

    if isinstance(
        receipt_or_statement,
        (
            ProgramZkpPublicInputs,
            ProgramZkpStatement,
            ProgramZkpVerificationReceipt,
            ProgramZkpTrace,
            ProgramZkpShadowEnvelope,
            ProgramZkpCapabilityConformanceReport,
        ),
    ):
        payload = receipt_or_statement.to_dict()
    elif isinstance(receipt_or_statement, Mapping):
        payload = dict(receipt_or_statement)
    else:
        raise ProgramAnalysisZkpError("unsupported artifact for non-claim assertion")
    claimed = payload.get("does_not_prove")
    if claimed is None:
        raise ProgramAnalysisZkpError("artifact is missing does_not_prove non-claims")
    if set(claimed) != set(TRACE_VALIDITY_DOES_NOT_PROVE):
        raise ProgramAnalysisZkpError(
            "artifact non-claim set drifted from TRACE_VALIDITY_DOES_NOT_PROVE"
        )
    scope = payload.get("trace_validity_scope", "")
    if TRACE_VALIDITY_SCOPE_STATEMENT not in str(scope):
        # Capability reports carry the scope under a dedicated key; tolerate
        # either embedding form so callers may assert non-claims uniformly.
        if "trace_validity_scope" not in payload and isinstance(
            receipt_or_statement, ProgramZkpCapabilityConformanceReport
        ):
            return
        raise ProgramAnalysisZkpError("artifact is missing trace validity scope statement")


def claim_level_for_verified_trace() -> ClaimLevel:
    """Return the only claim level a verified program-analysis ZK trace may use."""

    return ClaimLevel.ZK_TRACE_ATTESTED


def verdict_for_trace_attestation() -> ClaimVerdict:
    """Return the program-assurance verdict class used with ZK trace claims."""

    return ClaimVerdict.SATISFIED


def inconclusive_state_for_shadow() -> InconclusiveState:
    """Shadow/simulated ZK paths remain non-conclusive for semantic authority."""

    return InconclusiveState.NONE


# ---------------------------------------------------------------------------
# VFS-024: production capability, setup, ceremony, and verifier conformance
# ---------------------------------------------------------------------------


class ProgramZkpCapabilityDimension(str, Enum):
    """Probe dimensions that must pass before production ZK authority attaches."""

    EXECUTABLE_ARCHITECTURE = "executable_architecture"
    BACKEND = "backend"
    CIRCUIT_VERSION = "circuit_version"
    SETUP_ARTIFACTS = "setup_artifacts"
    CEREMONY = "ceremony"
    PROVING_KEY = "proving_key"
    VERIFYING_KEY = "verifying_key"
    PUBLIC_INPUT_CODEC = "public_input_codec"
    PROOF_SCHEMA = "proof_schema"
    INDEPENDENT_VERIFIER = "independent_verifier"
    BOUNDS = "bounds"
    CANCELLATION = "cancellation"


REQUIRED_CAPABILITY_DIMENSIONS: Final[tuple[ProgramZkpCapabilityDimension, ...]] = (
    tuple(ProgramZkpCapabilityDimension)
)


class ProgramZkpCapabilityStatus(str, Enum):
    """Status of one production capability probe dimension."""

    VERIFIED = "verified"
    AVAILABLE = "available"
    CONFIGURED = "configured"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    SIMULATED = "simulated"
    SHADOW = "shadow"
    FAILED = "failed"
    STALE = "stale"
    REJECTED = "rejected"


class ProgramZkpRolloutMode(str, Enum):
    """Rollout posture derived from capability probes (fail-closed to shadow)."""

    DISABLED = "disabled"
    SHADOW = "shadow"
    CANARY = "canary"
    ENFORCEMENT = "enforcement"


class ProgramZkpFieldEncodingKind(str, Enum):
    """Field encoding classes admitted (or rejected) for production circuits."""

    BN254_SHA256 = "bn254_sha256"
    PLACEHOLDER = "placeholder"
    NONZERO_ONLY_V1 = "nonzero_only_v1"
    UNKNOWN = "unknown"


class ProgramZkpCircuitFamily(str, Enum):
    """Circuit families; only program_contract_trace is production-compatible."""

    PROGRAM_CONTRACT_TRACE = "program_contract_trace"
    TDFOL_ONLY = "tdfol_only"
    NONZERO_ONLY_V1 = "nonzero_only_v1"
    UNKNOWN = "unknown"


class ProgramZkpAuthorityDenialReason(str, Enum):
    """Normative fail-closed reasons that block production ZK authority."""

    SIMULATED_DEFAULT = "simulated_default"
    KNOWLEDGE_GRAPH_FAIL_OPEN = "knowledge_graph_fail_open"
    PLACEHOLDER_FIELD_ENCODING = "placeholder_field_encoding"
    V1_NONZERO_ONLY_CIRCUIT = "v1_nonzero_only_circuit"
    INCOMPATIBLE_TDFOL_ONLY_CIRCUIT = "incompatible_tdfol_only_circuit"
    UNVERSIONED_ARTIFACT = "unversioned_artifact"
    MISSING_ARTIFACT = "missing_artifact"
    STALE_CAPABILITY = "stale_capability"
    BACKEND_NOT_CRYPTOGRAPHIC = "backend_not_cryptographic"
    CEREMONY_INELIGIBLE = "ceremony_ineligible"
    SETUP_INELIGIBLE = "setup_ineligible"
    INDEPENDENT_VERIFIER_ABSENT = "independent_verifier_absent"
    PROOF_SCHEMA_MISMATCH = "proof_schema_mismatch"
    CODEC_INCOMPATIBLE = "codec_incompatible"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    CANCELLED = "cancelled"
    CAPABILITY_LOSS = "capability_loss"
    CORRUPTED_PROOF = "corrupted_proof"
    CORRUPTED_KEY = "corrupted_key"
    CORRUPTED_INPUT = "corrupted_input"
    SEMANTIC_CLAIM_PROMOTION = "semantic_claim_promotion"
    SHADOW_ONLY_ROLLOUT = "shadow_only_rollout"
    PROBE_FAILED = "probe_failed"


_PRODUCTION_CAPABILITY_STATUSES: Final[frozenset[ProgramZkpCapabilityStatus]] = (
    frozenset(
        {
            ProgramZkpCapabilityStatus.VERIFIED,
            ProgramZkpCapabilityStatus.AVAILABLE,
        }
    )
)

_FAIL_CLOSED_CAPABILITY_STATUSES: Final[frozenset[ProgramZkpCapabilityStatus]] = (
    frozenset(
        {
            ProgramZkpCapabilityStatus.SIMULATED,
            ProgramZkpCapabilityStatus.SHADOW,
            ProgramZkpCapabilityStatus.FAILED,
            ProgramZkpCapabilityStatus.STALE,
            ProgramZkpCapabilityStatus.REJECTED,
            ProgramZkpCapabilityStatus.UNAVAILABLE,
            ProgramZkpCapabilityStatus.DEGRADED,
            ProgramZkpCapabilityStatus.CONFIGURED,
        }
    )
)


def field_element_from_text(value: str) -> int:
    """Map UTF-8 text to the BN254 scalar field via SHA-256 (mod P).

    This is the production field encoding for program_contract_trace public
    inputs.  Placeholder encodings must not be substituted.
    """

    text = _text(value, field_name="field_text")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % BN254_SCALAR_FIELD_MODULUS


def encode_public_input_field_vector(
    public_inputs: ProgramZkpPublicInputs | Mapping[str, str],
    *,
    field_encoding: ProgramZkpFieldEncodingKind | str = ProgramZkpFieldEncodingKind.BN254_SHA256,
) -> tuple[int, ...]:
    """Encode public inputs as BN254 field elements in codec order.

    Placeholder and nonzero-only encodings fail closed for production use.
    """

    encoding = _enum(
        field_encoding, ProgramZkpFieldEncodingKind, field_name="field_encoding"
    )
    if encoding is ProgramZkpFieldEncodingKind.PLACEHOLDER:
        raise ProgramZkpAuthorityError(
            "placeholder field encoding cannot encode production public inputs"
        )
    if encoding is ProgramZkpFieldEncodingKind.NONZERO_ONLY_V1:
        raise ProgramZkpAuthorityError(
            "v1 nonzero-only field encoding is insufficient for production authority"
        )
    if encoding is not ProgramZkpFieldEncodingKind.BN254_SHA256:
        raise ProgramZkpVersionError(
            "unsupported field encoding %s" % encoding.value
        )
    if isinstance(public_inputs, ProgramZkpPublicInputs):
        vector = public_inputs.public_input_vector
    elif isinstance(public_inputs, Mapping):
        vector = encode_public_input_vector(public_inputs)
    else:
        raise ProgramAnalysisZkpError("public_inputs must be ProgramZkpPublicInputs")
    return tuple(field_element_from_text(item) for item in vector)


def classify_circuit_family(circuit_id: str) -> ProgramZkpCircuitFamily:
    """Classify a circuit identity for production compatibility."""

    identity = _text(circuit_id, field_name="circuit_id").lower()
    if "tdfol" in identity and "program-contract-trace" not in identity:
        return ProgramZkpCircuitFamily.TDFOL_ONLY
    if "nonzero-only" in identity or identity.endswith("@nonzero-v1"):
        return ProgramZkpCircuitFamily.NONZERO_ONLY_V1
    if identity == PROGRAM_CONTRACT_TRACE_CIRCUIT_ID.lower() or (
        "program-contract-trace" in identity and "@" in identity
    ):
        return ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE
    return ProgramZkpCircuitFamily.UNKNOWN


def is_versioned_artifact_id(artifact_id: str) -> bool:
    """Return True when an artifact id carries an explicit version suffix."""

    text = (artifact_id or "").strip()
    if not text:
        return False
    # Require @N, @N.M, or trailing :sha256- style content pin after a version.
    if "@" in text:
        suffix = text.rsplit("@", 1)[-1]
        if suffix and suffix[0].isdigit():
            return True
    if ":sha256" in text.lower() or text.lower().startswith("sha256:"):
        return True
    return False


def proof_bytes_are_simulated(proof_data: bytes | bytearray | memoryview | str) -> bool:
    """Detect educational SIMZKP layouts that must never grant authority."""

    if isinstance(proof_data, str):
        stripped = proof_data.strip()
        hex_candidate = stripped[2:] if stripped.startswith(("0x", "0X")) else stripped
        try:
            if hex_candidate and len(hex_candidate) % 2 == 0:
                raw = bytes.fromhex(hex_candidate)
            else:
                raw = proof_data.encode("utf-8")
        except ValueError:
            raw = proof_data.encode("utf-8")
    elif isinstance(proof_data, (bytes, bytearray, memoryview)):
        raw = bytes(proof_data)
    else:
        raise ProgramAnalysisZkpError("proof_data must be bytes or hex string")
    if len(raw) == _SIMZKP_PROOF_LENGTH and raw[:8] == _SIMZKP_MAGIC:
        return True
    if raw.startswith(b"SIMZKP"):
        return True
    return False


def _path_if_exists(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.exists() else None


def _resolve_executable(
    names: Sequence[str],
    *,
    env_names: Sequence[str] = (),
) -> tuple[str | None, str]:
    for env_name in env_names:
        configured = os.environ.get(env_name, "").strip()
        if configured:
            path = Path(configured).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path), "configured_via_%s" % env_name
    for name in names:
        found = shutil.which(name)
        if found:
            return found, "path_lookup"
    return None, "not_found"


@dataclass(frozen=True)
class ProgramZkpCapabilityCheck(CanonicalContract):
    """One production capability probe result."""

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_CAPABILITY_CHECK_SCHEMA

    dimension: ProgramZkpCapabilityDimension
    status: ProgramZkpCapabilityStatus
    reason: str
    production_eligible: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)
    denial_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimension",
            _enum(self.dimension, ProgramZkpCapabilityDimension, field_name="dimension"),
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, ProgramZkpCapabilityStatus, field_name="status"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, field_name="reason", required=False)
        )
        object.__setattr__(
            self,
            "production_eligible",
            _boolean(self.production_eligible, field_name="production_eligible"),
        )
        evidence = self.evidence if isinstance(self.evidence, Mapping) else {}
        object.__setattr__(self, "evidence", MappingProxyType(dict(evidence)))
        denials: list[str] = []
        for item in self.denial_reasons or ():
            denials.append(_text(item, field_name="denial_reason"))
        object.__setattr__(self, "denial_reasons", tuple(denials))
        if self.production_eligible and self.status in _FAIL_CLOSED_CAPABILITY_STATUSES:
            raise ProgramZkpCapabilityError(
                "dimension %s cannot be production_eligible with status %s"
                % (self.dimension.value, self.status.value)
            )
        if self.production_eligible and self.status not in _PRODUCTION_CAPABILITY_STATUSES:
            raise ProgramZkpCapabilityError(
                "production_eligible requires verified or available status"
            )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "dimension": self.dimension.value,
            "status": self.status.value,
            "reason": self.reason,
            "production_eligible": self.production_eligible,
            "evidence": dict(self.evidence),
            "denial_reasons": list(self.denial_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramZkpCapabilityCheck":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("capability check payload must be a mapping")
        data = dict(payload)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("content_id", None)
        return cls(
            dimension=data.get("dimension", ""),
            status=data.get("status", ""),
            reason=data.get("reason", ""),
            production_eligible=data.get("production_eligible", False),
            evidence=data.get("evidence") or {},
            denial_reasons=tuple(data.get("denial_reasons") or ()),
        )


@dataclass(frozen=True)
class ProgramZkpCapabilityConformanceReport(CanonicalContract):
    """Aggregate production ZK capability and conformance probe report.

    Evidence identity: ``vfs/zk-capability-conformance@1``.
    """

    SCHEMA: ClassVar[str] = PROGRAM_ZKP_CAPABILITY_CONFORMANCE_SCHEMA

    checks: tuple[ProgramZkpCapabilityCheck, ...]
    backend_mode: ProgramZkpBackendMode = ProgramZkpBackendMode.SHADOW
    field_encoding: ProgramZkpFieldEncodingKind = ProgramZkpFieldEncodingKind.BN254_SHA256
    circuit_family: ProgramZkpCircuitFamily = ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE
    circuit_id: str = PROGRAM_CONTRACT_TRACE_CIRCUIT_ID
    circuit_version: int = PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION
    knowledge_graph_fail_open: bool = False
    stale: bool = False
    architecture: str = ""
    cancellation_supported: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_mode",
            _enum(self.backend_mode, ProgramZkpBackendMode, field_name="backend_mode"),
        )
        object.__setattr__(
            self,
            "field_encoding",
            _enum(
                self.field_encoding,
                ProgramZkpFieldEncodingKind,
                field_name="field_encoding",
            ),
        )
        object.__setattr__(
            self,
            "circuit_family",
            _enum(
                self.circuit_family, ProgramZkpCircuitFamily, field_name="circuit_family"
            ),
        )
        object.__setattr__(
            self, "circuit_id", _text(self.circuit_id, field_name="circuit_id")
        )
        if (
            isinstance(self.circuit_version, bool)
            or not isinstance(self.circuit_version, int)
            or self.circuit_version < 1
        ):
            raise ProgramZkpVersionError("circuit_version must be a positive integer")
        object.__setattr__(
            self,
            "knowledge_graph_fail_open",
            _boolean(
                self.knowledge_graph_fail_open, field_name="knowledge_graph_fail_open"
            ),
        )
        object.__setattr__(self, "stale", _boolean(self.stale, field_name="stale"))
        object.__setattr__(
            self,
            "architecture",
            _text(self.architecture, field_name="architecture", required=False),
        )
        object.__setattr__(
            self,
            "cancellation_supported",
            _boolean(self.cancellation_supported, field_name="cancellation_supported"),
        )
        normalized: list[ProgramZkpCapabilityCheck] = []
        if not isinstance(self.checks, Sequence) or isinstance(self.checks, (str, bytes)):
            raise ProgramAnalysisZkpError("checks must be a sequence")
        for raw in self.checks:
            check = (
                raw
                if isinstance(raw, ProgramZkpCapabilityCheck)
                else ProgramZkpCapabilityCheck.from_dict(raw)
            )
            normalized.append(check)
        # Stable order by required dimension order, then extras.
        order = {dim: index for index, dim in enumerate(REQUIRED_CAPABILITY_DIMENSIONS)}
        normalized.sort(key=lambda item: order.get(item.dimension, 1_000))
        object.__setattr__(self, "checks", tuple(normalized))
        notes = tuple(
            _text(item, field_name="note", required=False)
            for item in (self.notes or ())
            if str(item).strip()
        )
        object.__setattr__(self, "notes", notes)
        seen = {check.dimension for check in self.checks}
        missing = [dim for dim in REQUIRED_CAPABILITY_DIMENSIONS if dim not in seen]
        if missing:
            raise ProgramZkpCapabilityError(
                "capability report missing required dimensions: %s"
                % ", ".join(dim.value for dim in missing)
            )

    @property
    def capability_epoch(self) -> str:
        """Stable identity used to bind receipts and detect capability loss."""

        return self.content_id

    @property
    def checks_by_dimension(
        self,
    ) -> Mapping[ProgramZkpCapabilityDimension, ProgramZkpCapabilityCheck]:
        return {check.dimension: check for check in self.checks}

    @property
    def denial_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.stale:
            reasons.append(ProgramZkpAuthorityDenialReason.STALE_CAPABILITY.value)
        if self.knowledge_graph_fail_open:
            reasons.append(
                ProgramZkpAuthorityDenialReason.KNOWLEDGE_GRAPH_FAIL_OPEN.value
            )
        if self.backend_mode is ProgramZkpBackendMode.SIMULATED:
            reasons.append(ProgramZkpAuthorityDenialReason.SIMULATED_DEFAULT.value)
        if self.backend_mode is ProgramZkpBackendMode.SHADOW:
            reasons.append(ProgramZkpAuthorityDenialReason.SHADOW_ONLY_ROLLOUT.value)
        if self.field_encoding is ProgramZkpFieldEncodingKind.PLACEHOLDER:
            reasons.append(
                ProgramZkpAuthorityDenialReason.PLACEHOLDER_FIELD_ENCODING.value
            )
        if self.field_encoding is ProgramZkpFieldEncodingKind.NONZERO_ONLY_V1:
            reasons.append(
                ProgramZkpAuthorityDenialReason.V1_NONZERO_ONLY_CIRCUIT.value
            )
        if self.circuit_family is ProgramZkpCircuitFamily.TDFOL_ONLY:
            reasons.append(
                ProgramZkpAuthorityDenialReason.INCOMPATIBLE_TDFOL_ONLY_CIRCUIT.value
            )
        if self.circuit_family is ProgramZkpCircuitFamily.NONZERO_ONLY_V1:
            reasons.append(
                ProgramZkpAuthorityDenialReason.V1_NONZERO_ONLY_CIRCUIT.value
            )
        for check in self.checks:
            for reason in check.denial_reasons:
                if reason not in reasons:
                    reasons.append(reason)
            if not check.production_eligible:
                # Dimension-level failures contribute a generic probe reason once.
                if (
                    ProgramZkpAuthorityDenialReason.PROBE_FAILED.value not in reasons
                    and check.status
                    in {
                        ProgramZkpCapabilityStatus.FAILED,
                        ProgramZkpCapabilityStatus.REJECTED,
                        ProgramZkpCapabilityStatus.UNAVAILABLE,
                    }
                ):
                    reasons.append(ProgramZkpAuthorityDenialReason.PROBE_FAILED.value)
        return tuple(reasons)

    @property
    def production_eligible(self) -> bool:
        if self.stale:
            return False
        if self.knowledge_graph_fail_open:
            return False
        if self.backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC:
            return False
        if self.field_encoding is not ProgramZkpFieldEncodingKind.BN254_SHA256:
            return False
        if self.circuit_family is not ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE:
            return False
        if self.circuit_version != PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION:
            return False
        if not self.cancellation_supported:
            return False
        if not all(check.production_eligible for check in self.checks):
            return False
        return True

    @property
    def rollout_mode(self) -> ProgramZkpRolloutMode:
        if self.production_eligible:
            return ProgramZkpRolloutMode.ENFORCEMENT
        # Fail closed: non-production always rolls out as shadow-only.
        return ProgramZkpRolloutMode.SHADOW

    @property
    def shadow_only(self) -> bool:
        return not self.production_eligible

    @property
    def authoritative_allowed(self) -> bool:
        return self.production_eligible

    def require_production_eligible(self) -> None:
        if not self.production_eligible:
            reasons = ", ".join(self.denial_reasons) or "capability_probe_failed"
            raise ProgramZkpAuthorityError(
                "production ZK authority denied: %s" % reasons
            )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION,
            "evidence": PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE,
            "checks": [check.to_dict() for check in self.checks],
            "backend_mode": self.backend_mode.value,
            "field_encoding": self.field_encoding.value,
            "circuit_family": self.circuit_family.value,
            "circuit_id": self.circuit_id,
            "circuit_version": self.circuit_version,
            "knowledge_graph_fail_open": self.knowledge_graph_fail_open,
            "stale": self.stale,
            "architecture": self.architecture,
            "cancellation_supported": self.cancellation_supported,
            "notes": list(self.notes),
            "production_eligible": self.production_eligible,
            "rollout_mode": self.rollout_mode.value,
            "shadow_only": self.shadow_only,
            "authoritative_allowed": self.authoritative_allowed,
            "denial_reasons": list(self.denial_reasons),
            "does_not_prove": sorted(TRACE_VALIDITY_DOES_NOT_PROVE),
            "trace_validity_scope": TRACE_VALIDITY_SCOPE_STATEMENT,
        }

    def to_public_artifact(self) -> Dict[str, Any]:
        public = {**self.to_dict(), "capability_epoch": self.capability_epoch}
        reject_private_witness_from_public_payload(public)
        return public

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProgramZkpCapabilityConformanceReport":
        if not isinstance(payload, Mapping):
            raise ProgramAnalysisZkpError("capability report payload must be a mapping")
        data = dict(payload)
        reject_private_witness_from_public_payload(data)
        data.pop("schema", None)
        data.pop("contract_version", None)
        data.pop("evidence", None)
        data.pop("production_eligible", None)
        data.pop("rollout_mode", None)
        data.pop("shadow_only", None)
        data.pop("authoritative_allowed", None)
        data.pop("denial_reasons", None)
        data.pop("does_not_prove", None)
        data.pop("trace_validity_scope", None)
        claimed_id = data.pop("content_id", None) or data.pop("capability_epoch", None)
        result = cls(
            checks=tuple(data.get("checks") or ()),
            backend_mode=data.get("backend_mode", ProgramZkpBackendMode.SHADOW),
            field_encoding=data.get(
                "field_encoding", ProgramZkpFieldEncodingKind.BN254_SHA256
            ),
            circuit_family=data.get(
                "circuit_family", ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE
            ),
            circuit_id=data.get("circuit_id", PROGRAM_CONTRACT_TRACE_CIRCUIT_ID),
            circuit_version=data.get(
                "circuit_version", PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION
            ),
            knowledge_graph_fail_open=data.get("knowledge_graph_fail_open", False),
            stale=data.get("stale", False),
            architecture=data.get("architecture", ""),
            cancellation_supported=data.get("cancellation_supported", True),
            notes=tuple(data.get("notes") or ()),
        )
        if claimed_id and claimed_id != result.content_id:
            raise ProgramZkpTamperError(
                "forged capability conformance report identity rejected"
            )
        return result


def _check(
    dimension: ProgramZkpCapabilityDimension,
    *,
    status: ProgramZkpCapabilityStatus,
    reason: str,
    production_eligible: bool = False,
    evidence: Mapping[str, Any] | None = None,
    denial_reasons: Sequence[str] = (),
) -> ProgramZkpCapabilityCheck:
    return ProgramZkpCapabilityCheck(
        dimension=dimension,
        status=status,
        reason=reason,
        production_eligible=production_eligible,
        evidence=dict(evidence or {}),
        denial_reasons=tuple(denial_reasons),
    )


def _probe_executable_architecture(
    *,
    executable_names: Sequence[str] = ("provekit-cli", "provekit", "groth16"),
    env_names: Sequence[str] = (
        "IPFS_DATASETS_PROVEKIT_BINARY",
        "PROVEKIT_CLI",
        "IPFS_DATASETS_GROTH16_BINARY",
        "GROTH16_BINARY",
    ),
    architecture_override: str | None = None,
) -> ProgramZkpCapabilityCheck:
    arch = architecture_override or "%s-%s" % (platform.system(), platform.machine())
    path, source = _resolve_executable(executable_names, env_names=env_names)
    evidence = {
        "architecture": arch,
        "executable_path": path or "",
        "resolution": source,
        "machine": platform.machine(),
        "system": platform.system(),
    }
    if path is None:
        return _check(
            ProgramZkpCapabilityDimension.EXECUTABLE_ARCHITECTURE,
            status=ProgramZkpCapabilityStatus.UNAVAILABLE,
            reason="no production ZK executable discovered for architecture %s" % arch,
            evidence=evidence,
            denial_reasons=(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value,),
        )
    return _check(
        ProgramZkpCapabilityDimension.EXECUTABLE_ARCHITECTURE,
        status=ProgramZkpCapabilityStatus.AVAILABLE,
        reason="production ZK executable discovered on %s" % arch,
        production_eligible=True,
        evidence=evidence,
    )


def _probe_backend(
    *,
    backend_mode: ProgramZkpBackendMode,
    backend_id: str = "",
) -> ProgramZkpCapabilityCheck:
    identity = _text(backend_id, field_name="backend_id", required=False)
    lower = identity.lower()
    simulated_tokens = ("sim", "simulated", "mock", "fake", "demo", "educational")
    if backend_mode is ProgramZkpBackendMode.SIMULATED or any(
        token in lower.split(":") or token == lower for token in simulated_tokens
    ):
        return _check(
            ProgramZkpCapabilityDimension.BACKEND,
            status=ProgramZkpCapabilityStatus.SIMULATED,
            reason="simulated backend defaults cannot grant production authority",
            evidence={"backend_mode": backend_mode.value, "backend_id": identity},
            denial_reasons=(ProgramZkpAuthorityDenialReason.SIMULATED_DEFAULT.value,),
        )
    if backend_mode is ProgramZkpBackendMode.SHADOW:
        return _check(
            ProgramZkpCapabilityDimension.BACKEND,
            status=ProgramZkpCapabilityStatus.SHADOW,
            reason="shadow backend rollout is non-authoritative",
            evidence={"backend_mode": backend_mode.value, "backend_id": identity},
            denial_reasons=(ProgramZkpAuthorityDenialReason.SHADOW_ONLY_ROLLOUT.value,),
        )
    if backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC:
        return _check(
            ProgramZkpCapabilityDimension.BACKEND,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="backend is not cryptographic",
            evidence={"backend_mode": backend_mode.value, "backend_id": identity},
            denial_reasons=(
                ProgramZkpAuthorityDenialReason.BACKEND_NOT_CRYPTOGRAPHIC.value,
            ),
        )
    return _check(
        ProgramZkpCapabilityDimension.BACKEND,
        status=ProgramZkpCapabilityStatus.AVAILABLE,
        reason="cryptographic backend mode is selected",
        production_eligible=True,
        evidence={"backend_mode": backend_mode.value, "backend_id": identity},
    )


def _probe_circuit_version(
    *,
    circuit_id: str,
    circuit_version: int,
    circuit_family: ProgramZkpCircuitFamily,
) -> ProgramZkpCapabilityCheck:
    family = circuit_family
    denials: list[str] = []
    if family is ProgramZkpCircuitFamily.TDFOL_ONLY:
        denials.append(
            ProgramZkpAuthorityDenialReason.INCOMPATIBLE_TDFOL_ONLY_CIRCUIT.value
        )
    if family is ProgramZkpCircuitFamily.NONZERO_ONLY_V1:
        denials.append(ProgramZkpAuthorityDenialReason.V1_NONZERO_ONLY_CIRCUIT.value)
    if not is_versioned_artifact_id(circuit_id):
        denials.append(ProgramZkpAuthorityDenialReason.UNVERSIONED_ARTIFACT.value)
    if circuit_version != PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION:
        denials.append(ProgramZkpAuthorityDenialReason.UNVERSIONED_ARTIFACT.value)
    if family is not ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE:
        return _check(
            ProgramZkpCapabilityDimension.CIRCUIT_VERSION,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="circuit family %s is not production-compatible" % family.value,
            evidence={
                "circuit_id": circuit_id,
                "circuit_version": circuit_version,
                "circuit_family": family.value,
            },
            denial_reasons=denials
            or (ProgramZkpAuthorityDenialReason.PROBE_FAILED.value,),
        )
    if denials:
        return _check(
            ProgramZkpCapabilityDimension.CIRCUIT_VERSION,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="circuit identity or version is not production-admissible",
            evidence={
                "circuit_id": circuit_id,
                "circuit_version": circuit_version,
                "circuit_family": family.value,
            },
            denial_reasons=denials,
        )
    return _check(
        ProgramZkpCapabilityDimension.CIRCUIT_VERSION,
        status=ProgramZkpCapabilityStatus.VERIFIED,
        reason="program_contract_trace circuit version is pinned",
        production_eligible=True,
        evidence={
            "circuit_id": circuit_id,
            "circuit_version": circuit_version,
            "circuit_family": family.value,
            "max_trace_steps": PROGRAM_CONTRACT_TRACE_MAX_TRACE_STEPS,
            "canonical_trace_length": PROGRAM_CONTRACT_TRACE_CANONICAL_TRACE_LENGTH,
        },
    )


def _probe_setup_artifacts(
    *,
    setup_dir: str | Path | None,
    proving_key_id: str,
    verifying_key_id: str,
    require_files: Sequence[str] = (
        "proving_key.bin",
        "verifying_key.bin",
    ),
) -> ProgramZkpCapabilityCheck:
    denials: list[str] = []
    path = _path_if_exists(setup_dir)
    present: list[str] = []
    missing: list[str] = []
    if path is None:
        denials.append(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value)
        missing = list(require_files)
    else:
        for name in require_files:
            candidate = path / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                present.append(name)
            else:
                missing.append(name)
                denials.append(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value)
    if not is_versioned_artifact_id(proving_key_id) or not is_versioned_artifact_id(
        verifying_key_id
    ):
        denials.append(ProgramZkpAuthorityDenialReason.UNVERSIONED_ARTIFACT.value)
    evidence = {
        "setup_dir": str(path) if path else "",
        "present_files": present,
        "missing_files": missing,
        "proving_key_id": proving_key_id,
        "verifying_key_id": verifying_key_id,
    }
    if denials:
        return _check(
            ProgramZkpCapabilityDimension.SETUP_ARTIFACTS,
            status=ProgramZkpCapabilityStatus.UNAVAILABLE
            if ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value in denials
            else ProgramZkpCapabilityStatus.REJECTED,
            reason="setup artifacts are missing, empty, or unversioned",
            evidence=evidence,
            denial_reasons=tuple(dict.fromkeys(denials)),
        )
    return _check(
        ProgramZkpCapabilityDimension.SETUP_ARTIFACTS,
        status=ProgramZkpCapabilityStatus.AVAILABLE,
        reason="versioned setup artifacts are present",
        production_eligible=True,
        evidence=evidence,
    )


def _probe_ceremony(
    *,
    ceremony_id: str,
    ceremony_manifest: Mapping[str, Any] | None,
    ceremony_production_eligible: bool | None = None,
) -> ProgramZkpCapabilityCheck:
    denials: list[str] = []
    if not ceremony_id or not is_versioned_artifact_id(ceremony_id):
        denials.append(ProgramZkpAuthorityDenialReason.UNVERSIONED_ARTIFACT.value)
    production = False
    evidence: Dict[str, Any] = {"ceremony_id": ceremony_id}
    if ceremony_production_eligible is True:
        production = True
        evidence["ceremony_source"] = "explicit_admission"
    elif ceremony_manifest is not None:
        evidence["ceremony_source"] = "manifest"
        evidence["manifest_keys"] = sorted(str(k) for k in ceremony_manifest.keys())
        # Prefer shared MCP++ ceremony validation when available; never fail-open.
        try:
            from ipfs_datasets_py.logic.zkp.ceremony import (  # type: ignore
                validate_groth16_mpc_ceremony,
            )

            validation = validate_groth16_mpc_ceremony(dict(ceremony_manifest))
            production = bool(getattr(validation, "production_eligible", False))
            evidence["ceremony_valid"] = bool(getattr(validation, "valid", False))
            evidence["ceremony_cid"] = str(getattr(validation, "ceremony_cid", ""))
            if not production:
                denials.append(ProgramZkpAuthorityDenialReason.CEREMONY_INELIGIBLE.value)
        except Exception as exc:  # pragma: no cover - defensive import/validation path
            denials.append(ProgramZkpAuthorityDenialReason.CEREMONY_INELIGIBLE.value)
            evidence["ceremony_error"] = type(exc).__name__
    else:
        denials.append(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value)
        evidence["ceremony_source"] = "absent"
    if ceremony_production_eligible is False:
        production = False
        if ProgramZkpAuthorityDenialReason.CEREMONY_INELIGIBLE.value not in denials:
            denials.append(ProgramZkpAuthorityDenialReason.CEREMONY_INELIGIBLE.value)
    if production and not denials:
        return _check(
            ProgramZkpCapabilityDimension.CEREMONY,
            status=ProgramZkpCapabilityStatus.VERIFIED,
            reason="ceremony is production-eligible and versioned",
            production_eligible=True,
            evidence=evidence,
        )
    return _check(
        ProgramZkpCapabilityDimension.CEREMONY,
        status=ProgramZkpCapabilityStatus.REJECTED
        if denials
        else ProgramZkpCapabilityStatus.UNAVAILABLE,
        reason="ceremony is missing, unversioned, or not production-eligible",
        evidence=evidence,
        denial_reasons=tuple(dict.fromkeys(denials))
        or (ProgramZkpAuthorityDenialReason.CEREMONY_INELIGIBLE.value,),
    )


def _probe_key(
    dimension: ProgramZkpCapabilityDimension,
    *,
    key_id: str,
    key_material: bytes | bytearray | memoryview | str | None = None,
    expected_digest: str = "",
) -> ProgramZkpCapabilityCheck:
    denials: list[str] = []
    if not key_id:
        denials.append(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value)
    elif not is_versioned_artifact_id(key_id):
        denials.append(ProgramZkpAuthorityDenialReason.UNVERSIONED_ARTIFACT.value)
    evidence: Dict[str, Any] = {"key_id": key_id}
    if key_material is None:
        # Identity-only probe: versioned id is required; material optional.
        if denials:
            return _check(
                dimension,
                status=ProgramZkpCapabilityStatus.UNAVAILABLE,
                reason="key identity is missing or unversioned",
                evidence=evidence,
                denial_reasons=denials,
            )
        return _check(
            dimension,
            status=ProgramZkpCapabilityStatus.CONFIGURED,
            reason="key identity is versioned; material not supplied to this probe",
            production_eligible=False,
            evidence=evidence,
            denial_reasons=(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value,),
        )
    if isinstance(key_material, str):
        raw = key_material.encode("utf-8")
    else:
        raw = bytes(key_material)
    if not raw:
        denials.append(ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    evidence["key_digest"] = digest
    evidence["key_bytes"] = len(raw)
    if expected_digest:
        expected = _text(expected_digest, field_name="expected_digest")
        evidence["expected_digest"] = expected
        if digest != expected and expected != key_id and not key_id.endswith(
            digest.removeprefix("sha256:")
        ):
            denials.append(ProgramZkpAuthorityDenialReason.CORRUPTED_KEY.value)
    if denials:
        status = (
            ProgramZkpCapabilityStatus.REJECTED
            if ProgramZkpAuthorityDenialReason.CORRUPTED_KEY.value in denials
            else ProgramZkpCapabilityStatus.UNAVAILABLE
        )
        return _check(
            dimension,
            status=status,
            reason="key material failed integrity or version checks",
            evidence=evidence,
            denial_reasons=tuple(dict.fromkeys(denials)),
        )
    return _check(
        dimension,
        status=ProgramZkpCapabilityStatus.VERIFIED,
        reason="key material digests cleanly and identity is versioned",
        production_eligible=True,
        evidence=evidence,
    )


def _probe_public_input_codec(
    *,
    codec_id: str = PUBLIC_INPUT_CODEC_ID,
    codec_version: str = PUBLIC_INPUT_CODEC_VERSION,
    field_encoding: ProgramZkpFieldEncodingKind,
) -> ProgramZkpCapabilityCheck:
    denials: list[str] = []
    if codec_id != PUBLIC_INPUT_CODEC_ID or codec_version != PUBLIC_INPUT_CODEC_VERSION:
        denials.append(ProgramZkpAuthorityDenialReason.CODEC_INCOMPATIBLE.value)
    if field_encoding is ProgramZkpFieldEncodingKind.PLACEHOLDER:
        denials.append(
            ProgramZkpAuthorityDenialReason.PLACEHOLDER_FIELD_ENCODING.value
        )
    if field_encoding is ProgramZkpFieldEncodingKind.NONZERO_ONLY_V1:
        denials.append(ProgramZkpAuthorityDenialReason.V1_NONZERO_ONLY_CIRCUIT.value)
    if field_encoding is not ProgramZkpFieldEncodingKind.BN254_SHA256:
        if ProgramZkpAuthorityDenialReason.CODEC_INCOMPATIBLE.value not in denials:
            denials.append(ProgramZkpAuthorityDenialReason.CODEC_INCOMPATIBLE.value)
    evidence = {
        "codec_id": codec_id,
        "codec_version": codec_version,
        "field_encoding": field_encoding.value,
        "commitment_keys": list(PUBLIC_COMMITMENT_KEYS),
    }
    if denials:
        return _check(
            ProgramZkpCapabilityDimension.PUBLIC_INPUT_CODEC,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="public-input codec or field encoding is not production-admissible",
            evidence=evidence,
            denial_reasons=tuple(dict.fromkeys(denials)),
        )
    return _check(
        ProgramZkpCapabilityDimension.PUBLIC_INPUT_CODEC,
        status=ProgramZkpCapabilityStatus.VERIFIED,
        reason="canonical BN254 SHA-256 public-input codec is pinned",
        production_eligible=True,
        evidence=evidence,
    )


def _probe_proof_schema(
    *,
    proof_schema_id: str = PROGRAM_ZKP_PROOF_SCHEMA_ID,
    sample_proof: bytes | str | None = None,
) -> ProgramZkpCapabilityCheck:
    denials: list[str] = []
    if proof_schema_id != PROGRAM_ZKP_PROOF_SCHEMA_ID:
        denials.append(ProgramZkpAuthorityDenialReason.PROOF_SCHEMA_MISMATCH.value)
    evidence: Dict[str, Any] = {"proof_schema_id": proof_schema_id}
    if sample_proof is not None:
        simulated = proof_bytes_are_simulated(sample_proof)
        evidence["sample_is_simulated"] = simulated
        if simulated:
            denials.append(ProgramZkpAuthorityDenialReason.SIMULATED_DEFAULT.value)
            denials.append(ProgramZkpAuthorityDenialReason.CORRUPTED_PROOF.value)
    if denials:
        return _check(
            ProgramZkpCapabilityDimension.PROOF_SCHEMA,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="proof schema is mismatched or sample is simulated/corrupt",
            evidence=evidence,
            denial_reasons=tuple(dict.fromkeys(denials)),
        )
    return _check(
        ProgramZkpCapabilityDimension.PROOF_SCHEMA,
        status=ProgramZkpCapabilityStatus.VERIFIED,
        reason="production proof schema identity is pinned",
        production_eligible=True,
        evidence=evidence,
    )


def _probe_independent_verifier(
    *,
    independent_verifier_available: bool,
    verifier_id: str = "",
) -> ProgramZkpCapabilityCheck:
    if not independent_verifier_available:
        return _check(
            ProgramZkpCapabilityDimension.INDEPENDENT_VERIFIER,
            status=ProgramZkpCapabilityStatus.UNAVAILABLE,
            reason="independent verifier is not available",
            evidence={"verifier_id": verifier_id},
            denial_reasons=(
                ProgramZkpAuthorityDenialReason.INDEPENDENT_VERIFIER_ABSENT.value,
            ),
        )
    return _check(
        ProgramZkpCapabilityDimension.INDEPENDENT_VERIFIER,
        status=ProgramZkpCapabilityStatus.AVAILABLE,
        reason="independent verifier path is available",
        production_eligible=True,
        evidence={"verifier_id": verifier_id or "independent"},
    )


def _probe_bounds(
    *,
    max_trace_steps: int = PROGRAM_CONTRACT_TRACE_MAX_TRACE_STEPS,
    max_text_bytes: int = MAX_TEXT_BYTES,
    observed_trace_steps: int | None = None,
) -> ProgramZkpCapabilityCheck:
    denials: list[str] = []
    if max_trace_steps < PROGRAM_CONTRACT_TRACE_CANONICAL_TRACE_LENGTH:
        denials.append(ProgramZkpAuthorityDenialReason.BOUNDS_EXCEEDED.value)
    if max_trace_steps > PROGRAM_CONTRACT_TRACE_MAX_TRACE_STEPS:
        denials.append(ProgramZkpAuthorityDenialReason.BOUNDS_EXCEEDED.value)
    if observed_trace_steps is not None and (
        observed_trace_steps > max_trace_steps
        or observed_trace_steps != PROGRAM_CONTRACT_TRACE_CANONICAL_TRACE_LENGTH
    ):
        denials.append(ProgramZkpAuthorityDenialReason.BOUNDS_EXCEEDED.value)
    evidence = {
        "max_trace_steps": max_trace_steps,
        "canonical_trace_length": PROGRAM_CONTRACT_TRACE_CANONICAL_TRACE_LENGTH,
        "max_text_bytes": max_text_bytes,
        "observed_trace_steps": observed_trace_steps,
    }
    if denials:
        return _check(
            ProgramZkpCapabilityDimension.BOUNDS,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="resource or trace bounds are outside production envelope",
            evidence=evidence,
            denial_reasons=denials,
        )
    return _check(
        ProgramZkpCapabilityDimension.BOUNDS,
        status=ProgramZkpCapabilityStatus.VERIFIED,
        reason="production bounds match program_contract_trace envelope",
        production_eligible=True,
        evidence=evidence,
    )


def _probe_cancellation(
    *,
    cancellation_supported: bool,
    cancellation_event: threading.Event | None = None,
) -> ProgramZkpCapabilityCheck:
    if cancellation_event is not None and cancellation_event.is_set():
        return _check(
            ProgramZkpCapabilityDimension.CANCELLATION,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="capability probe cancelled before completion",
            evidence={"cancellation_supported": cancellation_supported, "cancelled": True},
            denial_reasons=(ProgramZkpAuthorityDenialReason.CANCELLED.value,),
        )
    if not cancellation_supported:
        return _check(
            ProgramZkpCapabilityDimension.CANCELLATION,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason="backend does not support cooperative cancellation",
            evidence={"cancellation_supported": False},
            denial_reasons=(ProgramZkpAuthorityDenialReason.CANCELLED.value,),
        )
    return _check(
        ProgramZkpCapabilityDimension.CANCELLATION,
        status=ProgramZkpCapabilityStatus.VERIFIED,
        reason="cooperative cancellation is supported",
        production_eligible=True,
        evidence={"cancellation_supported": True},
    )


def probe_program_analysis_zkp_capability(
    *,
    backend_mode: ProgramZkpBackendMode | str = ProgramZkpBackendMode.SHADOW,
    backend_id: str = "",
    circuit_id: str = PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
    circuit_version: int = PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION,
    circuit_family: ProgramZkpCircuitFamily | str | None = None,
    field_encoding: ProgramZkpFieldEncodingKind | str = (
        ProgramZkpFieldEncodingKind.BN254_SHA256
    ),
    proving_key_id: str = "",
    verifying_key_id: str = "",
    ceremony_id: str = "",
    ceremony_manifest: Mapping[str, Any] | None = None,
    ceremony_production_eligible: bool | None = None,
    setup_dir: str | Path | None = None,
    proving_key_material: bytes | str | None = None,
    verifying_key_material: bytes | str | None = None,
    proving_key_digest: str = "",
    verifying_key_digest: str = "",
    codec_id: str = PUBLIC_INPUT_CODEC_ID,
    codec_version: str = PUBLIC_INPUT_CODEC_VERSION,
    proof_schema_id: str = PROGRAM_ZKP_PROOF_SCHEMA_ID,
    sample_proof: bytes | str | None = None,
    independent_verifier_available: bool = False,
    verifier_id: str = "",
    knowledge_graph_fail_open: bool = False,
    stale: bool = False,
    cancellation_supported: bool = True,
    cancellation_event: threading.Event | None = None,
    architecture_override: str | None = None,
    max_trace_steps: int = PROGRAM_CONTRACT_TRACE_MAX_TRACE_STEPS,
    observed_trace_steps: int | None = None,
    notes: Sequence[str] = (),
    # Test/operator injection: when True, treat configured keys/setup as present
    # only if their production_eligible flags are also forced via material.
    force_checks: Mapping[str, Mapping[str, Any]] | None = None,
) -> ProgramZkpCapabilityConformanceReport:
    """Probe production ZK capability and publish explicit shadow/degraded status.

    Every required dimension is evaluated.  Simulated defaults, knowledge-graph
    fail-open fallback, placeholder encodings, incompatible circuits, missing
    or unversioned artifacts, and stale probes fail closed for authority.
    """

    mode = _enum(backend_mode, ProgramZkpBackendMode, field_name="backend_mode")
    encoding = _enum(
        field_encoding, ProgramZkpFieldEncodingKind, field_name="field_encoding"
    )
    family = (
        _enum(circuit_family, ProgramZkpCircuitFamily, field_name="circuit_family")
        if circuit_family is not None
        else classify_circuit_family(circuit_id)
    )
    if cancellation_event is not None and cancellation_event.is_set():
        # Cancellation aborts the probe with a complete but rejected report.
        cancelled = _probe_cancellation(
            cancellation_supported=cancellation_supported,
            cancellation_event=cancellation_event,
        )
        # Still emit every dimension so report construction succeeds.
        fail = lambda dimension, reason: _check(  # noqa: E731
            dimension,
            status=ProgramZkpCapabilityStatus.REJECTED,
            reason=reason,
            denial_reasons=(ProgramZkpAuthorityDenialReason.CANCELLED.value,),
        )
        checks = (
            fail(
                ProgramZkpCapabilityDimension.EXECUTABLE_ARCHITECTURE,
                "probe cancelled",
            ),
            fail(ProgramZkpCapabilityDimension.BACKEND, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.CIRCUIT_VERSION, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.SETUP_ARTIFACTS, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.CEREMONY, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.PROVING_KEY, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.VERIFYING_KEY, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.PUBLIC_INPUT_CODEC, "probe cancelled"),
            fail(ProgramZkpCapabilityDimension.PROOF_SCHEMA, "probe cancelled"),
            fail(
                ProgramZkpCapabilityDimension.INDEPENDENT_VERIFIER, "probe cancelled"
            ),
            fail(ProgramZkpCapabilityDimension.BOUNDS, "probe cancelled"),
            cancelled,
        )
        return ProgramZkpCapabilityConformanceReport(
            checks=checks,
            backend_mode=mode,
            field_encoding=encoding,
            circuit_family=family,
            circuit_id=circuit_id,
            circuit_version=circuit_version,
            knowledge_graph_fail_open=knowledge_graph_fail_open,
            stale=stale,
            architecture=architecture_override
            or "%s-%s" % (platform.system(), platform.machine()),
            cancellation_supported=cancellation_supported,
            notes=tuple(notes) + ("cancelled",),
        )

    checks = [
        _probe_executable_architecture(architecture_override=architecture_override),
        _probe_backend(backend_mode=mode, backend_id=backend_id),
        _probe_circuit_version(
            circuit_id=circuit_id,
            circuit_version=circuit_version,
            circuit_family=family,
        ),
        _probe_setup_artifacts(
            setup_dir=setup_dir,
            proving_key_id=proving_key_id,
            verifying_key_id=verifying_key_id,
        ),
        _probe_ceremony(
            ceremony_id=ceremony_id,
            ceremony_manifest=ceremony_manifest,
            ceremony_production_eligible=ceremony_production_eligible,
        ),
        _probe_key(
            ProgramZkpCapabilityDimension.PROVING_KEY,
            key_id=proving_key_id,
            key_material=proving_key_material,
            expected_digest=proving_key_digest,
        ),
        _probe_key(
            ProgramZkpCapabilityDimension.VERIFYING_KEY,
            key_id=verifying_key_id,
            key_material=verifying_key_material,
            expected_digest=verifying_key_digest,
        ),
        _probe_public_input_codec(
            codec_id=codec_id,
            codec_version=codec_version,
            field_encoding=encoding,
        ),
        _probe_proof_schema(
            proof_schema_id=proof_schema_id, sample_proof=sample_proof
        ),
        _probe_independent_verifier(
            independent_verifier_available=independent_verifier_available,
            verifier_id=verifier_id,
        ),
        _probe_bounds(
            max_trace_steps=max_trace_steps,
            observed_trace_steps=observed_trace_steps,
        ),
        _probe_cancellation(
            cancellation_supported=cancellation_supported,
            cancellation_event=cancellation_event,
        ),
    ]

    if force_checks:
        rewritten: list[ProgramZkpCapabilityCheck] = []
        for check in checks:
            override = force_checks.get(check.dimension.value)
            if not override:
                rewritten.append(check)
                continue
            rewritten.append(
                ProgramZkpCapabilityCheck(
                    dimension=check.dimension,
                    status=override.get("status", check.status),
                    reason=override.get("reason", check.reason),
                    production_eligible=override.get(
                        "production_eligible", check.production_eligible
                    ),
                    evidence=override.get("evidence", dict(check.evidence)),
                    denial_reasons=tuple(
                        override.get("denial_reasons", check.denial_reasons)
                    ),
                )
            )
        checks = rewritten

    return ProgramZkpCapabilityConformanceReport(
        checks=tuple(checks),
        backend_mode=mode,
        field_encoding=encoding,
        circuit_family=family,
        circuit_id=circuit_id,
        circuit_version=circuit_version,
        knowledge_graph_fail_open=knowledge_graph_fail_open,
        stale=stale,
        architecture=architecture_override
        or "%s-%s" % (platform.system(), platform.machine()),
        cancellation_supported=cancellation_supported,
        notes=tuple(notes),
    )


def build_production_ready_capability_fixture(
    *,
    proving_key_material: bytes = b"pk-fixture-material-v1",
    verifying_key_material: bytes = b"vk-fixture-material-v1",
    ceremony_id: str = "ceremony:program-contract-trace@1",
    proving_key_id: str = "pk:program-contract-trace@1:sha256-pk-fixture",
    verifying_key_id: str = "vk:program-contract-trace@1:sha256-vk-fixture",
    backend_id: str = "backend:provekit-groth16@1",
    verifier_id: str = "verifier:program-analysis-zkp-independent@1",
    architecture: str = "fixture-linux-x86_64",
) -> ProgramZkpCapabilityConformanceReport:
    """Construct an explicit production-eligible fixture for hermetic tests.

    Real environments must go through :func:`probe_program_analysis_zkp_capability`
    without ``force_checks``.  This helper never runs in production code paths
    unless an operator deliberately imports it.
    """

    pk_digest = "sha256:" + hashlib.sha256(proving_key_material).hexdigest()
    vk_digest = "sha256:" + hashlib.sha256(verifying_key_material).hexdigest()
    return probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        backend_id=backend_id,
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        circuit_version=PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION,
        field_encoding=ProgramZkpFieldEncodingKind.BN254_SHA256,
        proving_key_id=proving_key_id,
        verifying_key_id=verifying_key_id,
        ceremony_id=ceremony_id,
        ceremony_production_eligible=True,
        proving_key_material=proving_key_material,
        verifying_key_material=verifying_key_material,
        proving_key_digest=pk_digest,
        verifying_key_digest=vk_digest,
        independent_verifier_available=True,
        verifier_id=verifier_id,
        knowledge_graph_fail_open=False,
        stale=False,
        cancellation_supported=True,
        architecture_override=architecture,
        force_checks={
            ProgramZkpCapabilityDimension.EXECUTABLE_ARCHITECTURE.value: {
                "status": ProgramZkpCapabilityStatus.AVAILABLE.value,
                "reason": "fixture executable admitted for hermetic tests",
                "production_eligible": True,
                "evidence": {
                    "architecture": architecture,
                    "executable_path": "/fixture/provekit",
                    "resolution": "fixture",
                },
                "denial_reasons": (),
            },
            ProgramZkpCapabilityDimension.SETUP_ARTIFACTS.value: {
                "status": ProgramZkpCapabilityStatus.AVAILABLE.value,
                "reason": "fixture setup artifacts admitted for hermetic tests",
                "production_eligible": True,
                "evidence": {
                    "setup_dir": "/fixture/setup",
                    "present_files": ["proving_key.bin", "verifying_key.bin"],
                    "missing_files": [],
                    "proving_key_id": proving_key_id,
                    "verifying_key_id": verifying_key_id,
                },
                "denial_reasons": (),
            },
        },
    )


def grants_production_authority(
    receipt: ProgramZkpVerificationReceipt,
    capability: ProgramZkpCapabilityConformanceReport,
) -> bool:
    """Return True only when receipt and live capability both admit authority."""

    if not isinstance(receipt, ProgramZkpVerificationReceipt):
        raise ProgramAnalysisZkpError("receipt must be ProgramZkpVerificationReceipt")
    if not isinstance(capability, ProgramZkpCapabilityConformanceReport):
        raise ProgramAnalysisZkpError(
            "capability must be ProgramZkpCapabilityConformanceReport"
        )
    if not receipt.authoritative:
        return False
    if not capability.production_eligible:
        return False
    if receipt.capability_epoch != capability.capability_epoch:
        return False
    return True


def require_production_authority(
    receipt: ProgramZkpVerificationReceipt,
    capability: ProgramZkpCapabilityConformanceReport,
) -> None:
    """Fail closed unless the receipt still grants production authority."""

    if not grants_production_authority(receipt, capability):
        reasons = list(capability.denial_reasons)
        if receipt.capability_epoch != capability.capability_epoch:
            reasons.append(ProgramZkpAuthorityDenialReason.CAPABILITY_LOSS.value)
        if not receipt.authoritative:
            reasons.append(ProgramZkpAuthorityDenialReason.SHADOW_ONLY_ROLLOUT.value)
        raise ProgramZkpAuthorityError(
            "production ZK authority denied: %s"
            % (", ".join(reasons) or "not_authoritative")
        )


def invalidate_authority_on_capability_loss(
    receipt: ProgramZkpVerificationReceipt,
    *,
    previous_capability: ProgramZkpCapabilityConformanceReport,
    current_capability: ProgramZkpCapabilityConformanceReport,
) -> ProgramZkpVerificationReceipt:
    """Return a non-authoritative projection after capability loss.

    Prior authoritative receipts are invalidated when the capability epoch
    changes or the current probe is no longer production-eligible.
    """

    if not isinstance(receipt, ProgramZkpVerificationReceipt):
        raise ProgramAnalysisZkpError("receipt must be ProgramZkpVerificationReceipt")
    lost = (
        not current_capability.production_eligible
        or previous_capability.capability_epoch != current_capability.capability_epoch
        or receipt.capability_epoch != current_capability.capability_epoch
    )
    if not lost and receipt.authoritative:
        return receipt
    # Re-bind as non-authoritative independent rejection of authority projection.
    return ProgramZkpVerificationReceipt(
        statement=receipt.statement,
        verdict=receipt.verdict,
        verifier_id=receipt.verifier_id,
        verifying_key_id=receipt.verifying_key_id,
        circuit_id=receipt.circuit_id,
        public_input_digest=receipt.public_input_digest,
        ceremony_id=receipt.ceremony_id,
        public_input_codec_version=receipt.public_input_codec_version,
        backend_mode=receipt.backend_mode,
        capability_epoch=current_capability.capability_epoch,
        capability_production_eligible=False,
        proof_schema_id=receipt.proof_schema_id,
        independent_verifier=receipt.independent_verifier,
    )


def verify_program_zkp_independently(
    envelope: ProgramZkpShadowEnvelope,
    *,
    capability: ProgramZkpCapabilityConformanceReport,
    verifier_id: str,
    proof_bytes: bytes | bytearray | memoryview | str,
    verifying_key_material: bytes | bytearray | memoryview | str,
    public_inputs: ProgramZkpPublicInputs | None = None,
    expected_verifying_key_digest: str = "",
    cryptographic_verify: Callable[[bytes, bytes, tuple[int, ...]], bool] | None = None,
    cancellation_event: threading.Event | None = None,
) -> ProgramZkpVerificationReceipt:
    """Independently verify a proof against keys, codec, and live capability.

    Rejects simulated proofs, corrupted keys/inputs, schema mismatches, and
    non-production capability states.  When ``cryptographic_verify`` is omitted,
    structural production gates still run and the verdict is REJECTED unless a
    caller supplies a verified cryptographic callback (or the envelope is not
    cryptographic).  Deterministic receipt identity follows content addressing.
    """

    if not isinstance(envelope, ProgramZkpShadowEnvelope):
        raise ProgramAnalysisZkpError("envelope must be ProgramZkpShadowEnvelope")
    if not isinstance(capability, ProgramZkpCapabilityConformanceReport):
        raise ProgramAnalysisZkpError(
            "capability must be ProgramZkpCapabilityConformanceReport"
        )
    if cancellation_event is not None and cancellation_event.is_set():
        raise ProgramZkpCapabilityError("independent verification cancelled")

    pins = (
        public_inputs
        if isinstance(public_inputs, ProgramZkpPublicInputs)
        else envelope.statement.public_inputs
    )
    # Input binding
    if pins.public_input_digest != envelope.statement.public_input_digest:
        raise ProgramZkpTamperError("corrupted or drifted public inputs rejected")
    if pins.verifying_key_id != envelope.statement.public_inputs.verifying_key_id:
        raise ProgramZkpTamperError("corrupted verifying key identity rejected")
    if pins.circuit_id != envelope.statement.public_inputs.circuit_id:
        raise ProgramZkpTamperError("corrupted circuit identity rejected")
    if pins.ceremony_id != envelope.statement.public_inputs.ceremony_id:
        raise ProgramZkpTamperError("corrupted ceremony identity rejected")
    if pins.public_input_codec_id != PUBLIC_INPUT_CODEC_ID:
        raise ProgramZkpVersionError("public-input codec id is incompatible")
    if pins.public_input_codec_version != PUBLIC_INPUT_CODEC_VERSION:
        raise ProgramZkpVersionError("public-input codec version is incompatible")

    # Proof material
    if isinstance(proof_bytes, str):
        stripped = proof_bytes.strip()
        hex_candidate = stripped[2:] if stripped.startswith(("0x", "0X")) else stripped
        try:
            proof_raw = (
                bytes.fromhex(hex_candidate)
                if hex_candidate and len(hex_candidate) % 2 == 0
                else proof_bytes.encode("utf-8")
            )
        except ValueError:
            proof_raw = proof_bytes.encode("utf-8")
    else:
        proof_raw = bytes(proof_bytes)
    if not proof_raw:
        raise ProgramZkpTamperError("corrupted proof rejected: empty")
    if proof_bytes_are_simulated(proof_raw):
        raise ProgramZkpAuthorityError(
            "simulated proof layouts cannot pass independent verification"
        )

    # Key material
    if isinstance(verifying_key_material, str):
        key_raw = verifying_key_material.encode("utf-8")
    else:
        key_raw = bytes(verifying_key_material)
    if not key_raw:
        raise ProgramZkpTamperError("corrupted verifying key rejected: empty")
    key_digest = "sha256:" + hashlib.sha256(key_raw).hexdigest()
    if expected_verifying_key_digest:
        expected = _text(
            expected_verifying_key_digest, field_name="expected_verifying_key_digest"
        )
        if key_digest != expected:
            raise ProgramZkpTamperError("corrupted verifying key rejected: digest mismatch")

    # Field encoding / codec vector
    try:
        field_vector = encode_public_input_field_vector(
            pins, field_encoding=capability.field_encoding
        )
    except ProgramZkpAuthorityError:
        raise
    if any(value == 0 for value in field_vector):
        # Defensive: honest SHA-256 reduction is overwhelmingly nonzero; zero
        # vectors indicate placeholder/all-zero forgery paths.
        raise ProgramZkpTamperError("corrupted public-input field vector rejected")

    # Capability gate: production authority requires a live eligible report.
    production = capability.production_eligible
    if capability.stale:
        production = False
    if capability.knowledge_graph_fail_open:
        production = False

    verified = False
    if cryptographic_verify is not None:
        try:
            verified = bool(cryptographic_verify(proof_raw, key_raw, field_vector))
        except Exception as exc:
            raise ProgramZkpCapabilityError(
                "independent cryptographic verifier failed: %s" % type(exc).__name__
            ) from exc
    elif envelope.backend_mode is ProgramZkpBackendMode.CRYPTOGRAPHIC and production:
        # Structural gates passed but no cryptographic callback was supplied —
        # fail closed rather than invent a success.
        verified = False
    elif envelope.backend_mode in {
        ProgramZkpBackendMode.SHADOW,
        ProgramZkpBackendMode.SIMULATED,
    }:
        # Shadow/simulated independent "verification" is structural only.
        verified = True

    verdict = (
        ProgramZkpVerdict.VERIFIED if verified else ProgramZkpVerdict.REJECTED
    )
    return ProgramZkpVerificationReceipt(
        statement=envelope.statement,
        verdict=verdict,
        verifier_id=verifier_id,
        verifying_key_id=pins.verifying_key_id,
        circuit_id=pins.circuit_id,
        public_input_digest=envelope.statement.public_input_digest,
        ceremony_id=pins.ceremony_id,
        public_input_codec_version=pins.public_input_codec_version,
        backend_mode=envelope.backend_mode,
        capability_epoch=capability.capability_epoch,
        capability_production_eligible=bool(production and verified),
        independent_verifier=True,
    )


def record_production_program_zkp_verification(
    envelope: ProgramZkpShadowEnvelope,
    *,
    capability: ProgramZkpCapabilityConformanceReport,
    verifier_id: str,
    proof_bytes: bytes | bytearray | memoryview | str,
    verifying_key_material: bytes | bytearray | memoryview | str,
    cryptographic_verify: Callable[[bytes, bytes, tuple[int, ...]], bool],
    expected_verifying_key_digest: str = "",
    cancellation_event: threading.Event | None = None,
) -> ProgramZkpVerificationReceipt:
    """Verify and record a production path; requires eligible capability."""

    capability.require_production_eligible()
    if envelope.backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC:
        raise ProgramZkpAuthorityError(
            "production verification requires cryptographic backend mode"
        )
    receipt = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id=verifier_id,
        proof_bytes=proof_bytes,
        verifying_key_material=verifying_key_material,
        expected_verifying_key_digest=expected_verifying_key_digest,
        cryptographic_verify=cryptographic_verify,
        cancellation_event=cancellation_event,
    )
    if not receipt.authoritative:
        raise ProgramZkpAuthorityError(
            "independent verification did not produce an authoritative receipt"
        )
    # No semantic claim promotion: still only zk_trace_attested.
    if receipt.claim_level is not ClaimLevel.ZK_TRACE_ATTESTED:
        raise ProgramZkpClaimPromotionError(
            "production receipt cannot promote claim level"
        )
    return receipt


def rollout_mode_for_capability(
    capability: ProgramZkpCapabilityConformanceReport,
) -> ProgramZkpRolloutMode:
    """Return the fail-closed rollout mode for a capability report."""

    return capability.rollout_mode


def shadow_only_rollout(
    capability: ProgramZkpCapabilityConformanceReport,
) -> bool:
    """Return True when ZK must remain shadow-only (no production authority)."""

    return capability.shadow_only


__all__ = [
    "PROGRAM_ANALYSIS_ZKP_CONTRACT_VERSION",
    "CONTRACT_VERSION",
    "PUBLIC_INPUT_CODEC_ID",
    "PUBLIC_INPUT_CODEC_VERSION",
    "PUBLIC_COMMITMENT_KEYS",
    "TRACE_VALIDITY_DOES_NOT_PROVE",
    "TRACE_VALIDITY_SCOPE_STATEMENT",
    "SUPPORTED_TRACE_TRANSITIONS",
    "PROGRAM_ZKP_PUBLIC_INPUTS_SCHEMA",
    "PROGRAM_ZKP_STATEMENT_SCHEMA",
    "PROGRAM_ZKP_TRACE_SCHEMA",
    "PROGRAM_ZKP_WITNESS_POLICY_SCHEMA",
    "PROGRAM_ZKP_RECEIPT_SCHEMA",
    "PROGRAM_ZKP_SHADOW_ENVELOPE_SCHEMA",
    "PROGRAM_ZKP_CAPABILITY_CONFORMANCE_SCHEMA",
    "PROGRAM_ZKP_CAPABILITY_CHECK_SCHEMA",
    "PROGRAM_ZKP_PROOF_SCHEMA_ID",
    "PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE",
    "PROGRAM_CONTRACT_TRACE_CIRCUIT_ID",
    "PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION",
    "PROGRAM_CONTRACT_TRACE_MAX_TRACE_STEPS",
    "PROGRAM_CONTRACT_TRACE_CANONICAL_TRACE_LENGTH",
    "BN254_SCALAR_FIELD_MODULUS",
    "FIELD_ENCODING_BN254_SHA256",
    "REQUIRED_CAPABILITY_DIMENSIONS",
    "ProgramAnalysisZkpError",
    "ProgramZkpWitnessDisclosureError",
    "ProgramZkpTraceError",
    "ProgramZkpTamperError",
    "ProgramZkpReplayError",
    "ProgramZkpVersionError",
    "ProgramZkpClaimPromotionError",
    "ProgramZkpCapabilityError",
    "ProgramZkpAuthorityError",
    "ProgramZkpBackendMode",
    "ProgramZkpTrust",
    "ProgramZkpVerdict",
    "ProgramZkpCapabilityDimension",
    "ProgramZkpCapabilityStatus",
    "ProgramZkpRolloutMode",
    "ProgramZkpFieldEncodingKind",
    "ProgramZkpCircuitFamily",
    "ProgramZkpAuthorityDenialReason",
    "TraceState",
    "TraceTransitionKind",
    "ProgramZkpPublicInputs",
    "ProgramZkpTraceStep",
    "ProgramZkpTrace",
    "ProgramZkpWitnessPolicy",
    "PrivateProgramAnalysisWitness",
    "ProgramZkpStatement",
    "ProgramZkpProvingRequest",
    "ProgramZkpShadowEnvelope",
    "ProgramZkpVerificationReceipt",
    "ProgramZkpCapabilityCheck",
    "ProgramZkpCapabilityConformanceReport",
    "commitment_identity",
    "encode_public_input_vector",
    "public_input_vector_digest",
    "field_element_from_text",
    "encode_public_input_field_vector",
    "classify_circuit_family",
    "is_versioned_artifact_id",
    "proof_bytes_are_simulated",
    "supported_transition_table",
    "next_trace_state",
    "canonical_trace_transition_kinds",
    "trace_validity_does_not_prove",
    "reject_illegal_zk_claim_promotion",
    "build_program_zkp_public_inputs",
    "build_canonical_program_zkp_trace",
    "reject_private_witness_from_public_payload",
    "public_program_zkp_artifact",
    "public_artifact_contains",
    "prepare_program_analysis_zkp",
    "create_program_zkp_shadow_envelope",
    "record_program_zkp_verification",
    "record_production_program_zkp_verification",
    "probe_program_analysis_zkp_capability",
    "build_production_ready_capability_fixture",
    "verify_program_zkp_independently",
    "grants_production_authority",
    "require_production_authority",
    "invalidate_authority_on_capability_loss",
    "rollout_mode_for_capability",
    "shadow_only_rollout",
    "assert_trace_non_claims",
    "claim_level_for_verified_trace",
    "verdict_for_trace_attestation",
    "inconclusive_state_for_shadow",
]
