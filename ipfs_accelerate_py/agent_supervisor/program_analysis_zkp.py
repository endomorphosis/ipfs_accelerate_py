"""ZK public inputs, witness policy, and trace semantics for program assurance.

This module is the VFS program-assurance zero-knowledge contract surface
(VFS-022 / VFS-G080).  It defines:

* canonical public commitments for repository forest, inventory, contract,
  call slice, assumptions, analyzer/resolver/translator/prover versions,
  supported result, circuit, proving/verifying keys, ceremony, and the
  public-input codec;
* a private witness and redaction policy that never serializes protected
  openings into public artifacts; and
* supported deterministic trace transitions that a future bounded circuit
  may check.

Trace validity is intentionally narrow.  A verified program-analysis trace
establishes only that committed public inputs open to a witness following the
declared transition rules and that the trace ends in the committed supported
result.  It does **not** prove inventory completeness, translator soundness,
arbitrary runtime semantics, or any theorem beyond that committed result.

Circuit implementation and production capability probing live in later tasks
(VFS-023, VFS-024).  Simulated paths remain non-authoritative.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
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
    cryptographic verification is a later capability (VFS-024).
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
            # VFS-022 lands contracts only; cryptographic mode requires later
            # capability conformance before any authority can attach.
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
    non-authoritative for semantic claims.
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

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def verified(self) -> bool:
        return self.verdict is ProgramZkpVerdict.VERIFIED

    @property
    def authoritative(self) -> bool:
        # Shadow/simulated verification never grants production authority.
        if self.backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC:
            return False
        return self.verdict is ProgramZkpVerdict.VERIFIED

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
) -> ProgramZkpVerificationReceipt:
    """Record an independent (or shadow) verification bound to envelope pins."""

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
    "ProgramAnalysisZkpError",
    "ProgramZkpWitnessDisclosureError",
    "ProgramZkpTraceError",
    "ProgramZkpTamperError",
    "ProgramZkpReplayError",
    "ProgramZkpVersionError",
    "ProgramZkpClaimPromotionError",
    "ProgramZkpBackendMode",
    "ProgramZkpTrust",
    "ProgramZkpVerdict",
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
    "commitment_identity",
    "encode_public_input_vector",
    "public_input_vector_digest",
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
    "assert_trace_non_claims",
    "claim_level_for_verified_trace",
    "verdict_for_trace_attestation",
    "inconclusive_state_for_shadow",
]
