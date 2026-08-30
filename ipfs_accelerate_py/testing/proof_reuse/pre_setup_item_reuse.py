"""PCTDD-029: narrowly gated pre-setup whole-item reuse.

Accelerate-owned pytest integration that may reuse setup, call, and teardown
together *before* current setup, and only for populations that are explicitly
``pure`` or replay-safe *and* teardown-compatible.  Post-setup/pre-call reuse
remains the primary path; this gate never widens identity, proof, storage,
execution, scheduler, or publication authority.

Rules:

* Admission is authoritative only before current setup.
* Whole-item reuse is never pytest skip, never a skip marker, and never a
  substitute for opaque, unreviewed, incomplete, or teardown-incompatible
  populations.
* Signature, locator, execution-key, phase, or population disagreement force
  normal full execution of setup, call, and teardown.
* Fixture-proof-aware xdist, aggregate selected-test ZK, production ZK, key
  ceremony, and direct-execution profiles remain typed unavailable.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .signed_runner_attestations import (
    SignedRunnerAttestationResult,
    get_attached_signed_runner_result,
)

# This module is accelerate integration, not a pytest test module.
__test__ = False

PRE_SETUP_ITEM_REUSE_INTERFACE: Final = "PreSetupItemReuse@1"
PRE_SETUP_ITEM_REUSE_RESULT_INTERFACE: Final = "PreSetupItemReuseResult@1"
ADMITTED_ITEM_CERTIFICATE_INTERFACE: Final = "AdmittedItemCertificate@1"
PRE_SETUP_REUSE_POPULATION_INTERFACE: Final = "PreSetupReusePopulation@1"
PRE_SETUP_ITEM_REUSE_POLICY_INTERFACE: Final = "PreSetupItemReusePolicy@1"
PRE_SETUP_ITEM_REUSE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/pre-setup-item-reuse@1"
)
CLAIM_CLASS: Final = "IntegrityCommitment"
REUSED_PHASES: Final[tuple[str, ...]] = ("setup", "call", "teardown")
RUN_ACTION: Final = "RUN"
REUSE_ITEM_ACTION: Final = "REUSE_ITEM"
SCHEMA_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.pre_setup_item_reuse"
)
PREDECESSOR_ASSEMBLY_INTERFACE: Final = "SetupBoundExecutionKeyAssembly@1"
PREDECESSOR_ATTESTATION_INTERFACE: Final = "SignedRunnerAttestationBinding@1"
ITEM_ITEM_CERTIFICATE_ATTRIBUTE: Final = "_ipfs_proof_reuse_item_certificate"
ITEM_ITEM_CERTIFICATE_LOOKUP_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_item_certificate_lookup"
)
ITEM_POPULATION_ATTRIBUTE: Final = "_ipfs_proof_reuse_pre_setup_population"
ITEM_ITEM_REUSE_RESULT_ATTRIBUTE: Final = "_ipfs_proof_reuse_item_reuse_result"
ITEM_ITEM_REUSED_ATTRIBUTE: Final = "_ipfs_proof_reuse_item_reused"
ITEM_PHASE_PROBE_ATTRIBUTE: Final = "_ipfs_proof_reuse_item_phase_probe"
ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_composite_phase_receipt"
)
ITEM_LOCATOR_ATTRIBUTE: Final = "_ipfs_proof_reuse_locator"
ITEM_EXECUTION_KEY_ATTRIBUTE: Final = "_ipfs_proof_reuse_execution_key"
MAX_TEXT_CHARS: Final = 4_096
_DIGEST_PREFIX: Final = "sha256:"
_PHASE_PASS: Final = "pass"
PURE_REUSE_CLASS: Final = "pure"
EXPLICITLY_PURE_KIND: Final = "explicitly_pure"
REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND: Final = "replay_safe_teardown_compatible"

CLOSED_REUSE_CLASSES: Final[tuple[str, ...]] = (
    "pure",
    "deterministic_snapshot",
    "transactional",
    "idempotent_external",
    "effectful_replayable",
    "effectful_nonreplayable",
    "opaque",
)
REPLAY_SAFE_REUSE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "pure",
        "deterministic_snapshot",
        "transactional",
        "idempotent_external",
        "effectful_replayable",
    }
)
INELIGIBLE_REUSE_CLASSES: Final[frozenset[str]] = frozenset(
    {"opaque", "effectful_nonreplayable"}
)

REUSE_ESTABLISHES: Final = (
    "whole-item reuse is limited to explicitly pure or replay-safe "
    "teardown-compatible populations"
)
REUSE_DOES_NOT: Final = (
    "execution or semantics; skip; current-root publication; task "
    "completion; production ZK; opaque whole-item reuse; self-approval"
)

_PRIVATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private",
    "proving_key",
    "secret",
    "session",
    "signing_key",
    "token",
    "witness",
)

_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "fixture_proof_aware_xdist",
        "fixture_proof_aware_xdist_not_implemented",
        "fixture affinity and proof-cost placement remain a later xdist "
        "scheduling task; pre-setup whole-item reuse does not omit tests or "
        "merge pools",
    ),
    (
        "aggregate_selected_test_zk",
        "aggregate_selected_test_zk_missing",
        "aggregate selected-test ZK remains a versioned successor; whole-item "
        "reuse cannot upgrade leaf TestPassStatementV1 claims",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; pre-setup whole-item "
        "reuse cannot admit simulated, structural, or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by narrowly gated "
        "pre-setup whole-item reuse",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade whole-item reuse integrity commitments",
    ),
)


class PreSetupItemReuseError(ValueError):
    """Raised when narrowly gated pre-setup whole-item reuse is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise PreSetupItemReuseError(
            "floating-point values are not JSON-safe for pre-setup item reuse"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise PreSetupItemReuseError(
                    f"pre-setup item reuse rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise PreSetupItemReuseError(
            "pre-setup item reuse rejects secret or raw bytes"
        )
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise PreSetupItemReuseError(
        f"value of type {type(value).__name__} is not JSON-serializable"
    )


def canonical_public_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for a privacy-safe public payload."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def public_digest(value: Any) -> str:
    """Return ``sha256:<hex>`` of canonical public bytes."""

    return _DIGEST_PREFIX + hashlib.sha256(canonical_public_bytes(value)).hexdigest()


def classify_reuse_class(value: Any) -> str:
    """Map missing or unknown reuse labels onto the closed default ``opaque``."""

    if value is None:
        return "opaque"
    if not isinstance(value, str):
        raise PreSetupItemReuseError("reuse_class must be a string or omitted")
    text = value.strip()
    if not text or text not in CLOSED_REUSE_CLASSES:
        return "opaque"
    return text


REUSE_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": PRE_SETUP_ITEM_REUSE_POLICY_INTERFACE,
        "reuse_interface": PRE_SETUP_ITEM_REUSE_INTERFACE,
        "predecessor_assembly_interface": PREDECESSOR_ASSEMBLY_INTERFACE,
        "predecessor_attestation_interface": PREDECESSOR_ATTESTATION_INTERFACE,
        "reused_phases": list(REUSED_PHASES),
        "eligible_population_kinds": [
            EXPLICITLY_PURE_KIND,
            REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND,
        ],
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "normal_execution_fallback": True,
        "schema_authority": SCHEMA_AUTHORITY,
    }
)
DEFAULT_POLICY_CID: Final = public_digest(dict(REUSE_POLICY))


def _load_datasets_v2() -> Any | None:
    try:
        from ipfs_datasets_py.logic.zkp.pctdd import test_execution_key_v2 as contracts
    except Exception:
        return None
    return contracts


def _load_datasets_composite() -> Any | None:
    try:
        from ipfs_datasets_py.logic.zkp.pctdd import (
            composite_phase_receipt_contracts as contracts,
        )
    except Exception:
        return None
    return contracts


def datasets_contracts_available() -> bool:
    """Return whether datasets-owned V2 and composite codecs import."""

    return _load_datasets_v2() is not None and _load_datasets_composite() is not None


def record_typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    """Record a typed unavailable case without changing claim meaning."""

    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise PreSetupItemReuseError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-029 typed unavailable capabilities."""

    records = [
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    ]
    if not datasets_contracts_available():
        records.append(
            record_typed_unavailable(
                capability="datasets_v2_or_composite_contracts",
                reason_code="datasets_v2_or_composite_unresolved",
                message=(
                    "datasets TestExecutionKeyV2 or CompositePhaseReceipt "
                    "codecs did not import in this sealed environment; "
                    "pre-setup whole-item reuse forces full execution"
                ),
            )
        )
    return tuple(records)


def authority_descriptor() -> dict[str, Any]:
    """Return the reuse authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "reuse": SCHEMA_AUTHORITY,
        "schema_authority": SCHEMA_AUTHORITY,
        "does_not": REUSE_DOES_NOT,
        "establishes": REUSE_ESTABLISHES,
        "claim_class": CLAIM_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "reused_phases": list(REUSED_PHASES),
        "eligible_population_kinds": [
            EXPLICITLY_PURE_KIND,
            REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND,
        ],
        "normal_execution_fallback": True,
        "reuse_interface": PRE_SETUP_ITEM_REUSE_INTERFACE,
        "certificate_interface": ADMITTED_ITEM_CERTIFICATE_INTERFACE,
        "population_interface": PRE_SETUP_REUSE_POPULATION_INTERFACE,
        "predecessor_assembly_interface": PREDECESSOR_ASSEMBLY_INTERFACE,
        "predecessor_attestation_interface": PREDECESSOR_ATTESTATION_INTERFACE,
        "test_execution_key_v2": "TestExecutionKeyV2",
        "composite_phase_receipt": "CompositePhaseReceipt@1",
    }


def _require_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PreSetupItemReuseError(f"{field_name} must be text")
    text = value.strip()
    if text != value or len(text) > MAX_TEXT_CHARS:
        raise PreSetupItemReuseError(f"invalid {field_name}")
    if not text and not allow_empty:
        raise PreSetupItemReuseError(f"{field_name} is required")
    return text


def _public_attr(value: Any, *names: str) -> str:
    if value is None:
        return ""
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:MAX_TEXT_CHARS]
        enum_value = getattr(candidate, "value", None)
        if isinstance(enum_value, str) and enum_value.strip():
            return enum_value.strip()[:MAX_TEXT_CHARS]
    to_dict = getattr(value, "to_dict", None)
    payload: Any = value
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = value
    if isinstance(payload, Mapping):
        for name in names:
            candidate = payload.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:MAX_TEXT_CHARS]
            enum_value = getattr(candidate, "value", None)
            if isinstance(enum_value, str) and enum_value.strip():
                return enum_value.strip()[:MAX_TEXT_CHARS]
    return ""


def _phase_text(value: Any, field_name: str = "phase") -> str:
    if value is None:
        return ""
    text = getattr(value, "value", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(value, str):
        return value.strip()
    raise PreSetupItemReuseError(f"{field_name} must be text")


def _forbid_skip_production(
    *,
    may_authorize_skip: bool,
    production_admitted: bool,
    self_approved: bool,
    claim_unchanged: bool,
) -> None:
    if may_authorize_skip:
        raise PreSetupItemReuseError(
            "narrowly gated whole-item reuse must not authorize skip"
        )
    if production_admitted or self_approved or not claim_unchanged:
        raise PreSetupItemReuseError(
            "narrowly gated whole-item reuse cannot admit production, "
            "self-approve, or change claims"
        )


def _normalize_phases(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        phases = tuple(str(item).strip() for item in value if str(item).strip())
        return phases
    raise PreSetupItemReuseError("reused_phases must be text or a sequence of text")


def _is_whole_item_phases(phases: Sequence[str]) -> bool:
    return tuple(sorted(phases)) == tuple(sorted(REUSED_PHASES))


@dataclass
class PhaseExecutionProbe:
    """Mutable setup/call/teardown counters.  Not a pytest test class."""

    __test__ = False

    setup: int = 0
    call: int = 0
    teardown: int = 0

    def record_setup(self) -> None:
        self.setup += 1

    def record_call(self) -> None:
        self.call += 1

    def record_teardown(self) -> None:
        self.teardown += 1

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.setup, self.call, self.teardown)

    def to_dict(self) -> dict[str, int]:
        return {"setup": self.setup, "call": self.call, "teardown": self.teardown}


def attach_phase_probe(item: Any, probe: PhaseExecutionProbe | None = None) -> PhaseExecutionProbe:
    """Attach a phase-execution probe to *item*."""

    attached = probe if probe is not None else PhaseExecutionProbe()
    if not isinstance(attached, PhaseExecutionProbe):
        raise PreSetupItemReuseError("phase probe must be a PhaseExecutionProbe")
    try:
        setattr(item, ITEM_PHASE_PROBE_ATTRIBUTE, attached)
    except Exception as exc:
        raise PreSetupItemReuseError("unable to attach phase probe") from exc
    return attached


def get_phase_probe(item: Any) -> PhaseExecutionProbe | None:
    """Return the attached phase-execution probe, if any."""

    existing = getattr(item, ITEM_PHASE_PROBE_ATTRIBUTE, None)
    if isinstance(existing, PhaseExecutionProbe):
        return existing
    return None


@dataclass(frozen=True, slots=True)
class PreSetupReusePopulation:
    """Closed population gate for pre-setup whole-item reuse.

    Not a pytest test class.  Fail-closed: non-pure replay-safe classes require
    an explicit teardown-compatible declaration.
    """

    __test__ = False

    reuse_class: str
    teardown_compatible: bool = False
    teardown_compatible_explicit: bool = False
    reviewed: bool = False
    completeness: str = "unknown"
    reason: str = ""

    def __post_init__(self) -> None:
        classified = classify_reuse_class(self.reuse_class)
        object.__setattr__(self, "reuse_class", classified)
        completeness = str(self.completeness or "unknown").strip() or "unknown"
        object.__setattr__(self, "completeness", completeness[:32])
        object.__setattr__(self, "reason", str(self.reason or "")[:MAX_TEXT_CHARS])

    @property
    def interface(self) -> str:
        return PRE_SETUP_REUSE_POPULATION_INTERFACE

    @property
    def schema(self) -> str:
        return PRE_SETUP_ITEM_REUSE_SCHEMA

    @property
    def replay_safe(self) -> bool:
        return self.reuse_class in REPLAY_SAFE_REUSE_CLASSES

    @property
    def explicitly_pure(self) -> bool:
        return self.reuse_class == PURE_REUSE_CLASS

    @property
    def population_kind(self) -> str:
        if self.eligible and self.explicitly_pure:
            return EXPLICITLY_PURE_KIND
        if self.eligible:
            return REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND
        return ""

    @property
    def eligible(self) -> bool:
        return self.eligibility_reason == "eligible"

    @property
    def eligibility_reason(self) -> str:
        if self.reason:
            return self.reason
        if self.reuse_class in INELIGIBLE_REUSE_CLASSES:
            return f"reuse_class_{self.reuse_class}"
        if self.reuse_class not in REPLAY_SAFE_REUSE_CLASSES:
            return "reuse_class_unknown"
        if not self.reviewed:
            return "population_unreviewed"
        if self.completeness != "exact":
            return "population_incomplete"
        if self.explicitly_pure:
            if self.teardown_compatible_explicit and not self.teardown_compatible:
                return "pure_population_teardown_incompatible"
            return "eligible"
        if not self.teardown_compatible_explicit:
            return "teardown_compatible_not_explicit"
        if not self.teardown_compatible:
            return "teardown_not_compatible"
        return "eligible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "reuse_class": self.reuse_class,
            "teardown_compatible": self.teardown_compatible
            if self.explicitly_pure or self.teardown_compatible_explicit
            else False,
            "teardown_compatible_explicit": self.teardown_compatible_explicit,
            "reviewed": self.reviewed,
            "completeness": self.completeness,
            "replay_safe": self.replay_safe,
            "explicitly_pure": self.explicitly_pure,
            "eligible": self.eligible,
            "population_kind": self.population_kind,
            "reason": self.eligibility_reason if not self.eligible else "eligible",
        }


def classify_population(
    *,
    reuse_class: Any = None,
    teardown_compatible: bool | None = None,
    reviewed: bool = False,
    completeness: str = "unknown",
) -> PreSetupReusePopulation:
    """Classify a population.  Teardown compatibility is fail-closed."""

    explicit = teardown_compatible is not None
    classified = classify_reuse_class(reuse_class)
    if classified == PURE_REUSE_CLASS and not explicit:
        compatible = True
        explicit_flag = False
    else:
        compatible = bool(teardown_compatible) if explicit else False
        explicit_flag = explicit
    return PreSetupReusePopulation(
        reuse_class=classified,
        teardown_compatible=compatible,
        teardown_compatible_explicit=explicit_flag,
        reviewed=bool(reviewed),
        completeness=str(completeness or "unknown"),
    )


def population_from_item(item: Any) -> PreSetupReusePopulation | None:
    """Return an attached or reconstructed population, if any."""

    if item is None:
        return None
    attached = getattr(item, ITEM_POPULATION_ATTRIBUTE, None)
    if isinstance(attached, PreSetupReusePopulation):
        return attached
    if isinstance(attached, Mapping):
        try:
            return classify_population(
                reuse_class=attached.get("reuse_class"),
                teardown_compatible=attached.get("teardown_compatible"),
                reviewed=bool(attached.get("reviewed", False)),
                completeness=str(attached.get("completeness") or "unknown"),
            )
        except PreSetupItemReuseError:
            return None
    fixture = None
    for owner_name in (
        ITEM_EXECUTION_KEY_ATTRIBUTE,
        "_ipfs_proof_reuse_setup_bound_execution_key_v2",
    ):
        owner = getattr(item, owner_name, None)
        fixture = getattr(owner, "fixture", None)
        if fixture is not None:
            break
    if fixture is None:
        return None
    reuse_class = _public_attr(fixture, "reuse_class") or "opaque"
    completeness = _public_attr(fixture, "completeness") or "unknown"
    reviewed = bool(getattr(fixture, "reviewed", False))
    teardown_compatible = getattr(item, "_ipfs_proof_reuse_teardown_compatible", None)
    if teardown_compatible is None:
        teardown_compatible = getattr(fixture, "teardown_compatible", None)
    return classify_population(
        reuse_class=reuse_class,
        teardown_compatible=teardown_compatible if isinstance(teardown_compatible, bool) else None,
        reviewed=reviewed,
        completeness=completeness,
    )


def attach_population(item: Any, population: PreSetupReusePopulation) -> None:
    """Attach a public population gate without skip authority."""

    if item is None:
        return
    if not isinstance(population, PreSetupReusePopulation):
        raise PreSetupItemReuseError("population must be PreSetupReusePopulation")
    try:
        setattr(item, ITEM_POPULATION_ATTRIBUTE, population)
    except Exception:
        return


@dataclass(frozen=True, slots=True)
class AdmittedItemCertificate:
    """Public pins for one admitted whole-item reuse certificate.

    Setup, call, and teardown are reused together only when the population
    gate admits it.  Not a pytest test class.
    """

    __test__ = False

    execution_key_cid: str
    certificate_cid: str
    composite_phase_receipt_cid: str = ""
    locator_cid: str = ""
    setup_outcome: str = _PHASE_PASS
    call_outcome: str = _PHASE_PASS
    teardown_outcome: str = _PHASE_PASS
    signature_verified: bool = False
    admitted: bool = False
    reused_phases: tuple[str, ...] = REUSED_PHASES
    reuse_class: str = "opaque"
    teardown_compatible: bool = False
    population_kind: str = ""
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _forbid_skip_production(
            may_authorize_skip=self.may_authorize_skip,
            production_admitted=self.production_admitted,
            self_approved=self.self_approved,
            claim_unchanged=self.claim_unchanged,
        )
        object.__setattr__(
            self,
            "execution_key_cid",
            _require_text(self.execution_key_cid, "execution_key_cid", allow_empty=True),
        )
        object.__setattr__(
            self,
            "certificate_cid",
            _require_text(self.certificate_cid, "certificate_cid", allow_empty=True),
        )
        object.__setattr__(
            self,
            "composite_phase_receipt_cid",
            _require_text(
                self.composite_phase_receipt_cid,
                "composite_phase_receipt_cid",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "locator_cid",
            _require_text(self.locator_cid, "locator_cid", allow_empty=True),
        )
        object.__setattr__(
            self, "setup_outcome", _phase_text(self.setup_outcome, "setup_outcome")
        )
        object.__setattr__(
            self, "call_outcome", _phase_text(self.call_outcome, "call_outcome")
        )
        object.__setattr__(
            self,
            "teardown_outcome",
            _phase_text(self.teardown_outcome, "teardown_outcome"),
        )
        phases = _normalize_phases(self.reused_phases) or REUSED_PHASES
        object.__setattr__(self, "reused_phases", phases)
        object.__setattr__(self, "reuse_class", classify_reuse_class(self.reuse_class))
        object.__setattr__(
            self, "population_kind", str(self.population_kind or "")[:MAX_TEXT_CHARS]
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.admitted:
            if not self.signature_verified:
                raise PreSetupItemReuseError(
                    "admitted item certificate requires a verified signature"
                )
            if not self.execution_key_cid or not self.certificate_cid:
                raise PreSetupItemReuseError(
                    "admitted item certificate requires execution-key and certificate cids"
                )
            if (
                self.setup_outcome != _PHASE_PASS
                or self.call_outcome != _PHASE_PASS
                or self.teardown_outcome != _PHASE_PASS
            ):
                raise PreSetupItemReuseError(
                    "admitted item certificate requires an honest complete pass"
                )
            if not _is_whole_item_phases(self.reused_phases):
                raise PreSetupItemReuseError(
                    "admitted item certificate reuses the whole item"
                )
            if self.population_kind not in {
                EXPLICITLY_PURE_KIND,
                REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND,
            }:
                raise PreSetupItemReuseError(
                    "admitted item certificate requires an eligible population"
                )

    @property
    def interface(self) -> str:
        return ADMITTED_ITEM_CERTIFICATE_INTERFACE

    @property
    def schema(self) -> str:
        return PRE_SETUP_ITEM_REUSE_SCHEMA

    @property
    def reuses_item(self) -> bool:
        return self.admitted and _is_whole_item_phases(self.reused_phases)

    @property
    def reuses_setup(self) -> bool:
        return self.reuses_item

    @property
    def reuses_call(self) -> bool:
        return self.reuses_item

    @property
    def reuses_teardown(self) -> bool:
        return self.reuses_item

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "execution_key_cid": self.execution_key_cid,
            "certificate_cid": self.certificate_cid,
            "composite_phase_receipt_cid": self.composite_phase_receipt_cid,
            "locator_cid": self.locator_cid,
            "setup_outcome": self.setup_outcome,
            "call_outcome": self.call_outcome,
            "teardown_outcome": self.teardown_outcome,
            "signature_verified": self.signature_verified,
            "admitted": self.admitted,
            "reused_phases": list(self.reused_phases),
            "reuses_item": self.reuses_item,
            "reuses_setup": self.reuses_setup,
            "reuses_call": self.reuses_call,
            "reuses_teardown": self.reuses_teardown,
            "reuse_class": self.reuse_class,
            "teardown_compatible": self.teardown_compatible,
            "population_kind": self.population_kind,
            "may_authorize_skip": False,
            "production_admitted": False,
            "self_approved": False,
            "claim_unchanged": True,
            "diagnostics": dict(self.diagnostics),
        }


def _unadmitted_certificate(
    *,
    execution_key_cid: str = "",
    certificate_cid: str = "",
    composite_phase_receipt_cid: str = "",
    locator_cid: str = "",
    setup_outcome: str = "",
    call_outcome: str = "",
    teardown_outcome: str = "",
    signature_verified: bool = False,
    reuse_class: str = "opaque",
    teardown_compatible: bool = False,
    population_kind: str = "",
    reason: str,
) -> AdmittedItemCertificate:
    return AdmittedItemCertificate(
        execution_key_cid=execution_key_cid,
        certificate_cid=certificate_cid,
        composite_phase_receipt_cid=composite_phase_receipt_cid,
        locator_cid=locator_cid,
        setup_outcome=setup_outcome or _PHASE_PASS,
        call_outcome=call_outcome or _PHASE_PASS,
        teardown_outcome=teardown_outcome or _PHASE_PASS,
        signature_verified=signature_verified,
        admitted=False,
        reused_phases=REUSED_PHASES,
        reuse_class=reuse_class,
        teardown_compatible=teardown_compatible,
        population_kind=population_kind,
        diagnostics={"reason": reason},
    )


def admit_item_certificate(
    *,
    execution_key: Any = None,
    composite_receipt: Any = None,
    attestation_result: SignedRunnerAttestationResult | None = None,
    certificate_cid: str = "",
    population: PreSetupReusePopulation | None = None,
    reused_phases: Sequence[str] = REUSED_PHASES,
) -> AdmittedItemCertificate:
    """Admit a whole-item certificate or return an unadmitted record."""

    phases = _normalize_phases(reused_phases)
    if phases and not _is_whole_item_phases(phases):
        return _unadmitted_certificate(reason="reuse_phase_not_whole_item")
    resolved_population = population
    if resolved_population is None:
        reuse_class = _public_attr(getattr(execution_key, "fixture", None), "reuse_class")
        completeness = _public_attr(
            getattr(execution_key, "fixture", None), "completeness"
        ) or "unknown"
        reviewed = bool(getattr(getattr(execution_key, "fixture", None), "reviewed", False))
        resolved_population = classify_population(
            reuse_class=reuse_class or "opaque",
            reviewed=reviewed,
            completeness=completeness,
        )
    if not resolved_population.eligible:
        return _unadmitted_certificate(
            reuse_class=resolved_population.reuse_class,
            teardown_compatible=resolved_population.teardown_compatible,
            reason=resolved_population.eligibility_reason,
        )
    execution_key_cid = _public_attr(execution_key, "execution_key_cid", "content_id")
    locator_cid = _public_attr(execution_key, "locator_cid")
    composite_cid = _public_attr(
        composite_receipt, "receipt_cid", "composite_phase_receipt_cid"
    )
    setup_outcome = _phase_text(
        getattr(composite_receipt, "setup_outcome", _PHASE_PASS), "setup_outcome"
    )
    call_outcome = _phase_text(
        getattr(composite_receipt, "call_outcome", _PHASE_PASS), "call_outcome"
    )
    teardown_outcome = _phase_text(
        getattr(composite_receipt, "teardown_outcome", _PHASE_PASS),
        "teardown_outcome",
    )
    signature_verified = False
    attested_cid = certificate_cid
    if attestation_result is not None:
        if not isinstance(attestation_result, SignedRunnerAttestationResult):
            return _unadmitted_certificate(reason="attestation_result_invalid")
        if (
            not attestation_result.valid
            or attestation_result.may_authorize_skip
            or attestation_result.production_admitted
            or attestation_result.self_approved
        ):
            return _unadmitted_certificate(
                execution_key_cid=execution_key_cid,
                locator_cid=locator_cid,
                composite_phase_receipt_cid=composite_cid,
                setup_outcome=setup_outcome,
                call_outcome=call_outcome,
                teardown_outcome=teardown_outcome,
                reuse_class=resolved_population.reuse_class,
                teardown_compatible=resolved_population.teardown_compatible,
                reason=attestation_result.reason or "attestation_not_verified",
            )
        signature_verified = True
        binding = attestation_result.binding
        if binding is not None:
            if execution_key_cid and binding.test_execution_key_v2_cid != execution_key_cid:
                return _unadmitted_certificate(
                    execution_key_cid=execution_key_cid,
                    reason="execution_key_mismatch",
                )
            execution_key_cid = binding.test_execution_key_v2_cid
            composite_cid = composite_cid or binding.composite_phase_receipt_cid
            locator_cid = locator_cid or binding.locator_cid
        attestation = attestation_result.attestation
        if attestation is not None:
            attested_cid = attested_cid or _public_attr(attestation, "cid")
        pass_receipt = attestation_result.pass_receipt
        if pass_receipt is not None:
            setup_outcome = _phase_text(pass_receipt.setup_outcome, "setup_outcome")
            call_outcome = _phase_text(pass_receipt.call_outcome, "call_outcome")
            teardown_outcome = _phase_text(
                pass_receipt.teardown_outcome, "teardown_outcome"
            )
            locator_cid = locator_cid or _public_attr(pass_receipt, "locator_cid")
    composite_admitted = True
    if composite_receipt is not None:
        composite_admitted = bool(getattr(composite_receipt, "admitted", False))
        if getattr(composite_receipt, "may_authorize_skip", False):
            return _unadmitted_certificate(reason="composite_authorizes_skip")
        composite_key = _public_attr(composite_receipt, "execution_key_cid")
        if composite_key and execution_key_cid and composite_key != execution_key_cid:
            return _unadmitted_certificate(reason="execution_key_mismatch")
    if (
        signature_verified
        and composite_admitted
        and setup_outcome == _PHASE_PASS
        and call_outcome == _PHASE_PASS
        and teardown_outcome == _PHASE_PASS
        and execution_key_cid
        and attested_cid
        and resolved_population.eligible
    ):
        return AdmittedItemCertificate(
            execution_key_cid=execution_key_cid,
            certificate_cid=attested_cid,
            composite_phase_receipt_cid=composite_cid,
            locator_cid=locator_cid,
            setup_outcome=setup_outcome,
            call_outcome=call_outcome,
            teardown_outcome=teardown_outcome,
            signature_verified=True,
            admitted=True,
            reused_phases=REUSED_PHASES,
            reuse_class=resolved_population.reuse_class,
            teardown_compatible=True,
            population_kind=resolved_population.population_kind,
        )
    reason = "item_certificate_not_admitted"
    if not signature_verified:
        reason = "signature_not_verified"
    elif setup_outcome != _PHASE_PASS or call_outcome != _PHASE_PASS or teardown_outcome != _PHASE_PASS:
        reason = "item_phases_not_pass"
    elif not composite_admitted:
        reason = "composite_not_admitted"
    elif not execution_key_cid:
        reason = "execution_key_unbound"
    elif not attested_cid:
        reason = "certificate_cid_missing"
    return _unadmitted_certificate(
        execution_key_cid=execution_key_cid,
        certificate_cid=attested_cid,
        composite_phase_receipt_cid=composite_cid,
        locator_cid=locator_cid,
        setup_outcome=setup_outcome,
        call_outcome=call_outcome,
        teardown_outcome=teardown_outcome,
        signature_verified=signature_verified,
        reuse_class=resolved_population.reuse_class,
        teardown_compatible=resolved_population.teardown_compatible,
        population_kind=resolved_population.population_kind,
        reason=reason,
    )


def attach_item_certificate(item: Any, certificate: AdmittedItemCertificate) -> None:
    """Attach a public whole-item certificate without skip authority."""

    if item is None:
        return
    if not isinstance(certificate, AdmittedItemCertificate):
        raise PreSetupItemReuseError(
            "item certificate must be AdmittedItemCertificate"
        )
    if certificate.may_authorize_skip:
        raise PreSetupItemReuseError("attached item certificate cannot skip")
    try:
        setattr(item, ITEM_ITEM_CERTIFICATE_ATTRIBUTE, certificate)
    except Exception:
        return


def item_certificate_from_item(item: Any) -> AdmittedItemCertificate | None:
    """Return an attached or reconstructed whole-item certificate, if any."""

    if item is None:
        return None
    attached = getattr(item, ITEM_ITEM_CERTIFICATE_ATTRIBUTE, None)
    if isinstance(attached, AdmittedItemCertificate):
        return attached
    lookup = getattr(item, ITEM_ITEM_CERTIFICATE_LOOKUP_ATTRIBUTE, None)
    if callable(lookup):
        try:
            found = lookup(item)
        except Exception:
            found = None
        if isinstance(found, AdmittedItemCertificate):
            return found
    signed = get_attached_signed_runner_result(item)
    composite = getattr(item, ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE, None)
    key = getattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE, None)
    population = population_from_item(item)
    if signed is None and composite is None:
        return None
    try:
        return admit_item_certificate(
            execution_key=key,
            composite_receipt=composite,
            attestation_result=signed,
            population=population,
        )
    except PreSetupItemReuseError:
        return None


@dataclass(frozen=True, slots=True)
class PreSetupItemReuseResult:
    """Outcome of one narrowly gated pre-setup whole-item reuse attempt.

    ``reuses_item`` never implies pytest skip, production admission, or task
    completion.  Not a pytest test class.
    """

    __test__ = False

    reuses_item: bool
    reason: str
    certificate: AdmittedItemCertificate | None = None
    population: PreSetupReusePopulation | None = None
    execution_key_cid: str = ""
    lifecycle_phase: str = "pre_setup"
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    full_execution_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _forbid_skip_production(
            may_authorize_skip=self.may_authorize_skip,
            production_admitted=self.production_admitted,
            self_approved=self.self_approved,
            claim_unchanged=self.claim_unchanged,
        )
        object.__setattr__(self, "reason", str(self.reason or "")[:MAX_TEXT_CHARS])
        object.__setattr__(
            self,
            "execution_key_cid",
            _require_text(self.execution_key_cid, "execution_key_cid", allow_empty=True),
        )
        object.__setattr__(
            self, "full_execution_reasons", tuple(self.full_execution_reasons)
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.reuses_item:
            if self.certificate is None or not self.certificate.admitted:
                raise PreSetupItemReuseError(
                    "whole-item reuse requires an admitted item certificate"
                )
            if self.population is None or not self.population.eligible:
                raise PreSetupItemReuseError(
                    "whole-item reuse requires an eligible population"
                )
            if self.action != REUSE_ITEM_ACTION:
                raise PreSetupItemReuseError("admitted reuse action must be REUSE_ITEM")

    @property
    def interface(self) -> str:
        return PRE_SETUP_ITEM_REUSE_RESULT_INTERFACE

    @property
    def schema(self) -> str:
        return PRE_SETUP_ITEM_REUSE_SCHEMA

    @property
    def reuse_interface(self) -> str:
        return PRE_SETUP_ITEM_REUSE_INTERFACE

    @property
    def reused_phases(self) -> tuple[str, ...]:
        return REUSED_PHASES if self.reuses_item else ()

    @property
    def reuses_setup(self) -> bool:
        return self.reuses_item

    @property
    def reuses_call(self) -> bool:
        return self.reuses_item

    @property
    def reuses_teardown(self) -> bool:
        return self.reuses_item

    @property
    def action(self) -> str:
        return REUSE_ITEM_ACTION if self.reuses_item else RUN_ACTION

    @property
    def normal_execution_fallback(self) -> bool:
        return True

    @property
    def requires_full_execution(self) -> bool:
        return not self.reuses_item

    @property
    def claim_class(self) -> str:
        return CLAIM_CLASS

    @property
    def population_kind(self) -> str:
        if self.population is None:
            return ""
        return self.population.population_kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "reuse_interface": self.reuse_interface,
            "reuses_item": self.reuses_item,
            "reuses_setup": self.reuses_setup,
            "reuses_call": self.reuses_call,
            "reuses_teardown": self.reuses_teardown,
            "reused_phases": list(self.reused_phases),
            "action": self.action,
            "reason": self.reason,
            "execution_key_cid": self.execution_key_cid,
            "lifecycle_phase": self.lifecycle_phase,
            "population_kind": self.population_kind,
            "may_authorize_skip": False,
            "production_admitted": False,
            "self_approved": False,
            "claim_unchanged": True,
            "claim_class": self.claim_class,
            "normal_execution_fallback": True,
            "requires_full_execution": self.requires_full_execution,
            "full_execution_reasons": list(self.full_execution_reasons),
            "certificate": None if self.certificate is None else self.certificate.to_dict(),
            "population": None if self.population is None else self.population.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _run_result(
    *,
    reason: str,
    execution_key_cid: str = "",
    certificate: AdmittedItemCertificate | None = None,
    population: PreSetupReusePopulation | None = None,
    extra_reasons: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> PreSetupItemReuseResult:
    reasons = tuple(
        dict.fromkeys(
            [reason, *extra_reasons, "normal_execution_fallback"]
            if reason
            else list(extra_reasons) or ["normal_execution_fallback"]
        )
    )
    return PreSetupItemReuseResult(
        reuses_item=False,
        reason=reason,
        certificate=certificate,
        population=population,
        execution_key_cid=execution_key_cid,
        full_execution_reasons=reasons,
        diagnostics=dict(diagnostics or {}),
    )


def attach_item_reuse_result(item: Any, result: PreSetupItemReuseResult) -> None:
    """Attach a reuse decision without skip authority."""

    if item is None:
        return
    if not isinstance(result, PreSetupItemReuseResult):
        raise PreSetupItemReuseError("reuse result must be PreSetupItemReuseResult")
    try:
        setattr(item, ITEM_ITEM_REUSE_RESULT_ATTRIBUTE, result)
        if result.reuses_item:
            setattr(item, ITEM_ITEM_REUSED_ATTRIBUTE, True)
    except Exception:
        return


def get_attached_item_reuse_result(item: Any) -> PreSetupItemReuseResult | None:
    """Return the attached reuse result, if any."""

    existing = getattr(item, ITEM_ITEM_REUSE_RESULT_ATTRIBUTE, None)
    if isinstance(existing, PreSetupItemReuseResult):
        return existing
    return None


def item_reuses_whole_item(item: Any) -> bool:
    """Return whether *item* was admitted for whole-item reuse."""

    result = get_attached_item_reuse_result(item)
    return result is not None and result.reuses_item


def _locator_cid_from_item(item: Any) -> str:
    locator = getattr(item, ITEM_LOCATOR_ATTRIBUTE, None)
    text = _public_attr(locator, "locator_cid", "cid", "content_id")
    if text:
        return text
    if isinstance(locator, str) and locator.strip():
        return locator.strip()[:MAX_TEXT_CHARS]
    key = getattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE, None)
    return _public_attr(key, "locator_cid")


def _execution_key_cid_from_item(item: Any) -> str:
    key = getattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE, None)
    return _public_attr(key, "execution_key_cid", "content_id")


def evaluate_pre_setup_item_reuse(
    item: Any,
    *,
    certificate: AdmittedItemCertificate | None = None,
    population: PreSetupReusePopulation | None = None,
) -> PreSetupItemReuseResult:
    """Decide whole-item reuse before current setup.  Never skips."""

    resolved_population = (
        population if population is not None else population_from_item(item)
    )
    execution_key_cid = _execution_key_cid_from_item(item)
    if resolved_population is None:
        return _run_result(
            reason="population_missing",
            execution_key_cid=execution_key_cid,
        )
    if not resolved_population.eligible:
        return _run_result(
            reason=resolved_population.eligibility_reason,
            execution_key_cid=execution_key_cid,
            population=resolved_population,
        )
    resolved = (
        certificate if certificate is not None else item_certificate_from_item(item)
    )
    if resolved is None:
        return _run_result(
            reason="item_certificate_missing",
            execution_key_cid=execution_key_cid,
            population=resolved_population,
        )
    if not resolved.admitted or not resolved.signature_verified:
        return _run_result(
            reason=str(resolved.diagnostics.get("reason") or "item_certificate_not_admitted"),
            execution_key_cid=execution_key_cid,
            certificate=resolved,
            population=resolved_population,
        )
    if execution_key_cid and resolved.execution_key_cid != execution_key_cid:
        return _run_result(
            reason="execution_key_mismatch",
            execution_key_cid=execution_key_cid,
            certificate=resolved,
            population=resolved_population,
        )
    locator = _locator_cid_from_item(item)
    if resolved.locator_cid and locator and resolved.locator_cid != locator:
        return _run_result(
            reason="locator_mismatch",
            execution_key_cid=execution_key_cid,
            certificate=resolved,
            population=resolved_population,
        )
    if not _is_whole_item_phases(resolved.reused_phases):
        return _run_result(
            reason="reuse_phase_not_whole_item",
            execution_key_cid=execution_key_cid or resolved.execution_key_cid,
            certificate=resolved,
            population=resolved_population,
        )
    if resolved.population_kind != resolved_population.population_kind:
        return _run_result(
            reason="population_kind_mismatch",
            execution_key_cid=execution_key_cid or resolved.execution_key_cid,
            certificate=resolved,
            population=resolved_population,
        )
    if (
        resolved.setup_outcome != _PHASE_PASS
        or resolved.call_outcome != _PHASE_PASS
        or resolved.teardown_outcome != _PHASE_PASS
    ):
        return _run_result(
            reason="item_phases_not_pass",
            execution_key_cid=execution_key_cid or resolved.execution_key_cid,
            certificate=resolved,
            population=resolved_population,
        )
    result = PreSetupItemReuseResult(
        reuses_item=True,
        reason="admitted_item_certificate",
        certificate=resolved,
        population=resolved_population,
        execution_key_cid=execution_key_cid or resolved.execution_key_cid,
        diagnostics={
            "reused_phases": list(REUSED_PHASES),
            "population_kind": resolved_population.population_kind,
        },
    )
    attach_item_reuse_result(item, result)
    return result


def apply_admitted_item_reuse(item: Any) -> Callable[[], None] | None:
    """Replace setup and call so the whole item is reused.  Never a skip marker."""

    result = get_attached_item_reuse_result(item)
    if result is None or not result.reuses_item:
        return None
    original_setup = getattr(item, "setup", None)
    original_runtest = getattr(item, "runtest", None)

    def _reused_setup() -> None:
        try:
            setattr(item, ITEM_ITEM_REUSED_ATTRIBUTE, True)
        except Exception:
            pass

    def _reused_call() -> None:
        try:
            setattr(item, ITEM_ITEM_REUSED_ATTRIBUTE, True)
        except Exception:
            pass

    try:
        if original_setup is not None:
            item.setup = _reused_setup
        if original_runtest is not None:
            item.runtest = _reused_call
    except Exception:
        return None

    def restore() -> None:
        if original_setup is not None:
            try:
                item.setup = original_setup
            except Exception:
                pass
        if original_runtest is not None:
            try:
                item.runtest = original_runtest
            except Exception:
                pass

    return restore


def prepare_runtest_setup(item: Any) -> Callable[[], None] | None:
    """Evaluate whole-item reuse before setup and suppress the item body."""

    existing = get_attached_item_reuse_result(item)
    if existing is None:
        result = evaluate_pre_setup_item_reuse(item)
        attach_item_reuse_result(item, result)
    else:
        result = existing
    if not result.reuses_item:
        return None
    return apply_admitted_item_reuse(item)


def after_runtest_teardown(item: Any) -> PreSetupItemReuseResult | None:
    """Record that whole-item reuse omitted current teardown.  Never skip."""

    existing = get_attached_item_reuse_result(item)
    if existing is None or not existing.reuses_item:
        return existing
    updated = PreSetupItemReuseResult(
        reuses_item=True,
        reason=existing.reason,
        certificate=existing.certificate,
        population=existing.population,
        execution_key_cid=existing.execution_key_cid,
        lifecycle_phase="teardown",
        full_execution_reasons=existing.full_execution_reasons,
        diagnostics=dict(existing.diagnostics),
    )
    attach_item_reuse_result(item, updated)
    return updated


@dataclass(frozen=True, slots=True)
class PreSetupItemReuseLifecycleRecord:
    """Ordered pre-setup reuse-or-full-execution evidence.

    Not a pytest test class.
    """

    __test__ = False

    events: tuple[str, ...]
    counts: PhaseExecutionProbe
    reuse: PreSetupItemReuseResult
    population: PreSetupReusePopulation | None = None

    @property
    def reused_whole_item(self) -> bool:
        return (
            self.counts.as_tuple() == (0, 0, 0)
            and self.reuse.reuses_item
            and self.reuse.reused_phases == REUSED_PHASES
            and self.reuse.reuses_setup
            and self.reuse.reuses_call
            and self.reuse.reuses_teardown
            and not self.reuse.may_authorize_skip
        )

    @property
    def executed_all_phases_once(self) -> bool:
        return (
            self.counts.as_tuple() == (1, 1, 1)
            and not self.reuse.reuses_item
        )


def run_pre_setup_item_reuse_lifecycle(
    item: Any,
    *,
    certificate: AdmittedItemCertificate | None = None,
    population: PreSetupReusePopulation | None = None,
    setup: Callable[[], None] | None = None,
    call: Callable[[], None] | None = None,
    teardown: Callable[[], None] | None = None,
) -> PreSetupItemReuseLifecycleRecord:
    """Prove whole-item reuse omits every phase, else each phase runs once."""

    probe = attach_phase_probe(item)
    events: list[str] = []
    if population is not None:
        attach_population(item, population)
    if certificate is not None:
        attach_item_certificate(item, certificate)
    reuse = evaluate_pre_setup_item_reuse(
        item,
        certificate=certificate,
        population=population if population is not None else population_from_item(item),
    )
    attach_item_reuse_result(item, reuse)
    if reuse.reuses_item:
        apply_admitted_item_reuse(item)
        runtest = getattr(item, "runtest", None)
        if callable(runtest):
            runtest()
        setup_fn = getattr(item, "setup", None)
        if callable(setup_fn):
            setup_fn()
        events.append("reused_item")
        final = after_runtest_teardown(item)
        return PreSetupItemReuseLifecycleRecord(
            events=tuple(events),
            counts=probe,
            reuse=final if final is not None else reuse,
            population=reuse.population,
        )
    if setup is not None:
        setup()
    probe.record_setup()
    events.append("setup")
    runtest = getattr(item, "runtest", None)
    if call is not None:
        call()
    elif callable(runtest):
        runtest()
    probe.record_call()
    events.append("call")
    if teardown is not None:
        teardown()
    probe.record_teardown()
    events.append("teardown")
    return PreSetupItemReuseLifecycleRecord(
        events=tuple(events),
        counts=probe,
        reuse=reuse,
        population=reuse.population,
    )


__all__ = [
    "ADMITTED_ITEM_CERTIFICATE_INTERFACE",
    "CLAIM_CLASS",
    "CLOSED_REUSE_CLASSES",
    "DEFAULT_POLICY_CID",
    "EXPLICITLY_PURE_KIND",
    "INELIGIBLE_REUSE_CLASSES",
    "ITEM_ITEM_CERTIFICATE_ATTRIBUTE",
    "ITEM_ITEM_CERTIFICATE_LOOKUP_ATTRIBUTE",
    "ITEM_ITEM_REUSED_ATTRIBUTE",
    "ITEM_ITEM_REUSE_RESULT_ATTRIBUTE",
    "ITEM_PHASE_PROBE_ATTRIBUTE",
    "ITEM_POPULATION_ATTRIBUTE",
    "PREDECESSOR_ASSEMBLY_INTERFACE",
    "PREDECESSOR_ATTESTATION_INTERFACE",
    "PRE_SETUP_ITEM_REUSE_INTERFACE",
    "PRE_SETUP_ITEM_REUSE_POLICY_INTERFACE",
    "PRE_SETUP_ITEM_REUSE_RESULT_INTERFACE",
    "PRE_SETUP_REUSE_POPULATION_INTERFACE",
    "PURE_REUSE_CLASS",
    "REPLAY_SAFE_REUSE_CLASSES",
    "REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND",
    "REUSE_DOES_NOT",
    "REUSE_ESTABLISHES",
    "REUSE_ITEM_ACTION",
    "REUSE_POLICY",
    "REUSED_PHASES",
    "RUN_ACTION",
    "AdmittedItemCertificate",
    "PhaseExecutionProbe",
    "PreSetupItemReuseError",
    "PreSetupItemReuseLifecycleRecord",
    "PreSetupItemReuseResult",
    "PreSetupReusePopulation",
    "admit_item_certificate",
    "after_runtest_teardown",
    "apply_admitted_item_reuse",
    "attach_item_certificate",
    "attach_item_reuse_result",
    "attach_phase_probe",
    "attach_population",
    "authority_descriptor",
    "canonical_public_bytes",
    "classify_population",
    "classify_reuse_class",
    "datasets_contracts_available",
    "evaluate_pre_setup_item_reuse",
    "get_attached_item_reuse_result",
    "get_phase_probe",
    "item_certificate_from_item",
    "item_reuses_whole_item",
    "population_from_item",
    "prepare_runtest_setup",
    "public_digest",
    "record_typed_unavailable",
    "run_pre_setup_item_reuse_lifecycle",
    "typed_unavailable_records",
]
