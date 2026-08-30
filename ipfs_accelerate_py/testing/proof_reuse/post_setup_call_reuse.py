"""PCTDD-028: guarded post-setup, pre-call reuse.

Accelerate-owned pytest integration that reuses only the call phase after
current setup has assembled an exact ``TestExecutionKeyV2``.  An admitted
call certificate is a verified signed runner assertion over V2/composite
evidence.  Current setup and teardown still execute exactly once.
Reuse is never pytest skip, never whole-item skip, and never substitutes
setup, teardown, or finalizers.

Rules:

* Admission is authoritative only after current setup and before call.
* The certificate may cover a previous honest complete pass; only the call
  body is reused.  Setup and teardown remain current-run obligations.
* Signature, execution-key, locator, and call-phase disagreement force
  normal full execution of call.  Opaque or incomplete identities do too.
* Pre-setup whole-item reuse, fixture-proof-aware xdist, production ZK,
  key ceremony, and direct-execution profiles remain typed unavailable.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .setup_bound_execution_key import (
    AUTHORITATIVE_LIFECYCLE_PHASE,
    ITEM_SETUP_BOUND_PHASE_ATTRIBUTE,
    after_runtest_setup,
    before_runtest_call,
    before_runtest_setup,
    get_attached_setup_bound_assembly,
    get_attached_setup_bound_execution_key,
    get_setup_bound_phase,
    mark_setup_bound_phase,
)
from .signed_runner_attestations import (
    SignedRunnerAttestationResult,
    get_attached_signed_runner_result,
)

# This module is accelerate integration, not a pytest test module.
__test__ = False

POST_SETUP_CALL_REUSE_INTERFACE: Final = "PostSetupCallReuse@1"
POST_SETUP_CALL_REUSE_RESULT_INTERFACE: Final = "PostSetupCallReuseResult@1"
ADMITTED_CALL_CERTIFICATE_INTERFACE: Final = "AdmittedCallCertificate@1"
POST_SETUP_CALL_REUSE_POLICY_INTERFACE: Final = "PostSetupCallReusePolicy@1"
POST_SETUP_CALL_REUSE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/post-setup-call-reuse@1"
)
CLAIM_CLASS: Final = "IntegrityCommitment"
REUSED_PHASE: Final = "call"
RUN_ACTION: Final = "RUN"
REUSE_CALL_ACTION: Final = "REUSE_CALL"
SCHEMA_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.post_setup_call_reuse"
)
PREDECESSOR_ASSEMBLY_INTERFACE: Final = "SetupBoundExecutionKeyAssembly@1"
PREDECESSOR_ATTESTATION_INTERFACE: Final = "SignedRunnerAttestationBinding@1"
ITEM_CALL_CERTIFICATE_ATTRIBUTE: Final = "_ipfs_proof_reuse_call_certificate"
ITEM_CALL_CERTIFICATE_LOOKUP_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_call_certificate_lookup"
)
# Public composite pin name shared with xdist coordination; duplicated so this
# module stays import-cold and does not load xdist coordinators.
ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_composite_phase_receipt"
)
ITEM_CALL_REUSE_RESULT_ATTRIBUTE: Final = "_ipfs_proof_reuse_call_reuse_result"
ITEM_PHASE_PROBE_ATTRIBUTE: Final = "_ipfs_proof_reuse_phase_probe"
ITEM_CALL_REUSED_ATTRIBUTE: Final = "_ipfs_proof_reuse_call_reused"
MAX_TEXT_CHARS: Final = 4_096
_DIGEST_PREFIX: Final = "sha256:"
_PHASE_PASS: Final = "pass"

REUSE_ESTABLISHES: Final = (
    "an admitted call certificate reuses only call while current setup and "
    "teardown execute exactly once"
)
REUSE_DOES_NOT: Final = (
    "execution or semantics; skip; whole-item reuse; setup reuse; teardown "
    "reuse; current-root publication; task completion; production ZK; "
    "pre-setup item reuse; self-approval"
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
        "pre_setup_item_reuse",
        "pre_setup_item_reuse_not_implemented",
        "guarded post-setup reuse never skips setup; whole-item pre-setup "
        "reuse remains a later accelerate task",
    ),
    (
        "fixture_proof_aware_xdist",
        "fixture_proof_aware_xdist_not_implemented",
        "fixture affinity and proof-cost placement remain a later xdist "
        "scheduling task; call reuse does not omit tests or merge pools",
    ),
    (
        "aggregate_selected_test_zk",
        "aggregate_selected_test_zk_missing",
        "aggregate selected-test ZK remains a versioned successor; call "
        "reuse cannot upgrade leaf TestPassStatementV1 claims",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; guarded call "
        "reuse cannot admit simulated, structural, or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by guarded "
        "post-setup call reuse",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade call-reuse integrity commitments",
    ),
)


class PostSetupCallReuseError(ValueError):
    """Raised when guarded post-setup call reuse is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise PostSetupCallReuseError(
            "floating-point values are not JSON-safe for call reuse"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise PostSetupCallReuseError(
                    f"call reuse rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise PostSetupCallReuseError("call reuse rejects secret or raw bytes")
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise PostSetupCallReuseError(
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


REUSE_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": POST_SETUP_CALL_REUSE_POLICY_INTERFACE,
        "reuse_interface": POST_SETUP_CALL_REUSE_INTERFACE,
        "predecessor_assembly_interface": PREDECESSOR_ASSEMBLY_INTERFACE,
        "predecessor_attestation_interface": PREDECESSOR_ATTESTATION_INTERFACE,
        "reused_phase": REUSED_PHASE,
        "reuses_setup": False,
        "reuses_teardown": False,
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
        raise PostSetupCallReuseError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-028 typed unavailable capabilities."""

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
                    "guarded call reuse forces full execution of call"
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
        "reused_phase": REUSED_PHASE,
        "reuses_setup": False,
        "reuses_teardown": False,
        "normal_execution_fallback": True,
        "reuse_interface": POST_SETUP_CALL_REUSE_INTERFACE,
        "certificate_interface": ADMITTED_CALL_CERTIFICATE_INTERFACE,
        "predecessor_assembly_interface": PREDECESSOR_ASSEMBLY_INTERFACE,
        "predecessor_attestation_interface": PREDECESSOR_ATTESTATION_INTERFACE,
        "test_execution_key_v2": "TestExecutionKeyV2",
        "composite_phase_receipt": "CompositePhaseReceipt@1",
    }


def _require_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PostSetupCallReuseError(f"{field_name} must be text")
    text = value.strip()
    if text != value or len(text) > MAX_TEXT_CHARS:
        raise PostSetupCallReuseError(f"invalid {field_name}")
    if not text and not allow_empty:
        raise PostSetupCallReuseError(f"{field_name} is required")
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
    raise PostSetupCallReuseError(f"{field_name} must be text")


def _forbid_skip_production(
    *,
    may_authorize_skip: bool,
    production_admitted: bool,
    self_approved: bool,
    claim_unchanged: bool,
) -> None:
    if may_authorize_skip:
        raise PostSetupCallReuseError("guarded call reuse must not authorize skip")
    if production_admitted or self_approved or not claim_unchanged:
        raise PostSetupCallReuseError(
            "guarded call reuse cannot admit production, self-approve, or change claims"
        )


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
        raise PostSetupCallReuseError("phase probe must be a PhaseExecutionProbe")
    try:
        setattr(item, ITEM_PHASE_PROBE_ATTRIBUTE, attached)
    except Exception as exc:
        raise PostSetupCallReuseError("unable to attach phase probe") from exc
    return attached


def get_phase_probe(item: Any) -> PhaseExecutionProbe | None:
    """Return the attached phase-execution probe, if any."""

    existing = getattr(item, ITEM_PHASE_PROBE_ATTRIBUTE, None)
    if isinstance(existing, PhaseExecutionProbe):
        return existing
    return None


@dataclass(frozen=True, slots=True)
class AdmittedCallCertificate:
    """Public pins for one admitted call-only reuse certificate.

    Setup and teardown pins are recorded for honesty checks and are never
    reused.  Not a pytest test class.
    """

    __test__ = False

    execution_key_cid: str
    certificate_cid: str
    composite_phase_receipt_cid: str = ""
    locator_cid: str = ""
    call_outcome: str = _PHASE_PASS
    setup_outcome: str = _PHASE_PASS
    teardown_outcome: str = _PHASE_PASS
    signature_verified: bool = False
    admitted: bool = False
    reused_phase: str = REUSED_PHASE
    reuses_setup: bool = False
    reuses_teardown: bool = False
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
        if self.reuses_setup or self.reuses_teardown:
            raise PostSetupCallReuseError(
                "admitted call certificate cannot reuse setup or teardown"
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
            self, "call_outcome", _phase_text(self.call_outcome, "call_outcome")
        )
        object.__setattr__(
            self, "setup_outcome", _phase_text(self.setup_outcome, "setup_outcome")
        )
        object.__setattr__(
            self,
            "teardown_outcome",
            _phase_text(self.teardown_outcome, "teardown_outcome"),
        )
        reused = _require_text(self.reused_phase, "reused_phase", allow_empty=True)
        if reused and reused != REUSED_PHASE:
            raise PostSetupCallReuseError("admitted call certificate reuses only call")
        object.__setattr__(self, "reused_phase", reused or REUSED_PHASE)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.admitted:
            if not self.signature_verified:
                raise PostSetupCallReuseError(
                    "admitted call certificate requires a verified signature"
                )
            if not self.execution_key_cid or not self.certificate_cid:
                raise PostSetupCallReuseError(
                    "admitted call certificate requires execution-key and certificate cids"
                )
            if self.call_outcome != _PHASE_PASS:
                raise PostSetupCallReuseError(
                    "admitted call certificate requires an honest passing call"
                )
            if self.reused_phase != REUSED_PHASE:
                raise PostSetupCallReuseError(
                    "admitted call certificate reuses only call"
                )

    @property
    def interface(self) -> str:
        return ADMITTED_CALL_CERTIFICATE_INTERFACE

    @property
    def schema(self) -> str:
        return POST_SETUP_CALL_REUSE_SCHEMA

    @property
    def reuses_call(self) -> bool:
        return self.admitted and self.reused_phase == REUSED_PHASE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "execution_key_cid": self.execution_key_cid,
            "certificate_cid": self.certificate_cid,
            "composite_phase_receipt_cid": self.composite_phase_receipt_cid,
            "locator_cid": self.locator_cid,
            "call_outcome": self.call_outcome,
            "setup_outcome": self.setup_outcome,
            "teardown_outcome": self.teardown_outcome,
            "signature_verified": self.signature_verified,
            "admitted": self.admitted,
            "reused_phase": self.reused_phase,
            "reuses_call": self.reuses_call,
            "reuses_setup": False,
            "reuses_teardown": False,
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
    call_outcome: str = "",
    setup_outcome: str = "",
    teardown_outcome: str = "",
    signature_verified: bool = False,
    reason: str,
) -> AdmittedCallCertificate:
    return AdmittedCallCertificate(
        execution_key_cid=execution_key_cid,
        certificate_cid=certificate_cid,
        composite_phase_receipt_cid=composite_phase_receipt_cid,
        locator_cid=locator_cid,
        call_outcome=call_outcome or _PHASE_PASS,
        setup_outcome=setup_outcome or _PHASE_PASS,
        teardown_outcome=teardown_outcome or _PHASE_PASS,
        signature_verified=signature_verified,
        admitted=False,
        diagnostics={"reason": reason},
    )


def admit_call_certificate(
    *,
    execution_key: Any = None,
    composite_receipt: Any = None,
    attestation_result: SignedRunnerAttestationResult | None = None,
    certificate_cid: str = "",
    reused_phases: Sequence[str] = (REUSED_PHASE,),
    reuses_setup: bool = False,
    reuses_teardown: bool = False,
) -> AdmittedCallCertificate:
    """Admit a call-only certificate or return an unadmitted record."""

    phases = tuple(str(item).strip() for item in reused_phases if str(item).strip())
    if reuses_setup or reuses_teardown or any(phase != REUSED_PHASE for phase in phases):
        return _unadmitted_certificate(
            reason="reuse_phase_not_call_only",
        )
    execution_key_cid = _public_attr(execution_key, "execution_key_cid", "content_id")
    locator_cid = _public_attr(execution_key, "locator_cid")
    composite_cid = _public_attr(
        composite_receipt, "receipt_cid", "composite_phase_receipt_cid"
    )
    call_outcome = _phase_text(
        getattr(composite_receipt, "call_outcome", _PHASE_PASS), "call_outcome"
    )
    setup_outcome = _phase_text(
        getattr(composite_receipt, "setup_outcome", _PHASE_PASS), "setup_outcome"
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
                call_outcome=call_outcome,
                setup_outcome=setup_outcome,
                teardown_outcome=teardown_outcome,
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
            call_outcome = _phase_text(pass_receipt.call_outcome, "call_outcome")
            setup_outcome = _phase_text(pass_receipt.setup_outcome, "setup_outcome")
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
        and call_outcome == _PHASE_PASS
        and execution_key_cid
        and attested_cid
    ):
        return AdmittedCallCertificate(
            execution_key_cid=execution_key_cid,
            certificate_cid=attested_cid,
            composite_phase_receipt_cid=composite_cid,
            locator_cid=locator_cid,
            call_outcome=call_outcome,
            setup_outcome=setup_outcome,
            teardown_outcome=teardown_outcome,
            signature_verified=True,
            admitted=True,
        )
    reason = "call_certificate_not_admitted"
    if not signature_verified:
        reason = "signature_not_verified"
    elif call_outcome != _PHASE_PASS:
        reason = "call_phase_not_pass"
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
        call_outcome=call_outcome,
        setup_outcome=setup_outcome,
        teardown_outcome=teardown_outcome,
        signature_verified=signature_verified,
        reason=reason,
    )


def attach_call_certificate(item: Any, certificate: AdmittedCallCertificate) -> None:
    """Attach a public call certificate without skip authority."""

    if item is None:
        return
    if not isinstance(certificate, AdmittedCallCertificate):
        raise PostSetupCallReuseError("call certificate must be AdmittedCallCertificate")
    if certificate.may_authorize_skip or certificate.reuses_setup or certificate.reuses_teardown:
        raise PostSetupCallReuseError(
            "attached call certificate cannot skip or reuse setup/teardown"
        )
    try:
        setattr(item, ITEM_CALL_CERTIFICATE_ATTRIBUTE, certificate)
    except Exception:
        return


def call_certificate_from_item(item: Any) -> AdmittedCallCertificate | None:
    """Return an attached or reconstructed call certificate, if any."""

    if item is None:
        return None
    attached = getattr(item, ITEM_CALL_CERTIFICATE_ATTRIBUTE, None)
    if isinstance(attached, AdmittedCallCertificate):
        return attached
    lookup = getattr(item, ITEM_CALL_CERTIFICATE_LOOKUP_ATTRIBUTE, None)
    if callable(lookup):
        try:
            found = lookup(item)
        except Exception:
            found = None
        if isinstance(found, AdmittedCallCertificate):
            return found
    signed = get_attached_signed_runner_result(item)
    composite = getattr(item, ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE, None)
    key = get_attached_setup_bound_execution_key(item)
    if signed is None and composite is None:
        return None
    try:
        return admit_call_certificate(
            execution_key=key,
            composite_receipt=composite,
            attestation_result=signed,
        )
    except PostSetupCallReuseError:
        return None


@dataclass(frozen=True, slots=True)
class PostSetupCallReuseResult:
    """Outcome of one guarded post-setup, pre-call reuse attempt.

    ``reuses_call`` never implies pytest skip, setup reuse, teardown reuse,
    production admission, or task completion.  Not a pytest test class.
    """

    __test__ = False

    reuses_call: bool
    reason: str
    certificate: AdmittedCallCertificate | None = None
    execution_key_cid: str = ""
    lifecycle_phase: str = "call"
    current_setup_executed: bool = False
    current_teardown_executed: bool = False
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
        if self.reuses_call:
            if not self.current_setup_executed:
                raise PostSetupCallReuseError(
                    "call reuse requires current setup to have executed"
                )
            if self.certificate is None or not self.certificate.admitted:
                raise PostSetupCallReuseError(
                    "call reuse requires an admitted call certificate"
                )
            if self.certificate.reuses_setup or self.certificate.reuses_teardown:
                raise PostSetupCallReuseError(
                    "call reuse cannot reuse setup or teardown"
                )
            if self.action != REUSE_CALL_ACTION:
                raise PostSetupCallReuseError("admitted reuse action must be REUSE_CALL")

    @property
    def interface(self) -> str:
        return POST_SETUP_CALL_REUSE_RESULT_INTERFACE

    @property
    def schema(self) -> str:
        return POST_SETUP_CALL_REUSE_SCHEMA

    @property
    def reuse_interface(self) -> str:
        return POST_SETUP_CALL_REUSE_INTERFACE

    @property
    def reused_phase(self) -> str:
        return REUSED_PHASE if self.reuses_call else ""

    @property
    def reuses_setup(self) -> bool:
        return False

    @property
    def reuses_teardown(self) -> bool:
        return False

    @property
    def action(self) -> str:
        return REUSE_CALL_ACTION if self.reuses_call else RUN_ACTION

    @property
    def normal_execution_fallback(self) -> bool:
        return True

    @property
    def requires_full_execution(self) -> bool:
        return not self.reuses_call

    @property
    def claim_class(self) -> str:
        return CLAIM_CLASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "reuse_interface": self.reuse_interface,
            "reuses_call": self.reuses_call,
            "reuses_setup": False,
            "reuses_teardown": False,
            "reused_phase": self.reused_phase,
            "action": self.action,
            "reason": self.reason,
            "execution_key_cid": self.execution_key_cid,
            "lifecycle_phase": self.lifecycle_phase,
            "current_setup_executed": self.current_setup_executed,
            "current_teardown_executed": self.current_teardown_executed,
            "may_authorize_skip": False,
            "production_admitted": False,
            "self_approved": False,
            "claim_unchanged": True,
            "claim_class": self.claim_class,
            "normal_execution_fallback": True,
            "requires_full_execution": self.requires_full_execution,
            "full_execution_reasons": list(self.full_execution_reasons),
            "certificate": None if self.certificate is None else self.certificate.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _run_result(
    *,
    reason: str,
    execution_key_cid: str = "",
    current_setup_executed: bool = False,
    current_teardown_executed: bool = False,
    certificate: AdmittedCallCertificate | None = None,
    extra_reasons: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> PostSetupCallReuseResult:
    reasons = tuple(
        dict.fromkeys(
            [reason, *extra_reasons, "normal_execution_fallback"]
            if reason
            else list(extra_reasons) or ["normal_execution_fallback"]
        )
    )
    return PostSetupCallReuseResult(
        reuses_call=False,
        reason=reason,
        certificate=certificate,
        execution_key_cid=execution_key_cid,
        current_setup_executed=current_setup_executed,
        current_teardown_executed=current_teardown_executed,
        full_execution_reasons=reasons,
        diagnostics=dict(diagnostics or {}),
    )


def attach_call_reuse_result(item: Any, result: PostSetupCallReuseResult) -> None:
    """Attach a reuse decision without skip authority."""

    if item is None:
        return
    if not isinstance(result, PostSetupCallReuseResult):
        raise PostSetupCallReuseError("reuse result must be PostSetupCallReuseResult")
    try:
        setattr(item, ITEM_CALL_REUSE_RESULT_ATTRIBUTE, result)
        if result.reuses_call:
            setattr(item, ITEM_CALL_REUSED_ATTRIBUTE, True)
    except Exception:
        return


def get_attached_call_reuse_result(item: Any) -> PostSetupCallReuseResult | None:
    """Return the attached reuse result, if any."""

    existing = getattr(item, ITEM_CALL_REUSE_RESULT_ATTRIBUTE, None)
    if isinstance(existing, PostSetupCallReuseResult):
        return existing
    return None


def evaluate_post_setup_call_reuse(
    item: Any,
    *,
    certificate: AdmittedCallCertificate | None = None,
    current_setup_executed: bool | None = None,
) -> PostSetupCallReuseResult:
    """Decide call-only reuse after current setup.  Never skips."""

    if current_setup_executed is not None:
        setup_executed = bool(current_setup_executed)
    else:
        probe = get_phase_probe(item)
        phase = get_setup_bound_phase(item)
        setup_executed = bool(probe is not None and probe.setup > 0) or phase in {
            AUTHORITATIVE_LIFECYCLE_PHASE,
            "call",
            "teardown",
        }
    assembly = get_attached_setup_bound_assembly(item)
    key = get_attached_setup_bound_execution_key(item)
    execution_key_cid = _public_attr(key, "execution_key_cid", "content_id")
    if assembly is None or not getattr(assembly, "assembled_after_setup", False):
        return _run_result(
            reason="setup_bound_execution_key_missing",
            execution_key_cid=execution_key_cid,
            current_setup_executed=setup_executed,
        )
    if not setup_executed:
        return _run_result(
            reason="current_setup_not_executed",
            execution_key_cid=execution_key_cid,
            current_setup_executed=False,
        )
    if getattr(assembly, "requires_full_execution", True):
        return _run_result(
            reason="full_execution_required",
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
            extra_reasons=tuple(getattr(assembly, "full_execution_reasons", ()) or ()),
        )
    resolved = certificate if certificate is not None else call_certificate_from_item(item)
    if resolved is None:
        return _run_result(
            reason="call_certificate_missing",
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
        )
    if not resolved.admitted or not resolved.signature_verified:
        return _run_result(
            reason=str(resolved.diagnostics.get("reason") or "call_certificate_not_admitted"),
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
            certificate=resolved,
        )
    if resolved.execution_key_cid != execution_key_cid:
        return _run_result(
            reason="execution_key_mismatch",
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
            certificate=resolved,
        )
    locator = _public_attr(key, "locator_cid")
    if resolved.locator_cid and locator and resolved.locator_cid != locator:
        return _run_result(
            reason="locator_mismatch",
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
            certificate=resolved,
        )
    if resolved.reused_phase != REUSED_PHASE or resolved.reuses_setup or resolved.reuses_teardown:
        return _run_result(
            reason="reuse_phase_not_call_only",
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
            certificate=resolved,
        )
    if resolved.call_outcome != _PHASE_PASS:
        return _run_result(
            reason="call_phase_not_pass",
            execution_key_cid=execution_key_cid,
            current_setup_executed=True,
            certificate=resolved,
        )
    result = PostSetupCallReuseResult(
        reuses_call=True,
        reason="admitted_call_certificate",
        certificate=resolved,
        execution_key_cid=execution_key_cid,
        current_setup_executed=True,
        current_teardown_executed=False,
        diagnostics={"reused_phase": REUSED_PHASE},
    )
    attach_call_reuse_result(item, result)
    return result


def apply_admitted_call_reuse(item: Any) -> Callable[[], None] | None:
    """Replace ``item.runtest`` so only call is reused.  Never a skip marker."""

    result = get_attached_call_reuse_result(item)
    if result is None or not result.reuses_call:
        return None
    original = getattr(item, "runtest", None)

    def _reused_call() -> None:
        try:
            setattr(item, ITEM_CALL_REUSED_ATTRIBUTE, True)
        except Exception:
            pass

    try:
        item.runtest = _reused_call
    except Exception:
        return None

    def restore() -> None:
        if original is not None:
            try:
                item.runtest = original
            except Exception:
                pass

    return restore


def prepare_runtest_call(item: Any) -> Callable[[], None] | None:
    """Evaluate reuse after setup-bound assembly and suppress only call."""

    result = evaluate_post_setup_call_reuse(item)
    attach_call_reuse_result(item, result)
    if not result.reuses_call:
        return None
    return apply_admitted_call_reuse(item)


def before_runtest_teardown(item: Any) -> str:
    """Mark teardown; current teardown always runs."""

    mark_setup_bound_phase(item, "teardown")
    try:
        setattr(item, ITEM_SETUP_BOUND_PHASE_ATTRIBUTE, "teardown")
    except Exception:
        pass
    return "teardown"


def after_runtest_teardown(item: Any) -> PostSetupCallReuseResult | None:
    """Record that current teardown executed.  Never reuses teardown."""

    probe = get_phase_probe(item)
    if probe is not None and probe.teardown == 0:
        probe.record_teardown()
    existing = get_attached_call_reuse_result(item)
    if existing is None:
        return None
    updated = PostSetupCallReuseResult(
        reuses_call=existing.reuses_call,
        reason=existing.reason,
        certificate=existing.certificate,
        execution_key_cid=existing.execution_key_cid,
        lifecycle_phase="teardown",
        current_setup_executed=existing.current_setup_executed,
        current_teardown_executed=True,
        full_execution_reasons=existing.full_execution_reasons,
        diagnostics=dict(existing.diagnostics),
    )
    attach_call_reuse_result(item, updated)
    return updated


@dataclass(frozen=True, slots=True)
class PostSetupCallReuseLifecycleRecord:
    """Ordered setup -> reuse-or-call -> teardown evidence.

    Not a pytest test class.
    """

    __test__ = False

    events: tuple[str, ...]
    counts: PhaseExecutionProbe
    reuse: PostSetupCallReuseResult
    assembly: Any = None

    @property
    def reused_only_call(self) -> bool:
        return (
            self.counts.setup == 1
            and self.counts.call == 0
            and self.counts.teardown == 1
            and self.reuse.reuses_call
            and self.reuse.reused_phase == REUSED_PHASE
            and not self.reuse.reuses_setup
            and not self.reuse.reuses_teardown
            and self.reuse.current_setup_executed
            and self.reuse.current_teardown_executed
        )

    @property
    def executed_all_phases_once(self) -> bool:
        return (
            self.counts.setup == 1
            and self.counts.call == 1
            and self.counts.teardown == 1
            and not self.reuse.reuses_call
        )


def run_post_setup_call_reuse_lifecycle(
    item: Any,
    *,
    certificate: AdmittedCallCertificate | None = None,
    setup: Callable[[], None] | None = None,
    call: Callable[[], None] | None = None,
    teardown: Callable[[], None] | None = None,
    **key_fields: Any,
) -> PostSetupCallReuseLifecycleRecord:
    """Prove admitted call reuse skips only call, once each for setup/teardown."""

    probe = attach_phase_probe(item)
    events: list[str] = []
    before_runtest_setup(item)
    if setup is not None:
        setup()
    probe.record_setup()
    events.append("setup")
    assembly = after_runtest_setup(item, setup_failed=False, **key_fields)
    events.append("assembled")
    if certificate is not None:
        attach_call_certificate(item, certificate)
    before_runtest_call(item)
    reuse = evaluate_post_setup_call_reuse(
        item,
        certificate=certificate,
        current_setup_executed=True,
    )
    attach_call_reuse_result(item, reuse)
    if reuse.reuses_call:
        apply_admitted_call_reuse(item)
        runtest = getattr(item, "runtest", None)
        if callable(runtest):
            runtest()
        events.append("reused_call")
    else:
        if call is not None:
            call()
        probe.record_call()
        events.append("call")
    before_runtest_teardown(item)
    if teardown is not None:
        teardown()
    probe.record_teardown()
    events.append("teardown")
    final = after_runtest_teardown(item)
    return PostSetupCallReuseLifecycleRecord(
        events=tuple(events),
        counts=probe,
        reuse=final if final is not None else reuse,
        assembly=assembly,
    )


__all__ = [
    "ADMITTED_CALL_CERTIFICATE_INTERFACE",
    "CLAIM_CLASS",
    "DEFAULT_POLICY_CID",
    "ITEM_CALL_CERTIFICATE_ATTRIBUTE",
    "ITEM_CALL_CERTIFICATE_LOOKUP_ATTRIBUTE",
    "ITEM_CALL_REUSED_ATTRIBUTE",
    "ITEM_CALL_REUSE_RESULT_ATTRIBUTE",
    "ITEM_PHASE_PROBE_ATTRIBUTE",
    "POST_SETUP_CALL_REUSE_INTERFACE",
    "POST_SETUP_CALL_REUSE_POLICY_INTERFACE",
    "POST_SETUP_CALL_REUSE_RESULT_INTERFACE",
    "PREDECESSOR_ASSEMBLY_INTERFACE",
    "PREDECESSOR_ATTESTATION_INTERFACE",
    "REUSE_CALL_ACTION",
    "REUSE_DOES_NOT",
    "REUSE_ESTABLISHES",
    "REUSE_POLICY",
    "REUSED_PHASE",
    "RUN_ACTION",
    "AdmittedCallCertificate",
    "PhaseExecutionProbe",
    "PostSetupCallReuseError",
    "PostSetupCallReuseLifecycleRecord",
    "PostSetupCallReuseResult",
    "admit_call_certificate",
    "after_runtest_teardown",
    "apply_admitted_call_reuse",
    "attach_call_certificate",
    "attach_call_reuse_result",
    "attach_phase_probe",
    "authority_descriptor",
    "before_runtest_teardown",
    "call_certificate_from_item",
    "canonical_public_bytes",
    "datasets_contracts_available",
    "evaluate_post_setup_call_reuse",
    "get_attached_call_reuse_result",
    "get_phase_probe",
    "prepare_runtest_call",
    "public_digest",
    "record_typed_unavailable",
    "run_post_setup_call_reuse_lifecycle",
    "typed_unavailable_records",
]
