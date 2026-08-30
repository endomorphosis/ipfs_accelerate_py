"""PCTDD-032: bind current signed runner attestations to V2/composite evidence.

Accelerate-owned integration over the existing ``RunnerPassAttestation@1``
predecessor.  Runner signatures bind ``TestExecutionKeyV2`` and
``CompositePhaseReceipt@1`` to a verifier-selected key, issuer, epoch, and
policy.  The signature is a ``SignedExecutionReceipt``: a trusted issuer
assertion under explicit verifier policy, not independent execution proof.

Rules:

* Verifier-selected key, issuer, epoch, and policy are compared to V2 trust
  pins, composite runner-policy pins, and the signed attestation.  Any
  disagreement fails closed.
* Signatures never authorize pytest skip, never admit production, and never
  self-approve the task or change claim meaning.
* Production ZK, key ceremony, direct-execution, aggregate selected-test ZK,
  guarded post-setup reuse, and pre-setup item reuse remain typed unavailable.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ...agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    SignedTestPassReceiptV2,
    TestPassReceipt,
)
from .runner_pass_attestation import (
    RUNNER_PASS_ATTESTATION_INTERFACE,
    AttestationNonceRegistry,
    AttestationVerification,
    RunnerAttestationError,
    RunnerPassAttestation,
    RunnerPublicKey,
    RunnerTrustPolicy,
    attest_test_pass_receipt,
    dag_cbor_cid,
    verify_runner_pass_attestation_with_key,
)

# This module is accelerate integration, not a pytest test module.
__test__ = False

SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE: Final = (
    "SignedRunnerAttestationBinding@1"
)
SIGNED_RUNNER_ATTESTATION_RESULT_INTERFACE: Final = (
    "SignedRunnerAttestationResult@1"
)
SIGNED_RUNNER_ATTESTATION_POLICY_INTERFACE: Final = (
    "SignedRunnerAttestationPolicy@1"
)
SIGNED_RUNNER_ATTESTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/signed-runner-attestation-binding@1"
)
CLAIM_CLASS: Final = "IntegrityCommitment"
ATTESTATION_CLAIM_CLASS: Final = "SignedExecutionReceipt"
PREDECESSOR_INTERFACE: Final = RUNNER_PASS_ATTESTATION_INTERFACE
ATTESTATION_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.signed_runner_attestations"
)
SCHEMA_AUTHORITY: Final = ATTESTATION_AUTHORITY
ITEM_SIGNED_RUNNER_ATTESTATION_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_signed_runner_attestation"
)
ITEM_SIGNED_RUNNER_BINDING_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_signed_runner_binding"
)
ITEM_SIGNED_RUNNER_RESULT_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_signed_runner_result"
)
MAX_TEXT_CHARS: Final = 4_096
_DIGEST_PREFIX: Final = "sha256:"

ATTESTATION_ESTABLISHES: Final = (
    "runner signatures bind V2/composite evidence to verifier-selected key, "
    "issuer, epoch and policy"
)
ATTESTATION_DOES_NOT: Final = (
    "execution or semantics; skip; independent faithful execution; "
    "current-root publication; production ZK; adapter execution; aggregate ZK; "
    "self-approval"
)
SIGNED_EXECUTION_ESTABLISHES: Final = (
    "trusted issuer assertion under explicit verifier policy"
)
SIGNED_EXECUTION_DOES_NOT: Final = "independent faithful execution"

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
        "guarded_post_setup_reuse",
        "guarded_post_setup_reuse_not_implemented",
        "signed runner attestations bind V2/composite evidence only; guarded "
        "post-setup pre-call reuse remains a later accelerate task",
    ),
    (
        "pre_setup_item_reuse",
        "pre_setup_item_reuse_not_implemented",
        "signed runner attestations do not gate whole-item pre-setup reuse; "
        "that remains a later accelerate task",
    ),
    (
        "aggregate_selected_test_zk",
        "aggregate_selected_test_zk_missing",
        "aggregate selected-test ZK remains a versioned successor; signed "
        "runner attestations cannot upgrade leaf TestPassStatementV1 claims",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; signed runner "
        "attestations cannot admit simulated, structural, or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by signed runner "
        "attestation binding",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade signed runner attestation issuer assertions",
    ),
)


class SignedRunnerAttestationError(ValueError):
    """Raised when signed runner attestation binding is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise SignedRunnerAttestationError(
            "floating-point values are not JSON-safe for signed runner bindings"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise SignedRunnerAttestationError(
                    f"signed runner binding rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise SignedRunnerAttestationError(
            "signed runner binding rejects secret or raw bytes"
        )
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise SignedRunnerAttestationError(
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


ATTESTATION_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": SIGNED_RUNNER_ATTESTATION_POLICY_INTERFACE,
        "binding_interface": SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE,
        "predecessor_interface": PREDECESSOR_INTERFACE,
        "attestation_claim_class": ATTESTATION_CLAIM_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "normal_execution_fallback": True,
        "schema_authority": SCHEMA_AUTHORITY,
    }
)
DEFAULT_POLICY_CID: Final = public_digest(dict(ATTESTATION_POLICY))


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
        raise SignedRunnerAttestationError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-032 typed unavailable capabilities."""

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
                    "codecs did not import in this sealed environment; signed "
                    "runner binding cannot mint V2/composite evidence"
                ),
            )
        )
    return tuple(records)


def authority_descriptor() -> dict[str, Any]:
    """Return the attestation authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "attestation": ATTESTATION_AUTHORITY,
        "schema_authority": SCHEMA_AUTHORITY,
        "does_not": ATTESTATION_DOES_NOT,
        "establishes": ATTESTATION_ESTABLISHES,
        "claim_class": CLAIM_CLASS,
        "attestation_claim_class": ATTESTATION_CLAIM_CLASS,
        "signed_execution_establishes": SIGNED_EXECUTION_ESTABLISHES,
        "signed_execution_does_not": SIGNED_EXECUTION_DOES_NOT,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "normal_execution_fallback": True,
        "binding_interface": SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE,
        "predecessor_interface": PREDECESSOR_INTERFACE,
        "test_execution_key_v2": "TestExecutionKeyV2",
        "composite_phase_receipt": "CompositePhaseReceipt@1",
    }


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SignedRunnerAttestationError(f"{field_name} must be text")
    text = value.strip()
    if not text or text != value or len(text) > MAX_TEXT_CHARS:
        raise SignedRunnerAttestationError(f"invalid {field_name}")
    return text


def _public_attr(value: Any, *names: str) -> str:
    if value is None:
        return ""
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:MAX_TEXT_CHARS]
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
    return ""


def _nested_attr(value: Any, parent: str, *names: str) -> str:
    child = getattr(value, parent, None)
    if child is None and isinstance(value, Mapping):
        child = value.get(parent)
    if child is None:
        return ""
    return _public_attr(child, *names)


def v2_execution_key_cid(execution_key: Any) -> str:
    """Return the datasets-owned TestExecutionKeyV2 digest."""

    cid = _public_attr(execution_key, "execution_key_cid", "content_id")
    if not cid:
        raise SignedRunnerAttestationError("TestExecutionKeyV2 cid is missing")
    return cid


def composite_phase_receipt_cid(receipt: Any) -> str:
    """Return the datasets-owned CompositePhaseReceipt@1 digest."""

    cid = _public_attr(receipt, "receipt_cid", "composite_phase_receipt_cid")
    if not cid:
        raise SignedRunnerAttestationError("CompositePhaseReceipt cid is missing")
    return cid


def execution_key_cidv1(execution_key_cid: str) -> str:
    """Wrap a datasets V2 digest in the CIDv1 envelope signed by the runner."""

    return dag_cbor_cid(
        {
            "interface": "TestExecutionKeyV2",
            "test_execution_key_v2_cid": _require_text(
                execution_key_cid, "test_execution_key_v2_cid"
            ),
        }
    )


def composite_receipt_cidv1(receipt_cid: str) -> str:
    """Wrap a datasets composite digest in a CIDv1 envelope."""

    return dag_cbor_cid(
        {
            "interface": "CompositePhaseReceipt@1",
            "composite_phase_receipt_cid": _require_text(
                receipt_cid, "composite_phase_receipt_cid"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class VerifierSelectedTrust:
    """Verifier-owned key, issuer, epoch, and policy pins.

    Possession of a signature is not a trust decision.  The caller supplies
    the locally pinned policy and exact public-key material.
    """

    __test__ = False

    policy: RunnerTrustPolicy
    public_key: RunnerPublicKey
    issuer_id: str
    epoch: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RunnerTrustPolicy):
            raise SignedRunnerAttestationError("verifier policy must be RunnerTrustPolicy")
        if not isinstance(self.public_key, RunnerPublicKey):
            raise SignedRunnerAttestationError("verifier key must be RunnerPublicKey")
        object.__setattr__(self, "issuer_id", _require_text(self.issuer_id, "issuer_id"))
        object.__setattr__(self, "epoch", _require_text(self.epoch, "epoch"))
        if self.epoch != self.policy.active_key_epoch:
            raise SignedRunnerAttestationError(
                "verifier-selected epoch is not the active policy key epoch"
            )
        try:
            self.policy.key_for(self.public_key.cid, self.epoch, 0)
        except RunnerAttestationError:
            # Validity window is checked at attestation time; identity must exist.
            matches = [
                key
                for key in self.policy.keys
                if key.public_key_cid == self.public_key.cid and key.key_epoch == self.epoch
            ]
            if len(matches) != 1:
                raise SignedRunnerAttestationError(
                    "verifier-selected key is not in the pinned policy epoch"
                ) from None
            if matches[0].revoked or self.public_key.cid in self.policy.revoked_key_cids:
                raise SignedRunnerAttestationError(
                    "verifier-selected key is revoked"
                ) from None

    @property
    def policy_cid(self) -> str:
        return self.policy.cid

    @property
    def key_cid(self) -> str:
        return self.public_key.cid

    @property
    def trust_domain(self) -> str:
        return self.policy.trust_domain

    def to_dict(self) -> dict[str, str]:
        return {
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "key_cid": self.key_cid,
            "policy_cid": self.policy_cid,
            "trust_domain": self.trust_domain,
        }


def select_verifier_trust(
    *,
    policy: RunnerTrustPolicy,
    public_key: RunnerPublicKey,
    issuer_id: str,
    epoch: str | None = None,
) -> VerifierSelectedTrust:
    """Construct verifier-selected key/issuer/epoch/policy pins."""

    return VerifierSelectedTrust(
        policy=policy,
        public_key=public_key,
        issuer_id=issuer_id,
        epoch=epoch if epoch is not None else policy.active_key_epoch,
    )


def _v2_trust_pins(execution_key: Any) -> dict[str, str]:
    return {
        "policy_cid": _nested_attr(execution_key, "trust", "policy_cid"),
        "issuer_id": _nested_attr(execution_key, "trust", "issuer_id"),
        "epoch": _nested_attr(execution_key, "trust", "epoch"),
    }


def _composite_trust_pins(receipt: Any) -> dict[str, str]:
    return {
        "policy_cid": _public_attr(receipt, "policy_cid"),
        "issuer_id": _public_attr(receipt, "issuer_id"),
        "epoch": _public_attr(receipt, "epoch"),
        "execution_key_cid": _public_attr(receipt, "execution_key_cid"),
        "locator_cid": _public_attr(receipt, "locator_cid"),
    }


def _require_matching_pin(left: str, right: str, field_name: str) -> str:
    if not left or not right or left != right:
        raise SignedRunnerAttestationError(
            f"V2/composite {field_name} is not the verifier-selected pin"
        )
    return left


def bind_v2_composite_to_verifier(
    execution_key: Any,
    composite_receipt: Any,
    verifier: VerifierSelectedTrust,
) -> "SignedRunnerAttestationBinding":
    """Fail closed unless V2 and composite pins equal verifier selection."""

    try:
        v2_cid = v2_execution_key_cid(execution_key)
        composite_cid = composite_phase_receipt_cid(composite_receipt)
        v2_pins = _v2_trust_pins(execution_key)
        composite_pins = _composite_trust_pins(composite_receipt)
        if not v2_pins["policy_cid"] or not v2_pins["issuer_id"] or not v2_pins["epoch"]:
            raise SignedRunnerAttestationError("TestExecutionKeyV2 trust pins are unbound")
        if (
            not composite_pins["policy_cid"]
            or not composite_pins["issuer_id"]
            or not composite_pins["epoch"]
        ):
            raise SignedRunnerAttestationError(
                "CompositePhaseReceipt runner-policy pins are unbound"
            )
        policy_cid = _require_matching_pin(
            v2_pins["policy_cid"], verifier.policy_cid, "policy"
        )
        _require_matching_pin(composite_pins["policy_cid"], policy_cid, "policy")
        issuer_id = _require_matching_pin(
            v2_pins["issuer_id"], verifier.issuer_id, "issuer"
        )
        _require_matching_pin(composite_pins["issuer_id"], issuer_id, "issuer")
        epoch = _require_matching_pin(v2_pins["epoch"], verifier.epoch, "epoch")
        _require_matching_pin(composite_pins["epoch"], epoch, "epoch")
        composite_key = composite_pins["execution_key_cid"]
        if not composite_key or composite_key != v2_cid:
            raise SignedRunnerAttestationError(
                "composite execution_key_cid does not match TestExecutionKeyV2"
            )
        v2_locator = _public_attr(execution_key, "locator_cid")
        composite_locator = composite_pins["locator_cid"]
        if v2_locator and composite_locator and v2_locator != composite_locator:
            raise SignedRunnerAttestationError("V2 and composite locator pins disagree")
        if getattr(composite_receipt, "may_authorize_skip", False):
            raise SignedRunnerAttestationError("composite receipt must not authorize skip")
        if getattr(composite_receipt, "production_admitted", False):
            raise SignedRunnerAttestationError(
                "composite receipt must not admit production"
            )
        if getattr(execution_key, "may_authorize_skip", False):
            raise SignedRunnerAttestationError(
                "TestExecutionKeyV2 must not authorize skip"
            )
        if getattr(execution_key, "production_admitted", False):
            raise SignedRunnerAttestationError(
                "TestExecutionKeyV2 must not admit production"
            )
        return SignedRunnerAttestationBinding(
            test_execution_key_v2_cid=v2_cid,
            test_execution_key_v2_cidv1=execution_key_cidv1(v2_cid),
            composite_phase_receipt_cid=composite_cid,
            composite_phase_receipt_cidv1=composite_receipt_cidv1(composite_cid),
            verifier_key_cid=verifier.key_cid,
            verifier_issuer_id=issuer_id,
            verifier_epoch=epoch,
            verifier_policy_cid=policy_cid,
            locator_cid=v2_locator or composite_locator,
            trust_domain=verifier.trust_domain,
        )
    except SignedRunnerAttestationError:
        raise
    except (RunnerAttestationError, ValueError) as exc:
        raise SignedRunnerAttestationError(str(exc) or "binding failed") from exc


@dataclass(frozen=True, slots=True)
class SignedRunnerAttestationBinding:
    """Public V2/composite pins bound to verifier-selected trust.

    Not a pytest test class.  Contains no signature or witness bytes.
    """

    __test__ = False

    test_execution_key_v2_cid: str
    test_execution_key_v2_cidv1: str
    composite_phase_receipt_cid: str
    composite_phase_receipt_cidv1: str
    verifier_key_cid: str
    verifier_issuer_id: str
    verifier_epoch: str
    verifier_policy_cid: str
    locator_cid: str = ""
    trust_domain: str = ""
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    claim_class: str = ATTESTATION_CLAIM_CLASS

    def __post_init__(self) -> None:
        if self.may_authorize_skip:
            raise SignedRunnerAttestationError(
                "signed runner binding must not authorize skip"
            )
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise SignedRunnerAttestationError(
                "signed runner binding cannot admit production, self-approve, "
                "or change claims"
            )
        if self.claim_class != ATTESTATION_CLAIM_CLASS:
            raise SignedRunnerAttestationError(
                "signed runner binding claim class must remain SignedExecutionReceipt"
            )
        for name in (
            "test_execution_key_v2_cid",
            "test_execution_key_v2_cidv1",
            "composite_phase_receipt_cid",
            "composite_phase_receipt_cidv1",
            "verifier_key_cid",
            "verifier_issuer_id",
            "verifier_epoch",
            "verifier_policy_cid",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        if self.locator_cid:
            object.__setattr__(
                self, "locator_cid", _require_text(self.locator_cid, "locator_cid")
            )
        if self.trust_domain:
            object.__setattr__(
                self, "trust_domain", _require_text(self.trust_domain, "trust_domain")
            )

    @property
    def interface(self) -> str:
        return SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE

    @property
    def schema(self) -> str:
        return SIGNED_RUNNER_ATTESTATION_SCHEMA

    def unsigned_dict(self) -> dict[str, Any]:
        payload = {
            "claim_class": self.claim_class,
            "composite_phase_receipt_cid": self.composite_phase_receipt_cid,
            "composite_phase_receipt_cidv1": self.composite_phase_receipt_cidv1,
            "does_not": SIGNED_EXECUTION_DOES_NOT,
            "establishes": SIGNED_EXECUTION_ESTABLISHES,
            "interface": self.interface,
            "locator_cid": self.locator_cid,
            "may_authorize_skip": False,
            "predecessor_interface": PREDECESSOR_INTERFACE,
            "production_admitted": False,
            "schema": self.schema,
            "self_approved": False,
            "test_execution_key_v2_cid": self.test_execution_key_v2_cid,
            "test_execution_key_v2_cidv1": self.test_execution_key_v2_cidv1,
            "trust_domain": self.trust_domain,
            "verifier_epoch": self.verifier_epoch,
            "verifier_issuer_id": self.verifier_issuer_id,
            "verifier_key_cid": self.verifier_key_cid,
            "verifier_policy_cid": self.verifier_policy_cid,
        }
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["binding_cid"] = self.cid
        payload["claim_unchanged"] = True
        return payload

    @property
    def cid(self) -> str:
        return dag_cbor_cid(self.unsigned_dict())


def _phase_outcome(value: Any, field_name: str) -> PhaseOutcome:
    text = value.value if isinstance(value, PhaseOutcome) else str(value or "")
    try:
        return PhaseOutcome(text)
    except ValueError as exc:
        raise SignedRunnerAttestationError(f"unsupported {field_name}") from exc


def _require_complete_pass(composite_receipt: Any) -> None:
    admitted = bool(getattr(composite_receipt, "admitted", False))
    honest = bool(getattr(composite_receipt, "honest", False))
    all_pass = bool(getattr(composite_receipt, "all_phases_pass", False))
    if not (admitted and honest and all_pass):
        raise SignedRunnerAttestationError(
            "only admitted honest complete composite passes may be attested"
        )
    if getattr(composite_receipt, "disqualifying_bits", ()):
        raise SignedRunnerAttestationError(
            "disqualified composite receipts cannot be attested"
        )


def pass_receipt_from_v2_composite(
    execution_key: Any,
    composite_receipt: Any,
    verifier: VerifierSelectedTrust,
    binding: SignedRunnerAttestationBinding,
    *,
    nonce: str,
) -> TestPassReceipt:
    """Project a policy-bound TestPassReceipt@1 from V2/composite evidence."""

    _require_complete_pass(composite_receipt)
    locator = binding.locator_cid or _public_attr(execution_key, "locator_cid")
    if not locator:
        raise SignedRunnerAttestationError("locator_cid is required to attest")
    completeness = getattr(execution_key, "completeness_identity", None)
    return TestPassReceipt(
        execution_key_cid=binding.test_execution_key_v2_cidv1,
        locator_cid=locator,
        setup_outcome=_phase_outcome(
            getattr(composite_receipt, "setup_outcome", "pass"), "setup_outcome"
        ),
        call_outcome=_phase_outcome(
            getattr(composite_receipt, "call_outcome", "pass"), "call_outcome"
        ),
        teardown_outcome=_phase_outcome(
            getattr(composite_receipt, "teardown_outcome", "pass"), "teardown_outcome"
        ),
        static_trace_root_cid=_public_attr(completeness, "static_trace_root_cid"),
        runtime_trace_root_cid=_public_attr(completeness, "runtime_trace_root_cid"),
        completeness_receipt_cid=_public_attr(
            completeness, "completeness_policy_cid"
        ),
        runner_identity="runner:pytest",
        trust_domain=verifier.trust_domain,
        issuer_key_id=verifier.issuer_id,
        nonce=nonce,
        epoch_policy_id=verifier.epoch,
        policy_cid=verifier.policy_cid,
        admitted=True,
        metadata={
            "composite_phase_receipt_cid": binding.composite_phase_receipt_cid,
            "test_execution_key_v2_cid": binding.test_execution_key_v2_cid,
            "binding_cid": binding.cid,
        },
    )


@dataclass(frozen=True, slots=True)
class SignedRunnerAttestationResult:
    """Outcome of one V2/composite runner-attestation verification.

    Not a pytest test class.  ``valid`` never implies production admission,
    skip authority, or task completion.
    """

    __test__ = False

    valid: bool
    reason: str
    binding: SignedRunnerAttestationBinding | None = None
    attestation: RunnerPassAttestation | None = None
    signed_receipt: SignedTestPassReceiptV2 | None = None
    pass_receipt: TestPassReceipt | None = None
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    claim_class: str = ATTESTATION_CLAIM_CLASS

    def __post_init__(self) -> None:
        if self.may_authorize_skip:
            raise SignedRunnerAttestationError(
                "signed runner attestation must not authorize skip"
            )
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise SignedRunnerAttestationError(
                "signed runner attestation cannot admit production, "
                "self-approve, or change claims"
            )
        if self.claim_class != ATTESTATION_CLAIM_CLASS:
            raise SignedRunnerAttestationError(
                "signed runner attestation claim class must remain "
                "SignedExecutionReceipt"
            )
        object.__setattr__(self, "reason", str(self.reason or "")[:MAX_TEXT_CHARS])
        if self.valid and (
            self.binding is None
            or self.attestation is None
            or self.signed_receipt is None
            or self.pass_receipt is None
        ):
            raise SignedRunnerAttestationError(
                "valid signed runner attestation requires binding, attestation, "
                "signed receipt, and pass receipt"
            )

    @property
    def interface(self) -> str:
        return SIGNED_RUNNER_ATTESTATION_RESULT_INTERFACE

    @property
    def establishes(self) -> str:
        return SIGNED_EXECUTION_ESTABLISHES if self.valid else ""

    @property
    def does_not(self) -> str:
        return SIGNED_EXECUTION_DOES_NOT

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_class": self.claim_class,
            "claim_unchanged": True,
            "does_not": self.does_not,
            "establishes": self.establishes,
            "interface": self.interface,
            "may_authorize_skip": False,
            "production_admitted": False,
            "reason": self.reason,
            "self_approved": False,
            "valid": self.valid,
            "binding_cid": "" if self.binding is None else self.binding.cid,
            "attestation_cid": (
                "" if self.attestation is None else self.attestation.cid
            ),
        }


def _invalid(reason: str) -> SignedRunnerAttestationResult:
    return SignedRunnerAttestationResult(valid=False, reason=reason)


def attest_v2_composite_evidence(
    execution_key: Any,
    composite_receipt: Any,
    *,
    private_key: Any,
    verifier: VerifierSelectedTrust,
    issuance_nonce: str | None = None,
    issued_at: int | None = None,
    nonce_registry: AttestationNonceRegistry | None = None,
) -> tuple[SignedRunnerAttestationBinding, RunnerPassAttestation, TestPassReceipt]:
    """Sign admitted V2/composite evidence under verifier-selected trust."""

    try:
        binding = bind_v2_composite_to_verifier(
            execution_key, composite_receipt, verifier
        )
        nonce = issuance_nonce or binding.cid
        receipt = pass_receipt_from_v2_composite(
            execution_key,
            composite_receipt,
            verifier,
            binding,
            nonce=nonce,
        )
        attestation = attest_test_pass_receipt(
            receipt,
            private_key=private_key,
            policy=verifier.policy,
            candidate_context_cid=binding.cid,
            issuance_nonce=nonce,
            issued_at=issued_at,
            nonce_registry=nonce_registry,
        )
        if attestation.signer_key_cid != verifier.key_cid:
            raise SignedRunnerAttestationError(
                "attestation signer is not the verifier-selected key"
            )
        if attestation.key_epoch != verifier.epoch:
            raise SignedRunnerAttestationError(
                "attestation epoch is not the verifier-selected epoch"
            )
        if attestation.policy_cid != verifier.policy_cid:
            raise SignedRunnerAttestationError(
                "attestation policy is not the verifier-selected policy"
            )
        if attestation.execution_key_cid != binding.test_execution_key_v2_cidv1:
            raise SignedRunnerAttestationError(
                "attestation execution key is not the bound TestExecutionKeyV2"
            )
        if attestation.candidate_context_cid != binding.cid:
            raise SignedRunnerAttestationError(
                "attestation candidate context is not the V2/composite binding"
            )
        return binding, attestation, receipt
    except SignedRunnerAttestationError:
        raise
    except (RunnerAttestationError, ValueError) as exc:
        raise SignedRunnerAttestationError(str(exc) or "attestation failed") from exc


def verify_v2_composite_attestation(
    attestation: RunnerPassAttestation | bytes,
    *,
    execution_key: Any,
    composite_receipt: Any,
    verifier: VerifierSelectedTrust,
    pass_receipt: TestPassReceipt | None = None,
    now: int | None = None,
    nonce_registry: AttestationNonceRegistry | None = None,
    pinned_public_key_material: bytes | None = None,
) -> SignedRunnerAttestationResult:
    """Verify the signature and V2/composite/verifier pin agreement."""

    try:
        candidate = (
            attestation
            if isinstance(attestation, RunnerPassAttestation)
            else RunnerPassAttestation.from_bytes(attestation)
        )
        binding = bind_v2_composite_to_verifier(
            execution_key, composite_receipt, verifier
        )
    except (SignedRunnerAttestationError, RunnerAttestationError, ValueError) as exc:
        return _invalid(str(exc) or "attestation binding failed")
    if pass_receipt is None:
        try:
            pass_receipt = pass_receipt_from_v2_composite(
                execution_key,
                composite_receipt,
                verifier,
                binding,
                nonce=candidate.issuance_nonce,
            )
        except (SignedRunnerAttestationError, ValueError) as exc:
            return _invalid(str(exc) or "pass receipt reconstruction failed")
    try:
        if pinned_public_key_material is None:
            pinned_public_key_material = verifier.public_key.material
        verified: AttestationVerification = verify_runner_pass_attestation_with_key(
            candidate,
            receipt=pass_receipt,
            policy=verifier.policy,
            pinned_policy_cid=verifier.policy_cid,
            current_execution_key_cid=binding.test_execution_key_v2_cidv1,
            current_candidate_context_cid=binding.cid,
            pinned_public_key_material=pinned_public_key_material,
            now=now,
            nonce_registry=nonce_registry,
        )
    except (RunnerAttestationError, SignedRunnerAttestationError) as exc:
        return _invalid(str(exc))
    except Exception:
        return _invalid("attestation verification failed")
    if not verified.valid or verified.signed_receipt is None:
        return _invalid(verified.reason or "attestation_rejected")
    if candidate.signer_key_cid != verifier.key_cid:
        return _invalid("attestation signer is not the verifier-selected key")
    if candidate.key_epoch != verifier.epoch:
        return _invalid("attestation epoch is not the verifier-selected epoch")
    if candidate.policy_cid != verifier.policy_cid:
        return _invalid("attestation policy is not the verifier-selected policy")
    if pass_receipt.issuer_key_id != verifier.issuer_id:
        return _invalid("pass receipt issuer is not the verifier-selected issuer")
    if pass_receipt.epoch_policy_id != verifier.epoch:
        return _invalid("pass receipt epoch is not the verifier-selected epoch")
    signed = verified.signed_receipt
    if (
        signed.signer_key_cid != verifier.key_cid
        or signed.key_epoch != verifier.epoch
        or signed.trust_policy_cid != verifier.policy_cid
        or signed.execution_key_cid != binding.test_execution_key_v2_cidv1
        or signed.candidate_context_cid != binding.cid
    ):
        return _invalid("signed receipt does not bind verifier-selected trust")
    return SignedRunnerAttestationResult(
        valid=True,
        reason="verified",
        binding=binding,
        attestation=candidate,
        signed_receipt=signed,
        pass_receipt=pass_receipt,
    )


def attest_and_verify_v2_composite_evidence(
    execution_key: Any,
    composite_receipt: Any,
    *,
    private_key: Any,
    verifier: VerifierSelectedTrust,
    issuance_nonce: str | None = None,
    issued_at: int | None = None,
    nonce_registry: AttestationNonceRegistry | None = None,
    item: Any = None,
) -> SignedRunnerAttestationResult:
    """Sign then locally verify.  Cryptographic self-check is not task approval."""

    try:
        binding, attestation, receipt = attest_v2_composite_evidence(
            execution_key,
            composite_receipt,
            private_key=private_key,
            verifier=verifier,
            issuance_nonce=issuance_nonce,
            issued_at=issued_at,
            nonce_registry=nonce_registry,
        )
    except SignedRunnerAttestationError as exc:
        return _invalid(str(exc))
    result = verify_v2_composite_attestation(
        attestation,
        execution_key=execution_key,
        composite_receipt=composite_receipt,
        verifier=verifier,
        pass_receipt=receipt,
        now=issued_at,
        nonce_registry=nonce_registry,
        pinned_public_key_material=verifier.public_key.material,
    )
    if result.valid:
        attach_signed_runner_attestation(item, result)
    return result


def attach_signed_runner_attestation(item: Any, result: SignedRunnerAttestationResult) -> None:
    """Attach a verified public binding to *item* without skip authority."""

    if item is None or not result.valid:
        return
    if result.may_authorize_skip or result.production_admitted or result.self_approved:
        raise SignedRunnerAttestationError(
            "attached signed runner attestation cannot skip, admit, or self-approve"
        )
    try:
        setattr(item, ITEM_SIGNED_RUNNER_RESULT_ATTRIBUTE, result)
        if result.attestation is not None:
            setattr(item, ITEM_SIGNED_RUNNER_ATTESTATION_ATTRIBUTE, result.attestation)
        if result.binding is not None:
            setattr(item, ITEM_SIGNED_RUNNER_BINDING_ATTRIBUTE, result.binding)
    except Exception:
        return


def get_attached_signed_runner_result(item: Any) -> SignedRunnerAttestationResult | None:
    """Return the attached verification result, if any."""

    existing = getattr(item, ITEM_SIGNED_RUNNER_RESULT_ATTRIBUTE, None)
    if isinstance(existing, SignedRunnerAttestationResult):
        return existing
    return None


def signed_evidence_claim() -> dict[str, str]:
    """Return the SignedExecutionReceipt claim boundary."""

    return {
        "claim_class": ATTESTATION_CLAIM_CLASS,
        "establishes": SIGNED_EXECUTION_ESTABLISHES,
        "does_not": SIGNED_EXECUTION_DOES_NOT,
        "production_admitted": "false",
        "self_approved": "false",
    }


__all__ = [
    "ATTESTATION_CLAIM_CLASS",
    "ATTESTATION_DOES_NOT",
    "ATTESTATION_ESTABLISHES",
    "ATTESTATION_POLICY",
    "CLAIM_CLASS",
    "DEFAULT_POLICY_CID",
    "ITEM_SIGNED_RUNNER_ATTESTATION_ATTRIBUTE",
    "ITEM_SIGNED_RUNNER_BINDING_ATTRIBUTE",
    "ITEM_SIGNED_RUNNER_RESULT_ATTRIBUTE",
    "PREDECESSOR_INTERFACE",
    "SIGNED_EXECUTION_DOES_NOT",
    "SIGNED_EXECUTION_ESTABLISHES",
    "SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE",
    "SIGNED_RUNNER_ATTESTATION_POLICY_INTERFACE",
    "SIGNED_RUNNER_ATTESTATION_RESULT_INTERFACE",
    "SignedRunnerAttestationBinding",
    "SignedRunnerAttestationError",
    "SignedRunnerAttestationResult",
    "VerifierSelectedTrust",
    "attach_signed_runner_attestation",
    "attest_and_verify_v2_composite_evidence",
    "attest_v2_composite_evidence",
    "authority_descriptor",
    "bind_v2_composite_to_verifier",
    "canonical_public_bytes",
    "composite_phase_receipt_cid",
    "composite_receipt_cidv1",
    "datasets_contracts_available",
    "execution_key_cidv1",
    "get_attached_signed_runner_result",
    "public_digest",
    "record_typed_unavailable",
    "select_verifier_trust",
    "signed_evidence_claim",
    "pass_receipt_from_v2_composite",
    "typed_unavailable_records",
    "v2_execution_key_cid",
    "verify_v2_composite_attestation",
]
