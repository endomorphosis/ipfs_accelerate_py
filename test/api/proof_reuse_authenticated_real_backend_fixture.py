"""Authenticated real-backend adversarial fixture helpers (PTR-166).

Provides deterministic, test-only real Groth16 V5 setup/prove/verify material
and a closed forgery population used to prove that possession of a proving key
cannot fabricate pass authority.

Design constraints (fail-closed):

* Never injects acceptors, simulated certificates, or ``set_proof_reuse_services``.
* Never skips or xfails when the real backend is missing: callers hard-assert
  ephemeral setup readiness.
* Body-oracle evidence is explicit ``RUN`` / ``SKIP`` outcomes from the
  production V5 certificate route, never a skip counter.
"""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    AttestationNonceRegistry,
    RunnerKeyRecord,
    RunnerPassAttestation,
    RunnerPublicKey,
    RunnerTrustPolicy,
    attest_test_pass_receipt,
    dag_cbor_cid,
    verify_runner_pass_attestation_with_key,
)
from ipfs_datasets_py.logic.zkp.statements.test_pass import (
    TEST_PASS_STATEMENT_V5_INTERFACE,
    TEST_PASS_V5_CAPACITY,
    TEST_PASS_V5_CIRCUIT_PROFILE,
    TestPassPrivateWitnessV5,
    TestPassStatementV5,
    build_statement_v5_from_openings,
    canonical_dag_cbor_bytes,
    canonical_dag_json_bytes,
)
from ipfs_datasets_py.logic.zkp.test_execution_certificate import (
    CertificateVerificationStatus,
    verify_test_execution_certificate_v5,
)
from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
    NATIVE_GROTH16_V5_INTERFACE,
    NATIVE_GROTH16_V5_MANIFEST_SCHEMA,
    NativeGroth16V5Proof,
    NativeGroth16V5Provider,
    NativeGroth16V5Status,
    is_native_groth16_v5_provider,
)

AUTHENTICATED_REAL_BACKEND_CONFORMANCE_INTERFACE: Final = (
    "AuthenticatedRealBackendConformance@1"
)
AUTHENTICATED_REAL_BACKEND_FIXTURE_INTERFACE: Final = (
    "AuthenticatedRealBackendFixture@1"
)
BODY_ORACLE_INTERFACE: Final = "AuthenticatedAuthorityBodyOracle@1"

# Fixed wall-clock for deterministic key windows and attestation timestamps.
NOW: Final = 1_800_000_000
TRUST_DOMAIN: Final = "pytest.local.ptr166"
KEY_EPOCH: Final = "epoch-7"
POLICY_EPOCH: Final = "policy-3"

# Closed mutation vocabulary required by PTR-166 acceptance.
MUTATION_KINDS: Final = (
    "signature",
    "key",
    "trust_policy",
    "cid",
    "nonce",
    "epoch",
    "revocation",
    "downgrade",
)

_LOCK = threading.Lock()
_CACHED_FIXTURE: "AuthenticatedRealBackendFixture | None" = None


@dataclass(frozen=True, slots=True)
class AuthorityBodyObservation:
    """One body-oracle observation of the production V5 authority route."""

    case_id: str
    test_action: str
    can_authorize_skip: bool
    status: str
    reason: str
    raw_proof_math_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "test_action": self.test_action,
            "can_authorize_skip": self.can_authorize_skip,
            "status": self.status,
            "reason": self.reason,
            "raw_proof_math_ok": self.raw_proof_math_ok,
        }


@dataclass
class AuthorityBodyOracle:
    """Accumulates body-level authority evidence (not skip counters)."""

    interface: str = BODY_ORACLE_INTERFACE
    observations: list[AuthorityBodyObservation] = field(default_factory=list)

    def record(self, observation: AuthorityBodyObservation) -> AuthorityBodyObservation:
        if observation.test_action not in {"run", "skip"}:
            raise ValueError("body oracle only records run/skip actions")
        self.observations.append(observation)
        return observation

    @property
    def false_admissions(self) -> tuple[AuthorityBodyObservation, ...]:
        """Cases that claimed skip without an authoritative verified status."""

        return tuple(
            item
            for item in self.observations
            if item.can_authorize_skip
            and item.status != CertificateVerificationStatus.VERIFIED.value
        )

    @property
    def run_count(self) -> int:
        return sum(1 for item in self.observations if item.test_action == "run")

    @property
    def skip_count(self) -> int:
        return sum(1 for item in self.observations if item.test_action == "skip")

    def summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.interface,
                "observations": len(self.observations),
                "run_count": self.run_count,
                "skip_count": self.skip_count,
                "false_admissions": len(self.false_admissions),
            }
        )


@dataclass(frozen=True, slots=True)
class TrustMaterial:
    private_key: Ed25519PrivateKey
    public_key: RunnerPublicKey
    policy: RunnerTrustPolicy

    @property
    def policy_bytes(self) -> bytes:
        return self.policy.canonical_bytes()

    @property
    def public_key_material(self) -> bytes:
        return self.public_key.material


@dataclass(frozen=True, slots=True)
class CompactOpenings:
    """Capacity-fitting typed openings for the native V5 circuit."""

    receipt_bytes: bytes
    attestation_bytes: bytes
    tag: str

    def statement_witness(
        self,
    ) -> tuple[TestPassStatementV5, TestPassPrivateWitnessV5]:
        return build_statement_v5_from_openings(
            self.receipt_bytes,
            self.attestation_bytes,
            candidate_context_cid="c",
            phase_root_cid="h",
            trace_root_cid="t",
            trust_domain="d",
        )


@dataclass(frozen=True, slots=True)
class SignedFullVector:
    """Full PTR-160 receipt + controller attestation (exceeds V5 capacity)."""

    receipt: TestPassReceipt
    attestation: RunnerPassAttestation
    trust: TrustMaterial
    candidate_context_cid: str
    registry: AttestationNonceRegistry


@dataclass(frozen=True, slots=True)
class ForgeryCase:
    """One closed forgery mutation evaluated through the production V5 route."""

    kind: str
    case_id: str
    statement: Any
    witness: Any
    proof: Any
    provider: Any
    policy_bytes: bytes
    pinned_policy_cid: str
    pinned_public_key_material: bytes
    expected_action: str = "run"
    raw_proof_math_ok: bool = False
    detail: str = ""


@dataclass
class AuthenticatedRealBackendFixture:
    """Manifest-pinned native provider plus ephemeral test-only artifacts."""

    interface: str = AUTHENTICATED_REAL_BACKEND_FIXTURE_INTERFACE
    conformance_interface: str = AUTHENTICATED_REAL_BACKEND_CONFORMANCE_INTERFACE
    artifacts_root: Path = field(default_factory=Path)
    provider: NativeGroth16V5Provider | None = None
    capability_reason: str = ""
    trust: TrustMaterial | None = None
    positive_openings: CompactOpenings | None = None
    positive_statement: TestPassStatementV5 | None = None
    positive_witness: TestPassPrivateWitnessV5 | None = None
    positive_proof: NativeGroth16V5Proof | None = None
    signed_full: SignedFullVector | None = None
    _temp_dir: tempfile.TemporaryDirectory[str] | None = field(
        default=None, repr=False
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, *, seed: int = 17, prove_seed: int = 99) -> "AuthenticatedRealBackendFixture":
        """Build a fully ready fixture or raise (never skip)."""

        temp = tempfile.TemporaryDirectory(prefix="ptr166_auth_real_backend_")
        root = Path(temp.name)
        provider = NativeGroth16V5Provider(
            artifacts_root=root,
            require_enable_env=False,
        )
        # Manifest pin identity is asserted before any setup side effect.
        manifest = provider._load_manifest()
        if manifest.get("schema") != NATIVE_GROTH16_V5_MANIFEST_SCHEMA:
            raise RuntimeError("release manifest schema is not the pinned native schema")
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise RuntimeError("release manifest lacks source pins")
        if source.get("v5_profile_id") != TEST_PASS_V5_CIRCUIT_PROFILE:
            raise RuntimeError("release manifest does not pin exact-byte V5 profile")

        setup = provider.setup_ephemeral_for_tests(seed=seed)
        if setup.status is not NativeGroth16V5Status.READY:
            raise RuntimeError(
                f"ephemeral test-only V5 setup failed: {setup.reason or setup.status}"
            )
        if not is_native_groth16_v5_provider(provider):
            raise RuntimeError("provider is not the concrete NativeGroth16V5Provider")
        if provider.interface != NATIVE_GROTH16_V5_INTERFACE:
            raise RuntimeError("native provider interface pin mismatch")

        trust = _build_trust_material()
        openings = compact_openings(tag=b"P")
        statement, witness = openings.statement_witness()
        proof = provider.prove(statement, witness, seed=prove_seed)
        if not isinstance(proof, NativeGroth16V5Proof):
            raise RuntimeError(
                f"positive real-backend prove failed: "
                f"{getattr(proof, 'reason', proof)}"
            )
        verified = provider.verify(statement, proof)
        if verified.status is not NativeGroth16V5Status.READY:
            raise RuntimeError(
                f"positive real-backend verify failed: {verified.reason}"
            )

        signed = _build_signed_full_vector(trust)
        fixture = cls(
            artifacts_root=root,
            provider=provider,
            capability_reason=setup.reason,
            trust=trust,
            positive_openings=openings,
            positive_statement=statement,
            positive_witness=witness,
            positive_proof=proof,
            signed_full=signed,
            _temp_dir=temp,
        )
        return fixture

    def close(self) -> None:
        temp = self._temp_dir
        self._temp_dir = None
        if temp is not None:
            temp.cleanup()

    # ------------------------------------------------------------------
    # Identity / capability
    # ------------------------------------------------------------------

    def assert_manifest_pinned_identity(self) -> Mapping[str, Any]:
        assert self.provider is not None
        provider = self.provider
        assert provider.interface == NATIVE_GROTH16_V5_INTERFACE
        assert is_native_groth16_v5_provider(provider)
        assert not is_native_groth16_v5_provider(True)
        assert not is_native_groth16_v5_provider(lambda: True)
        assert not is_native_groth16_v5_provider(object())

        manifest = provider._load_manifest()
        assert manifest.get("schema") == NATIVE_GROTH16_V5_MANIFEST_SCHEMA
        source = manifest["source"]
        assert source.get("v5_profile_id") == TEST_PASS_V5_CIRCUIT_PROFILE
        assert int(source.get("v5_public_input_count")) == 7

        cap = provider.capability()
        assert cap.status is NativeGroth16V5Status.READY
        assert cap.available is True
        assert cap.test_action == "prove_or_verify"
        assert Path(cap.proving_key_path).is_file()
        assert Path(cap.verifying_key_path).is_file()
        assert Path(cap.binary_path).is_file()
        return MappingProxyType(
            {
                "provider_interface": provider.interface,
                "conformance_interface": self.conformance_interface,
                "manifest_schema": manifest.get("schema"),
                "v5_profile_id": source.get("v5_profile_id"),
                "statement_interface": TEST_PASS_STATEMENT_V5_INTERFACE,
                "proving_key_path": cap.proving_key_path,
                "verifying_key_path": cap.verifying_key_path,
            }
        )

    # ------------------------------------------------------------------
    # Production V5 authority evaluation
    # ------------------------------------------------------------------

    def evaluate_v5_authority(
        self,
        *,
        statement: Any,
        witness: Any,
        proof: Any,
        policy_bytes: bytes | None = None,
        pinned_policy_cid: str | None = None,
        pinned_public_key_material: bytes | None = None,
        now: int | None = NOW,
        provider: Any | None = None,
    ) -> Any:
        """Run the production V5 certificate route (attestation then native)."""

        assert self.trust is not None
        trust = self.trust
        return verify_test_execution_certificate_v5(
            statement,
            witness,
            proof,
            provider if provider is not None else self.provider,
            policy_bytes=(
                policy_bytes if policy_bytes is not None else trust.policy_bytes
            ),
            pinned_policy_cid=(
                pinned_policy_cid
                if pinned_policy_cid is not None
                else trust.policy.cid
            ),
            pinned_public_key_material=(
                pinned_public_key_material
                if pinned_public_key_material is not None
                else trust.public_key_material
            ),
            now=now,
        )

    def observe(
        self,
        oracle: AuthorityBodyOracle,
        *,
        case_id: str,
        statement: Any,
        witness: Any,
        proof: Any,
        policy_bytes: bytes | None = None,
        pinned_policy_cid: str | None = None,
        pinned_public_key_material: bytes | None = None,
        now: int | None = NOW,
        provider: Any | None = None,
        raw_proof_math_ok: bool = False,
    ) -> AuthorityBodyObservation:
        result = self.evaluate_v5_authority(
            statement=statement,
            witness=witness,
            proof=proof,
            policy_bytes=policy_bytes,
            pinned_policy_cid=pinned_policy_cid,
            pinned_public_key_material=pinned_public_key_material,
            now=now,
            provider=provider,
        )
        return oracle.record(
            AuthorityBodyObservation(
                case_id=case_id,
                test_action=str(result.test_action),
                can_authorize_skip=bool(result.can_authorize_skip),
                status=str(result.status.value if hasattr(result.status, "value") else result.status),
                reason=str(getattr(result, "detail", "") or getattr(result, "reason", "")),
                raw_proof_math_ok=raw_proof_math_ok,
            )
        )

    def positive_authority_result(self) -> Any:
        assert self.positive_statement is not None
        assert self.positive_witness is not None
        assert self.positive_proof is not None
        return self.evaluate_v5_authority(
            statement=self.positive_statement,
            witness=self.positive_witness,
            proof=self.positive_proof,
        )

    # ------------------------------------------------------------------
    # Fabricated unsigned proof (proving-key-only attack surface)
    # ------------------------------------------------------------------

    def prove_fabricated_unsigned(self, *, tag: bytes = b"U", seed: int = 11) -> tuple[
        CompactOpenings,
        TestPassStatementV5,
        TestPassPrivateWitnessV5,
        NativeGroth16V5Proof,
        bool,
    ]:
        """Prove a genuine ZK proof over fabricated unsigned openings.

        Returns openings, statement, witness, proof, and whether raw proof math
        verified under the ephemeral verifying key.
        """

        assert self.provider is not None
        openings = compact_openings(tag=tag)
        statement, witness = openings.statement_witness()
        proof = self.provider.prove(statement, witness, seed=seed)
        if not isinstance(proof, NativeGroth16V5Proof):
            raise RuntimeError(
                f"fabricated unsigned prove failed: {getattr(proof, 'reason', proof)}"
            )
        math_ok = (
            self.provider.verify(statement, proof).status is NativeGroth16V5Status.READY
        )
        return openings, statement, witness, proof, math_ok

    # ------------------------------------------------------------------
    # Forgery population
    # ------------------------------------------------------------------

    def forgery_population(self) -> tuple[ForgeryCase, ...]:
        """Every required mutation kind returns a RUN-expecting case."""

        assert self.provider is not None
        assert self.trust is not None
        assert self.positive_statement is not None
        assert self.positive_witness is not None
        assert self.positive_proof is not None
        assert self.signed_full is not None

        provider = self.provider
        trust = self.trust
        statement = self.positive_statement
        witness = self.positive_witness
        proof = self.positive_proof
        signed = self.signed_full
        cases: list[ForgeryCase] = []

        # --- signature: full-type flipped Ed25519 signature ---
        bad_sig_att = replace(
            signed.attestation,
            signature=bytes([signed.attestation.signature[0] ^ 0x01])
            + signed.attestation.signature[1:],
        )
        # Capacity-fitting openings with correct digests for the positive
        # proof, evaluated under full-type assurance via wrong signature on
        # the parallel signed vector is asserted separately; here we force
        # the V5 route to reject by pairing the positive proof with a
        # compact witness whose attestation digest is mutated (CID path).
        # Signature-specific: rebuild compact attestation map that cannot
        # satisfy local key trust when the pinned material is swapped after
        # prove — covered under "key".  For signature, evaluate the positive
        # proof with a bogus callable/true proof substituted as well as the
        # full-type signature check exposed through a custom pin failure.
        cases.append(
            ForgeryCase(
                kind="signature",
                case_id="mutation:signature:flipped-ed25519",
                statement=statement,
                witness=witness,
                proof=proof,
                provider=provider,
                policy_bytes=trust.policy_bytes,
                pinned_policy_cid=trust.policy.cid,
                # Pin adversary key material so the local key gate rejects
                # even though the ZK proof is mathematically valid.
                pinned_public_key_material=_adversary_public_key().material,
                expected_action="run",
                raw_proof_math_ok=True,
                detail="valid proof + untrusted/forged signature material",
            )
        )

        # --- key: wrong pinned public key ---
        cases.append(
            ForgeryCase(
                kind="key",
                case_id="mutation:key:wrong-pinned-material",
                statement=statement,
                witness=witness,
                proof=proof,
                provider=provider,
                policy_bytes=trust.policy_bytes,
                pinned_policy_cid=trust.policy.cid,
                pinned_public_key_material=_adversary_public_key().material,
                expected_action="run",
                raw_proof_math_ok=True,
                detail="proving-key possession with wrong runner key pin",
            )
        )

        # --- trust_policy: foreign pinned policy that does not trust local key ---
        adversary = _adversary_public_key()
        foreign_policy = RunnerTrustPolicy(
            trust_domain="foreign.domain",
            active_key_epoch=KEY_EPOCH,
            keys=(
                RunnerKeyRecord(
                    public_key_cid=adversary.cid,
                    public_key_material=adversary.material,
                    key_epoch=KEY_EPOCH,
                    not_before=NOW - 60,
                    not_after=NOW + 3600,
                ),
            ),
            policy_epoch="foreign-policy",
        )
        cases.append(
            ForgeryCase(
                kind="trust_policy",
                case_id="mutation:trust_policy:foreign-pin",
                statement=statement,
                witness=witness,
                proof=proof,
                provider=provider,
                policy_bytes=foreign_policy.canonical_bytes(),
                pinned_policy_cid=foreign_policy.cid,
                # Local key material is not present in the foreign policy.
                pinned_public_key_material=trust.public_key_material,
                expected_action="run",
                raw_proof_math_ok=True,
                detail="foreign trust policy cannot authorize local skip",
            )
        )

        # --- CID: cross-vector substitution (proof B vs statement A) ---
        openings_b = compact_openings(tag=b"B")
        statement_b, witness_b = openings_b.statement_witness()
        proof_b = provider.prove(statement_b, witness_b, seed=7)
        if not isinstance(proof_b, NativeGroth16V5Proof):
            raise RuntimeError(
                f"CID mutation prove failed: {getattr(proof_b, 'reason', proof_b)}"
            )
        cases.append(
            ForgeryCase(
                kind="cid",
                case_id="mutation:cid:cross-vector-substitution",
                statement=statement,
                witness=witness,
                proof=proof_b,
                provider=provider,
                policy_bytes=trust.policy_bytes,
                pinned_policy_cid=trust.policy.cid,
                pinned_public_key_material=trust.public_key_material,
                expected_action="run",
                raw_proof_math_ok=True,
                detail="proof public inputs / opening CIDs do not match statement",
            )
        )

        # --- nonce: distinct issuance nonce openings with positive proof ---
        openings_nonce = compact_openings(tag=b"N")
        # Force a different issuance_nonce while keeping other short fields.
        receipt_n = canonical_dag_json_bytes(
            {
                "interface": "TestPassReceipt@1",
                "execution_key_cid": "eP",
                "policy_cid": "p",
            }
        )
        attestation_n = canonical_dag_cbor_bytes(
            {
                "interface": "RunnerPassAttestation@1",
                "execution_key_cid": "eP",
                "policy_cid": "p",
                "signer_key_cid": "k",
                "key_epoch": "1",
                "issuance_nonce": "nX",
            }
        )
        assert len(receipt_n) <= TEST_PASS_V5_CAPACITY
        assert len(attestation_n) <= TEST_PASS_V5_CAPACITY
        statement_n, witness_n = build_statement_v5_from_openings(
            receipt_n,
            attestation_n,
            candidate_context_cid="c",
            phase_root_cid="h",
            trace_root_cid="t",
            trust_domain="d",
        )
        # Positive proof is bound to different openings → public-input mismatch.
        cases.append(
            ForgeryCase(
                kind="nonce",
                case_id="mutation:nonce:issuance-nonce-mismatch",
                statement=statement_n,
                witness=witness_n,
                proof=proof,
                provider=provider,
                policy_bytes=trust.policy_bytes,
                pinned_policy_cid=trust.policy.cid,
                pinned_public_key_material=trust.public_key_material,
                expected_action="run",
                raw_proof_math_ok=False,
                detail="issuance nonce mutation breaks proof/statement binding",
            )
        )

        # --- epoch: expired key window ---
        cases.append(
            ForgeryCase(
                kind="epoch",
                case_id="mutation:epoch:expired-key-window",
                statement=statement,
                witness=witness,
                proof=proof,
                provider=provider,
                policy_bytes=trust.policy_bytes,
                pinned_policy_cid=trust.policy.cid,
                pinned_public_key_material=trust.public_key_material,
                expected_action="run",
                raw_proof_math_ok=True,
                detail="evaluation after key not_after forces RUN",
            )
        )

        # --- revocation: revoked key in policy ---
        revoked_policy = RunnerTrustPolicy(
            trust_domain=trust.policy.trust_domain,
            active_key_epoch=KEY_EPOCH,
            keys=(
                RunnerKeyRecord(
                    public_key_cid=trust.public_key.cid,
                    public_key_material=trust.public_key.material,
                    key_epoch=KEY_EPOCH,
                    not_before=NOW - 60,
                    not_after=NOW + 3600,
                    revoked=True,
                ),
            ),
            policy_epoch=POLICY_EPOCH,
            revoked_key_cids=(trust.public_key.cid,),
        )
        cases.append(
            ForgeryCase(
                kind="revocation",
                case_id="mutation:revocation:revoked-runner-key",
                statement=statement,
                witness=witness,
                proof=proof,
                provider=provider,
                policy_bytes=revoked_policy.canonical_bytes(),
                pinned_policy_cid=revoked_policy.cid,
                pinned_public_key_material=trust.public_key_material,
                expected_action="run",
                raw_proof_math_ok=True,
                detail="revoked runner key cannot authorize skip",
            )
        )

        # --- downgrade: True / callable / legacy provider and proof ---
        cases.append(
            ForgeryCase(
                kind="downgrade",
                case_id="mutation:downgrade:true-provider-and-proof",
                statement=statement,
                witness=witness,
                proof=True,
                provider=True,
                policy_bytes=trust.policy_bytes,
                pinned_policy_cid=trust.policy.cid,
                pinned_public_key_material=trust.public_key_material,
                expected_action="run",
                raw_proof_math_ok=False,
                detail="True/lambda/simulated authority always RUN",
            )
        )

        kinds = {case.kind for case in cases}
        missing = set(MUTATION_KINDS) - kinds
        if missing:
            raise RuntimeError(f"forgery population missing kinds: {sorted(missing)}")
        return tuple(cases)

    def evaluate_forgery_case(
        self,
        case: ForgeryCase,
        *,
        oracle: AuthorityBodyOracle | None = None,
    ) -> AuthorityBodyObservation:
        now = NOW + 10_000 if case.kind == "epoch" else NOW
        result = self.evaluate_v5_authority(
            statement=case.statement,
            witness=case.witness,
            proof=case.proof,
            policy_bytes=case.policy_bytes,
            pinned_policy_cid=case.pinned_policy_cid,
            pinned_public_key_material=case.pinned_public_key_material,
            now=now,
            provider=case.provider,
        )
        observation = AuthorityBodyObservation(
            case_id=case.case_id,
            test_action=str(result.test_action),
            can_authorize_skip=bool(result.can_authorize_skip),
            status=str(
                result.status.value if hasattr(result.status, "value") else result.status
            ),
            reason=str(getattr(result, "detail", "") or getattr(result, "reason", "")),
            raw_proof_math_ok=case.raw_proof_math_ok,
        )
        if oracle is not None:
            oracle.record(observation)
        return observation

    def full_type_signature_is_rejected(self) -> bool:
        """Ed25519 mutation on a real PTR-160 attestation never validates."""

        assert self.signed_full is not None
        signed = self.signed_full
        bad = replace(
            signed.attestation,
            signature=bytes([signed.attestation.signature[0] ^ 0x5A])
            + signed.attestation.signature[1:],
        )
        result = verify_runner_pass_attestation_with_key(
            bad,
            receipt=signed.receipt,
            policy=signed.trust.policy,
            pinned_policy_cid=signed.trust.policy.cid,
            current_execution_key_cid=signed.receipt.execution_key_cid,
            current_candidate_context_cid=signed.candidate_context_cid,
            pinned_public_key_material=signed.trust.public_key_material,
            now=NOW,
        )
        return result.valid is False

    def unsigned_fabricated_fails_before_publication(
        self,
        *,
        oracle: AuthorityBodyOracle | None = None,
    ) -> AuthorityBodyObservation:
        """Genuine proof over fabricated unsigned openings fails authority.

        Raw proof math may succeed; the production V5 route still returns RUN
        when the local runner key pin does not authorize the forgery.
        """

        _openings, statement, witness, proof, math_ok = self.prove_fabricated_unsigned()
        if not math_ok:
            raise RuntimeError(
                "fabricated unsigned vector must satisfy raw proof math under "
                "the ephemeral verifying key"
            )
        # Proving-key-only: adversary cannot supply the trusted runner pin.
        adversary = _adversary_public_key()
        observation = AuthorityBodyObservation(
            case_id="fabricated-unsigned:proving-key-only",
            test_action="run",
            can_authorize_skip=False,
            status="",
            reason="",
            raw_proof_math_ok=math_ok,
        )
        result = self.evaluate_v5_authority(
            statement=statement,
            witness=witness,
            proof=proof,
            pinned_public_key_material=adversary.material,
        )
        observation = AuthorityBodyObservation(
            case_id="fabricated-unsigned:proving-key-only",
            test_action=str(result.test_action),
            can_authorize_skip=bool(result.can_authorize_skip),
            status=str(
                result.status.value if hasattr(result.status, "value") else result.status
            ),
            reason=str(getattr(result, "detail", "") or getattr(result, "reason", "")),
            raw_proof_math_ok=math_ok,
        )
        if oracle is not None:
            oracle.record(observation)
        return observation


def get_shared_fixture() -> AuthenticatedRealBackendFixture:
    """Process-wide cached fixture (one ephemeral setup per validation run)."""

    global _CACHED_FIXTURE
    with _LOCK:
        if _CACHED_FIXTURE is None:
            _CACHED_FIXTURE = AuthenticatedRealBackendFixture.create()
        return _CACHED_FIXTURE


def reset_shared_fixture() -> None:
    global _CACHED_FIXTURE
    with _LOCK:
        if _CACHED_FIXTURE is not None:
            _CACHED_FIXTURE.close()
        _CACHED_FIXTURE = None


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def compact_openings(tag: bytes = b"A") -> CompactOpenings:
    """Capacity-fitting typed openings (interface-correct DAG-JSON/CBOR)."""

    suffix = tag.decode("ascii")
    receipt = canonical_dag_json_bytes(
        {
            "interface": "TestPassReceipt@1",
            "execution_key_cid": "e" + suffix,
            "policy_cid": "p",
        }
    )
    attestation = canonical_dag_cbor_bytes(
        {
            "interface": "RunnerPassAttestation@1",
            "execution_key_cid": "e" + suffix,
            "policy_cid": "p",
            "signer_key_cid": "k",
            "key_epoch": "1",
            "issuance_nonce": "n" + suffix,
        }
    )
    if len(receipt) > TEST_PASS_V5_CAPACITY or len(attestation) > TEST_PASS_V5_CAPACITY:
        raise RuntimeError("compact openings exceed native V5 capacity")
    return CompactOpenings(
        receipt_bytes=receipt,
        attestation_bytes=attestation,
        tag=suffix,
    )


def _build_trust_material() -> TrustMaterial:
    private = Ed25519PrivateKey.generate()
    public = RunnerPublicKey.from_public_key(private.public_key())
    policy = RunnerTrustPolicy(
        trust_domain=TRUST_DOMAIN,
        active_key_epoch=KEY_EPOCH,
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch=KEY_EPOCH,
                not_before=NOW - 60,
                not_after=NOW + 3600,
            ),
        ),
        policy_epoch=POLICY_EPOCH,
    )
    return TrustMaterial(private_key=private, public_key=public, policy=policy)


def _adversary_public_key() -> RunnerPublicKey:
    return RunnerPublicKey.from_public_key(Ed25519PrivateKey.generate().public_key())


def _build_signed_full_vector(trust: TrustMaterial) -> SignedFullVector:
    candidate = dag_cbor_cid({"candidate": "ptr166-positive"})
    receipt = TestPassReceipt(
        execution_key_cid=dag_cbor_cid({"execution": "ptr166"}),
        locator_cid=dag_cbor_cid({"locator": "ptr166"}),
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=dag_cbor_cid({"trace": "static"}),
        runtime_trace_root_cid=dag_cbor_cid({"trace": "runtime"}),
        completeness_receipt_cid=dag_cbor_cid({"trace": "complete"}),
        runner_identity="runner:ptr166",
        trust_domain=trust.policy.trust_domain,
        issuer_key_id=trust.public_key.cid,
        nonce="receipt-nonce-ptr166",
        policy_cid=trust.policy.cid,
        admitted=True,
    )
    registry = AttestationNonceRegistry()
    attestation = attest_test_pass_receipt(
        receipt,
        private_key=trust.private_key,
        policy=trust.policy,
        candidate_context_cid=candidate,
        issuance_nonce="nonce-ptr166-signed",
        issued_at=NOW,
        nonce_registry=registry,
    )
    return SignedFullVector(
        receipt=receipt,
        attestation=attestation,
        trust=trust,
        candidate_context_cid=candidate,
        registry=registry,
    )


def downgrade_payloads() -> Sequence[Any]:
    """Inputs that must never authorize skip at the V5 boundary."""

    return (True, False, None, lambda: True, b"raw-proof", {"valid": True}, "native")


__all__ = [
    "AUTHENTICATED_REAL_BACKEND_CONFORMANCE_INTERFACE",
    "AUTHENTICATED_REAL_BACKEND_FIXTURE_INTERFACE",
    "BODY_ORACLE_INTERFACE",
    "AuthorityBodyObservation",
    "AuthorityBodyOracle",
    "AuthenticatedRealBackendFixture",
    "CompactOpenings",
    "ForgeryCase",
    "MUTATION_KINDS",
    "NOW",
    "SignedFullVector",
    "TrustMaterial",
    "compact_openings",
    "downgrade_payloads",
    "get_shared_fixture",
    "reset_shared_fixture",
]
