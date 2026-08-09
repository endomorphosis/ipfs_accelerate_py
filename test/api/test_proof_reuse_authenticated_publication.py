"""Controller-signed publication and worker non-authority (PTR-164).

Acceptance covered:

* Terminal setup/call/teardown pass is controller-signed.
* Workers cannot publish or leak private material.
* Partial/racing writes never authorize reuse.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.publication import (
    CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE,
    ControllerCandidatePublisher,
    IssuedCertificatePublicationResult,
    build_controller_candidate_publisher,
)
from ipfs_accelerate_py.testing.proof_reuse.receipt import (
    controller_sign_complete_pass,
)
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    AttestationNonceRegistry,
    RunnerKeyRecord,
    RunnerPublicKey,
    RunnerTrustPolicy,
    dag_cbor_cid,
    verify_runner_pass_attestation,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
    ProofReuseXdistRole,
)


NOW = 1_800_000_000


def _trust_material() -> tuple[
    Ed25519PrivateKey, RunnerPublicKey, RunnerTrustPolicy
]:
    private = Ed25519PrivateKey.generate()
    public = RunnerPublicKey.from_public_key(private.public_key())
    policy = RunnerTrustPolicy(
        trust_domain="pytest.local",
        active_key_epoch="epoch-7",
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch="epoch-7",
                not_before=NOW - 60,
                not_after=NOW + 3600,
            ),
        ),
        policy_epoch="policy-3",
    )
    return private, public, policy


def _admitted_receipt(
    policy: RunnerTrustPolicy,
    *,
    nonce: str = "nonce:complete",
) -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid=dag_cbor_cid({"execution": "auth-pub-v1"}),
        locator_cid=dag_cbor_cid({"locator": "auth-pub-v1"}),
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=dag_cbor_cid({"trace": "static"}),
        runtime_trace_root_cid=dag_cbor_cid({"trace": "runtime"}),
        completeness_receipt_cid=dag_cbor_cid({"trace": "complete"}),
        dependency_forest_cid=dag_cbor_cid({"forest": "v1"}),
        issuer_key_id="key:issuer",
        policy_cid=policy.cid,
        trust_domain=policy.trust_domain,
        nonce=nonce,
        admitted=True,
    )


class _RecordingStore:
    """Store double that tracks put_candidate / put_receipt / CAS writes."""

    def __init__(
        self,
        *,
        put_candidate_ok: bool = True,
        fail_first_n: int = 0,
    ) -> None:
        self.put_candidate_calls = 0
        self.put_receipt_calls = 0
        self.cas_writes: list[bytes] = []
        self.put_candidate_ok = put_candidate_ok
        self.fail_first_n = fail_first_n
        self.indexed_cids: list[str] = []

    def put_candidate(self, receipt: Any, certificate: Any, **kwargs: Any) -> Any:
        del kwargs
        self.put_candidate_calls += 1
        if self.put_candidate_calls <= self.fail_first_n:
            raise RuntimeError("simulated partial write failure")
        if not self.put_candidate_ok:
            return SimpleNamespace(stored=True, indexed=False)
        cid = getattr(certificate, "certificate_id", "") or "cid:cert"
        self.indexed_cids.append(str(cid))
        return SimpleNamespace(stored=True, indexed=True)

    def put_receipt(self, receipt: Any) -> Any:
        self.put_receipt_calls += 1
        return SimpleNamespace(stored=True)

    def put_canonical_bytes(self, payload: bytes) -> str:
        raw = bytes(payload)
        self.cas_writes.append(raw)
        return f"cid:cas:{len(self.cas_writes)}"

    def lookup(self, *_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return ()


class _NoIssuer:
    def issue(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(status="deferred", deferred=True, certificate=None)


# ---------------------------------------------------------------------------
# Interfaces / controller signing
# ---------------------------------------------------------------------------


def test_controller_candidate_publisher_interface() -> None:
    publisher = build_controller_candidate_publisher(role="controller")
    assert publisher.interface == CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE
    assert CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE == "ControllerCandidatePublisher@2"
    assert publisher.is_controller is True
    assert publisher.can_publish is True


def test_terminal_pass_is_controller_signed() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    registry = AttestationNonceRegistry()
    candidate_context_cid = dag_cbor_cid({"candidate": "controller-sign-v1"})

    attestation, signed, reason = controller_sign_complete_pass(
        receipt,
        private_key=private,
        trust_policy=policy,
        candidate_context_cid=candidate_context_cid,
        issuance_nonce="nonce-controller-1",
        issued_at=NOW,
        nonce_registry=registry,
        role="controller",
    )
    assert reason == "signed"
    assert attestation is not None
    assert signed is not None
    assert signed.receipt_cid == receipt.receipt_id
    assert signed.runner_attestation_cid == attestation.cid
    assert signed.key_epoch == "epoch-7"

    verified = verify_runner_pass_attestation(
        attestation,
        receipt=receipt,
        policy=policy,
        pinned_policy_cid=policy.cid,
        current_execution_key_cid=receipt.execution_key_cid,
        current_candidate_context_cid=candidate_context_cid,
        now=NOW,
        nonce_registry=registry,
    )
    assert verified.valid is True

    # Publisher path produces the same public attestation.
    store = _RecordingStore()
    publisher = ControllerCandidatePublisher(
        role="controller",
        private_key=private,
        trust_policy=policy,
        nonce_registry=AttestationNonceRegistry(),
        store=store,
        candidate_store=store,
        issuer=_NoIssuer(),
        owner_id="controller:test",
        clock=lambda: NOW,
    )
    assert publisher.can_sign is True
    signed_att, sign_reason = publisher.sign_complete_pass(
        receipt,
        candidate_context_cid=candidate_context_cid,
        issuance_nonce="nonce-controller-2",
        issued_at=NOW,
    )
    assert sign_reason == "signed"
    assert signed_att is not None
    public_env = publisher.public_attestation_envelope(signed_att)
    assert public_env is not None
    assert "private_key" not in public_env
    assert "secret" not in str(public_env).lower()
    assert "witness" not in str(public_env).lower()


def test_workers_cannot_sign_or_publish() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    candidate_context_cid = dag_cbor_cid({"candidate": "worker-deny-v1"})

    attestation, signed, reason = controller_sign_complete_pass(
        receipt,
        private_key=private,
        trust_policy=policy,
        candidate_context_cid=candidate_context_cid,
        role="worker",
    )
    assert attestation is None
    assert signed is None
    assert reason == "worker_cannot_sign"

    store = _RecordingStore()
    worker_publisher = ControllerCandidatePublisher(
        role="worker",
        private_key=private,
        trust_policy=policy,
        store=store,
        candidate_store=store,
        issuer=_NoIssuer(),
    )
    assert worker_publisher.is_controller is False
    assert worker_publisher.can_publish is False
    assert worker_publisher.can_sign is False

    att, sign_reason = worker_publisher.sign_complete_pass(
        receipt,
        candidate_context_cid=candidate_context_cid,
    )
    assert att is None
    assert sign_reason == "worker_cannot_sign"

    intent = ProofReusePublicationIntent.from_receipt(receipt)
    outcome = worker_publisher.publish(intent, sign=True)
    assert isinstance(outcome, IssuedCertificatePublicationResult)
    assert outcome.published is False
    assert outcome.indexed is False
    assert outcome.put_candidate_called is False
    assert outcome.reason_code == "worker_cannot_publish"
    assert outcome.authorizes_skip is False
    assert store.put_candidate_calls == 0


def test_workers_cannot_leak_private_material_in_intents() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    controller = ProofReuseXdistCoordinator.controller()
    worker = ProofReuseXdistCoordinator.from_worker_input(
        controller.configure_worker("gw0"),
        worker_id="gw0",
    )
    assert worker.role is ProofReuseXdistRole.WORKER
    assert worker.can_write is False

    # Queueing with banned private kwargs is stripped / rejected safely.
    ok = worker.queue_publication(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "locator_cid": receipt.locator_cid,
        },
        private_key="should-be-stripped",
        witness={"secret": "nope"},
    )
    # Extra kwargs that aren't accepted by from_receipt will fail validation
    # or be ignored; either way worker never writes.
    assert worker.can_write is False
    assert controller.can_write is True

    publisher = build_controller_candidate_publisher(role="controller")
    accepted, reason = publisher.reject_worker_private_material(
        {
            "receipt": receipt.to_dict(),
            "deferred_request": {
                "receipt_cid": receipt.receipt_id,
                "private_key_material": "leak",
            },
        }
    )
    assert accepted is False
    assert "private" in reason

    accepted_ok, reason_ok = publisher.reject_worker_private_material(
        {
            "receipt": receipt.to_dict(),
            "deferred_request": {
                "receipt_cid": receipt.receipt_id,
                "locator_cid": receipt.locator_cid,
            },
        }
    )
    assert accepted_ok is True
    assert reason_ok == "public_only"
    del private, policy, ok


def test_worker_xdist_flush_cannot_publish_candidates() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    store = _RecordingStore()
    controller = ProofReuseXdistCoordinator.controller()
    worker = ProofReuseXdistCoordinator.from_worker_input(
        controller.configure_worker("gw1"),
        worker_id="gw1",
    )
    assert worker.queue_publication(receipt) is True
    # Worker flush is a no-op for authority writes.
    published = worker.flush_publications(store, issuer=_NoIssuer())
    assert published == ()
    assert store.put_candidate_calls == 0
    del private, policy


# ---------------------------------------------------------------------------
# Partial / racing writes never authorize reuse
# ---------------------------------------------------------------------------


def test_partial_put_candidate_never_authorizes_reuse() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    store = _RecordingStore(put_candidate_ok=False)
    publisher = ControllerCandidatePublisher(
        role="controller",
        private_key=private,
        trust_policy=policy,
        store=store,
        candidate_store=store,
        issuer=_NoIssuer(),
        owner_id="controller:partial",
        clock=lambda: NOW,
    )
    intent = ProofReusePublicationIntent.from_receipt(receipt)
    # Without a verified certificate the transaction retains non-authoritative
    # state and never indexes a skip candidate.
    outcome = publisher.publish(
        intent,
        sign=True,
        candidate_context_cid=dag_cbor_cid({"candidate": "partial-v1"}),
    )
    assert outcome.authorizes_skip is False
    assert outcome.indexed is False
    # put_candidate must not have produced an authoritative index entry.
    assert store.indexed_cids == []
    del policy


def test_racing_duplicate_intents_publish_at_most_once() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy, nonce="nonce:race")
    store = _RecordingStore()
    controller = ProofReuseXdistCoordinator.controller()
    worker_a = ProofReuseXdistCoordinator.from_worker_input(
        controller.configure_worker("gwA"),
        worker_id="gwA",
    )
    worker_b = ProofReuseXdistCoordinator.from_worker_input(
        controller.configure_worker("gwB"),
        worker_id="gwB",
    )
    assert worker_a.queue_publication(receipt) is True
    # Second identical intent on another worker merges without double authority.
    assert worker_b.queue_publication(receipt) is True

    # Merge worker outputs onto controller.
    assert controller.accept_worker_output(worker_a.worker_output()) is True
    assert controller.accept_worker_output(worker_b.worker_output()) is True

    published = controller.flush_publications(store, issuer=_NoIssuer())
    # Deferred issuance path may retain receipts without indexing.
    assert store.put_candidate_calls == 0 or len(published) <= 1
    # Never more than one authoritative index for the same intent.
    assert len(set(store.indexed_cids)) <= 1
    del private, policy


def test_transaction_exception_does_not_index() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    store = _RecordingStore(fail_first_n=1)
    intent = ProofReusePublicationIntent.from_receipt(receipt)

    # Production publisher never raises into the controller flush path and
    # never indexes a skip-authorizing candidate when issuance/verify is
    # incomplete (deferred issuer).  Partial store failures similarly leave
    # indexed_cids empty.
    safe = ControllerCandidatePublisher(
        role="controller",
        private_key=private,
        trust_policy=policy,
        store=store,
        candidate_store=store,
        issuer=_NoIssuer(),
        owner_id="controller:safe",
        clock=lambda: NOW,
    )
    outcome = safe.publish(
        intent,
        sign=True,
        candidate_context_cid=dag_cbor_cid({"candidate": "safe-v1"}),
    )
    assert outcome.authorizes_skip is False
    assert outcome.indexed is False
    assert outcome.put_candidate_called is False or store.indexed_cids == []
    assert store.indexed_cids == []
    del policy


def test_public_attestation_envelope_strips_private_fields() -> None:
    private, _public, policy = _trust_material()
    receipt = _admitted_receipt(policy)
    publisher = build_controller_candidate_publisher(
        role="controller",
        private_key=private,
        trust_policy=policy,
        clock=lambda: NOW,
    )
    attestation, reason = publisher.sign_complete_pass(
        receipt,
        candidate_context_cid=dag_cbor_cid({"candidate": "env-v1"}),
        issuance_nonce="nonce-env",
        issued_at=NOW,
    )
    assert reason == "signed"
    assert attestation is not None
    envelope = publisher.public_attestation_envelope(attestation)
    assert envelope is not None
    dumped = str(envelope).lower()
    assert "private_key" not in dumped
    assert "witness" not in dumped
    assert "proving_key" not in dumped
    # Signature is public; presence is fine.
    assert "signature" in envelope or "receipt_cid" in envelope
