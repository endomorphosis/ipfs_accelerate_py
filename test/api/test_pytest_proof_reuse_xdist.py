"""Tests for single-authority xdist proof-reuse coordination (PTR-053)."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    ReuseAction,
    ReuseReasonCode,
    TestPassReceipt,
    TestProofCertificate,
)
from ipfs_accelerate_py.testing.proof_reuse.config import (
    ProofReuseConfig,
    ProofReuseMode,
)
from ipfs_accelerate_py.testing.proof_reuse.lookup import (
    ITEM_DECISION_ATTRIBUTE,
    SKIP_REASON_PREFIX,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    CONFIG_ATTRIBUTE,
    COORDINATOR_ATTRIBUTE,
    METRICS_ATTRIBUTE,
    pytest_configure_node,
    pytest_testnodedown,
)
import ipfs_accelerate_py.testing.proof_reuse.publication as publication_module
from ipfs_accelerate_py.testing.proof_reuse.publication import (
    Groth16ArtifactIdentityBindings,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import (
    ProofReuseMetricsSnapshot,
    ProofReuseOutcome,
    ProofReuseSessionMetrics,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    COORDINATION_UNAVAILABLE,
    WORKER_INPUT_KEY,
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
    ProofReuseXdistRole,
)


def _receipt(*, nonce: str = "nonce:one") -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid="cid:execution-key",
        locator_cid="cid:test-locator",
        nonce=nonce,
    )


def _certificate(receipt: TestPassReceipt) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=receipt.execution_key_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_system_id="proof:test",
    )


def _ready_bindings() -> Groth16ArtifactIdentityBindings:
    """Provenance-ready pins for an explicitly injected crypto test verifier."""

    return Groth16ArtifactIdentityBindings(
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        artifacts_root="/test-only/reviewed-groth16-v4",
        verifying_key_sha256="a" * 64,
        proving_key_sha256="b" * 64,
        provenance_ready=True,
        reason_code="test_fixture_provenance_ready",
    )


def _connected_worker(
    controller: ProofReuseXdistCoordinator,
    worker_id: str = "gw0",
    *,
    metrics: ProofReuseSessionMetrics | None = None,
) -> ProofReuseXdistCoordinator:
    return ProofReuseXdistCoordinator.from_worker_input(
        controller.configure_worker(worker_id),
        worker_id=worker_id,
        metrics=metrics,
    )


class _ReceiptStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.receipts: list[Any] = []

    def put_receipt(self, receipt: Any) -> Any:
        self.receipts.append(receipt)
        if self.fail:
            raise OSError("publication unavailable")
        return SimpleNamespace(stored=True)


class _CandidateStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, Any]] = []

    def put_candidate(
        self,
        receipt: Any,
        certificate: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(("put_candidate", (receipt, certificate, kwargs)))
        if self.fail:
            raise OSError("atomic publisher unavailable")
        return SimpleNamespace(stored=True, indexed=True)

    def put_receipt(self, receipt: Any) -> Any:
        self.calls.append(("put_receipt", receipt))
        return SimpleNamespace(stored=True)

    def put_certificate(self, certificate: Any) -> Any:
        self.calls.append(("put_certificate", certificate))
        return SimpleNamespace(stored=True)


def test_metrics_distinguish_every_public_outcome_without_private_data() -> None:
    metrics = ProofReuseSessionMetrics()
    metrics.predicted(reason_code="lookup_hit", bytes_read=13)
    metrics.verified(reason_code="proof_verified", latency_ms=2.5)
    metrics.skipped(reason_code="proof_cache_hit")
    metrics.executed(reason_code="real_execution", latency_ms=8)
    metrics.deferred(reason_code="certificate_deferred")
    metrics.degraded(reason_code=COORDINATION_UNAVAILABLE)

    payload = metrics.snapshot().to_dict()

    assert payload["counts"] == {
        "predicted": 1,
        "verified": 1,
        "skipped": 1,
        "executed": 1,
        "deferred": 1,
        "degraded": 1,
    }
    assert payload["verify_latency_ms"] == 2.5
    assert payload["execution_latency_ms"] == 8.0
    assert payload["bytes_read"] == 13
    serialized = json.dumps(payload, sort_keys=True)
    for private_field in ("nodeid", "path", "parameter", "stdout", "stderr"):
        assert private_field not in serialized


@pytest.mark.parametrize(
    "private_field",
    ["nodeid", "path", "parameters", "stdout", "longrepr"],
)
def test_worker_metrics_reject_private_or_unknown_fields(
    private_field: str,
) -> None:
    payload = ProofReuseSessionMetrics().snapshot().to_dict()
    payload[private_field] = "private test data"

    with pytest.raises(ValueError, match="private or unknown"):
        ProofReuseMetricsSnapshot.from_dict(payload)


def test_worker_metrics_reject_private_data_disguised_as_reason_code() -> None:
    payload = ProofReuseSessionMetrics().snapshot().to_dict()
    payload["reasons"] = {"test_customer_password": 1}

    with pytest.raises(ValueError, match="unsafe reason"):
        ProofReuseMetricsSnapshot.from_dict(payload)


def test_pending_ptr155_reason_is_privacy_safe_and_transportable() -> None:
    metrics = ProofReuseSessionMetrics()
    metrics.deferred(reason_code="positive_v4_publication_pending_ptr155")

    payload = metrics.snapshot().to_dict()

    assert payload["reasons"] == {
        "positive_v4_publication_pending_ptr155": 1,
    }
    assert ProofReuseMetricsSnapshot.from_dict(payload).reasons == payload["reasons"]


def test_metric_worker_packet_is_merged_exactly_once() -> None:
    controller_metrics = ProofReuseSessionMetrics()
    controller = ProofReuseXdistCoordinator.controller(
        metrics=controller_metrics
    )
    worker_metrics = ProofReuseSessionMetrics()
    worker_metrics.predicted()
    worker_metrics.verified(latency_ms=3)
    worker = _connected_worker(controller, metrics=worker_metrics)
    packet = worker.worker_output()

    assert controller.accept_worker_output(packet) is True
    assert controller.accept_worker_output(packet) is True
    assert controller_metrics.count(ProofReuseOutcome.PREDICTED) == 1
    assert controller_metrics.count(ProofReuseOutcome.VERIFIED) == 1


def test_workers_never_write_and_controller_publishes_one_receipt() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    first = _connected_worker(controller, "gw0")
    second = _connected_worker(controller, "gw1")
    assert first.worker_token != second.worker_token
    assert first.can_write is False
    assert second.can_write is False

    receipt = _receipt()
    assert first.queue_publication(receipt) is True
    assert second.queue_publication(receipt) is True
    assert controller.accept_worker_output(first.worker_output()) is True
    assert controller.accept_worker_output(second.worker_output()) is True
    assert controller.pending_publications == 1

    store = _ReceiptStore()
    published = controller.flush_publications(store)

    assert len(published) == 1
    assert len(store.receipts) == 1
    assert controller.pending_publications == 0
    assert controller.metrics.count(ProofReuseOutcome.DEFERRED) == 1
    assert controller.flush_publications(store) == ()
    assert len(store.receipts) == 1


def test_controller_rejects_tampering_and_disagreeing_authority() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    worker = _connected_worker(controller)
    worker.metrics.executed()
    packet = worker.worker_output()

    tampered = copy.deepcopy(packet)
    tampered["metrics"]["counts"]["executed"] = 100
    assert controller.accept_worker_output(tampered) is False

    disagreeing = copy.deepcopy(packet)
    disagreeing["controller_id"] = "controller:other"
    assert controller.accept_worker_output(disagreeing) is False

    assert controller.accept_worker_output(packet) is True
    assert controller.metrics.count(ProofReuseOutcome.EXECUTED) == 1


def test_workers_disagreeing_on_publication_payload_disable_all_writes() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    first = _connected_worker(controller, "gw0")
    second = _connected_worker(controller, "gw1")
    receipt = _receipt()
    assert first.queue_publication(
        receipt,
        deferred_request={"policy": "first"},
    )
    assert second.queue_publication(
        receipt,
        deferred_request={"policy": "second"},
    )

    assert controller.accept_worker_output(first.worker_output()) is True
    assert controller.accept_worker_output(second.worker_output()) is True

    assert controller.healthy is False
    assert controller.can_write is False
    assert controller.pending_publications == 0
    assert (
        controller.metrics.reasons["publication_intent_disagrees"] == 1
    )


def test_attached_fake_certificate_never_reaches_candidate_index() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    receipt = _receipt()
    certificate = _certificate(receipt)
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        certificate=certificate.to_dict(),
        certificate_cid=certificate.certificate_id,
    )
    assert controller.queue_publication(intent) is True
    store = _CandidateStore()
    issuer = SimpleNamespace(
        last_artifact_bindings=Groth16ArtifactIdentityBindings.unready(
            "hostile_unreviewed_artifacts"
        )
    )

    assert controller.flush_publications(store, issuer) == (intent.intent_id,)
    assert [name for name, _payload in store.calls] == ["put_receipt"]
    assert controller.healthy is True
    assert controller.pending_publications == 0


def test_attached_certificate_with_ready_pins_but_no_verifier_never_indexes() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    receipt = _receipt()
    certificate = _certificate(receipt)
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        certificate=certificate.to_dict(),
        certificate_cid=certificate.certificate_id,
    )
    assert controller.queue_publication(intent) is True
    store = _CandidateStore()
    issuer = SimpleNamespace(last_artifact_bindings=_ready_bindings())

    assert controller.flush_publications(store, issuer) == (intent.intent_id,)
    assert [name for name, _payload in store.calls] == ["put_receipt"]
    assert controller.healthy is True
    assert controller.pending_publications == 0


def test_transaction_unavailable_never_uses_legacy_issuer_candidate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = ProofReuseXdistCoordinator.controller()
    receipt = _receipt()
    certificate = _certificate(receipt)
    issued: list[Any] = []

    class _HostileIssuer:
        def issue(self, request: Any) -> Any:
            issued.append(request)
            return SimpleNamespace(
                status="certificate_issued",
                certificate=certificate.to_dict(),
            )

    intent = ProofReusePublicationIntent.from_receipt(receipt)
    assert controller.queue_publication(intent) is True
    store = _CandidateStore()
    monkeypatch.setitem(
        sys.modules,
        "ipfs_accelerate_py.testing.proof_reuse.publication",
        None,
    )

    assert controller.flush_publications(store, _HostileIssuer()) == (
        intent.intent_id,
    )
    assert issued == []
    assert [name for name, _payload in store.calls] == ["put_receipt"]
    assert controller.metrics.reasons["deferred_issuer_unavailable"] == 1


def test_missing_transaction_and_receipt_cache_stays_pending_without_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = ProofReuseXdistCoordinator.controller()
    intent = ProofReusePublicationIntent.from_receipt(_receipt())
    assert controller.queue_publication(intent) is True
    candidate_calls: list[Any] = []
    store = SimpleNamespace(
        put_candidate=lambda *args, **kwargs: candidate_calls.append(
            (args, kwargs)
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "ipfs_accelerate_py.testing.proof_reuse.publication",
        None,
    )

    assert controller.flush_publications(store, SimpleNamespace()) == ()
    assert candidate_calls == []
    assert controller.healthy is True
    assert controller.pending_publications == 1
    assert controller.metrics.count(ProofReuseOutcome.DEFERRED) == 1
    assert controller.metrics.reasons["cache_unavailable"] == 1


def test_retained_public_context_survives_worker_transport_for_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = ProofReuseXdistCoordinator.controller()
    worker = _connected_worker(controller)
    receipt = _receipt()
    retained_receipt = receipt.canonical_bytes()
    retained_candidate = json.dumps(
        {"padding": "c" * 5000},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert worker.queue_publication(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": receipt.execution_key_cid,
            "locator_cid": receipt.locator_cid,
            "retained_receipt_bytes_hex": retained_receipt.hex(),
            "retained_candidate_context_bytes_hex": retained_candidate.hex(),
        },
    )
    assert controller.accept_worker_output(worker.worker_output()) is True
    issued: list[Any] = []
    verified: list[Any] = []
    observed_requests: list[Any] = []
    observed_outcomes: list[Any] = []

    original_publish = (
        publication_module.ProofReuseControllerPublicationTransaction.publish_intent
    )

    def observe_request(self: Any, intent: Any, **kwargs: Any) -> Any:
        observed_requests.append(kwargs.get("deferred_request"))
        outcome = original_publish(self, intent, **kwargs)
        observed_outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(
        publication_module.ProofReuseControllerPublicationTransaction,
        "publish_intent",
        observe_request,
    )

    class _DeferredIssuer:
        def issue(self, request: Any) -> Any:
            issued.append(request)
            return SimpleNamespace(status="certificate_deferred")

        def verify_certificate_locally(self, *args: Any) -> Any:
            verified.append(args)
            return True

    store = _CandidateStore()
    published = controller.flush_publications(store, _DeferredIssuer())

    assert len(published) == 1
    assert issued == []
    assert verified == []
    assert len(observed_requests) == 1
    request = observed_requests[0]
    statement = getattr(request, "statement", None)
    public_inputs = getattr(statement, "public_inputs", None)
    if public_inputs is not None and hasattr(public_inputs, "to_dict"):
        public = public_inputs.to_dict()
    else:
        public = request
        if hasattr(public, "to_dict"):
            public = public.to_dict()
    assert public["retained_receipt_bytes_hex"] == retained_receipt.hex()
    assert (
        public["retained_candidate_context_bytes_hex"]
        == retained_candidate.hex()
    )
    assert [name for name, _payload in store.calls] == ["put_receipt"]
    assert len(observed_outcomes) == 1
    assert (
        observed_outcomes[0].reason_code
        == "positive_v4_publication_pending_ptr155"
    )


@pytest.mark.parametrize("attached", [False, True])
def test_ptr152_transaction_denies_positive_candidate_authority(
    attached: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    certificate = _certificate(receipt)
    bindings = _ready_bindings()
    verified: list[Any] = []
    issued: list[Any] = []
    observed_requests: list[Any] = []
    observed_outcomes: list[Any] = []

    class _TypedRequest:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload
            self.receipt_cid = str(payload["receipt_cid"])
            self.execution_key_cid = str(payload["execution_key_cid"])
            self.locator_cid = str(payload.get("locator_cid") or "")

        @classmethod
        def from_public_mapping(cls, payload: Any) -> "_TypedRequest":
            return cls(dict(payload))

    monkeypatch.setitem(
        sys.modules,
        "ipfs_datasets_py.logic.zkp.test_certificate_issuer",
        SimpleNamespace(
            DEFERRED_TEST_CERTIFICATE_REQUEST_INTERFACE=(
                "DeferredTestCertificateRequest@1"
            ),
            DeferredTestCertificateRequest=_TypedRequest,
        ),
    )

    original_publish = (
        publication_module.ProofReuseControllerPublicationTransaction.publish_intent
    )

    def observe_request(self: Any, intent: Any, **kwargs: Any) -> Any:
        observed_requests.append(kwargs.get("deferred_request"))
        outcome = original_publish(self, intent, **kwargs)
        observed_outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(
        publication_module.ProofReuseControllerPublicationTransaction,
        "publish_intent",
        observe_request,
    )

    class _Issuer:
        last_artifact_bindings = bindings

        def issue(self, request: Any) -> Any:
            issued.append(request)
            return SimpleNamespace(
                status="certificate_issued",
                certificate=certificate.to_dict(),
                certificate_cid=certificate.certificate_id,
            )

        def verify_certificate_locally(self, *args: Any) -> Any:
            verified.append(args)
            return SimpleNamespace(
                verified=True,
                authoritative=True,
                can_authorize_skip=True,
            )

    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        certificate=certificate.to_dict() if attached else None,
        certificate_cid=certificate.certificate_id if attached else "",
    )
    controller = ProofReuseXdistCoordinator.controller()
    assert controller.queue_publication(intent) is True
    store = _CandidateStore()

    assert controller.flush_publications(store, _Issuer()) == (intent.intent_id,)
    assert [name for name, _payload in store.calls] == ["put_receipt"]
    assert issued == []
    assert verified == []
    assert len(observed_requests) == 1
    request = observed_requests[0]
    assert isinstance(request, _TypedRequest)
    assert request.receipt_cid == receipt.receipt_id
    assert request.locator_cid == receipt.locator_cid
    assert "witness" not in json.dumps(request.payload)
    assert controller.healthy is True
    assert controller.pending_publications == 0
    assert len(observed_outcomes) == 1
    assert (
        observed_outcomes[0].reason_code
        == "positive_v4_publication_pending_ptr155"
    )


def test_pending_positive_v4_does_not_probe_atomic_candidate_failure() -> None:
    issued: list[Any] = []
    verified: list[Any] = []

    class _Issuer:
        last_artifact_bindings = _ready_bindings()

        def issue(self, request: Any) -> Any:
            issued.append(request)
            raise AssertionError("PTR-152 must not issue")

        def verify_certificate_locally(self, *args: Any) -> Any:
            verified.append(args)
            raise AssertionError("PTR-152 must not verify")

    issuer = _Issuer()
    controller = ProofReuseXdistCoordinator.controller()
    first_receipt = _receipt(nonce="nonce:first")
    first_certificate = _certificate(first_receipt)
    first = ProofReusePublicationIntent.from_receipt(
        first_receipt,
        certificate=first_certificate.to_dict(),
        certificate_cid=first_certificate.certificate_id,
    )
    second_receipt = _receipt(nonce="nonce:second")
    second_certificate = _certificate(second_receipt)
    second = ProofReusePublicationIntent.from_receipt(
        second_receipt,
        certificate=second_certificate.to_dict(),
        certificate_cid=second_certificate.certificate_id,
    )
    assert controller.queue_publication(first) is True
    assert controller.queue_publication(second) is True
    store = _CandidateStore(fail=True)

    assert controller.flush_publications(store, issuer) == (
        first.intent_id,
        second.intent_id,
    )
    assert issued == []
    assert verified == []
    assert controller.healthy is True
    assert controller.can_write is True
    assert controller.pending_publications == 0
    assert [name for name, _payload in store.calls] == [
        "put_receipt",
        "put_receipt",
    ]
    assert controller.flush_publications(store, issuer) == ()
    assert [name for name, _payload in store.calls] == [
        "put_receipt",
        "put_receipt",
    ]


def test_coordination_failure_removes_proof_skip_and_forces_execution() -> None:
    proof_skip = SimpleNamespace(
        name="skip",
        kwargs={"reason": f"{SKIP_REASON_PREFIX}cid:certificate"},
    )
    unrelated_skip = SimpleNamespace(
        name="skip",
        kwargs={"reason": "platform unavailable"},
    )
    item = SimpleNamespace(own_markers=[proof_skip, unrelated_skip])
    coordinator = ProofReuseXdistCoordinator.from_worker_input(
        None,
        worker_id="gw0",
    )

    assert coordinator.healthy is False
    assert coordinator.can_write is False
    coordinator.mark_controller_unavailable([item])

    assert item.own_markers == [unrelated_skip]
    decision = getattr(item, ITEM_DECISION_ATTRIBUTE)
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is ReuseReasonCode.COORDINATION_UNAVAILABLE
    assert coordinator.queue_publication(_receipt()) is False
    assert (
        coordinator.metrics.reasons[COORDINATION_UNAVAILABLE] == 1
    )


def test_worker_output_is_frozen_for_safe_transport_retry() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    worker = _connected_worker(controller)
    worker.metrics.executed()
    first = worker.worker_output()
    worker.metrics.executed()
    retried = worker.worker_output()

    assert retried == first
    assert controller.accept_worker_output(first) is True
    assert controller.accept_worker_output(retried) is True
    assert controller.metrics.count(ProofReuseOutcome.EXECUTED) == 1


def test_plugin_promotes_exactly_one_controller_and_scopes_worker_token() -> None:
    metrics = ProofReuseSessionMetrics()
    config = SimpleNamespace()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READWRITE),
    )
    setattr(config, METRICS_ATTRIBUTE, metrics)
    setattr(
        config,
        COORDINATOR_ATTRIBUTE,
        ProofReuseXdistCoordinator.standalone(metrics=metrics),
    )
    first = SimpleNamespace(
        config=config,
        gateway=SimpleNamespace(id="gw0"),
        workerinput={},
    )
    second = SimpleNamespace(
        config=config,
        gateway=SimpleNamespace(id="gw1"),
        workerinput={},
    )

    pytest_configure_node(first)
    pytest_configure_node(second)

    coordinator = getattr(config, COORDINATOR_ATTRIBUTE)
    assert coordinator.role is ProofReuseXdistRole.CONTROLLER
    first_payload = first.workerinput[WORKER_INPUT_KEY]
    second_payload = second.workerinput[WORKER_INPUT_KEY]
    assert first_payload["controller_id"] == second_payload["controller_id"]
    assert first_payload["session_id"] == second_payload["session_id"]
    assert first_payload["worker_token"] != second_payload["worker_token"]
    assert ProofReuseXdistCoordinator.from_worker_input(
        first_payload,
        worker_id="gw1",
    ).healthy is False


def test_worker_restart_invalidates_stale_worker_authority() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    original = _connected_worker(controller, "gw0")
    original.metrics.executed()
    stale_packet = original.worker_output()

    replacement = _connected_worker(controller, "gw0")
    replacement.metrics.executed()

    assert replacement.worker_token != original.worker_token
    assert controller.accept_worker_output(stale_packet) is False
    assert controller.accept_worker_output(replacement.worker_output()) is True
    assert controller.metrics.count(ProofReuseOutcome.EXECUTED) == 1


def test_observed_worker_crash_fences_controller_publication() -> None:
    metrics = ProofReuseSessionMetrics()
    coordinator = ProofReuseXdistCoordinator.controller(metrics=metrics)
    assert coordinator.queue_publication(_receipt()) is True
    config = SimpleNamespace()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READWRITE),
    )
    setattr(config, METRICS_ATTRIBUTE, metrics)
    setattr(config, COORDINATOR_ATTRIBUTE, coordinator)
    node = SimpleNamespace(config=config)

    pytest_testnodedown(node, RuntimeError("worker exited"))

    assert coordinator.healthy is False
    assert coordinator.can_write is False
    assert coordinator.pending_publications == 0
    assert metrics.reasons["worker_crash"] == 1


@pytest.mark.skipif(
    importlib.util.find_spec("xdist") is None,
    reason="pytest-xdist is not installed",
)
def test_enabled_plugin_runs_under_real_xdist(tmp_path: Path) -> None:
    test_file = tmp_path / "test_parallel.py"
    test_file.write_text(
        """
import os

def test_worker_one():
    assert os.environ["PYTEST_XDIST_WORKER"].startswith("gw")

def test_worker_two():
    assert os.environ["PYTEST_XDIST_WORKER"].startswith("gw")
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (package_root, environment.get("PYTHONPATH", ""))
        if part
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-p",
            "ipfs_accelerate_py.testing.proof_reuse.plugin",
            "--proof-reuse-mode=shadow",
            "-n",
            "2",
            str(test_file),
            "-q",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "2 passed" in output
    assert "proof reuse: predicted=0 verified=0 skipped=0 executed=2" in output
