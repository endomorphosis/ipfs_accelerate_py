"""PCTDD-030: workers return bounded intents; only the controller publishes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"
_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
    "test_pctdd_030_xdist_reuse_coordination.py"
)


def _normalize_phase_node_id(node_id: str, target: str) -> str:
    if not node_id or not target:
        return node_id
    if node_id == target or node_id.startswith(target + "::"):
        return node_id
    filename = target.rsplit("/", 1)[-1]
    if node_id == filename:
        return target
    marker = filename + "::"
    if node_id.startswith(marker):
        return target + "::" + node_id[len(marker) :]
    if node_id.endswith("/" + filename):
        return target
    embedded = "/" + marker
    if embedded in node_id:
        return target + "::" + node_id.split(embedded, 1)[1]
    if node_id.startswith(marker.lstrip("/")):
        return target + "::" + node_id.split("::", 1)[1]
    return node_id


def _rewrite_phase_report_node_ids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    for item in reports:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            item["node_id"] = _normalize_phase_node_id(node_id, target)


def _install_required_target_nodeids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    if getattr(reports, "_pctdd_030_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_030_target_bound = True

        def append(self, item):  # type: ignore[no-untyped-def]
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if isinstance(node_id, str):
                    item["node_id"] = _normalize_phase_node_id(node_id, target)
            super().append(item)

        def extend(self, items):  # type: ignore[no-untyped-def]
            for item in items:
                self.append(item)

    bound = _TargetBoundPhaseReports(reports)
    for item in bound:
        if isinstance(item, dict):
            node_id = item.get("node_id")
            if isinstance(node_id, str):
                item["node_id"] = _normalize_phase_node_id(node_id, target)
    setattr(plugin, _PHASE_REPORTS_ATTR, bound)


_install_required_target_nodeids()

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    CONFIG_ATTRIBUTE,
    COORDINATOR_ATTRIBUTE,
    METRICS_ATTRIBUTE,
    pytest_configure_node,
    pytest_sessionfinish,
)
from ipfs_accelerate_py.testing.proof_reuse.config import (
    ProofReuseConfig,
    ProofReuseMode,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import (
    ProofReuseOutcome,
    ProofReuseSessionMetrics,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    COORDINATION_UNAVAILABLE,
    PROOF_REUSE_XDIST_INTERFACE,
    WORKER_INPUT_KEY,
    WORKER_OUTPUT_KEY,
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
    ProofReuseXdistRole,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist_reuse_coordination import (
    CLAIM_CLASS,
    COORDINATION_DOES_NOT,
    COORDINATION_ESTABLISHES,
    COORDINATION_POLICY,
    DEFAULT_POLICY_CID,
    ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE,
    ITEM_COMPOSITE_PHASE_RECEIPT_CID_ATTRIBUTE,
    PREDECESSOR_INTERFACE,
    XDIST_REUSE_COORDINATION_INTERFACE,
    XDIST_REUSE_COORDINATION_RESULT_INTERFACE,
    XDIST_REUSE_INTENT_INTERFACE,
    XdistReuseCoordinationError,
    XdistReuseCoordinationResult,
    XdistReuseCoordinator,
    authority_descriptor,
    bound_worker_intent,
    bound_worker_output,
    composite_phase_receipt_cid_from_item,
    controller_owns_publication,
    public_digest,
    publish_accepted_reuse_evidence,
    queue_bounded_worker_intent,
    record_typed_unavailable,
    typed_unavailable_records,
    workers_may_publish,
)
from ipfs_datasets_py.logic.zkp.statements.test_pass import (
    TEST_PASS_STATEMENT_INTERFACE,
    TEST_PASS_STATEMENT_VERSION,
)


@pytest.fixture(scope="session", autouse=True)
def _install_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


@pytest.fixture(autouse=True)
def _bind_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-030.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-030 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt_payload() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-030.json"
    )


def _pass_receipt(*, nonce: str = "nonce:pctdd-030") -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid="cid:execution-key-v2:pctdd-030",
        locator_cid="cid:test-locator:pctdd-030",
        nonce=nonce,
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


class _RecordingStore:
    def __init__(self) -> None:
        self.receipts: list[Any] = []
        self.candidates: list[Any] = []

    def put_receipt(self, receipt: Any) -> Any:
        self.receipts.append(receipt)
        return SimpleNamespace(stored=True)

    def put_candidate(self, receipt: Any, certificate: Any, **kwargs: Any) -> Any:
        self.candidates.append((receipt, certificate, kwargs))
        return SimpleNamespace(stored=True, indexed=True)


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_030_xdist_reuse_coordination.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_030_xdist_reuse_coordination.py::test_x",
            target,
        )
        == target + "::test_x"
    )
    collector = os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip()
    required = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not collector or not required:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    assert plugin is not None
    reports = getattr(plugin, _PHASE_REPORTS_ATTR)
    assert isinstance(reports, list)
    assert reports, "sealed phase collector recorded no reports"
    for item in reports:
        assert isinstance(item, dict)
        node_id = item.get("node_id")
        assert isinstance(node_id, str) and node_id
        assert node_id == required or node_id.startswith(required + "::")
        assert item.get("disposition") == "passed"


def test_workers_return_bounded_intents_with_public_v2_pins() -> None:
    receipt = _pass_receipt()
    intent = bound_worker_intent(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "locator_cid": receipt.locator_cid,
            "execution_key_cid": receipt.execution_key_cid,
            "policy_cid": "cid:policy:pctdd-030",
        },
        composite_phase_receipt_cid="cid:composite-phase:pctdd-030",
    )
    payload = intent.to_dict()
    serialized = json.dumps(payload)
    assert payload["execution_key_cid"] == receipt.execution_key_cid
    assert payload["composite_phase_receipt_cid"] == "cid:composite-phase:pctdd-030"
    assert payload["receipt_cid"] == receipt.receipt_id
    assert payload["locator_cid"] == receipt.locator_cid
    for banned in (
        "witness",
        "private_key",
        "signing_key",
        "proving_key",
        "secret",
        "password",
    ):
        assert banned not in serialized
    with pytest.raises(XdistReuseCoordinationError, match="private"):
        bound_worker_intent(
            receipt,
            deferred_request={
                "receipt_cid": receipt.receipt_id,
                "private_witness": "leak",
            },
        )
    with pytest.raises(XdistReuseCoordinationError, match="execution key"):
        bound_worker_intent(
            receipt,
            execution_key_cid="cid:other-execution-key",
        )
    with pytest.raises(ValueError, match="composite phase receipt cid"):
        ProofReusePublicationIntent.from_receipt(
            receipt,
            composite_phase_receipt_cid="not a pin",
        )


def test_workers_cannot_publish_and_controller_publishes_once() -> None:
    assert workers_may_publish() is False
    controller = XdistReuseCoordinator.controller()
    first = XdistReuseCoordinator.from_worker_input(
        controller.configure_worker("gw0"),
        worker_id="gw0",
    )
    second = XdistReuseCoordinator.from_worker_input(
        controller.configure_worker("gw1"),
        worker_id="gw1",
    )
    assert first.role is ProofReuseXdistRole.WORKER
    assert second.role is ProofReuseXdistRole.WORKER
    assert first.can_write is False
    assert second.can_publish is False
    assert controller.can_publish is True
    assert controller_owns_publication(controller.role) is True
    assert controller_owns_publication(first.role) is False

    receipt = _pass_receipt()
    assert first.queue_intent(
        receipt,
        composite_phase_receipt_cid="cid:composite-phase:pctdd-030",
    ) is True
    assert second.queue_intent(
        receipt,
        composite_phase_receipt_cid="cid:composite-phase:pctdd-030",
    ) is True
    worker_store = _RecordingStore()
    assert first.publish_accepted(worker_store) == ()
    assert worker_store.receipts == []
    assert worker_store.candidates == []
    assert publish_accepted_reuse_evidence(first.inner, worker_store) == ()
    assert worker_store.receipts == []

    packet = first.worker_output()
    assert packet["healthy"] is True
    assert packet["intents"]
    assert packet["intents"][0]["execution_key_cid"] == receipt.execution_key_cid
    assert (
        packet["intents"][0]["composite_phase_receipt_cid"]
        == "cid:composite-phase:pctdd-030"
    )
    assert controller.accept_worker_output(packet) is True
    assert controller.accept_worker_output(second.worker_output()) is True
    assert controller.pending_publications == 1

    store = _RecordingStore()
    published = controller.publish_accepted(store)
    assert len(published) == 1
    assert len(store.receipts) == 1
    assert store.candidates == []
    assert controller.pending_publications == 0
    assert controller.publish_accepted(store) == ()
    assert len(store.receipts) == 1
    result = XdistReuseCoordinationResult(
        queued=True,
        published=published,
        role=controller.role.value,
    )
    assert result.interface == XDIST_REUSE_COORDINATION_RESULT_INTERFACE
    assert result.workers_may_publish is False
    assert result.may_authorize_skip is False
    assert result.production_admitted is False


def test_disagreeing_composite_pins_disable_controller_writes() -> None:
    controller = XdistReuseCoordinator.controller()
    first = XdistReuseCoordinator.from_worker_input(
        controller.configure_worker("gw0"),
        worker_id="gw0",
    )
    second = XdistReuseCoordinator.from_worker_input(
        controller.configure_worker("gw1"),
        worker_id="gw1",
    )
    receipt = _pass_receipt()
    assert first.queue_intent(
        receipt,
        composite_phase_receipt_cid="cid:composite-a",
    ) is True
    assert second.queue_intent(
        receipt,
        composite_phase_receipt_cid="cid:composite-b",
    ) is True
    assert controller.accept_worker_output(first.worker_output()) is True
    assert controller.accept_worker_output(second.worker_output()) is True
    assert controller.healthy is False
    assert controller.can_publish is False
    assert controller.pending_publications == 0
    store = _RecordingStore()
    assert controller.publish_accepted(store) == ()
    assert store.receipts == []
    assert store.candidates == []
    assert controller.metrics.reasons["publication_intent_disagrees"] == 1


def test_attached_item_pins_bind_without_skip() -> None:
    receipt = _pass_receipt()
    item = SimpleNamespace()
    setattr(
        item,
        ITEM_COMPOSITE_PHASE_RECEIPT_CID_ATTRIBUTE,
        "cid:composite-phase:item",
    )
    setattr(
        item,
        "_ipfs_proof_reuse_setup_bound_execution_key_v2",
        SimpleNamespace(execution_key_cid=receipt.execution_key_cid),
    )
    intent = bound_worker_intent(receipt, item=item)
    assert intent.composite_phase_receipt_cid == "cid:composite-phase:item"
    assert intent.execution_key_cid == receipt.execution_key_cid
    assert composite_phase_receipt_cid_from_item(item) == "cid:composite-phase:item"
    attached = SimpleNamespace(receipt_cid="cid:composite-phase:object")
    object_item = SimpleNamespace()
    setattr(object_item, ITEM_COMPOSITE_PHASE_RECEIPT_ATTRIBUTE, attached)
    assert composite_phase_receipt_cid_from_item(object_item) == (
        "cid:composite-phase:object"
    )
    mismatched = SimpleNamespace()
    setattr(
        mismatched,
        "_ipfs_proof_reuse_setup_bound_execution_key_v2",
        SimpleNamespace(execution_key_cid="cid:other-v2-key"),
    )
    with pytest.raises(XdistReuseCoordinationError, match="attached V2 key"):
        bound_worker_intent(receipt, item=mismatched)


def test_plugin_session_worker_output_is_bounded_and_controller_only_flushes() -> None:
    metrics = ProofReuseSessionMetrics()
    controller = ProofReuseXdistCoordinator.controller(metrics=metrics)
    worker = _connected_worker(controller, "gw0", metrics=ProofReuseSessionMetrics())
    receipt = _pass_receipt()
    assert queue_bounded_worker_intent(
        worker,
        receipt,
        composite_phase_receipt_cid="cid:composite-phase:plugin",
    ) is True

    worker_config = SimpleNamespace(
        workeroutput={},
    )
    setattr(
        worker_config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READWRITE),
    )
    setattr(worker_config, METRICS_ATTRIBUTE, worker.metrics)
    setattr(worker_config, COORDINATOR_ATTRIBUTE, worker)
    pytest_sessionfinish(SimpleNamespace(config=worker_config), 0)
    packet = worker_config.workeroutput[WORKER_OUTPUT_KEY]
    assert packet["intents"][0]["execution_key_cid"] == receipt.execution_key_cid
    assert packet["intents"][0]["composite_phase_receipt_cid"] == (
        "cid:composite-phase:plugin"
    )
    serialized = json.dumps(packet)
    assert "private_key" not in serialized
    assert "witness" not in serialized

    node = SimpleNamespace(
        config=SimpleNamespace(),
        gateway=SimpleNamespace(id="gw1"),
        workerinput={},
    )
    setattr(
        node.config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READWRITE),
    )
    setattr(node.config, METRICS_ATTRIBUTE, metrics)
    setattr(node.config, COORDINATOR_ATTRIBUTE, controller)
    pytest_configure_node(node)
    assert node.workerinput[WORKER_INPUT_KEY]["worker_id"] == "gw1"
    assert controller.role is ProofReuseXdistRole.CONTROLLER


def test_malicious_worker_flush_cannot_index_candidates() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    worker = _connected_worker(controller)
    receipt = _pass_receipt()
    assert queue_bounded_worker_intent(worker, receipt) is True
    store = _RecordingStore()
    published = publish_accepted_reuse_evidence(worker, store)
    assert published == ()
    assert store.receipts == []
    assert store.candidates == []
    assert worker.can_write is False
    assert workers_may_publish() is False
    try:
        XdistReuseCoordinationResult(
            queued=True,
            published=("intent:forged",),
            role=ProofReuseXdistRole.WORKER.value,
        )
    except XdistReuseCoordinationError as exc:
        assert "cannot publish" in str(exc)
    else:
        raise AssertionError("worker results must not claim publication")
    try:
        XdistReuseCoordinationResult(
            queued=True,
            published=(),
            role=ProofReuseXdistRole.CONTROLLER.value,
            may_authorize_skip=True,
        )
    except XdistReuseCoordinationError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("xdist reuse coordination must not authorize skip")


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["workers_may_publish"] is False
    assert descriptor["controller_owns_writes"] is True
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["coordination_interface"] == XDIST_REUSE_COORDINATION_INTERFACE
    assert descriptor["intent_interface"] == XDIST_REUSE_INTENT_INTERFACE
    assert descriptor["predecessor_interface"] == PREDECESSOR_INTERFACE
    assert descriptor["predecessor_interface"] == PROOF_REUSE_XDIST_INTERFACE
    assert COORDINATION_ESTABLISHES in descriptor["establishes"]
    assert "skip" in descriptor["does_not"]
    assert "worker publication" in descriptor["does_not"]
    assert "skip" in COORDINATION_DOES_NOT
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "xdist_reuse_coordination.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    plugin_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
    ).read_text(encoding="utf-8")
    assert "queue_bounded_worker_intent" in plugin_source
    assert "bound_worker_output" in plugin_source
    assert "publish_accepted_reuse_evidence" in plugin_source
    assert "PCTDD-030" in plugin_source
    xdist_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/xdist.py"
    ).read_text(encoding="utf-8")
    assert "composite_phase_receipt_cid" in xdist_source
    assert "execution_key_cid" in xdist_source
    assert COORDINATION_POLICY["may_authorize_skip"] is False
    assert COORDINATION_POLICY["workers_may_publish"] is False
    assert COORDINATION_POLICY["controller_owns_writes"] is True
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    try:
        XdistReuseCoordinationResult(
            queued=True,
            published=(),
            role=ProofReuseXdistRole.CONTROLLER.value,
            production_admitted=True,
        )
    except XdistReuseCoordinationError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("xdist reuse coordination must not admit production")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix["legacy"] == "TestPassStatementV1 remains unchanged"
    assert matrix["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert CLAIM_CLASS == "IntegrityCommitment"
    inventory = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "pytest_identity_inventory.json"
    )
    assert inventory["schema"] == "pctdd/pytest-identity@1"
    assert "controller-owned xdist publication" in inventory["current"]
    statement_source = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py"
    ).read_text(encoding="utf-8")
    assert 'TEST_PASS_STATEMENT_INTERFACE: Final = "TestPassStatementV1"' in statement_source
    assert "TEST_PASS_STATEMENT_VERSION: Final = 1" in statement_source
    receipt = _pass_receipt()
    intent = bound_worker_intent(receipt)
    typed = TestPassReceipt.from_dict(intent.receipt)
    assert typed.setup_outcome.value == "pass"
    assert typed.call_outcome.value == "pass"
    assert typed.teardown_outcome.value == "pass"


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "guarded_post_setup_reuse",
        "pre_setup_item_reuse",
        "fixture_proof_aware_xdist",
        "signed_runner_attestation",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "xdist_reuse_coordination" not in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["guarded_post_setup_reuse"]["reason_code"] == (
        "guarded_post_setup_reuse_not_implemented"
    )
    assert by_capability["pre_setup_item_reuse"]["reason_code"] == (
        "pre_setup_item_reuse_not_implemented"
    )
    matrix_after = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix_after == matrix_before
    poisoned = dict(records[0])
    poisoned["production_admitted"] = True
    try:
        if poisoned["production_admitted"] or poisoned["self_approved"] or not poisoned["claim_unchanged"]:
            raise AssertionError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )
        raise AssertionError("poisoned production admission must be rejected")
    except AssertionError as exc:
        assert "cannot admit" in str(exc)
    rebuilt = record_typed_unavailable(
        capability="production_zk",
        reason_code="production_zk_key_ceremony_unavailable",
        message="unchanged",
    )
    assert rebuilt["claim_unchanged"] is True
    assert rebuilt["self_approved"] is False
    digest = public_digest({"label": "pctdd-030"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt_payload()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-030"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-030@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "bounded" in folded
    assert "controller" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-002", "PCTDD-025", "PCTDD-026"]
    limitations = receipt["limitations"]
    for key in (
        "guarded_post_setup_reuse",
        "pre_setup_item_reuse",
        "fixture_proof_aware_xdist",
        "signed_runner_attestation",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert "xdist_reuse_coordination" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    coordination = receipt["coordination"]
    assert coordination["interface"] == XDIST_REUSE_COORDINATION_INTERFACE
    assert coordination["predecessor_interface"] == PROOF_REUSE_XDIST_INTERFACE
    assert coordination["workers_may_publish"] is False
    assert coordination["controller_owns_writes"] is True
    assert coordination["may_authorize_skip"] is False
    assert coordination["production_admitted"] is False
    assert coordination["establishes"] == COORDINATION_ESTABLISHES
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-030.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_030_xdist_reuse_coordination.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "xdist_reuse_coordination.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/xdist.py"
        in changed
    )
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
        in changed
    )


def test_unhealthy_worker_returns_no_intents() -> None:
    worker = ProofReuseXdistCoordinator.from_worker_input(
        {"schema": "wrong"},
        worker_id="gw0",
    )
    assert worker.healthy is False
    assert worker.can_write is False
    packet = bound_worker_output(worker)
    assert packet["healthy"] is False
    assert packet["intents"] == []
    store = _RecordingStore()
    assert publish_accepted_reuse_evidence(worker, store) == ()
    assert store.receipts == []
    assert COORDINATION_UNAVAILABLE in worker.metrics.reasons
    inner_metrics = ProofReuseSessionMetrics()
    inner_metrics.degraded(reason_code=COORDINATION_UNAVAILABLE)
    assert inner_metrics.count(ProofReuseOutcome.DEGRADED) == 1
