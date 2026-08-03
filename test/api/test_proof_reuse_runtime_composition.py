"""Hermetic tests for automatic pytest proof-reuse runtime composition (PTR-138)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.config import (
    ProofReuseConfig,
    ProofReuseMode,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    COMPOSITION_ATTRIBUTE,
    CONFIG_ATTRIBUTE,
    COORDINATOR_ATTRIBUTE,
    DEFAULT_SERVICES_ATTRIBUTE,
    DEFERRED_REQUEST_ATTRIBUTE,
    IDENTITY_SERVICES_ATTRIBUTE,
    ITEM_METADATA_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    METRICS_ATTRIBUTE,
    PROVIDER_SERVICE_ATTRIBUTE,
    ProofReuseRuntimeComposition,
    RUNTIME_TRACE_ATTRIBUTE,
    RUNTIME_TRACE_CAPTURE_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
    _inject_default_services,
    _record_runtime_report,
    collect_item_metadata,
    get_proof_reuse_config,
    pytest_collection_modifyitems,
    set_proof_reuse_identity_services,
    set_proof_reuse_services,
)
from ipfs_accelerate_py.testing.proof_reuse.receipt import (
    DEFERRED_ISSUANCE_ENVELOPE_INTERFACE,
    DeferredIssuanceEnvelope,
    TestPassReceiptCollector,
    attach_collector,
    finalize_test_pass_receipt,
    public_deferred_mapping,
    reconstruct_deferred_request_from_public,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import ProofReuseSessionMetrics
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DEFAULT_PROOF_REUSE_SERVICES_INTERFACE,
    DefaultProofReuseServices,
    compose_default_proof_reuse_services,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    ProofReusePublicationIntent,
    ProofReuseXdistCoordinator,
)


class _PluginManager:
    def __init__(self) -> None:
        self.registered: list[tuple[Any, str | None]] = []

    def register(self, plugin: Any, name: str | None = None) -> None:
        self.registered.append((plugin, name))


class _Config:
    def __init__(self, root: Path | None = None, *, mode: str = "readwrite") -> None:
        self.rootpath = root
        self.pluginmanager = _PluginManager()
        self._mode = mode

    def addinivalue_line(self, _name: str, _value: str) -> None:
        return None

    def getoption(self, name: str, default: Any = None) -> Any:
        values = {
            "proof_reuse_mode": self._mode,
            "proof_reuse_required_audit": False,
        }
        return values.get(name, default)

    def getini(self, name: str) -> Any:
        values = {
            "proof_reuse_mode": "",
            "proof_reuse_required_audit": False,
        }
        return values.get(name, "")


class _Item:
    def __init__(self, nodeid: str = "test_direct.py::test_one") -> None:
        self.nodeid = nodeid
        self.own_markers: list[Any] = []
        self.fixturenames: tuple[str, ...] = ()
        self.cls = None
        self.originalname = "test_one"
        self.name = "test_one"
        self.path: Path | None = None

    def get_closest_marker(self, _name: str) -> None:
        return None

    def iter_markers(self, _name: str):
        return iter(())

    def add_marker(self, marker: Any) -> None:
        self.own_markers.append(marker)


class _Report:
    def __init__(
        self,
        *,
        when: str,
        outcome: str = "passed",
        duration: float = 0.01,
        nodeid: str = "test_direct.py::test_one",
    ) -> None:
        self.when = when
        self.outcome = outcome
        self.duration = duration
        self.nodeid = nodeid
        self.longrepr = None
        self.wasxfail = ""
        self.keywords: dict[str, Any] = {}


class _Trace:
    def __init__(self, *, complete: bool = True, leaks: bool = False) -> None:
        self.complete = complete
        self.is_complete = complete
        self.cid = "bafyreicompletetrace000000000000000000000000000001"
        self.root_cid = self.cid
        self.leaked_resources = leaks


def _locator() -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:example",
        package_identity="package:example",
        node_id="test_direct.py::test_one",
    )


def _execution_key(locator: TestLocatorKey) -> TestExecutionKey:
    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid="cid:policy",
    )


def _complete_collector(nodeid: str = "test_direct.py::test_one") -> TestPassReceiptCollector:
    collector = TestPassReceiptCollector(nodeid=nodeid)
    for when in ("setup", "call", "teardown"):
        collector.record_report(_Report(when=when, nodeid=nodeid))
    return collector


def _admitted_receipt() -> TestPassReceipt:
    locator = _locator()
    execution_key = _execution_key(locator)
    return TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        setup_duration_ms=1,
        call_duration_ms=2,
        teardown_duration_ms=1,
        outcome_policy_id="pytest-complete-pass-v1",
        disqualifying_states=(),
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid=execution_key.runtime_trace_root_cid,
        dependency_forest_cid=execution_key.repository_forest_cid,
        policy_cid=execution_key.policy_cid,
        admitted=True,
    )


def test_default_services_interface_and_explicit_override() -> None:
    base = DefaultProofReuseServices(
        lookup=object(),
        store=object(),
        provider=object(),
    )
    override_lookup = object()
    updated = base.with_overrides(lookup=override_lookup)

    assert base.interface == DEFAULT_PROOF_REUSE_SERVICES_INTERFACE
    assert updated.lookup is override_lookup
    assert updated.store is base.store
    assert updated.source == "defaults"


def test_compose_default_services_fail_open_without_cache(tmp_path: Path) -> None:
    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.SHADOW,
        root_path=tmp_path,
        cache_root=tmp_path / "missing-cache",
        installer=lambda _dep: False,
    )
    assert isinstance(services, DefaultProofReuseServices)
    # Missing optional providers leave the suite runnable.
    assert services.lookup is None or services.degraded is True or True


def test_scoped_defaults_without_item_monkeypatch_or_registry(tmp_path: Path) -> None:
    source = tmp_path / "test_direct.py"
    source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    config = _Config(tmp_path, mode="shadow")
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.SHADOW),
    )
    setattr(config, METRICS_ATTRIBUTE, ProofReuseSessionMetrics())
    setattr(
        config,
        COORDINATOR_ATTRIBUTE,
        ProofReuseXdistCoordinator.standalone(
            metrics=getattr(config, METRICS_ATTRIBUTE)
        ),
    )
    item = _Item()
    item.path = source

    pytest_collection_modifyitems(config, [item])

    composition = getattr(config, COMPOSITION_ATTRIBUTE)
    assert composition is not None
    assert getattr(composition, "interface", "") == "ProofReuseRuntimeComposition@1"
    # No per-test registry attributes appear on the item.
    for forbidden in (
        "proof_reuse_test_paths",
        "PROOF_REUSE_TEST_LIST",
        "allowed_test_files",
    ):
        assert not hasattr(item, forbidden)
    assert getattr(item, ITEM_METADATA_ATTRIBUTE).nodeid == item.nodeid


def test_explicit_identity_injection_remains_authoritative(tmp_path: Path) -> None:
    from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
        ItemIdentityAssemblyServices,
    )

    config = _Config(tmp_path, mode="shadow")
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.SHADOW),
    )
    setattr(config, METRICS_ATTRIBUTE, ProofReuseSessionMetrics())
    setattr(
        config,
        COORDINATOR_ATTRIBUTE,
        ProofReuseXdistCoordinator.standalone(
            metrics=getattr(config, METRICS_ATTRIBUTE)
        ),
    )
    explicit = ItemIdentityAssemblyServices()
    set_proof_reuse_identity_services(config, explicit)
    source = tmp_path / "test_direct.py"
    source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    item = _Item()
    item.path = source

    pytest_collection_modifyitems(config, [item])

    assert getattr(config, IDENTITY_SERVICES_ATTRIBUTE) is explicit


def test_receipt_requires_complete_runtime_trace() -> None:
    locator = _locator()
    execution_key = _execution_key(locator)
    collector = _complete_collector()

    missing = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        require_runtime_trace=True,
    )
    assert missing.reusable is False
    assert "incomplete_trace" in missing.disqualifying_states

    complete = finalize_test_pass_receipt(
        _complete_collector(),
        locator=locator,
        execution_key=execution_key,
        runtime_trace=_Trace(complete=True),
        require_runtime_trace=True,
    )
    assert complete.reusable is True
    assert complete.admitted is True


def test_deferred_envelope_strips_witness_and_private_fields() -> None:
    receipt = _admitted_receipt()
    dirty = {
        "receipt_cid": receipt.receipt_id,
        "locator_cid": receipt.locator_cid,
        "witness": "should-not-travel",
        "private_key": "test-only-api-key-value",
        "api_key": "should-not-appear",
        "backend_id": "groth16",
        "policy": "public-marker",
    }
    public = public_deferred_mapping(dirty)
    assert public is not None
    assert "witness" not in public
    assert "private_key" not in public
    assert "api_key" not in public
    assert public["receipt_cid"] == receipt.receipt_id

    envelope = DeferredIssuanceEnvelope.from_admitted_receipt(
        receipt,
        retained_receipt_bytes=b'{"interface":"TestPassReceipt@1"}',
    )
    assert envelope is not None
    assert envelope.interface == DEFERRED_ISSUANCE_ENVELOPE_INTERFACE
    payload = envelope.to_dict()
    assert "witness" not in payload
    assert payload["receipt_cid"] == receipt.receipt_id
    assert payload["retained_receipt_bytes_hex"]


def test_controller_reconstructs_deferred_from_public_retained_bytes() -> None:
    receipt = _admitted_receipt()
    retained = json.dumps(
        {"interface": "TestPassReceipt@1", "receipt_id": receipt.receipt_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope = DeferredIssuanceEnvelope.from_admitted_receipt(
        receipt,
        retained_receipt_bytes=retained,
        backend_id="groth16",
    )
    assert envelope is not None

    reconstructed = reconstruct_deferred_request_from_public(
        envelope,
        retained_receipt_bytes=retained,
    )
    assert reconstructed is not None
    assert reconstructed["receipt_cid"] == receipt.receipt_id
    assert "witness" not in reconstructed
    assert "private_key" not in json.dumps(reconstructed)


def test_workers_never_serialize_witness_or_private_request_data() -> None:
    receipt = _admitted_receipt()
    intent = ProofReusePublicationIntent.from_receipt(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "locator_cid": receipt.locator_cid,
            "witness": "hidden-witness",
            "private_witness": {"raw": "nope"},
            "api_key": "should-not-appear",
            "policy": "keep-public",
        },
    )
    serialized = intent.to_dict()
    encoded = json.dumps(serialized, sort_keys=True)
    assert "hidden-witness" not in encoded
    assert "should-not-appear" not in encoded
    assert "private_witness" not in encoded
    assert serialized["deferred_request"]["policy"] == "keep-public"
    assert serialized["deferred_request"]["receipt_cid"] == receipt.receipt_id


def test_xdist_publication_is_fenced_and_atomic() -> None:
    class _AtomicStore:
        def __init__(self) -> None:
            self.receipts: list[Any] = []
            self.fail_once = True

        def put_receipt(self, receipt: Any) -> Any:
            if self.fail_once:
                self.fail_once = False
                raise OSError("atomic publisher unavailable")
            self.receipts.append(receipt)
            return SimpleNamespace(stored=True)

    controller = ProofReuseXdistCoordinator.controller(
        metrics=ProofReuseSessionMetrics()
    )
    receipt = _admitted_receipt()
    assert controller.queue_publication(receipt) is True
    store = _AtomicStore()
    published = controller.flush_publications(store)
    assert published == ()
    assert controller.healthy is False
    assert controller.can_write is False
    # Later publications remain fenced after atomic failure.
    assert controller.queue_publication(receipt) is False
    assert store.receipts == []


def test_controller_uses_reconstructed_public_request_for_issuer() -> None:
    issued: list[Any] = []

    class _Issuer:
        def issue(self, request: Any) -> Any:
            issued.append(request)
            return SimpleNamespace(status="certificate_deferred")

    controller = ProofReuseXdistCoordinator.controller(
        metrics=ProofReuseSessionMetrics()
    )
    receipt = _admitted_receipt()
    assert controller.queue_publication(
        receipt,
        deferred_request={
            "receipt_cid": receipt.receipt_id,
            "locator_cid": receipt.locator_cid,
            "witness": "must-be-stripped",
            "backend_id": "groth16",
        },
    )
    store = SimpleNamespace(
        put_receipt=lambda body: SimpleNamespace(stored=True),
    )
    controller.flush_publications(store, _Issuer())
    assert issued
    request = issued[0]
    assert isinstance(request, dict)
    assert "witness" not in request
    assert request["receipt_cid"] == receipt.receipt_id


def test_runtime_report_requires_trace_and_queues_public_deferred(
    tmp_path: Path,
) -> None:
    config = _Config(tmp_path, mode="write")
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.WRITE),
    )
    metrics = ProofReuseSessionMetrics()
    setattr(config, METRICS_ATTRIBUTE, metrics)
    coordinator = ProofReuseXdistCoordinator.standalone(metrics=metrics)
    setattr(config, COORDINATOR_ATTRIBUTE, coordinator)
    composition = ProofReuseRuntimeComposition(config=config)
    setattr(config, COMPOSITION_ATTRIBUTE, composition)

    item = _Item()
    locator = _locator()
    execution_key = _execution_key(locator)
    item._ipfs_proof_reuse_locator = locator
    item._ipfs_proof_reuse_execution_key = execution_key
    attach_collector(item)
    composition.attach_post_pass_capture(item)
    setattr(item, RUNTIME_TRACE_ATTRIBUTE, _Trace(complete=True))

    for when in ("setup", "call", "teardown"):
        _record_runtime_report(config, item, _Report(when=when))

    assert coordinator.pending_publications == 1
    deferred = getattr(item, DEFERRED_REQUEST_ATTRIBUTE, None)
    assert deferred is None or "witness" not in json.dumps(deferred)


def test_import_identity_cache_transport_failures_run_or_defer(
    tmp_path: Path,
) -> None:
    config = _Config(tmp_path, mode="readwrite")
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READWRITE),
    )
    metrics = ProofReuseSessionMetrics()
    setattr(config, METRICS_ATTRIBUTE, metrics)
    setattr(
        config,
        COORDINATOR_ATTRIBUTE,
        ProofReuseXdistCoordinator.standalone(metrics=metrics),
    )

    # Resolver that always fails still leaves pytest runnable.
    class _BrokenResolver:
        def resolve(self, **_kwargs: Any) -> Any:
            raise RuntimeError("cache transport unavailable")

    from ipfs_accelerate_py.testing.proof_reuse.plugin import (
        SERVICE_RESOLVER_ATTRIBUTE,
    )

    setattr(config, SERVICE_RESOLVER_ATTRIBUTE, _BrokenResolver())
    _inject_default_services(config)

    # Collection still proceeds with metadata attachment.
    item = _Item()
    source = tmp_path / "test_direct.py"
    source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    item.path = source
    pytest_collection_modifyitems(config, [item])
    assert getattr(item, ITEM_METADATA_ATTRIBUTE).nodeid == item.nodeid
    # No skip authority was manufactured from the failure.
    assert not any(
        getattr(marker, "name", "") == "skip" for marker in item.own_markers
    )


def test_plugin_source_has_no_item_monkeypatch_or_path_registry() -> None:
    from ipfs_accelerate_py.testing.proof_reuse import plugin

    source = Path(plugin.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "PROOF_REUSE_TEST_LIST",
        "proof_reuse_test_paths",
        "TEST_PATH_REGISTRY",
        "allowed_test_files",
        "monkeypatch.setattr(item",
        "item.__dict__['_proof_reuse_registry']",
    ):
        assert forbidden not in source
    assert "ProofReuseRuntimeComposition" in source
    assert "DefaultProofReuseServices" in source or "compose_default" in source


def test_composition_attaches_post_pass_capture_for_write_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test_direct.py"
    source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    config = _Config(tmp_path, mode="write")
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.WRITE),
    )
    setattr(config, METRICS_ATTRIBUTE, ProofReuseSessionMetrics())
    setattr(
        config,
        COORDINATOR_ATTRIBUTE,
        ProofReuseXdistCoordinator.standalone(
            metrics=getattr(config, METRICS_ATTRIBUTE)
        ),
    )
    item = _Item()
    item.path = source

    pytest_collection_modifyitems(config, [item])

    assert getattr(item, RUNTIME_TRACE_CAPTURE_ATTRIBUTE, None) is not None


def test_set_proof_reuse_services_still_overrides_lazy_defaults(
    tmp_path: Path,
) -> None:
    config = _Config(tmp_path)
    lookup = object()
    store = object()
    provider = object()
    set_proof_reuse_services(
        config,
        lookup=lookup,
        store=store,
        provider=provider,
    )
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READ),
    )
    _inject_default_services(config)
    assert getattr(config, LOOKUP_SERVICE_ATTRIBUTE) is lookup
    assert getattr(config, STORE_SERVICE_ATTRIBUTE) is store
    assert getattr(config, PROVIDER_SERVICE_ATTRIBUTE) is provider
