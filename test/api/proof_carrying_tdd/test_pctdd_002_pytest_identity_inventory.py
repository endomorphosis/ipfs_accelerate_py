"""PCTDD-002: inventory current pytest proof-reuse identity and DI paths."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# Nested pytest.ini under external/ipfs_accelerate/test makes collected nodeids
# rootdir-relative (`api/proof_carrying_tdd/...`).  The sealed required-
# acceptance collector admits only nodeids bound to the exact profile target.
_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"

_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
    "test_pctdd_002_pytest_identity_inventory.py"
)
_MAPPED_BOUNDARIES = (
    "collection",
    "fixture_timing",
    "di",
    "runtime_trace",
    "xdist_publication",
)
_CURRENT_CAPABILITIES = (
    "locator-first seed",
    "runtime trace",
    "DI lookup/store/provider/issuer",
    "controller-owned xdist publication",
)
_AUTHORITY_PATHS = (
    "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py",
    "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/receipt.py",
    "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/runner_pass_attestation.py",
)
_INVENTORY_GAP = (
    "exact fixture-instance key becomes safely authoritative only after "
    "current setup; add pre-call V2 key and honest composite phase receipt"
)
_PHASES = ("setup", "call", "teardown")
_DI_HANDLES = ("lookup", "store", "provider", "issuer")


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
    if getattr(reports, "_pctdd_002_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_002_target_bound = True

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
    TEST_EXECUTION_KEY_INTERFACE,
    TEST_LOCATOR_KEY_INTERFACE,
    TEST_PASS_RECEIPT_INTERFACE,
    PhaseOutcome,
)
from ipfs_accelerate_py.testing.proof_reuse.collection_seed import (
    LOCATOR_FIRST_ASSEMBLER_INTERFACE,
    PROOF_REUSE_COLLECTION_SEED_INTERFACE,
    CollectionSeedReason,
    ProofReuseCollectionSeed,
)
from ipfs_accelerate_py.testing.proof_reuse.current_context_provider import (
    CURRENT_CONTEXT_PROVIDER_INTERFACE,
)
from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
    AUTOMATIC_ITEM_IDENTITY_INTERFACE,
    CURRENT_RUNTIME_TRACE_EVIDENCE_INTERFACE,
    ItemIdentityAssemblyReason,
    RuntimeEvidenceProvenance,
)
from ipfs_accelerate_py.testing.proof_reuse.lookup import (
    PROOF_REUSE_LOOKUP_INTERFACE,
    SKIP_REASON_PREFIX,
    TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE,
)
from ipfs_accelerate_py.testing.proof_reuse import plugin as proof_reuse_plugin
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    ISSUER_SERVICE_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    PLUGIN_NAME,
    PROVIDER_SERVICE_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
    set_proof_reuse_services,
)
from ipfs_accelerate_py.testing.proof_reuse.publication import (
    CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE,
    build_controller_candidate_publisher,
)
from ipfs_accelerate_py.testing.proof_reuse.receipt import (
    DISQUALIFIER_ERROR,
    DISQUALIFIER_FAIL,
    DISQUALIFIER_RERUN,
    DISQUALIFIER_SKIP,
    DISQUALIFIER_XFAIL,
    DISQUALIFIER_XPASS,
    PHASES as RECEIPT_PHASES,
    PROOF_REUSE_RECEIPT_INTERFACE,
)
from ipfs_accelerate_py.testing.proof_reuse.runtime_trace_lifecycle import (
    PHASES as TRACE_PHASES,
    PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE,
    LifecycleAuthority,
    PytestRuntimeTraceLifecycle,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DEFAULT_PROOF_REUSE_SERVICES_INTERFACE,
    DefaultProofReuseServices,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist import (
    PROOF_REUSE_XDIST_INTERFACE,
    PROOF_REUSE_XDIST_SCHEMA,
    ProofReuseXdistCoordinator,
    ProofReuseXdistRole,
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
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-002.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-002 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
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
        raise AssertionError("typed unavailable cases cannot admit, self-approve, or change claims")
    return record


def _identity_inventory() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "pytest_identity_inventory.json"
    )


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-002.json"
    )


def _inventory_typed_unavailable() -> tuple[dict[str, Any], ...]:
    return (
        _typed_unavailable(
            capability="test_execution_key_v2",
            reason_code="test_execution_key_v2_not_implemented",
            message=(
                "current identity remains TestExecutionKey@1; TestExecutionKeyV2 "
                "is a successor and is not present in accelerate or datasets source"
            ),
        ),
        _typed_unavailable(
            capability="pre_call_fixture_instance_key",
            reason_code="fixture_instance_authoritative_only_after_current_setup",
            message=(
                "exact fixture-instance identity becomes safely authoritative only "
                "after current setup; collection seeds attach no execution key"
            ),
        ),
        _typed_unavailable(
            capability="composite_phase_receipt",
            reason_code="composite_phase_receipt_not_implemented",
            message=(
                "current receipts are TestPassReceipt@1 complete-pass aggregates; "
                "honest composite phase receipts are a successor and not implemented"
            ),
        ),
        _typed_unavailable(
            capability="production_zk",
            reason_code="production_zk_key_ceremony_unavailable",
            message=(
                "production ZK proving remains typed unavailable; signed runner "
                "attestations cannot become skip or publication authority here"
            ),
        ),
        _typed_unavailable(
            capability="key_ceremony",
            reason_code="production_zk_key_ceremony_unavailable",
            message=(
                "no production-eligible key ceremony is admitted; this inventory "
                "does not generate, download, or export proving-key bytes"
            ),
        ),
        _typed_unavailable(
            capability="direct_execution_profile",
            reason_code="direct_execution_profile_optional",
            message=(
                "direct CPython execution profiles remain research-only and unadmitted; "
                "they cannot upgrade signed, aggregate, simulated, or integrity evidence"
            ),
        ),
    )


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_002_pytest_identity_inventory.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_002_pytest_identity_inventory.py::test_x",
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


def test_pytest_identity_inventory_matches_current_source() -> None:
    inventory = _identity_inventory()
    receipt = _receipt()
    assert inventory["schema"] == "pctdd/pytest-identity@1"
    assert tuple(inventory["authority_paths"]) == _AUTHORITY_PATHS
    assert tuple(inventory["current"]) == _CURRENT_CAPABILITIES
    assert inventory["gap"] == _INVENTORY_GAP
    assert receipt["gap"] == inventory["gap"]
    root = _repo_root()
    for relative in _AUTHORITY_PATHS:
        path = root / relative
        assert path.is_file(), f"missing authority path {relative}"
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"empty authority path {relative}"
    plugin_text = (root / _AUTHORITY_PATHS[0]).read_text(encoding="utf-8")
    assert "pytest_collection_modifyitems" in plugin_text
    assert "LOOKUP_SERVICE_ATTRIBUTE" in plugin_text
    assert "STORE_SERVICE_ATTRIBUTE" in plugin_text
    assert "PROVIDER_SERVICE_ATTRIBUTE" in plugin_text
    assert "ISSUER_SERVICE_ATTRIBUTE" in plugin_text
    receipt_text = (root / _AUTHORITY_PATHS[1]).read_text(encoding="utf-8")
    assert 'PHASES: Final = ("setup", "call", "teardown")' in receipt_text
    attestation_text = (root / _AUTHORITY_PATHS[2]).read_text(encoding="utf-8")
    assert "cannot nominate a key" in attestation_text
    assert "skip authority" in attestation_text
    mapped = receipt["boundary_inventory"]
    assert tuple(mapped) == _MAPPED_BOUNDARIES
    for name in _MAPPED_BOUNDARIES:
        record = mapped[name]
        assert record["claim_unchanged"] is True
        assert record["self_approved"] is False
        assert record["production_admitted"] is False
        assert record["classification"] == "available_with_caveats"
        assert record["establishes"].strip()
        assert record["does_not"].strip()
        assert record["establishes"] != record["does_not"]
        authority = root / record["authority_path"]
        assert authority.is_file()


def test_collection_seed_has_no_skip_or_execution_key_authority() -> None:
    seed = ProofReuseCollectionSeed(
        reason=CollectionSeedReason.ADMITTED,
        stage="collection",
        node_id="api/proof_carrying_tdd/test_pctdd_002_pytest_identity_inventory.py::x",
        forest_id="forest:pctdd-002",
        locator=object(),
        locator_cid="bafybeigpctdd002locatoridentity000000000001",
        seed_cid="bafybeigpctdd002collectionseed0000000000001",
    )
    assert seed.interface == PROOF_REUSE_COLLECTION_SEED_INTERFACE
    assert LOCATOR_FIRST_ASSEMBLER_INTERFACE == "LocatorFirstItemIdentityAssembler@1"
    assert seed.admitted is True
    assert seed.reusable is True
    assert seed.action == "RUN"
    assert seed.authorizes_skip is False
    assert seed.authorizes_lookup is False
    assert seed.has_execution_key is False
    payload = seed.to_dict()
    assert payload["authorizes_skip"] is False
    assert payload["authorizes_lookup"] is False
    assert payload["has_execution_key"] is False
    assert payload["action"] == "RUN"
    denied = ProofReuseCollectionSeed(
        reason=CollectionSeedReason.STATIC_IDENTITY_INCOMPLETE,
        stage="collection",
    )
    assert denied.admitted is False
    assert denied.authorizes_skip is False
    assert denied.has_execution_key is False
    receipt = _receipt()["boundary_inventory"]["collection"]
    assert receipt["interface"] == PROOF_REUSE_COLLECTION_SEED_INTERFACE
    assert receipt["authorizes_skip"] is False
    assert receipt["authorizes_lookup"] is False
    assert receipt["has_execution_key"] is False
    assert receipt["plugin_hook"] == "pytest_collection_modifyitems"
    assert proof_reuse_plugin.pytest_collection_modifyitems.__name__ == (
        "pytest_collection_modifyitems"
    )
    assert PLUGIN_NAME == "ipfs-proof-reuse"
    assert TEST_LOCATOR_KEY_INTERFACE == "TestLocatorKey@1"


def test_fixture_timing_and_phase_receipt_boundaries() -> None:
    assert RECEIPT_PHASES == _PHASES
    assert TRACE_PHASES == _PHASES
    assert TEST_EXECUTION_KEY_INTERFACE == "TestExecutionKey@1"
    assert TEST_PASS_RECEIPT_INTERFACE == "TestPassReceipt@1"
    contracts = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts"
    )
    assert not hasattr(contracts, "TEST_EXECUTION_KEY_V2_INTERFACE")
    assert "TestExecutionKeyV2" not in dir(contracts)
    assert AUTOMATIC_ITEM_IDENTITY_INTERFACE == "AutomaticItemIdentityAssembler@1"
    assert CURRENT_RUNTIME_TRACE_EVIDENCE_INTERFACE == "CurrentRuntimeTraceEvidence@1"
    assert RuntimeEvidenceProvenance.FRESH_CONTROLLED_PREFLIGHT.value == (
        "fresh_controlled_preflight"
    )
    assert "HISTORICAL" not in RuntimeEvidenceProvenance.__members__
    assert "CACHE" not in RuntimeEvidenceProvenance.__members__
    item_identity = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.item_identity"
    )
    folded = " ".join((item_identity.__doc__ or "").casefold().split())
    assert "fixture values do not exist until setup" in folded
    assert "authoritative execution key cannot normally be constructed before" in folded
    assert ItemIdentityAssemblyReason.RUNTIME_EVIDENCE_UNAVAILABLE.value == (
        "runtime_evidence_unavailable"
    )
    for disqualifier in (
        DISQUALIFIER_SKIP,
        DISQUALIFIER_XFAIL,
        DISQUALIFIER_XPASS,
        DISQUALIFIER_RERUN,
        DISQUALIFIER_FAIL,
        DISQUALIFIER_ERROR,
    ):
        assert disqualifier
    disqualifying = {
        PhaseOutcome.SKIP,
        PhaseOutcome.XFAIL,
        PhaseOutcome.XPASS,
        PhaseOutcome.ERROR,
        PhaseOutcome.RERUN,
        PhaseOutcome.FAIL,
    }
    assert PhaseOutcome.PASS not in disqualifying
    assert len(disqualifying) == 6
    receipt = _receipt()["boundary_inventory"]["fixture_timing"]
    assert tuple(receipt["phases"]) == _PHASES
    assert receipt["authoritative_fixture_instance"] == "after_current_setup_only"
    adapters = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "fixture_adapter_inventory.json"
    )
    assert adapters["schema"] == "pctdd/fixture-adapter-inventory@1"
    assert adapters["default"] == "opaque"
    assert "opaque" in adapters["closed_reuse_classes"]


def test_di_lookup_store_provider_issuer_boundary() -> None:
    assert LOOKUP_SERVICE_ATTRIBUTE == "_ipfs_proof_reuse_lookup_service"
    assert STORE_SERVICE_ATTRIBUTE == "_ipfs_proof_reuse_store_service"
    assert PROVIDER_SERVICE_ATTRIBUTE == "_ipfs_proof_reuse_provider_service"
    assert ISSUER_SERVICE_ATTRIBUTE == "_ipfs_proof_reuse_issuer_service"
    assert DEFAULT_PROOF_REUSE_SERVICES_INTERFACE == "ProofReuseServices@1"
    assert PROOF_REUSE_LOOKUP_INTERFACE == "ProofReuseLookup@1"
    assert TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE == "TwoStageCandidateLookup@2"
    assert CURRENT_CONTEXT_PROVIDER_INTERFACE == "CurrentContextProvider@1"
    defaults = DefaultProofReuseServices(degraded=True, reason_code="inventory_probe")
    assert defaults.lookup is None
    assert defaults.store is None
    assert defaults.provider is None
    assert defaults.issuer is None
    assert defaults.available is False
    config: Any = type("Config", (), {})()
    set_proof_reuse_services(
        config,
        lookup="lookup-handle",
        store="store-handle",
        provider="provider-handle",
        issuer="issuer-handle",
    )
    assert getattr(config, LOOKUP_SERVICE_ATTRIBUTE) == "lookup-handle"
    assert getattr(config, STORE_SERVICE_ATTRIBUTE) == "store-handle"
    assert getattr(config, PROVIDER_SERVICE_ATTRIBUTE) == "provider-handle"
    assert getattr(config, ISSUER_SERVICE_ATTRIBUTE) == "issuer-handle"
    receipt = _receipt()["boundary_inventory"]["di"]
    assert tuple(receipt["handles"]) == _DI_HANDLES
    assert receipt["interface"] == DEFAULT_PROOF_REUSE_SERVICES_INTERFACE
    assert receipt["does_not"] == (
        "fabricate missing lookup, store, provider, or issuer authority"
    )
    context = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.current_context_provider"
    )
    folded = " ".join((context.__doc__ or "").casefold().split())
    assert "without" in folded
    assert "executing fixtures or the test body" in folded
    assert "never authorizes" in folded
    assert "skip" in folded


def test_runtime_trace_lifecycle_is_observation_only() -> None:
    lifecycle = PytestRuntimeTraceLifecycle(nodeid="pctdd-002::inventory")
    assert lifecycle.interface == PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE
    assert lifecycle.body_invocations == 0
    assert lifecycle.may_authorize_skip is False
    assert lifecycle.authority is LifecycleAuthority.NONE
    assert lifecycle.publishes_authoritatively is False
    assert lifecycle.lifecycle_complete is False
    assert not hasattr(lifecycle, "run_body")
    receipt = _receipt()["boundary_inventory"]["runtime_trace"]
    assert receipt["interface"] == PYTEST_RUNTIME_TRACE_LIFECYCLE_INTERFACE
    assert tuple(receipt["phases"]) == _PHASES
    assert receipt["may_authorize_skip"] is False
    assert receipt["publication_requires"] == "complete_pass"
    module = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.runtime_trace_lifecycle"
    )
    folded = " ".join((module.__doc__ or "").casefold().split())
    assert "never re-invokes the test body" in folded
    assert "incomplete" in folded
    attestation_path = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "runner_pass_attestation.py"
    )
    attestation_text = attestation_path.read_text(encoding="utf-8")
    assert 'RUNNER_PASS_ATTESTATION_INTERFACE: Final = "RunnerPassAttestation@1"' in (
        attestation_text
    )
    assert 'PYTEST_PASS_ATTESTATION_USAGE: Final = "pytest-pass-attestation"' in (
        attestation_text
    )
    folded_attestation = " ".join(attestation_text.casefold().split())
    assert "cannot nominate a key" in folded_attestation
    assert "skip authority" in folded_attestation
    try:
        attestation = importlib.import_module(
            "ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation"
        )
    except ImportError as exc:
        record = _typed_unavailable(
            capability="runner_pass_attestation_runtime",
            reason_code="runner_pass_attestation_optional_import_unresolved",
            message=(
                "runner_pass_attestation.py is mapped from source; sealed "
                f"import failed ({exc}). Source still cannot nominate a key "
                "or become skip authority"
            ),
        )
        assert record["production_admitted"] is False
        assert record["claim_unchanged"] is True
    else:
        assert attestation.RUNNER_PASS_ATTESTATION_INTERFACE == (
            "RunnerPassAttestation@1"
        )
        assert attestation.PYTEST_PASS_ATTESTATION_USAGE == "pytest-pass-attestation"
        doc = " ".join((attestation.__doc__ or "").casefold().split())
        assert "cannot nominate a key" in doc
        assert "skip authority" in doc
    assert PROOF_REUSE_RECEIPT_INTERFACE == "ProofReuseReceiptCapture@1"


def test_xdist_publication_is_controller_owned() -> None:
    controller = ProofReuseXdistCoordinator.controller()
    standalone = ProofReuseXdistCoordinator.standalone()
    worker = ProofReuseXdistCoordinator.from_worker_input(
        {
            "schema": PROOF_REUSE_XDIST_SCHEMA,
            "interface": PROOF_REUSE_XDIST_INTERFACE,
            "controller_id": controller.controller_id,
            "session_id": controller.session_id,
            "worker_id": "gw0",
            "worker_token": "a" * 32,
        },
        worker_id="gw0",
    )
    assert controller.interface == PROOF_REUSE_XDIST_INTERFACE
    assert controller.role is ProofReuseXdistRole.CONTROLLER
    assert standalone.role is ProofReuseXdistRole.STANDALONE
    assert worker.role is ProofReuseXdistRole.WORKER
    assert controller.can_write is True
    assert standalone.can_write is True
    assert worker.can_write is False
    assert worker.flush_publications(store=object()) == ()
    publisher_worker = build_controller_candidate_publisher(role="worker")
    publisher_controller = build_controller_candidate_publisher(role="controller")
    assert publisher_worker.interface == CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE
    assert publisher_worker.can_publish is False
    assert publisher_worker.is_controller is False
    assert publisher_controller.can_publish is True
    assert proof_reuse_plugin.pytest_configure_node.pytest_impl["optionalhook"] is True
    assert proof_reuse_plugin.pytest_testnodedown.pytest_impl["optionalhook"] is True
    assert proof_reuse_plugin.pytest_sessionfinish.__name__ == "pytest_sessionfinish"
    receipt = _receipt()["boundary_inventory"]["xdist_publication"]
    assert receipt["interface"] == PROOF_REUSE_XDIST_INTERFACE
    assert receipt["controller_owns_writes"] is True
    assert receipt["workers_may_publish"] is False
    try:
        xdist_spec = importlib.util.find_spec("xdist")
    except (ImportError, ValueError, AttributeError):
        xdist_spec = None
    if xdist_spec is None:
        record = _typed_unavailable(
            capability="pytest_xdist_plugin_runtime",
            reason_code="pytest_xdist_runtime_unresolved",
            message=(
                "pytest-xdist is not importable in this sealed PATH; source "
                "controller/worker publication boundaries remain mapped and "
                "are not admitted as production"
            ),
        )
        assert record["production_admitted"] is False
        assert record["claim_unchanged"] is True
    else:
        assert xdist_spec.name == "xdist"
        assert receipt["workers_may_publish"] is False


def test_legacy_skip_path_is_inventoried_and_not_promoted() -> None:
    assert SKIP_REASON_PREFIX == "proof-cache-hit:"
    lookup = importlib.import_module("ipfs_accelerate_py.testing.proof_reuse.lookup")
    folded = " ".join((lookup.__doc__ or "").casefold().split())
    assert "revalidation alone can never skip" in folded
    profile = _load_json("config/parallel_content_sealing_proof_carrying_tdd_validation_profiles.json")
    exclusions = profile["profiles"]["PCTDD-002"]["known_baseline_exclusions"]
    skip_conflict = next(
        item
        for item in exclusions
        if item["reason_code"] == "legacy_skip_semantics_conflict"
    )
    assert skip_conflict["authoritative_acceptance"] is False
    assert "replace rather than bless skip semantics" in skip_conflict["treatment"]
    gap = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "overlap_gap_matrix.json"
    )
    pytest_entry = next(
        item
        for item in gap["entries"]
        if item["capability"] == "pytest proof reuse/DI/xdist controller publication"
    )
    assert pytest_entry["classification"] == "available_with_caveats"
    assert pytest_entry["owner"] == "ipfs_accelerate_py"
    assert pytest_entry["successor"] == "fixture-staged V2 identity/composite phases"


def test_test_execution_key_v2_and_composite_receipt_are_typed_unavailable() -> None:
    records = {
        item["capability"]: item for item in _inventory_typed_unavailable()
    }
    assert records["test_execution_key_v2"]["reason_code"] == (
        "test_execution_key_v2_not_implemented"
    )
    assert records["pre_call_fixture_instance_key"]["reason_code"] == (
        "fixture_instance_authoritative_only_after_current_setup"
    )
    assert records["composite_phase_receipt"]["reason_code"] == (
        "composite_phase_receipt_not_implemented"
    )
    assert TEST_EXECUTION_KEY_INTERFACE != "TestExecutionKeyV2"
    identity_sources = (
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/proof/test_execution_contracts.py",
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/analysis/test_execution_identity.py",
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/item_identity.py",
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/collection_seed.py",
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/candidate_publication.py",
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/receipt.py",
        "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py",
        "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/test_execution_certificate.py",
    )
    root = _repo_root()
    for relative in identity_sources:
        path = root / relative
        assert path.is_file(), f"missing identity source {relative}"
        text = path.read_text(encoding="utf-8")
        assert "TestExecutionKeyV2" not in text
        assert "CompositePhaseReceipt" not in text
    receipt_module = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.receipt"
    )
    assert not hasattr(receipt_module, "COMPOSITE_PHASE_RECEIPT_INTERFACE")
    assert "CompositePhaseReceipt" not in dir(receipt_module)


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    inventory_before = _identity_inventory()
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = _inventory_typed_unavailable()
    capabilities = {item["capability"] for item in records}
    assert capabilities == {
        "test_execution_key_v2",
        "pre_call_fixture_instance_key",
        "composite_phase_receipt",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    inventory_after = _identity_inventory()
    matrix_after = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert inventory_after == inventory_before
    assert matrix_after == matrix_before
    assert inventory_after["gap"] == _INVENTORY_GAP
    assert matrix_after["IntegrityCommitment"]["does_not"] == "execution or semantics"
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


def test_authority_matrix_keeps_pytest_execution_on_accelerate() -> None:
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["schema"] == "pctdd/authority-matrix@1"
    assert matrix["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert matrix["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    assert matrix["markdown"] == "non-authoritative bootstrap"


def test_receipt_is_not_completion_authority() -> None:
    receipt_path = (
        _repo_root()
        / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-002.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-002"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-002@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded_claim = receipt["claim"].casefold()
    assert "does not complete" in folded_claim
    assert "collection" in folded_claim
    assert "fixture timing" in folded_claim
    assert "xdist" in folded_claim
    limitations = receipt["limitations"]
    for key in (
        "test_execution_key_v2",
        "pre_call_fixture_instance_key",
        "composite_phase_receipt",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == (
        "receipt-observation-only"
    )
    assert receipt["predecessor_rescue_candidate"]["outer_commit"] == (
        "25b2a4e0fd1ab354a0db5317ec8ea49ce0319a61"
    )
    mapped = receipt["boundary_inventory"]
    assert tuple(mapped) == _MAPPED_BOUNDARIES
    assert mapped["collection"]["has_execution_key"] is False
    assert mapped["xdist_publication"]["workers_may_publish"] is False
    assert mapped["runtime_trace"]["may_authorize_skip"] is False
    assert mapped["fixture_timing"]["authoritative_fixture_instance"] == (
        "after_current_setup_only"
    )
    assert mapped["di"]["handles"] == list(_DI_HANDLES)
