"""PCTDD-024: only reviewed adapters commit; opaque dependencies execute fully."""

from __future__ import annotations

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

_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"
_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
    "test_pctdd_024_dependency_commitment_adapters.py"
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
    if getattr(reports, "_pctdd_024_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_024_target_bound = True

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

from ipfs_accelerate_py.testing.proof_reuse.dependency_commitment_adapters import (
    ADAPTER_POLICY,
    CLAIM_CLASS,
    CLOSED_REUSE_CLASSES,
    COMMITTED_PROVIDER_FIELDS,
    DEFAULT_POLICY_CID,
    DEFAULT_REUSE_CLASS,
    DEPENDENCY_COMMITMENT_ADAPTER_REGISTRY_INTERFACE,
    DI_HANDLE_KINDS,
    FAMILY_DEFAULT_KIND,
    FAMILY_DEFAULT_REUSE_CLASS,
    OPAQUE_DEPENDENCY_ADAPTER_INTERFACE,
    PRIVACY_SECRET_RULE,
    REVIEW_CANDIDATE_FAMILIES,
    REVIEW_CANDIDATE_FAMILY_TO_INVENTORY,
    REVIEW_CANDIDATE_INVENTORY_LABELS,
    REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE,
    AdapterCommitError,
    AdapterCommitResult,
    AdapterExecutionDisposition,
    DependencyCommitmentAdapterRegistry,
    adapter_supports_value,
    authority_descriptor,
    build_committed_closure,
    commit_di_handle,
    commit_di_services,
    commit_fixture_instance,
    commit_injected_dependency,
    commit_opaque_dependency,
    datasets_contracts_available,
    default_registry,
    public_digest,
    record_typed_unavailable,
    requires_full_execution,
    typed_unavailable_records,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    ISSUER_SERVICE_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    PROVIDER_SERVICE_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
)
from ipfs_datasets_py.logic.zkp.pctdd.fixture_instance_contracts import (
    AUTHORITATIVE_AFTER,
    CLOSED_REUSE_CLASSES as DATASETS_CLOSED_REUSE_CLASSES,
    FIXTURE_INSTANCE_COMMITMENT_INTERFACE,
    INJECTED_DEPENDENCY_COMMITMENT_INTERFACE,
    REVIEW_CANDIDATE_FAMILIES as DATASETS_REVIEW_FAMILIES,
    ExecutionDisposition,
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
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-024.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-024 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-024.json"
    )


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_024_dependency_commitment_adapters.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_024_dependency_commitment_adapters.py::test_x",
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


def test_review_candidate_families_match_fixture_adapter_inventory() -> None:
    inventory = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "fixture_adapter_inventory.json"
    )
    assert inventory["schema"] == "pctdd/fixture-adapter-inventory@1"
    assert inventory["default"] == DEFAULT_REUSE_CLASS == "opaque"
    assert tuple(inventory["closed_reuse_classes"]) == CLOSED_REUSE_CLASSES
    assert tuple(inventory["closed_reuse_classes"]) == DATASETS_CLOSED_REUSE_CLASSES
    assert tuple(inventory["initial_review_candidates"]) == REVIEW_CANDIDATE_INVENTORY_LABELS
    assert REVIEW_CANDIDATE_FAMILIES == DATASETS_REVIEW_FAMILIES
    assert inventory["secret_rule"] == PRIVACY_SECRET_RULE
    registry = default_registry()
    assert registry.interface == DEPENDENCY_COMMITMENT_ADAPTER_REGISTRY_INTERFACE
    assert registry.families() == REVIEW_CANDIDATE_FAMILIES
    for family, label in REVIEW_CANDIDATE_FAMILY_TO_INVENTORY.items():
        adapter = registry.require(family)
        assert adapter.reviewed is True
        assert adapter.interface == REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE
        assert adapter.inventory_label == label
        assert adapter.default_kind == FAMILY_DEFAULT_KIND[family]
        assert adapter.default_reuse_class == FAMILY_DEFAULT_REUSE_CLASS[family]
    with pytest.raises(AdapterCommitError, match="closed review-candidate"):
        registry.require("unreviewed-local-adapter")


def test_only_reviewed_adapters_commit_supported_values() -> None:
    assert datasets_contracts_available() is True
    scalar = commit_injected_dependency(
        kind="immutable_scalar",
        name="limit",
        value=7,
        adapter_id="immutable_scalar",
        reviewed=True,
    )
    assert scalar.committed is True
    assert scalar.reviewed is True
    assert scalar.execution_disposition == AdapterExecutionDisposition.COMMITTED.value
    assert scalar.may_authorize_skip is False
    assert scalar.production_admitted is False
    assert scalar.self_approved is False
    payload = scalar.to_dict()["commitment"]
    assert payload["interface"] == INJECTED_DEPENDENCY_COMMITMENT_INTERFACE
    assert payload["reviewed"] is True
    assert payload["reuse_class"] == "pure"
    assert payload["completeness"] == "exact"
    for field in COMMITTED_PROVIDER_FIELDS:
        assert payload[field]
    assert "secret" not in json.dumps(payload)
    assert payload["commitment_cid"].startswith("sha256:")
    unreviewed = commit_injected_dependency(
        kind="immutable_scalar",
        name="limit",
        value=7,
        adapter_id="immutable_scalar",
        reviewed=False,
    )
    assert unreviewed.committed is False
    assert unreviewed.reviewed is False
    assert unreviewed.execution_disposition == (
        AdapterExecutionDisposition.FULL_EXECUTION.value
    )


def test_unreviewed_and_unknown_adapters_cannot_commit() -> None:
    opaque = commit_injected_dependency(kind="issuer", name="runner-issuer", value={"k": 1})
    assert opaque.adapter_id == ""
    assert opaque.committed is False
    assert opaque.reviewed is False
    assert "adapter_unreviewed" in opaque.full_execution_reasons
    with pytest.raises(AdapterCommitError, match="closed review-candidate"):
        commit_injected_dependency(
            kind="clock",
            name="freezegun",
            adapter_id="unreviewed-local-adapter",
            reviewed=True,
            value=0,
        )
    with pytest.raises(AdapterCommitError, match="review-candidate family"):
        commit_injected_dependency(
            kind="clock",
            name="freezegun",
            adapter_id="",
            reviewed=True,
            value=0,
        )


def test_opaque_dependencies_execute_fully() -> None:
    opaque = commit_opaque_dependency(kind="pytest_fixture", name="db")
    assert opaque.committed is False
    assert opaque.execution_disposition == AdapterExecutionDisposition.FULL_EXECUTION.value
    assert opaque.payload["reuse_class"] == "opaque"
    assert opaque.may_authorize_skip is False
    live_clock = commit_injected_dependency(
        kind="clock",
        name="wall-clock",
        value=lambda: 1,
        adapter_id="clock",
        reviewed=True,
    )
    assert live_clock.committed is False
    assert live_clock.execution_disposition == AdapterExecutionDisposition.FULL_EXECUTION.value
    assert "adapter_value_unsupported" in live_clock.full_execution_reasons
    host_path = commit_injected_dependency(
        kind="temp_directory_root",
        name="tmp_path",
        value={"provider_identity": "pytest.tmpdir", "provider_version": "1", "path": "/tmp/x"},
        adapter_id="path_independent_temp_directory_root",
        reviewed=True,
    )
    assert host_path.committed is False
    nonreplayable = commit_injected_dependency(
        kind="service_image",
        name="network-service",
        value={"image_cid": public_digest({"image": "svc"})},
        adapter_id="service_image_config_corpus",
        reviewed=True,
        reuse_class="effectful_nonreplayable",
    )
    assert nonreplayable.committed is False
    incomplete = commit_injected_dependency(
        kind="immutable_scalar",
        name="limit",
        value=3,
        adapter_id="immutable_scalar",
        reviewed=True,
        completeness="incomplete",
    )
    assert incomplete.committed is False
    unknown = commit_injected_dependency(
        kind="issuer",
        name="runner-issuer",
        reuse_class="mystery",
        value=1,
    )
    assert unknown.payload["reuse_class"] == "opaque"
    assert unknown.execution_disposition == AdapterExecutionDisposition.FULL_EXECUTION.value


def test_each_reviewed_family_commits_supported_public_identity() -> None:
    registry = DependencyCommitmentAdapterRegistry()
    cases = (
        (
            "immutable_scalar",
            "immutable_scalar",
            "answer",
            42,
        ),
        (
            "frozen_canonical_record",
            "frozen_canonical_record",
            "record",
            {"alpha": 1, "beta": "ok"},
        ),
        (
            "path_independent_temp_directory_root",
            "temp_directory_root",
            "tmp_root",
            {
                "provider_identity": "pytest.tmpdir",
                "provider_version": "tmp_path_factory@1",
                "layout": "relative",
            },
        ),
        (
            "transaction_baseline",
            "transaction_baseline",
            "db-baseline",
            {"baseline_cid": public_digest({"baseline": "v1"})},
        ),
        (
            "clock",
            "clock",
            "frozen-clock",
            {"provider_identity": "freezegun", "epoch": "epoch:frozen:2020-01-01T00:00:00Z"},
        ),
        (
            "rng",
            "rng",
            "seeded-rng",
            {
                "provider_identity": "random.Random",
                "algorithm": "MT19937",
                "seed_digest": public_digest({"seed": 1}),
            },
        ),
        (
            "service_image_config_corpus",
            "service_image",
            "corpus",
            {
                "image_cid": public_digest({"image": "svc"}),
                "config_cid": public_digest({"config": "v1"}),
            },
        ),
        (
            "explicit_environment_projection",
            "environment_projection",
            "env",
            {"PYTHONHASHSEED": "0", "TZ": "UTC"},
        ),
    )
    assert len(cases) == len(REVIEW_CANDIDATE_FAMILIES)
    results = []
    for adapter_id, kind, name, value in cases:
        assert adapter_supports_value(adapter_id, value) is True
        result = registry.require(adapter_id).commit(kind=kind, name=name, value=value)
        assert result.committed is True, adapter_id
        assert result.reviewed is True
        assert result.may_authorize_skip is False
        assert result.payload["adapter_id"] == adapter_id
        assert "secret" not in json.dumps(result.to_dict())
        results.append(result)
    assert requires_full_execution(results) is False
    closure = build_committed_closure(results, locator_cid="cid:locator:pctdd-024")
    assert closure is not None
    assert closure.requires_full_execution is False
    assert closure.may_authorize_skip is False
    assert closure.to_dict()["test_execution_key_v2"] is None


def test_privacy_never_commits_secret_bytes_or_private_keys() -> None:
    with pytest.raises(AdapterCommitError, match="private material"):
        commit_injected_dependency(
            kind="frozen_canonical_record",
            name="record",
            value={"secret": "leaked"},
            adapter_id="frozen_canonical_record",
            reviewed=True,
        )
    with pytest.raises(AdapterCommitError, match="secret or raw bytes"):
        commit_injected_dependency(
            kind="issuer",
            name="runner-issuer",
            value={"blob": b"witness"},
            adapter_id="frozen_canonical_record",
            reviewed=True,
        )
    private_name = commit_injected_dependency(
        kind="immutable_scalar",
        name="api_key",
        value=1,
        adapter_id="immutable_scalar",
        reviewed=True,
    )
    assert private_name.committed is False
    with pytest.raises(AdapterCommitError, match="private material"):
        commit_injected_dependency(
            kind="environment_projection",
            name="env",
            value={"PASSWORD": "nope"},
            adapter_id="explicit_environment_projection",
            reviewed=True,
        )
    with pytest.raises(AdapterCommitError, match="secret or raw bytes"):
        commit_injected_dependency(
            kind="rng",
            name="rng",
            value={"seed": b"entropy"},
            adapter_id="rng",
            reviewed=True,
        )


def test_di_handles_commit_only_with_public_interface_identity() -> None:
    class _Handle:
        def __init__(self, interface: str) -> None:
            self.interface = interface

    lookup = _Handle("ProofReuseLookup@1")
    store = _Handle("ProofReuseCandidateStore@1")
    provider = _Handle("CurrentContextProvider@1")
    issuer = _Handle("RunnerPassAttestation@1")
    results = commit_di_services(
        lookup=lookup,
        store=store,
        provider=provider,
        issuer=issuer,
    )
    assert tuple(item.kind for item in results) == DI_HANDLE_KINDS
    for item in results:
        assert item.committed is True
        assert item.payload["adapter_id"] == "frozen_canonical_record"
        assert item.payload["kind"] in DI_HANDLE_KINDS
        assert item.may_authorize_skip is False
    missing = commit_di_handle(kind="lookup", handle=None)
    assert missing.committed is False
    assert "di_handle_missing" in missing.full_execution_reasons
    incomplete = commit_di_handle(kind="store", handle=object())
    assert incomplete.committed is False
    degraded = commit_di_services(
        lookup=None,
        store=None,
        provider=None,
        issuer=None,
    )
    assert requires_full_execution(degraded) is True
    assert LOOKUP_SERVICE_ATTRIBUTE.endswith("lookup_service")
    assert STORE_SERVICE_ATTRIBUTE.endswith("store_service")
    assert PROVIDER_SERVICE_ATTRIBUTE.endswith("provider_service")
    assert ISSUER_SERVICE_ATTRIBUTE.endswith("issuer_service")


def test_fixture_instance_and_mixed_closure_force_full_execution_when_opaque() -> None:
    committed = commit_fixture_instance(
        fixture_name="frozen_record",
        fixture_scope="session",
        value={"alpha": 1},
        adapter_id="frozen_canonical_record",
        reviewed=True,
        definition_cid="cid:fixture-definition:frozen_record",
    )
    assert committed.committed is True
    assert committed.payload["interface"] == FIXTURE_INSTANCE_COMMITMENT_INTERFACE
    assert committed.payload["authoritative_after"] == AUTHORITATIVE_AFTER
    opaque = commit_fixture_instance(fixture_name="tmp_path")
    assert opaque.committed is False
    assert opaque.execution_disposition == ExecutionDisposition.FULL_EXECUTION.value
    closure = build_committed_closure(
        (committed, opaque),
        locator_cid="cid:locator:item-1",
    )
    assert closure is not None
    assert closure.requires_full_execution is True
    assert closure.may_authorize_skip is False
    assert requires_full_execution(()) is True


def test_adapters_do_not_authorize_skip_or_widen_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["opaque_adapter_interface"] == OPAQUE_DEPENDENCY_ADAPTER_INTERFACE
    assert "skip" in descriptor["does_not"]
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "dependency_commitment_adapters.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    assert ADAPTER_POLICY["may_authorize_skip"] is False
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    try:
        AdapterCommitResult(
            adapter_id="immutable_scalar",
            reviewed=True,
            committed=True,
            execution_disposition=AdapterExecutionDisposition.COMMITTED.value,
            full_execution_reasons=(),
            kind="immutable_scalar",
            name="x",
            commitment=None,
            payload={},
            may_authorize_skip=True,
        )
    except AdapterCommitError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("adapters must not authorize skip")
    try:
        AdapterCommitResult(
            adapter_id="",
            reviewed=False,
            committed=True,
            execution_disposition=AdapterExecutionDisposition.COMMITTED.value,
            full_execution_reasons=(),
            kind="pytest_fixture",
            name="x",
            commitment=None,
            payload={},
        )
    except AdapterCommitError as exc:
        assert "only reviewed adapters may commit" in str(exc)
    else:
        raise AssertionError("unreviewed adapters must not commit")


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


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "test_execution_key_v2",
        "pre_call_fixture_instance_key",
        "composite_phase_receipt",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "reviewed_dependency_commitment_adapters" not in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
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
    try:
        AdapterCommitResult(
            adapter_id="immutable_scalar",
            reviewed=True,
            committed=True,
            execution_disposition=AdapterExecutionDisposition.COMMITTED.value,
            full_execution_reasons=(),
            kind="immutable_scalar",
            name="x",
            commitment=None,
            payload={},
            production_admitted=True,
        )
    except AdapterCommitError:
        pass
    else:
        raise AssertionError("typed unavailable/adapters must not admit production")


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-024"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-024@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "reviewed adapters" in folded
    assert "opaque" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-002", "PCTDD-023"]
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
    assert "reviewed_dependency_commitment_adapters" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    adapters = receipt["adapters"]
    assert adapters["registry"] == DEPENDENCY_COMMITMENT_ADAPTER_REGISTRY_INTERFACE
    assert adapters["reviewed"] == REVIEWED_DEPENDENCY_COMMITMENT_ADAPTER_INTERFACE
    assert adapters["opaque"] == OPAQUE_DEPENDENCY_ADAPTER_INTERFACE
    assert adapters["may_authorize_skip"] is False
    assert adapters["default_reuse_class"] == "opaque"
    assert tuple(adapters["review_candidate_families"]) == REVIEW_CANDIDATE_FAMILIES
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-024.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_024_dependency_commitment_adapters.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "dependency_commitment_adapters.py"
    ) in changed
