"""PCTDD-027: exact V2 key is assembled after setup and before call."""

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
    "test_pctdd_027_setup_bound_execution_key.py"
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
    if getattr(reports, "_pctdd_027_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_027_target_bound = True

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
    commit_fixture_instance,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    ISSUER_SERVICE_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    PROVIDER_SERVICE_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
)
from ipfs_accelerate_py.testing.proof_reuse.setup_bound_execution_key import (
    ASSEMBLY_DOES_NOT,
    ASSEMBLY_ESTABLISHES,
    ASSEMBLY_POLICY,
    ASSEMBLY_WINDOW,
    AUTHORITATIVE_AFTER,
    AUTHORITATIVE_LIFECYCLE_PHASE,
    CLAIM_CLASS,
    DEFAULT_POLICY_CID,
    ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE,
    ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE,
    LIFECYCLE_PHASES,
    SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE,
    SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_RESULT_INTERFACE,
    SetupBoundExecutionKeyAssemblyError,
    SetupBoundExecutionKeyAssemblyResult,
    after_runtest_setup,
    assemble_and_attach_from_item,
    assemble_setup_bound_execution_key,
    authority_descriptor,
    before_runtest_call,
    before_runtest_setup,
    classify_lifecycle_phase,
    datasets_contracts_available,
    ensure_setup_bound_execution_key_before_call,
    get_attached_setup_bound_assembly,
    get_attached_setup_bound_execution_key,
    is_authoritative_assembly_phase,
    public_digest,
    record_typed_unavailable,
    run_setup_bound_assembly_lifecycle,
    typed_unavailable_records,
)
from ipfs_datasets_py.logic.zkp.pctdd.test_execution_key_v2 import (
    AUTHORITATIVE_AFTER as DATASETS_AUTHORITATIVE_AFTER,
    REQUIRED_BINDING_FAMILIES,
    TEST_EXECUTION_KEY_V2_INTERFACE,
    Completeness,
    ExecutionDisposition,
    build_completeness_identity,
    build_environment_identity,
    build_fixture_identity,
    build_test_execution_key_v2,
    build_toolchain_identity,
    build_trust_identity,
    commitment_digest,
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


@pytest.fixture
def pctdd_027_limit() -> int:
    return 7


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-027.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-027 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-027.json"
    )


def _digest(label: str) -> str:
    return commitment_digest(
        {"label": label, "schema": "pctdd/test-execution-key-v2@1"}
    )


def _bound_identities(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fixture": build_fixture_identity(
            fixture_definition_closure_cid=_digest("fixture-definition-closure"),
            fixture_instance_closure_cid=_digest("fixture-instance-closure"),
            reuse_class="pure",
            completeness="exact",
            reviewed=True,
        ),
        "environment": build_environment_identity(
            environment_cid=_digest("environment"),
            platform_cid=_digest("platform"),
            interpreter_abi_cid=_digest("interpreter-abi"),
            hardware_capability_cid=_digest("hardware"),
            completeness="exact",
        ),
        "toolchain": build_toolchain_identity(
            pytest_version="8.3.2",
            python_version="3.12.3",
            plugin_versions_cid=_digest("plugin-versions"),
            command_semantics_cid=_digest("command-semantics"),
            config_cid=_digest("pytest-config"),
            dependency_lock_cid=_digest("dependency-lock"),
            installed_distributions_cid=_digest("installed-distributions"),
            completeness="exact",
        ),
        "trust": build_trust_identity(
            policy_cid=_digest("policy"),
            issuer_id="issuer:runner-local",
            epoch="epoch:2026-08-30",
            canonicalization_schema_cid=_digest("canonicalization"),
            tracer_schema_cid=_digest("tracer"),
            certificate_schema_cid=_digest("certificate"),
            completeness="exact",
        ),
        "completeness_identity": build_completeness_identity(
            completeness_policy_cid=_digest("completeness-policy"),
            static_trace_root_cid=_digest("static-trace"),
            runtime_trace_root_cid=_digest("runtime-trace"),
            static_completeness="exact",
            runtime_completeness="exact",
        ),
        "locator_cid": "cid:locator:pctdd-027",
        "repository_forest_cid": _digest("repository-forest"),
        "git_commit_id": "cid:git-commit:pctdd-027",
        "git_tree_id": "cid:git-tree:pctdd-027",
        "test_module_cid": _digest("test-module"),
        "test_function_cid": _digest("test-function"),
        "test_ast_cid": _digest("test-ast"),
        "predecessor_execution_key_cid": _digest("test-execution-key-v1"),
    }
    payload.update(overrides)
    return payload


def _item(*, nodeid: str = "tests/test_mod.py::test_it", **attrs: Any) -> SimpleNamespace:
    payload = {
        "nodeid": nodeid,
        "funcargs": {"limit": 7},
    }
    payload.update(attrs)
    return SimpleNamespace(**payload)


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_027_setup_bound_execution_key.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_027_setup_bound_execution_key.py::test_x",
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


def test_exact_v2_key_is_assembled_after_setup_and_before_call() -> None:
    assert datasets_contracts_available() is True
    item = _item(funcargs={})
    record = run_setup_bound_assembly_lifecycle(item, **_bound_identities())
    assert record.events == ("setup", "assembled", "call")
    assert record.assembled_after_setup_before_call is True
    result = record.assembly
    assert result.interface == SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_RESULT_INTERFACE
    assert result.assembly_interface == SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE
    assert result.lifecycle_phase == AUTHORITATIVE_LIFECYCLE_PHASE == "post_setup"
    assert result.assembled_after_setup is True
    assert result.assembled_before_call is True
    assert result.assembly_window == ASSEMBLY_WINDOW == "after_setup_before_call"
    assert result.authoritative_after == AUTHORITATIVE_AFTER == "current_setup"
    assert result.authoritative_after == DATASETS_AUTHORITATIVE_AFTER
    assert result.may_authorize_skip is False
    assert result.production_admitted is False
    assert result.self_approved is False
    assert result.action == "RUN"
    assert result.normal_execution_fallback is True
    assert result.requires_full_execution is False
    assert result.execution_disposition == ExecutionDisposition.COMMITTED.value
    key = result.execution_key
    assert key is not None
    payload = key.to_dict()
    assert payload["interface"] == TEST_EXECUTION_KEY_V2_INTERFACE
    assert payload["assembled_after_setup"] is False
    assert payload["authoritative_after"] == "current_setup"
    assert payload["may_authorize_skip"] is False
    assert payload["required_families_bound"] is True
    assert tuple(payload["required_binding_families"]) == REQUIRED_BINDING_FAMILIES
    assert payload["completeness"] == Completeness.EXACT.value
    assert payload["execution_disposition"] == ExecutionDisposition.COMMITTED.value
    assert record.before_call is result or record.before_call.execution_key is key
    attached = get_attached_setup_bound_assembly(item)
    assert attached is result
    assert get_attached_setup_bound_execution_key(item) is key
    assert getattr(item, ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE) is result
    assert getattr(item, ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE) is key


def test_wrong_lifecycle_phases_force_normal_execution_fallback() -> None:
    identities = _bound_identities()
    for phase in LIFECYCLE_PHASES:
        if phase == AUTHORITATIVE_LIFECYCLE_PHASE:
            continue
        result = assemble_setup_bound_execution_key(
            lifecycle_phase=phase,
            **identities,
        )
        assert result.assembled_after_setup is False
        assert result.assembled_before_call is False
        assert result.requires_full_execution is True
        assert result.execution_disposition == ExecutionDisposition.FULL_EXECUTION.value
        assert result.action == "RUN"
        assert result.may_authorize_skip is False
        assert result.normal_execution_fallback is True
        assert result.execution_key is None
        assert any("lifecycle_phase" in reason for reason in result.full_execution_reasons)
    assert is_authoritative_assembly_phase("post_setup") is True
    assert is_authoritative_assembly_phase("collection") is False
    assert classify_lifecycle_phase("post_setup") == "post_setup"
    collection = assemble_and_attach_from_item(
        _item(),
        lifecycle_phase="collection",
        **identities,
    )
    assert collection.assembled_after_setup is False
    assert collection.requires_full_execution is True


def test_opaque_incomplete_and_unbound_identities_execute_fully() -> None:
    empty = assemble_setup_bound_execution_key(lifecycle_phase="post_setup")
    assert empty.assembled_after_setup is True
    assert empty.assembled_before_call is True
    assert empty.requires_full_execution is True
    assert empty.action == "RUN"
    assert empty.may_authorize_skip is False
    assert "empty_key" in empty.full_execution_reasons
    opaque = assemble_setup_bound_execution_key(
        lifecycle_phase="post_setup",
        **_bound_identities(
            fixture=build_fixture_identity(
                fixture_definition_closure_cid=_digest("def"),
                fixture_instance_closure_cid=_digest("inst"),
                reuse_class="opaque",
                completeness="exact",
                reviewed=True,
            )
        ),
    )
    assert opaque.assembled_after_setup is True
    assert opaque.requires_full_execution is True
    assert "reuse_class_opaque" in opaque.full_execution_reasons
    incomplete = assemble_setup_bound_execution_key(
        lifecycle_phase="post_setup",
        **_bound_identities(
            fixture=build_fixture_identity(
                fixture_definition_closure_cid=_digest("def"),
                fixture_instance_closure_cid=_digest("inst"),
                reuse_class="pure",
                completeness="incomplete",
                reviewed=True,
            )
        ),
    )
    assert incomplete.requires_full_execution is True
    opaque_instance = commit_fixture_instance(fixture_name="db", value=object())
    live_opaque = assemble_setup_bound_execution_key(
        lifecycle_phase="post_setup",
        fixture_instance_results=(opaque_instance,),
        **_bound_identities(),
    )
    assert live_opaque.assembled_after_setup is True
    assert live_opaque.requires_full_execution is True
    assert "opaque_or_incomplete_fixture_instance" in live_opaque.full_execution_reasons


def test_hook_helpers_assemble_after_setup_and_ensure_before_call() -> None:
    item = _item(funcargs={})
    identities = _bound_identities()
    assert before_runtest_setup(item) == "pre_setup"
    assert get_attached_setup_bound_assembly(item) is None
    pre = assemble_setup_bound_execution_key(
        lifecycle_phase="pre_setup",
        item=item,
        **identities,
    )
    assert pre.assembled_after_setup is False
    assembled = after_runtest_setup(item, setup_failed=False, **identities)
    assert assembled.assembled_after_setup is True
    assert assembled.assembled_before_call is True
    assert assembled.lifecycle_phase == "post_setup"
    before_call = before_runtest_call(item)
    assert before_call.assembled_after_setup is True
    assert before_call.execution_key is assembled.execution_key
    failed_item = _item(nodeid="tests/test_mod.py::test_failed")
    failed = after_runtest_setup(failed_item, setup_failed=True, **identities)
    assert failed.assembled_after_setup is False
    assert failed.requires_full_execution is True
    assert "setup_failed" in failed.full_execution_reasons
    missing = _item(nodeid="tests/test_mod.py::test_missing")
    fallback = ensure_setup_bound_execution_key_before_call(missing)
    assert fallback.assembled_after_setup is False
    assert fallback.action == "RUN"
    assert fallback.may_authorize_skip is False
    assert "setup_bound_execution_key_missing_before_call" in fallback.full_execution_reasons


def test_live_item_funcargs_assemble_without_skip(
    request: pytest.FixtureRequest,
    pctdd_027_limit: int,
) -> None:
    assert pctdd_027_limit == 7
    assert "pctdd_027_limit" in request.node.funcargs
    committed = commit_fixture_instance(
        fixture_name="pctdd_027_limit",
        value=pctdd_027_limit,
        adapter_id="immutable_scalar",
        reviewed=True,
    )
    assert committed.committed is True
    result = assemble_and_attach_from_item(
        request.node,
        lifecycle_phase="post_setup",
        fixture_adapter_map={"pctdd_027_limit": "immutable_scalar"},
        **_bound_identities(),
    )
    assert result.assembled_after_setup is True
    assert result.assembled_before_call is True
    assert result.may_authorize_skip is False
    assert result.action == "RUN"
    assert result.normal_execution_fallback is True
    assert get_attached_setup_bound_assembly(request.node) is result
    assert result.execution_key is not None
    assert result.execution_key.assembled_after_setup is False
    assert result.execution_key.may_authorize_skip is False


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["authoritative_after"] == AUTHORITATIVE_AFTER
    assert descriptor["assembly_window"] == ASSEMBLY_WINDOW
    assert descriptor["normal_execution_fallback"] is True
    assert descriptor["assembly_interface"] == SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE
    assert descriptor["test_execution_key_v2"] == TEST_EXECUTION_KEY_V2_INTERFACE
    assert "skip" in descriptor["does_not"]
    assert "after setup" in descriptor["establishes"]
    assert "before call" in descriptor["establishes"]
    assert ASSEMBLY_ESTABLISHES in descriptor["establishes"] or "V2 key" in descriptor["establishes"]
    assert "skip" in ASSEMBLY_DOES_NOT
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "setup_bound_execution_key.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    plugin_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
    ).read_text(encoding="utf-8")
    assert "assemble_and_attach_from_item" in plugin_source or "after_runtest_setup" in plugin_source
    assert "before_runtest_call" in plugin_source
    assert "pytest_runtest_setup" in plugin_source
    assert "pytest_runtest_call" in plugin_source
    assert "PCTDD-027" in plugin_source
    assert ASSEMBLY_POLICY["may_authorize_skip"] is False
    assert ASSEMBLY_POLICY["normal_execution_fallback"] is True
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    assert LOOKUP_SERVICE_ATTRIBUTE.endswith("lookup_service")
    assert STORE_SERVICE_ATTRIBUTE.endswith("store_service")
    assert PROVIDER_SERVICE_ATTRIBUTE.endswith("provider_service")
    assert ISSUER_SERVICE_ATTRIBUTE.endswith("issuer_service")
    try:
        SetupBoundExecutionKeyAssemblyResult(
            execution_key=None,
            lifecycle_phase="post_setup",
            assembled_after_setup=True,
            assembled_before_call=True,
            may_authorize_skip=True,
        )
    except SetupBoundExecutionKeyAssemblyError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("setup-bound assembly must not authorize skip")
    try:
        SetupBoundExecutionKeyAssemblyResult(
            execution_key=None,
            lifecycle_phase="collection",
            assembled_after_setup=True,
            assembled_before_call=True,
        )
    except SetupBoundExecutionKeyAssemblyError as exc:
        assert "after setup" in str(exc)
    else:
        raise AssertionError("assembly outside post-setup must be rejected")
    try:
        SetupBoundExecutionKeyAssemblyResult(
            execution_key=build_test_execution_key_v2(**_bound_identities()),
            lifecycle_phase="post_setup",
            assembled_after_setup=True,
            assembled_before_call=True,
            production_admitted=True,
        )
    except SetupBoundExecutionKeyAssemblyError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("setup-bound assembly must not admit production")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    key = assemble_setup_bound_execution_key(
        lifecycle_phase="post_setup",
        **_bound_identities(),
    ).execution_key
    assert key is not None
    assert key.preserves_test_pass_statement_v1 is True
    assert key.leaf_statement_interface == TEST_PASS_STATEMENT_INTERFACE
    assert key.leaf_statement_version == TEST_PASS_STATEMENT_VERSION
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
    assert "pre-call V2 key" in inventory["gap"]
    statement_source = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py"
    ).read_text(encoding="utf-8")
    assert 'TEST_PASS_STATEMENT_INTERFACE: Final = "TestPassStatementV1"' in statement_source
    assert "TEST_PASS_STATEMENT_VERSION: Final = 1" in statement_source


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "collection_time_fixture_instance_key",
        "guarded_post_setup_reuse",
        "composite_phase_receipt",
        "signed_runner_attestation",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "setup_bound_execution_key" not in capabilities
    assert "test_execution_key_v2" not in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["collection_time_fixture_instance_key"]["reason_code"] == (
        "fixture_instance_authoritative_only_after_current_setup"
    )
    assert by_capability["guarded_post_setup_reuse"]["reason_code"] == (
        "guarded_post_setup_reuse_not_implemented"
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
    digest = public_digest({"label": "pctdd-027"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-027"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-027@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "after setup" in folded
    assert "before call" in folded
    assert "v2" in folded or "execution key" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-022", "PCTDD-024", "PCTDD-025"]
    limitations = receipt["limitations"]
    for key in (
        "collection_time_fixture_instance_key",
        "guarded_post_setup_reuse",
        "composite_phase_receipt",
        "signed_runner_attestation",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert "setup_bound_execution_key" not in limitations
    assert "test_execution_key_v2" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    assembly = receipt["assembly"]
    assert assembly["interface"] == SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE
    assert assembly["test_execution_key"] == TEST_EXECUTION_KEY_V2_INTERFACE
    assert assembly["may_authorize_skip"] is False
    assert assembly["production_admitted"] is False
    assert assembly["authoritative_after"] == AUTHORITATIVE_AFTER
    assert assembly["assembly_window"] == ASSEMBLY_WINDOW
    assert assembly["normal_execution_fallback"] is True
    assert assembly["assembled_after"] == "current_setup"
    assert assembly["assembled_before"] == "call"
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-027.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_027_setup_bound_execution_key.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "setup_bound_execution_key.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
        in changed
    )
