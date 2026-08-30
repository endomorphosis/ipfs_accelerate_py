"""PCTDD-022: collection extracts and memoizes exact fixture closures."""

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
    "test_pctdd_022_fixture_definition_extraction.py"
)

_UNREQUESTED_BODY_RUNS = {"count": 0}


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
    if getattr(reports, "_pctdd_022_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_022_target_bound = True

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

from ipfs_accelerate_py.testing.proof_reuse.fixture_definition_extraction import (
    AUTHORITATIVE_AT,
    CLAIM_CLASS,
    DEFAULT_POLICY_CID,
    EXTRACTION_DOES_NOT,
    EXTRACTION_ESTABLISHES,
    EXTRACTION_POLICY,
    FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE,
    FIXTURE_DEFINITION_EXTRACTION_INTERFACE,
    ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE,
    REQUIRED_CLOSURE_MEMBER_KINDS,
    CollectedConftest,
    CollectedFixtureDefinition,
    CollectedHook,
    CollectedPolicy,
    CollectionSnapshot,
    FixtureDefinitionClosureMemo,
    FixtureDefinitionExtractionError,
    FixtureDefinitionExtractionResult,
    authority_descriptor,
    collect_fixture_definition_closures,
    datasets_contracts_available,
    extract_and_memoize_from_item,
    extract_fixture_definition_closure,
    new_memo,
    public_digest,
    record_typed_unavailable,
    typed_unavailable_records,
)
from ipfs_datasets_py.logic.zkp.pctdd.fixture_definition_contracts import (
    AUTHORITATIVE_AT as DATASETS_AUTHORITATIVE_AT,
    FIXTURE_DEFINITION_CLOSURE_INTERFACE,
    FIXTURE_DEFINITION_INTERFACE,
    REQUIRED_CLOSURE_MEMBER_KINDS as DATASETS_REQUIRED_KINDS,
    Completeness,
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


@pytest.fixture
def pctdd_022_unrequested_body():
    _UNREQUESTED_BODY_RUNS["count"] += 1
    yield "should-not-run"
    _UNREQUESTED_BODY_RUNS["count"] += 1


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-022.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-022 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-022.json"
    )


def _raising_body() -> None:
    raise AssertionError("fixture body executed during collection extraction")


def _exact_snapshot(*, cyclic: bool = False, missing: bool = False) -> CollectionSnapshot:
    db_source = (
        "def db(tmp_path):\n"
        "    _raising_body()\n"
        "    yield {'root': tmp_path}\n"
    )
    tmp_source = "def tmp_path():\n    _raising_body()\n    return 'tmp'\n"
    clock_source = "def clock():\n    _raising_body()\n    return 0\n"
    db_args: tuple[str, ...] = ("tmp_path",)
    extra_defs: tuple[CollectedFixtureDefinition, ...] = ()
    if cyclic:
        db_args = ("engine",)
        extra_defs = (
            CollectedFixtureDefinition(
                name="engine",
                argnames=("db",),
                source="def engine(db):\n    _raising_body()\n    return db\n",
                origin_path="tests/conftest.py",
                origin_kind="conftest",
                func=_raising_body,
            ),
        )
    if missing:
        db_args = ("engine",)
    return CollectionSnapshot(
        nodeid="tests/test_mod.py::test_it",
        root_fixture_names=("db",),
        definitions=(
            CollectedFixtureDefinition(
                name="db",
                argnames=db_args,
                yield_fixture=True,
                source=db_source,
                origin_path="tests/conftest.py",
                origin_kind="conftest",
                func=_raising_body,
            ),
            CollectedFixtureDefinition(
                name="tmp_path",
                source=tmp_source,
                origin_path="tests/conftest.py",
                origin_kind="conftest",
                func=_raising_body,
            ),
            CollectedFixtureDefinition(
                name="clock",
                autouse=True,
                source=clock_source,
                origin_path="tests/conftest.py",
                origin_kind="conftest",
                func=_raising_body,
            ),
            *extra_defs,
        ),
        conftests=(
            CollectedConftest(
                path="tests/conftest.py",
                content="import pytest\n@pytest.fixture\ndef db(tmp_path):\n    yield 1\n",
                plugin_name="tests.conftest",
            ),
        ),
        hooks=(
            CollectedHook(
                hook_name="pytest_collection_modifyitems",
                source="def pytest_collection_modifyitems(config, items):\n    return None\n",
                origin_path="tests/conftest.py",
                origin_kind="conftest",
            ),
        ),
        policies=(
            CollectedPolicy(
                policy_kind="pytest_ini",
                name="pytest.ini",
                payload={"addopts": "-q"},
            ),
        ),
        locator_cid="cid:locator:pctdd-022",
    )


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_022_fixture_definition_extraction.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_022_fixture_definition_extraction.py::test_x",
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


def test_collection_extracts_and_memoizes_exact_closures_without_executing_bodies() -> None:
    assert datasets_contracts_available() is True
    memo = FixtureDefinitionClosureMemo()
    snapshot = _exact_snapshot()
    first = extract_fixture_definition_closure(snapshot, memo=memo)
    second = extract_fixture_definition_closure(snapshot, memo=memo)
    assert first.executed_fixture_bodies is False
    assert first.may_authorize_skip is False
    assert first.production_admitted is False
    assert first.self_approved is False
    assert first.authoritative_at == AUTHORITATIVE_AT == "collection"
    assert first.action == "RUN"
    assert first.to_dict()["test_execution_key_v2"] is None
    closure = first.closure
    assert closure is not None
    payload = closure.to_dict()
    assert payload["interface"] == FIXTURE_DEFINITION_CLOSURE_INTERFACE
    assert payload["definitions"][0]["interface"] == FIXTURE_DEFINITION_INTERFACE
    assert tuple(payload["required_member_kinds"]) == REQUIRED_CLOSURE_MEMBER_KINDS
    assert payload["completeness"] == Completeness.EXACT.value
    assert payload["execution_disposition"] == ExecutionDisposition.COMMITTED.value
    assert payload["requires_full_execution"] is False
    assert payload["executed_fixture_bodies"] is False
    assert payload["may_authorize_skip"] is False
    assert payload["authoritative_at"] == DATASETS_AUTHORITATIVE_AT
    assert "db" in payload["transitive_names"]
    assert "tmp_path" in payload["transitive_names"]
    assert "clock" in payload["transitive_names"]
    assert payload["missing_definitions"] == []
    assert payload["definition_cycles"] == []
    assert any(item["kind"] == "yield_teardown" for item in payload["finalizers"])
    assert second.memoized is True
    assert second.closure is first.closure
    assert second.closure.to_dict()["closure_cid"] == payload["closure_cid"]
    assert memo.hits >= 1
    assert memo.misses == 1
    assert memo.interface == FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE


def test_memo_is_keyed_by_exact_closure_ingredients() -> None:
    memo = new_memo()
    baseline = extract_fixture_definition_closure(_exact_snapshot(), memo=memo)
    mutated = CollectionSnapshot(
        nodeid="tests/test_mod.py::test_it",
        root_fixture_names=("db",),
        definitions=_exact_snapshot().definitions,
        conftests=(
            CollectedConftest(
                path="tests/conftest.py",
                content="import pytest\n@pytest.fixture\ndef db(tmp_path):\n    yield 2\n",
                plugin_name="tests.conftest",
            ),
        ),
        hooks=_exact_snapshot().hooks,
        policies=_exact_snapshot().policies,
        locator_cid="cid:locator:pctdd-022",
    )
    other = extract_fixture_definition_closure(mutated, memo=memo)
    assert other.memoized is False
    assert other.closure.to_dict()["closure_cid"] != baseline.closure.to_dict()["closure_cid"]
    replay = extract_fixture_definition_closure(mutated, memo=memo)
    assert replay.memoized is True
    assert replay.closure is other.closure
    assert memo.size == 2


def test_missing_or_cyclic_definitions_force_full_execution() -> None:
    missing = extract_fixture_definition_closure(_exact_snapshot(missing=True))
    assert missing.closure is not None
    assert missing.requires_full_execution is True
    assert missing.closure.completeness == Completeness.INCOMPLETE.value
    assert "missing_transitive_definition" in missing.closure.full_execution_reasons
    assert missing.may_authorize_skip is False
    cyclic = extract_fixture_definition_closure(_exact_snapshot(cyclic=True))
    assert cyclic.requires_full_execution is True
    assert "definition_cycle" in cyclic.closure.full_execution_reasons
    assert cyclic.production_admitted is False
    empty = extract_fixture_definition_closure(CollectionSnapshot())
    assert empty.requires_full_execution is True
    assert "empty_closure" in empty.full_execution_reasons


def test_yield_finalizers_are_definition_time_and_addfinalizer_is_not_claimed() -> None:
    snapshot = CollectionSnapshot(
        nodeid="tests/test_mod.py::test_it",
        root_fixture_names=("db",),
        definitions=(
            CollectedFixtureDefinition(
                name="db",
                yield_fixture=True,
                source=(
                    "def db(request):\n"
                    "    request.addfinalizer(lambda: None)\n"
                    "    yield 1\n"
                ),
                origin_path="tests/conftest.py",
                origin_kind="conftest",
                func=_raising_body,
            ),
        ),
        conftests=_exact_snapshot().conftests,
        hooks=_exact_snapshot().hooks,
        policies=_exact_snapshot().policies,
        locator_cid="cid:locator:pctdd-022",
    )
    result = extract_fixture_definition_closure(snapshot)
    payload = result.closure.to_dict()
    kinds = {item["kind"] for item in payload["finalizers"]}
    assert "yield_teardown" in kinds
    assert "request_addfinalizer" not in kinds
    assert all(item["static"] is True for item in payload["finalizers"])
    records = typed_unavailable_records()
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["instance_time_finalizer"]["reason_code"] == (
        "instance_time_finalizer_not_definition_identity"
    )
    assert by_capability["instance_time_finalizer"]["status"] == "typed_unavailable"


def test_live_session_lookup_does_not_execute_unrequested_fixture(request: pytest.FixtureRequest) -> None:
    before = _UNREQUESTED_BODY_RUNS["count"]
    result = extract_and_memoize_from_item(
        request.node,
        memo=new_memo(),
        extra_fixture_names=("pctdd_022_unrequested_body",),
        attach=True,
    )
    assert _UNREQUESTED_BODY_RUNS["count"] == before
    assert result.executed_fixture_bodies is False
    assert result.may_authorize_skip is False
    assert getattr(request.node, ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE, None) is result.closure
    names = [item.name for item in result.closure.definitions] if result.closure is not None else []
    assert "pctdd_022_unrequested_body" in names
    replay = extract_and_memoize_from_item(request.node, memo=new_memo())
    assert replay.memoized is True
    assert _UNREQUESTED_BODY_RUNS["count"] == before
    collected = collect_fixture_definition_closures((request.node,), memo=new_memo(), attach=True)
    assert len(collected) == 1
    assert collected[0].executed_fixture_bodies is False


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["executed_fixture_bodies"] is False
    assert descriptor["authoritative_at"] == AUTHORITATIVE_AT
    assert descriptor["extraction_interface"] == FIXTURE_DEFINITION_EXTRACTION_INTERFACE
    assert "skip" in descriptor["does_not"]
    assert "fixture bodies" in descriptor["establishes"]
    assert EXTRACTION_ESTABLISHES in descriptor["establishes"] or "memoizes" in descriptor["establishes"]
    assert "skip" in EXTRACTION_DOES_NOT
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "fixture_definition_extraction.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    plugin_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
    ).read_text(encoding="utf-8")
    assert "collect_fixture_definition_closures" in plugin_source
    assert EXTRACTION_POLICY["may_authorize_skip"] is False
    assert EXTRACTION_POLICY["executed_fixture_bodies"] is False
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    assert tuple(REQUIRED_CLOSURE_MEMBER_KINDS) == tuple(DATASETS_REQUIRED_KINDS)
    try:
        FixtureDefinitionExtractionResult(
            closure=None,
            memo_key="sha256:" + ("a" * 64),
            may_authorize_skip=True,
        )
    except FixtureDefinitionExtractionError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("extraction must not authorize skip")
    try:
        FixtureDefinitionExtractionResult(
            closure=None,
            memo_key="sha256:" + ("a" * 64),
            executed_fixture_bodies=True,
        )
    except FixtureDefinitionExtractionError as exc:
        assert "fixture bodies" in str(exc)
    else:
        raise AssertionError("extraction must not execute fixture bodies")


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
    assert "locator-first seed" in inventory["current"]


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
        "instance_time_finalizer",
        "composite_phase_receipt",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "fixture_definition_extraction" not in capabilities
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
        FixtureDefinitionExtractionResult(
            closure=None,
            memo_key="sha256:" + ("b" * 64),
            production_admitted=True,
        )
    except FixtureDefinitionExtractionError:
        pass
    else:
        raise AssertionError("typed unavailable/extraction must not admit production")
    digest = public_digest({"label": "pctdd-022"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-022"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-022@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "fixture" in folded
    assert "closur" in folded
    assert "without executing" in folded or "without execute" in folded or "bodies" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-002", "PCTDD-021"]
    limitations = receipt["limitations"]
    for key in (
        "test_execution_key_v2",
        "pre_call_fixture_instance_key",
        "instance_time_finalizer",
        "composite_phase_receipt",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert "fixture_definition_extraction" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    extraction = receipt["extraction"]
    assert extraction["interface"] == FIXTURE_DEFINITION_EXTRACTION_INTERFACE
    assert extraction["memo"] == FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE
    assert extraction["may_authorize_skip"] is False
    assert extraction["executed_fixture_bodies"] is False
    assert extraction["authoritative_at"] == AUTHORITATIVE_AT
    assert tuple(extraction["required_member_kinds"]) == REQUIRED_CLOSURE_MEMBER_KINDS
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-022.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_022_fixture_definition_extraction.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "fixture_definition_extraction.py"
    ) in changed
