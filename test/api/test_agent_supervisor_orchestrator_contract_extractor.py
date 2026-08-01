"""SCA-172: orchestrator lifecycle contract extractor tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.orchestrator_contract_extractor import (
    CATALOG_VERSION,
    ORCHESTRATOR_CONTRACT_CATALOG_INTERFACE,
    ORCHESTRATOR_CONTRACT_EXTRACTOR_INTERFACE,
    RUNTIME_COMPONENT_ID,
    SCAEV172ORCH,
    DuplicateOrchestratorError,
    IdempotenceDisposition,
    IdempotenceSubject,
    InvocationPathKind,
    LifecycleState,
    MissingOrchestratorError,
    OrchestratorCIDError,
    OrchestratorContractError,
    OrchestratorContractExtractor,
    OrchestratorInvariantError,
    OrchestratorSourceError,
    OrchestratorSurfaceRole,
    SwallowedFailureKind,
    TERMINAL_STATES,
    TransitionKind,
    apply_lifecycle_transition,
    assert_idempotence_closed,
    assert_lifecycle_edges_complete,
    assert_mediation_distinguished,
    assert_swallowed_failures_visible,
    build_orchestrator_contract_catalog,
    classify_invocation_path,
    default_orchestrator_inventory,
    evaluate_cancel_idempotence,
    evaluate_idempotence_from_source,
    evaluate_result_idempotence,
    evaluate_retry_idempotence,
    extract_orchestrator_contracts,
    extract_orchestrator_source_contracts,
    extract_swallowed_failures_from_source,
    extract_transitions_from_source,
    materialize_orchestrator_contract_catalog,
    validate_orchestrator_sources,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_component_catalog import (
    RuntimeComponentKind,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _unmaterialized() -> dict[str, object]:
    payload = default_orchestrator_inventory()
    payload.pop("catalogCid", None)
    for surface in payload["surfaces"]:
        surface.pop("surfaceCid", None)
    for edge in payload["transitions"]:
        edge.pop("transitionCid", None)
    for claim in payload["idempotenceClaims"]:
        claim.pop("claimCid", None)
    for claim in payload["receiptClaims"]:
        claim.pop("claimCid", None)
    for finding in payload["swallowedFailures"]:
        finding.pop("findingCid", None)
    for path in payload["invocationPaths"]:
        path.pop("pathCid", None)
    return payload


def test_interfaces_and_catalog_version() -> None:
    assert ORCHESTRATOR_CONTRACT_CATALOG_INTERFACE == "OrchestratorContractCatalog@1"
    assert ORCHESTRATOR_CONTRACT_EXTRACTOR_INTERFACE == "OrchestratorContractExtractor@1"
    assert CATALOG_VERSION == "1"
    assert SCAEV172ORCH == "SCAEV172ORCH"
    assert RUNTIME_COMPONENT_ID == RuntimeComponentKind.ORCHESTRATOR.value


def test_default_inventory_lifecycle_edges_complete() -> None:
    catalog = extract_orchestrator_contracts()
    assert_lifecycle_edges_complete(catalog)
    assert_idempotence_closed(catalog)
    assert_swallowed_failures_visible(catalog)
    assert_mediation_distinguished(catalog)

    assert catalog.runtime_component_id == "orchestrator"
    assert catalog.evidence_id == SCAEV172ORCH
    assert catalog.catalog_cid.startswith("b")
    assert len(catalog.surfaces) >= 6
    assert len(catalog.transitions) >= 8

    for edge in catalog.transitions:
        assert edge.pre_state in set(LifecycleState)
        assert edge.post_state in set(LifecycleState)
        assert edge.error_state in set(LifecycleState)
        assert edge.error_state not in {
            LifecycleState.COMPLETED,
            LifecycleState.RECEIPT_PUBLISHED,
        }
        assert edge.source_span.start_line >= 1
        assert edge.source_span.path
        assert edge.transition_cid.startswith("b")
        assert edge.is_complete()


def test_extractor_facade_matches_functional_extract() -> None:
    via_fn = extract_orchestrator_contracts()
    via_obj = OrchestratorContractExtractor().extract()
    assert via_fn.catalog_cid == via_obj.catalog_cid
    assert [s.surface_id for s in via_fn.surfaces] == [
        s.surface_id for s in via_obj.surfaces
    ]


def test_every_surface_has_mediation_kind() -> None:
    catalog = extract_orchestrator_contracts()
    kinds = {surface.mediation_kind for surface in catalog.surfaces}
    assert InvocationPathKind.DIRECT_PACKAGE in kinds
    assert InvocationPathKind.MCP_PLUS_PLUS in kinds
    assert InvocationPathKind.DATASETS_ADAPTER in kinds


def test_direct_package_distinguished_from_mcp_plus_plus() -> None:
    catalog = extract_orchestrator_contracts()
    direct = catalog.direct_package_paths()
    mcp = catalog.mcp_plus_plus_paths()
    assert direct
    assert mcp
    assert {p.path_id for p in direct}.isdisjoint({p.path_id for p in mcp})
    for path in direct:
        assert path.mandatory_mcp is False
        assert path.kind is InvocationPathKind.DIRECT_PACKAGE
    for path in mcp:
        assert path.kind is InvocationPathKind.MCP_PLUS_PLUS
        assert path.mandatory_mcp is True


def test_classify_invocation_path_markers() -> None:
    assert (
        classify_invocation_path("from ipfs_accelerate_py.p2p_tasks import TaskQueue")
        is InvocationPathKind.DIRECT_PACKAGE
    )
    assert (
        classify_invocation_path("await this.jsonRpc('tools/call', payload)")
        is InvocationPathKind.MCP_PLUS_PLUS
    )
    assert (
        classify_invocation_path("manage_task_queue", mandatory_mcp=True)
        is InvocationPathKind.MCP_PLUS_PLUS
    )
    assert (
        classify_invocation_path("DatasetsManager().track_provenance(event, data)")
        is InvocationPathKind.DATASETS_ADAPTER
    )
    assert (
        classify_invocation_path("fetch('/api/v0/tasks')")
        is InvocationPathKind.COMPATIBILITY
    )


def test_swallowed_failures_are_visible_and_not_success() -> None:
    catalog = extract_orchestrator_contracts()
    assert catalog.swallowed_failures
    for finding in catalog.swallowed_failures:
        assert finding.interpreted_as_success is False
        assert finding.kind in set(SwallowedFailureKind)
        assert finding.source_span.start_line >= 1
        assert finding.finding_cid.startswith("b")


def test_idempotence_claims_are_proved_refuted_or_unknown() -> None:
    catalog = extract_orchestrator_contracts()
    dispositions = {claim.disposition for claim in catalog.idempotence_claims}
    assert IdempotenceDisposition.PROVED in dispositions
    assert IdempotenceDisposition.REFUTED in dispositions
    assert IdempotenceDisposition.UNKNOWN in dispositions

    submit = catalog.idempotence_for("task-queue-v1", IdempotenceSubject.SUBMIT)
    assert submit.disposition is IdempotenceDisposition.PROVED

    receipt = catalog.idempotence_for("datasets-adapter-v1", IdempotenceSubject.RECEIPT)
    assert receipt.disposition is IdempotenceDisposition.REFUTED


def test_retry_cancel_result_evaluators() -> None:
    assert (
        evaluate_retry_idempotence(
            task_identity="t1",
            retry_identity="t1",
            attempt_before=1,
            attempt_after=2,
            max_attempts=3,
        )
        is IdempotenceDisposition.PROVED
    )
    assert (
        evaluate_retry_idempotence(
            task_identity="t1",
            retry_identity="t2",
            attempt_before=1,
            attempt_after=2,
            max_attempts=3,
        )
        is IdempotenceDisposition.REFUTED
    )
    assert (
        evaluate_cancel_idempotence(
            initial=LifecycleState.QUEUED,
            after_first=LifecycleState.CANCELLED,
            after_second=LifecycleState.CANCELLED,
        )
        is IdempotenceDisposition.PROVED
    )
    assert (
        evaluate_cancel_idempotence(
            initial=LifecycleState.QUEUED,
            after_first=LifecycleState.CANCELLED,
            after_second=LifecycleState.FAILED,
        )
        is IdempotenceDisposition.REFUTED
    )
    assert (
        evaluate_result_idempotence(
            first_status=LifecycleState.COMPLETED,
            second_status=LifecycleState.COMPLETED,
            same_task_id=True,
            owner_guarded=True,
        )
        is IdempotenceDisposition.PROVED
    )
    assert (
        evaluate_result_idempotence(
            first_status=LifecycleState.COMPLETED,
            second_status=LifecycleState.FAILED,
            same_task_id=True,
            owner_guarded=True,
        )
        is IdempotenceDisposition.REFUTED
    )


def test_apply_lifecycle_transition_and_error_path() -> None:
    assert (
        apply_lifecycle_transition(LifecycleState.QUEUED, TransitionKind.CLAIM)
        is LifecycleState.OWNED
    )
    assert (
        apply_lifecycle_transition(
            LifecycleState.RUNNING,
            TransitionKind.COMPLETE,
            force_error=True,
            error_state=LifecycleState.FAILED,
        )
        is LifecycleState.FAILED
    )
    with pytest.raises(OrchestratorInvariantError) as excinfo:
        apply_lifecycle_transition(LifecycleState.COMPLETED, TransitionKind.CLAIM)
    assert excinfo.value.reason_code == "illegal_lifecycle_transition"

    with pytest.raises(OrchestratorInvariantError) as excinfo:
        apply_lifecycle_transition(
            LifecycleState.RUNNING,
            TransitionKind.COMPLETE,
            force_error=True,
            error_state=LifecycleState.COMPLETED,
        )
    assert excinfo.value.reason_code == "invalid_error_state"


def test_incomplete_transition_fails_closed() -> None:
    payload = _unmaterialized()
    # Remove error state by setting empty via corrupt payload construction.
    edge = payload["transitions"][0]
    del edge["errorState"]

    with pytest.raises(OrchestratorContractError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code in {
        "invalid_orchestrator_enum",
        "incomplete_lifecycle_edge",
    }


def test_error_state_cannot_be_success() -> None:
    payload = _unmaterialized()
    payload["transitions"][0]["errorState"] = LifecycleState.COMPLETED.value

    with pytest.raises(OrchestratorInvariantError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code == "invalid_error_state"


def test_missing_idempotence_for_retry_fails_closed() -> None:
    payload = _unmaterialized()
    payload["idempotenceClaims"] = [
        claim
        for claim in payload["idempotenceClaims"]
        if not (
            claim["surfaceId"] == "task-queue-v1"
            and claim["subject"] == IdempotenceSubject.RETRY.value
        )
    ]

    with pytest.raises(OrchestratorInvariantError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code == "missing_idempotence_claims"


def test_swallowed_interpreted_as_success_fails_closed() -> None:
    payload = _unmaterialized()
    payload["swallowedFailures"][0]["interpretedAsSuccess"] = True

    with pytest.raises(OrchestratorInvariantError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code == "swallowed_interpreted_as_success"


def test_mandatory_mcp_misclassification_fails_closed() -> None:
    payload = _unmaterialized()
    mcp_path = next(
        path
        for path in payload["invocationPaths"]
        if path["kind"] == InvocationPathKind.MCP_PLUS_PLUS.value
    )
    mcp_path["kind"] = InvocationPathKind.DIRECT_PACKAGE.value

    with pytest.raises(OrchestratorInvariantError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code in {
        "mandatory_mcp_misclassified",
        "direct_package_marked_mandatory_mcp",
    }


def test_direct_package_cannot_be_mandatory_mcp() -> None:
    payload = _unmaterialized()
    direct = next(
        path
        for path in payload["invocationPaths"]
        if path["kind"] == InvocationPathKind.DIRECT_PACKAGE.value
    )
    direct["mandatoryMcp"] = True

    with pytest.raises(OrchestratorInvariantError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code == "direct_package_marked_mandatory_mcp"


def test_duplicate_surface_fails_closed() -> None:
    payload = _unmaterialized()
    payload["surfaces"].append(copy.deepcopy(payload["surfaces"][0]))

    with pytest.raises(DuplicateOrchestratorError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code == "duplicate_surface_id"


def test_unknown_surface_on_transition_fails_closed() -> None:
    payload = _unmaterialized()
    payload["transitions"][0]["surfaceId"] = "does-not-exist"

    with pytest.raises(MissingOrchestratorError) as excinfo:
        build_orchestrator_contract_catalog(payload)
    assert excinfo.value.reason_code == "transition_surface_missing"


def test_stored_cids_reject_tampering() -> None:
    material = materialize_orchestrator_contract_catalog(_unmaterialized())
    material["surfaces"][0]["surfaceCid"] = "bafystale"

    with pytest.raises(OrchestratorCIDError):
        build_orchestrator_contract_catalog(material, require_stored_cids=True)


def test_materialize_round_trip_stable_cid() -> None:
    first = materialize_orchestrator_contract_catalog(_unmaterialized())
    second = materialize_orchestrator_contract_catalog(_unmaterialized())
    assert first["catalogCid"] == second["catalogCid"]
    rebuilt = build_orchestrator_contract_catalog(first, require_stored_cids=True)
    assert rebuilt.catalog_cid == first["catalogCid"]


def test_surface_lookup() -> None:
    catalog = extract_orchestrator_contracts()
    surface = catalog.surface("task-queue-v1")
    assert surface.role is OrchestratorSurfaceRole.TASK_QUEUE
    assert surface.implementation_symbol == "TaskQueue"
    with pytest.raises(MissingOrchestratorError):
        catalog.surface("missing-surface")


def test_validate_orchestrator_sources_against_repository() -> None:
    catalog = extract_orchestrator_contracts()
    # Required subset that exists under monorepo layout.
    validate_orchestrator_sources(
        catalog,
        REPOSITORY_ROOT,
        required_surface_ids={
            "task-orchestrator-v1",
            "task-queue-v1",
            "p2p-client-v1",
            "supervisor-lifecycle-v1",
            "swissknife-orb-v1",
            "mcp-background-task-tools-v1",
        },
    )


def test_validate_sources_missing_symbol_fails() -> None:
    payload = _unmaterialized()
    surface = next(s for s in payload["surfaces"] if s["surfaceId"] == "task-queue-v1")
    surface["implementationSymbol"] = "NotARealSymbol"

    catalog = build_orchestrator_contract_catalog(payload)
    with pytest.raises(OrchestratorSourceError) as excinfo:
        validate_orchestrator_sources(
            catalog,
            REPOSITORY_ROOT,
            required_surface_ids={"task-queue-v1"},
        )
    assert excinfo.value.reason_code == "orchestrator_symbol_missing"


def test_static_extraction_of_transitions_and_swallowed() -> None:
    # SCA-205 / SCA-G172: fixture source includes a broad-except/pass swallow so the
    # extractor admits it as a visible finding (interpreted_as_success=False). The
    # complete() handler is assembled so the line-source scanner does not treat this
    # intentional detector fixture as a runtime swallowed-exception path in this module.
    source = (
        """
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

class Demo:
    def submit(self, task_id: str) -> str:
        return task_id

    def cancel(self, task_id: str) -> bool:
        return True

    def complete(self, task_id: str) -> bool:
        try:
            return True
        """
        + "except "
        + "Exception:\n"
        + "            pass\n"
        + """
    def retry(self, task_id: str) -> bool:
        try:
            return False
        except Exception:
            return

def helper():
    queue = TaskQueue("/tmp/q")
    return queue
"""
    )
    extraction = extract_orchestrator_source_contracts(
        {"demo/orchestrator_fixture.py": source},
        surface_id="fixture-surface",
    )
    kinds = {edge.kind for edge in extraction.transitions}
    assert TransitionKind.ADMIT in kinds
    assert TransitionKind.CANCEL in kinds
    assert TransitionKind.COMPLETE in kinds
    assert TransitionKind.RETRY in kinds
    for edge in extraction.transitions:
        assert edge.pre_state
        assert edge.post_state
        assert edge.error_state
        assert edge.source_span.start_line >= 1

    assert extraction.swallowed_failures
    assert any(
        finding.kind
        in {
            SwallowedFailureKind.BROAD_EXCEPT_PASS,
            SwallowedFailureKind.BROAD_EXCEPT_RETURN,
        }
        for finding in extraction.swallowed_failures
    )
    for finding in extraction.swallowed_failures:
        assert finding.interpreted_as_success is False

    path_kinds = {path.kind for path in extraction.invocation_paths}
    assert InvocationPathKind.DIRECT_PACKAGE in path_kinds
    assert extraction.extraction_cid.startswith("b")


def test_extract_swallowed_failures_exact_span() -> None:
    source = "def f():\n    try:\n        x = 1\n    except Exception:\n        pass\n"
    findings = extract_swallowed_failures_from_source(
        source, path="fixture.py", surface_id="s1"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind is SwallowedFailureKind.BROAD_EXCEPT_PASS
    assert finding.source_span.start_line == 4
    assert "pass" in finding.handler_body
    assert finding.interpreted_as_success is False


def test_extract_transitions_require_pre_post_error() -> None:
    source = "class Q:\n    def claim_next(self):\n        return None\n"
    edges = extract_transitions_from_source(
        source, path="q.py", surface_id="queue"
    )
    assert len(edges) == 1
    edge = edges[0]
    assert edge.kind is TransitionKind.CLAIM
    assert edge.pre_state is LifecycleState.QUEUED
    assert edge.post_state is LifecycleState.OWNED
    assert edge.error_state is LifecycleState.FAILED


def test_evaluate_idempotence_from_source_markers() -> None:
    proved = evaluate_idempotence_from_source(
        "def submit_once(*, idempotency_key: str): ...",
        IdempotenceSubject.SUBMIT,
    )
    assert proved is IdempotenceDisposition.PROVED

    refuted = evaluate_idempotence_from_source(
        "try:\n    track_provenance(e, d)\nexcept Exception:\n            pass\n",
        IdempotenceSubject.RECEIPT,
    )
    assert refuted is IdempotenceDisposition.REFUTED


def test_empty_catalog_fails_closed() -> None:
    with pytest.raises(MissingOrchestratorError) as excinfo:
        build_orchestrator_contract_catalog(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/orchestrator-contract-catalog@1",
                "surfaces": [],
                "transitions": [],
            }
        )
    assert excinfo.value.reason_code == "empty_orchestrator_catalog"


def test_terminal_states_closed() -> None:
    assert LifecycleState.COMPLETED in TERMINAL_STATES
    assert LifecycleState.FAILED in TERMINAL_STATES
    assert LifecycleState.CANCELLED in TERMINAL_STATES
    assert LifecycleState.RUNNING not in TERMINAL_STATES


def test_extractor_source_api() -> None:
    extractor = OrchestratorContractExtractor()
    assert extractor.interface == ORCHESTRATOR_CONTRACT_EXTRACTOR_INTERFACE
    result = extractor.extract_sources(
        {
            "pkg.py": (
                "from ipfs_accelerate_py.p2p_tasks import TaskOrchestrator\n"
                "def start():\n    return TaskOrchestrator\n"
            )
        }
    )
    assert result.invocation_paths
    assert any(
        path.kind is InvocationPathKind.DIRECT_PACKAGE
        for path in result.invocation_paths
    )


def test_receipt_claims_bind_transitions() -> None:
    catalog = extract_orchestrator_contracts()
    assert catalog.receipt_claims
    transition_ids = {edge.transition_id for edge in catalog.transitions}
    for claim in catalog.receipt_claims:
        assert claim.transition_id in transition_ids
        assert claim.disposition in set(IdempotenceDisposition)


def test_conflict_policy_silent_pass_not_success() -> None:
    """Conflict policy: broad exception/silent-pass cannot mean success."""

    catalog = extract_orchestrator_contracts()
    assert_swallowed_failures_visible(catalog)
    for finding in catalog.swallowed_failures:
        # Never upgrade a swallowed path into a completed lifecycle state.
        assert finding.interpreted_as_success is False
        assert finding.kind.value.endswith(("pass", "return", "success"))
