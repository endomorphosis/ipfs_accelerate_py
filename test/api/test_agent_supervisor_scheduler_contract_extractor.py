"""SCA-173: scheduler authority and concurrency contract extractor tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.runtime_component_catalog import (
    RuntimeComponentKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.scheduler_contract_extractor import (
    CATALOG_VERSION,
    EFFECTFUL_TRANSITIONS,
    SCHEDULER_CONTRACT_CATALOG_INTERFACE,
    SCHEDULER_CONTRACT_EXTRACTOR_INTERFACE,
    DuplicateSchedulerError,
    EffectAttempt,
    InterleavingStep,
    LeaseFenceState,
    MissingSchedulerError,
    QueueBucket,
    RecoveryPath,
    ScheduledTask,
    SchedulerAuthorityError,
    SchedulerAuthorityKind,
    SchedulerCIDError,
    SchedulerContractExtractor,
    SchedulerInvariantError,
    SchedulerRelationKind,
    SchedulerRole,
    SchedulerSourceError,
    TransitionKind,
    apply_interleaving,
    assert_authority_partition,
    assert_lease_fence_invariants,
    build_scheduler_contract_catalog,
    check_recovery_path,
    classify_scheduler_authority,
    default_scheduler_inventory,
    enumerate_bounded_interleavings,
    evaluate_lease_fence_gate,
    extract_scheduler_contracts,
    materialize_scheduler_contract_catalog,
    validate_scheduler_sources,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _unmaterialized() -> dict[str, object]:
    payload = default_scheduler_inventory()
    payload.pop("catalogCid", None)
    for surface in payload["surfaces"]:
        surface.pop("surfaceCid", None)
        surface["authority"].pop("authorityCid", None)
    for relation in payload["relations"]:
        relation.pop("relationCid", None)
    for invariant in payload["invariants"]:
        invariant.pop("invariantCid", None)
    return payload


def test_interfaces_and_catalog_version() -> None:
    assert SCHEDULER_CONTRACT_CATALOG_INTERFACE == "SchedulerContractCatalog@1"
    assert SCHEDULER_CONTRACT_EXTRACTOR_INTERFACE == "SchedulerContractExtractor@1"
    assert CATALOG_VERSION == "1"


def test_default_inventory_classifies_every_surface() -> None:
    catalog = extract_scheduler_contracts()
    assert_authority_partition(catalog)
    assert_lease_fence_invariants(catalog)

    kinds = {surface.authority.kind for surface in catalog.surfaces}
    assert SchedulerAuthorityKind.CANONICAL in kinds
    assert SchedulerAuthorityKind.PROVED_ADAPTER in kinds
    assert SchedulerAuthorityKind.LEGACY_ONLY in kinds
    # Default inventory has no open contradictions.
    assert catalog.open_contradictions() == ()

    for surface in catalog.surfaces:
        assert surface.authority.kind in set(SchedulerAuthorityKind)
        assert surface.surface_cid.startswith("b")
        assert surface.authority.authority_cid.startswith("b")
        assert surface.version == surface.authority.version

    assert catalog.catalog_cid.startswith("b")
    assert catalog.runtime_component_id == "scheduler"
    assert RuntimeComponentKind.SCHEDULER.value == catalog.runtime_component_id


def test_extractor_facade_matches_functional_extract() -> None:
    via_fn = extract_scheduler_contracts()
    via_obj = SchedulerContractExtractor().extract()
    assert via_fn.catalog_cid == via_obj.catalog_cid
    assert [s.scheduler_id for s in via_fn.surfaces] == [
        s.scheduler_id for s in via_obj.surfaces
    ]


def test_proved_adapter_binds_versioned_contract_not_shared_name() -> None:
    catalog = extract_scheduler_contracts()
    adapter = catalog.surface("supervisor-resource-usage-adapter-v1")
    canonical = catalog.surface(adapter.authority.canonical_scheduler_id)

    assert adapter.authority.kind is SchedulerAuthorityKind.PROVED_ADAPTER
    assert canonical.authority.kind is SchedulerAuthorityKind.CANONICAL
    assert adapter.role is canonical.role is SchedulerRole.SUPERVISOR_RESOURCE
    assert adapter.authority.adapter_contract_id.startswith("adapter:")
    assert not adapter.authority.adapter_contract_id.startswith("name:")
    # Shared ResourceScheduler symbol alone is not authority.
    assert adapter.implementation_symbol != canonical.implementation_symbol
    assert classify_scheduler_authority(adapter, catalog) is (
        SchedulerAuthorityKind.PROVED_ADAPTER
    )


def test_name_only_adapter_contracts_fail_closed() -> None:
    payload = _unmaterialized()
    adapter = next(
        surface
        for surface in payload["surfaces"]
        if surface["schedulerId"] == "supervisor-resource-usage-adapter-v1"
    )
    adapter["authority"]["adapterContractId"] = "name:ResourceScheduler"

    with pytest.raises(SchedulerAuthorityError) as excinfo:
        build_scheduler_contract_catalog(payload)
    assert excinfo.value.reason_code == "name_only_adapter_forbidden"


def test_name_only_relation_proof_binding_fail_closed() -> None:
    payload = _unmaterialized()
    payload["relations"][0]["proofBinding"] = "name:P2PWorkflowScheduler"

    with pytest.raises(SchedulerAuthorityError) as excinfo:
        build_scheduler_contract_catalog(payload)
    assert excinfo.value.reason_code == "name_only_relation_forbidden"


def test_legacy_only_and_canonical_partition() -> None:
    catalog = extract_scheduler_contracts()
    legacy = catalog.surface("legacy-p2p-workflow-v1")
    assert legacy.authority.kind is SchedulerAuthorityKind.LEGACY_ONLY
    assert legacy.role is SchedulerRole.LEGACY_WORKFLOW

    # MCP workflow is the canonical consumer that imports the legacy surface.
    mcp = catalog.surface("mcp-workflow-adapter-v1")
    assert mcp.authority.kind is SchedulerAuthorityKind.CANONICAL
    rel = next(
        relation
        for relation in catalog.relations
        if relation.relation_id == "rel-mcp-workflow-delegates-legacy-v1"
    )
    assert rel.kind is SchedulerRelationKind.LEGACY_DELEGATION
    assert rel.source_version == mcp.version
    assert rel.target_version == legacy.version


def test_duplicate_role_canonical_fail_closed() -> None:
    payload = _unmaterialized()
    clone = copy.deepcopy(
        next(
            surface
            for surface in payload["surfaces"]
            if surface["schedulerId"] == "mcp-risk-v1"
        )
    )
    clone["schedulerId"] = "mcp-risk-duplicate"
    clone["authority"]["canonicalSchedulerId"] = "mcp-risk-duplicate"
    payload["surfaces"].append(clone)

    with pytest.raises(SchedulerAuthorityError) as excinfo:
        build_scheduler_contract_catalog(payload)
    assert excinfo.value.reason_code == "duplicate_role_canonical"


def test_adapter_cannot_target_different_role() -> None:
    payload = _unmaterialized()
    adapter = next(
        surface
        for surface in payload["surfaces"]
        if surface["schedulerId"] == "supervisor-resource-usage-adapter-v1"
    )
    adapter["authority"]["canonicalSchedulerId"] = "mcp-risk-v1"

    with pytest.raises(SchedulerAuthorityError) as excinfo:
        build_scheduler_contract_catalog(payload)
    assert excinfo.value.reason_code == "authority_role_mismatch"


def test_contradictory_surface_is_typed_and_grants_no_primary() -> None:
    payload = _unmaterialized()
    payload["surfaces"].append(
        {
            "schedulerId": "shadow-resource-v1",
            "displayName": "Unreviewed shadow resource scheduler",
            "role": SchedulerRole.SUPERVISOR_RESOURCE.value,
            "implementationSymbol": "ShadowResourceScheduler",
            "sourcePath": "ipfs_accelerate_py/agent_supervisor/shadow_scheduler.py",
            "packageId": "ipfs_accelerate_py",
            "version": "1",
            "concurrencyBound": 4,
            "supportsLease": True,
            "supportsFence": True,
            "authority": {
                "kind": SchedulerAuthorityKind.CONTRADICTORY.value,
                "canonicalSchedulerId": "supervisor-resource-v1",
                "decision": "unreviewed_shadow_copy",
                "adapterContractId": "",
                "version": "1",
                "sourcePath": (
                    "ipfs_accelerate_py/agent_supervisor/shadow_scheduler.py"
                ),
            },
        }
    )
    catalog = build_scheduler_contract_catalog(payload)
    contradictions = catalog.open_contradictions()
    assert len(contradictions) == 1
    assert contradictions[0].scheduler_id == "shadow-resource-v1"
    # Canonical for the role remains the reviewed primary.
    primary = catalog.canonical_for_role(SchedulerRole.SUPERVISOR_RESOURCE)
    assert primary.scheduler_id == "supervisor-resource-v1"


def test_lease_fence_dominates_effectful_transitions() -> None:
    catalog = extract_scheduler_contracts()
    surface = catalog.surface("supervisor-resource-v1")
    state = LeaseFenceState(
        owner_id="worker-a",
        fencing_token=7,
        lease_expires_at_ms=1000,
        now_ms=500,
        held=True,
    )

    allowed = evaluate_lease_fence_gate(
        state,
        EffectAttempt(
            transition=TransitionKind.START,
            actor_id="worker-a",
            presented_fencing_token=7,
            task_id="t1",
            effectful=True,
        ),
        surface=surface,
    )
    assert allowed.allowed is True
    assert allowed.reason_code == "lease_fence_ok"
    assert allowed.decision_cid.startswith("b")

    stale = evaluate_lease_fence_gate(
        state,
        EffectAttempt(
            transition=TransitionKind.COMPLETE,
            actor_id="worker-a",
            presented_fencing_token=6,
            task_id="t1",
            effectful=True,
        ),
        surface=surface,
    )
    assert stale.allowed is False
    assert stale.reason_code == "stale_fencing_token"

    wrong_owner = evaluate_lease_fence_gate(
        state,
        EffectAttempt(
            transition=TransitionKind.COMPLETE,
            actor_id="worker-b",
            presented_fencing_token=7,
            task_id="t1",
            effectful=True,
        ),
        surface=surface,
    )
    assert wrong_owner.allowed is False
    assert wrong_owner.reason_code == "lease_owner_mismatch"

    expired = evaluate_lease_fence_gate(
        LeaseFenceState(
            owner_id="worker-a",
            fencing_token=7,
            lease_expires_at_ms=100,
            now_ms=500,
            held=True,
        ),
        EffectAttempt(
            transition=TransitionKind.FAIL,
            actor_id="worker-a",
            presented_fencing_token=7,
            task_id="t1",
            effectful=True,
        ),
        surface=surface,
    )
    assert expired.allowed is False
    assert expired.reason_code == "lease_expired"

    observe = evaluate_lease_fence_gate(
        expired,
        EffectAttempt(
            transition=TransitionKind.OBSERVE,
            actor_id="anyone",
            presented_fencing_token=0,
            task_id="t1",
            effectful=False,
        ),
        surface=surface,
    )
    assert observe.allowed is True
    assert observe.reason_code == "observation_without_effect"


def test_effectful_transitions_are_gated() -> None:
    # Every effectful transition requires lease/fence; observe does not.
    assert TransitionKind.OBSERVE not in EFFECTFUL_TRANSITIONS
    for kind in (
        TransitionKind.ADMIT,
        TransitionKind.RESERVE,
        TransitionKind.START,
        TransitionKind.COMPLETE,
        TransitionKind.FAIL,
        TransitionKind.CANCEL,
        TransitionKind.RETRY,
        TransitionKind.CRASH_RECOVER,
        TransitionKind.HEARTBEAT,
    ):
        assert kind in EFFECTFUL_TRANSITIONS


def test_surface_without_lease_denies_effects() -> None:
    catalog = extract_scheduler_contracts()
    legacy = catalog.surface("legacy-p2p-workflow-v1")
    assert legacy.supports_fence is False
    decision = evaluate_lease_fence_gate(
        LeaseFenceState(
            owner_id="worker-a",
            fencing_token=1,
            lease_expires_at_ms=10_000,
            now_ms=1,
            held=True,
        ),
        EffectAttempt(
            transition=TransitionKind.START,
            actor_id="worker-a",
            presented_fencing_token=1,
            task_id="t1",
            effectful=True,
        ),
        surface=legacy,
    )
    assert decision.allowed is False
    assert decision.reason_code == "fence_unsupported_effect_denied"


def test_bounded_interleaving_conserves_admitted_work_and_terminals() -> None:
    initial = (
        ScheduledTask(task_id="a", bucket=QueueBucket.ADMITTED),
        ScheduledTask(task_id="b", bucket=QueueBucket.ADMITTED),
    )
    steps = (
        InterleavingStep(task_id="a", transition=TransitionKind.RESERVE),
        InterleavingStep(task_id="b", transition=TransitionKind.RESERVE),
        InterleavingStep(task_id="a", transition=TransitionKind.START),
        InterleavingStep(task_id="b", transition=TransitionKind.START),
        InterleavingStep(task_id="a", transition=TransitionKind.COMPLETE),
        InterleavingStep(task_id="b", transition=TransitionKind.FAIL),
    )
    trace = apply_interleaving(initial, steps, concurrency_bound=4)
    assert trace.conserved is True
    assert trace.reason_code == "queue_conserved"
    assert trace.initial_total == 2
    assert trace.final_total == 2
    assert trace.terminal_total == 2
    assert trace.active_total == 0
    assert trace.final_accounting.completed == 1
    assert trace.final_accounting.failed == 1
    assert trace.trace_cid.startswith("b")


def test_interleaving_rejects_duplicate_admit() -> None:
    initial = (ScheduledTask(task_id="a", bucket=QueueBucket.ADMITTED),)
    steps = (
        InterleavingStep(
            task_id="a",
            transition=TransitionKind.ADMIT,
            new_admission=True,
        ),
    )
    trace = apply_interleaving(initial, steps)
    assert trace.conserved is False
    assert trace.reason_code == "duplicate_task"


def test_interleaving_rejects_lost_task_transition() -> None:
    steps = (
        InterleavingStep(task_id="missing", transition=TransitionKind.START),
    )
    trace = apply_interleaving((), steps)
    assert trace.conserved is False
    assert trace.reason_code == "lost_task"
    assert "missing" in trace.lost_task_ids


def test_interleaving_lease_fence_blocks_effect() -> None:
    catalog = extract_scheduler_contracts()
    surface = catalog.surface("proof-scheduler-v1")
    initial = (ScheduledTask(task_id="p1", bucket=QueueBucket.RUNNING),)
    steps = (
        InterleavingStep(
            task_id="p1",
            transition=TransitionKind.COMPLETE,
            actor_id="other",
            presented_fencing_token=1,
        ),
    )
    lease = LeaseFenceState(
        owner_id="owner",
        fencing_token=1,
        lease_expires_at_ms=1000,
        now_ms=1,
        held=True,
    )
    trace = apply_interleaving(
        initial,
        steps,
        lease_state=lease,
        surface=surface,
    )
    assert trace.conserved is False
    assert trace.reason_code == "lease_owner_mismatch"


def test_retry_cancel_crash_paths_neither_duplicate_nor_lose_tasks() -> None:
    catalog = extract_scheduler_contracts()
    surface = catalog.surface("supervisor-resource-v1")
    lease = LeaseFenceState(
        owner_id="worker",
        fencing_token=3,
        lease_expires_at_ms=10_000,
        now_ms=100,
        held=True,
    )

    retry = RecoveryPath(
        path_id="retry-same-identity",
        kind="retry",
        scheduler_id=surface.scheduler_id,
        version=surface.version,
        steps=(
            InterleavingStep(
                task_id="job-1",
                transition=TransitionKind.RETRY,
                actor_id="worker",
                presented_fencing_token=3,
            ),
            InterleavingStep(
                task_id="job-1",
                transition=TransitionKind.START,
                actor_id="worker",
                presented_fencing_token=3,
            ),
            InterleavingStep(
                task_id="job-1",
                transition=TransitionKind.COMPLETE,
                actor_id="worker",
                presented_fencing_token=3,
            ),
        ),
    )
    retry_trace = check_recovery_path(
        retry,
        initial_tasks=(
            ScheduledTask(
                task_id="job-1",
                bucket=QueueBucket.RUNNING,
                owner_id="worker",
                fencing_token=3,
                identity_key="job-1",
            ),
        ),
        lease_state=lease,
        surface=surface,
    )
    assert retry_trace.conserved is True
    assert retry_trace.final_accounting.completed == 1
    assert retry_trace.final_total == 1

    cancel = RecoveryPath(
        path_id="cancel-running",
        kind="cancel",
        scheduler_id=surface.scheduler_id,
        version=surface.version,
        steps=(
            InterleavingStep(
                task_id="job-2",
                transition=TransitionKind.CANCEL,
                actor_id="worker",
                presented_fencing_token=3,
            ),
        ),
    )
    cancel_trace = check_recovery_path(
        cancel,
        initial_tasks=(
            ScheduledTask(
                task_id="job-2",
                bucket=QueueBucket.RUNNING,
                owner_id="worker",
                fencing_token=3,
            ),
        ),
        lease_state=lease,
        surface=surface,
    )
    assert cancel_trace.conserved is True
    assert cancel_trace.final_accounting.cancelled == 1

    crash = RecoveryPath(
        path_id="crash-requeue",
        kind="crash",
        scheduler_id=surface.scheduler_id,
        version=surface.version,
        steps=(
            InterleavingStep(
                task_id="job-3",
                transition=TransitionKind.CRASH_RECOVER,
                actor_id="worker",
                presented_fencing_token=3,
            ),
            InterleavingStep(
                task_id="job-3",
                transition=TransitionKind.RESERVE,
                actor_id="worker",
                presented_fencing_token=3,
            ),
            InterleavingStep(
                task_id="job-3",
                transition=TransitionKind.START,
                actor_id="worker",
                presented_fencing_token=3,
            ),
            InterleavingStep(
                task_id="job-3",
                transition=TransitionKind.COMPLETE,
                actor_id="worker",
                presented_fencing_token=3,
            ),
        ),
    )
    crash_trace = check_recovery_path(
        crash,
        initial_tasks=(
            ScheduledTask(
                task_id="job-3",
                bucket=QueueBucket.RUNNING,
                owner_id="worker",
                fencing_token=3,
                identity_key="job-3",
            ),
        ),
        lease_state=lease,
        surface=surface,
    )
    assert crash_trace.conserved is True
    assert crash_trace.final_accounting.completed == 1
    assert crash_trace.duplicate_task_ids == ()
    assert crash_trace.lost_task_ids == ()


def test_recovery_path_cannot_admit_fresh_identity() -> None:
    path = RecoveryPath(
        path_id="bad-retry-admit",
        kind="retry",
        scheduler_id="supervisor-resource-v1",
        version="1",
        steps=(
            InterleavingStep(
                task_id="fresh",
                transition=TransitionKind.ADMIT,
                new_admission=True,
            ),
        ),
    )
    with pytest.raises(SchedulerInvariantError) as excinfo:
        check_recovery_path(
            path,
            initial_tasks=(
                ScheduledTask(task_id="old", bucket=QueueBucket.RUNNING),
            ),
        )
    assert excinfo.value.reason_code == "recovery_duplicate_admit"


def test_enumerate_bounded_interleavings_all_conserve() -> None:
    traces = enumerate_bounded_interleavings(
        ("t1", "t2"),
        concurrency_bound=2,
        max_branching=512,
    )
    assert traces
    for trace in traces:
        assert trace.conserved is True
        assert trace.final_total == 2
        assert trace.active_total + trace.terminal_total == trace.final_total


def test_concurrency_bound_blocks_excess_admission() -> None:
    steps = (
        InterleavingStep(
            task_id="a", transition=TransitionKind.ADMIT, new_admission=True
        ),
        InterleavingStep(
            task_id="b", transition=TransitionKind.ADMIT, new_admission=True
        ),
    )
    trace = apply_interleaving((), steps, concurrency_bound=1)
    assert trace.conserved is False
    assert trace.reason_code == "concurrency_bound_exceeded"


def test_materialize_and_cid_tamper_fail_closed() -> None:
    material = materialize_scheduler_contract_catalog(default_scheduler_inventory())
    assert material["catalogCid"].startswith("b")
    rebuilt = build_scheduler_contract_catalog(material, require_stored_cids=True)
    assert rebuilt.catalog_cid == material["catalogCid"]

    material["surfaces"][0]["surfaceCid"] = "bafystale"
    with pytest.raises(SchedulerCIDError):
        build_scheduler_contract_catalog(material, require_stored_cids=True)


def test_duplicate_scheduler_id_fail_closed() -> None:
    payload = _unmaterialized()
    payload["surfaces"].append(copy.deepcopy(payload["surfaces"][0]))
    with pytest.raises(DuplicateSchedulerError):
        build_scheduler_contract_catalog(payload)


def test_empty_catalog_fail_closed() -> None:
    with pytest.raises(MissingSchedulerError):
        build_scheduler_contract_catalog({"surfaces": []})


def test_relation_version_must_bind_surface_version() -> None:
    payload = _unmaterialized()
    payload["relations"][0]["sourceVersion"] = "999"
    with pytest.raises(SchedulerAuthorityError) as excinfo:
        build_scheduler_contract_catalog(payload)
    assert excinfo.value.reason_code == "relation_version_mismatch"


def test_invariants_cover_lease_fence_and_queue_families() -> None:
    catalog = extract_scheduler_contracts()
    families = {invariant.family.value for invariant in catalog.invariants}
    assert "LeaseFenceBeforeEffect" in families
    assert "QueueAccountingConserved" in families
    assert "NoDuplicateOrLostWork" in families
    assert "CanonicalImplementationSelected" in families
    for invariant in catalog.invariants:
        assert invariant.bound >= 0
        assert invariant.version
        assert invariant.invariant_cid.startswith("b")
        for scheduler_id in invariant.scheduler_ids:
            catalog.surface(scheduler_id)


def test_validate_scheduler_sources_against_repository() -> None:
    catalog = extract_scheduler_contracts()
    # Validate a subset that lives under the monorepo external/ipfs_accelerate tree.
    validate_scheduler_sources(
        catalog,
        REPOSITORY_ROOT,
        required_scheduler_ids=(
            "deterministic-ownership-v1",
            "legacy-p2p-workflow-v1",
            "mcp-workflow-adapter-v1",
            "mcp-risk-v1",
            "supervisor-resource-v1",
            "supervisor-resource-usage-adapter-v1",
            "supervisor-provider-batch-v1",
            "validation-scheduler-v1",
            "proof-scheduler-v1",
        ),
    )


def test_missing_source_symbol_fail_closed() -> None:
    payload = _unmaterialized()
    payload["surfaces"] = [
        {
            "schedulerId": "missing-surface-v1",
            "displayName": "Missing surface",
            "role": SchedulerRole.DETERMINISTIC_OWNERSHIP.value,
            "implementationSymbol": "DoesNotExistSymbolXYZ",
            "sourcePath": (
                "ipfs_accelerate_py/p2p_tasks/deterministic_scheduler.py"
            ),
            "packageId": "ipfs_accelerate_py",
            "version": "1",
            "concurrencyBound": 1,
            "supportsLease": True,
            "supportsFence": True,
            "authority": {
                "kind": SchedulerAuthorityKind.CANONICAL.value,
                "canonicalSchedulerId": "missing-surface-v1",
                "decision": "fixture_missing_symbol",
                "adapterContractId": "",
                "version": "1",
                "sourcePath": (
                    "ipfs_accelerate_py/p2p_tasks/deterministic_scheduler.py"
                ),
            },
        }
    ]
    payload["relations"] = []
    payload["invariants"] = []
    catalog = build_scheduler_contract_catalog(payload)
    with pytest.raises(SchedulerSourceError) as excinfo:
        validate_scheduler_sources(catalog, REPOSITORY_ROOT)
    assert excinfo.value.reason_code == "scheduler_symbol_missing"


def test_swissknife_scheduler_source_resolves() -> None:
    catalog = extract_scheduler_contracts()
    validate_scheduler_sources(
        catalog,
        REPOSITORY_ROOT,
        required_scheduler_ids=("swissknife-mcp-scheduler-v1",),
    )


def test_display_name_never_joins_authority() -> None:
    payload = _unmaterialized()
    for surface in payload["surfaces"]:
        surface["displayName"] = "Shared Display Name"
    catalog = build_scheduler_contract_catalog(payload)
    # Still unique by id/CID; shared names are descriptive only.
    assert len({s.surface_cid for s in catalog.surfaces}) == len(catalog.surfaces)
    assert len({s.scheduler_id for s in catalog.surfaces}) == len(catalog.surfaces)
