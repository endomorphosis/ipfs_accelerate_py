"""IPS-032: incremental-versus-full planning and reuse explanations."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    EVIDENCE_SUBSET,
    REUSE_EVIDENCE_SUBSET,
    ParentSealContext,
    PlanMode,
    PlanningRequest,
    UnitPlanKind,
    UnitPlanningInput,
    create_incremental_plan,
    explain_reuse,
    plan_incremental_proof,
)

_PARENT = ParentSealContext(
    seal_cid="sha256:" + ("aa" * 32),
    repository_state_cid="sha256:" + ("bb" * 32),
    source_root_cid="sha256:" + ("cc" * 32),
)
_OLD = "sha256:" + ("bb" * 32)
_NEW = "sha256:" + ("dd" * 32)


def _unit(unit_id: str, **overrides: object) -> UnitPlanningInput:
    payload = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    return UnitPlanningInput(**payload)  # type: ignore[arg-type]


def test_evidence_subsets() -> None:
    assert EVIDENCE_SUBSET == "ips/incremental-plan@1"
    assert REUSE_EVIDENCE_SUBSET == "ips/reuse-explanation@1"


def test_first_state_chooses_full_fallback() -> None:
    plan = create_incremental_plan(
        None,
        _OLD,
        _NEW,
        units=(
            _unit("unit/a"),
            _unit("unit/b", preserved=False, invalidated=True, admitted=False),
        ),
    )
    assert plan.mode is PlanMode.FULL
    assert "first_state" in plan.fallback_reasons
    assert "missing_parent" in plan.fallback_reasons
    assert plan.reusable_unit_ids == ()
    assert "unit/a" in plan.invalidated_unit_ids
    assert plan.resources.prove_units == 2


def test_trust_or_schema_change_forces_full() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        {"schema_changed": True},
        units=(_unit("unit/a"),),
    )
    assert plan.mode is PlanMode.FULL
    assert "schema_change" in plan.fallback_reasons
    assert plan.reusable_unit_ids == ()


def test_candidate_presence_never_authorizes_reuse() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit(
                "unit/hint-only",
                admitted=False,
                candidate_present=True,
                cache_key_complete=True,
            ),
            _unit("unit/admitted"),
        ),
    )
    assert plan.mode is PlanMode.INCREMENTAL
    assert "unit/admitted" in plan.reusable_unit_ids
    assert "unit/hint-only" not in plan.reusable_unit_ids
    explanation = explain_reuse(plan, "unit/hint-only")
    assert explanation.reused is False
    assert explanation.candidate_present is True
    assert explanation.admitted is False
    assert "candidate" in explanation.reason


def test_unrelated_preserved_unit_reuses_only_when_admitted() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(
            _unit("unit/unrelated"),
            _unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
                cache_key_complete=True,
            ),
            _unit("unit/new", preserved=False, added=True, admitted=False),
            _unit("unit/gone", preserved=False, removed=True, admitted=False),
        ),
        changed_root_cids=("sha256:" + ("ee" * 32),),
    )
    assert plan.mode is PlanMode.INCREMENTAL
    assert plan.reusable_unit_ids == ("unit/unrelated",)
    assert "unit/changed" in plan.invalidated_unit_ids
    assert plan.added_unit_ids == ("unit/new",)
    assert plan.removed_unit_ids == ("unit/gone",)
    reused = explain_reuse(plan, "unit/unrelated")
    assert reused.reused is True
    assert reused.cache_key_complete is True
    assert reused.admitted is True


def test_incomplete_cache_key_cannot_be_reused() -> None:
    plan = create_incremental_plan(
        _PARENT,
        _OLD,
        _NEW,
        units=(_unit("unit/partial", cache_key_complete=False, admitted=True),),
    )
    assert plan.reusable_unit_ids == ()
    assert plan.units[0].kind is UnitPlanKind.REPROVE
    assert plan.units[0].reason == "incomplete_cache_key"


def test_plan_is_deterministic() -> None:
    units = (
        _unit("unit/b"),
        _unit("unit/a", preserved=False, invalidated=True, admitted=False),
    )
    first = create_incremental_plan(_PARENT, _OLD, _NEW, units=units)
    second = create_incremental_plan(_PARENT, _OLD, _NEW, units=tuple(reversed(units)))
    assert first.plan_cid() == second.plan_cid()
    assert first.to_canonical() == second.to_canonical()
    assert first.units[0].unit_id == "unit/a"


def test_required_plan_fields_are_present() -> None:
    plan = plan_incremental_proof(
        PlanningRequest(
            parent=_PARENT,
            old_repository_state_cid=_OLD,
            new_repository_state_cid=_NEW,
            units=(_unit("unit/a"),),
        )
    )
    payload = plan.to_canonical()
    for key in (
        "schema",
        "mode",
        "parent_seal_cid",
        "reusable_unit_ids",
        "invalidated_unit_ids",
        "added_unit_ids",
        "removed_unit_ids",
        "changed_root_cids",
        "fallback_reasons",
        "aggregation",
        "resources",
        "acceptance",
    ):
        assert key in payload
    assert payload["acceptance"]["forbid_candidate_only_reuse"] is True
    assert plan.resources.savings_cpu_ms >= 0
    assert plan.aggregation.recursive_verification is False
