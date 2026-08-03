"""Fail-closed coverage for support-behavior placement (RPR-038)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
)
from ipfs_accelerate_py.agent_supervisor.planning.implementation_site_admissibility import (
    PlacementDecision,
    PlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.support_behavior_placement import (
    EXISTING_ADMISSIBLE_IMPLEMENTATION,
    SUPPORT_BEHAVIOR_PLACEMENT_INTERFACE,
    ExistingImplementationFact,
    PlacementAnchor,
    PlacementAnchorKind,
    SupportBehaviorPlacement,
    SupportBehaviorPlacementAuthorityError,
    SupportBehaviorPlacementError,
    SupportPlacementAction,
    SupportPlacementCandidate,
    SupportPlacementDecision,
    SupportPlacementDisposition,
    support_placement_candidate_set_identity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-038",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-038",
        index_id="index:rpr-038",
        model_id="model:rpr-038",
        config_id="config:rpr-038",
        translator_id="translator:rpr-038",
        toolchain_id="toolchain:rpr-038",
        policy_id="policy:rpr-038",
    )


def _behavior(
    roots: PropagationAuthorityRoots, **extra: object
) -> RequiredBehaviorContract:
    values: dict[str, object] = {
        "roots": roots,
        "behavior_id": "behavior:SupportContext",
        "kind": BehaviorKind.CLASS,
        "subject_symbol_id": "symbol:SupportContext",
        "evidence_precedence": BehaviorEvidencePrecedence.REVIEWED_IDL,
        "field_refs": ("field:trace_id",),
        "constructor_refs": ("ctor:SupportContext",),
        "method_refs": ("method:with_span",),
        "invariant_refs": ("inv:non_empty_trace",),
        "effect_refs": ("effect:none",),
        "capability_refs": ("cap:context.read",),
        "proof_refs": ("proof:behavior:SupportContext",),
    }
    values.update(extra)
    return RequiredBehaviorContract(**values)


def _eligible_candidate(
    roots: PropagationAuthorityRoots,
    *,
    name: str = "owner_module",
    is_reuse: bool = False,
    **extra: object,
) -> SupportPlacementCandidate:
    anchor_kind = (
        PlacementAnchorKind.EXISTING_ADMISSIBLE_IMPLEMENTATION
        if is_reuse
        else PlacementAnchorKind.ARCHITECTURE_OWNERSHIP
    )
    path = f"pkg/support/{name}.py"
    values: dict[str, object] = {
        "roots": roots,
        "candidate_id": f"candidate:{name}",
        "behavior_id": "behavior:SupportContext",
        "subject_symbol_id": "symbol:SupportContext",
        "anchor_kind": anchor_kind,
        "anchor_id": f"anchor:{name}",
        "target_path": path,
        "placement_paths": (path,),
        "owner_id": f"owner:{name}",
        "module_owner_id": f"module:pkg.support.{name}",
        "language_runtime": "python",
        "proof_receipt_ids": (f"proof:placement:{name}",),
        "evidence_refs": (f"evidence:arch:{name}",),
        "nomination_source": "architecture",
        "ownership_exact": True,
        "owner_unambiguous": True,
        "visibility_route_satisfiable": True,
        "dependency_direction_legal": True,
        "dependency_acyclic": True,
        "registration_export_di_wiring_satisfiable": True,
        "capability_supported": True,
        "effect_supported": True,
        "resource_supported": True,
        "memory_supported": True,
        "mutation_authority_exact": True,
        "behavior_contract_fit": True,
        "behavior_proved": True,
        "lifecycle_supported": True,
        "site_placement_admitted": True,
        "is_reuse": is_reuse,
    }
    values.update(extra)
    return SupportPlacementCandidate(**values)


# ---------------------------------------------------------------------------
# Happy path / selection authority
# ---------------------------------------------------------------------------


def test_admits_unique_proved_site_and_selection_defines_paths(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(roots, name="context")
    decision = SupportBehaviorPlacement().decide(behavior, (candidate,))

    assert decision.disposition is SupportPlacementDisposition.ADMITTED
    assert decision.admitted is True
    assert decision.selected_candidate_id == candidate.candidate_id
    assert decision.target_path == "pkg/support/context.py"
    assert decision.placement_paths == ("pkg/support/context.py",)
    assert decision.action is SupportPlacementAction.PLACE_NEW
    assert decision.proof_receipt_ids == ("proof:placement:context",)
    assert decision.schema.endswith("support-placement-decision@1")
    assert decision.producer_id
    # Paths exist only because selection admitted them — not from nomination.
    assert decision.content_id


def test_prefers_existing_admissible_implementation_over_new_site(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    reuse = _eligible_candidate(
        roots,
        name="existing",
        is_reuse=True,
        candidate_id="candidate:reuse",
        score_vector=(1000, 100, 50, 5),
    )
    fresh = _eligible_candidate(
        roots,
        name="fresh",
        is_reuse=False,
        candidate_id="candidate:fresh",
        score_vector=(0, 90, 50, 5),
    )
    decision = SupportBehaviorPlacement().place(behavior, (fresh, reuse))

    assert decision.disposition is SupportPlacementDisposition.ADMITTED
    assert decision.selected_candidate_id == "candidate:reuse"
    assert decision.action is SupportPlacementAction.REUSE_EXISTING
    assert decision.placement_paths == ("pkg/support/existing.py",)


def test_enumerate_from_anchors_and_existing_implementations(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    anchors = (
        PlacementAnchor(
            roots,
            "anchor:decl",
            PlacementAnchorKind.DECLARATION,
            "pkg/support/context.py",
            "owner:pkg.support",
            "module:pkg.support.context",
            declaration_id="decl:SupportContext",
            registration_route_id="reg:support",
            evidence_refs=("evidence:decl",),
        ),
        PlacementAnchor(
            roots,
            "anchor:iface",
            PlacementAnchorKind.INTERFACE,
            "pkg/support/api.py",
            "owner:pkg.support",
            "module:pkg.support.api",
            interface_id="iface:SupportContext",
            export_route_id="export:api",
            evidence_refs=("evidence:iface",),
        ),
        PlacementAnchor(
            roots,
            "anchor:vector-noise",
            PlacementAnchorKind.SCHEMA,
            "pkg/support/schema.py",
            "owner:pkg.support",
            "module:pkg.support.schema",
            nomination_source="vector",
        ),
    )
    existing = (
        ExistingImplementationFact(
            roots,
            "impl:SupportContext",
            "pkg/support/legacy.py",
            "owner:pkg.support",
            "module:pkg.support.legacy",
            "symbol:SupportContext",
            admissible=True,
            behavior_contract_fit=True,
            site_placement_admitted=True,
            proof_receipt_ids=("proof:reuse:legacy",),
            evidence_refs=("evidence:reuse",),
        ),
    )
    engine = SupportBehaviorPlacement()
    enumerated = engine.enumerate_candidates(
        behavior, anchors, existing_implementations=existing, default_proved=True
    )

    kinds = {item.anchor_kind for item in enumerated}
    assert PlacementAnchorKind.DECLARATION in kinds
    assert PlacementAnchorKind.INTERFACE in kinds
    assert EXISTING_ADMISSIBLE_IMPLEMENTATION in kinds
    # Vector nomination must not become a candidate.
    assert all(item.nomination_source != "vector" for item in enumerated)
    assert any(item.is_reuse for item in enumerated)

    # Promote enumerated reuse (already proved) through admission.
    reuse = next(item for item in enumerated if item.is_reuse)
    decision = engine.decide(behavior, (reuse,))
    assert decision.disposition is SupportPlacementDisposition.ADMITTED
    assert decision.action is SupportPlacementAction.REUSE_EXISTING


# ---------------------------------------------------------------------------
# Hard exclusions / abstention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"read_only": True}, "generated_vendor_read_only_target"),
        ({"generated": True}, "generated_vendor_read_only_target"),
        ({"vendor": True}, "generated_vendor_read_only_target"),
        ({"dependency_acyclic": False}, "dependency_cycle"),
        ({"dependency_direction_legal": False}, "dependency_cycle"),
        ({"cross_root_write": True}, "cross_root_write"),
        ({"owner_unambiguous": False}, "missing_owner"),
        ({"ownership_exact": False}, "missing_owner"),
        ({"visibility_route_satisfiable": False}, "visibility_route_unsatisfied"),
        (
            {"registration_export_di_wiring_satisfiable": False},
            "registration_export_di_unsatisfied",
        ),
        (
            {"capability_supported": False},
            "capability_effect_resource_memory_unsupported",
        ),
        ({"effect_supported": False}, "capability_effect_resource_memory_unsupported"),
        ({"memory_supported": False}, "capability_effect_resource_memory_unsupported"),
        ({"mutation_authority_exact": False}, "mutation_authority_not_exact"),
        ({"behavior_contract_fit": False}, "behavior_contract_mismatch"),
        ({"behavior_proved": False}, "unproved_behavior"),
        ({"lifecycle_supported": False}, "unsupported_lifecycle_native_semantics"),
        (
            {"native_semantics_unsupported": True},
            "unsupported_lifecycle_native_semantics",
        ),
        ({"language_runtime": "native-rust"}, "language_runtime_unsupported"),
        ({"nomination_source": "llm"}, "vector_kg_llm_nomination_forbidden"),
        ({"nomination_source": "knowledge_graph"}, "vector_kg_llm_nomination_forbidden"),
        ({"site_placement_admitted": False}, "site_not_admitted"),
        ({"proof_receipt_ids": ()}, "missing_required_proof_receipt"),
        (
            {"target_path": "vendor/pkg/x.py", "placement_paths": ("vendor/pkg/x.py",)},
            "generated_vendor_read_only_target",
        ),
    ),
)
def test_excludes_unsupportable_and_forbidden_sites(
    roots: PropagationAuthorityRoots,
    change: dict[str, object],
    reason: str,
) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(roots, name="site")
    candidate = replace(candidate, **change)
    decision = SupportBehaviorPlacement().assess(behavior, (candidate,))

    assert decision.disposition in {
        SupportPlacementDisposition.ABSTAINED,
        SupportPlacementDisposition.REVIEW_ONLY,
    }
    assert decision.placement_paths == ()
    assert decision.selected_candidate_id == ""
    assert reason in decision.reason_codes


def test_ties_and_insufficient_margin_are_ambiguous(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    first = _eligible_candidate(
        roots,
        name="alpha",
        candidate_id="candidate:alpha",
        score_vector=(10, 10, 10),
    )
    second = _eligible_candidate(
        roots,
        name="beta",
        candidate_id="candidate:beta",
        score_vector=(10, 10, 10),
    )
    decision = SupportBehaviorPlacement().decide(behavior, (first, second))
    assert decision.disposition is SupportPlacementDisposition.AMBIGUOUS
    assert "rank_tie" in decision.reason_codes
    assert decision.placement_paths == ()

    close = replace(second, score_vector=(10, 10, 9))
    decision = SupportBehaviorPlacement(minimum_margin=5).decide(
        behavior, (first, close)
    )
    assert decision.disposition is SupportPlacementDisposition.AMBIGUOUS
    assert "insufficient_rank_margin" in decision.reason_codes
    assert decision.margin == 1


def test_unique_winner_with_sufficient_margin_admits(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    winner = _eligible_candidate(
        roots,
        name="winner",
        candidate_id="candidate:winner",
        score_vector=(50, 20, 10),
    )
    runner = _eligible_candidate(
        roots,
        name="runner",
        candidate_id="candidate:runner",
        score_vector=(40, 20, 10),
    )
    decision = SupportBehaviorPlacement(minimum_margin=5).evaluate(
        behavior, (runner, winner)
    )
    assert decision.disposition is SupportPlacementDisposition.ADMITTED
    assert decision.selected_candidate_id == "candidate:winner"
    assert decision.margin == 10
    assert decision.placement_paths == ("pkg/support/winner.py",)


def test_behavior_id_and_root_mismatches_abstain(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    other_roots = replace(
        roots,
        candidate_tree_id="tree:other",
        candidate_overlay_id="overlay:other",
        graph_id="graph:other",
    )
    mismatched = _eligible_candidate(roots, name="site", behavior_id="behavior:Other")
    decision = SupportBehaviorPlacement().decide(behavior, (mismatched,))
    assert decision.disposition is SupportPlacementDisposition.ABSTAINED
    assert "behavior_id_mismatch" in decision.reason_codes

    # Cross-root candidate set is rejected at identity construction.
    cross = _eligible_candidate(other_roots, name="cross")
    decision = SupportBehaviorPlacement().decide(behavior, (cross,))
    assert decision.disposition is SupportPlacementDisposition.ABSTAINED
    assert "invalid_admission_input" in decision.reason_codes or (
        "authority_roots_mismatch" in decision.reason_codes
    )


def test_write_policy_paths_gate_cross_root_writes(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(roots, name="context")
    decision = SupportBehaviorPlacement().decide(
        behavior,
        (candidate,),
        write_policy_paths=("pkg/other/only.py",),
    )
    assert decision.disposition is SupportPlacementDisposition.ABSTAINED
    assert "cross_root_write" in decision.reason_codes
    assert decision.placement_paths == ()

    decision = SupportBehaviorPlacement().decide(
        behavior,
        (candidate,),
        write_policy_paths=("pkg/support/context.py",),
    )
    assert decision.disposition is SupportPlacementDisposition.ADMITTED
    assert decision.placement_paths == ("pkg/support/context.py",)


def test_implementation_hypothesis_is_review_only(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(
        roots,
        evidence_precedence=BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
        implementation_hypothesis=True,
        proof_refs=(),
    )
    candidate = _eligible_candidate(roots, name="site")
    decision = SupportBehaviorPlacement().decide(behavior, (candidate,))
    assert decision.disposition is SupportPlacementDisposition.REVIEW_ONLY
    assert "implementation_hypothesis_not_authoritative" in decision.reason_codes
    assert decision.placement_paths == ()


def test_enumerate_rejects_implementation_hypothesis_authority(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(
        roots,
        evidence_precedence=BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
        implementation_hypothesis=True,
        proof_refs=(),
    )
    with pytest.raises(SupportBehaviorPlacementAuthorityError):
        SupportBehaviorPlacement().enumerate_candidates(behavior, ())


# ---------------------------------------------------------------------------
# Implementation-site admissibility join
# ---------------------------------------------------------------------------


def test_site_decision_join_accepts_admitted_site(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(
        roots, name="context", site_placement_admitted=False
    )
    site = PlacementDecision(
        PlacementDisposition.ADMITTED,
        "candidate-set:site",
        selected_candidate_id="repair:context",
        target_path="pkg/support/context.py",
        proof_receipt_ids=("proof:site:context",),
    )
    decision = SupportBehaviorPlacement().decide(
        behavior,
        (candidate,),
        site_decisions={candidate.candidate_id: site},
    )
    assert decision.disposition is SupportPlacementDisposition.ADMITTED


def test_site_decision_join_rejects_non_admitted_site(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(
        roots, name="context", site_placement_admitted=True
    )
    site = PlacementDecision(
        PlacementDisposition.ABSTAINED,
        "candidate-set:site",
        reason_codes=("no_admissible_implementation_site",),
    )
    decision = SupportBehaviorPlacement().decide(
        behavior,
        (candidate,),
        site_decisions={candidate.candidate_id: site},
    )
    assert decision.disposition is SupportPlacementDisposition.ABSTAINED
    assert "site_not_admitted" in decision.reason_codes


# ---------------------------------------------------------------------------
# Contract hygiene
# ---------------------------------------------------------------------------


def test_candidate_set_identity_is_stable_and_rejects_duplicates(
    roots: PropagationAuthorityRoots,
) -> None:
    first = _eligible_candidate(roots, name="a", candidate_id="candidate:a")
    second = _eligible_candidate(roots, name="b", candidate_id="candidate:b")
    left = support_placement_candidate_set_identity((first, second))
    right = support_placement_candidate_set_identity((second, first))
    assert left == right

    with pytest.raises(SupportBehaviorPlacementError):
        support_placement_candidate_set_identity((first, first))


def test_admitted_decision_cannot_omit_paths_or_proofs(
    roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(SupportBehaviorPlacementError):
        SupportPlacementDecision(
            SupportPlacementDisposition.ADMITTED,
            roots,
            "behavior:x",
            "set:1",
            selected_candidate_id="candidate:x",
            action=SupportPlacementAction.PLACE_NEW,
            target_path="pkg/x.py",
            placement_paths=(),
            proof_receipt_ids=("proof:1",),
        )


def test_non_admitted_decision_cannot_carry_paths(
    roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(SupportBehaviorPlacementError):
        SupportPlacementDecision(
            SupportPlacementDisposition.ABSTAINED,
            roots,
            "behavior:x",
            "set:1",
            selected_candidate_id="candidate:x",
            target_path="pkg/x.py",
            placement_paths=("pkg/x.py",),
        )


def test_placement_paths_must_include_target(
    roots: PropagationAuthorityRoots,
) -> None:
    with pytest.raises(SupportBehaviorPlacementError):
        SupportPlacementCandidate(
            roots=roots,
            candidate_id="candidate:bad",
            behavior_id="behavior:SupportContext",
            subject_symbol_id="symbol:SupportContext",
            anchor_kind=PlacementAnchorKind.DECLARATION,
            anchor_id="anchor:bad",
            target_path="pkg/a.py",
            placement_paths=("pkg/b.py",),
            owner_id="owner:a",
            module_owner_id="module:a",
        )


def test_interface_constant_and_aliases() -> None:
    assert SUPPORT_BEHAVIOR_PLACEMENT_INTERFACE == "SupportBehaviorPlacement@1"
    assert (
        SupportBehaviorPlacement.decide
        is SupportBehaviorPlacement.assess
        is SupportBehaviorPlacement.evaluate
        is SupportBehaviorPlacement.place
        is SupportBehaviorPlacement.admit
    )


def test_empty_candidate_set_abstains(roots: PropagationAuthorityRoots) -> None:
    behavior = _behavior(roots)
    decision = SupportBehaviorPlacement().decide(behavior, ())
    assert decision.disposition is SupportPlacementDisposition.ABSTAINED
    assert "no_placement_candidates" in decision.reason_codes


def test_dependency_graph_id_must_match_roots(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(roots, name="context")
    decision = SupportBehaviorPlacement().decide(
        behavior,
        (candidate,),
        dependency_graph_id="graph:stale",
    )
    assert decision.disposition is SupportPlacementDisposition.ABSTAINED
    assert "dependency_graph_root_mismatch" in decision.reason_codes


def test_decision_to_dict_is_deterministic(roots: PropagationAuthorityRoots) -> None:
    behavior = _behavior(roots)
    candidate = _eligible_candidate(roots, name="context")
    decision = SupportBehaviorPlacement().decide(behavior, (candidate,))
    payload = decision.to_dict()
    assert payload["disposition"] == "admitted"
    assert payload["placement_paths"] == ["pkg/support/context.py"]
    assert payload["content_id"] == decision.content_id
    assert payload["action"] == "place_new"
