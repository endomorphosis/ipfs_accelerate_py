"""PDR-054: independent multi-method repair candidate portfolio selection.

Covers:
* property-based / fuzz / concolic when supported (fixed seeds/budgets)
* mutation against independent oracle
* differential / metamorphic / sanitizer / static / model / proof / security
* flaky and unavailable lane recording
* reject self-authored tests and candidate-as-oracle
* require all hard obligations (no weighted averaging of hard failures)
* rank only hard-admissible by minimal blast radius then resource cost
* preserve correct abstention
* prove selection / replay identity
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.repair_candidate_portfolio import (
    DEFAULT_HARD_LANES,
    DEFAULT_SEED,
    OPTIONAL_CAPABILITY_LANES,
    PORTFOLIO_LANE_ORDER,
    PRODUCER_ID,
    REPAIR_CANDIDATE_DECISION_INTERFACE,
    REPAIR_CANDIDATE_PORTFOLIO_INTERFACE,
    CandidateEvaluation,
    HardObligation,
    IndependentOracle,
    LaneObservation,
    LaneOutcome,
    PortfolioAuthorityError,
    PortfolioCandidate,
    PortfolioDisposition,
    PortfolioLane,
    PortfolioReason,
    PortfolioRequest,
    PortfolioSeedBudget,
    RepairCandidateDecision,
    RepairCandidatePortfolio,
    RepairCandidatePortfolioError,
    all_lanes_supported,
    create_repair_candidate_portfolio,
    default_hard_obligations,
    derive_selection_replay_identities,
    evaluate_lane,
    evaluate_repair_candidate_portfolio,
    passing_observations,
    prove_selection_replay_identity,
    rank_hard_admissible,
    select_repair_candidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_oracle(**overrides: object) -> IndependentOracle:
    base: dict[str, object] = {
        "oracle_id": "oracle:held-out-v1",
        "source": "held_out_acceptance",
        "producer_id": "benchmark-oracle@1",
        "expectation_ids": ("expect:api", "expect:security"),
        "test_ids": ("test_hidden_api", "test_hidden_security"),
        "root_bindings": (("tree_id", "tree:fixture"), ("policy_id", "policy:fixture")),
    }
    base.update(overrides)
    return IndependentOracle(**base)  # type: ignore[arg-type]


def make_budget(**overrides: object) -> PortfolioSeedBudget:
    base: dict[str, object] = {
        "seed": DEFAULT_SEED,
        "max_property_cases": 64,
        "max_fuzz_inputs": 128,
        "max_concolic_paths": 32,
        "max_mutation_ops": 16,
        "max_wall_ms": 5_000,
        "max_resource_cost": 10_000,
    }
    base.update(overrides)
    return PortfolioSeedBudget(**base)  # type: ignore[arg-type]


def make_candidate(
    candidate_id: str = "cand:alpha",
    *,
    blast_radius: int = 2,
    resource_cost: int = 100,
    oracle_id: str = "oracle:held-out-v1",
    hard_status: str = "pass",
    optional_status: str = "pass",
    **overrides: object,
) -> PortfolioCandidate:
    observations = passing_observations(
        oracle_id=oracle_id,
        hard_status=hard_status,
        optional_status=optional_status,
    )
    base: dict[str, object] = {
        "candidate_id": candidate_id,
        "patch_cid": f"patch:{candidate_id}",
        "overlay_cid": f"overlay:{candidate_id}",
        "changed_paths": ("pkg/module.py",),
        "blast_radius": blast_radius,
        "resource_cost": resource_cost,
        "obligation_refs": ("ob:1",),
        "lane_support": all_lanes_supported(),
        "lane_observations": observations,
    }
    base.update(overrides)
    return PortfolioCandidate(**base)  # type: ignore[arg-type]


def make_request(
    candidates: list[PortfolioCandidate] | None = None,
    **overrides: object,
) -> PortfolioRequest:
    if candidates is None:
        resolved: tuple[PortfolioCandidate, ...] = (make_candidate(),)
    else:
        resolved = tuple(candidates)
    base: dict[str, object] = {
        "candidates": resolved,
        "oracle": make_oracle(),
        "hard_obligations": default_hard_obligations(),
        "budget": make_budget(),
        "capability_support": all_lanes_supported(),
        "repository_tree_id": "tree:fixture",
        "forest_id": "forest:fixture",
        "policy_id": "policy:fixture",
        "request_id": "req:pdr-054",
    }
    base.update(overrides)
    return PortfolioRequest(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Interface / contract surface
# ---------------------------------------------------------------------------


def test_interfaces_and_producer_constants() -> None:
    assert REPAIR_CANDIDATE_PORTFOLIO_INTERFACE == "RepairCandidatePortfolio@1"
    assert REPAIR_CANDIDATE_DECISION_INTERFACE == "RepairCandidateDecision@1"
    portfolio = create_repair_candidate_portfolio()
    assert portfolio.INTERFACE == REPAIR_CANDIDATE_PORTFOLIO_INTERFACE
    assert portfolio.producer_id == PRODUCER_ID


def test_lane_order_covers_acceptance_methods() -> None:
    values = {lane.value for lane in PORTFOLIO_LANE_ORDER}
    for required in (
        "property_based",
        "fuzz",
        "concolic",
        "mutation",
        "differential",
        "metamorphic",
        "sanitizer",
        "static",
        "model",
        "proof",
        "security",
    ):
        assert required in values
    assert PortfolioLane.MUTATION in DEFAULT_HARD_LANES
    assert PortfolioLane.PROOF in DEFAULT_HARD_LANES
    assert PortfolioLane.PROPERTY_BASED.value in OPTIONAL_CAPABILITY_LANES


# ---------------------------------------------------------------------------
# Happy path selection
# ---------------------------------------------------------------------------


def test_selects_hard_admissible_minimal_blast_then_cost() -> None:
    large = make_candidate("cand:large", blast_radius=10, resource_cost=50)
    cheap_but_wide = make_candidate("cand:wide", blast_radius=5, resource_cost=10)
    minimal = make_candidate("cand:minimal", blast_radius=1, resource_cost=200)
    same_blast_cheaper = make_candidate("cand:cheap", blast_radius=1, resource_cost=50)

    decision = select_repair_candidate(
        make_request(candidates=[large, cheap_but_wide, minimal, same_blast_cheaper])
    )

    assert decision.disposition is PortfolioDisposition.SELECTED
    assert decision.selected_candidate_id == "cand:cheap"
    assert decision.ranked_admissible == (
        "cand:cheap",
        "cand:minimal",
        "cand:wide",
        "cand:large",
    )
    assert PortfolioReason.ALL_HARD_OBLIGATIONS_MET.value in decision.reason_codes
    assert decision.proposal_only is True
    assert decision.write_authority is False
    assert decision.weighted_authority_used is False


def test_module_aliases_and_factory() -> None:
    request = make_request()
    a = select_repair_candidate(request)
    b = evaluate_repair_candidate_portfolio(request)
    c = create_repair_candidate_portfolio().select(request)
    assert a.selection_identity == b.selection_identity == c.selection_identity
    assert a.selected_candidate_id == "cand:alpha"


def test_runs_all_lanes_under_fixed_seed() -> None:
    decision = select_repair_candidate(make_request())
    assert len(decision.evaluations) == 1
    evaluation = decision.evaluations[0]
    assert evaluation.hard_admissible is True
    assert len(evaluation.lane_results) == len(PORTFOLIO_LANE_ORDER)
    for result in evaluation.lane_results:
        assert result.seed == DEFAULT_SEED
        assert result.outcome is LaneOutcome.PASS
    mutation = next(
        item
        for item in evaluation.lane_results
        if item.lane is PortfolioLane.MUTATION
    )
    assert mutation.oracle_id == "oracle:held-out-v1"


# ---------------------------------------------------------------------------
# Optional capability lanes (property / fuzz / concolic)
# ---------------------------------------------------------------------------


def test_records_unavailable_optional_lanes_without_blocking() -> None:
    support = all_lanes_supported()
    support["property_based"] = False
    support["fuzz"] = False
    support["concolic"] = False
    decision = select_repair_candidate(
        make_request(capability_support=support)
    )
    assert decision.disposition is PortfolioDisposition.SELECTED
    evaluation = decision.evaluations[0]
    unavailable = set(evaluation.unavailable_lanes)
    assert "property_based" in unavailable
    assert "fuzz" in unavailable
    assert "concolic" in unavailable
    assert evaluation.hard_admissible is True
    # Portfolio-level recording uses candidate:lane keys.
    assert any("property_based" in item for item in decision.unavailable_lanes)


def test_property_fuzz_concolic_pass_when_supported() -> None:
    candidate = make_candidate(optional_status="pass")
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    evaluation = decision.evaluations[0]
    for lane in (
        PortfolioLane.PROPERTY_BASED,
        PortfolioLane.FUZZ,
        PortfolioLane.CONCOLIC,
    ):
        result = next(item for item in evaluation.lane_results if item.lane is lane)
        assert result.outcome is LaneOutcome.PASS
        assert result.hard is False


# ---------------------------------------------------------------------------
# Hard obligations / no weighted averaging
# ---------------------------------------------------------------------------


def test_hard_failure_cannot_be_averaged_away_by_soft_score() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    # Proof fails hard, but carries a high soft_score that must not rescue it.
    observations["proof"] = {
        "supported": True,
        "status": "fail",
        "cases_run": 1,
        "budget_used": 1,
        "soft_score": 999_999,
        "reason_codes": ("proof_counterexample",),
    }
    candidate = make_candidate(
        "cand:soft-scored",
        blast_radius=0,
        resource_cost=0,
        lane_observations=observations,
    )
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert decision.selected_candidate_id == ""
    evaluation = decision.evaluations[0]
    assert evaluation.hard_admissible is False
    assert "proof" in evaluation.hard_failures
    proof = next(
        item for item in evaluation.lane_results if item.lane is PortfolioLane.PROOF
    )
    assert proof.outcome is LaneOutcome.FAIL
    assert PortfolioReason.HARD_FAILURE_NOT_AVERAGED.value in proof.reason_codes


def test_requires_all_default_hard_obligations() -> None:
    for lane in sorted(DEFAULT_HARD_LANES, key=lambda item: item.value):
        observations = passing_observations(oracle_id="oracle:held-out-v1")
        observations[lane.value] = {
            "supported": True,
            "status": "fail",
            "cases_run": 1,
            "budget_used": 1,
            "oracle_id": "oracle:held-out-v1"
            if lane is PortfolioLane.MUTATION
            else "",
        }
        candidate = make_candidate(
            f"cand:fail-{lane.value}",
            lane_observations=observations,
        )
        decision = select_repair_candidate(make_request(candidates=[candidate]))
        assert decision.disposition is PortfolioDisposition.ABSTAIN
        assert not decision.ranked_admissible
        assert lane.value in decision.evaluations[0].hard_failures


def test_custom_hard_obligations_subset() -> None:
    obligations = (
        HardObligation(obligation_id="hard:proof-only", lane=PortfolioLane.PROOF),
        HardObligation(obligation_id="hard:security-only", lane=PortfolioLane.SECURITY),
    )
    # Mutation fails but is not required under custom obligations.
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["mutation"] = {
        "supported": True,
        "status": "fail",
        "cases_run": 1,
        "budget_used": 1,
        "oracle_id": "oracle:held-out-v1",
    }
    candidate = make_candidate(lane_observations=observations)
    decision = select_repair_candidate(
        make_request(candidates=[candidate], hard_obligations=obligations)
    )
    assert decision.disposition is PortfolioDisposition.SELECTED
    assert "hard:proof-only" in decision.hard_obligation_ids


# ---------------------------------------------------------------------------
# Self-authored tests / candidate-as-oracle
# ---------------------------------------------------------------------------


def test_rejects_self_authored_tests_overlapping_oracle() -> None:
    candidate = make_candidate(
        "cand:self-test",
        authored_test_ids=("test_hidden_api", "test_local_only"),
    )
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.disposition is PortfolioDisposition.REJECT
    assert PortfolioReason.SELF_AUTHORED_TEST.value in decision.reason_codes
    assert decision.selected_candidate_id == ""


def test_rejects_lane_using_self_authored_tests() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["differential"] = {
        "supported": True,
        "status": "pass",
        "cases_run": 4,
        "budget_used": 4,
        "uses_self_authored_tests": True,
    }
    candidate = make_candidate(
        "cand:self-lane",
        lane_observations=observations,
    )
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.selected_candidate_id == ""
    evaluation = decision.evaluations[0]
    assert evaluation.hard_admissible is False
    assert PortfolioReason.SELF_AUTHORED_TEST.value in evaluation.rejection_reasons


def test_rejects_candidate_as_oracle_claim() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["mutation"] = {
        "supported": True,
        "status": "pass",
        "cases_run": 4,
        "budget_used": 4,
        "oracle_id": "oracle:held-out-v1",
        "candidate_claims_oracle": True,
    }
    candidate = make_candidate(
        "cand:self-oracle",
        lane_observations=observations,
    )
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.selected_candidate_id == ""
    assert PortfolioReason.CANDIDATE_AS_ORACLE.value in decision.evaluations[0].rejection_reasons


def test_rejects_claimed_oracle_ids_matching_portfolio_oracle() -> None:
    candidate = make_candidate(
        "cand:claim-oracle",
        claimed_oracle_ids=("oracle:held-out-v1",),
    )
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.selected_candidate_id == ""
    assert PortfolioReason.CANDIDATE_AS_ORACLE.value in decision.evaluations[0].rejection_reasons


def test_independent_oracle_rejects_candidate_source() -> None:
    with pytest.raises(RepairCandidatePortfolioError) as exc:
        IndependentOracle(
            oracle_id="oracle:bad",
            source="candidate_authored",
            producer_id="benchmark-oracle@1",
        )
    assert exc.value.reason_code == PortfolioReason.ORACLE_NOT_INDEPENDENT.value


def test_mutation_requires_matching_independent_oracle() -> None:
    observations = passing_observations(oracle_id="oracle:wrong")
    candidate = make_candidate(
        "cand:wrong-oracle",
        lane_observations=observations,
    )
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.selected_candidate_id == ""
    assert (
        PortfolioReason.ORACLE_NOT_INDEPENDENT.value
        in decision.evaluations[0].rejection_reasons
    )


def test_mutation_missing_oracle_id_fails() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["mutation"] = {
        "supported": True,
        "status": "pass",
        "cases_run": 4,
        "budget_used": 4,
        "oracle_id": "",
    }
    candidate = make_candidate(lane_observations=observations)
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert PortfolioReason.MISSING_ORACLE.value in decision.evaluations[0].rejection_reasons


# ---------------------------------------------------------------------------
# Flaky / unavailable hard lanes
# ---------------------------------------------------------------------------


def test_records_flaky_hard_lane_and_rejects_admission() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["security"] = {
        "supported": True,
        "status": "flaky",
        "cases_run": 3,
        "budget_used": 3,
    }
    candidate = make_candidate(lane_observations=observations)
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    evaluation = decision.evaluations[0]
    assert "security" in evaluation.flaky_lanes
    assert "security" in evaluation.hard_failures
    assert any("security" in item for item in decision.flaky_lanes)
    assert PortfolioReason.HARD_OBLIGATION_FLAKY.value in evaluation.rejection_reasons


def test_hard_unavailable_lane_blocks_admission() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["static"] = {
        "supported": False,
        "status": "unavailable",
        "cases_run": 0,
        "budget_used": 0,
    }
    candidate = make_candidate(lane_observations=observations)
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    evaluation = decision.evaluations[0]
    assert "static" in evaluation.unavailable_lanes
    assert "static" in evaluation.hard_failures


def test_capability_unavailable_hard_lane_recorded() -> None:
    support = all_lanes_supported()
    support["proof"] = False
    decision = select_repair_candidate(make_request(capability_support=support))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert any("proof" in item for item in decision.unavailable_lanes)


# ---------------------------------------------------------------------------
# Ranking boundaries
# ---------------------------------------------------------------------------


def test_ranks_only_hard_admissible_candidates() -> None:
    good_small = make_candidate("cand:good-small", blast_radius=3, resource_cost=30)
    good_large = make_candidate("cand:good-large", blast_radius=9, resource_cost=10)
    bad = make_candidate("cand:bad", blast_radius=0, resource_cost=0, hard_status="fail")
    decision = select_repair_candidate(
        make_request(candidates=[bad, good_large, good_small])
    )
    assert decision.ranked_admissible == ("cand:good-small", "cand:good-large")
    assert decision.selected_candidate_id == "cand:good-small"
    assert "cand:bad" not in decision.ranked_admissible
    # Bad candidate remains in evaluations for audit.
    assert {item.candidate_id for item in decision.evaluations} == {
        "cand:good-small",
        "cand:good-large",
        "cand:bad",
    }


def test_rank_hard_admissible_helper_stable_by_id() -> None:
    a = CandidateEvaluation(
        candidate_id="cand:b",
        hard_admissible=True,
        blast_radius=1,
        resource_cost=1,
    )
    b = CandidateEvaluation(
        candidate_id="cand:a",
        hard_admissible=True,
        blast_radius=1,
        resource_cost=1,
    )
    c = CandidateEvaluation(
        candidate_id="cand:z",
        hard_admissible=False,
        blast_radius=0,
        resource_cost=0,
    )
    ranked = rank_hard_admissible((a, b, c))
    assert [item.candidate_id for item in ranked] == ["cand:a", "cand:b"]


def test_resource_cost_over_budget_rejected() -> None:
    candidate = make_candidate("cand:expensive", resource_cost=50_000)
    decision = select_repair_candidate(
        make_request(candidates=[candidate], budget=make_budget(max_resource_cost=100))
    )
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert (
        PortfolioReason.BUDGET_EXCEEDED.value
        in decision.evaluations[0].rejection_reasons
    )


# ---------------------------------------------------------------------------
# Correct abstention
# ---------------------------------------------------------------------------


def test_preserves_correct_abstention_when_no_admissible() -> None:
    a = make_candidate("cand:a", hard_status="fail")
    b = make_candidate("cand:b", hard_status="fail")
    decision = select_repair_candidate(make_request(candidates=[a, b]))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert decision.selected_candidate_id == ""
    assert decision.ranked_admissible == ()
    assert PortfolioReason.NO_HARD_ADMISSIBLE.value in decision.reason_codes
    assert PortfolioReason.CORRECT_ABSTENTION.value in decision.reason_codes


def test_empty_portfolio_abstains() -> None:
    decision = select_repair_candidate(make_request(candidates=[]))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert PortfolioReason.NO_CANDIDATES.value in decision.reason_codes
    assert PortfolioReason.CORRECT_ABSTENTION.value in decision.reason_codes


def test_does_not_select_sole_inadmissible_despite_only_option() -> None:
    only = make_candidate("cand:only", blast_radius=0, resource_cost=0, hard_status="fail")
    decision = select_repair_candidate(make_request(candidates=[only]))
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert decision.selected_candidate_id == ""


# ---------------------------------------------------------------------------
# Selection / replay identity
# ---------------------------------------------------------------------------


def test_selection_and_replay_identity_stable_under_identical_inputs() -> None:
    request = make_request(
        candidates=[
            make_candidate("cand:a", blast_radius=2, resource_cost=20),
            make_candidate("cand:b", blast_radius=1, resource_cost=50),
        ]
    )
    first = select_repair_candidate(request)
    second = select_repair_candidate(request)
    assert first.selection_identity
    assert first.replay_identity
    assert first.selection_identity == second.selection_identity
    assert first.replay_identity == second.replay_identity
    assert prove_selection_replay_identity(first, second) is True
    # Derive functions agree with sealed fields.
    sel, rep = derive_selection_replay_identities(first)
    assert sel == first.selection_identity
    assert rep == first.replay_identity


def test_replay_identity_changes_when_ranking_inputs_change() -> None:
    base = [
        make_candidate("cand:a", blast_radius=2, resource_cost=20),
        make_candidate("cand:b", blast_radius=1, resource_cost=50),
    ]
    first = select_repair_candidate(make_request(candidates=base))
    # Swap costs so ranking changes.
    altered = [
        make_candidate("cand:a", blast_radius=2, resource_cost=20),
        make_candidate("cand:b", blast_radius=1, resource_cost=5),
    ]
    second = select_repair_candidate(make_request(candidates=altered))
    assert first.selected_candidate_id == "cand:b"
    assert second.selected_candidate_id == "cand:b"
    # Same selection but different cost evidence → different identities.
    # Actually both select b; change blast so selection changes.
    altered2 = [
        make_candidate("cand:a", blast_radius=0, resource_cost=20),
        make_candidate("cand:b", blast_radius=1, resource_cost=50),
    ]
    third = select_repair_candidate(make_request(candidates=altered2))
    assert third.selected_candidate_id == "cand:a"
    assert first.selection_identity != third.selection_identity
    assert prove_selection_replay_identity(first, third) is False


def test_forged_selection_identity_fails_proof() -> None:
    decision = select_repair_candidate(make_request())
    forged = RepairCandidateDecision(
        disposition=decision.disposition,
        reason_codes=decision.reason_codes,
        evaluations=decision.evaluations,
        ranked_admissible=decision.ranked_admissible,
        selected_candidate_id=decision.selected_candidate_id,
        selection_identity="forged-selection",
        replay_identity=decision.replay_identity,
        flaky_lanes=decision.flaky_lanes,
        unavailable_lanes=decision.unavailable_lanes,
        hard_obligation_ids=decision.hard_obligation_ids,
        seed=decision.seed,
        budget=decision.budget,
        oracle_id=decision.oracle_id,
    )
    # __post_init__ may overwrite empty identities but not non-empty forged ones.
    assert forged.selection_identity == "forged-selection"
    assert prove_selection_replay_identity(decision, forged) is False


def test_decision_round_trip_dict() -> None:
    decision = select_repair_candidate(make_request())
    restored = RepairCandidateDecision.from_dict(decision.to_dict())
    assert restored.disposition == decision.disposition
    assert restored.selected_candidate_id == decision.selected_candidate_id
    assert restored.selection_identity == decision.selection_identity
    assert restored.replay_identity == decision.replay_identity
    assert prove_selection_replay_identity(decision, restored) is True


# ---------------------------------------------------------------------------
# Authority / proposal-only boundaries
# ---------------------------------------------------------------------------


def test_candidate_cannot_claim_write_authority() -> None:
    with pytest.raises(PortfolioAuthorityError):
        PortfolioCandidate(
            candidate_id="cand:x",
            patch_cid="patch:x",
            write_authority=True,
        )


def test_decision_cannot_claim_completion_authority() -> None:
    decision = select_repair_candidate(make_request())
    with pytest.raises(PortfolioAuthorityError):
        RepairCandidateDecision(
            disposition=decision.disposition,
            reason_codes=decision.reason_codes,
            evaluations=decision.evaluations,
            ranked_admissible=decision.ranked_admissible,
            selected_candidate_id=decision.selected_candidate_id,
            grants_completion_authority=True,
            seed=decision.seed,
            budget=decision.budget,
            oracle_id=decision.oracle_id,
        )


def test_duplicate_candidate_ids_rejected() -> None:
    with pytest.raises(RepairCandidatePortfolioError) as exc:
        make_request(
            candidates=[
                make_candidate("cand:dup"),
                make_candidate("cand:dup", blast_radius=9),
            ]
        )
    assert exc.value.reason_code == PortfolioReason.DUPLICATE_CANDIDATE.value


# ---------------------------------------------------------------------------
# Budget / seed binding on lanes
# ---------------------------------------------------------------------------


def test_lane_budget_exceeded_fails_hard_lane() -> None:
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["mutation"] = {
        "supported": True,
        "status": "pass",
        "cases_run": 10_000,
        "budget_used": 10_000,
        "oracle_id": "oracle:held-out-v1",
    }
    candidate = make_candidate(lane_observations=observations)
    decision = select_repair_candidate(
        make_request(
            candidates=[candidate],
            budget=make_budget(max_mutation_ops=8),
        )
    )
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert PortfolioReason.BUDGET_EXCEEDED.value in decision.evaluations[0].rejection_reasons


def test_evaluate_lane_binds_seed_and_evidence() -> None:
    candidate = make_candidate()
    result = evaluate_lane(
        lane=PortfolioLane.DIFFERENTIAL,
        candidate=candidate,
        oracle=make_oracle(),
        budget=make_budget(seed=42),
        hard=True,
        capability_support=all_lanes_supported(),
    )
    assert result.seed == 42
    assert result.outcome is LaneOutcome.PASS
    assert result.evidence_id.startswith("b")
    assert result.hard is True


def test_lane_observation_from_mapping() -> None:
    obs = LaneObservation.from_mapping(
        {
            "supported": True,
            "status": "PASS",
            "cases_run": 3,
            "budget_used": 2,
            "soft_score": 10,
        }
    )
    assert obs.status == "pass"
    assert obs.soft_score == 10
    assert obs.to_dict()["cases_run"] == 3


# ---------------------------------------------------------------------------
# Differential / metamorphic / sanitizer coverage
# ---------------------------------------------------------------------------


def test_differential_metamorphic_sanitizer_static_model_security_run() -> None:
    decision = select_repair_candidate(make_request())
    lanes = {
        item.lane: item for item in decision.evaluations[0].lane_results
    }
    for lane in (
        PortfolioLane.DIFFERENTIAL,
        PortfolioLane.METAMORPHIC,
        PortfolioLane.SANITIZER,
        PortfolioLane.STATIC,
        PortfolioLane.MODEL,
        PortfolioLane.SECURITY,
    ):
        assert lanes[lane].outcome is LaneOutcome.PASS
        assert lanes[lane].seed == DEFAULT_SEED


def test_sanitizer_optional_fail_is_soft_debt_not_hard() -> None:
    # Sanitizer is not in DEFAULT_HARD_LANES.
    assert PortfolioLane.SANITIZER not in DEFAULT_HARD_LANES
    observations = passing_observations(oracle_id="oracle:held-out-v1")
    observations["sanitizer"] = {
        "supported": True,
        "status": "fail",
        "cases_run": 1,
        "budget_used": 1,
    }
    candidate = make_candidate(lane_observations=observations)
    decision = select_repair_candidate(make_request(candidates=[candidate]))
    assert decision.disposition is PortfolioDisposition.SELECTED
    evaluation = decision.evaluations[0]
    assert "sanitizer" in evaluation.soft_debt
    assert "sanitizer" not in evaluation.hard_failures


# ---------------------------------------------------------------------------
# Multi-candidate mixed portfolio
# ---------------------------------------------------------------------------


def test_mixed_portfolio_selects_best_admissible_records_rest() -> None:
    good = make_candidate("cand:good", blast_radius=4, resource_cost=40)
    self_test = make_candidate(
        "cand:self",
        blast_radius=0,
        resource_cost=0,
        authored_test_ids=("test_hidden_api",),
    )
    flaky = make_candidate("cand:flaky", blast_radius=1, resource_cost=1)
    flaky_obs = passing_observations(oracle_id="oracle:held-out-v1")
    flaky_obs["model"] = {
        "supported": True,
        "status": "flaky",
        "cases_run": 2,
        "budget_used": 2,
    }
    flaky = make_candidate(
        "cand:flaky",
        blast_radius=1,
        resource_cost=1,
        lane_observations=flaky_obs,
    )
    better = make_candidate("cand:better", blast_radius=2, resource_cost=40)

    decision = select_repair_candidate(
        make_request(candidates=[self_test, flaky, good, better])
    )
    assert decision.selected_candidate_id == "cand:better"
    assert decision.ranked_admissible == ("cand:better", "cand:good")
    assert any("flaky" in item for item in decision.flaky_lanes)
    # Self-authored rejection recorded among reasons for non-selection path audit.
    assert any(
        PortfolioReason.SELF_AUTHORED_TEST.value in item.rejection_reasons
        for item in decision.evaluations
    )


# ---------------------------------------------------------------------------
# Injectable lane runner
# ---------------------------------------------------------------------------


def test_custom_lane_runner_injection() -> None:
    calls: list[str] = []

    def runner(**kwargs: object) -> object:
        from ipfs_accelerate_py.agent_supervisor.validation.repair_candidate_portfolio import (
            LaneResult,
        )

        lane = kwargs["lane"]
        assert isinstance(lane, PortfolioLane)
        calls.append(lane.value)
        hard = bool(kwargs["hard"])
        budget = kwargs["budget"]
        assert isinstance(budget, PortfolioSeedBudget)
        # Fail security hard; pass everything else.
        outcome = (
            LaneOutcome.FAIL
            if lane is PortfolioLane.SECURITY
            else LaneOutcome.PASS
        )
        return LaneResult(
            lane=lane,
            outcome=outcome,
            hard=hard,
            seed=budget.seed,
            budget_used=1,
            cases_run=1,
            evidence_id=f"evidence:{lane.value}",
            reason_codes=(
                (PortfolioReason.LANE_FAIL.value,)
                if outcome is LaneOutcome.FAIL
                else (PortfolioReason.OK.value,)
            ),
            oracle_id="oracle:held-out-v1"
            if lane is PortfolioLane.MUTATION
            else "",
        )

    portfolio = RepairCandidatePortfolio(lane_runner=runner)
    decision = portfolio.evaluate(make_request())
    assert set(calls) == {lane.value for lane in PORTFOLIO_LANE_ORDER}
    assert decision.disposition is PortfolioDisposition.ABSTAIN
    assert "security" in decision.evaluations[0].hard_failures


# ---------------------------------------------------------------------------
# Content identity / serialization smoke
# ---------------------------------------------------------------------------


def test_candidate_and_oracle_content_ids_stable() -> None:
    oracle = make_oracle()
    again = make_oracle()
    assert oracle.content_id == again.content_id
    candidate = make_candidate()
    assert candidate.to_dict()["proposal_only"] is True
    assert candidate.content_id == PortfolioCandidate.from_dict(candidate.to_dict()).content_id


def test_request_content_id_changes_with_seed() -> None:
    a = make_request(budget=make_budget(seed=1))
    b = make_request(budget=make_budget(seed=2))
    assert a.content_id != b.content_id


def test_default_hard_obligations_nonempty_and_unique() -> None:
    obligations = default_hard_obligations()
    assert obligations
    ids = [item.obligation_id for item in obligations]
    assert len(ids) == len(set(ids))
    lanes = {item.lane for item in obligations}
    assert lanes == DEFAULT_HARD_LANES
