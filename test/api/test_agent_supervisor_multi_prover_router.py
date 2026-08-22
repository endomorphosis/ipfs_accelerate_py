from __future__ import annotations

import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ContractValidationError,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    AUTHORITATIVE_DISPOSITION_SCHEMA,
    AUTHORITY_LATTICE_SCHEMA,
    CHECKER_TRACE_SCHEMA,
    COUNTEREXAMPLE_TRACE_SCHEMA,
    DEFAULT_PROPERTY_POLICIES,
    HAMMER_TRACE_SCHEMA,
    AttemptOutcome,
    AuthoritativeDisposition,
    AuthorityClass,
    CheckerTrace,
    CounterexampleTrace,
    HammerTrace,
    MultiProverRouter,
    PortfolioResult,
    PortfolioVerdict,
    PropertyKind,
    PropertyObligation,
    PropertyPolicy,
    ProverLane,
    ProverOutput,
    ProverRole,
    authority_class_for_role,
    classify_property_kind,
    derive_authoritative_disposition,
    project_portfolio_traces,
    route_obligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.prover_conformance import (
    LEGACY_CEC_DCEC_WRAPPER,
)
from ipfs_accelerate_py.agent_supervisor.proof.prover_matrix_registry import (
    ProverMatrixSnapshot,
)


def _obligation(kind: PropertyKind) -> PropertyObligation:
    return PropertyObligation(
        obligation_id=f"obligation:{kind.value}",
        property_kind=kind,
        statement=f"reviewed {kind.value} property",
        premise_ids=("premise:a", "premise:b"),
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        (PropertyKind.FINITE_CONSTRAINT, ("z3", "cvc5")),
        (PropertyKind.STATE_MACHINE, ("tla_tlc", "apalache")),
        (PropertyKind.AUTHORIZATION, ("datalog_secpal",)),
        (PropertyKind.PROTOCOL, ("tamarin", "proverif")),
        (PropertyKind.HYPERPROPERTY, ("hyperltl_autohyper_mchyper",)),
        (PropertyKind.RUNTIME_TRACE, ("runtime_mtl",)),
        (PropertyKind.KERNEL_CHECK, ("lean", "coq", "isabelle")),
    ),
)
def test_routes_each_property_to_its_semantic_portfolio(
    kind: PropertyKind, expected: tuple[str, ...]
) -> None:
    plan = route_obligation(_obligation(kind))

    assert plan.prover_ids == expected
    assert plan.obligation.property_kind is kind
    assert plan.plan_id == plan.content_id
    assert all(lane.role.authoritative for lane in plan.lanes)


def test_dcec_tdfol_and_hammer_form_staged_reconstruction_portfolio() -> None:
    plan = MultiProverRouter().plan(_obligation(PropertyKind.TYPED_PLANNING))

    assert plan.prover_ids == (
        "dcec",
        "tdfol",
        "hammer",
        "vampire",
        "e",
        "z3",
        "lean",
        "coq",
        "isabelle",
    )
    roles = {lane.prover_id: lane.role for lane in plan.lanes}
    assert roles["dcec"] is ProverRole.DOMAIN_REASONER
    assert roles["tdfol"] is ProverRole.DOMAIN_REASONER
    assert roles["hammer"] is ProverRole.ORCHESTRATOR
    assert roles["vampire"] is ProverRole.CANDIDATE
    assert roles["lean"] is ProverRole.KERNEL
    assert [lane.stage for lane in plan.lanes] == [0, 0, 1, 2, 2, 2, 3, 3, 3]
    assert all(lane.requires_candidate for lane in plan.lanes if lane.role is ProverRole.KERNEL)


def test_code_obligation_requires_explicit_reviewed_property_classification() -> None:
    code = CodeProofObligation(
        repository_tree_id="tree:1",
        ast_scope_ids=("scope:1",),
        statement="x >= 0",
        template_id="nonnegative",
        template_version="1",
        template_semantic_hash="sha256:template",
        invariant_class="finite_constraint_satisfiability",
    )

    plan = MultiProverRouter().plan(code)

    assert plan.obligation.property_kind is PropertyKind.FINITE_CONSTRAINT
    assert plan.obligation.metadata["repository_tree_id"] == "tree:1"
    assert (
        classify_property_kind(metadata={"property_kind": "secpal"}) is PropertyKind.AUTHORIZATION
    )
    with pytest.raises(ContractValidationError, match="unsupported property"):
        classify_property_kind("theorem-looking-free-text")


def test_model_checking_authority_can_prove_but_candidate_cannot_self_promote() -> None:
    finite = MultiProverRouter().execute(
        _obligation(PropertyKind.FINITE_CONSTRAINT),
        lambda request, cancel: ProverOutput(AttemptOutcome.VERIFIED),
    )

    assert finite.verdict is PortfolioVerdict.PROVED
    assert finite.assurance is AssuranceLevel.SOLVER_CHECKED
    assert len(finite.authority_attempt_ids) == 2

    def planning_runner(request, cancel):
        if request.lane.role is ProverRole.KERNEL:
            return ProverOutput(AttemptOutcome.UNKNOWN, "kernel could not reconstruct")
        return ProverOutput(
            AttemptOutcome.VERIFIED,
            evidence={"premises": list(request.obligation.premise_ids)},
        )

    planning = MultiProverRouter().execute(
        _obligation(PropertyKind.TYPED_PLANNING), planning_runner
    )

    assert planning.verdict is PortfolioVerdict.INCONCLUSIVE
    assert planning.assurance is AssuranceLevel.UNVERIFIED
    assert not planning.authority_attempt_ids
    assert any(
        item.reported_outcome is AttemptOutcome.VERIFIED
        and item.effective_outcome is AttemptOutcome.CANDIDATE
        for item in planning.attempts
        if item.role
        in (
            ProverRole.DOMAIN_REASONER,
            ProverRole.ORCHESTRATOR,
            ProverRole.CANDIDATE,
        )
    )
    assert "no configured reconstruction" in planning.reason


def test_hammer_receives_domain_results_and_kernel_reconstruction_is_authority() -> None:
    seen: dict[str, tuple[str, ...]] = {}

    def runner(request, cancel):
        seen[request.prover_id] = tuple(item["prover_id"] for item in request.prior_attempts)
        if request.lane.role is ProverRole.KERNEL:
            if request.prover_id == "lean":
                return ProverOutput(
                    AttemptOutcome.VERIFIED,
                    evidence={"kernel_receipt_id": "lean:receipt:1"},
                )
            return ProverOutput(AttemptOutcome.UNSUPPORTED)
        return ProverOutput(AttemptOutcome.CANDIDATE)

    result = MultiProverRouter().execute(_obligation(PropertyKind.TEMPORAL_DEONTIC), runner)

    assert seen["hammer"] == ("dcec", "tdfol")
    assert set(seen["vampire"]) == {"dcec", "tdfol", "hammer"}
    assert set(seen["lean"]) == {"dcec", "tdfol", "hammer", "vampire", "e"}
    assert result.verdict is PortfolioVerdict.PROVED
    assert result.assurance is AssuranceLevel.KERNEL_VERIFIED
    authority = [
        item for item in result.attempts if item.attempt_id in result.authority_attempt_ids
    ]
    assert [item.prover_id for item in authority] == ["lean"]


@pytest.mark.parametrize(
    ("raw", "outcome", "verdict"),
    (
        ({"outcome": "unknown"}, AttemptOutcome.UNKNOWN, PortfolioVerdict.INCONCLUSIVE),
        ({"outcome": "unsupported"}, AttemptOutcome.UNSUPPORTED, PortfolioVerdict.UNSUPPORTED),
        (TimeoutError("bounded timeout"), AttemptOutcome.TIMEOUT, PortfolioVerdict.INCONCLUSIVE),
        ("not a typed result", AttemptOutcome.MALFORMED, PortfolioVerdict.ERROR),
    ),
)
def test_unknown_unsupported_timeout_and_malformed_fail_closed(
    raw, outcome: AttemptOutcome, verdict: PortfolioVerdict
) -> None:
    def runner(request, cancel):
        if isinstance(raw, BaseException):
            raise raw
        return raw

    result = MultiProverRouter().execute(_obligation(PropertyKind.AUTHORIZATION), runner)

    assert result.verdict is verdict
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert len(result.attempts) == 1
    assert result.attempts[0].effective_outcome is outcome
    assert result.fail_closed


def test_authority_disagreement_fails_closed_and_retains_both_attempts() -> None:
    def runner(request, cancel):
        if request.prover_id == "z3":
            return ProverOutput(AttemptOutcome.VERIFIED)
        return ProverOutput(
            AttemptOutcome.COUNTEREXAMPLE,
            evidence={"model": {"x": -1}},
            conclusive=True,
        )

    result = MultiProverRouter().execute(_obligation(PropertyKind.FINITE_CONSTRAINT), runner)

    assert result.verdict is PortfolioVerdict.INCONCLUSIVE
    assert result.disagreement
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert len(result.attempts) == len(result.plan.lanes) == 2
    assert {item.reported_outcome for item in result.attempts} == {
        AttemptOutcome.VERIFIED,
        AttemptOutcome.COUNTEREXAMPLE,
    }


def test_conclusive_counterexample_cancels_redundant_stages_and_retains_all() -> None:
    calls: list[str] = []

    def runner(request, cancel: threading.Event):
        calls.append(request.prover_id)
        if request.prover_id == "dcec":
            # Domain reasoners are candidate-only, so this cannot authoritatively
            # disprove the property despite claiming conclusiveness.
            return ProverOutput(AttemptOutcome.COUNTEREXAMPLE, conclusive=True)
        if request.prover_id == "tdfol":
            return ProverOutput(AttemptOutcome.CANDIDATE)
        if request.prover_id == "lean":
            return ProverOutput(AttemptOutcome.COUNTEREXAMPLE, conclusive=True)
        return ProverOutput(AttemptOutcome.CANDIDATE)

    # A custom kernel-check property demonstrates cancellation without racing
    # a candidate-only domain result.
    def kernel_runner(request, cancel):
        calls.append(request.prover_id)
        if request.prover_id == "lean":
            return ProverOutput(
                AttemptOutcome.COUNTEREXAMPLE,
                evidence={"kernel_error": "statement is false"},
                conclusive=True,
            )
        cancel.wait(0.2)
        return ProverOutput(AttemptOutcome.CANCELLED)

    result = MultiProverRouter().execute(_obligation(PropertyKind.KERNEL_CHECK), kernel_runner)

    assert result.verdict is PortfolioVerdict.DISPROVED
    assert result.counterexample_attempt_id
    assert len(result.attempts) == 3
    assert result.attempts[0].prover_id == "lean"
    assert all(
        item.effective_outcome in (AttemptOutcome.COUNTEREXAMPLE, AttemptOutcome.CANCELLED)
        for item in result.attempts
    )
    assert any(item.cancellation_requested for item in result.attempts[1:])


def test_strict_output_rejects_unknown_fields_and_oversized_evidence() -> None:
    router = MultiProverRouter(maximum_evidence_bytes=8)

    unknown = router.execute(
        _obligation(PropertyKind.AUTHORIZATION),
        lambda request, cancel: {"outcome": "verified", "provider_says": "proved"},
    )
    oversized = router.execute(
        _obligation(PropertyKind.AUTHORIZATION),
        lambda request, cancel: {
            "outcome": "verified",
            "evidence": {"payload": "too large"},
        },
    )

    assert unknown.attempts[0].effective_outcome is AttemptOutcome.MALFORMED
    assert oversized.attempts[0].effective_outcome is AttemptOutcome.MALFORMED
    assert unknown.verdict is oversized.verdict is PortfolioVerdict.ERROR


def test_matrix_and_legacy_conformance_gates_fail_closed_before_execution() -> None:
    calls: list[str] = []

    def runner(request, cancel):
        calls.append(request.prover_id)
        return ProverOutput(AttemptOutcome.VERIFIED)

    empty_matrix = ProverMatrixSnapshot(
        entries=(),
        generated_at="2026-07-23T00:00:00Z",
        duration_ms=0,
        self_tests_requested=True,
        bounded=True,
        max_self_tests=64,
        matrix_timeout_seconds=60.0,
        documentation_source=None,
    )
    matrix_gated = MultiProverRouter(matrix=empty_matrix).execute(
        _obligation(PropertyKind.AUTHORIZATION), runner
    )

    legacy_policy = PropertyPolicy(
        PropertyKind.AUTHORIZATION,
        (
            ProverLane(
                "legacy_dcec",
                ProverRole.MODEL_CHECKER,
                authority_capability="authorization_policy",
                translation_path_id=LEGACY_CEC_DCEC_WRAPPER,
            ),
        ),
    )
    conformance_gated = MultiProverRouter({PropertyKind.AUTHORIZATION: legacy_policy}).execute(
        _obligation(PropertyKind.AUTHORIZATION), runner
    )

    assert calls == []
    assert matrix_gated.verdict is PortfolioVerdict.UNSUPPORTED
    assert matrix_gated.attempts[0].effective_outcome is AttemptOutcome.UNAVAILABLE
    assert conformance_gated.verdict is PortfolioVerdict.UNSUPPORTED
    assert conformance_gated.attempts[0].effective_outcome is AttemptOutcome.UNSUPPORTED
    assert conformance_gated.attempts[0].conformance_gate_id


def test_conclusive_candidate_counterexample_stops_reconstruction() -> None:
    def runner(request, cancel):
        if request.prover_id == "hammer":
            return ProverOutput(AttemptOutcome.CANDIDATE)
        if request.prover_id == "vampire":
            return ProverOutput(
                AttemptOutcome.COUNTEREXAMPLE,
                evidence={"countermodel": {"premise": True, "goal": False}},
                conclusive=True,
            )
        cancel.wait(0.1)
        return ProverOutput(AttemptOutcome.CANCELLED)

    result = MultiProverRouter().execute(_obligation(PropertyKind.FIRST_ORDER_THEOREM), runner)

    assert result.verdict is PortfolioVerdict.DISPROVED
    vampire = next(item for item in result.attempts if item.prover_id == "vampire")
    assert vampire.role is ProverRole.CANDIDATE
    assert vampire.conclusive
    assert all(
        item.effective_outcome is AttemptOutcome.CANCELLED
        for item in result.attempts
        if item.role is ProverRole.KERNEL
    )


def test_kernel_required_code_obligation_adds_reconstruction_authorities() -> None:
    code = CodeProofObligation(
        repository_tree_id="tree:kernel",
        ast_scope_ids=("scope:kernel",),
        statement="x >= 0",
        template_id="nonnegative",
        template_version="1",
        template_semantic_hash="sha256:template",
        invariant_class="finite_constraint",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )

    def runner(request, cancel):
        if request.prover_id == "z3":
            return ProverOutput(AttemptOutcome.VERIFIED)
        if request.prover_id == "lean":
            return ProverOutput(AttemptOutcome.VERIFIED)
        return ProverOutput(AttemptOutcome.UNSUPPORTED)

    result = MultiProverRouter().execute(code, runner)

    assert result.plan.prover_ids == ("z3", "cvc5", "lean", "coq", "isabelle")
    assert result.verdict is PortfolioVerdict.PROVED
    assert result.assurance is AssuranceLevel.KERNEL_VERIFIED
    assert [
        item.prover_id
        for item in result.attempts
        if item.attempt_id in result.authority_attempt_ids
    ] == ["z3", "lean"]


def test_result_is_canonical_and_retains_plan_order() -> None:
    result = MultiProverRouter().execute(
        _obligation(PropertyKind.PROTOCOL),
        lambda request, cancel: ProverOutput(AttemptOutcome.UNKNOWN),
    )
    payload = result.to_record()

    assert payload["schema"] == PortfolioResult.SCHEMA
    assert payload["content_id"] == result.content_id
    assert [item["prover_id"] for item in payload["attempts"]] == [
        "tamarin",
        "proverif",
    ]
    assert PortfolioResult.from_dict(payload) == result
    assert set(DEFAULT_PROPERTY_POLICIES) == set(PropertyKind)


def test_trace_contract_schema_identities_are_declared_and_bound() -> None:
    assert HammerTrace.SCHEMA == HAMMER_TRACE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/hammer-trace@1"
    )
    assert CounterexampleTrace.SCHEMA == COUNTEREXAMPLE_TRACE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/counterexample-trace@1"
    )
    assert CheckerTrace.SCHEMA == CHECKER_TRACE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/checker-trace@1"
    )
    assert AuthoritativeDisposition.SCHEMA == AUTHORITATIVE_DISPOSITION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/authoritative-disposition@1"
    )

    result = MultiProverRouter().execute(
        _obligation(PropertyKind.AUTHORIZATION),
        lambda request, cancel: ProverOutput(AttemptOutcome.UNKNOWN),
    )
    disposition = derive_authoritative_disposition(result).to_dict()
    assert disposition["authority_lattice"] == AUTHORITY_LATTICE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/authority-lattice@1"
    )


def test_solver_counterexamples_retain_obligation_scope_and_bounds() -> None:
    obligation = PropertyObligation(
        obligation_id="obligation:bounded-safety",
        property_kind=PropertyKind.FIRST_ORDER_THEOREM,
        statement="No worker owns two tasks.",
        premise_ids=("premise:mutex",),
        metadata={
            "ast_scope_ids": ["src/mutex.py::acquire"],
            "finite_bounds": {"workers": 2, "steps": 8},
            "environment_lock_id": "lock:current",
        },
    )

    def runner(request, cancel):
        if request.prover_id == "vampire":
            return ProverOutput(
                AttemptOutcome.COUNTEREXAMPLE,
                evidence={"countermodel": {"owner": "w0"}},
                conclusive=True,
            )
        return ProverOutput(AttemptOutcome.CANDIDATE)

    result = MultiProverRouter().execute(obligation, runner)
    vampire = next(item for item in result.attempts if item.prover_id == "vampire")
    hammer, counterexamples, checkers = project_portfolio_traces(result)
    disposition = derive_authoritative_disposition(result)

    assert result.verdict is PortfolioVerdict.DISPROVED
    assert vampire.conclusive
    assert vampire.evidence["ast_scope_ids"] == ["src/mutex.py::acquire"]
    assert vampire.evidence["finite_bounds"] == {"workers": 2, "steps": 8}
    assert counterexamples[0].ast_scope_ids == ("src/mutex.py::acquire",)
    assert counterexamples[0].finite_bounds == {"workers": 2, "steps": 8}
    assert all(not item.authoritative for item in hammer)
    assert all(not item.accepted for item in checkers if item.role is ProverRole.KERNEL)
    assert disposition.verdict is PortfolioVerdict.DISPROVED
    assert disposition.fail_closed


def test_counterexample_bounds_or_scope_disagreement_is_not_conclusive() -> None:
    obligation = PropertyObligation(
        obligation_id="obligation:scoped",
        property_kind=PropertyKind.FINITE_CONSTRAINT,
        statement="x >= 0",
        metadata={
            "ast_scope_ids": ["src/state.py::step"],
            "finite_bounds": {"k": 3},
        },
    )

    def runner(request, cancel):
        if request.prover_id == "z3":
            return ProverOutput(
                AttemptOutcome.COUNTEREXAMPLE,
                evidence={"finite_bounds": {"k": 99}, "model": {"x": -1}},
                conclusive=True,
            )
        return ProverOutput(AttemptOutcome.UNKNOWN)

    result = MultiProverRouter().execute(obligation, runner)
    z3 = next(item for item in result.attempts if item.prover_id == "z3")

    assert z3.effective_outcome is AttemptOutcome.UNKNOWN
    assert not z3.conclusive
    assert result.verdict is not PortfolioVerdict.DISPROVED
    assert "bounds_disagree" in z3.detail


def test_resource_policy_denies_solver_lane_before_execution() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
        CHECKER_RESOURCE_SHARD,
        SOLVER_RESOURCE_SHARD,
        MultiProverResourceBudget,
        MultiProverResourceLease,
        resource_class_for_prover,
        resource_shard_for_class,
    )

    assert resource_shard_for_class(resource_class_for_prover("vampire")) == SOLVER_RESOURCE_SHARD
    assert resource_shard_for_class(resource_class_for_prover("lean")) == CHECKER_RESOURCE_SHARD

    lease = MultiProverResourceLease(MultiProverResourceBudget())
    lease.close()
    calls: list[str] = []

    def runner(request, cancel):
        calls.append(request.prover_id)
        return ProverOutput(AttemptOutcome.VERIFIED)

    result = MultiProverRouter(resource_lease=lease).execute(
        _obligation(PropertyKind.AUTHORIZATION), runner
    )

    assert calls == []
    assert result.verdict is PortfolioVerdict.INCONCLUSIVE
    assert result.attempts[0].effective_outcome is AttemptOutcome.BLOCKED
    assert "resource policy denied" in result.attempts[0].detail


def test_stale_kernel_version_certificate_cannot_self_verify() -> None:
    obligation = PropertyObligation(
        obligation_id="obligation:versioned",
        property_kind=PropertyKind.KERNEL_CHECK,
        statement="True",
        metadata={"kernel_version": "4.19.0", "environment_lock_id": "lock:current"},
    )

    def runner(request, cancel):
        if request.prover_id == "lean":
            return ProverOutput(
                AttemptOutcome.VERIFIED,
                evidence={"kernel_version": "1.0.0", "environment_lock_id": "lock:stale"},
            )
        return ProverOutput(AttemptOutcome.UNSUPPORTED)

    result = MultiProverRouter().execute(obligation, runner)
    lean = next(item for item in result.attempts if item.prover_id == "lean")
    disposition = derive_authoritative_disposition(result)

    assert lean.effective_outcome is AttemptOutcome.MALFORMED
    assert result.verdict is not PortfolioVerdict.PROVED
    assert not result.authority_attempt_ids
    assert disposition.assurance is AssuranceLevel.UNVERIFIED


def test_evidence_store_rejects_stale_environment_and_statement_reuse(tmp_path) -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.prover_evidence_store import (
        EvidenceLookupStatus,
        EvidenceRejectionReason,
        ProverEvidenceStore,
        build_prover_evidence_key,
        ConformanceBinding,
    )

    obligation = PropertyObligation(
        obligation_id="obligation:cached",
        property_kind=PropertyKind.STATE_MACHINE,
        statement="No two workers own the same task.",
        metadata={
            "repository_tree_id": "git-tree:abc123",
            "environment_lock_id": "lock:old",
        },
        required_assurance=AssuranceLevel.SOLVER_CHECKED,
    )

    def runner(request, cancel):
        return ProverOutput(AttemptOutcome.VERIFIED)

    result = MultiProverRouter().execute(obligation, runner)
    key = build_prover_evidence_key(
        property_class=PropertyKind.STATE_MACHINE,
        normalized_model={"statement": obligation.statement},
        translator_profile={"id": "tla", "version": "1"},
        assumptions=(),
        finite_bounds={"workers": 2},
        prover_versions={"tla_tlc": "2.19", "apalache": "0.45"},
        kernel_versions={"apalache-typechecker": "0.45"},
        policy={"id": result.plan.policy_id},
        repository_tree_id="git-tree:abc123",
        conformance_fixture_set_id="fixture-set:state-machine@8",
    )
    store = ProverEvidenceStore(tmp_path)
    stored = store.put(
        key,
        result,
        conformance=ConformanceBinding(
            fixture_set_id="fixture-set:state-machine@8",
            report_ids=("conformance-report:1",),
            passed=True,
            permitted_assurance=AssuranceLevel.SOLVER_CHECKED,
        ),
        metadata={"environment_lock_id": "lock:old"},
    )
    assert stored.stored

    stale_env = store.lookup(key, current_environment_lock_id="lock:current")
    stale_stmt = store.lookup(key, current_statement="A different theorem.")
    fresh = store.lookup(
        key,
        current_environment_lock_id="lock:old",
        current_statement=obligation.statement,
    )

    assert stale_env.status is EvidenceLookupStatus.REJECTED
    assert EvidenceRejectionReason.STALE_ENVIRONMENT.value in stale_env.reason_codes
    assert stale_stmt.status is EvidenceLookupStatus.REJECTED
    assert EvidenceRejectionReason.STALE_STATEMENT.value in stale_stmt.reason_codes
    assert fresh.status is EvidenceLookupStatus.HIT


def test_authority_lattice_keeps_hammer_candidates_off_the_proof_root() -> None:
    assert authority_class_for_role(ProverRole.ORCHESTRATOR) is AuthorityClass.CANDIDATE
    assert authority_class_for_role(ProverRole.CANDIDATE) is AuthorityClass.CANDIDATE
    assert authority_class_for_role(ProverRole.MODEL_CHECKER) is AuthorityClass.INDEPENDENT_CHECKER
    assert authority_class_for_role(ProverRole.KERNEL) is AuthorityClass.KERNEL
    assert not AuthorityClass.CANDIDATE.can_author_proof
    assert AuthorityClass.KERNEL.can_author_proof
