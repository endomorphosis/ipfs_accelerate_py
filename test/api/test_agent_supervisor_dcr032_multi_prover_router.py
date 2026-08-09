"""DCR-032: route obligations only to capability-qualified deterministic provers.

Acceptance:
* Required missing / unsupported / error backends fail closed.
* General LLM and remote nondeterministic providers are not representable.
* Only DCEC, TDFOL, SMT, theorem, and structural backends may be admitted, and
  only where declared fragments and self-tests match.
* Importability, simulated SAT, and unknown never count as proof.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    PROVER_PORTFOLIO_INTERFACE,
    SOLVER_READINESS_INTERFACE,
    AttemptOutcome,
    CapabilityAdmissionDecision,
    CapabilityAdmissionStatus,
    DeterministicBackendKind,
    MultiProverRouter,
    PortfolioVerdict,
    PropertyKind,
    PropertyObligation,
    PropertyPolicy,
    ProverCapabilityAdmission,
    ProverLane,
    ProverOutput,
    ProverRole,
    backend_is_representable,
    classify_deterministic_backend,
)
from ipfs_accelerate_py.agent_supervisor.proof.prover_matrix_registry import (
    ProverMatrixSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.proof.solver_readiness import (
    SolverAuthority,
    SolverBackendFamily,
    SolverBackendReadiness,
    SolverReadinessReport,
    SolverReadinessStatus,
)


def _obligation(
    kind: PropertyKind = PropertyKind.FINITE_CONSTRAINT,
    *,
    metadata: dict | None = None,
) -> PropertyObligation:
    return PropertyObligation(
        obligation_id=f"obligation:dcr032:{kind.value}",
        property_kind=kind,
        statement=f"deterministic {kind.value} property",
        premise_ids=("premise:a",),
        metadata=metadata or {},
    )


def _readiness(
    *,
    family: SolverBackendFamily,
    provider_id: str,
    status: SolverReadinessStatus = SolverReadinessStatus.AVAILABLE_CANDIDATE,
    supported: bool = True,
    self_test_passed: bool = True,
    reason_code: str = "ok",
    reason: str = "available",
) -> SolverBackendReadiness:
    return SolverBackendReadiness(
        family=family,
        status=status,
        provider_id=provider_id,
        capability_revision="test-revision",
        package_version="0.0-test",
        observations=(),
        reconstruction_compatible=True,
        reason_code=reason_code,
        reason=reason,
        supported=supported,
        authority=SolverAuthority.NON_AUTHORITATIVE,
        self_test_passed=self_test_passed,
    )


def _full_readiness_report(
    overrides: dict[SolverBackendFamily, SolverBackendReadiness] | None = None,
) -> SolverReadinessReport:
    defaults = {
        SolverBackendFamily.DCEC: _readiness(
            family=SolverBackendFamily.DCEC, provider_id="dcec"
        ),
        SolverBackendFamily.Z3: _readiness(
            family=SolverBackendFamily.Z3, provider_id="z3"
        ),
        SolverBackendFamily.TDFOL: _readiness(
            family=SolverBackendFamily.TDFOL, provider_id="tdfol"
        ),
        SolverBackendFamily.CEC: _readiness(
            family=SolverBackendFamily.CEC, provider_id="cec"
        ),
        SolverBackendFamily.HAMMER: _readiness(
            family=SolverBackendFamily.HAMMER, provider_id="hammer"
        ),
    }
    if overrides:
        defaults.update(overrides)
    return SolverReadinessReport(backends=tuple(defaults[f] for f in defaults))


def test_interfaces_are_declared() -> None:
    assert PROVER_PORTFOLIO_INTERFACE == "ProverPortfolio@1"
    assert SOLVER_READINESS_INTERFACE == "SolverReadiness@1"
    assert MultiProverRouter.INTERFACE == PROVER_PORTFOLIO_INTERFACE
    assert ProverCapabilityAdmission.INTERFACE == PROVER_PORTFOLIO_INTERFACE
    assert ProverCapabilityAdmission.READINESS_INTERFACE == SOLVER_READINESS_INTERFACE


@pytest.mark.parametrize(
    ("prover_id", "kind"),
    (
        ("dcec", DeterministicBackendKind.DCEC),
        ("tdfol", DeterministicBackendKind.TDFOL),
        ("z3", DeterministicBackendKind.SMT),
        ("cvc5", DeterministicBackendKind.SMT),
        ("hammer", DeterministicBackendKind.THEOREM),
        ("lean", DeterministicBackendKind.THEOREM),
        ("vampire", DeterministicBackendKind.THEOREM),
        ("tamarin", DeterministicBackendKind.STRUCTURAL),
        ("datalog_secpal", DeterministicBackendKind.STRUCTURAL),
    ),
)
def test_classifies_only_deterministic_backend_families(
    prover_id: str, kind: DeterministicBackendKind
) -> None:
    assert classify_deterministic_backend(prover_id) is kind
    assert backend_is_representable(prover_id)


@pytest.mark.parametrize(
    "prover_id",
    (
        "llm",
        "openai",
        "anthropic",
        "leanstral",
        "leanstral-local",
        "remote_service",
        "remote:prover",
        "gpt-4",
        "grok",
        "codex",
        "nondeterministic",
        "cloud_llm",
        "model_assistant",
        "litellm",
    ),
)
def test_general_llm_and_remote_nondeterministic_providers_are_not_representable(
    prover_id: str,
) -> None:
    assert not backend_is_representable(prover_id)
    with pytest.raises(ContractValidationError, match="not representable"):
        classify_deterministic_backend(prover_id)

    admission = ProverCapabilityAdmission(require_capability_evidence=False)
    decision = admission.admit_backend(prover_id)
    assert decision.status is CapabilityAdmissionStatus.UNREPRESENTABLE
    assert not decision.admitted
    assert decision.backend_kind is None


def test_model_assistant_role_is_unrepresentable_even_for_known_ids() -> None:
    admission = ProverCapabilityAdmission(require_capability_evidence=False)
    decision = admission.admit_backend("z3", role=ProverRole.MODEL_ASSISTANT)
    assert decision.status is CapabilityAdmissionStatus.UNREPRESENTABLE
    assert not decision.admitted


def test_required_missing_backend_fails_closed_before_runner() -> None:
    calls: list[str] = []
    empty_matrix = ProverMatrixSnapshot(
        entries=(),
        generated_at="2026-08-09T00:00:00Z",
        duration_ms=0,
        self_tests_requested=True,
        bounded=True,
        max_self_tests=64,
        matrix_timeout_seconds=60.0,
        documentation_source=None,
    )
    admission = ProverCapabilityAdmission(
        matrix=empty_matrix,
        required_backends=("z3",),
        require_capability_evidence=True,
        require_self_test=True,
    )
    router = MultiProverRouter(
        matrix=empty_matrix,
        capability_admission=admission,
    )

    def runner(request, cancel):
        calls.append(request.prover_id)
        return ProverOutput(AttemptOutcome.VERIFIED)

    result = router.execute(_obligation(PropertyKind.FINITE_CONSTRAINT), runner)

    assert calls == []
    assert result.fail_closed
    assert result.verdict is PortfolioVerdict.UNSUPPORTED
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert all(
        item.effective_outcome is AttemptOutcome.UNAVAILABLE for item in result.attempts
    )
    assert any("required backend" in item.detail for item in result.attempts)


def test_required_unsupported_backend_fails_closed() -> None:
    readiness = _full_readiness_report(
        {
            SolverBackendFamily.Z3: _readiness(
                family=SolverBackendFamily.Z3,
                provider_id="z3",
                status=SolverReadinessStatus.UNSUPPORTED,
                supported=False,
                self_test_passed=False,
                reason_code="missing_module",
                reason="z3 bridge absent",
            )
        }
    )
    admission = ProverCapabilityAdmission(
        readiness=readiness,
        required_backends=("z3",),
        require_capability_evidence=True,
        require_self_test=True,
    )
    decision = admission.admit_backend("z3", required=True)
    assert decision.status is CapabilityAdmissionStatus.MISSING
    assert not decision.admitted

    router = MultiProverRouter(capability_admission=admission)
    result = router.execute(
        _obligation(PropertyKind.FINITE_CONSTRAINT),
        lambda request, cancel: ProverOutput(AttemptOutcome.VERIFIED),
    )
    assert result.fail_closed
    assert result.verdict is PortfolioVerdict.UNSUPPORTED
    assert result.assurance is AssuranceLevel.UNVERIFIED


def test_required_error_backend_fails_closed() -> None:
    readiness = _full_readiness_report(
        {
            SolverBackendFamily.Z3: _readiness(
                family=SolverBackendFamily.Z3,
                provider_id="z3",
                status=SolverReadinessStatus.UNSUPPORTED,
                supported=False,
                self_test_passed=False,
                reason_code="probe_error",
                reason="z3 probe crashed",
            )
        }
    )
    admission = ProverCapabilityAdmission(
        readiness=readiness,
        required_backends=("z3",),
        require_capability_evidence=True,
    )
    decision = admission.admit_backend("z3", required=True)
    assert decision.status is CapabilityAdmissionStatus.ERROR
    assert decision.attempt_outcome is AttemptOutcome.ERROR

    router = MultiProverRouter(capability_admission=admission)
    result = router.execute(
        _obligation(PropertyKind.FINITE_CONSTRAINT),
        lambda request, cancel: ProverOutput(AttemptOutcome.VERIFIED),
    )
    assert result.fail_closed
    assert result.verdict is PortfolioVerdict.ERROR
    assert all(item.effective_outcome is AttemptOutcome.ERROR for item in result.attempts)


def test_logic_fragment_must_match_before_admission() -> None:
    admission = ProverCapabilityAdmission(
        require_capability_evidence=False,
        logic_fragment="smt",
    )
    smt = admission.admit_backend("z3")
    dcec = admission.admit_backend("dcec")
    assert smt.admitted
    assert smt.fragment_matched
    assert not dcec.admitted
    assert dcec.status is CapabilityAdmissionStatus.UNSUPPORTED
    assert "logic fragment" in dcec.detail


def test_obligation_metadata_required_backends_and_fragment_are_honored() -> None:
    admission = ProverCapabilityAdmission(
        require_capability_evidence=False,
        required_backends=(),
    )
    scoped = admission.with_obligation_context(
        _obligation(
            PropertyKind.TYPED_PLANNING,
            metadata={
                "logic_fragment": "deontic",
                "required_backends": ["dcec", "tdfol"],
            },
        )
    )
    assert scoped.logic_fragment == "deontic"
    assert scoped.required_backends == ("dcec", "tdfol")
    assert scoped.admit_backend("dcec").admitted
    assert scoped.admit_backend("tdfol").admitted
    assert not scoped.admit_backend("tamarin").admitted


def test_self_test_required_for_readiness_admission() -> None:
    readiness = _full_readiness_report(
        {
            SolverBackendFamily.Z3: _readiness(
                family=SolverBackendFamily.Z3,
                provider_id="z3",
                self_test_passed=False,
                reason_code="self_test_pending",
                reason="self-test not run",
            )
        }
    )
    admission = ProverCapabilityAdmission(
        readiness=readiness,
        require_self_test=True,
        require_capability_evidence=True,
    )
    decision = admission.admit_backend("z3")
    assert not decision.admitted
    assert decision.status is CapabilityAdmissionStatus.UNSUPPORTED
    assert "self-test" in decision.detail


def test_capability_qualified_smt_route_can_prove_with_authority() -> None:
    readiness = _full_readiness_report()
    admission = ProverCapabilityAdmission(
        readiness=readiness,
        require_capability_evidence=True,
        require_self_test=True,
    )
    router = MultiProverRouter(capability_admission=admission)

    # Finite-constraint portfolio is z3+cvc5.  Only z3 has readiness evidence;
    # cvc5 remains unsupported while z3 may still authoritatively prove.
    def runner(request, cancel):
        if request.prover_id == "z3":
            return ProverOutput(AttemptOutcome.VERIFIED)
        return ProverOutput(AttemptOutcome.UNSUPPORTED)

    result = router.execute(_obligation(PropertyKind.FINITE_CONSTRAINT), runner)
    assert result.verdict is PortfolioVerdict.PROVED
    assert result.assurance is AssuranceLevel.SOLVER_CHECKED
    assert result.fail_closed
    assert any(item.prover_id == "z3" for item in result.attempts)


def test_candidate_sat_without_reconstruction_never_proves() -> None:
    admission = ProverCapabilityAdmission(require_capability_evidence=False)
    router = MultiProverRouter(
        capability_admission=admission,
        require_deterministic_admission=True,
    )

    def runner(request, cancel):
        # Domain / candidate lanes report verified; kernels stay unknown.
        if request.lane.role is ProverRole.KERNEL:
            return ProverOutput(AttemptOutcome.UNKNOWN, "no reconstruction")
        return ProverOutput(
            AttemptOutcome.VERIFIED,
            evidence={"sat": True, "model": {"x": 1}},
        )

    result = router.execute(_obligation(PropertyKind.TYPED_PLANNING), runner)
    assert result.verdict is PortfolioVerdict.INCONCLUSIVE
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert not result.authority_attempt_ids
    assert result.fail_closed
    assert any(
        item.reported_outcome is AttemptOutcome.VERIFIED
        and item.effective_outcome is AttemptOutcome.CANDIDATE
        for item in result.attempts
    )


def test_unrepresentable_lane_is_gated_when_deterministic_admission_required() -> None:
    policy = PropertyPolicy(
        PropertyKind.AUTHORIZATION,
        (
            ProverLane("leanstral-draft", ProverRole.MODEL_ASSISTANT),
        ),
    )
    # Construction of a leanstral model-assistant lane is still possible for
    # non-portfolio draft paths, but deterministic admission rejects it.
    admission = ProverCapabilityAdmission(require_capability_evidence=False)
    decision = admission.admit_lane(policy.lanes[0])
    assert decision.status is CapabilityAdmissionStatus.UNREPRESENTABLE

    # Provide a complete policy map so MultiProverRouter construction succeeds.
    policies = dict(MultiProverRouter().policies)
    policies[PropertyKind.AUTHORIZATION] = policy
    router = MultiProverRouter(
        policies,
        capability_admission=admission,
    )
    result = router.execute(
        _obligation(PropertyKind.AUTHORIZATION),
        lambda request, cancel: ProverOutput(AttemptOutcome.VERIFIED),
    )
    assert result.fail_closed
    assert result.verdict is PortfolioVerdict.UNSUPPORTED
    assert result.attempts[0].effective_outcome is AttemptOutcome.UNSUPPORTED
    assert "not representable" in result.attempts[0].detail or (
        "model-assistant" in result.attempts[0].detail
    )


def test_importability_alone_never_counts_as_proof() -> None:
    """Discoverable backends without smoke/self-test evidence stay non-proof."""

    empty_matrix = ProverMatrixSnapshot(
        entries=(),
        generated_at="2026-08-09T00:00:00Z",
        duration_ms=0,
        self_tests_requested=True,
        bounded=True,
        max_self_tests=64,
        matrix_timeout_seconds=60.0,
        documentation_source=None,
    )
    router = MultiProverRouter(
        matrix=empty_matrix,
        require_deterministic_admission=True,
    )
    result = router.execute(
        _obligation(PropertyKind.AUTHORIZATION),
        lambda request, cancel: ProverOutput(AttemptOutcome.VERIFIED),
    )
    assert result.fail_closed
    assert result.verdict is not PortfolioVerdict.PROVED
    assert result.assurance is AssuranceLevel.UNVERIFIED


def test_unknown_outcome_never_promotes_to_proved() -> None:
    admission = ProverCapabilityAdmission(require_capability_evidence=False)
    router = MultiProverRouter(capability_admission=admission)
    result = router.execute(
        _obligation(PropertyKind.AUTHORIZATION),
        lambda request, cancel: ProverOutput(AttemptOutcome.UNKNOWN),
    )
    assert result.verdict is PortfolioVerdict.INCONCLUSIVE
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert result.fail_closed


def test_filter_lanes_retains_every_decision() -> None:
    admission = ProverCapabilityAdmission(require_capability_evidence=False)
    lanes = (
        ProverLane("z3", ProverRole.MODEL_CHECKER, authority_capability="finite"),
        ProverLane("dcec", ProverRole.DOMAIN_REASONER),
    )
    filtered = admission.filter_lanes(lanes)
    assert len(filtered) == 2
    assert all(
        isinstance(decision, CapabilityAdmissionDecision) for _, decision in filtered
    )
    assert filtered[0][1].admitted
    assert filtered[1][1].admitted
