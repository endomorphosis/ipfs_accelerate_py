from __future__ import annotations

import json
from dataclasses import replace

import ipfs_accelerate_py.agent_supervisor.federation.formal.models as formal_models
import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationLifecycleState,
)
from ipfs_accelerate_py.agent_supervisor.federation.formal import (
    ADVERSARIAL_PROPERTY,
    CASF_FORMAL_IDENTITY_SCHEMA,
    CASF_FORMAL_RECEIPT_SCHEMA,
    CASF_FORMAL_SUITE_SCHEMA,
    AdversarialMutation,
    ExternalCheckStatus,
    ExternalModelInvariant,
    FederationFormalError,
    FederationFormalIdentity,
    FederationFormalProperty,
    HermeticCheckStatus,
    build_federation_formal_suite,
    check_federation_formal_suite,
    check_federation_scenario,
    run_external_model_checks,
)
from ipfs_accelerate_py.agent_supervisor.federation.lifecycle import legal_transitions
from ipfs_accelerate_py.agent_supervisor.proof.prover_matrix_registry import (
    DEFAULT_PROVER_DEFINITIONS,
    CommandRequest,
    CommandResult,
    ProverMatrixEntry,
    ProverMatrixProbeConfig,
    ProverMatrixRegistry,
    ProverMatrixSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_state_model import (
    CounterexampleTrace,
    ModelCheckBounds,
    ModelCheckerTool,
    ModelCheckStatus,
    SupervisorStateModelChecker,
    TransitionRule,
)

SOURCE_REVISION = "b8328ec3a9cc066acfb3240d0e4b03d16950f5c7"
SOURCE_TREE = "1e26e0b1c7d7b8df9eafb1d2e7aede6bfea19233"
CASF_INVARIANT_NAMES = {
    FederationFormalProperty.EVENT_DELIVERY: "IdempotentAuthoritativeEffect",
    FederationFormalProperty.CLAIM_LEASE_FENCE: "CurrentFenceWins",
    FederationFormalProperty.LIFECYCLE: "LegalLifecycleTransition",
    FederationFormalProperty.SHARD_TRANSFER: "UniqueShardOwner",
    FederationFormalProperty.BUDGET_CONSERVATION: "BudgetConservation",
    FederationFormalProperty.CAUSAL_PROPAGATION: "CausalParentBeforeDependent",
}


def _identity() -> FederationFormalIdentity:
    return FederationFormalIdentity(
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        state_schema="casf-control-schema@3",
        generation_id="generation:casf-036-rescue",
        policy_id="policy:CASF-PLAN-R1",
        policy_revision=1,
        capability_ids=(
            "capability:bounded-local-proof",
            "capability:typed-quack-owner",
        ),
        federation_id="federation:causal-event-supervisor-v1",
        supervisor_ids=("supervisor:grok-build", "supervisor:codex-backup"),
        task_id="CASF-036",
        attempt_id="attempt:casf-036-codex-rescue-1",
        lease_id="lease:casf-036-codex-rescue-1",
        fencing_epoch=7,
        assignment_revision=3,
        worktree_id="worktree:casf-036-codex-rescue-20260824",
    )


def _bounds(*, max_steps: int = 12) -> ModelCheckBounds:
    return ModelCheckBounds(
        max_steps=max_steps,
        max_retries=1,
        max_fence=16,
        max_tasks=2,
        max_agents=2,
        max_states=16,
        max_transitions=64,
        max_evidence_ids=1,
    )


def _suite():
    return build_federation_formal_suite(_identity(), bounds=_bounds())


def _matrix_entry(prover_id: str, *, discovered: bool = False) -> ProverMatrixEntry:
    return ProverMatrixEntry(
        prover_id=prover_id,
        display_name=prover_id,
        family="state_machine",
        absent=not discovered,
        discovered=discovered,
        versioned=False,
        smoke_tested=False,
        translation_conformant=False,
        reconstruction_capable=False,
        authoritative_for=(),
        executable_path=(f"/tools/{prover_id}" if discovered else None),
        executable_version=None,
        package_module=None,
        package_version=None,
        reason=("discovered only" if discovered else "not installed"),
    )


def _matrix(*, discovered: bool = False) -> ProverMatrixSnapshot:
    return ProverMatrixSnapshot(
        entries=(
            _matrix_entry("tla_tlc", discovered=discovered),
            _matrix_entry("apalache", discovered=discovered),
        ),
        generated_at="2026-08-24T00:00:00Z",
        duration_ms=0,
        self_tests_requested=False,
        bounded=True,
        max_self_tests=2,
        matrix_timeout_seconds=2.0,
        documentation_source=None,
    )


def test_exact_identity_is_typed_canonical_and_tamper_evident() -> None:
    identity = _identity()
    payload = identity.to_dict()

    assert identity.schema == CASF_FORMAL_IDENTITY_SCHEMA
    assert payload["source_revision"] == SOURCE_REVISION
    assert payload["source_tree"] == SOURCE_TREE
    assert payload["task_id"] == "CASF-036"
    assert payload["fencing_epoch"] == 7
    assert FederationFormalIdentity.from_dict(payload) == identity
    assert json.loads(json.dumps(payload)) == payload

    payload["assignment_revision"] = 4
    with pytest.raises(FederationFormalError, match="does not match"):
        FederationFormalIdentity.from_dict(payload)

    with pytest.raises(FederationFormalError, match="CASF-036"):
        FederationFormalIdentity(
            **{
                **_identity().to_dict(include_identity=False),
                "task_id": "CASF-035",
            }
        )

    with pytest.raises(FederationFormalError, match="40-hex"):
        FederationFormalIdentity(
            **{
                **_identity().to_dict(include_identity=False),
                "source_tree": "current-tree",
            }
        )


def test_identity_wire_decoder_rejects_unknown_authority_shaped_fields() -> None:
    payload = _identity().to_dict()
    payload["authority_created"] = True

    with pytest.raises(FederationFormalError, match="unknown fields"):
        FederationFormalIdentity.from_dict(payload)


def test_suite_reuses_transition_rules_bounds_and_tla_generator_deterministically() -> None:
    first = _suite()
    second = _suite()

    assert first.schema == CASF_FORMAL_SUITE_SCHEMA
    assert first.suite_id == second.suite_id
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["bounded"] is True
    assert first.to_dict()["unbounded_proof"] is False
    assert first.to_dict()["authority_created"] is False
    assert {item.property for item in first.scenarios} == set(FederationFormalProperty)

    for scenario in first.scenarios:
        assert scenario.generated_model.bounds == first.bounds
        assert scenario.transition_schema.source_identity == first.identity.identity
        assert all(
            isinstance(rule, TransitionRule) for rule in scenario.transition_schema.transitions
        )
        assert scenario.generated_model.transition_schema == scenario.transition_schema
        assert scenario.transition_schema.schema_identity in scenario.generated_model.model_text
        assert "Finite explored bounds" in scenario.generated_model.model_text
        assert scenario.generated_model.to_dict()["unbounded_proof"] is False
        assert scenario.subject_ids["task"] == "CASF-036"
        assert scenario.subject_ids["attempt"] == first.identity.attempt_id
        assert scenario.subject_ids["lease"] == first.identity.lease_id


def test_lifecycle_model_is_derived_from_the_closed_runtime_transition_table() -> None:
    scenario = _suite().scenario(FederationFormalProperty.LIFECYCLE)
    modeled = {
        (source, target)
        for rule in scenario.transition_schema.transitions
        if rule.metadata["operation"] == "lifecycle"
        for source in rule.source_states
        for target in (rule.target_state,)
    }
    runtime = {
        (source.value, target.value)
        for source in FederationLifecycleState
        for target in legal_transitions(source)
    }

    assert modeled == runtime
    assert FederationLifecycleState.STOPPED.value in scenario.transition_schema.terminal_states
    assert "LifecycleDeclaredToCompleted" not in {
        rule.name for rule in scenario.transition_schema.transitions
    }


def test_all_six_hermetic_models_pass_only_as_bounded_evidence() -> None:
    suite = _suite()
    receipts = check_federation_formal_suite(suite)

    assert len(receipts) == 6
    assert {item.property for item in receipts} == set(FederationFormalProperty)
    assert all(item.status is HermeticCheckStatus.PASSED for item in receipts)
    assert all(item.passed and item.goal_reached for item in receipts)
    assert all(item.explored_states > 0 for item in receipts)
    assert all(item.to_dict()["bounded"] is True for item in receipts)
    assert all(item.to_dict()["unbounded_proof"] is False for item in receipts)
    assert all(item.to_dict()["authority_created"] is False for item in receipts)
    assert all(item.schema == CASF_FORMAL_RECEIPT_SCHEMA for item in receipts)
    assert all(item.counterexample is None for item in receipts)
    assert all(item.receipt_id == item.to_dict()["receipt_id"] for item in receipts)
    assert json.loads(json.dumps([item.to_dict() for item in receipts])) == [
        item.to_dict() for item in receipts
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_invariant", "expected_transition"),
    (
        (
            AdversarialMutation.DUPLICATE_EVENT_EFFECT,
            "IdempotentAuthoritativeEffect",
            "ApplyEffect",
        ),
        (
            AdversarialMutation.STALE_FENCE_MUTATION,
            "CurrentFenceWins",
            "AdversaryStaleFenceMutation",
        ),
        (
            AdversarialMutation.ILLEGAL_LIFECYCLE_TRANSITION,
            "LegalLifecycleTransition",
            "AdversaryDeclaredToCompleted",
        ),
        (
            AdversarialMutation.DUAL_SHARD_OWNER,
            "UniqueShardOwner",
            "AdversaryDualShardOwner",
        ),
        (
            AdversarialMutation.MINT_BUDGET,
            "BudgetConservation",
            "AdversaryMintBudget",
        ),
        (
            AdversarialMutation.ORPHAN_CAUSAL_PROPAGATION,
            "CausalParentBeforeDependent",
            "AdversaryOrphanCausalPropagation",
        ),
    ),
)
def test_adversarial_mutations_produce_exact_counterexample_traces(
    mutation: AdversarialMutation,
    expected_invariant: str,
    expected_transition: str,
) -> None:
    suite = _suite()
    property_ = ADVERSARIAL_PROPERTY[mutation]
    receipt = check_federation_scenario(
        suite.scenario(property_),
        mutation=mutation,
    )

    assert receipt.status is HermeticCheckStatus.COUNTEREXAMPLE
    assert not receipt.passed
    assert receipt.counterexample is not None
    assert receipt.counterexample.invariant == expected_invariant
    assert receipt.counterexample.mutation is mutation
    assert receipt.counterexample.trace[-1].transition == expected_transition
    assert receipt.counterexample.trace[-1].after_state_id == (
        receipt.counterexample.state.state_id
    )
    assert (
        receipt.counterexample.counterexample_id
        == (receipt.counterexample.to_dict()["counterexample_id"])
    )
    assert receipt.to_dict()["unbounded_proof"] is False


def test_mutation_cannot_be_applied_to_an_unrelated_scenario() -> None:
    with pytest.raises(FederationFormalError, match="different property"):
        check_federation_scenario(
            _suite().scenario(FederationFormalProperty.BUDGET_CONSERVATION),
            mutation=AdversarialMutation.DUAL_SHARD_OWNER,
        )


def test_insufficient_reachability_bound_is_inconclusive_never_passed() -> None:
    suite = build_federation_formal_suite(_identity(), bounds=_bounds(max_steps=1))
    scenario = suite.scenario(FederationFormalProperty.CAUSAL_PROPAGATION)
    receipt = check_federation_scenario(scenario)

    assert receipt.status is HermeticCheckStatus.INCONCLUSIVE
    assert not receipt.passed
    assert not receipt.goal_reached
    assert "not reachable" in receipt.reason


def test_fence_identity_must_fit_the_recorded_finite_bound() -> None:
    with pytest.raises(FederationFormalError, match="max_fence"):
        build_federation_formal_suite(
            _identity(),
            bounds=ModelCheckBounds(
                max_steps=12,
                max_retries=1,
                max_fence=8,
                max_tasks=2,
                max_agents=2,
                max_states=16,
                max_transitions=64,
                max_evidence_ids=1,
            ),
        )


def test_all_external_matrix_entries_claim_only_the_configured_generic_invariant() -> None:
    suite = _suite()
    receipts = run_external_model_checks(suite, matrix=_matrix())

    assert {
        (item.scenario_property, item.tool) for item in receipts
    } == {
        (property_, tool)
        for property_ in FederationFormalProperty
        for tool in (ModelCheckerTool.TLC, ModelCheckerTool.APALACHE)
    }
    for receipt in receipts:
        scenario = suite.scenario(receipt.scenario_property)
        model = scenario.generated_model
        configuration = model.configuration_for(receipt.tool)
        casf_invariant = CASF_INVARIANT_NAMES[receipt.scenario_property]

        assert receipt.property is ExternalModelInvariant.SUPERVISOR_SAFETY
        assert receipt.to_dict()["property"] == "Safety"
        assert receipt.to_dict()["property_scope"] == "generic_supervisor_state_model"
        assert receipt.to_dict()["external_model_satisfies_casf_property_alone"] is False
        assert model.model_text.count("\nSafety ==\n") == 1
        assert configuration.splitlines().count("INVARIANT Safety") == 1
        assert f"\n{casf_invariant} ==\n" not in model.model_text
        assert f"INVARIANT {casf_invariant}" not in configuration.splitlines()
        assert not receipt.casf_property_satisfied_by_pair
        assert not receipt.casf_property_passed


def test_absent_tlc_and_apalache_are_unavailable_and_never_run_or_pass() -> None:
    receipts = run_external_model_checks(_suite(), matrix=_matrix())

    assert len(receipts) == 12
    assert {item.tool for item in receipts} == {
        ModelCheckerTool.TLC,
        ModelCheckerTool.APALACHE,
    }
    assert all(item.status is ExternalCheckStatus.UNAVAILABLE for item in receipts)
    assert all(not item.ran and not item.passed for item in receipts)
    assert all(not item.model_check_receipt_id for item in receipts)
    assert all(not item.paired_hermetic_receipt_id for item in receipts)
    assert all(item.paired_hermetic_status is None for item in receipts)
    assert all(not item.casf_property_satisfied_by_pair for item in receipts)
    assert all(item.to_dict()["unbounded_proof"] is False for item in receipts)


def test_discovery_without_matrix_self_test_is_not_run_and_never_passed() -> None:
    receipts = run_external_model_checks(_suite(), matrix=_matrix(discovered=True))

    assert len(receipts) == 12
    assert all(item.status is ExternalCheckStatus.NOT_RUN for item in receipts)
    assert all(not item.ran and not item.passed for item in receipts)
    assert all(not item.casf_property_satisfied_by_pair for item in receipts)
    assert all(item.matrix_entry_state == "discovered" for item in receipts)
    assert all("smoke-tested" in item.reason for item in receipts)


class _MatrixRuntime:
    def __init__(self) -> None:
        self.requests: list[CommandRequest] = []

    @staticmethod
    def which(name: str) -> str:
        del name
        return "/bin/true"

    def run(self, request: CommandRequest) -> CommandResult:
        self.requests.append(request)
        if "--version" in request.command or "version" in request.command:
            return CommandResult(0, stdout="checker 1.0\n")
        if "check" in request.command:
            return CommandResult(0, stdout="verification result: pass\n")
        return CommandResult(
            0,
            stdout="model checking completed. no error has been found.\n",
        )


def _qualified_matrix(runtime: _MatrixRuntime) -> ProverMatrixSnapshot:
    definitions = tuple(
        item for item in DEFAULT_PROVER_DEFINITIONS if item.prover_id in {"tla_tlc", "apalache"}
    )
    return ProverMatrixRegistry(
        definitions,
        config=ProverMatrixProbeConfig(
            run_self_tests=True,
            max_self_tests=2,
            documentation_path="does-not-exist.md",
        ),
        which=runtime.which,
        find_spec=lambda _name: None,
        distribution_version=lambda _name: "",
        command_runner=runtime.run,
    ).probe()


def test_external_pass_requires_both_qualified_matrix_and_executed_receipt() -> None:
    matrix_runtime = _MatrixRuntime()
    matrix = _qualified_matrix(matrix_runtime)
    assert all(entry.smoke_tested for entry in matrix.entries)
    assert all(entry.translation_conformant for entry in matrix.entries)
    assert all("bounded_state_machine" in entry.authoritative_for for entry in matrix.entries)

    checker_runtime = _MatrixRuntime()
    checker = SupervisorStateModelChecker(command_runner=checker_runtime.run)
    receipts = run_external_model_checks(
        _suite(),
        matrix=matrix,
        checker=checker,
    )

    assert len(receipts) == 12
    assert all(item.status is ExternalCheckStatus.PASSED for item in receipts)
    assert all(item.ran and item.passed for item in receipts)
    assert all(
        item.property is ExternalModelInvariant.SUPERVISOR_SAFETY for item in receipts
    )
    assert {item.scenario_property for item in receipts} == set(FederationFormalProperty)
    assert all(item.model_check_receipt_id for item in receipts)
    assert all(not item.paired_hermetic_receipt_id for item in receipts)
    assert all(item.paired_hermetic_status is None for item in receipts)
    assert all(not item.casf_property_satisfied_by_pair for item in receipts)
    assert all(item.to_dict()["unbounded_proof"] is False for item in receipts)
    assert len(checker_runtime.requests) == 24  # version plus bounded check per result


def test_casf_property_satisfaction_requires_exact_trusted_hermetic_pairing() -> None:
    suite = _suite()
    hermetic_receipts = check_federation_formal_suite(suite)
    checker_runtime = _MatrixRuntime()
    receipts = run_external_model_checks(
        suite,
        matrix=_qualified_matrix(_MatrixRuntime()),
        checker=SupervisorStateModelChecker(command_runner=checker_runtime.run),
        hermetic_receipts=hermetic_receipts,
    )
    hermetic_by_property = {item.property: item for item in hermetic_receipts}

    assert len(receipts) == 12
    assert all(
        item.passed
        and item.casf_property_satisfied_by_pair
        and item.casf_property_passed
        for item in receipts
    )
    assert all(
        item.paired_hermetic_receipt_id
        == hermetic_by_property[item.scenario_property].receipt_id
        for item in receipts
    )
    assert all(
        item.paired_hermetic_status
        is hermetic_by_property[item.scenario_property].status
        for item in receipts
    )
    assert all(item.property is ExternalModelInvariant.SUPERVISOR_SAFETY for item in receipts)

    unavailable = run_external_model_checks(
        suite,
        matrix=_matrix(),
        hermetic_receipts=hermetic_receipts,
    )
    assert all(item.paired_hermetic_receipt_id for item in unavailable)
    assert all(item.paired_hermetic_status is HermeticCheckStatus.PASSED for item in unavailable)
    assert all(not item.passed for item in unavailable)
    assert all(not item.casf_property_satisfied_by_pair for item in unavailable)

    forged = replace(
        hermetic_receipts[0],
        explored_states=hermetic_receipts[0].explored_states + 1,
    )
    with pytest.raises(FederationFormalError, match="trusted finite exploration"):
        run_external_model_checks(
            suite,
            matrix=_qualified_matrix(_MatrixRuntime()),
            checker=SupervisorStateModelChecker(command_runner=_MatrixRuntime().run),
            hermetic_receipts=(forged,),
        )


def test_generic_external_pass_cannot_promote_inconclusive_hermetic_pair() -> None:
    suite = build_federation_formal_suite(_identity(), bounds=_bounds(max_steps=1))
    hermetic_receipts = check_federation_formal_suite(suite)
    hermetic_by_property = {item.property: item for item in hermetic_receipts}
    inconclusive_properties = {
        item.property
        for item in hermetic_receipts
        if item.status is HermeticCheckStatus.INCONCLUSIVE
    }
    assert inconclusive_properties

    receipts = run_external_model_checks(
        suite,
        matrix=_qualified_matrix(_MatrixRuntime()),
        checker=SupervisorStateModelChecker(command_runner=_MatrixRuntime().run),
        hermetic_receipts=hermetic_receipts,
    )

    assert all(item.passed for item in receipts)
    for receipt in receipts:
        paired = hermetic_by_property[receipt.scenario_property]
        assert receipt.paired_hermetic_receipt_id == paired.receipt_id
        assert receipt.paired_hermetic_status is paired.status
        assert receipt.casf_property_satisfied_by_pair is paired.passed
        if receipt.scenario_property in inconclusive_properties:
            assert not receipt.casf_property_passed

    inconclusive_external = next(
        item for item in receipts if item.scenario_property in inconclusive_properties
    )
    with pytest.raises(FederationFormalError, match="exact passed hermetic receipt"):
        replace(
            inconclusive_external,
            casf_property_satisfied_by_pair=True,
        )

    inconclusive_hermetic = next(
        item
        for item in hermetic_receipts
        if item.status is HermeticCheckStatus.INCONCLUSIVE
    )
    forged_pass = replace(inconclusive_hermetic, status=HermeticCheckStatus.PASSED)
    with pytest.raises(FederationFormalError, match="trusted finite exploration"):
        run_external_model_checks(
            suite,
            matrix=_qualified_matrix(_MatrixRuntime()),
            checker=SupervisorStateModelChecker(command_runner=_MatrixRuntime().run),
            hermetic_receipts=(forged_pass,),
        )


def test_counterexample_hermetic_receipt_cannot_be_used_as_a_successful_pair() -> None:
    suite = _suite()
    counterexample = check_federation_scenario(
        suite.scenario(FederationFormalProperty.EVENT_DELIVERY),
        mutation=AdversarialMutation.DUPLICATE_EVENT_EFFECT,
    )
    assert counterexample.status is HermeticCheckStatus.COUNTEREXAMPLE

    with pytest.raises(FederationFormalError, match="trusted finite exploration"):
        run_external_model_checks(
            suite,
            matrix=_qualified_matrix(_MatrixRuntime()),
            checker=SupervisorStateModelChecker(command_runner=_MatrixRuntime().run),
            hermetic_receipts=(counterexample,),
        )


def test_checker_runtime_unavailability_cannot_be_promoted_by_matrix_capability() -> None:
    matrix = _qualified_matrix(_MatrixRuntime())
    checker = SupervisorStateModelChecker(
        command_runner=lambda _request: CommandResult(
            returncode=None,
            error="executable disappeared",
        )
    )
    receipts = run_external_model_checks(_suite(), matrix=matrix, checker=checker)

    assert all(item.status is ExternalCheckStatus.ERROR for item in receipts)
    assert all(item.ran for item in receipts)
    assert all(not item.passed for item in receipts)
    assert all(not item.casf_property_satisfied_by_pair for item in receipts)


def test_checker_subclass_cannot_replace_the_trusted_execution_boundary() -> None:
    class ExplodingChecker(SupervisorStateModelChecker):
        def check(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise RuntimeError("execution boundary failed")

    with pytest.raises(FederationFormalError, match="exact SupervisorStateModelChecker"):
        run_external_model_checks(
            _suite(),
            matrix=_qualified_matrix(_MatrixRuntime()),
            checker=ExplodingChecker(),
        )


def test_subclass_cannot_promote_a_dataclass_replace_forgery() -> None:
    class ForgingChecker(SupervisorStateModelChecker):
        def check(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            receipt = super().check(*args, **kwargs)
            return replace(
                receipt,
                status=ModelCheckStatus.PASSED,
                executable="",
                tool_version="",
                version_returncode=None,
                version_stdout="",
                version_stderr="",
                command=(),
                version_command=(),
                returncode=None,
                stdout="",
                stderr="",
                checked_safety_properties=(),
                checked_liveness_properties=(),
                counterexample=None,
            )

    matrix = _qualified_matrix(_MatrixRuntime())
    checker = ForgingChecker(command_runner=_MatrixRuntime().run)
    scenario = _suite().scenarios[0]
    forged = checker.check(
        scenario.generated_model,
        tool=ModelCheckerTool.TLC,
        executable=matrix.entry("tla_tlc").executable_path,
    )
    assert forged.passed
    assert not forged.executable
    assert not forged.version_command and not forged.command
    assert forged.version_returncode is None and forged.returncode is None
    assert not forged.checked_safety_properties
    assert not forged.checked_liveness_properties

    with pytest.raises(FederationFormalError, match="exact SupervisorStateModelChecker"):
        run_external_model_checks(_suite(), matrix=matrix, checker=checker)


@pytest.mark.parametrize(
    "malformation",
    (
        "empty_execution_evidence",
        "wrong_executable",
        "empty_commands",
        "failed_version",
        "failed_check",
        "missing_properties",
        "status_inconsistent_output",
        "version_drift",
        "counterexample_on_pass",
    ),
)
def test_malformed_passed_receipts_are_fail_closed_as_not_run(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    trusted_check = formal_models._TRUSTED_MODEL_CHECK

    def malformed_check(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        receipt = trusted_check(self, *args, **kwargs)
        if malformation == "empty_execution_evidence":
            changes = {
                "executable": "",
                "tool_version": "",
                "version_returncode": None,
                "version_stdout": "",
                "version_stderr": "",
                "command": (),
                "version_command": (),
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "checked_safety_properties": (),
                "checked_liveness_properties": (),
            }
        elif malformation == "wrong_executable":
            changes = {"executable": "/tools/forged-checker"}
        elif malformation == "empty_commands":
            changes = {"command": (), "version_command": ()}
        elif malformation == "failed_version":
            changes = {"version_returncode": 1}
        elif malformation == "failed_check":
            changes = {"returncode": 1}
        elif malformation == "missing_properties":
            changes = {
                "checked_safety_properties": (),
                "checked_liveness_properties": (),
            }
        elif malformation == "status_inconsistent_output":
            changes = {"stdout": "ambiguous checker output\n", "stderr": ""}
        elif malformation == "version_drift":
            changes = {
                "tool_version": "forged checker 9.0",
                "version_stdout": "forged checker 9.0\n",
            }
        else:
            changes = {"counterexample": CounterexampleTrace(raw="forged trace")}
        return replace(receipt, **changes)

    monkeypatch.setattr(formal_models, "_TRUSTED_MODEL_CHECK", malformed_check)
    receipts = run_external_model_checks(
        _suite(),
        matrix=_qualified_matrix(_MatrixRuntime()),
        checker=SupervisorStateModelChecker(command_runner=_MatrixRuntime().run),
    )

    assert len(receipts) == 12
    assert all(item.status is ExternalCheckStatus.NOT_RUN for item in receipts)
    assert all(not item.ran and not item.passed for item in receipts)
    assert all(not item.model_check_receipt_id for item in receipts)
    assert all("valid execution receipt" in item.reason for item in receipts)


def test_external_checker_must_be_the_canonical_typed_checker() -> None:
    with pytest.raises(FederationFormalError, match="SupervisorStateModelChecker"):
        run_external_model_checks(
            _suite(),
            matrix=_qualified_matrix(_MatrixRuntime()),
            checker=object(),  # type: ignore[arg-type]
        )
