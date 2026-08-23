from __future__ import annotations

import copy

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ConditionOperator,
    EffectClass,
    ProcedureAuthorityEnvelope,
    ProcedureBranch,
    ProcedureContractError,
    ProcedureEffect,
    ProcedureLocal,
    ProcedureObservation,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    RiskClass,
    StepOperation,
    ValueType,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.procedure_ir import (
    ProcedureDataflowError,
    ProcedureEffectError,
    ProcedureGraphError,
    ProcedureIRParser,
    ProcedureIRValidationError,
    ProcedureScopeError,
    ProcedureValidationRetentionError,
    parse_procedure_spec,
    validate_procedure_spec,
)


def valid_spec() -> ProcedureSpec:
    bindings = ArtifactBindings(
        repository_id="repo-main",
        repository_commit="abc123",
        tree_id="tree-abc123",
        objective_id="PCPC-G000",
        task_id="PCPC-004",
        contract_revision="procedure-contracts-v1",
        policy_revision="authority-policy-v1",
        environment_id="python312-linux-lock1",
    )
    precondition = ProcedurePrecondition(
        condition_id="precondition.current-tree",
        binding="binding:tree_id",
        operator=ConditionOperator.CURRENT,
        evidence_producer="tree-verifier@1",
        evidence_type="current-tree-receipt@1",
    )
    postcondition = ProcedurePostcondition(
        condition_id="postcondition.tests-admitted",
        binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
        evidence_producer="postcondition-checker@1",
        evidence_type="postcondition-receipt@1",
    )
    test_observation = ProcedureObservation(
        observation_id="observation.tests",
        producer_contract="test-runner@1",
        output_binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
        evidence_type="test-receipt@1",
    )
    post_observation = ProcedureObservation(
        observation_id="observation.postcondition",
        producer_contract="postcondition-checker@1",
        output_binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
        evidence_type="postcondition-receipt@1",
    )
    validation_effect = ProcedureEffect(
        effect_id="effect.validation", effect_class=EffectClass.VALIDATION
    )
    receipt_effect = ProcedureEffect(
        effect_id="effect.receipt", effect_class=EffectClass.RECEIPT_EMIT
    )
    read = ProcedureStep(
        step_id="read",
        operation=StepOperation.READ_STATE,
        operation_contract="state-reader@1",
        output_bindings={"state": "local:state"},
        next_step_id="tests",
    )
    tests = ProcedureStep(
        step_id="tests",
        operation=StepOperation.RUN_SELECTED_TESTS,
        operation_contract="test-runner@1",
        input_bindings={"state": "local:state"},
        output_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("observation.tests",),
        next_step_id="postcondition",
    )
    check = ProcedureStep(
        step_id="postcondition",
        operation=StepOperation.CHECK_POSTCONDITION,
        operation_contract="postcondition-checker@1",
        input_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("observation.postcondition",),
        next_step_id="receipt",
    )
    receipt = ProcedureStep(
        step_id="receipt",
        operation=StepOperation.EMIT_RECEIPT,
        operation_contract="receipt-emitter@1",
        declared_effect_ids=("effect.receipt",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("receipt.execution",),
    )
    authority = ProcedureAuthorityEnvelope(
        authority_policy_revision="authority-policy-v1",
        requirement_ids=("authority.execute",),
        required_capability_ids=("capability.tests",),
        allowed_operations=tuple(step.operation for step in (read, tests, check, receipt)),
        risk_ceiling=RiskClass.OBSERVATION_ONLY,
    )
    resources = ProcedureResourceEnvelope(
        wall_time_ms=60_000,
        cpu_time_ms=60_000,
        memory_bytes=128_000_000,
        disk_bytes=128_000_000,
        model_token_limit=0,
        model_call_limit=0,
        subprocess_limit=4,
    )
    validation = ProcedureValidationPlan(
        required_step_ids=("tests", "postcondition"),
        required_observation_ids=("observation.tests", "observation.postcondition"),
        required_test_contracts=("focused-tests@1",),
    )
    return ProcedureSpec(
        bindings=bindings,
        name="focused-validation-procedure",
        version=ProcedureVersion(major=1),
        task_family_id="IMPORT_PURITY_REPAIR",
        entry_step_id="read",
        locals=(
            ProcedureLocal("state", ValueType.STRUCTURED),
            ProcedureLocal("test-result", ValueType.STRUCTURED),
        ),
        preconditions=(precondition,),
        declared_reads=("ipfs_accelerate_py/agent_supervisor/example.py",),
        declared_effects=(validation_effect, receipt_effect),
        steps=(read, tests, check, receipt),
        postconditions=(postcondition,),
        observations=(test_observation, post_observation),
        validation=validation,
        authority=authority,
        resources=resources,
        terminal_step_ids=("receipt",),
        scope_paths=("ipfs_accelerate_py/agent_supervisor",),
        provenance_cids=("accepted-trajectory-cid",),
    )


def decode_modified(spec: ProcedureSpec, mutate: object) -> ProcedureSpec:
    payload = copy.deepcopy(spec.to_dict())
    mutate(payload)  # type: ignore[operator]
    return ProcedureSpec.from_dict(payload)


def test_parser_round_trip_is_deterministic_for_mapping_json_and_bytes() -> None:
    spec = valid_spec()
    assert ProcedureIRParser.parse(spec) is spec
    assert parse_procedure_spec(spec.to_dict()) == spec
    assert parse_procedure_spec(spec.to_json()) == spec
    assert parse_procedure_spec(spec.canonical_bytes()) == spec
    assert parse_procedure_spec(spec.to_json()).content_id == spec.content_id


def test_parser_rejects_duplicate_json_fields_floats_and_unknown_normative_fields() -> None:
    with pytest.raises(ProcedureIRValidationError, match="duplicate"):
        parse_procedure_spec('{"schema":"x","schema":"y"}')
    with pytest.raises(ProcedureIRValidationError, match="floating"):
        parse_procedure_spec('{"schema":"x","ratio":1.5}')

    payload = valid_spec().to_dict()
    payload["executable_policy"] = "never"
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        parse_procedure_spec(payload)


def test_parser_rejects_forbidden_and_unknown_operation_categories() -> None:
    payload = valid_spec().to_dict()
    payload["steps"][0]["operation"] = "ARBITRARY_SHELL"
    with pytest.raises(ProcedureContractError):
        parse_procedure_spec(payload)
    payload["steps"][0]["operation"] = "SOME_FUTURE_OPERATION"
    with pytest.raises(ProcedureContractError):
        parse_procedure_spec(payload)


def test_graph_rejects_unreachable_step() -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["steps"].append(  # type: ignore[union-attr]
            ProcedureStep("orphan", StepOperation.READ_STATE, "state-reader@1").to_dict()
        )

    with pytest.raises(ProcedureGraphError, match="unreachable"):
        validate_procedure_spec(decode_modified(valid_spec(), mutate))


def test_graph_rejects_cycle_without_a_declared_bounded_loop() -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["steps"][3]["next_step_id"] = "read"  # type: ignore[index]

    with pytest.raises(ProcedureGraphError):
        validate_procedure_spec(decode_modified(valid_spec(), mutate))


def test_dataflow_rejects_undeclared_and_not_definitely_initialized_locals() -> None:
    def undeclared(payload: dict[str, object]) -> None:
        payload["steps"][1]["input_bindings"] = {"state": "local:missing"}  # type: ignore[index]

    with pytest.raises(ProcedureDataflowError, match="undeclared"):
        validate_procedure_spec(decode_modified(valid_spec(), undeclared))

    def branch_before_initialization(payload: dict[str, object]) -> None:
        branch = ProcedureBranch(
            branch_id="branch",
            observation_id="observation.tests",
            true_step_id="read",
            false_step_id="receipt",
        )
        payload["branches"] = [branch.to_dict()]
        payload["entry_step_id"] = "branch"

    with pytest.raises(ProcedureDataflowError, match="definite initialization"):
        validate_procedure_spec(decode_modified(valid_spec(), branch_before_initialization))


def test_effect_class_must_match_operation_and_have_authority_and_evidence() -> None:
    def wrong_effect(payload: dict[str, object]) -> None:
        payload["steps"][1]["operation"] = StepOperation.APPLY_APPROVED_PATCH_TEMPLATE.value  # type: ignore[index]

    with pytest.raises(ProcedureEffectError):
        validate_procedure_spec(decode_modified(valid_spec(), wrong_effect))

    def no_authority(payload: dict[str, object]) -> None:
        payload["steps"][1]["required_authority_ids"] = []  # type: ignore[index]

    with pytest.raises(ProcedureEffectError, match="authority"):
        validate_procedure_spec(decode_modified(valid_spec(), no_authority))

    def no_evidence(payload: dict[str, object]) -> None:
        payload["steps"][1]["evidence_outputs"] = []  # type: ignore[index]

    with pytest.raises(ProcedureIRValidationError):
        validate_procedure_spec(decode_modified(valid_spec(), no_evidence))


def test_scope_validation_rejects_declared_read_and_effect_outside_scope() -> None:
    def read_escape(payload: dict[str, object]) -> None:
        payload["declared_reads"] = ["docs/outside.md"]

    with pytest.raises(ProcedureScopeError):
        validate_procedure_spec(decode_modified(valid_spec(), read_escape))

    spec = valid_spec()
    write = ProcedureEffect(
        "effect.write", EffectClass.REPOSITORY_WRITE, targets=("docs/outside.md",)
    )
    payload = spec.to_dict()
    payload["declared_effects"].append(write.to_dict())
    patch_step = ProcedureStep(
        "patch",
        StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
        "approved-patch-template@1",
        declared_effect_ids=("effect.write",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("patch-receipt",),
        next_step_id="tests",
    )
    payload["steps"].append(patch_step.to_dict())
    payload["steps"][0]["next_step_id"] = "patch"
    payload["authority"]["allowed_operations"].append(
        StepOperation.APPLY_APPROVED_PATCH_TEMPLATE.value
    )
    with pytest.raises(ProcedureScopeError):
        validate_procedure_spec(ProcedureSpec.from_dict(payload))


def test_scope_glob_is_segment_aware_and_double_star_is_recursive() -> None:
    def shallow_scope(payload: dict[str, object]) -> None:
        payload["scope_paths"] = ["ipfs_accelerate_py/*"]

    with pytest.raises(ProcedureScopeError):
        validate_procedure_spec(decode_modified(valid_spec(), shallow_scope))

    def recursive_scope(payload: dict[str, object]) -> None:
        payload["scope_paths"] = ["ipfs_accelerate_py/**"]

    recursive = decode_modified(valid_spec(), recursive_scope)
    assert validate_procedure_spec(recursive) is recursive


def test_branch_cannot_hide_required_validation() -> None:
    spec = valid_spec()
    payload = spec.to_dict()
    payload["observations"].append(
        ProcedureObservation(
            observation_id="observation.branch",
            producer_contract="state-reader@1",
            output_binding="local:state",
            operator=ConditionOperator.EXISTS,
            evidence_type="state-observation@1",
        ).to_dict()
    )
    payload["steps"][0]["evidence_outputs"] = ["observation.branch"]
    payload["steps"][0]["next_step_id"] = "branch"
    payload["branches"] = [
        ProcedureBranch(
            branch_id="branch",
            observation_id="observation.branch",
            true_step_id="tests",
            false_step_id="receipt",
        ).to_dict()
    ]
    with pytest.raises(ProcedureValidationRetentionError, match="bypasses"):
        validate_procedure_spec(ProcedureSpec.from_dict(payload))


def test_validation_plan_cannot_replace_postcondition_gate_with_only_a_cheap_check() -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["validation"]["required_step_ids"] = ["tests"]  # type: ignore[index]

    with pytest.raises(ProcedureValidationRetentionError, match="postcondition"):
        validate_procedure_spec(decode_modified(valid_spec(), mutate))


def test_required_observation_must_come_from_declared_external_producer() -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["observations"][0]["producer_contract"] = "forged-producer@1"  # type: ignore[index]

    with pytest.raises(ProcedureIRValidationError, match="external producer"):
        validate_procedure_spec(decode_modified(valid_spec(), mutate))


def test_stale_authority_policy_binding_is_rejected() -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["authority"]["authority_policy_revision"] = "old-policy-v0"  # type: ignore[index]

    with pytest.raises(ProcedureEffectError, match="stale"):
        validate_procedure_spec(decode_modified(valid_spec(), mutate))
