from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.anti_unification import (
    FORBIDDEN_GENERALIZATIONS,
    UNIFIER_REVISION,
    AntiUnificationError,
    AntiUnificationPattern,
    LostDetailDisposition,
    LostDetailKind,
    PatternStatus,
    ProcedureAntiUnifier,
    StepPresence,
    UnsafeMergeClass,
    anti_unify,
    anti_unify_procedures,
    anti_unify_trajectories,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    EpisodeKind,
    ExecutionTrajectory,
    FailureTransition,
    FamilyMembershipClass,
    HoleType,
    IdempotencyClass,
    ProcedureAuthorityEnvelope,
    ProcedureEffect,
    ProcedureLocal,
    ProcedureObservation,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    RiskClass,
    StepOperation,
    TaskFamily,
    TaskFamilyBoundary,
    TaskFamilyMembership,
    TraceEventStatus,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
    ValueType,
    parse_procedure_artifact,
)


def bindings(**changes: object) -> ArtifactBindings:
    values: dict[str, object] = {
        "repository_id": "repo-main",
        "repository_commit": "abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G000",
        "task_id": "PCPC-012",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    values.update(changes)
    return ArtifactBindings(**values)


def family(**changes: object) -> TaskFamily:
    values: dict[str, object] = {
        "bindings": bindings(),
        "name": "IMPORT_PURITY_REPAIR",
        "goal_semantics": ("restore-import-purity",),
        "precondition_shape": ("import-side-effect-observed",),
        "affected_artifact_classes": ("python-source",),
        "effect_classes": (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
        "required_operation_contracts": ("approved-patch-template@1", "test-runner@1"),
        "validation_structure": ("focused-tests", "postcondition-check"),
        "failure_signatures": ("import-side-effect",),
        "postcondition_shape": ("import-is-pure",),
        "rollback_structure": ("restore-exact-tree",),
        "boundary": TaskFamilyBoundary(
            positive_member_cids=("positive-a",),
            negative_example_cids=("negative-a",),
            boundary_example_cids=("boundary-a",),
            unknown_case_cids=("unknown-a",),
            risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
            permitted_repositories=("repo-main",),
            permitted_languages=("python",),
            permitted_frameworks=("pytest",),
            permitted_effect_classes=(
                EffectClass.REPOSITORY_WRITE,
                EffectClass.VALIDATION,
                EffectClass.OBSERVE,
                EffectClass.RECEIPT_EMIT,
                EffectClass.MODEL_REQUEST,
                EffectClass.ROLLBACK,
            ),
        ),
    }
    values.update(changes)
    return TaskFamily(**values)


def _step(
    sequence: int,
    operation: StepOperation,
    contract: str,
    *,
    initial: str,
    terminal: str,
    effects: tuple[str, ...] = (),
    validation: tuple[str, ...] = (),
    observations: tuple[str, ...] = (),
    hole_type: str = "",
    status: TraceEventStatus = TraceEventStatus.SUCCEEDED,
) -> TrajectoryStep:
    return TrajectoryStep(
        sequence=sequence,
        operation=operation,
        operation_contract=contract,
        initial_state_cid=initial,
        terminal_state_cid=terminal,
        observation_cids=observations or (f"observation-{sequence}",),
        effect_ids=effects,
        validation_receipt_cids=validation,
        hole_type=hole_type,
        status=status,
    )


def _default_ops(
    *,
    patch_contract: str = "approved-patch-template@1",
    include_query: bool = False,
    include_tests: bool = True,
    include_scope: bool = False,
    include_hole: bool = False,
    hole_type: str = HoleType.CLASSIFY_FAILURE.value,
    invert_read_patch: bool = False,
    patch_effects: tuple[str, ...] = ("repository_write",),
) -> list[tuple[StepOperation, str, tuple[str, ...], tuple[str, ...], str]]:
    ops: list[tuple[StepOperation, str, tuple[str, ...], tuple[str, ...], str]] = []
    read = (StepOperation.READ_STATE, "state-reader@1", ("observe",), (), "")
    patch = (
        StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
        patch_contract,
        patch_effects,
        (),
        "",
    )
    if invert_read_patch:
        ops.extend((patch, read))
    else:
        ops.append(read)
        if include_query:
            ops.append(
                (StepOperation.QUERY_AST_INDEX, "ast-index@1", ("observe",), (), "")
            )
        if include_hole:
            ops.append(
                (
                    StepOperation.REQUEST_TYPED_MODEL_HOLE,
                    "typed-hole-service@1",
                    ("model-request",),
                    ("hole-validation",),
                    hole_type,
                )
            )
        ops.append(patch)
    if include_scope:
        ops.append((StepOperation.CHECK_SCOPE, "scope-checker@1", ("validation",), ("scope-receipt",), ""))
    if include_tests:
        ops.append(
            (
                StepOperation.RUN_SELECTED_TESTS,
                "test-runner@1",
                ("validation",),
                ("test-receipt",),
                "",
            )
        )
    ops.append(
        (
            StepOperation.CHECK_POSTCONDITION,
            "postcondition-checker@1",
            ("validation",),
            ("postcondition-receipt",),
            "",
        )
    )
    ops.append(
        (StepOperation.EMIT_RECEIPT, "receipt-emitter@1", ("receipt_emit",), (), "")
    )
    return ops


def trajectory(
    *,
    source_episode_cid: str = "accepted-receipt-a",
    status: TrajectoryTerminalStatus = TrajectoryTerminalStatus.ACCEPTED,
    accepted_criterion_ids: tuple[str, ...] = ("criterion-a",),
    include_query: bool = False,
    include_tests: bool = True,
    include_scope: bool = False,
    include_hole: bool = False,
    hole_type: str = HoleType.CLASSIFY_FAILURE.value,
    invert_read_patch: bool = False,
    patch_contract: str = "approved-patch-template@1",
    patch_effects: tuple[str, ...] = ("repository_write",),
    family_hint: str = "IMPORT_PURITY_REPAIR",
    bind: ArtifactBindings | None = None,
    failed_test: bool = False,
) -> ExecutionTrajectory:
    ops = _default_ops(
        patch_contract=patch_contract,
        include_query=include_query,
        include_tests=include_tests,
        include_scope=include_scope,
        include_hole=include_hole,
        hole_type=hole_type,
        invert_read_patch=invert_read_patch,
        patch_effects=patch_effects,
    )
    steps = []
    validation_cids: list[str] = []
    for index, (operation, contract, effects, validation, hole) in enumerate(ops):
        initial = f"state-{source_episode_cid}-{index}"
        terminal = f"state-{source_episode_cid}-{index + 1}"
        step_status = TraceEventStatus.SUCCEEDED
        if failed_test and operation is StepOperation.RUN_SELECTED_TESTS:
            step_status = TraceEventStatus.FAILED
        steps.append(
            _step(
                index,
                operation,
                contract,
                initial=initial,
                terminal=terminal,
                effects=effects,
                validation=validation,
                hole_type=hole,
                status=step_status,
            )
        )
        validation_cids.extend(validation)
    if not validation_cids:
        validation_cids.append("fallback-validation")
    rejection = ""
    if status is TrajectoryTerminalStatus.REJECTED:
        rejection = "typed-rejection"
    return ExecutionTrajectory(
        bindings=bind or bindings(),
        source_episode_cid=source_episode_cid,
        source_episode_kind=EpisodeKind.ACCEPTED_TASK_RECEIPT
        if status is TrajectoryTerminalStatus.ACCEPTED
        else EpisodeKind.FAILED_RECOVERED_EXECUTION
        if status is TrajectoryTerminalStatus.FAILED_RECOVERED
        else EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT
        if status is TrajectoryTerminalStatus.ROLLED_BACK
        else EpisodeKind.REJECTED_TASK_RECORD,
        initial_abstract_state_cid=steps[0].initial_state_cid,
        terminal_abstract_state_cid=steps[-1].terminal_state_cid,
        objective_criterion_ids=("criterion-a", "criterion-b"),
        task_family_hint=family_hint,
        steps=tuple(steps),
        outcome=TrajectoryOutcome(
            status=status,
            accepted_criterion_ids=accepted_criterion_ids
            if status is TrajectoryTerminalStatus.ACCEPTED
            else (),
            validation_receipt_cids=tuple(dict.fromkeys(validation_cids)),
            proof_receipt_cids=(),
            rejection_reason_code=rejection,
        ),
        total_cost_units=len(steps),
        total_tokens=0,
        total_latency_ms=10 * len(steps),
        human_interventions=0,
    )


def procedure_spec(
    *,
    name: str = "focused-validation-procedure",
    provenance: str = "accepted-trajectory-cid",
    write_target: str = "ipfs_accelerate_py/agent_supervisor/example.py",
    include_write: bool = True,
    authority_ids: tuple[str, ...] = ("authority.execute",),
    include_authority_check: bool = False,
    credential_binding: bool = False,
    patch_contract: str = "approved-patch-template@1",
    bind: ArtifactBindings | None = None,
) -> ProcedureSpec:
    used_bindings = bind or bindings()
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
    effects = [
        ProcedureEffect(effect_id="effect.validation", effect_class=EffectClass.VALIDATION),
        ProcedureEffect(effect_id="effect.receipt", effect_class=EffectClass.RECEIPT_EMIT),
    ]
    if include_write:
        effects.insert(
            0,
            ProcedureEffect(
                effect_id="effect.write",
                effect_class=EffectClass.REPOSITORY_WRITE,
                targets=(write_target,),
                reversible=True,
            ),
        )
    read_inputs = {"state": "local:state"}
    if credential_binding:
        read_inputs["api_key"] = "binding:redacted"
    steps: list[ProcedureStep] = [
        ProcedureStep(
            step_id="read",
            operation=StepOperation.READ_STATE,
            operation_contract="state-reader@1",
            output_bindings={"state": "local:state"},
            input_bindings=read_inputs if credential_binding else {},
            next_step_id="authority" if include_authority_check else ("patch" if include_write else "tests"),
            idempotency=IdempotencyClass.PURE,
        )
    ]
    if include_authority_check:
        steps.append(
            ProcedureStep(
                step_id="authority",
                operation=StepOperation.CHECK_AUTHORITY,
                operation_contract="authority-checker@1",
                required_authority_ids=authority_ids,
                next_step_id="patch" if include_write else "tests",
            )
        )
    if include_write:
        steps.append(
            ProcedureStep(
                step_id="patch",
                operation=StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
                operation_contract=patch_contract,
                input_bindings={"state": "local:state"},
                declared_effect_ids=("effect.write",),
                required_authority_ids=authority_ids,
                next_step_id="tests",
                idempotency=IdempotencyClass.IDEMPOTENCY_KEY_REQUIRED,
                failure_transition=FailureTransition.ROLLBACK,
                failure_target="rollback",
            )
        )
    tests = ProcedureStep(
        step_id="tests",
        operation=StepOperation.RUN_SELECTED_TESTS,
        operation_contract="test-runner@1",
        input_bindings={"state": "local:state"},
        output_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=authority_ids,
        evidence_outputs=("observation.tests",),
        next_step_id="postcondition",
    )
    check = ProcedureStep(
        step_id="postcondition",
        operation=StepOperation.CHECK_POSTCONDITION,
        operation_contract="postcondition-checker@1",
        input_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=authority_ids,
        evidence_outputs=("observation.postcondition",),
        next_step_id="receipt",
    )
    receipt = ProcedureStep(
        step_id="receipt",
        operation=StepOperation.EMIT_RECEIPT,
        operation_contract="receipt-emitter@1",
        declared_effect_ids=("effect.receipt",),
        required_authority_ids=authority_ids,
        evidence_outputs=("receipt.execution",),
    )
    steps.extend((tests, check, receipt))
    rollback = ()
    if include_write:
        rollback = (
            ProcedureRollback(
                rollback_id="rollback.restore-tree",
                trigger_effect_ids=("effect.write",),
                step_ids=("patch",),
                verification_observation_ids=("observation.tests",),
                exact_target_cid="tree-abc123",
            ),
        )
    authority = ProcedureAuthorityEnvelope(
        authority_policy_revision=used_bindings.policy_revision,
        requirement_ids=authority_ids,
        required_capability_ids=("capability.tests",),
        allowed_operations=tuple(step.operation for step in steps),
        risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
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
        required_proof_contracts=("scope-proof@1",),
    )
    return ProcedureSpec(
        bindings=used_bindings,
        name=name,
        version=ProcedureVersion(major=1),
        task_family_id="IMPORT_PURITY_REPAIR",
        entry_step_id="read",
        locals=(
            ProcedureLocal("state", ValueType.STRUCTURED),
            ProcedureLocal("test-result", ValueType.STRUCTURED),
        ),
        preconditions=(precondition,),
        declared_reads=(write_target,),
        declared_effects=tuple(effects),
        steps=tuple(steps),
        postconditions=(postcondition,),
        observations=(test_observation, post_observation),
        validation=validation,
        rollback=rollback,
        authority=authority,
        resources=resources,
        terminal_step_ids=("receipt",),
        scope_paths=("ipfs_accelerate_py/agent_supervisor",),
        provenance_cids=(provenance,),
    )


def _violations(result) -> set[str]:
    return {item.violation_class.value for item in result.counterexamples}


def test_identical_positive_traces_infer_constants_and_preserve_order() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(source_episode_cid="accepted-receipt-b")
    result = anti_unify_trajectories(family(), (left, right))

    assert result.admitted is True
    assert result.pattern.status is PatternStatus.CANDIDATE
    assert result.pattern.unifier_revision == UNIFIER_REVISION
    assert result.pattern.required_operations == (
        "READ_STATE",
        "APPLY_APPROVED_PATCH_TEMPLATE",
        "RUN_SELECTED_TESTS",
        "CHECK_POSTCONDITION",
        "EMIT_RECEIPT",
    )
    assert all(step.presence is StepPresence.REQUIRED for step in result.pattern.steps)
    assert result.pattern.optional_branches == ()
    assert result.pattern.parameters == ()
    assert result.pattern.holes == ()
    assert result.pattern.postconditions == ("criterion-a",)
    assert "RUN_SELECTED_TESTS" in result.pattern.validation_operations
    assert "CHECK_POSTCONDITION" in result.pattern.validation_operations
    assert result.retained_validations == result.pattern.validation_operations
    assert result.retained_postconditions == result.pattern.postconditions
    artifact = result.pattern_artifact
    assert artifact.state is ArtifactState.CANDIDATE
    decoded = parse_procedure_artifact(artifact.to_dict())
    assert decoded == artifact
    assert AntiUnificationPattern.from_artifact(decoded).required_operations == (
        result.pattern.required_operations
    )
    boundary = parse_procedure_artifact(result.boundary_artifact.to_dict())
    assert boundary.facts["forbidden_generalizations"] == FORBIDDEN_GENERALIZATIONS


def test_state_cid_differences_are_recorded_as_lost_instance_state() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(source_episode_cid="accepted-receipt-b")
    result = ProcedureAntiUnifier().anti_unify(family(), (left, right))
    kinds = {item.kind for item in result.lost_details}
    assert LostDetailKind.INSTANCE_STATE in kinds
    assert all(
        item.disposition is LostDetailDisposition.INSTANCE_STATE
        for item in result.lost_details
        if item.kind is LostDetailKind.INSTANCE_STATE
    )
    assert result.admitted is True


def test_differing_patch_contracts_become_closed_parameters() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        patch_contract="approved-patch-template@2",
    )
    result = anti_unify(family(), (left, right))
    assert result.admitted is True
    assert result.pattern.parameters
    names = {item.source_field for item in result.pattern.parameters}
    assert "operation_contract" in names
    allowed = result.pattern.parameters[0].allowed_values
    assert "approved-patch-template@1" in allowed
    assert "approved-patch-template@2" in allowed
    assert all(item.value_type is not ValueType.RELATIVE_PATH for item in result.pattern.parameters)
    assert any(item.kind is LostDetailKind.PARAMETER for item in result.lost_details)


def test_optional_query_becomes_a_bounded_optional_branch() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a", include_query=True)
    right = trajectory(source_episode_cid="accepted-receipt-b")
    result = anti_unify(family(), (left, right))
    assert result.admitted is True
    assert len(result.pattern.optional_branches) == 1
    branch = result.pattern.optional_branches[0]
    assert branch.operation is StepOperation.QUERY_AST_INDEX
    assert branch.predecessor_operation == "READ_STATE"
    assert branch.successor_operation == "APPLY_APPROVED_PATCH_TEMPLATE"
    assert any(item.kind is LostDetailKind.OPTIONAL_BRANCH for item in result.lost_details)
    assert "QUERY_AST_INDEX" not in result.pattern.required_operations
    assert "RUN_SELECTED_TESTS" in result.pattern.required_operations


def test_uncertain_model_hole_presence_becomes_a_typed_hole() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a", include_hole=True)
    right = trajectory(source_episode_cid="accepted-receipt-b")
    result = anti_unify(family(), (left, right))
    assert result.pattern.holes
    assert all(item.hole_type.value not in {"AUTHORITY_DECISION", "TEST_OMISSION"} for item in result.pattern.holes)
    assert any(item.presence is StepPresence.HOLE for item in result.pattern.steps)
    assert any(item.kind is LostDetailKind.TYPED_HOLE for item in result.lost_details)
    assert "RUN_SELECTED_TESTS" in result.retained_validations
    assert result.pattern.postconditions


def test_differing_allowed_hole_types_become_parameters_and_typed_holes() -> None:
    left = trajectory(
        source_episode_cid="accepted-receipt-a",
        include_hole=True,
        hole_type=HoleType.CLASSIFY_FAILURE.value,
    )
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        include_hole=True,
        hole_type=HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE.value,
    )
    result = anti_unify(family(), (left, right))
    assert result.admitted is True
    assert result.pattern.holes
    assert any(item.source_field.endswith("hole_type") for item in result.pattern.holes)
    assert any(item.source_field == "hole_type" for item in result.pattern.parameters)


def test_omitted_tests_produce_counterexamples_and_cannot_drop_validation() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        include_tests=False,
        include_scope=True,
    )
    result = anti_unify(family(), (left, right))
    assert result.admitted is False
    assert result.pattern.status is PatternStatus.REJECTED
    assert UnsafeMergeClass.OMITTED_TEST.value in _violations(result)
    assert "RUN_SELECTED_TESTS" in result.pattern.validation_operations
    assert "CHECK_POSTCONDITION" in result.pattern.validation_operations
    assert result.pattern_artifact.state is ArtifactState.REJECTED
    assert result.counterexample_artifacts
    decoded = parse_procedure_artifact(result.counterexample_artifacts[0].to_dict())
    assert decoded.state is ArtifactState.REJECTED
    assert "path" not in {item.source_field for item in result.pattern.parameters}


def test_missing_postconditions_produce_counterexamples_and_are_retained() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        accepted_criterion_ids=("criterion-a", "criterion-extra"),
    )
    result = anti_unify(family(), (left, right))
    assert result.admitted is False
    assert UnsafeMergeClass.MISSING_POSTCONDITION.value in _violations(result)
    assert "criterion-a" in result.pattern.postconditions
    assert "criterion-extra" in result.pattern.postconditions
    assert result.retained_postconditions == result.pattern.postconditions


def test_path_disagreements_are_never_generalized() -> None:
    left = procedure_spec(provenance="trace-a")
    right = procedure_spec(
        name="focused-validation-procedure-b",
        provenance="trace-b",
        write_target="ipfs_accelerate_py/agent_supervisor/other.py",
    )
    result = anti_unify_procedures(family(), (left, right))
    assert result.admitted is False
    assert UnsafeMergeClass.PATH_GENERALIZATION.value in _violations(result)
    assert all(item.value_type is not ValueType.RELATIVE_PATH for item in result.pattern.parameters)
    assert any(item.kind is LostDetailKind.PATH for item in result.lost_details)
    assert "ipfs_accelerate_py/agent_supervisor/example.py" not in {
        value
        for parameter in result.pattern.parameters
        for value in parameter.allowed_values
    }


def test_credentials_are_never_generalized() -> None:
    left = procedure_spec(provenance="trace-a", credential_binding=True)
    right = procedure_spec(
        name="focused-validation-procedure-b",
        provenance="trace-b",
        credential_binding=True,
    )
    result = anti_unify_procedures(family(), (left, right))
    assert result.admitted is False
    assert UnsafeMergeClass.CREDENTIAL_GENERALIZATION.value in _violations(result)
    assert all(not str(item.name).startswith("api_key") for item in result.pattern.parameters)
    assert any(item.kind is LostDetailKind.CREDENTIAL for item in result.lost_details)


def test_authority_splits_and_uncertain_authority_produce_counterexamples() -> None:
    left = procedure_spec(provenance="trace-a", include_authority_check=True)
    right = procedure_spec(
        name="focused-validation-procedure-b",
        provenance="trace-b",
        include_authority_check=False,
    )
    result = anti_unify_procedures(family(), (left, right))
    assert result.admitted is False
    assert UnsafeMergeClass.UNCERTAIN_AUTHORITY.value in _violations(result)

    split = anti_unify_procedures(
        family(),
        (
            procedure_spec(provenance="trace-c", authority_ids=("authority.execute",)),
            procedure_spec(
                name="focused-validation-procedure-d",
                provenance="trace-d",
                authority_ids=("authority.execute", "authority.merge"),
            ),
        ),
    )
    assert split.admitted is False
    assert UnsafeMergeClass.AUTHORITY_SPLIT.value in _violations(split)


def test_effect_splits_produce_counterexamples() -> None:
    left = procedure_spec(provenance="trace-a", include_write=True)
    right = procedure_spec(
        name="focused-validation-procedure-b",
        provenance="trace-b",
        include_write=False,
    )
    result = anti_unify_procedures(family(), (left, right))
    assert result.admitted is False
    assert UnsafeMergeClass.EFFECT_SPLIT.value in _violations(result)
    assert any(item.kind is LostDetailKind.EFFECT for item in result.lost_details)


def test_validation_order_inversion_is_an_unsafe_merge() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right_ops = _default_ops()
    # Place tests before the patch.
    read, patch, tests, post, receipt = right_ops
    inverted = trajectory(source_episode_cid="accepted-receipt-b")
    inverted = replace(
        inverted,
        steps=(
            _step(0, read[0], read[1], initial="s0", terminal="s1", effects=read[2]),
            _step(
                1,
                tests[0],
                tests[1],
                initial="s1",
                terminal="s2",
                effects=tests[2],
                validation=tests[3],
            ),
            _step(2, patch[0], patch[1], initial="s2", terminal="s3", effects=patch[2]),
            _step(
                3,
                post[0],
                post[1],
                initial="s3",
                terminal="s4",
                effects=post[2],
                validation=post[3],
            ),
            _step(4, receipt[0], receipt[1], initial="s4", terminal="s5", effects=receipt[2]),
        ),
        initial_abstract_state_cid="s0",
        terminal_abstract_state_cid="s5",
    )
    result = anti_unify(family(), (left, inverted))
    assert result.admitted is False
    assert UnsafeMergeClass.VALIDATION_SPLIT.value in _violations(result)
    assert "RUN_SELECTED_TESTS" in result.pattern.validation_operations


def test_non_validation_order_uncertainty_becomes_a_typed_hole() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(source_episode_cid="accepted-receipt-b", invert_read_patch=True)
    result = anti_unify(family(), (left, right))
    assert any(item.reason == "order" for item in result.pattern.holes)
    assert any(item.kind is LostDetailKind.ORDER for item in result.lost_details)
    assert "RUN_SELECTED_TESTS" in result.retained_validations
    assert "CHECK_POSTCONDITION" in result.retained_validations


def test_failure_transitions_are_preserved_from_recovered_traces() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(
        source_episode_cid="recovered-receipt-b",
        status=TrajectoryTerminalStatus.FAILED_RECOVERED,
        accepted_criterion_ids=(),
        failed_test=True,
    )
    result = anti_unify(family(), (left, right))
    assert "retry" in result.pattern.failure_transitions or "failed" in result.pattern.failure_transitions
    assert "RUN_SELECTED_TESTS" in result.pattern.validation_operations
    assert result.pattern.preconditions == ("criterion-a", "criterion-b")


def test_preconditions_are_unioned_and_shared_values_stay_constants() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(source_episode_cid="accepted-receipt-b")
    result = anti_unify(family(), (left, right))
    assert result.pattern.preconditions == ("criterion-a", "criterion-b")
    assert result.pattern.constants["task_family_id"] == "IMPORT_PURITY_REPAIR"
    assert result.pattern.constants["required_operations"] == result.pattern.required_operations


def test_rejects_single_trace_wrong_family_and_non_positive_membership() -> None:
    value = family()
    one = trajectory()
    with pytest.raises(AntiUnificationError, match="at least two"):
        anti_unify(value, (one,))
    with pytest.raises(AntiUnificationError, match="task-family hint"):
        anti_unify(
            value,
            (
                trajectory(source_episode_cid="a", family_hint="OTHER_FAMILY"),
                trajectory(source_episode_cid="b", family_hint="OTHER_FAMILY"),
            ),
        )
    membership = TaskFamilyMembership(
        bindings=bindings(),
        task_family_cid=value.content_id,
        trajectory_cid=one.content_id,
        membership=FamilyMembershipClass.NEGATIVE,
        evidence_cids=("membership-evidence",),
        classifier_revision="TaskFamilyClassifier@1",
    )
    with pytest.raises(AntiUnificationError, match="positive"):
        anti_unify(
            value,
            (one, trajectory(source_episode_cid="accepted-receipt-b")),
            memberships=(membership,),
        )


def test_rejected_trajectories_cannot_enter_positive_anti_unification() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(
        source_episode_cid="rejected-receipt-b",
        status=TrajectoryTerminalStatus.REJECTED,
        accepted_criterion_ids=(),
    )
    with pytest.raises(AntiUnificationError, match="accepted, recovered, or rolled-back"):
        anti_unify(family(), (left, right))


def test_policy_revision_mismatch_is_an_authority_split() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a")
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        bind=bindings(policy_revision="authority-policy-v2"),
    )
    result = anti_unify(family(), (left, right))
    assert result.admitted is False
    assert UnsafeMergeClass.AUTHORITY_SPLIT.value in _violations(result)


def test_forbidden_hole_types_fail_closed() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a", include_hole=True)
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        include_hole=True,
        hole_type="AUTHORITY_DECISION",
    )
    with pytest.raises(AntiUnificationError, match="forbidden hole"):
        anti_unify(family(), (left, right))


def test_every_lost_detail_is_recorded_and_round_trips() -> None:
    left = trajectory(source_episode_cid="accepted-receipt-a", include_query=True)
    right = trajectory(
        source_episode_cid="accepted-receipt-b",
        patch_contract="approved-patch-template@2",
        bind=bindings(tree_id="tree-other"),
    )
    result = anti_unify(family(), (left, right))
    assert result.lost_details
    kinds = {item.kind for item in result.lost_details}
    assert LostDetailKind.OPTIONAL_BRANCH in kinds
    assert LostDetailKind.PARAMETER in kinds
    assert LostDetailKind.BINDING in kinds
    restored = AntiUnificationPattern.from_artifact(result.pattern_artifact)
    assert restored.lost_details == result.pattern.lost_details
    assert restored.parameters == result.pattern.parameters
    assert result.pattern_artifact.facts["forbidden_generalizations"] == FORBIDDEN_GENERALIZATIONS


def test_result_is_deterministic() -> None:
    traces = (
        trajectory(source_episode_cid="accepted-receipt-a", include_query=True),
        trajectory(source_episode_cid="accepted-receipt-b", include_hole=True),
    )
    first = ProcedureAntiUnifier(emitted_at_ms=12).anti_unify(family(), traces)
    second = ProcedureAntiUnifier(emitted_at_ms=12).anti_unify(family(), traces)
    assert first.pattern_artifact.content_id == second.pattern_artifact.content_id
    assert first.boundary_artifact.content_id == second.boundary_artifact.content_id
    assert [item.to_facts() for item in first.counterexamples] == [
        item.to_facts() for item in second.counterexamples
    ]
