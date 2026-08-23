from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    FORBIDDEN_STEP_OPERATIONS,
    ProcedureContractError,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.tool_synthesis import (
    ALLOWED_EFFECTS,
    APPROVED_REPAIR_TEMPLATE_IDS,
    COMPILER_REVISION,
    DEFAULT_CERTIFICATE_ISSUER,
    DSL_REVISION,
    GRAMMAR_REVISION,
    MAX_OPERATIONS,
    MIN_REPETITIONS,
    REQUIRED_TOOL_DECLARATION_FIELDS,
    RISK_CEILING,
    VALIDATOR_REVISION,
    DeterministicToolDsl,
    FixtureKind,
    GeneratedToolCandidate,
    GeneratedToolCertificate,
    GeneratedToolCompiler,
    GeneratedToolInvocationReceipt,
    GeneratedToolSpec,
    ToolPromotionAction,
    ToolPromotionDecision,
    ToolReason,
    ToolResourceBound,
    ToolSynthesisRefusal,
    TransformationDsl,
    TransformationOp,
    TransformationOpcode,
    TransformationProgram,
    TranslationKind,
    TranslationStatus,
    TranslationValidationReceipt,
    TranslationValidator,
    reviewed_template_library,
)


SCOPE = "ipfs_accelerate_py/agent_supervisor/procedure_compiler"


def _bindings(**changes: object) -> ArtifactBindings:
    values: dict[str, object] = {
        "repository_id": "repo-main",
        "repository_commit": "commit-abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G030",
        "task_id": "PCPC-023",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    values.update(changes)
    return ArtifactBindings(**values)


def _library():
    return reviewed_template_library()


def _template(template_id: str = "tool.project-rename-sort"):
    for item in _library():
        if item.template_id == template_id:
            return item
    raise AssertionError(f"missing reviewed template {template_id}")


def _compiler() -> GeneratedToolCompiler:
    return GeneratedToolCompiler()


def _compile(
    template_id: str = "tool.project-rename-sort",
    *,
    occurrence_count: int = 2,
    **changes: object,
) -> GeneratedToolCandidate:
    template = _template(template_id)
    values: dict[str, object] = {
        "bindings": _bindings(),
        "program": template.program,
        "occurrence_count": occurrence_count,
        "template_id": template_id,
    }
    values.update(changes)
    return _compiler().compile(**values)


def _validate(candidate: GeneratedToolCandidate) -> TranslationValidationReceipt:
    return TranslationValidator().validate(candidate)


def _certificate(
    candidate: GeneratedToolCandidate,
    validation: TranslationValidationReceipt | None = None,
    **changes: object,
) -> GeneratedToolCertificate:
    receipt = validation or _validate(candidate)
    values: dict[str, object] = {
        "candidate": candidate,
        "validation": receipt,
        "now_ms": 1_000,
        "expires_at_ms": 86_400_000,
    }
    values.update(changes)
    return TranslationValidator().issue_certificate(**values)


def test_reviewed_library_is_closed_and_declares_required_fields() -> None:
    library = _library()
    assert library
    assert {item.template_id for item in library} == {
        "tool.project-rename-sort",
        "tool.filter-dedupe-join",
        "tool.qualify-in-scope",
    }
    assert set(REQUIRED_TOOL_DECLARATION_FIELDS) == {
        "schema",
        "effects",
        "path_limits",
        "resources",
        "tests",
        "adversarial_fixtures",
    }
    for template in library:
        assert template.program.grammar_revision == GRAMMAR_REVISION
        assert template.program.input_schema_ref
        assert template.program.output_schema_ref
        assert template.program.scope_paths
        assert set(template.program.effect_classes) <= ALLOWED_EFFECTS
        assert template.risk_class is RISK_CEILING
        assert template.tests
        assert template.adversarial_fixtures
        assert all(item.kind is FixtureKind.TEST and not item.expect_reject for item in template.tests)
        assert all(
            item.kind is FixtureKind.ADVERSARIAL and item.expect_reject
            for item in template.adversarial_fixtures
        )
        assert set(template.approved_repair_template_ids) <= set(APPROVED_REPAIR_TEMPLATE_IDS)


def test_dsl_alias_and_grammar_bound_reject_unbounded_programs() -> None:
    assert DeterministicToolDsl is TransformationDsl
    assert TransformationDsl.revision == DSL_REVISION
    template = _template()
    ops = list(template.program.operations)
    extra = TransformationOp(opcode=TransformationOpcode.SORT, operand="symbol")
    unbounded = ops + [extra] * (MAX_OPERATIONS)
    with pytest.raises(ToolSynthesisRefusal) as caught:
        TransformationProgram(
            operations=tuple(unbounded),
            input_schema_ref=template.program.input_schema_ref,
            output_schema_ref=template.program.output_schema_ref,
            scope_paths=template.program.scope_paths,
        )
    assert caught.value.reason_code is ToolReason.GRAMMAR_UNBOUNDED


@pytest.mark.parametrize("opcode", sorted(FORBIDDEN_STEP_OPERATIONS) + ["eval", "exec", "os.system"])
def test_arbitrary_code_and_shell_opcodes_are_rejected(opcode: str) -> None:
    with pytest.raises(ToolSynthesisRefusal) as caught:
        TransformationOp(opcode=opcode, operand=("module",))
    assert caught.value.reason_code is ToolReason.FORBIDDEN_OPCODE


def test_executable_operands_and_unapproved_repair_templates_are_rejected() -> None:
    with pytest.raises(ToolSynthesisRefusal) as caught:
        TransformationOp(
            opcode=TransformationOpcode.PROJECT,
            operand={"python_source": "print(1)"},
        )
    assert caught.value.reason_code is ToolReason.ARBITRARY_CODE

    with pytest.raises(ToolSynthesisRefusal) as caught_shell:
        TransformationOp(
            opcode=TransformationOpcode.RENAME,
            operand={"name": "eval(payload)"},
        )
    assert caught_shell.value.reason_code is ToolReason.ARBITRARY_CODE

    with pytest.raises(ToolSynthesisRefusal) as caught_template:
        TransformationOp(
            opcode=TransformationOpcode.MAP_APPROVED_TEMPLATE,
            operand="repair.invented-arbitrary-patch",
        )
    assert caught_template.value.reason_code is ToolReason.UNAPPROVED_TEMPLATE


def test_generated_tools_cannot_declare_effect_or_risk_escalation() -> None:
    template = _template()
    with pytest.raises(ToolSynthesisRefusal) as caught:
        TransformationProgram(
            operations=template.program.operations,
            input_schema_ref=template.program.input_schema_ref,
            output_schema_ref=template.program.output_schema_ref,
            scope_paths=template.program.scope_paths,
            effect_classes=(EffectClass.REPOSITORY_WRITE,),
        )
    assert caught.value.reason_code is ToolReason.EFFECT_ESCALATION


def test_repeated_reviewed_transformations_yield_candidate_tools_only() -> None:
    template = _template()
    candidate = _compile(
        observed_programs=(template.program, template.program, template.program),
        occurrence_count=None,
    )
    spec = GeneratedToolSpec(
        bindings=candidate.bindings,
        tool_id=candidate.tool_id,
        template_id=candidate.template_id,
        program=candidate.program,
        optimized_translation_id=candidate.optimized_translation_id,
        resource_bound=candidate.resource_bound,
        test_ids=candidate.test_ids,
        adversarial_fixture_ids=candidate.adversarial_fixture_ids,
        occurrence_count=candidate.occurrence_count,
    )
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.optimized_status is TranslationStatus.CANDIDATE
    assert candidate.can_authorize is False
    assert candidate.can_promote is False
    assert candidate.occurrence_count == 3
    assert spec.state is ArtifactState.CANDIDATE
    assert spec.can_promote is False
    assert spec.missing_declaration_fields() == ()
    assert spec.compiler_revision == COMPILER_REVISION
    with pytest.raises(FrozenInstanceError):
        candidate.state = ArtifactState.PROMOTED  # type: ignore[misc]


def test_one_off_transformations_do_not_synthesize_a_tool() -> None:
    with pytest.raises(ToolSynthesisRefusal) as caught:
        _compile(occurrence_count=1)
    assert caught.value.reason_code is ToolReason.INSUFFICIENT_REPETITION
    assert MIN_REPETITIONS == 2

    with pytest.raises(ToolSynthesisRefusal) as missing:
        _compiler().compile(bindings=_bindings(), program=_template().program)
    assert missing.value.reason_code is ToolReason.INSUFFICIENT_REPETITION


def test_unreviewed_programs_cannot_be_generated() -> None:
    template = _template()
    invented = TransformationProgram(
        operations=(TransformationOp(opcode=TransformationOpcode.SORT, operand="symbol"),),
        input_schema_ref=template.program.input_schema_ref,
        output_schema_ref=template.program.output_schema_ref,
        scope_paths=template.program.scope_paths,
    )
    with pytest.raises(ToolSynthesisRefusal) as caught:
        _compiler().compile(
            bindings=_bindings(),
            program=invented,
            occurrence_count=4,
        )
    assert caught.value.reason_code is ToolReason.UNKNOWN_TEMPLATE


def test_path_and_resource_bounds_fail_closed() -> None:
    dsl = TransformationDsl()
    template = _template()
    with pytest.raises(ToolSynthesisRefusal) as path_caught:
        dsl.interpret(
            template.program,
            (
                {
                    "module": "pkg.mod",
                    "name": "alpha",
                    "path": "tmp/escape.py",
                },
            ),
            resource_bound=template.resource_bound,
        )
    assert path_caught.value.reason_code is ToolReason.PATH_ESCAPE

    with pytest.raises(ToolSynthesisRefusal) as resource_caught:
        dsl.interpret(
            template.program,
            template.tests[0].input_value,
            resource_bound=ToolResourceBound(max_items=1, max_operations=8),
        )
    assert resource_caught.value.reason_code is ToolReason.RESOURCE_BOUND


def test_dsl_and_optimized_python_are_differentially_exact() -> None:
    compiler = _compiler()
    validator = TranslationValidator()
    for template in _library():
        candidate = compiler.compile(
            bindings=_bindings(),
            program=template.program,
            occurrence_count=2,
            template_id=template.template_id,
        )
        receipt = validator.validate(candidate)
        assert receipt.equivalent is True
        assert receipt.dsl_output_digest == receipt.optimized_output_digest
        assert receipt.can_authorize is False
        assert receipt.can_promote is False
        assert receipt.state is ArtifactState.CANDIDATE
        assert set(receipt.tests_passed) == {item.fixture_id for item in template.tests}
        assert set(receipt.adversarial_passed) == {
            item.fixture_id for item in template.adversarial_fixtures
        }
        dsl_result = TransformationDsl().interpret(
            candidate.program,
            template.tests[0].input_value,
            resource_bound=candidate.resource_bound,
        )
        optimized = validator.evaluate_optimized(candidate, template.tests[0].input_value)
        assert dsl_result.value == optimized.value
        assert dsl_result.translation_kind is TranslationKind.INTERPRETED_DSL
        assert optimized.translation_kind is TranslationKind.OPTIMIZED_PYTHON


def test_mismatched_optimized_translation_is_not_equivalent() -> None:
    candidate = _compile()
    tampered = GeneratedToolCandidate(
        bindings=candidate.bindings,
        tool_id=candidate.tool_id,
        spec_cid=candidate.spec_cid,
        template_id=candidate.template_id,
        program=candidate.program,
        optimized_translation_id="opt.filter-dedupe-join",
        resource_bound=candidate.resource_bound,
        occurrence_count=candidate.occurrence_count,
        test_ids=candidate.test_ids,
        adversarial_fixture_ids=candidate.adversarial_fixture_ids,
    )
    with pytest.raises(ToolSynthesisRefusal) as caught:
        TranslationValidator().validate(tampered)
    assert caught.value.reason_code is ToolReason.TRANSLATION_MISMATCH


def test_optimized_python_stays_candidate_without_certificate() -> None:
    candidate = _compile()
    validation = _validate(candidate)
    decision = _compiler().promote_optimized(candidate, validation, None)
    assert decision.action is ToolPromotionAction.REFUSE
    assert decision.reason_code is ToolReason.MISSING_CERTIFICATE
    assert decision.optimized_status is TranslationStatus.REJECTED
    assert decision.tool_state is ArtifactState.CANDIDATE
    assert candidate.optimized_status is TranslationStatus.CANDIDATE
    assert candidate.state is ArtifactState.CANDIDATE


def test_optimized_python_stays_candidate_without_differential_validation() -> None:
    candidate = _compile()
    certificate = _certificate(candidate)
    decision = _compiler().promote_optimized(candidate, None, certificate)
    assert decision.action is ToolPromotionAction.REFUSE
    assert decision.reason_code is ToolReason.MISSING_VALIDATION
    assert decision.tool_state is ArtifactState.CANDIDATE


def test_optimized_python_is_promoted_only_after_validation_and_certificate() -> None:
    candidate = _compile()
    validation = _validate(candidate)
    certificate = _certificate(candidate, validation)
    decision = _compiler().promote_optimized(
        candidate, validation, certificate, now_ms=2_000
    )
    assert decision.action is ToolPromotionAction.PROMOTE_OPTIMIZED
    assert decision.reason_code is ToolReason.ADMITTED
    assert decision.optimized_status is TranslationStatus.PROMOTED
    assert decision.tool_state is ArtifactState.CANDIDATE
    assert decision.can_promote_tool is False
    assert decision.can_authorize is False
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.optimized_status is TranslationStatus.CANDIDATE
    assert certificate.state is ArtifactState.VERIFIED
    assert certificate.can_promote is False
    assert certificate.can_authorize is False
    assert certificate.translation_equivalent is True

    dsl_receipt = _compiler().invoke(
        candidate, _template().tests[0].input_value
    )
    assert dsl_receipt.refused is False
    assert dsl_receipt.translation_kind is TranslationKind.INTERPRETED_DSL
    assert dsl_receipt.state is ArtifactState.CANDIDATE

    optimized_receipt = _compiler().invoke(
        candidate,
        _template().tests[0].input_value,
        translation=TranslationKind.OPTIMIZED_PYTHON,
        promotion=decision,
        certificate=certificate,
    )
    assert optimized_receipt.refused is False
    assert optimized_receipt.translation_kind is TranslationKind.OPTIMIZED_PYTHON
    assert optimized_receipt.output_digest == dsl_receipt.output_digest
    assert optimized_receipt.can_authorize is False


def test_optimized_invocation_without_promotion_is_refused() -> None:
    candidate = _compile()
    receipt = _compiler().invoke(
        candidate,
        _template().tests[0].input_value,
        translation=TranslationKind.OPTIMIZED_PYTHON,
    )
    assert receipt.refused is True
    assert receipt.reason_code == ToolReason.OPTIMIZED_NOT_PROMOTED.value
    assert receipt.state is ArtifactState.CANDIDATE


def test_self_issued_and_forged_certificates_fail_closed() -> None:
    candidate = _compile()
    validation = _validate(candidate)
    with pytest.raises(ToolSynthesisRefusal) as self_issued:
        TranslationValidator().issue_certificate(
            candidate,
            validation,
            now_ms=1,
            expires_at_ms=10,
            issuer="self",
        )
    assert self_issued.value.reason_code is ToolReason.SELF_ISSUED

    with pytest.raises(ToolSynthesisRefusal) as compiler_issued:
        TranslationValidator(issuer=COMPILER_REVISION)
    assert compiler_issued.value.reason_code is ToolReason.SELF_ISSUED

    certificate = _certificate(candidate, validation)
    payload = certificate.to_dict()
    payload["issuer"] = "forged-issuer@1"
    payload["statement_digest"] = certificate.statement_digest
    with pytest.raises(ToolSynthesisRefusal) as forged:
        GeneratedToolCertificate.from_dict(payload)
    assert forged.value.reason_code is ToolReason.FORGED_CERTIFICATE


def test_expired_certificate_cannot_promote_optimized_python() -> None:
    candidate = _compile()
    validation = _validate(candidate)
    certificate = _certificate(candidate, validation)
    decision = _compiler().promote_optimized(
        candidate, validation, certificate, now_ms=certificate.expires_at_ms
    )
    assert decision.action is ToolPromotionAction.REFUSE
    assert decision.reason_code is ToolReason.STALE_CERTIFICATE
    assert decision.optimized_status is TranslationStatus.REJECTED


def test_certificate_and_candidate_cannot_leave_non_promoting_states() -> None:
    candidate = _compile()
    with pytest.raises(ToolSynthesisRefusal) as promoted_candidate:
        GeneratedToolCandidate(
            bindings=candidate.bindings,
            tool_id=candidate.tool_id,
            spec_cid=candidate.spec_cid,
            template_id=candidate.template_id,
            program=candidate.program,
            optimized_translation_id=candidate.optimized_translation_id,
            resource_bound=candidate.resource_bound,
            occurrence_count=candidate.occurrence_count,
            test_ids=candidate.test_ids,
            adversarial_fixture_ids=candidate.adversarial_fixture_ids,
            state=ArtifactState.PROMOTED,
        )
    assert promoted_candidate.value.reason_code is ToolReason.CANDIDATE_TIER_REQUIRED

    with pytest.raises(ToolSynthesisRefusal) as promoted_optimized:
        GeneratedToolCandidate(
            bindings=candidate.bindings,
            tool_id=candidate.tool_id,
            spec_cid=candidate.spec_cid,
            template_id=candidate.template_id,
            program=candidate.program,
            optimized_translation_id=candidate.optimized_translation_id,
            resource_bound=candidate.resource_bound,
            occurrence_count=candidate.occurrence_count,
            test_ids=candidate.test_ids,
            adversarial_fixture_ids=candidate.adversarial_fixture_ids,
            optimized_status=TranslationStatus.PROMOTED,
        )
    assert promoted_optimized.value.reason_code is ToolReason.PROMOTION_FORBIDDEN

    validation = _validate(candidate)
    certificate = _certificate(candidate, validation)
    with pytest.raises(ToolSynthesisRefusal) as cert_promote:
        GeneratedToolCertificate(
            bindings=certificate.bindings,
            tool_id=certificate.tool_id,
            spec_cid=certificate.spec_cid,
            candidate_cid=certificate.candidate_cid,
            template_id=certificate.template_id,
            validation_cid=certificate.validation_cid,
            test_ids=certificate.test_ids,
            adversarial_fixture_ids=certificate.adversarial_fixture_ids,
            translation_equivalent=True,
            issuer=certificate.issuer,
            issued_at_ms=certificate.issued_at_ms,
            expires_at_ms=certificate.expires_at_ms,
            can_promote=True,
        )
    assert cert_promote.value.reason_code is ToolReason.AUTHORITY_REJECTED


def test_artifacts_round_trip_and_parse_through_the_compiler_schema() -> None:
    candidate = _compile("tool.qualify-in-scope")
    validation = _validate(candidate)
    certificate = _certificate(candidate, validation)
    decision = _compiler().promote_optimized(candidate, validation, certificate, now_ms=2_000)
    receipt = _compiler().invoke(
        candidate,
        _template("tool.qualify-in-scope").tests[0].input_value,
        translation=TranslationKind.OPTIMIZED_PYTHON,
        promotion=decision,
        certificate=certificate,
    )
    spec = GeneratedToolSpec(
        bindings=candidate.bindings,
        tool_id=candidate.tool_id,
        template_id=candidate.template_id,
        program=candidate.program,
        optimized_translation_id=candidate.optimized_translation_id,
        resource_bound=candidate.resource_bound,
        test_ids=candidate.test_ids,
        adversarial_fixture_ids=candidate.adversarial_fixture_ids,
        occurrence_count=candidate.occurrence_count,
    )

    for artifact in (spec, candidate, validation, certificate, decision, receipt):
        decoded = artifact.from_dict(artifact.to_dict())
        assert decoded == artifact
        parsed = parse_procedure_artifact(artifact.to_dict())
        assert parsed == artifact
        assert parsed.content_id == artifact.content_id
        payload = artifact.to_dict()
        payload["unknown_field"] = "nope"
        with pytest.raises(ProcedureContractError):
            artifact.from_dict(payload)


def test_qualify_template_reuses_approved_repair_templates() -> None:
    template = _template("tool.qualify-in-scope")
    assert "repair.qualify-symbol" in template.approved_repair_template_ids
    candidate = _compile("tool.qualify-in-scope")
    result = TransformationDsl().interpret(
        candidate.program,
        template.tests[0].input_value,
        resource_bound=candidate.resource_bound,
    )
    assert result.value[0]["qualified"] == "pkg.mod:HoleRequest"
    assert all(_SCOPE in path or path.startswith(_SCOPE) for path in result.paths_read)


def test_filter_join_template_is_pure_and_deterministic() -> None:
    template = _template("tool.filter-dedupe-join")
    first = TransformationDsl().interpret(
        template.program, template.tests[0].input_value, resource_bound=template.resource_bound
    )
    second = TransformationDsl().interpret(
        template.program, template.tests[0].input_value, resource_bound=template.resource_bound
    )
    assert first.value == second.value == template.tests[0].expected_output
    assert first.value == {"joined": "pkg.a:pkg.b"}


def test_invocation_receipt_records_typed_refusals() -> None:
    candidate = _compile()
    receipt = _compiler().invoke(
        candidate,
        ({"module": "pkg.mod", "name": "alpha", "path": "tmp/escape.py"},),
    )
    assert receipt.refused is True
    assert receipt.reason_code == ToolReason.PATH_ESCAPE.value
    assert receipt.can_authorize is False
    parsed = GeneratedToolInvocationReceipt.from_dict(receipt.to_dict())
    assert parsed == receipt


def test_promotion_decision_cannot_promote_the_tool_artifact() -> None:
    candidate = _compile()
    validation = _validate(candidate)
    certificate = _certificate(candidate, validation)
    with pytest.raises(ToolSynthesisRefusal) as caught:
        ToolPromotionDecision(
            bindings=candidate.bindings,
            candidate_cid=candidate.content_id,
            certificate_cid=certificate.content_id,
            validation_cid=validation.content_id,
            action=ToolPromotionAction.PROMOTE_OPTIMIZED,
            reason_code=ToolReason.ADMITTED,
            optimized_status=TranslationStatus.PROMOTED,
            tool_state=ArtifactState.PROMOTED,
        )
    assert caught.value.reason_code is ToolReason.PROMOTION_FORBIDDEN


def test_default_issuer_is_the_independent_validator() -> None:
    assert DEFAULT_CERTIFICATE_ISSUER == "translation-validator@1"
    assert VALIDATOR_REVISION == "TranslationValidator@1"
    certificate = _certificate(_compile())
    assert certificate.issuer == DEFAULT_CERTIFICATE_ISSUER
    assert certificate.validator_revision == VALIDATOR_REVISION
