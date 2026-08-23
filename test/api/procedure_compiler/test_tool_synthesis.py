from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ProcedureContractError,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.tool_synthesis import (
    ALLOWED_OPCODES,
    APPROVED_REPAIR_TEMPLATE_IDS,
    APPROVED_TEMPLATE_LIBRARY,
    CERTIFICATE_ISSUER,
    COMPILER_REVISION,
    DSL_REVISION,
    GRAMMAR_ID,
    VALIDATOR_REVISION,
    FixtureKind,
    GeneratedToolCandidate,
    GeneratedToolCertificate,
    GeneratedToolCompiler,
    GeneratedToolSpec,
    ToolGrammarError,
    ToolPromotionError,
    ToolRepresentation,
    ToolResourceEnvelope,
    ToolSafetyError,
    ToolSynthesisError,
    ToolSynthesisReason,
    ToolTranslationError,
    TransformationDsl,
    TransformationOpcode,
    TransformationStep,
    TranslationValidator,
)


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


def _compiler() -> GeneratedToolCompiler:
    return GeneratedToolCompiler()


def _validator() -> TranslationValidator:
    return TranslationValidator()


def _compile_template(template_id: str = "normalize-identifier-record", **changes: object):
    values: dict[str, object] = {
        "bindings": _bindings(),
        "tool_id": f"tool.{template_id}",
        "template_id": template_id,
    }
    values.update(changes)
    return _compiler().compile(**values)


def test_reviewed_grammar_and_template_library_are_closed() -> None:
    assert TransformationDsl.grammar_id == GRAMMAR_ID
    assert TransformationDsl.revision == DSL_REVISION
    assert ALLOWED_OPCODES == {item.value for item in TransformationOpcode}
    assert set(APPROVED_TEMPLATE_LIBRARY) == {
        "normalize-identifier-record",
        "project-receipt-fields",
        "map-closed-status",
        "scope-relative-path",
        "select-approved-repair-template",
    }
    repair = APPROVED_TEMPLATE_LIBRARY["select-approved-repair-template"]
    assert repair.repair_template is True
    assert APPROVED_REPAIR_TEMPLATE_IDS == (
        "repair.replace-import-path",
        "repair.normalize-identifier",
        "repair.add-missing-test-name",
    )
    for template in APPROVED_TEMPLATE_LIBRARY.values():
        assert template.effect_classes == (EffectClass.OBSERVE,)
        assert template.resources.subprocess_limit == 0
        assert template.resources.network_request_limit == 0
        assert any(item.kind is FixtureKind.TEST for item in template.fixtures)
        assert any(item.kind is FixtureKind.ADVERSARIAL for item in template.fixtures)


def test_repeated_traces_synthesize_candidate_tools_only() -> None:
    traces = (
        (
            {"record_id": "Receipt.One", "status": "pending", "extra": "drop-me"},
            {"extra": "drop-me", "record_id": "receipt.one", "status": "candidate"},
        ),
        (
            {"record_id": "Receipt.Two", "status": "done"},
            {"record_id": "receipt.two", "status": "accepted"},
        ),
    )
    compiled = _compiler().synthesize(traces, bindings=_bindings(), tool_id="tool.from-traces")
    assert compiled.spec.state is ArtifactState.CANDIDATE
    assert compiled.candidate.state is ArtifactState.CANDIDATE
    assert compiled.candidate.representation is ToolRepresentation.INTERPRETED_DSL
    assert compiled.candidate.can_promote is False
    assert compiled.candidate.can_authorize is False
    assert compiled.spec.can_grant_authority is False
    assert compiled.spec.template_id == "normalize-identifier-record"
    again = _compiler().synthesize(traces, bindings=_bindings(), tool_id="tool.from-traces")
    assert again.candidate.content_id == compiled.candidate.content_id
    assert again.spec.content_id == compiled.spec.content_id


def test_interpreted_dsl_invokes_deterministically_and_round_trips() -> None:
    compiled = _compile_template()
    result = _compiler().invoke(
        compiled.candidate,
        {"record_id": "Receipt.One", "status": "pending", "extra": "keep"},
    )
    assert dict(result.output) == {
        "extra": "keep",
        "record_id": "receipt.one",
        "status": "candidate",
    }
    assert result.receipt.accepted is True
    assert result.receipt.state is ArtifactState.CANDIDATE
    assert result.receipt.can_authorize is False
    assert result.receipt.compiler_revision == COMPILER_REVISION

    decoded_spec = GeneratedToolSpec.from_dict(compiled.spec.to_dict())
    assert decoded_spec == compiled.spec
    parsed_spec = parse_procedure_artifact(compiled.spec.to_dict())
    assert isinstance(parsed_spec, GeneratedToolSpec)
    decoded_candidate = GeneratedToolCandidate.from_dict(compiled.candidate.to_dict())
    assert decoded_candidate == compiled.candidate
    parsed_candidate = parse_procedure_artifact(compiled.candidate.to_dict())
    assert isinstance(parsed_candidate, GeneratedToolCandidate)
    parsed_receipt = parse_procedure_artifact(result.receipt.to_dict())
    assert parsed_receipt == result.receipt
    with pytest.raises(FrozenInstanceError):
        compiled.candidate.state = ArtifactState.PROMOTED  # type: ignore[misc]


def test_optimized_python_stays_candidate_until_certificate_and_promotion() -> None:
    compiled = _compile_template("project-receipt-fields")
    compiler = _compiler()
    optimized = compiler.optimize(compiled)
    assert optimized.representation is ToolRepresentation.OPTIMIZED_PYTHON
    assert optimized.state is ArtifactState.CANDIDATE
    assert optimized.predecessor_cid == compiled.candidate.content_id
    assert optimized.certificate_cid == ""
    assert optimized.can_promote is False

    validation = _validator().validate_and_certify(compiled.candidate, optimized)
    assert validation.receipt.equivalent is True
    assert validation.receipt.state is ArtifactState.CANDIDATE
    assert validation.receipt.can_promote is False
    assert validation.certificate is not None
    certificate = validation.certificate
    assert certificate.state is ArtifactState.VERIFIED
    assert certificate.issuer == CERTIFICATE_ISSUER
    assert certificate.validator_revision == VALIDATOR_REVISION
    assert certificate.can_authorize is False
    assert certificate.can_promote is False
    parsed_certificate = parse_procedure_artifact(certificate.to_dict())
    assert isinstance(parsed_certificate, GeneratedToolCertificate)
    assert parsed_certificate == certificate

    promoted = _validator().promote(optimized, certificate, validation.receipt)
    assert promoted.state is ArtifactState.PROMOTED
    assert promoted.representation is ToolRepresentation.OPTIMIZED_PYTHON
    assert promoted.certificate_cid == certificate.content_id
    assert promoted.translation_receipt_cid == validation.receipt.content_id
    assert promoted.predecessor_cid == optimized.content_id
    assert promoted.can_authorize is False
    invoked = compiler.invoke(
        promoted,
        {
            "receipt_id": "receipt-1",
            "tree_id": "tree-1",
            "status": "accepted",
            "producer": "tests@1",
            "note": "omit",
        },
    )
    assert dict(invoked.output) == {
        "producer": "tests@1",
        "receipt_id": "receipt-1",
        "status": "accepted",
        "tree_id": "tree-1",
    }


def test_interpreted_dsl_cannot_be_promoted() -> None:
    compiled = _compile_template("map-closed-status")
    optimized = _compiler().optimize(compiled)
    validation = _validator().validate_and_certify(compiled.candidate, optimized)
    assert validation.certificate is not None
    with pytest.raises(ToolPromotionError, match="interpreted DSL"):
        _validator().promote(compiled.candidate, validation.certificate, validation.receipt)
    with pytest.raises(ToolPromotionError, match="candidate-tier"):
        replace(compiled.candidate, state=ArtifactState.PROMOTED)


def test_promotion_requires_exact_certificate_binding() -> None:
    compiled = _compile_template("map-closed-status")
    optimized = _compiler().optimize(compiled)
    validation = _validator().validate_and_certify(compiled.candidate, optimized)
    assert validation.certificate is not None
    promoted = _validator().promote(optimized, validation.certificate, validation.receipt)
    assert promoted.state is ArtifactState.PROMOTED

    other = _compiler().optimize(_compile_template("map-closed-status", tool_id="tool.other"))
    with pytest.raises(ToolPromotionError, match="does not bind this optimized candidate"):
        _validator().promote(other, validation.certificate, validation.receipt)
    with pytest.raises(ToolTranslationError, match="exact differential"):
        _validator().certify(
            replace(
                validation.receipt,
                equivalent=False,
                reason_code=ToolSynthesisReason.TRANSLATION_MISMATCH,
                failed_fixture_ids=("x",),
            ),
            optimized,
        )


def test_inequivalent_optimized_python_is_not_certified_or_promoted() -> None:
    compiled = _compile_template("map-closed-status")
    optimized = _compiler().optimize(compiled)
    mutated = replace(
        optimized,
        transformations=(
            TransformationStep(
                opcode=TransformationOpcode.SET_LITERAL,
                field="status",
                parameters={"value": "forged"},
            ),
        ),
    )
    receipt = _validator().validate(compiled.candidate, mutated)
    assert receipt.equivalent is False
    assert receipt.failed_fixture_ids
    assert receipt.reason_code is ToolSynthesisReason.TRANSLATION_MISMATCH
    validation = _validator().validate_and_certify(compiled.candidate, mutated)
    assert validation.certificate is None
    with pytest.raises(ToolTranslationError, match="exact differential"):
        _validator().certify(receipt, mutated)


def test_unknown_opcode_and_arbitrary_code_or_shell_are_refused() -> None:
    with pytest.raises(ToolSafetyError) as python_error:
        TransformationStep(opcode="arbitrary-python", parameters={})
    assert python_error.value.reason_code is ToolSynthesisReason.ARBITRARY_CODE
    with pytest.raises(ToolSafetyError) as shell_error:
        TransformationStep(opcode="ARBITRARY_SHELL", parameters={})
    assert shell_error.value.reason_code is ToolSynthesisReason.ARBITRARY_SHELL
    with pytest.raises(ToolSafetyError) as unknown:
        TransformationStep(opcode="eval", parameters={})
    assert unknown.value.reason_code is ToolSynthesisReason.ARBITRARY_CODE
    with pytest.raises(ToolSafetyError, match="arbitrary code"):
        TransformationStep(
            opcode=TransformationOpcode.SET_LITERAL,
            field="status",
            parameters={"value": "ok", "python_source": "import os"},
        )
    with pytest.raises(ToolSafetyError, match="arbitrary shell"):
        TransformationStep(
            opcode=TransformationOpcode.SET_LITERAL,
            field="status",
            parameters={"value": "ok", "shell_command": "rm -rf /"},
        )


def test_effect_path_and_resource_escalation_are_fail_closed() -> None:
    with pytest.raises(ToolSafetyError, match="escalate effects"):
        _compile_template(effect_classes=(EffectClass.REPOSITORY_WRITE,))
    with pytest.raises(ToolSafetyError, match="escape the template scope"):
        _compile_template(path_prefixes=("docs/architecture",))
    with pytest.raises(ToolSafetyError, match="escape the template scope"):
        _compile_template(path_prefixes=("ipfs_accelerate_py",))
    with pytest.raises(ProcedureContractError, match="network_request_limit"):
        ToolResourceEnvelope(network_request_limit=1)
    with pytest.raises(ProcedureContractError, match="subprocess_limit"):
        ToolResourceEnvelope(subprocess_limit=1)
    compiled = _compile_template("scope-relative-path")
    with pytest.raises(ToolSafetyError) as escaped:
        _compiler().invoke(compiled.candidate, {"path": "../secrets"})
    assert escaped.value.reason_code is ToolSynthesisReason.PATH_ESCAPE


def test_schema_enum_and_unreviewed_template_failures_are_refused() -> None:
    compiled = _compile_template("map-closed-status")
    with pytest.raises(ToolSynthesisError) as closed:
        _compiler().invoke(compiled.candidate, {"status": "invented"})
    assert closed.value.reason_code is ToolSynthesisReason.ENUM_CLOSED
    with pytest.raises(ToolGrammarError, match="reviewed library"):
        _compiler().compile(
            bindings=_bindings(),
            tool_id="tool.unknown",
            template_id="invented-template",
        )
    with pytest.raises(ToolGrammarError, match="do not match a reviewed template"):
        _compiler().synthesize(
            (({"status": "done"}, {"status": "not-the-template"}),),
            bindings=_bindings(),
        )
    receipt_tool = _compile_template("project-receipt-fields")
    with pytest.raises(ToolSynthesisError) as missing:
        _compiler().invoke(receipt_tool.candidate, {"receipt_id": "receipt-1", "status": "accepted"})
    assert missing.value.reason_code is ToolSynthesisReason.MISSING_FIELD


def test_repair_template_selection_stays_inside_approved_library() -> None:
    compiled = _compile_template("select-approved-repair-template")
    result = _compiler().invoke(
        compiled.candidate,
        {
            "template_id": "repair.replace-import-path",
            "target_path": "procedure_compiler/contracts.py",
            "note": "omit",
        },
    )
    assert dict(result.output) == {
        "target_path": "ipfs_accelerate_py/agent_supervisor/procedure_compiler/contracts.py",
        "template_id": "repair.replace-import-path",
    }
    with pytest.raises(ToolGrammarError) as unknown:
        _compiler().invoke(
            compiled.candidate,
            {
                "template_id": "repair.invented-shell",
                "target_path": "procedure_compiler/contracts.py",
            },
        )
    assert unknown.value.reason_code is ToolSynthesisReason.UNKNOWN_TEMPLATE


def test_apply_template_expands_from_the_reviewed_library() -> None:
    compiled = _compiler().compile(
        bindings=_bindings(),
        tool_id="tool.apply-status",
        transformations=(
            TransformationStep(
                opcode=TransformationOpcode.APPLY_TEMPLATE,
                parameters={"template_id": "map-closed-status"},
            ),
        ),
        path_prefixes=("ipfs_accelerate_py/agent_supervisor",),
        fixtures=APPROVED_TEMPLATE_LIBRARY["map-closed-status"].fixtures,
        input_schema_ref="schema.status.in",
        output_schema_ref="schema.status.out",
    )
    assert compiled.spec.transformations[0].opcode is TransformationOpcode.APPLY_TEMPLATE
    assert all(step.opcode is not TransformationOpcode.APPLY_TEMPLATE for step in compiled.candidate.transformations)
    result = _compiler().invoke(compiled.candidate, {"status": "failed", "other": "keep"})
    assert dict(result.output) == {"other": "keep", "status": "rejected"}


def test_translation_validation_rejects_adversarial_and_requires_passing_tests() -> None:
    compiled = _compile_template("normalize-identifier-record")
    optimized = _compiler().optimize(compiled)
    receipt = _validator().validate(compiled.candidate, optimized)
    assert receipt.equivalent is True
    assert "normalize-identifier-record.ok" in receipt.passed_fixture_ids
    assert "normalize-identifier-record.unknown-status" in receipt.adversarial_rejected_ids
    assert not receipt.failed_fixture_ids
    with pytest.raises(ToolTranslationError, match="requires fixtures"):
        _validator().validate(
            replace(compiled.candidate, fixtures=(), test_fixture_ids=(), adversarial_fixture_ids=()),
            replace(optimized, fixtures=(), test_fixture_ids=(), adversarial_fixture_ids=()),
            fixtures=(),
        )


def test_empty_or_unbounded_programs_are_refused() -> None:
    with pytest.raises(ToolGrammarError, match="at least one transformation"):
        TransformationDsl().parse(())
    too_many = tuple(
        TransformationStep(opcode=TransformationOpcode.SORT_KEYS)
        for _ in range(17)
    )
    with pytest.raises(ToolSynthesisError) as bound:
        TransformationDsl().parse(too_many)
    assert bound.value.reason_code is ToolSynthesisReason.RESOURCE_EXCEEDED
    with pytest.raises(ToolGrammarError, match="requires a field"):
        TransformationStep(opcode=TransformationOpcode.LOWERCASE)
