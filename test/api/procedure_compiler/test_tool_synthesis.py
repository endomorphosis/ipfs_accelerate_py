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
    ALLOWED_EFFECT_CLASSES,
    APPROVED_REPAIR_TEMPLATE_IDS,
    COMPILER_REVISION,
    DETERMINISTIC_TOOL_DSL_REVISION,
    DSL_REVISION,
    FORBIDDEN_TRANSFORMATION_OPS,
    GRAMMAR_REVISION,
    MAX_TOOL_STEPS,
    VALIDATOR_REVISION,
    DeterministicToolDsl,
    FixtureKind,
    GeneratedToolCandidate,
    GeneratedToolCertificate,
    GeneratedToolCompiler,
    GeneratedToolInvocationReceipt,
    GeneratedToolSpec,
    TemplateLibrary,
    ToolFixture,
    ToolGrammarError,
    ToolPromotionError,
    ToolReason,
    ToolResourceEnvelope,
    ToolSafetyError,
    ToolSynthesisError,
    ToolSynthesisRequest,
    ToolTranslationError,
    TransformationDsl,
    TransformationOp,
    TransformationProgram,
    TransformationStep,
    TranslationReceipt,
    TranslationStatus,
    TranslationValidator,
    default_adversarial_fixtures,
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


def _happy_mapping(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "path": f"{SCOPE}/tool_synthesis.py",
        "kind": "keep",
        "id": "symbol-a",
        "extra": "ignored",
    }
    values.update(changes)
    return values


def _happy_sequence() -> tuple[dict[str, object], ...]:
    return (
        {"id": "b", "kind": "drop", "path": f"{SCOPE}/b.py"},
        {"id": "a", "kind": "keep", "path": f"{SCOPE}/a.py"},
        {"id": "c", "kind": "keep", "path": f"{SCOPE}/c.py"},
    )


def _request(**changes: object) -> ToolSynthesisRequest:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "tool_id": "tool.select-rename",
        "input_schema_ref": "schema.tool-in",
        "output_schema_ref": "schema.tool-out",
        "scope_paths": (SCOPE,),
        "template_id": "select-and-rename",
        "template_parameters": {
            "fields": ("path", "kind"),
            "mapping": {"path": "scope_path"},
        },
        "fixtures": (
            ToolFixture(fixture_id="happy.select", kind=FixtureKind.HAPPY, payload=_happy_mapping()),
        ),
    }
    values.update(changes)
    return ToolSynthesisRequest(**values)


def _synthesize(**changes: object):
    return GeneratedToolCompiler().synthesize(_request(**changes))


def _qualify(**changes: object):
    compiler = GeneratedToolCompiler()
    result = compiler.synthesize(_request(**changes))
    receipt = compiler.validator.validate(
        result.spec, result.candidate, result.optimized, result.fixtures
    )
    certificate = compiler.certify(result, receipt)
    return compiler, result, receipt, certificate


def test_reviewed_grammar_and_template_library_are_closed() -> None:
    dsl = TransformationDsl()
    assert dsl.revision == DSL_REVISION
    assert DeterministicToolDsl is TransformationDsl
    assert DETERMINISTIC_TOOL_DSL_REVISION == "DeterministicToolDsl@1"
    assert set(dsl.library.template_ids) >= {
        "select-and-rename",
        "scoped-path-normalize",
        "bounded-sort-limit",
        "canonical-digest",
        "filter-identifier-set",
        "prefix-scoped-path",
        "select-filter-limit",
        "repair-template-path-scope",
    }
    program = dsl.instantiate(
        "select-and-rename",
        {"fields": ("path", "kind"), "mapping": {"path": "scope_path"}},
    )
    assert program.grammar_revision == GRAMMAR_REVISION
    assert program.operations == (TransformationOp.SELECT_FIELDS, TransformationOp.RENAME_FIELDS)
    with pytest.raises(ToolGrammarError) as unknown_template:
        dsl.instantiate("invented-template", {})
    assert unknown_template.value.reason_code is ToolReason.UNKNOWN_TEMPLATE
    with pytest.raises(ToolSafetyError) as shell:
        TransformationStep(operation="ARBITRARY_SHELL")
    assert shell.value.reason_code is ToolReason.ARBITRARY_SHELL
    with pytest.raises(ToolSafetyError) as python:
        TransformationProgram.from_record(
            {"steps": [{"operation": "ARBITRARY_PYTHON", "parameters": {}}]}
        )
    assert python.value.reason_code is ToolReason.ARBITRARY_CODE
    assert "ARBITRARY_SHELL" in FORBIDDEN_TRANSFORMATION_OPS
    with pytest.raises(ToolGrammarError):
        TransformationStep(operation="invented-op")


def test_schema_effect_path_and_resource_bounds_fail_closed() -> None:
    with pytest.raises(ToolSynthesisError) as schema:
        _request(input_schema_ref="")
    assert schema.value.reason_code is ToolReason.SCHEMA_MISSING
    with pytest.raises(ToolSynthesisError) as effect:
        _request(effect_class=EffectClass.REPOSITORY_WRITE)
    assert effect.value.reason_code is ToolReason.EFFECT_FORBIDDEN
    assert EffectClass.REPOSITORY_WRITE not in ALLOWED_EFFECT_CLASSES
    with pytest.raises(ProcedureContractError):
        _request(scope_paths=("../secrets",))
    with pytest.raises(ToolGrammarError):
        ToolResourceEnvelope(max_steps=MAX_TOOL_STEPS + 1)
    with pytest.raises(ToolSynthesisError) as unknown_repair:
        _request(repair_template_ids=("template.invented",))
    assert unknown_repair.value.reason_code is ToolReason.UNKNOWN_TEMPLATE


def test_repeated_pure_transformations_yield_candidate_tools_only() -> None:
    first = _synthesize()
    second = _synthesize()
    assert first.spec.content_id == second.spec.content_id
    assert first.candidate.content_id == second.candidate.content_id
    assert first.optimized.content_id == second.optimized.content_id
    assert first.candidate.state is ArtifactState.CANDIDATE
    assert first.optimized.state is ArtifactState.CANDIDATE
    assert first.spec.state is ArtifactState.CANDIDATE
    assert first.candidate.representation == "dsl"
    assert first.optimized.representation == "optimized-python"
    assert first.candidate.validated is False
    assert first.optimized.certified is False
    assert first.spec.can_promote is False
    assert first.candidate.can_authorize is False
    assert first.optimized.can_grant_authority is False
    parsed_spec = parse_procedure_artifact(first.spec.to_dict())
    assert isinstance(parsed_spec, GeneratedToolSpec)
    assert parsed_spec == first.spec
    parsed_candidate = parse_procedure_artifact(first.candidate.to_dict())
    assert isinstance(parsed_candidate, GeneratedToolCandidate)
    with pytest.raises(FrozenInstanceError):
        first.candidate.state = ArtifactState.PROMOTED  # type: ignore[misc]


def test_dsl_interpreter_is_deterministic_and_scope_preserving() -> None:
    dsl = TransformationDsl()
    program = dsl.instantiate(
        "select-and-rename",
        {"fields": ("path", "kind"), "mapping": {"path": "scope_path"}},
    )
    output = dsl.interpret(program, _happy_mapping(), scope_paths=(SCOPE,))
    assert output == {"scope_path": f"{SCOPE}/tool_synthesis.py", "kind": "keep"}
    again = dsl.interpret(program, _happy_mapping(), scope_paths=(SCOPE,))
    assert again == output
    path_program = dsl.instantiate("scoped-path-normalize", {"field": "path"})
    with pytest.raises(ToolSafetyError) as escaped:
        dsl.interpret(
            path_program,
            {"path": "other_pkg/outside.py", "kind": "keep"},
            scope_paths=(SCOPE,),
        )
    assert escaped.value.reason_code is ToolReason.PATH_ESCAPE


def test_optimized_python_is_restricted_and_never_executes_arbitrary_code() -> None:
    compiler = GeneratedToolCompiler()
    result = compiler.synthesize(_request())
    source = result.optimized.optimized_translation
    assert "def transform(value, env):" in source
    assert "import" not in source
    assert "exec" not in source
    assert "os.system" not in source
    with pytest.raises(ToolSafetyError) as injected:
        compiler.dsl.interpret_python(
            "import os\ndef transform(value, env):\n    return os.system('true')\n",
            _happy_mapping(),
            scope_paths=(SCOPE,),
        )
    assert injected.value.reason_code is ToolReason.ARBITRARY_CODE
    with pytest.raises(ToolSafetyError):
        compiler.dsl.interpret_python(
            "def transform(value, env):\n    return __import__('os')\n",
            _happy_mapping(),
            scope_paths=(SCOPE,),
        )


def test_translation_validator_requires_exact_differential_equivalence() -> None:
    compiler, result, receipt, certificate = _qualify()
    assert receipt.status is TranslationStatus.ACCEPTED
    assert receipt.exact is True
    assert receipt.reason_code is ToolReason.ACCEPTED
    assert receipt.state is ArtifactState.VERIFIED
    assert receipt.can_promote is False
    assert receipt.can_authorize is False
    assert receipt.validator_revision == VALIDATOR_REVISION
    assert receipt.adversarial_fixture_ids
    assert "happy.select" in receipt.matched_fixture_ids
    assert not receipt.mismatched_fixture_ids
    parsed = TranslationReceipt.from_dict(receipt.to_dict())
    assert parsed == receipt
    dsl_out = compiler.dsl.interpret(
        result.spec.program, _happy_mapping(), scope_paths=(SCOPE,)
    )
    fused_out = compiler.dsl.interpret(
        result.spec.optimized_program, _happy_mapping(), scope_paths=(SCOPE,)
    )
    python_out = compiler.dsl.interpret_python(
        result.optimized.optimized_translation,
        _happy_mapping(),
        scope_paths=(SCOPE,),
    )
    assert dsl_out == fused_out == python_out
    assert certificate.accepted is True
    assert certificate.state is ArtifactState.VERIFIED
    assert certificate.grants_promotion is False
    assert certificate.grants_authority is False
    assert certificate.spec_cid == result.spec.content_id
    assert certificate.translation_cid == receipt.content_id
    parsed_cert = parse_procedure_artifact(certificate.to_dict())
    assert isinstance(parsed_cert, GeneratedToolCertificate)


def test_optimized_python_is_promoted_only_after_validation_and_certificate() -> None:
    compiler, result, receipt, certificate = _qualify()
    rejected_certificate = replace(certificate, accepted=False, state=ArtifactState.REJECTED)
    with pytest.raises(ToolPromotionError) as missing_cert:
        compiler.promote_optimized(result.optimized, receipt, rejected_certificate)
    assert missing_cert.value.reason_code is ToolReason.CERTIFICATE_REJECTED
    with pytest.raises(ToolPromotionError):
        compiler.promote_optimized(result.candidate, receipt, certificate)
    rejected = TranslationValidator().validate(
        result.spec,
        result.candidate,
        replace(
            result.optimized,
            optimized_translation=(
                "def transform(value, env):\n    current = op_identity(value)\n    return current\n"
            ),
            translation_digest="",
            spec_cid=result.spec.content_id,
            program=result.optimized.program,
        ),
        result.fixtures,
    )
    assert rejected.status is TranslationStatus.REJECTED
    assert rejected.exact is False
    with pytest.raises(ToolPromotionError) as mismatch:
        compiler.promote_optimized(result.optimized, rejected, certificate)
    assert mismatch.value.reason_code in {
        ToolReason.TRANSLATION_MISMATCH,
        ToolReason.BINDING_MISMATCH,
    }
    promoted = compiler.promote_optimized(result.optimized, receipt, certificate)
    assert promoted.state is ArtifactState.PROMOTED
    assert promoted.validated is True
    assert promoted.certified is True
    assert promoted.can_authorize is False
    assert promoted.can_skip_validation is False
    assert promoted.representation == "optimized-python"
    decoded = GeneratedToolCandidate.from_dict(promoted.to_dict())
    assert decoded == promoted


def test_promotion_without_certificate_or_exact_translation_is_refused() -> None:
    compiler = GeneratedToolCompiler()
    result = compiler.synthesize(_request())
    receipt = compiler.validator.validate(
        result.spec, result.candidate, result.optimized, result.fixtures
    )
    with pytest.raises(ToolPromotionError) as missing:
        compiler.promote_optimized(result.optimized, receipt, None)  # type: ignore[arg-type]
    assert missing.value.reason_code is ToolReason.MISSING_CERTIFICATE
    with pytest.raises(ToolSynthesisError):
        compiler.certify(result, replace(receipt, status=TranslationStatus.REJECTED, exact=False, reason_code=ToolReason.TRANSLATION_MISMATCH, state=ArtifactState.REJECTED))


def test_fused_sort_limit_and_path_prefix_match_the_dsl() -> None:
    compiler = GeneratedToolCompiler()
    result = compiler.synthesize(
        _request(
            tool_id="tool.sort-limit",
            template_id="bounded-sort-limit",
            template_parameters={"key": "id", "limit": 2},
            fixtures=(
                ToolFixture(
                    fixture_id="happy.sort",
                    kind=FixtureKind.HAPPY,
                    payload=_happy_sequence(),
                ),
            ),
        )
    )
    assert result.spec.optimized_program.operations == (TransformationOp.SORT_LIMIT,)
    receipt = compiler.validator.validate(
        result.spec, result.candidate, result.optimized, result.fixtures
    )
    assert receipt.accepted
    output = compiler.dsl.interpret(
        result.spec.program, _happy_sequence(), scope_paths=(SCOPE,)
    )
    assert [item["id"] for item in output] == ["a", "b"]
    path_result = compiler.synthesize(
        _request(
            tool_id="tool.prefix-path",
            template_id="prefix-scoped-path",
            template_parameters={"field": "path", "prefix": SCOPE},
            fixtures=(
                ToolFixture(
                    fixture_id="happy.prefix",
                    kind=FixtureKind.HAPPY,
                    payload={"path": "tool_synthesis.py", "kind": "keep"},
                ),
            ),
        )
    )
    assert path_result.spec.optimized_program.operations == (TransformationOp.PREFIX_NORMALIZE,)
    path_receipt = compiler.validator.validate(
        path_result.spec,
        path_result.candidate,
        path_result.optimized,
        path_result.fixtures,
    )
    assert path_receipt.accepted
    prefixed = compiler.dsl.interpret(
        path_result.spec.program,
        {"path": "tool_synthesis.py", "kind": "keep"},
        scope_paths=(SCOPE,),
    )
    assert prefixed["path"] == f"{SCOPE}/tool_synthesis.py"


def test_repair_template_path_scope_reuses_approved_templates() -> None:
    assert "template.path-scope" in APPROVED_REPAIR_TEMPLATE_IDS
    result = _synthesize(
        tool_id="tool.repair-scope",
        template_id="repair-template-path-scope",
        template_parameters={
            "field": "path",
            "prefix": SCOPE,
            "fields": ("path", "kind"),
        },
        repair_template_ids=("template.path-scope", "template.import-purity"),
        fixtures=(
            ToolFixture(
                fixture_id="happy.repair",
                kind=FixtureKind.HAPPY,
                payload={"path": "tool_synthesis.py", "kind": "keep", "id": "x"},
            ),
        ),
    )
    assert result.spec.repair_template_ids == (
        "template.path-scope",
        "template.import-purity",
    )
    output = TransformationDsl().interpret(
        result.spec.program,
        {"path": "tool_synthesis.py", "kind": "keep", "id": "x"},
        scope_paths=(SCOPE,),
    )
    assert output == {"path": f"{SCOPE}/tool_synthesis.py", "kind": "keep"}


def test_adversarial_fixtures_are_required_and_refuse_injection() -> None:
    fixtures = default_adversarial_fixtures(scope_paths=(SCOPE,))
    kinds = {item.kind for item in fixtures}
    assert FixtureKind.ADVERSARIAL in kinds
    compiler, result, receipt, _certificate = _qualify()
    assert set(receipt.adversarial_fixture_ids) <= {item.fixture_id for item in result.fixtures}
    dsl = TransformationDsl()
    with pytest.raises(ToolSafetyError) as shell:
        dsl.interpret(
            result.spec.program,
            {"kind": "arbitrary-shell"},
            scope_paths=(SCOPE,),
        )
    assert shell.value.reason_code is ToolReason.ARBITRARY_SHELL
    with pytest.raises(ProcedureContractError):
        dsl.interpret(
            result.spec.program,
            {"path": "../secrets", "kind": "keep"},
            scope_paths=(SCOPE,),
        )
    happy_only = (
        ToolFixture(fixture_id="happy.only", kind=FixtureKind.HAPPY, payload=_happy_mapping()),
    )
    with pytest.raises(ToolTranslationError) as missing_adv:
        TranslationValidator().validate(
            result.spec, result.candidate, result.optimized, happy_only
        )
    assert missing_adv.value.reason_code is ToolReason.ADVERSARIAL_FAILURE


def test_certified_invocation_receipt_cannot_authorize_or_complete() -> None:
    compiler, result, _receipt, certificate = _qualify()
    receipt = compiler.invoke(
        result.spec,
        result.candidate,
        certificate,
        _happy_mapping(),
        fixture_id="happy.select",
    )
    assert isinstance(receipt, GeneratedToolInvocationReceipt)
    assert receipt.state is ArtifactState.CANDIDATE
    assert receipt.can_authorize is False
    assert receipt.can_establish_proof is False
    assert receipt.can_establish_postcondition is False
    assert receipt.can_establish_completion is False
    assert receipt.can_promote is False
    parsed = parse_procedure_artifact(receipt.to_dict())
    assert isinstance(parsed, GeneratedToolInvocationReceipt)
    promoted = compiler.promote_optimized(result.optimized, _receipt, certificate)
    optimized_receipt = compiler.invoke(
        result.spec,
        promoted,
        certificate,
        _happy_mapping(),
        use_optimized=True,
    )
    assert optimized_receipt.representation == "optimized-python"
    assert optimized_receipt.output_digest == receipt.output_digest
    with pytest.raises(ToolPromotionError):
        compiler.invoke(
            result.spec,
            result.optimized,
            certificate,
            _happy_mapping(),
            use_optimized=True,
        )


def test_unknown_fields_floats_and_executable_values_are_rejected() -> None:
    result = _synthesize()
    payload = result.spec.to_dict()
    payload["claim_completion"] = True
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        GeneratedToolSpec.from_dict(payload)
    with pytest.raises(ProcedureContractError):
        ToolFixture(fixture_id="bad.float", kind=FixtureKind.HAPPY, payload={"ratio": 0.5})
    with pytest.raises(ToolSafetyError):
        TransformationDsl().parse_program(
            {"steps": [{"operation": "eval", "parameters": {}}]}
        )


def test_filter_and_canonical_digest_templates_are_pure() -> None:
    compiler = GeneratedToolCompiler()
    filtered = compiler.synthesize(
        _request(
            tool_id="tool.filter",
            template_id="filter-identifier-set",
            template_parameters={"field": "kind", "allowed": ("keep",)},
            fixtures=(
                ToolFixture(
                    fixture_id="happy.filter",
                    kind=FixtureKind.HAPPY,
                    payload=_happy_sequence(),
                ),
            ),
        )
    )
    kept = compiler.dsl.interpret(
        filtered.spec.program, _happy_sequence(), scope_paths=(SCOPE,)
    )
    assert [item["id"] for item in kept] == ["a", "c"]
    digest_result = compiler.synthesize(
        _request(
            tool_id="tool.digest",
            template_id="canonical-digest",
            template_parameters={"fields": ("id", "kind")},
        )
    )
    digest = compiler.dsl.interpret(
        digest_result.spec.program, _happy_mapping(), scope_paths=(SCOPE,)
    )
    again = compiler.dsl.interpret(
        digest_result.spec.program, _happy_mapping(), scope_paths=(SCOPE,)
    )
    assert digest == again
    assert isinstance(digest, str) and digest
    receipt = compiler.validator.validate(
        digest_result.spec,
        digest_result.candidate,
        digest_result.optimized,
        digest_result.fixtures,
    )
    assert receipt.accepted


def test_resource_exhaustion_and_missing_tests_fail_closed() -> None:
    dsl = TransformationDsl()
    program = TransformationProgram(
        steps=tuple(
            TransformationStep(TransformationOp.IDENTITY)
            for _ in range(4)
        )
    )
    with pytest.raises(ToolSynthesisError) as exhausted:
        dsl.interpret(
            program,
            _happy_mapping(),
            scope_paths=(SCOPE,),
            resources=ToolResourceEnvelope(max_steps=2),
        )
    assert exhausted.value.reason_code is ToolReason.RESOURCE_EXCEEDED
    with pytest.raises(ToolGrammarError):
        TransformationProgram(steps=())
    library = TemplateLibrary()
    assert library.contains("select-and-rename")
    assert library.revision == "transformation-template-library@1"
    assert GeneratedToolCompiler().revision == COMPILER_REVISION
