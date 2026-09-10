"""SAWM-021 deterministic repair operators and bounded CEGIS tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_cegis import (
    PROGRAM_WORLD_CEGIS_INTERFACE,
    CegisBounds,
    CegisDisposition,
    CegisRoute,
    CounterexampleKind,
    ProgramWorldCEGIS,
    ProgramWorldCegisAuthorityError,
    ProgramWorldCounterexample,
    RepairCounterexampleRefiner,
    synthesize_program_world_repair,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators import (
    CLOSED_OPERATOR_VOCABULARY,
    DEFAULT_PROTECTED_PATHS,
    PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE,
    SAWM_CEGIS_REPAIR_EVIDENCE,
    OperatorApplicationDisposition,
    OperatorReason,
    ProgramWorldRepairAuthorityError,
    ProgramWorldRepairOperatorKind,
    ProgramWorldRepairOperatorRegistry,
    ProgramWorldRepairStaleError,
    RepairCandidateScopeGate,
    ScopeGateDisposition,
    apply_program_world_repair_operator,
    build_default_program_world_operator_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATORS_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/autonomous_repair/program_world_operators.py"
)
CEGIS_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/autonomous_repair/program_world_cegis.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _env() -> str:
    return _cid("env-v1")


def _names(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    return functions, classes


def _apply(**overrides: Any):
    fields: dict[str, Any] = {
        "operator": ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN,
        "source": "def ping():\n    return x + 0\n",
        "path": "pkg/mod.py",
        "write_scope": ("pkg/",),
        "environment_binding_cid": _env(),
        "before": "x + 0",
        "after": "x",
    }
    fields.update(overrides)
    return apply_program_world_repair_operator(**fields)


def _cx(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "counterexample_id": "cx:span-1",
        "kind": CounterexampleKind.SPAN.value,
        "path": "pkg/mod.py",
        "before": "x + 0",
        "expected_after": "x",
        "evidence_cid": _cid("cx-span-1"),
        "obligation_id": "ob:span-1",
    }
    payload.update(overrides)
    return payload


def _synthesize(**overrides: Any):
    fields: dict[str, Any] = {
        "source": "def ping():\n    return x + 0\n",
        "path": "pkg/mod.py",
        "write_scope": ("pkg/",),
        "environment_binding_cid": _env(),
        "counterexamples": [_cx()],
        "obligations": ("ob:span-1",),
    }
    fields.update(overrides)
    return synthesize_program_world_repair(**fields)


def test_public_interfaces_and_symbols_are_present() -> None:
    operator_fns, operator_cls = _names(OPERATORS_PATH)
    cegis_fns, cegis_cls = _names(CEGIS_PATH)
    assert "apply_program_world_repair_operator" in operator_fns
    assert "synthesize_program_world_repair" in cegis_fns
    assert "ProgramWorldRepairOperatorRegistry" in operator_cls
    assert "RepairCandidateScopeGate" in operator_cls
    assert "ProgramWorldCEGIS" in cegis_cls
    assert "RepairCounterexampleRefiner" in cegis_cls
    assert PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE == (
        "ProgramWorldRepairOperatorRegistry@1"
    )
    assert PROGRAM_WORLD_CEGIS_INTERFACE == "ProgramWorldCEGIS@1"
    assert SAWM_CEGIS_REPAIR_EVIDENCE == "sawm/cegis-repair@1"


def test_import_has_no_io_or_thread_side_effects() -> None:
    before = {thread.name for thread in threading.enumerate()}
    operators = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators"
    )
    cegis = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_cegis"
    )
    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert inspect.isfunction(operators.apply_program_world_repair_operator)
    assert inspect.isfunction(cegis.synthesize_program_world_repair)
    assert inspect.isclass(operators.ProgramWorldRepairOperatorRegistry)
    assert inspect.isclass(operators.RepairCandidateScopeGate)
    assert inspect.isclass(cegis.ProgramWorldCEGIS)
    assert inspect.isclass(cegis.RepairCounterexampleRefiner)


def test_closed_operator_vocabulary_is_finite_and_rejects_unknown_kinds() -> None:
    registry = build_default_program_world_operator_registry()
    assert set(registry.kinds()) == set(CLOSED_OPERATOR_VOCABULARY)
    assert set(CLOSED_OPERATOR_VOCABULARY) == {
        "replace_exact_span",
        "rename_exact_symbol",
        "add_unique_argument",
        "add_unique_import",
        "add_unique_registration",
        "equality_rewrite",
        "restore_tracked_artifact",
    }
    receipt = _apply(operator="llm_freeform_patch", after="y")
    assert receipt.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.OPERATOR_NOT_REVIEWED.value in receipt.reason_codes
    assert receipt.grants_write_authority is False
    with pytest.raises(ProgramWorldCegisAuthorityError):
        _synthesize(operators=("llm_freeform_patch",))


def test_unique_exact_span_replacement_is_deterministic() -> None:
    first = _apply()
    second = _apply()
    assert first.disposition == OperatorApplicationDisposition.UNIQUE_REPAIR.value
    assert first.unique
    assert first.sketch is not None
    assert first.sketch.after_source == "def ping():\n    return x\n"
    assert first.sketch.proposal_only is True
    assert first.independently_admitted is False
    assert first.grants_write_authority is False
    assert first.grants_proof_authority is False
    assert first.grants_semantic_authority is False
    assert first.content_id == second.content_id
    assert first.sketch.sketch_cid == second.sketch.sketch_cid
    encoded = first.to_dict()
    assert encoded["proposal_only"] is True
    assert encoded["independently_admitted"] is False


def test_ambiguous_multi_match_abstains() -> None:
    source = "def ping():\n    return x + 0 + (x + 0)\n"
    receipt = _apply(source=source, before="x + 0", after="x")
    assert receipt.disposition == OperatorApplicationDisposition.AMBIGUOUS.value
    assert OperatorReason.AMBIGUOUS_MATCH.value in receipt.reason_codes
    assert receipt.sketch is None
    assert receipt.residual_question


def test_analytical_unique_repair_runs_before_cegis_and_model() -> None:
    model = {
        "operator": "replace_exact_span",
        "before": "x + 0",
        "after": "z",
        "evidence_cid": _cid("model-1"),
    }
    result = _synthesize(model_proposal=model)
    assert result.analytical_ran_first is True
    assert result.model_calls == 0
    assert result.disposition == CegisDisposition.UNIQUE_REPAIR.value
    assert result.route == CegisRoute.ANALYTICAL_UNIQUE.value
    assert result.sketch is not None
    assert result.sketch.after_source == "def ping():\n    return x\n"
    assert "z" not in result.sketch.after_source
    assert result.proposal_only is True
    again = ProgramWorldCEGIS(expected_environment_binding_cid=_env()).synthesize(
        source="def ping():\n    return x + 0\n",
        path="pkg/mod.py",
        write_scope=("pkg/",),
        environment_binding_cid=_env(),
        counterexamples=[_cx()],
        model_proposal=model,
    )
    assert again.route == CegisRoute.ANALYTICAL_UNIQUE.value
    assert again.sketch is not None
    assert again.sketch.sketch_cid == result.sketch.sketch_cid


def test_equality_rewrite_uses_unique_normal_form() -> None:
    receipt = _apply(
        operator=ProgramWorldRepairOperatorKind.EQUALITY_REWRITE,
        before="x + 0",
        after="",
    )
    assert receipt.disposition == OperatorApplicationDisposition.UNIQUE_REPAIR.value
    assert receipt.sketch is not None
    assert receipt.sketch.after == "x"
    assert "x + 0" not in receipt.sketch.after_source


def test_unique_argument_registration_and_restore_operators() -> None:
    added = _apply(
        operator=ProgramWorldRepairOperatorKind.ADD_UNIQUE_ARGUMENT,
        source="def ping():\n    return target()\n",
        before="target",
        after="",
        argument_name="timeout",
        argument_value="1",
    )
    assert added.unique
    assert added.sketch is not None
    assert "target(timeout=1)" in added.sketch.after_source
    registered = _apply(
        operator=ProgramWorldRepairOperatorKind.ADD_UNIQUE_REGISTRATION,
        source="REGISTRY = {\n    'ping': ping,\n}\n",
        before="",
        after="",
        registration_anchor="    'ping': ping,",
        registration_payload="    'pong': pong,",
    )
    assert registered.unique
    assert registered.sketch is not None
    assert "'pong': pong" in registered.sketch.after_source
    restored = _apply(
        operator=ProgramWorldRepairOperatorKind.RESTORE_TRACKED_ARTIFACT,
        source="def ping():\n    return 2\n",
        before="",
        after="def ping():\n    return 1\n",
    )
    assert restored.unique
    assert restored.sketch is not None
    assert restored.sketch.after_source == "def ping():\n    return 1\n"


def test_unique_symbol_rename_and_unique_import() -> None:
    renamed = _apply(
        operator=ProgramWorldRepairOperatorKind.RENAME_EXACT_SYMBOL,
        source="def ping():\n    return pong()\n",
        before="",
        after="",
        symbol_from="pong",
        symbol_to="pong_v2",
    )
    assert renamed.unique
    assert renamed.sketch is not None
    assert "pong_v2()" in renamed.sketch.after_source
    imported = _apply(
        operator=ProgramWorldRepairOperatorKind.ADD_UNIQUE_IMPORT,
        source="def ping():\n    return 1\n",
        before="",
        after="",
        import_module="pkg.util",
        import_name="helper",
    )
    assert imported.unique
    assert imported.sketch is not None
    assert "from pkg.util import helper" in imported.sketch.after_source


def test_candidates_must_stay_in_write_scope() -> None:
    receipt = _apply(path="other/mod.py", write_scope=("pkg/",))
    assert receipt.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.PATH_NOT_IN_SCOPE.value in receipt.reason_codes
    assert receipt.scope is not None
    assert receipt.scope.in_scope is False
    result = _synthesize(path="other/mod.py", write_scope=("pkg/",))
    assert result.disposition in {
        CegisDisposition.RESIDUAL.value,
        CegisDisposition.ABSTAINED.value,
        CegisDisposition.REJECTED.value,
        CegisDisposition.IDENTICAL_RETRY_BLOCKED.value,
    }
    assert result.sketch is None
    assert any(
        OperatorReason.PATH_NOT_IN_SCOPE.value in item.reason_codes
        for item in result.failed_attempts
    )


def test_protected_and_trusted_paths_cannot_be_edited() -> None:
    protected = DEFAULT_PROTECTED_PATHS[2]
    receipt = _apply(path=protected, write_scope=("docs/", "pkg/"))
    assert receipt.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.PROTECTED_PATH.value in receipt.reason_codes
    result = _synthesize(
        path=protected,
        write_scope=("docs/", "pkg/"),
        counterexamples=[_cx(path=protected)],
    )
    assert result.sketch is None
    assert any(
        OperatorReason.PROTECTED_PATH.value in item.reason_codes
        for item in result.failed_attempts
    )


def test_test_weakening_is_rejected() -> None:
    source = "def test_ping():\n    assert ping() == 1\n"
    skip = _apply(
        source=source,
        path="test/api/test_ping.py",
        write_scope=("test/",),
        before="def test_ping():\n    assert ping() == 1\n",
        after="import pytest\n@pytest.mark.skip\ndef test_ping():\n    assert ping() == 1\n",
    )
    assert skip.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.TEST_WEAKENING.value in skip.reason_codes
    dropped = _apply(
        source=source,
        path="test/api/test_ping.py",
        write_scope=("test/",),
        before="    assert ping() == 1\n",
        after="    pass\n",
    )
    assert dropped.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.TEST_WEAKENING.value in dropped.reason_codes


def test_arbitrary_execution_is_rejected() -> None:
    receipt = _apply(
        before="return x + 0",
        after="return eval('x')",
    )
    assert receipt.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.ARBITRARY_EXECUTION.value in receipt.reason_codes
    shell = _apply(
        before="return x + 0",
        after="return subprocess.call('rm -rf /', shell=True)",
    )
    assert shell.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.ARBITRARY_EXECUTION.value in shell.reason_codes


def test_authority_claims_and_obligation_waivers_are_rejected() -> None:
    claimed = _apply(metadata={"write_authority": True, "admission": True})
    assert claimed.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.AUTHORITY_CLAIM.value in claimed.reason_codes
    receipt = _apply(metadata={"skip_proof": True, "waive_obligations": ["ob:span-1"]})
    assert receipt.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.OBLIGATION_WAIVER.value in receipt.reason_codes
    widened = _apply(metadata={"extra_paths": ["other/mod.py"]})
    assert widened.disposition == OperatorApplicationDisposition.REJECTED.value
    assert OperatorReason.SCOPE_WIDENING.value in widened.reason_codes


def test_scope_gate_enforces_type_effect_proof_and_test_preservation() -> None:
    gate = RepairCandidateScopeGate(
        write_scope=("pkg/",),
        required_effects=("pure_local",),
        required_obligations=("ob:span-1",),
        required_tests=("test_ping",),
    )
    admitted = gate.admit(
        path="pkg/mod.py",
        before_source="def ping():\n    return x + 0\n",
        after_source="def ping():\n    return x\n",
        obligations=("ob:span-1",),
        tests=("test_ping",),
        effects=("pure_local",),
    )
    assert admitted.disposition == ScopeGateDisposition.ADMITTED.value
    dropped = gate.admit(
        path="pkg/mod.py",
        before_source="def ping():\n    return x + 0\n",
        after_source="def ping():\n    return x\n",
        obligations=(),
        tests=(),
        effects=("network",),
    )
    assert dropped.disposition != ScopeGateDisposition.ADMITTED.value
    assert OperatorReason.PROOF_GATE_FAILED.value in dropped.reason_codes
    assert OperatorReason.TEST_GATE_FAILED.value in dropped.reason_codes
    effectful = gate.admit(
        path="pkg/mod.py",
        before_source="def ping():\n    return x + 0\n",
        after_source="def ping():\n    print(x)\n    return x\n",
        obligations=("ob:span-1",),
        tests=("test_ping",),
    )
    assert OperatorReason.EFFECT_VIOLATION.value in effectful.reason_codes


def test_unsupported_language_and_stale_environment_fail_closed() -> None:
    js = _apply(language="javascript")
    assert js.disposition == OperatorApplicationDisposition.UNSUPPORTED.value
    assert any(code.startswith("language_unsupported") for code in js.reason_codes)
    with pytest.raises(ProgramWorldRepairStaleError):
        _apply(expected_environment_binding_cid=_cid("env-other"))


def test_bounded_iterations_candidates_and_time() -> None:
    bounds = CegisBounds(max_iterations=1, max_candidates=1, max_wall_time_ms=5_000)
    result = _synthesize(
        source="def ping():\n    return opaque(x)\n",
        counterexamples=[_cx(before="opaque(x)", expected_after="")],
        bounds=bounds,
    )
    assert result.bounds.max_iterations == 1
    assert result.bounds.max_candidates == 1
    assert result.bounds.max_model_calls == 0
    assert result.candidates_considered >= 1
    assert result.disposition in {
        CegisDisposition.BUDGET_EXHAUSTED.value,
        CegisDisposition.RESIDUAL.value,
        CegisDisposition.ABSTAINED.value,
        CegisDisposition.SYNTHESIZED.value,
        CegisDisposition.UNIQUE_REPAIR.value,
        CegisDisposition.IDENTICAL_RETRY_BLOCKED.value,
    }
    if result.disposition == CegisDisposition.BUDGET_EXHAUSTED.value:
        assert result.strategy_change_required is True
        assert result.residual_question
    with pytest.raises(Exception):
        CegisBounds(max_iterations=99)
    with pytest.raises(ProgramWorldCegisAuthorityError):
        CegisBounds(max_model_calls=1)


def test_counterexample_refinement_is_monotonic_and_preserves_evidence() -> None:
    original = ProgramWorldCounterexample.from_dict(_cx())
    sketch_source = "def ping():\n    return x + 0\n"
    child = RepairCounterexampleRefiner().refine(
        original,
        after_source=sketch_source,
        failed_sketch=None,
    )
    assert child is not None
    assert child.parent_id == original.counterexample_id
    assert child.evidence_cid != original.evidence_cid
    failed = _synthesize(
        source="def ping():\n    return opaque(x)\n",
        counterexamples=[_cx(before="opaque(x)", expected_after="eval('x')")],
    )
    assert failed.sketch is None
    assert failed.failed_attempts
    assert failed.counterexample_cids
    retry = _synthesize(
        source="def ping():\n    return opaque(x)\n",
        counterexamples=[_cx(before="opaque(x)", expected_after="eval('x')")],
        prior_failures=failed.failed_attempts,
    )
    assert len(retry.failed_attempts) >= len(failed.failed_attempts)
    assert retry.counterexample_cids
    assert failed.counterexample_cids[0] in retry.counterexample_cids


def test_failures_preserve_evidence_and_block_identical_model_retries() -> None:
    source = "def ping():\n    return opaque(x)\n"
    failing_cx = _cx(before="opaque(x)", expected_after="eval('x')")
    model = {
        "operator": "replace_exact_span",
        "before": "opaque(x)",
        "after": "eval('x')",
        "evidence_cid": _cid("cx-span-1"),
    }
    first = _synthesize(
        source=source,
        counterexamples=[failing_cx],
        model_proposal=model,
    )
    assert first.failed_attempts
    assert first.model_calls == 0
    assert first.sketch is None
    retry = _synthesize(
        source=source,
        counterexamples=[failing_cx],
        prior_failures=first.failed_attempts,
        model_proposal=model,
    )
    assert retry.disposition == CegisDisposition.IDENTICAL_RETRY_BLOCKED.value
    assert retry.strategy_change_required is True
    assert retry.failed_attempts
    assert len(retry.failed_attempts) >= len(first.failed_attempts)
    assert "identical_model_retry_without_new_evidence" in retry.reason_codes
    fresh = _synthesize(
        source=source,
        counterexamples=[
            _cx(
                before="opaque(x)",
                expected_after="hidden(x)",
                evidence_cid=_cid("cx-span-2"),
                counterexample_id="cx:span-2",
            )
        ],
        prior_failures=first.failed_attempts,
        model_proposal={
            "operator": "replace_exact_span",
            "before": "opaque(x)",
            "after": "hidden(x)",
            "evidence_cid": _cid("cx-span-2"),
        },
    )
    assert fresh.disposition in {
        CegisDisposition.UNIQUE_REPAIR.value,
        CegisDisposition.SYNTHESIZED.value,
    }
    assert fresh.sketch is not None
    assert "hidden(x)" in fresh.sketch.after_source
    assert fresh.independently_admitted is False


def test_unsupported_repair_becomes_explicit_residual_question() -> None:
    result = _synthesize(
        source="def ping():\n    return mystery(x)\n",
        counterexamples=[
            {
                "counterexample_id": "cx:unknown",
                "kind": CounterexampleKind.PROOF.value,
                "path": "pkg/mod.py",
                "evidence_cid": _cid("cx-unknown"),
                "witness": "no unique operator applies",
            }
        ],
    )
    assert result.disposition == CegisDisposition.RESIDUAL.value
    assert result.route == CegisRoute.RESIDUAL_QUESTION.value
    assert result.sketch is None
    assert result.residual_question
    assert OperatorReason.RESIDUAL_QUESTION.value in result.reason_codes
    assert result.analytical_ran_first is True
    assert result.proposal_only is True


def test_registry_identity_is_stable_and_descriptors_require_uniqueness() -> None:
    first = ProgramWorldRepairOperatorRegistry.default()
    second = ProgramWorldRepairOperatorRegistry.default()
    assert first.registry_cid == second.registry_cid
    for descriptor in first.descriptors:
        assert descriptor.uniqueness_required is True
        assert descriptor.review_ref == SAWM_CEGIS_REPAIR_EVIDENCE
        rebuilt = descriptor.from_dict(descriptor.to_dict())
        assert rebuilt.descriptor_cid == descriptor.descriptor_cid
    with pytest.raises(ProgramWorldRepairAuthorityError):
        type(first.descriptors[0])(
            operator_id="replace_exact_span",
            kind=ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN,
            uniqueness_required=False,
        )


def test_controller_binds_freshness_and_does_not_self_admit() -> None:
    controller = ProgramWorldCEGIS(expected_environment_binding_cid=_env())
    result = controller.synthesize(
        source="def ping():\n    return x + 0\n",
        path="pkg/mod.py",
        write_scope=("pkg/",),
        environment_binding_cid=_env(),
        counterexamples=[_cx()],
    )
    assert result.disposition == CegisDisposition.UNIQUE_REPAIR.value
    assert result.independently_admitted is False
    assert result.grants_write_authority is False
    with pytest.raises(ProgramWorldRepairStaleError):
        controller.synthesize(
            source="def ping():\n    return x + 0\n",
            path="pkg/mod.py",
            write_scope=("pkg/",),
            environment_binding_cid=_cid("env-other"),
            counterexamples=[_cx()],
        )
