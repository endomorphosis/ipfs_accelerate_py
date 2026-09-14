"""SAWM-021 deterministic repair operators and bounded CEGIS tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_cegis import (
    ANALYTICAL_PROCEDURE,
    CEGIS_PROCEDURE,
    MODEL_STRATEGY,
    PROGRAM_WORLD_CEGIS_INTERFACE,
    REPAIR_CANDIDATE_SCOPE_GATE_INTERFACE,
    REPAIR_COUNTEREXAMPLE_REFINER_INTERFACE,
    RESIDUAL_PROCEDURE,
    SAWM_CEGIS_REPAIR_EVIDENCE,
    ProgramWorldCEGIS,
    ProgramWorldCegisAuthorityError,
    ProgramWorldCegisBounds,
    ProgramWorldCounterexample,
    ProgramWorldRepairDisposition,
    ProgramWorldRepairRequest,
    RepairCandidateScopeGate,
    RepairCounterexampleRefiner,
    synthesize_program_world_repair,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators import (
    DEFAULT_PROTECTED_PATHS,
    EXISTING_OPERATOR_KIND_ALIASES,
    FAMILY_TO_OPERATORS,
    PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE,
    ProgramWorldDefectFamily,
    ProgramWorldOperatorDisposition,
    ProgramWorldOperatorKind,
    ProgramWorldRepairOperatorRegistry,
    apply_program_world_repair_operator,
    failure_fingerprint,
    list_program_world_repair_operators,
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

SCOPE = ("pkg/",)
SOURCE = "def ping():\n    return missing()\n"


def _names(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    return functions, classes


def _cx(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "counterexample_id": "cx:ping-missing",
        "path": "pkg/mod.py",
        "family": ProgramWorldDefectFamily.WRONG_SYMBOL.value,
        "witness": "missing",
    }
    fields.update(overrides)
    return fields


def _request(**overrides: Any) -> ProgramWorldRepairRequest:
    fields: dict[str, Any] = {
        "path": "pkg/mod.py",
        "source_text": SOURCE,
        "counterexample": _cx(),
        "admitted_scope": SCOPE,
        "symbol": "missing",
        "replacement": "pong",
        "proof_obligation_ids": ("obligation:preserve-public-compat",),
        "test_obligation_ids": ("obligation:no-new-test-counterexample",),
        "environment_binding_cid": "env:v1",
        "tree_id": "tree:current",
    }
    fields.update(overrides)
    return ProgramWorldRepairRequest.from_mapping(fields)


def test_public_interfaces_and_symbols_are_present() -> None:
    operator_fns, operator_cls = _names(OPERATORS_PATH)
    cegis_fns, cegis_cls = _names(CEGIS_PATH)
    assert "apply_program_world_repair_operator" in operator_fns
    assert "synthesize_program_world_repair" in cegis_fns
    assert "ProgramWorldRepairOperatorRegistry" in operator_cls
    assert "ProgramWorldCEGIS" in cegis_cls
    assert "RepairCandidateScopeGate" in cegis_cls
    assert "RepairCounterexampleRefiner" in cegis_cls
    assert PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE == (
        "ProgramWorldRepairOperatorRegistry@1"
    )
    assert PROGRAM_WORLD_CEGIS_INTERFACE == "ProgramWorldCEGIS@1"
    assert REPAIR_CANDIDATE_SCOPE_GATE_INTERFACE == "RepairCandidateScopeGate@1"
    assert REPAIR_COUNTEREXAMPLE_REFINER_INTERFACE == "RepairCounterexampleRefiner@1"
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
    assert inspect.isclass(cegis.ProgramWorldCEGIS)
    assert inspect.isclass(cegis.RepairCandidateScopeGate)
    assert inspect.isclass(cegis.RepairCounterexampleRefiner)


def test_closed_operator_vocabulary_extends_existing_kinds() -> None:
    registry = ProgramWorldRepairOperatorRegistry()
    kinds = {item.value for item in registry.kinds()}
    assert kinds == {item.value for item in ProgramWorldOperatorKind}
    assert kinds == set(EXISTING_OPERATOR_KIND_ALIASES)
    listed = list_program_world_repair_operators()
    assert {item.kind for item in listed} == set(registry.kinds())
    for spec in listed:
        assert spec.proposal_only is True
        assert spec.grants_write_authority is False
        assert spec.existing_kind_alias == EXISTING_OPERATOR_KIND_ALIASES[spec.kind.value]
    encoded = registry.to_dict()
    assert encoded["interface"] == PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE
    assert encoded["grants_write_authority"] is False
    assert FAMILY_TO_OPERATORS[ProgramWorldDefectFamily.UNSUPPORTED] == ()


def test_unique_exact_bytes_repair_is_deterministic() -> None:
    source = "value = 1\nkeep = 2\n"
    first = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text=source,
        before_span="value = 1",
        after_span="value = 2",
        admitted_scope=SCOPE,
    )
    second = apply_program_world_repair_operator(
        operator_kind="replace_exact_bytes",
        path="pkg/mod.py",
        source_text=source,
        before_span="value = 1",
        after_span="value = 2",
        admitted_scope=SCOPE,
    )
    assert first.disposition is ProgramWorldOperatorDisposition.UNIQUE
    assert first.sketch is not None
    assert first.after_source == "value = 2\nkeep = 2\n"
    assert first.sketch.sketch_cid == second.sketch.sketch_cid
    assert first.content_id == second.content_id
    assert first.proposal_only is True
    assert first.independently_admitted is False
    assert first.grants_write_authority is False


def test_non_unique_span_abstains_as_ambiguous() -> None:
    source = "value = 1\nvalue = 1\n"
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text=source,
        before_span="value = 1",
        after_span="value = 2",
        admitted_scope=SCOPE,
    )
    assert result.disposition is ProgramWorldOperatorDisposition.AMBIGUOUS
    assert result.sketch is None
    assert "span_not_unique" in result.reason_codes


def test_rename_exact_symbol_is_unique_and_token_aware() -> None:
    source = "def ping():\n    missing = missing\n    return missing\n"
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL,
        path="pkg/mod.py",
        source_text=source,
        symbol="missing",
        replacement="pong",
        admitted_scope=SCOPE,
    )
    assert result.unique
    assert "missing" not in result.after_source
    assert "pong" in result.after_source
    assert result.sketch is not None
    assert result.sketch.occurrence_count == 3


def test_add_import_inserts_canonically() -> None:
    source = "import os\n\ndef ping():\n    return pong()\n"
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.ADD_IMPORT,
        path="pkg/mod.py",
        source_text=source,
        import_module="pkg.other",
        import_name="pong",
        admitted_scope=SCOPE,
    )
    assert result.unique
    assert "from pkg.other import pong" in result.after_source.splitlines()
    again = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.ADD_IMPORT,
        path="pkg/mod.py",
        source_text=result.after_source,
        import_module="pkg.other",
        import_name="pong",
        admitted_scope=SCOPE,
    )
    assert again.disposition is ProgramWorldOperatorDisposition.ABSTAINED
    assert "import_already_present" in again.reason_codes


def test_equality_rewrite_uses_landed_egraph() -> None:
    source = "value = x + 0\n"
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.EQUALITY_REWRITE,
        path="pkg/mod.py",
        source_text=source,
        source_term="x + 0",
        target_term="x",
        admitted_scope=SCOPE,
    )
    assert result.unique
    assert result.after_source == "value = x\n"
    unproved = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.EQUALITY_REWRITE,
        path="pkg/mod.py",
        source_text=source,
        source_term="x + 0",
        target_term="x + 1",
        admitted_scope=SCOPE,
    )
    assert unproved.disposition is ProgramWorldOperatorDisposition.ABSTAINED
    assert unproved.sketch is None


def test_analytical_procedures_run_before_cegis_and_model() -> None:
    model = {
        "operator_kind": "replace_exact_bytes",
        "before_span": "missing()",
        "after_span": "model_fix()",
        "grants_write_authority": False,
    }
    receipt = synthesize_program_world_repair(_request(model_proposal=model))
    assert receipt.analytical_first is True
    assert receipt.procedures_attempted[0] == ANALYTICAL_PROCEDURE
    assert receipt.selected_procedure == ANALYTICAL_PROCEDURE
    assert CEGIS_PROCEDURE not in receipt.procedures_attempted
    assert RESIDUAL_PROCEDURE not in receipt.procedures_attempted
    assert receipt.disposition is ProgramWorldRepairDisposition.UNIQUE
    assert receipt.application is not None
    assert receipt.application.operator_kind is ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL
    assert "model_fix()" not in (receipt.application.after_source or "")
    assert "pong()" in (receipt.application.after_source or "")
    assert receipt.model_call_count == 0
    assert receipt.proposal_only is True
    assert receipt.independently_admitted is False
    assert receipt.grants_write_authority is False


def test_cegis_enumerates_after_non_unique_analytical_family() -> None:
    source = "missing = 1\nmissing = 1\ndef ping():\n    return missing\n"
    receipt = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=_cx(family=ProgramWorldDefectFamily.UNIQUE_BYTES.value),
            before_span="missing = 1",
            after_span="pong = 1",
            symbol="missing",
            replacement="pong",
        )
    )
    assert receipt.analytical_first is True
    assert receipt.procedures_attempted[0] == ANALYTICAL_PROCEDURE
    assert CEGIS_PROCEDURE in receipt.procedures_attempted
    assert receipt.disposition in {
        ProgramWorldRepairDisposition.UNIQUE,
        ProgramWorldRepairDisposition.SUPPORTED,
    }
    assert receipt.application is not None
    assert receipt.application.operator_kind is ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL
    assert "missing" not in receipt.application.after_source


def test_scope_gate_rejects_out_of_scope_and_protected_paths() -> None:
    out_of_scope = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="other/mod.py",
        source_text="value = 1\n",
        before_span="value = 1",
        after_span="value = 2",
        admitted_scope=SCOPE,
    )
    assert out_of_scope.disposition is ProgramWorldOperatorDisposition.OUT_OF_SCOPE
    protected = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="docs/architecture/semantic_addressed_world_model.todo.md",
        source_text="- Status: todo\n",
        before_span="todo",
        after_span="done",
        admitted_scope=("docs/architecture/",),
    )
    assert protected.disposition is ProgramWorldOperatorDisposition.REJECTED
    assert "protected_path" in protected.reason_codes
    unspecified = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text="value = 1\n",
        before_span="value = 1",
        after_span="value = 2",
        admitted_scope=(),
    )
    assert unspecified.disposition is ProgramWorldOperatorDisposition.OUT_OF_SCOPE
    assert "scope_unspecified" in unspecified.reason_codes
    assert DEFAULT_PROTECTED_PATHS


def test_cannot_weaken_tests() -> None:
    before = (
        "def test_ping():\n"
        "    assert ping() == 1\n"
        "    assert ping() != 0\n"
    )
    after = (
        "def test_ping():\n"
        "    assert True\n"
        "    pytest.skip('repaired')\n"
    )
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="test/api/semantic_world/test_ping.py",
        source_text=before,
        before_span=before,
        after_span=after,
        admitted_scope=("test/api/semantic_world/",),
    )
    assert result.disposition is ProgramWorldOperatorDisposition.REJECTED
    assert any(code.startswith("test_weakening") for code in result.reason_codes)
    deleted = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="test/api/semantic_world/test_ping.py",
        source_text=before,
        before_span="def test_ping():\n    assert ping() == 1\n    assert ping() != 0\n",
        after_span="def helper():\n    return 1\n",
        admitted_scope=("test/api/semantic_world/",),
    )
    assert deleted.disposition is ProgramWorldOperatorDisposition.REJECTED
    assert "test_weakening:test_removed" in deleted.reason_codes


def test_cannot_weaken_authority_or_trusted_paths() -> None:
    authority = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text="value = 1\n",
        before_span="value = 1",
        after_span="value = 2",
        admitted_scope=SCOPE,
        metadata={"grants_write_authority": True, "self_approve": True},
    )
    assert authority.disposition is ProgramWorldOperatorDisposition.REJECTED
    assert any(code.startswith("authority_claim:") for code in authority.reason_codes)
    trusted = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/authority.py",
        source_text="policy = 'deny'\n",
        before_span="deny",
        after_span="allow",
        admitted_scope=SCOPE,
        trusted_paths=("pkg/authority.py",),
    )
    assert trusted.disposition is ProgramWorldOperatorDisposition.REJECTED
    assert "trusted_path" in trusted.reason_codes
    receipt = synthesize_program_world_repair(
        _request(metadata={"completion_authority": True})
    )
    assert receipt.disposition is ProgramWorldRepairDisposition.REJECTED
    assert any("authority_claim" in code for code in receipt.reason_codes)


def test_arbitrary_execution_is_rejected() -> None:
    source = "def ping():\n    return 1\n"
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text=source,
        before_span="return 1",
        after_span="return eval('2')",
        admitted_scope=SCOPE,
    )
    assert result.disposition is ProgramWorldOperatorDisposition.REJECTED
    assert any(code.startswith("arbitrary_execution") for code in result.reason_codes)
    subprocess_import = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text=source,
        before_span=source,
        after_span="import subprocess\n\ndef ping():\n    subprocess.call(['true'])\n",
        admitted_scope=SCOPE,
    )
    assert subprocess_import.disposition is ProgramWorldOperatorDisposition.REJECTED


def test_unknown_operator_cannot_expand_grammar() -> None:
    with pytest.raises(ProgramWorldCegisAuthorityError):
        synthesize_program_world_repair(
            _request(operator_kinds=("llm_rewrite",))
        )
    receipt = synthesize_program_world_repair(
        _request(
            source_text="value = 1\nvalue = 1\n",
            counterexample=_cx(family=ProgramWorldDefectFamily.UNSUPPORTED.value),
            symbol="",
            replacement="",
            model_proposal={"operator_kind": "llm_rewrite", "after_span": "pass"},
        )
    )
    assert receipt.disposition is ProgramWorldRepairDisposition.REJECTED
    assert "grammar_expansion" in receipt.reason_codes
    assert receipt.model_call_count == 0
    assert receipt.independently_admitted is False


def test_bounded_iterations_and_candidates() -> None:
    bounds = ProgramWorldCegisBounds(
        max_iterations=1,
        max_candidates_per_iteration=1,
        max_identical_failures=1,
        wall_time_ms=5_000,
    )
    receipt = synthesize_program_world_repair(
        _request(
            source_text="value = 1\nvalue = 1\n",
            counterexample=_cx(family=ProgramWorldDefectFamily.UNIQUE_BYTES.value),
            before_span="value = 1",
            after_span="value = 2",
            symbol="",
            replacement="",
            operator_kinds=("replace_exact_bytes",),
            bounds=bounds,
        )
    )
    assert receipt.disposition in {
        ProgramWorldRepairDisposition.BUDGET_EXHAUSTED,
        ProgramWorldRepairDisposition.RESIDUAL,
        ProgramWorldRepairDisposition.ABSTAINED,
        ProgramWorldRepairDisposition.IDENTICAL_FAILURE,
    }
    assert receipt.model_call_count == 0
    assert receipt.bounds.max_iterations == 1
    assert receipt.bounds.max_candidates_per_iteration == 1
    assert receipt.bounds.max_model_calls == 0
    assert "evidence_preserved" in receipt.reason_codes
    assert receipt.preserved_evidence_cids


def test_wall_time_bound_is_enforced() -> None:
    class Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            current = self.t
            self.t += 10.0
            return current

    bounds = ProgramWorldCegisBounds(max_iterations=8, wall_time_ms=1)
    receipt = ProgramWorldCEGIS(monotonic_clock=Clock()).synthesize(
        _request(
            source_text="value = 1\nvalue = 1\n",
            counterexample=_cx(family=ProgramWorldDefectFamily.UNIQUE_BYTES.value),
            before_span="value = 1",
            after_span="value = 2",
            symbol="",
            replacement="",
            operator_kinds=("replace_exact_bytes",),
            bounds=bounds,
        )
    )
    assert receipt.disposition is ProgramWorldRepairDisposition.BUDGET_EXHAUSTED
    assert "wall_time_exhausted" in receipt.reason_codes
    assert receipt.residual is not None


def test_counterexample_refinement_records_new_evidence() -> None:
    cx = ProgramWorldCounterexample.from_mapping(_cx())
    application = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        path="pkg/mod.py",
        source_text=SOURCE,
        before_span="missing()",
        after_span="eval('pong')",
        admitted_scope=SCOPE,
    )
    gate = RepairCandidateScopeGate()
    request = _request(before_span="missing()", after_span="eval('pong')")
    gates = gate.evaluate(application, request, before_source=SOURCE)
    refined, progress = RepairCounterexampleRefiner().refine(
        cx, failed_application=application, gate_results=gates
    )
    assert progress is True
    assert refined.evidence_cid != cx.evidence_cid
    assert refined.counterexample_id == cx.counterexample_id
    assert any("gate_fail" in item for item in refined.effect_markers + refined.abstract_markers)
    again, second = RepairCounterexampleRefiner().refine(refined)
    assert second is False
    assert again.evidence_cid == refined.evidence_cid


def test_failures_preserve_evidence_and_block_identical_retries() -> None:
    source = "value = 1\nvalue = 1\n"
    first = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=_cx(family=ProgramWorldDefectFamily.UNIQUE_BYTES.value),
            before_span="value = 1",
            after_span="value = 2",
            symbol="",
            replacement="",
            operator_kinds=("replace_exact_bytes",),
        )
    )
    assert first.disposition in {
        ProgramWorldRepairDisposition.RESIDUAL,
        ProgramWorldRepairDisposition.ABSTAINED,
        ProgramWorldRepairDisposition.IDENTICAL_FAILURE,
        ProgramWorldRepairDisposition.BUDGET_EXHAUSTED,
    }
    assert first.failure_fingerprints
    assert first.preserved_evidence_cids
    assert first.residual is not None
    assert first.residual.requires_new_evidence is True
    assert first.residual.model_may_self_approve is False
    second = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=_cx(family=ProgramWorldDefectFamily.UNIQUE_BYTES.value),
            before_span="value = 1",
            after_span="value = 2",
            symbol="",
            replacement="",
            operator_kinds=("replace_exact_bytes",),
            prior_failure_fingerprints=first.failure_fingerprints,
            prior_evidence_cids=first.preserved_evidence_cids,
            bounds=ProgramWorldCegisBounds(max_iterations=2, max_identical_failures=1),
        )
    )
    assert second.disposition in {
        ProgramWorldRepairDisposition.IDENTICAL_FAILURE,
        ProgramWorldRepairDisposition.RESIDUAL,
        ProgramWorldRepairDisposition.ABSTAINED,
        ProgramWorldRepairDisposition.BUDGET_EXHAUSTED,
    }
    assert "evidence_preserved" in second.reason_codes
    assert set(first.preserved_evidence_cids) <= set(second.preserved_evidence_cids)


def test_identical_model_retry_without_new_evidence_is_blocked() -> None:
    source = "value = 1\nvalue = 1\n"
    cx = _cx(family=ProgramWorldDefectFamily.UNSUPPORTED.value)
    evidence = ProgramWorldCounterexample.from_mapping(cx, default_path="pkg/mod.py")
    fingerprint = failure_fingerprint(
        counterexample_id=evidence.counterexample_id,
        operator_kind=MODEL_STRATEGY,
        sketch_cid="",
        evidence_cid=evidence.evidence_cid,
        strategy=MODEL_STRATEGY,
    )
    receipt = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=cx,
            symbol="",
            replacement="",
            prior_failure_fingerprints=(fingerprint,),
            prior_evidence_cids=(evidence.evidence_cid,),
            model_proposal={
                "operator_kind": "replace_exact_bytes",
                "before_span": "value = 1",
                "after_span": "value = 2",
            },
        )
    )
    assert receipt.disposition is ProgramWorldRepairDisposition.REJECTED
    assert "identical_model_retry_without_new_evidence" in receipt.reason_codes
    assert receipt.model_call_count == 0
    assert receipt.residual is not None
    assert receipt.residual.requires_new_evidence is True


def test_strategy_change_is_allowed_when_operator_differs() -> None:
    source = "def ping():\n    return missing()\n"
    first = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=_cx(family=ProgramWorldDefectFamily.UNIQUE_BYTES.value),
            before_span="missing()",
            after_span="eval('x')",
            operator_kinds=("replace_exact_bytes",),
            symbol="",
            replacement="",
        )
    )
    assert first.failure_fingerprints
    second = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=_cx(family=ProgramWorldDefectFamily.WRONG_SYMBOL.value),
            symbol="missing",
            replacement="pong",
            prior_failure_fingerprints=first.failure_fingerprints,
            prior_evidence_cids=first.preserved_evidence_cids,
        )
    )
    assert second.disposition is ProgramWorldRepairDisposition.UNIQUE
    assert second.application is not None
    assert second.application.operator_kind is ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL


def test_unsupported_family_becomes_residual_question() -> None:
    receipt = synthesize_program_world_repair(
        _request(
            counterexample=_cx(family=ProgramWorldDefectFamily.UNSUPPORTED.value),
            symbol="",
            replacement="",
            operator_kinds=("replace_exact_bytes",),
        )
    )
    assert receipt.disposition is ProgramWorldRepairDisposition.RESIDUAL
    assert receipt.residual is not None
    assert receipt.residual.decision_relevant is True
    assert receipt.residual.model_may_propose is True
    assert receipt.residual.model_may_self_approve is False
    assert receipt.application is None
    assert receipt.selected_procedure == RESIDUAL_PROCEDURE
    assert receipt.procedures_attempted[0] == ANALYTICAL_PROCEDURE


def test_model_proposal_after_exhaustion_stays_proposal_only() -> None:
    source = "value = unique_token_xyz\n"
    receipt = synthesize_program_world_repair(
        _request(
            source_text=source,
            counterexample=_cx(family=ProgramWorldDefectFamily.UNSUPPORTED.value),
            symbol="",
            replacement="",
            model_proposal={
                "operator_kind": "replace_exact_bytes",
                "before_span": "unique_token_xyz",
                "after_span": "repaired_token_xyz",
            },
        )
    )
    assert receipt.disposition is ProgramWorldRepairDisposition.SUPPORTED
    assert receipt.selected_procedure == RESIDUAL_PROCEDURE
    assert receipt.procedures_attempted[0] == ANALYTICAL_PROCEDURE
    assert CEGIS_PROCEDURE in receipt.procedures_attempted
    assert receipt.application is not None
    assert "repaired_token_xyz" in receipt.application.after_source
    assert receipt.proposal_only is True
    assert receipt.independently_admitted is False
    assert receipt.grants_write_authority is False
    assert receipt.grants_completion_authority is False
    assert "cannot_self_approve" in receipt.reason_codes
    assert receipt.model_call_count == 0


def test_restore_tracked_artifact_requires_digest() -> None:
    source = "old = 1\n"
    restored = "new = 2\n"
    mismatch = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.RESTORE_TRACKED_ARTIFACT,
        path="pkg/mod.py",
        source_text=source,
        after_span=restored,
        artifact_digest="sha256:deadbeef",
        admitted_scope=SCOPE,
    )
    assert mismatch.disposition is ProgramWorldOperatorDisposition.REJECTED
    import hashlib

    digest = "sha256:" + hashlib.sha256(restored.encode("utf-8")).hexdigest()
    ok = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.RESTORE_TRACKED_ARTIFACT,
        path="pkg/mod.py",
        source_text=source,
        after_span=restored,
        artifact_digest=digest,
        admitted_scope=SCOPE,
        trusted_paths=("pkg/mod.py",),
    )
    assert ok.unique
    assert ok.after_source == restored


def test_add_argument_requires_unique_function() -> None:
    source = "def ping(x):\n    return x\n"
    result = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.ADD_ARGUMENT,
        path="pkg/mod.py",
        source_text=source,
        symbol="ping",
        argument="y",
        admitted_scope=SCOPE,
    )
    assert result.unique
    assert "def ping(x, y):" in result.after_source
    duplicate = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.ADD_ARGUMENT,
        path="pkg/mod.py",
        source_text="def ping(x):\n    return x\ndef ping(z):\n    return z\n",
        symbol="ping",
        argument="y",
        admitted_scope=SCOPE,
    )
    assert duplicate.disposition is ProgramWorldOperatorDisposition.AMBIGUOUS


def test_receipt_roundtrip_is_content_addressed() -> None:
    receipt = synthesize_program_world_repair(_request())
    encoded = receipt.to_dict()
    assert encoded["interface"] == PROGRAM_WORLD_CEGIS_INTERFACE
    assert encoded["evidence"] == SAWM_CEGIS_REPAIR_EVIDENCE
    assert encoded["analytical_first"] is True
    assert encoded["model_call_count"] == 0
    assert encoded["proposal_only"] is True
    again = synthesize_program_world_repair(_request())
    assert again.content_id == receipt.content_id
    bounds = ProgramWorldCegisBounds.from_mapping(receipt.bounds.to_dict())
    assert bounds.content_id == receipt.bounds.content_id


def test_scope_gate_and_refiner_are_usable_independently() -> None:
    application = apply_program_world_repair_operator(
        operator_kind=ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL,
        path="pkg/mod.py",
        source_text=SOURCE,
        symbol="missing",
        replacement="pong",
        admitted_scope=SCOPE,
    )
    request = _request()
    gates = RepairCandidateScopeGate().evaluate(
        application, request, before_source=SOURCE
    )
    assert gates
    assert {item.kind.value for item in gates} >= {
        "scope",
        "type",
        "effect",
        "proof",
        "test",
        "authority",
        "protected_path",
    }
    assert all(item.passed for item in gates)
    cx = ProgramWorldCounterexample.from_mapping(_cx())
    refined, progress = RepairCounterexampleRefiner().refine(cx)
    assert progress is False
    assert refined.evidence_cid == cx.evidence_cid


def test_no_llm_or_provider_imports_in_source() -> None:
    for path in (OPERATORS_PATH, CEGIS_PATH):
        text = path.read_text(encoding="utf-8").casefold()
        assert "openai" not in text
        assert "anthropic" not in text
        assert "llm_router" not in text
        assert "subprocess" not in text or "forbidden" in text
