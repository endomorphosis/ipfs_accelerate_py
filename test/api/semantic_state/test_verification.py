"""SCH-008 verification runner, typed timeout/cancel, and producer-oracle metrics."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    TestSelectionRef,
    VerificationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
    FALLBACK_BOTH,
    FALLBACK_FULL_PYTEST,
    FALLBACK_NONE,
    CommandBinding,
    CommandKind,
    HarnessAssurancePolicy,
    MaterializedCommand,
    TypedTimeout,
    materialize_selection_commands,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
    ADAPTER_ID,
    FullSuiteComparison,
    NormalizedOutcome,
    NormalizedRunFacts,
    OracleApplicability,
    ProverResult,
    PytestResult,
    SEMANTIC_VERIFICATION_INTERFACE,
    StaticCheckResult,
    VERIFICATION_SCHEMA,
    VerificationCancelled,
    VerificationError,
    VerificationRunner,
    VerificationStatus,
    VerificationTimeout,
    compare_full_suite,
    compute_new_regressions,
    normalize_run_facts,
    verification_descriptor,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_state/verification.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _binding() -> CommandBinding:
    return CommandBinding.from_dict(
        {
            "tree_cid": _cid("tree"),
            "config_cid": _cid("config"),
            "dependency_lock_cid": _cid("lock"),
            "toolchain_cid": _cid("toolchain"),
            "policy_cid": _cid("policy"),
            "interface_cid": _cid("interface"),
        }
    )


def _selection(
    *,
    pytest_nodes: tuple[str, ...] = ("tests/test_mod.py::test_a",),
    proof_ids: tuple[str, ...] = (),
    fallback: str = FALLBACK_NONE,
    fallback_reasons: tuple[str, ...] = (),
) -> Any:
    import types

    curr = _cid("curr-root")
    prev = _cid("prev-root")
    return types.SimpleNamespace(
        selection_cid=_cid(f"sel|{fallback}|{','.join(pytest_nodes)}"),
        previous_root_cid=prev,
        current_root_cid=curr,
        selected_pytest_node_ids=pytest_nodes,
        selected_proof_ids=proof_ids,
        reason_paths=(
            types.SimpleNamespace(path_cid=_cid("reason-path")),
        ),
        covered_seed_obligation_ids=(),
        unresolved_obligation_ids=(),
        known_test_universe_cid=_cid("universe"),
        known_test_universe_count=4,
        fallback=fallback,
        fallback_reasons=fallback_reasons,
        policy_cid=_cid("sel-policy"),
    )


def _outcome(
    node_id: str,
    status: str,
    fingerprint: str | None = None,
) -> NormalizedOutcome:
    return NormalizedOutcome(
        node_id=node_id,
        status=status,
        failure_fingerprint=fingerprint,
    )


def _facts(run_id: str, *outcomes: NormalizedOutcome) -> NormalizedRunFacts:
    return NormalizedRunFacts(run_id=run_id, outcomes=outcomes)


# ---------------------------------------------------------------------------
# Module authority
# ---------------------------------------------------------------------------


def test_verification_module_forbids_impact_reselection() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "run_impact_selected(" not in source
    tree = ast.parse(source)
    names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_impact_selected" not in names
    assert "select_tests_and_proofs" not in names


def test_descriptor_pins_interface() -> None:
    descriptor = verification_descriptor()
    assert descriptor["interface"] == SEMANTIC_VERIFICATION_INTERFACE
    assert descriptor["adapter_id"] == ADAPTER_ID
    assert "unavailable_prover_as_passed" in set(descriptor["forbids"])
    assert "fabricated_100_percent_empty_oracle" in set(descriptor["forbids"])


# ---------------------------------------------------------------------------
# Runner stages
# ---------------------------------------------------------------------------


def test_static_check_and_pytest_results_bind_selection_and_tree(
    tmp_path: Path,
) -> None:
    selection = _selection()
    policy = HarnessAssurancePolicy(
        require_static_checks=True,
        static_check_commands=("python3.12 -m compileall pkg",),
    )
    runner = VerificationRunner(assurance=policy)

    def _pass(command: MaterializedCommand) -> Mapping[str, Any]:
        return {"passed": True, "returncode": 0}

    report = runner.run(
        selection,
        binding=_binding(),
        workspace_path=tmp_path,
        runner=_pass,
        prover_available=False,
    )
    assert report["interface"] == SEMANTIC_VERIFICATION_INTERFACE
    assert report["selection_cid"] == selection.selection_cid
    assert report["binding"]["tree_cid"] == _binding().tree_cid
    assert report["reason_path_cids"]
    static = report["static_checks"]
    assert len(static) == 1
    assert static[0]["passed"] is True
    assert static[0]["selection_cid"] == selection.selection_cid
    pytest_results = report["pytest"]
    assert len(pytest_results) == 1
    assert pytest_results[0]["kind"] == CommandKind.PYTEST_NODE.value
    assert pytest_results[0]["passed"] is True


def test_unavailable_prover_is_typed_and_not_passed(tmp_path: Path) -> None:
    selection = _selection(proof_ids=("proof:1", "proof:2"))
    runner = VerificationRunner()
    report = runner.run(
        selection,
        binding=_binding(),
        workspace_path=tmp_path,
        runner=lambda command: {"passed": True, "returncode": 0},
        prover_available=False,
    )
    proofs = report["proofs"]
    assert len(proofs) == 2
    for item in proofs:
        assert item["status"] == VerificationStatus.UNAVAILABLE.value
        assert item["passed"] is False
    # Acceptance must not treat unavailable proofs as success.
    assert report["acceptance_eligible"] is False
    assert report["passed"] is False


def test_prover_result_rejects_unavailable_as_passed_semantics() -> None:
    result = ProverResult(
        command_identity="cmd",
        proof_id="p1",
        status=VerificationStatus.UNAVAILABLE.value,
        binding=_binding(),
        selection_cid=_cid("sel"),
    )
    assert result.passed is False
    assert result.to_dict()["passed"] is False


def test_typed_timeout_raises_from_runner(tmp_path: Path) -> None:
    selection = _selection()
    runner = VerificationRunner()

    def _timeout(command: MaterializedCommand) -> Mapping[str, Any]:
        raise VerificationTimeout(
            "budget exceeded",
            timeout=command.timeout,
            command_identity=command.command_identity,
        )

    with pytest.raises(VerificationTimeout) as excinfo:
        runner.run(
            selection,
            binding=_binding(),
            workspace_path=tmp_path,
            runner=_timeout,
        )
    assert isinstance(excinfo.value.timeout, TypedTimeout)
    assert excinfo.value.reason_code == "verification_timeout"


def test_typed_cancellation_raises(tmp_path: Path) -> None:
    selection = _selection()
    runner = VerificationRunner()
    token = CancellationToken("v-cancel")
    token.cancel(cancellation_id="v-cancel", reason="drain")
    with pytest.raises(VerificationCancelled) as excinfo:
        runner.run(
            selection,
            binding=_binding(),
            workspace_path=tmp_path,
            cancellation=token,
            runner=lambda command: {"passed": True, "returncode": 0},
        )
    assert excinfo.value.cancellation_id == "v-cancel"


def test_simulated_cannot_be_acceptance_eligible(tmp_path: Path) -> None:
    selection = _selection()
    runner = VerificationRunner()
    report = runner.run(
        selection,
        binding=_binding(),
        workspace_path=tmp_path,
        runner=lambda command: {"passed": True, "returncode": 0},
        prover_available=False,
        simulated=True,
    )
    # No proofs selected => proofs_all_available_and_passed True, but simulated.
    assert report["simulated"] is True
    assert report["acceptance_eligible"] is False


def test_build_verification_receipt_binds_command_and_selection() -> None:
    selection = _selection()
    ref = TestSelectionRef.from_dict(
        {
            "selection_cid": selection.selection_cid,
            "previous_semantic_state_root_cid": selection.previous_root_cid,
            "current_semantic_state_root_cid": selection.current_root_cid,
        }
    )
    runner = VerificationRunner()
    binding = _binding()
    receipt = runner.build_verification_receipt(
        binding=binding,
        selection_ref=ref,
        command_identity="sch-cmd:abc",
        exit_code=0,
        output_artifact_cids=(_cid("out"),),
        simulated=False,
        fresh=True,
        acceptance_eligible=True,
    )
    assert isinstance(receipt, VerificationReceipt)
    assert receipt.tree_cid == binding.tree_cid
    assert receipt.config_cid == binding.config_cid
    assert receipt.selection_ref.selection_cid == selection.selection_cid
    assert receipt.acceptance_eligible is True
    with pytest.raises(VerificationError, match="simulated"):
        runner.build_verification_receipt(
            binding=binding,
            selection_ref=ref,
            command_identity="x",
            exit_code=0,
            simulated=True,
            acceptance_eligible=True,
        )


# ---------------------------------------------------------------------------
# Controlled producer-oracle metrics / false negatives
# ---------------------------------------------------------------------------


def test_false_negative_is_authored_oracle_miss() -> None:
    """FN = oracle node absent from effective selected membership."""

    selection = _selection(
        pytest_nodes=("tests/test_mod.py::test_a",),
        fallback=FALLBACK_NONE,
    )
    baseline = _facts(
        "base",
        _outcome("tests/test_mod.py::test_a", "passed"),
        _outcome("tests/test_mod.py::test_b", "passed"),
        _outcome("tests/test_other.py::test_c", "passed"),
    )
    selected = _facts(
        "selected",
        _outcome("tests/test_mod.py::test_a", "failed", "fp-a"),
    )
    candidate = _facts(
        "full",
        _outcome("tests/test_mod.py::test_a", "failed", "fp-a"),
        _outcome("tests/test_mod.py::test_b", "failed", "fp-b"),
        _outcome("tests/test_other.py::test_c", "passed"),
    )
    comparison = compare_full_suite(
        selection,
        baseline_full=baseline,
        selected_run=selected,
        candidate_full=candidate,
        authored_oracle=(
            "tests/test_mod.py::test_a",
            "tests/test_mod.py::test_b",
        ),
    )
    assert comparison.applicability == OracleApplicability.APPLICABLE.value
    assert comparison.false_negatives == ("tests/test_mod.py::test_b",)
    assert comparison.true_positives == ("tests/test_mod.py::test_a",)
    assert comparison.zero_false_negatives is False
    assert comparison.supports_100_percent_recall is False
    # Missed regression for test_b (new fail not selected).
    assert "tests/test_mod.py::test_b" in comparison.missed_regressions


def test_100_percent_recall_when_all_oracle_nodes_selected() -> None:
    selection = _selection(
        pytest_nodes=(
            "tests/test_mod.py::test_a",
            "tests/test_mod.py::test_b",
        ),
        fallback=FALLBACK_NONE,
    )
    baseline = _facts(
        "base",
        _outcome("tests/test_mod.py::test_a", "passed"),
        _outcome("tests/test_mod.py::test_b", "passed"),
        _outcome("tests/test_other.py::test_c", "passed"),
    )
    selected = _facts(
        "selected",
        _outcome("tests/test_mod.py::test_a", "failed", "fp-a"),
        _outcome("tests/test_mod.py::test_b", "failed", "fp-b"),
    )
    candidate = _facts(
        "full",
        _outcome("tests/test_mod.py::test_a", "failed", "fp-a"),
        _outcome("tests/test_mod.py::test_b", "failed", "fp-b"),
        _outcome("tests/test_other.py::test_c", "passed"),
    )
    comparison = compare_full_suite(
        selection,
        baseline_full=baseline,
        selected_run=selected,
        candidate_full=candidate,
        authored_oracle=(
            "tests/test_mod.py::test_a",
            "tests/test_mod.py::test_b",
        ),
    )
    assert comparison.false_negatives == ()
    assert comparison.missed_regressions == ()
    assert comparison.fixture_recall_bp == 10_000
    assert comparison.zero_false_negatives is True
    assert comparison.supports_100_percent_recall is True
    # Unaffected passing test_c is not a true positive.
    assert "tests/test_other.py::test_c" not in comparison.true_positives


def test_empty_authored_oracle_is_not_applicable_not_fabricated_100() -> None:
    selection = _selection()
    baseline = _facts("base", _outcome("t::a", "passed"))
    selected = _facts("sel", _outcome("t::a", "passed"))
    candidate = _facts("full", _outcome("t::a", "passed"))
    comparison = compare_full_suite(
        selection,
        baseline_full=baseline,
        selected_run=selected,
        candidate_full=candidate,
        authored_oracle=None,
    )
    assert comparison.applicability == OracleApplicability.NOT_APPLICABLE.value
    assert comparison.fixture_recall_bp is None
    assert comparison.supports_100_percent_recall is False
    assert comparison.false_negatives == ()


def test_full_pytest_fallback_covers_membership_without_false_negatives() -> None:
    selection = _selection(
        pytest_nodes=(),  # producer clears selected under full fallback
        fallback=FALLBACK_FULL_PYTEST,
        fallback_reasons=("unknown_test_universe",),
    )
    baseline = _facts(
        "base",
        _outcome("t::a", "passed"),
        _outcome("t::b", "passed"),
    )
    selected = _facts(
        "sel",
        _outcome("t::a", "failed", "x"),
        _outcome("t::b", "passed"),
    )
    candidate = _facts(
        "full",
        _outcome("t::a", "failed", "x"),
        _outcome("t::b", "passed"),
    )
    comparison = compare_full_suite(
        selection,
        baseline_full=baseline,
        selected_run=selected,
        candidate_full=candidate,
        authored_oracle=("t::a",),
    )
    assert comparison.fallback_rate_bp == 10_000
    assert comparison.false_negatives == ()
    assert comparison.supports_100_percent_recall is True
    assert comparison.selected_count == comparison.full_count


def test_known_baseline_failure_is_not_new_regression() -> None:
    baseline = _facts(
        "base",
        _outcome("t::a", "failed", "same"),
        _outcome("t::b", "passed"),
    )
    candidate = _facts(
        "full",
        _outcome("t::a", "failed", "same"),
        _outcome("t::b", "failed", "new"),
    )
    new = compute_new_regressions(baseline, candidate)
    assert new == ("t::b",)


def test_normalize_run_facts_is_deterministic() -> None:
    facts = normalize_run_facts(
        "run-1",
        [
            {"node_id": "b::2", "status": "passed", "failure_fingerprint": None},
            {"node_id": "a::1", "status": "failed", "failure_fingerprint": "fp"},
        ],
    )
    again = normalize_run_facts(
        "run-1",
        [
            {"node_id": "a::1", "status": "failed", "failure_fingerprint": "fp"},
            {"node_id": "b::2", "status": "passed", "failure_fingerprint": None},
        ],
    )
    assert facts.facts_cid == again.facts_cid
    assert [item.node_id for item in facts.outcomes] == ["a::1", "b::2"]


def test_runner_compare_full_suite_method(tmp_path: Path) -> None:
    selection = _selection(
        pytest_nodes=("t::a", "t::b"),
    )
    runner = VerificationRunner()
    comparison = runner.compare_full_suite(
        selection,
        baseline_full=_facts(
            "b",
            _outcome("t::a", "passed"),
            _outcome("t::b", "passed"),
        ),
        selected_run=_facts(
            "s",
            _outcome("t::a", "failed", "f"),
            _outcome("t::b", "failed", "g"),
        ),
        candidate_full=_facts(
            "c",
            _outcome("t::a", "failed", "f"),
            _outcome("t::b", "failed", "g"),
        ),
        authored_oracle=("t::a", "t::b"),
    )
    assert isinstance(comparison, FullSuiteComparison)
    assert comparison.supports_100_percent_recall is True
    payload = comparison.to_dict()
    assert payload["schema"] == VERIFICATION_SCHEMA
    assert payload["zero_false_negatives"] is True


def test_result_records_round_trip_fields() -> None:
    binding = _binding()
    static = StaticCheckResult(
        command_identity="static-1",
        shell_command="python3.12 -m compileall pkg",
        status=VerificationStatus.PASSED.value,
        exit_code=0,
        timed_out=False,
        cancelled=False,
        binding=binding,
        selection_cid=_cid("sel"),
        timeout=TypedTimeout(seconds=30, stage="static_check"),
    )
    assert static.passed is True
    assert static.to_dict()["timeout"]["seconds"] == 30.0

    pytest_result = PytestResult(
        command_identity="py-1",
        shell_command="python3.12 -m pytest -q -- t::a",
        status=VerificationStatus.FAILED.value,
        exit_code=1,
        timed_out=False,
        cancelled=False,
        node_ids=("t::a",),
        binding=binding,
        selection_cid=_cid("sel"),
        outcomes=(_outcome("t::a", "failed", "fp"),),
    )
    assert pytest_result.passed is False
    assert pytest_result.to_dict()["outcomes"][0]["node_id"] == "t::a"


def test_predicted_symbols_exported() -> None:
    import ipfs_accelerate_py.agent_supervisor.semantic_state.verification as mod

    for name in (
        "VerificationRunner",
        "StaticCheckResult",
        "PytestResult",
        "ProverResult",
        "FullSuiteComparison",
        "compare_full_suite",
    ):
        assert hasattr(mod, name), name
