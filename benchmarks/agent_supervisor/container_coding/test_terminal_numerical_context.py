"""Authored numerical source through native projections, lakes and planning."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version

SOURCE = """import numpy as np

def candidate(matrix):
    values, vectors = np.linalg.eig(matrix)
    chosen = np.argmax(np.abs(values))
    return values[chosen], vectors[:, chosen]
"""


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    root = tmp_path / "code"
    root.mkdir()
    (root / "sample.py").write_text(SOURCE)
    hashes = {"sample.py": hashlib.sha256(SOURCE.encode()).hexdigest()}
    receipt = numerical.prepare_numerical_context(repository=root, source_hashes=hashes,
        output=tmp_path / "observations", profile=numerical.PROFILE)
    return root, receipt


def test_actual_projected_context_reopens_both_native_lakes(context):
    root, receipt = context
    assert numerical.validate_numerical_context(repository=root, expected_receipt=receipt) == receipt
    summary = receipt["summary"]
    assert summary["functions"] == summary["structurally_complete_functions"] == 1
    assert summary["families_with_artifacts"] == ["first_order", "higher_order", "program"]
    assert summary["pending_obligation_count"] > 0
    assert summary["compound_contract_count"] == 1
    assert summary["compound_contracts"][0]["kind"] == "selected_eigenpair"
    assert summary["compound_contracts"][0]["path"] == "sample.py"
    assert summary["omitted_compound_contract_summaries"] == 0
    assert summary["source_obligations_discharged"] == 0
    assert summary["source_semantics_verified"] is summary["proof_authority"] is False
    assert receipt["hydration"]["ducklake_verified"] is True
    assert receipt["hydration"]["ducklake_catalogs"] == receipt["hydration"]["ducklake_links"] == 3
    assert receipt["provider_calls"] == receipt["model_inference_calls"] == 0
    assert len(numerical._bytes(summary)) <= 8192


@pytest.mark.parametrize("mutation", ["source", "report", "lake", "receipt", "producer"])
def test_stale_source_artifact_lake_receipt_and_producer_rejected(context, monkeypatch, mutation):
    root, receipt = context
    output = numerical.Path(receipt["output"])
    if mutation == "producer":
        monkeypatch.setattr(numerical, "_producers", lambda: {"different": "producer"})
    elif mutation == "receipt":
        receipt = deepcopy(receipt)
        receipt["summary"]["source_semantics_verified"] = True
    else:
        path = {"source": root / "sample.py", "report": output / "report-0000.json",
                "lake": output / "metadata-lake/metadata.ducklake"}[mutation]
        with path.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError):
        numerical.validate_numerical_context(repository=root, expected_receipt=receipt)


def test_rehashed_forged_formula_still_fails_deterministic_source_replay(context):
    root, receipt = context
    output = numerical.Path(receipt["output"])
    path = output / "report-0000.json"
    report = json.loads(path.read_bytes())
    report["coverage"]["source_obligations_discharged"] = 10
    path.write_bytes(numerical._bytes(report))
    index = json.loads((output / "reports.json").read_bytes())
    index["sample.py"]["sha256"] = numerical._sha(path.read_bytes())
    (output / "reports.json").write_bytes(numerical._bytes(index))
    receipt = deepcopy(receipt)
    receipt["files"] = numerical._inventory(output)
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    (output / "receipt.json").write_bytes(numerical._bytes(unsigned))
    receipt["receipt_sha256"] = numerical._sha(numerical._bytes(unsigned))
    with pytest.raises(ValueError, match="source, interpretation or artifact"):
        numerical.validate_numerical_context(repository=root, expected_receipt=receipt)


def test_source_mutation_during_native_hydration_cannot_return_current(context, monkeypatch):
    root, receipt = context
    hydrate = numerical._hydrate
    def changed(output, record):
        result = hydrate(output, record)
        (root / "sample.py").write_text(SOURCE + "\n# concurrent edit\n")
        return result
    monkeypatch.setattr(numerical, "_hydrate", changed)
    with pytest.raises(ValueError, match="signed declaration"):
        numerical.validate_numerical_context(repository=root, expected_receipt=receipt)


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py", "./sample.py", ".runtime/sample.py", "sample.txt"])
def test_noncanonical_source_names_rejected_before_persistence(tmp_path, name):
    root = tmp_path / "code"
    root.mkdir()
    with pytest.raises(ValueError, match="canonical declared Python"):
        numerical.prepare_numerical_context(repository=root, source_hashes={name: "a"*64},
            output=tmp_path / "context", profile=numerical.PROFILE)
    assert not (tmp_path / "context").exists()


def test_summary_reaches_router_before_goals_and_retains_admitted_index(original, monkeypatch):
    root, instruction, state = original
    (root / "bottle.py").write_text(SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state, numerical_formalization_profile=numerical.PROFILE)
    formal = receipt["numerical_formalization"]
    assert set(formal["source_hashes"]) == {"bottle.py"}
    assert receipt["canonical_tasks_created"] is False
    assert receipt["provider_calls"] == 0
    observed = []
    def router(prompt, **kwargs):
        observed.append(prompt)
        assert "terminal-numerical-formalization-planning-summary@1" in prompt
        assert "eigenpair_residual" in prompt
        assert "selected_eigenpair" in prompt
        assert "selected_eigenpair_nonzero" in prompt
        assert '"source_semantics_verified":false' in prompt.replace(" ", "")
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    result = prep.plan(state, provider_callable=router)
    assert result["qualified"], result
    assert len(observed) == 1
    admitted = prep.context(state=state)
    assert admitted["initial_indexes_reused"] is True
    assert admitted["numerical_formalization"] == formal
    persisted = json.loads((root / ".runtime/terminal-context/result.json").read_bytes())
    assert persisted["numerical_formalization"] == formal
    assert numerical.validate_numerical_context(repository=root, expected_receipt=formal) == formal


def test_selected_context_cannot_disappear_before_planning(original, monkeypatch):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state, numerical_formalization_profile=numerical.PROFILE)
    (state / "numerical-formalization-selection.json").write_text(json.dumps({"profile": "different"}))
    _version(monkeypatch)
    with pytest.raises(ValueError, match="selected public program population"):
        prep.plan(state, provider_callable=lambda *a, **k: pytest.fail("stale context reached provider"))
    assert not (state / "planner-invoked.json").exists()


def test_explicit_selection_preserves_legacy_path(original):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state)
    assert "numerical_formalization" not in receipt
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    assert all(s["schema"] != "terminal-numerical-formalization-planning-summary@1" for s in loaded["summaries"])
    assert not (state / "numerical-formalization-selection.json").exists()


def test_compound_contract_budget_accounts_for_every_omitted_row():
    from ipfs_datasets_py.logic.software_verification.numerical_logic import formalize_numerical_source
    reports = {f"file_{index:02d}.py": formalize_numerical_source(SOURCE, path=f"file_{index:02d}.py")
               for index in range(12)}
    summary = numerical._summary(reports, "authored-tree")
    assert summary["compound_contract_count"] == 12
    assert 0 < len(summary["compound_contracts"]) <= 4
    assert len(summary["compound_contracts"]) + summary["omitted_compound_contract_summaries"] == 12
    assert len(summary["pending_obligations"]) + summary["omitted_obligation_summaries"] == summary["pending_obligation_count"]
    assert len(numerical._bytes(summary)) <= 8192
    assert summary["source_obligations_discharged"] == 0
