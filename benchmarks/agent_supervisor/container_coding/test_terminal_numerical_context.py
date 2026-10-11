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

ASSERTION_SOURCE = SOURCE + """
def check_candidate(matrix):
    value, vector = candidate(matrix)
    assert not np.allclose(vector, 0)
    assert not np.isnan(value)
    assert not np.isinf(value)
    assert np.allclose(matrix @ vector, value * vector)
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
    assert summary["source_unit_count"] == summary["linked_source_unit_count"] == 1
    assert summary["unmatched_source_unit_count"] == 0
    link = summary["source_unit_links"][0]
    assert link["scalar_guard_reason"] == "two_distinct_parameters_required"
    assert summary["source_unit_links_establish_learned_coverage"] is False
    index = json.loads((numerical.Path(receipt["output"]) / "reports.json").read_bytes())
    retained = json.loads((numerical.Path(receipt["output"]) / index["sample.py"]["unit_links_artifact"]).read_bytes())
    assert link["unit_id"] == retained["links"][0]["unit_id"]
    assert link["function_id"] == retained["links"][0]["function_id"]
    assert link["link_report_sha256"] == retained["report_sha256"]


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


@pytest.mark.parametrize("owner", ["security_formula_corpus.py", "security_formalization_evaluation.py",
                                  "source_screening.py"])
def test_unit_extraction_helper_mutation_invalidates_prepared_context(context, monkeypatch, owner):
    root, receipt = context
    read = numerical._read
    touched = []

    def changed(path, *args, **kwargs):
        raw = read(path, *args, **kwargs)
        if numerical.Path(path).name == owner:
            touched.append(owner)
            return raw + b"\n# authored changed producer observation\n"
        return raw

    monkeypatch.setattr(numerical, "_read", changed)
    with pytest.raises(ValueError, match="artifacts or producers changed"):
        numerical.validate_numerical_context(repository=root, expected_receipt=receipt)
    assert touched == [owner]


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
        assert "source_unit_links" in prompt
        assert "deterministic_source_projection" in prompt
        assert formal["summary"]["source_unit_links"][0]["unit_id"] in prompt
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
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_selection
    bundle = admitted["context_bundle"]
    wire = json.loads((root / bundle["artifact"]).read_bytes())
    assert wire["schema"] == "supervisor-task-context-nominations@4"
    selected = load_task_context_selection(repository=root, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=admitted["task_cid"], task_id=persisted["task_id"])
    assert selected["numerical_formalization"] == formal
    evidence = selected["worker_evidence"]
    assert evidence["task_cid"] == admitted["task_cid"]
    assert evidence["context_bundle"] == bundle
    assert evidence["contexts"][0]["summary"] == formal["summary"]
    assert evidence["contexts"][0]["summary"]["source_unit_links"] == formal["summary"]["source_unit_links"]
    assert evidence["source_obligations_discharged"] == 0


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


@pytest.mark.parametrize("mutation", ["function", "source", "proof", "missing_artifact"])
def test_rehashed_unit_link_mutations_fail_source_replay(context, mutation):
    root, receipt = context
    output = numerical.Path(receipt["output"])
    index = json.loads((output / "reports.json").read_bytes())
    ref = index["sample.py"]
    path = output / ref["unit_links_artifact"]
    links = json.loads(path.read_bytes())
    if mutation == "missing_artifact":
        ref["unit_links_artifact"] = "report-0000.json"
    else:
        if mutation == "function": links["links"][0]["function_id"] = "foreign-function"
        elif mutation == "source": links["source_sha256"] = "0" * 64
        else: links["proof_authority"] = True
        links["report_sha256"] = numerical._sha(numerical._bytes({k: v for k, v in links.items() if k != "report_sha256"}))
        path.write_bytes(numerical._bytes(links))
        ref["unit_links_sha256"] = numerical._sha(path.read_bytes())
    (output / "reports.json").write_bytes(numerical._bytes(index))
    receipt = deepcopy(receipt)
    receipt["files"] = numerical._inventory(output)
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    (output / "receipt.json").write_bytes(numerical._bytes(unsigned))
    receipt["receipt_sha256"] = numerical._sha(numerical._bytes(unsigned))
    with pytest.raises(ValueError, match="numerical unit.link"):
        numerical.validate_numerical_context(repository=root, expected_receipt=receipt)


def test_unit_summary_accounts_for_unmatched_and_omitted_units():
    from ipfs_datasets_py.logic.software_verification.numerical_logic import formalize_numerical_source
    source = "def outer(x):\n    def inner(y):\n        return y\n    return x\n"
    sources = {f"file_{i}.py": source for i in range(3)}
    reports = {p: formalize_numerical_source(t, path=p) for p, t in sources.items()}
    links = numerical._function_unit_links(sources, reports)
    summary = numerical._summary(reports, "authored-tree", links)
    assert summary["source_unit_count"] == 6
    assert summary["linked_source_unit_count"] == summary["unmatched_source_unit_count"] == 3
    assert len(summary["source_unit_links"]) + summary["omitted_source_unit_link_summaries"] == 6
    assert len(numerical._bytes(summary)) <= 8192
    assert summary["autoencoder_coverage_changed"] is False
    assert summary["source_obligations_discharged"] == 0


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


def test_assertion_requirements_reopen_native_lakes_and_replay_predicates(tmp_path, monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    root = tmp_path / "code"
    root.mkdir()
    (root / "checks.py").write_text(ASSERTION_SOURCE)
    receipt = numerical.prepare_numerical_context(repository=root,
        source_hashes={"checks.py": hashlib.sha256(ASSERTION_SOURCE.encode()).hexdigest()},
        output=tmp_path / "observations", profile=numerical.PROFILE)
    summary = receipt["summary"]
    assert summary["assertion_requirement_count"] == 4
    rows = summary["assertion_requirements"]
    assert rows and len(rows) + summary["omitted_assertion_requirement_summaries"] == 4
    assert all(row["path"] == "checks.py" and row["origin"] == "source_assertion" for row in rows)
    assert all(row["status"] == "pending" and not row["proof_authority"] for row in rows)
    assert all(row["command_id"] and row["predicate_expression_id"] for row in rows)
    assert all(row["symbol_definition_ids"] for row in rows)
    assert all("argument_expression_ids" not in row["predicate"] for row in rows)
    assert receipt["hydration"]["ducklake_verified"] is True
    assert numerical.validate_numerical_context(repository=root, expected_receipt=receipt) == receipt

    # Rehashing persisted bytes cannot promote a different assertion to source evidence.
    output = numerical.Path(receipt["output"])
    path = output / "report-0000.json"
    report = json.loads(path.read_bytes())
    report["assertion_requirements"][0]["predicate_expression_id"] = "forged-predicate"
    path.write_bytes(numerical._bytes(report))
    index = json.loads((output / "reports.json").read_bytes())
    index["checks.py"]["sha256"] = numerical._sha(path.read_bytes())
    (output / "reports.json").write_bytes(numerical._bytes(index))
    receipt = deepcopy(receipt)
    receipt["files"] = numerical._inventory(output)
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    (output / "receipt.json").write_bytes(numerical._bytes(unsigned))
    receipt["receipt_sha256"] = numerical._sha(numerical._bytes(unsigned))
    with pytest.raises(ValueError, match="source, interpretation or artifact"):
        numerical.validate_numerical_context(repository=root, expected_receipt=receipt)


def test_assertion_summary_reaches_planner_before_goals(original, monkeypatch):
    root, instruction, state = original
    (root / "bottle.py").write_text(ASSERTION_SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state, numerical_formalization_profile=numerical.PROFILE)
    formal = receipt["numerical_formalization"]
    assert formal["summary"]["assertion_requirement_count"] == 4
    observed = []
    def router(prompt, **kwargs):
        observed.append(prompt)
        request = json.loads(prompt)
        if request.get("schema") == "intent-plan-provider-request@1":
            request = request["graph_request"]
        summaries = request["constraints"]["constraint_summaries"]
        supplied = [row for row in summaries if isinstance(row, dict)
                    and row.get("schema") == formal["summary"]["schema"]]
        assert supplied == [formal["summary"]]
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    result = prep.plan(state, provider_callable=router)
    assert result["qualified"], json.dumps(result, indent=2)
    assert len(observed) == 1
    admitted = prep.context(state=state)
    assert admitted["initial_indexes_reused"] is True
    assert admitted["numerical_formalization"] == formal
    assert numerical.validate_numerical_context(repository=root, expected_receipt=formal) == formal


def test_assertion_summary_budget_counts_upstream_omissions():
    from ipfs_datasets_py.logic.software_verification.numerical_logic import formalize_numerical_source
    source = ASSERTION_SOURCE + """
def another_check(value):
    assert not np.isnan(value)
    assert not np.isinf(value)
"""
    reports = {f"file_{index:02d}.py": formalize_numerical_source(source, path=f"file_{index:02d}.py")
               for index in range(3)}
    assert all(report["summary"]["omitted_assertion_requirement_summaries"] > 0 for report in reports.values())
    summary = numerical._summary(reports, "authored-assertion-tree")
    assert summary["assertion_requirement_count"] == 18
    assert 0 < len(summary["assertion_requirements"]) <= 4
    assert len(summary["assertion_requirements"]) + summary["omitted_assertion_requirement_summaries"] == 18
    assert len(summary["compound_contracts"]) + summary["omitted_compound_contract_summaries"] == 3
    assert len(summary["pending_obligations"]) + summary["omitted_obligation_summaries"] == summary["pending_obligation_count"]
    assert len(numerical._bytes(summary)) <= 8192
    assert summary["source_obligations_discharged"] == 0
