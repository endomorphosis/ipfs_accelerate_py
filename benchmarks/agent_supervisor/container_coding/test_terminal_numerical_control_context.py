"""Selected control grammar through native context, lake and planning owners."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version


SOURCE = '''def scan(values, limit):
    total = 0
    for value in values:
        if value < 0:
            continue
        if value > limit:
            break
        total = total + value
    else:
        total = total + 1
    while total < limit:
        total = total + 1
    assert total >= 0, f"invalid total: {total}"
    return total
'''

HISTORICAL_SOURCE = """import numpy as np

def choose(value):
    assert np.isfinite(value), f'nonfinite: {value}'
    return np.sqrt(value) if value > 0 else np.abs(value)
"""


@pytest.mark.parametrize("profile, expected", [
    (native.PROFILE, "46aa10055716290998b1bd3dc0b2b68d9f12b7f37ab40448e6e9e0dc145d4db8"),
    (native.GUARDED_PROFILE, "8ccc62143c3194ac6bc3d32cc6a5f35b41cb0c8dbfbb1da405ca81f92147956e"),
])
def test_earlier_profile_planning_summary_retains_prior_bytes(profile, expected):
    # Captured from the unmodified summary implementation at a9bcde965ac8.
    # No self/producer hashes or filesystem paths enter this planner fixture.
    _, owner = native._modules(profile)
    report = owner.formalize_numerical_source(HISTORICAL_SOURCE, path="sample.py")
    summary = native._summary({"sample.py": report}, "authored-tree", profile=profile)
    assert native._sha(native._bytes(summary)) == expected


@pytest.fixture
def control_context(tmp_path, monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.py").write_text(SOURCE)
    receipt = native.prepare_numerical_context(repository=root,
        source_hashes={"sample.py": native._sha(SOURCE.encode())},
        output=tmp_path / "control", profile=native.CONTROL_PROFILE)
    return root, receipt


def _report(receipt):
    output = native.Path(receipt["output"])
    index = json.loads((output / "reports.json").read_bytes())
    return json.loads((output / index["sample.py"]["artifact"]).read_bytes())


def test_control_context_replays_lakes_and_keeps_unresolved_loop_references(control_context):
    root, receipt = control_context
    assert native.validate_numerical_context(repository=root, expected_receipt=receipt) == receipt
    summary, report = receipt["summary"], _report(receipt)
    assert receipt["profile"] == summary["profile"] == receipt["record"]["profile"] == native.CONTROL_PROFILE
    assert receipt["hydration"]["ducklake_verified"] is True
    assert receipt["hydration"]["ducklake_catalogs"] == receipt["hydration"]["ducklake_links"] == 3
    assert summary["structurally_complete_functions"] == summary["functions"] == 1
    assert len(native._bytes(summary)) <= 8192
    assert "Loop" in summary["interpretation"] and "termination" in summary["interpretation"]
    assert report["structural"]["control_regions"]
    assert report["structural"]["command_contexts"]
    full = {row["obligation_id"]: row for row in report["pending_obligations"]}
    assert any(row["loop_region_ids"] for row in full.values())
    assert summary["pending_obligations"][0]["proof_phase"] == "activation_entry"
    for row in summary["pending_obligations"]:
        source = full[row["obligation_id"]]
        assert row["control_region_ids"] == source["control_region_ids"]
        assert row["loop_region_ids"] == source["loop_region_ids"]
        assert row["cfg_ids"] == sorted({value["cfg_id"] for value in source["cfg_occurrences"]})
        assert row["cfg_block_ids"] == sorted({value["block_id"] for value in source["cfg_occurrences"]})
        assert row["cfg_command_ids"] == sorted({value["command_id"] for value in source["cfg_occurrences"]})
        assert "cfg_occurrences" not in row and "evaluation_contexts" not in row
        assert row["evaluation_guard_verified"] is row["standalone_program_evaluation_complete"] is False
        if "proof_scope" in source:
            assert row["evaluation_context_role"] == "source_anchor_only"
            assert row["logical_guard_scope"] == "unresolved_proof_scope_applicability"
            assert row["proof_phase"] == source["proof_scope"]["proof_phase"]
            assert row["proof_scope_control_region_id"] == source["proof_scope"]["control_region_id"]
            assert row["proof_phase_established"] is row["proof_applicability_verified"] is False
    assert summary["source_obligations_discharged"] == receipt["model_inference_calls"] == receipt["provider_calls"] == 0
    assert summary["source_semantics_verified"] is summary["proof_authority"] is False
    assert len(summary["pending_obligations"]) + summary["omitted_obligation_summaries"] == len(full)
    assert summary["control_reference_count"] == 1
    assert summary["omitted_control_reference_summaries"] == 0
    reference = summary["control_references"][0]
    regions = report["structural"]["control_regions"]
    loops = {row["region_id"] for row in regions if row["kind"] in {"for_statement", "while_statement"}}
    assert len(reference["loop_region_ids"]) + reference["omitted_loop_region_ids"] == len(loops)
    assert reference["loop_region_ids"] == sorted(loops)[:4]
    controls = {row["region_id"] for row in regions}
    assert reference["control_region_ids"] == sorted(controls, key=lambda key: (key not in loops, key))[:4]


def test_control_paging_retains_exact_cfg_occurrences_and_iteration_contexts(control_context):
    root, receipt = control_context
    expected = _report(receipt)["pending_obligations"]
    rows, cursor = [], None
    for _ in range(len(expected) + 1):
        page = native.retrieve_numerical_obligations(repository=root, expected_receipt=receipt,
            path="sample.py", cursor=cursor, limit=2)
        assert len(native._page_bytes(page)) <= 8192
        assert page["source_tree_id"] == receipt["record"]["source_tree_id"]
        rows.extend(page["rows"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("control obligation pagination did not terminate")
    assert rows == expected
    assert any(row["loop_region_ids"] and row["cfg_occurrences"] for row in rows)
    assert all(row["evaluation_guard_verified"] is False for row in rows)
    (root / "sample.py").write_text(SOURCE + "\n# changed source\n")
    with pytest.raises(ValueError, match="source"):
        native.retrieve_numerical_obligations(repository=root, expected_receipt=receipt, path="sample.py")


@pytest.mark.parametrize("profile", [native.PROFILE, native.GUARDED_PROFILE])
def test_control_profile_cannot_reuse_earlier_tree_or_cursor(control_context, profile):
    root, selected = control_context
    earlier = native.prepare_numerical_context(repository=root, source_hashes=selected["source_hashes"],
        output=root.parent / "earlier", profile=profile)
    assert earlier["record"]["source_tree_id"] != selected["record"]["source_tree_id"]
    assert earlier["record"]["task_id"] != selected["record"]["task_id"]
    page = native.retrieve_numerical_obligations(repository=root, expected_receipt=selected,
        path="sample.py", limit=1)
    assert page["next_cursor"] is not None
    with pytest.raises(ValueError, match="selected context receipt"):
        native.retrieve_numerical_obligations(repository=root, expected_receipt=earlier,
            path="sample.py", limit=1, cursor=page["next_cursor"])


@pytest.mark.parametrize("profile", [native.PROFILE, native.GUARDED_PROFILE,
                                     "python-numerical-control-structure@4", True, None])
def test_changed_profile_cannot_relabel_control_receipt(control_context, profile):
    root, receipt = control_context
    changed = deepcopy(receipt)
    changed["profile"] = profile
    raw = native._bytes({k: v for k, v in changed.items() if k != "receipt_sha256"})
    (native.Path(receipt["output"]) / "receipt.json").write_bytes(raw)
    changed["receipt_sha256"] = native._sha(raw)
    with pytest.raises(ValueError):
        native.validate_numerical_context(repository=root, expected_receipt=changed)


@pytest.mark.parametrize("basename", ["numerical_source_v3.py", "numerical_logic_v3.py",
                                     "numerical_source_v2.py", "numerical_logic_v2.py",
                                     "numerical_source.py", "numerical_logic.py"])
def test_control_context_pins_selected_and_reused_implementations(control_context, monkeypatch, basename):
    root, receipt = control_context
    read = native._read
    def changed(path, *args, **kwargs):
        raw = read(path, *args, **kwargs)
        return raw + b"\n# authored producer drift\n" if native.Path(path).name == basename else raw
    monkeypatch.setattr(native, "_read", changed)
    with pytest.raises(ValueError, match="artifacts or producers changed"):
        native.validate_numerical_context(repository=root, expected_receipt=receipt)


def test_control_summary_charges_omitted_rows_without_stripping_loop_scope(control_context):
    _, receipt = control_context
    full = _report(receipt)
    reports = {f"long-name-{i}-" + "x" * 300 + ".py": full for i in range(12)}
    compact = native._summary(reports, "authored-tree", profile=native.CONTROL_PROFILE)
    assert len(native._bytes(compact)) <= 8192
    assert compact["omitted_obligation_summaries"] > 0
    assert len(compact["pending_obligations"]) + compact["omitted_obligation_summaries"] == compact["pending_obligation_count"]
    assert len(compact["control_references"]) + compact["omitted_control_reference_summaries"] == len(reports)
    for row in compact["pending_obligations"]:
        assert "loop_region_ids" in row and "cfg_block_ids" in row


def test_control_advice_reaches_planner_before_goals_then_admitted_indexes(original, monkeypatch):
    root, instruction, state = original
    (root / "bottle.py").write_text(SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    initial = prep.initial_context(state=state, numerical_formalization_profile=native.CONTROL_PROFILE)
    formal = initial["numerical_formalization"]
    assert initial["canonical_tasks_created"] is False and initial["provider_calls"] == 0
    calls = []
    def authored_planner(prompt, **kwargs):
        calls.append(prompt)
        assert native.CONTROL_PROFILE in prompt
        assert "Loop" in prompt and "termination" in prompt
        assert "loop_region_ids" in prompt and "cfg_block_ids" in prompt
        assert '"evaluation_guard_verified":false' in prompt.replace(" ", "")
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    result = prep.plan(state, provider_callable=authored_planner)
    assert result["qualified"], result
    assert len(calls) == 1
    admitted = prep.context(state=state)
    assert admitted["initial_indexes_reused"] is True
    assert admitted["numerical_formalization"] == formal
    assert native.validate_numerical_context(repository=root, expected_receipt=formal) == formal
