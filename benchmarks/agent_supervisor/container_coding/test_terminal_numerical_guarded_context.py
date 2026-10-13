"""Selected guarded grammar through the actual numerical context owners."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version


SOURCE = '''import numpy as np

def choose(value):
    return np.sqrt(value) if value > 0 else np.abs(value)

def check(value):
    assert np.isfinite(value), f"nonfinite value: {value}"
    return value

def still_fenced(items):
    for item in items:
        return item
'''


@pytest.fixture
def guarded_context(tmp_path, monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    root = tmp_path / "source"
    root.mkdir()
    (root / "sample.py").write_text(SOURCE)
    receipt = native.prepare_numerical_context(repository=root,
        source_hashes={"sample.py": native._sha(SOURCE.encode())},
        output=tmp_path / "guarded", profile=native.GUARDED_PROFILE)
    return root, receipt


def report(receipt):
    output = native.Path(receipt["output"])
    index = json.loads((output / "reports.json").read_bytes())
    return json.loads((output / index["sample.py"]["artifact"]).read_bytes())


def test_guarded_context_replays_lakes_and_retains_conditional_planning_scope(guarded_context):
    root, receipt = guarded_context
    assert native.validate_numerical_context(repository=root, expected_receipt=receipt) == receipt
    summary = receipt["summary"]
    assert receipt["profile"] == summary["profile"] == receipt["record"]["profile"] == native.GUARDED_PROFILE
    assert receipt["hydration"]["ducklake_verified"] is True
    assert receipt["hydration"]["ducklake_catalogs"] == receipt["hydration"]["ducklake_links"] == 3
    assert len(native._bytes(summary)) <= 8192
    assert "deferred until selected" in summary["interpretation"]
    assert "deferred until assertion failure" in summary["interpretation"]
    assert summary["pending_obligation_count"] > 0
    assert len(summary["pending_obligations"]) + summary["omitted_obligation_summaries"] == summary["pending_obligation_count"]
    full = {row["obligation_id"]: row for row in report(receipt)["pending_obligations"]}
    for row in summary["pending_obligations"]:
        assert row["guarded_region_ids"] == full[row["obligation_id"]]["guarded_region_ids"]
        assert row["evaluation_guard_interpretation"] == full[row["obligation_id"]]["evaluation_guard_interpretation"]
        assert row["evaluation_guard_verified"] is row["standalone_program_evaluation_complete"] is False
        assert "evaluation_contexts" not in row
    assert summary["partial_or_unsupported_functions"] >= 1  # The loop remains outside this grammar.
    assert summary["source_obligations_discharged"] == receipt["model_inference_calls"] == receipt["provider_calls"] == 0
    assert summary["source_semantics_verified"] is summary["proof_authority"] is False


def test_guarded_paging_preserves_full_contexts_and_owner_bound_continuations(guarded_context):
    root, receipt = guarded_context
    expected = report(receipt)["pending_obligations"]
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
        pytest.fail("guarded obligation pagination did not terminate")
    assert rows == expected
    guarded = [row for row in rows if row["guarded_region_ids"]]
    assert guarded and all(row["evaluation_contexts"] for row in guarded)
    assert all(row["evaluation_guard_verified"] is False for row in rows)
    (root / "sample.py").write_text(SOURCE + "\n# changed source\n")
    with pytest.raises(ValueError, match="source"):
        native.retrieve_numerical_obligations(repository=root, expected_receipt=receipt, path="sample.py")


def test_guarded_selection_cannot_reuse_legacy_tree_or_cursor(guarded_context):
    root, selected = guarded_context
    legacy = native.prepare_numerical_context(repository=root, source_hashes=selected["source_hashes"],
        output=root.parent / "legacy", profile=native.PROFILE)
    expected_v1 = native.content_identity(dict(schema=native.SCHEMA, sources=legacy["source_hashes"]))
    assert legacy["record"]["source_tree_id"] == expected_v1
    assert "profile" not in legacy["record"]
    assert legacy["record"]["source_tree_id"] != selected["record"]["source_tree_id"]
    assert legacy["record"]["task_id"] != selected["record"]["task_id"]
    page = native.retrieve_numerical_obligations(repository=root, expected_receipt=selected,
        path="sample.py", limit=1)
    assert page["next_cursor"] is not None
    with pytest.raises(ValueError, match="selected context receipt"):
        native.retrieve_numerical_obligations(repository=root, expected_receipt=legacy,
            path="sample.py", limit=1, cursor=page["next_cursor"])


@pytest.mark.parametrize("profile", [native.PROFILE, "python-numerical-structure@2", True, None])
def test_changed_profile_cannot_relabel_guarded_receipt(guarded_context, profile):
    root, receipt = guarded_context
    changed = deepcopy(receipt)
    changed["profile"] = profile
    body = {k: v for k, v in changed.items() if k != "receipt_sha256"}
    raw = native._bytes(body)
    (native.Path(receipt["output"]) / "receipt.json").write_bytes(raw)
    changed["receipt_sha256"] = native._sha(raw)
    with pytest.raises(ValueError):
        native.validate_numerical_context(repository=root, expected_receipt=changed)


@pytest.mark.parametrize("basename", ["numerical_source_v2.py", "numerical_logic_v2.py",
                                     "numerical_source.py", "numerical_logic.py"])
def test_guarded_context_pins_selected_and_reused_implementations(guarded_context, monkeypatch, basename):
    root, receipt = guarded_context
    read = native._read
    def changed(path, *args, **kwargs):
        raw = read(path, *args, **kwargs)
        return raw + b"\n# authored producer drift\n" if native.Path(path).name == basename else raw
    monkeypatch.setattr(native, "_read", changed)
    with pytest.raises(ValueError, match="artifacts or producers changed"):
        native.validate_numerical_context(repository=root, expected_receipt=receipt)


def test_guarded_summary_charges_omitted_rows_without_stripping_guard_fields(guarded_context):
    _, receipt = guarded_context
    full = report(receipt)
    summaries = {f"long-name-{i}-" + "x" * 300 + ".py": full for i in range(12)}
    compact = native._summary(summaries, "authored-tree", profile=native.GUARDED_PROFILE)
    assert len(native._bytes(compact)) <= 8192
    assert compact["omitted_obligation_summaries"] > 0
    assert len(compact["pending_obligations"]) + compact["omitted_obligation_summaries"] == compact["pending_obligation_count"]
    for row in compact["pending_obligations"]:
        assert "guarded_region_ids" in row and row["evaluation_guard_verified"] is False


def test_guarded_advice_reaches_planner_before_goals_with_deferred_semantics(original, monkeypatch):
    root, instruction, state = original
    (root / "bottle.py").write_text(SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    initial = prep.initial_context(state=state, numerical_formalization_profile=native.GUARDED_PROFILE)
    formal = initial["numerical_formalization"]
    assert initial["canonical_tasks_created"] is False and initial["provider_calls"] == 0
    calls = []
    def authored_planner(prompt, **kwargs):
        calls.append(prompt)
        assert native.GUARDED_PROFILE in prompt
        assert "deferred until selected" in prompt and "deferred until assertion failure" in prompt
        assert "guarded_region_ids" in prompt
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
