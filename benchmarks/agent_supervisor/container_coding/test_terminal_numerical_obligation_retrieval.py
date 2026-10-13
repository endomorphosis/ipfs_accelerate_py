"""Current-owner pagination, exact types and source-edit fences with real lakes."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_context import context


def _report(receipt):
    output = numerical.Path(receipt["output"])
    index = json.loads((output / "reports.json").read_bytes())
    return json.loads((output / index["sample.py"]["artifact"]).read_bytes())


def _page(root, receipt, **kwargs):
    return numerical.retrieve_numerical_obligations(repository=root,
        expected_receipt=receipt, path="sample.py", **kwargs)


@pytest.mark.parametrize("max_bytes", [4096, 8192])
def test_all_rows_retrieved_exactly_once_with_owner_bound_utf8_budget(context, max_bytes):
    root, receipt = context
    report = _report(receipt)
    rows, cursor, seen = [], None, set()
    for _ in range(257):
        page = _page(root, receipt, cursor=cursor, max_bytes=max_bytes)
        assert page["schema"] == numerical.PAGE_SCHEMA
        assert page["context_receipt_sha256"] == receipt["receipt_sha256"]
        assert page["source_tree_id"] == receipt["record"]["source_tree_id"]
        assert page["binding"]["source_sha256"] == receipt["source_hashes"]["sample.py"]
        assert page["binding"]["report_id"] == report["report_sha256"]
        assert len(numerical._page_bytes(page)) <= max_bytes
        assert page["source_obligations_discharged"] == 0
        assert page["proof_authority"] is False
        rows.extend(page["rows"])
        cursor = page["next_cursor"]
        if cursor is None:
            assert page["exhausted"] is True
            break
        assert page["exhausted"] is False
        identity = numerical._page_bytes(cursor)
        assert identity not in seen
        seen.add(identity)
    else:
        pytest.fail("pagination did not terminate")
    assert rows == report["pending_obligations"]


def test_function_kind_filter_is_exact_and_cannot_reuse_another_query(context):
    root, receipt = context
    report = _report(receipt)
    row = report["pending_obligations"][0]
    function_id = row["function_ids"][0]
    page = _page(root, receipt, function_id=function_id, kind=row["kind"])
    expected = [r for r in report["pending_obligations"]
                if function_id in r["function_ids"] and r["kind"] == row["kind"]]
    assert page["matched_count"] == len(expected)
    assert page["rows"] == expected[:page["returned_count"]]
    first = _page(root, receipt, limit=1)
    assert first["next_cursor"] is not None
    for changed in ({"limit": 2}, {"limit": 1, "kind": row["kind"]},
                    {"limit": 1, "function_id": function_id}, {"limit": 1, "max_bytes": 4096}):
        with pytest.raises(ValueError):
            _page(root, receipt, cursor=first["next_cursor"], **changed)


def test_continuation_cannot_cross_owner_receipts_with_identical_source(context):
    root, receipt = context
    first = _page(root, receipt, limit=1)
    other = numerical.prepare_numerical_context(repository=root, source_hashes=receipt["source_hashes"],
        output=root.parent / "other-context", profile=numerical.PROFILE)
    assert other["record"]["source_tree_id"] == receipt["record"]["source_tree_id"]
    with pytest.raises(ValueError, match="selected context receipt"):
        _page(root, other, cursor=first["next_cursor"], limit=1)


@pytest.mark.parametrize("cursor", [False, {}, {"context_receipt_sha256": "foreign", "cursor": {}},
                                      {"context_receipt_sha256": "foreign", "cursor": None}])
def test_malformed_owner_continuation_refused(context, cursor):
    with pytest.raises(ValueError, match="continuation"):
        _page(*context, cursor=cursor)


@pytest.mark.parametrize("path", ["../sample.py", "/sample.py", "./sample.py", "other.py", True])
def test_retrieval_cannot_expand_selected_path_population(context, path):
    root, receipt = context
    with pytest.raises(ValueError, match="exact selected source path"):
        numerical.retrieve_numerical_obligations(repository=root, expected_receipt=receipt, path=path)


@pytest.mark.parametrize("change", ["summary_flag", "summary_count", "record_flag", "hydration_flag"])
def test_rehashed_bool_integer_substitution_is_not_exact_evidence(context, change):
    root, receipt = context
    changed = deepcopy(receipt)
    if change == "summary_flag":
        changed["summary"]["proof_authority"] = 0
    elif change == "summary_count":
        assert changed["summary"]["functions"] == 1
        changed["summary"]["functions"] = True
    elif change == "record_flag":
        changed["record"]["proposal_only"] = 1
    else:
        changed["hydration"]["ducklake_verified"] = 1
    body = {k: v for k, v in changed.items() if k != "receipt_sha256"}
    raw = numerical._bytes(body)
    (numerical.Path(changed["output"]) / "receipt.json").write_bytes(raw)
    changed["receipt_sha256"] = numerical._sha(raw)
    with pytest.raises(ValueError, match="planning/world evidence"):
        _page(root, changed)


@pytest.mark.parametrize("change", ["source", "report", "producer"])
def test_mutation_during_datasets_projection_cannot_escape_currentness_fence(context, monkeypatch, change):
    from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager
    root, receipt = context
    original = pager.page_numerical_obligations
    def mutated(**kwargs):
        result = original(**kwargs)
        if change == "source":
            with (root / "sample.py").open("a") as stream:
                stream.write("\n# concurrent edit\n")
        elif change == "report":
            with (numerical.Path(receipt["output"]) / "report-0000.json").open("a") as stream:
                stream.write("\n")
        else:
            monkeypatch.setattr(numerical, "_producers", lambda *args: {"changed": "producer"})
        return result
    monkeypatch.setattr(pager, "page_numerical_obligations", mutated)
    with pytest.raises(ValueError):
        _page(root, receipt)


@pytest.mark.parametrize("change", ["index_locator", "report_bytes"])
def test_reread_artifacts_are_bound_before_following_locators_or_parsing(context, monkeypatch, change):
    root, receipt = context
    output = numerical.Path(receipt["output"])
    original = numerical.validate_numerical_context
    outside = root.parent / "not-a-selected-report.json"
    outside.write_text("{}")
    def after_validation(**kwargs):
        result = original(**kwargs)
        if change == "index_locator":
            path = output / "reports.json"
            index = json.loads(path.read_bytes())
            index["sample.py"]["artifact"] = str(outside)
            path.write_bytes(numerical._bytes(index))
        else:
            (output / "report-0000.json").write_text("not JSON")
        return result
    original_read = numerical._read
    def bounded_read(path, *args):
        assert numerical.Path(path) != outside, "followed an unbound external artifact locator"
        return original_read(path, *args)
    monkeypatch.setattr(numerical, "validate_numerical_context", after_validation)
    monkeypatch.setattr(numerical, "_read", bounded_read)
    with pytest.raises(ValueError, match="numerical retrieval (index|report) changed"):
        _page(root, receipt)


@pytest.mark.parametrize("kwargs", [{"limit": True}, {"limit": 0}, {"limit": 9},
                                    {"max_bytes": True}, {"max_bytes": 0}, {"max_bytes": 8193}])
def test_invalid_budgets_fail_before_owner_io(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(numerical, "validate_numerical_context",
        lambda **kwargs: pytest.fail("invalid budget reached owner IO"))
    with pytest.raises(ValueError, match="page .* limit"):
        numerical.retrieve_numerical_obligations(repository=tmp_path,
            expected_receipt=None, path="sample.py", **kwargs)
