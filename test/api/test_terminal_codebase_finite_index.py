"""Native finite cache and symbolic planning controls, independent of fitting."""
from copy import deepcopy
import json
from pathlib import Path

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_finite_index as cache
from ipfs_datasets_py.logic.software_contracts.cache import CacheIntegrityError
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from .test_finite_integer_codebase import (
    VIEW, finite_prepared, finite_tools, finite_match, finite_source, finite_git, finite_text,
)


@pytest.fixture
def retained_finite(finite_prepared, finite_tools):
    match = finite_match(finite_prepared, finite_tools)
    assert match["observation"]["status"] == "observed", match
    output = finite_prepared["output"].parent / "finite-index"
    manifest = cache.persist_finite_evidence_index(match=match, output=output)
    return finite_prepared, match, manifest


def query(prepared, expected):
    return cache.query_finite_evidence_index(index=prepared["index"], repository=prepared["repository"],
        expected_head=prepared["expected_head"], expected=expected, scheduler=prepared["scheduler"])


def reseal(manifest):
    body = deepcopy(manifest)
    body["artifacts"] = [cache._pin(pin["path"]) for pin in body["artifacts"]]
    body.pop("manifest_id", None)
    body["manifest_id"] = cache.digest(body)
    (Path(body["output"]) / "manifest.json").write_bytes(cache.wire(body) + b"\n")
    return body


def test_real_native_cache_query_retains_only_historical_facts_after_reopen(retained_finite):
    prepared, match, manifest = retained_finite
    result = query(prepared, manifest)
    assert result["current_facts"] == []
    assert result["historical_fact_ids"] == [fact["fact_id"] for fact in match["current_facts"]]
    assert result["live_source_checked_before_and_after"] is True
    assert result["counts"] == {"observations": 5, "facts": 1, "clauses": 2}
    assert result["cache_authority"] == "historical_candidate_only"
    assert result["fresh_observation_required"] is True
    assert all(result[name] is False for name in ("proof_authority", "execution_authority", "completion_authority"))
    with duckdb.connect(str(Path(manifest["output"]) / "finite.duckdb"), read_only=True,
                        config={"threads": 1, "memory_limit": "64MB"}) as cx:
        assert cx.execute("SELECT input,output FROM observation_rows ORDER BY ordinal").fetchall() == [
            (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)]
        assert cx.execute("SELECT count(*) FROM clause_rows").fetchone()[0] == 2
    assert query(prepared, manifest) == result
    receipt = match["observation"]
    assert prepared["index"].artifacts.get(cid_for_structured(receipt)) == receipt
    assert cid_for_structured(receipt) != receipt["result_cid"]


@pytest.mark.parametrize("control", ["observation_missing", "observation_extra", "observation_value",
    "fact_missing", "fact_payload", "clause_missing", "clause_payload", "match_payload", "renamed_column", "extra_table"])
def test_resealed_db_pin_does_not_hide_native_row_or_schema_corruption(retained_finite, control):
    prepared, _, manifest = retained_finite
    with duckdb.connect(str(Path(manifest["output"]) / "finite.duckdb"), config={"threads": 1, "memory_limit": "64MB"}) as cx:
        if control == "observation_missing":
            cx.execute("DELETE FROM observation_rows WHERE ordinal=4")
        elif control == "observation_extra":
            cx.execute("INSERT INTO observation_rows VALUES (99,99,100,'{}')")
        elif control == "observation_value":
            cx.execute("UPDATE observation_rows SET output=output+1 WHERE ordinal=0")
        elif control == "fact_missing":
            cx.execute("DELETE FROM fact_rows")
        elif control == "fact_payload":
            cx.execute("UPDATE fact_rows SET payload_json='{}'")
        elif control == "clause_missing":
            cx.execute("DELETE FROM clause_rows WHERE ordinal=0")
        elif control == "clause_payload":
            cx.execute("UPDATE clause_rows SET payload_json='{}' WHERE ordinal=0")
        elif control == "match_payload":
            cx.execute("UPDATE match_record SET payload_json='{}'")
        elif control == "renamed_column":
            cx.execute("ALTER TABLE observation_rows RENAME COLUMN output TO caller_output")
        else:
            cx.execute("CREATE TABLE caller_extra (value INTEGER)")
    expected = reseal(manifest)
    assert expected["artifacts"][0]["sha256"] != manifest["artifacts"][0]["sha256"]
    with pytest.raises(ValueError, match="native|schema|rows"):
        query(prepared, expected)


@pytest.mark.parametrize("control", ["wrong_count", "bool_count", "wrong_root", "raw_pin"])
def test_resealed_manifest_refuses_counts_root_or_bytes(retained_finite, control):
    prepared, _, manifest = retained_finite
    expected = deepcopy(manifest)
    if control == "wrong_count":
        expected["counts"]["clauses"] = 1
    elif control == "bool_count":
        expected["counts"]["facts"] = True
    elif control == "wrong_root":
        expected["head"]["snapshot_cid"] = "foreign-source-root"
    else:
        (Path(manifest["output"]) / "match.json").write_bytes(b"{}\n")
        with pytest.raises(ValueError, match="artifact"):
            query(prepared, expected)
        return
    expected = reseal(expected)
    with pytest.raises(ValueError):
        query(prepared, expected)


def test_same_head_dirty_source_cannot_query_retained_finite_rows(retained_finite):
    prepared, _, manifest = retained_finite
    commit = finite_git(prepared["repository"], "rev-parse", "HEAD")
    (prepared["repository"] / "calc.py").write_bytes(finite_source(2))
    assert finite_git(prepared["repository"], "rev-parse", "HEAD") == commit
    with pytest.raises(StaleCodebaseError):
        query(prepared, manifest)
    assert prepared["index"].current(VIEW) == prepared["expected_head"]


def test_immutable_receipt_corruption_refuses_historical_query(retained_finite):
    prepared, match, manifest = retained_finite
    receipt_cid = cid_for_structured(match["observation"])
    prepared["index"].artifacts.path_for(receipt_cid).write_bytes(b"{}")
    with pytest.raises(CacheIntegrityError):
        query(prepared, manifest)


def test_typed_output_preserves_exact_integer_beyond_signed_64_bits(finite_prepared, finite_tools):
    offset = 2**63 - 1
    (finite_prepared["repository"] / "calc.py").write_bytes(finite_source(offset))
    finite_prepared["expected_head"] = finite_prepared["index"].prepare_current(
        finite_prepared["repository"], repository_id=VIEW, operation_id="large-offset",
        expected_head=finite_prepared["expected_head"], scheduler=finite_prepared["scheduler"]).head
    match = finite_match(finite_prepared, finite_tools, source_text=finite_text(offset=offset))
    assert match["residual_clause_ids"] == []
    output = finite_prepared["output"].parent / "large-offset-index"
    manifest = cache.persist_finite_evidence_index(match=match, output=output)
    with duckdb.connect(str(output / "finite.duckdb"), read_only=True,
                        config={"threads": 1, "memory_limit": "64MB"}) as cx:
        assert cx.execute("SELECT output FROM observation_rows WHERE input=2").fetchone()[0] == 2**63 + 1
    result = query(finite_prepared, manifest)
    assert len(result["historical_fact_ids"]) == 2
    assert result["current_facts"] == []
