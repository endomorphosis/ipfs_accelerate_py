"""Current-scope DuckDB ranking preserves native bindings and lexical order."""
from dataclasses import replace
import re
from types import MappingProxyType

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_candidate_index import (
    MAX_ROWS, build_candidate_rows, rank_candidate_ids,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _scan_scoped_sources
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider


@pytest.fixture(scope="module")
def native():
    sources = {
        "alpha.py": b"raise RuntimeError('source must never execute')\n\ndef fix_header(value):\n    return value + 1\n",
        "zeta.py": "def café(value):\n    return value\n\ndef alpha(value):\n    return value\n".encode(),
    }
    state = _scan_scoped_sources(sources, repository_id="candidate-index-test", max_symbols=16)
    provider = IpfsDatasetsSemanticStateProvider()
    bundle = provider.build_semantic_state(state)
    rows = build_candidate_rows(symbols=state.symbols, bundle=bundle, scope_cid="scope:current")
    return dict(sources=sources, state=state, bundle=bundle, rows=rows)


def rank_args(native, **changes):
    return dict(rows=native["rows"], repository_id=native["state"].repository_id,
                scope_cid="scope:current", semantic_root_cid=native["bundle"].root.root_cid,
                query="fix_header", **changes)


def expected(rows, query):
    terms = set(re.findall(r"[a-z][a-z0-9_]+", query.lower()))
    return [row["stable_symbol_id"] for row in sorted(rows, key=lambda row: (
        -len(terms.intersection(re.findall(r"[a-z][a-z0-9_]+",
            (row["qualified_name"] + " " + row["path"]).lower()))),
        row["path"], row["qualified_name"], row["stable_symbol_id"]))]


def test_complete_native_projection_is_deterministic_and_does_not_execute_source(native):
    rows = native["rows"]
    assert len(rows) == len(native["state"].symbols)
    assert rows == build_candidate_rows(symbols=tuple(reversed(native["state"].symbols)),
        bundle=native["bundle"], scope_cid="scope:current")
    assert all(row["source_cid"] and row["capsule_cid"] for row in rows)
    assert all(set(row) == {"repository_id", "scope_cid", "semantic_root_cid",
        "stable_symbol_id", "path", "qualified_name", "source_cid", "version_cid",
        "capsule_cid"} for row in rows)


@pytest.mark.parametrize("query", ["fix_header", "alpha zeta", "", "Kelvin CAFÉ İ alpha",
    "'; DROP TABLE semantic_capsule_candidates; --", "alpha alpha alpha"])
def test_sql_matches_legacy_order_including_unicode_and_query_metacharacters(native, query, tmp_path):
    args = {**rank_args(native), "query": query}
    assert rank_candidate_ids(**args) == expected(native["rows"], query)
    assert rank_candidate_ids(**args, database_path=tmp_path / "metadata.duckdb") == expected(native["rows"], query)


def test_foreign_stale_and_previous_population_cannot_extend_current_selection(native, tmp_path):
    database = tmp_path / "metadata.duckdb"
    args = rank_args(native, database_path=database)
    initial = rank_candidate_ids(**args)
    foreign = [{**row, field: "other:" + row[field]} for field in
               ("repository_id", "scope_cid", "semantic_root_cid") for row in native["rows"]]
    assert rank_candidate_ids(**{**args, "rows": native["rows"] + foreign}) == initial
    subset = native["rows"][:1]
    assert rank_candidate_ids(**{**args, "rows": subset + foreign}) == [subset[0]["stable_symbol_id"]]
    assert rank_candidate_ids(**{**args, "rows": foreign}) == []
    # Replays reconstruct their current input; persisted history is unnecessary.
    database.unlink()
    assert rank_candidate_ids(**rank_args(native)) == initial


def test_current_scope_replaces_only_itself_and_coexists_with_meta_catalog(native, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex
    database = tmp_path / "metadata.duckdb"
    index = SupervisorMetaIndex(database)
    catalog = index.register_catalog(kind="ast", locator_ref="native-blocks", project=False)
    rank_candidate_ids(**rank_args(native, database_path=database))
    older = [{**row, "semantic_root_cid": "root:older"} for row in native["rows"]]
    rank_candidate_ids(**{**rank_args(native, database_path=database),
                          "rows": older, "semantic_root_cid": "root:older"})
    with duckdb.connect(str(database), config={"threads": 1}) as connection:
        assert connection.execute("SELECT COUNT(*) FROM semantic_capsule_candidates").fetchone()[0] == 2 * len(older)
        assert connection.execute("SELECT catalog_id FROM catalogs").fetchone()[0] == catalog["catalog_id"]


def test_full_ranking_is_not_truncated_to_worker_capsule_limit(native):
    base = native["rows"][0]
    rows = [{**base, "stable_symbol_id": f"symbol:{number:04}",
             "qualified_name": f"scope.symbol_{number:04}"} for number in range(MAX_ROWS)]
    result = rank_candidate_ids(**{**rank_args(native), "rows": rows})
    assert len(result) == MAX_ROWS
    assert result == expected(rows, "fix_header")


def test_database_has_one_thread_bounded_memory_and_no_spill(native, monkeypatch):
    connect = duckdb.connect
    settings = []
    def inspect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        settings.append(connection.execute("""SELECT current_setting('threads'),
            current_setting('memory_limit'), current_setting('temp_directory'),
            current_setting('max_temp_directory_size')""").fetchone())
        return connection
    monkeypatch.setattr(duckdb, "connect", inspect)
    rank_candidate_ids(**rank_args(native))
    threads, memory, temporary, spill = settings[0]
    amount, unit = memory.split()
    assert threads == 1 and unit in {"MiB", "MB"} and float(amount) <= 64
    assert temporary == "" and spill == "0 bytes"


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "foreign", "changed_source", "not_native"])
def test_projection_refuses_population_or_native_symbol_binding_changes(native, mutation):
    symbols = list(native["state"].symbols)
    if mutation == "partial":
        symbols.pop()
    elif mutation == "duplicate":
        symbols.append(symbols[0])
    elif mutation == "foreign":
        symbols[0] = replace(symbols[0], source_cid=native["rows"][-1]["source_cid"])
    elif mutation == "changed_source":
        changed = dict(native["sources"])
        changed["alpha.py"] = changed["alpha.py"].replace(b"value + 1", b"value + 7")
        state = _scan_scoped_sources(changed, repository_id=native["state"].repository_id, max_symbols=16)
        symbols = list(state.symbols)
    else:
        symbols[0] = {"stable_id": symbols[0].stable_id}
    with pytest.raises((ValueError, KeyError)):
        build_candidate_rows(symbols=symbols, bundle=native["bundle"], scope_cid="scope:current")


@pytest.mark.parametrize("target", ["root", "index", "capsule"])
def test_projection_independently_refuses_tampered_native_blocks(native, target):
    bundle = native["bundle"]
    cid = {"root": bundle.root.root_cid, "index": bundle.root.capsule_index_cid,
           "capsule": native["rows"][0]["capsule_cid"]}[target]
    broken = object.__new__(type(bundle))
    object.__setattr__(broken, "root", bundle.root)
    object.__setattr__(broken, "blocks", MappingProxyType({**bundle.blocks, cid: b"{}"}))
    with pytest.raises(ValueError):
        build_candidate_rows(symbols=native["state"].symbols, bundle=broken, scope_cid="scope:current")


@pytest.mark.parametrize("mutation", ["too_many", "query_bytes", "nontext_query", "extra_field",
    "duplicate", "escape_path", "absolute_path", "oversized_field", "empty_identity"])
def test_invalid_bounds_and_rows_fail_before_database_creation(native, tmp_path, mutation):
    args = rank_args(native, database_path=tmp_path / "absent.duckdb")
    if mutation == "too_many":
        args["rows"] = native["rows"] * (MAX_ROWS + 1)
    elif mutation == "query_bytes":
        args["query"] = "λ" * 4097
    elif mutation == "nontext_query":
        args["query"] = True
    elif mutation == "duplicate":
        args["rows"] = [native["rows"][0]] * 2
    elif mutation == "empty_identity":
        args["repository_id"] = ""
    else:
        changed = dict(native["rows"][0])
        changed.update({"extra_field": {"authority": True}, "escape_path": {"path": "../outside"},
            "absolute_path": {"path": "/outside"}, "oversized_field": {"qualified_name": "q" * 8193}}[mutation])
        args["rows"] = [changed]
    with pytest.raises(ValueError):
        rank_candidate_ids(**args)
    assert not args["database_path"].exists()


def test_no_extra_gate_control_database(native, tmp_path):
    with pytest.raises(ValueError, match="control.duckdb"):
        rank_candidate_ids(**rank_args(native, database_path=tmp_path / "control.duckdb"))
    assert not (tmp_path / "control.duckdb").exists()
    alias = tmp_path / "alias.duckdb"
    alias.symlink_to(tmp_path / "control.duckdb")
    with pytest.raises(ValueError, match="control.duckdb"):
        rank_candidate_ids(**rank_args(native, database_path=alias))
    assert not (tmp_path / "control.duckdb").exists()


def test_total_metadata_byte_budget_is_enforced_before_database_creation(native, tmp_path):
    base = native["rows"][0]
    rows = [{**base, "stable_symbol_id": f"symbol:{number}",
             "qualified_name": "q" * 8192} for number in range(MAX_ROWS)]
    database = tmp_path / "absent.duckdb"
    with pytest.raises(ValueError, match="metadata byte bound"):
        rank_candidate_ids(**{**rank_args(native, database_path=database), "rows": rows})
    assert not database.exists()
