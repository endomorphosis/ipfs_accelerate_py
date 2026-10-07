"""Qualification must account for every allowed input and native symbol name."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify


@pytest.mark.parametrize("name,source", [("bad.py", "def incomplete(:\n"),
                                        ("unsupported.xyz", "opaque input")])
def test_incomplete_index_cannot_claim_complete_permitted_scope(tmp_path, name, source):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ok.py").write_text("def work():\n    return 1\n")
    (repo / name).write_text(source)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        qualify(repo, tmp_path / "out", ["ok.py", name], "work")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name,qualified", [("pkg/__init__.py", "pkg.work"),
                                            ("__init__.py", "work")])
def test_initializer_names_follow_native_index(tmp_path, name, qualified):
    repo = tmp_path / "repo"
    source = repo / name
    source.parent.mkdir(parents=True)
    source.write_text("def work():\n    return 1\n")
    result = qualify(repo, tmp_path / "out", [name], "work")
    assert result["status"] == "qualified"
    assert result["symbols"] == result["native_fact_rows_replayed"] == 1
    assert result["input_statuses"] == {name: "success"}
    assert result["hits"]["hits"][0]["row"]["qualified_symbol"] == qualified


@pytest.mark.parametrize("query", ["Schedule the jobs within their deadlines", "...!?"])
def test_no_lexical_overlap_retains_index_and_replayable_empty_context(tmp_path, query):
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
        prepare_code_retrieval_context, load_code_retrieval_context,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "worker.py"
    source.write_text("def calculate():\n    return 1\n")
    output = tmp_path / "out"
    result = qualify(repo, output, ["worker.py"], query)
    assert result["status"] == "qualified"
    assert result["symbols"] == result["native_fact_rows_replayed"] == 1
    assert result["complete_permitted_scope"] is True
    assert result["ducklake"]["status"] == "projected"
    assert result["metadata_retrieval"]["worker.py"]["n"] > 0
    assert result["semantic_authority"] is False
    with duckdb.connect(str(output / "vectors.duckdb"), read_only=True,
                        config={"threads": 1}) as connection:
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
            "SELECT payload FROM snapshots WHERE id=?", [result["index_id"]]).fetchone()[0]))
    hits = CodeVectorSearchResult.from_dict(result["hits"])
    assert hits.hits == () and hits.complete and hits.searched_row_count == 1
    assert hits == snapshot.search(hits.query)
    prepared = prepare_code_retrieval_context(repository=repo, task_id="RET-001",
        query_text=query, snapshot=snapshot, result=hits, output=repo / "retrieval.json")
    args = dict(repository=repo, task_id="RET-001", artifact="retrieval.json",
                expected_sha256=prepared["sha256"])
    context = json.loads(load_code_retrieval_context(**args))
    assert context["status"] == "current" and context["hits"] == []
    assert context["semantic_authority"] is context["completion_authority"] is False
    source.write_text("def calculate():\n    return 2\n")
    stale = json.loads(load_code_retrieval_context(**args))
    assert stale["status"] == "stale" and stale["hits"] == []
    assert stale["stale_paths"] == ["worker.py"]
