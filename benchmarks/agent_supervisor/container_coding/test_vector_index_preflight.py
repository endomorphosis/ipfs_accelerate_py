"""Qualification must account for every allowed input and native symbol name."""
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
