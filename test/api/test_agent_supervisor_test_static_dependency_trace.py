"""PTR-020 contract tests for deterministic static dependency tracing."""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    AnalysisASTIndex,
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_static_dependency_trace import (
    STATIC_TEST_DEPENDENCY_TRACE_INTERFACE,
    STATIC_TEST_DEPENDENCY_TRACE_SCHEMA,
    STATIC_TRACE_ANALYZER_INTERFACE,
    STATIC_TRACE_LIMITS_INTERFACE,
    StaticTestDependencyTracer,
    StaticTraceError,
    StaticTraceLimits,
    trace_static_dependencies,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_python_ast_blob_record,
)
from multiformats import CID


def _index(
    root: Path,
    sources: Mapping[str, str],
    *,
    order: Iterable[str] | None = None,
) -> AnalysisASTIndex:
    records = []
    for relative in order if order is not None else sources:
        source = sources[relative]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        records.append(
            (
                relative,
                build_python_ast_blob_record(
                    source,
                    blob_identity=f"blob:{relative}:{hashlib.sha256(source.encode()).hexdigest()}",
                ),
            )
        )
    return build_analysis_ast_index(records)


def _frontier_kinds(trace: object) -> set[str]:
    return {item.kind for item in trace.unknown_frontier}  # type: ignore[attr-defined]


def _assert_profile_cid(trace_cid: str, canonical_bytes: bytes) -> None:
    parsed = CID.decode(trace_cid)
    assert parsed.version == 1
    assert parsed.base.name == "base32"
    assert parsed.codec.name == "dag-json"
    assert parsed.hashfun.name == "sha2-256"
    assert bytes(parsed.raw_digest) == hashlib.sha256(canonical_bytes).digest()


def test_trace_is_deterministic_canonical_content_addressed_and_body_free(
    tmp_path: Path,
) -> None:
    sources = {
        "tests/test_demo.py": """from app.util import helper

def test_demo(answer):
    assert helper() == answer
""",
        "tests/conftest.py": """@pytest.fixture
def answer():
    return 42
""",
        "app/util.py": """def helper():
    ignored_body_literal = \"BODY-MUST-NOT-BE-PERSISTED\"
    return 42
""",
    }
    forward = _index(tmp_path, sources)
    reverse = _index(tmp_path, sources, order=reversed(tuple(sources)))

    first = trace_static_dependencies(
        forward,
        "tests/test_demo.py",
        repository_root=tmp_path,
        test_symbol="test_demo",
    )
    second = StaticTestDependencyTracer(reverse, tmp_path).trace(
        "tests/test_demo.py", node_id="tests/test_demo.py::test_demo[param-value]"
    )

    assert first.cid == second.cid
    assert first.canonical_bytes == second.canonical_bytes
    assert first.complete and first.completeness == "complete"
    assert first.interface == STATIC_TEST_DEPENDENCY_TRACE_INTERFACE
    assert first.schema == STATIC_TEST_DEPENDENCY_TRACE_SCHEMA
    assert first.verify() is first
    assert (
        json.dumps(
            first.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        == first.canonical_bytes
    )
    _assert_profile_cid(first.cid, first.canonical_bytes)

    payload = first.to_dict()
    assert payload["limits"]["interface"] == STATIC_TRACE_LIMITS_INTERFACE
    assert payload["analyzer"]["interface"] == STATIC_TRACE_ANALYZER_INTERFACE
    assert payload["analyzer_cid"].startswith("b")
    assert {edge["kind"] for edge in payload["dependencies"]["edges"]} == {
        "decorator",
        "fixture",
        "import",
    }
    assert all("source" not in row for row in payload["dependencies"]["nodes"])
    serialized = first.canonical_bytes.decode()
    assert "BODY-MUST-NOT-BE-PERSISTED" not in serialized
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"max_files": 0}, "max_files"),
        ({"max_edges": True}, "max_edges"),
        ({"max_source_bytes": 100_000_000}, "max_source_bytes"),
        ({"max_text_chars": 8}, "max_text_chars"),
    ],
)
def test_trace_limits_are_hard_validated(changes: dict[str, object], match: str) -> None:
    with pytest.raises(StaticTraceError, match=match):
        StaticTraceLimits(**changes)  # type: ignore[arg-type]


def test_fixtures_hooks_plugins_configuration_and_data_are_closed(
    tmp_path: Path,
) -> None:
    sources = {
        "tests/test_integration.py": """def test_integration(dependent):
    with open(\"payload.txt\") as stream:
        assert stream.read() == dependent
""",
        "tests/conftest.py": """pytest_plugins = (\"plugins.local\",)

@pytest.fixture
def base():
    return \"payload\"

@pytest.fixture
def dependent(base):
    return base

def pytest_collection_modifyitems(items):
    pass
""",
        "plugins/local.py": 'PLUGIN_NAME = "local"\n',
    }
    index = _index(tmp_path, sources)
    (tmp_path / "tests/payload.txt").write_text("payload", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '-q'\n", encoding="utf-8"
    )

    trace = trace_static_dependencies(
        index,
        "tests/test_integration.py",
        repository_root=tmp_path,
        test_symbol="test_integration",
    )

    assert trace.complete
    payload = trace.to_dict()
    edges = payload["dependencies"]["edges"]
    assert {edge["kind"] for edge in edges} == {
        "config",
        "data",
        "decorator",
        "fixture",
        "hook",
        "plugin",
    }
    nodes = payload["dependencies"]["nodes"]
    assert {node["path"] for node in nodes} == {
        "plugins/local.py",
        "pyproject.toml",
        "tests/conftest.py",
        "tests/payload.txt",
        "tests/test_integration.py",
    }
    assert {node["kind"] for node in nodes} == {"data", "source"}
    assert payload["health"]["source_hashes_verified"] is True


def test_dynamic_import_reflection_opaque_decorator_and_missing_import_are_unknown(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "test_unknown.py": """import absent_package
from importlib import import_module as load

@custom_decorator
def test_unknown(module_name):
    imported = load(module_name)
    return getattr(imported, module_name)
"""
        },
    )

    trace = trace_static_dependencies(
        index,
        "test_unknown.py",
        repository_root=tmp_path,
        test_symbol="test_unknown",
    )

    assert not trace.complete
    assert {
        "dynamic_import",
        "reflection",
        "opaque_decorator",
        "missing_file",
        "unresolved_fixture",
    }.issubset(_frontier_kinds(trace))
    assert all(
        item.frontier_id.startswith("static-frontier:sha256:") for item in trace.unknown_frontier
    )


def test_native_extension_import_is_an_explicit_unknown_frontier(tmp_path: Path) -> None:
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    (tmp_path / f"native_dependency{suffix}").write_bytes(b"not executed")
    index = _index(
        tmp_path, {"test_native.py": "import native_dependency\n\ndef test_native(): pass\n"}
    )

    trace = trace_static_dependencies(
        index, "test_native.py", repository_root=tmp_path, test_symbol="test_native"
    )

    assert "native_code" in _frontier_kinds(trace)
    native = next(item for item in trace.unknown_frontier if item.kind == "native_code")
    assert native.target == "native_dependency"


def test_path_reads_close_over_data_and_path_writes_are_uncontrolled(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "test_paths.py": """from pathlib import Path


def test_paths():
    assert Path("payload.txt").read_text() == "payload"
    Path("output.txt").open("w")
""",
        },
    )
    (tmp_path / "payload.txt").write_text("payload", encoding="utf-8")

    trace = trace_static_dependencies(
        index, "test_paths.py", repository_root=tmp_path, test_symbol="test_paths"
    )

    dependencies = trace.to_dict()["dependencies"]
    nodes = dependencies["nodes"]
    assert any(node["kind"] == "data" and node["path"] == "payload.txt" for node in nodes)
    assert not any(node["path"] == "output.txt" for node in nodes)
    assert any(
        edge["kind"] == "effect" and edge["target_symbol"] == "filesystem_write"
        for edge in dependencies["edges"]
    )
    assert any(
        item.kind == "uncontrolled_effect" and item.target == "filesystem_write"
        for item in trace.unknown_frontier
    )


def test_analysis_file_and_frontier_bounds_are_explicit_and_deterministic(
    tmp_path: Path,
) -> None:
    sources = {
        "test_bound.py": """import one
import two
import three

def test_bound(): pass
""",
        "one.py": "ONE = 1\n",
        "two.py": "TWO = 2\n",
        "three.py": "THREE = 3\n",
    }
    index = _index(tmp_path, sources)
    limits = StaticTraceLimits(max_files=1, max_frontier=1)

    first = trace_static_dependencies(
        index,
        "test_bound.py",
        repository_root=tmp_path,
        test_symbol="test_bound",
        limits=limits,
    )
    second = trace_static_dependencies(
        index,
        "test_bound.py",
        repository_root=tmp_path,
        test_symbol="test_bound",
        limits=limits,
    )

    assert first.cid == second.cid
    assert len(first.unknown_frontier) == 1
    assert first.unknown_frontier[0].kind == "analysis_bound"
    assert first.unknown_frontier[0].target == "frontier"
    assert first.to_dict()["health"]["analysis_bounds_reached"] == [
        "files",
        "frontier",
    ]


def test_stale_ast_index_fails_closed_without_parsing_new_source(tmp_path: Path) -> None:
    index = _index(tmp_path, {"test_stale.py": "def test_stale(): return 1\n"})
    (tmp_path / "test_stale.py").write_text(
        "def test_stale(): return 'changed body'\n", encoding="utf-8"
    )

    trace = trace_static_dependencies(
        index, "test_stale.py", repository_root=tmp_path, test_symbol="test_stale"
    )

    assert _frontier_kinds(trace) == {"stale_ast_index"}
    assert trace.analyzed_file_count == 0
    assert trace.to_dict()["health"]["source_hashes_verified"] is False
    assert "changed body" not in trace.canonical_bytes.decode()


def test_missing_and_unparseable_root_files_have_typed_frontiers(tmp_path: Path) -> None:
    missing_index = build_analysis_ast_index([])
    missing = trace_static_dependencies(missing_index, "test_missing.py", repository_root=tmp_path)
    assert _frontier_kinds(missing) == {"missing_file"}

    broken_source = "def test_broken(:\n"
    broken_index = _index(tmp_path, {"test_broken.py": broken_source})
    broken = trace_static_dependencies(broken_index, "test_broken.py", repository_root=tmp_path)
    assert _frontier_kinds(broken) == {"parse_error"}
    assert broken.to_dict()["health"]["parser_healthy"] is False


def test_relative_import_resolution_and_import_cycles_terminate(tmp_path: Path) -> None:
    sources = {
        "pkg/test_relative.py": "from .helper import value\n\ndef test_relative(): assert value == 1\n",
        "pkg/helper.py": "from .other import other\nvalue = other\n",
        "pkg/other.py": "from .helper import value\nother = 1\n",
    }
    index = _index(tmp_path, sources)

    trace = trace_static_dependencies(
        index,
        "pkg/test_relative.py",
        repository_root=tmp_path,
        test_symbol="test_relative",
    )

    assert trace.complete
    assert trace.analyzed_file_count == 3
    assert {
        edge["target_path"]
        for edge in trace.to_dict()["dependencies"]["edges"]
        if edge["kind"] == "import"
    } == {"pkg/helper.py", "pkg/other.py"}


def test_identity_provider_cannot_substitute_noncanonical_trace_bytes(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path, {"test_identity.py": "def test_identity(): pass\n"})

    with pytest.raises(StaticTraceError, match="analyzer identity bytes"):
        trace_static_dependencies(
            index,
            "test_identity.py",
            repository_root=tmp_path,
            identity_minter=lambda _value: mint_content_identity({"substituted": True}),
        )
