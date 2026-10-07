"""Source-bound package dependencies; no imported-value or execution proof."""
from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_dependency_graph import (
    PathSource, ProgramDependencyGraph,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_graph import (
    Completeness, ProgramEdgeKind, ProgramGraphRoots, ProgramNodeKind,
)


def _builder(sources, *, previous=None):
    roots = ProgramGraphRoots(forest_id="forest:package", tree_id="tree:package",
        overlay_id="overlay:package", coverage_id="coverage:package",
        included_roots=tuple(sorted(sources)))
    builder = ProgramDependencyGraph(roots, previous=previous)
    builder.build([PathSource(path=path, source=source, language="python")
                   for path, source in sources.items()], previous=previous)
    return builder


def _dependencies(graph, path):
    nodes = {node.node_id: node for node in graph.nodes}
    return [(nodes[edge.target].path, edge) for edge in graph.edges
            if edge.kind is ProgramEdgeKind.DEPENDS_ON
            and nodes[edge.source].kind is ProgramNodeKind.IMPORT
            and nodes[edge.source].path == path]


def _module_dependencies(graph, path):
    nodes = {node.node_id: node for node in graph.nodes}
    return [(nodes[edge.target].path, edge) for edge in graph.edges
            if edge.kind is ProgramEdgeKind.DEPENDS_ON
            and nodes[edge.source].kind is ProgramNodeKind.MODULE
            and nodes[edge.source].path == path]


def _sources(caller, statement):
    sources = {
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": '"""Inert package."""\n',
        "pkg/helpers.py": "def transform(value):\n    return value\n",
        "pkg/sub/helpers.py": "def transform(value):\n    return value\n",
        "helpers.py": "def transform(value):\n    return value\n",
    }
    sources[caller] = statement + "\ndef answer(value):\n    return normalize(value)\n"
    return sources


@pytest.mark.parametrize("caller,statement,target,parents", [
    ("answer.py", "from pkg.helpers import transform as normalize", "pkg/helpers.py", {"pkg/__init__.py"}),
    ("pkg/answer.py", "from .helpers import transform as normalize", "pkg/helpers.py", {"pkg/__init__.py"}),
    ("pkg/sub/answer.py", "from ..helpers import transform as normalize", "pkg/helpers.py", {"pkg/__init__.py", "pkg/sub/__init__.py"}),
    ("pkg/sub/answer.py", "from .helpers import transform as normalize", "pkg/sub/helpers.py", {"pkg/__init__.py", "pkg/sub/__init__.py"}),
    ("pkg/sub/__init__.py", "from ..helpers import transform as normalize", "pkg/helpers.py", {"pkg/__init__.py", "pkg/sub/__init__.py"}),
])
def test_import_resolution_uses_caller_package_and_parent_initializers(caller, statement, target, parents):
    graph = _builder(_sources(caller, statement)).graph
    dependencies = _dependencies(graph, caller)
    assert {path for path, _ in dependencies} == {target, *parents}
    assert "helpers.py" not in {path for path, _ in dependencies}
    assert not graph.frontier_refs
    assert all(edge.attributes["resolution_scope"] == "source_module_dependency" for _, edge in dependencies)
    aliases = [node for node in graph.nodes if node.kind is ProgramNodeKind.ALIAS
               and node.path == caller and node.name == "normalize"]
    assert len(aliases) == 1
    assert aliases[0].attributes["target"] == target.removesuffix(".py").replace("/", ".") + ".transform"


@pytest.mark.parametrize("statement,binding", [
    ("import pkg.sub.helpers as helpers", "pkg.sub.helpers"),
    ("import pkg.sub.helpers", "pkg"),
])
def test_direct_dotted_import_retains_every_package_dependency(statement, binding):
    sources = _sources("answer.py", statement)
    graph = _builder(sources).graph
    assert {path for path, _ in _dependencies(graph, "answer.py")} == {
        "pkg/__init__.py", "pkg/sub/__init__.py", "pkg/sub/helpers.py"}
    alias, = [node for node in graph.nodes if node.kind is ProgramNodeKind.ALIAS and node.path == "answer.py"]
    assert alias.attributes["target"] == binding


@pytest.mark.parametrize("caller,statement", [
    ("answer.py", "from .helpers import transform as normalize"),
    ("pkg/answer.py", "from ..helpers import transform as normalize"),
    ("pkg/sub/answer.py", "from ...helpers import transform as normalize"),
    ("__init__.py", "from .helpers import transform as normalize"),
])
def test_relative_overrun_retains_frontier_and_never_uses_root_helper(caller, statement):
    graph = _builder(_sources(caller, statement)).graph
    assert any(ref.startswith("invalid_relative_import:") for ref in graph.frontier_refs)
    assert not _dependencies(graph, caller)
    assert graph.complete is False


@pytest.mark.parametrize("mutation,reason", [
    ("missing_parent", "missing_import_module:"),
    ("missing_caller_parent", "missing_import_module:"),
    ("missing_target", "missing_import_module:"),
    ("module_package_collision", "ambiguous_import_module:"),
    ("leaf_collision", "ambiguous_import_module:"),
    ("nonpackage_parent", "nonpackage_import_parent:"),
])
def test_unresolved_regular_package_population_never_gets_complete_edges(mutation, reason):
    sources = _sources("pkg/sub/answer.py", "from ..helpers import transform as normalize")
    if mutation == "missing_parent":
        del sources["pkg/__init__.py"]
    elif mutation == "missing_caller_parent":
        del sources["pkg/sub/__init__.py"]
    elif mutation == "missing_target":
        del sources["pkg/helpers.py"]
    elif mutation == "module_package_collision":
        sources["pkg.py"] = ""
    elif mutation == "leaf_collision":
        sources["pkg/helpers/__init__.py"] = ""
    elif mutation == "nonpackage_parent":
        del sources["pkg/__init__.py"]
        sources["pkg.py"] = ""
    graph = _builder(sources).graph
    assert any(ref.startswith(reason) for ref in graph.frontier_refs)
    assert graph.complete is False
    assert "pkg/helpers.py" not in {path for path, _ in _dependencies(graph, "pkg/sub/answer.py")}


def test_package_child_attribute_is_only_a_possible_dependency():
    graph = _builder({"pkg/__init__.py": "child = 3\n", "pkg/child.py": "",
                      "answer.py": "from pkg import child\n"}).graph
    dependencies = dict(_dependencies(graph, "answer.py"))
    assert dependencies["pkg/__init__.py"].attributes["resolved_import"] is True
    assert dependencies["pkg/__init__.py"].attributes["package_initializer"] is True
    assert dependencies["pkg/child.py"].completeness is Completeness.PARTIAL
    assert "resolved_import" not in dependencies["pkg/child.py"].attributes
    assert "possible_submodule_import:answer.py:pkg.child" in graph.frontier_refs
    assert not graph.complete


def test_wildcard_and_missing_external_modules_keep_open_frontiers():
    graph = _builder({"pkg/__init__.py": "", "answer.py": "from pkg import *\nimport unavailable\n"}).graph
    assert "wildcard_import:answer.py:pkg" in graph.frontier_refs
    assert "missing_import_module:answer.py:unavailable" in graph.frontier_refs
    assert not graph.complete


def test_empty_initializer_is_parsed_and_changed_initializer_invalidates_graph():
    sources = _sources("pkg/sub/answer.py", "from ..helpers import transform as normalize")
    warm = _builder(sources)
    assert not warm.graph.frontier_refs
    changed = {**sources, "pkg/__init__.py": '"""Changed source preimage."""\n'}
    incremental = _builder(changed, previous=warm)
    clean = _builder(changed)
    assert incremental.graph.graph_id == clean.graph.graph_id != warm.graph.graph_id
    old_init, = [node for node in warm.graph.nodes if node.kind is ProgramNodeKind.MODULE
                 and node.path == "pkg/__init__.py"]
    new_init, = [node for node in clean.graph.nodes if node.kind is ProgramNodeKind.MODULE
                 and node.path == "pkg/__init__.py"]
    assert old_init.source_sha256 != new_init.source_sha256
    assert old_init.content_id != new_init.content_id
    # Reverse traversal from the initializer reaches the import in its consumer.
    import_nodes = {edge.source for edge in clean.graph.edges_to(new_init.node_id)
                    if edge.kind is ProgramEdgeKind.DEPENDS_ON}
    assert any(node.node_id in import_nodes and node.path == "pkg/sub/answer.py"
               for node in clean.graph.nodes)


def test_dynamic_initializer_is_not_closed_by_structural_import_resolution():
    sources = _sources("answer.py", "from pkg.helpers import transform as normalize")
    sources["pkg/__init__.py"] = "exec('pass')\n"
    graph = _builder(sources).graph
    assert any(ref.startswith("dynamic:") for ref in graph.frontier_refs)
    assert not graph.complete


@pytest.mark.parametrize("body", [
    "def answer(value):\n    return value\n",
    "from other.helpers import transform as normalize\ndef answer(value):\n    return normalize(value)\n",
])
def test_module_loading_exposes_own_initializers_for_absolute_or_absent_imports(body):
    sources = {"pkg/__init__.py": "", "pkg/sub/__init__.py": "",
               "pkg/sub/answer.py": body, "other/__init__.py": "",
               "other/helpers.py": "def transform(value):\n    return value\n"}
    before = _builder(sources).graph
    assert before.complete
    assert {path for path, _ in _module_dependencies(before, "pkg/sub/answer.py")} == {
        "pkg/__init__.py", "pkg/sub/__init__.py"}
    assert {path for path, _ in _module_dependencies(before, "pkg/sub/__init__.py")} == {
        "pkg/__init__.py"}
    assert not _module_dependencies(before, "pkg/__init__.py")
    assert all(edge.attributes["implicit_package_context"] for _, edge in
               _module_dependencies(before, "pkg/sub/answer.py"))
    after = _builder({**sources, "pkg/sub/__init__.py": '"""Changed caller context."""\n'}).graph
    assert before.graph_id != after.graph_id
    init, = [node for node in after.nodes if node.kind is ProgramNodeKind.MODULE
             and node.path == "pkg/sub/__init__.py"]
    reverse_ids = {edge.source for edge in after.edges_to(init.node_id)
                   if edge.kind is ProgramEdgeKind.DEPENDS_ON}
    assert any(node.node_id in reverse_ids and node.kind is ProgramNodeKind.MODULE
               and node.path == "pkg/sub/answer.py" for node in after.nodes)


@pytest.mark.parametrize("population,reason", [
    ({}, "missing_import_module:"),
    ({"pkg.py": ""}, "nonpackage_import_context:"),
    ({"pkg.py": "", "pkg/__init__.py": ""}, "ambiguous_import_module:"),
])
def test_no_import_module_with_unresolved_package_context_stays_open(population, reason):
    graph = _builder({**population, "pkg/answer.py": "def answer(value):\n    return value\n"}).graph
    assert any(ref.startswith(reason) for ref in graph.frontier_refs)
    assert not graph.complete
    assert not _module_dependencies(graph, "pkg/answer.py")
