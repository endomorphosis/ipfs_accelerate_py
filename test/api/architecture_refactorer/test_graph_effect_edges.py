"""Hermetic PCAR-003 effect, schema, test, and proof edge tests."""

from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureIR,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.graph_builder import (
    EXTRACTOR_IDENTITY,
    call_targets,
    edges_of,
    extract_architecture_graph,
    nodes_of,
)

_TREE = "pcar-003-effect-fixture-tree"
_FRESHNESS = "pcar-003-effect-fixture"

_EFFECT_MODULE = '''import json
import logging
import subprocess
from pathlib import Path

SCHEMA = {"type": "object", "properties": {"id": {"type": "integer"}}}


def operation(fn):
    return fn


@operation
def load_config(path: str) -> dict:
    text = Path(path).read_text()
    return json.loads(text)


@operation
def save_config(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload))


def mutate_handle(path: str) -> str:
    with open(path, "w") as handle:
        handle.write("ok")
    with open(path) as handle:
        return handle.read()


def run_tool() -> None:
    subprocess.run(["echo", "ok"], check=True)


def observe() -> None:
    logging.info("seen")
    print("seen")


def persist(conn, row) -> None:
    conn.execute("insert", row)
    conn.commit()


class Store:
    def dispatch(self, name: str) -> None:
        action = getattr(self, name)
        action()

    def write_state(self) -> None:
        self.conn.execute("update")

    def read_state(self) -> None:
        logging.info("read")
'''

_TEST_MODULE = '''from pkg.effects import load_config, save_config


def test_load_config() -> None:
    assert load_config("cfg.json") == {}


def test_save_config() -> None:
    save_config("cfg.json", {})
'''

_PROOF_JSON = {
    "proves": "pkg.effects.load_config",
    "obligation": "load_config deserializes SCHEMA",
}


def _graph() -> ArchitectureIR:
    return extract_architecture_graph(
        {
            "pkg/__init__.py": "",
            "pkg/effects.py": _EFFECT_MODULE,
            "test/test_effects.py": _TEST_MODULE,
            "proofs/load_config.proof.json": json.dumps(_PROOF_JSON),
            "pkg/effects.schema.json": json.dumps(
                {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}
            ),
        },
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )


def _node(graph: ArchitectureIR, kind: NodeKind, identity: str):
    exact = f"n:{kind.value}:{identity}"
    for node in graph.nodes:
        if node.node_id == exact:
            return node
    matches = [node for node in graph.nodes if node.kind is kind and identity in node.node_id]
    assert matches, f"missing {kind.value} containing {identity!r}"
    return matches[0]


def _symbol(graph: ArchitectureIR, qualified: str):
    return _node(graph, NodeKind.SYMBOL, qualified)


def _edges_from(graph: ArchitectureIR, source: str, *kinds: EdgeKind):
    wanted = frozenset(kinds) if kinds else None
    return [
        edge
        for edge in graph.edges
        if edge.source == source and (wanted is None or edge.kind in wanted)
    ]


def test_effect_edges_cover_read_write_mutate_observe_persist_and_serde() -> None:
    graph = _graph()
    load = _symbol(graph, "pkg.effects.load_config").node_id
    save = _symbol(graph, "pkg.effects.save_config").node_id
    mutate = _symbol(graph, "pkg.effects.mutate_handle").node_id
    tool = _symbol(graph, "pkg.effects.run_tool").node_id
    observe = _symbol(graph, "pkg.effects.observe").node_id
    persist = _symbol(graph, "pkg.effects.persist").node_id
    assert any(edge.kind is EdgeKind.READS for edge in _edges_from(graph, load))
    assert any(edge.kind is EdgeKind.DESERIALIZES for edge in _edges_from(graph, load))
    assert any(edge.kind is EdgeKind.WRITES for edge in _edges_from(graph, save))
    assert any(edge.kind is EdgeKind.SERIALIZES for edge in _edges_from(graph, save))
    mutate_kinds = {edge.kind for edge in _edges_from(graph, mutate)}
    assert EdgeKind.WRITES in mutate_kinds
    assert EdgeKind.READS in mutate_kinds
    assert any(edge.kind is EdgeKind.EXECUTES for edge in _edges_from(graph, tool))
    assert any(edge.kind is EdgeKind.OBSERVES for edge in _edges_from(graph, observe))
    persist_kinds = {edge.kind for edge in _edges_from(graph, persist)}
    assert EdgeKind.MUTATES in persist_kinds
    assert EdgeKind.PERSISTS in persist_kinds
    effect_nodes = nodes_of(graph, NodeKind.EFFECT)
    assert effect_nodes
    classes = {node.node_id for node in effect_nodes}
    assert any("filesystem" in node_id for node_id in classes)
    assert any("process" in node_id for node_id in classes)
    assert any("logging" in node_id for node_id in classes)
    assert any("state" in node_id for node_id in classes)
    for edge in edges_of(
        graph,
        EdgeKind.READS,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.OBSERVES,
        EdgeKind.PERSISTS,
        EdgeKind.SERIALIZES,
        EdgeKind.DESERIALIZES,
        EdgeKind.EXECUTES,
    ):
        assert edge.provenance.extractor_identity == EXTRACTOR_IDENTITY
        target = next(node for node in graph.nodes if node.node_id == edge.target)
        assert target.kind in {
            NodeKind.EFFECT,
            NodeKind.STATE,
            NodeKind.SCHEMA,
            NodeKind.SYMBOL,
            NodeKind.OPERATION,
            NodeKind.MODULE,
            NodeKind.ENTRYPOINT,
        }


def test_schema_operation_test_and_proof_edges() -> None:
    graph = _graph()
    schema_nodes = nodes_of(graph, NodeKind.SCHEMA)
    assert any("pkg.effects.SCHEMA" in node.node_id or "effects.schema.json" in node.node_id for node in schema_nodes)
    json_schema = _node(graph, NodeKind.SCHEMA, "pkg/effects.schema.json")
    assert json_schema.provenance.span.path == "pkg/effects.schema.json"
    operations = {node.node_id for node in nodes_of(graph, NodeKind.OPERATION)}
    assert any("pkg.effects.load_config" in node_id for node_id in operations)
    assert any("pkg.effects.save_config" in node_id for node_id in operations)
    load_op = _node(graph, NodeKind.OPERATION, "pkg.effects.load_config")
    load_sym = _symbol(graph, "pkg.effects.load_config")
    implements = edges_of(graph, EdgeKind.IMPLEMENTS)
    assert any(edge.source == load_sym.node_id and edge.target == load_op.node_id for edge in implements)
    test_load = _node(graph, NodeKind.TEST, "test.test_effects.test_load_config")
    tests = edges_of(graph, EdgeKind.TESTS)
    assert any(edge.source == test_load.node_id and edge.target == load_sym.node_id for edge in tests)
    proof = _node(graph, NodeKind.PROOF, "proofs/load_config.proof.json")
    proves = edges_of(graph, EdgeKind.PROVES)
    assert any(edge.source == proof.node_id and edge.target == load_sym.node_id for edge in proves)
    proven = [edge for edge in proves if edge.source == proof.node_id]
    assert proven
    assert all(edge.provenance.confidence is Confidence.EXACT for edge in proven)


def test_open_without_literal_mode_stays_conservative_or_exact_by_default() -> None:
    graph = extract_architecture_graph(
        {
            "pkg/io.py": (
                "def write_unknown(path, mode):\n"
                "    handle = open(path, mode)\n"
                "    handle.write('x')\n"
                "    return handle.read()\n"
            )
        },
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    writer = _symbol(graph, "pkg.io.write_unknown")
    open_effects = [
        edge
        for edge in _edges_from(graph, writer.node_id, EdgeKind.READS, EdgeKind.WRITES, EdgeKind.MUTATES)
        if edge.provenance.span.start_line == 2
    ]
    assert open_effects
    assert all(edge.provenance.confidence in {Confidence.CONSERVATIVE, Confidence.EXACT} for edge in open_effects)
    writes = _edges_from(graph, writer.node_id, EdgeKind.WRITES)
    reads = _edges_from(graph, writer.node_id, EdgeKind.READS)
    assert writes
    assert reads


def test_dynamic_effect_dispatch_widens_to_effectful_methods() -> None:
    graph = _graph()
    dispatch = _symbol(graph, "pkg.effects.Store.dispatch")
    write_state = _symbol(graph, "pkg.effects.Store.write_state")
    read_state = _symbol(graph, "pkg.effects.Store.read_state")
    targets = set(call_targets(graph, dispatch.node_id))
    assert write_state.node_id in targets
    assert read_state.node_id in targets
    dynamic_calls = [
        edge
        for edge in edges_of(graph, EdgeKind.CALLS)
        if edge.source == dispatch.node_id
    ]
    assert all(edge.provenance.confidence is not Confidence.EXACT for edge in dynamic_calls)
    write_effects = _edges_from(graph, write_state.node_id, EdgeKind.MUTATES)
    read_effects = _edges_from(graph, read_state.node_id, EdgeKind.OBSERVES)
    assert write_effects
    assert read_effects
    assert any(edge.provenance.confidence is Confidence.OPAQUE for edge in dynamic_calls)


def test_heuristic_and_opaque_effect_facts_are_not_promoted_to_exact() -> None:
    graph = _graph()
    non_exact = [
        edge
        for edge in graph.edges
        if edge.provenance.confidence in {Confidence.HEURISTIC, Confidence.OPAQUE}
    ]
    assert non_exact
    assert all(edge.provenance.confidence is not Confidence.EXACT for edge in non_exact)
    for node in nodes_of(graph, NodeKind.EFFECT):
        if "dynamic" in node.node_id:
            assert node.provenance.confidence in {Confidence.CONSERVATIVE, Confidence.OPAQUE}
