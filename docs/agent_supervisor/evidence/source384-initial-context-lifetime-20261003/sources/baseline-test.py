"""Native context producers with weak lifetime witnesses; no GC or RSS claims."""
import weakref
from pathlib import Path

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import vector_index_preflight
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot


class _WeakDict(dict):
    pass


class _WeakText(str):
    pass


class _ViewWitness:
    def __init__(self, native):
        self.native = native

    @property
    def root(self):
        return self.native.root


def _track_native_construction(monkeypatch):
    refs = {}

    def track(name, value):
        refs[name] = weakref.ref(value)
        return value

    native_connect = duckdb.connect

    class Reader:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, query, parameters):
            assert query == "SELECT payload FROM snapshots WHERE id=?"
            self.connection.execute(query, parameters)
            return self

        def fetchone(self):
            row = self.connection.fetchone()
            return (track("serialized_snapshot", _WeakText(row[0])),)

    def connect(path, *args, **kwargs):
        connection = native_connect(path, *args, **kwargs)
        if kwargs.get("read_only") and Path(path).name == "vectors.duckdb":
            return Reader(connection)
        return connection

    monkeypatch.setattr(duckdb, "connect", connect)
    native_retrieval = initial.prepare_code_retrieval_context

    def retrieval(**kwargs):
        track("snapshot", kwargs["snapshot"])
        assert isinstance(kwargs["snapshot"], CodeVectorIndexSnapshot)
        return track("retrieval", _WeakDict(native_retrieval(**kwargs)))

    monkeypatch.setattr(initial, "prepare_code_retrieval_context", retrieval)
    native_semantic = initial.prepare_semantic_context
    monkeypatch.setattr(initial, "prepare_semantic_context", lambda **kwargs:
                        track("semantic_summary", _WeakDict(native_semantic(**kwargs))))
    native_vector = vector_index_preflight.qualify
    monkeypatch.setattr(vector_index_preflight, "qualify", lambda *args, **kwargs:
                        track("indexed", _WeakDict(native_vector(*args, **kwargs))))
    native_view = initial._semantic_view

    def view(*args, **kwargs):
        payload, verified, reader = native_view(*args, **kwargs)
        return (track("semantic_payload", _WeakDict(payload)),
                track("semantic_view", _ViewWitness(verified)), track("semantic_reader", reader))

    monkeypatch.setattr(initial, "_semantic_view", view)
    native_capture = initial.capture_intent_world_snapshot
    monkeypatch.setattr(initial, "capture_intent_world_snapshot", lambda *args, **kwargs:
                        track("world_capture", _WeakDict(native_capture(*args, **kwargs))))
    return refs


@pytest.mark.parametrize("released", ["serialized_snapshot", "snapshot", "semantic_payload",
    "semantic_view", "semantic_reader", "world_capture", "semantic_summary", "retrieval", "indexed"])
def test_native_construction_releases_finished_objects_before_fresh_replay(original, monkeypatch, released):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    refs = _track_native_construction(monkeypatch)
    native_replay = initial._load_initial_context
    observed = []

    def replay(**kwargs):
        # Evaluate before the fresh loader creates a second set of native views.
        assert released in refs
        assert refs[released]() is None, f"{released} remains live at final replay"
        observed.append(released)
        return native_replay(**kwargs)

    monkeypatch.setattr(initial, "_load_initial_context", replay)
    result = prep.initial_context(state=state)
    assert observed == [released]
    assert result["provider_calls"] == 0
    assert result["canonical_tasks_created"] is False
    assert result["full_capsules"] == 1
    assert not (root / "report.jsonl").exists()
    assert not (state / "planner-invoked.json").exists()
    # A later call still reconstructs and validates rather than reusing the
    # released construction objects or suppressing the fresh boundary.
    loaded = native_replay(state=state, prepared=prepared, require_empty_owner=True, result=result)
    assert loaded["receipt"] == result
    assert loaded["semantic"]["semantic_root_cid"] == result["semantic_root_cid"]
    assert loaded["indexed"]["index_id"] == result["index_id"]
    assert loaded["capture"]["snapshot"]["snapshot_cid"] == result["world_snapshot_cid"]


def test_fresh_replay_failure_still_prevents_success_marker(original, monkeypatch):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    native_replay = initial._load_initial_context
    seen = []

    def changed_source(**kwargs):
        seen.append(True)
        (root / "bottle.py").write_text("def changed():\n    return 'changed after capture'\n")
        return native_replay(**kwargs)

    monkeypatch.setattr(initial, "_load_initial_context", changed_source)
    with pytest.raises(ValueError):
        prep.initial_context(state=state)
    assert seen == [True]
    assert not (state / "initial-context-result.json").exists()
    assert not (state / "planner-invoked.json").exists()
