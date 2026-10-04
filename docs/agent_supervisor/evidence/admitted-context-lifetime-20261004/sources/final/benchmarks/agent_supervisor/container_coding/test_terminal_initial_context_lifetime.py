"""Native context producers with weak ownership witnesses; no RSS claims."""
import gc
import weakref
from pathlib import Path

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import vector_index_preflight
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_graph
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot


class _WeakDict(dict):
    pass


class _WeakText(str):
    pass


class _WeakList(list):
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
    "semantic_view", "world_capture", "semantic_summary", "retrieval", "indexed"])
def test_native_construction_releases_finished_objects_before_fresh_replay(original, monkeypatch, released):
    # The reader is also dropped by this caller, but downstream native owner
    # internals or test instrumentation can outlive this thin Path-only local.
    # Its global destruction is deliberately not this caller contract.
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
    assert not (root / "report.jsonl").exists()
    assert not (state / "planner-invoked.json").exists()
    # A later call still reconstructs and validates rather than reusing the
    # released construction objects or suppressing the fresh boundary.
    loaded = native_replay(state=state, prepared=prepared, require_empty_owner=True, result=result)
    assert loaded["receipt"] == result
    assert result["full_capsules"] == loaded["descriptor"]["semantic"]["capsules"]
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


def test_selected_source384_manifest_temporary_released_before_final_replay(original, monkeypatch):
    """Real signed map; authored neural/validation seams isolate its lifetime."""
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source_owner
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    config = state / "authored-lifetime-config.json"
    config.write_text('{"fixture":"lifetime-only; no checkpoint selected"}')
    native_manifest = local._manifest
    refs, calls = [], []

    def manifest(*args, **kwargs):
        payload, profile, current = native_manifest(*args, **kwargs)
        current = _WeakDict(current)
        refs.append(weakref.ref(current))
        return payload, profile, current

    def neural_seam(**kwargs):
        assert kwargs["source_hashes"] == {key: value["sha256"] for key, value in
            prepared["manifest"]["payload"]["sources"].items()}
        calls.append(kwargs["config_path"])
        return {"fixture": "lifetime-only", "execution_authority": False, "completion_authority": False}

    monkeypatch.setattr(local, "_manifest", manifest)
    monkeypatch.setattr(source_owner, "prepare_source384_context", neural_seam)

    def replay(**kwargs):
        assert refs and all(ref() is None for ref in refs)
        assert kwargs["result"]["source384_context"]["execution_authority"] is False
        return {}

    monkeypatch.setattr(initial, "_load_initial_context", replay)
    result = prep.initial_context(state=state, source384_config=config)
    assert calls == [config]
    assert result["provider_calls"] == 0
    assert result["source384_context"]["completion_authority"] is False


@pytest.mark.parametrize("refuse_gate", [False, True])
def test_admitted_construction_releases_native_view_before_fresh_gate(original, monkeypatch, refuse_gate):
    """Native persisted world and semantic payload; only the refusal is authored."""
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    initial_receipt = prep.initial_context(state=state)
    admission = local.admit_local_benchmark_plan(graph=_proposal_graph(prepared), manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)

    refs, capture_refs, contract_refs = [], {}, []
    native_view = initial._semantic_view

    def view(*args, **kwargs):
        payload, verified, reader = native_view(*args, **kwargs)
        payload, verified = _WeakDict(payload), _ViewWitness(verified)
        refs.append({"payload": weakref.ref(payload), "view": weakref.ref(verified)})
        return payload, verified, reader

    monkeypatch.setattr(initial, "_semantic_view", view)
    native_capture = initial.capture_intent_world_snapshot

    def capture(*args, **kwargs):
        payload = _WeakDict(native_capture(*args, **kwargs))
        assert kwargs["task_cids"]
        records = _WeakList(payload["plan_projection"]["tasks"])
        records[0] = _WeakDict(records[0])
        payload["plan_projection"]["tasks"] = records
        capture_refs.update(capture=weakref.ref(payload), records=weakref.ref(records),
                            record=weakref.ref(records[0]))
        return payload

    monkeypatch.setattr(initial, "capture_intent_world_snapshot", capture)
    native_contract = local._contract

    def contract(*args, **kwargs):
        payload, manifest, profile, current = native_contract(*args, **kwargs)
        payload = _WeakDict(payload)
        contract_refs.append(weakref.ref(payload))
        return payload, manifest, profile, current

    monkeypatch.setattr(local, "_contract", contract)
    native_load = initial.load_initial_context
    observed = []

    def fresh_gate(**kwargs):
        # One view came from nomination staging, the second from admitted
        # world construction. Assert this caller's completed construction;
        # returned native/instrumentation frames can survive in collectable
        # cycles. Test-only collection proves the caller no longer owns these
        # objects; it does not promise immediate reclamation or reduced RSS.
        gc.collect()
        assert len(refs) == 2
        assert refs[1]["payload"]() is None
        assert refs[1]["view"]() is None
        assert set(capture_refs) == {"capture", "records", "record"}
        assert all(ref() is None for ref in capture_refs.values())
        assert contract_refs and all(ref() is None for ref in contract_refs)
        assert kwargs["require_empty_owner"] is False
        observed.append(True)
        if refuse_gate:
            raise ValueError("authored fresh admission refusal")
        return native_load(**kwargs)

    monkeypatch.setattr(initial, "load_initial_context", fresh_gate)
    if refuse_gate:
        with pytest.raises(ValueError, match="authored fresh admission refusal"):
            prep.context(state=state)
        assert not (root / ".runtime/terminal-context/result.json").exists()
        assert not (root / ".runtime/terminal-context-bundle.json").exists()
        assert not (state / "context-result.json").exists()
        assert not (state / "doctor-repair-eligibility.json").exists()
    else:
        result = prep.context(state=state)
        assert result["initial_indexes_reused"] is True
        assert result["semantic_root_cid"] == initial_receipt["semantic_root_cid"]
        assert result["world_snapshot_cid"] != initial_receipt["world_snapshot_cid"]
        assert result["new_embedding_calls"] == 0
        assert result["provider_calls"] == 0
        assert (root / ".runtime/terminal-context/result.json").is_file()
        assert (root / ".runtime/terminal-context-bundle.json").is_file()
    assert observed == [True]
    # The private admitted capture precedes the gate; its existence is not a
    # published task context or authority to execute the worker.
    assert (root / ".runtime/terminal-context/world/intent-world.json").is_file()
    assert not (root / "report.jsonl").exists()
    assert not (state / "planner-invoked.json").exists()
