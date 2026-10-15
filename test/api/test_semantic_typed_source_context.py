"""Authored wrapper boundary controls; native producer verification is substituted.

The real bounded codec file reader is retained. Native current-source verification,
bundle loading, and typed graph projection are explicit authored seams, so these
controls establish wrapper ordering/bindings only. The companion native producer
probe must establish actual source verification and retained-bundle integration.
No provider, database, scanner, model, source edit authority, or policy is used.
"""

from __future__ import annotations

from base64 import b32encode
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_source_context as wrapper


TASK = "TASK-AUTHORED-TYPED-SOURCE"
ARTIFACT = ".runtime/semantic/worker-context.json"
GRAPH_ROOT = "decision-root:authored-typed-source"
SOURCE = b"def helper(value):\n    return value * 2\n\ndef target(value):\n    return helper(value) + 2\n"


def _cid(label):
    """A valid RAW CIDv1 for an explicitly authored identity, not a native root."""
    return "b" + b32encode(b"\x01\x55\x12\x20" + sha256(label.encode()).digest()).decode().lower().rstrip("=")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@pytest.fixture
def authored(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    source_path = repository / "operations.py"
    source_path.write_bytes(SOURCE)
    artifact_path = repository / ARTIFACT
    artifact_path.parent.mkdir(parents=True)
    root_cid = _cid("authored-semantic-root")
    target, helper = (_cid(name) for name in ("authored-target", "authored-helper"))
    capsule_rows = [{"stable_symbol_id": sid, "capsule_cid": _cid("capsule-" + sid)}
                    for sid in sorted((target, helper))]
    manifest = {"operations.py": {"sha256": sha256(SOURCE).hexdigest(),
                                   "source_cid": _cid("authored-retained-source")}}
    payload = {
        "schema": "supervisor-semantic-worker-context@1", "task_id": TASK,
        "semantic_root_cid": root_cid, "scope_cid": _cid("authored-source-scope"),
        "manifest": manifest, "capsules": capsule_rows, "admissions": [],
        "required_raw_paths": ["operations.py"],
        "raw_sources": {"operations.py": SOURCE.decode()},
        "completion_authority": False,
    }
    artifact_path.write_bytes(_encoded(payload))
    # The wrapper delegates bundle verification and graph semantics. An opaque
    # authored object ensures these tests cannot accidentally claim native work.
    bundle = SimpleNamespace(root=SimpleNamespace(root_cid=root_cid), authored_fixture=True)
    proposal = {
        "schema": "authored-typed-semantic-dependency-proposal@1",
        "semantic_root_cid": root_cid, "graph_root_id": GRAPH_ROOT,
        "full_symbol_ids": sorted((target, helper)),
        "nominee_symbol_ids": sorted((target, helper)),
        "graph": {"authored_fixture": True, "nodes": [
            {"record": {"native_symbol_id": sid, "source_module_path": "operations.py",
                        "source_cid": manifest["operations.py"]["source_cid"]}}
            for sid in sorted((target, helper))]}, "source_freshness_verified": False,
        "dependency_semantics_complete": False, "source_substitution_authority": False,
        "execution_authority": False, "proof_authority": False,
        "completion_authority": False, "deployment_ready": False,
        "runtime_activation": False,
    }
    state = SimpleNamespace(repository=repository, source_path=source_path,
                            artifact_path=artifact_path, payload=payload,
                            root_cid=root_cid, target=target, helper=helper,
                            bundle=bundle, proposal=proposal,
                            verified_calls=[], load_calls=[], projection_calls=[],
                            after_projection=None)

    def verified(root, *, artifact, text, task_id, current):
        state.verified_calls.append({"root": root, "artifact": artifact,
                                     "task_id": task_id, "current": current})
        assert current is True
        raw = wrapper.codec._read(root, artifact)
        if raw != text.encode():
            raise ValueError("authored verifier: retained parent changed")
        value = json.loads(text)
        if value.get("task_id") != task_id:
            raise ValueError("authored verifier: foreign task")
        for path, binding in value["manifest"].items():
            original = SOURCE if path == "operations.py" else None
            current_bytes = wrapper.codec._read(root, path)
            if original is None or current_bytes != original or sha256(current_bytes).hexdigest() != binding["sha256"]:
                raise ValueError("authored verifier: current source stale")
        return value, tuple(sorted(row["stable_symbol_id"] for row in value.get("capsules", [])))

    def load(repository, artifact, semantic_root_cid):
        state.load_calls.append((repository, artifact, semantic_root_cid))
        return bundle

    def project(**kwargs):
        state.projection_calls.append(kwargs)
        assert kwargs["bundle"] is bundle
        if state.after_projection is not None:
            state.after_projection()
        return deepcopy(state.proposal)

    monkeypatch.setattr(wrapper.codec, "_verified_semantic", verified)
    monkeypatch.setattr(wrapper, "_load_bundle", load)
    monkeypatch.setattr(wrapper, "nominate_typed_semantic_dependencies", project)
    return state


def _project(case, **changes):
    arguments = {
        "repository": case.repository, "artifact": ARTIFACT,
        "expected_sha256": sha256(case.artifact_path.read_bytes()).hexdigest(),
        "task_id": TASK, "expected_semantic_root_cid": case.root_cid,
        "graph_root_id": GRAPH_ROOT, "seed_symbol_ids": (case.target,),
    }
    arguments.update(changes)
    return wrapper.project_source_bound_typed_links(**arguments)


def test_authored_wrapper_verifies_current_source_before_and_after_projection(authored):
    source_before = authored.source_path.read_bytes()
    artifact_before = authored.artifact_path.read_bytes()
    payload_before = deepcopy(authored.payload)
    result = _project(authored)
    assert len(authored.verified_calls) == 2
    assert all(call == {"root": authored.repository, "artifact": ARTIFACT,
                        "task_id": TASK, "current": True} for call in authored.verified_calls)
    assert len(authored.projection_calls) == len(authored.load_calls) == 1
    assert authored.load_calls == [(authored.repository, ARTIFACT, authored.root_cid)]
    call = authored.projection_calls[0]
    assert call["graph_root_id"] == GRAPH_ROOT and call["seed_symbol_ids"] == (authored.target,)
    assert result["source_freshness_verified"] is True
    assert result["source_check_scope"] == "opening_and_closing_current_native_verification"
    assert result["owner_task_policy_verified"] is False
    assert result["source_binding"] == {
        "task_id": TASK, "artifact": ARTIFACT,
        "artifact_sha256": sha256(artifact_before).hexdigest(),
        "semantic_root_cid": authored.root_cid, "scope_cid": authored.payload["scope_cid"],
        "source_manifest": authored.payload["manifest"],
        "required_raw_paths": ["operations.py"], "raw_source_paths": ["operations.py"],
    }
    for field in ("source_substitution_authority", "execution_authority", "proof_authority",
                  "completion_authority", "deployment_ready", "runtime_activation"):
        assert result[field] is False
    assert authored.source_path.read_bytes() == source_before
    assert authored.artifact_path.read_bytes() == artifact_before
    assert authored.payload == payload_before
    assert SOURCE.decode() not in json.dumps(result)


@pytest.mark.parametrize("change", ["hash", "semantic_root", "task"])
def test_foreign_expected_bindings_refuse_before_typed_projection(authored, change):
    arguments = {
        "hash": {"expected_sha256": "0" * 64},
        "semantic_root": {"expected_semantic_root_cid": _cid("foreign-root")},
        "task": {"task_id": "TASK-FOREIGN"},
    }[change]
    with pytest.raises(ValueError):
        _project(authored, **arguments)
    assert authored.projection_calls == []
    assert authored.source_path.read_bytes() == SOURCE


def test_preexisting_current_source_change_refuses_before_projection(authored):
    authored.source_path.write_bytes(SOURCE + b"\n# externally changed source\n")
    with pytest.raises(ValueError):
        _project(authored)
    assert authored.projection_calls == []
    assert len(authored.verified_calls) == 1


@pytest.mark.parametrize("required", ["operations.py", {"operations.py": True},
    ["operations.py", "operations.py"], ["foreign.py"], [True]])
def test_malformed_or_foreign_required_raw_paths_refuse_before_projection(authored, required):
    payload = deepcopy(authored.payload)
    payload["required_raw_paths"] = required
    authored.artifact_path.write_bytes(_encoded(payload))
    with pytest.raises(ValueError, match="required raw paths"):
        _project(authored)
    assert authored.projection_calls == []


def test_required_path_without_literal_source_delivery_refuses(authored):
    payload = deepcopy(authored.payload)
    payload["raw_sources"].clear()
    authored.artifact_path.write_bytes(_encoded(payload))
    with pytest.raises(ValueError, match="literal delivery"):
        _project(authored)
    assert authored.projection_calls == []


@pytest.mark.parametrize("damage", ["foreign_path", "foreign_cid", "missing_source_cid"])
def test_graph_symbol_sources_must_match_current_verified_frontier(authored, damage):
    record = authored.proposal["graph"]["nodes"][0]["record"]
    if damage == "foreign_path":
        record["source_module_path"] = "foreign.py"
    else:
        record["source_cid"] = None if damage == "missing_source_cid" else _cid("foreign-source")
    with pytest.raises(ValueError, match="current source frontier"):
        _project(authored)


@pytest.mark.parametrize("changed", ["source", "artifact"])
def test_closing_fence_refuses_changes_during_typed_projection(authored, changed):
    source_before = authored.source_path.read_bytes()
    artifact_before = authored.artifact_path.read_bytes()
    expected = sha256(artifact_before).hexdigest()
    if changed == "source":
        authored.after_projection = lambda: authored.source_path.write_bytes(source_before + b"\n# changed during projection\n")
    else:
        authored.after_projection = lambda: authored.artifact_path.write_bytes(artifact_before + b"\n")
    try:
        with pytest.raises(ValueError):
            _project(authored, expected_sha256=expected)
        assert len(authored.projection_calls) == 1
        # Refusal is observational; the wrapper must not revert another writer.
        assert (authored.source_path.read_bytes() != source_before if changed == "source"
                else authored.artifact_path.read_bytes() != artifact_before)
    finally:
        authored.source_path.write_bytes(source_before)
        authored.artifact_path.write_bytes(artifact_before)


@pytest.mark.parametrize("shape", ["worker_projection", "role_view", "decision_packet"])
def test_shortened_or_projected_parent_is_not_a_full_source_inventory(authored, shape):
    payload = deepcopy(authored.payload)
    if shape == "worker_projection":
        payload["worker_projection"] = {"schema": "supervisor-semantic-worker-projection@1"}
    elif shape == "role_view":
        payload["schema"] = "supervisor-semantic-worker-role-view@1"
    else:
        payload["schema"] = "supervisor-semantic-decision-packet@1"
    authored.artifact_path.write_bytes(_encoded(payload))
    with pytest.raises(ValueError):
        _project(authored)
    assert authored.projection_calls == []


@pytest.mark.parametrize("population", ["capsules_missing", "capsules_duplicate", "proposal_foreign"])
def test_complete_native_capsule_population_is_not_narrowed_or_aliased(authored, population):
    if population == "proposal_foreign":
        authored.proposal["full_symbol_ids"] = sorted((authored.target, _cid("foreign-symbol")))
    else:
        payload = deepcopy(authored.payload)
        payload["capsules"] = ([payload["capsules"][0]] if population == "capsules_missing"
                               else [payload["capsules"][0], payload["capsules"][0]])
        authored.artifact_path.write_bytes(_encoded(payload))
    with pytest.raises(ValueError):
        _project(authored)


@pytest.mark.parametrize("artifact", ["../outside.json", "/tmp/foreign.json", ".runtime/semantic/../worker-context.json"])
def test_real_codec_reader_refuses_noncanonical_or_escaping_artifact_path(authored, artifact):
    with pytest.raises(ValueError):
        _project(authored, artifact=artifact)
    assert authored.projection_calls == []


def test_real_codec_reader_refuses_artifact_symlink(authored):
    original = authored.artifact_path.read_bytes()
    target = authored.artifact_path.with_name("foreign-parent.json")
    target.write_bytes(original)
    authored.artifact_path.unlink()
    authored.artifact_path.symlink_to(target.name)
    try:
        with pytest.raises((ValueError, OSError)):
            _project(authored, expected_sha256=sha256(original).hexdigest())
        assert authored.projection_calls == []
    finally:
        authored.artifact_path.unlink()
        authored.artifact_path.write_bytes(original)


def _scoped_source_seam(authored):
    """Complete authored identity metadata independent of partial graph nodes."""
    proposal = authored.proposal
    records = deepcopy([node["record"] for node in proposal["graph"]["nodes"]])
    capsules = {row["stable_symbol_id"]:row["capsule_cid"] for row in authored.payload["capsules"]}
    for record in records:
        record["capsule_cid"] = capsules[record["native_symbol_id"]]
    proposal.update(schema="supervisor-semantic-typed-dependency-proposal@2",
                    adapter_version="native-typed-dependency-proposal@2", graph_scope="task_closure",
                    root_symbol_records=records, root_symbol_records_count=len(records),
                    root_symbol_records_sha256=sha256(_encoded(records)).hexdigest(),
                    nominee_symbol_ids=[authored.target], materialized_symbol_ids=[authored.target],
                    omitted_root_symbol_ids=[authored.helper])
    proposal["graph"]["nodes"] = [{"record":deepcopy(record)} for record in records
                                  if record["native_symbol_id"] == authored.target]


def test_scoped_source_joins_full_parent_frontier_without_full_graph(authored):
    _scoped_source_seam(authored)
    result = _project(authored, graph_scope="task_closure")
    assert result["schema"] == wrapper.SCOPED_SCHEMA
    assert result["root_symbol_records_count"] == 2 and len(result["graph"]["nodes"]) == 1
    assert authored.projection_calls[0]["graph_scope"] == "task_closure"
    assert len(authored.verified_calls) == 2


@pytest.mark.parametrize("damage", ["old_schema", "foreign_root_record", "root_hash", "graph_record"])
def test_scoped_source_refuses_foreign_or_incomplete_full_metadata(authored, damage):
    _scoped_source_seam(authored)
    proposal = authored.proposal
    if damage == "old_schema":
        proposal["schema"] = "supervisor-semantic-typed-dependency-proposal@1"
    elif damage == "foreign_root_record":
        for record in proposal["root_symbol_records"]:
            if record["native_symbol_id"] == authored.helper:
                record["source_module_path"] = "foreign.py"
        proposal["root_symbol_records_sha256"] = sha256(_encoded(proposal["root_symbol_records"])).hexdigest()
    elif damage == "root_hash":
        proposal["root_symbol_records_sha256"] = "0" * 64
    else:
        proposal["graph"]["nodes"][0]["record"]["capsule_cid"] = _cid("foreign-graph-capsule")
    with pytest.raises(ValueError):
        _project(authored, graph_scope="task_closure")


@pytest.mark.parametrize("scope", [None, True, "", "Task_closure", "full"])
def test_invalid_graph_scope_refuses_before_source_work(authored, scope):
    with pytest.raises(ValueError, match="graph scope"):
        _project(authored, graph_scope=scope)
    assert authored.verified_calls == authored.projection_calls == []



def _authored_cache_seam(authored, monkeypatch):
    """Exercise real HMAC custody around an explicitly authored native seam."""
    authored.repository.chmod(0o755)
    (authored.repository / ".runtime").chmod(0o755)
    blocks = {authored.root_cid: _encoded({"capsule_index_cid": "authored-capsule-index"}),
              "authored-omitted-block": b"omitted authored bytes"}
    monkeypatch.setattr(wrapper, "_capture_blocks", lambda *args: deepcopy(blocks))
    monkeypatch.setattr(wrapper, "_typed_validator_code", lambda: {"schema": "authored-code-binding@1",
                                                                 "sha256": "a" * 64})
    return blocks


def test_authenticated_proposal_hit_keeps_fresh_source_fences(authored, monkeypatch):
    _authored_cache_seam(authored, monkeypatch)
    first = _project(authored)
    assert len(authored.projection_calls) == 1
    def forbid(**kwargs):
        raise AssertionError("a genuine cache hit must avoid the authored projection seam")
    monkeypatch.setattr(wrapper, "nominate_typed_semantic_dependencies", forbid)
    second = _project(authored)
    assert first == second
    assert len(authored.verified_calls) == 4
    assert len(authored.load_calls) == 1
    assert all(call["current"] is True for call in authored.verified_calls)
    assert second["proof_authority"] is second["completion_authority"] is False


def test_proposal_cache_opt_out_uses_full_projection_without_cache_io(authored, monkeypatch):
    _authored_cache_seam(authored, monkeypatch)
    first = _project(authored)
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE", "0")
    def forbid(*args):
        raise AssertionError("opt-out must not inspect the cache or its fingerprint")
    monkeypatch.setattr(wrapper, "_capture_blocks", forbid)
    monkeypatch.setattr(wrapper, "_load_cached_typed_proposal", forbid)
    monkeypatch.setattr(wrapper, "_publish_cached_typed_proposal", forbid)
    assert _project(authored) == first
    assert len(authored.projection_calls) == len(authored.load_calls) == 2


@pytest.mark.parametrize("changed", ["omitted_block", "validator", "seed", "graph_root", "node_limit"])
def test_source_cache_changes_require_full_projection(authored, monkeypatch, changed):
    blocks = _authored_cache_seam(authored, monkeypatch)
    _project(authored)
    changes = {}
    if changed == "omitted_block":
        blocks["authored-omitted-block"] += b" changed"
    elif changed == "validator":
        monkeypatch.setattr(wrapper, "_typed_validator_code", lambda: {"schema": "authored-code-binding@1",
                                                                     "sha256": "b" * 64})
    elif changed == "seed":
        changes["seed_symbol_ids"] = (authored.helper,)
    elif changed == "graph_root":
        changes["graph_root_id"] = GRAPH_ROOT + ":another"
    else:
        changes["max_graph_nodes"] = 1024
    _project(authored, **changes)
    assert len(authored.projection_calls) == len(authored.load_calls) == 2
    assert len(authored.verified_calls) == 4


def test_valid_cache_cannot_skip_current_source_drift(authored, monkeypatch):
    _authored_cache_seam(authored, monkeypatch)
    _project(authored)
    authored.source_path.write_bytes(SOURCE + b"\n# changed after cache publication\n")
    with pytest.raises(ValueError, match="current source stale"):
        _project(authored)
    assert len(authored.projection_calls) == 1


@pytest.mark.parametrize("changed", ["omitted_block", "validator", "cache_record", "owner_key"])
def test_cache_hit_rechecks_closing_population_code_and_authentication(authored, monkeypatch, changed):
    blocks = _authored_cache_seam(authored, monkeypatch)
    _project(authored)
    original = wrapper.codec._verified_semantic
    calls = 0
    def closing_change(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            if changed == "omitted_block":
                blocks["authored-omitted-block"] += b" changed during hit"
            elif changed == "validator":
                monkeypatch.setattr(wrapper, "_typed_validator_code", lambda: {"schema": "authored-code-binding@1",
                                                                             "sha256": "c" * 64})
            else:
                directory = authored.repository / ".runtime" / "semantic-typed-verification-cache"
                path = (directory / ".owner-key" if changed == "owner_key"
                        else next(directory.glob("*.json")))
                raw = path.read_bytes()
                path.chmod(0o600)
                path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
                path.chmod(0o400)
        return value
    monkeypatch.setattr(wrapper.codec, "_verified_semantic", closing_change)
    with pytest.raises(ValueError, match="population or validators changed|cache changed"):
        _project(authored)
    assert calls == 2 and len(authored.projection_calls) == 1


def test_missing_cache_eligibility_still_runs_native_projection(authored, monkeypatch):
    _authored_cache_seam(authored, monkeypatch)
    def unavailable(*args):
        raise ValueError("unavailable conservative code/source capture")
    monkeypatch.setattr(wrapper, "_typed_validator_code", unavailable)
    result = _project(authored)
    assert len(authored.projection_calls) == len(authored.load_calls) == 1
    assert result["source_freshness_verified"] is True
    assert not (authored.repository / ".runtime" / "semantic-typed-verification-cache").exists()



@pytest.mark.parametrize("error_type", [ValueError, ImportError])
def test_unavailable_code_capture_uses_full_native_fallback(authored, monkeypatch, error_type):
    _authored_cache_seam(authored, monkeypatch)
    def unavailable():
        raise error_type("optional cache validator capture unavailable")
    monkeypatch.setattr(wrapper, "_typed_validator_code", unavailable)
    result = _project(authored)
    assert result["source_freshness_verified"] is True
    assert len(authored.projection_calls) == len(authored.load_calls) == 1
    assert len(authored.verified_calls) == 2


@pytest.mark.parametrize("cache_hit", [False, True])
def test_closing_capture_unavailability_requires_full_native_evidence(authored, monkeypatch, cache_hit):
    _authored_cache_seam(authored, monkeypatch)
    if cache_hit:
        _project(authored)
    calls = 0
    def closing_unavailable():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ImportError("closing optional dependency unavailable")
        return {"schema": "authored-code-binding@1", "sha256": "a" * 64}
    monkeypatch.setattr(wrapper, "_typed_validator_code", closing_unavailable)
    if cache_hit:
        with pytest.raises(ValueError, match="closing capture unavailable"):
            _project(authored)
    else:
        assert _project(authored)["source_freshness_verified"] is True
        assert not (authored.repository / ".runtime" / "semantic-typed-verification-cache").exists()
    assert len(authored.projection_calls) == 1
    assert calls == 2


def test_code_custody_refuses_sources_mutable_by_supervisor(tmp_path):
    from os import geteuid
    path = tmp_path / "owned-validator.py"
    path.write_text("VALUE = 1\n")
    if geteuid() != 0:
        with pytest.raises(ValueError, match="single-owner validator input"):
            wrapper._typed_code_protected(path.stat(), 0, directory=False)
    # Mode guard is independent of UID, including test runs invoked as root.
    path.chmod(0o666)
    with pytest.raises(ValueError, match="single-owner validator input"):
        wrapper._typed_code_protected(path.stat(), geteuid(), directory=False)



@pytest.mark.parametrize("status", [
    "CapEff: 1\nCapPrm: 0\nCapAmb: 0\n",
    "CapEff: 0\nCapPrm: 1\nCapAmb: 0\n",
    "CapEff: 0\nCapPrm: 0\nCapAmb: 1\n",
    "CapEff: 0\nCapPrm: 0\n",
    "CapEff: 0\nCapEff: 0\nCapPrm: 0\nCapAmb: 0\n",
])
def test_capability_custody_cannot_authorize_immutable_code_cache(monkeypatch, status):
    from io import StringIO
    monkeypatch.setattr(wrapper, "open", lambda *args, **kwargs: StringIO(status), raising=False)
    with pytest.raises(ValueError):
        wrapper._typed_validator_code()


def test_bytecode_write_preference_does_not_change_immutable_code_binding(tmp_path, monkeypatch):
    """Authored hashing seam; actual protected no-B entry is qualified separately."""
    import sys
    from types import SimpleNamespace
    packages = {name: tmp_path / name for name in wrapper._typed_code_native_packages}
    dependencies = {name: tmp_path / name for name in wrapper._typed_code_minimal_cid_dependency_trees}
    files = {"typing_extensions": tmp_path / "typing_extensions.py"}
    interpreter = __import__("pathlib").Path(sys.executable).resolve()
    def tree(path, uid, budget, *, reject_bytecode):
        budget["files"] += 1
        budget["bytes"] += 10
        return {"path": str(path), "sha256": "a" * 64}, {"__init__.py"}
    def file(path, uid, budget):
        budget["files"] += 1
        budget["bytes"] += 10
        return {"path": str(path), "sha256": "b" * 64}
    monkeypatch.setattr(wrapper, "_typed_code_tree", tree)
    monkeypatch.setattr(wrapper, "_typed_code_file", file)
    monkeypatch.setattr(wrapper, "_typed_code_origins", lambda *args: 0)
    actual = sys
    class SourceOnlySys:
        pycache_prefix = None
        dont_write_bytecode = False
        def __getattr__(self, key):
            return getattr(actual, key)
    proxy = SourceOnlySys()
    monkeypatch.setattr(wrapper, "sys", proxy)
    kwargs = dict(package_roots=packages, dependency_roots=dependencies, dependency_files=files,
                  interpreter_path=interpreter, trusted_uid=0, profile={"schema": "authored-profile@1"})
    first = wrapper._typed_code_fingerprint_native_validator_code(**kwargs)["binding"]
    proxy.dont_write_bytecode = True
    assert wrapper._typed_code_fingerprint_native_validator_code(**kwargs)["binding"] == first
    proxy.pycache_prefix = "/unadmitted-bytecode"
    with pytest.raises(ValueError, match="alternate pycache"):
        wrapper._typed_code_fingerprint_native_validator_code(**kwargs)



@pytest.mark.parametrize("cache_hit", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_closing_retained_block_capture_failure_always_refuses(authored, monkeypatch, cache_hit, error_type):
    blocks = _authored_cache_seam(authored, monkeypatch)
    if cache_hit:
        _project(authored)
    calls = 0
    def failed_closing_capture(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("closing complete retained population changed or missing")
        return deepcopy(blocks)
    monkeypatch.setattr(wrapper, "_capture_blocks", failed_closing_capture)
    with pytest.raises(error_type, match="retained population changed or missing"):
        _project(authored)
    assert calls == 2
    assert len(authored.projection_calls) == 1
    if not cache_hit:
        assert not (authored.repository / ".runtime" / "semantic-typed-verification-cache").exists()


# Append to test_semantic_typed_source_context.py; wrapper/pytest imports already exist.
# These are authored origin-boundary controls, not native cache qualification.

@pytest.fixture
def authored_namespace_origin(tmp_path, monkeypatch):
    import importlib.machinery as machinery
    import importlib.util as util
    import sys
    from types import SimpleNamespace

    prefix = "authored_native"
    name = prefix + ".synthetic_namespace"
    assert prefix not in sys.modules
    root = (tmp_path / prefix).resolve()
    root.mkdir()
    init = root / "__init__.py"
    init.write_text("# Synthetic origin fixture; never executed.\n")
    namespace_dir = root / "synthetic_namespace"
    namespace_dir.mkdir()
    regular_file = root / "helper.py"
    regular_file.write_text("VALUE = 1\n")
    parent = util.module_from_spec(util.spec_from_file_location(prefix, init))
    monkeypatch.setitem(sys.modules, prefix, parent)
    spec = machinery.PathFinder.find_spec(name, parent.__path__)
    assert spec is not None and spec.origin is None
    module = util.module_from_spec(spec)
    locations = module.__path__
    assert type(module.__loader__) is machinery.NamespaceLoader
    assert locations is spec.submodule_search_locations is module.__loader__._path
    finder = locations._path_finder
    expected_finder = machinery.PathFinder._get_spec
    assert finder.__self__ is expected_finder.__self__
    assert finder.__func__ is expected_finder.__func__
    assert tuple(locations) == (str(namespace_dir),)

    class ObservedSys:
        modules = {name: module}
        def __getattr__(self, key):
            return getattr(sys, key)

    observed = ObservedSys()
    monkeypatch.setattr(wrapper, "sys", observed)
    monkeypatch.setattr(wrapper, "_typed_code_native_packages", frozenset({prefix}))
    members = {"/", "synthetic_namespace/", "__init__.py", "helper.py"}
    return SimpleNamespace(prefix=prefix, name=name, root=root,
                           namespace_dir=namespace_dir, parent=parent, module=module,
                           locations=locations, observed=observed, members=members,
                           trees={prefix: (root, members)})


def test_exact_stdlib_single_root_namespace_origin_is_accepted(authored_namespace_origin):
    state = authored_namespace_origin
    assert wrapper._typed_code_origins(state.trees, {}) == 1


def test_namespace_and_regular_module_subsets_are_validated_independently(authored_namespace_origin):
    import importlib.util as util
    state = authored_namespace_origin
    regular_name = state.prefix + ".helper"
    regular = util.module_from_spec(util.spec_from_file_location(
        regular_name, state.root / "helper.py"))
    state.observed.modules[regular_name] = regular
    assert wrapper._typed_code_origins(state.trees, {}) == 2
    del state.observed.modules[state.name]
    assert wrapper._typed_code_origins(state.trees, {}) == 1


@pytest.mark.parametrize("error_type", [AttributeError, TypeError])
def test_unsupported_namespace_inspection_refuses_optional_cache(
        authored_namespace_origin, monkeypatch, error_type):
    state = authored_namespace_origin
    def unsupported(*args):
        raise error_type("authored unsupported private import machinery")
    monkeypatch.setattr(wrapper, "_typed_code_namespace", unsupported)
    with pytest.raises(ValueError, match="namespace inspection is unsupported"):
        wrapper._typed_code_origins(state.trees, {})


@pytest.mark.parametrize("mutation", [
    "alternative_contained_root", "extra_foreign_root", "duplicate_root", "zero_roots",
    "relative_root", "symlink_root", "absent_directory", "init_source", "init_bytecode",
    "init_extension", "custom_loader", "mismatched_loader", "spec_name", "module_name",
    "module_package", "spec_origin", "module_file", "has_location", "cached_origin",
    "detached_spec_path", "custom_path_iterator", "path_subclass", "custom_path_finder",
    "path_name",
    "spoofed_path_finder", "missing_loader_path", "missing_path_name",
    "missing_path_finder", "missing_stored_paths",
])
def test_namespace_origin_refuses_unadmitted_provenance(authored_namespace_origin, mutation):
    import importlib.machinery as machinery
    state = authored_namespace_origin
    module, locations, spec = state.module, state.locations, state.module.__spec__
    if mutation == "alternative_contained_root":
        alternative = state.root / "alternative_namespace"
        alternative.mkdir()
        state.members.add("alternative_namespace/")
        locations._path = [str(alternative)]
    elif mutation == "extra_foreign_root":
        foreign = state.root.parent / "foreign_namespace"
        foreign.mkdir()
        locations._path = [str(state.namespace_dir), str(foreign)]
    elif mutation == "duplicate_root":
        locations._path = [str(state.namespace_dir), str(state.namespace_dir)]
    elif mutation == "zero_roots":
        locations._path = []
    elif mutation == "relative_root":
        locations._path = ["synthetic_namespace"]
    elif mutation == "symlink_root":
        target = state.root / "real_target"
        state.namespace_dir.rename(target)
        state.namespace_dir.symlink_to(target, target_is_directory=True)
    elif mutation == "absent_directory":
        state.members.remove("synthetic_namespace/")
    elif mutation in {"init_source", "init_bytecode", "init_extension"}:
        suffix = {"init_source": ".py", "init_bytecode": ".pyc",
                  "init_extension": machinery.EXTENSION_SUFFIXES[0]}[mutation]
        filename = "__init__" + suffix
        (state.namespace_dir / filename).write_bytes(b"# Never imported\n")
        state.members.add("synthetic_namespace/" + filename)
    elif mutation == "custom_loader":
        spec.loader = module.__loader__ = object()
    elif mutation == "mismatched_loader":
        module.__loader__ = object()
    elif mutation == "spec_name":
        spec.name = state.prefix + ".other"
    elif mutation == "module_name":
        module.__name__ = state.prefix + ".other"
    elif mutation == "module_package":
        module.__package__ = state.prefix
    elif mutation == "spec_origin":
        spec.origin = "namespace"
    elif mutation == "module_file":
        module.__file__ = str(state.namespace_dir)
    elif mutation == "has_location":
        spec.has_location = True
    elif mutation == "cached_origin":
        spec._cached = str(state.namespace_dir / "injected.pyc")
    elif mutation == "detached_spec_path":
        spec.submodule_search_locations = type(locations)(
            state.name, [str(state.namespace_dir)], machinery.PathFinder._get_spec)
    elif mutation == "custom_path_iterator":
        class UntrustedPath:
            def __iter__(self):
                raise AssertionError("An untrusted namespace iterator was executed")
        path = UntrustedPath()
        module.__path__ = spec.submodule_search_locations = module.__loader__._path = path
    elif mutation == "path_subclass":
        class UntrustedPathSubclass(type(locations)):
            def __iter__(self):
                raise AssertionError("An untrusted namespace subclass was executed")
        path = UntrustedPathSubclass(state.name, [str(state.namespace_dir)],
                                     machinery.PathFinder._get_spec)
        module.__path__ = spec.submodule_search_locations = module.__loader__._path = path
    elif mutation == "custom_path_finder":
        def finder(*args):
            raise AssertionError("An untrusted namespace search callback was executed")
        locations._path_finder = finder
        # Force recalculation if provenance checks accidentally iterate first.
        locations._last_parent_path = ()
    elif mutation == "path_name":
        locations._name = state.prefix + ".other"
    elif mutation == "spoofed_path_finder":
        expected = machinery.PathFinder._get_spec
        class SpoofedFinder:
            __self__ = expected.__self__
            __func__ = expected.__func__
            def __call__(self, *args):
                raise AssertionError("A spoofed namespace callback was executed")
        locations._path_finder = SpoofedFinder()
        locations._last_parent_path = ()
    elif mutation == "missing_loader_path":
        del module.__loader__._path
    elif mutation == "missing_path_name":
        del locations._name
    elif mutation == "missing_path_finder":
        del locations._path_finder
    elif mutation == "missing_stored_paths":
        del locations._path
    else:
        raise AssertionError("Unknown mutation")
    with pytest.raises(ValueError):
        wrapper._typed_code_origins(state.trees, {})
