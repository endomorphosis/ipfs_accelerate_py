"""Signed terminal scope, complete inventories, and honest checkpoint attribution."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import original, prepare, git
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_context import selected_config
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as owner
from ipfs_accelerate_py.agent_supervisor.runtime import source384_program_scope as scope_owner
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles


@pytest.fixture
def signed_case(tmp_path):
    args = original(tmp_path)
    args[3]["outputs"] = [dict(path="source.py", effect="modify", media_type="text/x-python")]
    prepared = prepare(args)
    root, _, state, _ = args
    hashes = {name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()}
    return SimpleNamespace(root=root, prepared=prepared, hashes=hashes, output=state / "scoped-source384")


def _scope(c):
    return scope_owner.program_scope(repository=c.root, source_hashes=c.hashes,
        envelope=c.prepared["manifest"], current=True)


def test_signed_scope_accounts_for_every_source_and_only_excludes_fixed_support(signed_case):
    c = signed_case
    scope = _scope(c)
    assert scope["schema"] == scope_owner.SCHEMA
    assert scope["program_paths"] == ["source.py"]
    assert {row["path"] for row in scope["harness_support"]} == {profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE}
    assert set(scope["program_paths"]) | {row["path"] for row in scope["harness_support"]} == set(c.hashes)
    inventory = [dict(path=name, disposition="captured", sha256=sha) for name, sha in c.hashes.items()]
    assert owner._program_python_paths(inventory, scope) == ["source.py"]
    assert profiles.SMOKE in owner._program_python_paths(inventory)
    inference = dict(report=dict(preparation=dict(paths=["source.py"], max_functions=1024, max_selected_units=128)))
    owner._validate_population(inference, inventory, scope)
    inference["report"]["preparation"]["paths"].insert(0, profiles.SMOKE)
    with pytest.raises(ValueError, match="complete declared Python"):
        owner._validate_population(inference, inventory, scope)


@pytest.mark.parametrize("name", [profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE, "source.py"])
def test_initial_scope_rejects_changed_signed_bytes(signed_case, name):
    c = signed_case
    (c.root / name).write_bytes((c.root / name).read_bytes() + b"\n")
    with pytest.raises(ValueError):
        _scope(c)


@pytest.mark.parametrize("damage", ["signature", "task", "evidence", "extra_source", "missing_source"])
def test_scope_refuses_altered_manifest_and_population(signed_case, damage):
    c = signed_case
    manifest = deepcopy(c.prepared["manifest"])
    if damage == "signature":
        manifest["binding"]["signature"] = "invalid"
    elif damage == "task":
        manifest["payload"]["tasks"][0]["validations"][0]["argv"] = ["true"]
    elif damage == "evidence":
        manifest["payload"]["planning_inputs"]["selected_evidence"] = []
    elif damage == "extra_source":
        c.hashes["ignored.py"] = "f" * 64
    else:
        c.hashes.pop("source.py")
    with pytest.raises((ValueError, KeyError)):
        scope_owner.program_scope(repository=c.root, source_hashes=c.hashes,
            envelope=manifest, current=True)


@pytest.mark.parametrize("empty", [False, True])
def test_absent_program_abstains_before_output_or_checkpoint(tmp_path, monkeypatch, empty):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as models
    args = original(tmp_path, empty=empty)
    if not empty:
        root = args[0]
        git(root, "mv", "source.py", "source.txt")
        git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "data input")
        args[3]["input_paths"] = ["source.txt"]
    prepared = prepare(args)
    root, _, state, _ = args
    output = state / "no-inference"
    monkeypatch.setattr(models, "register_shared_parent", lambda *a, **k: pytest.fail("checkpoint registered"))
    with pytest.raises(ValueError, match="abstained: no declared Python program inputs; checkpoint not consumed"):
        owner.prepare_source384_context(repository=root,
            source_hashes={name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()},
            output=output, config_path=state / "unused.json", manifest_envelope=prepared["manifest"])
    assert not output.exists()


def test_profile_cannot_silently_fall_back_to_inferencing_support(signed_case):
    c = signed_case
    with pytest.raises(ValueError, match="requires verified program scope"):
        owner.prepare_source384_context(repository=c.root, source_hashes=c.hashes,
            output=c.output, config_path=c.output / "unused.json")
    assert not c.output.exists()


@pytest.fixture
def numerical_case(signed_case, tmp_path, monkeypatch):
    """Authored numerical seam; native manifest/signatures/current source stay real."""
    from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as models
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
    from ipfs_datasets_py.logic.software_contracts import codebase_resources as resources
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    c = signed_case
    cid = cid_for_structured({"authored": True})
    head = CodebaseHead("authored", 1, cid, cid, f"rev:authored:snapshot:{cid}", cid)
    config = dict(schema="terminal-source384-config@1", checkpoint_path="authored",
        checkpoint_sha256="c" * 64, embedding_snapshot="authored")
    c.config = tmp_path / "config.json"
    c.config.write_bytes(owner._raw(config))
    c.calls = []
    @contextmanager
    def lease(**kwargs):
        yield object()
    @contextmanager
    def owners(output):
        yield SimpleNamespace(prepare_current=lambda *a, **k: SimpleNamespace(head=head)), object()
    def inference(index, root, **kwargs):
        c.calls.append(kwargs)
        return dict(native_worker_executed=True, inference_executed=True,
            report=dict(key=dict(source_head=head.to_dict(), version_id="authored",
                original_checkpoint_sha256="c" * 64), preparation=dict(paths=kwargs["paths"],
                max_functions=1024, max_selected_units=128), output=dict(model_loads=1)))
    monkeypatch.setattr(owner, "load_source384_config", lambda *a: deepcopy(config))
    monkeypatch.setattr(owner, "_owners", owners)
    monkeypatch.setattr(owner, "_inventory", lambda index, head, hashes:
        [dict(path=name, sha256=sha, disposition="captured") for name, sha in sorted(hashes.items())])
    monkeypatch.setattr(owner, "_summary", lambda inference, **kwargs: dict(authored_summary=kwargs))
    monkeypatch.setattr(resources, "acquire_codebase_resources", lease)
    monkeypatch.setattr(models, "register_shared_parent", lambda *a, **k: "authored")
    monkeypatch.setattr(units, "infer_shared_parent_units", inference)
    monkeypatch.setattr(units, "validate_shared_parent_units", lambda *a, **k: None)
    c.prepare = lambda: owner.prepare_source384_context(repository=c.root, source_hashes=c.hashes,
        output=c.output, config_path=c.config, manifest_envelope=c.prepared["manifest"])
    return c


def test_scoped_inference_replay_successor_keep_full_ledger_without_support_inference(numerical_case):
    c = numerical_case
    initial = c.prepare()
    assert initial["source_hashes"] == c.hashes
    assert {row["path"] for row in initial["source_inventory"]} == set(c.hashes)
    assert c.calls[0]["paths"] == ["source.py"]
    assert owner.validate_source384_context(repository=c.root, expected_receipt=initial) == initial
    assert len(c.calls) == 1
    (c.root / "source.py").write_text("def public_source():\n    return 2\n")
    git(c.root, "add", "source.py")
    git(c.root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "candidate")
    hashes = {name: owner._sha((c.root / name).read_bytes()) for name in c.hashes}
    assert owner.validate_historical_source384_selection(repository=c.root, expected_receipt=initial) == initial
    with pytest.raises(ValueError, match="differs from signed population"):
        owner.validate_source384_context(repository=c.root, expected_receipt=initial)
    output = c.output.with_name("successor")
    successor = owner.prepare_source384_successor_context(repository=c.root, source_hashes=hashes,
        output=output, predecessor_receipt=initial)
    assert len(c.calls) == 2 and c.calls[-1]["paths"] == ["source.py"]
    assert successor["source_hashes"] == hashes
    assert successor["program_scope"]["selection_sha256"] == initial["program_scope"]["selection_sha256"]
    assert successor["program_scope"]["source_population_sha256"] != initial["program_scope"]["source_population_sha256"]
    assert owner.validate_source384_context(repository=c.root, expected_receipt=successor) == successor
    assert len(c.calls) == 2


@pytest.mark.parametrize("damage", ["selection_bytes", "signature_rebound", "scope_program", "scope_support",
    "scope_population", "scope_authority", "scope_extra", "scope_removed", "support_bytes"])
def test_scope_replay_refuses_tampering_without_neural_replay(numerical_case, damage):
    c = numerical_case
    receipt = c.prepare()
    if damage == "selection_bytes":
        (c.output / scope_owner.ARTIFACT).write_text("{}")
    elif damage == "signature_rebound":
        manifest = deepcopy(c.prepared["manifest"])
        manifest["binding"]["signature"] = "invalid"
        raw = owner._raw(manifest)
        (c.output / scope_owner.ARTIFACT).write_bytes(raw)
        receipt["program_scope"]["selection_sha256"] = owner._sha(raw)
    elif damage == "scope_program":
        receipt["program_scope"]["program_paths"] = [profiles.SMOKE]
    elif damage == "scope_support":
        receipt["program_scope"]["harness_support"][0]["sha256"] = "0" * 64
    elif damage == "scope_population":
        receipt["program_scope"]["source_population_sha256"] = "0" * 64
    elif damage == "scope_authority":
        receipt["program_scope"]["proof_authority"] = True
    elif damage == "scope_extra":
        receipt["program_scope"]["unknown"] = False
    elif damage == "scope_removed":
        receipt.pop("program_scope")
    else:
        (c.root / profiles.SMOKE).write_text("pass\n")
    (c.output / "receipt.json").write_bytes(owner._raw(receipt))
    with pytest.raises((ValueError, KeyError)):
        owner.validate_source384_context(repository=c.root, expected_receipt=receipt)
    assert len(c.calls) == 1


def test_real_cpu_checkpoint_inference_selects_program_and_replays_without_model(signed_case, selected_config, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as native
    c = signed_case
    from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
    result = prep.initial_context(state=c.output.parent, source384_config=selected_config)
    receipt = result["source384_context"]
    c.output = Path(receipt["output"])
    inference = json.loads((c.output / "inference.json").read_bytes())
    assert inference["native_worker_executed"] is inference["inference_executed"] is True
    assert inference["report"]["preparation"]["paths"] == ["source.py"]
    assert inference["report"]["output"]["model_loads"] == 1
    assert receipt["summary"]["program_source_files"] == receipt["summary"]["inference_python_files"] == 1
    assert receipt["summary"]["harness_support_files"] == 3
    assert set(receipt["source_hashes"]) == set(c.hashes)
    assert {row["path"] for row in receipt["source_inventory"]} == set(c.hashes)
    assert all(row.get("path") != profiles.SMOKE for row in receipt["summary"]["candidate_samples"])
    monkeypatch.setattr(native, "_worker", lambda *a, **k: pytest.fail("observation invoked neural worker"))
    assert owner.validate_source384_context(repository=c.root, expected_receipt=receipt) == receipt


def test_selection_bounds_precede_signature_or_manifest_replay(signed_case, monkeypatch):
    monkeypatch.setattr(scope_owner, "MAX_BYTES", 32)
    monkeypatch.setattr(scope_owner.local, "_manifest", lambda *a, **k: pytest.fail("manifest verification before bound"))
    with pytest.raises(ValueError, match="bounded Source384 selection"):
        _scope(signed_case)


@pytest.mark.parametrize("damage", ["hardlink", "parent_replacement"])
def test_shared_support_reader_rejects_hardlinks_and_parent_path_swap(tmp_path, monkeypatch, damage):
    import os
    from ipfs_accelerate_py.agent_supervisor.runtime import terminal_source_partition as shared
    root = tmp_path / "root"
    root.mkdir()
    source = root / "support.json"
    source.write_bytes(b"canonical")
    sources = {source.name: {"sha256": owner._sha(source.read_bytes())}}
    if damage == "hardlink":
        os.link(source, tmp_path / "alias")
    else:
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        (replacement / source.name).write_bytes(b"different")
        fstat = shared.os.fstat
        calls = []
        def swap(fd):
            result = fstat(fd)
            calls.append(fd)
            if len(calls) == 2:
                root.rename(tmp_path / "previous")
                replacement.rename(root)
            return result
        monkeypatch.setattr(shared.os, "fstat", swap)
    with pytest.raises(ValueError, match="bounded regular|changed during verification"):
        shared._read(root, source.name, sources, 1024)
