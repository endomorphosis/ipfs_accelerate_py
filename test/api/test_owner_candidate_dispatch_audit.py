"""Exact pre-dispatch bytes remain reportable after candidate edits and cleanup.

Real Git, native capsule/formal production, failed child wait, typed dispatch
preparation and durable files run normally. Container UID and cleanup executable
are authored by the existing fixture; no provider or hidden task is invoked.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from test.api.test_owner_candidate_dispatch import (
    prepared_candidate, failed_retirement, captured_service, original_argv)  # noqa: F401
from test.api.test_owner_candidate_round_context import source_service, native, SOURCE, CHANGED  # noqa: F401
from test.api.test_owner_candidate_round_context import waited
from test.api.test_owner_candidate_round_prompt import public_service  # noqa: F401
from test.api.test_router_public_instruction import selected, scenario  # noqa: F401


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def retained(handle):
    manifest = json.loads(handle.manifest_bytes)
    rows = {**manifest["artifacts"], **{"source:" + name: row
        for name, row in manifest["candidate_sources"].items()}}
    return manifest, {name: (handle.manifest_path.parent / row["path"]).read_bytes()
                      for name, row in rows.items()}


def test_exact_owner_private_archive_survives_candidate_edits_and_runtime_cleanup(prepared_candidate):
    launch, _, prepared, _ = prepared_candidate
    handle = prepared.audit
    manifest, data = retained(handle)
    session = prepared.source_broker.descriptor["session_id"]
    assert handle.manifest_path.name == audit.audit_manifest_name(session)
    assert handle.manifest_path.stat().st_mode & 0o777 == 0o400
    directory = handle.manifest_path.parent / audit.audit_directory_name(session)
    assert directory.stat().st_mode & 0o777 == 0o700
    assert set(manifest["artifacts"]) == set(audit.FILENAMES)
    for row in [*manifest["artifacts"].values(), *manifest["candidate_sources"].values()]:
        path = handle.manifest_path.parent / row["path"]
        raw = path.read_bytes()
        assert path.parent == directory and stat.S_IMODE(path.stat().st_mode) == 0o400
        assert row == dict(path=path.relative_to(handle.manifest_path.parent).as_posix(),
                           sha256=sha(raw), bytes=len(raw))
    assert handle.manifest_sha256 == sha(handle.manifest_path.read_bytes())
    assert manifest["schema"] == audit.SCHEMA
    assert all(manifest[key] == value for key, value in audit.FLAGS.items())
    assert data["candidate_input"] == prepared.prompt.provider_prompt == prepared.handoff.provider_prompt
    assert data["native_prompt"] == launch.source_broker.prompt.encode()
    assert data["context"] == prepared.snapshot.metadata_bytes
    assert data["handoff"] == prepared.handoff.path.read_bytes()
    assert data["descriptor"] == prepared.source_broker.descriptor_path.read_bytes()
    assert data["original_requirements"] == launch._original_requirements.metadata_bytes
    assert data["original_public_instruction"] == data["original_public_artifact"] == b""
    assert json.loads(data["public_sources"]) == {}
    assert data["source:mod.py"] == CHANGED
    assert set(manifest["candidate_sources"]) == set(prepared.snapshot.metadata["source_hashes"])
    assert data["formal"] == prepared.snapshot.artifacts["formal"]
    report = json.loads(data["formal"])["reports"]["mod.py"]
    definitions = {row["definition"] for row in report["projections"]["lean"]["bindings"]}
    assert "SelectionMin" in definitions and "SelectionMax" not in definitions
    assert manifest["binding"]["candidate_input_sha256"] == sha(data["candidate_input"])
    assert manifest["binding"]["round_context_sha256"] == sha(data["context"])
    assert handle.observation["audit_manifest"] == dict(path=handle.manifest_path.name,
        sha256=handle.manifest_sha256, bytes=len(handle.manifest_bytes))
    assert handle.observation["retained_files"] == len(data)
    assert handle.observation["retained_bytes"] == sum(map(len, data.values()))
    assert not prepared.source_broker._registered.is_set()
    assert data["native_prompt"] not in audit.canonical_json(handle.observation)
    (launch.source_broker.workspace / "mod.py").write_bytes(CHANGED + b"\n# next-round edit\n")
    prepared.close()
    assert not prepared.handoff.path.exists()
    assert not prepared.source_broker.descriptor_path.exists()
    assert retained(handle) == (manifest, data)
    assert (launch.source_broker.repository / "mod.py").read_bytes() == SOURCE
    with pytest.raises(FrozenInstanceError):
        handle.manifest_bytes = b"changed"
    handle.observation["proof_authority"] = True
    assert handle.observation["proof_authority"] is False


def test_duplicate_retention_preserves_existing_files_and_refuses_serialized_authority(prepared_candidate):
    _, _, prepared, _ = prepared_candidate
    before = retained(prepared.audit)
    kwargs = dict(snapshot=prepared.snapshot, prompt=prepared.prompt, source_broker=prepared.source_broker,
                  admission=prepared.admission, handoff=prepared.handoff)
    with pytest.raises(FileExistsError):
        audit.persist_candidate_dispatch_audit(**kwargs)
    assert retained(prepared.audit) == before
    for key, replacement in (("snapshot", prepared.snapshot.metadata),
                             ("admission", json.loads(prepared.admission._binding_bytes)),
                             ("handoff", prepared.handoff.metadata)):
        with pytest.raises(ValueError, match="exact live"):
            audit.persist_candidate_dispatch_audit(**{**kwargs, key: replacement})
    assert retained(prepared.audit) == before


@pytest.mark.parametrize("fault", ["write", "source", "mode", "bytes", "deadline", "interrupt"])
def test_partial_retention_rolls_back_and_prevents_preparation(failed_retirement, monkeypatch, fault):
    launch, retired, _ = failed_retirement
    parent = launch.source_broker.boundary.parent / "worker-observations"
    sentinel = parent / "unrelated-owner-evidence"
    sentinel.write_bytes(b"retain pre-existing evidence")
    writer = audit.storage._write
    calls = []
    def failing(path, raw, created):
        writer(path, raw, created)
        calls.append(path)
        if len(calls) == 4:
            if fault == "write":
                raise OSError("authored audit write failure")
            if fault == "interrupt":
                raise KeyboardInterrupt("authored audit cancellation")
            if fault == "source":
                target = launch.source_broker.workspace / "mod.py"
                target.write_bytes(target.read_bytes() + b"\n# drift during archive\n")
            elif fault == "mode":
                path.chmod(0o640)
            elif fault == "bytes":
                path.chmod(0o600)
                path.write_bytes(raw + b"changed")
                path.chmod(0o400)
            elif fault == "deadline":
                monkeypatch.setattr(audit.time, "monotonic_ns", lambda: 10**30)
    monkeypatch.setattr(audit.storage, "_write", failing)
    expected = KeyboardInterrupt if fault == "interrupt" else (ValueError, OSError)
    with pytest.raises(expected):
        owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    assert len(calls) >= 4
    assert launch._state.phase == "failed"
    assert launch._state.candidate_broker._closed
    assert not launch._state.candidate_broker._registered.is_set()
    assert not tuple(parent.glob("candidate-audit-*"))
    assert not tuple(parent.glob("candidate-dispatch-*"))
    assert sentinel.read_bytes() == b"retain pre-existing evidence"


def test_successful_audit_is_present_before_spawn_and_survives_spawn_failure(prepared_candidate, monkeypatch):
    _, _, prepared, _ = prepared_candidate
    manifest, data = retained(prepared.audit)
    calls = []
    def no_spawn(*args, **kwargs):
        calls.append(True)
        assert retained(prepared.audit) == (manifest, data)
        raise OSError("authored unavailable worker executable")
    monkeypatch.setattr(owner, "_spawn_worker", no_spawn)
    with pytest.raises(OSError, match="unavailable worker executable"):
        prepared.start()
    assert calls == [True]
    assert retained(prepared.audit) == (manifest, data)
    assert not prepared.handoff.path.exists()
    assert not prepared.source_broker.descriptor_path.exists()
    assert prepared.audit.observation["dispatch_reload_allowed"] is False


def test_manifest_paths_require_exact_session_identifiers():
    for value in (None, "../escape", "a" * 31, "A" * 32, 0, True):
        with pytest.raises(ValueError):
            audit.audit_manifest_name(value)


def test_signed_public_requirements_are_retained_exactly_after_candidate_edits(public_service, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
    broker, selected_args, _ = public_service
    broker.original_requirements = prompts.capture_public_requirements(source_broker=broker, **selected_args)
    argv = ["--semantic-repository", str(broker.repository), "--source-view-profile", broker._binding["profile"],
            "--provider", "codex_cli", "--model", "gpt-6.1-sol", "--timeout", "180"]
    for flag, name in (("--public-instruction-artifact", "artifact"),
            ("--public-instruction-sha256", "expected_sha256"), ("--public-instruction-task-cid", "task_cid")):
        argv.extend([flag, str(selected_args[name])])
    launch = owner.capture_original_coding_launch(source_broker=broker, argv=argv,
        absolute_deadline=time.monotonic() + 179)
    waited(broker, close=False)
    (broker.workspace / "answer.py").write_bytes(CHANGED)
    monkeypatch.setattr(owner, "_isolated_cleanup", lambda: subprocess.CompletedProcess(["authored-cleanup"], 0))
    retired = owner.retire_coding_worker(launch=launch, process=broker._process, source_broker=broker)
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=retired)
    try:
        _, data = retained(prepared.audit)
        original = broker.original_requirements
        assert data["original_public_instruction"] == original.public_instruction_block != b""
        assert data["original_public_artifact"] == selected_args["artifact"].read_bytes()
        assert data["public_sources"] == original._public_sources_bytes
        assert json.loads(data["original_requirements"])["selection"] == "public-instruction"
        assert json.loads(data["candidate_input"])["original_public_instruction_block"].encode() == data["original_public_instruction"]
        assert data["source:answer.py"] == CHANGED
        prepared.close()
        assert retained(prepared.audit)[1] == data
    finally:
        prepared.close()
