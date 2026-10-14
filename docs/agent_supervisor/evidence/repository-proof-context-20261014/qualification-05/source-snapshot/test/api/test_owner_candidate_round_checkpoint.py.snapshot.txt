"""Owner-private candidate retention from real failed child and source capture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_checkpoint as checkpoint
from test.api.test_worker_source_broker import service, allocated, native  # noqa: F401


@pytest.fixture
def failed_service(service):
    code = ("import sys; from pathlib import Path; sys.stdin.read(); p=Path('mod.py'); "
            "p.write_bytes(p.read_bytes()+b'\\n# unfinished authored candidate\\n'); raise SystemExit(7)")
    child = subprocess.Popen([sys.executable, "-c", code], cwd=service.workspace,
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    try:
        service.note_spawned(child)
        child.communicate("go", timeout=10)
        assert child.returncode == 7
        service.close()
        yield service
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def persist(service, **kwargs):
    return checkpoint.persist_failed_worker_candidate_context(source_broker=service,
        prior_returncode=7, absolute_deadline=kwargs.pop("deadline", time.monotonic() + 60), **kwargs)


def paths(service):
    parent = Path(service.boundary).parent / "worker-observations"
    stem = "candidate-round-" + service.descriptor["session_id"]
    return parent, parent / stem, parent / (stem + ".json")


def test_failed_candidate_has_private_hash_bound_artifacts_without_prompt_or_authority(failed_service):
    baseline = (failed_service.repository / "mod.py").read_bytes()
    observation = persist(failed_service)
    parent, directory, manifest_path = paths(failed_service)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    assert observation["schema"] == checkpoint.OBSERVATION_SCHEMA
    assert observation["status"] == "prepared"
    assert observation["checkpoint_manifest"] == dict(path=manifest_path.name,
        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    assert manifest["observation"] == {key: value for key, value in observation.items()
        if key != "checkpoint_manifest"}
    assert manifest["schema"] == checkpoint.SCHEMA
    assert manifest["context"]["round_number"] == 2
    assert set(manifest["artifacts"]) == {"summary", "capsules", "formal"}
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o400
    assert manifest_path.stat().st_nlink == 1
    for key, row in manifest["artifacts"].items():
        path = parent / row["path"]
        artifact = path.read_bytes()
        assert path.parent == directory
        assert stat.S_IMODE(path.stat().st_mode) == 0o400 and path.stat().st_nlink == 1
        assert row["bytes"] == len(artifact) and row["sha256"] == hashlib.sha256(artifact).hexdigest()
        assert json.loads(artifact)["workspace_generation"] == observation["workspace_generation"]
    assert observation["artifacts_count"] == 3
    assert observation["artifact_bytes"] == sum(row["bytes"] for row in manifest["artifacts"].values())
    assert failed_service.prompt.encode() not in raw
    assert (failed_service.repository / "mod.py").read_bytes() == baseline
    assert (failed_service.workspace / "mod.py").read_bytes() != baseline
    for key, value in checkpoint.FLAGS.items():
        assert manifest[key] == observation[key] == value
    assert "accepted" not in manifest


def test_repeated_persistence_cannot_replace_prior_checkpoint(failed_service):
    first = persist(failed_service)
    parent, directory, manifest = paths(failed_service)
    existing = {path: path.read_bytes() for path in [manifest, *directory.iterdir()]}
    with pytest.raises(FileExistsError):
        persist(failed_service)
    assert existing == {path: path.read_bytes() for path in existing}
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == first["checkpoint_manifest"]["sha256"]


@pytest.mark.parametrize("failure", ["source_drift", "deadline", "write_error", "manifest_error"])
def test_partial_refresh_is_removed_when_freshness_or_publication_fails(failed_service, monkeypatch, failure):
    parent, directory, manifest = paths(failed_service)
    deadline = time.monotonic() + 60
    observed = [time.monotonic()]
    original = checkpoint._write
    def write(path, raw, created):
        original(path, raw, created)
        if failure == "manifest_error" and path == manifest:
            raise OSError("authored manifest failure")
        if path.parent != directory:
            return
        if failure == "source_drift":
            target = failed_service.workspace / "mod.py"
            target.write_bytes(target.read_bytes() + b"\n# changed during persistence\n")
        elif failure == "deadline":
            observed[0] = deadline + 1
        elif failure == "write_error":
            raise OSError("authored file failure")
    if failure == "deadline":
        monkeypatch.setattr(checkpoint, "time", SimpleNamespace(monotonic=lambda: observed[0]))
    monkeypatch.setattr(checkpoint, "_write", write)
    with pytest.raises((ValueError, OSError, TimeoutError)):
        persist(failed_service, deadline=deadline)
    assert not manifest.exists() and not directory.exists()
    assert failed_service.audit_path.is_file()


def test_preexisting_manifest_is_retained_on_publication_collision(failed_service):
    parent, directory, manifest = paths(failed_service)
    manifest.write_bytes(b"existing unrelated artifact")
    manifest.chmod(0o400)
    with pytest.raises(FileExistsError):
        persist(failed_service)
    assert manifest.read_bytes() == b"existing unrelated artifact"
    assert not directory.exists()


@pytest.mark.parametrize("deadline", [None, True, float("inf"), float("nan"), 0.0])
def test_invalid_or_expired_budget_cannot_create_files(failed_service, deadline):
    parent, directory, manifest = paths(failed_service)
    with pytest.raises((ValueError, TimeoutError)):
        persist(failed_service, deadline=deadline)
    assert not directory.exists() and not manifest.exists()
