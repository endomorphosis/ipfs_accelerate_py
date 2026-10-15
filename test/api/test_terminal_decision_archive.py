"""Selected worker archives bind current declared capabilities before dispatch.

Authored archives exercise the existing bounded manifest/tar reader. No
provider, Docker environment, planning call, or benchmark verifier is launched.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from test.api.test_terminal_timeout_configuration import (
    _authored_prepare_dataset, _timeout_runtime_archive,
)

DECISION = {
    "semantic_decision_policy_artifact": "/opt/ipfs-supervisor/task-policy/decision.json",
    "semantic_decision_policy_sha256": "d" * 64,
}


def _decision_runtime_archive(tmp_path, *, fault=None):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    members = []
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as original:
        for member in original:
            members.append((member.name, original.extractfile(member).read()))
    selected = benchmark._decision_runtime_sources()
    target = next(iter(selected))
    for name, source in selected.items():
        raw = source.read_bytes()
        row = {"path": name, "bytes": len(raw), "sha256": benchmark._hash(source), "mode": 0o644}
        if name != target or fault != "missing_row":
            manifest["files"].append(row)
        if name == target and fault == "duplicate_row":
            manifest["files"].append(dict(row))
        if name == target and fault == "stale_row":
            row["sha256"] = "0" * 64
        if name == target and fault == "missing_member":
            continue
        if name == target and fault == "stale_payload":
            raw = (b"#" if raw[:1] != b"#" else b" ") + raw[1:]
        members.append((name, raw))
        if name == target and fault == "duplicate_member":
            members.append((name, raw))
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for name, raw in members:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), 0o644
            bundle.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    benchmark._json(archive / "manifest.json", manifest)
    return archive, manifest


def test_selected_archive_checks_actual_current_members_and_leaves_legacy_reader_unchanged(tmp_path):
    archive, manifest = _decision_runtime_archive(tmp_path)
    legacy = benchmark._require_current_timeout_runtime(archive, manifest)
    assert set(legacy) == {benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name
                           for name in benchmark.TIMEOUT_RUNTIME_SOURCE_NAMES}
    verified = benchmark._require_current_timeout_runtime(
        archive, manifest, selected_sources=benchmark._decision_runtime_sources())
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    assert verified == expected
    assert len(verified) == len(legacy) + len(benchmark._decision_runtime_sources())
    assert {name: verified[name] for name in legacy} == legacy


@pytest.mark.parametrize("fault", [
    "missing_row", "missing_member", "stale_row", "stale_payload", "duplicate_row", "duplicate_member",
])
def test_self_consistent_archive_digest_cannot_hide_missing_or_changed_worker_capability(tmp_path, fault):
    archive, manifest = _decision_runtime_archive(tmp_path, fault=fault)
    assert benchmark._hash(archive / "runtime.tar.gz") == manifest["archive_sha256"]
    # Deadline sources remain current; the opt-in capability check is needed.
    assert benchmark._require_current_timeout_runtime(archive, manifest)
    with pytest.raises(ValueError, match="selected decision"):
        benchmark._require_current_timeout_runtime(
            archive, manifest, selected_sources=benchmark._decision_runtime_sources())


def test_selected_preparation_refuses_deadline_only_archive_before_output_or_process(tmp_path, monkeypatch):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "output"
    def unexpected(*args, **kwargs):
        pytest.fail("missing worker capability reached Harbor dry-run")
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **DECISION)
    assert not output.exists()


def _prepared_decision_trial(tmp_path, monkeypatch):
    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="Authored dry run.", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **DECISION)
    assert prepared["runtime_decision_source_sha256"] == {
        name: benchmark._hash(source) for name, source in benchmark._decision_runtime_sources().items()
    }
    return output, prepared


@pytest.mark.parametrize("fault", ["missing", "incomplete", "changed", "local_source_changed"])
def test_execute_refuses_missing_or_changed_selected_source_pins_before_invocation(tmp_path, monkeypatch, fault):
    output, prepared = _prepared_decision_trial(tmp_path, monkeypatch)
    pins = prepared["runtime_decision_source_sha256"]
    target = next(iter(pins))
    if fault == "missing":
        prepared.pop("runtime_decision_source_sha256")
    elif fault == "incomplete":
        pins.pop(target)
    elif fault == "changed":
        pins[target] = "0" * 64
    else:
        original = benchmark._hash
        selected = benchmark._decision_runtime_sources()[target]
        monkeypatch.setattr(benchmark, "_hash", lambda path: "0" * 64 if Path(path) == selected else original(path))
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("changed decision pins reached process invocation"))
    with pytest.raises(ValueError, match="decision source pins"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_execute_replays_selected_members_for_externally_retained_preparation(tmp_path, monkeypatch):
    output, prepared = _prepared_decision_trial(tmp_path, monkeypatch)
    archive = Path(prepared["archive"])
    manifest = json.loads((archive / "manifest.json").read_text())
    target = next(iter(benchmark._decision_runtime_sources()))
    rows = []
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as original:
        for member in original:
            if member.name != target:
                rows.append((member.name, original.extractfile(member).read()))
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for name, raw in rows:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), 0o644
            bundle.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    benchmark._json(archive / "manifest.json", manifest)
    prepared.update(archive_sha256=manifest["archive_sha256"],
                    manifest_sha256=benchmark._hash(archive / "manifest.json"))
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("missing selected member reached process invocation"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()
