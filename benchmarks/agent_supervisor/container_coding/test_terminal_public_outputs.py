"""Authored public files only; no benchmark private or verifier state fixtures."""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_public_outputs as evidence


class LocalEnvironment:
    def __init__(self, app, mutate=None):
        self.app, self.calls, self.mutate = app, [], mutate

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        argv = shlex.split(kwargs["command"])
        assert argv[:4] == [evidence.PYTHON, "-I", "-S", "-c"]
        assert kwargs["user"] == "supervisor" and kwargs["cwd"] == "/"
        assert kwargs["timeout_sec"] == 15
        assert argv[4] == evidence.CAPTURE_SCRIPT
        script = argv[4].replace('root = "/app"', 'root = ' + repr(str(self.app)))
        if self.mutate:
            script = self.mutate(script)
        result = subprocess.run([sys.executable, "-I", "-S", "-c", script],
            capture_output=True, text=True, timeout=5, cwd="/")
        return SimpleNamespace(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)


def _fixture(tmp_path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(); logs.mkdir()
    (logs / "instruction.md").write_text("Authored public instructions only.\n")
    return app, logs, LocalEnvironment(app)


def test_exact_public_bytes_before_after_and_bounded_diff_are_retained_without_execution(tmp_path):
    app, logs, environment = _fixture(tmp_path)
    original = b"raise RuntimeError('candidate must never execute')\n"
    final = original + b"# authored output\n\xff\x00"
    (app / "bottle.py").write_bytes(original)
    (app / "private-state").mkdir()
    (app / "private-state" / "never-read.txt").write_text("not a public output")
    before = asyncio.run(evidence.capture_public_inputs(environment, logs))
    assert before["files"]["report.jsonl"] == {"status": "missing"}
    (app / "bottle.py").write_bytes(final)
    report = b'{"diagnostic":"authored"}\n'
    (app / "report.jsonl").write_bytes(report)
    captured = asyncio.run(evidence.export_public_outputs(environment, logs))
    root = logs / "public-output-evidence"
    assert (root / "before/bottle.py").read_bytes() == original
    assert (root / "after/bottle.py").read_bytes() == final
    assert (root / "after/report.jsonl").read_bytes() == report
    assert captured["files"]["bottle.py"]["sha256"] == hashlib.sha256(final).hexdigest()
    assert captured["before_receipt_sha256"] == before["receipt_sha256"]
    assert captured["public_instruction"]["sha256"] == hashlib.sha256((logs / "instruction.md").read_bytes()).hexdigest()
    assert captured["candidate_code_executed"] is False and captured["private_state_accessed"] is False
    assert captured["verifier_state_accessed"] is False and captured["signed"] is False
    assert captured["quiescence"] == "caller_lifecycle_receipt_required"
    assert len(environment.calls) == 2
    assert {path.name for path in (root / "after").iterdir()} == {"bottle.py", "report.jsonl", "receipt.json"}
    assert b"not a public output" not in b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "oversized"])
def test_unsafe_public_name_is_rejected_before_copy(tmp_path, kind):
    app, logs, environment = _fixture(tmp_path)
    path = app / "bottle.py"
    if kind in {"symlink", "hardlink"}:
        other = tmp_path / "not-public.txt"; other.write_text("must not copy")
        path.symlink_to(other) if kind == "symlink" else os.link(other, path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        with path.open("wb") as stream:
            stream.truncate(evidence.FILES["bottle.py"] + 1)
    captured = asyncio.run(evidence.export_public_outputs(environment, logs))
    assert captured["files"]["bottle.py"]["status"] == "rejected"
    assert not (logs / "public-output-evidence/after/bottle.py").exists()


def test_app_directory_symlink_does_not_allow_reading_outside_declared_root(tmp_path):
    target, logs = tmp_path / "other", tmp_path / "logs"
    target.mkdir(); logs.mkdir()
    (target / "bottle.py").write_text("must not copy")
    app = tmp_path / "app"; app.symlink_to(target, target_is_directory=True)
    captured = asyncio.run(evidence.export_public_outputs(LocalEnvironment(app), logs))
    assert all(item == {"status": "unavailable", "reason": "app_directory_unavailable"}
               for item in captured["files"].values())


def test_file_changed_between_read_and_path_check_is_not_exported(tmp_path):
    app, logs, _environment = _fixture(tmp_path)
    (app / "bottle.py").write_text("authored public source\n")
    def mutate(script):
        return script.replace('                current = os.stat(',
            '                os.utime(name, ns=(after.st_atime_ns, after.st_mtime_ns + 1), dir_fd=directory)\n'
            '                current = os.stat(')
    captured = asyncio.run(evidence.export_public_outputs(LocalEnvironment(app, mutate), logs))
    assert captured["files"]["bottle.py"] == {"status": "rejected", "reason": "changed_during_capture"}
    assert not (logs / "public-output-evidence/after/bottle.py").exists()


def test_missing_baseline_is_explicit_and_no_old_trial_is_overwritten(tmp_path):
    app, logs, environment = _fixture(tmp_path)
    (app / "bottle.py").write_text("public output")
    captured = asyncio.run(evidence.export_public_outputs(environment, logs))
    assert captured["before_receipt_sha256"] is None
    assert captured["diffs"]["bottle.py"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="never overwrite"):
        asyncio.run(evidence.export_public_outputs(environment, logs))
    assert len(environment.calls) == 1


def test_baseline_mutation_is_detected_before_final_capture(tmp_path):
    app, logs, environment = _fixture(tmp_path)
    (app / "bottle.py").write_text("first public source")
    asyncio.run(evidence.capture_public_inputs(environment, logs))
    path = logs / "public-output-evidence/before/bottle.py"
    path.chmod(0o644); path.write_text("tampered baseline")
    with pytest.raises(ValueError, match="baseline bytes differ"):
        asyncio.run(evidence.export_public_outputs(environment, logs))
    assert len(environment.calls) == 1


@pytest.mark.parametrize("kind", ["foreign_file", "hash", "size", "duplicate", "status", "base64"])
def test_untrusted_capture_envelope_cannot_expand_scope_or_rebind_bytes(kind):
    raw = b"public bytes"
    value = {"schema": evidence.SCHEMA, "files": {
        "bottle.py": {"status": "captured", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "base64": base64.b64encode(raw).decode()}, "report.jsonl": {"status": "missing"}}}
    if kind == "foreign_file":
        value["files"]["owner.json"] = {"status": "missing"}
    elif kind == "hash":
        value["files"]["bottle.py"]["sha256"] = "0" * 64
    elif kind == "size":
        value["files"]["bottle.py"]["bytes"] += 1
    elif kind == "status":
        value["files"]["bottle.py"] = {"status": "proved"}
    elif kind == "base64":
        value["files"]["bottle.py"]["base64"] = "?"
    encoded = json.dumps(value)
    if kind == "duplicate":
        encoded = encoded.replace('"schema":', '"schema":"duplicate", "schema":')
    with pytest.raises(ValueError):
        evidence._validate_envelope(encoded)


def test_diff_limits_are_explicit_and_full_source_bytes_remain_separate():
    raw, descriptor = evidence._diff(b"old\n", b"x" * (evidence.MAX_DIFF_BYTES * 2), "bottle.py")
    assert len(raw) <= evidence.MAX_DIFF_BYTES and descriptor["truncated"] is True
    raw, descriptor = evidence._diff(b"old\n", b"line\n" * (evidence.MAX_DIFF_LINES + 1), "bottle.py")
    assert raw is None and descriptor == {"status": "omitted", "reason": "line_bound_exceeded"}


def test_host_output_symlink_is_rejected_before_environment_access(tmp_path):
    app, logs, environment = _fixture(tmp_path)
    target = tmp_path / "other"; target.mkdir()
    (logs / "public-output-evidence").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        asyncio.run(evidence.capture_public_inputs(environment, logs))
    assert not environment.calls


def test_preexisting_diff_directory_cannot_redirect_evidence_writes(tmp_path):
    app, logs, environment = _fixture(tmp_path)
    (app / "bottle.py").write_text("public source")
    asyncio.run(evidence.capture_public_inputs(environment, logs))
    target = tmp_path / "other"; target.mkdir()
    (logs / "public-output-evidence/diffs").symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        asyncio.run(evidence.export_public_outputs(environment, logs))
    assert not list(target.iterdir())
