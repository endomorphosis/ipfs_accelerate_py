"""Configured handoffs remain bounded diagnostics alongside ordinary job state."""

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import diagnostic_handoff as handoff
from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair


@pytest.fixture
def sample(tmp_path):
    directory = tmp_path / "repairs/doep"
    directory.mkdir(parents=True, mode=0o700)
    report = directory / "root-recovery/report.json"
    report.parent.mkdir()
    # Even explicitly referenced reports can contain private appendices. Only
    # their digest/path is carried, never values or model-written instructions.
    report.write_text(
        json.dumps(
            {
                "summary": "Authorization: Bearer fixture_private_token",
                "next_action": "EXECUTE_UNSAFE_OLD_SCRIPT",
            }
        )
    )
    board = {
        "id": "doep",
        "cwd": str(tmp_path),
        "hold_files": [],
        "diagnostic_handoff": "monitoring-handoff.json",
    }
    observation = {
        "board_id": "doep",
        "health": "degraded",
        "details": {
            "source_heads": {".": "a" * 40, "external/runtime": "b" * 40},
            "source_integrity": {"configured": True, "valid": True},
        },
    }
    now = time.time()
    heads, integrity = handoff.source_binding(observation)
    value = {
        "schema": handoff.SCHEMA,
        "board_id": "doep",
        "observed_at": now - 10,
        "expires_at": now + 3600,
        "source_heads": heads,
        "source_integrity_sha256": integrity,
        "completed_stages": ["source_qualification"],
        "reports": [
            {
                "path": "root-recovery/report.json",
                "stage": "source_qualification",
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        ],
    }
    path = directory / board["diagnostic_handoff"]
    path.write_text(json.dumps(value))
    return board, directory, observation, now, value, path, report


def load(sample):
    board, directory, observation, now, *_ = sample
    return handoff.load_diagnostic_handoff(
        board, directory, observation, observed_at=now - 1, now=now
    )


def test_actual_prompt_carries_only_closed_stage_and_exact_references(sample):
    board, directory, observation, _now, _value, _path, report = sample
    result = load(sample)
    assert result["status"] == "available" and result["diagnostic_only"] is True
    assert result["report_references"][0]["path"] == str(report)
    prompt = repair.repair_prompt(
        board,
        {"observation": observation},
        {"repair_worker": {"cwd": board["cwd"]}},
        directory / "next.json",
        {"continuation": {"next_action": "KEEP_PRIOR_JOB"}},
        result,
    )
    assert "source_qualification" in prompt and str(report) in prompt
    assert "KEEP_PRIOR_JOB" in prompt
    assert (
        "fixture_private_token" not in prompt
        and "EXECUTE_UNSAFE_OLD_SCRIPT" not in prompt
    )
    assert (
        "no authority" in prompt
        and "Never execute a referenced old recovery tool" in prompt
    )
    assert "PCPR remains held" in prompt and "Never start llama-server" in prompt


@pytest.mark.parametrize(
    "change",
    [
        "stale",
        "future",
        "lifetime",
        "bool_time",
        "nan_time",
        "wrong_board",
        "heads",
        "integrity",
        "stage_prose",
        "unknown_field",
        "too_many_refs",
        "missing_report",
        "wrong_digest",
        "oversized_report",
        "oversized_handoff",
        "missing_handoff",
        "malformed_json",
        "duplicate_json",
        "source_missing",
        "foreign_probe",
        "old_probe",
        "path_escape",
        "absolute_path",
        "vault_path",
        "ref_symlink",
        "dir_symlink",
        "handoff_symlink",
        "fifo",
        "hardlink",
        "writable_report",
        "missing_stage_report",
        "nonprivate_root",
    ],
)
def test_unavailable_handoff_never_carries_prose_or_changes_job(sample, change):
    board, directory, observation, now, value, path, report = sample
    if change == "stale":
        value["expires_at"] = now - 1
    if change == "future":
        value["observed_at"] = now + 1
    if change == "lifetime":
        value["expires_at"] = now + 90000
    if change == "bool_time":
        value["observed_at"] = True
    if change == "nan_time":
        value["observed_at"] = float("nan")
    if change == "wrong_board":
        value["board_id"] = "sawm"
    if change == "heads":
        value["source_heads"] = {".": "c" * 40}
    if change == "integrity":
        value["source_integrity_sha256"] = "f" * 64
    if change == "stage_prose":
        value["completed_stages"] = ["fixture_private_token"]
    if change == "unknown_field":
        value["env"] = "fixture_private_token"
    if change == "too_many_refs":
        value["reports"] *= 9
    if change == "missing_report":
        report.unlink()
    if change == "wrong_digest":
        value["reports"][0]["sha256"] = "f" * 64
    if change == "oversized_report":
        report.write_bytes(b"x" * (1024 * 1024 + 1))
    if change == "source_missing":
        observation["details"].pop("source_heads")
    if change == "foreign_probe":
        observation["board_id"] = "sawm"
    if change == "path_escape":
        value["reports"][0]["path"] = "../sawm/report.json"
    if change == "absolute_path":
        value["reports"][0]["path"] = str(report)
    if change == "vault_path":
        value["reports"][0]["path"] = "private-vault/report.json"
    if change == "ref_symlink":
        report.rename(report.with_suffix(".kept"))
        report.symlink_to(report.with_suffix(".kept"))
    if change == "dir_symlink":
        report.parent.rename(directory / "kept")
        report.parent.symlink_to(directory / "kept")
    if change == "fifo":
        report.unlink()
        os.mkfifo(report)
    if change == "hardlink":
        os.link(report, report.with_suffix(".link"))
    if change == "writable_report":
        report.chmod(0o666)
    if change == "nonprivate_root":
        directory.chmod(0o775)
    if change == "missing_stage_report":
        value["completed_stages"].append("graceful_closure")
    path.write_text(json.dumps(value))
    if change == "oversized_handoff":
        path.write_bytes(b"x" * 32769)
    if change == "missing_handoff":
        path.unlink()
    if change == "malformed_json":
        path.write_text("{fixture_private_token")
    if change == "duplicate_json":
        path.write_text('{"schema":"x","schema":"fixture_private_token"}')
    if change == "handoff_symlink":
        path.rename(directory / "kept.json")
        path.symlink_to(directory / "kept.json")
    job = directory / "job.json"
    job.write_text('{"attempts":7,"status":"queued","next_attempt_at":9999}')
    before = job.read_bytes()
    result = handoff.load_diagnostic_handoff(
        board,
        directory,
        observation,
        observed_at=now - (181 if change == "old_probe" else 1),
        now=now,
    )
    assert result["status"] == "unavailable"
    assert set(result) == {"status", "diagnostic_only", "reason"}
    assert "fixture_private_token" not in json.dumps(result)
    assert job.read_bytes() == before


@pytest.mark.parametrize("replace", ["file_bytes", "file_entry", "directory"])
def test_changed_reference_during_read_is_omitted(sample, monkeypatch, replace):
    _board, directory, _observation, _now, _value, _path, report = sample
    real_read = handoff.os.read
    changed = []

    def reading(fd, size):
        data = real_read(fd, size)
        if not changed and Path(os.readlink(f"/proc/self/fd/{fd}")) == report:
            changed.append(True)
            if replace == "file_bytes":
                report.write_bytes(b"x" * report.stat().st_size)
            if replace == "file_entry":
                report.rename(report.with_suffix(".old"))
                report.write_bytes(data)
            if replace == "directory":
                report.parent.rename(directory / "old")
                report.parent.mkdir()
                report.write_bytes(data)
        return data

    monkeypatch.setattr(handoff.os, "read", reading)
    assert load(sample)["status"] == "unavailable"
    assert changed


def test_optional_budget_and_absent_config_do_not_block_prior_continuation(
    sample, monkeypatch
):
    board, directory, observation, now, *_ = sample
    calls = [0]

    def monotonic():
        calls[0] += 1
        return 0 if calls[0] == 1 else 3

    monkeypatch.setattr(handoff.time, "monotonic", monotonic)
    assert load(sample)["reason"] == "read_budget_exceeded"
    board.pop("diagnostic_handoff")
    assert (
        handoff.load_diagnostic_handoff(
            board, directory, observation, observed_at=now, now=now
        )
        == {}
    )


def test_actual_run_job_keeps_old_report_and_validates_configured_handoff(
    sample, monkeypatch
):
    # Handoff continuity should not depend on unrelated host disk allocations.
    monkeypatch.setattr(repair.os, "fstatvfs", lambda _fd: SimpleNamespace(
        f_frsize=4096, f_bavail=1 << 30, f_favail=1 << 20))
    board, directory, observation, _now, *_ = sample
    board["probe"] = {"argv": ["probe"]}
    cfg = {
        "state_dir": str(directory.parents[1]),
        "boards": [board],
        "repair_worker": {
            "cwd": board["cwd"],
            "argv": ["fixture-coder"],
            "retry_seconds": 100,
        },
    }
    prior = directory / "report-old.json"
    repair.write_json(
        prior, {"status": "blocked", "next_action": "KEEP_REAL_PREDECESSOR"}
    )
    job = directory / "job.json"
    repair.write_json(
        job,
        {
            "board_id": "doep",
            "status": "queued",
            "attempts": 2,
            "report_path": str(prior),
            "latest_incident": {"observation": observation},
        },
    )

    def command(spec, **_):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 3, "stdout": "inactive"}
        return {"returncode": 0, "stdout": json.dumps(observation)}

    prompts = []

    class Process:
        def __init__(self, *_args, **kwargs):
            prompts.append(kwargs["stdin"].read().decode())

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair.shutil, "which", lambda _: "/fixture/coder")
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    first = repair.run_job(cfg, board, job)
    assert first["status"] == "queued"
    assert (
        "source_qualification" in prompts[0] and "KEEP_REAL_PREDECESSOR" in prompts[0]
    )
    assert "fixture_private_token" not in prompts[0]
    # The following worker gets an unavailable root handoff, and still uses
    # the last usable job report after its predecessor failed to write one.
    sample[5].unlink()
    repair.run_job(cfg, board, job)
    durable = repair.read_json(job)
    assert '"status": "unavailable"' in prompts[1]
    assert "KEEP_REAL_PREDECESSOR" in prompts[1]
    assert durable["attempts"] == 4 and durable["last_valid_report_path"] == str(prior)
    assert durable["verification"]["verified"] is False
