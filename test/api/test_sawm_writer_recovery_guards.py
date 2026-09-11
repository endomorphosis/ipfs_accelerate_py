"""Bounded operator guard inputs grant no callback or source authority."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import sawm_writer_recovery as recovery


def test_namespace_mismatch_is_unknown_before_kernel_absence(tmp_path, monkeypatch):
    binding = recovery.process.observe_process(os.getpid())
    real = os.readlink
    monkeypatch.setattr(
        os,
        "readlink",
        lambda path: (
            "foreign" if str(path) == f"/proc/{os.getpid()}/ns/pid" else real(path)
        ),
    )
    with pytest.raises(
        recovery.process.GracefulRecoveryUnverified, match="namespace_unknown"
    ):
        recovery.canonical_writer_lost(tmp_path / "unopened.duckdb", binding)


@pytest.mark.parametrize("kind", ["fifo", "symlink", "oversize"])
def test_recovery_artifact_read_is_bounded_regular_only(tmp_path, kind):
    path = tmp_path / "artifact"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_bytes(b"x" * 17)
    with pytest.raises(
        (
            recovery.process.GracefulRecoveryUnverified,
            recovery.observation.OwnerObservationUnavailable,
        )
    ):
        recovery.file_binding(path, 16)


@pytest.mark.parametrize("changed", ["hold", "config", "unit"])
def test_actual_configured_hold_and_inactive_job_are_rechecked(
    tmp_path, monkeypatch, changed
):
    run = tmp_path / "run"
    run.mkdir()
    hold = run / "HOLD"
    hold.write_text("disposable reviewed recovery\n")
    config = {"runtime_paths": {"root": "run"}}
    watchdog = tmp_path / "fleet.json"
    watchdog.write_text(
        json.dumps(
            {
                "boards": [
                    {"id": "sawm", "cwd": str(tmp_path), "hold_files": [str(hold)]}
                ]
            }
        )
    )
    active = {"value": False}
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            stdout="ActiveState=active\nMainPID=900\n"
            if active["value"]
            else "ActiveState=inactive\nMainPID=0\n"
        ),
    )
    gate = recovery.startup_exclusion(tmp_path, config, watchdog)
    gate()
    if changed == "hold":
        hold.write_text("changed")
    elif changed == "config":
        watchdog.write_text("{}")
    else:
        active["value"] = True
    with pytest.raises(recovery.process.GracefulRecoveryUnverified):
        gate()


def test_unconfigured_hold_never_excludes_native_startup(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "HOLD").touch()
    watchdog = tmp_path / "fleet.json"
    watchdog.write_text(
        json.dumps({"boards": [{"id": "sawm", "cwd": str(tmp_path), "hold_files": []}]})
    )
    with pytest.raises(
        recovery.process.GracefulRecoveryUnverified, match="not_configured"
    ):
        recovery.startup_exclusion(
            tmp_path, {"runtime_paths": {"root": "run"}}, watchdog
        )


def test_live_marker_disappearance_is_not_accepted_as_intentional_cleanup(
    tmp_path, monkeypatch
):
    binding = dataclasses.asdict(recovery.process.observe_process(os.getpid()))
    # Our observer excludes itself; use a separate isolated process-census fixture
    # to exercise the concrete live-marker rule without spawning a native board.
    binding["pid"] = 991991
    expected = {
        "owner": binding,
        "controller": binding,
        "lanes": [],
        "markers": {"missing.pid": {"pid": 991991, "file": {}}},
    }
    real_iter = Path.iterdir
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda self: (
            iter([Path("/proc/991991")]) if self == Path("/proc") else real_iter(self)
        ),
    )
    monkeypatch.setattr(
        recovery.process,
        "_stat",
        lambda _: ["S", "1", str(binding["process_group"]), str(binding["session"])],
    )
    monkeypatch.setattr(os, "readlink", lambda _: str(tmp_path))
    monkeypatch.setattr(recovery.process, "_read_proc", lambda _: b"fixture\0")
    monkeypatch.setattr(recovery.process, "require_exact_process", lambda _: None)
    with pytest.raises(
        recovery.process.GracefulRecoveryUnverified, match="marker_absent"
    ):
        recovery.scoped_census(tmp_path, expected)


@pytest.mark.parametrize("lineage", ["group", "session", "parent"])
@pytest.mark.parametrize("denied", [False, True])
def test_reparented_or_renamed_scoped_actor_is_never_unrelated(
    tmp_path, monkeypatch, lineage, denied
):
    binding = dataclasses.asdict(recovery.process.observe_process(os.getpid()))
    binding.update(pid=991991, process_group=991991, session=991991)
    expected = {"owner": binding, "controller": binding, "lanes": [], "markers": {}}
    fields = ["S", "1", "881881", "881881", *(["0"] * 15), "123456"]
    fields[{"group": 2, "session": 3, "parent": 1}[lineage]] = "991991"
    real_iter = Path.iterdir
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda self: (
            iter([Path("/proc/771771")]) if self == Path("/proc") else real_iter(self)
        ),
    )
    monkeypatch.setattr(recovery.process, "_stat", lambda _: fields)

    def cwd(_):
        if denied:
            raise PermissionError("disposable inaccessible descendant")
        return "/changed-unrelated-cwd"

    monkeypatch.setattr(os, "readlink", cwd)
    monkeypatch.setattr(recovery.process, "_read_proc", lambda _: b"renamed\0")
    reason = "observation_unavailable" if denied else "additional_native_scope_process"
    with pytest.raises(recovery.process.GracefulRecoveryUnverified, match=reason):
        recovery.scoped_census(tmp_path, expected)


def test_unrelated_unobservable_actor_is_recorded_without_global_exclusion(
    tmp_path, monkeypatch
):
    binding = dataclasses.asdict(recovery.process.observe_process(os.getpid()))
    binding.update(pid=991991, process_group=991991, session=991991)
    expected = {"owner": binding, "controller": binding, "lanes": [], "markers": {}}
    real_iter = Path.iterdir
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda self: (
            iter([Path("/proc/771771")]) if self == Path("/proc") else real_iter(self)
        ),
    )
    monkeypatch.setattr(
        recovery.process, "_stat", lambda _: ["S", "1", "881881", "881881"]
    )

    def denied(_):
        raise PermissionError("disposable unrelated peer")

    monkeypatch.setattr(os, "readlink", denied)
    result = recovery.scoped_census(tmp_path, expected)
    assert result["unavailable_unrelated_pids"] == [771771]
    assert result["global_process_visibility_claimed"] is False


def test_late_source_drift_denies_owner_and_controller_signals(tmp_path, monkeypatch):
    binding = dataclasses.asdict(recovery.process.observe_process(os.getpid()))
    expected = {
        "owner": binding,
        "controller": binding,
        "lanes": [],
        "source": {"head": "reviewed"},
        "custody": {"lost": True},
    }
    config = {
        "quack_owner": {"database_path": "control.duckdb"},
        "runtime_paths": {"state": "state"},
    }
    monkeypatch.setattr(recovery, "inspect", lambda *_: expected)
    monkeypatch.setattr(recovery, "source_binding", lambda *_: {"head": "changed"})
    signals = []
    monkeypatch.setattr(
        recovery.signal, "pidfd_send_signal", lambda *args: signals.append(args)
    )

    def native_boundary(**kwargs):
        kwargs["effect_gate"]()
        raise AssertionError("changed source passed the actual effect gate")

    monkeypatch.setattr(
        recovery.process, "gracefully_close_native_lanes", native_boundary
    )
    with pytest.raises(
        recovery.process.GracefulRecoveryUnverified, match="source_binding_changed"
    ):
        recovery.close_reviewed(
            root=tmp_path,
            config_path=tmp_path / "config.json",
            config=config,
            expected=expected,
            startup_exclusion_gate=lambda: None,
            record_phase=lambda _: None,
        )
    assert signals == []


@pytest.mark.parametrize(
    "failure", ["collision", "parent_symlink", "replacement", "journal_error"]
)
def test_exclusive_durable_journal_preserves_foreign_and_partial_artifacts(
    tmp_path, failure
):
    target = tmp_path / "journal.jsonl"
    if failure == "collision":
        target.write_text("foreign")
        with pytest.raises(FileExistsError):  # noqa: SIM117 - explicit entry-failure boundary
            with recovery.recovery_journal(target, "a" * 64):
                raise AssertionError("foreign journal admitted")
        assert target.read_text() == "foreign"
        return
    if failure == "parent_symlink":
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(
            recovery.process.GracefulRecoveryUnverified, match="parent_symlink"
        ), recovery.recovery_journal(link / target.name, "a" * 64):
            raise AssertionError("symlink parent admitted")
        assert not target.exists()
        return
    with pytest.raises((ValueError, recovery.process.GracefulRecoveryUnverified)):  # noqa: SIM117 - preserve yielded prefix on caller failure
        with recovery.recovery_journal(target, "a" * 64) as record:
            record("controller_suspend_prepared")
            if failure == "replacement":
                target.rename(tmp_path / "preserved-prefix")
                target.write_text("foreign")
                record("controller_all_threads_stopped")
            raise ValueError("disposable journal consumer failure")
    if failure == "replacement":
        assert target.read_text() == "foreign"
        target = tmp_path / "preserved-prefix"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["controller_suspend_prepared"]
    assert rows[0]["callback_settlement_authority"] is False
