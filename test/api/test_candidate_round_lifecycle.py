"""Actual registered children and native owner handles anchor retained history."""
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_lifecycle as lifecycle
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_checkpoint as storage
from test.api.test_owner_candidate_dispatch import captured_service, failed_retirement, prepared_candidate  # noqa: F401
from test.api.test_owner_candidate_round_context import source_service, native  # noqa: F401
from test.api.test_terminal_candidate_attempt_accounting import observation


def row(broker, number, *, retired=True):
    return dict(round_number=number, process_started=broker._process is not None,
        session_id=broker.descriptor["session_id"], descriptor_sha256=broker.descriptor_sha256,
        returncode=broker._process.returncode if broker._process is not None else None,
        wait_completed=broker._process is not None, retirement_observed=retired)


def retained(launch, value):
    receipt = lifecycle.persist(source_broker=launch.source_broker, observation=value)
    path = launch.source_broker.descriptor_path.parent / receipt["path"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert path.stat().st_uid == path.stat().st_gid == os.getuid()
    assert path.stat().st_nlink == 1 and path.stat().st_size == receipt["bytes"]
    return path, lifecycle.parse_retention(path.read_bytes())


def test_actual_second_child_cannot_be_omitted_from_history(prepared_candidate, monkeypatch):
    launch, _, prepared, _ = prepared_candidate
    monkeypatch.setattr(dispatch, "_spawn_worker", lambda command, *, stdin, workspace:
        subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(.1)"], stdin=stdin, cwd=workspace))
    running = prepared.start()
    assert running.process.wait(timeout=running.remaining_seconds) == 0
    dispatch.retire_coding_worker(launch=launch, process=running.process, source_broker=running.source_broker)
    prepared.close()
    # Historical evidence remains valid after legitimate edits and lease expiry;
    # it cannot be reloaded to admit work or assert source equivalence.
    (launch.source_broker.workspace / "mod.py").write_text("# authored post-dispatch edit\n")
    monkeypatch.setattr(dispatch, "time", SimpleNamespace(monotonic=lambda: launch.absolute_deadline+1))
    path, value = retained(launch, observation([row(launch.source_broker, 1)]))
    assert len(value["observation"]["rounds"]) == 2
    second = value["observation"]["rounds"][1]
    assert second["process_started"] is True and second["returncode"] == 0
    assert second["retirement_observed"] is True and second["wait_completed"] is False
    assert value["dispatch_reload_allowed"] is value["completion_authority"] is False
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        lifecycle.persist(source_broker=launch.source_broker, observation=observation([row(launch.source_broker, 1)]))
    assert path.read_bytes() == original


def test_failed_registration_is_retained_as_unknown_not_absent(prepared_candidate, monkeypatch):
    launch, _, prepared, _ = prepared_candidate
    children = []
    def spawn(command, *, stdin, workspace):
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(30)"], stdin=stdin, cwd=workspace)
        children.append(child)
        return child
    def refuse(child):
        raise ValueError("authored registration failure")
    monkeypatch.setattr(dispatch, "_spawn_worker", spawn)
    monkeypatch.setattr(prepared.source_broker, "note_spawned", refuse)
    with pytest.raises(ValueError, match="registration failure"):
        prepared.start()
    assert len(children) == 1 and children[0].poll() is not None
    _, value = retained(launch, observation([row(launch.source_broker, 1)]))
    second = value["observation"]["rounds"][1]
    assert second["process_started"] is second["returncode"] is None
    assert second["retirement_observed"] is second["wait_completed"] is False


@pytest.mark.parametrize("change", ["fabricated_second", "false_returncode", "sibling_session", "false_process"])
def test_caller_rows_cannot_rewrite_native_child_history(failed_retirement, change):
    launch, _, _ = failed_retirement
    value = observation([row(launch.source_broker, 1)])
    if change == "fabricated_second":
        value["rounds"].append(dict(value["rounds"][0], round_number=2, session_id="f"*32))
    elif change == "false_returncode":
        value["rounds"][0]["returncode"] = 0
    elif change == "sibling_session":
        value["rounds"][0]["session_id"] = "f"*32
    else:
        value["rounds"][0]["process_started"] = False
    with pytest.raises(ValueError, match="lifecycle binding"):
        lifecycle.persist(source_broker=launch.source_broker, observation=value)
    assert not list(launch.source_broker.descriptor_path.parent.glob("candidate-rounds-*"))


def test_unclosed_broker_cannot_publish_retired_history(captured_service):
    launch, _ = captured_service
    value = observation([row(launch.source_broker, 1, retired=False)])
    with pytest.raises(ValueError):
        lifecycle.persist(source_broker=launch.source_broker, observation=value)
    assert not list(launch.source_broker.descriptor_path.parent.glob("candidate-rounds-*"))


def test_uncaptured_source_broker_cannot_mint_owner_evidence(source_service):
    source_service.close()
    with pytest.raises(ValueError, match="lifecycle binding"):
        lifecycle.persist(source_broker=source_service, observation=observation([row(source_service, 1, retired=False)]))


def test_persistence_failure_removes_only_new_file(failed_retirement, monkeypatch):
    launch, _, _ = failed_retirement
    parent = launch.source_broker.descriptor_path.parent
    sibling = parent / "unrelated-retained-evidence"
    sibling.write_bytes(b"authored sibling")
    def fail(path):
        raise OSError("authored directory fsync failure")
    monkeypatch.setattr(storage, "_sync_directory", fail)
    with pytest.raises(OSError, match="fsync failure"):
        lifecycle.persist(source_broker=launch.source_broker, observation=observation([row(launch.source_broker, 1)]))
    assert not list(parent.glob("candidate-rounds-*")) and sibling.read_bytes() == b"authored sibling"


def test_native_prepared_archive_without_final_lifecycle_keeps_usage_unknown(prepared_candidate):
    from benchmarks.agent_supervisor.container_coding import terminal_candidate_attempt_accounting as accounting
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage
    from test.api.test_terminal_candidate_attempt_accounting import receipt
    launch, _, prepared, _ = prepared_candidate
    original = launch.source_broker
    parent = original.descriptor_path.parent
    # The real preparation retained source/IR/input bytes before child dispatch.
    # Omitting final owner persistence models interruption at this boundary.
    assert prepared.audit.manifest_path.exists()
    assert not list(parent.glob("candidate-rounds-*"))
    prepared.close()
    assert prepared.audit.manifest_path.exists()
    retained_fields = dict(task_id=original.descriptor["task_id"],
        native_prompt_sha256=original.descriptor["native_prompt_sha256"],
        observation=observation([row(original, 1)]))
    first = receipt(retained_fields, 1)
    first["workspace"] = str(original.workspace)
    report = dict(provider_invocations=[first])
    assert measured_usage(report)["total_tokens"] == 110
    value = accounting.apply(report, parent=parent)
    assert value["status"] == "unavailable" and value["potential_unreceipted_attempt"] is True
    usage = measured_usage(report)
    assert usage["provider_calls"] == 1 and usage["total_tokens"] is None
    assert usage["all_invocations_receipted"] is False
