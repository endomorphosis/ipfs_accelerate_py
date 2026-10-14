"""Real process and replay invariants for the opt-in output helper."""
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


PATH = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/runtime/tool_output_artifact.py"
SPEC = importlib.util.spec_from_file_location("tool_output_artifact_under_test", PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "outputs"
    helper.create_store(path)
    return path


def capture(store, script, **options):
    return helper.run(store=store, argv=[sys.executable, "-c", script], **options)


def test_separate_streams_failure_exit_and_private_content_free_receipt(store):
    result = capture(store, 'import sys; print("STDOUT_SECRET"); print("ERR_SECRET",file=sys.stderr); sys.exit(7)')
    assert result["complete"] and result["cleanup_complete"]
    assert result["exit_code"] == 7
    assert result["streams"]["stdout"]["head_text"] == "STDOUT_SECRET\n"
    assert result["streams"]["stderr"]["head_text"] == "ERR_SECRET\n"
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in store.iterdir())
    receipt = next(store.glob("capture-*.json")).read_text()
    assert "STDOUT_SECRET" not in receipt and "ERR_SECRET" not in receipt
    summary = helper.summarize_store(store)
    assert summary["capture_count"] == summary["complete_capture_count"] == 1
    assert summary["command_failure_count"] == 1
    assert not summary["raw_content_exported"]


def test_hidden_middle_is_exactly_retrievable_and_pages_reconstruct(store):
    content = b"HEAD" + b"x" * 9000 + b"CRITICAL_MIDDLE" + b"y" * 9000 + b"TAIL"
    result = capture(store, 'import os; os.write(1,b"HEAD"+b"x"*9000+b"CRITICAL_MIDDLE"+b"y"*9000+b"TAIL")', preview_bytes=64)
    stream = result["streams"]["stdout"]
    assert "CRITICAL_MIDDLE" not in stream["head_text"] + stream["tail_text"]
    assert stream["omitted_bytes"] == len(content) - 32
    pages = []
    offset = 0
    while offset < len(content):
        page = helper.read(store=store, sha256=stream["sha256"], offset=offset, length=4096)
        pages.append(base64.b64decode(page["data_base64"]))
        offset = page["next_offset"]
    assert b"".join(pages) == content
    assert page["eof"]
    summary = helper.summarize_store(store)
    assert summary["read_count"] == len(pages)
    assert summary["read_bytes"] == len(content)


def test_binary_and_utf8_boundary_have_exact_base64(store):
    content = b"\xff\x00" + "💡".encode() * 10
    result = capture(store, f'import os; os.write(1,{content!r})', preview_bytes=10)
    stream = result["streams"]["stdout"]
    assert stream["display_encoding"] == "utf-8-with-replacement"
    assert base64.b64decode(stream["head_base64"]) == content[:3]
    page = helper.read(store=store, sha256=stream["sha256"], length=100)
    assert page["text"] is None
    assert base64.b64decode(page["data_base64"]) == content


@pytest.mark.parametrize("count,complete", [(8 * 1024 * 1024, True), (8 * 1024 * 1024 + 1, False)])
def test_real_combined_output_cap(store, count, complete):
    result = capture(store, f'import os; data=b"z"*65536; n={count};\nwhile n:\n chunk=data[:min(n,len(data))]; n-=os.write(1,chunk)', timeout=10)
    assert result["complete"] is complete
    assert result["cleanup_complete"]
    if complete:
        assert result["retained_bytes"] == count
        page = helper.read(store=store, sha256=result["streams"]["stdout"]["sha256"], offset=count - 3, length=3)
        assert page["text"] == "zzz"
    else:
        assert result["termination_reason"] == "output_limit"
        assert result["retained_bytes"] == 0
        assert not list(store.glob("*.bin"))
        assert not list(store.glob(".partial-*"))


def test_infinite_flood_kills_group_and_keeps_no_partial_artifact(store):
    started = time.monotonic()
    result = capture(store, 'import os,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN);\nwhile True: os.write(2,b"x"*65536)', timeout=10)
    assert time.monotonic() - started < 5
    assert result["termination_reason"] == "output_limit"
    assert result["exit_code"] == -signal.SIGKILL
    assert result["cleanup_complete"]
    assert not list(store.glob("*.bin"))


def test_output_cap_is_combined_across_both_streams(store):
    result = capture(store, 'import os; os.write(1,b"a"*(4*1024*1024)); os.write(2,b"b"*(4*1024*1024+1))', timeout=10)
    assert result["termination_reason"] == "output_limit"
    assert result["cleanup_complete"] and result["retained_bytes"] == 0


def test_timeout_stops_process_and_reports_partial_observation(store):
    started = time.monotonic()
    result = capture(store, 'import os,time; os.write(1,b"before wait"); time.sleep(60)', timeout=0.15)
    assert time.monotonic() - started < 3
    assert result["termination_reason"] == "timeout"
    assert not result["complete"] and result["cleanup_complete"]
    assert result["observed_bytes"] == 11
    assert result["retained_bytes"] == 0


def test_descendant_retained_pipe_cannot_stall_after_parent_exit(store, tmp_path):
    pid_file = tmp_path / "child.pid"
    script = ('import os,time,signal; pid=os.fork();\n'
              'if pid==0:\n signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)\n'
              f'else:\n open({str(pid_file)!r},"w").write(str(pid)); os._exit(0)')
    started = time.monotonic()
    result = capture(store, script, timeout=5)
    assert time.monotonic() - started < 4
    assert result["termination_reason"] == "descendant_pipe_stall"
    assert result["exit_code"] == 0 and not result["complete"]
    assert result["cleanup_complete"]
    child = Path("/proc") / pid_file.read_text() / "stat"
    assert not child.exists() or child.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_closed_pipes_still_obey_command_deadline(store):
    result = capture(store, 'import os,time; os.close(1); os.close(2); time.sleep(60)', timeout=0.1)
    assert result["termination_reason"] == "timeout"
    assert result["cleanup_complete"]


def test_failed_process_observation_still_kills_and_reaps_owned_child(store, monkeypatch):
    def unavailable(group):
        raise PermissionError("injected /proc failure")
    monkeypatch.setattr(helper, "_alive_group", unavailable)
    result = capture(store, 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)', timeout=0.2)
    assert result["exit_code"] == -signal.SIGKILL
    assert result["termination_reason"] == "cleanup_failed"
    assert not result["cleanup_complete"] and not result["complete"]
    assert not list(store.glob("*.bin"))


@pytest.mark.parametrize("stage", ["spawn", "fence"])
def test_cancellation_during_custody_transitions_does_not_leak_child(store, monkeypatch, stage):
    state = {"cancelled": False}
    if stage == "spawn":
        original = helper.subprocess.Popen
        def start(*args, **kwargs):
            process = original(*args, **kwargs)
            state["cancelled"] = True
            return process
        monkeypatch.setattr(helper.subprocess, "Popen", start)
        script = "import time; time.sleep(60)"
    else:
        original = helper._alive_group
        def inspect(group):
            state["cancelled"] = True
            return original(group)
        monkeypatch.setattr(helper, "_alive_group", inspect)
        script = "print('finished')"
    result = capture(store, script, cancel_check=lambda: state["cancelled"])
    assert result["termination_reason"] == "cancelled"
    assert result["cleanup_complete"] and not result["complete"]
    assert result["retained_bytes"] == 0


def test_command_argv_not_shell_and_stdin_closed(store, tmp_path):
    marker = tmp_path / "should-not-exist"
    argument = f"$(touch {marker})"
    result = helper.run(store=store, argv=[sys.executable, "-c", "import sys; print(sys.argv[1]); print(repr(sys.stdin.read()))", argument])
    assert argument in result["streams"]["stdout"]["head_text"]
    assert "''" in result["streams"]["stdout"]["head_text"]
    assert not marker.exists()


def test_missing_executable_returns_explicit_incomplete_receipt(store):
    result = helper.run(store=store, argv=["/definitely/missing/tool"])
    assert not result["complete"] and result["exit_code"] is None
    assert result["termination_reason"] == "spawn_error"
    assert helper.summarize_store(store)["incomplete_capture_count"] == 1


@pytest.mark.parametrize("mode", ["alter", "truncate", "symlink", "hardlink"])
def test_replay_rejects_tampered_or_linked_artifact(store, tmp_path, mode):
    result = capture(store, 'print("exact output")')
    digest = result["streams"]["stdout"]["sha256"]
    path = store / (digest + ".bin")
    if mode == "alter":
        path.write_bytes(b"x" * path.stat().st_size)
    elif mode == "truncate":
        path.write_bytes(b"")
    elif mode == "symlink":
        path.unlink(); path.symlink_to(tmp_path / "outside")
    else:
        os.link(path, tmp_path / "duplicate")
    with pytest.raises((ValueError, OSError)):
        helper.read(store=store, sha256=digest)
    assert not (store / "reads.json").exists()


def test_replay_detects_mutation_during_read(store, monkeypatch):
    result = capture(store, 'print("original bytes")')
    digest = result["streams"]["stdout"]["sha256"]
    path = store / (digest + ".bin")
    original = helper.os.read
    altered = False
    def mutate(fd, count):
        nonlocal altered
        raw = original(fd, count)
        if raw and not altered:
            altered = True
            path.write_bytes(b"changed bytes\n")
        return raw
    monkeypatch.setattr(helper.os, "read", mutate)
    with pytest.raises(ValueError, match="identity or digest"):
        helper.read(store=store, sha256=digest)


@pytest.mark.parametrize("change", ["raw_extra", "count", "authority", "digest", "duplicate"])
def test_summary_rejects_malformed_worker_receipts(store, change):
    result = capture(store, 'print("payload")')
    path = next(store.glob("capture-*.json"))
    value = json.loads(path.read_bytes())
    if change == "raw_extra":
        value["command"] = "must not be accepted"
    elif change == "count":
        value["retained_bytes"] += 1
    elif change == "authority":
        value["proof_authority"] = True
    elif change == "digest":
        artifact = store / (result["streams"]["stdout"]["sha256"] + ".bin")
        artifact.write_bytes(b"corrupt")
    else:
        path.write_bytes(path.read_bytes()[:-1] + b',"complete":true}')
    if change != "duplicate":
        path.write_bytes(helper._json(value))
    with pytest.raises((ValueError, OSError)):
        helper.summarize_store(store)


@pytest.mark.parametrize("value", ["../outside", "a" * 63, "A" * 64, False])
def test_replay_rejects_nonidentity(store, value):
    with pytest.raises(ValueError):
        helper.read(store=store, sha256=value)


def test_store_root_permissions_symlinks_and_special_entries(tmp_path):
    store = tmp_path / "store"; helper.create_store(store)
    store.chmod(0o755)
    with pytest.raises(ValueError):
        helper.summarize_store(store)
    store.chmod(0o700)
    alias = tmp_path / "alias"; alias.symlink_to(store, target_is_directory=True)
    with pytest.raises(OSError):
        helper.summarize_store(alias)
    os.mkfifo(store / "fifo", 0o600)
    with pytest.raises(ValueError):
        helper.summarize_store(store)


def test_lock_contention_is_bounded(store, monkeypatch):
    monkeypatch.setattr(helper, "LOCK_TIMEOUT_SECONDS", 0.05)
    with (store / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="timed out"):
            helper.summarize_store(store)


def test_capture_count_cap_deduplicates_bytes_but_still_limits_runs(store):
    for _ in range(helper.MAX_CAPTURES):
        assert capture(store, 'print("same")')["complete"]
    assert len(list(store.glob("*.bin"))) == 2
    with pytest.raises(ValueError, match="quota exhausted"):
        capture(store, 'print("same")')


def test_artifact_count_cap_is_enforced_before_spawn(store, tmp_path):
    for index in range(helper.MAX_ARTIFACTS):
        data = str(index).encode()
        path = store / (hashlib.sha256(data).hexdigest() + ".bin")
        path.write_bytes(data); path.chmod(0o600)
    marker = tmp_path / "spawned"
    with pytest.raises(ValueError, match="quota exhausted"):
        capture(store, f'open({str(marker)!r},"w").write("oops")')
    assert not marker.exists()


def test_combined_disk_quota_stops_capture_without_retaining_prefix(store, monkeypatch):
    monkeypatch.setattr(helper, "MAX_STORE_BYTES", 2 * helper.MAX_RECEIPT_BYTES + 10)
    result = capture(store, 'import os; os.write(1,b"a"*20)')
    assert result["termination_reason"] == "store_limit"
    assert result["retained_bytes"] == 0 and not list(store.glob("*.bin"))


def test_real_disk_write_failure_cleans_partial_capture(store, monkeypatch):
    def fail(*args):
        raise OSError("disk is full")
    monkeypatch.setattr(helper.os, "write", fail)
    result = capture(store, 'print("payload")')
    assert result["termination_reason"] == "capture_io_error"
    assert result["cleanup_complete"] and not result["complete"]
    assert not list(store.glob(".partial-*"))


def test_preview_and_retrieval_serialization_remain_bounded(store):
    result = capture(store, 'import os; os.write(1,b"\\x00"*8192); os.write(2,b"\\xff"*8192)', preview_bytes=8192)
    assert sum(s["preview_bytes"] for s in result["streams"].values()) == 8192
    assert len(helper._json(result)) < 80000
    page = helper.read(store=store, sha256=result["streams"]["stdout"]["sha256"], length=8192)
    assert len(helper._json(page)) < 70000


@pytest.mark.parametrize("options", [{"preview_bytes": True}, {"preview_bytes": 8193},
    {"timeout": float("nan")}, {"timeout": 121}, {"timeout": True}])
def test_reject_bad_bounds_before_process(store, options):
    with pytest.raises(ValueError):
        capture(store, 'raise Exception("must not run")', **options)
    assert not list(store.glob("capture-*.json"))


@pytest.mark.parametrize("missing", ["linux", "waitid", "WNOWAIT"])
def test_unsupported_process_custody_platform_rejected_before_spawn(store, monkeypatch, missing):
    if missing == "linux":
        monkeypatch.setattr(helper.sys, "platform", "darwin")
    elif missing == "waitid":
        monkeypatch.setattr(helper.os, "waitid", None)
    else:
        monkeypatch.delattr(helper.os, "WNOWAIT")
    with pytest.raises(ValueError, match="requires Linux"):
        capture(store, "raise Exception('must not run')")
    assert not list(store.glob("capture-*.json"))


def test_cleanup_requires_original_directory_identity(tmp_path):
    path = tmp_path / "owned"
    identity = helper.create_store(path)["store_identity"]
    capture(path, 'print("retained")')
    assert helper.remove_store(path, expected_identity=identity)["removed"]
    assert not path.exists()
    helper.create_store(path)
    # The filesystem may immediately reuse the inode; an altered supplied
    # identity deterministically checks the refusal contract.
    with pytest.raises(ValueError, match="identity changed"):
        helper.remove_store(path, expected_identity={"device": identity["device"], "inode": -1 + identity["inode"]})
    assert path.is_dir()


def test_cleanup_refuses_unknown_entry_instead_of_recursing(tmp_path):
    path = tmp_path / "owned"; identity = helper.create_store(path)["store_identity"]
    (path / "nested").mkdir()
    with pytest.raises(ValueError):
        helper.remove_store(path, expected_identity=identity)
    assert (path / "nested").exists()


def test_cleanup_removes_only_bounded_crash_partials(tmp_path):
    path = tmp_path / "owned"; identity = helper.create_store(path)["store_identity"]
    for name in (".partial-" + "a" * 32 + ".stdout", ".receipt-" + "b" * 32):
        artifact = path / name; artifact.write_bytes(b"unfinished"); artifact.chmod(0o600)
    with pytest.raises(ValueError, match="unfinished store entry"):
        helper.summarize_store(path)
    assert helper.remove_store(path, expected_identity=identity)["removed"]
    assert not path.exists()


def test_cli_sigterm_fences_term_ignoring_command_and_records_incomplete(store, tmp_path):
    marker = tmp_path / "pid"
    script = ('import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
              f'open({str(marker)!r},"w").write(str(os.getpid())); time.sleep(60)')
    process = subprocess.Popen([sys.executable, str(PATH), "run", "--store", str(store),
        "--timeout", "60", "--", sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        command_pid = int(marker.read_text())
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        result = json.loads(stdout)
        assert process.returncode == 1 and not stderr
        assert result["termination_reason"] == "cancelled"
        assert result["cleanup_complete"] and not result["complete"]
        assert result["exit_code"] == -signal.SIGKILL
        assert not Path(f"/proc/{command_pid}").exists()
        assert helper.summarize_store(store)["incomplete_capture_count"] == 1
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=5)
        if marker.exists() and Path(f"/proc/{marker.read_text()}").exists():
            try:
                os.killpg(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_direct_cli_real_capture_and_read(store):
    proc = subprocess.run([sys.executable, str(PATH), "run", "--store", str(store),
        "--timeout", "2", "--preview-bytes", "64", "--", sys.executable, "-c", 'print("CLI")'],
        capture_output=True, check=True, timeout=10)
    result = json.loads(proc.stdout)
    digest = result["streams"]["stdout"]["sha256"]
    proc = subprocess.run([sys.executable, str(PATH), "read", "--store", str(store),
        "--sha256", digest, "--offset", "0", "--length", "4"], capture_output=True, check=True, timeout=10)
    assert json.loads(proc.stdout)["text"] == "CLI\n"


def cli_capture(store, script, *, response_format="compact", preview_bytes=2048, timeout=2):
    options = [] if response_format is None else ["--response-format", response_format]
    process = subprocess.run([sys.executable, str(PATH), "run", "--store", str(store),
        "--timeout", str(timeout), "--preview-bytes", str(preview_bytes), *options,
        "--", sys.executable, "-c", script], capture_output=True, timeout=10)
    assert not process.stderr
    return process, json.loads(process.stdout)


def represented_bytes(value):
    assert ("text" in value) != ("data_base64" in value)
    return value["text"].encode("utf-8") if "text" in value else base64.b64decode(value["data_base64"])


@pytest.mark.parametrize("content", [b"", b"ordinary text\n", "snowman ☃ light 💡\n".encode(),
    b"\xff\x80\x00", b'\x00\n\r\t"\\'])
def test_compact_cli_fully_shown_bytes_are_exactly_one_representation(store, content):
    process, result = cli_capture(store, f'import os; os.write(1,{content!r})')
    assert process.returncode == 0
    assert result["schema"] == "tool-output-artifact-capture@2"
    assert "capture_id" not in result and "preview_budget_bytes" not in result
    stream = result["streams"]["stdout"]
    assert represented_bytes(stream) == content
    assert stream["bytes"] == len(content) and stream["omitted_bytes"] == 0
    assert set(stream) == {"bytes", "omitted_bytes", "text" if "text" in stream else "data_base64"}
    assert result["streams"]["stderr"] == {"bytes": 0, "omitted_bytes": 0, "text": ""}
    assert result["proof_authority"] is result["completion_authority"] is False
    # Existing persisted receipts remain version one with their original keys.
    receipt = json.loads(next(store.glob("capture-*.json")).read_bytes())
    assert receipt["schema"] == "tool-output-artifact-capture@1"
    assert {"capture_id", "preview_budget_bytes"} <= set(receipt)
    assert set(receipt["streams"]["stdout"]) == {
        "sha256", "bytes", "replayable", "omitted_bytes", "preview_bytes"}


@pytest.mark.parametrize("content,preview_bytes,binary_segments", [
    (b"A" * 100 + b"B" * 100, 20, False),
    ("💡".encode() * 10, 12, True),
    (b"\xff" * 40, 12, True),
])
def test_compact_cli_omitted_previews_preserve_exact_segments_and_replay_handle(
        store, content, preview_bytes, binary_segments):
    process, result = cli_capture(store, f'import os; os.write(1,{content!r})', preview_bytes=preview_bytes)
    assert process.returncode == 0 and result["complete"]
    stream = result["streams"]["stdout"]
    limit = (preview_bytes + 1) // 2
    head_length, tail_length = (limit + 1) // 2, limit // 2
    if binary_segments:
        assert "head_text" not in stream and "tail_text" not in stream
        head, tail = base64.b64decode(stream["head_base64"]), base64.b64decode(stream["tail_base64"])
    else:
        assert "head_base64" not in stream and "tail_base64" not in stream
        head, tail = stream["head_text"].encode(), stream["tail_text"].encode()
    assert head == content[:head_length] and tail == content[-tail_length:]
    assert stream["omitted_bytes"] == len(content) - head_length - tail_length
    assert stream["sha256"] == hashlib.sha256(content).hexdigest()
    assert "replayable" not in stream and "display_encoding" not in stream


@pytest.mark.parametrize("offset,length,has_text", [(0, 4, True), (1, 2, False), (4, 4, True)])
def test_compact_cli_read_codepoint_boundaries_are_exact(store, offset, length, has_text):
    content = "💡💡".encode()
    _, capture_result = cli_capture(store, f'import os; os.write(1,{content!r})', preview_bytes=4)
    digest = capture_result["streams"]["stdout"]["sha256"]
    process = subprocess.run([sys.executable, str(PATH), "read", "--store", str(store),
        "--sha256", digest, "--offset", str(offset), "--length", str(length),
        "--response-format", "compact"], capture_output=True, check=True, timeout=10)
    result = json.loads(process.stdout)
    assert result["schema"] == "tool-output-artifact-page@2"
    assert represented_bytes(result) == content[offset:offset + length]
    assert ("text" in result) is has_text
    assert result["sha256"] == digest
    assert result["offset"] == offset and result["length"] == length
    assert result["total_bytes"] == len(content) and result["next_offset"] == offset + length
    assert result["eof"] is (offset + length == len(content))
    assert result["proof_authority"] is result["completion_authority"] is False


def test_compact_cli_keeps_failure_exit_and_timeout_custody_fields(store):
    process, result = cli_capture(store, 'import sys; print("failed"); sys.exit(7)')
    assert process.returncode == 1 and result["exit_code"] == 7 and result["complete"]
    assert result["cleanup_complete"] and result["termination_reason"] is None
    process, result = cli_capture(store,
        'import os,time; os.write(1,b"x"*100); time.sleep(60)', timeout=0.1, preview_bytes=8)
    assert process.returncode == 1 and not result["complete"]
    assert result["termination_reason"] == "timeout" and result["cleanup_complete"]
    assert result["retained_bytes"] == 0 and result["observed_bytes"] == 100
    assert "sha256" not in result["streams"]["stdout"]
    assert "replayable" not in result["streams"]["stdout"]
    assert result["streams"]["stdout"]["omitted_bytes"] == 96


def test_compact_format_is_pure_and_legacy_dictionaries_and_receipts_unchanged(store):
    original = capture(store, 'import os; os.write(1,b"a"*4096)')
    digest = original["streams"]["stdout"]["sha256"]
    page = helper.read(store=store, sha256=digest, offset=1024, length=2048)
    before_store = {path.name: path.read_bytes() for path in store.iterdir()}
    for result in (original, page):
        serialized = helper._json(result)
        assert helper.format_response(result) is result
        assert helper.format_response(result, "legacy") is result
        compact = helper.format_response(result, "compact")
        assert compact["schema"].endswith("@2")
        assert helper._json(result) == serialized
    assert {path.name: path.read_bytes() for path in store.iterdir()} == before_store
    assert page["schema"] == "tool-output-artifact-page@1"
    assert "text" in page and "data_base64" in page
    assert helper.summarize_store(store)["read_count"] == 1


def test_cli_default_and_explicit_legacy_keep_identical_serialization_contract(store, monkeypatch, capsys):
    result = capture(store, 'print("legacy")')
    monkeypatch.setattr(helper, "run", lambda **options: result)
    expected = helper._json(result).decode("ascii") + "\n"
    assert helper.main(["run", "--store", str(store), "--", "unused"]) == 0
    assert capsys.readouterr().out == expected
    assert helper.main(["run", "--store", str(store), "--response-format", "legacy", "--", "unused"]) == 0
    assert capsys.readouterr().out == expected


def test_invalid_cli_response_format_rejected_before_command_runs(store, tmp_path):
    marker = tmp_path / "must-not-run"
    process = subprocess.run([sys.executable, str(PATH), "run", "--store", str(store),
        "--response-format", "unknown", "--", sys.executable, "-c",
        f'open({str(marker)!r},"w").write("bad")'], capture_output=True, timeout=10)
    assert process.returncode == 2 and not marker.exists()
    assert not list(store.glob("capture-*.json"))


@pytest.mark.parametrize("response_format", [None, True, [], "unknown"])
def test_pure_projection_rejects_invalid_format(response_format):
    with pytest.raises(ValueError, match="supported response format"):
        helper.format_response({}, response_format)


def test_compact_projection_rejects_inconsistent_preview_census(store):
    result = capture(store, 'print("exact")')
    result["streams"]["stdout"]["omitted_bytes"] = 1
    with pytest.raises(ValueError, match="byte census differs"):
        helper.format_response(result, "compact")


def test_compact_projection_rejects_oversized_page_before_decoding():
    result = {"schema": "tool-output-artifact-page@1", "sha256": "a" * 64,
        "offset": 0, "length": 8193, "total_bytes": 8193, "next_offset": 8193,
        "eof": True, "proof_authority": False, "completion_authority": False,
        "text": None, "data_base64": base64.b64encode(b"x" * 8193).decode()}
    with pytest.raises(ValueError, match="bounded exact page|byte bound"):
        helper.format_response(result, "compact")
