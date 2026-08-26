from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_casf_owner_management as management,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_reconciliation_lifecycle as reconciliation,
)

_GENERATION_ID = "eaaef-management-test-001"
_CID_A = "sha256:" + "a" * 64
_CID_B = "sha256:" + "b" * 64
_CID_C = "sha256:" + "c" * 64
_BINDING_CID = "sha256:" + "d" * 64
_SNAPSHOT_BINDINGS_CID = "sha256:" + "e" * 64


def _owner_birth() -> dict[str, Any]:
    observed = reconciliation.inspect_process_birth(os.getpid())
    assert observed is not None
    return observed.to_dict()


def _private_state(tmp_path: Path) -> Path:
    state_dir = tmp_path / "casf-owner"
    state_dir.mkdir(mode=0o700, parents=True)
    state_dir.chmod(0o700)
    return state_dir


def _committed_server(
    tmp_path: Path,
) -> tuple[
    management.CASFOwnerManagementServer,
    management.CASFOwnerManagementClient,
]:
    state_dir = _private_state(tmp_path)
    holder: dict[str, management.CASFOwnerManagementServer] = {}

    def _request_stop() -> None:
        def _complete() -> None:
            holder["server"].mark_stopped()

        threading.Thread(target=_complete, daemon=True).start()

    server = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=_request_stop,
        stop_timeout_seconds=5,
    )
    holder["server"] = server
    server.start()
    server.mark_committed(
        owner_start_receipt_cid=_CID_A,
        final_record_cid=_CID_B,
        commit_receipt_cid=_CID_C,
    )
    client = management.CASFOwnerManagementClient(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        timeout_seconds=5,
    )
    return server, client


def test_management_status_and_stop_are_exact_private_operations(
    tmp_path: Path,
) -> None:
    server, client = _committed_server(tmp_path)
    try:
        status = client.status_snapshot()
        assert status == server._snapshot()
        assert status["phase"] == "committed"
        assert status["owner_process_birth"] == _owner_birth()
        assert status["provider_process_started"] is False
        assert status["task_state_mutated"] is False
        encoded_status = json.dumps(status, sort_keys=True)
        for forbidden in (
            "database_path",
            "control.duckdb",
            "transport_token",
            "key_sha256",
            "management.key",
            "SELECT ",
        ):
            assert forbidden not in encoded_status
        capsule_raw = (client.state_dir / management.MANAGEMENT_CAPSULE_NAME).read_bytes()
        assert client.key not in capsule_raw
        assert client.key.hex().encode("ascii") not in capsule_raw

        result = client.stop()
        assert result["committed_owner_stopped"] is True
        assert result["exclusive_owner_lease_released"] is True
        intent = management._validate_stop_intent(
            management._read_private(
                client.state_dir, management.MANAGEMENT_STOP_INTENT_NAME
            ),
            generation_id=_GENERATION_ID,
        )
        retained = management._validate_stop_result(
            management._read_private(
                client.state_dir, management.MANAGEMENT_STOP_RESULT_NAME
            ),
            generation_id=_GENERATION_ID,
        )
        assert intent["stop_request_cid"] == retained["stop_request_cid"]
        assert intent["stop_request_id"] == retained["stop_request_id"]
        for name in (
            management.MANAGEMENT_KEY_NAME,
            management.MANAGEMENT_CAPSULE_NAME,
            management.MANAGEMENT_STOP_INTENT_NAME,
            management.MANAGEMENT_STOP_RESULT_NAME,
        ):
            assert stat.S_IMODE((client.state_dir / name).stat().st_mode) == 0o600
        assert stat.S_IMODE(client.state_dir.stat().st_mode) == 0o700
    finally:
        server.close()


def test_management_stop_adopts_durable_result_after_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, client = _committed_server(tmp_path)
    original_round_trip = client._round_trip

    def _lose_response(request: dict[str, Any]) -> dict[str, Any]:
        response = original_round_trip(request)
        if request["operation"] == "stop":
            raise management.EAAEFCASFOwnerManagementError(
                "injected response loss after broker effect"
            )
        return response

    monkeypatch.setattr(client, "_round_trip", _lose_response)
    try:
        result = client.stop()
        assert result["committed_owner_stopped"] is True
        assert result["exclusive_owner_lease_released"] is True
        assert server.stop_response_sent.is_set()
    finally:
        server.close()


def test_management_stop_is_unavailable_before_owner_commit(tmp_path: Path) -> None:
    state_dir = _private_state(tmp_path)
    server = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=lambda: pytest.fail("provisional stop callback was invoked"),
        stop_timeout_seconds=1,
    )
    server.start()
    client = management.CASFOwnerManagementClient(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        timeout_seconds=1,
    )
    try:
        assert client.status_snapshot()["phase"] == "provisional"
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="owner_not_committed",
        ):
            client.stop()
        assert not (state_dir / management.MANAGEMENT_STOP_INTENT_NAME).exists()
        assert not (state_dir / management.MANAGEMENT_STOP_RESULT_NAME).exists()
    finally:
        server.close()


@pytest.mark.parametrize(
    "tamper",
    ["birth", "key", "capsule_mode", "key_mode", "binding", "snapshot"],
)
def test_management_reattach_rejects_stale_or_tampered_artifacts(
    tmp_path: Path,
    tamper: str,
) -> None:
    server, client = _committed_server(tmp_path)
    state_dir = client.state_dir
    capsule_path = state_dir / management.MANAGEMENT_CAPSULE_NAME
    key_path = state_dir / management.MANAGEMENT_KEY_NAME
    expected_binding_cid = _BINDING_CID
    expected_snapshot_bindings_cid = _SNAPSHOT_BINDINGS_CID
    try:
        if tamper == "birth":
            capsule = management._decode_object(
                capsule_path.read_bytes(), noun="test capsule"
            )
            capsule["owner_process_birth"]["boot_id"] = "stale-boot"
            body = dict(capsule)
            body.pop("capsule_cid")
            capsule["capsule_cid"] = management._cid(body)
            capsule_path.write_bytes(
                management._canonical_bytes(capsule, noun="tampered capsule")
            )
        elif tamper == "key":
            key_path.write_bytes(hashlib.sha256(b"wrong-generation-key").digest())
        elif tamper == "capsule_mode":
            capsule_path.chmod(0o640)
        elif tamper == "key_mode":
            key_path.chmod(0o640)
        elif tamper == "binding":
            expected_binding_cid = "sha256:" + "f" * 64
        else:
            expected_snapshot_bindings_cid = "sha256:" + "f" * 64
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="stale|divergent|not private",
        ):
            management.CASFOwnerManagementClient(
                generation_id=_GENERATION_ID,
                binding_cid=expected_binding_cid,
                snapshot_bindings_cid=expected_snapshot_bindings_cid,
                state_dir=state_dir,
                timeout_seconds=1,
            )
    finally:
        server.close()


def test_management_endpoint_rejects_authenticated_authority_shape_injection(
    tmp_path: Path,
) -> None:
    server, client = _committed_server(tmp_path)
    try:
        request = management._build_request(
            generation_id=_GENERATION_ID,
            capsule_cid=client.capsule["capsule_cid"],
            key=client.key,
            operation="status.snapshot",
            arguments={},
        )
        request["arguments"] = {"database_path": "/tmp/attacker/control.duckdb"}
        cid_body = dict(request)
        cid_body.pop("mac")
        cid_body.pop("request_cid")
        request["request_cid"] = management._cid(cid_body)
        unsigned = dict(request)
        unsigned.pop("mac")
        request["mac"] = management._mac(client.key, unsigned)
        channel = client._connect()
        try:
            channel.settimeout(1)
            management._send_packet(
                channel,
                management._canonical_bytes(request, noun="authority attack"),
            )
            with pytest.raises(
                management.EAAEFCASFOwnerManagementError,
                match="closed early",
            ):
                management._recv_packet(channel)
        finally:
            channel.close()
        assert server._snapshot()["phase"] == "committed"
        assert not (client.state_dir / management.MANAGEMENT_STOP_INTENT_NAME).exists()
    finally:
        server.close()


def test_management_rejects_replayed_authenticated_request_nonce(
    tmp_path: Path,
) -> None:
    server, client = _committed_server(tmp_path)
    try:
        request = management._build_request(
            generation_id=_GENERATION_ID,
            capsule_cid=client.capsule["capsule_cid"],
            key=client.key,
            operation="status.snapshot",
            arguments={},
        )
        first = client._round_trip(request)
        assert first["ok"] is True
        replay = client._round_trip(request)
        assert replay["ok"] is False
        assert replay["error_code"] == "request_nonce_replayed"
        assert replay["result"] == {}
    finally:
        server.close()


def test_management_rejects_cross_generation_capsule_key_swap(
    tmp_path: Path,
) -> None:
    first, first_client = _committed_server(tmp_path / "first")
    second_state = _private_state(tmp_path / "second")
    second = management.CASFOwnerManagementServer(
        generation_id="eaaef-management-test-002",
        binding_cid="sha256:" + "1" * 64,
        snapshot_bindings_cid="sha256:" + "2" * 64,
        state_dir=second_state,
        owner_process_birth=_owner_birth(),
        request_stop=lambda: None,
        stop_timeout_seconds=1,
    )
    second.start()
    try:
        (first_client.state_dir / management.MANAGEMENT_KEY_NAME).write_bytes(
            (second_state / management.MANAGEMENT_KEY_NAME).read_bytes()
        )
        (first_client.state_dir / management.MANAGEMENT_CAPSULE_NAME).write_bytes(
            (second_state / management.MANAGEMENT_CAPSULE_NAME).read_bytes()
        )
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="stale or divergent",
        ):
            management.CASFOwnerManagementClient(
                generation_id=_GENERATION_ID,
                binding_cid=_BINDING_CID,
                snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
                state_dir=first_client.state_dir,
                timeout_seconds=1,
            )
    finally:
        first.close()
        second.close()


def test_management_capsule_and_key_are_fresh_generation_write_once(
    tmp_path: Path,
) -> None:
    state_dir = _private_state(tmp_path)
    first = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=lambda: None,
        stop_timeout_seconds=1,
    )
    first.start()
    first_key = first.key
    first.close()

    second = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=lambda: None,
        stop_timeout_seconds=1,
    )
    with pytest.raises(
        management.EAAEFCASFOwnerManagementError,
        match="already exists",
    ):
        second.start()
    assert management._read_private(state_dir, management.MANAGEMENT_KEY_NAME) == first_key


def test_management_request_requires_exact_peer_process_birth(tmp_path: Path) -> None:
    server, client = _committed_server(tmp_path)
    try:
        request = management._build_request(
            generation_id=_GENERATION_ID,
            capsule_cid=client.capsule["capsule_cid"],
            key=client.key,
            operation="status.snapshot",
            arguments={},
        )
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="authentication differs",
        ):
            management._validate_request(
                management._canonical_bytes(request, noun="peer mismatch request"),
                generation_id=_GENERATION_ID,
                capsule_cid=client.capsule["capsule_cid"],
                key=client.key,
                peer_pid=os.getpid() + 1,
            )
    finally:
        server.close()


def test_management_server_rejects_wrong_peer_uid(tmp_path: Path) -> None:
    server, _client = _committed_server(tmp_path)

    class _WrongUIDPeer:
        @staticmethod
        def getsockopt(_level: int, _option: int, _size: int) -> bytes:
            return management._PEER_CREDENTIALS.pack(
                os.getpid(), os.geteuid() + 1, os.getegid()
            )

    try:
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="peer credentials differ",
        ):
            server._peer_pid(_WrongUIDPeer())  # type: ignore[arg-type]
    finally:
        server.close()


def test_management_birth_corroboration_allows_only_linux_reparenting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = _owner_birth()
    reparented = dict(sealed)
    reparented["parent_pid"] = sealed["parent_pid"] + 1
    monkeypatch.setattr(
        management,
        "_observed_process_birth",
        lambda _pid: dict(reparented),
    )
    assert management._corroborates_process_birth(sealed) is True

    for field, replacement in (
        ("pid", sealed["pid"] + 1),
        ("start_time_ticks", sealed["start_time_ticks"] + 1),
        ("boot_id", "different-boot"),
        ("argv_sha256", "sha256:" + "f" * 64),
    ):
        changed = dict(reparented)
        changed[field] = replacement
        monkeypatch.setattr(
            management,
            "_observed_process_birth",
            lambda _pid, value=changed: dict(value),
        )
        assert management._corroborates_process_birth(sealed) is False


def test_management_reattaches_after_one_shot_caller_is_reparented(
    tmp_path: Path,
) -> None:
    state_dir = _private_state(tmp_path)
    broker_helper = r'''
import os
import sys
import threading
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime import eaaef_casf_owner_management as management
from ipfs_accelerate_py.agent_supervisor.runtime import eaaef_reconciliation_lifecycle as reconciliation

state_dir = Path(sys.argv[1])
birth = reconciliation.inspect_process_birth(os.getpid())
if birth is None:
    raise SystemExit(71)
exiting = threading.Event()
holder = {}

def request_stop():
    def finish():
        holder["server"].mark_stopped()
        exiting.set()
    threading.Thread(target=finish, daemon=True).start()

server = management.CASFOwnerManagementServer(
    generation_id="eaaef-management-reparent-001",
    binding_cid="sha256:" + "1" * 64,
    snapshot_bindings_cid="sha256:" + "2" * 64,
    state_dir=state_dir,
    owner_process_birth=birth.to_dict(),
    request_stop=request_stop,
    stop_timeout_seconds=10,
)
holder["server"] = server
server.start()
server.mark_committed(
    owner_start_receipt_cid="sha256:" + "a" * 64,
    final_record_cid="sha256:" + "b" * 64,
    commit_receipt_cid="sha256:" + "c" * 64,
)
ready = os.open(
    state_dir / "test-ready",
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o600,
)
os.write(ready, b"ready")
os.close(ready)
if not exiting.wait(30):
    raise SystemExit(72)
server.stop_response_sent.wait(2)
server.close()
'''
    launcher_helper = r'''
import subprocess
import sys
import time
from pathlib import Path

state_dir = Path(sys.argv[1])
child = subprocess.Popen(
    [sys.executable, "-B", "-c", sys.argv[2], str(state_dir)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
    start_new_session=True,
)
deadline = time.monotonic() + 10
while time.monotonic() < deadline and not (state_dir / "test-ready").exists():
    if child.poll() is not None:
        raise SystemExit(73)
    time.sleep(0.01)
if not (state_dir / "test-ready").exists():
    raise SystemExit(74)
print(child.pid, flush=True)
'''
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            launcher_helper,
            str(state_dir),
            broker_helper,
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert launcher.stdout is not None
    raw_pid = launcher.stdout.readline().strip()
    launcher.wait(timeout=15)
    if launcher.returncode != 0:
        assert launcher.stderr is not None
        pytest.fail("one-shot launcher failed: " + launcher.stderr.read())
    broker_pid = int(raw_pid)
    client: management.CASFOwnerManagementClient | None = None
    try:
        client = management.CASFOwnerManagementClient(
            generation_id="eaaef-management-reparent-001",
            binding_cid="sha256:" + "1" * 64,
            snapshot_bindings_cid="sha256:" + "2" * 64,
            state_dir=state_dir,
            timeout_seconds=10,
        )
        sealed_birth = dict(client.owner_process_birth)
        deadline = time.monotonic() + 5
        observed: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            observed = management._observed_process_birth(broker_pid)
            if observed is not None and observed["parent_pid"] != sealed_birth["parent_pid"]:
                break
            time.sleep(0.01)
        assert observed is not None
        assert observed["parent_pid"] != sealed_birth["parent_pid"]
        assert client.status_snapshot()["phase"] == "committed"
        assert client.stop()["committed_owner_stopped"] is True
        assert client.wait_dead(10)
    finally:
        observed = management._observed_process_birth(broker_pid)
        if observed is not None:
            os.kill(broker_pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while (
                time.monotonic() < deadline
                and management._observed_process_birth(broker_pid) is not None
            ):
                time.sleep(0.01)
