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
    *,
    authentication_timeout_seconds: float = 1.0,
    stop_delay_seconds: float = 0.0,
    stop_timeout_seconds: float = 5.0,
    client_timeout_seconds: float = 5.0,
) -> tuple[
    management.CASFOwnerManagementServer,
    management.CASFOwnerManagementClient,
]:
    state_dir = _private_state(tmp_path)
    holder: dict[str, management.CASFOwnerManagementServer] = {}

    def _request_stop() -> None:
        def _complete() -> None:
            if stop_delay_seconds:
                time.sleep(stop_delay_seconds)
            holder["server"].mark_stopped()

        threading.Thread(target=_complete, daemon=True).start()

    server = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=_request_stop,
        stop_timeout_seconds=stop_timeout_seconds,
        authentication_timeout_seconds=authentication_timeout_seconds,
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
        timeout_seconds=client_timeout_seconds,
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
            binding_cid=_BINDING_CID,
            snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
            capsule_cid=client.capsule["capsule_cid"],
            key=client.key,
        )
        retained = management._validate_stop_result(
            management._read_private(
                client.state_dir, management.MANAGEMENT_STOP_RESULT_NAME
            ),
            generation_id=_GENERATION_ID,
            binding_cid=_BINDING_CID,
            snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
            capsule_cid=client.capsule["capsule_cid"],
            key=client.key,
        )
        assert intent["stop_request_cid"] == retained["stop_request_cid"]
        assert intent["stop_request_id"] == retained["stop_request_id"]
        assert intent["capsule_cid"] == client.capsule["capsule_cid"]
        assert retained["stop_intent_cid"] == intent["intent_cid"]
        tampered = dict(retained)
        tampered["final_record_cid"] = _CID_A
        tampered_body = dict(tampered)
        tampered_body.pop("result_mac")
        tampered_body.pop("result_cid")
        tampered["result_cid"] = management._cid(tampered_body)
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="stop result differs",
        ):
            management._validate_stop_result(
                management._canonical_bytes(tampered, noun="tampered stop result"),
                generation_id=_GENERATION_ID,
                binding_cid=_BINDING_CID,
                snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
                capsule_cid=client.capsule["capsule_cid"],
                key=client.key,
            )
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


def test_management_stop_gets_fresh_adoption_window_after_transport_timeout(
    tmp_path: Path,
) -> None:
    server, client = _committed_server(
        tmp_path,
        stop_delay_seconds=0.15,
        stop_timeout_seconds=1.0,
        client_timeout_seconds=0.1,
    )
    try:
        started = time.monotonic()
        result = client.stop()
        elapsed = time.monotonic() - started
        assert elapsed >= 0.1
        assert elapsed < 0.4
        assert result["committed_owner_stopped"] is True
        assert result["exclusive_owner_lease_released"] is True
    finally:
        server.close()


def test_management_stop_adopts_result_after_authenticated_unavailable(
    tmp_path: Path,
) -> None:
    server, client = _committed_server(
        tmp_path,
        stop_delay_seconds=0.2,
        stop_timeout_seconds=0.1,
        client_timeout_seconds=0.5,
    )
    try:
        started = time.monotonic()
        result = client.stop()
        elapsed = time.monotonic() - started
        assert elapsed >= 0.1
        assert elapsed < 0.7
        assert result["committed_owner_stopped"] is True
        assert result["exclusive_owner_lease_released"] is True
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


def test_rejected_provisional_stop_nonce_cannot_be_replayed_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(management, "_MAX_STATUS_REQUEST_NONCES", 1)
    state_dir = _private_state(tmp_path)
    holder: dict[str, management.CASFOwnerManagementServer] = {}

    def _request_stop() -> None:
        threading.Thread(
            target=holder["server"].mark_stopped,
            daemon=True,
        ).start()

    server = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=_request_stop,
        stop_timeout_seconds=1,
    )
    holder["server"] = server
    server.start()
    client = management.CASFOwnerManagementClient(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        timeout_seconds=1,
    )
    rejected = management._build_request(
        generation_id=_GENERATION_ID,
        capsule_cid=client.capsule["capsule_cid"],
        key=client.key,
        operation="stop",
        arguments={
            "stop_request_id": "1" * 64,
            "owner_start_receipt_cid": "sha256:" + "7" * 64,
            "final_record_cid": "sha256:" + "8" * 64,
            "commit_receipt_cid": "sha256:" + "9" * 64,
        },
    )
    rejected_raw = management._canonical_bytes(
        rejected,
        noun="provisional stop replay",
    )
    try:
        first = client._round_trip(rejected)
        assert first["ok"] is False
        assert first["error_code"] == "owner_not_committed"
        assert client.status_snapshot()["phase"] == "provisional"
        assert rejected["request_nonce"] not in server._status_request_nonces
        server.mark_committed(
            owner_start_receipt_cid=_CID_A,
            final_record_cid=_CID_B,
            commit_receipt_cid=_CID_C,
        )
        assert (
            management._canonical_bytes(rejected, noun="committed stop replay")
            == rejected_raw
        )
        replay = client._round_trip(rejected)
        assert replay["ok"] is False
        assert replay["error_code"] == "stop_request_diverged"
        assert not server.stop_requested.is_set()
        assert not (state_dir / management.MANAGEMENT_STOP_INTENT_NAME).exists()
        assert client.stop()["committed_owner_stopped"] is True
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


def test_stalled_drip_frame_cannot_block_authenticated_status_or_stop(
    tmp_path: Path,
) -> None:
    server, client = _committed_server(
        tmp_path,
        authentication_timeout_seconds=0.5,
    )
    stalled = client._connect()
    stopped = threading.Event()
    drip_errors: list[BaseException] = []

    def _drip() -> None:
        try:
            while not stopped.wait(0.04):
                stalled.sendall(b" ")
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            drip_errors.append(exc)

    try:
        stalled.sendall(management._FRAME_HEADER.pack(1024) + b"{")
        dripper = threading.Thread(target=_drip, daemon=True)
        dripper.start()
        time.sleep(0.03)
        with server._connection_gate:
            assert 1 <= len(server._connection_threads) <= management._MAX_CONNECTION_WORKERS
        started = time.monotonic()
        assert client.status_snapshot()["phase"] == "committed"
        assert client.stop()["committed_owner_stopped"] is True
        assert time.monotonic() - started < 0.3
        assert dripper.is_alive()
        assert not drip_errors
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with server._connection_gate:
                if not server._connections:
                    break
            time.sleep(0.01)
        with server._connection_gate:
            assert not server._connections
            assert not server._connection_threads
        deadline = time.monotonic() + 0.3
        while not drip_errors and dripper.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert drip_errors
    finally:
        stopped.set()
        try:
            stalled.close()
        except OSError:
            pass
        if "dripper" in locals():
            dripper.join(timeout=1)
            assert not dripper.is_alive()
        server.close()
        with server._connection_gate:
            assert not server._connections
            assert not server._connection_threads
    assert not drip_errors or all(
        isinstance(exc, (BrokenPipeError, ConnectionError, OSError))
        for exc in drip_errors
    )


def test_status_nonce_window_is_bounded_and_reserves_first_exact_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(management, "_MAX_STATUS_REQUEST_NONCES", 2)
    server, client = _committed_server(tmp_path)
    try:
        statuses = [
            management._build_request(
                generation_id=_GENERATION_ID,
                capsule_cid=client.capsule["capsule_cid"],
                key=client.key,
                operation="status.snapshot",
                arguments={},
            )
            for _index in range(3)
        ]
        assert client._round_trip(statuses[0])["ok"] is True
        assert client._round_trip(statuses[1])["ok"] is True
        assert client._round_trip(statuses[2])["ok"] is True
        assert len(server._status_request_nonces) == 2
        assert statuses[0]["request_nonce"] not in server._status_request_nonces
        replay = client._round_trip(statuses[1])
        assert replay["ok"] is False
        assert replay["error_code"] == "request_nonce_replayed"
        assert client._round_trip(statuses[0])["ok"] is True
        assert len(server._status_request_nonces) == 2
        assert server._stop_request_nonce is None

        result = client.stop()
        assert result["committed_owner_stopped"] is True
        assert server._stop_request_nonce is not None
        assert len(server._status_request_nonces) == 2
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


@pytest.mark.parametrize(
    ("artifact_name", "payload"),
    [
        (management.MANAGEMENT_KEY_NAME, b"k" * 32),
        (management.MANAGEMENT_CAPSULE_NAME, b"{}"),
    ],
)
def test_partial_management_artifact_quarantines_generation(
    tmp_path: Path,
    artifact_name: str,
    payload: bytes,
) -> None:
    state_dir = _private_state(tmp_path)
    artifact = state_dir / artifact_name
    artifact.write_bytes(payload)
    artifact.chmod(0o600)
    server = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=lambda: pytest.fail("partial generation requested stop"),
        stop_timeout_seconds=1,
    )
    with pytest.raises(
        management.EAAEFCASFOwnerManagementError,
        match="already exists",
    ):
        server.start()
    assert server.key == b""
    assert server.capsule == {}
    assert server.listener is None
    assert server.thread is None

    retry = management.CASFOwnerManagementServer(
        generation_id=_GENERATION_ID,
        binding_cid=_BINDING_CID,
        snapshot_bindings_cid=_SNAPSHOT_BINDINGS_CID,
        state_dir=state_dir,
        owner_process_birth=_owner_birth(),
        request_stop=lambda: pytest.fail("quarantined generation requested stop"),
        stop_timeout_seconds=1,
    )
    with pytest.raises(
        management.EAAEFCASFOwnerManagementError,
        match="already exists",
    ):
        retry.start()
    assert retry.key == b""
    assert retry.listener is None


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
        result = client.stop()
        assert result["committed_owner_stopped"] is True
        assert client.wait_dead(10)
        adopted = management.CASFOwnerManagementClient.adopt_completed_stop(
            generation_id="eaaef-management-reparent-001",
            binding_cid="sha256:" + "1" * 64,
            snapshot_bindings_cid="sha256:" + "2" * 64,
            state_dir=state_dir,
        )
        assert adopted == result
        with pytest.raises(
            management.EAAEFCASFOwnerManagementError,
            match="stale or divergent",
        ):
            management.CASFOwnerManagementClient(
                generation_id="eaaef-management-reparent-001",
                binding_cid="sha256:" + "1" * 64,
                snapshot_bindings_cid="sha256:" + "2" * 64,
                state_dir=state_dir,
                timeout_seconds=1,
            )
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
