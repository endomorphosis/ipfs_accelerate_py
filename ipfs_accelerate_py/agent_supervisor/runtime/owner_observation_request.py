"""Request a new retained-owner sample without granting task or query authority.

Linux peer credentials bind a request to an exact live process. The receiver is
created only inside an already admitted owner and holds a frozen launch scope.
Replies acknowledge a coalesced request, never an observation or acceptance.
The native sampler keeps its existing cadence, connection and failure budgets.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import struct
import threading
import uuid
from collections.abc import Mapping
from typing import Any

from ..merge.worktree_lifecycle import read_process_birth

SCHEMA = "ipfs_accelerate_py/agent-supervisor/owner-observation-request@1"
MAX_PACKET = 16384
_SCOPE_FIELDS = {"schema", "program_id", "owner_identity", "process_birth",
                 "source_head", "source_tree", "launch_admission_id"}


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _birth(pid: int) -> dict[str, Any]:
    birth = read_process_birth(pid)
    if birth is None or birth.start_time_ticks <= 0 or not birth.boot_id:
        raise ValueError("process birth unavailable")
    return birth.to_dict()


def observation_scope(*, program_id: str, owner_identity: Mapping[str, Any],
                      source_head: str, source_tree: str,
                      launch_admission_id: str) -> dict[str, Any]:
    return _validate_scope({"schema": SCHEMA, "program_id": program_id,
        "owner_identity": dict(owner_identity), "process_birth": _birth(os.getpid()),
        "source_head": source_head, "source_tree": source_tree,
        "launch_admission_id": launch_admission_id})


def _validate_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    if set(scope) != _SCOPE_FIELDS or scope.get("schema") != SCHEMA:
        raise ValueError("observation scope schema differs")
    for key in ("program_id", "launch_admission_id"):
        if not isinstance(scope[key], str) or not scope[key] or len(scope[key]) > 256:
            raise ValueError("observation scope identity invalid")
    for key in ("source_head", "source_tree"):
        if not isinstance(scope[key], str) or not re.fullmatch(r"[0-9a-f]{40}", scope[key]):
            raise ValueError("observation source identity invalid")
    birth = scope["process_birth"]
    if (not isinstance(birth, dict) or set(birth) !=
            {"pid", "start_time_ticks", "boot_id", "parent_pid"}
            or any(type(birth[key]) is not int or birth[key] <= 0
                   for key in ("pid", "start_time_ticks", "parent_pid"))
            or not isinstance(birth["boot_id"], str) or not birth["boot_id"]
            or not isinstance(scope["owner_identity"], dict) or not scope["owner_identity"]):
        raise ValueError("observation owner identity invalid")
    raw = _encoded(scope)
    if len(raw) > MAX_PACKET // 2:
        raise ValueError("observation scope exceeds bound")
    return json.loads(raw)


def _address(scope: Mapping[str, Any]) -> str:
    # Abstract socket: no mutable filesystem endpoint and no stale socket unlink.
    return "\0ipfs-observe-" + hashlib.sha256(_encoded(scope)).hexdigest()


def _peer(connection: socket.socket) -> tuple[int, int]:
    pid, uid, _ = struct.unpack("3i", connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    return pid, uid


def _receive(connection: socket.socket) -> dict[str, Any]:
    raw, _, flags, _ = connection.recvmsg(MAX_PACKET)
    if flags & socket.MSG_TRUNC or not raw:
        raise ValueError("observation packet exceeds bound")
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("observation packet repeats a field")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=object_pairs)
    if not isinstance(result, dict):
        raise ValueError("observation packet invalid")
    return result


class OwnerObservationRequests:
    """One request slot; call poll only from the retained native owner loop."""

    def __init__(self, scope: Mapping[str, Any]):
        self.scope = _validate_scope(scope)
        if _encoded(self.scope["process_birth"]) != _encoded(_birth(os.getpid())):
            raise ValueError("observation listener must be the admitted owner process")
        self.pending = threading.Event()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self.listener.bind(_address(self.scope))
            self.listener.listen(4)
            self.listener.setblocking(False)
        except BaseException:
            self.listener.close()
            raise

    def sample_started(self) -> None:
        # Only the existing sampler consumes this request. It does not reset
        # its previous sample, progress timestamp, or outage budget.
        self.pending.clear()

    def poll(self, *, observer_available: bool) -> None:
        # Observation hints are optional. A transient descriptor shortage,
        # aborted peer or socket cleanup error cannot retire the native owner.
        try:
            self._poll_one(observer_available=observer_available)
        except OSError:
            return

    def _poll_one(self, *, observer_available: bool) -> None:
        try:
            connection, _ = self.listener.accept()
        except BlockingIOError:
            return
        with connection:
            connection.settimeout(0.02)
            reason = "request_rejected"
            nonce = None
            try:
                peer_pid, peer_uid = _peer(connection)
                packet = _receive(connection)
                if (peer_uid != os.getuid() or set(packet) !=
                        {"schema", "scope", "requester_birth", "nonce"}
                        or packet["schema"] != SCHEMA
                        or _encoded(packet["scope"]) != _encoded(self.scope)
                        or _encoded(packet["requester_birth"]) != _encoded(_birth(peer_pid))
                        or not isinstance(packet["nonce"], str)
                        or not re.fullmatch(r"[0-9a-f]{32}", packet["nonce"])
                        or _encoded(_birth(os.getpid())) != _encoded(self.scope["process_birth"])):
                    raise ValueError("observation request binding differs")
                nonce = packet["nonce"]
                if observer_available is not True:
                    reason = "observer_unavailable"
                else:
                    self.pending.set()
                    reason = "queued_for_next_sample"
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                pass
            response = {"schema": SCHEMA, "request_nonce": nonce,
                "accepted": reason == "queued_for_next_sample", "reason": reason,
                "authenticated_observation": False, "completion_authority": False,
                "mutation_authority": False}
            try:
                connection.sendall(_encoded(response))
            except OSError:
                pass

    def close(self) -> None:
        # Listener cleanup must not bypass the native owner's scheduler,
        # monitor and state-server cleanup if the descriptor is already gone.
        try:
            self.listener.close()
        except OSError:
            pass


def request_observation(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Return only a request acknowledgment; native status admission is separate."""
    expected = _validate_scope(scope)
    nonce = uuid.uuid4().hex
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(1.0)
        connection.connect(_address(expected))
        peer_pid, peer_uid = _peer(connection)
        if (peer_uid != os.getuid()
                or _encoded(_birth(peer_pid)) != _encoded(expected["process_birth"])):
            raise ValueError("observation receiver birth differs")
        connection.sendall(_encoded({"schema": SCHEMA, "scope": expected,
            "requester_birth": _birth(os.getpid()), "nonce": nonce}))
        response = _receive(connection)
    if (set(response) != {"schema", "request_nonce", "accepted", "reason",
            "authenticated_observation", "completion_authority", "mutation_authority"}
            or response.get("schema") != SCHEMA
            or response.get("request_nonce") != nonce
            or type(response.get("accepted")) is not bool
            or response.get("reason") not in {"queued_for_next_sample", "observer_unavailable"}
            or response["accepted"] is not (response["reason"] == "queued_for_next_sample")
            or any(response.get(key) is not False for key in
                   ("authenticated_observation", "completion_authority", "mutation_authority"))):
        raise ValueError("observation acknowledgment invalid")
    return response
