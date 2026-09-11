"""SAWM native operational dispatch pause; never a task or closure capability.

Native actors authenticate with their own inherited owner credential plus kernel
peer birth. The public operator may request a pause, but an ACK only retains
that request. Every current lane must subsequently reach its preclaim boundary.
Retained claims may still execute their existing obligations. No lease is
released and no process is signalled by this module.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import os
import re
import socket
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import owner_status_observation as observation

SCHEMA = "ipfs_accelerate_py/sawm-native-dispatch-drain@1"
PROGRAM = "semantic-addressed-world-model-v1"
# At most sixteen bounded 6 KiB observations plus the existing lane roster.
MAX_PACKET = 131072
MAX_LANES = 16
FRESH_SECONDS = 45.0
IO_SECONDS = 1.0


class DispatchObservationUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("native dispatch observation unavailable")


def _raw(value: Any) -> bytes:
    data = observation._encoded(value)
    if len(data) > MAX_PACKET:
        raise DispatchObservationUnavailable()
    return data


def _birth(pid: int) -> dict[str, Any]:
    return observation._birth(pid)


def _parent(pid: int) -> int:
    raw = observation._read_bounded(Path(f"/proc/{pid}/stat"), 8192).decode()
    return int(raw.rsplit(") ", 1)[1].split()[1])


def _command_sha256(pid: int) -> str:
    return hashlib.sha256(
        observation._read_bounded(Path(f"/proc/{pid}/cmdline"), 65536)
    ).hexdigest()


def _same_birth(value: Any) -> bool:
    return (
        type(value) is dict
        and set(value) == {"pid", "start_time_ticks", "boot_id"}
        and type(value.get("pid")) is int
        and value["pid"] > 0
        and _raw(value) == _raw(_birth(value["pid"]))
    )


def _receive(connection: socket.socket) -> dict[str, Any]:
    data, _, flags, _ = connection.recvmsg(MAX_PACKET)
    if not data or flags & socket.MSG_TRUNC:
        raise DispatchObservationUnavailable()
    return observation._decode(data)


def _address(scope: Mapping[str, Any]) -> str:
    return "\0ipfs-native-drain-" + observation._digest(scope)


def _proof(token: str, packet: Mapping[str, Any]) -> str:
    return hmac.new(token.encode(), _raw(packet), hashlib.sha256).hexdigest()


def _master(path: Path) -> dict[str, Any]:
    raw = observation._read_bounded(path, 32)
    if not raw.strip().isdigit():
        raise DispatchObservationUnavailable()
    birth = _birth(int(raw.strip()))
    if observation._read_bounded(path, 32) != raw:
        raise DispatchObservationUnavailable()
    return birth


class DrainState:
    """Retained in the owner across optional listener replacement."""

    def __init__(self, maximum_lanes: int):
        if type(maximum_lanes) is not int or not 1 <= maximum_lanes <= MAX_LANES:
            raise DispatchObservationUnavailable()
        self.maximum_lanes = maximum_lanes
        self.epoch = 0
        self.request_id = ""
        self.master_birth: dict[str, Any] | None = None
        self.master_ack_epoch = -1
        self.master_seen = 0.0
        self.lanes: dict[str, dict[str, Any]] = {}
        self.nonces: dict[str, float] = {}

    def roster(self, master: dict[str, Any], rows: Any) -> None:
        if type(rows) is not list or len(rows) > self.maximum_lanes:
            raise DispatchObservationUnavailable()
        current = {}
        for row in rows:
            if (
                type(row) is not dict
                or set(row) != {"lane", "supervisor_birth"}
                or not isinstance(row["lane"], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", row["lane"])
                or row["lane"] in current
                or not _same_birth(row["supervisor_birth"])
                or _parent(row["supervisor_birth"]["pid"]) != master["pid"]
            ):
                raise DispatchObservationUnavailable()
            previous = self.lanes.get(row["lane"])
            if (
                self.master_birth == master
                and previous is not None
                and previous["supervisor_birth"] == row["supervisor_birth"]
            ):
                current[row["lane"]] = previous
            else:
                current[row["lane"]] = {
                    **row,
                    "daemon_birth": None,
                    "ack_epoch": -1,
                    "last_seen": 0.0,
                    "phase": "unknown",
                    "command_sha256": "",
                    "attempt_observation": None,
                }
        self.lanes = current
        self.master_birth = master
        self.master_ack_epoch = self.epoch if self.request_id else -1
        self.master_seen = time.monotonic()

    def register_daemon(
        self, supervisor: dict[str, Any], daemon: Any, command_sha256: Any
    ) -> None:
        matches = [
            row for row in self.lanes.values() if row["supervisor_birth"] == supervisor
        ]
        if (
            len(matches) != 1
            or not _same_birth(supervisor)
            or not _same_birth(daemon)
            or _parent(daemon["pid"]) != supervisor["pid"]
            or not isinstance(command_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", command_sha256)
            or _command_sha256(daemon["pid"]) != command_sha256
        ):
            raise DispatchObservationUnavailable()
        row = matches[0]
        if row["daemon_birth"] != daemon or row["command_sha256"] != command_sha256:
            row.update(
                daemon_birth=daemon,
                command_sha256=command_sha256,
                ack_epoch=-1,
                last_seen=0.0,
                phase="unknown",
                attempt_observation=None,
            )

    def boundary(self, daemon: dict[str, Any], phase: str) -> None:
        if phase not in {"preclaim", "retained_work", "reconciling"} or not _same_birth(
            daemon
        ):
            raise DispatchObservationUnavailable()
        matches = [
            row
            for row in self.lanes.values()
            if row["supervisor_birth"]["pid"] == _parent(daemon["pid"])
            and _same_birth(row["supervisor_birth"])
        ]
        if len(matches) != 1 or not _same_birth(self.master_birth):
            raise DispatchObservationUnavailable()
        row = matches[0]
        if (
            row["daemon_birth"] != daemon
            or _command_sha256(daemon["pid"]) != row["command_sha256"]
        ):
            raise DispatchObservationUnavailable()
        if _parent(row["supervisor_birth"]["pid"]) != self.master_birth["pid"]:
            raise DispatchObservationUnavailable()
        row["phase"] = phase
        row["attempt_observation"] = None
        row["last_seen"] = time.monotonic()
        row["ack_epoch"] = self.epoch if self.request_id and phase == "preclaim" else -1

    def projection(self) -> dict[str, Any]:
        now = time.monotonic()
        rows = []
        for name, row in sorted(self.lanes.items()):
            try:
                fresh = (
                    now - row["last_seen"] <= FRESH_SECONDS
                    and _same_birth(row["supervisor_birth"])
                    and _same_birth(row["daemon_birth"])
                    and _command_sha256(row["daemon_birth"]["pid"])
                    == row["command_sha256"]
                    and _parent(row["daemon_birth"]["pid"])
                    == row["supervisor_birth"]["pid"]
                    and _parent(row["supervisor_birth"]["pid"])
                    == self.master_birth["pid"]
                )
            except (
                OSError,
                ValueError,
                TypeError,
                observation.OwnerObservationUnavailable,
            ):
                fresh = False
            rows.append(
                {
                    "lane": name,
                    "supervisor_birth": row["supervisor_birth"],
                    "daemon_birth": row["daemon_birth"],
                    "fresh": bool(fresh),
                    "pause_epoch_acknowledged": bool(
                        fresh and self.request_id and row["ack_epoch"] == self.epoch
                    ),
                    "retained_work_reported": row["phase"] == "retained_work",
                    "last_observed_phase": row["phase"],
                    "attempt_observation": copy.deepcopy(row.get("attempt_observation"))
                    if fresh and row["phase"] == "retained_work" else None,
                }
            )
        try:
            master_fresh = (
                _same_birth(self.master_birth)
                and now - self.master_seen <= FRESH_SECONDS
                and self.master_ack_epoch == self.epoch
            )
        except (
            OSError,
            ValueError,
            TypeError,
            observation.OwnerObservationUnavailable,
        ):
            master_fresh = False
        paused = bool(
            self.request_id
            and master_fresh
            and len(rows) == self.maximum_lanes
            and all(row["pause_epoch_acknowledged"] for row in rows)
        )
        return {
            "request_id": self.request_id,
            "drain_epoch": self.epoch,
            "drain_requested": bool(self.request_id),
            "dispatch_pause_observed": paused,
            "master_birth": self.master_birth,
            "master_fresh": bool(master_fresh),
            "lanes": rows,
            "callback_custody_known": False,
            "terminal_custody": "unknown",
            "task_authority": False,
            "completion_authority": False,
            "source_transition_authority": False,
            "signals_sent": False,
        }


class NativeDrainService:
    def __init__(self, observer: Any, configuration: Mapping[str, Any]):
        self.observer = observer
        self.scope = observer.scope
        self.server = observer.server
        # The canonical path comes from the retained admitted owner config.
        root = Path(self.server.config.database_path).resolve()
        relative = Path(str(configuration["quack_owner"]["database_path"]))
        for _ in relative.parts:
            root = root.parent
        self.master_path = (
            root
            / str(configuration["runtime_paths"]["state"])
            / "configured-board-master.pid"
        )
        state = getattr(self.server, "_native_dispatch_drain_state", None)
        if state is None:
            state = DrainState(configuration["max_lanes"])
            self.server._native_dispatch_drain_state = state
        self.state = state
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self.listener.bind(_address(self.scope))
            self.listener.listen(8)
            self.listener.setblocking(False)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            self.listener.close()
        except OSError:
            pass

    def handle(self, packet: dict[str, Any], pid: int, uid: int) -> dict[str, Any]:
        observation._custody(self.scope, self.observer.database)
        if (
            self.server.lifecycle.value != "ready"
            or uid != self.scope["uid"]
            or set(packet)
            != {
                "schema",
                "scope_cid",
                "nonce",
                "birth",
                "operation",
                "body",
                "sent_at",
                "proof",
            }
            or packet["schema"] != SCHEMA
            or packet["scope_cid"] != observation._digest(self.scope)
            or not isinstance(packet["nonce"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", packet["nonce"])
            or _raw(packet["birth"]) != _raw(_birth(pid))
            or type(packet["body"]) is not dict
            or type(packet["sent_at"]) not in (float, int)
            or not 0 <= time.time() - packet["sent_at"] <= IO_SECONDS + 2
        ):
            raise DispatchObservationUnavailable()
        if observation._encoded(
            {k: self.server.identity.to_dict()[k] for k in observation._OWNER_FIELDS}
        ) != observation._encoded(self.scope["owner_identity"]):
            raise DispatchObservationUnavailable()
        # Stage every operational update. A failed final identity check must
        # never clear a retained pause or publish an unverified roster.
        candidate = copy.deepcopy(self.state)
        nonce_key = observation._digest(
            {"birth": packet["birth"], "nonce": packet["nonce"]}
        )
        now = time.monotonic()
        candidate.nonces = {
            key: stamp
            for key, stamp in candidate.nonces.items()
            if now - stamp <= IO_SECONDS + 2
        }
        if nonce_key in candidate.nonces or len(candidate.nonces) >= 256:
            raise DispatchObservationUnavailable()
        candidate.nonces[nonce_key] = now
        operation, body = packet["operation"], packet["body"]
        master = _master(self.master_path)
        if operation in {
            "coordinator_boundary",
            "supervisor_boundary",
            "lane_boundary",
            "custody_boundary",
        }:
            unsigned = {k: v for k, v in packet.items() if k != "proof"}
            token = self.server._vault.resolve(self.server.secret_handle)
            if not isinstance(packet["proof"], str) or not hmac.compare_digest(
                packet["proof"], _proof(token, unsigned)
            ):
                raise DispatchObservationUnavailable()
            if operation == "coordinator_boundary":
                if set(body) != {"lanes"} or packet["birth"] != master:
                    raise DispatchObservationUnavailable()
                candidate.roster(master, body["lanes"])
            elif operation == "supervisor_boundary":
                if (
                    set(body) != {"daemon_birth", "command_sha256"}
                    or candidate.master_birth != master
                ):
                    raise DispatchObservationUnavailable()
                candidate.register_daemon(
                    packet["birth"], body["daemon_birth"], body["command_sha256"]
                )
            elif operation == "custody_boundary":
                from .attempt_custody_observation import validate_observation

                if set(body) != {"observation"} or candidate.master_birth != master:
                    raise DispatchObservationUnavailable()
                observed = (
                    validate_observation(body["observation"])
                    if body["observation"] is not None else None
                )
                candidate.boundary(packet["birth"], "retained_work")
                matching = [row for row in candidate.lanes.values()
                            if row["daemon_birth"] == packet["birth"]]
                if len(matching) != 1:
                    raise DispatchObservationUnavailable()
                matching[0]["attempt_observation"] = observed
            else:
                if set(body) != {"phase"} or candidate.master_birth != master:
                    raise DispatchObservationUnavailable()
                candidate.boundary(packet["birth"], body["phase"])
        elif operation in {"request", "status", "release"}:
            if (
                packet["proof"] != ""
                or set(body) != {"master_birth", "request_id"}
                or _raw(body["master_birth"]) != _raw(master)
            ):
                raise DispatchObservationUnavailable()
            if operation == "request":
                if body["request_id"] != "":
                    raise DispatchObservationUnavailable()
                if not candidate.request_id:
                    candidate.epoch += 1
                    candidate.request_id = uuid.uuid4().hex
            elif operation == "release":
                if (
                    not candidate.request_id
                    or body["request_id"] != candidate.request_id
                ):
                    raise DispatchObservationUnavailable()
                candidate.epoch += 1
                candidate.request_id = ""
            elif body["request_id"] != "":
                raise DispatchObservationUnavailable()
        else:
            raise DispatchObservationUnavailable()
        response = {
            "schema": SCHEMA,
            "scope_cid": packet["scope_cid"],
            "nonce": packet["nonce"],
            "request_received": True,
            "new_dispatch_permitted": not bool(candidate.request_id),
            "reason": "native_dispatch_pause_requested"
            if candidate.request_id
            else "native_dispatch_not_paused",
            "observed_at": time.time(),
            "state": candidate.projection(),
        }
        _raw(response)
        observation._custody(self.scope, self.observer.database)
        if observation._encoded(
            {k: self.server.identity.to_dict()[k] for k in observation._OWNER_FIELDS}
        ) != observation._encoded(self.scope["owner_identity"]):
            raise DispatchObservationUnavailable()
        if _birth(pid) != packet["birth"] or _master(self.master_path) != master:
            raise DispatchObservationUnavailable()
        self.state.__dict__.update(candidate.__dict__)
        return response

    def poll(self) -> None:
        try:
            connection, _ = self.listener.accept()
            with connection:
                connection.settimeout(0.02)
                pid, uid = observation._peer(connection)
                try:
                    response = self.handle(_receive(connection), pid, uid)
                except Exception:  # noqa: BLE001 - optional observation must not retire native work
                    response = {
                        "schema": SCHEMA,
                        "request_received": False,
                        "new_dispatch_permitted": False,
                        "reason": "native_dispatch_observation_unavailable",
                    }
                connection.sendall(_raw(response))
        except Exception:  # noqa: BLE001 - optional observation must not retire native work
            return


class NativeDispatchClient:
    def __init__(
        self,
        *,
        database: Path,
        state_dir: Path,
        configuration: Mapping[str, Any],
        source_head: str,
        source_tree: str,
        token: str = "",
    ):
        self.database, self.state_dir = database, state_dir
        self.configuration = dict(configuration)
        self.source_head, self.source_tree = source_head, source_tree
        self._token = token
        self._owner_scope: dict[str, Any] | None = None

    def _scope(self) -> dict[str, Any]:
        scope = observation._validated_scope(
            observation._decode(
                observation._read_bounded(self.state_dir / observation.DESCRIPTOR, 8192)
            )
        )
        owner = self.configuration["quack_owner"]
        expected_owner = {
            "store_id": str(owner["store_id"]),
            "repository_id": str(owner["repository_id"]),
            "generation": int(
                self.configuration["database_program"]["store_generation"]
            ),
        }
        if (
            scope["program_id"] != PROGRAM
            or self.configuration["board_namespace"] != PROGRAM
            or scope["configuration_cid"] != observation._digest(self.configuration)
            or scope["source_head"] != self.source_head
            or scope["source_tree"] != self.source_tree
            or _raw({k: scope["owner_identity"].get(k) for k in expected_owner})
            != _raw(expected_owner)
            or self._owner_scope is not None
            and _raw(scope) != _raw(self._owner_scope)
        ):
            raise DispatchObservationUnavailable()
        observation._custody(scope, self.database)
        self._owner_scope = scope
        return scope

    def exchange(self, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            scope = self._scope()
            packet = {
                "schema": SCHEMA,
                "scope_cid": observation._digest(scope),
                "nonce": uuid.uuid4().hex,
                "birth": _birth(os.getpid()),
                "operation": operation,
                "body": body,
                "sent_at": time.time(),
            }
            packet["proof"] = (
                _proof(self._token, packet) if operation.endswith("_boundary") else ""
            )
            started = time.monotonic()
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
                connection.settimeout(IO_SECONDS)
                connection.connect(_address(scope))
                peer = (scope["owner_birth"]["pid"], scope["uid"])
                if observation._peer(connection) != peer:
                    raise DispatchObservationUnavailable()
                connection.sendall(_raw(packet))
                reply = _receive(connection)
                if observation._peer(connection) != peer:
                    raise DispatchObservationUnavailable()
            observation._custody(scope, self.database)
            if (
                set(reply)
                != {
                    "schema",
                    "scope_cid",
                    "nonce",
                    "request_received",
                    "new_dispatch_permitted",
                    "reason",
                    "observed_at",
                    "state",
                }
                or reply["schema"] != SCHEMA
                or reply["scope_cid"] != packet["scope_cid"]
                or reply["nonce"] != packet["nonce"]
                or reply["request_received"] is not True
                or type(reply["new_dispatch_permitted"]) is not bool
                or type(reply["observed_at"]) not in (float, int)
                or not 0 <= time.time() - reply["observed_at"] <= IO_SECONDS + 1
                or time.monotonic() - started > IO_SECONDS + 1
                or type(reply["state"]) is not dict
                or any(
                    reply["state"].get(k) is not False
                    for k in [
                        "callback_custody_known",
                        "task_authority",
                        "completion_authority",
                        "source_transition_authority",
                        "signals_sent",
                    ]
                )
                or reply["state"].get("terminal_custody") != "unknown"
            ):
                raise DispatchObservationUnavailable()
            from .attempt_custody_observation import validate_observation

            for row in reply["state"].get("lanes", []):
                if row.get("attempt_observation") is not None:
                    if row.get("fresh") is not True or row.get("retained_work_reported") is not True:
                        raise DispatchObservationUnavailable()
                    validate_observation(row["attempt_observation"])
            return reply
        except Exception:  # noqa: BLE001 - optional observation must not retire native work
            raise DispatchObservationUnavailable() from None

    def before_claim(self) -> dict[str, Any]:
        try:
            return self.exchange("lane_boundary", {"phase": "preclaim"})
        except DispatchObservationUnavailable:
            return {
                "new_dispatch_permitted": False,
                "reason": "native_dispatch_observation_unavailable",
            }

    def reconciliation_started(self) -> None:
        try:
            self.exchange("lane_boundary", {"phase": "reconciling"})
        except DispatchObservationUnavailable:
            pass

    def register_daemon(self, birth: Mapping[str, Any], command_sha256: str) -> None:
        try:
            self.exchange(
                "supervisor_boundary",
                {"daemon_birth": dict(birth), "command_sha256": command_sha256},
            )
        except DispatchObservationUnavailable:
            pass

    def retained_work(self, _attempt: Any) -> None:
        try:
            self.exchange("lane_boundary", {"phase": "retained_work"})
        except DispatchObservationUnavailable:
            pass  # Missing diagnostics never cancel an already admitted obligation.

    def retained_attempt_custody(self, daemon: Any, attempt: Any) -> None:
        """Relay only this admitted lane's typed read, without new credentials.

        An unavailable read clears the prior observation and still reports
        retained work. Neither the relay nor its control owner settles it.
        """
        from .attempt_custody_observation import (
            AttemptObservationUnavailable, observe_attempt,
        )

        try:
            observed = observe_attempt(daemon, attempt)
        except AttemptObservationUnavailable:
            observed = None
        try:
            self.exchange("custody_boundary", {"observation": observed})
        except DispatchObservationUnavailable:
            pass

    def coordinator_boundary(self, processes: Mapping[str, Any]) -> dict[str, Any]:
        try:
            rows = [
                {"lane": name, "supervisor_birth": _birth(process.pid)}
                for name, process in sorted(processes.items())
                if process.poll() is None
            ]
            return self.exchange("coordinator_boundary", {"lanes": rows})
        except Exception:  # noqa: BLE001 - optional observation must not retire native work
            return {
                "new_dispatch_permitted": False,
                "reason": "native_dispatch_observation_unavailable",
            }


def from_native_admission(
    *, admission: Any, repo_root: Path
) -> NativeDispatchClient | None:
    """Called only after the existing sealed FD/native bootstrap verified admission.

    Re-read the exact admitted config artifact, rather than trusting an argv
    enable flag or the source context in the mutable endpoint descriptor.
    """
    if admission is None:
        return None
    from .configured_board_live_capsule import (
        ConfiguredBoardLiveCapsuleAdmission,
        _cid,
        _protected_control_json,
        parse_configured_board_live_capsule_admission,
    )

    if type(admission) is not ConfiguredBoardLiveCapsuleAdmission:
        raise DispatchObservationUnavailable()
    if admission.board_namespace != PROGRAM:
        return None
    admission = parse_configured_board_live_capsule_admission(admission.as_dict())
    configuration, raw_configuration = _protected_control_json(
        repo_root, admission, admission.config_path, maximum=4 * 1024 * 1024
    )
    if (
        admission.configuration_root
        != _cid({"bytes_sha256": hashlib.sha256(raw_configuration).hexdigest()})
        or int(configuration["database_program"]["store_generation"])
        != admission.database_authority["store_generation"]
        or configuration["quack_owner"]["store_id"]
        != admission.database_authority["store_id"]
    ):
        raise DispatchObservationUnavailable()
    owner = configuration["quack_owner"]
    handle = str(admission.database_authority["endpoint_secret_handle"])
    if (
        configuration["board_namespace"] != PROGRAM
        or not handle.startswith("env://")
        or owner["secret_handle"] != handle
    ):
        raise DispatchObservationUnavailable()
    token = os.environ.get(handle.removeprefix("env://"), "")
    if not token:
        raise DispatchObservationUnavailable()
    return NativeDispatchClient(
        database=repo_root / owner["database_path"],
        state_dir=repo_root / owner["state_dir"],
        configuration=configuration,
        source_head=admission.source_head,
        source_tree=admission.source_tree,
        token=token,
    )
