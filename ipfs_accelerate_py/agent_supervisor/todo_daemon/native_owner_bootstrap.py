"""Native CLI adapter for the existing private, PID-bound owner bootstrap."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time

from ..merge.worktree_lifecycle import read_process_birth
from ..merge.database_worktree_registry import process_birth_id

from ..task_sources.state_owner_bootstrap import request_state_owner_bootstrap
from ..task_sources.quack_state_client import QuackStateClient
from ..task_sources.typed_database_task_source import TypedDatabaseTaskSource

STARTUP_WAIT_ENV = "IPFS_ACCELERATE_STATE_OWNER_BOOTSTRAP_TIMEOUT_MS"


def _startup_wait_seconds():
    value = os.environ.get(STARTUP_WAIT_ENV)
    if value is None:
        return 30.0
    if (not re.fullmatch(r"[0-9]{1,6}", value)
            or not 2_000 <= int(value) <= 120_000):
        raise ValueError("native bootstrap wait must be in 2000..120000 milliseconds")
    return int(value) / 1000


class NativeOwnerHeartbeat:
    """Live owner read evidence, independent of completed daemon passes.

    Every sequence requires a successful native typed owner generation read.
    This observation grants no task progress, completion or dispatch authority.
    A separate native client avoids blocking the main client's request lock.
    """

    def __init__(self, *, credentials, state_dir: Path, state_prefix: str,
                 interval_seconds: float = 0.25):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", state_prefix):
            raise ValueError("invalid live owner heartbeat prefix")
        if not 0.1 <= interval_seconds <= 1.0:
            raise ValueError("live owner heartbeat interval must be bounded")
        self.credentials = credentials
        self.path = Path(state_dir) / f"{state_prefix}_native_owner_heartbeat.json"
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread = None
        self.sequence = 0
        self.last_error = ""
        self.client = QuackStateClient(
            owner_id=credentials.client_id, store_id=credentials.store_id,
            process_birth_id=credentials.process_birth_id,
        )

    def start(self):
        self.client.attach(self.credentials.endpoint, server_id=self.credentials.server_id)
        self.source = TypedDatabaseTaskSource(
            self.client, execution_route_policy=self.credentials.execution_route_policy,
        )
        self._publish()
        self.thread = threading.Thread(target=self._run, name="native-owner-heartbeat", daemon=True)
        self.thread.start()
        return self

    def _publish(self):
        binding = self.source.require_quack_authority_binding(
            expected_endpoint=self.credentials.endpoint,
            expected_process_instance_id=self.credentials.process_birth_id,
            bootstrap_credentials=self.credentials,
        )
        birth = read_process_birth(os.getpid())
        if birth is None or process_birth_id(birth) != self.credentials.process_birth_id:
            raise ValueError("live owner heartbeat process birth changed")
        self.sequence += 1
        payload = {
            "schema": "native-typed-owner-live-observation@1",
            "observed_at_ms": time.time_ns() // 1_000_000, "sequence": self.sequence,
            "process_birth": birth.to_dict(), "process_birth_id": process_birth_id(birth),
            **{key: binding[key] for key in ("store_id", "server_id", "generation", "fence_epoch", "route_policy_id")},
            "client_id": self.credentials.client_id, "owner_read_succeeded": True,
            "completion_authority": False, "task_progress_authority": False,
            "dispatch_authority": False,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".tmp-{os.getpid()}")
        with temporary.open("w") as stream:
            os.chmod(temporary, 0o600)
            json.dump(payload, stream, sort_keys=True)
        os.replace(temporary, self.path)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._publish()
            except Exception as exc:
                # Preserve the last real observation; its age fails health.
                self.last_error = type(exc).__name__

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.client.detach()


def database_owner_bootstrap_kwargs(args, program) -> dict:
    descriptor = getattr(args, "state_owner_bootstrap_fd", -1)
    client_id = str(getattr(args, "state_owner_client_id", "") or "")
    if descriptor == -1 and not client_id:
        return {}
    if (type(descriptor) is not int or descriptor < 3 or not client_id
            or program is None or program.authority_mode != "quack"
            or program.task_source_kind != "duckdb"):
        raise ValueError("native owner bootstrap requires its descriptor, client and Quack program")
    credentials = request_state_owner_bootstrap(
        descriptor, client_id=client_id, store_id=program.store_id,
        timeout_seconds=_startup_wait_seconds(),
    )
    if credentials.endpoint != program.quack_endpoint:
        raise ValueError("bootstrap owner endpoint differs from the launch program")
    credentials.install_environment()
    client = QuackStateClient(
        owner_id=credentials.client_id, store_id=credentials.store_id,
        process_birth_id=credentials.process_birth_id,
    )
    try:
        client.attach(credentials.endpoint, server_id=credentials.server_id)
        source = TypedDatabaseTaskSource(client, execution_route_policy=credentials.execution_route_policy)
        source.require_quack_authority_binding(
            expected_endpoint=program.quack_endpoint,
            expected_process_instance_id=credentials.process_birth_id,
            bootstrap_credentials=credentials,
        )
    except BaseException:
        client.detach()
        raise
    return {
        "task_source": source, "close_task_source": True,
        "process_instance_id": credentials.process_birth_id,
        "state_owner_bootstrap_credentials": credentials,
    }


def observe_typed_database_without_dispatch(daemon) -> dict:
    """Read actual queue state without creating a noop execution attempt."""
    if type(daemon.task_source) is not TypedDatabaseTaskSource:
        raise ValueError("observation mode requires the native typed owner")
    daemon.open()
    tasks = daemon.task_source.ready_tasks(limit=256)
    return {
        "schema": "native-typed-daemon-observation@1",
        "ready_task_count": len(tasks.tasks), "selection_idle_reason": "implementation_disabled",
        "unchanged": True, "write_count": 0, "provider_dispatched": False,
        "attempt_consumed": False, "completion_authority": False,
    }
