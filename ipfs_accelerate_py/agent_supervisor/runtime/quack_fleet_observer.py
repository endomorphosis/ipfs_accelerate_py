"""Read admitted native board sources and persist observations via Quack."""
# Retain datetime parsing compatibility with supported Python runtimes.
# ruff: noqa: UP017, FURB162
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..federation.fleet_observation import (
    OPERATION,
    SCHEMA,
    FleetObservationStore,
    _fleet_templates,
    canonical,
)
from ..task_sources.quack_state_client import QuackStateClient
from ..task_sources.typed_state_owner import TypedStateOwnerConnection


def _doep_native_authority(native: Mapping[str, Any], identity: Mapping[str, Any], *, now: datetime) -> dict[str, Any]:
    """Admit the owner-local Quack monitor's exact fresh broker-bound receipt."""
    live, broker, handoff = (native.get(key, {}) for key in ("task_authority", "bootstrap_broker", "handoff"))
    if not all(isinstance(item, Mapping) for item in (live, broker, handoff)):
        return {}
    owner = handoff.get("owner_identity", {})
    if not isinstance(owner, Mapping):
        return {}
    launch = handoff.get("launch_id")
    birth = identity.get("process_birth", {})
    pid = birth.get("pid")
    if (native.get("schema") != "ipfs_accelerate_py/agent-supervisor/doep-bootstrap-handoff@1"
        or live.get("schema") != "ipfs_accelerate_py/agent-supervisor/doep-live-status@1"
        or broker.get("schema") != "ipfs_accelerate_py/agent-supervisor/doep-bootstrap-broker@1"
        or live.get("credential_transport") != "private_inherited_socket"
        or live.get("owner_ready") is not True or broker.get("ready") is not True
        or native.get("operator_alive") is not True or live.get("raw_token_in_evidence") is not False
        or not launch or live.get("launch_id") != launch or broker.get("launch_id") != launch
        or not pid or any(item.get(key) != pid for item, key in ((native, "operator_pid"), (live, "monitor_pid"), (broker, "operator_pid"), (handoff, "operator_pid")))
        or any(owner.get(key) != identity.get(key) for key in ("server_id", "database_uuid", "generation", "process_birth_id", "listen_uri"))
        or live.get("owner_server_id") != identity.get("server_id") or broker.get("server_id") != identity.get("server_id")
        or broker.get("state_owner_process_birth_id") != identity.get("process_birth_id")
        or any(not live.get(key) or live.get(key) != handoff.get(key) for key in ("plan_root_cid", "repository_tree_id"))):
        return {}
    try:
        age = (now - datetime.fromisoformat(str(live.get("updated_at", "")).replace("Z", "+00:00"))).total_seconds()
    except (TypeError, ValueError):
        return {}
    return dict(live) if 0 <= age <= 20 else {}


def _pctdd_native_authority(native: Mapping[str, Any], identity: Mapping[str, Any], *, query_seconds: float) -> dict[str, Any]:
    """PCTDD performs a fresh authenticated query, not a cached status read."""
    candidate, state = native.get("task_authority", {}), native.get("state_owner", {})
    if not isinstance(candidate, Mapping) or not isinstance(state, Mapping):
        return {}
    lifecycle = state.get("authoritative_lifecycle", {})
    if not isinstance(lifecycle, Mapping):
        return {}
    returned = candidate.get("identity", {})
    fields = ("server_id", "database_uuid", "generation", "process_birth_id", "listen_uri", "store_id")
    if (native.get("schema") != "ipfs_accelerate_py/agent-supervisor/parallel-content-sealing-proof-carrying-tdd-operator@1"
        or not 0 <= query_seconds <= 30 or candidate.get("available") is not True or candidate.get("authenticated_query") is not True
        or candidate.get("direct_database_file_open") is not False or candidate.get("transport") != "quack_loopback_token_attach"
        or lifecycle.get("available") is not True or lifecycle.get("reason") != "authenticated_live_quack_query"
        or lifecycle.get("direct_database_file_open") is not False or state.get("lifecycle_consistent") is not True
        or not isinstance(returned, Mapping) or any(returned.get(key) != identity.get(key) for key in fields)
        or any(lifecycle.get("latest", {}).get(key) != identity.get(key) for key in fields)):
        return {}
    return dict(candidate)


def read_native_source(board: Mapping[str, Any], *, adapter: Any = None) -> dict[str, Any]:
    """Reuse native credential admission; never read task counts from replicas."""
    if adapter is None:
        from ..rescue import live_board_probe as adapter
    result = {"schema": SCHEMA, "source_id": board["id"], "observed_at": "", "availability": "unavailable",
              "source_identity": {}, "native_receipt": {}, "reason": "native_source_unavailable", "completion_authority": False}
    try:
        if not Path(board["database_path"]).is_file() or not Path(board["config_path"]).is_file():
            result["reason"] = "native_configuration_or_database_missing"
            return result
        status = adapter.read_json(Path(board["owner_status_path"]))
        identity = status.get("identity", {})
        birth = identity.get("process_birth", {})
        if status.get("lifecycle") != "ready" or not adapter.birth_matches(adapter.process_identity(birth.get("pid")), birth):
            result["reason"] = "native_owner_not_ready_or_alive"
            return result
        if identity.get("listen_uri") != board["quack_endpoint"]:
            result["reason"] = "native_owner_endpoint_mismatch"
            return result
        result["source_identity"] = {key: identity[key] for key in ("database_uuid", "generation", "process_birth_id", "listen_uri") if key in identity}
        query_started = time.monotonic()
        native, error, _attempts = adapter._status_with_receipt_retry(board, birth)
        query_seconds = time.monotonic() - query_started
        latest = adapter.read_json(Path(board["owner_status_path"]))
        latest_identity = latest.get("identity", {})
        if latest.get("lifecycle") != "ready" or not adapter.birth_matches(adapter.process_identity(birth.get("pid")), birth) or any(latest_identity.get(key) != value for key, value in result["source_identity"].items()):
            result["reason"] = "native_owner_changed_during_query"
            return result
        if error:
            result["reason"] = error
            return result
        authority = {}
        if native.get("schema") == "ipfs_accelerate_py/agent-supervisor/database-board-status@1":
            authority = adapter._database_board_authority(native, board, latest, time.time())
        elif board["id"] == "pctdd":
            authority = _pctdd_native_authority(native, latest_identity, query_seconds=query_seconds)
        elif board["id"] == "doep":
            authority = _doep_native_authority(native, latest_identity, now=datetime.now(timezone.utc))
        elif board["id"] == "aseh":
            receipt = native.get("receipt", {})
            samples = receipt.get("samples", [])
            if native.get("broker_authenticated_receipt") is True and receipt.get("broker_authenticated") is True and samples:
                candidate = samples[-1].get("authority", {})
                binding = candidate.get("owner_binding", {})
                started_ms = candidate.get("query_started_at_ms")
                fresh = type(started_ms) is int and 0 <= time.time() - started_ms / 1000 <= 30
                same_owner = isinstance(binding, Mapping) and all(binding.get(key) == latest_identity.get(key) for key in ("server_id", "database_uuid", "generation", "process_birth_id", "listen_uri", "store_id"))
                if candidate.get("available") is True and candidate.get("transport") == "quack" and candidate.get("credential_path") == "sealed_memfd_broker" and fresh and same_owner:
                    authority = candidate
        else:
            candidate = native.get("task_authority", {})
            age = native.get("status_age_seconds")
            current = type(age) in (int, float) and 0 <= age <= 30
            if candidate.get("available") is True and current and (candidate.get("authenticated_query") is True or candidate.get("transport") == "exclusive_owner_authenticated_quack_projection"):
                authority = candidate
        if not authority:
            result["reason"] = "native_projection_not_direct_query" if board["id"] == "spar" else "native_quack_read_not_admitted"
            return result
        # Retain the actual native authority payload plus digest of its operator
        # envelope. A daemon/watchdog projection cannot enter this branch.
        result.update(availability="available", reason="", native_receipt={
            "authority": authority, "operator_envelope_sha256": "sha256:" + hashlib.sha256(canonical(native).encode()).hexdigest(),
            "admission": "native_operator_authenticated_quack_read"})
        if board["id"] == "doep":
            sample_time = datetime.fromisoformat(authority["updated_at"].replace("Z", "+00:00"))
            result["native_receipt"]["valid_until"] = (sample_time + timedelta(seconds=20)).isoformat()
        elif board["id"] == "aseh":
            result["native_receipt"]["valid_until"] = datetime.fromtimestamp(authority["query_started_at_ms"] / 1000 + 30, timezone.utc).isoformat()
        elif board["id"] == "pctdd":
            result["native_receipt"]["valid_until"] = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(availability="unavailable", native_receipt={}, reason=f"native_read_failed:{type(error).__name__}")
        return result
    finally:
        result["observed_at"] = datetime.now(timezone.utc).isoformat()


class FleetObserver:
    """Bounded source fanout; the native owner retains control and write authority."""
    def __init__(self, server: Any, inventory_path: Path, output_path: Path, *, poll_seconds: float = 10):
        if not 5 <= poll_seconds <= 300:
            raise ValueError("fleet poll interval must be between 5 and 300 seconds")
        self.server, self.inventory_path, self.output_path = server, inventory_path, output_path
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.last_progress = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="quack-fleet-observer", daemon=True)

    def _client(self) -> QuackStateClient:
        identity = self.server.identity
        if identity is None:
            raise ValueError("native owner identity unavailable")
        client_id = "fleet:observation-writer"
        templates = _fleet_templates()
        token = self.server.issue_typed_client_grant(client_id=client_id, process_birth_id=identity.process_birth_id,
                    allowed_operations=("whoami_metadata", "load_store_generation", "txn_load_generation", "txn_lookup_idempotency",
                                        "txn_advance_store_revision", "txn_record_idempotency", *(item.name for item in templates)),
                    allowed_command_operations=(OPERATION,), ttl_seconds=600)
        def connect(_endpoint):
            return TypedStateOwnerConnection(socket_path=self.server.typed_command_socket_path(), token=token,
                    client_id=client_id, process_birth_id=identity.process_birth_id, store_id=identity.store_id)
        client = QuackStateClient(owner_id=client_id, store_id=identity.store_id, process_birth_id=identity.process_birth_id, connection_factory=connect)
        try:
            client.attach(identity.listen_uri, server_id=identity.server_id)
        except BaseException:
            client.close()
            raise
        return client

    def cycle(self) -> dict[str, Any]:
        inventory = json.loads(self.inventory_path.read_text())
        if inventory.get("schema") != "ipfs_accelerate_py/taskboard-fleet-inventory@1":
            raise ValueError("native fleet inventory required")
        boards = inventory.get("boards", [])
        ids = [board["id"] for board in boards]
        if not 1 <= len(boards) <= 4096 or len(set(ids)) != len(ids):
            raise ValueError("bounded unique native source inventory required")
        with ThreadPoolExecutor(max_workers=min(16, len(boards))) as executor:
            samples = list(executor.map(read_native_source, boards))
        client = self._client()
        try:
            store = FleetObservationStore(client)
            for sample in samples:
                store.record(sample)
            view = store.view(ids)
            view["observed_at"] = datetime.now(timezone.utc).isoformat()
            view["registration_count"] = len(ids)
            view["source_admission"] = {sample["source_id"]: sample["availability"] for sample in samples}
            return view
        finally:
            client.close()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = self.cycle()
            except Exception as error:  # noqa: BLE001 - isolate one bounded observation cycle
                result = {"schema": "ipfs_accelerate_py/agent-supervisor/fleet-observer-error@1", "error": type(error).__name__,
                          "reason": str(error)[:512], "completion_authority": False}
            self.last_progress = time.monotonic()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, self.output_path)
            self.stop_event.wait(self.poll_seconds)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
