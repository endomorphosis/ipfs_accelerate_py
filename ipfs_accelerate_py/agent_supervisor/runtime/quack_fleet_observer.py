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
    validate_observation,
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


def _configured_source_integrity_verified(board: Mapping[str, Any], adapter: Any) -> bool:
    """Configured source scopes require the immutable adapter's bounded check."""
    if board.get("source_integrity_paths") is None:
        return True
    check = getattr(adapter, "_source_integrity", None)
    if not callable(check):
        return False
    try:
        integrity = check(board)
    except Exception:  # noqa: BLE001 - an unavailable guard cannot admit a source
        return False
    return (isinstance(integrity, Mapping) and integrity.get("configured") is True
            and integrity.get("valid") is True)


def read_native_source(board: Mapping[str, Any], *, adapter: Any = None) -> dict[str, Any]:
    """Reuse native credential admission; never read task counts from replicas."""
    if adapter is None:
        from ..rescue import live_board_probe as adapter
    result = {"schema": SCHEMA, "source_id": board["id"], "observed_at": "", "availability": "unavailable",
              "source_identity": {}, "native_receipt": {}, "reason": "native_source_unavailable", "completion_authority": False}
    try:
        if not _configured_source_integrity_verified(board, adapter):
            result["reason"] = "source_integrity_not_verified"
            return result
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
        query_finished_at = datetime.now(timezone.utc)
        if not _configured_source_integrity_verified(board, adapter):
            result["reason"] = "source_integrity_not_verified"
            return result
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
            result["native_receipt"]["valid_until"] = (query_finished_at + timedelta(seconds=30)).isoformat()
        return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(availability="unavailable", native_receipt={}, reason=f"native_read_failed:{type(error).__name__}")
        return result
    finally:
        result["observed_at"] = datetime.now(timezone.utc).isoformat()


class _SourcePolls:
    """Keep one read per source in flight without a fleet-wide cycle barrier."""
    def __init__(self, poll_seconds: float, *, max_workers: int = 16, executor: Any = None):
        self.poll_seconds, self.max_workers = poll_seconds, max_workers
        self.executor = executor or ThreadPoolExecutor(max_workers=max_workers)
        self.pending: dict[str, Any] = {}
        self.keys: dict[str, str] = {}
        self.next_poll: dict[str, float] = {}

    def step(self, boards: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        current = {board["id"]: board for board in boards}
        keys = {identifier: canonical(board) for identifier, board in current.items()}
        for identifier, key in keys.items():
            if self.keys.get(identifier) != key:
                self.next_poll[identifier] = 0
        self.keys = keys
        self.next_poll = {identifier: self.next_poll.get(identifier, 0) for identifier in current}
        samples = []
        for identifier, (key, future) in list(self.pending.items()):
            if identifier not in current:
                future.cancel()
            if not future.done():
                continue
            del self.pending[identifier]
            # A result from a removed or changed inventory binding is history
            # of an obsolete selection, not a current source observation.
            if keys.get(identifier) != key:
                continue
            self.next_poll[identifier] = now + self.poll_seconds
            try:
                sample = validate_observation(future.result())
                if sample["source_id"] != identifier:
                    raise ValueError("native source returned a foreign observation")
            except Exception as error:  # noqa: BLE001 - contain one source reader
                sample = {"schema": SCHEMA, "source_id": identifier,
                          "observed_at": datetime.now(timezone.utc).isoformat(),
                          "availability": "unavailable", "source_identity": {}, "native_receipt": {},
                          "reason": f"native_read_failed:{type(error).__name__}", "completion_authority": False}
            samples.append(sample)
        # Oldest due source first; never queue an unbounded number of threads
        # or duplicate a slow source while its previous read is still running.
        for identifier in sorted(current, key=lambda item: self.next_poll[item]):
            if len(self.pending) >= self.max_workers:
                break
            if identifier in self.pending or self.next_poll[identifier] > now:
                continue
            self.pending[identifier] = (keys[identifier], self.executor.submit(read_native_source, current[identifier]))
        return samples

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


class FleetObserver:
    """Bounded source fanout; the native owner retains control and write authority."""
    def __init__(self, server: Any, inventory_path: Path, output_path: Path, *, poll_seconds: float = 10,
                 source_workers: int = 16):
        if not 5 <= poll_seconds <= 300:
            raise ValueError("fleet poll interval must be between 5 and 300 seconds")
        if type(source_workers) is not int or not 1 <= source_workers <= 256:
            raise ValueError("source workers must be between 1 and 256")
        self.server, self.inventory_path, self.output_path = server, inventory_path, output_path
        self.poll_seconds = poll_seconds
        self.source_workers = source_workers
        self._polls: _SourcePolls | None = None
        self._writer_client: QuackStateClient | None = None
        self._writer_grant: Any = None
        self._writer_token: str | None = None
        self._writer_owner_key: tuple[Any, ...] | None = None
        self.stop_event = threading.Event()
        self.last_progress = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="quack-fleet-observer", daemon=True)

    def _close_writer_client(self) -> None:
        client, self._writer_client = self._writer_client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _retire_writer(self, *, revoke: bool = True) -> None:
        self._close_writer_client()
        grant, self._writer_grant = self._writer_grant, None
        self._writer_token = None
        self._writer_owner_key = None
        if revoke and grant is not None:
            try:
                self.server.revoke_typed_client_grant(grant.grant_id)
            except Exception:
                # A stopped/replaced owner has already invalidated its grants.
                pass

    def _client(self) -> QuackStateClient:
        identity = self.server.identity
        if identity is None:
            self._retire_writer(revoke=False)
            raise ValueError("native owner identity unavailable")
        owner_key = tuple(getattr(identity, key) for key in (
            "server_id", "store_id", "database_uuid", "generation", "fence_epoch",
            "process_birth_id", "listen_uri"))
        if self._writer_owner_key is not None and self._writer_owner_key != owner_key:
            # Never revoke an old owner's grant ID in a replacement gateway.
            self._retire_writer(revoke=False)
        now_ms = int(time.time() * 1000)
        if self._writer_grant is not None:
            remaining_ms = self._writer_grant.expires_at - now_ms
            if remaining_ms <= 0:
                self._retire_writer()
            elif remaining_ms <= 120_000 or self._writer_client is None:
                try:
                    self._writer_grant = self.server.renew_typed_client_grant(
                        self._writer_grant.grant_id, ttl_seconds=600)
                except Exception:
                    # One failure per cycle; do not issue capabilities in a
                    # tight retry loop when native ownership is unavailable.
                    self._retire_writer()
                    raise
        client_id = "fleet:observation-writer"
        if self._writer_grant is None:
            templates = _fleet_templates()
            token, grant = self.server.issue_typed_client_grant_record(
                client_id=client_id, process_birth_id=identity.process_birth_id,
                allowed_operations=("whoami_metadata", "load_store_generation", "txn_load_generation", "txn_lookup_idempotency",
                                    "txn_advance_store_revision", "txn_record_idempotency", *(item.name for item in templates)),
                allowed_command_operations=(OPERATION,), ttl_seconds=600)
            self._writer_token, self._writer_grant = token, grant
            self._writer_owner_key = owner_key
        if self._writer_client is not None:
            return self._writer_client
        token = self._writer_token
        def connect(_endpoint):
            return TypedStateOwnerConnection(socket_path=self.server.typed_command_socket_path(), token=token,
                    client_id=client_id, process_birth_id=identity.process_birth_id, store_id=identity.store_id)
        client = QuackStateClient(owner_id=client_id, store_id=identity.store_id, process_birth_id=identity.process_birth_id, connection_factory=connect)
        try:
            client.attach(identity.listen_uri, server_id=identity.server_id)
        except BaseException:
            client.close()
            raise
        self._writer_client = client
        return client

    def cycle(self) -> dict[str, Any] | None:
        inventory = json.loads(self.inventory_path.read_text())
        if inventory.get("schema") != "ipfs_accelerate_py/taskboard-fleet-inventory@1":
            raise ValueError("native fleet inventory required")
        boards = inventory.get("boards", [])
        ids = [board["id"] for board in boards]
        if not 1 <= len(boards) <= 4096 or len(set(ids)) != len(ids):
            raise ValueError("bounded unique native source inventory required")
        if self._polls is None:
            self._polls = _SourcePolls(self.poll_seconds, max_workers=self.source_workers)
        samples = self._polls.step(boards, time.monotonic())
        if not samples:
            return None
        client = self._client()
        try:
            store = FleetObservationStore(client)
            for sample in samples:
                store.record(sample)
            view = store.view(ids)
            view["observed_at"] = datetime.now(timezone.utc).isoformat()
            view["registration_count"] = len(ids)
            view["source_admission"] = {identifier: "available" if item["available"] else "unavailable"
                                        for identifier, item in view["sources"].items()}
            return view
        except BaseException:
            # Retry the connection on a later cycle using the same still-live
            # owner grant. Poll cadence and the 600-second health bound remain.
            self._close_writer_client()
            raise

    def _run(self):
        try:
            while not self.stop_event.is_set():
                delay = 0.25
                try:
                    result = self.cycle()
                    if result is not None:
                        self.last_progress = time.monotonic()
                except Exception as error:  # noqa: BLE001 - isolate one bounded observation cycle
                    delay = self.poll_seconds
                    result = {"schema": "ipfs_accelerate_py/agent-supervisor/fleet-observer-error@1", "error": type(error).__name__,
                              "reason": str(error)[:512], "completion_authority": False}
                if result is not None:
                    self.output_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.output_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                    os.replace(temporary, self.output_path)
                self.stop_event.wait(delay)
        finally:
            self._retire_writer()
            if self._polls is not None:
                self._polls.close()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
