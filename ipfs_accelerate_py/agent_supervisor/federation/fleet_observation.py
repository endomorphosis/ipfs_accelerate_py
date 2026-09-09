"""Typed, non-authoritative fleet observations in the existing Quack store.

Snapshots are ordinary content-addressed artifacts with federation history
receipts. They never mutate source tasks, leases, acceptance, or semantic roots.
"""
# Retain datetime parsing compatibility with supported Python runtimes.
# ruff: noqa: UP017, FURB162
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ..task_sources.control_plane_contracts import (
    CommandKind,
    StateAuthorityClass,
    StateCommand,
)
from ..task_sources.quack_state_client import (
    QuackStateClient,
    StatementKind,
    StatementTemplate,
    TransportMode,
)

SCHEMA = "ipfs_accelerate_py/agent-supervisor/fleet-source-observation@1"
OPERATION = "fleet.observation.record"
MUTATIONS = frozenset({"fleet_insert_observation_artifact", "fleet_insert_observation_receipt"})
MAX_BYTES = 262144


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "source_id", "observed_at", "availability", "source_identity", "native_receipt", "reason", "completion_authority"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != SCHEMA:
        raise ValueError("closed fleet observation required")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", str(value["source_id"])):
        raise ValueError("invalid observation source id")
    if value["completion_authority"] is not False or value["availability"] not in {"available", "unavailable"}:
        raise ValueError("fleet observations cannot establish completion authority")
    timestamp = datetime.fromisoformat(str(value["observed_at"]).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("observation time must include a timezone")
    if not isinstance(value["source_identity"], Mapping) or not isinstance(value["native_receipt"], Mapping):
        raise TypeError("native observation identity and receipt must be objects")
    if value["availability"] == "available":
        identity = value["source_identity"]
        if not all(identity.get(key) for key in ("database_uuid", "generation", "process_birth_id", "listen_uri")):
            raise ValueError("admitted observation requires exact native owner identity")
        if not value["native_receipt"] or value["reason"]:
            raise ValueError("available observation requires native receipt and no error")
    elif value["native_receipt"] or not isinstance(value["reason"], str) or not value["reason"]:
        raise ValueError("unavailable observation cannot copy an old receipt as current")
    if len(canonical(value).encode()) > MAX_BYTES:
        raise ValueError("fleet observation exceeds bounded artifact size")
    return dict(value)


def observation_cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def _fleet_templates() -> tuple[StatementTemplate, ...]:
    def template(name, sql, parameters, *, query=False):
        return StatementTemplate(name=name, sql=sql, parameter_names=tuple(parameters),
                                 kind=StatementKind.QUERY if query else StatementKind.MUTATION,
                                 description="closed observational fleet projection")
    return (
        template("fleet_insert_observation_artifact", """
            INSERT INTO artifacts (cid, media_type, byte_length, digest, storage_uri, kind, created_at, provenance_json, payload_json)
            VALUES (?, 'application/json', ?, ?, ?, 'fleet_source_observation', ?, '{"completion_authority":false}', ?)
            """, ("cid", "byte_length", "digest", "storage_uri", "created_at", "payload_json")),
        template("fleet_insert_observation_receipt", """
            INSERT INTO federation_receipts (federation_receipt_id, tenant_id, federation_id, receipt_kind,
                federation_revision, control_plane_generation, event_watermark, issuer_id, content_ref, recorded_at)
            VALUES (?, 'local_observations', 'fleet:taskboard-observations', 'fleet_source_observation', ?, ?, 0, ?, ?, ?)
            """, ("federation_receipt_id", "federation_revision", "control_plane_generation", "issuer_id", "content_ref", "recorded_at")),
        template("fleet_select_source_observation", """
            SELECT cid, payload_json FROM artifacts WHERE kind = 'fleet_source_observation' AND storage_uri = ?
            ORDER BY created_at DESC, cid DESC LIMIT 1
            """, ("storage_uri",), query=True),
        template("fleet_select_last_admitted_observation", """
            SELECT cid, payload_json FROM artifacts WHERE kind = 'fleet_source_observation' AND storage_uri = ?
            AND json_extract_string(payload_json, '$.availability') = 'available'
            ORDER BY created_at DESC, cid DESC LIMIT 1
            """, ("storage_uri",), query=True),
    )


def validate_owner_manifest(command: Any, manifest: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    """Owner-side coupling: exactly one immutable artifact and matching receipt."""
    domain = [(name, bound) for name, bound in manifest if name in MUTATIONS]
    if [name for name, _ in domain] != ["fleet_insert_observation_artifact", "fleet_insert_observation_receipt"]:
        raise ValueError("fleet observation requires exactly one ordered artifact and receipt")
    artifact, receipt = domain[0][1], domain[1][1]
    value = validate_observation(json.loads(artifact["payload_json"]))
    payload, cid = canonical(value), observation_cid(value)
    expected_artifact = {"cid": cid, "byte_length": len(payload.encode()), "digest": cid,
                         "storage_uri": "fleet-source:" + value["source_id"], "created_at": value["observed_at"], "payload_json": payload}
    expected_receipt = {"federation_receipt_id": "fleet-receipt:" + cid,
                        "federation_revision": command.expected_revision + 1,
                        "control_plane_generation": command.expected_generation,
                        "issuer_id": command.parameters["issuer_id"], "content_ref": cid, "recorded_at": value["observed_at"]}
    if artifact != expected_artifact or receipt != expected_receipt or command.parameters.get("observation_cid") != cid:
        raise ValueError("fleet observation manifest differs from its content-addressed command")


class FleetObservationStore:
    def __init__(self, client: QuackStateClient):
        if not isinstance(client, QuackStateClient) or not client.attached or client.session.transport_mode is not TransportMode.QUACK:
            raise ValueError("fleet projection requires attached native Quack client")
        self.client = client
        for template in _fleet_templates():
            if template.name not in client.list_templates():
                client.register_template(template)
        client.seal_templates()

    def record(self, observation: Mapping[str, Any]) -> Any:
        value = validate_observation(observation)
        payload, cid = canonical(value), observation_cid(value)
        client, live = self.client, self.client.load_generation()
        command = StateCommand(command_id="fleet-command:" + cid, command_kind=CommandKind.APPEND,
                               store_id=client.store_id, session_id=client.session.session_id,
                               expected_generation=live.generation, expected_revision=live.revision, fence_epoch=live.fence_epoch,
                               idempotency_key="fleet-observation:" + cid, authority_class=StateAuthorityClass.DIAGNOSTIC,
                               parameters={"operation": OPERATION, "observation_cid": cid, "issuer_id": client.owner_id})
        def apply(_txn, active, generation):
            client.execute("fleet_insert_observation_artifact", {"cid": cid, "byte_length": len(payload.encode()), "digest": cid,
                           "storage_uri": "fleet-source:" + value["source_id"], "created_at": value["observed_at"], "payload_json": payload})
            client.execute("fleet_insert_observation_receipt", {"federation_receipt_id": "fleet-receipt:" + cid,
                           "federation_revision": generation.revision + 1, "control_plane_generation": generation.generation,
                           "issuer_id": active.parameters["issuer_id"], "content_ref": cid, "recorded_at": value["observed_at"]})
            return {"observation_cid": cid, "source_id": value["source_id"], "completion_authority": False}
        return client.submit_command(command, apply=apply)

    def view(self, source_ids: Sequence[str], *, max_age_seconds: float = 60, now: datetime | None = None) -> dict[str, Any]:
        if not 1 <= max_age_seconds <= 300:
            raise ValueError("source freshness must be bounded")
        observed = now or datetime.now(timezone.utc)
        if len(source_ids) > 4096 or len(set(source_ids)) != len(source_ids):
            raise ValueError("bounded unique source ids required")
        sources = {}
        for source_id in source_ids:
            selected = {}
            for key, operation in (("current", "fleet_select_source_observation"), ("last_admitted", "fleet_select_last_admitted_observation")):
                rows = self.client.execute(operation, {"storage_uri": "fleet-source:" + source_id})
                if rows:
                    value = validate_observation(json.loads(rows[0]["payload_json"]))
                    if observation_cid(value) != rows[0]["cid"]:
                        raise ValueError("stored observation checksum differs")
                    selected[key] = value
            current = selected.get("current")
            age = None if current is None else (observed - datetime.fromisoformat(current["observed_at"].replace("Z", "+00:00"))).total_seconds()
            selected["available"] = bool(current and current["availability"] == "available" and age is not None and 0 <= age <= max_age_seconds)
            valid_until = (current or {}).get("native_receipt", {}).get("valid_until")
            if valid_until:
                try:
                    selected["available"] &= observed <= datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    selected["available"] = False
            selected["age_seconds"] = age
            selected["reason"] = "" if selected["available"] else (
                "source_observation_absent" if current is None else
                "source_observation_stale" if age is None or not 0 <= age <= max_age_seconds or current["availability"] == "available" else current["reason"])
            sources[source_id] = selected
        return {"schema": "ipfs_accelerate_py/agent-supervisor/fleet-aggregate-view@1", "sources": sources,
                "control_store_id": self.client.store_id, "control_generation": self.client.load_generation().generation,
                "completion_authority": False, "ducklake_history_required_for_control": False}
