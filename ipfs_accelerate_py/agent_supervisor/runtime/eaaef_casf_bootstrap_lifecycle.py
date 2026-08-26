"""Persistent CASF/Quack owner for the bounded EAAEF bootstrap prefix.

The public EAAEF owner protocol deliberately carries no database path, raw
token, SQL, or process-control authority.  The concrete lifecycle in this
module keeps those resources inside a separately launched local broker.  That
broker acquires :class:`ExclusiveOwnerLease` before the caller performs the
offline population and passes the *same held lease* to
``QuackStateServer.start_with_acquired_lease`` after the durable offline
record is present.

Only the bootstrap prefix is implemented.  This module is not the statically
opened production owner, does not qualify the independent Plan-R2 transport,
and exposes no public status, stop, launch, provider, or generic SQL surface.
The private committed-owner management path exposes only cached typed status
and exact stop for bounded local recovery; it is not reachable through
``EAAEFTypedReconciliationOwner``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from ..merge.worktree_lifecycle import current_process_birth
from ..task_sources.eaaef_casf_bootstrap_owner import (
    _REGISTRY_FIELDS,
    EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
    EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
    EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
    EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS,
    EAAEFCASFBootstrapBinding,
    EAAEFCASFBootstrapOwnerError,
    EAAEFCASFBootstrapRegistry,
    _verified,
)
from .eaaef_casf_owner_management import (
    CASFOwnerManagementClient,
    CASFOwnerManagementServer,
)

EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE: Final = (
    "EAAEFCASFBootstrapOwnerBroker@1"
)
EAAEF_CASF_BOOTSTRAP_BROKER_INIT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-bootstrap-owner-broker-init@1"
)
EAAEF_CASF_BOOTSTRAP_BROKER_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-bootstrap-owner-broker-request@1"
)
EAAEF_CASF_BOOTSTRAP_BROKER_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-bootstrap-owner-broker-response@1"
)
EAAEF_CASF_BOOTSTRAP_SNAPSHOT_BINDINGS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-bootstrap-snapshot-bindings@1"
)

_MAX_FRAME_BYTES: Final = 256 * 1024
_FRAME_HEADER = struct.Struct("!I")
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_GENERATION_RE: Final = re.compile(r"^eaaef-[a-z0-9][a-z0-9.-]{7,95}$")
_DID_RE: Final = re.compile(r"^did:key:z[A-Za-z0-9]{8,511}$")
_SAFE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}$")
_PROCESS_BIRTH_FIELDS: Final = frozenset(
    {"pid", "start_time_ticks", "parent_pid", "boot_id", "argv_sha256"}
)
_BINDING_FIELDS: Final = frozenset(
    {
        "generation_id",
        "source_head",
        "source_tree",
        "source_forest_root",
        "board_cid",
        "population_cid",
        "bootstrap_population_cid",
        "plan_r1_cid",
        "database_path",
        "owner_state_dir",
    }
)
_SNAPSHOT_BINDING_FIELDS: Final = frozenset(
    {
        "schema",
        "bootstrap_admission_cid",
        "r1_launch_capsule_cid",
        "quack_owner_qualification_cid",
        "quack_command_fabric_qualification_cid",
        "owner_principal_did",
        "shard_id",
        "store_id",
        "lease_id",
        "expected_event_cursor",
        "request_id",
        "idempotency_key",
        "issued_at_ms",
        "deadline_ms",
        "expires_at_ms",
        "one_use_nonce",
        "bindings_cid",
    }
)
_INIT_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "binding",
        "snapshot_bindings",
        "caller_process_birth",
        "init_cid",
    }
)
_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "sequence",
        "operation",
        "arguments",
        "request_cid",
    }
)
_RESPONSE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "sequence",
        "operation",
        "request_cid",
        "ok",
        "error_code",
        "result",
        "response_cid",
    }
)
_ARGUMENT_FIELDS: Final = {
    "start_after_offline_commit": frozenset(
        {"absence_attestation_cid", "offline_materialization_receipt_cid"}
    ),
    "abort_started_owner": frozenset(
        {"owner_start_receipt_cid", "abort_reason_code"}
    ),
    "commit_started_owner": frozenset(
        {"owner_start_receipt_cid", "final_record_cid"}
    ),
}
_PUBLIC_OPERATIONS: Final = frozenset(
    {
        "start_after_offline_commit",
        "abort_started_owner",
        "commit_started_owner",
    }
)
_ERROR_CODES: Final = frozenset(
    {
        "broker_bootstrap_invalid",
        "broker_caller_unavailable",
        "broker_contention",
        "broker_frame_invalid",
        "broker_frame_diverged",
        "broker_offline_commit_invalid",
        "broker_owner_start_failed",
        "broker_commit_invalid",
        "broker_abort_failed",
        "broker_internal_failure",
    }
)


class EAAEFCASFBootstrapBrokerError(EAAEFCASFBootstrapOwnerError):
    """The private CASF bootstrap broker failed closed."""


def _canonical_bytes(value: Any, *, noun: str, maximum: int = _MAX_FRAME_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} is not canonical JSON") from exc
    if not encoded or len(encoded) > maximum:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} exceeds its byte bound")
    return encoded


def _cid(value: Any) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(
        _canonical_bytes(value, noun="broker content identity")
    ).hexdigest()


def _decode_canonical_object(
    raw: object,
    *,
    noun: str,
    maximum: int = _MAX_FRAME_BYTES,
) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise EAAEFCASFBootstrapBrokerError(
            f"{noun} is not bounded canonical bytes"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} is not canonical JSON") from exc
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise EAAEFCASFBootstrapBrokerError(f"{noun} is not an exact object")
    if _canonical_bytes(value, noun=noun, maximum=maximum) != raw:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} bytes are not canonical")
    return value


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = channel.recv(size - len(output))
        if not chunk:
            raise EOFError("bootstrap broker caller channel closed")
        output.extend(chunk)
    return bytes(output)


def _recv_packet(channel: socket.socket) -> bytes:
    header = _recv_exact(channel, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size < 1 or size > _MAX_FRAME_BYTES:
        raise EAAEFCASFBootstrapBrokerError("broker frame length is invalid")
    return _recv_exact(channel, size)


def _recv_packet_buffered(channel: socket.socket, buffer: bytearray) -> bytes:
    """Receive one stream frame while retaining partial bytes across timeouts."""

    while len(buffer) < _FRAME_HEADER.size:
        chunk = channel.recv(_FRAME_HEADER.size - len(buffer))
        if not chunk:
            raise EOFError("bootstrap broker caller channel closed")
        buffer.extend(chunk)
    (size,) = _FRAME_HEADER.unpack(buffer[: _FRAME_HEADER.size])
    if size < 1 or size > _MAX_FRAME_BYTES:
        raise EAAEFCASFBootstrapBrokerError("broker frame length is invalid")
    frame_size = _FRAME_HEADER.size + size
    while len(buffer) < frame_size:
        chunk = channel.recv(frame_size - len(buffer))
        if not chunk:
            raise EOFError("bootstrap broker caller channel closed")
        buffer.extend(chunk)
    raw = bytes(buffer[_FRAME_HEADER.size:frame_size])
    del buffer[:frame_size]
    return raw


def _send_packet(channel: socket.socket, raw: bytes) -> None:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_FRAME_BYTES:
        raise EAAEFCASFBootstrapBrokerError("broker frame is not bounded bytes")
    channel.sendall(_FRAME_HEADER.pack(len(raw)) + raw)


def _assert_no_public_authority(value: Any, *, noun: str) -> None:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    try:
        reconciliation._assert_no_boundary_authority(value, path=noun)  # noqa: SLF001
    except reconciliation.EAAEFReconciliationIdentityError as exc:
        raise EAAEFCASFBootstrapBrokerError(
            f"{noun} exposes forbidden authority"
        ) from exc


def _verified_self_addressed(
    raw: Mapping[str, Any],
    *,
    fields: frozenset[str],
    cid_field: str,
    noun: str,
) -> dict[str, Any]:
    value = dict(raw)
    body = dict(value)
    claimed = body.pop(cid_field, "")
    if set(value) != fields or type(claimed) is not str or claimed != _cid(body):
        raise EAAEFCASFBootstrapBrokerError(f"{noun} identity differs")
    return value


def _exact_process_birth(raw: object, *, noun: str) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != _PROCESS_BIRTH_FIELDS:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} shape differs")
    if (
        any(type(raw[field]) is not int for field in ("pid", "start_time_ticks", "parent_pid"))
        or any(type(raw[field]) is not str for field in ("boot_id", "argv_sha256"))
        or int(raw["pid"]) < 1
        or int(raw["start_time_ticks"]) < 1
        or int(raw["parent_pid"]) < 0
        or not _SHA256_RE.fullmatch(str(raw["argv_sha256"]))
    ):
        raise EAAEFCASFBootstrapBrokerError(f"{noun} types differ")
    return dict(raw)


def _sealed_local_path(raw: object, *, noun: str, require_directory: bool) -> Path:
    if type(raw) is not str or not raw or "\x00" in raw:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} is invalid")
    lexical = Path(os.path.abspath(raw))
    if not lexical.is_absolute() or lexical.resolve(strict=False) != lexical:
        raise EAAEFCASFBootstrapBrokerError(f"{noun} is not an exact sealed path")
    for candidate in (lexical, *lexical.parents):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise EAAEFCASFBootstrapBrokerError(f"{noun} contains a symlink")
    if require_directory:
        try:
            metadata = os.lstat(lexical)
        except OSError as exc:
            raise EAAEFCASFBootstrapBrokerError(f"{noun} is unavailable") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise EAAEFCASFBootstrapBrokerError(f"{noun} is not private")
    return lexical


@dataclass(frozen=True, slots=True)
class EAAEFCASFBootstrapSnapshotBindings:
    """Code-bound, non-secret inputs for the unsigned bootstrap snapshot.

    Supplying these identities does not verify their external signatures and
    therefore cannot qualify the final reconciliation owner.  The later
    independently signed Plan-R2 channel remains a separate mandatory gate.
    """

    bootstrap_admission_cid: str
    r1_launch_capsule_cid: str
    quack_owner_qualification_cid: str
    quack_command_fabric_qualification_cid: str
    owner_principal_did: str
    shard_id: str
    store_id: str
    lease_id: str
    expected_event_cursor: str
    request_id: str
    idempotency_key: str
    issued_at_ms: int
    deadline_ms: int
    expires_at_ms: int
    one_use_nonce: str

    def __post_init__(self) -> None:
        cids = (
            self.bootstrap_admission_cid,
            self.r1_launch_capsule_cid,
            self.quack_owner_qualification_cid,
            self.quack_command_fabric_qualification_cid,
        )
        identifiers = (
            self.shard_id,
            self.store_id,
            self.lease_id,
            self.expected_event_cursor,
            self.request_id,
            self.idempotency_key,
            self.one_use_nonce,
        )
        integers = (self.issued_at_ms, self.deadline_ms, self.expires_at_ms)
        if (
            any(type(item) is not str or not _SHA256_RE.fullmatch(item) for item in cids)
            or type(self.owner_principal_did) is not str
            or not _DID_RE.fullmatch(self.owner_principal_did)
            or any(type(item) is not str or not _SAFE_ID_RE.fullmatch(item) for item in identifiers)
            or any(type(item) is not int or item < 1 for item in integers)
            or self.issued_at_ms >= self.expires_at_ms
            or self.deadline_ms < self.issued_at_ms
            or self.deadline_ms > self.expires_at_ms
        ):
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap snapshot bindings are invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": EAAEF_CASF_BOOTSTRAP_SNAPSHOT_BINDINGS_SCHEMA,
            "bootstrap_admission_cid": self.bootstrap_admission_cid,
            "r1_launch_capsule_cid": self.r1_launch_capsule_cid,
            "quack_owner_qualification_cid": self.quack_owner_qualification_cid,
            "quack_command_fabric_qualification_cid": (
                self.quack_command_fabric_qualification_cid
            ),
            "owner_principal_did": self.owner_principal_did,
            "shard_id": self.shard_id,
            "store_id": self.store_id,
            "lease_id": self.lease_id,
            "expected_event_cursor": self.expected_event_cursor,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "issued_at_ms": self.issued_at_ms,
            "deadline_ms": self.deadline_ms,
            "expires_at_ms": self.expires_at_ms,
            "one_use_nonce": self.one_use_nonce,
        }
        value["bindings_cid"] = _cid(value)
        return value

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any]
    ) -> EAAEFCASFBootstrapSnapshotBindings:
        value = _verified_self_addressed(
            raw,
            fields=_SNAPSHOT_BINDING_FIELDS,
            cid_field="bindings_cid",
            noun="CASF bootstrap snapshot bindings",
        )
        if value.get("schema") != EAAEF_CASF_BOOTSTRAP_SNAPSHOT_BINDINGS_SCHEMA:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap snapshot binding schema differs"
            )
        try:
            return cls(
                bootstrap_admission_cid=value["bootstrap_admission_cid"],
                r1_launch_capsule_cid=value["r1_launch_capsule_cid"],
                quack_owner_qualification_cid=value["quack_owner_qualification_cid"],
                quack_command_fabric_qualification_cid=value[
                    "quack_command_fabric_qualification_cid"
                ],
                owner_principal_did=value["owner_principal_did"],
                shard_id=value["shard_id"],
                store_id=value["store_id"],
                lease_id=value["lease_id"],
                expected_event_cursor=value["expected_event_cursor"],
                request_id=value["request_id"],
                idempotency_key=value["idempotency_key"],
                issued_at_ms=value["issued_at_ms"],
                deadline_ms=value["deadline_ms"],
                expires_at_ms=value["expires_at_ms"],
                one_use_nonce=value["one_use_nonce"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap snapshot bindings differ"
            ) from exc


def _binding_to_mapping(binding: EAAEFCASFBootstrapBinding) -> dict[str, Any]:
    if type(binding) is not EAAEFCASFBootstrapBinding:
        raise EAAEFCASFBootstrapBrokerError("CASF bootstrap binding is not exact")
    return {
        "generation_id": binding.generation_id,
        "source_head": binding.source_head,
        "source_tree": binding.source_tree,
        "source_forest_root": binding.source_forest_root,
        "board_cid": binding.board_cid,
        "population_cid": binding.population_cid,
        "bootstrap_population_cid": binding.bootstrap_population_cid,
        "plan_r1_cid": binding.plan_r1_cid,
        "database_path": str(binding.database_path),
        "owner_state_dir": str(binding.owner_state_dir),
    }


def _binding_from_mapping(raw: Mapping[str, Any]) -> EAAEFCASFBootstrapBinding:
    value = dict(raw)
    if set(value) != _BINDING_FIELDS:
        raise EAAEFCASFBootstrapBrokerError("CASF bootstrap binding shape differs")
    generation = value.get("generation_id")
    source_head = value.get("source_head")
    source_tree = value.get("source_tree")
    cid_fields = (
        "source_forest_root",
        "board_cid",
        "population_cid",
        "bootstrap_population_cid",
        "plan_r1_cid",
    )
    if (
        type(generation) is not str
        or not _GENERATION_RE.fullmatch(generation)
        or type(source_head) is not str
        or not _GIT_OID_RE.fullmatch(source_head)
        or type(source_tree) is not str
        or not _GIT_OID_RE.fullmatch(source_tree)
        or any(type(value.get(name)) is not str or not _SHA256_RE.fullmatch(value[name]) for name in cid_fields)
    ):
        raise EAAEFCASFBootstrapBrokerError("CASF bootstrap binding identity differs")
    database_path = _sealed_local_path(
        value.get("database_path"), noun="CASF bootstrap database path", require_directory=False
    )
    owner_state_dir = _sealed_local_path(
        value.get("owner_state_dir"), noun="CASF bootstrap owner path", require_directory=False
    )
    generation_dir = _sealed_local_path(
        str(database_path.parent), noun="CASF bootstrap generation", require_directory=True
    )
    if (
        database_path.parent != generation_dir
        or owner_state_dir.parent != generation_dir
        or database_path.name != "control.duckdb"
        or owner_state_dir.name != "casf-owner"
    ):
        raise EAAEFCASFBootstrapBrokerError("CASF bootstrap local path binding differs")
    return EAAEFCASFBootstrapBinding(
        generation_id=generation,
        source_head=source_head,
        source_tree=source_tree,
        source_forest_root=value["source_forest_root"],
        board_cid=value["board_cid"],
        population_cid=value["population_cid"],
        bootstrap_population_cid=value["bootstrap_population_cid"],
        plan_r1_cid=value["plan_r1_cid"],
        database_path=database_path,
        owner_state_dir=owner_state_dir,
    )


def _build_init_frame(
    *,
    binding: EAAEFCASFBootstrapBinding,
    snapshot_bindings: EAAEFCASFBootstrapSnapshotBindings,
) -> dict[str, Any]:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    caller_birth = reconciliation.inspect_process_birth(os.getpid())
    if caller_birth is None:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap caller process birth is unavailable"
        )
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_BOOTSTRAP_BROKER_INIT_SCHEMA,
        "interface": EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE,
        "binding": _binding_to_mapping(binding),
        "snapshot_bindings": snapshot_bindings.to_dict(),
        "caller_process_birth": caller_birth.to_dict(),
    }
    value["init_cid"] = _cid(value)
    return value


def _validate_init_frame(
    raw: bytes,
) -> tuple[EAAEFCASFBootstrapBinding, EAAEFCASFBootstrapSnapshotBindings, dict[str, Any], str]:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    value = _verified_self_addressed(
        _decode_canonical_object(raw, noun="CASF bootstrap broker init"),
        fields=_INIT_FIELDS,
        cid_field="init_cid",
        noun="CASF bootstrap broker init",
    )
    if (
        value.get("schema") != EAAEF_CASF_BOOTSTRAP_BROKER_INIT_SCHEMA
        or value.get("interface") != EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE
        or not isinstance(value.get("binding"), Mapping)
        or not isinstance(value.get("snapshot_bindings"), Mapping)
    ):
        raise EAAEFCASFBootstrapBrokerError("CASF bootstrap broker init differs")
    caller = _exact_process_birth(
        value.get("caller_process_birth"), noun="CASF bootstrap caller process birth"
    )
    observed = reconciliation.inspect_process_birth(int(caller["pid"]))
    if observed is None or observed.to_dict() != caller or os.getppid() != int(caller["pid"]):
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap caller process birth is not corroborated"
        )
    return (
        _binding_from_mapping(value["binding"]),
        EAAEFCASFBootstrapSnapshotBindings.from_mapping(value["snapshot_bindings"]),
        caller,
        str(value["init_cid"]),
    )


def _build_request(
    *,
    generation_id: str,
    sequence: int,
    operation: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if operation not in _ARGUMENT_FIELDS or set(arguments) != _ARGUMENT_FIELDS[operation]:
        raise EAAEFCASFBootstrapBrokerError("broker request arguments differ")
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_BOOTSTRAP_BROKER_REQUEST_SCHEMA,
        "interface": EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE,
        "generation_id": generation_id,
        "sequence": sequence,
        "operation": operation,
        "arguments": dict(arguments),
    }
    _assert_no_public_authority(value, noun="CASF bootstrap broker request")
    value["request_cid"] = _cid(value)
    return value


def _validate_request(raw: bytes, *, generation_id: str) -> dict[str, Any]:
    value = _verified_self_addressed(
        _decode_canonical_object(raw, noun="CASF bootstrap broker request"),
        fields=_REQUEST_FIELDS,
        cid_field="request_cid",
        noun="CASF bootstrap broker request",
    )
    operation = value.get("operation")
    arguments = value.get("arguments")
    if (
        value.get("schema") != EAAEF_CASF_BOOTSTRAP_BROKER_REQUEST_SCHEMA
        or value.get("interface") != EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE
        or value.get("generation_id") != generation_id
        or type(value.get("sequence")) is not int
        or int(value["sequence"]) < 1
        or type(operation) is not str
        or operation not in _ARGUMENT_FIELDS
        or type(arguments) is not dict
        or set(arguments) != _ARGUMENT_FIELDS[operation]
    ):
        raise EAAEFCASFBootstrapBrokerError("broker request shape differs")
    _assert_no_public_authority(value, noun="CASF bootstrap broker request")
    return value


def _build_response(
    *,
    generation_id: str,
    sequence: int,
    operation: str,
    request_cid: str,
    ok: bool,
    result: Mapping[str, Any] | None = None,
    error_code: str = "",
) -> dict[str, Any]:
    if (
        type(sequence) is not int
        or sequence < 0
        or type(operation) is not str
        or type(request_cid) is not str
        or (request_cid and not _SHA256_RE.fullmatch(request_cid))
        or type(ok) is not bool
        or (ok and error_code)
        or (not ok and error_code not in _ERROR_CODES)
    ):
        raise EAAEFCASFBootstrapBrokerError("broker response inputs differ")
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_BOOTSTRAP_BROKER_RESPONSE_SCHEMA,
        "interface": EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE,
        "generation_id": generation_id,
        "sequence": sequence,
        "operation": operation,
        "request_cid": request_cid,
        "ok": ok,
        "error_code": error_code,
        "result": {} if result is None else dict(result),
    }
    _assert_no_public_authority(value, noun="CASF bootstrap broker response")
    value["response_cid"] = _cid(value)
    return value


def _validate_response(
    raw: bytes,
    *,
    generation_id: str,
    sequence: int,
    operation: str,
    request_cid: str,
) -> dict[str, Any]:
    value = _verified_self_addressed(
        _decode_canonical_object(raw, noun="CASF bootstrap broker response"),
        fields=_RESPONSE_FIELDS,
        cid_field="response_cid",
        noun="CASF bootstrap broker response",
    )
    if (
        value.get("schema") != EAAEF_CASF_BOOTSTRAP_BROKER_RESPONSE_SCHEMA
        or value.get("interface") != EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE
        or value.get("generation_id") != generation_id
        or value.get("sequence") != sequence
        or value.get("operation") != operation
        or value.get("request_cid") != request_cid
        or type(value.get("ok")) is not bool
        or type(value.get("error_code")) is not str
        or type(value.get("result")) is not dict
        or (value["ok"] is True and value["error_code"])
        or (value["ok"] is False and value["error_code"] not in _ERROR_CODES)
    ):
        raise EAAEFCASFBootstrapBrokerError("broker response identity differs")
    _assert_no_public_authority(value, noun="CASF bootstrap broker response")
    return value


def _receipt(value: dict[str, Any], cid_field: str) -> dict[str, Any]:
    value[cid_field] = _cid(value)
    return value


def _absence_receipt(binding: EAAEFCASFBootstrapBinding) -> dict[str, Any]:
    return _receipt(
        {
            "schema": EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": binding.generation_id,
            "source_forest_root": binding.source_forest_root,
            "owner_absent": True,
            "exclusive_owner_lease_held": True,
            "observed_owner_process_birth": None,
        },
        "attestation_cid",
    )


def _read_registry_record(binding: EAAEFCASFBootstrapBinding) -> dict[str, Any]:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    path = binding.database_path.parent / "bootstrap-owner.json"
    value = _verified(
        reconciliation._private_json_object(  # noqa: SLF001
            path, noun="CASF bootstrap broker registry record"
        ),
        schema=EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
        cid_field="record_cid",
        fields=_REGISTRY_FIELDS,
        noun="CASF bootstrap broker registry record",
    )
    EAAEFCASFBootstrapRegistry._validate_phase_record(value)  # noqa: SLF001
    if value.get("generation_id") != binding.generation_id:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap broker registry identity differs"
        )
    return value


def _assert_owner_lease_released(binding: EAAEFCASFBootstrapBinding) -> None:
    lock_path = binding.database_path.with_name(
        f".{binding.database_path.name}.state-owner.lock"
    )
    marker_path = binding.database_path.with_name(
        f".{binding.database_path.name}.state-owner.json"
    )
    try:
        os.lstat(marker_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF owner marker release is unconfirmed"
        ) from exc
    else:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF owner marker remains after abort"
        )
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF owner lease release is unconfirmed"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        current = os.stat(lock_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise EAAEFCASFBootstrapBrokerError(
                "CASF owner lock identity differs after abort"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF owner lease remains held after abort"
            ) from exc
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _require_private_database(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap offline database is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_size < 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap offline database is not private"
        )


def _prepare_private_owner_state(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
        metadata = os.lstat(path)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap owner state is unsafe"
            )
        os.chmod(path, 0o700)
        current = os.lstat(path)
        if (
            stat.S_IMODE(current.st_mode) != 0o700
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap owner state identity changed"
            )
    except OSError as exc:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap owner state is unavailable"
        ) from exc


def _build_snapshot(
    *,
    binding: EAAEFCASFBootstrapBinding,
    snapshot_bindings: EAAEFCASFBootstrapSnapshotBindings,
    identity: Any,
) -> dict[str, Any]:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    if (
        str(getattr(identity, "store_id", "")) != snapshot_bindings.store_id
        or type(getattr(identity, "generation", None)) is not int
        or int(identity.generation) < 1
        or type(getattr(identity, "fence_epoch", None)) is not int
        or int(identity.fence_epoch) < 1
        or type(getattr(identity, "revision", None)) is not int
        or int(identity.revision) < 0
    ):
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap Quack identity differs from snapshot bindings"
        )
    value: dict[str, Any] = {
        "schema": reconciliation.EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA,
        "source_head": binding.source_head,
        "source_tree": binding.source_tree,
        "source_forest_root": binding.source_forest_root,
        "board_cid": binding.board_cid,
        "reconciliation_population_cid": binding.population_cid,
        "bootstrap_population_cid": binding.bootstrap_population_cid,
        "bootstrap_task_count": reconciliation.EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": reconciliation.EAAEF_PLAN_R2_TASK_COUNT,
        "terminal_statuses_imported": 0,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_owner_absent_during_materialization": True,
        "owner_started_after_bootstrap": True,
        "direct_database_mutation_after_owner_start": False,
        "bootstrap_admission_cid": snapshot_bindings.bootstrap_admission_cid,
        "r1_launch_capsule_cid": snapshot_bindings.r1_launch_capsule_cid,
        "quack_owner_qualification_cid": (
            snapshot_bindings.quack_owner_qualification_cid
        ),
        "quack_command_fabric_qualification_cid": (
            snapshot_bindings.quack_command_fabric_qualification_cid
        ),
        "owner_principal_did": snapshot_bindings.owner_principal_did,
        "shard_id": snapshot_bindings.shard_id,
        "store_id": snapshot_bindings.store_id,
        "owner_generation": int(identity.generation),
        "expected_epoch": int(identity.fence_epoch),
        "fencing_token": int(identity.fence_epoch),
        "lease_id": snapshot_bindings.lease_id,
        "expected_version": int(identity.revision),
        "expected_active_plan_cid": binding.plan_r1_cid,
        "expected_active_plan_root_cid": binding.plan_r1_cid,
        "expected_active_plan_revision": 1,
        "expected_event_cursor": snapshot_bindings.expected_event_cursor,
        "expected_semantic_root_cid": binding.source_forest_root,
        "request_id": snapshot_bindings.request_id,
        "idempotency_key": snapshot_bindings.idempotency_key,
        "deadline_ms": snapshot_bindings.deadline_ms,
        "issued_at_ms": snapshot_bindings.issued_at_ms,
        "expires_at_ms": snapshot_bindings.expires_at_ms,
        "one_use_nonce": snapshot_bindings.one_use_nonce,
    }
    value["snapshot_cid"] = reconciliation._cid(value)  # noqa: SLF001
    reconciliation._assert_no_boundary_authority(  # noqa: SLF001
        value, path="CASF bootstrap snapshot"
    )
    return value


def _start_receipt(
    *,
    binding: EAAEFCASFBootstrapBinding,
    absence: Mapping[str, Any],
    offline_receipt_cid: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    birth = reconciliation.inspect_process_birth(os.getpid())
    if birth is None:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap owner process birth is unavailable"
        )
    return _receipt(
        {
            "schema": EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": binding.generation_id,
            "source_forest_root": binding.source_forest_root,
            "population_cid": binding.population_cid,
            "absence_attestation_cid": absence["attestation_cid"],
            "offline_materialization_receipt_cid": offline_receipt_cid,
            "owner_started_after_bootstrap": True,
            "exclusive_owner_lease_handoff_complete": True,
            "owner_start_commit_pending": True,
            "provider_process_started": False,
            "owner_process_birth": birth.to_dict(),
            "bootstrap_snapshot": dict(snapshot),
        },
        "start_receipt_cid",
    )


def _commit_receipt(
    *,
    binding: EAAEFCASFBootstrapBinding,
    start_receipt: Mapping[str, Any],
    final_record_cid: str,
) -> dict[str, Any]:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    birth = reconciliation.inspect_process_birth(os.getpid())
    if birth is None:
        raise EAAEFCASFBootstrapBrokerError(
            "CASF bootstrap committed owner process birth is unavailable"
        )
    return _receipt(
        {
            "schema": EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": binding.generation_id,
            "owner_start_receipt_cid": start_receipt["start_receipt_cid"],
            "final_record_cid": final_record_cid,
            "owner_commit_completed": True,
            "owner_process_birth": birth.to_dict(),
            "owner_process_alive": True,
            "provider_process_started": False,
        },
        "commit_receipt_cid",
    )


def _safe_error_response(
    *,
    generation_id: str,
    raw: bytes,
    error_code: str,
) -> bytes:
    sequence = 0
    operation = "invalid"
    request_cid = ""
    try:
        value = _decode_canonical_object(raw, noun="rejected broker request")
        if type(value.get("sequence")) is int and int(value["sequence"]) >= 0:
            sequence = int(value["sequence"])
        if type(value.get("operation")) is str and _SAFE_ID_RE.fullmatch(value["operation"]):
            operation = str(value["operation"])
        if type(value.get("request_cid")) is str and _SHA256_RE.fullmatch(value["request_cid"]):
            request_cid = str(value["request_cid"])
    except EAAEFCASFBootstrapBrokerError:
        pass
    return _canonical_bytes(
        _build_response(
            generation_id=generation_id,
            sequence=sequence,
            operation=operation,
            request_cid=request_cid,
            ok=False,
            error_code=error_code,
        ),
        noun="CASF bootstrap broker error response",
    )


def _cleanup_owner(server: Any, lease: Any) -> None:
    if server is not None:
        try:
            server.stop()
        except BaseException:
            pass
    if lease is not None:
        try:
            if lease.held:
                lease.release(fence_token=lease.fence_token)
        except BaseException:
            pass


def _broker_child(control_fd: int, caller_death_fd: int) -> int:
    from .quack_state_server import ExclusiveOwnerLease, build_server

    channel = socket.socket(fileno=control_fd)
    caller_dead = threading.Event()
    committed = threading.Event()
    stop_after_commit = threading.Event()
    server: Any = None
    lease: ExclusiveOwnerLease | None = None
    binding: EAAEFCASFBootstrapBinding | None = None
    death_thread: threading.Thread | None = None
    management: CASFOwnerManagementServer | None = None

    def _watch_caller() -> None:
        try:
            while os.read(caller_death_fd, 1):
                pass
        except OSError:
            pass
        finally:
            caller_dead.set()

    def _signal_stop(_signum: int, _frame: object) -> None:
        stop_after_commit.set()
        try:
            channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _request_management_stop() -> None:
        stop_after_commit.set()
        try:
            channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    try:
        init_raw = _recv_packet(channel)
        binding, snapshot_bindings, _caller_birth, init_cid = _validate_init_frame(init_raw)
        death_thread = threading.Thread(
            target=_watch_caller,
            name="eaaef-bootstrap-caller-watch",
            daemon=True,
        )
        death_thread.start()
        _prepare_private_owner_state(binding.owner_state_dir)
        server = build_server(
            database_path=binding.database_path,
            state_dir=binding.owner_state_dir,
            host="127.0.0.1",
            port=0,
            repository_id="repository:ipfs_accelerate_py",
            store_id=snapshot_bindings.store_id,
            allow_experimental=False,
            allow_legacy_board_unstall=False,
        )
        from . import eaaef_reconciliation_lifecycle as reconciliation

        observed_owner_birth = reconciliation.inspect_process_birth(os.getpid())
        if observed_owner_birth is None:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap management owner birth is unavailable"
            )
        management = CASFOwnerManagementServer(
            generation_id=binding.generation_id,
            binding_cid=_cid(_binding_to_mapping(binding)),
            snapshot_bindings_cid=snapshot_bindings.to_dict()["bindings_cid"],
            state_dir=binding.owner_state_dir,
            owner_process_birth=observed_owner_birth.to_dict(),
            request_stop=_request_management_stop,
            stop_timeout_seconds=600.0,
        )
        management.start()
        birth = current_process_birth()
        lease = ExclusiveOwnerLease(
            lock_path=server.owner_lock_path(),
            marker_path=server.owner_marker_path(),
        )
        lease.acquire(
            server_id=f"eaaef-bootstrap:{binding.generation_id}",
            process_birth=birth,
            database_path=binding.database_path,
            generation=1,
        )
        absence = _absence_receipt(binding)
        held_response = _build_response(
            generation_id=binding.generation_id,
            sequence=0,
            operation="hold_exclusive_bootstrap",
            request_cid=init_cid,
            ok=True,
            result=absence,
        )
        _send_packet(
            channel,
            _canonical_bytes(held_response, noun="CASF bootstrap held response"),
        )

        journal: dict[int, tuple[bytes, bytes]] = {}
        start_receipt: dict[str, Any] | None = None
        final_record_cid = ""
        terminal = False
        while not terminal:
            if caller_dead.is_set() and not committed.is_set():
                break
            try:
                request_raw = _recv_packet(channel)
            except (EOFError, OSError):
                if committed.is_set():
                    # The durable owner_started record is the transfer point.
                    # Caller death before it aborts; caller death after it
                    # must not silently tear down the committed state owner.
                    while not stop_after_commit.wait(timeout=60.0):
                        pass
                break
            try:
                request = _validate_request(
                    request_raw, generation_id=binding.generation_id
                )
            except EAAEFCASFBootstrapBrokerError:
                _send_packet(
                    channel,
                    _safe_error_response(
                        generation_id=binding.generation_id,
                        raw=request_raw,
                        error_code="broker_frame_invalid",
                    ),
                )
                if committed.is_set():
                    stop_after_commit.set()
                break
            sequence = int(request["sequence"])
            previous = journal.get(sequence)
            if previous is not None:
                if previous[0] == request_raw:
                    _send_packet(channel, previous[1])
                    continue
                _send_packet(
                    channel,
                    _safe_error_response(
                        generation_id=binding.generation_id,
                        raw=request_raw,
                        error_code="broker_frame_diverged",
                    ),
                )
                if committed.is_set():
                    stop_after_commit.set()
                break
            expected_sequence = (
                3
                if committed.is_set()
                else (2 if start_receipt is not None else 1)
            )
            if sequence != expected_sequence:
                _send_packet(
                    channel,
                    _safe_error_response(
                        generation_id=binding.generation_id,
                        raw=request_raw,
                        error_code="broker_frame_diverged",
                    ),
                )
                if committed.is_set():
                    stop_after_commit.set()
                break
            operation = str(request["operation"])
            arguments = dict(request["arguments"])
            response: dict[str, Any]
            try:
                if operation == "start_after_offline_commit":
                    if start_receipt is not None or committed.is_set():
                        raise EAAEFCASFBootstrapBrokerError(
                            "owner start is not in its initial phase"
                        )
                    record = _read_registry_record(binding)
                    offline_cid = arguments["offline_materialization_receipt_cid"]
                    if (
                        record.get("phase") != "offline_committed"
                        or record.get("absence_attestation_cid")
                        != absence["attestation_cid"]
                        or record.get("offline_materialization_receipt_cid") != offline_cid
                        or arguments.get("absence_attestation_cid")
                        != absence["attestation_cid"]
                        or type(offline_cid) is not str
                        or not _SHA256_RE.fullmatch(offline_cid)
                    ):
                        raise EAAEFCASFBootstrapBrokerError(
                            "offline commit identity differs"
                        )
                    _require_private_database(binding.database_path)
                    identity = server.start_with_acquired_lease(lease)
                    if caller_dead.is_set():
                        raise EAAEFCASFBootstrapBrokerError(
                            "caller died during owner start"
                        )
                    snapshot = _build_snapshot(
                        binding=binding,
                        snapshot_bindings=snapshot_bindings,
                        identity=identity,
                    )
                    start_receipt = _start_receipt(
                        binding=binding,
                        absence=absence,
                        offline_receipt_cid=offline_cid,
                        snapshot=snapshot,
                    )
                    response = _build_response(
                        generation_id=binding.generation_id,
                        sequence=sequence,
                        operation=operation,
                        request_cid=request["request_cid"],
                        ok=True,
                        result=start_receipt,
                    )
                elif operation == "abort_started_owner":
                    expected_start = (
                        "" if start_receipt is None else start_receipt["start_receipt_cid"]
                    )
                    if (
                        arguments.get("owner_start_receipt_cid") != expected_start
                        or type(arguments.get("abort_reason_code")) is not str
                        or not _SAFE_ID_RE.fullmatch(arguments["abort_reason_code"])
                    ):
                        raise EAAEFCASFBootstrapBrokerError(
                            "owner abort identity differs"
                        )
                    _cleanup_owner(server, lease)
                    response = _build_response(
                        generation_id=binding.generation_id,
                        sequence=sequence,
                        operation=operation,
                        request_cid=request["request_cid"],
                        ok=True,
                        result={
                            "owner_abort_acknowledged": True,
                            "owner_start_receipt_cid": expected_start,
                        },
                    )
                    terminal = True
                elif operation == "commit_started_owner":
                    if start_receipt is None or committed.is_set():
                        raise EAAEFCASFBootstrapBrokerError(
                            "owner commit is not in its provisional phase"
                        )
                    record = _read_registry_record(binding)
                    final_record_cid = arguments["final_record_cid"]
                    ready = server.ready()
                    if (
                        arguments.get("owner_start_receipt_cid")
                        != start_receipt["start_receipt_cid"]
                        or record.get("phase") != "owner_started"
                        or record.get("owner_start_receipt_cid")
                        != start_receipt["start_receipt_cid"]
                        or record.get("record_cid") != final_record_cid
                        or type(final_record_cid) is not str
                        or not _SHA256_RE.fullmatch(final_record_cid)
                        or ready.get("ready") is not True
                        or ready.get("live") is not True
                    ):
                        raise EAAEFCASFBootstrapBrokerError(
                            "owner commit identity differs"
                        )
                    commit = _commit_receipt(
                        binding=binding,
                        start_receipt=start_receipt,
                        final_record_cid=final_record_cid,
                    )
                    management.mark_committed(
                        owner_start_receipt_cid=start_receipt[
                            "start_receipt_cid"
                        ],
                        final_record_cid=final_record_cid,
                        commit_receipt_cid=commit["commit_receipt_cid"],
                    )
                    committed.set()
                    response = _build_response(
                        generation_id=binding.generation_id,
                        sequence=sequence,
                        operation=operation,
                        request_cid=request["request_cid"],
                        ok=True,
                        result=commit,
                    )
                else:  # pragma: no cover - exact validator closes this branch.
                    raise EAAEFCASFBootstrapBrokerError("broker operation differs")
            except BaseException:
                code = {
                    "start_after_offline_commit": "broker_owner_start_failed",
                    "abort_started_owner": "broker_abort_failed",
                    "commit_started_owner": "broker_commit_invalid",
                }.get(operation, "broker_internal_failure")
                response = _build_response(
                    generation_id=binding.generation_id,
                    sequence=sequence,
                    operation=operation,
                    request_cid=request["request_cid"],
                    ok=False,
                    error_code=code,
                )
                if committed.is_set():
                    stop_after_commit.set()
                terminal = True
            response_raw = _canonical_bytes(
                response, noun="CASF bootstrap broker operation response"
            )
            journal[sequence] = (request_raw, response_raw)
            try:
                _send_packet(channel, response_raw)
            except (BrokenPipeError, ConnectionError, OSError):
                if committed.is_set():
                    while not stop_after_commit.wait(timeout=60.0):
                        pass
                break
        return 0
    except BaseException:
        return 70
    finally:
        cleanup_required = not committed.is_set() or stop_after_commit.is_set()
        if cleanup_required:
            _cleanup_owner(server, lease)
        if (
            management is not None
            and binding is not None
            and management.stop_requested.is_set()
            and cleanup_required
        ):
            try:
                _assert_owner_lease_released(binding)
                management.mark_stopped()
                management.stop_response_sent.wait(timeout=2.0)
            except EAAEFCASFBootstrapOwnerError:
                pass
        if management is not None:
            management.close()
        try:
            channel.close()
        except OSError:
            pass
        try:
            os.close(caller_death_fd)
        except OSError:
            pass
        if death_thread is not None and death_thread.is_alive():
            death_thread.join(timeout=0.1)


class _BrokerClient:
    __slots__ = (
        "binding",
        "channel",
        "death_writer",
        "process",
        "absence",
        "start_receipt",
        "final_record_cid",
        "sequence",
        "committed",
        "closed",
        "timeout_seconds",
        "receive_buffer",
    )

    def __init__(
        self,
        *,
        binding: EAAEFCASFBootstrapBinding,
        channel: socket.socket,
        death_writer: int,
        process: subprocess.Popen[bytes],
        absence: Mapping[str, Any],
        timeout_seconds: float,
    ) -> None:
        self.binding = binding
        self.channel = channel
        self.death_writer = death_writer
        self.process = process
        self.absence = dict(absence)
        self.start_receipt: dict[str, Any] | None = None
        self.final_record_cid = ""
        self.sequence = 0
        self.committed = False
        self.closed = False
        self.timeout_seconds = timeout_seconds
        self.receive_buffer = bytearray()

    def _exchange(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        sequence: int | None = None,
    ) -> dict[str, Any]:
        if self.closed:
            raise EAAEFCASFBootstrapBrokerError("CASF bootstrap broker is closed")
        selected_sequence = self.sequence + 1 if sequence is None else sequence
        request = _build_request(
            generation_id=self.binding.generation_id,
            sequence=selected_sequence,
            operation=operation,
            arguments=arguments,
        )
        request_raw = _canonical_bytes(
            request, noun="CASF bootstrap broker operation request"
        )
        prior_timeout = self.channel.gettimeout()
        self.channel.settimeout(self.timeout_seconds)
        try:
            # Once sendall succeeds the complete length-prefixed request is in
            # the local kernel stream.  A delayed response must not cause a
            # duplicate request/response pair to remain queued for the next
            # sequence; retry only the bounded receive.
            _send_packet(self.channel, request_raw)
            last_error: BaseException | None = None
            for _attempt in range(2):
                try:
                    response_raw = _recv_packet_buffered(
                        self.channel, self.receive_buffer
                    )
                    response = _validate_response(
                        response_raw,
                        generation_id=self.binding.generation_id,
                        sequence=selected_sequence,
                        operation=operation,
                        request_cid=request["request_cid"],
                    )
                    if response["ok"] is not True:
                        raise EAAEFCASFBootstrapBrokerError(
                            "CASF bootstrap broker rejected operation: "
                            + str(response["error_code"])
                        )
                    if sequence is None:
                        self.sequence = selected_sequence
                    return dict(response["result"])
                except TimeoutError as exc:
                    last_error = exc
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap broker response timed out"
            ) from last_error
        except (EOFError, BrokenPipeError, ConnectionError, OSError) as exc:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap broker response is unavailable"
            ) from exc
        finally:
            self.channel.settimeout(prior_timeout)

    def exchange_raw_for_test(self, raw: bytes) -> bytes:
        """Exercise the decoder without granting this method public authority."""

        if self.closed:
            raise EAAEFCASFBootstrapBrokerError("CASF bootstrap broker is closed")
        _send_packet(self.channel, raw)
        return _recv_packet_buffered(self.channel, self.receive_buffer)

    def close_descriptors(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.channel.close()
        except OSError:
            pass
        try:
            os.close(self.death_writer)
        except OSError:
            pass

    def wait_dead(self, timeout: float) -> bool:
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True


class _PersistentCASFBootstrapGuard(AbstractContextManager["_PersistentCASFBootstrapGuard"]):
    INTERFACE: ClassVar[str] = EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE

    def __init__(
        self,
        *,
        lifecycle: QuackEAAEFCASFBootstrapOwnerLifecycle,
        binding: EAAEFCASFBootstrapBinding,
    ) -> None:
        self._lifecycle = lifecycle
        self._binding = binding
        self._broker: _BrokerClient | None = None
        self._entered = False
        self._terminal = False

    def __enter__(self) -> _PersistentCASFBootstrapGuard:
        if self._entered:
            raise EAAEFCASFBootstrapBrokerError("CASF bootstrap guard is one-shot")
        self._entered = True
        self._broker = self._lifecycle._open_broker(self._binding)  # noqa: SLF001
        return self

    def __exit__(self, *_args: object) -> None:
        broker = self._broker
        if broker is None:
            return
        if not self._terminal and not broker.committed:
            try:
                self.abort_started_owner(
                    start_receipt=broker.start_receipt,
                    reason_code="bootstrap_guard_scope_exited",
                )
            except EAAEFCASFBootstrapOwnerError:
                broker.close_descriptors()

    def _required_broker(self) -> _BrokerClient:
        if not self._entered or self._broker is None:
            raise EAAEFCASFBootstrapBrokerError("CASF bootstrap guard is not held")
        return self._broker

    def owner_absence_attestation(self) -> Mapping[str, Any]:
        return dict(self._required_broker().absence)

    def start_after_offline_commit(
        self,
        *,
        offline_materialization_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        broker = self._required_broker()
        if self._terminal or broker.start_receipt is not None:
            raise EAAEFCASFBootstrapBrokerError("CASF owner start is not available")
        receipt_cid = offline_materialization_receipt.get("receipt_cid")
        if type(receipt_cid) is not str or not _SHA256_RE.fullmatch(receipt_cid):
            raise EAAEFCASFBootstrapBrokerError(
                "offline materialization receipt identity is invalid"
            )
        _assert_no_public_authority(
            offline_materialization_receipt,
            noun="offline materialization receipt",
        )
        started = broker._exchange(  # noqa: SLF001
            "start_after_offline_commit",
            {
                "absence_attestation_cid": broker.absence["attestation_cid"],
                "offline_materialization_receipt_cid": receipt_cid,
            },
        )
        broker.start_receipt = dict(started)
        return dict(started)

    @staticmethod
    def _abort_receipt(
        *,
        binding: EAAEFCASFBootstrapBinding,
        start_receipt: Mapping[str, Any] | None,
        reason_code: str,
    ) -> dict[str, Any]:
        start_cid = ""
        birth: dict[str, Any] | None = None
        if start_receipt is not None:
            candidate = start_receipt.get("start_receipt_cid")
            if type(candidate) is str and _SHA256_RE.fullmatch(candidate):
                start_cid = candidate
            raw_birth = start_receipt.get("owner_process_birth")
            if raw_birth is not None:
                birth = _exact_process_birth(
                    raw_birth, noun="aborted CASF owner process birth"
                )
        return _receipt(
            {
                "schema": EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
                "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
                "generation_id": binding.generation_id,
                "owner_start_receipt_cid": start_cid,
                "abort_reason_code": reason_code,
                "owner_abort_completed": True,
                "remaining_started_owner_count": 0,
                "owner_process_birth": birth,
                "owner_process_alive": False,
                "task_state_mutated": False,
            },
            "abort_receipt_cid",
        )

    def abort_started_owner(
        self,
        *,
        start_receipt: Mapping[str, Any] | None,
        reason_code: str,
    ) -> Mapping[str, Any]:
        broker = self._required_broker()
        if broker.committed:
            raise EAAEFCASFBootstrapBrokerError(
                "committed CASF owner cannot use provisional abort"
            )
        if type(reason_code) is not str or not _SAFE_ID_RE.fullmatch(reason_code):
            raise EAAEFCASFBootstrapBrokerError("CASF owner abort reason is invalid")
        start_cid = ""
        if start_receipt is not None:
            candidate = start_receipt.get("start_receipt_cid")
            if type(candidate) is str and _SHA256_RE.fullmatch(candidate):
                start_cid = candidate
        if not self._terminal and broker.process.poll() is None:
            try:
                broker._exchange(  # noqa: SLF001
                    "abort_started_owner",
                    {
                        "owner_start_receipt_cid": start_cid,
                        "abort_reason_code": reason_code,
                    },
                )
            except EAAEFCASFBootstrapBrokerError:
                pass
        broker.close_descriptors()
        if not broker.wait_dead(self._lifecycle.shutdown_timeout_seconds):
            raise EAAEFCASFBootstrapBrokerError(
                "CASF provisional owner did not exit after abort"
            )
        _assert_owner_lease_released(self._binding)
        if start_receipt is not None:
            birth = _exact_process_birth(
                start_receipt.get("owner_process_birth"),
                noun="aborted CASF owner process birth",
            )
            from . import eaaef_reconciliation_lifecycle as reconciliation

            if reconciliation.inspect_process_birth(int(birth["pid"])) is not None:
                raise EAAEFCASFBootstrapBrokerError(
                    "CASF provisional owner remains alive after abort"
                )
        self._terminal = True
        self._lifecycle._forget_broker(self._binding.generation_id, broker)  # noqa: SLF001
        return self._abort_receipt(
            binding=self._binding,
            start_receipt=start_receipt,
            reason_code=reason_code,
        )

    def commit_started_owner(
        self,
        *,
        start_receipt: Mapping[str, Any],
        final_record_cid: str,
    ) -> Mapping[str, Any]:
        broker = self._required_broker()
        if (
            self._terminal
            or broker.start_receipt is None
            or dict(start_receipt) != broker.start_receipt
            or type(final_record_cid) is not str
            or not _SHA256_RE.fullmatch(final_record_cid)
        ):
            raise EAAEFCASFBootstrapBrokerError("CASF owner commit binding differs")
        result = broker._exchange(  # noqa: SLF001
            "commit_started_owner",
            {
                "owner_start_receipt_cid": start_receipt["start_receipt_cid"],
                "final_record_cid": final_record_cid,
            },
        )
        broker.committed = True
        broker.final_record_cid = final_record_cid
        self._terminal = True
        return result


class QuackEAAEFCASFBootstrapOwnerLifecycle:
    """Launch one persistent broker that owns offline and live Quack phases."""

    INTERFACE: ClassVar[str] = EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE
    QUALIFICATION_STATUS: ClassVar[str] = (
        EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS
    )

    def __init__(
        self,
        *,
        snapshot_bindings: EAAEFCASFBootstrapSnapshotBindings,
        startup_timeout_seconds: float = 180.0,
        operation_timeout_seconds: float = 180.0,
        shutdown_timeout_seconds: float = 30.0,
    ) -> None:
        if type(snapshot_bindings) is not EAAEFCASFBootstrapSnapshotBindings:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap lifecycle requires exact snapshot bindings"
            )
        for selected in (
            startup_timeout_seconds,
            operation_timeout_seconds,
            shutdown_timeout_seconds,
        ):
            if isinstance(selected, bool) or not isinstance(selected, (int, float)) or not 0.1 <= float(selected) <= 600.0:
                raise EAAEFCASFBootstrapBrokerError(
                    "CASF bootstrap lifecycle timeout is invalid"
                )
        self.snapshot_bindings = snapshot_bindings
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.operation_timeout_seconds = float(operation_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._gate = threading.Lock()
        self._brokers: dict[str, _BrokerClient] = {}
        self._management_clients: dict[str, CASFOwnerManagementClient] = {}
        self._management_bindings: dict[str, EAAEFCASFBootstrapBinding] = {}

    def hold_exclusive_bootstrap(
        self,
        binding: EAAEFCASFBootstrapBinding,
    ) -> AbstractContextManager[_PersistentCASFBootstrapGuard]:
        if type(binding) is not EAAEFCASFBootstrapBinding:
            raise EAAEFCASFBootstrapBrokerError("CASF bootstrap binding is not exact")
        # Round-trip through the exact private capsule validator before launch.
        _binding_from_mapping(_binding_to_mapping(binding))
        return _PersistentCASFBootstrapGuard(lifecycle=self, binding=binding)

    def _open_broker(self, binding: EAAEFCASFBootstrapBinding) -> _BrokerClient:
        with self._gate:
            if binding.generation_id in self._brokers:
                raise EAAEFCASFBootstrapBrokerError(
                    "CASF bootstrap generation already has a local broker"
                )
            parent_channel, child_channel = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_STREAM
            )
            death_reader, death_writer = os.pipe2(
                getattr(os, "O_CLOEXEC", 0)
            )
            module_root = Path(__file__).resolve().parents[3]
            command = (
                sys.executable,
                "-B",
                "-m",
                "ipfs_accelerate_py.agent_supervisor.runtime.eaaef_casf_bootstrap_lifecycle",
                "--broker-child",
                str(child_channel.fileno()),
                str(death_reader),
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=module_root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(child_channel.fileno(), death_reader),
                    start_new_session=True,
                )
            except BaseException:
                parent_channel.close()
                child_channel.close()
                os.close(death_reader)
                os.close(death_writer)
                raise
            child_channel.close()
            os.close(death_reader)
            init = _build_init_frame(
                binding=binding,
                snapshot_bindings=self.snapshot_bindings,
            )
            init_raw = _canonical_bytes(init, noun="CASF bootstrap broker init")
            prior_timeout = parent_channel.gettimeout()
            parent_channel.settimeout(self.startup_timeout_seconds)
            try:
                _send_packet(parent_channel, init_raw)
                response_raw = _recv_packet(parent_channel)
                response = _validate_response(
                    response_raw,
                    generation_id=binding.generation_id,
                    sequence=0,
                    operation="hold_exclusive_bootstrap",
                    request_cid=init["init_cid"],
                )
                if response["ok"] is not True:
                    raise EAAEFCASFBootstrapBrokerError(
                        "CASF bootstrap broker refused its lease"
                    )
            except BaseException:
                parent_channel.close()
                os.close(death_writer)
                try:
                    process.wait(timeout=self.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=self.shutdown_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=self.shutdown_timeout_seconds)
                raise
            finally:
                if parent_channel.fileno() >= 0:
                    parent_channel.settimeout(prior_timeout)
            broker = _BrokerClient(
                binding=binding,
                channel=parent_channel,
                death_writer=death_writer,
                process=process,
                absence=response["result"],
                timeout_seconds=self.operation_timeout_seconds,
            )
            self._brokers[binding.generation_id] = broker
            return broker

    def _forget_broker(self, generation_id: str, broker: _BrokerClient) -> None:
        with self._gate:
            if self._brokers.get(generation_id) is broker:
                self._brokers.pop(generation_id, None)

    @staticmethod
    def _validate_management_status_binding(
        binding: EAAEFCASFBootstrapBinding,
        status: Mapping[str, Any],
    ) -> None:
        record = _read_registry_record(binding)
        if (
            status.get("generation_id") != binding.generation_id
            or status.get("phase") != "committed"
            or status.get("owner_committed") is not True
            or status.get("owner_process_alive") is not True
            or status.get("provider_process_started") is not False
            or status.get("task_state_mutated") is not False
            or record.get("phase") != "owner_started"
            or record.get("source_forest_root") != binding.source_forest_root
            or record.get("population_cid") != binding.population_cid
            or record.get("owner_lifecycle_interface")
            != EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE
            or record.get("owner_start_receipt_cid")
            != status.get("owner_start_receipt_cid")
            or record.get("record_cid") != status.get("final_record_cid")
            or record.get("owner_process_birth")
            != status.get("owner_process_birth")
        ):
            raise EAAEFCASFBootstrapBrokerError(
                "committed CASF management binding differs"
            )

    def reattach_committed_owner(
        self,
        binding: EAAEFCASFBootstrapBinding,
    ) -> Mapping[str, Any]:
        """Adopt an exact live broker using only its sealed private capsule."""

        if type(binding) is not EAAEFCASFBootstrapBinding:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap binding is not exact"
            )
        _binding_from_mapping(_binding_to_mapping(binding))
        with self._gate:
            existing = self._management_clients.get(binding.generation_id)
            existing_binding = self._management_bindings.get(binding.generation_id)
        if existing is not None:
            if existing_binding != binding:
                raise EAAEFCASFBootstrapBrokerError(
                    "committed CASF management binding changed"
                )
            status = existing.status_snapshot()
            self._validate_management_status_binding(binding, status)
            return status
        client = CASFOwnerManagementClient(
            generation_id=binding.generation_id,
            binding_cid=_cid(_binding_to_mapping(binding)),
            snapshot_bindings_cid=self.snapshot_bindings.to_dict()[
                "bindings_cid"
            ],
            state_dir=binding.owner_state_dir,
            timeout_seconds=self.operation_timeout_seconds,
        )
        status = client.status_snapshot()
        self._validate_management_status_binding(binding, status)
        with self._gate:
            raced = self._management_clients.get(binding.generation_id)
            if raced is None:
                self._management_clients[binding.generation_id] = client
                self._management_bindings[binding.generation_id] = binding
                selected = client
            else:
                if self._management_bindings.get(binding.generation_id) != binding:
                    raise EAAEFCASFBootstrapBrokerError(
                        "committed CASF management binding changed"
                    )
                selected = raced
        if selected is not client:
            status = selected.status_snapshot()
            self._validate_management_status_binding(binding, status)
        return status

    def committed_owner_status(self, generation_id: str) -> Mapping[str, Any]:
        """Return one authenticated cached owner status; never query DuckDB."""

        with self._gate:
            broker = self._brokers.get(generation_id)
            binding = self._management_bindings.get(generation_id)
        if binding is None and broker is not None and broker.committed:
            binding = broker.binding
        if binding is None:
            raise EAAEFCASFBootstrapBrokerError(
                "committed CASF bootstrap owner is not locally bound"
            )
        return self.reattach_committed_owner(binding)

    def shutdown_committed_owner(self, generation_id: str) -> Mapping[str, Any]:
        """Stop one exact owner over its private authenticated management path."""

        with self._gate:
            broker = self._brokers.get(generation_id)
            client = self._management_clients.get(generation_id)
            binding = self._management_bindings.get(generation_id)
        if binding is None and broker is not None and broker.committed:
            binding = broker.binding
        if binding is None:
            raise EAAEFCASFBootstrapBrokerError(
                "committed CASF bootstrap owner is not locally bound"
            )
        if client is None:
            self.reattach_committed_owner(binding)
            with self._gate:
                client = self._management_clients.get(generation_id)
        if client is None:  # pragma: no cover - guarded insertion above.
            raise EAAEFCASFBootstrapBrokerError(
                "committed CASF management client is unavailable"
            )
        result = client.stop()
        if not client.wait_dead(self.shutdown_timeout_seconds):
            raise EAAEFCASFBootstrapBrokerError(
                "committed CASF bootstrap owner did not stop"
            )
        _assert_owner_lease_released(binding)
        if broker is not None:
            broker.close_descriptors()
            if not broker.wait_dead(self.shutdown_timeout_seconds):
                raise EAAEFCASFBootstrapBrokerError(
                    "committed CASF bootstrap broker did not exit"
                )
            self._forget_broker(generation_id, broker)
        with self._gate:
            if self._management_clients.get(generation_id) is client:
                self._management_clients.pop(generation_id, None)
                self._management_bindings.pop(generation_id, None)
        return result

    def adopt_completed_owner_stop(
        self,
        binding: EAAEFCASFBootstrapBinding,
    ) -> Mapping[str, Any]:
        """Verify a dead owner's terminal proof without reviving its generation.

        Successful adoption proves that the capsule-bound owner durably wrote
        its exact stop intent and result and released the exclusive lease.  The
        stopped generation remains quarantined; subsequent work requires a
        fresh generation and cannot reuse this capsule or its authority.
        """

        if type(binding) is not EAAEFCASFBootstrapBinding:
            raise EAAEFCASFBootstrapBrokerError(
                "CASF bootstrap binding is not exact"
            )
        _binding_from_mapping(_binding_to_mapping(binding))
        result = CASFOwnerManagementClient.adopt_completed_stop(
            generation_id=binding.generation_id,
            binding_cid=_cid(_binding_to_mapping(binding)),
            snapshot_bindings_cid=self.snapshot_bindings.to_dict()[
                "bindings_cid"
            ],
            state_dir=binding.owner_state_dir,
        )
        record = _read_registry_record(binding)
        if (
            result.get("generation_id") != binding.generation_id
            or record.get("phase") != "owner_started"
            or record.get("source_forest_root") != binding.source_forest_root
            or record.get("population_cid") != binding.population_cid
            or record.get("owner_lifecycle_interface")
            != EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE
            or record.get("owner_start_receipt_cid")
            != result.get("owner_start_receipt_cid")
            or record.get("record_cid") != result.get("final_record_cid")
            or record.get("owner_process_birth")
            != result.get("owner_process_birth")
        ):
            raise EAAEFCASFBootstrapBrokerError(
                "completed CASF owner stop binding differs"
            )
        _assert_owner_lease_released(binding)
        with self._gate:
            broker = self._brokers.get(binding.generation_id)
            client = self._management_clients.get(binding.generation_id)
            retained_binding = self._management_bindings.get(
                binding.generation_id
            )
            if retained_binding is not None and retained_binding != binding:
                raise EAAEFCASFBootstrapBrokerError(
                    "completed CASF owner stop binding changed"
                )
            if client is not None and client.is_alive():
                raise EAAEFCASFBootstrapBrokerError(
                    "completed CASF owner stop remains live"
                )
            self._management_clients.pop(binding.generation_id, None)
            self._management_bindings.pop(binding.generation_id, None)
        if broker is not None:
            broker.close_descriptors()
            if not broker.wait_dead(self.shutdown_timeout_seconds):
                raise EAAEFCASFBootstrapBrokerError(
                    "completed CASF bootstrap broker remains live"
                )
            self._forget_broker(binding.generation_id, broker)
        return result

    def committed_generation_ids(self) -> tuple[str, ...]:
        """Return local identities only; no database authority crosses."""

        with self._gate:
            retained = {
                generation_id
                for generation_id, broker in self._brokers.items()
                if broker.committed and broker.process.poll() is None
            }
            retained.update(
                generation_id
                for generation_id, client in self._management_clients.items()
                if client.is_alive()
            )
        return tuple(sorted(retained))


def _main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "--broker-child":
        return 64
    try:
        control_fd = int(argv[1])
        caller_death_fd = int(argv[2])
    except ValueError:
        return 64
    if control_fd < 3 or caller_death_fd < 3 or control_fd == caller_death_fd:
        return 64
    return _broker_child(control_fd, caller_death_fd)


if __name__ == "__main__":  # pragma: no cover - exercised through the broker.
    raise SystemExit(_main(sys.argv[1:]))


__all__ = (
    "EAAEFCASFBootstrapBrokerError",
    "EAAEFCASFBootstrapSnapshotBindings",
    "EAAEF_CASF_BOOTSTRAP_BROKER_INIT_SCHEMA",
    "EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE",
    "EAAEF_CASF_BOOTSTRAP_BROKER_REQUEST_SCHEMA",
    "EAAEF_CASF_BOOTSTRAP_BROKER_RESPONSE_SCHEMA",
    "EAAEF_CASF_BOOTSTRAP_SNAPSHOT_BINDINGS_SCHEMA",
    "QuackEAAEFCASFBootstrapOwnerLifecycle",
)
