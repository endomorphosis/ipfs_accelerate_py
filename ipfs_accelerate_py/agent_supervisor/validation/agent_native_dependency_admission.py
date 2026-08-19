"""Independent, source-addressed admission for the supervisor native runtime.

The native dependency pin describes reviewed DuckDB extension bytes and their
ABI.  A pin is evidence, not authority.  This module binds that exact pin to a
single plan-bound child birth under a short-lived independent signature.  It
does not carry a filesystem path, descriptor, token, callback, or process
launch authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ...llm_router import (
    AgentSupervisorNativeDependencyPin,
    parse_agent_supervisor_native_dependency_pin,
)
from ..control.profile_authority import LocalProfileTampered, verify_did_key_signature
from .eaaef_bootstrap_gateway_launch import EAAEF_BOARD_NAMESPACE

AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_INTERFACE: Final = (
    "AgentSupervisorNativeDependencyAdmission@1"
)
AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/native-dependency-admission@1"
)
AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_REVIEWER_ROLE: Final = (
    "independent_agent_supervisor_native_dependency_reviewer"
)
AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_PATH_TEMPLATE: Final = (
    "agent-supervisor-native-dependency-admission--<source_head>--"
    "<plan_root_sha256>--<lane_session_sha256>--g<lane_generation>.json"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_DID_KEY = re.compile(r"did:key:z[1-9A-HJ-NP-Za-km-z]{20,200}\Z")
_MAX_FILE_BYTES = 256 * 1024
_MAX_LIFETIME_MS = 15 * 60 * 1000
_VERIFIED_TOKEN = object()

_BINDING_FIELDS: Final = frozenset(
    {
        "board_namespace",
        "source_head",
        "source_tree",
        "configuration_root",
        "accepted_control_plane_capsule_id",
        "accepted_control_plane_pin_cid",
        "active_plan_root_cid",
        "active_plan_revision",
        "active_plan_revision_cid",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "process_birth_nonce",
        "expected_process_uid",
        "expected_parent_pid",
        "expected_parent_process_start_time_ticks",
        "expected_executable_sha256",
        "launch_argv_cid",
    }
)
_UNSIGNED_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        *_BINDING_FIELDS,
        "native_dependency_pin",
        "native_dependency_pin_cid",
        "sealed_descriptor_required",
        "ambient_loader_environment_allowed",
        "raw_path_authority",
        "launch_authority_granted",
        "admission_outcome",
        "issued_at_ms",
        "expires_at_ms",
        "issuance_nonce",
        "reviewer_did",
        "reviewer_role",
    }
)
_SEALED_FIELDS: Final = _UNSIGNED_FIELDS | {"reviewer_signature", "admission_cid"}


class AgentSupervisorNativeDependencyAdmissionError(ValueError):
    """The native admission is malformed, untrusted, stale, or mismatched."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission is not canonical JSON"
        ) from exc


def _detached(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("ascii"))


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: object, noun: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AgentSupervisorNativeDependencyAdmissionError(f"{noun} is invalid")
    return value


def _git(value: object, noun: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT.fullmatch(value) is None:
        raise AgentSupervisorNativeDependencyAdmissionError(f"{noun} is invalid")
    return value


def _identifier(value: object, noun: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise AgentSupervisorNativeDependencyAdmissionError(f"{noun} is invalid")
    return value


def _positive(value: object, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgentSupervisorNativeDependencyAdmissionError(f"{noun} is invalid")
    return value


def _nonnegative(value: object, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentSupervisorNativeDependencyAdmissionError(f"{noun} is invalid")
    return value


def _did(value: object, noun: str) -> str:
    if not isinstance(value, str) or _DID_KEY.fullmatch(value) is None:
        raise AgentSupervisorNativeDependencyAdmissionError(f"{noun} is invalid")
    return value


def agent_supervisor_native_dependency_admission_relative_path(
    source_head: str,
    active_plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    *,
    registry_prefix: str = "authority/eaaef",
) -> Path:
    """Return the deterministic, relative location of one signed admission."""

    head = _git(source_head, "source_head")
    root = _sha(active_plan_root_cid, "active_plan_root_cid").split(":", 1)[1]
    lane = _identifier(lane_session_id, "lane_session_id")
    generation = _positive(lane_generation, "lane_generation")
    prefix = Path(registry_prefix)
    if (
        prefix.is_absolute()
        or not prefix.parts
        or any(part in {"", ".", ".."} for part in prefix.parts)
        or any(_SAFE_ID.fullmatch(part) is None for part in prefix.parts)
    ):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission registry prefix is invalid"
        )
    lane_digest = hashlib.sha256(lane.encode("ascii")).hexdigest()
    return prefix / (
        f"agent-supervisor-native-dependency-admission--{head}--{root}--"
        f"{lane_digest}--g{generation}.json"
    )


def seal_agent_supervisor_native_dependency_admission(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    """Attach one signature and content identity without granting authority."""

    if not isinstance(statement, Mapping) or set(statement) != _UNSIGNED_FIELDS:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission statement shape is invalid"
        )
    if not isinstance(reviewer_signature, str) or not reviewer_signature:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission signature is invalid"
        )
    sealed = {**_detached(dict(statement)), "reviewer_signature": reviewer_signature}
    sealed["admission_cid"] = _cid(sealed)
    return MappingProxyType(sealed)


class VerifiedAgentSupervisorNativeDependencyAdmission(Mapping[str, Any]):
    """Exact immutable result of signature, pin, and birth-binding verification."""

    __slots__ = ("_value", "_pin", "_coordinates")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        pin: AgentSupervisorNativeDependencyPin,
        coordinates: Mapping[str, Any] | None = None,
    ) -> None:
        if token is not _VERIFIED_TOKEN:
            raise TypeError(
                "verified native dependency admissions come from the exact verifier"
            )
        self._value = MappingProxyType(_detached(dict(value)))
        self._pin = pin
        self._coordinates = MappingProxyType(dict(coordinates or {}))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        return _detached(value) if isinstance(value, (dict, list)) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def native_dependency_pin(self) -> AgentSupervisorNativeDependencyPin:
        return self._pin

    @property
    def admission_cid(self) -> str:
        """Return the signed identity consumed by the v2 lane admission."""

        return str(self._value["admission_cid"])

    def to_dict(self) -> dict[str, Any]:
        return _detached(dict(self._value))

    def reverify(self, *, now_ms: int) -> VerifiedAgentSupervisorNativeDependencyAdmission:
        if not self._coordinates:
            raise AgentSupervisorNativeDependencyAdmissionError(
                "native admission has no retained source coordinates"
            )
        arguments = dict(self._coordinates)
        arguments["now_ms"] = _positive(now_ms, "now_ms")
        checked = load_and_verify_agent_supervisor_native_dependency_admission(
            **arguments
        )
        if checked.to_dict() != self.to_dict():
            raise AgentSupervisorNativeDependencyAdmissionError(
                "source-reverified native admission changed"
            )
        return checked


def verify_agent_supervisor_native_dependency_admission(
    value: Mapping[str, Any],
    *,
    trusted_reviewer_dids: Sequence[str],
    expected_native_dependency_pin: AgentSupervisorNativeDependencyPin,
    expected_bindings: Mapping[str, Any],
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedAgentSupervisorNativeDependencyAdmission:
    """Verify one exact native pin admission for one planned process birth."""

    now = _positive(now_ms, "now_ms")
    if not isinstance(value, Mapping) or set(value) != _SEALED_FIELDS:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission shape is invalid"
        )
    if not isinstance(expected_bindings, Mapping) or set(expected_bindings) != _BINDING_FIELDS:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "expected native dependency bindings are incomplete"
        )
    if type(expected_native_dependency_pin) is not AgentSupervisorNativeDependencyPin:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "expected native dependency pin must be the exact closed type"
        )
    admitted = _detached(dict(value))
    expected = _detached(dict(expected_bindings))
    if any(admitted.get(field) != expected[field] for field in _BINDING_FIELDS):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission birth binding differs"
        )
    if (
        admitted.get("schema") != AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_SCHEMA
        or admitted.get("interface")
        != AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_INTERFACE
        or admitted.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or admitted.get("admission_outcome") != "admitted"
        or admitted.get("sealed_descriptor_required") is not True
        or admitted.get("ambient_loader_environment_allowed") is not False
        or admitted.get("raw_path_authority") is not False
        or admitted.get("launch_authority_granted") is not False
    ):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission policy is invalid"
        )
    _git(admitted.get("source_head"), "source_head")
    _git(admitted.get("source_tree"), "source_tree")
    for name in (
        "configuration_root",
        "accepted_control_plane_capsule_id",
        "accepted_control_plane_pin_cid",
        "active_plan_root_cid",
        "active_plan_revision_cid",
        "slice_manifest_cid",
        "expected_executable_sha256",
        "launch_argv_cid",
        "native_dependency_pin_cid",
    ):
        _sha(admitted.get(name), name)
    for name in (
        "slice_id",
        "lane_id",
        "lane_session_id",
        "process_instance_id",
        "process_birth_nonce",
        "issuance_nonce",
    ):
        _identifier(admitted.get(name), name)
    for name in (
        "active_plan_revision",
        "lane_generation",
        "expected_parent_pid",
        "expected_parent_process_start_time_ticks",
    ):
        _positive(admitted.get(name), name)
    _nonnegative(admitted.get("expected_process_uid"), "expected_process_uid")
    issued = _positive(admitted.get("issued_at_ms"), "issued_at_ms")
    expires = _positive(admitted.get("expires_at_ms"), "expires_at_ms")
    if issued > now or now >= expires or issued >= expires or expires - issued > _MAX_LIFETIME_MS:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission lifetime is invalid"
        )
    reviewer = _did(admitted.get("reviewer_did"), "reviewer_did")
    trusted = tuple(_did(item, "trusted reviewer DID") for item in trusted_reviewer_dids)
    forbidden = frozenset(
        _did(item, "forbidden reviewer DID") for item in forbidden_reviewer_dids
    )
    if (
        not trusted
        or len(set(trusted)) != len(trusted)
        or reviewer not in trusted
        or reviewer in forbidden
        or admitted.get("reviewer_role")
        != AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_REVIEWER_ROLE
    ):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission reviewer is unauthorized"
        )
    claimed = admitted.get("admission_cid")
    signed = dict(admitted)
    signed.pop("admission_cid", None)
    signature = signed.pop("reviewer_signature", None)
    if claimed != _cid({**signed, "reviewer_signature": signature}):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission self-address is invalid"
        )
    if not isinstance(signature, str) or not signature:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission is unsigned"
        )
    try:
        verify_did_key_signature(
            identity_did=reviewer,
            payload=signed,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission signature is invalid"
        ) from exc
    try:
        observed_pin = parse_agent_supervisor_native_dependency_pin(
            admitted.get("native_dependency_pin")
        )
        expected_pin = parse_agent_supervisor_native_dependency_pin(
            expected_native_dependency_pin.as_dict()
        )
    except ValueError as exc:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency pin is invalid"
        ) from exc
    if (
        observed_pin.as_dict() != expected_pin.as_dict()
        or admitted["native_dependency_pin_cid"] != observed_pin.dependency_id
        or admitted["expected_executable_sha256"]
        != observed_pin.python_executable_sha256
    ):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native dependency admission does not bind the exact pin"
        )
    return VerifiedAgentSupervisorNativeDependencyAdmission(
        _VERIFIED_TOKEN,
        admitted,
        observed_pin,
    )


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AgentSupervisorNativeDependencyAdmissionError(
                "native admission JSON contains a duplicate key"
            )
        result[key] = value
    return result


def _open_canonical_source(repo_root: Path, relative: Path) -> tuple[dict[str, Any], str]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission no-follow source reads are unavailable"
        )
    root = Path(repo_root)
    if not root.is_absolute() or ".." in root.parts or relative.is_absolute():
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source root is invalid"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(root.anchor, directory_flags)
    try:
        for component in root.parts[1:]:
            if component in {"", ".", ".."}:
                raise AgentSupervisorNativeDependencyAdmissionError(
                    "native admission source path is invalid"
                )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.geteuid()
            or status.st_mode & 0o022
        ):
            raise AgentSupervisorNativeDependencyAdmissionError(
                "native admission source root is not owner-controlled"
            )
        for component in relative.parts[:-1]:
            if component in {"", ".", ".."}:
                raise AgentSupervisorNativeDependencyAdmissionError(
                    "native admission source path is invalid"
                )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or status.st_mode & 0o022
            ):
                raise AgentSupervisorNativeDependencyAdmissionError(
                    "native admission source directory is not owner-controlled"
                )
        leaf = os.open(relative.name, flags, dir_fd=descriptor)
        try:
            before = os.fstat(leaf)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o077
                or not 0 < before.st_size <= _MAX_FILE_BYTES
            ):
                raise AgentSupervisorNativeDependencyAdmissionError(
                    "native admission source file is not stable"
                )
            raw = b""
            while len(raw) < before.st_size:
                chunk = os.pread(leaf, before.st_size - len(raw), len(raw))
                if not chunk:
                    break
                raw += chunk
            after = os.fstat(leaf)
            named = os.stat(
                relative.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        finally:
            os.close(leaf)
    except OSError as exc:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source is unavailable"
        ) from exc
    finally:
        os.close(descriptor)
    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_gid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(after) != identity(named)
    ):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source changed while reading"
        )
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source is invalid JSON"
        ) from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source is not canonical"
        )
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def load_and_verify_agent_supervisor_native_dependency_admission(
    repo_root: Path,
    *,
    source_head: str,
    active_plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    registry_prefix: str,
    expected_file_sha256: str,
    trusted_reviewer_dids: Sequence[str],
    expected_native_dependency_pin: AgentSupervisorNativeDependencyPin,
    expected_bindings: Mapping[str, Any],
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedAgentSupervisorNativeDependencyAdmission:
    """No-follow load, hash-pin, and fully verify one native admission file."""

    relative = agent_supervisor_native_dependency_admission_relative_path(
        source_head,
        active_plan_root_cid,
        lane_session_id,
        lane_generation,
        registry_prefix=registry_prefix,
    )
    value, observed_sha = _open_canonical_source(Path(repo_root), relative)
    if observed_sha != _sha(expected_file_sha256, "expected_file_sha256"):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source hash differs"
        )
    checked = verify_agent_supervisor_native_dependency_admission(
        value,
        trusted_reviewer_dids=trusted_reviewer_dids,
        expected_native_dependency_pin=expected_native_dependency_pin,
        expected_bindings=expected_bindings,
        now_ms=now_ms,
        forbidden_reviewer_dids=forbidden_reviewer_dids,
    )
    if (
        checked["source_head"] != source_head
        or checked["active_plan_root_cid"] != active_plan_root_cid
        or checked["lane_session_id"] != lane_session_id
        or checked["lane_generation"] != lane_generation
    ):
        raise AgentSupervisorNativeDependencyAdmissionError(
            "native admission source coordinates differ"
        )
    coordinates = {
        "repo_root": Path(repo_root),
        "source_head": source_head,
        "active_plan_root_cid": active_plan_root_cid,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "registry_prefix": registry_prefix,
        "expected_file_sha256": expected_file_sha256,
        "trusted_reviewer_dids": tuple(trusted_reviewer_dids),
        "expected_native_dependency_pin": expected_native_dependency_pin,
        "expected_bindings": _detached(dict(expected_bindings)),
        "forbidden_reviewer_dids": tuple(forbidden_reviewer_dids),
    }
    return VerifiedAgentSupervisorNativeDependencyAdmission(
        _VERIFIED_TOKEN,
        checked.to_dict(),
        checked.native_dependency_pin,
        coordinates,
    )


__all__ = [
    "AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_INTERFACE",
    "AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_PATH_TEMPLATE",
    "AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_REVIEWER_ROLE",
    "AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_SCHEMA",
    "AgentSupervisorNativeDependencyAdmissionError",
    "VerifiedAgentSupervisorNativeDependencyAdmission",
    "agent_supervisor_native_dependency_admission_relative_path",
    "load_and_verify_agent_supervisor_native_dependency_admission",
    "seal_agent_supervisor_native_dependency_admission",
    "verify_agent_supervisor_native_dependency_admission",
]
