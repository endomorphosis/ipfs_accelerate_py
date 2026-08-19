"""Source-addressed per-birth admission for the EAAEF bootstrap gateway.

The bootstrap operational capability authorizes one board owner.  It does not
authorize an arbitrary child process to use that owner's command ingress.  A
lane therefore needs a short-lived, signed, per-process-birth authority plus
two independent pieces of evidence: a verifier receipt for the exact lane and
a merge admission for the exact reviewed runtime sources.

The positive result in this module can only be produced by loading four
hash-pinned, canonical files (the operational capability and the three lane
records) through no-follow ``openat`` walks.  The result retains all source
coordinates and re-opens and re-verifies them on demand.  It contains no
secret, command callback, database path, or signing key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ...llm_router import parse_agent_supervisor_native_dependency_pin
from ..control.profile_authority import LocalProfileTampered, verify_did_key_signature
from .agent_native_dependency_admission import (
    AgentSupervisorNativeDependencyAdmissionError,
    VerifiedAgentSupervisorNativeDependencyAdmission,
    load_and_verify_agent_supervisor_native_dependency_admission,
)
from .eaaef_bootstrap_gateway_launch import (
    EAAEF_BOARD_NAMESPACE,
    EAAEFBootstrapGatewayLaunchError,
    VerifiedEAAEFBootstrapOperationalCapability,
    load_eaaef_bootstrap_operational_capability,
    verify_eaaef_bootstrap_operational_capability,
)

EAAEF_LANE_AUTHORITY_INTERFACE: Final = "EAAEFBootstrapLaneAuthority@1"
EAAEF_LANE_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-lane-authority@1"
)
EAAEF_LANE_AUTHORITY_REVIEW_ROLE: Final = "independent_eaaef_lane_authority_reviewer"
EAAEF_LANE_VERIFIER_RECEIPT_INTERFACE: Final = "EAAEFBootstrapLaneVerifierReceipt@1"
EAAEF_LANE_VERIFIER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-lane-verifier-receipt@1"
)
EAAEF_LANE_VERIFIER_REVIEW_ROLE: Final = "independent_eaaef_lane_runtime_verifier"
EAAEF_LANE_MERGE_ADMISSION_INTERFACE: Final = "EAAEFBootstrapLaneMergeAdmission@1"
EAAEF_LANE_MERGE_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-lane-merge-admission@1"
)
EAAEF_LANE_MERGE_ADMISSION_REVIEW_ROLE: Final = "independent_eaaef_lane_merge_admission_reviewer"
EAAEF_LANE_AUTHORITY_V2_INTERFACE: Final = "EAAEFBootstrapLaneAuthority@2"
EAAEF_LANE_AUTHORITY_V2_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-lane-authority@2"
)
EAAEF_LANE_VERIFIER_RECEIPT_V2_INTERFACE: Final = (
    "EAAEFBootstrapLaneVerifierReceipt@2"
)
EAAEF_LANE_VERIFIER_RECEIPT_V2_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-lane-verifier-receipt@2"
)
EAAEF_LANE_MERGE_ADMISSION_V2_INTERFACE: Final = (
    "EAAEFBootstrapLaneMergeAdmission@2"
)
EAAEF_LANE_MERGE_ADMISSION_V2_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-lane-merge-admission@2"
)
EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_INTERFACE: Final = (
    "EAAEFQuackClientFactoryQualification@1"
)
EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-quack-client-factory-qualification@1"
)
EAAEF_QUACK_CLIENT_FACTORY_REVIEW_ROLE: Final = (
    "independent_eaaef_quack_client_factory_reviewer"
)
EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_INTERFACE: Final = (
    "EAAEFContainerDispatcherFactoryQualification@1"
)
EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-container-dispatcher-factory-qualification@1"
)
EAAEF_CONTAINER_DISPATCHER_FACTORY_REVIEW_ROLE: Final = (
    "independent_eaaef_container_dispatcher_factory_reviewer"
)
EAAEF_CONTAINER_DYNAMIC_SERVICE_INTERFACE: Final = "EAAEFContainerDynamicService@1"
EAAEF_LANE_AUTHORITY_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-lane-authority--<source_head>--<plan_root_sha256>--"
    "<lane_session_sha256>--g<lane_generation>.json"
)
EAAEF_LANE_VERIFIER_RECEIPT_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-lane-verifier-receipt--<lane_authority_sha256>.json"
)
EAAEF_LANE_MERGE_ADMISSION_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-lane-merge-admission--<lane_authority_sha256>.json"
)
EAAEF_LANE_AUTHORITY_V2_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-lane-authority-v2--<source_head>--<plan_root_sha256>--"
    "<lane_session_sha256>--g<lane_generation>.json"
)
EAAEF_LANE_VERIFIER_RECEIPT_V2_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-lane-verifier-receipt-v2--<lane_authority_sha256>.json"
)
EAAEF_LANE_MERGE_ADMISSION_V2_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-lane-merge-admission-v2--<lane_authority_sha256>.json"
)
EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_PATH_TEMPLATE: Final = (
    "eaaef-quack-client-factory-qualification--<source_head>--"
    "<lane_session_sha256>--g<lane_generation>.json"
)
EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_PATH_TEMPLATE: Final = (
    "eaaef-container-dispatcher-factory-qualification--<source_head>--"
    "<lane_session_sha256>--g<lane_generation>.json"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_MAX_FILE_BYTES = 256 * 1024
_MAX_LIFETIME_MS = 15 * 60 * 1000
_LANE_ROLE = "database_implementation_daemon"
_VERIFIER_CHECKS: Final = (
    "exact_birth_identity",
    "exact_operational_capability_join",
    "exact_source_hashes",
    "no_callback_dispatch",
    "no_direct_database",
    "no_raw_token",
)
_VERIFIER_CHECKS_V2: Final = (
    *_VERIFIER_CHECKS,
    "exact_plan_revision_slice_lane",
    "exact_slice_task_population",
    "expected_parent_process_birth",
    "expected_child_executable",
    "exact_launch_argv",
)

_LANE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "source_head",
        "source_tree",
        "active_plan_root_cid",
        "active_plan_revision",
        "operational_capability_cid",
        "operational_capability_file_sha256",
        "gateway_binding_cid",
        "owner_principal_did",
        "owner_session_id",
        "owner_generation",
        "fence_epoch",
        "lane_principal_did",
        "lane_role",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "process_birth_nonce",
        "journal_namespace",
        "direct_database_open",
        "arbitrary_sql_enabled",
        "callback_dispatch_enabled",
        "raw_token_available",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "authority_cid",
    }
)
_VERIFIER_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "lane_authority_cid",
        "lane_authority_file_sha256",
        "operational_capability_cid",
        "gateway_binding_cid",
        "source_head",
        "source_tree",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "process_birth_nonce",
        "runtime_gateway_source_sha256",
        "lane_validation_source_sha256",
        "checks",
        "verification_outcome",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "receipt_cid",
    }
)
_MERGE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "lane_authority_cid",
        "lane_authority_file_sha256",
        "verifier_receipt_cid",
        "verifier_receipt_file_sha256",
        "operational_capability_cid",
        "operational_capability_file_sha256",
        "gateway_binding_cid",
        "source_head",
        "source_tree",
        "active_plan_root_cid",
        "active_plan_revision",
        "operation_vocabulary_cid",
        "command_fabric_qualification_cid",
        "borrowed_transaction_adapter_qualification_cid",
        "runtime_gateway_source_sha256",
        "lane_validation_source_sha256",
        "admission_outcome",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "admission_cid",
    }
)

_BIRTH_PLAN_FIELDS: Final = frozenset(
    {
        "active_plan_revision_cid",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "task_ids",
        "task_cids",
        "expected_process_uid",
        "expected_parent_pid",
        "expected_parent_process_start_time_ticks",
        "expected_executable_sha256",
        "launch_argv_cid",
        "native_dependency_admission_cid",
        "native_dependency_admission_file_sha256",
        "quack_client_factory_qualification_cid",
        "quack_client_factory_qualification_file_sha256",
        "container_dispatcher_factory_qualification_cid",
        "container_dispatcher_factory_qualification_file_sha256",
        "command_secret_descriptor_sha256",
        "command_secret_generation",
        "state_secret_descriptor_sha256",
        "state_secret_generation",
    }
)
_LANE_V2_FIELDS: Final = _LANE_FIELDS | _BIRTH_PLAN_FIELDS
_VERIFIER_V2_FIELDS: Final = _VERIFIER_FIELDS | _BIRTH_PLAN_FIELDS
_MERGE_V2_FIELDS: Final = _MERGE_FIELDS | _BIRTH_PLAN_FIELDS
_SOURCE_COORDINATE_ARGUMENT_FIELDS: Final = frozenset(
    {
        "source_head",
        "plan_root_cid",
        "lane_session_id",
        "lane_generation",
        "registry_prefix",
        "operational_capability_registry_prefix",
        "expected_operational_capability_file_sha256",
        "expected_lane_authority_file_sha256",
        "expected_verifier_receipt_file_sha256",
        "expected_merge_admission_file_sha256",
        "trusted_operational_reviewer_dids",
        "trusted_authorization_service_reviewer_dids",
        "trusted_lane_authority_reviewer_dids",
        "trusted_lane_verifier_reviewer_dids",
        "trusted_lane_merge_reviewer_dids",
        "expected_operational_bindings",
        "forbidden_reviewer_dids",
        "artifact_version",
    }
)
_NATIVE_SOURCE_COORDINATE_FIELDS: Final = frozenset(
    {
        "source_head",
        "active_plan_root_cid",
        "lane_session_id",
        "lane_generation",
        "registry_prefix",
        "expected_file_sha256",
        "trusted_reviewer_dids",
        "expected_native_dependency_pin",
        "expected_bindings",
        "forbidden_reviewer_dids",
    }
)
_QUALIFICATION_SOURCE_COORDINATE_FIELDS: Final = frozenset(
    {"registry_prefix", "expected_file_sha256", "trusted_reviewer_dids"}
)
_QUACK_CLIENT_FACTORY_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "source_head",
        "source_tree",
        "active_plan_root_cid",
        "active_plan_revision",
        "active_plan_revision_cid",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "task_ids",
        "task_cids",
        "operational_capability_cid",
        "gateway_binding_cid",
        "command_fabric_qualification_cid",
        "native_dependency_admission_cid",
        "native_dependency_admission_file_sha256",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "process_birth_nonce",
        "command_endpoint",
        "command_secret_handle",
        "command_secret_generation",
        "command_secret_descriptor_sha256",
        "state_endpoint",
        "state_secret_handle",
        "state_secret_generation",
        "state_secret_descriptor_sha256",
        "quack_extension_relative_path",
        "quack_extension_sha256",
        "secret_resolver_mode",
        "raw_token_argv_enabled",
        "raw_token_environment_enabled",
        "raw_token_path_enabled",
        "runtime_gateway_source_sha256",
        "lane_validation_source_sha256",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "qualification_cid",
    }
)
_CONTAINER_DISPATCHER_FACTORY_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "source_head",
        "source_tree",
        "active_plan_root_cid",
        "active_plan_revision",
        "active_plan_revision_cid",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "task_ids",
        "task_cids",
        "operational_capability_cid",
        "gateway_binding_cid",
        "native_dependency_admission_cid",
        "native_dependency_admission_file_sha256",
        "quack_client_factory_qualification_cid",
        "quack_client_factory_qualification_file_sha256",
        "lane_session_id",
        "lane_generation",
        "process_instance_id",
        "process_birth_nonce",
        "worker_principal_did",
        "services",
        "dispatcher_source_sha256",
        "runtime_gateway_source_sha256",
        "lane_validation_source_sha256",
        "dynamic_per_attempt_verification_required",
        "dynamic_per_attempt_merge_admission_required",
        "static_lane_evidence_is_attempt_success",
        "caller_callbacks_allowed",
        "direct_container_launch_allowed",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "qualification_cid",
    }
)
_DYNAMIC_SERVICE_FIELDS: Final = frozenset(
    {
        "interface",
        "endpoint",
        "service_principal_did",
        "expected_server_uid",
        "expected_server_pid",
        "expected_server_process_start_time_ticks",
        "methods",
        "peer_credentials_required",
        "response_signature_verification_required",
        "request_lane_reverification_required",
        "maximum_request_bytes",
        "maximum_response_bytes",
        "request_timeout_ms",
    }
)
_DYNAMIC_SERVICE_METHODS: Final = MappingProxyType(
    {
        "worker": ("packet", "qualify", "launch"),
        "verifier": ("verify",),
        "merge": ("observe_merge",),
        "host_source": ("observe_source",),
    }
)


class EAAEFLaneGatewayAdmissionError(EAAEFBootstrapGatewayLaunchError):
    """A lane authority or one of its source/evidence joins was invalid."""


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
        raise EAAEFLaneGatewayAdmissionError("value is not canonical JSON") from exc


def _detached(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(_canonical_bytes(dict(value)).decode("ascii"))
    if not isinstance(result, dict):  # pragma: no cover - mapping input is an object.
        raise EAAEFLaneGatewayAdmissionError("record is not an object")
    return result


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: object, noun: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not a full sha256 identity")
    return text


def _git(value: object, noun: str) -> str:
    text = str(value or "")
    if _GIT_OBJECT.fullmatch(text) is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not a Git object identity")
    return text


def _identifier(value: object, noun: str) -> str:
    text = str(value or "")
    if _SAFE_ID.fullmatch(text) is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not a bounded identifier")
    return text


def _service_endpoint(value: object, noun: str) -> str:
    endpoint = str(value or "")
    if not endpoint.startswith("unix:/") or "\x00" in endpoint:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} must be an absolute Unix endpoint")
    path = Path(endpoint.removeprefix("unix:"))
    if (
        not path.is_absolute()
        or path.parts[:2] != ("/", "run")
        or ".." in path.parts
        or len(os.fsencode(path)) > 100
    ):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} must be a bounded /run endpoint")
    return endpoint


def _did(value: object, noun: str) -> str:
    text = str(value or "")
    if not text.startswith("did:key:z") or len(text.encode("utf-8")) > 512:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not an Ed25519 did:key")
    return text


def _signature(value: object, noun: str) -> str:
    text = str(value or "")
    if not text or "\x00" in text or len(text.encode("utf-8")) > 512:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not a bounded detached signature")
    return text


def _positive(value: object, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} must be a positive integer")
    return value


def _nonnegative(value: object, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} must be a non-negative integer")
    return value


def _task_population(
    task_ids: object,
    task_cids: object,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if (
        not isinstance(task_ids, list)
        or not isinstance(task_cids, list)
        or not task_ids
        or len(task_ids) != len(task_cids)
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "lane task IDs/CIDs must be one exact nonempty population"
        )
    ids = tuple(_identifier(item, "task_id") for item in task_ids)
    cids = tuple(_identifier(item, "task_cid") for item in task_cids)
    if len(set(ids)) != len(ids) or len(set(cids)) != len(cids):
        raise EAAEFLaneGatewayAdmissionError("lane task IDs/CIDs must be unique")
    return ids, cids


def eaaef_launch_argv_cid(argv: Sequence[str]) -> str:
    """Return the exact ordered argv identity signed before ``Popen``."""

    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or any(
            not isinstance(item, str)
            or not item
            or "\x00" in item
            or len(item.encode("utf-8")) > 16_384
            for item in argv
        )
        or len(argv) > 1_024
    ):
        raise EAAEFLaneGatewayAdmissionError("launch argv is not an exact bounded vector")
    return _cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-launch-argv@1",
            "argv": list(argv),
        }
    )


def _linux_process_start_time_ticks(pid: int) -> int:
    checked = _positive(pid, "process pid")
    try:
        raw = Path(f"/proc/{checked}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        if close < 1 or len(fields) <= 19 or fields[0] == "Z":
            raise ValueError("malformed or zombie process stat")
        return _positive(int(fields[19]), "process start time ticks")
    except (OSError, UnicodeError, ValueError) as exc:
        raise EAAEFLaneGatewayAdmissionError("process birth is unavailable") from exc


def eaaef_current_executable_sha256() -> str:
    """Hash the kernel-selected current executable, not an argv path."""

    try:
        descriptor = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
    except OSError as exc:
        raise EAAEFLaneGatewayAdmissionError("current executable is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise EAAEFLaneGatewayAdmissionError("current executable is not regular")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _relative_prefix(registry_prefix: str) -> Path:
    prefix = Path(str(registry_prefix or ""))
    if prefix.is_absolute() or not prefix.parts or ".." in prefix.parts:
        raise EAAEFLaneGatewayAdmissionError(
            "lane authority registry prefix is not repository-relative"
        )
    return prefix


def _open_source_root(
    repo_root: str | Path,
    noun: str,
    *,
    unsafe_write_mask: int = 0o022,
) -> int:
    """Return a stable root descriptor reached without following any link."""

    root = Path(os.path.abspath(os.fspath(repo_root)))
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} requires nofollow openat")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    descriptor = os.open("/", flags)
    try:
        for part in root.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        pathname = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(pathname.st_mode)
            or metadata.st_dev != pathname.st_dev
            or metadata.st_ino != pathname.st_ino
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & unsafe_write_mask
        ):
            raise EAAEFLaneGatewayAdmissionError(f"{noun} ownership is unsafe")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validated_repo_root(repo_root: str | Path) -> Path:
    """Open every absolute root component without following a link."""

    root = Path(os.path.abspath(os.fspath(repo_root)))
    try:
        descriptor = _open_source_root(root, "lane authority source root")
    except OSError as exc:
        raise EAAEFLaneGatewayAdmissionError(
            "lane authority source root contains an unavailable component"
        ) from exc
    try:
        return root
    finally:
        os.close(descriptor)


def _lane_session_digest(lane_session_id: str) -> str:
    lane = _identifier(lane_session_id, "lane_session_id")
    return hashlib.sha256(lane.encode("utf-8")).hexdigest()


def eaaef_lane_authority_relative_path(
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    *,
    registry_prefix: str,
) -> Path:
    """Return the only accepted source path for one per-birth authority."""

    head = _git(source_head, "source_head")
    plan = _sha(plan_root_cid, "plan_root_cid").removeprefix("sha256:")
    generation = _positive(lane_generation, "lane_generation")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-bootstrap-lane-authority--{head}--{plan}--"
        f"{_lane_session_digest(lane_session_id)}--g{generation}.json"
    )


def eaaef_lane_verifier_receipt_relative_path(
    lane_authority_cid: str, *, registry_prefix: str
) -> Path:
    authority = _sha(lane_authority_cid, "lane_authority_cid").removeprefix("sha256:")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-bootstrap-lane-verifier-receipt--{authority}.json"
    )


def eaaef_lane_merge_admission_relative_path(
    lane_authority_cid: str, *, registry_prefix: str
) -> Path:
    authority = _sha(lane_authority_cid, "lane_authority_cid").removeprefix("sha256:")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-bootstrap-lane-merge-admission--{authority}.json"
    )


def eaaef_lane_authority_v2_relative_path(
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    *,
    registry_prefix: str,
) -> Path:
    """Return the only accepted source path for one v2 lane authority."""

    head = _git(source_head, "source_head")
    plan = _sha(plan_root_cid, "plan_root_cid").removeprefix("sha256:")
    generation = _positive(lane_generation, "lane_generation")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-bootstrap-lane-authority-v2--{head}--{plan}--"
        f"{_lane_session_digest(lane_session_id)}--g{generation}.json"
    )


def eaaef_lane_verifier_receipt_v2_relative_path(
    lane_authority_cid: str, *, registry_prefix: str
) -> Path:
    authority = _sha(lane_authority_cid, "lane_authority_cid").removeprefix("sha256:")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-bootstrap-lane-verifier-receipt-v2--{authority}.json"
    )


def eaaef_lane_merge_admission_v2_relative_path(
    lane_authority_cid: str, *, registry_prefix: str
) -> Path:
    authority = _sha(lane_authority_cid, "lane_authority_cid").removeprefix("sha256:")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-bootstrap-lane-merge-admission-v2--{authority}.json"
    )


def eaaef_quack_client_factory_qualification_relative_path(
    source_head: str,
    lane_session_id: str,
    lane_generation: int,
    *,
    registry_prefix: str,
) -> Path:
    """Return the source address for one planned-birth client qualification."""

    head = _git(source_head, "source_head")
    generation = _positive(lane_generation, "lane_generation")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-quack-client-factory-qualification--{head}--"
        f"{_lane_session_digest(lane_session_id)}--g{generation}.json"
    )


def eaaef_container_dispatcher_factory_qualification_relative_path(
    source_head: str,
    lane_session_id: str,
    lane_generation: int,
    *,
    registry_prefix: str,
) -> Path:
    head = _git(source_head, "source_head")
    generation = _positive(lane_generation, "lane_generation")
    return _relative_prefix(registry_prefix) / (
        f"eaaef-container-dispatcher-factory-qualification--{head}--"
        f"{_lane_session_digest(lane_session_id)}--g{generation}.json"
    )


def _load_source_record(
    repo_root: Path,
    relative: Path,
    *,
    expected_file_sha256: str,
    expected_fields: frozenset[str],
    noun: str,
) -> tuple[dict[str, Any], str]:
    """Read one canonical owner-only file without following registry links."""

    root_path = Path(os.path.abspath(os.fspath(repo_root)))
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} requires nofollow openat")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    file_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    try:
        directory_fd = _open_source_root(root_path, f"{noun} root")
    except OSError as exc:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} root is unavailable") from exc
    try:
        for part in relative.parts[:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise EAAEFLaneGatewayAdmissionError(f"{noun} parent is unavailable") from exc
            try:
                metadata = os.fstat(next_fd)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o022
                ):
                    raise EAAEFLaneGatewayAdmissionError(f"{noun} parent ownership is unsafe")
            except BaseException:
                os.close(next_fd)
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            descriptor = os.open(relative.name, file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise EAAEFLaneGatewayAdmissionError(f"{noun} is unavailable") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_FILE_BYTES
                or stat.S_IMODE(before.st_mode) & 0o077
            ):
                raise EAAEFLaneGatewayAdmissionError(f"{noun} is not an owner-only regular file")
            raw = b""
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                raw += chunk
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            pathname = os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)
    identity = lambda item: (  # noqa: E731 - immutable stat identity.
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(pathname)
        or stat.S_ISLNK(pathname.st_mode)
    ):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} changed during stable read")
    observed_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    if observed_sha != _sha(expected_file_sha256, f"expected {noun} file sha256"):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} file identity changed")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not JSON") from exc
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} shape is not exact")
    if raw != _canonical_bytes(value):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is not canonical JSON")
    return value, observed_sha


def _load_immutable_source_bytes(
    repo_root: Path,
    relative: Path,
    *,
    expected_file_sha256: str,
    noun: str,
) -> bytes:
    """Read one owner-controlled immutable binary through no-follow openat."""

    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} path is not repository-relative")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} requires nofollow openat")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    file_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    try:
        directory_fd = _open_source_root(repo_root, f"{noun} root")
    except OSError as exc:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} root is unavailable") from exc
    try:
        for part in relative.parts[:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise EAAEFLaneGatewayAdmissionError(f"{noun} parent is unavailable") from exc
            metadata = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                os.close(next_fd)
                raise EAAEFLaneGatewayAdmissionError(f"{noun} parent ownership is unsafe")
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            descriptor = os.open(relative.name, file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise EAAEFLaneGatewayAdmissionError(f"{noun} is unavailable") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or not 0 < before.st_size <= 256 * 1024 * 1024
                or stat.S_IMODE(before.st_mode) & 0o022
            ):
                raise EAAEFLaneGatewayAdmissionError(
                    f"{noun} is not an immutable owner-controlled regular file"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            pathname = os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)
    identity = lambda item: (  # noqa: E731 - immutable stat identity.
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(pathname)
        or stat.S_ISLNK(pathname.st_mode)
        or "sha256:" + hashlib.sha256(raw).hexdigest()
        != _sha(expected_file_sha256, f"expected {noun} sha256")
    ):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} changed or differs from its pin")
    return raw


def _source_sha(path: Path, noun: str) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} requires nofollow source access")
    try:
        parent_fd = _open_source_root(
            path.parent,
            f"{noun} parent",
            unsafe_write_mask=0o002,
        )
    except OSError as exc:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is unavailable") from exc
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | nofollow,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        os.close(parent_fd)
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or stat.S_IMODE(before.st_mode) & 0o002
        ):
            raise EAAEFLaneGatewayAdmissionError(f"{noun} is not a regular source file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        pathname = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    identity = lambda item: (  # noqa: E731 - immutable stat identity.
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(pathname)
        or stat.S_ISLNK(pathname.st_mode)
    ):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} changed during read")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def eaaef_lane_gateway_source_identities() -> Mapping[str, str]:
    """Return the exact validation/runtime source hashes reviewers must sign."""

    validation_path = Path(os.path.abspath(__file__))
    runtime_path = validation_path.parent.parent / "runtime" / "eaaef_bootstrap_gateway.py"
    return MappingProxyType(
        {
            "lane_validation_source_sha256": _source_sha(validation_path, "lane validation source"),
            "runtime_gateway_source_sha256": _source_sha(runtime_path, "runtime gateway source"),
        }
    )


def eaaef_container_dispatcher_source_sha256() -> str:
    """Return the exact dispatcher implementation source identity."""

    validation_path = Path(os.path.abspath(__file__))
    dispatcher_path = (
        validation_path.parent.parent
        / "todo_daemon"
        / "external_agent_container_dispatcher.py"
    )
    return _source_sha(dispatcher_path, "external container dispatcher source")


def _seal(
    statement: Mapping[str, Any],
    *,
    signature: str,
    fields: frozenset[str],
    signature_field: str,
    cid_field: str,
    noun: str,
) -> Mapping[str, Any]:
    if not isinstance(statement, Mapping) or set(statement) != (
        fields - {signature_field, cid_field}
    ):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} statement shape is invalid")
    plain = _detached(statement)
    signed = {**plain, signature_field: _signature(signature, f"{noun} signature")}
    result = {**signed, cid_field: _cid(signed)}
    return MappingProxyType(result)


def seal_eaaef_lane_authority(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_LANE_FIELDS,
        signature_field="reviewer_signature",
        cid_field="authority_cid",
        noun="lane authority",
    )


def seal_eaaef_lane_verifier_receipt(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_VERIFIER_FIELDS,
        signature_field="reviewer_signature",
        cid_field="receipt_cid",
        noun="lane verifier receipt",
    )


def seal_eaaef_lane_merge_admission(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_MERGE_FIELDS,
        signature_field="reviewer_signature",
        cid_field="admission_cid",
        noun="lane merge admission",
    )


def seal_eaaef_lane_authority_v2(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_LANE_V2_FIELDS,
        signature_field="reviewer_signature",
        cid_field="authority_cid",
        noun="v2 lane authority",
    )


def seal_eaaef_lane_verifier_receipt_v2(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_VERIFIER_V2_FIELDS,
        signature_field="reviewer_signature",
        cid_field="receipt_cid",
        noun="v2 lane verifier receipt",
    )


def seal_eaaef_lane_merge_admission_v2(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_MERGE_V2_FIELDS,
        signature_field="reviewer_signature",
        cid_field="admission_cid",
        noun="v2 lane merge admission",
    )


def seal_eaaef_quack_client_factory_qualification(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_QUACK_CLIENT_FACTORY_FIELDS,
        signature_field="reviewer_signature",
        cid_field="qualification_cid",
        noun="Quack client factory qualification",
    )


def seal_eaaef_container_dispatcher_factory_qualification(
    statement: Mapping[str, Any], *, reviewer_signature: str
) -> Mapping[str, Any]:
    return _seal(
        statement,
        signature=reviewer_signature,
        fields=_CONTAINER_DISPATCHER_FACTORY_FIELDS,
        signature_field="reviewer_signature",
        cid_field="qualification_cid",
        noun="container dispatcher factory qualification",
    )


def _verify_signature(
    value: Mapping[str, Any],
    *,
    reviewer_did: str,
    signature_field: str,
    cid_field: str,
    noun: str,
) -> None:
    signed = dict(value)
    claimed = str(signed.pop(cid_field, ""))
    signature = signed.pop(signature_field, None)
    if claimed != _cid({**signed, signature_field: signature}):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} self-address is invalid")
    if not isinstance(signature, str) or not signature:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} is unsigned")
    try:
        verify_did_key_signature(
            identity_did=reviewer_did,
            payload=signed,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise EAAEFLaneGatewayAdmissionError(f"{noun} reviewer signature is invalid") from exc


def _trusted_reviewer(value: object, trusted: Sequence[str], noun: str) -> str:
    reviewer = _did(value, f"{noun} reviewer_did")
    if reviewer not in frozenset(_did(item, f"trusted {noun} reviewer") for item in trusted):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} reviewer is untrusted")
    return reviewer


def _lifetime(value: Mapping[str, Any], now_ms: int, noun: str) -> tuple[int, int]:
    issued = _positive(value.get("issued_at_ms"), f"{noun} issued_at_ms")
    expires = _positive(value.get("expires_at_ms"), f"{noun} expires_at_ms")
    if (
        issued > now_ms
        or now_ms >= expires
        or issued >= expires
        or expires - issued > _MAX_LIFETIME_MS
    ):
        raise EAAEFLaneGatewayAdmissionError(f"{noun} lifetime is invalid")
    return issued, expires


def _verify_lane(
    lane: Mapping[str, Any],
    *,
    capability: VerifiedEAAEFBootstrapOperationalCapability,
    operational_file_sha: str,
    trusted_reviewers: Sequence[str],
    now_ms: int,
    artifact_version: int = 1,
) -> dict[str, Any]:
    value = _detached(lane)
    cap = dict(capability)
    service = cap.get("command_authorization_service")
    if not isinstance(service, Mapping):
        raise EAAEFLaneGatewayAdmissionError(
            "verified operational capability lost its signer service"
        )
    reviewer = _trusted_reviewer(value.get("reviewer_did"), trusted_reviewers, "lane authority")
    issued, expires = _lifetime(value, now_ms, "lane authority")
    lane_principal = _did(value.get("lane_principal_did"), "lane_principal_did")
    owner = _did(value.get("owner_principal_did"), "owner_principal_did")
    expected_schema = (
        EAAEF_LANE_AUTHORITY_V2_SCHEMA
        if artifact_version == 2
        else EAAEF_LANE_AUTHORITY_SCHEMA
    )
    expected_interface = (
        EAAEF_LANE_AUTHORITY_V2_INTERFACE
        if artifact_version == 2
        else EAAEF_LANE_AUTHORITY_INTERFACE
    )
    if (
        value.get("schema") != expected_schema
        or value.get("interface") != expected_interface
        or value.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or value.get("reviewer_role") != EAAEF_LANE_AUTHORITY_REVIEW_ROLE
        or value.get("lane_role") != _LANE_ROLE
        or value.get("direct_database_open") is not False
        or value.get("arbitrary_sql_enabled") is not False
        or value.get("callback_dispatch_enabled") is not False
        or value.get("raw_token_available") is not False
        or lane_principal != cap.get("command_principal_did")
        or owner != cap.get("owner_principal_did")
        or reviewer
        in {
            owner,
            lane_principal,
            cap.get("reviewer_did"),
            service.get("reviewer_did"),
            service.get("service_principal_did"),
            service.get("approver_principal_did"),
            service.get("authorized_client_principal_did"),
        }
        or value.get("source_head") != cap.get("source_head")
        or value.get("source_tree") != cap.get("source_tree")
        or value.get("active_plan_root_cid") != cap.get("active_plan_root_cid")
        or value.get("active_plan_revision") != cap.get("active_plan_revision")
        or value.get("operational_capability_cid") != cap.get("capability_cid")
        or value.get("operational_capability_file_sha256") != operational_file_sha
        or value.get("gateway_binding_cid") != cap.get("gateway_binding_cid")
        or value.get("owner_session_id") != cap.get("owner_session_id")
        or value.get("owner_generation") != cap.get("owner_generation")
        or value.get("fence_epoch") != cap.get("fence_epoch")
        or value.get("lane_session_id") == value.get("owner_session_id")
        or not int(cap["issued_at_ms"]) <= issued < expires <= int(cap["expires_at_ms"])
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "lane authority differs from the verified operational capability"
        )
    for field in (
        "lane_session_id",
        "process_instance_id",
        "process_birth_nonce",
        "journal_namespace",
    ):
        _identifier(value.get(field), field)
    if value["process_instance_id"] in {
        value["lane_session_id"],
        value["owner_session_id"],
    }:
        raise EAAEFLaneGatewayAdmissionError(
            "lane process identity is not distinct from session identity"
        )
    _positive(value.get("lane_generation"), "lane_generation")
    _git(value.get("source_head"), "source_head")
    _git(value.get("source_tree"), "source_tree")
    _sha(value.get("active_plan_root_cid"), "active_plan_root_cid")
    _sha(value.get("operational_capability_cid"), "operational_capability_cid")
    _sha(value.get("operational_capability_file_sha256"), "operational capability file sha256")
    _sha(value.get("gateway_binding_cid"), "gateway_binding_cid")
    if artifact_version == 2:
        for field in (
            "active_plan_revision_cid",
            "slice_manifest_cid",
            "native_dependency_admission_cid",
            "native_dependency_admission_file_sha256",
            "quack_client_factory_qualification_cid",
            "quack_client_factory_qualification_file_sha256",
            "container_dispatcher_factory_qualification_cid",
            "container_dispatcher_factory_qualification_file_sha256",
        ):
            _sha(value.get(field), field)
        for field in ("slice_id", "lane_id"):
            _identifier(value.get(field), field)
        _task_population(value.get("task_ids"), value.get("task_cids"))
        _nonnegative(value.get("expected_process_uid"), "expected_process_uid")
        _positive(value.get("expected_parent_pid"), "expected_parent_pid")
        _positive(
            value.get("expected_parent_process_start_time_ticks"),
            "expected_parent_process_start_time_ticks",
        )
        _sha(value.get("expected_executable_sha256"), "expected_executable_sha256")
        _sha(value.get("launch_argv_cid"), "launch_argv_cid")
        _sha(
            value.get("command_secret_descriptor_sha256"),
            "command_secret_descriptor_sha256",
        )
        _sha(
            value.get("state_secret_descriptor_sha256"),
            "state_secret_descriptor_sha256",
        )
        _positive(value.get("command_secret_generation"), "command_secret_generation")
        _positive(value.get("state_secret_generation"), "state_secret_generation")
    _verify_signature(
        value,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="authority_cid",
        noun="lane authority",
    )
    return value


def _verify_verifier(
    receipt: Mapping[str, Any],
    *,
    lane: Mapping[str, Any],
    lane_file_sha: str,
    source_ids: Mapping[str, str],
    trusted_reviewers: Sequence[str],
    forbidden_dids: frozenset[str],
    now_ms: int,
    artifact_version: int = 1,
) -> dict[str, Any]:
    value = _detached(receipt)
    reviewer = _trusted_reviewer(value.get("reviewer_did"), trusted_reviewers, "lane verifier")
    issued, expires = _lifetime(value, now_ms, "lane verifier receipt")
    joins = {
        "board_namespace": lane["board_namespace"],
        "lane_authority_cid": lane["authority_cid"],
        "lane_authority_file_sha256": lane_file_sha,
        "operational_capability_cid": lane["operational_capability_cid"],
        "gateway_binding_cid": lane["gateway_binding_cid"],
        "source_head": lane["source_head"],
        "source_tree": lane["source_tree"],
        "lane_session_id": lane["lane_session_id"],
        "lane_generation": lane["lane_generation"],
        "process_instance_id": lane["process_instance_id"],
        "process_birth_nonce": lane["process_birth_nonce"],
        **dict(source_ids),
    }
    if artifact_version == 2:
        joins.update({name: lane[name] for name in _BIRTH_PLAN_FIELDS})
    expected_schema = (
        EAAEF_LANE_VERIFIER_RECEIPT_V2_SCHEMA
        if artifact_version == 2
        else EAAEF_LANE_VERIFIER_RECEIPT_SCHEMA
    )
    expected_interface = (
        EAAEF_LANE_VERIFIER_RECEIPT_V2_INTERFACE
        if artifact_version == 2
        else EAAEF_LANE_VERIFIER_RECEIPT_INTERFACE
    )
    expected_checks = list(_VERIFIER_CHECKS_V2 if artifact_version == 2 else _VERIFIER_CHECKS)
    if (
        value.get("schema") != expected_schema
        or value.get("interface") != expected_interface
        or value.get("reviewer_role") != EAAEF_LANE_VERIFIER_REVIEW_ROLE
        or value.get("verification_outcome") != "admitted"
        or value.get("checks") != expected_checks
        or reviewer in forbidden_dids
        or not int(lane["issued_at_ms"]) <= issued < expires <= int(lane["expires_at_ms"])
        or any(value.get(name) != expected for name, expected in joins.items())
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "lane verifier receipt is not independently and exactly bound"
        )
    _verify_signature(
        value,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="receipt_cid",
        noun="lane verifier receipt",
    )
    return value


def _verify_merge(
    admission: Mapping[str, Any],
    *,
    lane: Mapping[str, Any],
    lane_file_sha: str,
    verifier: Mapping[str, Any],
    verifier_file_sha: str,
    capability: VerifiedEAAEFBootstrapOperationalCapability,
    source_ids: Mapping[str, str],
    trusted_reviewers: Sequence[str],
    forbidden_dids: frozenset[str],
    now_ms: int,
    artifact_version: int = 1,
) -> dict[str, Any]:
    value = _detached(admission)
    reviewer = _trusted_reviewer(
        value.get("reviewer_did"), trusted_reviewers, "lane merge admission"
    )
    issued, expires = _lifetime(value, now_ms, "lane merge admission")
    cap = dict(capability)
    joins = {
        "board_namespace": lane["board_namespace"],
        "lane_authority_cid": lane["authority_cid"],
        "lane_authority_file_sha256": lane_file_sha,
        "verifier_receipt_cid": verifier["receipt_cid"],
        "verifier_receipt_file_sha256": verifier_file_sha,
        "operational_capability_cid": lane["operational_capability_cid"],
        "operational_capability_file_sha256": lane["operational_capability_file_sha256"],
        "gateway_binding_cid": lane["gateway_binding_cid"],
        "source_head": lane["source_head"],
        "source_tree": lane["source_tree"],
        "active_plan_root_cid": lane["active_plan_root_cid"],
        "active_plan_revision": lane["active_plan_revision"],
        "operation_vocabulary_cid": cap["operation_vocabulary_cid"],
        "command_fabric_qualification_cid": cap["command_fabric_qualification_cid"],
        "borrowed_transaction_adapter_qualification_cid": cap[
            "borrowed_transaction_adapter_qualification_cid"
        ],
        **dict(source_ids),
    }
    if artifact_version == 2:
        joins.update({name: lane[name] for name in _BIRTH_PLAN_FIELDS})
    expected_schema = (
        EAAEF_LANE_MERGE_ADMISSION_V2_SCHEMA
        if artifact_version == 2
        else EAAEF_LANE_MERGE_ADMISSION_SCHEMA
    )
    expected_interface = (
        EAAEF_LANE_MERGE_ADMISSION_V2_INTERFACE
        if artifact_version == 2
        else EAAEF_LANE_MERGE_ADMISSION_INTERFACE
    )
    if (
        value.get("schema") != expected_schema
        or value.get("interface") != expected_interface
        or value.get("reviewer_role") != EAAEF_LANE_MERGE_ADMISSION_REVIEW_ROLE
        or value.get("admission_outcome") != "admitted"
        or reviewer in forbidden_dids
        or not max(int(lane["issued_at_ms"]), int(verifier["issued_at_ms"]))
        <= issued
        < expires
        <= min(int(lane["expires_at_ms"]), int(verifier["expires_at_ms"]))
        or any(value.get(name) != expected for name, expected in joins.items())
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "lane merge admission is not independently and exactly bound"
        )
    _verify_signature(
        value,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="admission_cid",
        noun="lane merge admission",
    )
    return value


def _load_and_verify(
    *,
    repo_root: Path,
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    registry_prefix: str,
    operational_capability_registry_prefix: str,
    expected_operational_capability_file_sha256: str,
    expected_lane_authority_file_sha256: str,
    expected_verifier_receipt_file_sha256: str,
    expected_merge_admission_file_sha256: str,
    trusted_operational_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    trusted_lane_authority_reviewer_dids: Sequence[str],
    trusted_lane_verifier_reviewer_dids: Sequence[str],
    trusted_lane_merge_reviewer_dids: Sequence[str],
    expected_operational_bindings: Mapping[str, Any],
    forbidden_reviewer_dids: Sequence[str],
    now_ms: int,
    artifact_version: int = 1,
) -> dict[str, Any]:
    if artifact_version not in {1, 2}:
        raise EAAEFLaneGatewayAdmissionError("unsupported lane artifact version")
    repo_root = _validated_repo_root(repo_root)
    operational_raw, operational_file_sha, operational_path = (
        load_eaaef_bootstrap_operational_capability(
            repo_root,
            source_head=source_head,
            plan_root_cid=plan_root_cid,
            registry_prefix=operational_capability_registry_prefix,
            expected_file_sha256=expected_operational_capability_file_sha256,
        )
    )
    pinned_operational, pinned_operational_sha = _load_source_record(
        repo_root,
        Path(operational_path),
        expected_file_sha256=expected_operational_capability_file_sha256,
        expected_fields=frozenset(operational_raw),
        noun="operational capability",
    )
    if (
        pinned_operational != dict(operational_raw)
        or pinned_operational_sha != operational_file_sha
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "operational capability changed between source-safe reads"
        )
    capability = verify_eaaef_bootstrap_operational_capability(
        operational_raw,
        trusted_reviewer_dids=trusted_operational_reviewer_dids,
        trusted_authorization_service_reviewer_dids=(trusted_authorization_service_reviewer_dids),
        now_ms=now_ms,
        expected=expected_operational_bindings,
        forbidden_reviewer_dids=forbidden_reviewer_dids,
    )
    lane_relative = (
        eaaef_lane_authority_v2_relative_path(
            source_head,
            plan_root_cid,
            lane_session_id,
            lane_generation,
            registry_prefix=registry_prefix,
        )
        if artifact_version == 2
        else eaaef_lane_authority_relative_path(
            source_head,
            plan_root_cid,
            lane_session_id,
            lane_generation,
            registry_prefix=registry_prefix,
        )
    )
    lane_raw, lane_file_sha = _load_source_record(
        repo_root,
        lane_relative,
        expected_file_sha256=expected_lane_authority_file_sha256,
        expected_fields=_LANE_V2_FIELDS if artifact_version == 2 else _LANE_FIELDS,
        noun="lane authority",
    )
    lane = _verify_lane(
        lane_raw,
        capability=capability,
        operational_file_sha=operational_file_sha,
        trusted_reviewers=trusted_lane_authority_reviewer_dids,
        now_ms=now_ms,
        artifact_version=artifact_version,
    )
    if lane["lane_session_id"] != lane_session_id or lane["lane_generation"] != lane_generation:
        raise EAAEFLaneGatewayAdmissionError(
            "loaded lane authority differs from its source address"
        )
    source_ids = eaaef_lane_gateway_source_identities()
    verifier_relative = (
        eaaef_lane_verifier_receipt_v2_relative_path(
            lane["authority_cid"], registry_prefix=registry_prefix
        )
        if artifact_version == 2
        else eaaef_lane_verifier_receipt_relative_path(
            lane["authority_cid"], registry_prefix=registry_prefix
        )
    )
    verifier_raw, verifier_file_sha = _load_source_record(
        repo_root,
        verifier_relative,
        expected_file_sha256=expected_verifier_receipt_file_sha256,
        expected_fields=(
            _VERIFIER_V2_FIELDS if artifact_version == 2 else _VERIFIER_FIELDS
        ),
        noun="lane verifier receipt",
    )
    service = capability.get("command_authorization_service")
    if not isinstance(service, Mapping):
        raise EAAEFLaneGatewayAdmissionError(
            "verified operational capability lost its signer service"
        )
    forbidden = frozenset(
        {
            *(_did(item, "forbidden reviewer") for item in forbidden_reviewer_dids),
            _did(lane["reviewer_did"], "lane reviewer"),
            _did(lane["owner_principal_did"], "owner principal"),
            _did(lane["lane_principal_did"], "lane principal"),
            _did(capability["reviewer_did"], "operational reviewer"),
            _did(
                capability["command_authorization_service"]["reviewer_did"],
                "authorization service reviewer",
            ),
            _did(service["service_principal_did"], "authorization service"),
            _did(service["approver_principal_did"], "authorization approver"),
            _did(
                service["authorized_client_principal_did"],
                "authorization service client",
            ),
        }
    )
    verifier = _verify_verifier(
        verifier_raw,
        lane=lane,
        lane_file_sha=lane_file_sha,
        source_ids=source_ids,
        trusted_reviewers=trusted_lane_verifier_reviewer_dids,
        forbidden_dids=forbidden,
        now_ms=now_ms,
        artifact_version=artifact_version,
    )
    merge_relative = (
        eaaef_lane_merge_admission_v2_relative_path(
            lane["authority_cid"], registry_prefix=registry_prefix
        )
        if artifact_version == 2
        else eaaef_lane_merge_admission_relative_path(
            lane["authority_cid"], registry_prefix=registry_prefix
        )
    )
    merge_raw, merge_file_sha = _load_source_record(
        repo_root,
        merge_relative,
        expected_file_sha256=expected_merge_admission_file_sha256,
        expected_fields=_MERGE_V2_FIELDS if artifact_version == 2 else _MERGE_FIELDS,
        noun="lane merge admission",
    )
    merge = _verify_merge(
        merge_raw,
        lane=lane,
        lane_file_sha=lane_file_sha,
        verifier=verifier,
        verifier_file_sha=verifier_file_sha,
        capability=capability,
        source_ids=source_ids,
        trusted_reviewers=trusted_lane_merge_reviewer_dids,
        forbidden_dids=forbidden | {_did(verifier["reviewer_did"], "lane verifier")},
        now_ms=now_ms,
        artifact_version=artifact_version,
    )
    return {
        "artifact_version": artifact_version,
        "capability": capability,
        "lane_authority": lane,
        "verifier_receipt": verifier,
        "merge_admission": merge,
        "source_identities": dict(source_ids),
        "source_paths": {
            "operational_capability": operational_path,
            "lane_authority": lane_relative.as_posix(),
            "verifier_receipt": verifier_relative.as_posix(),
            "merge_admission": merge_relative.as_posix(),
        },
        "file_sha256": {
            "operational_capability": operational_file_sha,
            "lane_authority": lane_file_sha,
            "verifier_receipt": verifier_file_sha,
            "merge_admission": merge_file_sha,
        },
    }


_VERIFIED_ADMISSION_TOKEN = object()
_VERIFIED_EXPIRED_RECOVERY_TOKEN = object()
_VERIFIED_EXPIRED_RECOVERY_V2_TOKEN = object()
_VERIFIED_ADMISSION_V2_TOKEN = object()
_VERIFIED_PROCESS_BIRTH_TOKEN = object()
_SOURCE_COORDINATES_TOKEN = object()
_DEPENDENCY_SOURCE_COORDINATES_TOKEN = object()
_VERIFIED_SOURCE_ARTIFACTS_TOKEN = object()
_VERIFIED_QUACK_CLIENT_FACTORY_TOKEN = object()
_VERIFIED_CONTAINER_DISPATCHER_FACTORY_TOKEN = object()


class EAAEFLaneAdmissionSourceCoordinates(Mapping[str, Any]):
    """Non-authoritative source coordinates for child-side signed re-open."""

    __slots__ = ("_value",)

    def __init__(self, token: object, value: Mapping[str, Any]) -> None:
        if token is not _SOURCE_COORDINATES_TOKEN:
            raise TypeError("lane source coordinates come from the exact parser")
        self._value = MappingProxyType(_detached(value))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def to_dict(self) -> dict[str, Any]:
        return _detached(self._value)


class EAAEFLaneRuntimeDependencySourceCoordinates(Mapping[str, Any]):
    """Transport-only coordinates for child-side signed source re-open."""

    __slots__ = ("_value",)

    def __init__(self, token: object, value: Mapping[str, Any]) -> None:
        if token is not _DEPENDENCY_SOURCE_COORDINATES_TOKEN:
            raise TypeError("dependency source coordinates come from the exact parser")
        self._value = MappingProxyType(_detached(value))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        return _detached(value) if isinstance(value, (dict, list)) else value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def to_dict(self) -> dict[str, Any]:
        return _detached(self._value)


class VerifiedEAAEFLaneRuntimeAdmission(Mapping[str, Any]):
    """Exact typed result of the four-file, three-reviewer source join."""

    __slots__ = ("_value", "_coordinates")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        coordinates: Mapping[str, Any],
    ) -> None:
        if token is not _VERIFIED_ADMISSION_TOKEN:
            raise TypeError("verified EAAEF lane admissions come from the source loader")
        self._value = MappingProxyType(_detached(value))
        self._coordinates = MappingProxyType(dict(coordinates))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def operational_capability(self) -> VerifiedEAAEFBootstrapOperationalCapability:
        capability = self._coordinates["verified_operational_capability"]
        if type(capability) is not VerifiedEAAEFBootstrapOperationalCapability:
            raise EAAEFLaneGatewayAdmissionError("retained operational capability lost exact type")
        return capability

    def reverify(self, *, now_ms: int) -> VerifiedEAAEFLaneRuntimeAdmission:
        """Re-open all sources and require the originally admitted identities."""

        arguments = dict(self._coordinates["loader_arguments"])
        arguments["now_ms"] = _positive(now_ms, "now_ms")
        checked = _load_and_verify(**arguments)
        expected = {
            "lane_authority_cid": self["lane_authority_cid"],
            "verifier_receipt_cid": self["verifier_receipt_cid"],
            "merge_admission_cid": self["merge_admission_cid"],
            "operational_capability_cid": self["operational_capability_cid"],
            "process_birth_nonce": self["process_birth_nonce"],
        }
        observed = _admission_projection(checked)
        if any(observed[name] != value for name, value in expected.items()):
            raise EAAEFLaneGatewayAdmissionError(
                "source-reverified lane admission identity changed"
            )
        return VerifiedEAAEFLaneRuntimeAdmission(
            _VERIFIED_ADMISSION_TOKEN,
            observed,
            {
                **dict(self._coordinates),
                "verified_operational_capability": checked["capability"],
            },
        )


class VerifiedEAAEFLaneRuntimeAdmissionV2(Mapping[str, Any]):
    """Exact v2 lane admission with intrinsic slice and launch bindings."""

    __slots__ = ("_value", "_coordinates")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        coordinates: Mapping[str, Any],
    ) -> None:
        if token is not _VERIFIED_ADMISSION_V2_TOKEN:
            raise TypeError("verified v2 EAAEF lane admissions come from the source loader")
        self._value = MappingProxyType(_detached(value))
        self._coordinates = MappingProxyType(dict(coordinates))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def operational_capability(self) -> VerifiedEAAEFBootstrapOperationalCapability:
        capability = self._coordinates["verified_operational_capability"]
        if type(capability) is not VerifiedEAAEFBootstrapOperationalCapability:
            raise EAAEFLaneGatewayAdmissionError(
                "retained v2 operational capability lost exact type"
            )
        return capability

    def reverify(self, *, now_ms: int) -> VerifiedEAAEFLaneRuntimeAdmissionV2:
        arguments = dict(self._coordinates["loader_arguments"])
        arguments["now_ms"] = _positive(now_ms, "now_ms")
        checked = _load_and_verify(**arguments)
        observed = _admission_projection(checked)
        if observed != dict(self._value):
            raise EAAEFLaneGatewayAdmissionError(
                "source-reverified v2 lane admission changed"
            )
        return VerifiedEAAEFLaneRuntimeAdmissionV2(
            _VERIFIED_ADMISSION_V2_TOKEN,
            observed,
            {
                **dict(self._coordinates),
                "verified_operational_capability": checked["capability"],
            },
        )


class VerifiedEAAEFProcessBirth(Mapping[str, Any]):
    """Current child process joined to one exact signed v2 lane birth."""

    __slots__ = ("_value", "_admission_cid")

    def __init__(self, token: object, value: Mapping[str, Any], admission_cid: str) -> None:
        if token is not _VERIFIED_PROCESS_BIRTH_TOKEN:
            raise TypeError("verified EAAEF process births come from the OS verifier")
        self._value = MappingProxyType(_detached(value))
        self._admission_cid = _sha(admission_cid, "lane merge admission CID")

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def admission_cid(self) -> str:
        return self._admission_cid


class VerifiedEAAEFQuackClientFactoryQualification(Mapping[str, Any]):
    """Signed exact Quack extension and sealed-handle factory qualification."""

    __slots__ = ("_value", "_coordinates", "_extension_bytes")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        coordinates: Mapping[str, Any],
        extension_bytes: bytes,
    ) -> None:
        if token is not _VERIFIED_QUACK_CLIENT_FACTORY_TOKEN:
            raise TypeError("verified Quack client qualifications come from the source loader")
        self._value = MappingProxyType(_detached(value))
        self._coordinates = MappingProxyType(dict(coordinates))
        self._extension_bytes = bytes(extension_bytes)

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def qualification_cid(self) -> str:
        return str(self._value["qualification_cid"])

    def reverify(
        self, *, now_ms: int
    ) -> VerifiedEAAEFQuackClientFactoryQualification:
        checked = load_and_verify_eaaef_quack_client_factory_qualification(
            **dict(self._coordinates),
            now_ms=_positive(now_ms, "now_ms"),
        )
        if dict(checked) != dict(self._value):
            raise EAAEFLaneGatewayAdmissionError(
                "source-reverified Quack client qualification changed"
            )
        return checked


class VerifiedEAAEFContainerDispatcherFactoryQualification(Mapping[str, Any]):
    """Signed endpoints for dynamic worker, verifier, and merge evidence."""

    __slots__ = ("_value", "_coordinates")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        coordinates: Mapping[str, Any],
    ) -> None:
        if token is not _VERIFIED_CONTAINER_DISPATCHER_FACTORY_TOKEN:
            raise TypeError("verified dispatcher qualifications come from the source loader")
        self._value = MappingProxyType(_detached(value))
        self._coordinates = MappingProxyType(dict(coordinates))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def qualification_cid(self) -> str:
        return str(self._value["qualification_cid"])

    def reverify(
        self, *, now_ms: int
    ) -> VerifiedEAAEFContainerDispatcherFactoryQualification:
        checked = load_and_verify_eaaef_container_dispatcher_factory_qualification(
            **dict(self._coordinates),
            now_ms=_positive(now_ms, "now_ms"),
        )
        if dict(checked) != dict(self._value):
            raise EAAEFLaneGatewayAdmissionError(
                "source-reverified dispatcher qualification changed"
            )
        return checked


class VerifiedEAAEFLaneRuntimeSourceArtifacts:
    """Exact v2 lane/native/Quack/dispatcher results from one coordinate reopen."""

    __slots__ = (
        "admission",
        "native_admission",
        "quack_qualification",
        "dispatcher_qualification",
    )

    def __init__(
        self,
        token: object,
        *,
        admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
        native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
        quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
        dispatcher_qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
    ) -> None:
        if token is not _VERIFIED_SOURCE_ARTIFACTS_TOKEN:
            raise TypeError("runtime source artifacts come from the exact source loader")
        if (
            type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2
            or type(native_admission)
            is not VerifiedAgentSupervisorNativeDependencyAdmission
            or type(quack_qualification)
            is not VerifiedEAAEFQuackClientFactoryQualification
            or type(dispatcher_qualification)
            is not VerifiedEAAEFContainerDispatcherFactoryQualification
        ):
            raise EAAEFLaneGatewayAdmissionError(
                "runtime source artifact bundle rejects substitutes"
            )
        self.admission = admission
        self.native_admission = native_admission
        self.quack_qualification = quack_qualification
        self.dispatcher_qualification = dispatcher_qualification


def _native_admission_source_file_sha256(
    admission: VerifiedAgentSupervisorNativeDependencyAdmission,
) -> str:
    """Read the loader-retained hash from an exact, unforgeable native result."""

    if type(admission) is not VerifiedAgentSupervisorNativeDependencyAdmission:
        raise EAAEFLaneGatewayAdmissionError("exact native admission is required")
    coordinates = admission._coordinates
    if not coordinates:
        raise EAAEFLaneGatewayAdmissionError(
            "native admission lacks source-addressed loader coordinates"
        )
    return _sha(
        coordinates.get("expected_file_sha256"),
        "native dependency admission file sha256",
    )


class VerifiedEAAEFExpiredLaneRecoveryAdmission(Mapping[str, Any]):
    """Source-verified historical lane usable only in dead-lane observations."""

    __slots__ = ("_value", "_coordinates")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        coordinates: Mapping[str, Any],
    ) -> None:
        if token is not _VERIFIED_EXPIRED_RECOVERY_TOKEN:
            raise TypeError("expired EAAEF lane recovery admissions come from the source loader")
        self._value = MappingProxyType(_detached(value))
        self._coordinates = MappingProxyType(dict(coordinates))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def reverify_for_recovery(self, *, now_ms: int) -> VerifiedEAAEFExpiredLaneRecoveryAdmission:
        current = _positive(now_ms, "recovery now_ms")
        arguments = dict(self._coordinates["loader_arguments"])
        checked = _load_and_verify(**arguments)
        observed = _admission_projection(checked)
        if observed != dict(self._value) or current < int(observed["expires_at_ms"]):
            raise EAAEFLaneGatewayAdmissionError(
                "expired lane recovery admission changed or is not historical"
            )
        return VerifiedEAAEFExpiredLaneRecoveryAdmission(
            _VERIFIED_EXPIRED_RECOVERY_TOKEN,
            observed,
            self._coordinates,
        )


class VerifiedEAAEFExpiredLaneRecoveryAdmissionV2(Mapping[str, Any]):
    """Historical v2 lane admitted only for exact dead-lane recovery."""

    __slots__ = ("_value", "_coordinates")

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        coordinates: Mapping[str, Any],
    ) -> None:
        if token is not _VERIFIED_EXPIRED_RECOVERY_V2_TOKEN:
            raise TypeError("expired v2 lane admissions come from the source loader")
        self._value = MappingProxyType(_detached(value))
        self._coordinates = MappingProxyType(dict(coordinates))

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def reverify_for_recovery(
        self, *, now_ms: int
    ) -> VerifiedEAAEFExpiredLaneRecoveryAdmissionV2:
        current = _positive(now_ms, "recovery now_ms")
        checked = _load_and_verify(**dict(self._coordinates["loader_arguments"]))
        observed = _admission_projection(checked)
        if observed != dict(self._value) or current < int(observed["expires_at_ms"]):
            raise EAAEFLaneGatewayAdmissionError(
                "expired v2 lane recovery admission changed or is not historical"
            )
        return VerifiedEAAEFExpiredLaneRecoveryAdmissionV2(
            _VERIFIED_EXPIRED_RECOVERY_V2_TOKEN,
            observed,
            self._coordinates,
        )


def _admission_projection(checked: Mapping[str, Any]) -> dict[str, Any]:
    lane = checked["lane_authority"]
    verifier = checked["verifier_receipt"]
    merge = checked["merge_admission"]
    artifact_version = int(checked.get("artifact_version", 1))
    result = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-lane-runtime-admission@2"
            if artifact_version == 2
            else "ipfs_accelerate_py/agent-supervisor/eaaef-lane-runtime-admission@1"
        ),
        "interface": (
            "EAAEFLaneRuntimeAdmission@2"
            if artifact_version == 2
            else "EAAEFLaneRuntimeAdmission@1"
        ),
        "board_namespace": lane["board_namespace"],
        "source_head": lane["source_head"],
        "source_tree": lane["source_tree"],
        "active_plan_root_cid": lane["active_plan_root_cid"],
        "active_plan_revision": lane["active_plan_revision"],
        "operational_capability_cid": lane["operational_capability_cid"],
        "gateway_binding_cid": lane["gateway_binding_cid"],
        "lane_authority_cid": lane["authority_cid"],
        "verifier_receipt_cid": verifier["receipt_cid"],
        "merge_admission_cid": merge["admission_cid"],
        "owner_principal_did": lane["owner_principal_did"],
        "owner_session_id": lane["owner_session_id"],
        "owner_generation": lane["owner_generation"],
        "fence_epoch": lane["fence_epoch"],
        "lane_principal_did": lane["lane_principal_did"],
        "lane_session_id": lane["lane_session_id"],
        "lane_generation": lane["lane_generation"],
        "process_instance_id": lane["process_instance_id"],
        "process_birth_nonce": lane["process_birth_nonce"],
        "journal_namespace": lane["journal_namespace"],
        "issued_at_ms": max(lane["issued_at_ms"], verifier["issued_at_ms"], merge["issued_at_ms"]),
        "expires_at_ms": min(
            lane["expires_at_ms"], verifier["expires_at_ms"], merge["expires_at_ms"]
        ),
        "source_paths": checked["source_paths"],
        "file_sha256": checked["file_sha256"],
        "source_identities": checked["source_identities"],
    }
    if artifact_version == 2:
        result.update({name: lane[name] for name in _BIRTH_PLAN_FIELDS})
    return result


def load_and_verify_eaaef_lane_runtime_admission(
    repo_root: str | Path,
    *,
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    registry_prefix: str,
    operational_capability_registry_prefix: str,
    expected_operational_capability_file_sha256: str,
    expected_lane_authority_file_sha256: str,
    expected_verifier_receipt_file_sha256: str,
    expected_merge_admission_file_sha256: str,
    trusted_operational_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    trusted_lane_authority_reviewer_dids: Sequence[str],
    trusted_lane_verifier_reviewer_dids: Sequence[str],
    trusted_lane_merge_reviewer_dids: Sequence[str],
    expected_operational_bindings: Mapping[str, Any],
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedEAAEFLaneRuntimeAdmission:
    """Load and verify the complete per-birth runtime admission bundle."""

    root = _validated_repo_root(repo_root)
    loader_arguments = {
        "repo_root": root,
        "source_head": source_head,
        "plan_root_cid": plan_root_cid,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "registry_prefix": registry_prefix,
        "operational_capability_registry_prefix": (operational_capability_registry_prefix),
        "expected_operational_capability_file_sha256": (
            expected_operational_capability_file_sha256
        ),
        "expected_lane_authority_file_sha256": expected_lane_authority_file_sha256,
        "expected_verifier_receipt_file_sha256": (expected_verifier_receipt_file_sha256),
        "expected_merge_admission_file_sha256": (expected_merge_admission_file_sha256),
        "trusted_operational_reviewer_dids": tuple(trusted_operational_reviewer_dids),
        "trusted_authorization_service_reviewer_dids": tuple(
            trusted_authorization_service_reviewer_dids
        ),
        "trusted_lane_authority_reviewer_dids": tuple(trusted_lane_authority_reviewer_dids),
        "trusted_lane_verifier_reviewer_dids": tuple(trusted_lane_verifier_reviewer_dids),
        "trusted_lane_merge_reviewer_dids": tuple(trusted_lane_merge_reviewer_dids),
        "expected_operational_bindings": _detached(expected_operational_bindings),
        "forbidden_reviewer_dids": tuple(forbidden_reviewer_dids),
        "now_ms": _positive(now_ms, "now_ms"),
    }
    checked = _load_and_verify(**loader_arguments)
    projection = _admission_projection(checked)
    if not projection["issued_at_ms"] <= now_ms < projection["expires_at_ms"]:
        raise EAAEFLaneGatewayAdmissionError("joined lane admission has no common live interval")
    return VerifiedEAAEFLaneRuntimeAdmission(
        _VERIFIED_ADMISSION_TOKEN,
        projection,
        {
            "loader_arguments": loader_arguments,
            "verified_operational_capability": checked["capability"],
        },
    )


def load_and_verify_eaaef_lane_runtime_admission_v2(
    repo_root: str | Path,
    *,
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    registry_prefix: str,
    operational_capability_registry_prefix: str,
    expected_operational_capability_file_sha256: str,
    expected_lane_authority_file_sha256: str,
    expected_verifier_receipt_file_sha256: str,
    expected_merge_admission_file_sha256: str,
    trusted_operational_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    trusted_lane_authority_reviewer_dids: Sequence[str],
    trusted_lane_verifier_reviewer_dids: Sequence[str],
    trusted_lane_merge_reviewer_dids: Sequence[str],
    expected_operational_bindings: Mapping[str, Any],
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedEAAEFLaneRuntimeAdmissionV2:
    """Load v2 authority/verifier/merge artifacts without a v1 upgrade."""

    root = _validated_repo_root(repo_root)
    loader_arguments = {
        "repo_root": root,
        "source_head": source_head,
        "plan_root_cid": plan_root_cid,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "registry_prefix": registry_prefix,
        "operational_capability_registry_prefix": operational_capability_registry_prefix,
        "expected_operational_capability_file_sha256": (
            expected_operational_capability_file_sha256
        ),
        "expected_lane_authority_file_sha256": expected_lane_authority_file_sha256,
        "expected_verifier_receipt_file_sha256": expected_verifier_receipt_file_sha256,
        "expected_merge_admission_file_sha256": expected_merge_admission_file_sha256,
        "trusted_operational_reviewer_dids": tuple(trusted_operational_reviewer_dids),
        "trusted_authorization_service_reviewer_dids": tuple(
            trusted_authorization_service_reviewer_dids
        ),
        "trusted_lane_authority_reviewer_dids": tuple(
            trusted_lane_authority_reviewer_dids
        ),
        "trusted_lane_verifier_reviewer_dids": tuple(
            trusted_lane_verifier_reviewer_dids
        ),
        "trusted_lane_merge_reviewer_dids": tuple(trusted_lane_merge_reviewer_dids),
        "expected_operational_bindings": _detached(expected_operational_bindings),
        "forbidden_reviewer_dids": tuple(forbidden_reviewer_dids),
        "now_ms": _positive(now_ms, "now_ms"),
        "artifact_version": 2,
    }
    checked = _load_and_verify(**loader_arguments)
    projection = _admission_projection(checked)
    if not projection["issued_at_ms"] <= now_ms < projection["expires_at_ms"]:
        raise EAAEFLaneGatewayAdmissionError(
            "joined v2 lane admission has no common live interval"
        )
    return VerifiedEAAEFLaneRuntimeAdmissionV2(
        _VERIFIED_ADMISSION_V2_TOKEN,
        projection,
        {
            "loader_arguments": loader_arguments,
            "verified_operational_capability": checked["capability"],
        },
    )


def load_and_verify_eaaef_quack_client_factory_qualification(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
    native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
    registry_prefix: str,
    expected_file_sha256: str,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> VerifiedEAAEFQuackClientFactoryQualification:
    """Load the independently signed extension/sealed-handle qualification."""

    if (
        type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2
        or type(native_admission) is not VerifiedAgentSupervisorNativeDependencyAdmission
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "Quack client qualification requires exact lane and native admissions"
        )
    current = _positive(now_ms, "now_ms")
    checked_admission = admission.reverify(now_ms=current)
    try:
        checked_native = native_admission.reverify(now_ms=current)
    except AgentSupervisorNativeDependencyAdmissionError as exc:
        raise EAAEFLaneGatewayAdmissionError(
            "native dependency admission failed source re-verification"
        ) from exc
    lane_arguments = dict(admission._coordinates["loader_arguments"])
    lane_arguments["now_ms"] = current
    lane_bundle = _load_and_verify(**lane_arguments)
    native_file_sha256 = _native_admission_source_file_sha256(checked_native)
    if (
        _sha(expected_file_sha256, "Quack qualification file sha256")
        != checked_admission["quack_client_factory_qualification_file_sha256"]
        or native_file_sha256
        != checked_admission["native_dependency_admission_file_sha256"]
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "Quack qualification coordinates differ from signed v2 source pins"
        )
    repo_root = Path(lane_arguments["repo_root"])
    relative = eaaef_quack_client_factory_qualification_relative_path(
        str(checked_admission["source_head"]),
        str(checked_admission["lane_session_id"]),
        int(checked_admission["lane_generation"]),
        registry_prefix=registry_prefix,
    )
    raw, _file_sha = _load_source_record(
        repo_root,
        relative,
        expected_file_sha256=expected_file_sha256,
        expected_fields=_QUACK_CLIENT_FACTORY_FIELDS,
        noun="Quack client factory qualification",
    )
    value = _detached(raw)
    reviewer = _trusted_reviewer(
        value.get("reviewer_did"),
        trusted_reviewer_dids,
        "Quack client factory qualification",
    )
    issued, expires = _lifetime(value, current, "Quack client factory qualification")
    capability = checked_admission.operational_capability
    service = capability.get("command_authorization_service")
    if not isinstance(service, Mapping):
        raise EAAEFLaneGatewayAdmissionError(
            "operational capability lost its authorization service"
        )
    forbidden = {
        lane_bundle["lane_authority"]["reviewer_did"],
        lane_bundle["verifier_receipt"]["reviewer_did"],
        lane_bundle["merge_admission"]["reviewer_did"],
        capability["reviewer_did"],
        service["reviewer_did"],
        service["service_principal_did"],
        service["approver_principal_did"],
        service["authorized_client_principal_did"],
        checked_admission["owner_principal_did"],
        checked_admission["lane_principal_did"],
        checked_native["reviewer_did"],
    }
    joins = {
        "board_namespace": checked_admission["board_namespace"],
        "source_head": checked_admission["source_head"],
        "source_tree": checked_admission["source_tree"],
        "active_plan_root_cid": checked_admission["active_plan_root_cid"],
        "active_plan_revision": checked_admission["active_plan_revision"],
        "active_plan_revision_cid": checked_admission["active_plan_revision_cid"],
        "slice_manifest_cid": checked_admission["slice_manifest_cid"],
        "slice_id": checked_admission["slice_id"],
        "lane_id": checked_admission["lane_id"],
        "task_ids": checked_admission["task_ids"],
        "task_cids": checked_admission["task_cids"],
        "operational_capability_cid": checked_admission["operational_capability_cid"],
        "gateway_binding_cid": checked_admission["gateway_binding_cid"],
        "command_fabric_qualification_cid": capability[
            "command_fabric_qualification_cid"
        ],
        "native_dependency_admission_cid": checked_admission[
            "native_dependency_admission_cid"
        ],
        "native_dependency_admission_file_sha256": native_file_sha256,
        "lane_session_id": checked_admission["lane_session_id"],
        "lane_generation": checked_admission["lane_generation"],
        "process_instance_id": checked_admission["process_instance_id"],
        "process_birth_nonce": checked_admission["process_birth_nonce"],
        "command_endpoint": capability["command_endpoint"],
        "command_secret_handle": capability["command_secret_handle"],
        "command_secret_generation": checked_admission["command_secret_generation"],
        "command_secret_descriptor_sha256": checked_admission[
            "command_secret_descriptor_sha256"
        ],
        "state_endpoint": capability["state_endpoint"],
        "state_secret_handle": capability["state_secret_handle"],
        "state_secret_generation": checked_admission["state_secret_generation"],
        "state_secret_descriptor_sha256": checked_admission[
            "state_secret_descriptor_sha256"
        ],
        **dict(checked_admission["source_identities"]),
    }
    native_joins = {
        "board_namespace": checked_admission["board_namespace"],
        "source_head": checked_admission["source_head"],
        "source_tree": checked_admission["source_tree"],
        "active_plan_root_cid": checked_admission["active_plan_root_cid"],
        "active_plan_revision": checked_admission["active_plan_revision"],
        "active_plan_revision_cid": checked_admission["active_plan_revision_cid"],
        "slice_manifest_cid": checked_admission["slice_manifest_cid"],
        "slice_id": checked_admission["slice_id"],
        "lane_id": checked_admission["lane_id"],
        "lane_session_id": checked_admission["lane_session_id"],
        "lane_generation": checked_admission["lane_generation"],
        "process_instance_id": checked_admission["process_instance_id"],
        "process_birth_nonce": checked_admission["process_birth_nonce"],
        "expected_process_uid": checked_admission["expected_process_uid"],
        "expected_parent_pid": checked_admission["expected_parent_pid"],
        "expected_parent_process_start_time_ticks": checked_admission[
            "expected_parent_process_start_time_ticks"
        ],
        "expected_executable_sha256": checked_admission["expected_executable_sha256"],
        "launch_argv_cid": checked_admission["launch_argv_cid"],
    }
    if (
        value.get("schema") != EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_SCHEMA
        or value.get("interface") != EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_INTERFACE
        or value.get("reviewer_role") != EAAEF_QUACK_CLIENT_FACTORY_REVIEW_ROLE
        or reviewer in forbidden
        or value.get("secret_resolver_mode") != "inherited_write_sealed_memfd"
        or value.get("raw_token_argv_enabled") is not False
        or value.get("raw_token_environment_enabled") is not False
        or value.get("raw_token_path_enabled") is not False
        or value.get("qualification_cid")
        != checked_admission["quack_client_factory_qualification_cid"]
        or checked_native["admission_cid"]
        != checked_admission["native_dependency_admission_cid"]
        or any(checked_native.get(name) != expected for name, expected in native_joins.items())
        or not issued <= int(checked_admission["issued_at_ms"])
        < int(checked_admission["expires_at_ms"])
        <= expires
        or any(value.get(name) != expected for name, expected in joins.items())
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "Quack client factory qualification is not independently bound"
        )
    extension_relative = Path(str(value.get("quack_extension_relative_path") or ""))
    extension_bytes = _load_immutable_source_bytes(
        repo_root,
        extension_relative,
        expected_file_sha256=str(value.get("quack_extension_sha256") or ""),
        noun="qualified Quack extension",
    )
    _verify_signature(
        value,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="qualification_cid",
        noun="Quack client factory qualification",
    )
    return VerifiedEAAEFQuackClientFactoryQualification(
        _VERIFIED_QUACK_CLIENT_FACTORY_TOKEN,
        value,
        {
            "admission": admission,
            "native_admission": native_admission,
            "registry_prefix": registry_prefix,
            "expected_file_sha256": expected_file_sha256,
            "trusted_reviewer_dids": tuple(trusted_reviewer_dids),
        },
        extension_bytes,
    )


def load_and_verify_eaaef_container_dispatcher_factory_qualification(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
    native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
    quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
    registry_prefix: str,
    expected_file_sha256: str,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> VerifiedEAAEFContainerDispatcherFactoryQualification:
    """Load independently signed dynamic service identities for each attempt."""

    if (
        type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2
        or type(native_admission) is not VerifiedAgentSupervisorNativeDependencyAdmission
        or type(quack_qualification) is not VerifiedEAAEFQuackClientFactoryQualification
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "dispatcher qualification requires exact lane/native/Quack artifacts"
        )
    current = _positive(now_ms, "now_ms")
    checked_lane = admission.reverify(now_ms=current)
    try:
        checked_native = native_admission.reverify(now_ms=current)
    except AgentSupervisorNativeDependencyAdmissionError as exc:
        raise EAAEFLaneGatewayAdmissionError(
            "dispatcher native admission failed source re-verification"
        ) from exc
    checked_quack = quack_qualification.reverify(now_ms=current)
    lane_arguments = dict(admission._coordinates["loader_arguments"])
    lane_arguments["now_ms"] = current
    lane_bundle = _load_and_verify(**lane_arguments)
    native_file_sha256 = _native_admission_source_file_sha256(checked_native)
    quack_file_sha256 = _sha(
        checked_quack._coordinates.get("expected_file_sha256"),
        "Quack qualification file sha256",
    )
    if (
        _sha(expected_file_sha256, "dispatcher qualification file sha256")
        != checked_lane["container_dispatcher_factory_qualification_file_sha256"]
        or native_file_sha256
        != checked_lane["native_dependency_admission_file_sha256"]
        or quack_file_sha256
        != checked_lane["quack_client_factory_qualification_file_sha256"]
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "dispatcher qualification coordinates differ from signed v2 source pins"
        )
    repo_root = Path(lane_arguments["repo_root"])
    relative = eaaef_container_dispatcher_factory_qualification_relative_path(
        str(checked_lane["source_head"]),
        str(checked_lane["lane_session_id"]),
        int(checked_lane["lane_generation"]),
        registry_prefix=registry_prefix,
    )
    raw, _file_sha = _load_source_record(
        repo_root,
        relative,
        expected_file_sha256=expected_file_sha256,
        expected_fields=_CONTAINER_DISPATCHER_FACTORY_FIELDS,
        noun="container dispatcher factory qualification",
    )
    value = _detached(raw)
    reviewer = _trusted_reviewer(
        value.get("reviewer_did"),
        trusted_reviewer_dids,
        "container dispatcher factory qualification",
    )
    issued, expires = _lifetime(
        value, current, "container dispatcher factory qualification"
    )
    capability = checked_lane.operational_capability
    service = capability.get("command_authorization_service")
    if not isinstance(service, Mapping):
        raise EAAEFLaneGatewayAdmissionError(
            "operational capability lost its authorization service"
        )
    forbidden = {
        lane_bundle["lane_authority"]["reviewer_did"],
        lane_bundle["verifier_receipt"]["reviewer_did"],
        lane_bundle["merge_admission"]["reviewer_did"],
        capability["reviewer_did"],
        service["reviewer_did"],
        service["service_principal_did"],
        service["approver_principal_did"],
        service["authorized_client_principal_did"],
        checked_lane["owner_principal_did"],
        checked_lane["lane_principal_did"],
        checked_native["reviewer_did"],
        checked_quack["reviewer_did"],
    }
    joins = {
        "board_namespace": checked_lane["board_namespace"],
        "source_head": checked_lane["source_head"],
        "source_tree": checked_lane["source_tree"],
        "active_plan_root_cid": checked_lane["active_plan_root_cid"],
        "active_plan_revision": checked_lane["active_plan_revision"],
        "active_plan_revision_cid": checked_lane["active_plan_revision_cid"],
        "slice_manifest_cid": checked_lane["slice_manifest_cid"],
        "slice_id": checked_lane["slice_id"],
        "lane_id": checked_lane["lane_id"],
        "task_ids": checked_lane["task_ids"],
        "task_cids": checked_lane["task_cids"],
        "operational_capability_cid": checked_lane["operational_capability_cid"],
        "gateway_binding_cid": checked_lane["gateway_binding_cid"],
        "native_dependency_admission_cid": checked_native["admission_cid"],
        "native_dependency_admission_file_sha256": native_file_sha256,
        "quack_client_factory_qualification_cid": checked_quack.qualification_cid,
        "quack_client_factory_qualification_file_sha256": quack_file_sha256,
        "lane_session_id": checked_lane["lane_session_id"],
        "lane_generation": checked_lane["lane_generation"],
        "process_instance_id": checked_lane["process_instance_id"],
        "process_birth_nonce": checked_lane["process_birth_nonce"],
        "worker_principal_did": capability["worker_principal_did"],
        "dispatcher_source_sha256": eaaef_container_dispatcher_source_sha256(),
        **dict(checked_lane["source_identities"]),
    }
    services = value.get("services")
    if not isinstance(services, Mapping) or set(services) != set(_DYNAMIC_SERVICE_METHODS):
        raise EAAEFLaneGatewayAdmissionError(
            "dispatcher dynamic service set is not exact"
        )
    principals: dict[str, str] = {}
    endpoints: set[str] = set()
    for name, expected_methods in _DYNAMIC_SERVICE_METHODS.items():
        descriptor = services.get(name)
        if not isinstance(descriptor, Mapping) or set(descriptor) != _DYNAMIC_SERVICE_FIELDS:
            raise EAAEFLaneGatewayAdmissionError(
                f"dispatcher {name} service descriptor is not exact"
            )
        principal = _did(descriptor.get("service_principal_did"), f"{name} service")
        endpoint = _service_endpoint(descriptor.get("endpoint"), f"{name} service endpoint")
        if (
            descriptor.get("interface") != EAAEF_CONTAINER_DYNAMIC_SERVICE_INTERFACE
            or descriptor.get("methods") != list(expected_methods)
            or descriptor.get("peer_credentials_required") is not True
            or descriptor.get("response_signature_verification_required") is not True
            or descriptor.get("request_lane_reverification_required") is not True
            or _nonnegative(descriptor.get("expected_server_uid"), "service uid")
            != os.geteuid()
            or _positive(descriptor.get("expected_server_pid"), "service pid") < 1
            or _positive(
                descriptor.get("expected_server_process_start_time_ticks"),
                "service process birth",
            )
            < 1
            or not 1
            <= _positive(descriptor.get("maximum_request_bytes"), "maximum request")
            <= 1024 * 1024
            or not 1
            <= _positive(descriptor.get("maximum_response_bytes"), "maximum response")
            <= 8 * 1024 * 1024
            or not 1
            <= _positive(descriptor.get("request_timeout_ms"), "request timeout")
            <= 60_000
        ):
            raise EAAEFLaneGatewayAdmissionError(
                f"dispatcher {name} service policy is invalid"
            )
        principals[name] = principal
        if endpoint in endpoints:
            raise EAAEFLaneGatewayAdmissionError(
                "dispatcher services require distinct endpoints"
            )
        endpoints.add(endpoint)
    if (
        len(set(principals.values())) != len(principals)
        or principals["worker"] != capability["worker_principal_did"]
        or principals["worker"]
        in (forbidden - {str(checked_lane["lane_principal_did"])})
        or {principals["verifier"], principals["merge"], principals["host_source"]}
        & forbidden
        or value.get("schema") != EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_SCHEMA
        or value.get("interface")
        != EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_INTERFACE
        or value.get("reviewer_role") != EAAEF_CONTAINER_DISPATCHER_FACTORY_REVIEW_ROLE
        or reviewer in forbidden | set(principals.values())
        or value.get("dynamic_per_attempt_verification_required") is not True
        or value.get("dynamic_per_attempt_merge_admission_required") is not True
        or value.get("static_lane_evidence_is_attempt_success") is not False
        or value.get("caller_callbacks_allowed") is not False
        or value.get("direct_container_launch_allowed") is not False
        or value.get("qualification_cid")
        != checked_lane["container_dispatcher_factory_qualification_cid"]
        or not issued <= int(checked_lane["issued_at_ms"])
        < int(checked_lane["expires_at_ms"])
        <= expires
        or any(value.get(name) != expected for name, expected in joins.items())
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "dispatcher qualification is not independently and exactly bound"
        )
    _verify_signature(
        value,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="qualification_cid",
        noun="container dispatcher factory qualification",
    )
    return VerifiedEAAEFContainerDispatcherFactoryQualification(
        _VERIFIED_CONTAINER_DISPATCHER_FACTORY_TOKEN,
        value,
        {
            "admission": admission,
            "native_admission": native_admission,
            "quack_qualification": quack_qualification,
            "registry_prefix": registry_prefix,
            "expected_file_sha256": expected_file_sha256,
            "trusted_reviewer_dids": tuple(trusted_reviewer_dids),
        },
    )


def verify_eaaef_current_process_birth(
    admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
) -> VerifiedEAAEFProcessBirth:
    """Join the current daemon process to the signed parent/exe/argv birth."""

    if type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2:
        raise EAAEFLaneGatewayAdmissionError(
            "process birth requires an exact v2 lane admission"
        )
    checked = admission.reverify(now_ms=time.time_ns() // 1_000_000)
    parent_pid = os.getppid()
    observations = {
        "process_uid": os.geteuid(),
        "parent_pid": parent_pid,
        "parent_process_start_time_ticks": _linux_process_start_time_ticks(parent_pid),
        "executable_sha256": eaaef_current_executable_sha256(),
        "launch_argv_cid": eaaef_launch_argv_cid(sys.argv),
    }
    expected = {
        "process_uid": checked["expected_process_uid"],
        "parent_pid": checked["expected_parent_pid"],
        "parent_process_start_time_ticks": checked[
            "expected_parent_process_start_time_ticks"
        ],
        "executable_sha256": checked["expected_executable_sha256"],
        "launch_argv_cid": checked["launch_argv_cid"],
    }
    if observations != expected:
        raise EAAEFLaneGatewayAdmissionError(
            "current process differs from the signed v2 birth context"
        )
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-verified-process-birth@1",
        "interface": "VerifiedEAAEFProcessBirth@1",
        "lane_authority_cid": checked["lane_authority_cid"],
        "lane_merge_admission_cid": checked["merge_admission_cid"],
        "lane_session_id": checked["lane_session_id"],
        "lane_generation": checked["lane_generation"],
        "process_instance_id": checked["process_instance_id"],
        "process_birth_nonce": checked["process_birth_nonce"],
        "pid": os.getpid(),
        "process_start_time_ticks": _linux_process_start_time_ticks(os.getpid()),
        **observations,
    }
    return VerifiedEAAEFProcessBirth(
        _VERIFIED_PROCESS_BIRTH_TOKEN,
        {**body, "birth_cid": _cid(body)},
        str(checked["merge_admission_cid"]),
    )


def eaaef_lane_admission_source_coordinates(
    admission: VerifiedEAAEFLaneRuntimeAdmission
    | VerifiedEAAEFLaneRuntimeAdmissionV2,
) -> EAAEFLaneAdmissionSourceCoordinates:
    """Project non-authoritative coordinates for a child source re-open."""

    if type(admission) not in {
        VerifiedEAAEFLaneRuntimeAdmission,
        VerifiedEAAEFLaneRuntimeAdmissionV2,
    }:
        raise EAAEFLaneGatewayAdmissionError("source coordinates require an exact admission")
    arguments = dict(admission._coordinates["loader_arguments"])
    arguments.pop("repo_root", None)
    arguments.pop("now_ms", None)
    arguments.setdefault("artifact_version", 1)
    if set(arguments) != _SOURCE_COORDINATE_ARGUMENT_FIELDS:
        raise EAAEFLaneGatewayAdmissionError("retained source coordinates are incomplete")
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-lane-source-coordinates@1",
        "interface": "EAAEFLaneAdmissionSourceCoordinates@1",
        **_detached(arguments),
    }
    return EAAEFLaneAdmissionSourceCoordinates(
        _SOURCE_COORDINATES_TOKEN,
        {**body, "coordinates_cid": _cid(body)},
    )


def parse_eaaef_lane_admission_source_coordinates(
    value: Mapping[str, Any],
) -> EAAEFLaneAdmissionSourceCoordinates:
    """Parse coordinates only; this never verifies or grants lane authority."""

    if not isinstance(value, Mapping):
        raise EAAEFLaneGatewayAdmissionError("lane source coordinates are not an object")
    payload = _detached(value)
    expected = _SOURCE_COORDINATE_ARGUMENT_FIELDS | {
        "schema",
        "interface",
        "coordinates_cid",
    }
    body = {name: item for name, item in payload.items() if name != "coordinates_cid"}
    if (
        set(payload) != expected
        or payload.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/eaaef-lane-source-coordinates@1"
        or payload.get("interface") != "EAAEFLaneAdmissionSourceCoordinates@1"
        or payload.get("artifact_version") not in {1, 2}
        or payload.get("coordinates_cid") != _cid(body)
    ):
        raise EAAEFLaneGatewayAdmissionError("lane source coordinates are not exact")
    return EAAEFLaneAdmissionSourceCoordinates(_SOURCE_COORDINATES_TOKEN, payload)


def load_and_verify_eaaef_lane_runtime_admission_from_coordinates(
    repo_root: str | Path,
    *,
    coordinates: EAAEFLaneAdmissionSourceCoordinates,
    now_ms: int,
) -> VerifiedEAAEFLaneRuntimeAdmission | VerifiedEAAEFLaneRuntimeAdmissionV2:
    """Re-open signed artifacts; coordinates themselves are never authority."""

    if type(coordinates) is not EAAEFLaneAdmissionSourceCoordinates:
        raise EAAEFLaneGatewayAdmissionError("child re-open requires parsed coordinates")
    arguments = {
        name: coordinates[name] for name in _SOURCE_COORDINATE_ARGUMENT_FIELDS
    }
    version = int(arguments.pop("artifact_version"))
    loader = (
        load_and_verify_eaaef_lane_runtime_admission_v2
        if version == 2
        else load_and_verify_eaaef_lane_runtime_admission
    )
    return loader(
        repo_root,
        **arguments,
        now_ms=_positive(now_ms, "now_ms"),
    )


def eaaef_lane_runtime_dependency_source_coordinates(
    *,
    admission: VerifiedEAAEFLaneRuntimeAdmissionV2,
    native_admission: VerifiedAgentSupervisorNativeDependencyAdmission,
    quack_qualification: VerifiedEAAEFQuackClientFactoryQualification,
    dispatcher_qualification: VerifiedEAAEFContainerDispatcherFactoryQualification,
) -> EAAEFLaneRuntimeDependencySourceCoordinates:
    """Project source coordinates only; no path, token, callback, or authority."""

    if (
        type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2
        or type(native_admission)
        is not VerifiedAgentSupervisorNativeDependencyAdmission
        or type(quack_qualification)
        is not VerifiedEAAEFQuackClientFactoryQualification
        or type(dispatcher_qualification)
        is not VerifiedEAAEFContainerDispatcherFactoryQualification
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "dependency coordinates require exact verified source artifacts"
        )
    native = dict(native_admission._coordinates)
    quack = dict(quack_qualification._coordinates)
    dispatcher = dict(dispatcher_qualification._coordinates)
    if not native or not quack or not dispatcher:
        raise EAAEFLaneGatewayAdmissionError(
            "dependency artifacts lack retained source coordinates"
        )
    native.pop("repo_root", None)
    expected_pin = native.pop("expected_native_dependency_pin", None)
    if not hasattr(expected_pin, "as_dict"):
        raise EAAEFLaneGatewayAdmissionError(
            "native source coordinates lost their exact dependency pin"
        )
    native["expected_native_dependency_pin"] = expected_pin.as_dict()
    quack_coordinates = {
        name: quack[name] for name in _QUALIFICATION_SOURCE_COORDINATE_FIELDS
    }
    dispatcher_coordinates = {
        name: dispatcher[name] for name in _QUALIFICATION_SOURCE_COORDINATE_FIELDS
    }
    if (
        set(native) != _NATIVE_SOURCE_COORDINATE_FIELDS
        or set(quack_coordinates) != _QUALIFICATION_SOURCE_COORDINATE_FIELDS
        or set(dispatcher_coordinates) != _QUALIFICATION_SOURCE_COORDINATE_FIELDS
        or native_admission["admission_cid"]
        != admission["native_dependency_admission_cid"]
        or native["expected_file_sha256"]
        != admission["native_dependency_admission_file_sha256"]
        or quack_qualification.qualification_cid
        != admission["quack_client_factory_qualification_cid"]
        or quack_coordinates["expected_file_sha256"]
        != admission["quack_client_factory_qualification_file_sha256"]
        or dispatcher_qualification.qualification_cid
        != admission["container_dispatcher_factory_qualification_cid"]
        or dispatcher_coordinates["expected_file_sha256"]
        != admission["container_dispatcher_factory_qualification_file_sha256"]
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "dependency source coordinates differ from the signed v2 lane pins"
        )
    body = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-lane-runtime-dependency-source-coordinates@1"
        ),
        "interface": "EAAEFLaneRuntimeDependencySourceCoordinates@1",
        "lane": eaaef_lane_admission_source_coordinates(admission).to_dict(),
        "native": _detached(native),
        "quack": _detached(quack_coordinates),
        "dispatcher": _detached(dispatcher_coordinates),
    }
    return EAAEFLaneRuntimeDependencySourceCoordinates(
        _DEPENDENCY_SOURCE_COORDINATES_TOKEN,
        {**body, "coordinates_cid": _cid(body)},
    )


def parse_eaaef_lane_runtime_dependency_source_coordinates(
    value: Mapping[str, Any],
) -> EAAEFLaneRuntimeDependencySourceCoordinates:
    """Parse transport coordinates; signed loaders still grant all authority."""

    if not isinstance(value, Mapping):
        raise EAAEFLaneGatewayAdmissionError(
            "runtime dependency source coordinates are not an object"
        )
    payload = _detached(value)
    body = {name: item for name, item in payload.items() if name != "coordinates_cid"}
    if (
        set(payload)
        != {"schema", "interface", "lane", "native", "quack", "dispatcher", "coordinates_cid"}
        or payload.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-lane-runtime-dependency-source-coordinates@1"
        )
        or payload.get("interface")
        != "EAAEFLaneRuntimeDependencySourceCoordinates@1"
        or payload.get("coordinates_cid") != _cid(body)
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "runtime dependency source coordinates are not exact"
        )
    lane_coordinates = parse_eaaef_lane_admission_source_coordinates(payload["lane"])
    native = payload.get("native")
    quack = payload.get("quack")
    dispatcher = payload.get("dispatcher")
    if (
        lane_coordinates["artifact_version"] != 2
        or not isinstance(native, Mapping)
        or set(native) != _NATIVE_SOURCE_COORDINATE_FIELDS
        or not isinstance(quack, Mapping)
        or set(quack) != _QUALIFICATION_SOURCE_COORDINATE_FIELDS
        or not isinstance(dispatcher, Mapping)
        or set(dispatcher) != _QUALIFICATION_SOURCE_COORDINATE_FIELDS
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "runtime dependency source coordinate members are incomplete"
        )
    try:
        parse_agent_supervisor_native_dependency_pin(
            native["expected_native_dependency_pin"]
        )
    except ValueError as exc:
        raise EAAEFLaneGatewayAdmissionError(
            "native dependency source coordinate pin is invalid"
        ) from exc
    for item, noun in (
        (native["expected_file_sha256"], "native source file sha256"),
        (quack["expected_file_sha256"], "Quack source file sha256"),
        (dispatcher["expected_file_sha256"], "dispatcher source file sha256"),
    ):
        _sha(item, noun)
    return EAAEFLaneRuntimeDependencySourceCoordinates(
        _DEPENDENCY_SOURCE_COORDINATES_TOKEN, payload
    )


def load_and_verify_eaaef_lane_runtime_source_artifacts(
    repo_root: str | Path,
    *,
    coordinates: EAAEFLaneRuntimeDependencySourceCoordinates,
    now_ms: int,
) -> VerifiedEAAEFLaneRuntimeSourceArtifacts:
    """Re-open the exact v2 lane, native, Quack, and dispatcher artifacts."""

    if type(coordinates) is not EAAEFLaneRuntimeDependencySourceCoordinates:
        raise EAAEFLaneGatewayAdmissionError(
            "runtime source loader requires exact parsed coordinates"
        )
    current = _positive(now_ms, "now_ms")
    lane_coordinates = parse_eaaef_lane_admission_source_coordinates(
        coordinates["lane"]
    )
    admission = load_and_verify_eaaef_lane_runtime_admission_from_coordinates(
        repo_root,
        coordinates=lane_coordinates,
        now_ms=current,
    )
    if type(admission) is not VerifiedEAAEFLaneRuntimeAdmissionV2:
        raise EAAEFLaneGatewayAdmissionError(
            "runtime dependency source loader requires v2 lane authority"
        )
    native_arguments = dict(coordinates["native"])
    native_arguments["expected_native_dependency_pin"] = (
        parse_agent_supervisor_native_dependency_pin(
            native_arguments["expected_native_dependency_pin"]
        )
    )
    native = load_and_verify_agent_supervisor_native_dependency_admission(
        Path(repo_root),
        **native_arguments,
        now_ms=current,
    )
    quack_arguments = dict(coordinates["quack"])
    quack = load_and_verify_eaaef_quack_client_factory_qualification(
        admission=admission,
        native_admission=native,
        **quack_arguments,
        now_ms=current,
    )
    dispatcher_arguments = dict(coordinates["dispatcher"])
    dispatcher = load_and_verify_eaaef_container_dispatcher_factory_qualification(
        admission=admission,
        native_admission=native,
        quack_qualification=quack,
        **dispatcher_arguments,
        now_ms=current,
    )
    return VerifiedEAAEFLaneRuntimeSourceArtifacts(
        _VERIFIED_SOURCE_ARTIFACTS_TOKEN,
        admission=admission,
        native_admission=native,
        quack_qualification=quack,
        dispatcher_qualification=dispatcher,
    )


def load_and_verify_eaaef_expired_lane_recovery_admission(
    repo_root: str | Path,
    *,
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    registry_prefix: str,
    operational_capability_registry_prefix: str,
    expected_operational_capability_file_sha256: str,
    expected_lane_authority_file_sha256: str,
    expected_verifier_receipt_file_sha256: str,
    expected_merge_admission_file_sha256: str,
    trusted_operational_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    trusted_lane_authority_reviewer_dids: Sequence[str],
    trusted_lane_verifier_reviewer_dids: Sequence[str],
    trusted_lane_merge_reviewer_dids: Sequence[str],
    expected_operational_bindings: Mapping[str, Any],
    authority_verification_ms: int,
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedEAAEFExpiredLaneRecoveryAdmission:
    """Reconstruct a signed expired lane for read-only dead-lane recovery.

    ``authority_verification_ms`` must fall inside every originally signed
    lifetime and ``now_ms`` must be at or after the joined expiry.  The result
    has a distinct exact type and cannot satisfy the live gateway factory.
    """

    root = _validated_repo_root(repo_root)
    historical = _positive(authority_verification_ms, "authority_verification_ms")
    current = _positive(now_ms, "recovery now_ms")
    loader_arguments = {
        "repo_root": root,
        "source_head": source_head,
        "plan_root_cid": plan_root_cid,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "registry_prefix": registry_prefix,
        "operational_capability_registry_prefix": (operational_capability_registry_prefix),
        "expected_operational_capability_file_sha256": (
            expected_operational_capability_file_sha256
        ),
        "expected_lane_authority_file_sha256": expected_lane_authority_file_sha256,
        "expected_verifier_receipt_file_sha256": (expected_verifier_receipt_file_sha256),
        "expected_merge_admission_file_sha256": (expected_merge_admission_file_sha256),
        "trusted_operational_reviewer_dids": tuple(trusted_operational_reviewer_dids),
        "trusted_authorization_service_reviewer_dids": tuple(
            trusted_authorization_service_reviewer_dids
        ),
        "trusted_lane_authority_reviewer_dids": tuple(trusted_lane_authority_reviewer_dids),
        "trusted_lane_verifier_reviewer_dids": tuple(trusted_lane_verifier_reviewer_dids),
        "trusted_lane_merge_reviewer_dids": tuple(trusted_lane_merge_reviewer_dids),
        "expected_operational_bindings": _detached(expected_operational_bindings),
        "forbidden_reviewer_dids": tuple(forbidden_reviewer_dids),
        "now_ms": historical,
    }
    checked = _load_and_verify(**loader_arguments)
    projection = _admission_projection(checked)
    if not (
        int(projection["issued_at_ms"]) <= historical < int(projection["expires_at_ms"]) <= current
    ):
        raise EAAEFLaneGatewayAdmissionError(
            "expired lane recovery times do not prove a historical live admission"
        )
    return VerifiedEAAEFExpiredLaneRecoveryAdmission(
        _VERIFIED_EXPIRED_RECOVERY_TOKEN,
        projection,
        {"loader_arguments": loader_arguments},
    )


def load_and_verify_eaaef_expired_lane_recovery_admission_v2(
    repo_root: str | Path,
    *,
    source_head: str,
    plan_root_cid: str,
    lane_session_id: str,
    lane_generation: int,
    registry_prefix: str,
    operational_capability_registry_prefix: str,
    expected_operational_capability_file_sha256: str,
    expected_lane_authority_file_sha256: str,
    expected_verifier_receipt_file_sha256: str,
    expected_merge_admission_file_sha256: str,
    trusted_operational_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    trusted_lane_authority_reviewer_dids: Sequence[str],
    trusted_lane_verifier_reviewer_dids: Sequence[str],
    trusted_lane_merge_reviewer_dids: Sequence[str],
    expected_operational_bindings: Mapping[str, Any],
    authority_verification_ms: int,
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedEAAEFExpiredLaneRecoveryAdmissionV2:
    """Reconstruct only an originally-live, now-expired v2 lane."""

    historical = _positive(authority_verification_ms, "authority_verification_ms")
    current = _positive(now_ms, "recovery now_ms")
    live = load_and_verify_eaaef_lane_runtime_admission_v2(
        repo_root,
        source_head=source_head,
        plan_root_cid=plan_root_cid,
        lane_session_id=lane_session_id,
        lane_generation=lane_generation,
        registry_prefix=registry_prefix,
        operational_capability_registry_prefix=(
            operational_capability_registry_prefix
        ),
        expected_operational_capability_file_sha256=(
            expected_operational_capability_file_sha256
        ),
        expected_lane_authority_file_sha256=expected_lane_authority_file_sha256,
        expected_verifier_receipt_file_sha256=(
            expected_verifier_receipt_file_sha256
        ),
        expected_merge_admission_file_sha256=expected_merge_admission_file_sha256,
        trusted_operational_reviewer_dids=trusted_operational_reviewer_dids,
        trusted_authorization_service_reviewer_dids=(
            trusted_authorization_service_reviewer_dids
        ),
        trusted_lane_authority_reviewer_dids=trusted_lane_authority_reviewer_dids,
        trusted_lane_verifier_reviewer_dids=trusted_lane_verifier_reviewer_dids,
        trusted_lane_merge_reviewer_dids=trusted_lane_merge_reviewer_dids,
        expected_operational_bindings=expected_operational_bindings,
        forbidden_reviewer_dids=forbidden_reviewer_dids,
        now_ms=historical,
    )
    if not int(live["issued_at_ms"]) <= historical < int(
        live["expires_at_ms"]
    ) <= current:
        raise EAAEFLaneGatewayAdmissionError(
            "expired v2 lane times do not prove a historical live admission"
        )
    return VerifiedEAAEFExpiredLaneRecoveryAdmissionV2(
        _VERIFIED_EXPIRED_RECOVERY_V2_TOKEN,
        dict(live),
        {"loader_arguments": dict(live._coordinates["loader_arguments"])},
    )


__all__ = (
    "EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_INTERFACE",
    "EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_PATH_TEMPLATE",
    "EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_SCHEMA",
    "EAAEF_CONTAINER_DISPATCHER_FACTORY_REVIEW_ROLE",
    "EAAEF_CONTAINER_DYNAMIC_SERVICE_INTERFACE",
    "EAAEF_LANE_AUTHORITY_INTERFACE",
    "EAAEF_LANE_AUTHORITY_PATH_TEMPLATE",
    "EAAEF_LANE_AUTHORITY_REVIEW_ROLE",
    "EAAEF_LANE_AUTHORITY_SCHEMA",
    "EAAEF_LANE_AUTHORITY_V2_INTERFACE",
    "EAAEF_LANE_AUTHORITY_V2_PATH_TEMPLATE",
    "EAAEF_LANE_AUTHORITY_V2_SCHEMA",
    "EAAEF_LANE_MERGE_ADMISSION_INTERFACE",
    "EAAEF_LANE_MERGE_ADMISSION_PATH_TEMPLATE",
    "EAAEF_LANE_MERGE_ADMISSION_REVIEW_ROLE",
    "EAAEF_LANE_MERGE_ADMISSION_SCHEMA",
    "EAAEF_LANE_MERGE_ADMISSION_V2_INTERFACE",
    "EAAEF_LANE_MERGE_ADMISSION_V2_PATH_TEMPLATE",
    "EAAEF_LANE_MERGE_ADMISSION_V2_SCHEMA",
    "EAAEF_LANE_VERIFIER_RECEIPT_INTERFACE",
    "EAAEF_LANE_VERIFIER_RECEIPT_PATH_TEMPLATE",
    "EAAEF_LANE_VERIFIER_RECEIPT_SCHEMA",
    "EAAEF_LANE_VERIFIER_REVIEW_ROLE",
    "EAAEF_LANE_VERIFIER_RECEIPT_V2_INTERFACE",
    "EAAEF_LANE_VERIFIER_RECEIPT_V2_PATH_TEMPLATE",
    "EAAEF_LANE_VERIFIER_RECEIPT_V2_SCHEMA",
    "EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_INTERFACE",
    "EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_PATH_TEMPLATE",
    "EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_SCHEMA",
    "EAAEF_QUACK_CLIENT_FACTORY_REVIEW_ROLE",
    "EAAEFLaneAdmissionSourceCoordinates",
    "EAAEFLaneGatewayAdmissionError",
    "EAAEFLaneRuntimeDependencySourceCoordinates",
    "VerifiedEAAEFContainerDispatcherFactoryQualification",
    "VerifiedEAAEFExpiredLaneRecoveryAdmission",
    "VerifiedEAAEFExpiredLaneRecoveryAdmissionV2",
    "VerifiedEAAEFLaneRuntimeAdmission",
    "VerifiedEAAEFLaneRuntimeAdmissionV2",
    "VerifiedEAAEFLaneRuntimeSourceArtifacts",
    "VerifiedEAAEFProcessBirth",
    "VerifiedEAAEFQuackClientFactoryQualification",
    "eaaef_container_dispatcher_factory_qualification_relative_path",
    "eaaef_container_dispatcher_source_sha256",
    "eaaef_current_executable_sha256",
    "eaaef_lane_authority_relative_path",
    "eaaef_lane_authority_v2_relative_path",
    "eaaef_lane_admission_source_coordinates",
    "eaaef_lane_gateway_source_identities",
    "eaaef_lane_merge_admission_relative_path",
    "eaaef_lane_merge_admission_v2_relative_path",
    "eaaef_lane_runtime_dependency_source_coordinates",
    "eaaef_lane_verifier_receipt_relative_path",
    "eaaef_lane_verifier_receipt_v2_relative_path",
    "eaaef_launch_argv_cid",
    "eaaef_quack_client_factory_qualification_relative_path",
    "load_and_verify_eaaef_container_dispatcher_factory_qualification",
    "load_and_verify_eaaef_expired_lane_recovery_admission",
    "load_and_verify_eaaef_expired_lane_recovery_admission_v2",
    "load_and_verify_eaaef_lane_runtime_admission",
    "load_and_verify_eaaef_lane_runtime_admission_from_coordinates",
    "load_and_verify_eaaef_lane_runtime_admission_v2",
    "load_and_verify_eaaef_lane_runtime_source_artifacts",
    "load_and_verify_eaaef_quack_client_factory_qualification",
    "parse_eaaef_lane_admission_source_coordinates",
    "parse_eaaef_lane_runtime_dependency_source_coordinates",
    "seal_eaaef_container_dispatcher_factory_qualification",
    "seal_eaaef_lane_authority",
    "seal_eaaef_lane_authority_v2",
    "seal_eaaef_lane_merge_admission",
    "seal_eaaef_lane_merge_admission_v2",
    "seal_eaaef_lane_verifier_receipt",
    "seal_eaaef_lane_verifier_receipt_v2",
    "seal_eaaef_quack_client_factory_qualification",
    "verify_eaaef_current_process_birth",
)
