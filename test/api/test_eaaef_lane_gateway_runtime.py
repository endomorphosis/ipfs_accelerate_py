from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import eaaef_bootstrap_gateway as runtime
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    authorized_state_command_signing_payload,
    seal_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandClient,
    QuackReadClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
    quack_daemon_operation_command_vocabulary,
    quack_daemon_operation_intent,
    quack_daemon_operation_intent_from_envelope,
    quack_daemon_state_command_parameters,
    require_quack_daemon_command_gateway,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    agent_native_dependency_admission as native,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_bootstrap_gateway_launch as launch,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_lane_gateway_admission as lane,
)
from test.api.test_agent_supervisor_native_dependency_admission import _pin as _native_pin
from test.api.test_eaaef_bootstrap_gateway_launch import (
    NOW_MS,
    _client,
    _expected_bindings,
    _key,
    _signature,
    _signed_capability,
)

from ipfs_accelerate_py import llm_router


def _file_sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_record(root: Path, relative: Path, value: object) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        current.chmod(0o700)
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    path.write_bytes(raw)
    path.chmod(0o600)
    return _file_sha(raw)


def _admission_bundle(
    tmp_path: Path,
    *,
    capability_bundle: tuple[dict[str, object], dict[str, object]] | None = None,
    lane_session_id: str = "session:eaaef-lane-0-birth-17",
    lane_generation: int = 17,
    merge_expires_at_ms: int = NOW_MS + 20_000,
    birth_overrides: dict[str, object] | None = None,
) -> tuple[
    lane.VerifiedEAAEFLaneRuntimeAdmissionV2,
    dict[str, object],
    dict[str, object],
]:
    tmp_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp_path.chmod(0o700)
    if capability_bundle is None:
        capability, base_context = _signed_capability()
    else:
        capability, base_context = capability_bundle
    context = dict(base_context)
    registry = "authority/eaaef"
    operational_relative = launch.eaaef_bootstrap_operational_capability_relative_path(
        str(capability["source_head"]),
        str(capability["active_plan_root_cid"]),
        registry_prefix=registry,
    )
    operational_sha = _write_record(tmp_path, operational_relative, capability)

    lane_key, lane_reviewer = _key()
    verifier_key, verifier_reviewer = _key()
    merge_key, merge_reviewer = _key()
    lane_tag = lane_session_id.rsplit(":", 1)[-1]
    source_ids = dict(lane.eaaef_lane_gateway_source_identities())
    lane_statement: dict[str, object] = {
        "schema": lane.EAAEF_LANE_AUTHORITY_V2_SCHEMA,
        "interface": lane.EAAEF_LANE_AUTHORITY_V2_INTERFACE,
        "board_namespace": capability["board_namespace"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "active_plan_root_cid": capability["active_plan_root_cid"],
        "active_plan_revision": capability["active_plan_revision"],
        "active_plan_revision_cid": "sha256:" + "8" * 64,
        "slice_manifest_cid": "sha256:" + "9" * 64,
        "slice_id": "slice:eaaef:0",
        "lane_id": "lane:eaaef:0",
        "task_ids": ["task:eaaef:1"],
        "task_cids": ["task:eaaef:1"],
        "operational_capability_cid": capability["capability_cid"],
        "operational_capability_file_sha256": operational_sha,
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "owner_principal_did": capability["owner_principal_did"],
        "owner_session_id": capability["owner_session_id"],
        "owner_generation": capability["owner_generation"],
        "fence_epoch": capability["fence_epoch"],
        "lane_principal_did": capability["command_principal_did"],
        "lane_role": "database_implementation_daemon",
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": f"process:{lane_tag}",
        "process_birth_nonce": f"birth:{lane_tag}",
        "journal_namespace": f"journal:{lane_tag}",
        "expected_process_uid": os.geteuid(),
        "expected_parent_pid": os.getppid(),
        "expected_parent_process_start_time_ticks": lane._linux_process_start_time_ticks(
            os.getppid()
        ),
        "expected_executable_sha256": lane.eaaef_current_executable_sha256(),
        "launch_argv_cid": lane.eaaef_launch_argv_cid(sys.argv),
        "native_dependency_admission_cid": "sha256:" + "a" * 64,
        "native_dependency_admission_file_sha256": "sha256:" + "f" * 64,
        "quack_client_factory_qualification_cid": "sha256:" + "d" * 64,
        "quack_client_factory_qualification_file_sha256": "sha256:" + "1" * 64,
        "container_dispatcher_factory_qualification_cid": "sha256:" + "e" * 64,
        "container_dispatcher_factory_qualification_file_sha256": "sha256:" + "2" * 64,
        "command_secret_descriptor_sha256": "sha256:" + "b" * 64,
        "command_secret_generation": 1,
        "state_secret_descriptor_sha256": "sha256:" + "c" * 64,
        "state_secret_generation": 1,
        "direct_database_open": False,
        "arbitrary_sql_enabled": False,
        "callback_dispatch_enabled": False,
        "raw_token_available": False,
        "issued_at_ms": NOW_MS - 400,
        "expires_at_ms": NOW_MS + 40_000,
        "reviewer_did": lane_reviewer,
        "reviewer_role": lane.EAAEF_LANE_AUTHORITY_REVIEW_ROLE,
    }
    lane_statement.update(birth_overrides or {})
    lane_authority = dict(
        lane.seal_eaaef_lane_authority_v2(
            lane_statement,
            reviewer_signature=_signature(lane_key, lane_statement),
        )
    )
    lane_relative = lane.eaaef_lane_authority_v2_relative_path(
        str(capability["source_head"]),
        str(capability["active_plan_root_cid"]),
        lane_session_id,
        lane_generation,
        registry_prefix=registry,
    )
    lane_sha = _write_record(tmp_path, lane_relative, lane_authority)

    verifier_statement: dict[str, object] = {
        "schema": lane.EAAEF_LANE_VERIFIER_RECEIPT_V2_SCHEMA,
        "interface": lane.EAAEF_LANE_VERIFIER_RECEIPT_V2_INTERFACE,
        "board_namespace": capability["board_namespace"],
        "lane_authority_cid": lane_authority["authority_cid"],
        "lane_authority_file_sha256": lane_sha,
        "operational_capability_cid": capability["capability_cid"],
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": lane_statement["process_instance_id"],
        "process_birth_nonce": lane_statement["process_birth_nonce"],
        **{name: lane_statement[name] for name in lane._BIRTH_PLAN_FIELDS},
        **source_ids,
        "checks": list(lane._VERIFIER_CHECKS_V2),
        "verification_outcome": "admitted",
        "issued_at_ms": NOW_MS - 300,
        "expires_at_ms": NOW_MS + 30_000,
        "reviewer_did": verifier_reviewer,
        "reviewer_role": lane.EAAEF_LANE_VERIFIER_REVIEW_ROLE,
    }
    verifier = dict(
        lane.seal_eaaef_lane_verifier_receipt_v2(
            verifier_statement,
            reviewer_signature=_signature(verifier_key, verifier_statement),
        )
    )
    verifier_relative = lane.eaaef_lane_verifier_receipt_v2_relative_path(
        str(lane_authority["authority_cid"]), registry_prefix=registry
    )
    verifier_sha = _write_record(tmp_path, verifier_relative, verifier)

    merge_statement: dict[str, object] = {
        "schema": lane.EAAEF_LANE_MERGE_ADMISSION_V2_SCHEMA,
        "interface": lane.EAAEF_LANE_MERGE_ADMISSION_V2_INTERFACE,
        "board_namespace": capability["board_namespace"],
        "lane_authority_cid": lane_authority["authority_cid"],
        "lane_authority_file_sha256": lane_sha,
        "verifier_receipt_cid": verifier["receipt_cid"],
        "verifier_receipt_file_sha256": verifier_sha,
        "operational_capability_cid": capability["capability_cid"],
        "operational_capability_file_sha256": operational_sha,
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "active_plan_root_cid": capability["active_plan_root_cid"],
        "active_plan_revision": capability["active_plan_revision"],
        "operation_vocabulary_cid": capability["operation_vocabulary_cid"],
        "command_fabric_qualification_cid": capability["command_fabric_qualification_cid"],
        "borrowed_transaction_adapter_qualification_cid": capability[
            "borrowed_transaction_adapter_qualification_cid"
        ],
        **{name: lane_statement[name] for name in lane._BIRTH_PLAN_FIELDS},
        **source_ids,
        "admission_outcome": "admitted",
        "issued_at_ms": NOW_MS - 200,
        "expires_at_ms": merge_expires_at_ms,
        "reviewer_did": merge_reviewer,
        "reviewer_role": lane.EAAEF_LANE_MERGE_ADMISSION_REVIEW_ROLE,
    }
    merge = dict(
        lane.seal_eaaef_lane_merge_admission_v2(
            merge_statement,
            reviewer_signature=_signature(merge_key, merge_statement),
        )
    )
    merge_relative = lane.eaaef_lane_merge_admission_v2_relative_path(
        str(lane_authority["authority_cid"]), registry_prefix=registry
    )
    merge_sha = _write_record(tmp_path, merge_relative, merge)
    loader_arguments = {
        "repo_root": tmp_path,
        "source_head": str(capability["source_head"]),
        "plan_root_cid": str(capability["active_plan_root_cid"]),
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "registry_prefix": registry,
        "operational_capability_registry_prefix": registry,
        "expected_operational_capability_file_sha256": operational_sha,
        "expected_lane_authority_file_sha256": lane_sha,
        "expected_verifier_receipt_file_sha256": verifier_sha,
        "expected_merge_admission_file_sha256": merge_sha,
        "trusted_operational_reviewer_dids": [context["operational_reviewer"]],
        "trusted_authorization_service_reviewer_dids": [context["service_reviewer"]],
        "trusted_lane_authority_reviewer_dids": [lane_reviewer],
        "trusted_lane_verifier_reviewer_dids": [verifier_reviewer],
        "trusted_lane_merge_reviewer_dids": [merge_reviewer],
        "expected_operational_bindings": _expected_bindings(capability),
    }
    admission = lane.load_and_verify_eaaef_lane_runtime_admission_v2(
        **loader_arguments,
        now_ms=NOW_MS,
    )
    context["artifact_paths"] = {
        "lane": tmp_path / lane_relative,
        "verifier": tmp_path / verifier_relative,
        "merge": tmp_path / merge_relative,
    }
    context["lane_loader_arguments"] = loader_arguments
    context["lane_statement"] = dict(lane_statement)
    return admission, capability, context


def _qualified_source_artifacts(
    tmp_path: Path,
) -> tuple[
    lane.VerifiedEAAEFLaneRuntimeSourceArtifacts,
    runtime.EAAEFSealedQuackSecretDescriptor,
    runtime.EAAEFSealedQuackSecretDescriptor,
    dict[str, object],
]:
    """Build genuine signed source artifacts without opening a service/client."""

    tmp_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp_path.chmod(0o700)
    capability, base_context = _signed_capability()
    context = dict(base_context)
    verified_capability = launch.verify_eaaef_bootstrap_operational_capability(
        capability,
        trusted_reviewer_dids=(context["operational_reviewer"],),
        trusted_authorization_service_reviewer_dids=(context["service_reviewer"],),
        now_ms=NOW_MS,
        expected=_expected_bindings(capability),
    )
    lane_session_id = "session:eaaef-qualified-birth-23"
    lane_generation = 23
    process_instance_id = "process:eaaef-qualified-birth-23"
    process_birth_nonce = "birth:eaaef:qualified:23"
    active_plan_revision_cid = "sha256:" + "8" * 64
    slice_manifest_cid = "sha256:" + "9" * 64
    launch_argv_cid = lane.eaaef_launch_argv_cid(sys.argv)
    command_secret = runtime.create_eaaef_sealed_quack_secret_descriptor(
        operational_capability=verified_capability,
        purpose="command",
        lane_session_id=lane_session_id,
        lane_generation=lane_generation,
        process_instance_id=process_instance_id,
        process_birth_nonce=process_birth_nonce,
        secret_generation=1,
        token="command-token-" + "a" * 64,
    )
    state_secret = runtime.create_eaaef_sealed_quack_secret_descriptor(
        operational_capability=verified_capability,
        purpose="state",
        lane_session_id=lane_session_id,
        lane_generation=lane_generation,
        process_instance_id=process_instance_id,
        process_birth_nonce=process_birth_nonce,
        secret_generation=1,
        token="state-token-" + "b" * 64,
    )

    pin = _native_pin()
    native_key, native_reviewer = _key()
    native_bindings: dict[str, object] = {
        "board_namespace": capability["board_namespace"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "configuration_root": capability["configuration_root"],
        "accepted_control_plane_capsule_id": capability["configured_board_capsule_cid"],
        "accepted_control_plane_pin_cid": "sha256:" + "5" * 64,
        "active_plan_root_cid": capability["active_plan_root_cid"],
        "active_plan_revision": capability["active_plan_revision"],
        "active_plan_revision_cid": active_plan_revision_cid,
        "slice_manifest_cid": slice_manifest_cid,
        "slice_id": "slice:eaaef:0",
        "lane_id": "lane:eaaef:0",
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": process_instance_id,
        "process_birth_nonce": process_birth_nonce,
        "expected_process_uid": os.geteuid(),
        "expected_parent_pid": os.getppid(),
        "expected_parent_process_start_time_ticks": lane._linux_process_start_time_ticks(
            os.getppid()
        ),
        "expected_executable_sha256": lane.eaaef_current_executable_sha256(),
        "launch_argv_cid": launch_argv_cid,
    }
    native_statement: dict[str, object] = {
        "schema": native.AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_SCHEMA,
        "interface": native.AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_INTERFACE,
        **native_bindings,
        "native_dependency_pin": pin.as_dict(),
        "native_dependency_pin_cid": pin.dependency_id,
        "sealed_descriptor_required": True,
        "ambient_loader_environment_allowed": False,
        "raw_path_authority": False,
        "launch_authority_granted": False,
        "admission_outcome": "admitted",
        "issued_at_ms": NOW_MS - 1_000,
        "expires_at_ms": NOW_MS + 45_000,
        "issuance_nonce": "nonce:eaaef-native-qualified-23",
        "reviewer_did": native_reviewer,
        "reviewer_role": (
            native.AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_REVIEWER_ROLE
        ),
    }
    native_record = dict(
        native.seal_agent_supervisor_native_dependency_admission(
            native_statement,
            reviewer_signature=_signature(native_key, native_statement),
        )
    )
    registry = "authority/eaaef"
    native_relative = native.agent_supervisor_native_dependency_admission_relative_path(
        str(capability["source_head"]),
        str(capability["active_plan_root_cid"]),
        lane_session_id,
        lane_generation,
        registry_prefix=registry,
    )
    native_file_sha = _write_record(tmp_path, native_relative, native_record)
    native_admission = native.load_and_verify_agent_supervisor_native_dependency_admission(
        tmp_path,
        source_head=str(capability["source_head"]),
        active_plan_root_cid=str(capability["active_plan_root_cid"]),
        lane_session_id=lane_session_id,
        lane_generation=lane_generation,
        registry_prefix=registry,
        expected_file_sha256=native_file_sha,
        trusted_reviewer_dids=(native_reviewer,),
        expected_native_dependency_pin=pin,
        expected_bindings=native_bindings,
        now_ms=NOW_MS,
    )

    extension_relative = Path("authority/eaaef/quack/test-quack.duckdb_extension")
    extension_bytes = b"qualified-test-quack-extension"
    extension_path = tmp_path / extension_relative
    extension_path.parent.mkdir(parents=True, mode=0o700)
    (tmp_path / "authority").chmod(0o700)
    (tmp_path / "authority" / "eaaef").chmod(0o700)
    extension_path.parent.chmod(0o700)
    extension_path.write_bytes(extension_bytes)
    extension_path.chmod(0o400)
    source_ids = dict(lane.eaaef_lane_gateway_source_identities())
    quack_key, quack_reviewer = _key()
    quack_statement: dict[str, object] = {
        "schema": lane.EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_SCHEMA,
        "interface": lane.EAAEF_QUACK_CLIENT_FACTORY_QUALIFICATION_INTERFACE,
        "board_namespace": capability["board_namespace"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "active_plan_root_cid": capability["active_plan_root_cid"],
        "active_plan_revision": capability["active_plan_revision"],
        "active_plan_revision_cid": active_plan_revision_cid,
        "slice_manifest_cid": slice_manifest_cid,
        "slice_id": "slice:eaaef:0",
        "lane_id": "lane:eaaef:0",
        "task_ids": ["task:eaaef:1"],
        "task_cids": ["task:eaaef:1"],
        "operational_capability_cid": capability["capability_cid"],
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "command_fabric_qualification_cid": capability[
            "command_fabric_qualification_cid"
        ],
        "native_dependency_admission_cid": native_record["admission_cid"],
        "native_dependency_admission_file_sha256": native_file_sha,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": process_instance_id,
        "process_birth_nonce": process_birth_nonce,
        "command_endpoint": capability["command_endpoint"],
        "command_secret_handle": capability["command_secret_handle"],
        "command_secret_generation": 1,
        "command_secret_descriptor_sha256": command_secret.sha256,
        "state_endpoint": capability["state_endpoint"],
        "state_secret_handle": capability["state_secret_handle"],
        "state_secret_generation": 1,
        "state_secret_descriptor_sha256": state_secret.sha256,
        "quack_extension_relative_path": extension_relative.as_posix(),
        "quack_extension_sha256": _file_sha(extension_bytes),
        "secret_resolver_mode": "inherited_write_sealed_memfd",
        "raw_token_argv_enabled": False,
        "raw_token_environment_enabled": False,
        "raw_token_path_enabled": False,
        **source_ids,
        "issued_at_ms": NOW_MS - 1_000,
        "expires_at_ms": NOW_MS + 45_000,
        "reviewer_did": quack_reviewer,
        "reviewer_role": lane.EAAEF_QUACK_CLIENT_FACTORY_REVIEW_ROLE,
    }
    quack_record = dict(
        lane.seal_eaaef_quack_client_factory_qualification(
            quack_statement,
            reviewer_signature=_signature(quack_key, quack_statement),
        )
    )
    quack_relative = lane.eaaef_quack_client_factory_qualification_relative_path(
        str(capability["source_head"]),
        lane_session_id,
        lane_generation,
        registry_prefix=registry,
    )
    quack_file_sha = _write_record(tmp_path, quack_relative, quack_record)

    verifier_service_key, verifier_service_did = _key()
    merge_service_key, merge_service_did = _key()
    host_service_key, host_service_did = _key()
    service_dids = {
        "worker": str(capability["worker_principal_did"]),
        "verifier": verifier_service_did,
        "merge": merge_service_did,
        "host_source": host_service_did,
    }
    service_methods = {
        "worker": ["packet", "qualify", "launch"],
        "verifier": ["verify"],
        "merge": ["observe_merge"],
        "host_source": ["observe_source"],
    }
    services = {
        name: {
            "interface": lane.EAAEF_CONTAINER_DYNAMIC_SERVICE_INTERFACE,
            "endpoint": f"unix:/run/eaaef/test-{name}-23.sock",
            "service_principal_did": service_dids[name],
            "expected_server_uid": os.geteuid(),
            "expected_server_pid": os.getpid(),
            "expected_server_process_start_time_ticks": (
                lane._linux_process_start_time_ticks(os.getpid())
            ),
            "methods": methods,
            "peer_credentials_required": True,
            "response_signature_verification_required": True,
            "request_lane_reverification_required": True,
            "maximum_request_bytes": 65_536,
            "maximum_response_bytes": 262_144,
            "request_timeout_ms": 500,
        }
        for name, methods in service_methods.items()
    }
    dispatcher_key, dispatcher_reviewer = _key()
    dispatcher_statement: dict[str, object] = {
        "schema": lane.EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_SCHEMA,
        "interface": lane.EAAEF_CONTAINER_DISPATCHER_FACTORY_QUALIFICATION_INTERFACE,
        "board_namespace": capability["board_namespace"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "active_plan_root_cid": capability["active_plan_root_cid"],
        "active_plan_revision": capability["active_plan_revision"],
        "active_plan_revision_cid": active_plan_revision_cid,
        "slice_manifest_cid": slice_manifest_cid,
        "slice_id": "slice:eaaef:0",
        "lane_id": "lane:eaaef:0",
        "task_ids": ["task:eaaef:1"],
        "task_cids": ["task:eaaef:1"],
        "operational_capability_cid": capability["capability_cid"],
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "native_dependency_admission_cid": native_record["admission_cid"],
        "native_dependency_admission_file_sha256": native_file_sha,
        "quack_client_factory_qualification_cid": quack_record["qualification_cid"],
        "quack_client_factory_qualification_file_sha256": quack_file_sha,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": process_instance_id,
        "process_birth_nonce": process_birth_nonce,
        "worker_principal_did": capability["worker_principal_did"],
        "services": services,
        "dispatcher_source_sha256": lane.eaaef_container_dispatcher_source_sha256(),
        **source_ids,
        "dynamic_per_attempt_verification_required": True,
        "dynamic_per_attempt_merge_admission_required": True,
        "static_lane_evidence_is_attempt_success": False,
        "caller_callbacks_allowed": False,
        "direct_container_launch_allowed": False,
        "issued_at_ms": NOW_MS - 1_000,
        "expires_at_ms": NOW_MS + 45_000,
        "reviewer_did": dispatcher_reviewer,
        "reviewer_role": lane.EAAEF_CONTAINER_DISPATCHER_FACTORY_REVIEW_ROLE,
    }
    dispatcher_record = dict(
        lane.seal_eaaef_container_dispatcher_factory_qualification(
            dispatcher_statement,
            reviewer_signature=_signature(dispatcher_key, dispatcher_statement),
        )
    )
    dispatcher_relative = (
        lane.eaaef_container_dispatcher_factory_qualification_relative_path(
            str(capability["source_head"]),
            lane_session_id,
            lane_generation,
            registry_prefix=registry,
        )
    )
    dispatcher_file_sha = _write_record(
        tmp_path, dispatcher_relative, dispatcher_record
    )

    admission, _capability, lane_context = _admission_bundle(
        tmp_path,
        capability_bundle=(capability, base_context),
        lane_session_id=lane_session_id,
        lane_generation=lane_generation,
        birth_overrides={
            "active_plan_revision_cid": active_plan_revision_cid,
            "slice_manifest_cid": slice_manifest_cid,
            "process_instance_id": process_instance_id,
            "process_birth_nonce": process_birth_nonce,
            "launch_argv_cid": launch_argv_cid,
            "native_dependency_admission_cid": native_record["admission_cid"],
            "native_dependency_admission_file_sha256": native_file_sha,
            "quack_client_factory_qualification_cid": quack_record[
                "qualification_cid"
            ],
            "quack_client_factory_qualification_file_sha256": quack_file_sha,
            "container_dispatcher_factory_qualification_cid": dispatcher_record[
                "qualification_cid"
            ],
            "container_dispatcher_factory_qualification_file_sha256": (
                dispatcher_file_sha
            ),
            "command_secret_descriptor_sha256": command_secret.sha256,
            "state_secret_descriptor_sha256": state_secret.sha256,
        },
    )
    quack_qualification = lane.load_and_verify_eaaef_quack_client_factory_qualification(
        admission=admission,
        native_admission=native_admission,
        registry_prefix=registry,
        expected_file_sha256=quack_file_sha,
        trusted_reviewer_dids=(quack_reviewer,),
        now_ms=NOW_MS,
    )
    dispatcher_qualification = (
        lane.load_and_verify_eaaef_container_dispatcher_factory_qualification(
            admission=admission,
            native_admission=native_admission,
            quack_qualification=quack_qualification,
            registry_prefix=registry,
            expected_file_sha256=dispatcher_file_sha,
            trusted_reviewer_dids=(dispatcher_reviewer,),
            now_ms=NOW_MS,
        )
    )
    dependency_coordinates = lane.eaaef_lane_runtime_dependency_source_coordinates(
        admission=admission,
        native_admission=native_admission,
        quack_qualification=quack_qualification,
        dispatcher_qualification=dispatcher_qualification,
    )
    artifacts = lane.load_and_verify_eaaef_lane_runtime_source_artifacts(
        tmp_path,
        coordinates=lane.parse_eaaef_lane_runtime_dependency_source_coordinates(
            dependency_coordinates.to_dict()
        ),
        now_ms=NOW_MS,
    )
    context.update(lane_context)
    context.update(
        {
            "native_pin": pin,
            "native_bindings": native_bindings,
            "extension_path": extension_path,
            "quack_file_sha": quack_file_sha,
            "dispatcher_file_sha": dispatcher_file_sha,
            "dependency_coordinates": dependency_coordinates,
            "verifier_service_key": verifier_service_key,
            "verifier_service_did": verifier_service_did,
        }
    )
    return artifacts, command_secret, state_secret, context


def _envelope(
    intent: dict[str, object],
    capability: dict[str, object],
    context: dict[str, object],
    *,
    serial: int,
) -> object:
    operation = str(intent["operation"])
    arguments = dict(intent["arguments"])
    authority = arguments.get("task_authority_binding")
    if isinstance(authority, dict):
        lease_id = str(authority["lease_id"])
        scope_id = str(authority["task_cid"])
        fencing_token = int(authority["fencing_token"])
    else:
        lease_id = str(capability["lease_id"])
        scope_id = str(capability["board_scope"])
        fencing_token = int(capability["fencing_token"])
    request_id = f"request:eaaef-runtime:{serial}"
    idempotency_key = f"idempotency:eaaef-runtime:{serial}"
    deadline_ms = NOW_MS + 5_000
    parameters = dict(
        quack_daemon_state_command_parameters(
            intent,
            request_id=request_id,
            principal_did=str(context["worker_did"]),
            authority_ref_cid=str(context["policy"].authority_ref_cid),
            lease_id=lease_id,
            scope_id=scope_id,
            deadline_ms=deadline_ms,
            fencing_token=fencing_token,
            idempotency_key=idempotency_key,
        )
    )
    parameters["authorization_request_cid"] = "sha256:" + format(serial, "064x")
    kind = CommandKind(quack_daemon_operation_command_vocabulary()[operation])
    command = StateCommand(
        command_id=f"{request_id}:{operation.replace('.', '-')}",
        command_kind=kind,
        store_id=str(capability["store_id"]),
        session_id=lease_id,
        expected_generation=int(capability["owner_generation"]),
        expected_revision=1,
        fence_epoch=int(capability["fence_epoch"]),
        idempotency_key=idempotency_key,
        parameters=parameters,
    )
    prepared = authorized_state_command_signing_payload(
        request_id=request_id,
        submission_id=f"submission:eaaef-runtime:{serial}",
        ingress_slot=serial,
        principal_did=str(context["worker_did"]),
        approver_did=str(context["approver_did"]),
        authority_ref_cid=str(context["policy"].authority_ref_cid),
        board_namespace=str(capability["board_namespace"]),
        shard_id=str(capability["shard_id"]),
        owner_principal_did=str(context["owner_did"]),
        lease_id=lease_id,
        scope_id=scope_id,
        effect=f"control-plane/{kind.value}",
        issued_at_ms=NOW_MS - 100,
        expires_at_ms=NOW_MS + 10_000,
        deadline_ms=deadline_ms,
        one_use_nonce=f"nonce:eaaef-runtime:{serial}",
        command=command,
    )
    return seal_authorized_state_command(
        prepared,
        approver_signature=_signature(context["approver_key"], dict(prepared)),
    )


def _receipt(envelope: object, *, value: object) -> dict[str, object]:
    intent = dict(quack_daemon_operation_intent_from_envelope(envelope))
    command = envelope.command
    return {
        "submission_id": envelope.submission_id,
        "envelope_cid": envelope.envelope_cid,
        "request_id": envelope.request_id,
        "principal_did": envelope.principal_did,
        "approver_did": envelope.approver_did,
        "authority_ref_cid": envelope.authority_ref_cid,
        "lease_id": envelope.lease_id,
        "one_use_nonce": envelope.one_use_nonce,
        "command_id": command.command_id,
        "idempotency_key": command.idempotency_key,
        "outcome": "accepted",
        "changed": True,
        "revision": 1,
        "generation": command.expected_generation,
        "fence_epoch": command.fence_epoch,
        "result_json": json.dumps(
            {
                "daemon_operation": intent["operation"],
                "intent_cid": intent["intent_cid"],
                "value": value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "error": "",
        "submitted_at": NOW_MS,
        "applied_at": NOW_MS,
    }


def _clients(capability: dict[str, object]) -> tuple[QuackCommandClient, QuackReadClient]:
    command = object.__new__(QuackCommandClient)
    command._endpoint = str(capability["command_endpoint"])
    command._closed = False
    read = object.__new__(QuackReadClient)
    read._endpoint = str(capability["state_endpoint"])
    read._closed = False
    return command, read


def test_lane_admission_is_source_addressed_independent_and_reverified(
    tmp_path: Path,
) -> None:
    admission, capability, context = _admission_bundle(tmp_path)
    assert admission["process_instance_id"] == "process:eaaef-lane-0-birth-17"
    assert admission["lane_session_id"] != capability["owner_session_id"]
    assert (
        len(
            {
                context["operational_reviewer"],
                context["service_reviewer"],
                admission["lane_authority_cid"],
                admission["verifier_receipt_cid"],
                admission["merge_admission_cid"],
            }
        )
        == 5
    )
    assert (
        admission.reverify(now_ms=NOW_MS)["process_birth_nonce"]
        == (admission["process_birth_nonce"])
    )

    linked_root = tmp_path.parent / f"{tmp_path.name}-linked-root"
    linked_root.symlink_to(tmp_path, target_is_directory=True)
    linked_arguments = dict(context["lane_loader_arguments"])
    linked_arguments["repo_root"] = linked_root
    with pytest.raises(lane.EAAEFLaneGatewayAdmissionError, match="unavailable"):
        lane.load_and_verify_eaaef_lane_runtime_admission(
            **linked_arguments,
            now_ms=NOW_MS,
        )

    lane_path = context["artifact_paths"]["lane"]
    original = lane_path.read_bytes()
    lane_path.unlink()
    attacker = tmp_path / "attacker-lane.json"
    attacker.write_bytes(original)
    attacker.chmod(0o600)
    lane_path.symlink_to(attacker)
    with pytest.raises(lane.EAAEFLaneGatewayAdmissionError, match="unavailable"):
        admission.reverify(now_ms=NOW_MS)


def test_signed_runtime_dependencies_reopen_and_lazy_factories_have_no_live_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, command_secret, state_secret, context = _qualified_source_artifacts(
        tmp_path / "sources"
    )
    monkeypatch.setattr(runtime.time, "time_ns", lambda: NOW_MS * 1_000_000)
    admission = artifacts.admission
    assert type(admission) is lane.VerifiedEAAEFLaneRuntimeAdmissionV2
    assert (
        type(artifacts.native_admission)
        is native.VerifiedAgentSupervisorNativeDependencyAdmission
    )
    assert (
        type(artifacts.quack_qualification)
        is lane.VerifiedEAAEFQuackClientFactoryQualification
    )
    assert (
        type(artifacts.dispatcher_qualification)
        is lane.VerifiedEAAEFContainerDispatcherFactoryQualification
    )
    birth = lane.verify_eaaef_current_process_birth(admission)
    sealed_clients = runtime.bind_eaaef_sealed_quack_client_descriptors(
        admission=admission,
        process_birth=birth,
        command_descriptor=command_secret.descriptor,
        state_descriptor=state_secret.descriptor,
    )
    assert not hasattr(sealed_clients, "token")
    assert "command-token" not in repr(sealed_clients)

    descriptor = llm_router.AgentSupervisorNativeDependencyDescriptor(
        schema="ipfs_accelerate_py.agent_supervisor.native-dependency-descriptor@1",
        descriptor=999,
        st_dev=1,
        st_ino=1,
        st_mode=stat.S_IFREG | 0o500,
        st_uid=os.geteuid(),
        st_nlink=0,
        size_bytes=context["native_pin"].size_bytes,
        payload_sha256=context["native_pin"].payload_sha256,
        seals=15,
    )
    native_launch = llm_router.AgentSupervisorNativeDependencyLaunch(
        schema="ipfs_accelerate_py.agent_supervisor.native-dependency-launch@1",
        accepted_authorization_id=artifacts.native_admission.admission_cid,
        pin=context["native_pin"],
        descriptor=descriptor,
    )
    native_path = "/proc/self/fd/999"
    native_module = types.ModuleType("_duckdb")
    native_module.__file__ = native_path
    native_module.__version__ = context["native_pin"].distribution_version

    class FixedTemplateConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.closed = False

        def execute(self, statement: str, *_args: object) -> FixedTemplateConnection:
            self.statements.append(statement)
            return self

        def close(self) -> None:
            self.closed = True

    connections: list[FixedTemplateConnection] = []

    def connect(**_kwargs: object) -> FixedTemplateConnection:
        connection = FixedTemplateConnection()
        connections.append(connection)
        return connection

    native_module.connect = connect
    monkeypatch.setitem(sys.modules, "_duckdb", native_module)
    monkeypatch.setitem(sys.modules, "duckdb", native_module)
    monkeypatch.setattr(
        runtime,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda _launch: native_path,
    )
    monkeypatch.setattr(
        runtime.socket,
        "socket",
        lambda *_args, **_kwargs: pytest.fail(
            "lazy dependency factory opened a dynamic service socket"
        ),
    )
    dispatcher_factory = runtime.create_eaaef_container_dispatcher_factory(
        admission=admission,
        process_birth=birth,
        native_admission=artifacts.native_admission,
        quack_qualification=artifacts.quack_qualification,
        qualification=artifacts.dispatcher_qualification,
    )
    assert type(dispatcher_factory) is runtime.EAAEFContainerDispatcherFactory

    class SignedResponseConnection:
        def __init__(self, *, invalid_signature: bool = False) -> None:
            self._response = b""
            self._invalid_signature = invalid_signature

        def sendall(self, raw: bytes) -> None:
            size = runtime.struct.unpack("!I", raw[:4])[0]
            request = json.loads(raw[4:])
            assert size == len(raw) - 4
            signed_response = {
                "schema": runtime.EAAEF_CONTAINER_DYNAMIC_SERVICE_RESPONSE_SCHEMA,
                "interface": "EAAEFContainerDynamicService@1",
                "service": "verifier",
                "method": "verify",
                "request_cid": request["request_cid"],
                "service_principal_did": context["verifier_service_did"],
                "result": {"verified": True},
                "issued_at_ms": NOW_MS - 1,
                "expires_at_ms": NOW_MS + 1_000,
                "response_nonce": "response:eaaef:verifier:1",
            }
            signature = _signature(
                context["verifier_service_key"], signed_response
            )
            if self._invalid_signature:
                signature = "not-a-signature"
            response_body = {**signed_response, "service_signature": signature}
            response = {
                **response_body,
                "response_cid": runtime._content_cid(
                    response_body, "test dynamic service response"
                ),
            }
            encoded = runtime._canonical_bytes(
                response, "test dynamic service response"
            )
            self._response = runtime.struct.pack("!I", len(encoded)) + encoded

        def recv(self, size: int) -> bytes:
            result, self._response = self._response[:size], self._response[size:]
            return result

        def close(self) -> None:
            return None

    verifier_client = dispatcher_factory._services["verifier"]
    monkeypatch.setattr(
        runtime._EAAEFContainerDynamicServiceClient,
        "_connect",
        lambda _self: SignedResponseConnection(),
    )
    assert verifier_client.request("verify", {"proposal": "sha256:" + "3" * 64}) == {
        "verified": True
    }
    monkeypatch.setattr(
        runtime._EAAEFContainerDynamicServiceClient,
        "_connect",
        lambda _self: SignedResponseConnection(invalid_signature=True),
    )
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayError, match="signature"):
        verifier_client.request("verify", {"proposal": "sha256:" + "3" * 64})

    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    dependency_factory = runtime.create_eaaef_lane_runtime_dependency_factory(
        admission=admission,
        process_birth=birth,
        native_admission=artifacts.native_admission,
        native_launch=native_launch,
        native_module=native_module,
        quack_qualification=artifacts.quack_qualification,
        sealed_descriptors=sealed_clients,
        dispatcher_qualification=artifacts.dispatcher_qualification,
        authorization_client=_client(
            dict(admission.operational_capability),
            context,
            clock_ms=lambda: NOW_MS,
        ),
        journal_parent_directory=journal_parent,
    )
    assert type(dependency_factory) is runtime.EAAEFLaneRuntimeDependencyFactory
    assert not dependency_factory.journal_relative_path.is_absolute()

    bundle = dependency_factory.build()
    assert type(bundle) is runtime.EAAEFLaneRuntimeDependencyBundle
    assert type(bundle.gateway) is runtime.EAAEFBootstrapCommandGateway
    assert (
        type(bundle.container_dispatcher)
        is runtime.ExternalAgentContainerWorkerDispatcher
    )
    assert bundle.gateway.attached is False
    production = bundle.gateway.require_production_admission()
    assert production["process_birth_cid"] == birth["birth_cid"]
    assert production["plan_r2_enabled"] is False
    assert bundle.gateway.evidence()["production_blockers"] == []
    assert len(connections) == 2
    assert all("/proc/self/fd/" in connection.statements[0] for connection in connections)

    v1 = object.__new__(lane.VerifiedEAAEFLaneRuntimeAdmission)
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayError, match="rejects"):
        runtime.create_eaaef_container_dispatcher_factory(
            admission=v1,
            process_birth=birth,
            native_admission=artifacts.native_admission,
            quack_qualification=artifacts.quack_qualification,
            qualification=artifacts.dispatcher_qualification,
        )

    coordinates = context["dependency_coordinates"].to_dict()
    coordinates["native"]["expected_file_sha256"] = "sha256:" + "0" * 64
    coordinate_body = {
        name: value for name, value in coordinates.items() if name != "coordinates_cid"
    }
    coordinates["coordinates_cid"] = lane._cid(coordinate_body)
    parsed = lane.parse_eaaef_lane_runtime_dependency_source_coordinates(coordinates)
    with pytest.raises(
        native.AgentSupervisorNativeDependencyAdmissionError,
        match="source hash differs",
    ):
        lane.load_and_verify_eaaef_lane_runtime_source_artifacts(
            tmp_path / "sources", coordinates=parsed, now_ms=NOW_MS
        )

    extension_path = context["extension_path"]
    extension_raw = extension_path.read_bytes()
    extension_path.unlink()
    outside = tmp_path / "substituted-extension"
    outside.write_bytes(extension_raw)
    outside.chmod(0o400)
    extension_path.symlink_to(outside)
    with pytest.raises(lane.EAAEFLaneGatewayAdmissionError, match="unavailable"):
        artifacts.quack_qualification.reverify(now_ms=NOW_MS)

    bundle.close()
    assert bundle.gateway._dispatcher._transport._closed is True
    with pytest.raises(QuackDaemonGatewayError, match="failed attach"):
        bundle.gateway.attach()
    sealed_clients.close()
    command_secret.close()
    state_secret.close()


def test_expired_lane_is_reconstructed_only_as_typed_recovery_evidence(
    tmp_path: Path,
) -> None:
    _admission, _capability, context = _admission_bundle(tmp_path)
    expired = lane.load_and_verify_eaaef_expired_lane_recovery_admission_v2(
        **context["lane_loader_arguments"],
        authority_verification_ms=NOW_MS,
        now_ms=NOW_MS + 20_000,
    )
    assert type(expired) is lane.VerifiedEAAEFExpiredLaneRecoveryAdmissionV2
    assert (
        expired.reverify_for_recovery(now_ms=NOW_MS + 20_000)["lane_session_id"]
        == expired["lane_session_id"]
    )
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayError, match="rejects"):
        runtime.create_eaaef_bootstrap_command_gateway(
            admission=expired,
            authorization_client={},
            transport={},
            journal={},
        )


def test_active_gateway_retains_exact_expired_lane_binding_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capability, base_context = _signed_capability()
    source_root = tmp_path / "sources"
    prior, _capability, prior_context = _admission_bundle(
        source_root,
        capability_bundle=(capability, base_context),
        lane_session_id="session:eaaef-lane-prior-birth-3",
        lane_generation=3,
        merge_expires_at_ms=NOW_MS + 5_000,
    )
    active, _capability, active_context = _admission_bundle(
        source_root,
        capability_bundle=(capability, base_context),
        lane_session_id="session:eaaef-lane-active-birth-4",
        lane_generation=4,
        merge_expires_at_ms=NOW_MS + 20_000,
    )
    expired = lane.load_and_verify_eaaef_expired_lane_recovery_admission_v2(
        **prior_context["lane_loader_arguments"],
        authority_verification_ms=NOW_MS,
        now_ms=NOW_MS + 6_000,
    )
    monkeypatch.setattr(
        runtime.time,
        "time_ns",
        lambda: (NOW_MS + 6_000) * 1_000_000,
    )
    authorization_client = _client(
        capability,
        active_context,
        clock_ms=lambda: NOW_MS + 6_000,
    )
    command, read = _clients(capability)
    transport = runtime.bind_eaaef_bootstrap_command_transport(
        command_client=command,
        read_client=read,
        admission=active,
    )
    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    journal = runtime.open_eaaef_exact_envelope_journal(
        journal_parent,
        admission=active,
    )
    gateway = runtime.create_eaaef_bootstrap_command_gateway(
        admission=active,
        authorization_client=authorization_client,
        transport=transport,
        journal=journal,
        recovery_admissions=(expired,),
    )
    arguments = gateway._dispatcher.expired_running_arguments(
        limit=10,
        now_ms=NOW_MS + 6_000,
    )
    recovered_lane = arguments["recovery_authority"]["lane_bindings"][0]
    assert recovered_lane["lane_session_id"] == prior["lane_session_id"]
    assert recovered_lane["lane_generation"] == prior["lane_generation"]
    assert recovered_lane["process_instance_id"] == prior["process_instance_id"]
    assert recovered_lane["lane_session_id"] != active["lane_session_id"]


def test_plan_revision_journal_restarts_exactly_and_rejects_divergence(
    tmp_path: Path,
) -> None:
    admission, capability, context = _admission_bundle(tmp_path / "sources")
    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    journal_relative = runtime.eaaef_exact_envelope_journal_relative_path(admission)
    assert not journal_relative.is_absolute() and len(journal_relative.parts) == 1
    journal = runtime.open_eaaef_exact_envelope_journal(journal_parent, admission=admission)
    assert journal._store.root == journal_parent / journal_relative
    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="coordination.settle_claim",
            arguments={"claim_id": "claim:1"},
        )
    )
    first = _envelope(
        intent,
        capability,
        context,
        serial=1,
    )
    second = _envelope(
        intent,
        capability,
        context,
        serial=2,
    )
    operation_key = "sha256:" + "3" * 64
    journal.prepare(
        operation_key=operation_key,
        operation="coordination.settle_claim",
        intent_cid=str(intent["intent_cid"]),
        envelope=first,
    )
    # Crash after the lane pointer but before the secondary operation index.
    journal._store.clear_continuation(journal._continuation_key(operation_key))
    restarted = runtime.open_eaaef_exact_envelope_journal(journal_parent, admission=admission)
    loaded, receipt = restarted.lookup(
        operation_key=operation_key,
        operation="coordination.settle_claim",
        intent_cid=str(intent["intent_cid"]),
    )
    assert loaded.to_dict() == first.to_dict()
    assert receipt is None
    prepared_state = dict(
        restarted._store.load_continuation(restarted._continuation_key(operation_key))
    )
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayDiverged):
        restarted.prepare(
            operation_key=operation_key,
            operation="coordination.settle_claim",
            intent_cid=str(intent["intent_cid"]),
            envelope=second,
        )
    durable_receipt = {
        "submission_id": first.submission_id,
        "envelope_cid": first.envelope_cid,
        "status": "accepted",
    }
    restarted.commit_receipt(
        operation_key=operation_key,
        operation="coordination.settle_claim",
        intent_cid=str(intent["intent_cid"]),
        envelope=first,
        receipt=durable_receipt,
    )
    third = runtime.open_eaaef_exact_envelope_journal(journal_parent, admission=admission)
    assert (
        dict(
            third.lookup(
                operation_key=operation_key,
                operation="coordination.settle_claim",
                intent_cid=str(intent["intent_cid"]),
            )[1]
        )
        == durable_receipt
    )

    # Inverse crash: committed operation durable, lane pointer not yet cleared.
    # A different operation heals the stale pointer without reapplying it.
    third._store.put_continuation(third._pending_key(), prepared_state)
    other_intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.ready",
            arguments={"limit": 1},
        )
    )
    assert (
        third.lookup(
            operation_key="sha256:" + "4" * 64,
            operation="task.ready",
            intent_cid=str(other_intent["intent_cid"]),
        )
        is None
    )
    assert third._store.load_continuation(third._pending_key()) is None

    symlink = tmp_path / "journal-link"
    symlink.symlink_to(journal_parent, target_is_directory=True)
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayError, match="unsafe"):
        runtime.open_eaaef_exact_envelope_journal(symlink, admission=admission)

    outside = tmp_path / "outside-cas-record"
    outside.write_text("not journal state", encoding="utf-8")
    (third._store.cas_dir / "linked-record").symlink_to(outside)
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayError, match="unsafe"):
        runtime.open_eaaef_exact_envelope_journal(journal_parent, admission=admission)


def test_response_loss_restart_reuses_self_revoking_settle_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission, capability, context = _admission_bundle(tmp_path / "sources")
    monkeypatch.setattr(runtime.time, "time_ns", lambda: NOW_MS * 1_000_000)
    authorization_client = _client(capability, context, clock_ms=lambda: NOW_MS)
    issued: list[object] = []

    def authorize(_self: object, intent: object) -> object:
        envelope = _envelope(dict(intent), capability, context, serial=len(issued) + 1)
        issued.append(envelope)
        return envelope

    monkeypatch.setattr(type(authorization_client), "authorize", authorize)
    appended: list[object] = []
    visible_receipts: list[dict[str, object]] = []

    def append(_self: object, envelope: object) -> None:
        appended.append(envelope)
        raise RuntimeError("owner committed and response was lost")

    def receipts(_self: object) -> tuple[dict[str, object], ...]:
        return tuple(visible_receipts)

    monkeypatch.setattr(QuackCommandClient, "append", append)
    monkeypatch.setattr(QuackReadClient, "list_recent_receipts", receipts)
    monkeypatch.setattr(QuackCommandClient, "close", lambda _self: None)
    monkeypatch.setattr(QuackReadClient, "close", lambda _self: None)

    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    journal = runtime.open_eaaef_exact_envelope_journal(journal_parent, admission=admission)
    command, read = _clients(capability)
    transport = runtime.bind_eaaef_bootstrap_command_transport(
        command_client=command,
        read_client=read,
        admission=admission,
        maximum_wait_ms=1,
        poll_interval_ms=1,
    )
    gateway = runtime.create_eaaef_bootstrap_command_gateway(
        admission=admission,
        authorization_client=authorization_client,
        transport=transport,
        journal=journal,
    )
    gateway.attach()
    claim = {
        "task_cid": "task:eaaef:1",
        "claim_id": "claim:eaaef:1",
        "attempt_id": "attempt:eaaef:1",
        "attempt_number": 1,
        "lease_id": "lease:eaaef:task:1",
        "owner_session_id": admission["lane_session_id"],
        "fencing_token": 13,
        "fence_epoch": admission["fence_epoch"],
    }
    gateway._dispatcher._register_identity(claim)
    settle_kwargs = {
        "expected_revision": 4,
        "settled_at_ms": NOW_MS,
    }
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayAmbiguous):
        gateway.coordinator.settle_task_claim(claim, **settle_kwargs)
    assert len(issued) == 1
    exact = appended[0]
    before_replay = len(appended)
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayAmbiguous):
        gateway.coordinator.settle_task_claim(
            claim,
            expected_revision=4,
            settled_at_ms=NOW_MS + 1,
        )
    assert len(issued) == 1
    assert len(appended) == before_replay + 1
    assert appended[-1].to_dict() == exact.to_dict()

    collision = _receipt(exact, value={"task_cid": claim["task_cid"], "settled": True})
    collision["envelope_cid"] = "sha256:" + "f" * 64
    visible_receipts.append(collision)
    restarted_journal = runtime.open_eaaef_exact_envelope_journal(
        journal_parent, admission=admission
    )
    command2, read2 = _clients(capability)
    transport2 = runtime.bind_eaaef_bootstrap_command_transport(
        command_client=command2,
        read_client=read2,
        admission=admission,
        maximum_wait_ms=1,
        poll_interval_ms=1,
    )
    restarted = runtime.create_eaaef_bootstrap_command_gateway(
        admission=admission,
        authorization_client=authorization_client,
        transport=transport2,
        journal=restarted_journal,
    )
    restarted.attach()
    restarted._dispatcher._register_identity(claim)
    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayDiverged,
        match="submission_id collision",
    ):
        restarted.coordinator.settle_task_claim(claim, **settle_kwargs)
    visible_receipts.clear()
    visible_receipts.append(_receipt(exact, value={"task_cid": claim["task_cid"], "settled": True}))
    monkeypatch.setattr(
        runtime.time,
        "time_ns",
        lambda: (NOW_MS + 11_000) * 1_000_000,
    )
    result = restarted.coordinator.settle_task_claim(claim, **settle_kwargs)
    assert result == {"task_cid": claim["task_cid"], "settled": True}
    assert len(issued) == 1
    assert appended[-1].to_dict() == exact.to_dict()


def test_factory_is_r1_only_typed_and_never_exposes_database_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission, capability, context = _admission_bundle(tmp_path / "sources")
    monkeypatch.setattr(runtime.time, "time_ns", lambda: NOW_MS * 1_000_000)
    authorization_client = _client(capability, context, clock_ms=lambda: NOW_MS)
    command, read = _clients(capability)
    monkeypatch.setattr(QuackCommandClient, "close", lambda _self: None)
    monkeypatch.setattr(QuackReadClient, "close", lambda _self: None)
    transport = runtime.bind_eaaef_bootstrap_command_transport(
        command_client=command, read_client=read, admission=admission
    )
    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    journal = runtime.open_eaaef_exact_envelope_journal(journal_parent, admission=admission)
    gateway = runtime.create_eaaef_bootstrap_command_gateway(
        admission=admission,
        authorization_client=authorization_client,
        transport=transport,
        journal=journal,
    )
    assert (
        require_quack_daemon_command_gateway(
            gateway, expected_command_endpoint=str(capability["command_endpoint"])
        )
        is gateway
    )
    assert gateway.merge_repository is None
    assert gateway.plan_repository is None
    assert not hasattr(gateway.task_source, "execute")
    assert not hasattr(gateway.coordinator, "connection")
    assert not hasattr(gateway.execution_repository, "database_path")
    with pytest.raises(runtime.EAAEFBootstrapExcludedOperation):
        gateway.task_source.materialize({})
    with pytest.raises(Exception, match="production no-go"):
        gateway.require_production_admission()
    with pytest.raises(runtime.EAAEFBootstrapRuntimeGatewayError, match="rejects"):
        runtime.create_eaaef_bootstrap_command_gateway(
            admission=dict(admission),
            authorization_client=authorization_client,
            transport=transport,
            journal=journal,
        )
    with pytest.raises(
        runtime.EAAEFBootstrapRuntimeGatewayError,
        match="rejects callbacks",
    ):
        runtime.EAAEFBootstrapCommandTransport(
            runtime._TRANSPORT_FACTORY_TOKEN,
            command_client=lambda _value: None,
            read_client=lambda: (),
            admission=admission,
            maximum_wait_ms=1,
            poll_interval_ms=1,
        )
