from __future__ import annotations

import base64
import json
import os
import stat
import struct
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationControlPlanePin,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    QUACK_STATE_REPOSITORY_INTERFACE,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
    EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS,
    EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE,
    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
    EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
    eaaef_bootstrap_handler_source_evidence,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    _REQUIRED_COLUMNS,
    EAAEF_OPERATIONAL_PROFILE_ID,
    eaaef_operation_vocabulary_cid,
    eaaef_operational_profile_contract,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    QuackCommandAuthorizationPolicy,
    authorized_state_command_signing_payload,
    seal_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    quack_daemon_operation_command_vocabulary,
    quack_daemon_operation_intent,
    quack_daemon_state_command_parameters,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_content_cid,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_bootstrap_gateway_launch as launch,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_configured_board_capsule as configured_capsule,
)

NOW_MS = 1_800_000_000_000


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _key() -> tuple[Ed25519PrivateKey, str]:
    key = Ed25519PrivateKey.generate()
    return key, ed25519_did_key(key.public_key())


def _signature(key: Ed25519PrivateKey, value: object) -> str:
    return base64.b64encode(key.sign(launch._canonical_bytes(value))).decode("ascii")


def _pin() -> AgentImplementationControlPlanePin:
    return AgentImplementationControlPlanePin(
        schema="ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2",
        runner_path="/sealed/grok_cli_runner.py",
        runner_sha256=_sha("1"),
        capsule_root="/sealed",
        capsule_id=_sha("2"),
        source_head="3" * 40,
        source_tree="4" * 40,
        archive_sha256=_sha("5"),
    )


def _signed_capability() -> tuple[dict[str, object], dict[str, object]]:
    operational_reviewer_key, operational_reviewer = _key()
    service_reviewer_key, service_reviewer = _key()
    _service_key, service_did = _key()
    approver_key, approver_did = _key()
    _worker_key, worker_did = _key()
    _owner_key, owner_did = _key()
    vocabulary_cid = eaaef_operation_vocabulary_cid(
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
    )
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=launch.EAAEF_BOARD_NAMESPACE,
        shard_id="control-shard-0",
        store_id="eaaef-control-run-v6",
        authority_ref_cid=_sha("6"),
        owner_principal_did=owner_did,
        owner_generation=6,
        fence_epoch=9,
        trusted_approver_dids=frozenset({approver_did}),
        authorized_principal_dids=frozenset({worker_did}),
        allowed_command_kinds=frozenset(
            CommandKind(quack_daemon_operation_command_vocabulary()[operation])
            for operation in EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
        ),
    )
    service_statement: dict[str, object] = {
        "schema": launch.EAAEF_COMMAND_AUTHORIZATION_SERVICE_CAPABILITY_SCHEMA,
        "interface": launch.EAAEF_COMMAND_AUTHORIZATION_SERVICE_INTERFACE,
        "board_namespace": launch.EAAEF_BOARD_NAMESPACE,
        "transport_kind": "private_unix_length_prefixed_json",
        "endpoint": "unix:/run/eaaef/command-authorizer.sock",
        "service_principal_did": service_did,
        "approver_principal_did": approver_did,
        "authorized_client_principal_did": worker_did,
        "authorization_policy_cid": policy.policy_cid,
        "request_schema": launch.EAAEF_COMMAND_AUTHORIZATION_REQUEST_SCHEMA,
        "response_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "peer_credentials_required": True,
        "response_signature_verification_required": True,
        "private_key_available_to_child": False,
        "raw_token_available_to_child": False,
        "dynamic_endpoint_allowed": False,
        "maximum_request_bytes": 65_536,
        "maximum_response_bytes": 65_536,
        "request_timeout_ms": 500,
        "expected_server_uid": os.geteuid(),
        "expected_server_pid": os.getpid(),
        "expected_server_process_start_time_ticks": (
            launch._linux_process_start_time_ticks(os.getpid())
        ),
        "issuance_nonce": "service-capability-nonce-1",
        "issued_at_ms": NOW_MS - 1_000,
        "expires_at_ms": NOW_MS + 60_000,
        "reviewer_did": service_reviewer,
        "reviewer_role": launch.EAAEF_COMMAND_AUTHORIZATION_SERVICE_REVIEW_ROLE,
    }
    service = dict(
        launch.seal_eaaef_command_authorization_service_capability(
            service_statement,
            reviewer_signature=_signature(service_reviewer_key, service_statement),
        )
    )
    profile = dict(
        eaaef_operational_profile_contract(
            operation_vocabulary_cid=vocabulary_cid
        )
    )
    profile_evidence: dict[str, object] = {
        **profile,
        "valid": True,
        "schema_fingerprint": canonical_content_cid(
            {"schema": "test-eaaef-profile", "version": 2}
        ),
        "required_index_set_cid": canonical_content_cid(
            {"schema": "test-required-index-set", "version": 2}
        ),
        "required_columns": {
            table: list(columns) for table, columns in _REQUIRED_COLUMNS.items()
        },
    }
    profile_evidence["verification_cid"] = canonical_content_cid(
        profile_evidence
    )
    gateway_binding_source = {
        "board_namespace": launch.EAAEF_BOARD_NAMESPACE,
        "shard_id": "control-shard-0",
        "command_endpoint": "quack:127.0.0.1:19495",
        "state_endpoint": "quack:127.0.0.1:19496",
        "store_id": "eaaef-control-run-v6",
        "store_generation": "eaaef-run-v6",
        "owner_principal_did": owner_did,
        "owner_session_id": "session:eaaef-owner-v6",
        "owner_generation": 6,
        "command_principal_did": worker_did,
        "worker_principal_did": worker_did,
        "authorization_policy_cid": policy.policy_cid,
        "command_fabric_qualification_cid": _sha("e"),
        "borrowed_transaction_adapter_qualification_cid": _sha("d"),
        "materialization_operational_profile_cid": profile_evidence[
            "verification_cid"
        ],
        "fence_epoch": 9,
        "control_plane_schema_version": QUACK_STATE_REPOSITORY_INTERFACE,
        "state_schema_revision": EAAEF_OPERATIONAL_PROFILE_ID,
        "operational_profile_id": EAAEF_OPERATIONAL_PROFILE_ID,
        "schema_revision": EAAEF_OPERATIONAL_PROFILE_ID,
        "operation_vocabulary_cid": vocabulary_cid,
        "borrowed_transaction_handler_interface": (
            EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
        ),
    }
    gateway_binding_cid = launch.eaaef_bootstrap_gateway_binding_cid(
        gateway_binding_source
    )
    handler_evidence = eaaef_bootstrap_handler_source_evidence(
        board_namespace=launch.EAAEF_BOARD_NAMESPACE,
        shard_id="control-shard-0",
    )
    statement: dict[str, object] = {
        "schema": launch.EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_SCHEMA,
        "interface": launch.EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_INTERFACE,
        "board_namespace": launch.EAAEF_BOARD_NAMESPACE,
        "source_head": "3" * 40,
        "source_tree": "4" * 40,
        "configuration_root": _sha("7"),
        "bootstrap_admission_receipt_cid": _sha("8"),
        "configured_board_capsule_cid": _sha("9"),
        "materialization_receipt_cid": _sha("a"),
        "materialization_database_program_binding_cid": _sha("b"),
        "materialization_operational_profile_cid": profile_evidence[
            "verification_cid"
        ],
        "operational_profile_verification": profile_evidence,
        "operation_vocabulary_cid": vocabulary_cid,
        "borrowed_transaction_handler_interface": (
            EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
        ),
        "borrowed_transaction_handler_source_evidence_cid": (
            handler_evidence["handler_source_evidence_cid"]
        ),
        "borrowed_transaction_adapter_qualification_cid": _sha("d"),
        "gateway_interface": EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE,
        "gateway_binding_cid": gateway_binding_cid,
        "control_plane_schema_version": QUACK_STATE_REPOSITORY_INTERFACE,
        "state_schema_revision": EAAEF_OPERATIONAL_PROFILE_ID,
        "operational_profile_id": EAAEF_OPERATIONAL_PROFILE_ID,
        "authorization_policy": policy.to_dict(),
        "authorization_policy_cid": policy.policy_cid,
        "command_fabric_qualification_cid": _sha("e"),
        "command_authorization_service": service,
        "active_plan_root_cid": _sha("f"),
        "active_plan_revision": 2,
        "frontier_cid": _sha("0"),
        "operations": sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS),
        "excluded_operations": sorted(EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS),
        "store_id": "eaaef-control-run-v6",
        "store_generation": "eaaef-run-v6",
        "schema_revision": EAAEF_OPERATIONAL_PROFILE_ID,
        "board_scope": (
            f"board:{launch.EAAEF_BOARD_NAMESPACE}:control-shard-0"
        ),
        "shard_id": "control-shard-0",
        "owner_principal_did": owner_did,
        "owner_session_id": "session:eaaef-owner-v6",
        "owner_generation": 6,
        "lease_id": "lease:eaaef-board-shard-v6",
        "lease_kind": "board_shard_scheduler",
        "lease_mode": "shared_scheduler",
        "command_principal_did": worker_did,
        "fencing_token": 11,
        "fence_epoch": 9,
        "command_endpoint": "quack:127.0.0.1:19495",
        "command_secret_handle": "secret-handle:eaaef-quack-ingress-v6",
        "state_endpoint": "quack:127.0.0.1:19496",
        "state_secret_handle": "secret-handle:eaaef-quack-state-v6",
        "worker_principal_did": worker_did,
        "issuance_nonce": "bootstrap-operational-nonce-1",
        "issued_at_ms": NOW_MS - 500,
        "expires_at_ms": NOW_MS + 50_000,
        "production_admitted": True,
        "owner_provisions_lease_after_capability_verification": True,
        "owner_renews_lease_after_reverification": True,
        "materializer_mints_external_lease": False,
        "direct_database_open": False,
        "portal_fallback": False,
        "local_sidecar_writes": False,
        "arbitrary_sql_enabled": False,
        "reviewer_did": operational_reviewer,
        "reviewer_role": launch.EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_REVIEW_ROLE,
    }
    capability = dict(
        launch.seal_eaaef_bootstrap_operational_capability(
            statement,
            reviewer_signature=_signature(operational_reviewer_key, statement),
        )
    )
    context = {
        "operational_reviewer": operational_reviewer,
        "operational_reviewer_key": operational_reviewer_key,
        "service_reviewer": service_reviewer,
        "service_reviewer_key": service_reviewer_key,
        "approver_key": approver_key,
        "approver_did": approver_did,
        "worker_did": worker_did,
        "owner_did": owner_did,
        "policy": policy,
    }
    return capability, context


def _expected_bindings(capability: dict[str, object]) -> dict[str, object]:
    return {
        field: deepcopy(capability[field])
        for field in launch._EXPECTED_OPERATIONAL_BINDING_FIELDS
    }


def _base_live_seal(
    capability: dict[str, object],
    pin: AgentImplementationControlPlanePin,
) -> configured_capsule.VerifiedExternalAgentConfiguredBoardLiveSeal:
    base_report: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-configured-board-live-seal-verification@1"
        ),
        "valid": True,
        "source_head": pin.source_head,
        "source_tree": pin.source_tree,
        "configuration_root": capability["configuration_root"],
        "bootstrap_admission_receipt_cid": capability[
            "bootstrap_admission_receipt_cid"
        ],
        "configured_board_capsule_cid": capability[
            "configured_board_capsule_cid"
        ],
        "accepted_control_plane_pin_cid": launch._cid(pin.as_dict()),
        "active_plan": {
            "plan_root_cid": capability["active_plan_root_cid"],
            "revision": capability["active_plan_revision"],
        },
        "frontier_cid": capability["frontier_cid"],
        "authority_mutated": False,
        "process_started": False,
    }
    base_report["verification_cid"] = launch._cid(base_report)
    # Unit tests use the verifier-owned token constructor to model the output
    # of the much larger admission/capsule fixture.  Production code cannot
    # obtain this exact type from external JSON.
    return configured_capsule.VerifiedExternalAgentConfiguredBoardLiveSeal(
        configured_capsule._VERIFIED_LIVE_SEAL_TOKEN,
        base_report,
    )


def _verified_live_seal(
    capability: dict[str, object],
    context: dict[str, object],
    pin: AgentImplementationControlPlanePin,
) -> launch.VerifiedEAAEFBootstrapGatewayLiveSeal:
    base = _base_live_seal(capability, pin)
    relative_path = launch.eaaef_bootstrap_operational_capability_relative_path(
        str(capability["source_head"]),
        str(capability["active_plan_root_cid"]),
        registry_prefix="data/eaaef/authority",
    ).as_posix()
    capability_file_sha256 = "sha256:" + launch.hashlib.sha256(
        launch._canonical_bytes(capability)
    ).hexdigest()
    return launch.verify_eaaef_bootstrap_gateway_live_seal(
        base,
        operational_capability=capability,
        operational_capability_file_sha256=capability_file_sha256,
        operational_capability_relative_path=relative_path,
        authority_registry_prefix="data/eaaef/authority",
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
        expected_operational_bindings=_expected_bindings(capability),
        now_ms=NOW_MS,
    )


def _launch_authority(
    capability: dict[str, object],
    context: dict[str, object],
    pin: AgentImplementationControlPlanePin,
) -> tuple[dict[str, object], launch.VerifiedEAAEFBootstrapGatewayLiveSeal]:
    live = _verified_live_seal(capability, context, pin)
    authority = dict(
        launch.build_eaaef_bootstrap_gateway_launch_authority(
            live,
            accepted_control_plane_pin=pin,
            now_ms=NOW_MS,
        )
    )
    return authority, live


def _unsigned(value: dict[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"reviewer_signature", "capability_cid"}
    }


def _replace_signed_service(
    capability: dict[str, object],
    context: dict[str, object],
    service_updates: dict[str, object],
    *,
    service_key: Ed25519PrivateKey | None = None,
) -> dict[str, object]:
    service_statement = _unsigned(
        dict(capability["command_authorization_service"])
    )
    service_statement.update(service_updates)
    key = service_key or context["service_reviewer_key"]
    service = dict(
        launch.seal_eaaef_command_authorization_service_capability(
            service_statement,
            reviewer_signature=_signature(key, service_statement),
        )
    )
    operational_statement = _unsigned(capability)
    operational_statement["command_authorization_service"] = service
    return dict(
        launch.seal_eaaef_bootstrap_operational_capability(
            operational_statement,
            reviewer_signature=_signature(
                context["operational_reviewer_key"], operational_statement
            ),
        )
    )


def _replace_signed_operational(
    capability: dict[str, object],
    context: dict[str, object],
    updates: dict[str, object],
    *,
    operational_key: Ed25519PrivateKey | None = None,
) -> dict[str, object]:
    statement = _unsigned(capability)
    statement.update(updates)
    statement["gateway_binding_cid"] = (
        launch.eaaef_bootstrap_gateway_binding_cid(statement)
    )
    key = operational_key or context["operational_reviewer_key"]
    return dict(
        launch.seal_eaaef_bootstrap_operational_capability(
            statement,
            reviewer_signature=_signature(key, statement),
        )
    )


def _client(
    capability: dict[str, object],
    context: dict[str, object],
    *,
    clock_ms: object,
    monotonic_ms: object | None = None,
) -> launch.EAAEFCommandAuthorizationServiceClient:
    if monotonic_ms is None:
        def default_monotonic_ms() -> int:
            return launch.time.monotonic_ns() // 1_000_000

        monotonic_ms = default_monotonic_ms
    return launch.EAAEFCommandAuthorizationServiceClient.from_signed_operational_capability(
        operational_capability=capability,
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
        expected=_expected_bindings(capability),
        clock_ms=clock_ms,
        monotonic_ms=monotonic_ms,
    )


def _safe_socket_stat(*, inode: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        st_mode=stat.S_IFSOCK | 0o600,
        st_uid=os.geteuid(),
        st_nlink=1,
        st_dev=7,
        st_ino=inode,
    )


def _safe_parent_stat() -> SimpleNamespace:
    return SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o755,
        st_uid=0,
        st_nlink=2,
        st_dev=7,
        st_ino=50,
    )


def _authorized_response(
    capability: dict[str, object],
    context: dict[str, object],
    request: dict[str, object],
    *,
    expires_at_ms: int = NOW_MS + 10_000,
    parameter_updates: dict[str, object] | None = None,
) -> tuple[dict[str, object], object]:
    policy = context["policy"]
    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.get",
            arguments={"task_cid": "task:eaaef:1"},
        )
    )
    deadline_ms = NOW_MS + 5_000
    parameters = dict(quack_daemon_state_command_parameters(
        intent,
        request_id=request["request_id"],
        principal_did=context["worker_did"],
        authority_ref_cid=policy.authority_ref_cid,
        lease_id=capability["lease_id"],
        scope_id="task:eaaef:1",
        deadline_ms=deadline_ms,
        fencing_token=capability["fencing_token"],
        idempotency_key="idempotency:eaaef:1",
    ))
    parameters["authorization_request_cid"] = request["request_cid"]
    parameters.update(parameter_updates or {})
    command = StateCommand(
        command_id=f"{request['request_id']}:task-get",
        command_kind=CommandKind.OBSERVE,
        store_id=policy.store_id,
        session_id=capability["lease_id"],
        expected_generation=policy.owner_generation,
        expected_revision=1,
        fence_epoch=policy.fence_epoch,
        idempotency_key="idempotency:eaaef:1",
        parameters=parameters,
    )
    prepared = authorized_state_command_signing_payload(
        request_id=request["request_id"],
        submission_id="submission:eaaef:1",
        ingress_slot=1,
        principal_did=context["worker_did"],
        approver_did=context["approver_did"],
        authority_ref_cid=policy.authority_ref_cid,
        board_namespace=policy.board_namespace,
        shard_id=policy.shard_id,
        owner_principal_did=context["owner_did"],
        lease_id=capability["lease_id"],
        scope_id="task:eaaef:1",
        effect="control-plane/observe",
        issued_at_ms=NOW_MS - 100,
        expires_at_ms=expires_at_ms,
        deadline_ms=deadline_ms,
        one_use_nonce=request["request_nonce"],
        command=command,
    )
    signature = _signature(context["approver_key"], dict(prepared))
    return intent, seal_authorized_state_command(
        prepared,
        approver_signature=signature,
    )


def test_signed_operational_capability_binds_exact_31_operation_profile() -> None:
    capability, context = _signed_capability()

    verified = launch.verify_eaaef_bootstrap_operational_capability(
        capability,
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[
            context["service_reviewer"]
        ],
        expected=_expected_bindings(capability),
        now_ms=NOW_MS,
    )

    assert verified["production_admitted"] is True
    assert len(verified["operations"]) == 31
    assert len(verified["excluded_operations"]) == 8
    assert verified["materializer_mints_external_lease"] is False
    assert verified["lease_kind"] == "board_shard_scheduler"
    assert verified["lease_mode"] == "shared_scheduler"


def test_stable_gateway_binding_and_strong_submission_are_exact() -> None:
    capability, context = _signed_capability()
    verified = launch.verify_eaaef_bootstrap_operational_capability(
        capability,
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
        expected=_expected_bindings(capability),
        now_ms=NOW_MS,
    )
    assert type(verified) is launch.VerifiedEAAEFBootstrapOperationalCapability
    binding = dict(launch.eaaef_bootstrap_gateway_binding(verified))
    assert set(binding) == {
        "schema",
        "interface",
        "board_namespace",
        "shard_id",
        "command_endpoint",
        "state_endpoint",
        "store_id",
        "store_generation",
        "owner_principal_did",
        "owner_session_id",
        "owner_generation",
        "command_principal_did",
        "authorization_policy_cid",
        "command_fabric_qualification_cid",
        "borrowed_transaction_adapter_qualification_cid",
        "operational_profile_verification_cid",
        "fence_epoch",
        "control_plane_schema_version",
        "state_schema_revision",
        "operational_profile_id",
        "operation_vocabulary_cid",
        "handler_interface",
        "handler_schema",
    }
    assert launch.eaaef_bootstrap_gateway_binding_cid(verified) == capability[
        "gateway_binding_cid"
    ]

    request = {
        "request_id": "authorization-request:strong-owner-verifier",
        "request_nonce": "authorization-nonce:strong-owner-verifier",
        "request_cid": _sha("c"),
    }
    intent, envelope = _authorized_response(capability, context, request)
    admitted = launch.verify_eaaef_bootstrap_operation_submission(
        envelope,
        intent,
        verified_capability=verified,
        authorization_policy=context["policy"],
        now_ms=NOW_MS,
    )
    assert admitted["operation"] == "task.get"
    assert admitted["authorization_request_cid"] == request["request_cid"]
    assert envelope.command.session_id == capability["lease_id"]
    assert envelope.command.command_id == f"{request['request_id']}:task-get"

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="typed verified capability",
    ):
        launch.verify_eaaef_bootstrap_operation_submission(
            envelope,
            intent,
            verified_capability=dict(verified),  # type: ignore[arg-type]
            authorization_policy=context["policy"],
            now_ms=NOW_MS,
        )


def test_strong_submission_rejects_envelope_subclass_serialization_override() -> None:
    capability, context = _signed_capability()
    verified = launch.verify_eaaef_bootstrap_operational_capability(
        capability,
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
        expected=_expected_bindings(capability),
        now_ms=NOW_MS,
    )
    request = {
        "request_id": "authorization-request:subclass-smuggling",
        "request_nonce": "authorization-nonce:subclass-smuggling",
        "request_cid": _sha("e"),
    }
    intent, envelope = _authorized_response(capability, context, request)

    class SerializationOverrideEnvelope(type(envelope)):
        def unsigned_payload(self) -> dict[str, object]:
            return envelope.unsigned_payload()

    subclassed = SerializationOverrideEnvelope.from_dict(envelope.to_dict())
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="exact AuthorizedStateCommand@1",
    ):
        launch.verify_eaaef_bootstrap_operation_submission(
            subclassed,
            intent,
            verified_capability=verified,
            authorization_policy=context["policy"],
            now_ms=NOW_MS,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("state_endpoint", "quack:127.0.0.1:19596"),
        ("store_generation", "eaaef-run-v7"),
        ("command_fabric_qualification_cid", _sha("1")),
        ("borrowed_transaction_adapter_qualification_cid", _sha("2")),
        ("materialization_operational_profile_cid", _sha("3")),
    ],
)
def test_resigned_stable_binding_field_requires_a_new_gateway_identity(
    field: str,
    replacement: object,
) -> None:
    capability, context = _signed_capability()
    statement = _unsigned(capability)
    statement[field] = replacement
    rebound = dict(
        launch.seal_eaaef_bootstrap_operational_capability(
            statement,
            reviewer_signature=_signature(
                context["operational_reviewer_key"], statement
            ),
        )
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="gateway binding is invalid",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            rebound,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(rebound),
            now_ms=NOW_MS,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("source_tree", "0" * 40, "self-address|signature"),
        ("operation_vocabulary_cid", _sha("1"), "protocol identities|vocabulary"),
        ("lease_mode", "exclusive", "lease"),
        ("materializer_mints_external_lease", True, "authority"),
        ("portal_fallback", True, "authority"),
    ],
)
def test_operational_capability_tampering_fails_closed(
    field: str,
    replacement: object,
    match: str,
) -> None:
    capability, context = _signed_capability()
    expected = _expected_bindings(capability)
    capability[field] = replacement

    with pytest.raises(launch.EAAEFBootstrapGatewayLaunchError, match=match):
        launch.verify_eaaef_bootstrap_operational_capability(
            capability,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[
                context["service_reviewer"]
            ],
            expected=expected,
            now_ms=NOW_MS,
        )


def test_launch_authority_is_path_free_canonical_and_pin_bound() -> None:
    capability, context = _signed_capability()
    pin = _pin()
    live = _verified_live_seal(capability, context, pin)
    authority = dict(
        launch.build_eaaef_bootstrap_gateway_launch_authority(
            live,
            accepted_control_plane_pin=pin,
            now_ms=NOW_MS,
        )
    )
    encoded = launch.canonical_eaaef_bootstrap_gateway_launch_authority_json(
        authority,
        accepted_control_plane_pin=pin,
        verified_live_seal=live,
        now_ms=NOW_MS,
    )

    assert "secret-handle:eaaef-quack-ingress-v6" in encoded
    assert "callback" not in encoded.lower()
    assert "database_path" not in encoded
    decoded = json.loads(encoded)
    assert "raw_token" not in decoded
    assert "private_key" not in decoded
    assert "authorization_callback" not in decoded
    assert launch.parse_eaaef_bootstrap_gateway_launch_authority(
        encoded,
        accepted_control_plane_pin=pin,
        verified_live_seal=live,
        now_ms=NOW_MS,
    )["authority_cid"] == authority["authority_cid"]

    tampered = deepcopy(authority)
    tampered["source_tree"] = "0" * 40
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="binding|control plane",
    ):
        launch.parse_eaaef_bootstrap_gateway_launch_authority(
            tampered,
            accepted_control_plane_pin=pin,
            verified_live_seal=live,
            now_ms=NOW_MS,
        )


def test_source_addressed_path_has_no_dynamic_caller_component() -> None:
    path = launch.eaaef_bootstrap_operational_capability_relative_path(
        "3" * 40,
        _sha("f"),
        registry_prefix="data/eaaef/authority",
    )
    assert path.as_posix() == (
        "data/eaaef/authority/"
        "eaaef-bootstrap-daemon-operational-capability--"
        + "3" * 40
        + "--"
        + "f" * 64
        + ".json"
    )
    with pytest.raises(launch.EAAEFBootstrapGatewayLaunchError, match="relative"):
        launch.eaaef_bootstrap_operational_capability_relative_path(
            "3" * 40,
            _sha("f"),
            registry_prefix="/tmp/attacker",
        )


def test_authorization_client_contains_no_signer_or_raw_secret_surface() -> None:
    capability, context = _signed_capability()
    client = launch.EAAEFCommandAuthorizationServiceClient.from_signed_operational_capability(
        operational_capability=capability,
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
        expected=_expected_bindings(capability),
        clock_ms=lambda: NOW_MS,
    )

    assert not hasattr(client, "private_key")
    assert not hasattr(client, "token")
    assert not hasattr(client, "callback")
    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="unavailable",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


def test_client_rejects_same_shaped_mapping_without_signed_factory() -> None:
    capability, context = _signed_capability()

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="signed factory",
    ):
        launch.EAAEFCommandAuthorizationServiceClient(
            object(),
            operational_capability=capability,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[
                context["service_reviewer"]
            ],
            expected=_expected_bindings(capability),
            forbidden_reviewer_dids=(),
            clock_ms=lambda: NOW_MS,
            monotonic_ms=lambda: 1,
        )


def test_client_reverifies_service_freshness_before_every_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    clock = [NOW_MS]
    client = _client(capability, context, clock_ms=lambda: clock[0])
    clock[0] = int(
        capability["command_authorization_service"]["expires_at_ms"]
    )
    monkeypatch.setattr(
        launch.socket,
        "socket",
        lambda *_args, **_kwargs: pytest.fail("expired client opened a socket"),
    )

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="re-verification",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


def test_operational_and_service_reviewers_must_be_independent() -> None:
    capability, context = _signed_capability()
    service_reviewer = context["operational_reviewer"]
    crossed = _replace_signed_service(
        capability,
        context,
        {"reviewer_did": service_reviewer},
        service_key=context["operational_reviewer_key"],
    )

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="service capability binding|role",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            crossed,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[service_reviewer],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )


def test_capability_lifetimes_are_bounded_and_nested() -> None:
    capability, context = _signed_capability()
    service = capability["command_authorization_service"]
    oversized_service = _replace_signed_service(
        capability,
        context,
        {
            "expires_at_ms": int(service["issued_at_ms"])
            + launch._MAX_SERVICE_CAPABILITY_LIFETIME_MS
            + 1
        },
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="service capability binding",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            oversized_service,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[
                context["service_reviewer"]
            ],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )

    longer_than_service = _replace_signed_operational(
        capability,
        context,
        {"expires_at_ms": int(service["expires_at_ms"]) + 1},
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="lifetime",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            longer_than_service,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[
                context["service_reviewer"]
            ],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )

    service_starts_after_operational = _replace_signed_service(
        capability,
        context,
        {"issued_at_ms": int(capability["issued_at_ms"]) + 1},
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="lifetime",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            service_starts_after_operational,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )


def test_authorization_policy_rejects_command_kind_superset() -> None:
    capability, context = _signed_capability()
    old_policy = context["policy"]
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=old_policy.board_namespace,
        shard_id=old_policy.shard_id,
        store_id=old_policy.store_id,
        authority_ref_cid=old_policy.authority_ref_cid,
        owner_principal_did=old_policy.owner_principal_did,
        owner_generation=old_policy.owner_generation,
        fence_epoch=old_policy.fence_epoch,
        trusted_approver_dids=old_policy.trusted_approver_dids,
        authorized_principal_dids=old_policy.authorized_principal_dids,
        allowed_command_kinds=frozenset(CommandKind),
    )
    rebound_service = _replace_signed_service(
        capability,
        context,
        {"authorization_policy_cid": policy.policy_cid},
    )
    rebound = _replace_signed_operational(
        rebound_service,
        context,
        {
            "authorization_policy": policy.to_dict(),
            "authorization_policy_cid": policy.policy_cid,
        },
    )

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="authorization policy differs",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            rebound,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[
                context["service_reviewer"]
            ],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"request_timeout_ms": 30_001}, "request_timeout_ms"),
        ({"endpoint": "unix:/run/" + "x" * 100}, "bounded /run"),
    ],
)
def test_service_transport_bounds_are_signed_and_fail_closed(
    updates: dict[str, object],
    match: str,
) -> None:
    capability, context = _signed_capability()
    rebound = _replace_signed_service(capability, context, updates)

    with pytest.raises(launch.EAAEFBootstrapGatewayLaunchError, match=match):
        launch.verify_eaaef_bootstrap_operational_capability(
            rebound,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[
                context["service_reviewer"]
            ],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )


class _FakeSocket:
    def __init__(
        self,
        *,
        peer_pid: int,
        peer_uid: int,
        recv_error: BaseException | None = None,
        response: bytes = b"",
        response_factory: object = None,
    ) -> None:
        self.peer_pid = peer_pid
        self.peer_uid = peer_uid
        self.recv_error = recv_error
        self.response = response
        self.response_factory = response_factory
        self.timeout: float | None = None

    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def connect(self, _path: str) -> None:
        return None

    def getsockopt(self, *_args: object) -> bytes:
        return struct.pack("3i", self.peer_pid, self.peer_uid, os.getegid())

    def sendall(self, payload: bytes) -> None:
        if callable(self.response_factory):
            length = struct.unpack("!I", payload[:4])[0]
            request = json.loads(payload[4 : 4 + length].decode("ascii"))
            response = self.response_factory(request)
            self.response = struct.pack("!I", len(response)) + response

    def recv(self, count: int) -> bytes:
        if self.recv_error is not None:
            raise self.recv_error
        chunk = self.response[:count]
        self.response = self.response[count:]
        return chunk


def _install_safe_lstat(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_lstat(path: object) -> SimpleNamespace:
        if str(path).endswith(".sock"):
            return _safe_socket_stat()
        return _safe_parent_stat()

    monkeypatch.setattr(launch.os, "lstat", fake_lstat)


def test_socket_swap_is_detected_before_any_authorization_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    observations = [0]

    def fake_lstat(path: object) -> SimpleNamespace:
        if not str(path).endswith(".sock"):
            return _safe_parent_stat()
        observations[0] += 1
        return _safe_socket_stat(inode=100 + observations[0])

    fake_socket = _FakeSocket(peer_pid=os.getpid(), peer_uid=os.geteuid())
    monkeypatch.setattr(launch.os, "lstat", fake_lstat)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="changed during connect",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


def test_peer_credentials_must_match_signed_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid() + 1,
    )
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="peer process identity differs",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


def test_signed_socket_deadline_stops_slowloris_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        recv_error=TimeoutError("slow peer"),
    )
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)
    monkeypatch.setattr(
        launch,
        "_linux_process_start_time_ticks",
        lambda _pid: capability["command_authorization_service"][
            "expected_server_process_start_time_ticks"
        ],
    )

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="transport failed",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})
    assert 0 < float(fake_socket.timeout or 0) <= 0.5


def test_valid_response_is_accepted_only_with_exact_peer_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    returned: list[object] = []

    def response_factory(request: dict[str, object]) -> bytes:
        _intent, envelope = _authorized_response(capability, context, request)
        returned.append(envelope)
        return launch._canonical_bytes(envelope.to_dict())

    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.get",
            arguments={"task_cid": "task:eaaef:1"},
        )
    )
    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        response_factory=response_factory,
    )
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    result = client.authorize(intent)

    assert result.envelope_cid == returned[0].envelope_cid
    assert 0 < float(fake_socket.timeout or 0) <= 0.5


def test_authorization_request_uses_a_deep_canonical_intent_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.get",
            arguments={"task_cid": "task:eaaef:1"},
        )
    )

    def response_factory(request: dict[str, object]) -> bytes:
        intent["arguments"]["task_cid"] = "task:caller-mutated-after-send"
        _returned_intent, envelope = _authorized_response(
            capability,
            context,
            request,
        )
        return launch._canonical_bytes(envelope.to_dict())

    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        response_factory=response_factory,
    )
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    assert client.authorize(intent).request_id.startswith("authorization-request:")


@pytest.mark.parametrize("fragment", [b"\x00\x00", struct.pack("!I", 8) + b"{}"])
def test_partial_authorization_header_or_body_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    fragment: bytes,
) -> None:
    capability, context = _signed_capability()
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        response=fragment,
    )
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="partial response",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


def test_peer_process_birth_must_match_signed_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    fake_socket = _FakeSocket(peer_pid=os.getpid(), peer_uid=os.geteuid())
    signed_ticks = int(
        capability["command_authorization_service"][
            "expected_server_process_start_time_ticks"
        ]
    )
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)
    monkeypatch.setattr(
        launch,
        "_linux_process_start_time_ticks",
        lambda _pid: signed_ticks + 1,
    )

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="peer process identity differs",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


def test_plain_mapping_cannot_bypass_live_seal_at_build_or_child_parse() -> None:
    capability, context = _signed_capability()
    pin = _pin()
    authority, live = _launch_authority(capability, context, pin)
    exact_reproducer = {
        "bootstrap_operational_capability": {
            "capability_cid": _sha("d")
        }
    }

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="exact verified live-seal token",
    ):
        launch.build_eaaef_bootstrap_gateway_launch_authority(
            exact_reproducer,  # type: ignore[arg-type]
            accepted_control_plane_pin=pin,
            now_ms=NOW_MS,
        )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="freshly verified gateway live-seal token",
    ):
        launch.parse_eaaef_bootstrap_gateway_launch_authority(
            authority,
            accepted_control_plane_pin=pin,
            verified_live_seal=dict(live),  # type: ignore[arg-type]
            now_ms=NOW_MS,
        )


def test_outer_authority_cannot_be_rebound_to_another_signed_capability() -> None:
    capability_a, context_a = _signed_capability()
    capability_b = _replace_signed_operational(
        capability_a,
        context_a,
        {"issuance_nonce": "bootstrap-operational-nonce-2"},
    )
    pin = _pin()
    authority, live = _launch_authority(capability_a, context_a, pin)
    crossed = deepcopy(authority)
    crossed["operational_capability"] = capability_b
    crossed["operational_capability_cid"] = capability_b["capability_cid"]
    crossed_body = {
        key: value for key, value in crossed.items() if key != "authority_cid"
    }
    crossed["authority_cid"] = launch._cid(crossed_body)

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="signed live identities",
    ):
        launch.parse_eaaef_bootstrap_gateway_launch_authority(
            crossed,
            accepted_control_plane_pin=pin,
            verified_live_seal=live,
            now_ms=NOW_MS,
        )


def test_launch_json_is_canonical_bounded_duplicate_free_and_pin_typed() -> None:
    capability, context = _signed_capability()
    pin = _pin()
    authority, live = _launch_authority(capability, context, pin)
    encoded = launch._canonical_bytes(authority).decode("ascii")
    duplicate = encoded[:-1] + ',"schema":"duplicate"}'

    for malformed, match in (
        (json.dumps(authority, indent=2), "canonical"),
        (duplicate, "duplicate JSON key"),
    ):
        with pytest.raises(launch.EAAEFBootstrapGatewayLaunchError, match=match):
            launch.parse_eaaef_bootstrap_gateway_launch_authority(
                malformed,
                accepted_control_plane_pin=pin,
                verified_live_seal=live,
                now_ms=NOW_MS,
            )

    oversized = {**authority, "oversized": "x" * launch._MAX_CAPABILITY_BYTES}
    with pytest.raises(launch.EAAEFBootstrapGatewayLaunchError, match="oversized"):
        launch.parse_eaaef_bootstrap_gateway_launch_authority(
            oversized,
            accepted_control_plane_pin=pin,
            verified_live_seal=live,
            now_ms=NOW_MS,
        )

    class FakePin:
        def as_dict(self) -> dict[str, object]:
            return pin.as_dict()

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="exact closed type",
    ):
        launch.parse_eaaef_bootstrap_gateway_launch_authority(
            authority,
            accepted_control_plane_pin=FakePin(),
            verified_live_seal=live,
            now_ms=NOW_MS,
        )


def test_source_addressed_loader_is_canonical_pinned_and_nofollow(
    tmp_path: Path,
) -> None:
    capability, _context = _signed_capability()
    relative = launch.eaaef_bootstrap_operational_capability_relative_path(
        str(capability["source_head"]),
        str(capability["active_plan_root_cid"]),
        registry_prefix="data/eaaef/authority",
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    for parent in (tmp_path / "data", tmp_path / "data/eaaef", path.parent):
        parent.chmod(0o700)
    raw = launch._canonical_bytes(capability)
    path.write_bytes(raw)
    path.chmod(0o600)
    file_sha = "sha256:" + launch.hashlib.sha256(raw).hexdigest()

    loaded, observed_sha, observed_path = (
        launch.load_eaaef_bootstrap_operational_capability(
            tmp_path,
            source_head=str(capability["source_head"]),
            plan_root_cid=str(capability["active_plan_root_cid"]),
            registry_prefix="data/eaaef/authority",
            expected_file_sha256=file_sha,
        )
    )
    assert dict(loaded) == capability
    assert observed_sha == file_sha
    assert observed_path == relative.as_posix()

    tmp_path.chmod(0o777)
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="root ownership is unsafe",
    ):
        launch.load_eaaef_bootstrap_operational_capability(
            tmp_path,
            source_head=str(capability["source_head"]),
            plan_root_cid=str(capability["active_plan_root_cid"]),
            registry_prefix="data/eaaef/authority",
        )
    tmp_path.chmod(0o700)

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="file identity changed",
    ):
        launch.load_eaaef_bootstrap_operational_capability(
            tmp_path,
            source_head=str(capability["source_head"]),
            plan_root_cid=str(capability["active_plan_root_cid"]),
            registry_prefix="data/eaaef/authority",
            expected_file_sha256=_sha("0"),
        )

    path.parent.chmod(0o777)
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="parent ownership is unsafe",
    ):
        launch.load_eaaef_bootstrap_operational_capability(
            tmp_path,
            source_head=str(capability["source_head"]),
            plan_root_cid=str(capability["active_plan_root_cid"]),
            registry_prefix="data/eaaef/authority",
        )

    path.parent.chmod(0o700)
    attacker = tmp_path / "attacker-capability.json"
    attacker.write_bytes(raw)
    attacker.chmod(0o600)
    path.unlink()
    path.symlink_to(attacker)
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="unavailable",
    ):
        launch.load_eaaef_bootstrap_operational_capability(
            tmp_path,
            source_head=str(capability["source_head"]),
            plan_root_cid=str(capability["active_plan_root_cid"]),
            registry_prefix="data/eaaef/authority",
        )


def test_gateway_live_seal_joins_file_hash_to_canonical_capability_bytes() -> None:
    capability, context = _signed_capability()
    pin = _pin()
    relative = launch.eaaef_bootstrap_operational_capability_relative_path(
        str(capability["source_head"]),
        str(capability["active_plan_root_cid"]),
        registry_prefix="data/eaaef/authority",
    ).as_posix()

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="detached from its canonical bytes",
    ):
        launch.verify_eaaef_bootstrap_gateway_live_seal(
            _base_live_seal(capability, pin),
            operational_capability=capability,
            operational_capability_file_sha256=_sha("2"),
            operational_capability_relative_path=relative,
            authority_registry_prefix="data/eaaef/authority",
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected_operational_bindings=_expected_bindings(capability),
            now_ms=NOW_MS,
        )

@pytest.mark.parametrize("set_name", ["authorized_principal_dids", "trusted_approver_dids"])
def test_authorization_policy_rejects_resigned_identity_supersets(
    set_name: str,
) -> None:
    capability, context = _signed_capability()
    old_policy = context["policy"]
    _extra_key, extra_did = _key()
    values = {
        "trusted_approver_dids": old_policy.trusted_approver_dids,
        "authorized_principal_dids": old_policy.authorized_principal_dids,
    }
    values[set_name] = frozenset({*values[set_name], extra_did})
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=old_policy.board_namespace,
        shard_id=old_policy.shard_id,
        store_id=old_policy.store_id,
        authority_ref_cid=old_policy.authority_ref_cid,
        owner_principal_did=old_policy.owner_principal_did,
        owner_generation=old_policy.owner_generation,
        fence_epoch=old_policy.fence_epoch,
        trusted_approver_dids=values["trusted_approver_dids"],
        authorized_principal_dids=values["authorized_principal_dids"],
        allowed_command_kinds=old_policy.allowed_command_kinds,
    )
    rebound_service = _replace_signed_service(
        capability,
        context,
        {"authorization_policy_cid": policy.policy_cid},
    )
    rebound = _replace_signed_operational(
        rebound_service,
        context,
        {
            "authorization_policy": policy.to_dict(),
            "authorization_policy_cid": policy.policy_cid,
        },
    )

    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="authorization policy differs|role or lifetime",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            rebound,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(rebound),
            now_ms=NOW_MS,
        )


def test_operational_profile_evidence_is_closed_and_cross_bound() -> None:
    capability, context = _signed_capability()
    profile = deepcopy(capability["operational_profile_verification"])
    profile.pop("required_index_set_cid")
    missing_index = _replace_signed_operational(
        capability,
        context,
        {"operational_profile_verification": profile},
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="operational profile verification",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            missing_index,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(missing_index),
            now_ms=NOW_MS,
        )

    crossed = _replace_signed_operational(
        capability,
        context,
        {"materialization_operational_profile_cid": _sha("1")},
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="operational profile verification",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            crossed,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(crossed),
            now_ms=NOW_MS,
        )


def test_service_expiry_during_io_is_rechecked_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    clock = [NOW_MS]

    def response_factory(request: dict[str, object]) -> bytes:
        _intent, envelope = _authorized_response(capability, context, request)
        clock[0] = int(
            capability["command_authorization_service"]["expires_at_ms"]
        )
        return launch._canonical_bytes(envelope.to_dict())

    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        response_factory=response_factory,
    )
    client = _client(capability, context, clock_ms=lambda: clock[0])
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.get",
            arguments={"task_cid": "task:eaaef:1"},
        )
    )
    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="re-verification",
    ):
        client.authorize(intent)


def test_authorization_transport_uses_one_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    monotonic = [0]

    def advancing_clock() -> int:
        monotonic[0] += 100
        return monotonic[0]

    class DripSocket(_FakeSocket):
        def recv(self, count: int) -> bytes:
            return super().recv(min(count, 1))

    fake_socket = DripSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        response=b"\x00\x00\x00\x01x",
    )
    client = _client(
        capability,
        context,
        clock_ms=lambda: NOW_MS,
        monotonic_ms=advancing_clock,
    )
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="absolute request deadline expired",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})


@pytest.mark.parametrize("mode", ["request_correlation", "expiry_window", "malformed_intent"])
def test_signed_response_must_be_freshly_correlated_and_capability_bounded(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    capability, context = _signed_capability()

    def response_factory(request: dict[str, object]) -> bytes:
        signed_request = dict(request)
        expires = NOW_MS + 10_000
        updates: dict[str, object] | None = None
        if mode == "request_correlation":
            signed_request["request_id"] = "authorization-request:previous"
            signed_request["request_nonce"] = "authorization-nonce:previous"
            signed_request["request_cid"] = _sha("c")
        elif mode == "expiry_window":
            expires = int(
                capability["command_authorization_service"]["expires_at_ms"]
            ) + 1
        else:
            updates = {"daemon_operation": "not-in-the-closed-vocabulary"}
        _intent, envelope = _authorized_response(
            capability,
            context,
            signed_request,
            expires_at_ms=expires,
            parameter_updates=updates,
        )
        return launch._canonical_bytes(envelope.to_dict())

    fake_socket = _FakeSocket(
        peer_pid=os.getpid(),
        peer_uid=os.geteuid(),
        response_factory=response_factory,
    )
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)
    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.get",
            arguments={"task_cid": "task:eaaef:1"},
        )
    )

    match = (
        "malformed operation intent"
        if mode == "malformed_intent"
        else "request/capability bound"
    )
    with pytest.raises(launch.EAAEFCommandAuthorizationServiceError, match=match):
        client.authorize(intent)


@pytest.mark.parametrize(
    "forbidden_role",
    [
        "operational_reviewer",
        "service_reviewer",
        "service_principal_did",
        "approver_principal_did",
        "authorized_client_principal_did",
        "owner_principal_did",
    ],
)
def test_reviewer_and_service_roles_cannot_collide_or_enter_forbidden_roles(
    forbidden_role: str,
) -> None:
    capability, context = _signed_capability()
    service = capability["command_authorization_service"]
    collided = _replace_signed_service(
        capability,
        context,
        {"service_principal_did": service["approver_principal_did"]},
    )
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="service capability binding",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            collided,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(capability),
            now_ms=NOW_MS,
        )

    role_dids = {
        "operational_reviewer": context["operational_reviewer"],
        "service_reviewer": context["service_reviewer"],
        "service_principal_did": service["service_principal_did"],
        "approver_principal_did": service["approver_principal_did"],
        "authorized_client_principal_did": service[
            "authorized_client_principal_did"
        ],
        "owner_principal_did": capability["owner_principal_did"],
    }
    with pytest.raises(
        launch.EAAEFBootstrapGatewayLaunchError,
        match="authority is invalid|service capability binding",
    ):
        launch.verify_eaaef_bootstrap_operational_capability(
            capability,
            trusted_reviewer_dids=[context["operational_reviewer"]],
            trusted_authorization_service_reviewer_dids=[context["service_reviewer"]],
            expected=_expected_bindings(capability),
            forbidden_reviewer_dids=[role_dids[forbidden_role]],
            now_ms=NOW_MS,
        )


def test_peer_pid_must_match_signed_service_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, context = _signed_capability()
    client = _client(capability, context, clock_ms=lambda: NOW_MS)
    fake_socket = _FakeSocket(
        peer_pid=os.getpid() + 1,
        peer_uid=os.geteuid(),
    )
    _install_safe_lstat(monkeypatch)
    monkeypatch.setattr(launch.socket, "socket", lambda *_args: fake_socket)

    with pytest.raises(
        launch.EAAEFCommandAuthorizationServiceError,
        match="peer process identity differs",
    ):
        client.authorize({"schema": "QuackDaemonOperationIntent@1"})
