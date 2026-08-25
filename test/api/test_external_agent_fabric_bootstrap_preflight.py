from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_fabric_bootstrap as preflight,
)

from ipfs_accelerate_py import agent_implementation_route as route_module

NOW_MS = 1_800_000_000_000


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value):
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _board():
    value = {
        "schema": preflight.EAAEF_BOARD_SCHEMA,
        "board_namespace": preflight.EAAEF_BOARD_NAMESPACE,
        "plan_revision": "EAAEF-PLAN-R1",
        "parent_objective": "ExternalAgentAutonomousExecutionFabric",
        "source_forest_root": "sha256:" + "1" * 64,
        "goals": [{"goal_id": "EAAEF-G000"}],
        "tasks": [
            {
                "stable_task_id": preflight.EAAEF_BOOTSTRAP_TASK_ID,
                "task_spec_cid": "sha256:" + "a" * 64,
                "idempotency_key": "sha256:" + "b" * 64,
                "provider_policy": "no provider invocation",
                "model_route": "host-controlled deterministic admission",
                "source_control_plane_schema_version": (
                    "datasets-authoritative-operational-v1"
                ),
                "resource_request": {
                    "cpu_millicores": 1000,
                    "ram_mib": 2048,
                    "network": "deny",
                },
            }
        ],
    }
    value["board_cid"] = _cid(value)
    return value


def _materialization(board):
    source_generation = {
        "ipfs_accelerate_py": {
            "head": "2" * 40,
            "tree": "3" * 40,
        }
    }
    source_generation["source_generation_cid"] = _cid(source_generation)
    value = {
        "schema": preflight.EAAEF_MATERIALIZATION_SCHEMA,
        "authority_mode": "embedded",
        "maximum_writer_processes": 1,
        "continuous_quack_authority": False,
        "ducklake_authority": False,
        "process_started": False,
        "board_validation": {
            "valid": True,
            # Structural board validation is intentionally not launch
            # authority.  The bootstrap preflight joins the independent
            # provider/image/container evidence below.
            "live_launch_allowed": False,
            "board_cid": board["board_cid"],
            "source_forest_root": board["source_forest_root"],
        },
        "source_generation": source_generation,
        "source_head": "2" * 40,
        "source_tree": "3" * 40,
        "population_cid": "sha256:" + "4" * 64,
        "plan_root_cid": "sha256:" + "5" * 64,
        "control_projection": {
            "projection_root": "sha256:" + "c" * 64,
        },
        "coordination_projection": {
            "projection_root": "sha256:" + "d" * 64,
        },
        "execution_projection": {
            "projection_root": "sha256:" + "e" * 64,
        },
    }
    value["receipt_cid"] = _cid(value)
    return value


def _image_qualification(*, expires_at_ms=NOW_MS + 1000):
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    value = {
        "schema": preflight.EAAEF_CONTAINER_IMAGE_QUALIFICATION_SCHEMA,
        "image_digest": "sha256:" + "6" * 64,
        "image_label": "eaaef-bootstrap-20260818",
        "image_os": "linux",
        "image_architecture": "amd64",
        "sbom_digest": "sha256:" + "7" * 64,
        "sbom_format": "spdx-json",
        "sbom_bytes": 4096,
        "toolchain_versions": {"python": "3.12.3", "git": "2.43.0"},
        "workload_class": "bootstrap_diagnostic_only",
        "task_dispatch_verified": False,
        "execution_mode": "rootless_engine",
        "rootless_supported": True,
        "rootless_verified": True,
        "nonroot_hardening_verified": True,
        "daemon_identity_cid": "sha256:" + "0" * 64,
        "daemon_policy_cid": "sha256:" + "a" * 64,
        "reviewer_identity_did": reviewer,
        "reviewer_role": "independent_security_reviewer",
        "verified_at_ms": NOW_MS - 1000,
        "expires_at_ms": expires_at_ms,
    }
    signature = key.sign(_canonical(value))
    value["reviewer_signature"] = base64.b64encode(signature).decode("ascii")
    value["qualification_cid"] = _cid(value)
    return value, reviewer


def _container_profile(image_digest):
    value = {
        "schema": preflight.EAAEF_CONTAINER_PROFILE_SCHEMA,
        "runtime": "oci",
        "workload_class": "bootstrap_diagnostic_only",
        "task_dispatch_admitted": False,
        "execution_mode": "rootless_engine",
        "rootless_supported": True,
        "daemon_identity_cid": "sha256:" + "0" * 64,
        "daemon_policy_cid": "sha256:" + "a" * 64,
        "bootstrap_policy_cid": preflight.EAAEF_BOOTSTRAP_POLICY_CID,
        "rootful_fallback_admitted": False,
        "image_digest": image_digest,
        "rootless": True,
        "nonroot_user": "1000:1000",
        "read_only_base": True,
        "network_mode": "none",
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
        "pids_limit": 256,
        "cpu_limit": 2,
        "memory_limit_bytes": 4 * 1024**3,
        "disk_limit_bytes": 16 * 1024**3,
        "maximum_parallel_workers": 0,
        "maximum_parallel_containers": 1,
        "gpu": {"mode": "none", "device_ids": [], "memory_limit_bytes": 0},
        "privileged": False,
        "host_pid": False,
        "host_ipc": False,
        "devices": [],
        "docker_socket_mounted": False,
        "inherit_host_environment": False,
        "environment": dict(preflight._EXPECTED_CONTAINER_ENV),
        "mounts": [
            {
                "source_identity": "sha256:" + "8" * 64,
                "target": "/workspace",
                "read_only": False,
                "kind": "worktree",
            },
        ],
    }
    value["profile_cid"] = _cid(value)
    return value


def _sealed_eaaef_route(
    board,
    receipt,
    profile,
    *,
    invocation_expires_at_ms=NOW_MS + 1000,
):
    bounds = route_module.AgentImplementationAuthorityBounds(
        repository_cid=preflight.eaaef_repository_binding_cid(
            board=board,
            materialization_receipt=receipt,
        ),
        baseline_commit=receipt["source_head"],
        effects=preflight.EAAEF_REQUIRED_ROUTE_EFFECTS,
        budget_cid=preflight.eaaef_provider_budget_binding_cid(board=board),
        resource_cid=profile["profile_cid"],
        authority_cid="sha256:" + "f" * 64,
    )
    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_identity = ed25519_did_key(reviewer_key.public_key())
    artifact_route = {
        "route_id": preflight.EAAEF_ROUTE_ID,
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.6",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    review_values = {
        "board_namespace": preflight.EAAEF_BOARD_NAMESPACE,
        "authorization_kind": "explicit_operator_override",
        "source_head": receipt["source_head"],
        "source_tree": receipt["source_tree"],
        "route": artifact_route,
        "authority_bounds": bounds.as_dict(),
        "reviewer_identity": reviewer_identity,
        "reviewer_provider": "local_operator",
        "reviewer_profile_id": "unit-profile",
        "reviewer_profile_content_id": "sha256:" + "2" * 64,
        "reviewer_lifecycle_anchor_id": "3" * 64,
        "reviewer_lifecycle_generation": 1,
        "reviewer_witness_path": (
            route_module._EAAEF_AGENT_LIFECYCLE_WITNESS_PREFIX
            + receipt["source_tree"]
            + "-unit.json"
        ),
        "reviewer_witness_sha256": "sha256:" + "4" * 64,
        "lifecycle_root_identity_did": "did:key:zUnitLifecycleRoot",
        "lifecycle_witness_nonce": "unit-nonce",
        "lifecycle_root_pin_path": (
            route_module.eaaef_agent_lifecycle_root_pin_path(
                receipt["source_tree"]
            )
        ),
        "lifecycle_root_pin_sha256": "sha256:" + "5" * 64,
        "authorized_at_ms": NOW_MS - 2000,
        "fallback_implementer_identity": "codex",
    }
    review_payload = route_module.agent_implementation_route_review_payload(
        **review_values
    )
    reviewer_signature = base64.b64encode(
        reviewer_key.sign(_canonical(review_payload))
    ).decode("ascii")
    values = {
        "schema": route_module._AGENT_ROUTE_AUTHORIZATION_SCHEMA,
        "board_namespace": preflight.EAAEF_BOARD_NAMESPACE,
        "artifact_path": route_module.eaaef_agent_route_authorization_path(
            receipt["source_tree"]
        ),
        "artifact_sha256": "sha256:" + "1" * 64,
        "authorization_kind": "explicit_operator_override",
        "source_head": receipt["source_head"],
        "source_tree": receipt["source_tree"],
        "reviewer_identity": reviewer_identity,
        "reviewer_provider": "local_operator",
        "reviewer_signature": reviewer_signature,
        "reviewer_profile_id": review_values["reviewer_profile_id"],
        "reviewer_profile_content_id": review_values[
            "reviewer_profile_content_id"
        ],
        "reviewer_lifecycle_anchor_id": review_values[
            "reviewer_lifecycle_anchor_id"
        ],
        "reviewer_lifecycle_generation": review_values[
            "reviewer_lifecycle_generation"
        ],
        "reviewer_witness_path": review_values["reviewer_witness_path"],
        "reviewer_witness_sha256": review_values[
            "reviewer_witness_sha256"
        ],
        "lifecycle_root_identity_did": review_values[
            "lifecycle_root_identity_did"
        ],
        "lifecycle_witness_nonce": review_values[
            "lifecycle_witness_nonce"
        ],
        "lifecycle_root_pin_path": review_values[
            "lifecycle_root_pin_path"
        ],
        "lifecycle_root_pin_sha256": review_values[
            "lifecycle_root_pin_sha256"
        ],
        "authorized_at_ms": review_values["authorized_at_ms"],
        "fallback_implementer_identity": "codex",
        "authority_bounds": bounds.as_dict(),
    }
    authorization_id = route_module._agent_implementation_route_id(values)
    authorization = route_module.AgentImplementationRouteAuthorization(
        board_namespace=values["board_namespace"],
        artifact_path=values["artifact_path"],
        artifact_sha256=values["artifact_sha256"],
        authorization_kind=values["authorization_kind"],
        source_head=values["source_head"],
        source_tree=values["source_tree"],
        authorization_id=authorization_id,
        reviewer_identity=values["reviewer_identity"],
        reviewer_provider=values["reviewer_provider"],
        reviewer_signature=values["reviewer_signature"],
        reviewer_profile_id=values["reviewer_profile_id"],
        reviewer_profile_content_id=values["reviewer_profile_content_id"],
        reviewer_lifecycle_anchor_id=values["reviewer_lifecycle_anchor_id"],
        reviewer_lifecycle_generation=values[
            "reviewer_lifecycle_generation"
        ],
        reviewer_witness_path=values["reviewer_witness_path"],
        reviewer_witness_sha256=values["reviewer_witness_sha256"],
        lifecycle_root_identity_did=values["lifecycle_root_identity_did"],
        lifecycle_witness_nonce=values["lifecycle_witness_nonce"],
        lifecycle_root_pin_path=values["lifecycle_root_pin_path"],
        lifecycle_root_pin_sha256=values["lifecycle_root_pin_sha256"],
        authorized_at_ms=values["authorized_at_ms"],
        fallback_implementer_identity=values[
            "fallback_implementer_identity"
        ],
        authority_bounds=bounds,
        _validation_seal=route_module._agent_implementation_private_seal(
            values
        ),
    )
    route = route_module.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        authorization=authorization,
    )
    control_plane = route_module.AgentImplementationControlPlanePin(
        schema=route_module._AGENT_CONTROL_PLANE_PIN_SCHEMA,
        runner_path="/opt/ipfs-task-tools/agent-runner.pyz",
        runner_sha256="sha256:" + "6" * 64,
        capsule_root="/opt/ipfs-task-tools/capsule",
        capsule_id="sha256:" + "7" * 64,
        source_head=receipt["source_head"],
        source_tree=receipt["source_tree"],
        archive_sha256="sha256:" + "8" * 64,
    )
    invocation = route_module.AgentImplementationInvocationBinding(
        schema=preflight.EAAEF_INVOCATION_BINDING_SCHEMA,
        invocation_id="sha256:" + "9" * 64,
        logical_attempt_id="sha256:" + "a" * 64,
        task_id=preflight.EAAEF_BOOTSTRAP_TASK_ID,
        attempt=1,
        task_revision_cid=board["tasks"][0]["task_spec_cid"],
        prompt_cid="sha256:" + "b" * 64,
        worktree_id="sha256:" + "c" * 64,
        workspace_path="/tmp/eaaef-unit-workspace",
        repository_cid=bounds.repository_cid,
        baseline_commit=bounds.baseline_commit,
        effects=bounds.effects,
        scope_cid="sha256:" + "d" * 64,
        budget_cid=bounds.budget_cid,
        resource_cid=bounds.resource_cid,
        authority_cid=bounds.authority_cid,
        route_id=route.route_id,
        primary_provider_id=route.primary_provider_id,
        primary_model_id=route.primary_model_id,
        fallback_provider_id=route.fallback_provider_id,
        fallback_model_id=route.fallback_model_id,
        fallback_reasoning_effort=route.fallback_reasoning_effort,
        fallback_implementer_identity=route.fallback_implementer_identity,
        reviewer_identity=reviewer_identity,
        reviewer_provider="local_operator",
        profile_id="unit-profile",
        profile_identity_did=reviewer_identity,
        profile_lifecycle_anchor_id="3" * 64,
        profile_lifecycle_generation=1,
        profile_dir="/tmp/eaaef-unit-profile",
        profile_lifecycle_dir="/tmp/eaaef-unit-lifecycle",
        issued_at_ms=NOW_MS - 1000,
        expires_at_ms=invocation_expires_at_ms,
        provider_attempt_store="/tmp/eaaef-unit-attempts",
        provider_attempt_store_identity="sha256:" + "e" * 64,
        control_plane=control_plane,
        reviewer_signature="",
    )
    invocation = replace(
        invocation,
        reviewer_signature=base64.b64encode(
            reviewer_key.sign(_canonical(invocation.signed_payload()))
        ).decode("ascii"),
    )
    return replace(route, invocation_binding=invocation)


def _qualification(
    board,
    receipt,
    route,
    image,
    reviewer,
    profile,
    *,
    prepare_now_ms=NOW_MS,
    expires_at_ms=NOW_MS + 500,
):
    key = Ed25519PrivateKey.generate()
    signer = ed25519_did_key(key.public_key())
    worker_principal = ed25519_did_key(
        Ed25519PrivateKey.generate().public_key()
    )
    provider_principal = ed25519_did_key(
        Ed25519PrivateKey.generate().public_key()
    )
    prepared = preflight.prepare_eaaef_provider_container_qualification(
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        worker_principal_did=worker_principal,
        provider_principal_did=provider_principal,
        signer_identity_did=signer,
        admitted_at_ms=NOW_MS - 500,
        expires_at_ms=expires_at_ms,
        now_ms=prepare_now_ms,
        trusted_image_reviewer_dids=[reviewer],
    )
    signature = base64.b64encode(
        key.sign(
            preflight.eaaef_provider_container_qualification_signing_bytes(
                prepared
            )
        )
    ).decode("ascii")
    return (
        preflight.seal_eaaef_provider_container_qualification(
            prepared_payload=prepared,
            signer_signature=signature,
        ),
        signer,
    )


def _evaluate(*, board=None, receipt=None, route=None, image=None, reviewer="", profile=None, qualification=None, qualification_signer="", **kwargs):
    board = board or _board()
    receipt = receipt or _materialization(board)
    if image is None:
        image, reviewer = _image_qualification()
    profile = profile or _container_profile(image["image_digest"])
    return preflight.evaluate_external_agent_fabric_bootstrap_preflight(
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        trusted_image_reviewer_dids=[reviewer],
        now_ms=NOW_MS,
        expected_worker_principal_did=(
            str(qualification.get("worker_principal_did") or "")
            if isinstance(qualification, dict)
            else ""
        ),
        expected_provider_principal_did=(
            str(qualification.get("provider_principal_did") or "")
            if isinstance(qualification, dict)
            else ""
        ),
        provider_container_qualification=qualification,
        trusted_qualification_signer_dids=[qualification_signer],
        **kwargs,
    )


def test_complete_non_authority_evidence_reports_only_missing_eaaef_route():
    decision = _evaluate()

    assert decision.allowed is False
    assert decision.blockers == (
        "eaaef_scoped_provider_authorization_missing",
        "provider_container_qualification_missing",
        "provider_task_dispatch_not_admitted",
    )
    payload = decision.as_dict()
    assert payload["board_cid"]
    assert payload["materialization_receipt_cid"]
    assert payload["configured_board_capsule_gate_bypassed"] is False
    assert payload["authority_mutated"] is False
    assert payload["process_started"] is False
    assert payload["preflight_cid"] == _cid(
        {key: value for key, value in payload.items() if key != "preflight_cid"}
    )


def test_structural_materialization_does_not_require_live_launch_decision():
    board = _board()
    receipt = _materialization(board)

    decision = _evaluate(board=board, receipt=receipt)

    assert decision.blockers == (
        "eaaef_scoped_provider_authorization_missing",
        "provider_container_qualification_missing",
        "provider_task_dispatch_not_admitted",
    )


def test_structural_materialization_may_omit_live_launch_decision():
    board = _board()
    receipt = _materialization(board)
    receipt["board_validation"].pop("live_launch_allowed")
    receipt.pop("receipt_cid")
    receipt["receipt_cid"] = _cid(receipt)

    decision = _evaluate(board=board, receipt=receipt)

    assert decision.blockers == (
        "eaaef_scoped_provider_authorization_missing",
        "provider_container_qualification_missing",
        "provider_task_dispatch_not_admitted",
    )


def test_network_bridge_and_docker_socket_are_fail_closed():
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    profile["network_mode"] = "bridge"
    profile["docker_socket_mounted"] = True
    profile.pop("profile_cid")
    profile["profile_cid"] = _cid(profile)

    decision = _evaluate(
        image=image,
        reviewer=reviewer,
        profile=profile,
    )

    assert "container_profile_invalid" in decision.blockers


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pids_limit", 4097),
        ("cpu_limit", 65),
        ("cpu_limit", float("inf")),
        ("memory_limit_bytes", 257 * 1024**3),
        ("disk_limit_bytes", 2 * 1024**4 + 1),
        ("maximum_parallel_workers", 1),
        ("maximum_parallel_containers", 6),
    ),
)
def test_container_resources_are_finite_and_bounded(field, value):
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    profile[field] = value
    if value != float("inf"):
        profile.pop("profile_cid")
        profile["profile_cid"] = _cid(profile)

    decision = _evaluate(image=image, reviewer=reviewer, profile=profile)

    assert "container_profile_invalid" in decision.blockers


def test_container_mount_targets_are_allowlisted():
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    profile["mounts"][0]["target"] = "/etc/ld.so.preload"
    profile.pop("profile_cid")
    profile["profile_cid"] = _cid(profile)

    decision = _evaluate(image=image, reviewer=reviewer, profile=profile)

    assert "container_profile_invalid" in decision.blockers


def test_untrusted_or_tampered_sbom_qualification_is_rejected():
    image, _reviewer = _image_qualification()
    image["sbom_bytes"] += 1
    image.pop("qualification_cid")
    image["qualification_cid"] = _cid(image)
    profile = _container_profile(image["image_digest"])

    decision = _evaluate(image=image, reviewer="did:key:zUntrusted", profile=profile)

    assert "oci_image_qualification_invalid" in decision.blockers


def test_configured_board_capsule_gate_cannot_be_bypassed():
    decision = _evaluate(configured_board_capsule_gate_bypassed=True)

    assert "configured_board_capsule_gate_bypass_prohibited" in decision.blockers
    assert decision.as_dict()["configured_board_capsule_gate_bypassed"] is False


def test_multi_supervisor_mode_is_outside_this_preflight():
    decision = _evaluate(launch_mode="configured_multi_supervisor")

    assert "direct_single_supervisor_launch_mode_required" in decision.blockers


def test_board_and_materialization_identity_tampering_is_detected():
    board = _board()
    receipt = _materialization(board)
    board["tasks"] = deepcopy(board["tasks"]) + [{"stable_task_id": "EAAEF-999"}]

    decision = _evaluate(board=board, receipt=receipt)

    assert "board_missing_or_invalid" in decision.blockers


def test_historical_materialization_v1_cannot_satisfy_live_admission():
    board = _board()
    receipt = _materialization(board)
    receipt["schema"] = (
        "ipfs_accelerate_py/agent-supervisor/"
        "external-agent-autonomous-execution-fabric-materialization@1"
    )
    receipt.pop("receipt_cid")
    receipt["receipt_cid"] = _cid(receipt)

    decision = _evaluate(board=board, receipt=receipt)

    assert "materialization_receipt_missing_or_invalid" in decision.blockers


def test_independently_signed_provider_container_qualification_is_nonlaunching():
    board = _board()
    receipt = _materialization(board)
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    route = _sealed_eaaef_route(board, receipt, profile)
    qualification, signer = _qualification(
        board,
        receipt,
        route,
        image,
        reviewer,
        profile,
    )

    verification = preflight.verify_eaaef_provider_container_qualification(
        qualification=qualification,
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        trusted_qualification_signer_dids=[signer],
        trusted_image_reviewer_dids=[reviewer],
        expected_worker_principal_did=qualification["worker_principal_did"],
        expected_provider_principal_did=qualification["provider_principal_did"],
        now_ms=NOW_MS,
    )
    decision = _evaluate(
        board=board,
        receipt=receipt,
        route=route,
        image=image,
        reviewer=reviewer,
        profile=profile,
        qualification=qualification,
        qualification_signer=signer,
    )

    assert verification.valid is True
    assert verification.blockers == ()
    assert verification.as_dict()["authority_mutated"] is False
    assert verification.as_dict()["process_started"] is False
    assert verification.workload_class == "bootstrap_diagnostic_only"
    assert verification.task_dispatch_admitted is False
    assert verification.maximum_parallel_workers == 0
    assert verification.maximum_parallel_containers == 1
    assert verification.worker_principal_did.startswith("did:key:z")
    assert verification.provider_principal_did.startswith("did:key:z")
    assert verification.provider_principal_did != verification.worker_principal_did
    assert decision.allowed is False
    assert decision.blockers == ("provider_task_dispatch_not_admitted",)
    assert decision.as_dict()["authority_mutated"] is False
    assert decision.as_dict()["process_started"] is False


def test_provider_container_effect_binding_cannot_be_resigned_to_widen_scope():
    board = _board()
    receipt = _materialization(board)
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    route = _sealed_eaaef_route(board, receipt, profile)
    qualification, signer = _qualification(
        board,
        receipt,
        route,
        image,
        reviewer,
        profile,
    )
    qualification["route_effects"] = [
        *qualification["route_effects"],
        "push",
    ]
    qualification.pop("signer_signature")
    qualification.pop("receipt_cid")
    signer_key = Ed25519PrivateKey.generate()
    replacement_signer = ed25519_did_key(signer_key.public_key())
    qualification["signer_identity_did"] = replacement_signer
    signature = base64.b64encode(
        signer_key.sign(
            preflight.eaaef_provider_container_qualification_signing_bytes(
                qualification
            )
        )
    ).decode("ascii")
    qualification = preflight.seal_eaaef_provider_container_qualification(
        prepared_payload=qualification,
        signer_signature=signature,
    )

    verification = preflight.verify_eaaef_provider_container_qualification(
        qualification=qualification,
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        trusted_qualification_signer_dids=[signer, replacement_signer],
        trusted_image_reviewer_dids=[reviewer],
        expected_worker_principal_did=qualification["worker_principal_did"],
        expected_provider_principal_did=qualification["provider_principal_did"],
        now_ms=NOW_MS,
    )

    assert verification.valid is False
    assert "provider_container_binding_mismatch" in verification.blockers
    assert "provider_container_signature_invalid" not in verification.blockers


def test_provider_container_service_principals_are_exact_and_independent():
    board = _board()
    receipt = _materialization(board)
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    route = _sealed_eaaef_route(board, receipt, profile)
    qualification, signer = _qualification(
        board,
        receipt,
        route,
        image,
        reviewer,
        profile,
    )

    verification = preflight.verify_eaaef_provider_container_qualification(
        qualification=qualification,
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        trusted_qualification_signer_dids=[signer],
        trusted_image_reviewer_dids=[reviewer],
        expected_worker_principal_did=qualification["worker_principal_did"],
        expected_provider_principal_did=ed25519_did_key(
            Ed25519PrivateKey.generate().public_key()
        ),
        now_ms=NOW_MS,
    )

    assert verification.valid is False
    assert "provider_container_service_principals_invalid" in verification.blockers


def test_provider_container_reviewer_must_differ_from_image_reviewer():
    board = _board()
    receipt = _materialization(board)
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    route = _sealed_eaaef_route(board, receipt, profile)

    with pytest.raises(
        ValueError,
        match="provider_container_reviewer_not_independent",
    ):
        preflight.prepare_eaaef_provider_container_qualification(
            board=board,
            materialization_receipt=receipt,
            route_plan=route,
            image_qualification=image,
            container_profile=profile,
            worker_principal_did=ed25519_did_key(
                Ed25519PrivateKey.generate().public_key()
            ),
            provider_principal_did=ed25519_did_key(
                Ed25519PrivateKey.generate().public_key()
            ),
            signer_identity_did=reviewer,
            admitted_at_ms=NOW_MS - 500,
            expires_at_ms=NOW_MS + 500,
            now_ms=NOW_MS,
            trusted_image_reviewer_dids=[reviewer],
        )


def test_provider_container_signature_and_source_binding_are_fail_closed():
    board = _board()
    receipt = _materialization(board)
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    route = _sealed_eaaef_route(board, receipt, profile)
    qualification, signer = _qualification(
        board,
        receipt,
        route,
        image,
        reviewer,
        profile,
    )
    qualification["source_tree"] = "9" * 40
    qualification.pop("receipt_cid")
    qualification["receipt_cid"] = _cid(qualification)

    verification = preflight.verify_eaaef_provider_container_qualification(
        qualification=qualification,
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        trusted_qualification_signer_dids=[signer],
        trusted_image_reviewer_dids=[reviewer],
        expected_worker_principal_did=qualification["worker_principal_did"],
        expected_provider_principal_did=qualification["provider_principal_did"],
        now_ms=NOW_MS,
    )

    assert verification.valid is False
    assert "provider_container_signature_invalid" in verification.blockers
    assert "provider_container_binding_mismatch" in verification.blockers


def test_image_qualification_expiry_is_half_open():
    image, reviewer = _image_qualification(expires_at_ms=NOW_MS)
    profile = _container_profile(image["image_digest"])

    decision = _evaluate(
        image=image,
        reviewer=reviewer,
        profile=profile,
    )

    assert "oci_image_qualification_invalid" in decision.blockers


def test_provider_container_qualification_expiry_is_half_open():
    board = _board()
    receipt = _materialization(board)
    image, reviewer = _image_qualification()
    profile = _container_profile(image["image_digest"])
    route = _sealed_eaaef_route(board, receipt, profile)
    qualification, signer = _qualification(
        board,
        receipt,
        route,
        image,
        reviewer,
        profile,
        prepare_now_ms=NOW_MS - 1,
        expires_at_ms=NOW_MS,
    )

    verification = preflight.verify_eaaef_provider_container_qualification(
        qualification=qualification,
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        trusted_qualification_signer_dids=[signer],
        trusted_image_reviewer_dids=[reviewer],
        expected_worker_principal_did=qualification["worker_principal_did"],
        expected_provider_principal_did=qualification["provider_principal_did"],
        now_ms=NOW_MS,
    )

    assert verification.valid is False
    assert "provider_container_qualification_time_invalid" in (
        verification.blockers
    )


def _worker_image_qualification(*, expires_at_ms=NOW_MS + 1000):
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    value = {
        "schema": preflight.EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA,
        "image_digest": "sha256:" + "6" * 64,
        "image_label": "eaaef-worker-qualified-fixture",
        "image_os": "linux",
        "image_architecture": "amd64",
        "sbom_digest": "sha256:" + "7" * 64,
        "sbom_format": "spdx-json",
        "sbom_bytes": 4096,
        "toolchain_versions": {
            "python": "3.12.3",
            "git": "2.43.0",
            "codex": "0.147.0",
            "grok": "1.0.5",
        },
        "workload_class": "agent_worker",
        "task_dispatch_verified": True,
        "execution_mode": "rootless_engine",
        "rootless_supported": True,
        "rootless_verified": True,
        "nonroot_hardening_verified": True,
        "daemon_identity_cid": "sha256:" + "0" * 64,
        "daemon_policy_cid": "sha256:" + "a" * 64,
        "credential_disposition": "clean_no_credentials",
        "credential_disposition_evidence_cid": "sha256:" + "b" * 64,
        "reproducible_build_evidence_cid": "sha256:" + "c" * 64,
        "reproducible_build_count": 2,
        "network_policy_cid": "sha256:" + "d" * 64,
        "reviewer_identity_did": reviewer,
        "reviewer_role": "independent_security_reviewer",
        "verified_at_ms": NOW_MS - 1000,
        "expires_at_ms": expires_at_ms,
    }
    value["reviewer_signature"] = base64.b64encode(
        key.sign(
            preflight.eaaef_worker_image_qualification_signing_bytes(value)
        )
    ).decode("ascii")
    value["qualification_cid"] = _cid(value)
    return value, reviewer


def _worker_container_profile(image, worker_principal, provider_principal):
    key = Ed25519PrivateKey.generate()
    reviewer = ed25519_did_key(key.public_key())
    value = _container_profile(image["image_digest"])
    value.pop("profile_cid")
    value.update(
        {
            "schema": preflight.EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA,
            "workload_class": "agent_worker",
            "task_dispatch_admitted": True,
            "network_mode": "policy_proxy_only",
            "maximum_parallel_workers": 2,
            "maximum_parallel_containers": 3,
            "image_qualification_cid": image["qualification_cid"],
            "sbom_digest": image["sbom_digest"],
            "toolchain_versions": dict(image["toolchain_versions"]),
            "network_policy_cid": image["network_policy_cid"],
            "worker_principal_did": worker_principal,
            "provider_principal_did": provider_principal,
            "reviewer_identity_did": reviewer,
            "reviewer_role": "independent_container_security_reviewer",
            "reviewed_at_ms": NOW_MS - 750,
            "expires_at_ms": NOW_MS + 750,
        }
    )
    resource_body = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "external-agent-worker-resource-profile@1"
        ),
        "pids_limit": value["pids_limit"],
        "cpu_limit": value["cpu_limit"],
        "memory_limit_bytes": value["memory_limit_bytes"],
        "disk_limit_bytes": value["disk_limit_bytes"],
        "maximum_parallel_workers": value["maximum_parallel_workers"],
        "maximum_parallel_containers": value[
            "maximum_parallel_containers"
        ],
        "gpu": value["gpu"],
    }
    value["resource_profile_cid"] = _cid(resource_body)
    value["reviewer_signature"] = base64.b64encode(
        key.sign(preflight.eaaef_worker_container_profile_signing_bytes(value))
    ).decode("ascii")
    value["profile_cid"] = _cid(value)
    return value, reviewer


def _closed_worker_chain():
    board = _board()
    receipt = _materialization(board)
    image, image_reviewer = _worker_image_qualification()
    worker_principal = ed25519_did_key(
        Ed25519PrivateKey.generate().public_key()
    )
    provider_principal = ed25519_did_key(
        Ed25519PrivateKey.generate().public_key()
    )
    profile, profile_reviewer = _worker_container_profile(
        image, worker_principal, provider_principal
    )
    route = _sealed_eaaef_route(board, receipt, profile)
    signer_key = Ed25519PrivateKey.generate()
    signer = ed25519_did_key(signer_key.public_key())
    prepared = preflight.prepare_eaaef_provider_container_qualification(
        board=board,
        materialization_receipt=receipt,
        route_plan=route,
        image_qualification=image,
        container_profile=profile,
        worker_principal_did=worker_principal,
        provider_principal_did=provider_principal,
        signer_identity_did=signer,
        admitted_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 500,
        now_ms=NOW_MS,
        trusted_image_reviewer_dids=[image_reviewer],
        trusted_container_profile_reviewer_dids=[profile_reviewer],
    )
    signature = base64.b64encode(
        signer_key.sign(
            preflight.eaaef_provider_container_qualification_signing_bytes(
                prepared
            )
        )
    ).decode("ascii")
    qualification = preflight.seal_eaaef_provider_container_qualification(
        prepared_payload=prepared,
        signer_signature=signature,
    )
    return {
        "board": board,
        "receipt": receipt,
        "image": image,
        "image_reviewer": image_reviewer,
        "profile": profile,
        "profile_reviewer": profile_reviewer,
        "route": route,
        "qualification": qualification,
        "signer": signer,
        "worker_principal": worker_principal,
        "provider_principal": provider_principal,
    }


def _evaluate_worker_chain(chain):
    return preflight.evaluate_external_agent_fabric_bootstrap_preflight(
        board=chain["board"],
        materialization_receipt=chain["receipt"],
        route_plan=chain["route"],
        image_qualification=chain["image"],
        container_profile=chain["profile"],
        trusted_image_reviewer_dids=[chain["image_reviewer"]],
        trusted_container_profile_reviewer_dids=[
            chain["profile_reviewer"]
        ],
        now_ms=NOW_MS,
        expected_worker_principal_did=chain["worker_principal"],
        expected_provider_principal_did=chain["provider_principal"],
        provider_container_qualification=chain["qualification"],
        trusted_qualification_signer_dids=[chain["signer"]],
    )


def test_closed_worker_contract_chain_is_dispatch_admitted_but_effect_free():
    chain = _closed_worker_chain()
    decision = _evaluate_worker_chain(chain)

    assert decision.allowed is True
    assert decision.blockers == ()
    assert decision.image_digest == chain["image"]["image_digest"]
    assert decision.container_profile_cid == chain["profile"]["profile_cid"]
    assert decision.as_dict()["authority_mutated"] is False
    assert decision.as_dict()["process_started"] is False

    verification = preflight.verify_eaaef_provider_container_qualification(
        qualification=chain["qualification"],
        board=chain["board"],
        materialization_receipt=chain["receipt"],
        route_plan=chain["route"],
        image_qualification=chain["image"],
        container_profile=chain["profile"],
        trusted_qualification_signer_dids=[chain["signer"]],
        trusted_image_reviewer_dids=[chain["image_reviewer"]],
        trusted_container_profile_reviewer_dids=[
            chain["profile_reviewer"]
        ],
        expected_worker_principal_did=chain["worker_principal"],
        expected_provider_principal_did=chain["provider_principal"],
        now_ms=NOW_MS,
    )
    assert verification.valid is True
    assert verification.image_digest == chain["image"]["image_digest"]
    assert verification.container_profile_cid == chain["profile"]["profile_cid"]
    assert verification.maximum_parallel_workers == 2


@pytest.mark.parametrize("failure", ("expiry", "identity", "schema"))
def test_worker_contract_expiry_identity_and_schema_are_fail_closed(failure):
    chain = _closed_worker_chain()
    if failure == "expiry":
        chain["profile"]["expires_at_ms"] = NOW_MS
    elif failure == "identity":
        chain["profile"]["worker_principal_did"] = ed25519_did_key(
            Ed25519PrivateKey.generate().public_key()
        )
    else:
        chain["profile"]["schema"] = preflight.EAAEF_CONTAINER_PROFILE_SCHEMA
    chain["profile"].pop("profile_cid")
    chain["profile"]["profile_cid"] = _cid(chain["profile"])

    decision = _evaluate_worker_chain(chain)

    assert decision.allowed is False
    assert "container_profile_invalid" in decision.blockers
