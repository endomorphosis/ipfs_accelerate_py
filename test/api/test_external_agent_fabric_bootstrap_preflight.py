from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_fabric_bootstrap as preflight,
)

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
        "tasks": [{"stable_task_id": "EAAEF-001"}],
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
    }
    value["receipt_cid"] = _cid(value)
    return value


def _image_qualification():
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
        "rootless_verified": True,
        "reviewer_identity_did": reviewer,
        "reviewer_role": "independent_security_reviewer",
        "verified_at_ms": NOW_MS - 1000,
        "expires_at_ms": NOW_MS + 1000,
    }
    signature = key.sign(_canonical(value))
    value["reviewer_signature"] = base64.b64encode(signature).decode("ascii")
    value["qualification_cid"] = _cid(value)
    return value, reviewer


def _container_profile(image_digest):
    value = {
        "schema": preflight.EAAEF_CONTAINER_PROFILE_SCHEMA,
        "runtime": "oci",
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
            {
                "source_identity": "sha256:" + "9" * 64,
                "target": "/opt/codex-home/auth.json",
                "read_only": True,
                "kind": "provider_auth",
            },
        ],
    }
    value["profile_cid"] = _cid(value)
    return value


def _evaluate(*, board=None, receipt=None, route=None, image=None, reviewer="", profile=None, **kwargs):
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
        **kwargs,
    )


def test_complete_non_authority_evidence_reports_only_missing_eaaef_route():
    decision = _evaluate()

    assert decision.allowed is False
    assert decision.blockers == ("eaaef_scoped_provider_authorization_missing",)
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

    assert decision.blockers == ("eaaef_scoped_provider_authorization_missing",)


def test_structural_materialization_may_omit_live_launch_decision():
    board = _board()
    receipt = _materialization(board)
    receipt["board_validation"].pop("live_launch_allowed")
    receipt.pop("receipt_cid")
    receipt["receipt_cid"] = _cid(receipt)

    decision = _evaluate(board=board, receipt=receipt)

    assert decision.blockers == ("eaaef_scoped_provider_authorization_missing",)


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
    profile["mounts"][1]["target"] = "/etc/ld.so.preload"
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
