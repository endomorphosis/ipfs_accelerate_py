"""Contracts for epic S host-gated admission-evidence tasks.

These tests are host-controlled and must pass without live supervisor launch.
Missing signed artifacts are represented as typed receipts, not xfail/skip.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.validation import eaaef_host_admission
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_bootstrap_admission as bootstrap_admission,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
    BUNDLE_SCHEMA,
    BUNDLE_SIGNATURES_SCHEMA,
    RECEIPT_DIR,
    RECEIPT_FILES,
    RECEIPT_SCHEMA,
    REQUIRED_QUACK_EXTENSION_FINGERPRINT,
    REQUIRED_QUACK_EXTENSION_VERSION,
    REQUIRED_QUACK_PLATFORM,
    admission_bundle_review_payload,
    admission_bundle_target_decision,
    cid,
    classify_blocker,
    closing_task_ids,
    collect_host_admission_receipts,
    eaaef_checkout_has_only_generated_receipt_drift,
    verify_admission_bundle_receipt,
    verify_current_admission_bundle_receipt,
    verify_host_admission_task_receipt,
    verify_prebootstrap_admission_statement,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    plan_automatic_implementation_rescue,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
BOARD = CAMPAIGN / "task_board.json"
HOST_EVIDENCE_IDS = [f"EAAEF-{number}" for number in range(180, 192)]


def _prebootstrap_statement(
    *,
    source_head: str,
    source_tree: str,
    board_cid: str,
    materialization_cid: str,
) -> dict[str, object]:
    issued_at_ms = int(time.time() * 1000)
    value: dict[str, object] = {
        field: "" for field in bootstrap_admission._STATEMENT_FIELDS
    }
    value.update(
        {
            "schema": bootstrap_admission.EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA,
            "task_id": "EAAEF-000",
            "board_namespace": (
                "external-agent-autonomous-execution-fabric-v1"
            ),
            "decision": "no_go",
            "outcome": "mutation_not_admitted",
            "blockers": ["EAAEF-191 host admission bundle pending"],
            "board_cid": board_cid,
            "source_head": source_head,
            "source_tree": source_tree,
            "materialization_receipt_cid": materialization_cid,
            "materialization_store_generation": "eaaef-test-run-v1",
            "materialization_database_program_binding_cid": (
                "sha256:" + "6" * 64
            ),
            "materialization_bootstrap_profile_cid": "sha256:" + "7" * 64,
            "materialization_operational_profile_cid": "sha256:" + "8" * 64,
            "provider_qualification_expires_at_ms": 0,
            "provider_maximum_parallel_workers": 0,
            "provider_maximum_parallel_containers": 0,
            "provider_task_dispatch_admitted": False,
            "quack_qualification_expires_at_ms": 0,
            "quack_epoch": 0,
            "quack_fence": 0,
            "authority": dict(bootstrap_admission._EXPECTED_AUTHORITY),
            "one_use_nonce": "prebootstrap-test-nonce",
            "issued_at_ms": issued_at_ms,
            "expires_at_ms": issued_at_ms + 3_600_000,
        }
    )
    value.pop("statement_cid", None)
    value["statement_cid"] = bootstrap_admission._cid(value)
    return value


def _board() -> dict:
    return json.loads(BOARD.read_text(encoding="utf-8"))


def _tasks() -> dict[str, dict]:
    return {task["stable_task_id"]: task for task in _board()["tasks"]}


def _receipt(name: str) -> dict:
    path = RECEIPT_DIR / name
    assert path.is_file(), f"missing host-admission receipt {name}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("process_started") is not True
    assert payload.get("supervisor_process_started") is not True
    assert payload.get("self_signed") is not True
    assert payload.get("receipt_cid", "").startswith("sha256:")
    return payload


def test_inventory_host_evidence_is_bootstrap_ready_frontier() -> None:
    board = _board()
    tasks = _tasks()
    for task_id in HOST_EVIDENCE_IDS:
        task = tasks[task_id]
        assert task["initial_population"] is True
        assert task["is_schedulable"] is True
        assert task["epic"] == "S"
        assert task["resource_request"]["supervisor_processes"] == 0
        assert task["resource_request"]["provider_concurrency"] == 0
    ready = [
        task_id
        for task_id, task in tasks.items()
        if task["status"] == "todo"
        and task["is_schedulable"]
        and not task["dependencies"]
    ]
    assert ready == ["EAAEF-180", "EAAEF-181", "EAAEF-182", "EAAEF-183"]
    assert "EAAEF-191" in tasks["EAAEF-000"]["dependencies"]
    assert board["goals"]
    assert any(goal["goal_id"] == "EAAEF-G190" for goal in board["goals"])
    owned = tasks["EAAEF-180"]["owned_files"]
    assert "ipfs_accelerate_py/agent_supervisor/validation/eaaef_host_admission.py" in owned
    assert "scripts/collect_eaaef_host_admission_receipts.py" in owned


def test_inventory_classifies_ingest_failures_as_host_bootstrap_recovery() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": "ControlPlaneIdentityError: output path is not a safe identifier",
        }
    )
    assert plan.action is AutoRescueAction.HOST_BOOTSTRAP_RECOVERY
    assert plan.max_provider_rescue_passes == 0
    assert classify_blocker("output path is not a safe identifier") == "auto_recoverable"
    assert classify_blocker("eaaef_scoped_provider_authorization_missing") == (
        "host_gated_external_authority"
    )
    assert "EAAEF-184" in closing_task_ids("eaaef_scoped_provider_authorization_missing")
    evidence_plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": (
                "independently signed network-none worker image, SBOM and five "
                "slot identities are absent"
            ),
        }
    )
    assert evidence_plan.action is AutoRescueAction.HOST_EVIDENCE_MATERIALIZE
    assert evidence_plan.max_provider_rescue_passes == 0
    assert "EAAEF-185" in closing_task_ids(
        "container_policy.bootstrap_image_digest is not a full sha256 identity"
    )


def test_inventory_receipt_classifies_launch_plan_blockers() -> None:
    payload = _receipt("blocker_inventory.json")
    assert payload["schema"] == RECEIPT_SCHEMA
    assert payload["task_id"] == "EAAEF-180"
    items = payload["evidence"]["items"]
    assert items
    classes = {item["class"] for item in items}
    assert "host_gated_external_authority" in classes
    assert all(item["closing_task_ids"] for item in items)


def test_principals_receipt_contract() -> None:
    payload = _receipt("runtime_principals.json")
    principals = payload["evidence"]["principals"]
    dids = [item["did"] for item in principals]
    roles = [item["role"] for item in principals]
    assert roles == ["worker", "provider", "quack_owner"]
    assert len(set(dids)) == 3
    assert all(did.startswith("did:key:z") for did in dids)
    assert payload["evidence"]["secret_material_exported"] is False
    assert payload["evidence"]["admitted_authority"] is False
    dumped = json.dumps(payload)
    assert "BEGIN PRIVATE" not in dumped
    assert "PKCS8" not in dumped
    assert "private_key_pkcs8" not in dumped


def test_duckdb_quack_receipt_contract() -> None:
    payload = _receipt("duckdb_quack_155.json")
    evidence = payload["evidence"]
    assert evidence["required_duckdb"] == "1.5.5"
    assert evidence["required_quack"] == "1.5.5+core"
    assert evidence["network_install_attempted"] is False
    if evidence["observed_duckdb"] != "1.5.5":
        assert evidence["silent_substitution_refused"] is True
        assert payload["decision"] == "typed_missing"
    elif payload["decision"] == "admitted":
        assert evidence["under_approved_import_root"] is True
        assert evidence["quack_probe"]["passes_health_check"] is True
        assert evidence["quack_probe"]["extension"]["installed_from"] == "core"


def test_duckdb_quack_probe_requires_the_exact_host_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import quack_capabilities

    observed: dict[str, object] = {}
    site_packages = tmp_path / "site-packages"
    module_path = site_packages / "duckdb/__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# pinned duckdb package\n", encoding="utf-8")
    native_module_path = site_packages / "_duckdb.test.so"
    native_module_path.write_bytes(b"pinned native duckdb")
    extension_dir = tmp_path / ".duckdb/extensions/v1.5.5/linux_arm64"
    extension_dir.mkdir(parents=True)
    extension_path = extension_dir / "quack.duckdb_extension"
    httpfs_path = extension_dir / "httpfs.duckdb_extension"
    extension_path.write_bytes(b"pinned quack")
    httpfs_path.write_bytes(b"pinned httpfs")

    class _Report:
        passes_health_check = True
        extension = SimpleNamespace(
            installed_from="core",
            extension_version=REQUIRED_QUACK_EXTENSION_VERSION,
            install_path=str(extension_path),
        )
        extension_fingerprint = REQUIRED_QUACK_EXTENSION_FINGERPRINT
        platform_name = "linux"
        platform_machine = "aarch64"

        def to_dict(self) -> dict:
            return {
                "passes_health_check": True,
                "extension": {
                    "installed_from": "core",
                    "extension_version": REQUIRED_QUACK_EXTENSION_VERSION,
                    "install_path": str(extension_path),
                },
                "extension_fingerprint": REQUIRED_QUACK_EXTENSION_FINGERPRINT,
                "platform_name": "linux",
                "platform_machine": "aarch64",
                "profile": observed["profile"].to_dict(),
            }

    def _probe(**kwargs):
        observed.update(kwargs)
        return _Report()

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(
            __version__="1.5.5",
            __file__=str(module_path),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "_duckdb",
        SimpleNamespace(__file__=str(native_module_path)),
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "APPROVED_IMPORT_ROOT",
        site_packages,
    )
    monkeypatch.setattr(quack_capabilities, "probe_quack_capabilities", _probe)

    evidence = eaaef_host_admission.probe_duckdb_quack()

    profile = observed["profile"]
    assert profile.pinned_duckdb_version == "1.5.5"
    assert profile.pinned_extension_fingerprint == REQUIRED_QUACK_EXTENSION_FINGERPRINT
    assert profile.pinned_platform == REQUIRED_QUACK_PLATFORM
    assert profile.allow_experimental_within_minor is False
    assert observed["allow_network_install"] is False
    assert evidence["decision"] == "admitted"
    assert evidence["required_quack_extension_version"] == (
        REQUIRED_QUACK_EXTENSION_VERSION
    )
    assert evidence["required_quack_extension_fingerprint"] == (
        REQUIRED_QUACK_EXTENSION_FINGERPRINT
    )
    assert evidence["required_quack_platform"] == REQUIRED_QUACK_PLATFORM
    assert evidence["observed_module_sha256"] == (
        eaaef_host_admission._stable_regular_file_sha256(module_path)
    )
    assert evidence["observed_native_module_sha256"] == (
        eaaef_host_admission._stable_regular_file_sha256(native_module_path)
    )
    assert evidence["quack_extension_sha256"] == (
        eaaef_host_admission._stable_regular_file_sha256(extension_path)
    )
    assert evidence["httpfs_extension_sha256"] == (
        eaaef_host_admission._stable_regular_file_sha256(httpfs_path)
    )


def test_engine_mode_receipt_contract() -> None:
    payload = _receipt("engine_mode.json")
    evidence = payload["evidence"]
    assert evidence["docker_socket_mounted"] is False
    assert evidence["supervisor_started"] is False
    if evidence.get("rootless") is True and payload["decision"] == "admitted":
        assert evidence.get("host_docker_socket_used") is not True
        assert str(evidence.get("docker_host") or "") != "unix:///var/run/docker.sock"
    else:
        fallback = evidence["fallback_package"]
        assert fallback["signed"] is False
        assert fallback["docker_socket_mount"] == "prohibited"
        assert fallback["independent_security_review_required"] is True
        assert payload["decision"] == "typed_missing"


def test_provider_authorization_receipt_contract() -> None:
    payload = _receipt("provider_authorization.json")
    evidence = payload["evidence"]
    assert evidence["self_signed_rejected"] is True
    assert evidence["supervisor_signed"] is False
    assert evidence["configured_board_launch"] is False
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert evidence["reviewer_provider"] == "local_operator"
        assert evidence["route_id"] == (
            "agent-supervisor-eaaef-v1-grok46-terra56-high-auth-or-hard-quota-v1"
        )
        assert str(evidence.get("artifact_path") or "").endswith(".json")
        assert str(evidence.get("authorization_id") or "").startswith("sha256:")
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_worker_image_receipt_contract() -> None:
    payload = _receipt("worker_image.json")
    evidence = payload["evidence"]
    assert evidence["live_dispatch_claimed"] is False
    assert evidence.get("configured_board_launch") is not True
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert str(evidence.get("image_digest") or "").startswith("sha256:")
        assert len(str(evidence.get("image_digest") or "")) == 71
        assert evidence["required_worker_slots"] == 5
        assert len(evidence.get("slot_identities") or ()) == 5
        assert evidence.get("supervisor_signed") is False
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_container_profile_receipt_contract() -> None:
    payload = _receipt("container_profile.json")
    evidence = payload["evidence"]
    assert evidence.get("configured_board_launch") is not True
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert str(evidence.get("schema") or "").endswith("launch@2")
        assert evidence.get("nonroot_user") == "65532:65532"
        assert evidence.get("read_only_base") is True
        assert evidence.get("cap_drop") == ["ALL"]
        assert evidence.get("supervisor_signed") is False
        assert evidence.get("live_dispatch_claimed") is False
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_worker_network_receipt_contract() -> None:
    payload = _receipt("worker_network.json")
    evidence = payload["evidence"]
    assert evidence["required_lanes"] == 5
    assert evidence.get("configured_board_launch") is not True
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert len(evidence.get("lane_ids") or ()) == 5
        assert len(set(evidence.get("lane_ids") or ())) == 5
        assert evidence.get("docker_network_internal") is True
        assert evidence.get("connect_only_443") is True
        assert evidence.get("create_start_restart_reverification_required") is True
        assert evidence.get("child_propagation_status") == "admitted"
        assert evidence.get("supervisor_signed") is False
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_command_fabric_receipt_contract() -> None:
    payload = _receipt("command_fabric_endpoints.json")
    evidence = payload["evidence"]
    assert evidence.get("implemented_unqualified_fail_closed_admitted") is False
    assert evidence.get("configured_board_launch") is not True
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert evidence["child_adapter_status"] == "admitted"
        assert str(evidence.get("command_authorizer_endpoint") or "").startswith("unix://")
        assert str(evidence.get("quack_ingress_endpoint") or "").startswith("unix://")
        assert str(evidence.get("dispatcher_endpoint") or "").startswith("unix://")
        assert evidence.get("supervisor_signed") is False
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_command_fabric_endpoint_requires_a_live_owner_only_listener(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    directory = runtime / "eaaef-cf"
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    endpoint = eaaef_host_admission._unix_endpoint("authorizer")
    path = Path(endpoint.removeprefix("unix://"))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        server.listen(1)
        os.chmod(path, 0o600)
        assert eaaef_host_admission._unix_listener_live(endpoint) is True
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        assert eaaef_host_admission._unix_listener_live(endpoint) is True
    finally:
        server.close()
    assert eaaef_host_admission._unix_listener_live(endpoint) is False


def test_connect_only_command_fabric_listeners_never_admit_runtime_authority(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "eaaef-cf"
    directory.mkdir(mode=0o700)
    servers: list[socket.socket] = []
    evidence: dict[str, str] = {}
    try:
        for field, name in (
            ("command_authorizer_endpoint", "authorizer"),
            ("quack_ingress_endpoint", "ingress"),
            ("quack_projection_endpoint", "projection"),
            ("dispatcher_endpoint", "dispatcher"),
        ):
            path = directory / f"{name}.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)
            os.chmod(path, 0o600)
            servers.append(server)
            evidence[field] = f"unix://{path}"

        assert all(
            eaaef_host_admission._unix_listener_live(endpoint)
            for endpoint in evidence.values()
        )
        assert eaaef_host_admission.command_fabric_endpoints_live(evidence) is False
    finally:
        for server in servers:
            server.close()


def test_host_collector_never_mints_command_fabric_authority_from_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "command-fabric-endpoints.json"
    monkeypatch.setattr(
        eaaef_host_admission,
        "COMMAND_FABRIC_ARTIFACT",
        artifact,
    )

    evidence = eaaef_host_admission.materialize_command_fabric(
        duckdb={"decision": "admitted"},
    )

    assert evidence["decision"] == "typed_missing"
    assert "authenticated" in str(evidence.get("reason") or "")
    assert artifact.exists() is False


def test_native_lane_receipt_contract() -> None:
    payload = _receipt("native_lane_dispatcher.json")
    evidence = payload["evidence"]
    assert evidence.get("configured_board_launch") is not True
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert evidence["native_dependency_admission"] == (
            "AgentSupervisorNativeDependencyAdmission@1"
        )
        assert evidence["lane_authority"] == "EAAEFBootstrapLaneAuthority@2"
        assert str(evidence.get("quack_extension_sha256") or "").startswith("sha256:")
        assert evidence.get("supervisor_signed") is False
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_plan_r2_receipt_contract() -> None:
    payload = _receipt("plan_r2_remote_owner.json")
    evidence = payload["evidence"]
    assert evidence["r1_evidence_promotes_r2"] is False
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert evidence["allowed_operations"] == [
            "plan_r2.prepare",
            "plan_r2.apply",
            "plan_r2.observe",
        ]
        assert evidence.get("supervisor_signed") is False
    else:
        assert payload["decision"] == "typed_missing"
        assert evidence["independent_signature_present"] is False


def test_admission_bundle_receipt_contract() -> None:
    payload = _receipt("admission_bundle.json")
    evidence = payload["evidence"]
    assert payload["schema"] == BUNDLE_SCHEMA
    assert payload["decision"] in {"no_go", "admitted"}
    assert evidence["prospective_supervisor_signature_rejected"] is True
    assert evidence["launch_plan_allowed"] is False
    child_cids = evidence["child_receipt_cids"]
    for task_id, filename in RECEIPT_FILES.items():
        if task_id == "EAAEF-191":
            continue
        child = _receipt(filename)
        assert child_cids[task_id] == child["receipt_cid"]
    if evidence.get("independent_signature_present") is True:
        assert evidence["independent_operator_signature"]
        assert evidence["independent_security_reviewer_signature"]
        assert evidence.get("operator_did", "").startswith("did:key:z")
        assert evidence.get("security_reviewer_did", "").startswith("did:key:z")
    else:
        assert evidence["independent_operator_signature"] == ""
        assert evidence["independent_security_reviewer_signature"] == ""
    if payload["decision"] == "admitted":
        assert evidence["independent_signature_present"] is True
        assert evidence.get("configured_board_launch") is not True
    assert _tasks()["EAAEF-191"]["completion_mode"] == "manual"
    assert _tasks()["EAAEF-183"]["completion_mode"] == "auto"
    assert _tasks()["EAAEF-184"]["completion_mode"] == "auto"


def test_admission_review_binds_source_children_and_host_gate_inventory() -> None:
    child_decisions = {
        task_id: ("admitted" if task_id not in {"EAAEF-180", "EAAEF-181"} else "held")
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    child_cids = {
        task_id: "sha256:" + f"{index:064x}"
        for index, task_id in enumerate(child_decisions, start=1)
    }
    bootstrap_cid = "sha256:" + "a" * 64
    materialization_cid = "sha256:" + "b" * 64
    assert admission_bundle_target_decision(
        child_decisions=child_decisions,
        bootstrap_admission_preflight_valid=True,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
    ) == "admitted"

    assert admission_bundle_target_decision(
        child_decisions=child_decisions,
        bootstrap_admission_preflight_valid=False,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
    ) == "no_go"

    common = {
        "child_decisions": child_decisions,
        "child_receipt_cids": child_cids,
        "decision": "admitted",
        "launch_plan_allowed": False,
        "source_head": "1" * 40,
        "source_tree": "2" * 40,
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "board_cid": "sha256:" + "c" * 64,
        "bootstrap_admission_statement_cid": bootstrap_cid,
        "materialization_receipt_cid": materialization_cid,
        "inventory_open_host_gated": (),
    }
    review = admission_bundle_review_payload(**common)
    changed = admission_bundle_review_payload(**{**common, "source_head": "3" * 40})
    assert review != changed
    changed = admission_bundle_review_payload(
        **{
            **common,
            "child_receipt_cids": {**child_cids, "EAAEF-190": "sha256:" + "d" * 64},
        }
    )
    assert review != changed
    changed = admission_bundle_review_payload(
        **{
            **common,
            "inventory_open_host_gated": ("bootstrap bundle pending",),
        }
    )
    assert review != changed


def test_prebootstrap_statement_requires_canonical_current_materialization() -> None:
    source_head = "1" * 40
    source_tree = "2" * 40
    board_cid = "sha256:" + "3" * 64
    materialization_cid = "sha256:" + "4" * 64
    statement = _prebootstrap_statement(
        source_head=source_head,
        source_tree=source_tree,
        board_cid=board_cid,
        materialization_cid=materialization_cid,
    )
    verified = verify_prebootstrap_admission_statement(
        statement=statement,
        expected_source_head=source_head,
        expected_source_tree=source_tree,
        expected_board_namespace=(
            "external-agent-autonomous-execution-fabric-v1"
        ),
        expected_board_cid=board_cid,
        expected_materialization_receipt_cid=materialization_cid,
        now_ms=int(statement["issued_at_ms"]) + 1,
    )
    assert verified["statement_cid"] == statement["statement_cid"]

    forged = {**statement, "statement_cid": "sha256:" + "f" * 64}
    with pytest.raises(
        bootstrap_admission.ExternalAgentBootstrapAdmissionError,
        match="admission statement is invalid",
    ):
        verify_prebootstrap_admission_statement(
            statement=forged,
            expected_source_head=source_head,
            expected_source_tree=source_tree,
            expected_board_namespace=(
                "external-agent-autonomous-execution-fabric-v1"
            ),
            expected_board_cid=board_cid,
            expected_materialization_receipt_cid=materialization_cid,
            now_ms=int(statement["issued_at_ms"]) + 1,
        )

    stale = {**statement, "materialization_receipt_cid": "sha256:" + "e" * 64}
    stale.pop("statement_cid")
    stale["statement_cid"] = bootstrap_admission._cid(stale)
    with pytest.raises(
        bootstrap_admission.ExternalAgentBootstrapAdmissionError,
        match="binding differs",
    ):
        verify_prebootstrap_admission_statement(
            statement=stale,
            expected_source_head=source_head,
            expected_source_tree=source_tree,
            expected_board_namespace=(
                "external-agent-autonomous-execution-fabric-v1"
            ),
            expected_board_cid=board_cid,
            expected_materialization_receipt_cid=materialization_cid,
            now_ms=int(statement["issued_at_ms"]) + 1,
        )

    overlong = {
        **statement,
        "expires_at_ms": int(statement["issued_at_ms"]) + 3_600_001,
    }
    overlong.pop("statement_cid")
    overlong["statement_cid"] = bootstrap_admission._cid(overlong)
    with pytest.raises(
        bootstrap_admission.ExternalAgentBootstrapAdmissionError,
        match="binding differs",
    ):
        verify_prebootstrap_admission_statement(
            statement=overlong,
            expected_source_head=source_head,
            expected_source_tree=source_tree,
            expected_board_namespace=(
                "external-agent-autonomous-execution-fabric-v1"
            ),
            expected_board_cid=board_cid,
            expected_materialization_receipt_cid=materialization_cid,
            now_ms=int(overlong["issued_at_ms"]) + 1,
        )


def test_tracked_admission_word_is_not_current_launch_authority() -> None:
    board = _board()
    verification = verify_admission_bundle_receipt(
        expected_source_head="0" * 40,
        expected_source_tree="1" * 40,
        expected_board_namespace=str(board["board_namespace"]),
        expected_board_cid=str(board["board_cid"]),
    )
    assert verification["admitted"] is False
    assert verification["blockers"]


def test_current_launch_authority_rejects_non_receipt_checkout_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(
        ["git", "init", "-q"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "eaaef-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "EAAEF Test"],
        cwd=tmp_path,
        check=True,
    )
    runtime = tmp_path / "ipfs_accelerate_py/runtime.py"
    runtime.parent.mkdir()
    runtime.write_text("SOURCE = 'committed'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=tmp_path,
        check=True,
    )
    receipt = (
        tmp_path
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "receipts/host_admission/blocker_inventory.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    fsmonitor_sentinel = tmp_path.parent / "hostile-fsmonitor-ran"
    fsmonitor = tmp_path.parent / "hostile-fsmonitor.sh"
    fsmonitor.write_text(
        "#!/bin/sh\n"
        f"printf bad > {str(fsmonitor_sentinel)!r}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fsmonitor.chmod(0o700)
    subprocess.run(
        ["git", "config", "core.fsmonitor", str(fsmonitor)],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker-selected-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-selected-worktree"))
    assert eaaef_checkout_has_only_generated_receipt_drift(tmp_path) is True
    assert not fsmonitor_sentinel.exists()

    test_git_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_WORK_TREE"}
    }
    subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "update-index",
            "--assume-unchanged",
            "ipfs_accelerate_py/runtime.py",
        ],
        cwd=tmp_path,
        check=True,
        env=test_git_environment,
    )
    runtime.write_text("SOURCE = 'dirty runtime'\n", encoding="utf-8")
    assert eaaef_checkout_has_only_generated_receipt_drift(tmp_path) is False
    assert not fsmonitor_sentinel.exists()
    with pytest.raises(RuntimeError, match="non-receipt source drift"):
        verify_current_admission_bundle_receipt(tmp_path)
    assert not fsmonitor_sentinel.exists()


def _write_current_task_receipt(
    receipt_dir: Path,
    *,
    task_id: str,
    decision: str,
    source_head: str,
    source_tree: str,
    board_namespace: str,
    board_cid: str,
    overrides: dict | None = None,
) -> dict:
    payload = {
        "schema": RECEIPT_SCHEMA if task_id != "EAAEF-191" else BUNDLE_SCHEMA,
        "task_id": task_id,
        "receipt_name": RECEIPT_FILES[task_id],
        "decision": decision,
        "process_started": False,
        "supervisor_process_started": False,
        "self_signed": False,
        "independent_signatures": [],
        "source_head": source_head,
        "source_tree": source_tree,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
        "evidence": {},
        **(overrides or {}),
    }
    payload["receipt_cid"] = cid(payload)
    (receipt_dir / RECEIPT_FILES[task_id]).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def test_single_task_verifier_accepts_only_the_task_decision(tmp_path: Path) -> None:
    expected = {
        "receipt_dir": tmp_path,
        "expected_source_head": "1" * 40,
        "expected_source_tree": "2" * 40,
        "expected_board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "expected_board_cid": "sha256:" + "3" * 64,
    }
    accepted_decisions = {
        "EAAEF-180": "inventory",
        "EAAEF-181": "bound_unadmitted",
        **{f"EAAEF-{number}": "admitted" for number in range(182, 191)},
    }
    for task_id, decision in accepted_decisions.items():
        _write_current_task_receipt(
            tmp_path,
            task_id=task_id,
            decision=decision,
            source_head=expected["expected_source_head"],
            source_tree=expected["expected_source_tree"],
            board_namespace=expected["expected_board_namespace"],
            board_cid=expected["expected_board_cid"],
        )
        verification = verify_host_admission_task_receipt(
            task_id=task_id,
            **expected,
        )
        assert verification == {
            "valid": True,
            "decision": decision,
            "blockers": [],
        }

        _write_current_task_receipt(
            tmp_path,
            task_id=task_id,
            decision="typed_missing",
            source_head=expected["expected_source_head"],
            source_tree=expected["expected_source_tree"],
            board_namespace=expected["expected_board_namespace"],
            board_cid=expected["expected_board_cid"],
        )
        rejected = verify_host_admission_task_receipt(task_id=task_id, **expected)
        assert rejected["valid"] is False
        assert f"{task_id} host receipt decision is not {decision}" in rejected[
            "blockers"
        ]


def test_single_task_verifier_binds_identity_source_and_launch_separation(
    tmp_path: Path,
) -> None:
    task_id = "EAAEF-182"
    expected = {
        "receipt_dir": tmp_path,
        "expected_source_head": "1" * 40,
        "expected_source_tree": "2" * 40,
        "expected_board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "expected_board_cid": "sha256:" + "3" * 64,
    }
    payload = _write_current_task_receipt(
        tmp_path,
        task_id=task_id,
        decision="admitted",
        source_head="4" * 40,
        source_tree="5" * 40,
        board_namespace="wrong-board",
        board_cid="sha256:" + "6" * 64,
        overrides={
            "schema": "wrong-schema",
            "task_id": "EAAEF-183",
            "receipt_name": "wrong.json",
            "process_started": True,
            "supervisor_process_started": None,
            "self_signed": True,
        },
    )
    payload["receipt_cid"] = "sha256:" + "7" * 64
    (tmp_path / RECEIPT_FILES[task_id]).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verification = verify_host_admission_task_receipt(task_id=task_id, **expected)
    assert verification["valid"] is False
    for blocker in (
        "EAAEF-182 host receipt schema differs",
        "EAAEF-182 host receipt task identity differs",
        "EAAEF-182 host receipt filename differs",
        "EAAEF-182 host receipt CID differs",
        "EAAEF-182 host receipt source_head differs",
        "EAAEF-182 host receipt source_tree differs",
        "EAAEF-182 host receipt board_namespace differs",
        "EAAEF-182 host receipt board_cid differs",
        "EAAEF-182 host receipt launch-separation field process_started differs",
        "EAAEF-182 host receipt launch-separation field supervisor_process_started differs",
        "EAAEF-182 host receipt launch-separation field self_signed differs",
    ):
        assert blocker in verification["blockers"]


def test_bundle_task_verifier_requires_full_admission_verification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tmp_path.chmod(0o700)
    expected = {
        "receipt_dir": tmp_path,
        "final_dir": tmp_path,
        "final_root": tmp_path,
        "expected_source_head": "1" * 40,
        "expected_source_tree": "2" * 40,
        "expected_board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "expected_board_cid": "sha256:" + "3" * 64,
    }
    bundle = _write_current_task_receipt(
        tmp_path,
        task_id="EAAEF-191",
        decision="admitted",
        source_head=expected["expected_source_head"],
        source_tree=expected["expected_source_tree"],
        board_namespace=expected["expected_board_namespace"],
        board_cid=expected["expected_board_cid"],
    )
    source_bundle_path, _source_signatures_path = (
        eaaef_host_admission.source_addressed_admission_bundle_paths(
            final_dir=tmp_path,
            source_head=expected["expected_source_head"],
        )
    )
    source_bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_bundle_path.chmod(0o400)
    calls = []

    def _reject_bundle(**kwargs):
        calls.append(kwargs)
        return {
            "admitted": False,
            "decision": "admitted",
            "target_decision": "admitted",
            "blockers": ["signed child receipt replay"],
        }

    monkeypatch.setattr(
        eaaef_host_admission,
        "verify_admission_bundle_receipt",
        _reject_bundle,
    )
    rejected = verify_host_admission_task_receipt(task_id="EAAEF-191", **expected)
    assert rejected["valid"] is False
    assert "signed child receipt replay" in rejected["blockers"]
    assert calls == [{**expected, "require_source_addressed": True}]

    monkeypatch.setattr(
        eaaef_host_admission,
        "verify_admission_bundle_receipt",
        lambda **_kwargs: {
            "admitted": True,
            "decision": "admitted",
            "target_decision": "admitted",
            "bundle_receipt_cid": bundle["receipt_cid"],
            "blockers": [],
        },
    )
    admitted = verify_host_admission_task_receipt(task_id="EAAEF-191", **expected)
    assert admitted == {
        "valid": True,
        "decision": "admitted",
        "blockers": [],
        "receipt_cid": bundle["receipt_cid"],
    }
    monkeypatch.setattr(
        eaaef_host_admission,
        "verify_admission_bundle_receipt",
        lambda **_kwargs: {
            "admitted": True,
            "decision": "admitted",
            "target_decision": "admitted",
            "bundle_receipt_cid": "sha256:" + "f" * 64,
            "blockers": [],
        },
    )
    mismatched_cid = verify_host_admission_task_receipt(
        task_id="EAAEF-191", **expected
    )
    assert mismatched_cid["valid"] is False
    assert "EAAEF-191 verified bundle receipt CID differs" in mismatched_cid[
        "blockers"
    ]
    monkeypatch.setattr(
        eaaef_host_admission,
        "verify_admission_bundle_receipt",
        lambda **_kwargs: {
            "admitted": True,
            "decision": "admitted",
            "target_decision": "admitted",
            "bundle_receipt_cid": bundle["receipt_cid"],
            "blockers": [],
        },
    )
    source_bundle_path.chmod(0o600)
    insecure = verify_host_admission_task_receipt(
        task_id="EAAEF-191", **expected
    )
    assert insecure["valid"] is False
    assert "EAAEF-191 host receipt is unavailable or malformed" in insecure[
        "blockers"
    ]


def test_current_signed_bundle_rejects_child_receipt_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tmp_path.chmod(0o700)
    source_head = "1" * 40
    source_tree = "2" * 40
    board_namespace = "external-agent-autonomous-execution-fabric-v1"
    board_cid = "sha256:" + "3" * 64
    materialization_cid = "sha256:" + "5" * 64
    bootstrap_statement = _prebootstrap_statement(
        source_head=source_head,
        source_tree=source_tree,
        board_cid=board_cid,
        materialization_cid=materialization_cid,
    )
    bootstrap_cid = str(bootstrap_statement["statement_cid"])
    open_host_gates = ["EAAEF-191 host admission bundle pending"]
    operator_key = Ed25519PrivateKey.generate()
    reviewer_key = Ed25519PrivateKey.generate()
    operator_did = ed25519_did_key(operator_key.public_key())
    reviewer_did = ed25519_did_key(reviewer_key.public_key())
    monkeypatch.setattr(
        eaaef_host_admission,
        "TRUSTED_OPERATOR_DIDS",
        (operator_did,),
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "TRUSTED_SECURITY_REVIEWER_DIDS",
        (reviewer_did,),
    )

    child_decisions: dict[str, str] = {}
    child_receipt_cids: dict[str, str] = {}
    for task_id, filename in RECEIPT_FILES.items():
        if task_id == "EAAEF-191":
            continue
        decision = (
            "admitted"
            if task_id not in {"EAAEF-180", "EAAEF-181"}
            else "inventory"
        )
        child = {
            "schema": RECEIPT_SCHEMA,
            "task_id": task_id,
            "receipt_name": filename,
            "decision": decision,
            "process_started": False,
            "supervisor_process_started": False,
            "self_signed": False,
            "independent_signatures": [],
            "source_head": source_head,
            "source_tree": source_tree,
            "board_namespace": board_namespace,
            "board_cid": board_cid,
            "evidence": (
                {
                    "bootstrap_admission_statement": bootstrap_statement,
                    "items": [
                        {
                            "blocker": "EAAEF-191 host admission bundle pending",
                            "class": "host_gated_external_authority",
                            "closing_task_ids": ["EAAEF-191"],
                        }
                    ],
                }
                if task_id == "EAAEF-180"
                else {}
            ),
        }
        child["receipt_cid"] = cid(child)
        (tmp_path / filename).write_text(
            json.dumps(child, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        child_decisions[task_id] = decision
        child_receipt_cids[task_id] = str(child["receipt_cid"])

    review = admission_bundle_review_payload(
        child_decisions=child_decisions,
        child_receipt_cids=child_receipt_cids,
        decision="admitted",
        launch_plan_allowed=False,
        source_head=source_head,
        source_tree=source_tree,
        board_namespace=board_namespace,
        board_cid=board_cid,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
        inventory_open_host_gated=open_host_gates,
    )
    canonical_review = json.dumps(
        review,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    operator_signature = base64.b64encode(
        operator_key.sign(canonical_review)
    ).decode("ascii")
    reviewer_payload = {
        **review,
        "operator_did": operator_did,
        "operator_signature": operator_signature,
    }
    reviewer_signature = base64.b64encode(
        reviewer_key.sign(
            json.dumps(
                reviewer_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
    ).decode("ascii")
    signatures = {
        "schema": BUNDLE_SIGNATURES_SCHEMA,
        "operator_did": operator_did,
        "operator_signature": operator_signature,
        "security_reviewer_did": reviewer_did,
        "security_reviewer_signature": reviewer_signature,
        "payload_sha256": cid(review),
        "supervisor_signed": False,
        "configured_board_launch": False,
        "decision": "admitted",
    }
    (tmp_path / "admission_bundle.signatures.json").write_text(
        json.dumps(signatures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "task_id": "EAAEF-191",
        "receipt_name": RECEIPT_FILES["EAAEF-191"],
        "decision": "admitted",
        "process_started": False,
        "supervisor_process_started": False,
        "self_signed": False,
        "independent_signatures": [],
        "source_head": source_head,
        "source_tree": source_tree,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
        "evidence": {
            "child_receipt_cids": child_receipt_cids,
            "launch_plan_allowed": False,
            "bootstrap_admission_statement_cid": bootstrap_cid,
            "materialization_receipt_cid": materialization_cid,
            "independent_operator_signature": operator_signature,
            "independent_security_reviewer_signature": reviewer_signature,
            "operator_did": operator_did,
            "security_reviewer_did": reviewer_did,
            "independent_signature_present": True,
            "prospective_supervisor_signature_rejected": True,
            "inventory_open_host_gated": open_host_gates,
        },
    }
    bundle["receipt_cid"] = cid(bundle)
    (tmp_path / RECEIPT_FILES["EAAEF-191"]).write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected = {
        "receipt_dir": tmp_path,
        "final_dir": tmp_path,
        "final_root": tmp_path,
        "expected_source_head": source_head,
        "expected_source_tree": source_tree,
        "expected_board_namespace": board_namespace,
        "expected_board_cid": board_cid,
        "prebootstrap_statement_now_ms": (
            int(bootstrap_statement["issued_at_ms"]) + 1
        ),
    }
    assert verify_admission_bundle_receipt(**expected)["admitted"] is True
    strict_without_source = verify_admission_bundle_receipt(
        **expected,
        require_source_addressed=True,
    )
    assert strict_without_source["admitted"] is False
    assert (
        "EAAEF-191 source-addressed final artifacts are unavailable"
        in strict_without_source["blockers"]
    )
    source_bundle_path, source_signatures_path = (
        eaaef_host_admission.source_addressed_admission_bundle_paths(
            final_dir=tmp_path,
            source_head=source_head,
        )
    )
    source_bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_signatures_path.write_text(
        json.dumps(signatures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_bundle_path.chmod(0o400)
    source_signatures_path.chmod(0o400)
    for task_id, filename in RECEIPT_FILES.items():
        if task_id == "EAAEF-191":
            continue
        snapshot = eaaef_host_admission.source_addressed_child_receipt_path(
            final_dir=tmp_path,
            source_head=source_head,
            task_id=task_id,
        )
        snapshot.write_bytes((tmp_path / filename).read_bytes())
        snapshot.chmod(0o400)
    assert (
        verify_admission_bundle_receipt(
            **expected,
            require_source_addressed=True,
        )["admitted"]
        is True
    )
    expired_expected = {
        **expected,
        "prebootstrap_statement_now_ms": int(
            bootstrap_statement["expires_at_ms"]
        )
        + 1,
    }
    assert verify_admission_bundle_receipt(**expired_expected)["admitted"] is False
    historical = verify_admission_bundle_receipt(
        **expired_expected,
        require_source_addressed=True,
    )
    assert historical["admitted"] is True
    assert historical["blockers"] == []
    source_bundle_path.unlink()
    source_signatures_path.unlink()

    bundle_path = tmp_path / RECEIPT_FILES["EAAEF-191"]
    mismatched_bundle = {
        **bundle,
        "evidence": {
            **bundle["evidence"],
            "bootstrap_admission_statement_cid": "sha256:" + "f" * 64,
        },
    }
    mismatched_bundle.pop("receipt_cid", None)
    mismatched_bundle["receipt_cid"] = cid(mismatched_bundle)
    bundle_path.write_text(
        json.dumps(mismatched_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mismatched = verify_admission_bundle_receipt(**expected)
    assert mismatched["admitted"] is False
    assert mismatched["target_decision"] == "no_go"
    assert "EAAEF-191 pre-bootstrap statement identity differs" in mismatched[
        "blockers"
    ]
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    signatures_path = tmp_path / "admission_bundle.signatures.json"
    mislabeled = {**signatures, "configured_board_launch": True}
    signatures_path.write_text(
        json.dumps(mislabeled, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert verify_admission_bundle_receipt(**expected)["admitted"] is False
    signatures_path.write_text(
        json.dumps(signatures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay_path = tmp_path / RECEIPT_FILES["EAAEF-190"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["evidence"] = {"replayed": True}
    replay.pop("receipt_cid")
    replay["receipt_cid"] = cid(replay)
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejected = verify_admission_bundle_receipt(**expected)
    assert rejected["admitted"] is False
    assert "EAAEF-191 child receipt identities differ" in rejected["blockers"]


def test_collector_records_live_allowed_plan_without_launching() -> None:
    receipts = collect_host_admission_receipts(
        launch_plan={"allowed": True, "process_started": False, "blockers": []}
    )
    assert receipts["EAAEF-191"]["process_started"] is False
    assert receipts["EAAEF-191"]["evidence"]["launch_plan_allowed"] is False
    assert receipts["EAAEF-180"]["evidence"]["launch_plan_allowed"] is False


def test_collector_refuses_a_plan_that_started_a_process() -> None:
    try:
        collect_host_admission_receipts(
            launch_plan={"allowed": False, "process_started": True, "blockers": []}
        )
    except RuntimeError as exc:
        assert "started a process" in str(exc)
    else:
        raise AssertionError("collector accepted a process-starting plan")
