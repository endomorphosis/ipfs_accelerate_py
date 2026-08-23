from __future__ import annotations

import base64
import json
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_bootstrap_admission as admission,
)

NOW_MS = 1_800_000_000_000
REPO_ROOT = Path(__file__).resolve().parents[2]


def _materialization_receipt() -> dict[str, object]:
    config = json.loads(
        (
            REPO_ROOT
            / "config/external_agent_autonomous_execution_fabric_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    raw_bootstrap = dict(config["bootstrap_database_program"])
    raw_operational = dict(config["database_program"])
    bootstrap = DatabaseProgramConfig.from_mapping(raw_bootstrap).to_dict()
    operational = DatabaseProgramConfig.from_mapping(raw_operational).to_dict()
    command_fabric = dict(config["operational_command_fabric"])
    bindings: dict[str, object] = {
        "schema": admission.EAAEF_DATABASE_PROGRAM_BINDINGS_SCHEMA,
        "bootstrap": bootstrap,
        "bootstrap_source_cid": admission._cid(raw_bootstrap),
        "bootstrap_profile_cid": admission._cid(bootstrap),
        "operational": operational,
        "operational_source_cid": admission._cid(raw_operational),
        "operational_database_program_profile_cid": admission._cid(operational),
        "operational_command_fabric": command_fabric,
        "operational_profile_cid": admission._cid(command_fabric),
        "operational_child_adapter_status": "implemented_unqualified_fail_closed",
        "materializer_opens_operational_profile": False,
        "direct_file_fallback": False,
    }
    bindings["binding_cid"] = admission._cid(bindings)
    receipt: dict[str, object] = {
        "schema": admission.EAAEF_MATERIALIZATION_RECEIPT_SCHEMA_V2,
        "source_head": "1" * 40,
        "source_tree": "2" * 40,
        "source_generation": {"source_generation_cid": "sha256:" + "3" * 64},
        "board_validation": {
            "board_cid": "sha256:" + "4" * 64,
            "board_namespace": admission.EAAEF_BOARD_NAMESPACE,
        },
        "population_cid": "sha256:" + "5" * 64,
        "plan_root_cid": "sha256:" + "6" * 64,
        "control_projection": {"projection_root": "sha256:" + "7" * 64},
        "coordination_projection": {"projection_root": "sha256:" + "8" * 64},
        "execution_projection": {"projection_root": "sha256:" + "9" * 64},
        "database_program_bindings": bindings,
    }
    receipt["receipt_cid"] = admission._cid(receipt)
    return receipt


def _readdress_materialization_receipt(receipt: dict[str, object]) -> None:
    bindings = receipt["database_program_bindings"]
    assert isinstance(bindings, dict)
    command_fabric = bindings["operational_command_fabric"]
    assert isinstance(command_fabric, dict)
    bindings["operational_profile_cid"] = admission._cid(command_fabric)
    bindings.pop("binding_cid", None)
    bindings["binding_cid"] = admission._cid(bindings)
    receipt.pop("receipt_cid", None)
    receipt["receipt_cid"] = admission._cid(receipt)


def test_materialization_projection_binds_command_fabric_v2_board_and_shard() -> None:
    receipt = _materialization_receipt()

    result = admission._materialization_projection(receipt)

    assert result["materialization_receipt_cid"] == receipt["receipt_cid"]


def test_materialization_projection_accepts_188_admitted_child_adapter() -> None:
    receipt = _materialization_receipt()
    bindings = receipt["database_program_bindings"]
    assert isinstance(bindings, dict)
    command_fabric = bindings["operational_command_fabric"]
    assert isinstance(command_fabric, dict)
    command_fabric["child_adapter_status"] = "admitted"
    bindings["operational_child_adapter_status"] = "admitted"
    _readdress_materialization_receipt(receipt)

    result = admission._materialization_projection(receipt)

    assert result["materialization_receipt_cid"] == receipt["receipt_cid"]


def test_materialization_projection_rejects_child_adapter_status_mismatch() -> None:
    receipt = _materialization_receipt()
    bindings = receipt["database_program_bindings"]
    assert isinstance(bindings, dict)
    bindings["operational_child_adapter_status"] = "admitted"
    _readdress_materialization_receipt(receipt)

    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="materialization_database_program_binding_invalid",
    ):
        admission._materialization_projection(receipt)


@pytest.mark.parametrize("field", ["board_namespace", "shard_id"])
def test_materialization_projection_rejects_missing_command_fabric_identity(
    field: str,
) -> None:
    receipt = _materialization_receipt()
    bindings = receipt["database_program_bindings"]
    assert isinstance(bindings, dict)
    command_fabric = bindings["operational_command_fabric"]
    assert isinstance(command_fabric, dict)
    command_fabric.pop(field)
    _readdress_materialization_receipt(receipt)

    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="materialization_database_program_binding_invalid",
    ):
        admission._materialization_projection(receipt)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("board_namespace", "other-board"), ("shard_id", "control-shard-1")],
)
def test_materialization_projection_rejects_command_fabric_identity_mismatch(
    field: str,
    replacement: str,
) -> None:
    receipt = _materialization_receipt()
    bindings = receipt["database_program_bindings"]
    assert isinstance(bindings, dict)
    command_fabric = bindings["operational_command_fabric"]
    assert isinstance(command_fabric, dict)
    command_fabric[field] = replacement
    _readdress_materialization_receipt(receipt)

    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="materialization_database_program_binding_invalid",
    ):
        admission._materialization_projection(receipt)


def _statement(*, admitted: bool = True) -> dict[str, object]:
    def sha(token: str) -> str:
        return "sha256:" + token * 64

    value: dict[str, object] = {
        "schema": admission.EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA,
        "task_id": "EAAEF-000",
        "board_namespace": admission.EAAEF_BOARD_NAMESPACE,
        "decision": "admitted" if admitted else "no_go",
        "outcome": "accepted" if admitted else "mutation_not_admitted",
        "blockers": [] if admitted else ["qualification_missing"],
        "board_cid": sha("1"),
        "source_head": "2" * 40,
        "source_tree": "3" * 40,
        "source_generation_cid": sha("4"),
        "materialization_receipt_cid": sha("5"),
        "materialization_store_generation": "eaaef-run-v5",
        "materialization_database_program_binding_cid": sha("5"),
        "materialization_bootstrap_profile_cid": sha("4"),
        "materialization_operational_profile_cid": sha("3"),
        "population_cid": sha("6"),
        "plan_root_cid": sha("7"),
        "control_projection_root": sha("8"),
        "coordination_projection_root": sha("9"),
        "execution_projection_root": sha("a"),
        "provider_container_qualification_cid": sha("b"),
        "provider_container_verification_cid": sha("c"),
        "provider_qualification_signer_did": "did:key:zProvider",
        "image_qualification_reviewer_did": "did:key:zImage",
        "provider_qualification_expires_at_ms": NOW_MS + 100_000,
        "provider_maximum_parallel_workers": 5,
        "provider_maximum_parallel_containers": 5,
        "provider_worker_principal_did": "did:key:zWorker",
        "provider_principal_did": "did:key:zProviderService",
        "provider_task_dispatch_admitted": True,
        "provider_workload_class": "agent_worker",
        "quack_owner_qualification_cid": sha("d"),
        "quack_owner_verification_cid": sha("e"),
        "quack_qualification_reviewer_did": "did:key:zQuack",
        "quack_qualification_expires_at_ms": NOW_MS + 100_000,
        "quack_owner_principal_did": "did:key:zOwner",
        "container_profile_cid": sha("f"),
        "image_digest": sha("0"),
        "quack_shard_id": "eaaef-control",
        "quack_epoch": 7,
        "quack_fence": 11,
        "authority": {
            "launch_mode": "configured_board_multi_supervisor",
            "maximum_lanes": 5,
            "actual_lanes_bounded_by_qualified_resources": True,
            "mutable_coordination_authority": "one_fenced_quack_owner",
            "direct_duckdb_file_open": False,
            "ducklake_current_authority": False,
            "automatic_protected_branch_merge": False,
        },
        "one_use_nonce": "operator-nonce-1",
        "issued_at_ms": NOW_MS - 1000,
        "expires_at_ms": NOW_MS + 100_000,
    }
    value["statement_cid"] = admission._cid(value)
    return value


def _signed_approval(
    statement: dict[str, object],
    role: str,
    key: Ed25519PrivateKey,
) -> tuple[dict[str, object], str]:
    identity = ed25519_did_key(key.public_key())
    value = admission.prepare_external_agent_bootstrap_approval(
        statement,
        role=role,
        identity_did=identity,
        issued_at_ms=NOW_MS - 500,
        expires_at_ms=NOW_MS + 50_000,
    )
    value["signature"] = base64.b64encode(
        key.sign(admission._canonical_bytes(value))
    ).decode("ascii")
    return value, identity


def _receipt(*, admitted: bool = True):
    statement = _statement(admitted=admitted)
    operator, operator_did = _signed_approval(
        statement, "independent_operator", Ed25519PrivateKey.generate()
    )
    security, security_did = _signed_approval(
        statement,
        "independent_security_reviewer",
        Ed25519PrivateKey.generate(),
    )
    receipt = admission.assemble_external_agent_bootstrap_admission(
        statement,
        operator_approval=operator,
        security_approval=security,
        trusted_operator_dids=[operator_did],
        trusted_security_reviewer_dids=[security_did],
        now_ms=NOW_MS,
    )
    return receipt, operator_did, security_did


def test_prepare_emits_no_go_when_runtime_principals_are_absent() -> None:
    receipt = _materialization_receipt()
    board = {
        "board_namespace": admission.EAAEF_BOARD_NAMESPACE,
        "board_cid": receipt["board_validation"]["board_cid"],
    }

    statement = admission.prepare_external_agent_bootstrap_admission(
        board=board,
        materialization_receipt=receipt,
        provider_container_qualification=None,
        route_plan=None,
        image_qualification=None,
        container_profile=None,
        quack_owner_qualification=None,
        trusted_provider_signer_dids=(),
        trusted_image_reviewer_dids=(),
        trusted_container_profile_reviewer_dids=(),
        trusted_quack_reviewer_dids=(),
        expected_worker_principal_did="",
        expected_provider_principal_did="",
        expected_source_commit=str(receipt["source_head"]),
        expected_source_tree=str(receipt["source_tree"]),
        one_use_nonce="nonce-missing-principals",
        issued_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 60_000,
    )

    assert statement["decision"] == "no_go"
    assert statement["outcome"] == "mutation_not_admitted"
    assert "worker_network_runtime_principals_unavailable" in statement["blockers"]
    assert "quack_owner_qualification_missing" in statement["blockers"]
    assert "provider_container_qualification_missing" in statement["blockers"]
    assert "quack_owner_board_identity_mismatch" not in statement["blockers"]
    assert statement["provider_task_dispatch_admitted"] is False


def test_two_independent_approvals_bind_exact_admission_statement() -> None:
    receipt, operator, security = _receipt()

    result = admission.verify_external_agent_bootstrap_admission(
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
    )

    assert result["valid"] is True
    assert result["admitted"] is True
    assert result["maximum_lanes"] == 5
    assert result["authority_mutated"] is False
    assert result["process_started"] is False


def test_no_go_is_signed_but_cannot_satisfy_positive_gate() -> None:
    receipt, operator, security = _receipt(admitted=False)

    with pytest.raises(admission.ExternalAgentBootstrapAdmissionError, match="typed no-go"):
        admission.verify_external_agent_bootstrap_admission(
            receipt,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )
    result = admission.verify_external_agent_bootstrap_admission(
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
        require_admitted=False,
    )
    assert result["admitted"] is False


def test_receipt_tamper_and_self_approval_are_rejected() -> None:
    receipt, operator, security = _receipt()
    tampered = deepcopy(receipt)
    tampered["quack_fence"] = 12
    with pytest.raises(admission.ExternalAgentBootstrapAdmissionError, match="self-address"):
        admission.verify_external_agent_bootstrap_admission(
            tampered,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )

    same_key = Ed25519PrivateKey.generate()
    statement = _statement()
    operator_approval, same_did = _signed_approval(
        statement, "independent_operator", same_key
    )
    security_approval, _ = _signed_approval(
        statement, "independent_security_reviewer", same_key
    )
    with pytest.raises(admission.ExternalAgentBootstrapAdmissionError, match="independent"):
        admission.assemble_external_agent_bootstrap_admission(
            statement,
            operator_approval=operator_approval,
            security_approval=security_approval,
            trusted_operator_dids=[same_did],
            trusted_security_reviewer_dids=[same_did],
            now_ms=NOW_MS,
        )


@pytest.mark.parametrize(
    "principal_field",
    [
        "provider_worker_principal_did",
        "provider_principal_did",
        "quack_owner_principal_did",
    ],
)
def test_runtime_principal_cannot_approve_its_own_launch(
    principal_field: str,
) -> None:
    runtime_key = Ed25519PrivateKey.generate()
    runtime_did = ed25519_did_key(runtime_key.public_key())
    statement = _statement()
    statement[principal_field] = runtime_did
    statement["statement_cid"] = admission._cid(
        {
            key: value
            for key, value in statement.items()
            if key != "statement_cid"
        }
    )
    operator_approval, _ = _signed_approval(
        statement,
        "independent_operator",
        runtime_key,
    )
    security_approval, security_did = _signed_approval(
        statement,
        "independent_security_reviewer",
        Ed25519PrivateKey.generate(),
    )
    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="runtime principals",
    ):
        admission.assemble_external_agent_bootstrap_admission(
            statement,
            operator_approval=operator_approval,
            security_approval=security_approval,
            trusted_operator_dids=[runtime_did],
            trusted_security_reviewer_dids=[security_did],
            now_ms=NOW_MS,
        )


def test_create_once_publication_is_idempotency_fail_closed(tmp_path: Path) -> None:
    receipt, operator, security = _receipt()
    relative = admission.external_agent_bootstrap_admission_relative_path(
        str(receipt["source_head"])
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True, mode=0o700)
    for parent in target.parents:
        if parent == tmp_path:
            break
        parent.chmod(0o700)
    result = admission.publish_external_agent_bootstrap_admission(
        tmp_path,
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
    )
    original = target.read_bytes()
    assert result["receipt_cid"] == receipt["receipt_cid"]
    with pytest.raises(admission.ExternalAgentBootstrapAdmissionError, match="overwrite"):
        admission.publish_external_agent_bootstrap_admission(
            tmp_path,
            receipt,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )
    assert target.read_bytes() == original


def test_publication_rejects_outside_and_symlinked_parent(
    tmp_path: Path,
) -> None:
    receipt, operator, security = _receipt()
    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="reviewed EAAEF prefix",
    ):
        admission.external_agent_bootstrap_admission_relative_path(
            str(receipt["source_head"]),
            registry_prefix="../outside",
        )

    managed = tmp_path / "outside-managed"
    managed.mkdir(mode=0o700)
    prefix_parent = tmp_path / "data/agent_supervisor"
    prefix_parent.mkdir(parents=True, mode=0o700)
    (tmp_path / "data").chmod(0o700)
    prefix_parent.chmod(0o700)
    linked = prefix_parent / "external_agent_autonomous_execution_fabric"
    linked.symlink_to(managed, target_is_directory=True)
    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="parent is unavailable",
    ):
        admission.publish_external_agent_bootstrap_admission(
            tmp_path,
            receipt,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
        )
    assert not list(managed.iterdir())


def test_create_once_issuer_diagnoses_current_head_without_publishing() -> None:
    import importlib.util

    script = REPO_ROOT / "scripts/issue_eaaef_bootstrap_admission.py"
    spec = importlib.util.spec_from_file_location("eaaef_bootstrap_admission_issue", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.diagnose()
    assert report["process_started"] is False
    assert report["configured_board_launch"] is False
    assert report["rematerialize"] is False
    assert report.get("published") is not True
    assert report["exists"] is False
    relative = report["relative_path"]
    assert relative.endswith(f"bootstrap-admission--{report['source_head']}.json")
    assert not (REPO_ROOT / relative).exists()
    assert "materialization_source_or_board_mismatch" in report["blockers"]
    assert report["would_publish"] is False
