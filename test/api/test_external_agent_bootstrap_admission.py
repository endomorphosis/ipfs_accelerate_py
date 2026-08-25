from __future__ import annotations

import base64
import hashlib
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


def test_create_once_publication_accepts_exact_replay_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    authority_root = tmp_path / "authority"
    repo_root.mkdir(mode=0o700)
    receipt, operator, security = _receipt()
    relative = admission.external_agent_bootstrap_admission_relative_path(
        str(receipt["source_head"])
    )
    target = authority_root / relative.name
    result = admission.publish_external_agent_bootstrap_admission(
        repo_root,
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
        authority_root=authority_root,
    )
    original = target.read_bytes()
    assert result["receipt_cid"] == receipt["receipt_cid"]

    replay = admission.publish_external_agent_bootstrap_admission(
        repo_root,
        receipt,
        trusted_operator_dids=[operator],
        trusted_security_reviewer_dids=[security],
        now_ms=NOW_MS,
        authority_root=authority_root,
    )
    assert replay["receipt_cid"] == receipt["receipt_cid"]
    assert target.read_bytes() == original

    conflicting, conflicting_operator, conflicting_security = _receipt()
    with pytest.raises(
        admission.ExternalAgentBootstrapAdmissionError,
        match="already contains different bytes",
    ):
        admission.publish_external_agent_bootstrap_admission(
            repo_root,
            conflicting,
            trusted_operator_dids=[conflicting_operator],
            trusted_security_reviewer_dids=[conflicting_security],
            now_ms=NOW_MS,
            authority_root=authority_root,
        )
    assert target.read_bytes() == original


def test_publication_rejects_outside_and_symlinked_parent(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "checkout"
    repo_root.mkdir(mode=0o700)
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
    linked = tmp_path / "authority"
    linked.symlink_to(managed, target_is_directory=True)
    with pytest.raises(OSError, match="Not a directory"):
        admission.publish_external_agent_bootstrap_admission(
            repo_root,
            receipt,
            trusted_operator_dids=[operator],
            trusted_security_reviewer_dids=[security],
            now_ms=NOW_MS,
            authority_root=linked,
        )
    assert not list(managed.iterdir())


def test_create_once_issuer_diagnoses_current_head_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    script = REPO_ROOT / "scripts/issue_eaaef_bootstrap_admission.py"
    spec = importlib.util.spec_from_file_location("eaaef_bootstrap_admission_issue", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authority_root = tmp_path / "authority"
    monkeypatch.setattr(module, "AUTHORITY_ROOT_OVERRIDE", authority_root)
    report = module.diagnose()
    assert report["process_started"] is False
    assert report["configured_board_launch"] is False
    assert report["rematerialize"] is False
    assert report.get("published") is not True
    assert report["exists"] is False
    relative = report["relative_path"]
    assert relative.endswith(f"bootstrap-admission--{report['source_head']}.json")
    assert not (authority_root / Path(relative).name).exists()
    assert "materialization_source_or_board_mismatch" in report["blockers"]
    assert report["would_publish"] is False


def _load_bootstrap_issuer():
    import importlib.util

    script = REPO_ROOT / "scripts/issue_eaaef_bootstrap_admission.py"
    spec = importlib.util.spec_from_file_location(
        "eaaef_bootstrap_admission_input_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_issuer_loads_only_exact_source_addressed_admission_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_bootstrap_issuer()
    source_head = "1" * 40
    source_tree = "2" * 40
    provider = {
        "source_head": source_head,
        "source_tree": source_tree,
        "qualification_cid": "sha256:" + "a" * 64,
    }
    quack = {
        "source": {"commit": source_head, "tree": source_tree},
        "receipt_cid": "sha256:" + "b" * 64,
    }
    route_binding = {"route": "signed-binding"}
    image = {"qualification_cid": "sha256:" + "3" * 64}
    profile = {"profile_cid": "sha256:" + "4" * 64}
    bundle = {
        "schema": issuer._INPUT_BUNDLE_SCHEMA,
        "source_head": source_head,
        "source_tree": source_tree,
        "provider_container_qualification_cid": provider["qualification_cid"],
        "quack_owner_qualification_cid": quack["receipt_cid"],
        "route_binding": route_binding,
        "image_qualification": image,
        "container_profile": profile,
    }
    bundle["input_cid"] = "sha256:" + hashlib.sha256(
        issuer._canonical_bytes(bundle)
    ).hexdigest()
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    payloads = {
        "provider_container": provider,
        "quack_owner": quack,
        "admission_inputs": bundle,
    }
    for kind, payload in payloads.items():
        path = authority / issuer._authority_artifact_name(
            kind,
            source_head,
            source_tree,
        )
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        path.chmod(0o400)
    resolved_route = object()
    calls: list[dict[str, object]] = []

    def resolve(binding, **kwargs):
        calls.append({"binding": binding, **kwargs})
        return resolved_route

    monkeypatch.setattr(issuer, "AUTHORITY_DIR", authority)
    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    monkeypatch.setattr(
        issuer.routes,
        "resolve_agent_implementation_route_binding",
        resolve,
    )
    result = issuer._load_admission_inputs(
        source_head=source_head,
        source_tree=source_tree,
        now_ms=NOW_MS,
        qualification_trust={
            "schema": issuer._QUALIFICATION_TRUST_SCHEMA,
            "trusted_provider_signer_dids": ["did:key:zProviderSigner"],
            "trusted_image_reviewer_dids": ["did:key:zImage"],
            "trusted_container_profile_reviewer_dids": ["did:key:zProfile"],
            "trusted_quack_reviewer_dids": ["did:key:zQuack"],
        },
    )

    assert result["provider_container_qualification"] == provider
    assert result["quack_owner_qualification"] == quack
    assert result["route_plan"] is resolved_route
    assert result["image_qualification"] == image
    assert result["container_profile"] == profile
    assert result["trusted_provider_signer_dids"] == (
        "did:key:zProviderSigner",
    )
    assert result["trusted_image_reviewer_dids"] == ("did:key:zImage",)
    assert result["trusted_container_profile_reviewer_dids"] == (
        "did:key:zProfile",
    )
    assert result["trusted_quack_reviewer_dids"] == ("did:key:zQuack",)
    assert calls == [
        {
            "binding": route_binding,
            "repo_root": tmp_path,
            "now_ms": NOW_MS,
            "max_age_ms": 5 * 60 * 1000,
        }
    ]

    stale_provider = {**provider, "source_tree": "9" * 40}
    provider_path = (
        authority
        / issuer._authority_artifact_name(
            "provider_container",
            source_head,
            source_tree,
        )
    )
    provider_path.chmod(0o600)
    provider_path.write_text(json.dumps(stale_provider), encoding="utf-8")
    provider_path.chmod(0o400)
    with pytest.raises(RuntimeError, match="qualifications are stale"):
        issuer._load_admission_inputs(
            source_head=source_head,
            source_tree=source_tree,
            now_ms=NOW_MS,
            qualification_trust={
                "schema": issuer._QUALIFICATION_TRUST_SCHEMA,
                "trusted_provider_signer_dids": ["did:key:zProviderSigner"],
                "trusted_image_reviewer_dids": ["did:key:zImage"],
                "trusted_container_profile_reviewer_dids": ["did:key:zProfile"],
                "trusted_quack_reviewer_dids": ["did:key:zQuack"],
            },
        )


def test_issuer_requires_pairwise_distinct_explicit_qualification_trust() -> None:
    issuer = _load_bootstrap_issuer()
    with pytest.raises(RuntimeError, match="trust is unavailable"):
        issuer._explicit_reviewer_trust({})
    with pytest.raises(RuntimeError, match="trust roles overlap"):
        issuer._explicit_reviewer_trust(
            {
                "schema": issuer._QUALIFICATION_TRUST_SCHEMA,
                "trusted_provider_signer_dids": ["did:key:zProvider"],
                "trusted_image_reviewer_dids": ["did:key:zShared"],
                "trusted_container_profile_reviewer_dids": ["did:key:zShared"],
                "trusted_quack_reviewer_dids": ["did:key:zQuack"],
            }
        )

    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/external_agent_autonomous_execution_fabric_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(RuntimeError, match="trust is unavailable"):
        issuer._explicit_reviewer_trust(
            scheduler["bootstrap_qualification_trust"]
        )


def test_issuer_resolves_configured_and_valid_recovery_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_bootstrap_issuer()
    config = {
        "bootstrap_database_program": {
            "store_generation": "eaaef-run-v7",
            "runtime_registry_path": (
                "data/agent_supervisor/"
                "external_agent_autonomous_execution_fabric/run-v7/registry"
            ),
        }
    }
    cursor = tmp_path / "generation-cursor.json"
    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    monkeypatch.setattr(issuer, "CURSOR_PATH", cursor)

    assert issuer._receipt_path(config) == (
        tmp_path
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "run-v7/registry/bootstrap-materialization.json"
    )
    cursor.write_text(
        json.dumps(
            {
                "schema": issuer._GENERATION_CURSOR_SCHEMA,
                "configured_generation": "eaaef-run-v7",
                "active_generation": "eaaef-run-v9",
                "process_started": False,
            }
        ),
        encoding="utf-8",
    )
    assert issuer._receipt_path(config) == (
        tmp_path
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "run-v9/registry/bootstrap-materialization.json"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "wrong"},
        {"configured_generation": "eaaef-run-v8"},
        {"active_generation": "eaaef-run-v8/../../outside"},
        {"active_generation": "eaaef-run-v6"},
        {"process_started": True},
    ],
)
def test_issuer_rejects_invalid_generation_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    issuer = _load_bootstrap_issuer()
    config = {
        "bootstrap_database_program": {
            "store_generation": "eaaef-run-v7",
            "runtime_registry_path": (
                "data/agent_supervisor/"
                "external_agent_autonomous_execution_fabric/run-v7/registry"
            ),
        }
    }
    cursor = tmp_path / "generation-cursor.json"
    payload = {
        "schema": issuer._GENERATION_CURSOR_SCHEMA,
        "configured_generation": "eaaef-run-v7",
        "active_generation": "eaaef-run-v8",
        "process_started": False,
        **overrides,
    }
    cursor.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    monkeypatch.setattr(issuer, "CURSOR_PATH", cursor)

    with pytest.raises(RuntimeError, match="generation cursor is invalid"):
        issuer._receipt_path(config)


def test_issuer_input_failure_never_publishes_create_once_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_bootstrap_issuer()
    source_head = "1" * 40
    source_tree = "2" * 40
    publication_attempts: list[str] = []
    config = {
        "configured_board_live_seal": {
            "trusted_operator_dids": ["did:key:zOperator"],
            "trusted_security_reviewer_dids": ["did:key:zSecurity"],
        },
        "bootstrap_qualification_trust": {},
    }
    monkeypatch.setattr(
        issuer,
        "diagnose",
        lambda: {
            "source_head": source_head,
            "source_tree": source_tree,
            "identity_blockers": [],
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        issuer,
        "_load",
        lambda path: config if path == issuer.CONFIG_PATH else {},
    )
    monkeypatch.setattr(issuer, "_receipt_path", lambda: Path("receipt.json"))
    monkeypatch.setattr(
        issuer,
        "_load_admission_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("qualification trust unavailable")
        ),
    )
    monkeypatch.setattr(
        issuer,
        "publish_external_agent_bootstrap_admission",
        lambda *_args, **_kwargs: publication_attempts.append("published"),
    )

    result = issuer.issue()

    assert result["published"] is False
    assert result["would_publish"] is False
    assert "qualification trust unavailable" in result["blockers"]
    assert publication_attempts == []


def test_issuer_prepares_admitted_statement_but_requires_separate_approvals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_bootstrap_issuer()
    source_head = "1" * 40
    source_tree = "2" * 40
    operator_did = "did:key:zOperator"
    security_did = "did:key:zSecurity"
    verified_inputs = {
        "provider_container_qualification": {"provider": "qualification"},
        "route_plan": object(),
        "image_qualification": {"image": "qualification"},
        "container_profile": {"container": "profile"},
        "quack_owner_qualification": {"quack": "qualification"},
        "trusted_provider_signer_dids": (operator_did,),
        "trusted_image_reviewer_dids": (security_did,),
        "trusted_container_profile_reviewer_dids": ("did:key:zProfile",),
        "trusted_quack_reviewer_dids": (security_did,),
    }
    captured: dict[str, object] = {}
    config = {
        "configured_board_live_seal": {
            "trusted_operator_dids": [operator_did],
            "trusted_security_reviewer_dids": [security_did],
        }
    }

    monkeypatch.setattr(
        issuer,
        "diagnose",
        lambda: {
            "source_head": source_head,
            "source_tree": source_tree,
            "identity_blockers": [],
            "blockers": [],
        },
    )
    monkeypatch.setattr(issuer, "_load", lambda path: config if path == issuer.CONFIG_PATH else {})
    monkeypatch.setattr(issuer, "_receipt_path", lambda: Path("receipt.json"))
    monkeypatch.setattr(
        issuer,
        "_load_admission_inputs",
        lambda **_kwargs: verified_inputs,
    )
    monkeypatch.setattr(issuer, "_principal_did", lambda role: f"did:key:z{role}")

    def prepare(**kwargs):
        captured.update(kwargs)
        return {"decision": "admitted"}

    monkeypatch.setattr(issuer, "prepare_external_agent_bootstrap_admission", prepare)
    publication_attempts: list[str] = []
    monkeypatch.setattr(
        issuer,
        "publish_external_agent_bootstrap_admission",
        lambda *_args, **_kwargs: publication_attempts.append("published"),
    )

    result = issuer.issue()

    assert result["published"] is False
    assert result["statement_decision"] == "admitted"
    assert result["prepared_statement"] == {"decision": "admitted"}
    assert (
        "separate operator and security approval artifacts are required"
        in result["blockers"]
    )
    assert publication_attempts == []
    for name, value in verified_inputs.items():
        assert captured[name] is value or captured[name] == value


def test_issuer_reuses_exact_prepared_statement_for_external_review_and_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_bootstrap_issuer()
    source_head = "1" * 40
    source_tree = "2" * 40
    operator_key = Ed25519PrivateKey.generate()
    security_key = Ed25519PrivateKey.generate()
    operator_did = ed25519_did_key(operator_key.public_key())
    security_did = ed25519_did_key(security_key.public_key())
    config = {
        "configured_board_live_seal": {
            "trusted_operator_dids": [operator_did],
            "trusted_security_reviewer_dids": [security_did],
        },
        "bootstrap_qualification_trust": {},
    }
    verified_inputs = {
        "provider_container_qualification": {},
        "route_plan": object(),
        "image_qualification": {},
        "container_profile": {},
        "quack_owner_qualification": {},
        "trusted_provider_signer_dids": ("did:key:zProviderReviewer",),
        "trusted_image_reviewer_dids": ("did:key:zImageReviewer",),
        "trusted_container_profile_reviewer_dids": (
            "did:key:zProfileReviewer",
        ),
        "trusted_quack_reviewer_dids": ("did:key:zQuackReviewer",),
    }
    monkeypatch.setattr(
        issuer,
        "diagnose",
        lambda: {
            "source_head": source_head,
            "source_tree": source_tree,
            "identity_blockers": [],
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        issuer,
        "_load",
        lambda path: config if path == issuer.CONFIG_PATH else {},
    )
    monkeypatch.setattr(issuer, "_receipt_path", lambda: Path("receipt.json"))
    monkeypatch.setattr(
        issuer,
        "_load_admission_inputs",
        lambda **_kwargs: verified_inputs,
    )
    monkeypatch.setattr(
        issuer,
        "_principal_did",
        lambda role: f"did:key:zRuntime{role}",
    )
    monkeypatch.setattr(issuer.time, "time", lambda: NOW_MS / 1000)
    monkeypatch.setattr(
        issuer,
        "_admission_expiry_ms",
        lambda **_kwargs: NOW_MS + 100_000,
    )
    preparation_calls: list[tuple[str, int, int]] = []
    drift = {"enabled": False}

    def prepare(**kwargs):
        preparation_calls.append(
            (
                kwargs["one_use_nonce"],
                kwargs["issued_at_ms"],
                kwargs["expires_at_ms"],
            )
        )
        value = _statement()
        value.update(
            {
                "source_head": kwargs["expected_source_commit"],
                "source_tree": kwargs["expected_source_tree"],
                "one_use_nonce": kwargs["one_use_nonce"],
                "issued_at_ms": kwargs["issued_at_ms"],
                "expires_at_ms": kwargs["expires_at_ms"],
                "provider_qualification_expires_at_ms": NOW_MS + 100_000,
                "quack_qualification_expires_at_ms": NOW_MS + 100_000,
                "provider_workload_class": (
                    "drifted" if drift["enabled"] else "agent_worker"
                ),
            }
        )
        value["statement_cid"] = admission._cid(
            {key: item for key, item in value.items() if key != "statement_cid"}
        )
        return value

    monkeypatch.setattr(
        issuer,
        "prepare_external_agent_bootstrap_admission",
        prepare,
    )
    repo_root = tmp_path / "checkout"
    authority_root = tmp_path / "authority"
    repo_root.mkdir(mode=0o700)
    monkeypatch.setattr(issuer, "ROOT", repo_root)
    monkeypatch.setattr(issuer, "AUTHORITY_ROOT_OVERRIDE", authority_root)

    prepared_result = issuer.issue()
    statement = prepared_result["prepared_statement"]
    assert prepared_result["published"] is False
    assert statement["decision"] == "admitted"

    operator = admission.prepare_external_agent_bootstrap_approval(
        statement,
        role="independent_operator",
        identity_did=operator_did,
        issued_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 50_000,
    )
    operator["signature"] = base64.b64encode(
        operator_key.sign(admission._canonical_bytes(operator))
    ).decode("ascii")
    security = admission.prepare_external_agent_bootstrap_approval(
        statement,
        role="independent_security_reviewer",
        identity_did=security_did,
        issued_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 50_000,
    )
    security["signature"] = base64.b64encode(
        security_key.sign(admission._canonical_bytes(security))
    ).decode("ascii")
    relative = admission.external_agent_bootstrap_admission_relative_path(source_head)
    target = authority_root / relative.name

    published = issuer.issue(
        prepared_statement=statement,
        operator_approval=operator,
        security_approval=security,
    )

    assert published["published"] is True
    assert target.is_file()
    assert preparation_calls[0] == preparation_calls[1]

    replayed = issuer.issue(
        prepared_statement=statement,
        operator_approval=operator,
        security_approval=security,
    )
    assert replayed["published"] is True
    assert replayed["receipt_cid"] == published["receipt_cid"]
    assert target.is_file()

    drift["enabled"] = True
    rejected = issuer.issue(prepared_statement=statement)
    assert rejected["published"] is False
    assert (
        "prepared bootstrap admission statement differs from current inputs"
        in rejected["blockers"]
    )
