"""Tests for the IPFS Kit VFS assurance locked profile (LPR-027)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout import (
    AdversarialInjection,
    AssuranceRolloutMode,
    GateKind,
    SymbolicAssurancePublicAPI,
    build_frozen_adversarial_population,
    evaluate_adversarial_gates,
    evaluate_symbolic_assurance_rollout,
    freeze_multi_repository_fixture,
    project_bounded_findings,
    project_bounded_receipts,
    project_bounded_status,
    run_symbolic_assurance_e2e,
    verify_adversarial_e2e_report,
    verify_symbolic_assurance_rollout,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_kit_vfs_assurance import (
    CLOSED_ADAPTERS,
    CONFIG_SCHEMA,
    IpfsKitVfsAssuranceError,
    build_ipfs_kit_vfs_assurance_profile,
    lazy_import_adapter,
    load_assurance_config,
    optional_providers_loaded,
    resolve_safe_root,
    run_contracts,
    run_rollout,
    run_verify,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "ipfs_kit_vfs_symbolic_assurance.json"

ORIGINAL_SCHEMAS = {
    "adversarial_e2e_gate": "vfs/adversarial-e2e-gate@1",
    "shadow_rollout_report": "vfs/shadow-rollout-report@1",
    "rollout_decision": "vfs/symbolic-rollout-decision@1",
    "control_request": "vfs/symbolic-control-request@1",
    "control_result": "vfs/symbolic-control-result@1",
    "bounded_status": "vfs/symbolic-bounded-status@1",
    "bounded_findings": "vfs/symbolic-bounded-findings@1",
    "bounded_receipts": "vfs/symbolic-bounded-receipts@1",
    "public_api": "vfs/symbolic-public-api@1",
}

ORIGINAL_IDS = {
    "behavior_id": "behavior:vfs-symbolic-assurance-rollout@1",
    "objective_id": "VFS-G130",
    "objective_revision": "VFS-G130@vfs-036",
    "requirement_id": "vfs-036:adversarial-e2e-control-parity-recovery-rollback",
}


def test_config_is_immutable_bounded_and_content_identified():
    config = load_assurance_config(CONFIG_PATH)
    assert config.raw["schema"] == CONFIG_SCHEMA
    assert config.content_id.startswith("sha256:")
    assert config.content_id == config.raw["content_id"]
    assert set(config.adapters) == CLOSED_ADAPTERS
    assert all(spec.lazy for spec in config.adapters.values())
    assert all(not spec.optional_provider for spec in config.adapters.values())
    assert config.safe_relative_roots
    assert all(
        not root.startswith("/") and ".." not in Path(root).parts
        for root in config.safe_relative_roots
    )
    assert config.profile.automatic_mutation_enabled is False
    assert config.profile.default_mode is AssuranceRolloutMode.SHADOW
    assert not any(config.authority_flags.values())


def test_build_ipfs_kit_vfs_assurance_profile_preserves_original_identity():
    profile = build_ipfs_kit_vfs_assurance_profile(CONFIG_PATH)
    assert profile.behavior_id == ORIGINAL_IDS["behavior_id"]
    assert profile.objective_id == ORIGINAL_IDS["objective_id"]
    assert profile.objective_revision == ORIGINAL_IDS["objective_revision"]
    assert profile.requirement_id == ORIGINAL_IDS["requirement_id"]
    for key, value in ORIGINAL_SCHEMAS.items():
        assert getattr(profile.schemas, key) == value
    assert profile.gate_by_kind(GateKind.SEEDED_DRIFT).gate_id == "vfs_seeded_drift"
    assert (
        profile.gate_by_kind(GateKind.SEEDED_DRIFT).expected_outcome
        == "seeded-vfs-drift-detected"
    )


def test_locked_profile_preserves_operation_invariant_error_mappings():
    config = load_assurance_config(CONFIG_PATH)
    mappings = config.operation_invariant_error_mappings
    assert "read" in mappings["operations"]
    assert "write" in mappings["operations"]
    assert "path-normalized" in mappings["invariants"]
    assert "not-found" in mappings["error_codes"]
    assert any(v["vector_id"] == "read-empty" for v in mappings["canonical_vectors"])
    projected = run_contracts(config=config)
    assert projected["operations"] == list(mappings["operations"])
    assert projected["invariants"] == list(mappings["invariants"])
    assert projected["error_codes"] == list(mappings["error_codes"])
    assert projected["canonical_vectors"] == list(mappings["canonical_vectors"])
    assert projected["authority_flags"] == dict(config.authority_flags)


def test_vfs_profile_rollout_preserves_schemas_projections_and_authority():
    config = load_assurance_config(CONFIG_PATH)
    profile = config.profile
    fixture, report, binding, policy = build_frozen_adversarial_population(
        profile=profile
    )
    assert report.passed
    assert report.to_dict()["schema"] == ORIGINAL_SCHEMAS["adversarial_e2e_gate"]
    assert report.to_dict()["objective_id"] == ORIGINAL_IDS["objective_id"]
    assert report.to_dict()["requirement_id"] == ORIGINAL_IDS["requirement_id"]
    assert {item.gate_id for item in report.observations} == set(
        profile.required_gate_ids
    )
    assert "vfs_seeded_drift" in profile.required_gate_ids
    assert verify_adversarial_e2e_report(report)

    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    assert decision.effective_mode is AssuranceRolloutMode.ASSIST
    assert decision.to_dict()["schema"] == ORIGINAL_SCHEMAS["rollout_decision"]
    assert not decision.automatic_mutation_enabled
    assert not decision.authoritative
    assert not decision.completion_authoritative
    assert verify_symbolic_assurance_rollout(
        decision, report, binding=binding, policy=policy
    )

    status = project_bounded_status(decision)
    findings = project_bounded_findings(decision)
    receipts = project_bounded_receipts(decision)
    assert status["schema"] == ORIGINAL_SCHEMAS["bounded_status"]
    assert findings["schema"] == ORIGINAL_SCHEMAS["bounded_findings"]
    assert receipts["schema"] == ORIGINAL_SCHEMAS["bounded_receipts"]
    assert status["behavior_id"] == ORIGINAL_IDS["behavior_id"]
    assert status["automatic_mutation_enabled"] is False
    assert findings["finding_count"] == 0
    assert receipts["receipt_count"] >= 4


def test_vfs_fixture_population_matches_locked_default_repositories():
    config = load_assurance_config(CONFIG_PATH)
    fixture = freeze_multi_repository_fixture(profile=config.profile)
    ids = set(fixture.repository_ids)
    assert "repository:swissknife@fixture" in ids
    assert "repository:ipfs-accelerate-py@fixture" in ids
    assert "repository:ipfs-kit-py@fixture" in ids
    assert "repository:ipfs-datasets-py@fixture" in ids
    assert fixture.fixture_id == "fixture:vfs-adversarial-e2e@1"
    assert fixture.total_excluded_paths > 0


def test_public_api_discovery_uses_vfs_profile():
    profile = build_ipfs_kit_vfs_assurance_profile(CONFIG_PATH)
    discovery = SymbolicAssurancePublicAPI.discovery(profile)
    assert discovery["schema"] == ORIGINAL_SCHEMAS["public_api"]
    assert discovery["behavior_id"] == ORIGINAL_IDS["behavior_id"]
    assert discovery["objective_id"] == ORIGINAL_IDS["objective_id"]
    assert discovery["requirement_id"] == ORIGINAL_IDS["requirement_id"]
    assert ORIGINAL_SCHEMAS["adversarial_e2e_gate"] in discovery["evidence_schemas"]
    assert "vfs_seeded_drift" in discovery["required_gates"]
    assert discovery["automatic_mutation_enabled"] is False
    assert discovery["optional_providers_loaded"] is False


def test_seeded_drift_injection_targets_vfs_gate_id():
    config = load_assurance_config(CONFIG_PATH)
    fixture = freeze_multi_repository_fixture(profile=config.profile)
    report = evaluate_adversarial_gates(
        fixture,
        profile=config.profile,
        injection=AdversarialInjection(miss_seeded_drift=True),
    )
    assert not report.passed
    failed = {item.gate_id for item in report.observations if not item.passed}
    assert failed == {"vfs_seeded_drift"}


def test_run_rollout_and_verify_through_integration():
    config = load_assurance_config(CONFIG_PATH)
    payload = run_rollout(config=config, desired_mode="assist")
    assert payload["adversarial_e2e_gate"]["schema"] == ORIGINAL_SCHEMAS["adversarial_e2e_gate"]
    assert payload["shadow_rollout_report"]["schema"] == ORIGINAL_SCHEMAS["shadow_rollout_report"]
    assert payload["decision"]["effective_mode"] == "assist"
    assert payload["automatic_mutation_enabled"] is False
    verified = run_verify(config=config)
    assert verified["verified"] is True
    assert verified["effective_mode"] == "shadow"
    assert verified["status"]["schema"] == ORIGINAL_SCHEMAS["bounded_status"]


def test_lazy_adapters_do_not_load_optional_providers():
    config = load_assurance_config(CONFIG_PATH)
    before = optional_providers_loaded()
    for name in ("rollout", "verify", "contracts", "differential", "parity", "benchmark", "pilot"):
        factory = lazy_import_adapter(name, config=config)
        assert callable(factory) or factory is not None
    assert optional_providers_loaded() == before


def test_unsafe_roots_are_rejected():
    config = load_assurance_config(CONFIG_PATH)
    with pytest.raises(IpfsKitVfsAssuranceError):
        resolve_safe_root("../etc", config=config)
    with pytest.raises(IpfsKitVfsAssuranceError):
        resolve_safe_root("/tmp", config=config)
    resolved = resolve_safe_root("config", config=config, checkout_root=REPO_ROOT)
    assert resolved == (REPO_ROOT / "config").resolve()


def test_config_rejects_open_adapter_registry(tmp_path: Path):
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["adapter_registry"]["extra"] = {
        "module": "os",
        "factory": "system",
        "lazy": True,
        "optional_provider": False,
    }
    raw.pop("content_id", None)
    path = tmp_path / "bad.json"
    # Write without valid content_id so either registry or content_id fails.
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(IpfsKitVfsAssuranceError):
        load_assurance_config(path)
