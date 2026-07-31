"""Tests for generic symbolic assurance rollout control (LPR-027)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout import (
    DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA,
    DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA,
    AdversarialInjection,
    AssuranceRolloutMode,
    ControlRequest,
    GateKind,
    GateStatus,
    SymbolicAssurancePublicAPI,
    SymbolicAssuranceRolloutError,
    build_default_rollout_binding,
    build_default_rollout_policy,
    build_frozen_adversarial_population,
    build_generic_rollout_profile,
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

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_MODULE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "control"
    / "symbolic_assurance_rollout.py"
)

_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife|"
    r"ipfs_kit|SWISSKNIFE_ROOT|IPFS_ACCELERATE_ROOT|argparse|__main__)\b"
)


def _population(**kwargs):
    return build_frozen_adversarial_population(**kwargs)


def test_generic_module_contains_no_domain_literals() -> None:
    text = ROLLOUT_MODULE.read_text(encoding="utf-8")
    hits = _FORBIDDEN_GENERIC.findall(text)
    assert hits == [], f"generic control embeds domain literals: {hits}"


def test_freeze_multi_repository_fixture_is_reproducible() -> None:
    profile = build_generic_rollout_profile()
    first = freeze_multi_repository_fixture(profile=profile)
    second = freeze_multi_repository_fixture(profile=profile)
    assert first.fixture_cid == second.fixture_cid
    assert first.forest_id == second.forest_id
    assert len(first.repositories) == 4
    assert first.total_included_paths > 0
    assert first.total_excluded_paths > 0


def test_adversarial_gates_cover_every_required_case() -> None:
    fixture, report, binding, policy = _population()
    assert report.passed
    assert report.to_dict()["schema"] == DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA
    assert not report.automatic_mutation_enabled
    assert {item.gate_id for item in report.observations} == set(
        report.profile.required_gate_ids
    )
    assert all(item.status is GateStatus.PASSED for item in report.observations)
    assert all(not item.authoritative for item in report.observations)
    assert verify_adversarial_e2e_report(report)

    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.SHADOW,
    )
    assert decision.effective_mode is AssuranceRolloutMode.SHADOW
    assert not decision.automatic_mutation_enabled
    assert not decision.authoritative
    assert not decision.completion_authoritative
    assert verify_symbolic_assurance_rollout(
        decision, report, binding=binding, policy=policy
    )


@pytest.mark.parametrize(
    ("flag", "kind"),
    [
        ("allow_stale_cache_hit", GateKind.STALE_CACHE_REJECTION),
        ("allow_corrupt_cache_hit", GateKind.CORRUPT_CACHE_REJECTION),
        ("wrong_contract_match", GateKind.CONTRACT_PRECISION),
        ("accept_wrong_proof", GateKind.WRONG_PROOF),
        ("promote_unknown_proof", GateKind.UNKNOWN_PROOF),
        ("accept_mcp_mock", GateKind.MCP_MOCK),
        ("accept_mcp_bypass", GateKind.MCP_BYPASS),
        ("miss_seeded_drift", GateKind.SEEDED_DRIFT),
        ("emit_vulnerability_false_positive", GateKind.VULNERABILITY_FALSE_POSITIVE),
        ("nondeterministic_tasks", GateKind.TASK_DETERMINISM),
        ("expand_authority_on_provider_loss", GateKind.PROVIDER_LOSS),
        ("restart_diverges", GateKind.RESTART_REPLAY),
        ("ignore_lease_fence", GateKind.LEASE_FENCE_LOSS),
        ("silent_merge_conflict", GateKind.MERGE_CONFLICT),
        ("unbounded_refill", GateKind.BOUNDED_REFILL),
        ("refill_busywork_after_exhaustion", GateKind.REFILL_EXHAUSTION),
        ("skip_rollback", GateKind.ROLLBACK),
        ("control_surface_divergence", GateKind.CONTROL_PARITY),
        ("corrupt_second_cid_pass", GateKind.REPRODUCIBLE_CIDS),
        ("omit_exclusion", GateKind.INVENTORY_EXCLUSIONS),
    ],
)
def test_each_adversarial_injection_fails_exactly_the_targeted_gate(flag, kind):
    injection = AdversarialInjection(**{flag: True})
    profile = build_generic_rollout_profile()
    fixture = freeze_multi_repository_fixture(profile=profile)
    report = evaluate_adversarial_gates(
        fixture, profile=profile, injection=injection
    )
    assert not report.passed
    failed = {item.gate_id for item in report.observations if not item.passed}
    target = profile.gate_by_kind(kind).gate_id
    assert target in failed
    assert report.observation_by_kind(
        GateKind.AUTOMATIC_MUTATION_DISABLED
    ).passed


def test_simulated_forged_and_tampered_zk_cannot_become_authoritative():
    fixture, report, binding, policy = _population()
    for kind in (GateKind.SIMULATED_ZK, GateKind.FORGED_ZK, GateKind.TAMPERED_ZK):
        observation = report.observation_by_kind(kind)
        assert observation.passed
        assert not observation.authoritative
    with pytest.raises(SymbolicAssuranceRolloutError):
        evaluate_adversarial_gates(
            fixture, injection=AdversarialInjection(force_authoritative_zk=True)
        )
    with pytest.raises(SymbolicAssuranceRolloutError):
        evaluate_adversarial_gates(
            fixture, injection=AdversarialInjection(force_automatic_mutation=True)
        )


def test_assist_promotes_only_when_all_gates_pass_and_automatic_stays_shadow():
    fixture, report, binding, policy = _population()
    assist = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    assert assist.effective_mode is AssuranceRolloutMode.ASSIST
    assert not assist.rollback_applied
    assert not assist.automatic_mutation_enabled

    automatic = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.AUTOMATIC,
    )
    assert automatic.effective_mode is AssuranceRolloutMode.SHADOW
    assert automatic.rollback_applied
    assert "automatic-mutation-disabled" in automatic.reason_codes

    auto_policy = build_default_rollout_policy(
        profile=report.profile, approve_assist=True, approve_automatic=True
    )
    still_shadow = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=auto_policy,
        desired_mode=AssuranceRolloutMode.AUTOMATIC,
    )
    assert still_shadow.effective_mode is AssuranceRolloutMode.SHADOW
    assert still_shadow.rollback_applied


def test_regression_returns_effective_rollout_to_shadow():
    fixture, clean, binding, policy = _population(
        observed_at="2026-07-29T00:00:00Z"
    )
    prior = evaluate_symbolic_assurance_rollout(
        clean,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    assert prior.effective_mode is AssuranceRolloutMode.ASSIST

    regressed = evaluate_adversarial_gates(
        fixture,
        profile=clean.profile,
        injection=AdversarialInjection(accept_wrong_proof=True),
        observed_at="2026-07-29T01:00:00Z",
    )
    decision = evaluate_symbolic_assurance_rollout(
        regressed,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
        prior_gate_report=clean,
    )
    assert decision.effective_mode is AssuranceRolloutMode.SHADOW
    assert decision.rollback_applied
    assert "assurance-regression" in decision.reason_codes


def test_stale_binding_or_policy_mismatch_returns_to_shadow():
    fixture, report, binding, policy = _population()
    stale = replace(binding, forest_id="sha256:" + "0" * 64)
    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=stale,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    assert decision.effective_mode is AssuranceRolloutMode.SHADOW
    assert "stale-binding:forest_id" in decision.reason_codes


def test_python_cli_and_mcp_publish_equivalent_bounded_projections():
    fixture, report, binding, policy = _population()
    request = ControlRequest(
        action="assist",
        schema=report.profile.schemas.control_request,
        version=report.profile.schemas.version,
    )
    results = []
    for adapter in ("python", "cli", "mcp"):
        api = SymbolicAssurancePublicAPI(
            report, binding=binding, policy=policy, initial_mode="shadow"
        )
        result = getattr(api, adapter)(request.to_dict())
        results.append(result.to_dict())

    assert results[0] == results[1] == results[2]
    assert results[0]["decision"]["effective_mode"] == "assist"
    assert results[0]["status"]["automatic_mutation_enabled"] is False
    assert results[0]["receipts"]["receipt_count"] >= 4


def test_public_discovery_is_lazy_and_provider_free():
    discovery = SymbolicAssurancePublicAPI.discovery()
    assert set(discovery["surfaces"]) == {"python", "cli", "mcp"}
    assert discovery["automatic_mutation_enabled"] is False
    assert discovery["optional_providers_loaded"] is False
    assert discovery["processes_started"] is False
    assert DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA in discovery["evidence_schemas"]
    assert DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA in discovery["evidence_schemas"]

    script = """
import json, sys, os
os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"
forbidden = ("torch", "transformers", "openai", "neo4j", "duckdb", "anthropic")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}
from ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout import (
    SymbolicAssurancePublicAPI,
    freeze_multi_repository_fixture,
    evaluate_adversarial_gates,
    run_symbolic_assurance_e2e,
)
after = {name for name in sys.modules if name.split(".")[0] in forbidden}
fixture = freeze_multi_repository_fixture()
report = evaluate_adversarial_gates(fixture)
payload = run_symbolic_assurance_e2e(desired_mode="shadow")
print(json.dumps({
    "discovery": SymbolicAssurancePublicAPI.discovery(),
    "added": sorted(after - before),
    "passed": report.passed,
    "auto": payload["automatic_mutation_enabled"],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert not payload["discovery"]["optional_providers_loaded"]
    assert payload["passed"]
    assert payload["auto"] is False


def test_control_status_findings_receipts_and_rollback_paths():
    fixture, report, binding, policy = _population()
    api = SymbolicAssurancePublicAPI(report, binding=binding, policy=policy)
    assert api.status().decision.effective_mode is AssuranceRolloutMode.SHADOW
    promoted = api.execute("assist")
    assert promoted.decision.effective_mode is AssuranceRolloutMode.ASSIST
    assert promoted.changed
    assert api.findings().findings["finding_count"] == 0
    assert any(
        item["receipt_kind"] == "adversarial-e2e-gate"
        for item in api.receipts().receipts["receipts"]
    )
    rolled_back = api.rollback()
    assert rolled_back.decision.effective_mode is AssuranceRolloutMode.SHADOW


def test_failed_gates_project_into_bounded_findings():
    fixture, report, binding, policy = _population(
        injection=AdversarialInjection(
            accept_wrong_proof=True,
            miss_seeded_drift=True,
        )
    )
    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    findings = project_bounded_findings(decision)
    assert findings["finding_count"] == 2
    assert decision.effective_mode is AssuranceRolloutMode.SHADOW


def test_run_e2e_emits_full_evidence_bundle():
    payload = run_symbolic_assurance_e2e(desired_mode="assist")
    assert payload["adversarial_e2e_gate"]["schema"] == DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA
    assert payload["shadow_rollout_report"]["schema"] == DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA
    assert payload["shadow_rollout_report"]["effective_mode"] == "assist"
    assert payload["automatic_mutation_enabled"] is False


def test_non_vfs_profile_traverses_same_public_api():
    """Hermetic non-domain profile proves the engine is parameterized."""

    profile = build_generic_rollout_profile(
        profile_id="profile:widget-assurance@1",
        behavior_id="behavior:widget-assurance@1",
        objective_id="WIDGET-G001",
        objective_revision="WIDGET-G001@1",
        requirement_id="widget:adversarial-e2e",
        default_fixture_repositories={
            "repository:widget-a@fixture": {
                "src/widget.py": "def spin():\n    return 1\n",
                "node_modules/x/index.js": "module.exports=1\n",
            },
            "repository:widget-b@fixture": {
                "lib/gadget.py": "def tick():\n    return 2\n",
                "__pycache__/x.pyc": "bin",
            },
        },
        default_fixture_id="fixture:widget-e2e@1",
    )
    fixture, report, binding, policy = build_frozen_adversarial_population(
        profile=profile
    )
    assert report.passed
    assert report.profile.profile_id == "profile:widget-assurance@1"
    assert report.to_dict()["objective_id"] == "WIDGET-G001"
    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    assert decision.effective_mode is AssuranceRolloutMode.ASSIST
    api = SymbolicAssurancePublicAPI(
        report, binding=binding, policy=policy, initial_mode="shadow"
    )
    result = api.execute("assist")
    assert result.decision.effective_mode is AssuranceRolloutMode.ASSIST
    discovery = SymbolicAssurancePublicAPI.discovery(profile)
    assert discovery["behavior_id"] == "behavior:widget-assurance@1"
    assert discovery["objective_id"] == "WIDGET-G001"
    status = project_bounded_status(decision)
    assert status["behavior_id"] == "behavior:widget-assurance@1"
    assert project_bounded_receipts(decision)["receipt_count"] >= 4


def test_off_and_shadow_modes_never_gain_authority():
    fixture, report, binding, policy = _population()
    off = evaluate_symbolic_assurance_rollout(
        report, binding=binding, policy=policy, desired_mode=AssuranceRolloutMode.OFF
    )
    assert off.effective_mode is AssuranceRolloutMode.OFF
    shadow = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=AssuranceRolloutMode.SHADOW,
    )
    assert shadow.effective_mode is AssuranceRolloutMode.SHADOW


def test_default_mode_is_shadow_and_mutation_disabled():
    profile = build_generic_rollout_profile()
    assert profile.default_mode is AssuranceRolloutMode.SHADOW
    assert profile.automatic_mutation_enabled is False
    with pytest.raises(SymbolicAssuranceRolloutError):
        build_generic_rollout_profile(automatic_mutation_enabled=True)
