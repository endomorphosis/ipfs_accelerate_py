"""Adversarial end-to-end assurance, control parity, recovery, and rollback.

Covers VFS-036 / VFS-G130: frozen multi-repository CIDs, inventory, cache,
proof/ZK, MCP, VFS drift, security false positives, task determinism, provider
loss, restart/replay, lease/fence, merge, refill, rollback, and Python/CLI/MCP
parity without provider imports or process starts during discovery.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_symbolic_rollout import (
    ADVERSARIAL_E2E_GATE_SCHEMA,
    REQUIRED_ADVERSARIAL_GATES,
    SHADOW_ROLLOUT_REPORT_SCHEMA,
    AdversarialGateId,
    AdversarialInjection,
    GateStatus,
    VFS_SYMBOLIC_BEHAVIOR_ID,
    VFS_SYMBOLIC_OBJECTIVE_ID,
    VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
    VfsControlRequest,
    VfsRolloutMode,
    VfsSymbolicPublicAPI,
    VfsSymbolicRolloutError,
    build_default_vfs_binding,
    build_default_vfs_policy,
    build_frozen_adversarial_population,
    evaluate_adversarial_gates,
    evaluate_vfs_symbolic_rollout,
    freeze_multi_repository_fixture,
    project_bounded_findings,
    project_bounded_receipts,
    project_bounded_status,
    run_vfs_symbolic_assurance_e2e,
    verify_adversarial_e2e_report,
    verify_vfs_symbolic_rollout,
)


def _population(**kwargs):
    return build_frozen_adversarial_population(**kwargs)


def test_freeze_multi_repository_fixture_is_reproducible_across_independent_freezes():
    first = freeze_multi_repository_fixture()
    second = freeze_multi_repository_fixture()
    assert first.fixture_cid == second.fixture_cid
    assert first.forest_id == second.forest_id
    assert len(first.repositories) == 4
    assert first.total_included_paths > 0
    assert first.total_excluded_paths > 0
    for left, right in zip(first.repositories, second.repositories, strict=True):
        assert left.content_cid == right.content_cid
        assert left.tree_id == right.tree_id
        assert left.included_paths == right.included_paths
        assert left.excluded_paths == right.excluded_paths
        assert set(left.included_paths).isdisjoint(left.excluded_paths)


def test_adversarial_e2e_gate_covers_every_required_case_and_passes_cleanly():
    fixture, report, binding, policy = _population()
    assert report.passed
    assert report.to_dict()["schema"] == ADVERSARIAL_E2E_GATE_SCHEMA
    assert not report.automatic_mutation_enabled
    assert {item.gate_id for item in report.observations} == set(
        REQUIRED_ADVERSARIAL_GATES
    )
    assert len(report.observations) == len(REQUIRED_ADVERSARIAL_GATES)
    assert all(item.status is GateStatus.PASSED for item in report.observations)
    assert all(not item.authoritative for item in report.observations)
    assert verify_adversarial_e2e_report(report)

    decision = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.SHADOW,
    )
    assert decision.effective_mode is VfsRolloutMode.SHADOW
    assert not decision.automatic_mutation_enabled
    assert not decision.authoritative
    assert not decision.completion_authoritative
    assert verify_vfs_symbolic_rollout(
        decision, report, binding=binding, policy=policy
    )


def test_complete_inventory_and_exclusions_are_policy_bound():
    fixture = freeze_multi_repository_fixture()
    for repo in fixture.repositories:
        assert repo.exhaustive
        for path in repo.excluded_paths:
            assert any(
                path == prefix.rstrip("/") or path.startswith(prefix)
                for prefix in fixture.exclusion_prefixes
            )
        for path in repo.included_paths:
            assert not any(
                path == prefix.rstrip("/") or path.startswith(prefix)
                for prefix in fixture.exclusion_prefixes
            )
    report = evaluate_adversarial_gates(fixture)
    assert report.observation(AdversarialGateId.COMPLETE_INVENTORY).passed
    assert report.observation(AdversarialGateId.INVENTORY_EXCLUSIONS).passed
    assert report.observation(AdversarialGateId.INCREMENTAL_REUSE).passed


@pytest.mark.parametrize(
    ("flag", "gate_id"),
    [
        ("allow_stale_cache_hit", AdversarialGateId.STALE_CACHE_REJECTION),
        ("allow_corrupt_cache_hit", AdversarialGateId.CORRUPT_CACHE_REJECTION),
        ("wrong_contract_match", AdversarialGateId.CONTRACT_PRECISION),
        ("accept_wrong_proof", AdversarialGateId.WRONG_PROOF),
        ("promote_unknown_proof", AdversarialGateId.UNKNOWN_PROOF),
        ("accept_mcp_mock", AdversarialGateId.MCP_MOCK),
        ("accept_mcp_bypass", AdversarialGateId.MCP_BYPASS),
        ("miss_seeded_drift", AdversarialGateId.VFS_SEEDED_DRIFT),
        (
            "emit_vulnerability_false_positive",
            AdversarialGateId.VULNERABILITY_FALSE_POSITIVE,
        ),
        ("nondeterministic_tasks", AdversarialGateId.TASK_DETERMINISM),
        ("expand_authority_on_provider_loss", AdversarialGateId.PROVIDER_LOSS),
        ("restart_diverges", AdversarialGateId.RESTART_REPLAY),
        ("ignore_lease_fence", AdversarialGateId.LEASE_FENCE_LOSS),
        ("silent_merge_conflict", AdversarialGateId.MERGE_CONFLICT),
        ("unbounded_refill", AdversarialGateId.BOUNDED_REFILL),
        ("refill_busywork_after_exhaustion", AdversarialGateId.REFILL_EXHAUSTION),
        ("skip_rollback", AdversarialGateId.ROLLBACK),
        ("control_surface_divergence", AdversarialGateId.CONTROL_PARITY),
        ("corrupt_second_cid_pass", AdversarialGateId.REPRODUCIBLE_CIDS),
        ("omit_exclusion", AdversarialGateId.INVENTORY_EXCLUSIONS),
    ],
)
def test_each_adversarial_injection_fails_exactly_the_targeted_gate(flag, gate_id):
    injection = AdversarialInjection(**{flag: True})
    fixture = freeze_multi_repository_fixture()
    report = evaluate_adversarial_gates(fixture, injection=injection)
    assert not report.passed
    failed = {item.gate_id for item in report.observations if not item.passed}
    assert gate_id in failed
    # Targeted failure should not silently pass the automatic-mutation gate.
    assert report.observation(
        AdversarialGateId.AUTOMATIC_MUTATION_DISABLED
    ).passed


def test_simulated_forged_and_tampered_zk_cannot_become_authoritative():
    fixture = freeze_multi_repository_fixture()
    report = evaluate_adversarial_gates(fixture)
    for gate_id in (
        AdversarialGateId.SIMULATED_ZK,
        AdversarialGateId.FORGED_ZK,
        AdversarialGateId.TAMPERED_ZK,
    ):
        observation = report.observation(gate_id)
        assert observation.passed
        assert not observation.authoritative
    with pytest.raises(VfsSymbolicRolloutError):
        evaluate_adversarial_gates(
            fixture, injection=AdversarialInjection(force_authoritative_zk=True)
        )
    with pytest.raises(VfsSymbolicRolloutError):
        evaluate_adversarial_gates(
            fixture,
            injection=AdversarialInjection(force_automatic_mutation=True),
        )


def test_assist_promotes_only_when_all_gates_pass_and_automatic_stays_shadow():
    fixture, report, binding, policy = _population()
    assist = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.ASSIST,
    )
    assert assist.effective_mode is VfsRolloutMode.ASSIST
    assert not assist.rollback_applied
    assert not assist.automatic_mutation_enabled

    automatic = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.AUTOMATIC,
    )
    assert automatic.effective_mode is VfsRolloutMode.SHADOW
    assert automatic.rollback_applied
    assert "automatic-mutation-disabled" in automatic.reason_codes

    # Even with automatic approved in policy, mutation stays disabled and
    # effective mode cannot become automatic.
    auto_policy = build_default_vfs_policy(
        approve_assist=True, approve_automatic=True
    )
    still_shadow = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=auto_policy,
        desired_mode=VfsRolloutMode.AUTOMATIC,
    )
    assert still_shadow.effective_mode is VfsRolloutMode.SHADOW
    assert still_shadow.rollback_applied


def test_regression_returns_effective_rollout_to_shadow():
    fixture, clean, binding, policy = _population(
        observed_at="2026-07-29T00:00:00Z"
    )
    prior = evaluate_vfs_symbolic_rollout(
        clean,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.ASSIST,
    )
    assert prior.effective_mode is VfsRolloutMode.ASSIST

    regressed = evaluate_adversarial_gates(
        fixture,
        injection=AdversarialInjection(accept_wrong_proof=True),
        observed_at="2026-07-29T01:00:00Z",
    )
    decision = evaluate_vfs_symbolic_rollout(
        regressed,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.ASSIST,
        prior_gate_report=clean,
    )
    assert decision.effective_mode is VfsRolloutMode.SHADOW
    assert decision.rollback_applied
    assert "assurance-regression" in decision.reason_codes
    assert any(
        code.startswith("gate-failed:wrong_proof") for code in decision.reason_codes
    )


def test_stale_binding_or_policy_mismatch_returns_to_shadow():
    fixture, report, binding, policy = _population()
    stale = replace(binding, forest_id="sha256:" + "0" * 64)
    decision = evaluate_vfs_symbolic_rollout(
        report,
        binding=stale,
        policy=policy,
        desired_mode=VfsRolloutMode.ASSIST,
    )
    assert decision.effective_mode is VfsRolloutMode.SHADOW
    assert "stale-binding:forest_id" in decision.reason_codes

    foreign_policy = replace(
        policy,
        policy_id="policy:other@1",
        policy_revision="sha256:other",
    )
    blocked = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=foreign_policy,
        desired_mode=VfsRolloutMode.ASSIST,
    )
    assert blocked.effective_mode is VfsRolloutMode.SHADOW
    assert "stale-binding:rollout-policy" in blocked.reason_codes


def test_python_cli_and_mcp_publish_equivalent_bounded_status_findings_receipts():
    fixture, report, binding, policy = _population()
    request = VfsControlRequest(action="assist")
    results = []
    for adapter in ("python", "cli", "mcp"):
        api = VfsSymbolicPublicAPI(
            report, binding=binding, policy=policy, initial_mode="shadow"
        )
        result = getattr(api, adapter)(request.to_dict())
        results.append(result.to_dict())

    assert results[0] == results[1] == results[2]
    assert results[0]["decision"]["effective_mode"] == "assist"
    assert results[0]["decision"]["requirement_id"] == (
        VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID
    )
    assert results[0]["status"]["schema"].endswith("bounded-status@1")
    assert results[0]["findings"]["schema"].endswith("bounded-findings@1")
    assert results[0]["receipts"]["schema"].endswith("bounded-receipts@1")
    assert results[0]["status"]["automatic_mutation_enabled"] is False
    assert results[0]["receipts"]["receipt_count"] >= 4

    # Projections are canonically identical when rebuilt from the decision.
    api = VfsSymbolicPublicAPI(report, binding=binding, policy=policy)
    api.execute("assist")
    decision = api.decision
    assert project_bounded_status(decision) == api.status().status
    assert project_bounded_findings(decision)["finding_count"] == 0
    assert project_bounded_receipts(decision)["receipt_count"] >= 4


def test_public_discovery_is_lazy_and_provider_free():
    discovery = VfsSymbolicPublicAPI.discovery()
    assert set(discovery["surfaces"]) == {"python", "cli", "mcp"}
    assert set(discovery["actions"]) == {
        "off",
        "shadow",
        "assist",
        "automatic",
        "status",
        "findings",
        "receipts",
        "explanation",
        "rollback",
    }
    assert discovery["behavior_id"] == VFS_SYMBOLIC_BEHAVIOR_ID
    assert discovery["objective_id"] == VFS_SYMBOLIC_OBJECTIVE_ID
    assert discovery["requirement_id"] == VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID
    assert discovery["automatic_mutation_enabled"] is False
    assert discovery["optional_providers_loaded"] is False
    assert discovery["processes_started"] is False
    assert ADVERSARIAL_E2E_GATE_SCHEMA in discovery["evidence_schemas"]
    assert SHADOW_ROLLOUT_REPORT_SCHEMA in discovery["evidence_schemas"]
    assert set(discovery["required_gates"]) == {
        item.value for item in REQUIRED_ADVERSARIAL_GATES
    }

    script = """
import json, sys
forbidden = ("torch", "transformers", "openai", "neo4j", "duckdb", "anthropic")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}
from ipfs_accelerate_py.agent_supervisor.vfs_symbolic_rollout import (
    VfsSymbolicPublicAPI,
    freeze_multi_repository_fixture,
    evaluate_adversarial_gates,
    run_vfs_symbolic_assurance_e2e,
)
after = {name for name in sys.modules if name.split(".")[0] in forbidden}
fixture = freeze_multi_repository_fixture()
report = evaluate_adversarial_gates(fixture)
payload = run_vfs_symbolic_assurance_e2e(desired_mode="shadow")
print(json.dumps({
    "discovery": VfsSymbolicPublicAPI.discovery(),
    "added": sorted(after - before),
    "passed": report.passed,
    "fixture_cid": fixture.fixture_cid,
    "e2e_passed": payload["adversarial_e2e_gate"]["passed"],
    "auto": payload["automatic_mutation_enabled"],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert not payload["discovery"]["optional_providers_loaded"]
    assert not payload["discovery"]["processes_started"]
    assert payload["passed"]
    assert payload["e2e_passed"]
    assert payload["auto"] is False


def test_control_status_findings_receipts_and_rollback_paths():
    fixture, report, binding, policy = _population()
    api = VfsSymbolicPublicAPI(report, binding=binding, policy=policy)
    assert api.status().decision.effective_mode is VfsRolloutMode.SHADOW
    promoted = api.execute("assist")
    assert promoted.decision.effective_mode is VfsRolloutMode.ASSIST
    assert promoted.changed

    findings = api.findings()
    assert findings.findings["finding_count"] == 0
    receipts = api.receipts()
    assert any(
        item["receipt_kind"] == "adversarial-e2e-gate"
        for item in receipts.receipts["receipts"]
    )
    explanation = api.explanation()
    assert VFS_SYMBOLIC_BEHAVIOR_ID in explanation.explanation

    rolled_back = api.rollback()
    assert rolled_back.decision.effective_mode is VfsRolloutMode.SHADOW
    assert rolled_back.decision.affected_behavior_ids == (
        VFS_SYMBOLIC_BEHAVIOR_ID,
    )

    # Live re-evaluation on status after binding drift.
    api.execute("assist")
    api.binding = replace(api.binding, forest_id="sha256:" + "a" * 64)
    regressed = api.status()
    assert regressed.decision.effective_mode is VfsRolloutMode.SHADOW
    assert regressed.decision.rollback_applied


def test_failed_gates_project_into_bounded_findings():
    fixture, report, binding, policy = _population(
        injection=AdversarialInjection(
            accept_wrong_proof=True,
            miss_seeded_drift=True,
        )
    )
    decision = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.ASSIST,
    )
    findings = project_bounded_findings(decision)
    assert findings["finding_count"] == 2
    gate_ids = {item["gate_id"] for item in findings["findings"]}
    assert gate_ids == {"wrong_proof", "vfs_seeded_drift"}
    assert decision.effective_mode is VfsRolloutMode.SHADOW


def test_run_vfs_symbolic_assurance_e2e_emits_full_evidence_bundle():
    payload = run_vfs_symbolic_assurance_e2e(desired_mode="assist")
    assert payload["adversarial_e2e_gate"]["schema"] == ADVERSARIAL_E2E_GATE_SCHEMA
    assert payload["shadow_rollout_report"]["schema"] == SHADOW_ROLLOUT_REPORT_SCHEMA
    assert payload["shadow_rollout_report"]["effective_mode"] == "assist"
    assert payload["automatic_mutation_enabled"] is False
    assert payload["decision"]["effective_mode"] == "assist"
    assert payload["status"]["fixture_cid"] == payload["fixture"]["fixture_cid"]
    assert payload["receipts"]["receipt_count"] >= 4


def test_custom_multi_repo_fixture_freezes_and_excludes_correctly():
    fixture = freeze_multi_repository_fixture(
        {
            "repository:alpha": {
                "src/main.py": "print('a')\n",
                "node_modules/x/index.js": "module.exports=1\n",
            },
            "repository:beta": {
                "lib/core.py": "x=1\n",
                "__pycache__/core.pyc": "bin",
            },
        },
        fixture_id="fixture:custom-e2e@1",
        fixture_revision="rev:custom",
    )
    assert fixture.fixture_id == "fixture:custom-e2e@1"
    assert len(fixture.repositories) == 2
    alpha = next(
        item for item in fixture.repositories if item.repository_id == "repository:alpha"
    )
    assert "src/main.py" in alpha.included_paths
    assert "node_modules/x/index.js" in alpha.excluded_paths
    # Self-replay still yields CID parity for custom fixtures.
    report = evaluate_adversarial_gates(fixture)
    assert report.observation(AdversarialGateId.REPRODUCIBLE_CIDS).passed


def test_off_and_shadow_modes_never_gain_authority():
    fixture, report, binding, policy = _population()
    off = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.OFF,
    )
    assert off.effective_mode is VfsRolloutMode.OFF
    assert not off.rollback_applied

    shadow = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=VfsRolloutMode.SHADOW,
    )
    assert shadow.effective_mode is VfsRolloutMode.SHADOW
    assert not shadow.rollback_applied


def test_task_determinism_gate_is_stable_across_rebuilds():
    fixture = freeze_multi_repository_fixture()
    first = evaluate_adversarial_gates(fixture)
    second = evaluate_adversarial_gates(fixture)
    task_first = first.observation(AdversarialGateId.TASK_DETERMINISM)
    task_second = second.observation(AdversarialGateId.TASK_DETERMINISM)
    assert task_first.evidence_ids == task_second.evidence_ids
    assert task_first.observation_id == task_second.observation_id
    assert first.report_id == second.report_id
