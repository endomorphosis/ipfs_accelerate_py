"""Tests for Quack security, concurrency, conflict, and restart chaos (DQP-034).

Acceptance:

* No provider/LLM process obtains token or arbitrary SQL
* Unauthorized/cross-root/file/extension queries fail before effect
* Same-row conflicts are bounded
* Stale clients cannot write after restart/rotation
* Live tests cannot silently skip when profile claims compatible

Evidence subset: authn/authz defaults, loopback, TLS boundary statement,
Python-UDF limitation, retry jitter, split brain, restore, denial, secret scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    QuackCapabilityStatus,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientSQLError,
    open_embedded_client,
)
from ipfs_accelerate_py.agent_supervisor.validation.quack_chaos import (
    DEFAULT_CLIENT_COUNT,
    GOAL_ID,
    QUACK_CHAOS_REPORT_INTERFACE,
    QUACK_SECURITY_POLICY_INTERFACE,
    REQUIRED_SCENARIOS,
    TASK_ID,
    AuthorizationAction,
    AuthorizationReason,
    ChaosScenarioId,
    ChaosVerdict,
    QuackChaosLiveGateError,
    QuackChaosReport,
    QuackChaosScenarioError,
    QuackSecurityPolicy,
    ScenarioMode,
    ScenarioOutcome,
    assert_chaos_passed,
    authorize_statement,
    default_security_policy,
    enforce_live_gate,
    prepare_chaos_database,
    profile_claims_compatible,
    run_quack_chaos_suite,
    scan_for_secrets,
    scenario_forbidden_surface,
    scenario_hot_row_conflict,
    scenario_live_gate_policy,
    scenario_loopback_bind,
    scenario_provider_env_isolation,
    scenario_raw_sql_rejection,
    scenario_retry_jitter,
    scenario_stale_after_rotation,
    scenario_token_isolation,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for Quack chaos hermetic tests",
)


# ---------------------------------------------------------------------------
# Policy / authorization unit proofs
# ---------------------------------------------------------------------------


def test_security_policy_interface_and_defaults() -> None:
    policy = default_security_policy()
    assert policy.INTERFACE == QUACK_SECURITY_POLICY_INTERFACE
    assert policy.require_loopback is True
    assert policy.provider_receives_token is False
    assert policy.provider_receives_arbitrary_sql is False
    assert policy.tls_required_for_remote is True
    assert policy.client_count == DEFAULT_CLIENT_COUNT
    assert "TLS" in policy.tls_boundary_statement
    assert "Python UDF" in policy.python_udf_limitation_statement
    payload = policy.to_dict()
    assert payload["interface"] == QUACK_SECURITY_POLICY_INTERFACE
    assert payload["provider_receives_token"] is False


def test_security_policy_rejects_provider_token_admission() -> None:
    with pytest.raises(ValueError, match="provider token"):
        QuackSecurityPolicy(provider_receives_token=True)
    with pytest.raises(ValueError, match="provider token"):
        QuackSecurityPolicy(provider_receives_arbitrary_sql=True)


def test_authorize_statement_denies_before_effect() -> None:
    samples = {
        "SELECT 1": AuthorizationReason.ARBITRARY_SQL,
        "INSTALL quack": AuthorizationReason.EXTENSION_SURFACE,
        "LOAD '/tmp/evil.duckdb_extension'": AuthorizationReason.EXTENSION_SURFACE,
        "SELECT * FROM read_csv_auto('/etc/passwd')": AuthorizationReason.CROSS_ROOT,
        "COPY tasks TO '../../../tmp/out.csv'": AuthorizationReason.CROSS_ROOT,
        "CREATE FUNCTION evil AS (x) -> python_eval(x)": AuthorizationReason.PYTHON_UDF,
        "DROP TABLE tasks": AuthorizationReason.FORBIDDEN_VERB,
        "SELECT 1; DROP TABLE tasks": AuthorizationReason.MULTI_STATEMENT,
    }
    for sql, expected in samples.items():
        decision = authorize_statement(sql)
        assert decision.action is AuthorizationAction.DENY, sql
        assert decision.effect_attempted is False, sql
        assert decision.reason is expected, (sql, decision.reason)

    # ATTACH is denied before effect via one of the closed surface reasons.
    attach = authorize_statement("ATTACH '/tmp/x.duckdb' AS evil")
    assert attach.denied and not attach.effect_attempted
    assert attach.reason in {
        AuthorizationReason.FORBIDDEN_VERB,
        AuthorizationReason.EXTENSION_SURFACE,
        AuthorizationReason.FILE_PATH,
        AuthorizationReason.CROSS_ROOT,
    }

    admitted = authorize_statement(
        "SELECT task_cid FROM tasks WHERE task_cid = ?",
        via_template=True,
    )
    assert admitted.allowed
    assert admitted.reason is AuthorizationReason.ADMITTED_TEMPLATE


def test_provider_env_and_token_isolation(tmp_path: Path) -> None:
    result = scenario_provider_env_isolation(
        policy=default_security_policy(),
        raw_token="chaos-token-value-xyz",
    )
    assert result.passed
    assert "QUACK_TOKEN" not in result.metrics["env_keys"]

    token_result = scenario_token_isolation(tmp_path, policy=default_security_policy())
    assert token_result.passed, token_result.detail
    findings = scan_for_secrets(
        {"token": "raw-secret", "ok": True},
        known_secrets=["raw-secret"],
    )
    assert findings


def test_forbidden_surface_and_loopback() -> None:
    forbidden = scenario_forbidden_surface()
    assert forbidden.passed, forbidden.detail
    assert forbidden.denials
    loopback = scenario_loopback_bind()
    assert loopback.passed, loopback.detail


def test_raw_sql_client_rejection(tmp_path: Path) -> None:
    db = prepare_chaos_database(tmp_path / "raw")
    result = scenario_raw_sql_rejection(db)
    assert result.passed, result.detail
    client = open_embedded_client(db, owner_id="test:raw", seed_generation=False)
    try:
        with pytest.raises(QuackClientSQLError):
            client.execute_sql("SELECT 1")
    finally:
        client.close()


def test_retry_jitter_bounded() -> None:
    result = scenario_retry_jitter(default_security_policy())
    assert result.passed, result.detail
    delays = result.metrics["delays"]
    assert delays[0] == 0.0
    assert all(0.0 <= delay <= 0.4 for delay in delays)


def test_live_gate_rejects_compatible_skip() -> None:
    with pytest.raises(QuackChaosLiveGateError):
        enforce_live_gate(
            profile_compatible=True,
            scenario_id=ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
            mode=ScenarioMode.SKIPPED,
            outcome=ScenarioOutcome.SKIPPED,
        )
    # Non-compatible may skip without raising.
    enforce_live_gate(
        profile_compatible=False,
        scenario_id=ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
        mode=ScenarioMode.SKIPPED,
        outcome=ScenarioOutcome.SKIPPED,
    )
    gate = scenario_live_gate_policy(
        profile_compatible=False,
        observed_modes={ChaosScenarioId.TOKEN_ISOLATION.value: ScenarioMode.HERMETIC.value},
    )
    assert gate.passed, gate.detail
    assert profile_claims_compatible(QuackCapabilityStatus.COMPATIBLE.value) is True
    assert profile_claims_compatible("install-required") is False


def test_hot_row_conflict_bounded(tmp_path: Path) -> None:
    db = prepare_chaos_database(tmp_path / "hot", task_count=8)
    result = scenario_hot_row_conflict(db, policy=default_security_policy())
    assert result.passed, result.detail
    assert result.metrics["accepted"] == 1
    assert result.metrics["max_attempts_observed"] <= default_security_policy().max_retry_attempts


def test_stale_after_rotation(tmp_path: Path) -> None:
    workspace = tmp_path / "rotation"
    db = prepare_chaos_database(workspace / "db", task_count=8)
    result = scenario_stale_after_rotation(
        db, workspace, policy=default_security_policy()
    )
    assert result.passed, result.detail
    assert result.metrics["live_generation"] > result.metrics["pre_generation"]


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------


def test_full_quack_chaos_suite_passes(tmp_path: Path) -> None:
    report = run_quack_chaos_suite(tmp_path / "suite")
    assert isinstance(report, QuackChaosReport)
    assert report.INTERFACE == QUACK_CHAOS_REPORT_INTERFACE
    assert report.task_id == TASK_ID
    assert report.goal_id == GOAL_ID
    assert report.live_gate_enforced is True
    assert report.duckdb_available is True
    assert report.policy.provider_receives_token is False

    present = {item.scenario_id for item in report.scenarios}
    assert set(REQUIRED_SCENARIOS).issubset(present)

    # No silent skips under a compatible-claiming injected profile either.
    if report.profile_claims_compatible:
        assert not report.skipped_scenarios

    assert_chaos_passed(report)
    assert report.verdict is ChaosVerdict.PASSED
    assert report.passed

    # Acceptance-facing scenario spot checks.
    for scenario_id in (
        ChaosScenarioId.TOKEN_ISOLATION,
        ChaosScenarioId.PROVIDER_ENV_ISOLATION,
        ChaosScenarioId.RAW_SQL_REJECTION,
        ChaosScenarioId.FORBIDDEN_SURFACE,
        ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
        ChaosScenarioId.HOT_ROW_CONFLICT,
        ChaosScenarioId.LOST_REPLY_IDEMPOTENCY,
        ChaosScenarioId.STALE_AFTER_ROTATION,
        ChaosScenarioId.SERVER_RESTART_STALE,
        ChaosScenarioId.CREDENTIAL_ROTATION,
        ChaosScenarioId.SECRET_SCAN,
        ChaosScenarioId.LIVE_GATE_POLICY,
        ChaosScenarioId.SPLIT_BRAIN_OWNERSHIP,
        ChaosScenarioId.TLS_BOUNDARY,
        ChaosScenarioId.PYTHON_UDF_LIMITATION,
        ChaosScenarioId.DENIAL_LOGGING,
    ):
        item = report.scenario(scenario_id)
        assert item is not None, scenario_id
        assert item.passed, (scenario_id, item.detail)
        assert item.mode is not ScenarioMode.SKIPPED

    payload = report.to_dict()
    assert payload["verdict"] == "passed"
    assert payload["failed_count"] == 0
    # Report serialization must not embed a raw token-looking secret from the vault.
    serialized = json.dumps(payload)
    assert "super-secret" not in serialized
    content_id = report.content_id()
    assert content_id.startswith("sha256:")


def test_assert_chaos_passed_fails_closed() -> None:
    policy = default_security_policy()
    report = QuackChaosReport(
        verdict=ChaosVerdict.FAILED,
        policy=policy,
        scenarios=(),
        profile_status="compatible",
        profile_claims_compatible=True,
        live_gate_enforced=True,
        duckdb_available=True,
    )
    with pytest.raises(QuackChaosScenarioError):
        assert_chaos_passed(report)


def test_suite_with_forced_compatible_profile_cannot_skip(tmp_path: Path) -> None:
    """When the harness is told the profile is compatible, skips are fatal."""

    from ipfs_accelerate_py.agent_supervisor.validation.quack_chaos import (
        QuackChaosHarness,
        _compatible_capability_report,
    )

    harness = QuackChaosHarness(
        workspace=tmp_path / "compat",
        capability_probe=_compatible_capability_report,
    )
    report = harness.run()
    assert report.profile_claims_compatible is True
    assert report.skipped_scenarios == ()
    assert_chaos_passed(report)
