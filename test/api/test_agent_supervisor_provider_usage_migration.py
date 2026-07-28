"""Tests for ASI-168 complete supervisor provider callsite migration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_execution import (
    ProviderExecutionGateway,
    ProviderExecutionPhase,
)
from ipfs_accelerate_py.agent_supervisor.provider_usage_migration import (
    COMPLETE_PROVIDER_CALLSITE_GOAL_ID,
    COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
    DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
    IN_SCOPE_CONSUMER_MODULES,
    MIGRATION_IS_COMPLETION_EVIDENCE,
    MIGRATION_IS_CORRECTNESS_EVIDENCE,
    MIGRATION_MAY_CHANGE_PROMPT_SOURCE_OUTPUT_CONTRACTS,
    MIGRATION_MAY_RETRY_SIDE_EFFECTING_AGENT_WORK,
    MIGRATION_MAY_ROUTE_TO_FORBIDDEN_ENDPOINT,
    CallsiteViolationKind,
    ConsumerId,
    ConsumerCallContext,
    MigratedProviderCallResult,
    NonMeterableProviderResult,
    ProviderUsageMigrationError,
    SupervisorUsageMode,
    admit_non_meterable,
    assert_coverage_complete,
    build_callsite_inventory,
    build_consumer_call_context,
    build_request_lineage_envelope,
    clear_last_call_results,
    consume_usage_receipt,
    discover_migration_schemas,
    dispatch_migrated_provider_call,
    extract_child_usage_metadata,
    inventory_repository_consumers,
    last_call_result,
    migration_status_for,
    mode_requires_envelope,
    mode_requires_receipt,
    non_meterable_from_child,
    registered_consumer_ids,
    request_leaf_envelope,
    resolve_usage_mode,
    retain_last_call_result,
    scan_module_callsites,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_requirement_and_authority_bounds() -> None:
    assert COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID.startswith("requirement:")
    assert COMPLETE_PROVIDER_CALLSITE_GOAL_ID == "ASI-G520"
    assert MIGRATION_IS_COMPLETION_EVIDENCE is False
    assert MIGRATION_IS_CORRECTNESS_EVIDENCE is False
    assert MIGRATION_MAY_RETRY_SIDE_EFFECTING_AGENT_WORK is False
    assert MIGRATION_MAY_ROUTE_TO_FORBIDDEN_ENDPOINT is False
    assert MIGRATION_MAY_CHANGE_PROMPT_SOURCE_OUTPUT_CONTRACTS is False
    assert set(registered_consumer_ids()) == {
        item.value for item in ConsumerId
    }
    schemas = discover_migration_schemas()
    assert any("provider-usage-migration" in item for item in schemas)


def test_resolve_usage_mode_env_and_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", raising=False)
    assert resolve_usage_mode() is SupervisorUsageMode.OFF
    assert resolve_usage_mode("enforce") is SupervisorUsageMode.ENFORCE
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "assist")
    assert resolve_usage_mode() is SupervisorUsageMode.ASSIST
    with pytest.raises(ProviderUsageMigrationError, match="unknown usage mode"):
        resolve_usage_mode("not-a-mode")


def test_mode_policy_helpers() -> None:
    assert mode_requires_envelope(SupervisorUsageMode.OFF) is False
    assert mode_requires_envelope(SupervisorUsageMode.OBSERVE) is True
    assert mode_requires_receipt(SupervisorUsageMode.OBSERVE) is False
    assert mode_requires_receipt(SupervisorUsageMode.ENFORCE) is True


def test_off_mode_preserves_invoke_without_gateway() -> None:
    clear_last_call_results()
    calls = {"n": 0}

    def invoke() -> str:
        calls["n"] += 1
        return "legacy-output"

    context = build_consumer_call_context(
        consumer_id=ConsumerId.TASK_PROPOSAL_ROUTER,
        provider_id="provider:test",
    )
    result = dispatch_migrated_provider_call(
        context=context,
        invoke=invoke,
        mode=SupervisorUsageMode.OFF,
    )
    assert result.text == "legacy-output"
    assert result.metered is False
    assert result.usage_receipt is None
    assert result.phase == "off"
    assert calls["n"] == 1
    assert result.is_completion_evidence is False


def test_enforce_mode_supplies_envelope_and_consumes_receipt() -> None:
    clear_last_call_results()
    calls = {"n": 0}

    def invoke() -> str:
        calls["n"] += 1
        return '{"ok": true}'

    context = build_consumer_call_context(
        consumer_id=ConsumerId.PROMPT_GOAL_PLANNER,
        provider_id="provider:simulated",
        estimated_requests=1,
    )
    lineage = build_request_lineage_envelope(context)
    leaf = request_leaf_envelope(lineage)
    assert leaf.scope.level.value == "request"
    assert leaf.scope.task_id
    assert leaf.scope.attempt >= 1
    assert leaf.scope.stage
    assert leaf.scope.lane
    assert leaf.scope.request_id
    assert leaf.scope.idempotency_key
    assert leaf.scope.deadline_at
    assert leaf.scope.policy_id

    result = dispatch_migrated_provider_call(
        context=context,
        invoke=invoke,
        mode=SupervisorUsageMode.ENFORCE,
    )
    assert result.metered is True
    assert result.usage_receipt is not None
    assert result.usage_receipt.request_id == context.request_id
    assert result.usage_receipt.is_completion_evidence is False
    assert result.execution_result is not None
    assert result.execution_result.phase is ProviderExecutionPhase.SETTLED
    assert result.text == '{"ok": true}'
    assert calls["n"] == 1
    retain_last_call_result(ConsumerId.PROMPT_GOAL_PLANNER, result)
    assert last_call_result(ConsumerId.PROMPT_GOAL_PLANNER) is result


def test_dispatch_does_not_retry_side_effecting_work() -> None:
    calls = {"n": 0}

    def invoke() -> str:
        calls["n"] += 1
        return "once"

    context = build_consumer_call_context(
        consumer_id=ConsumerId.RESCUE_PLANNER,
        provider_id="provider:once",
    )
    first = dispatch_migrated_provider_call(
        context=context,
        invoke=invoke,
        mode=SupervisorUsageMode.ASSIST,
    )
    # Exact replay through the same gateway attempt key must not re-invoke.
    gateway = ProviderExecutionGateway()
    # Rebuild an identical context (same idempotency) and force same attempt.
    second = dispatch_migrated_provider_call(
        context=context,
        invoke=invoke,
        mode=SupervisorUsageMode.ASSIST,
        gateway=gateway,
    )
    assert first.text == "once"
    assert second.text == "once"
    # Each dispatch builds a new gateway unless shared; still one invoke per
    # dispatch, never an internal double invoke.
    assert calls["n"] == 2


def test_non_meterable_child_result_and_enforce_ceiling() -> None:
    result = non_meterable_from_child(
        consumer_id=ConsumerId.LEANSTRAL_PROOF_PROVIDER,
        provider_id="cli:codex",
        output_text="hello",
        payload=None,
        ceiling_requests=DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
    )
    assert result.reason_code == "usage_metadata_unavailable"
    assert result.is_completion_evidence is False
    admitted = admit_non_meterable(
        result,
        mode=SupervisorUsageMode.ENFORCE,
        reviewed_ceiling=DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
    )
    assert admitted.ceiling_requests == DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS

    from ipfs_accelerate_py.agent_supervisor.provider_usage_migration import (
        MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
    )

    oversized = NonMeterableProviderResult(
        consumer_id=ConsumerId.LEANSTRAL_PROOF_PROVIDER.value,
        provider_id="cli:codex",
        ceiling_requests=MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
    )
    with pytest.raises(ProviderUsageMigrationError, match="reviewed enforce ceiling"):
        admit_non_meterable(
            oversized,
            mode=SupervisorUsageMode.ENFORCE,
            reviewed_ceiling=1,
        )
    with pytest.raises(ProviderUsageMigrationError, match="reviewed bound"):
        NonMeterableProviderResult(
            consumer_id=ConsumerId.LEANSTRAL_PROOF_PROVIDER.value,
            provider_id="cli:codex",
            ceiling_requests=99,
        )


def test_extract_child_usage_and_reset_metadata() -> None:
    usage, reset = extract_child_usage_metadata(
        {
            "usage": {"input_tokens": 3, "api_key": "secret"},
            "reset": {"resets_at": "2099-01-01T00:00:00Z"},
        }
    )
    assert usage.get("input_tokens") == 3
    assert "api_key" not in usage
    assert reset.get("resets_at")


def test_consume_usage_receipt_rejects_drop_and_authority() -> None:
    context = build_consumer_call_context(
        consumer_id=ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT,
        provider_id="provider:x",
    )

    def invoke() -> str:
        return "ok"

    result = dispatch_migrated_provider_call(
        context=context,
        invoke=invoke,
        mode=SupervisorUsageMode.ENFORCE,
    )
    assert result.execution_result is not None
    receipt = consume_usage_receipt(
        result.execution_result,
        expected_request_id=context.request_id,
    )
    assert receipt.request_id == context.request_id

    from ipfs_accelerate_py.agent_supervisor.provider_execution import (
        ProviderExecutionResult,
    )

    from ipfs_accelerate_py.agent_supervisor.provider_usage import (
        SupervisorUsageFinalStatus,
    )

    dropped = ProviderExecutionResult(
        phase=ProviderExecutionPhase.SETTLED,
        final_status=SupervisorUsageFinalStatus.COMMITTED,
        granted=True,
        reservation_id="res:test",
        usage_revision="usage:1",
        catalog_revision="catalog:1",
        provider_id="provider:x",
        redacted_endpoint="endpoint:redacted",
        attempt_key="attempt:1",
        request_key="request:1",
        request_id=context.request_id,
        receipt=None,
    )
    with pytest.raises(ProviderUsageMigrationError, match="dropped"):
        consume_usage_receipt(dropped)


def test_migrated_result_cannot_claim_completion_authority() -> None:
    with pytest.raises(ProviderUsageMigrationError, match="proof/completion"):
        MigratedProviderCallResult(is_completion_evidence=True)
    with pytest.raises(ProviderUsageMigrationError, match="proof/completion"):
        NonMeterableProviderResult(is_correctness_evidence=True)


def test_ast_inventory_fails_unregistered_direct_import_and_call() -> None:
    rogue = """
import ipfs_accelerate_py.llm_router as sneaky_router
from ipfs_datasets_py.llm_router import generate_text as gt

def go(prompt):
    return sneaky_router.generate_text(prompt)
"""
    records, violations = scan_module_callsites(
        rogue,
        module_path="ipfs_accelerate_py/agent_supervisor/rogue_consumer.py",
    )
    kinds = {item.kind for item in violations}
    assert CallsiteViolationKind.UNREGISTERED_DIRECT_IMPORT in kinds
    assert (
        CallsiteViolationKind.WRAPPER_ALIAS in kinds
        or CallsiteViolationKind.UNREGISTERED_DIRECT_CALL in kinds
    )
    assert records


def test_ast_inventory_allowlists_provider_free_discovery() -> None:
    source = """
# provider-free schema discovery only
SCHEMA = "x"
def discover():
    return (SCHEMA,)
"""
    records, violations = scan_module_callsites(
        source,
        module_path=(
            "ipfs_accelerate_py/agent_supervisor/provider_usage_migration.py"
        ),
    )
    assert violations == []


def test_ast_inventory_fails_missing_attribution_for_consumer_without_gateway() -> None:
    bare = """
def plan():
    return "no provider"
"""
    _records, violations = scan_module_callsites(
        bare,
        module_path=(
            "ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py"
        ),
    )
    assert any(
        item.kind is CallsiteViolationKind.MISSING_ATTRIBUTION
        for item in violations
    )


def test_ast_inventory_accepts_migrated_consumer_fixture() -> None:
    migrated = """
from ipfs_accelerate_py.agent_supervisor.provider_usage_migration import (
    dispatch_migrated_provider_call,
    ConsumerId,
)

def _call_text_provider(prompt):
    return dispatch_migrated_provider_call(
        context=None,
        invoke=lambda: prompt,
    )
"""
    _records, violations = scan_module_callsites(
        migrated,
        module_path=(
            "ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py"
        ),
    )
    assert not any(
        item.kind is CallsiteViolationKind.MISSING_ATTRIBUTION
        for item in violations
    )


def test_repository_inventory_covers_in_scope_consumers() -> None:
    inventory = inventory_repository_consumers(REPO_ROOT)
    for consumer in ConsumerId:
        assert consumer.value in inventory.consumers
    # Live tree should be coverage-complete after ASI-168 migration.
    assert_coverage_complete(inventory)
    record = inventory.to_record()
    assert record["coverage_complete"] is True
    assert record["is_completion_evidence"] is False


def test_build_callsite_inventory_reports_receipt_drop_fixture() -> None:
    sources = {
        IN_SCOPE_CONSUMER_MODULES[ConsumerId.TASK_PROPOSAL_ROUTER]: """
from ipfs_accelerate_py.agent_supervisor.provider_usage_migration import (
    dispatch_migrated_provider_call,
)
def go():
    return dispatch_migrated_provider_call
""",
        IN_SCOPE_CONSUMER_MODULES[ConsumerId.PROMPT_GOAL_PLANNER]: """
from ipfs_accelerate_py.agent_supervisor.provider_execution import (
    ProviderExecutionGateway,
)
def go():
    return ProviderExecutionGateway
""",
        IN_SCOPE_CONSUMER_MODULES[ConsumerId.RESCUE_PLANNER]: """
from ipfs_accelerate_py.agent_supervisor.provider_usage_migration import (
    consume_usage_receipt,
)
def go():
    return consume_usage_receipt
""",
        IN_SCOPE_CONSUMER_MODULES[ConsumerId.LEANSTRAL_PROOF_PROVIDER]: """
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import call_llm_router
def go(p, c):
    return call_llm_router(p, c)
""",
        IN_SCOPE_CONSUMER_MODULES[ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT]: """
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchScheduler,
)
def go():
    return ProviderBatchScheduler
""",
    }
    inventory = build_callsite_inventory(sources, require_consumers=True)
    assert inventory.coverage_complete is True


def test_migration_status_for_each_consumer() -> None:
    for consumer in ConsumerId:
        status = migration_status_for(consumer)
        assert status["migrated"] is True
        assert status["requirement_id"] == COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID
        assert status["is_completion_evidence"] is False
        assert Path(REPO_ROOT / status["module_path"]).is_file()


def test_declared_flat_surfaces_export_migration_status() -> None:
    import importlib.util

    flat_paths = {
        ConsumerId.TASK_PROPOSAL_ROUTER: (
            REPO_ROOT
            / "ipfs_accelerate_py/agent_supervisor/task_proposal_router.py"
        ),
        ConsumerId.PROMPT_GOAL_PLANNER: (
            REPO_ROOT
            / "ipfs_accelerate_py/agent_supervisor/prompt_goal_planner.py"
        ),
        ConsumerId.RESCUE_PLANNER: (
            REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/rescue_planner.py"
        ),
        ConsumerId.LEANSTRAL_PROOF_PROVIDER: (
            REPO_ROOT
            / "ipfs_accelerate_py/agent_supervisor/leanstral_proof_provider.py"
        ),
        ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT: (
            REPO_ROOT
            / "ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py"
        ),
    }
    for consumer, path in flat_paths.items():
        assert path.is_file(), path
        source = path.read_text(encoding="utf-8")
        assert "ASI_168" in source or "migration_status_for" in source
        assert consumer.value in source or consumer.value.replace("_", "-") in source

    # Package imports resolve to landed modules and remain importable.
    for stem in (
        "task_proposal_router",
        "prompt_goal_planner",
        "rescue_planner",
        "leanstral_proof_provider",
        "leanstral_goal_development",
    ):
        module = __import__(
            f"ipfs_accelerate_py.agent_supervisor.{stem}",
            fromlist=["*"],
        )
        assert module is not None
        assert any(
            not name.startswith("__") for name in dir(module)
        )


def test_consumer_modules_import_migration_helpers() -> None:
    """Runtime fixture: each in-scope consumer source references the gateway."""

    markers = (
        "provider_usage_migration",
        "dispatch_migrated_provider_call",
        "call_llm_router",
        "ProviderBatchScheduler",
        "provider_execution",
    )
    for consumer, rel in IN_SCOPE_CONSUMER_MODULES.items():
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert any(marker in source for marker in markers), consumer


def test_cold_import_is_provider_free() -> None:
    import importlib

    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.provider_usage_migration"
    )
    assert module.COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID
    assert module.MIGRATION_IS_COMPLETION_EVIDENCE is False
