"""DCR-001 contract tests for deterministic repair's no-LLM boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.no_llm_policy import (
    DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE,
    DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA,
    NO_LLM_EXECUTION_GUARD_INTERFACE,
    DeterministicRepairAuthorityPolicy,
    NoLlmExecutionDenied,
    NoLlmExecutionGuard,
    RepairAuthorityDisposition,
    RepairExecutionRoute,
)


@pytest.fixture(autouse=True)
def isolated_process_guard() -> None:
    NoLlmExecutionGuard.reset_audit_for_testing()
    yield
    NoLlmExecutionGuard.reset_audit_for_testing()


def policy() -> DeterministicRepairAuthorityPolicy:
    return DeterministicRepairAuthorityPolicy(
        local_logic_pins=frozenset({"logic:doctor-ir@1"}),
        prover_subprocess_pins=frozenset({"prover:lean@4.19"}),
        loopback_mcp_pins=frozenset({"mcp:repair-local@1"}),
    )


def test_interface_identities_and_only_explicit_pinned_routes_admit() -> None:
    subject = policy()
    assert NO_LLM_EXECUTION_GUARD_INTERFACE == "NoLlmExecutionGuard@1"
    assert DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE == "DeterministicRepairAuthorityPolicy@1"
    assert subject.authorize(
        RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC,
        pin="logic:doctor-ir@1",
    ).allowed
    assert subject.authorize(
        RepairExecutionRoute.PROVER_SUBPROCESS,
        pin="prover:lean@4.19",
    ).allowed
    assert subject.authorize(
        RepairExecutionRoute.LOOPBACK_MCP,
        pin="mcp:repair-local@1",
        endpoint="http://127.0.0.1:8765/mcp",
    ).allowed


@pytest.mark.parametrize(
    "route",
    (
        "llm",
        "model.call",
        "indirect_model_delegate",
        "retry.remote_provider",
        "rescue-provider",
        "residual_llm_authorized",
        "self-improvement.model",
        "provider_alias",
    ),
)
def test_model_and_remote_negative_aliases_deny_before_callback(route: str) -> None:
    invoked: list[str] = []
    with pytest.raises(NoLlmExecutionDenied, match="forbidden_model_or_remote_route"):
        policy().invoke(route, lambda: invoked.append("called"), pin="forged")
    assert invoked == []
    counters = NoLlmExecutionGuard.audit_snapshot()
    assert counters["denied_total"] == 1
    assert counters["denied_by_reason"]["forbidden_model_or_remote_route"] == 1


def test_unpinned_and_non_loopback_routes_fail_closed_before_invocation() -> None:
    invoked: list[str] = []
    with pytest.raises(NoLlmExecutionDenied, match="unpinned_local_logic"):
        policy().invoke("deterministic_local_logic", lambda: invoked.append("bad"))
    with pytest.raises(NoLlmExecutionDenied, match="non_loopback_mcp_endpoint"):
        policy().invoke(
            "loopback_mcp",
            lambda: invoked.append("bad"),
            pin="mcp:repair-local@1",
            endpoint="https://example.invalid/mcp",
        )
    with pytest.raises(NoLlmExecutionDenied, match="non_loopback_mcp_endpoint"):
        policy().invoke(
            "loopback_mcp",
            lambda: invoked.append("bad"),
            pin="mcp:repair-local@1",
            endpoint="http://userinfo@127.0.0.1:8765/mcp",
        )
    assert invoked == []


def test_absent_allowlists_disable_categories_and_pinned_names_are_not_routes() -> None:
    disabled = DeterministicRepairAuthorityPolicy()
    with pytest.raises(NoLlmExecutionDenied, match="unpinned_local_logic"):
        disabled.invoke(
            "deterministic_local_logic",
            lambda: None,
            pin="logic:model-checker@1",
        )

    # The word "model" in an explicit deterministic *pin* is not a route
    # alias.  It remains admissible when the local logic capability is pinned.
    named_logic = DeterministicRepairAuthorityPolicy(
        local_logic_pins=frozenset({"logic:model-checker@1"}),
    )
    assert (
        named_logic.invoke(
            "deterministic_local_logic",
            lambda: "local result",
            pin="logic:model-checker@1",
        )
        == "local result"
    )


def test_nested_guards_are_process_wide_and_cannot_open_an_unguarded_gap() -> None:
    subject = policy()
    outer = NoLlmExecutionGuard()
    inner = NoLlmExecutionGuard()
    assert not NoLlmExecutionGuard.active()
    with outer:
        assert NoLlmExecutionGuard.active()
        with inner:
            assert NoLlmExecutionGuard.audit_snapshot()["active_depth"] == 2
            with pytest.raises(NoLlmExecutionDenied):
                subject.authorize("retry_model", pin="forged")
        assert NoLlmExecutionGuard.active()
    assert not NoLlmExecutionGuard.active()
    assert NoLlmExecutionGuard.audit_snapshot()["denied_by_route"]["retry_model"] == 1


def test_abstain_and_defer_are_typed_terminals_not_model_fallbacks() -> None:
    subject = policy()
    abstained = subject.abstain("ambiguous proof obligation")
    deferred = subject.defer("local prover capability unavailable")
    assert abstained.disposition is RepairAuthorityDisposition.ABSTAIN
    assert deferred.disposition is RepairAuthorityDisposition.DEFER
    assert abstained.terminal and deferred.terminal
    assert not abstained.to_dict()["fallback_authorized"]
    assert NoLlmExecutionGuard.audit_snapshot()["denied_total"] == 0


def test_reviewed_authority_config_is_exact_and_bootstraps_with_no_self_authority(
    tmp_path: Path,
) -> None:
    workspace = Path(__file__).resolve().parents[4]
    config = workspace / "config/deterministic_contract_repair_authority.json"
    subject = DeterministicRepairAuthorityPolicy.from_file(config)
    assert DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA in subject.to_dict()["schema"]
    assert subject.to_dict()["model_call_budget"] == 0
    assert subject.to_dict()["local_logic_pins"] == []
    with pytest.raises(NoLlmExecutionDenied, match="unpinned_local_logic"):
        subject.invoke("deterministic_local_logic", lambda: None, pin="self-authored")

    drift = subject.to_dict()
    drift["unknown_authority"] = True
    with pytest.raises(ValueError, match="fields must match exactly"):
        DeterministicRepairAuthorityPolicy.from_mapping(drift)

    duplicate = tmp_path / "duplicate-authority.json"
    duplicate.write_text(
        '{"schema":"one","schema":"two"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unreadable"):
        DeterministicRepairAuthorityPolicy.from_file(duplicate)
