"""DCR-082 deterministic-only restart and recovery contract tests."""

from __future__ import annotations

import ast
import hashlib
import inspect

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_recovery import (
    DeterministicRecoveryError,
    REPAIR_RECOVERY_INTERFACE,
    RecoveryDisposition,
    RecoveryJournalEntry,
    RecoveryRequest,
    recover_repair_state,
    replay_recovery,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.failure_replan_policy import (
    FailureReplanPolicy,
    ProviderRetryAuthorization,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.analytical_close_executor import (
    AnalyticalCloseExecutor,
    AnalyticalClosePlan,
    AnalyticalEdit,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
)


def _entry(state: str, *, sequence: int = 1, **values: object) -> RecoveryJournalEntry:
    base: dict[str, object] = {
        "transition_id": "transition:one",
        "sequence": sequence,
        "state": state,
        "operation_id": "operation:analytical-close",
        "gate_id": "gate:validated",
    }
    base.update(values)
    return RecoveryJournalEntry(**base)  # type: ignore[arg-type]


def _request(*entries: RecoveryJournalEntry, **values: object) -> RecoveryRequest:
    base: dict[str, object] = {
        "task_id": "DCR-082",
        "run_id": "run:one",
        "required_gate_id": "gate:validated",
        "journal": entries,
    }
    base.update(values)
    return RecoveryRequest(**base)  # type: ignore[arg-type]


def test_interface_and_cold_source_have_no_provider_or_model_route() -> None:
    assert REPAIR_RECOVERY_INTERFACE == "RepairRecovery@1"
    source = inspect.getsource(
        __import__(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_recovery",
            fromlist=["unused"],
        )
    )
    imports = [
        alias.name.lower()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    for marker in ("openai", "anthropic", "litellm", "grok_cli"):
        assert not any(marker in name for name in imports)


def test_committed_receipt_is_replayed_not_reconstructed_or_mutated(tmp_path) -> None:
    request = _request(
        _entry("intent"),
        _entry("mutation_applied", sequence=2, mutation_id="mutation:one"),
        _entry(
            "receipt_committed",
            sequence=3,
            mutation_id="mutation:one",
            receipt_id="receipt:real",
        ),
    )
    first = recover_repair_state(request, state_path=tmp_path / "recovery.json")
    second = recover_repair_state(request, state_path=tmp_path / "recovery.json")

    assert first == second
    assert first.disposition is RecoveryDisposition.REPLAYED_RECEIPT
    assert first.replayed_receipt_id == "receipt:real"
    assert first.mutation_authorized is False
    assert first.runtime_model_calls == 0


def test_crash_after_mutation_requires_rollback_and_never_double_mutates() -> None:
    decision = replay_recovery(
        _request(
            _entry("intent"),
            _entry("mutation_applied", sequence=2, mutation_id="mutation:one"),
        )
    )

    assert decision.disposition is RecoveryDisposition.ROLLBACK_REQUIRED
    assert decision.mutation_id == "mutation:one"
    assert decision.mutation_authorized is False


def test_analytical_receipt_replay_does_not_apply_the_edit_twice(tmp_path) -> None:
    target = tmp_path / "repair.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    original = target.read_text(encoding="utf-8")
    plan = AnalyticalClosePlan(
        edits=(
            AnalyticalEdit(
                path="repair.py",
                start=0,
                end=len(original),
                replacement="VALUE = 2\n",
                before_hash=hashlib.sha256(original.encode("utf-8")).hexdigest(),
            ),
        ),
        plan_cid="plan:one",
        task_cid="task:one",
    )
    executor = AnalyticalCloseExecutor(tmp_path)
    receipt = executor.apply(plan)
    replayed = executor.replay_receipt(receipt, plan)

    assert replayed.receipt_id == receipt.receipt_id
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


@pytest.mark.parametrize("failure_kind", ("transient_io", "transient_timeout", "transient_resource"))
def test_only_closed_typed_transient_failures_can_retry(failure_kind: str) -> None:
    decision = replay_recovery(_request(_entry("failed", failure_kind=failure_kind)))
    assert decision.disposition is RecoveryDisposition.RETRY_DETERMINISTIC
    assert decision.runtime_model_calls == 0


def test_expired_lease_and_gate_mismatch_fail_closed() -> None:
    expired = replay_recovery(_request(_entry("intent"), lease_expired=True))
    assert expired.disposition is RecoveryDisposition.DEFER_LEASE_EXPIRED
    with pytest.raises(DeterministicRecoveryError, match="gate mismatch"):
        replay_recovery(_request(_entry("intent", gate_id="gate:weakened")))


def test_fabricated_receipt_and_provider_authorization_are_rejected() -> None:
    with pytest.raises(DeterministicRecoveryError, match="committed receipt"):
        _entry("receipt_committed", mutation_id="mutation:one")
    with pytest.raises(DeterministicRecoveryError, match="provider/model"):
        _entry("intent", route_kind="provider")
    denied = FailureReplanPolicy().authorize_llm_retry({"disposition": "abstain_review"})
    assert denied.authorized is False
    with pytest.raises(Exception, match="disabled"):
        ProviderRetryAuthorization(
            authorized=True,
            reason_code="forbidden",
            disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        )
