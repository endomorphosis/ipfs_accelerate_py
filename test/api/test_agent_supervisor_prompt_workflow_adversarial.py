"""Adversarial non-compensable fixtures for prompt bootstrap and rescue."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.prompt_workflow_benchmark import (
    REQUIRED_ADVERSARIAL_FIXTURES,
    REQUIRED_OPTIONAL_DEPENDENCIES,
    AdversarialFixture,
    PromptWorkflowBenchmark,
    TerminalOutcome,
    build_frozen_prompt_workflow_benchmark,
    recompute_prompt_workflow_gate,
)


def test_every_required_adversarial_fixture_is_non_compensable():
    benchmark = build_frozen_prompt_workflow_benchmark()
    report = recompute_prompt_workflow_gate(benchmark)

    assert set(report.adversarial_fixtures_passed) == {
        item.value for item in REQUIRED_ADVERSARIAL_FIXTURES
    }
    assert report.adversarial_passed
    assert report.secret_hygiene_passed
    assert report.passed

    target = next(
        receipt
        for receipt in benchmark.receipts
        if receipt.adversarial_fixture is not None
    )
    escaped = PromptWorkflowBenchmark(
        tuple(
            replace(
                receipt,
                metrics=replace(
                    receipt.metrics,
                    escape_count=1,
                    terminal_result=TerminalOutcome.ACCEPTED.value,
                ),
            )
            if receipt.receipt_id == target.receipt_id
            else receipt
            for receipt in benchmark.receipts
        )
    )
    failed = recompute_prompt_workflow_gate(escaped)
    assert not failed.passed
    assert (
        f"adversarial-escape:{target.adversarial_fixture.value}"
        in failed.failure_codes
    )


def test_missing_fixture_and_secret_emission_cannot_be_averaged_away():
    benchmark = build_frozen_prompt_workflow_benchmark()
    omitted = PromptWorkflowBenchmark(
        tuple(
            receipt
            for receipt in benchmark.receipts
            if receipt.adversarial_fixture
            is not REQUIRED_ADVERSARIAL_FIXTURES[0]
        )
    )
    report = recompute_prompt_workflow_gate(omitted)
    assert not report.passed
    assert any(
        code.startswith("missing-adversarial-fixture:")
        for code in report.failure_codes
    )

    target = next(
        receipt
        for receipt in benchmark.receipts
        if receipt.adversarial_fixture is AdversarialFixture.SECRET_LEAK
    )
    leaked = PromptWorkflowBenchmark(
        tuple(
            replace(
                receipt,
                metrics=replace(receipt.metrics, secret_bytes_emitted=16),
            )
            if receipt.receipt_id == target.receipt_id
            else receipt
            for receipt in benchmark.receipts
        )
    )
    leak_report = recompute_prompt_workflow_gate(leaked)
    assert not leak_report.passed
    assert not leak_report.secret_hygiene_passed
    assert "secret-hygiene" in leak_report.failure_codes
    assert (
        f"adversarial-escape:{AdversarialFixture.SECRET_LEAK.value}"
        in leak_report.failure_codes
    )


def test_scope_policy_authority_completion_and_sql_cases_are_required():
    required = {
        AdversarialFixture.PROMPT_INJECTION,
        AdversarialFixture.PATH_ESCAPE,
        AdversarialFixture.SYMLINK_ESCAPE,
        AdversarialFixture.FORGED_CID,
        AdversarialFixture.SCHEMA_DOWNGRADE,
        AdversarialFixture.SQL_INJECTION,
        AdversarialFixture.PID_REUSE,
        AdversarialFixture.PROCESS_ESCAPE,
        AdversarialFixture.POLICY_WEAKENING,
        AdversarialFixture.AUTHORIZATION_BYPASS,
        AdversarialFixture.PERMIT_FORGERY,
        AdversarialFixture.COMPLETION_FORGERY,
        AdversarialFixture.MANDATORY_EVIDENCE_OMISSION,
        AdversarialFixture.SHELL_RESCUE_PROPOSAL,
    }
    benchmark = build_frozen_prompt_workflow_benchmark()
    present = {
        receipt.adversarial_fixture
        for receipt in benchmark.receipts
        if receipt.adversarial_fixture is not None
    }
    assert required.issubset(present)
    for fixture in required:
        receipt = next(
            item
            for item in benchmark.receipts
            if item.adversarial_fixture is fixture
        )
        assert receipt.metrics.escape_count == 0
        assert receipt.metrics.terminal_result in {
            TerminalOutcome.REJECTED.value,
            TerminalOutcome.DENIED.value,
            TerminalOutcome.FAIL_CLOSED.value,
            TerminalOutcome.QUARANTINED.value,
            TerminalOutcome.DEGRADED.value,
        }


def test_optional_dependency_degradation_is_explicit_and_lazy():
    benchmark = build_frozen_prompt_workflow_benchmark()
    report = recompute_prompt_workflow_gate(benchmark)
    assert report.lazy_discovery_passed
    assert report.deterministic_degraded_passed
    assert set(report.optional_dependencies_passed) == {
        item.value for item in REQUIRED_OPTIONAL_DEPENDENCIES
    }
    for receipt in benchmark.receipts:
        assert receipt.lazy_discovery
        if receipt.optional_dependency is not None:
            assert receipt.degraded_local
            assert receipt.deterministic_replay_id
            assert receipt.metrics.escape_count == 0
