"""VGO-086: adversarial patch acceptance and isolation.

Attack fixtures remain compact data and temporary-repository diffs.  Tests
never execute modeled HTML, commands, tools, host paths, or credential
payloads.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.authority import (
    GUI_ACCEPTANCE_AUTHORITY_INTERFACE,
    AuthorityReasonCode,
    AuthorityVerdict,
    GuiAcceptanceAuthority,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.improvement_loop import (
    GUI_IMPROVEMENT_DECISION_INTERFACE,
    ImprovementDecisionKind,
    ImprovementReasonCode,
    default_verified_gui_optimizer,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.patch_scope import (
    GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
    GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
    GUI_PATCH_SCOPE_DECISION_INTERFACE,
    GUI_PATCH_SCOPE_GATE_INTERFACE,
    PatchScopeReasonCode,
    default_patch_scope_gate,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.proposal import (
    DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.run_journal import (
    JournalPhase,
    JournalReasonCode,
    ResumeAction,
    RunStatus,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.worktree_executor import (
    ApplicationDisposition,
    CleanupState,
    GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE,
    GuiIsolatedWorktreeExecutor,
    HOST_GIT_EXECUTABLE,
    HostGitResult,
    HostGitRunner,
    WorktreeExecutorReasonCode,
    default_isolated_worktree_executor,
    sealed_git_environment,
)
from ipfs_datasets_py.logic.gui_optimizer.models import GuiImprovementReceipt
from ipfs_datasets_py.logic.gui_optimizer.schema import GUI_IMPROVEMENT_RECEIPT_INTERFACE

IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"
IN_SCOPE_TEST = "swissknife/test/browser/agent-supervisor-console-gateway.test.ts"
ORIGINAL = "export const label = 'old';\n"
UPDATED = "export const label = 'accessible';\n"
REVISION = "b" * 40
STALE_REVISION = "c" * 40
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "gui_optimizer"
    / "adversarial-proposals.json"
)
SUITE_FIELDS = frozenset(
    {
        "application_id",
        "cases",
        "conflict_policy",
        "interface",
        "required_case_ids",
        "schema_version",
        "screen_id",
        "suite_id",
        "task_id",
    }
)
CASE_FIELDS = frozenset(
    {
        "added_lines",
        "application_ids",
        "authorizes",
        "case_id",
        "change_kind",
        "deleted_lines",
        "expected_decision",
        "expected_reason_codes",
        "family",
        "halt_after_phase",
        "hard_gate",
        "isolate_canonical",
        "kind",
        "marker",
        "never_auto_accept",
        "operation",
        "path",
        "surface",
        "title",
    }
)
DECISIONS = frozenset({"rejected", "review_required", "pending"})
SURFACES = frozenset({"scope_gate", "loop", "executor", "journal"})
_FORBIDDEN_EXECUTION_MARKERS = (
    "<script",
    "javascript:",
    "eval(",
    "subprocess",
    "/bin/sh",
    "curl ",
    "wget ",
)
_VALIDATION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
GUI_ACCEPTANCE_DECISION_INTERFACE = GUI_IMPROVEMENT_DECISION_INTERFACE


def load_suite() -> dict[str, Any]:
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if type(raw) is not dict:
        raise AssertionError("adversarial-proposals.json must be a JSON object")
    return raw


def _reject_unknown(payload: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise AssertionError(f"unknown {label} field(s): {', '.join(unknown)}")


@pytest.fixture(scope="module")
def suite() -> dict[str, Any]:
    payload = load_suite()
    _reject_unknown(payload, SUITE_FIELDS, "GuiAdversarialProposalSuite")
    return payload


@pytest.fixture(scope="module")
def cases(suite: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_cases = suite["cases"]
    if type(raw_cases) is not list or not raw_cases:
        raise AssertionError("suite.cases must be a non-empty JSON array")
    decoded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if type(item) is not dict:
            raise AssertionError(f"cases[{index}] must be a JSON object")
        _reject_unknown(item, CASE_FIELDS, f"cases[{index}]")
        case_id = item.get("case_id")
        if type(case_id) is not str or not case_id:
            raise AssertionError(f"cases[{index}].case_id must be a nonempty string")
        if case_id in seen:
            raise AssertionError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        if item.get("authorizes") is not False:
            raise AssertionError(f"{case_id} must declare authorizes=false")
        if item.get("never_auto_accept") is not True:
            raise AssertionError(f"{case_id} must declare never_auto_accept=true")
        if item.get("expected_decision") not in DECISIONS:
            raise AssertionError(f"{case_id} has unknown expected_decision")
        if item.get("surface") not in SURFACES:
            raise AssertionError(f"{case_id} has unknown surface")
        codes = item.get("expected_reason_codes")
        if type(codes) is not list or any(type(code) is not str for code in codes):
            raise AssertionError(f"{case_id} expected_reason_codes must be strings")
        decoded.append(item)
    required = suite["required_case_ids"]
    if type(required) is not list or sorted(required) != sorted(seen):
        raise AssertionError("required_case_ids must equal the unique case_id set")
    return tuple(decoded)


def _by_id(cases: tuple[dict[str, Any], ...], case_id: str) -> dict[str, Any]:
    return next(item for item in cases if item["case_id"] == case_id)


def _proposal(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "proposal_id": "proposal:adversarial",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective": "modeled-attack: keep the goal form labeled",
        "intended_file_paths": [IN_SCOPE],
        "intended_component_ids": ["comp:goal-form"],
        "acceptance_criteria": ["modeled-attack: reject forbidden patches"],
        "expected_test_ids": ["test:goal-form-a11y"],
        "expected_screenshot_ids": ["screenshot:keyboard-desktop"],
        "state_effect_ids": ["state:ready"],
        "visual_effect_summary": "modeled-attack: no execution",
        "route_kind": "deterministic_transform",
        "context_pack_id": "pack:adversarial",
        "decision": "pending",
        "analysis_classification": "exact",
        "verification_status": "unverified",
        "interface": GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
        "schema_version": GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
    }
    payload.update(overrides)
    return payload


def _hunk_from_case(case: Mapping[str, Any]) -> dict[str, Any]:
    operation = case.get("operation", "modify")
    added = case.get("added_lines", 0 if operation == "delete" else 1)
    deleted = case.get("deleted_lines", 1 if operation == "delete" else 0)
    marker = str(case.get("marker", "modeled-attack"))
    hunk: dict[str, Any] = {
        "path": case.get("path", IN_SCOPE),
        "operation": operation,
        "added_lines": added,
        "deleted_lines": deleted,
        "diff_text": f"- {marker}" if operation == "delete" else f"+ {marker}",
    }
    if "change_kind" in case:
        hunk["change_kinds"] = [case["change_kind"]]
    return hunk


def _observation(case: Mapping[str, Any]) -> dict[str, Any]:
    hunk = _hunk_from_case(case)
    apps = case.get("application_ids") or ["app:agent-supervisor"]
    payload: dict[str, Any] = {
        "hunks": [hunk],
        "touched_component_ids": ["comp:goal-form"],
        "touched_state_effect_ids": ["state:ready"],
        "touched_test_ids": [],
        "touched_screenshot_ids": [],
        "application_ids": list(apps),
        "action_binding_ids": [],
        "action_contract_evidence": [],
        "visual_effect_observed": False,
        "unresolved_paths": [],
    }
    if hunk["path"] == IN_SCOPE_TEST:
        payload["touched_component_ids"] = []
        payload["touched_state_effect_ids"] = []
    return payload


def _invalidation() -> dict[str, Any]:
    return {
        "plan_id": "invalidate:adversarial",
        "change_set_id": "changeset:adversarial",
        "reasons": ["component_changed"],
        "affected_component_ids": ["comp:goal-form"],
        "affected_scenario_ids": ["scenario:keyboard-only"],
        "affected_check_ids": ["check:direct-tests"],
        "fallback_triggered": False,
        "fallback_explanation": "",
        "interface": "UiInvalidationPlan@1",
        "schema_version": "ui-invalidation-plan/v1",
        "confidence": "exact",
    }


def _evaluate_scope(case: Mapping[str, Any]):
    proposal = _proposal()
    if case.get("path") == IN_SCOPE_TEST:
        proposal = _proposal(
            intended_file_paths=[IN_SCOPE, IN_SCOPE_TEST],
            expected_test_ids=[],
        )
    return default_patch_scope_gate().evaluate_request(
        {
            "proposal": proposal,
            "observation": _observation(case),
            "invalidation": _invalidation(),
        }
    )


def _assert_scope_decision(case: Mapping[str, Any], decision: Any) -> None:
    encoded = decision.to_dict()
    assert encoded["interface"] == GUI_PATCH_SCOPE_DECISION_INTERFACE
    assert decision.allowed is False
    expected = case["expected_decision"]
    if expected == "rejected":
        assert decision.verdict is AuthorityVerdict.REJECT
        assert decision.rejected
    elif expected == "review_required":
        assert decision.verdict is AuthorityVerdict.REQUIRE_HUMAN_REVIEW
        assert decision.requires_human_review
    else:
        raise AssertionError(f"{case['case_id']} unexpected scope decision {expected}")
    for code in case["expected_reason_codes"]:
        assert code in decision.reason_codes


def _source() -> dict[str, Any]:
    return {
        "path": IN_SCOPE,
        "content": (
            "const deprecatedTitle = title;\n"
            "<label>Goal</label>\n"
            "export const GoalForm = () => null;\n"
        ),
        "component_id": "comp:goal-form",
        "editable": True,
    }


def _loop_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run:adversarial",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective_id": "objective:accessible-name",
        "objective": "Ensure the goal form has an accessible name.",
        "source_revision": REVISION,
        "canonical_branch": "main",
        "canonical_revision": REVISION,
        "canonical_porcelain": "",
        "attempt": 1,
        "context_pack": {
            "pack_id": "pack:adversarial",
            "application_id": "app:agent-supervisor",
            "screen_id": "screen:agent-supervisor",
            "objective": "Repair the goal form label.",
            "raw_sources": [_source()],
            "analysis_classification": "exact",
            "verification_status": "unverified",
            "escalation_conditions": [],
            "formal_invariant_failures": [],
            "acceptance_criteria": ["crit:accessible-name"],
        },
        "transformations": [
            {
                "kind": "label",
                "path": IN_SCOPE,
                "find": "<label>Goal</label>",
                "replace": '<label for="goal">Goal</label>',
                "expected_count": 1,
                "interface": DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE,
                "schema_version": (
                    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
                    "deterministic-transformation@1"
                ),
            }
        ],
        "intended_file_paths": [IN_SCOPE],
        "intended_component_ids": ["comp:goal-form"],
        "acceptance_criteria": ["crit:accessible-name"],
        "expected_test_ids": ["test:goal-form-a11y"],
        "expected_screenshot_ids": ["screenshot:keyboard-desktop"],
        "state_effect_ids": ["state:ready"],
        "analysis_classification": "exact",
        "route_kind": "deterministic_transform",
        "baseline": {"violations": ["missing-name"]},
        "baseline_metrics": {
            "accessible_name_coverage": 0.4,
            "critical_accessibility_violations": 2,
            "confirmation_bypass_count": 1,
            "interaction_step_count": 6,
        },
        "candidate_metrics": {
            "accessible_name_coverage": 1.0,
            "critical_accessibility_violations": 0,
            "confirmation_bypass_count": 0,
            "interaction_step_count": 3,
        },
        "objective_metric_id": "accessible_name_coverage",
        "impact": {"affected_component_ids": ["comp:goal-form"]},
        "invalidation": {
            "plan_id": "invalidate:adversarial",
            "fallback_triggered": False,
        },
        "application": {
            "applied": True,
            "promoted": False,
            "disposition": "applied",
            "reason_codes": ["applied"],
        },
        "check_execution": {
            "acceptance_blocked": False,
            "executed_check_ids": ["check:direct-tests"],
            "failed_required_check_ids": [],
            "fallback_applied": False,
        },
        "evidence": {
            "visual_receipt_ids": ["visual:goal-form"],
            "accessibility_receipt_ids": ["a11y:goal-form"],
            "interaction_receipt_ids": ["interaction:goal-form"],
            "constraint_receipt_ids": ["constraint:goal-form"],
            "invalidation_plan_id": "invalidate:adversarial",
            "context_pack_id": "pack:adversarial",
        },
        "hard_gates": {
            "accessibility_regression": False,
            "security_regression": False,
            "confirmation_regression": False,
        },
    }
    payload.update(overrides)
    return payload


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = sealed_git_environment()
    env.update(
        {
            "GIT_AUTHOR_NAME": "vgo-test",
            "GIT_AUTHOR_EMAIL": "vgo-test@example.invalid",
            "GIT_COMMITTER_NAME": "vgo-test",
            "GIT_COMMITTER_EMAIL": "vgo-test@example.invalid",
        }
    )
    completed = subprocess.run(
        [HOST_GIT_EXECUTABLE, "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {completed.stderr or completed.stdout}"
        )
    return completed


def _require_host_git() -> None:
    if not (Path(HOST_GIT_EXECUTABLE).is_file() and os.access(HOST_GIT_EXECUTABLE, os.X_OK)):
        raise AssertionError(
            "dependency gap: sealed validation PATH must provide /usr/bin/git"
        )


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    worktrees = tmp_path / "worktrees"
    repo.mkdir()
    worktrees.mkdir()
    target = repo / "swissknife" / "web" / "js" / "apps"
    target.mkdir(parents=True)
    (target / "agent-supervisor.js").write_text(ORIGINAL, encoding="utf-8")
    init = _git(repo, "init", "-b", "main", check=False)
    if init.returncode != 0:
        _git(repo, "init")
        _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "add", IN_SCOPE)
    _git(repo, "commit", "-m", "baseline")
    return repo, worktrees


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "--verify", "HEAD").stdout.strip()


def _assert_canonical_untouched(repo: Path, revision: str) -> None:
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    assert _head(repo) == revision
    assert _git(repo, "status", "--porcelain=v1", "-uall").stdout == ""
    assert (repo / IN_SCOPE).read_text(encoding="utf-8") == ORIGINAL


def _modify_diff(*, path: str, marker: str) -> str:
    if path == IN_SCOPE:
        return (
            f"--- a/{IN_SCOPE}\n"
            f"+++ b/{IN_SCOPE}\n"
            "@@ -1,1 +1,2 @@\n"
            f" {ORIGINAL.rstrip(chr(10))}\n"
            f"+{marker}\n"
        )
    return (
        f"--- a/{IN_SCOPE}\n"
        f"+++ b/{IN_SCOPE}\n"
        "@@ -1,1 +1,1 @@\n"
        f"-{ORIGINAL.rstrip(chr(10))}\n"
        f"+{UPDATED.rstrip(chr(10))}\n"
        f"--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1,1 @@\n"
        f"+{marker}\n"
    )


class _FailApplyRunner(HostGitRunner):
    def run(self, argv, *, cwd, input_text=None):  # type: ignore[no-untyped-def]
        if argv and argv[0] == "apply":
            return HostGitResult(
                argv=(HOST_GIT_EXECUTABLE, *tuple(argv)),
                returncode=1,
                stdout="",
                stderr="injected apply failure",
            )
        return super().run(argv, cwd=cwd, input_text=input_text)


def test_suite_is_closed_compact_and_non_executing(
    suite: dict[str, Any], cases: tuple[dict[str, Any], ...]
) -> None:
    assert suite["interface"] == "GuiAdversarialProposalSuite@1"
    assert suite["schema_version"] == "gui-adversarial-proposal-suite/v1"
    assert suite["task_id"] == "VGO-086"
    assert suite["application_id"] == "app:agent-supervisor"
    assert "never execute" in suite["conflict_policy"]
    blob = FIXTURE_PATH.read_text(encoding="utf-8")
    assert FIXTURE_PATH.stat().st_size < 12000
    lowered = blob.lower()
    for marker in _FORBIDDEN_EXECUTION_MARKERS:
        assert marker not in lowered
    assert GUI_PATCH_SCOPE_GATE_INTERFACE == "GuiPatchScopeGate@1"
    assert GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE == "GuiIsolatedWorktreeExecutor@1"
    assert GUI_ACCEPTANCE_DECISION_INTERFACE == "GuiImprovementDecision@1"
    assert GUI_ACCEPTANCE_AUTHORITY_INTERFACE == "GuiAcceptanceAuthority@1"
    assert default_patch_scope_gate().interface == GUI_PATCH_SCOPE_GATE_INTERFACE
    assert default_isolated_worktree_executor().interface == (
        GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE
    )
    _require_host_git()


@pytest.mark.parametrize("case_id", [
    "adv:out-of-scope",
    "adv:test-deletion",
    "adv:arbitrary-html",
    "adv:authority-weakening",
    "adv:credential-weakening",
    "adv:check-weakening",
    "adv:unrelated-app",
    "adv:excess-size",
    "adv:unverified-binding",
])
def test_scope_gate_rejects_or_reviews_forbidden_proposals(
    cases: tuple[dict[str, Any], ...], case_id: str
) -> None:
    case = _by_id(cases, case_id)
    decision = _evaluate_scope(case)
    _assert_scope_decision(case, decision)


def test_confirmation_weakening_never_auto_accepts(
    cases: tuple[dict[str, Any], ...]
) -> None:
    case = _by_id(cases, "adv:click-confirmation-bypass")
    decision = _evaluate_scope(case)
    assert decision.allowed is False
    assert PatchScopeReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value in (
        decision.reason_codes
    ) or "confirmation_weakening" in decision.reason_codes


def test_loop_hard_gates_block_a11y_and_confirmation_regressions(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    for case_id, gate, metric_key in (
        (
            "adv:aesthetic-a11y-regression",
            "accessibility_regression",
            ImprovementReasonCode.ACCESSIBILITY_REGRESSION.value,
        ),
        (
            "adv:click-confirmation-bypass",
            "confirmation_regression",
            ImprovementReasonCode.CONFIRMATION_REGRESSION.value,
        ),
    ):
        case = _by_id(cases, case_id)
        optimizer = default_verified_gui_optimizer(tmp_path / case_id)
        run = optimizer.improve(
            _loop_request(
                run_id=f"run:{case_id}",
                hard_gates={gate: True},
                candidate_metrics={
                    "accessible_name_coverage": 1.0,
                    "critical_accessibility_violations": (
                        3 if gate == "accessibility_regression" else 0
                    ),
                    "confirmation_bypass_count": (
                        4 if gate == "confirmation_regression" else 0
                    ),
                    "interaction_step_count": 2,
                },
            )
        )
        assert run.decision.interface == GUI_ACCEPTANCE_DECISION_INTERFACE
        assert run.decision.kind is ImprovementDecisionKind.REJECT
        assert run.decision.accepted is False
        assert metric_key in run.decision.reason_codes
        assert run.promoted is False
        assert run.canonical_mutated is False
        assert run.receipt is not None
        decoded = GuiImprovementReceipt.from_dict(dict(run.receipt))
        assert decoded.interface == GUI_IMPROVEMENT_RECEIPT_INTERFACE
        assert decoded.decision.value == "reject"
        assert decoded.rejection_reasons
        for code in case["expected_reason_codes"]:
            assert code in run.decision.reason_codes


def test_check_weakening_blocks_acceptance(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    case = _by_id(cases, "adv:check-weakening")
    optimizer = default_verified_gui_optimizer(tmp_path / "check-weakening")
    run = optimizer.improve(
        _loop_request(
            run_id="run:check-weakening",
            check_execution={
                "acceptance_blocked": True,
                "executed_check_ids": ["check:direct-tests"],
                "failed_required_check_ids": ["check:direct-tests"],
                "fallback_applied": True,
            },
        )
    )
    assert run.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.REQUIRED_CHECK_FAILED.value in run.decision.reason_codes
    assert run.receipt is not None
    assert case["never_auto_accept"] is True


def test_acceptance_authority_blocks_a11y_security_and_confirmation() -> None:
    authority = GuiAcceptanceAuthority()
    a11y = authority.evaluate({"accessibility_regression": True})
    assert a11y.verdict is AuthorityVerdict.REJECT
    assert AuthorityReasonCode.ACCESSIBILITY_REGRESSION.value in a11y.reason_codes
    security = authority.evaluate({"security_regression": True})
    assert security.verdict is AuthorityVerdict.REJECT
    assert AuthorityReasonCode.SECURITY_REGRESSION.value in security.reason_codes
    confirmation = authority.evaluate(
        {
            "confirmation_required": True,
            "confirmation_granted": False,
            "intended_action_id": "action:dispatch",
        }
    )
    assert confirmation.verdict is AuthorityVerdict.REJECT
    assert AuthorityReasonCode.CONFIRMATION_REQUIRED.value in confirmation.reason_codes
    assert a11y.allowed is False
    assert security.allowed is False
    assert confirmation.allowed is False


def test_interrupted_run_then_resume_rejects_regression(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    case = _by_id(cases, "adv:interrupted-optimization")
    optimizer = default_verified_gui_optimizer(tmp_path / "interrupt")
    halted = optimizer.improve(
        _loop_request(
            run_id="run:interrupt",
            halt_after_phase=case["halt_after_phase"],
            hard_gates={"accessibility_regression": True},
        )
    )
    assert halted.decision.kind is ImprovementDecisionKind.PENDING
    assert halted.status is RunStatus.INTERRUPTED
    assert halted.receipt is None
    assert halted.promoted is False
    assert halted.canonical_mutated is False
    resumed = optimizer.improve(
        _loop_request(
            run_id="run:interrupt",
            resume=True,
            hard_gates={"accessibility_regression": True},
        )
    )
    assert resumed.run_id == "run:interrupt"
    assert resumed.decision.kind is ImprovementDecisionKind.REJECT
    assert ImprovementReasonCode.ACCESSIBILITY_REGRESSION.value in (
        resumed.decision.reason_codes
    )
    assert resumed.receipt is not None
    assert resumed.canonical_mutated is False


def test_deterministic_rerun_returns_same_terminal_identity(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    case = _by_id(cases, "adv:deterministic-rerun")
    optimizer = default_verified_gui_optimizer(tmp_path / "rerun")
    first = optimizer.improve(
        _loop_request(
            run_id="run:rerun",
            hard_gates={case["hard_gate"]: True},
        )
    )
    second = optimizer.improve(
        _loop_request(
            run_id="run:rerun",
            hard_gates={case["hard_gate"]: True},
        )
    )
    assert first.decision.kind is ImprovementDecisionKind.REJECT
    assert first.terminal_receipt_cid == second.terminal_receipt_cid
    assert first.receipt == second.receipt
    assert second.status is RunStatus.COMPLETED
    assert first.canonical_mutated is False
    assert second.canonical_mutated is False


def test_stale_journal_recovery_rejects(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    case = _by_id(cases, "adv:stale-journal")
    optimizer = default_verified_gui_optimizer(tmp_path / "stale")
    halted = optimizer.improve(
        _loop_request(run_id="run:stale", halt_after_phase="context_pack")
    )
    assert halted.status is RunStatus.INTERRUPTED
    stale = optimizer.improve(
        _loop_request(
            run_id="run:stale",
            resume=True,
            source_revision=STALE_REVISION,
            canonical_revision=STALE_REVISION,
        )
    )
    assert stale.decision.kind is ImprovementDecisionKind.REJECT
    assert JournalReasonCode.REVISION_MISMATCH.value in stale.decision.reason_codes
    for code in case["expected_reason_codes"]:
        assert code in stale.decision.reason_codes
    assert stale.canonical_mutated is False
    journal = optimizer.journal
    opened = journal.open_run(
        run_id="run:stale-worktree",
        application_id="app:agent-supervisor",
        screen_id="screen:agent-supervisor",
        objective_id="objective:accessible-name",
        source_revision=REVISION,
        canonical_branch="main",
        canonical_revision=REVISION,
        worktree_path="/isolated/worktree-a",
        worktree_revision=REVISION,
        worktree_lease_id="lease:vgo-1",
    )
    journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.ISOLATED_WORKTREE,
        effect_id="effect:worktree:apply",
        payload={"applied": True, "promoted": False},
    )
    foreign = journal.decide_resume(
        run_id="run:stale-worktree",
        source_revision=REVISION,
        canonical_branch="main",
        canonical_revision=REVISION,
        worktree_path="/isolated/worktree-b",
        worktree_revision=REVISION,
        worktree_lease_id="lease:vgo-1",
        process_alive=False,
    )
    assert foreign.action is ResumeAction.REJECT
    assert JournalReasonCode.FOREIGN_WORKTREE.value in foreign.reason_codes
    missing = journal.decide_resume(
        run_id="run:stale-worktree",
        source_revision=REVISION,
        canonical_branch="main",
        canonical_revision=REVISION,
        process_alive=False,
    )
    assert missing.action is ResumeAction.REJECT
    assert JournalReasonCode.STALE_WORKTREE.value in missing.reason_codes


def test_every_case_rejects_or_reviews_with_reason_codes(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    for case in cases:
        surface = case["surface"]
        if surface == "scope_gate":
            decision = _evaluate_scope(case)
            assert decision.allowed is False
            assert decision.verdict is not AuthorityVerdict.ALLOW
            for code in case["expected_reason_codes"]:
                assert code in decision.reason_codes
            continue
        if surface == "loop" and case["expected_decision"] != "pending":
            gates = {case["hard_gate"]: True} if case.get("hard_gate") else {
                "accessibility_regression": True
            }
            run = default_verified_gui_optimizer(tmp_path / case["case_id"]).improve(
                _loop_request(run_id=f"run:{case['case_id']}", hard_gates=gates)
            )
            assert run.decision.accepted is False
            assert run.decision.kind is ImprovementDecisionKind.REJECT
            assert run.receipt is not None
            assert run.canonical_mutated is False
            decoded = GuiImprovementReceipt.from_dict(dict(run.receipt))
            assert decoded.decision.value == "reject"
            assert decoded.rejection_reasons
            for code in case["expected_reason_codes"]:
                assert code in run.decision.reason_codes
            continue
        assert case["expected_decision"] != "accepted"
        assert case["never_auto_accept"] is True


def test_rejected_and_interrupted_patches_leave_canonical_untouched(
    tmp_path: Path, cases: tuple[dict[str, Any], ...]
) -> None:
    _require_host_git()
    repo, worktrees = _init_repo(tmp_path)
    before = _head(repo)
    isolation = _by_id(cases, "adv:rejected-patch-isolation")
    html = _by_id(cases, "adv:arbitrary-html")
    executor = default_isolated_worktree_executor()
    rejected = executor.apply(
        {
            "repository_path": str(repo),
            "worktree_parent": str(worktrees),
            "source_revision": before,
            "proposal": _proposal(),
            "diff_text": _modify_diff(
                path=isolation["path"], marker=isolation["marker"]
            ),
            "observation": {
                "touched_component_ids": ["comp:goal-form"],
                "touched_state_effect_ids": ["state:ready"],
                "touched_test_ids": [],
                "touched_screenshot_ids": [],
                "application_ids": ["app:agent-supervisor"],
                "action_binding_ids": [],
                "action_contract_evidence": [],
                "visual_effect_observed": False,
                "unresolved_paths": [],
            },
            "invalidation": _invalidation(),
            "task_id": "VGO-086",
            "attempt": 1,
            "lane_id": "vgo-lane-0",
        }
    )
    assert rejected.disposition is ApplicationDisposition.REJECTED
    assert rejected.applied is False
    assert rejected.promoted is False
    assert rejected.cleanup_state is CleanupState.NEVER_CREATED
    assert WorktreeExecutorReasonCode.SCOPE_REJECTED.value in rejected.reason_codes
    for code in isolation["expected_reason_codes"]:
        assert code in rejected.reason_codes
    _assert_canonical_untouched(repo, before)
    html_receipt = executor.apply(
        {
            "repository_path": str(repo),
            "worktree_parent": str(worktrees),
            "source_revision": before,
            "proposal": _proposal(),
            "diff_text": _modify_diff(path=IN_SCOPE, marker=html["marker"]),
            "observation": {
                "touched_component_ids": ["comp:goal-form"],
                "touched_state_effect_ids": ["state:ready"],
                "touched_test_ids": [],
                "touched_screenshot_ids": [],
                "application_ids": ["app:agent-supervisor"],
                "action_binding_ids": [],
                "action_contract_evidence": [],
                "visual_effect_observed": False,
                "unresolved_paths": [],
            },
            "invalidation": _invalidation(),
            "task_id": "VGO-086",
            "attempt": 1,
            "lane_id": "vgo-lane-0",
        }
    )
    assert html_receipt.applied is False
    assert html_receipt.promoted is False
    assert PatchScopeReasonCode.ARBITRARY_HTML_EXECUTION.value in (
        html_receipt.reason_codes
    )
    _assert_canonical_untouched(repo, before)
    interrupted = GuiIsolatedWorktreeExecutor(git_runner=_FailApplyRunner()).apply(
        {
            "repository_path": str(repo),
            "worktree_parent": str(worktrees),
            "source_revision": before,
            "proposal": _proposal(),
            "diff_text": (
                f"--- a/{IN_SCOPE}\n"
                f"+++ b/{IN_SCOPE}\n"
                "@@ -1,1 +1,1 @@\n"
                f"-{ORIGINAL.rstrip(chr(10))}\n"
                f"+{UPDATED.rstrip(chr(10))}\n"
            ),
            "observation": {
                "touched_component_ids": ["comp:goal-form"],
                "touched_state_effect_ids": ["state:ready"],
                "touched_test_ids": [],
                "touched_screenshot_ids": [],
                "application_ids": ["app:agent-supervisor"],
                "action_binding_ids": [],
                "action_contract_evidence": [],
                "visual_effect_observed": True,
                "unresolved_paths": [],
            },
            "invalidation": _invalidation(),
            "task_id": "VGO-086",
            "attempt": 1,
            "lane_id": "vgo-lane-0",
        }
    )
    assert interrupted.disposition is ApplicationDisposition.INTERRUPTED
    assert interrupted.applied is False
    assert interrupted.promoted is False
    _assert_canonical_untouched(repo, before)
    assert list(worktrees.iterdir()) == []


def test_validation_environment_is_sealed() -> None:
    assert _VALIDATION_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    env = sealed_git_environment()
    assert env["PATH"] == _VALIDATION_PATH
    assert "GIT_DIR" not in env
    assert FIXTURE_PATH.is_file()
    assert FIXTURE_PATH.name == "adversarial-proposals.json"
    _require_host_git()
