"""ASE-011 reusable goal/task/profile plan lint tests."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.inference_explain import (
    INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.plan_lint import (
    PLAN_LINT_REQUIREMENT_ID,
    REQUIRED_PROFILE_FIELDS,
    PlanLintError,
    PlanLintKind,
    PlanLintReport,
    SupervisorPlanDocument,
    lint_plan,
    lint_supervisor_plan,
)

PROMPT_BODY = "Do not leak this plan prompt body into lint findings."


def _goal(
    goal_id: str,
    *,
    parent: str = "",
    depends_on: tuple[str, ...] = (),
    title: str = "",
    acceptance: str = "Done when tests pass",
) -> dict[str, object]:
    return {
        "goal_id": goal_id,
        "title": title or f"Goal {goal_id}",
        "acceptance": acceptance,
        "parent": parent,
        "depends_on": list(depends_on),
    }


def _task(
    task_id: str,
    *,
    goal_id: str = "G1",
    depends_on: tuple[str, ...] = (),
    predicted_files: tuple[str, ...] = (),
    validation_commands: tuple[str, ...] = (
        "python -m pytest test/api/test_example.py -q",
    ),
    outputs: tuple[str, ...] = (),
    title: str = "",
    acceptance: str = "Acceptance criteria met",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "title": title or f"Task {task_id}",
        "goal_id": goal_id,
        "acceptance": acceptance,
        "outputs": list(outputs or (f"out/{task_id}.py",)),
        "predicted_files": list(
            predicted_files or (f"ipfs_accelerate_py/{task_id}.py",)
        ),
        "validation_commands": list(validation_commands),
        "depends_on": list(depends_on),
    }


def _profile(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "profile_name": "local-worktree",
        "mode": "worktree",
        "repository_root": "/home/dev/src/project",
        "state_root": "/home/dev/.local/state/project",
        "run_namespace": "fixture-run",
        "policy_cid": "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "principal_ref": "did:key:local-owner",
        "effect_ceiling_cid": "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "task_source_kind": "dual",
        "task_source_cid": "baguqeeraccccccccccccccccccccccccccccccccccccccccccccc",
        "provider_route": {"selected_provider": "grok"},
        "validation_profile_cid": "baguqeeraddddddddddddddddddddddddddddddddddddddddddddd",
        "worktree_strategy": "isolated",
        "expected_effects": ["inspect_repository", "edit_isolated_worktree"],
        "credential_handles": ["vault:provider-token"],
        "supervisor_argv": ["ipfs-accelerate", "supervisor", "run"],
    }
    values.update(overrides)
    return values


def _clean_plan() -> dict[str, object]:
    return {
        "plan_id": "plan-clean",
        "goals": [
            _goal("G1"),
            _goal("G2", parent="G1", depends_on=("G1",)),
        ],
        "tasks": [
            _task("T1", goal_id="G1", predicted_files=("pkg/a.py",)),
            _task(
                "T2",
                goal_id="G2",
                depends_on=("T1",),
                predicted_files=("pkg/b.py",),
                outputs=("pkg/b.py",),
            ),
        ],
        "profile": _profile(),
    }


def test_requirement_ids_align_with_inference_explain() -> None:
    assert (
        INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID
        == "inference_explain.INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID"
    )
    assert PLAN_LINT_REQUIREMENT_ID.startswith("requirement:")


def test_clean_plan_is_accepted_and_deterministic() -> None:
    plan = _clean_plan()
    first = lint_supervisor_plan(plan)
    second = lint_plan(copy.deepcopy(plan))
    assert isinstance(first, PlanLintReport)
    assert first.accepted is True
    assert first.total_finding_count == 0
    assert first.goal_count == 2
    assert first.task_count == 2
    assert first.profile_present is True
    assert first.content_id == second.content_id
    assert first.to_json(indent=None) == second.to_json(indent=None)
    assert first.requirement_id == INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID
    # Read-only: caller document is unchanged.
    assert plan == _clean_plan()


def test_duplicate_goal_and_task_ids() -> None:
    plan = {
        "plan_id": "plan-dup",
        "goals": [_goal("G1"), _goal("G1", title="Other")],
        "tasks": [
            _task("T1", predicted_files=("a.py",)),
            _task("T1", predicted_files=("b.py",)),
        ],
    }
    report = lint_supervisor_plan(plan)
    kinds = set(report.kinds)
    assert PlanLintKind.DUPLICATE.value in kinds
    codes = {item.code for item in report.findings}
    assert "duplicate_goal_id" in codes
    assert "duplicate_task_id" in codes


def test_unknown_dependencies_and_parents() -> None:
    plan = {
        "plan_id": "plan-unknown",
        "goals": [_goal("G1", parent="G-missing", depends_on=("G-other",))],
        "tasks": [
            _task("T1", goal_id="G-missing", depends_on=("T-missing",)),
        ],
    }
    report = lint_supervisor_plan(plan)
    codes = {item.code for item in report.findings}
    assert "unknown_parent_goal" in codes
    assert "unknown_goal_dependency" in codes
    assert "task_unknown_goal" in codes
    assert "unknown_task_dependency" in codes
    assert PlanLintKind.UNKNOWN.value in report.kinds


def test_cyclic_goal_and_task_graphs() -> None:
    plan = {
        "plan_id": "plan-cyclic",
        "goals": [
            _goal("G1", depends_on=("G2",)),
            _goal("G2", depends_on=("G1",)),
            _goal("G3", parent="G4"),
            _goal("G4", parent="G3"),
        ],
        "tasks": [
            _task("T1", goal_id="G1", depends_on=("T2",), predicted_files=("t1.py",)),
            _task("T2", goal_id="G1", depends_on=("T1",), predicted_files=("t2.py",)),
            _task("T3", goal_id="G1", depends_on=("T3",), predicted_files=("t3.py",)),
        ],
    }
    report = lint_supervisor_plan(plan)
    codes = {item.code for item in report.findings}
    assert "cyclic_goal_dependency" in codes
    assert "cyclic_goal_parent_hierarchy" in codes
    assert "cyclic_task_dependency" in codes
    assert "task_self_dependency" in codes
    assert PlanLintKind.CYCLIC.value in report.kinds


def test_missing_required_metadata() -> None:
    plan = {
        "plan_id": "plan-missing",
        "goals": [
            {"goal_id": "G1"},  # missing title/acceptance
        ],
        "tasks": [
            {
                "task_id": "T1",
                "goal_id": "G1",
                # missing title, acceptance, outputs, predicted_files, validation
            }
        ],
        "profile": {"profile_name": "incomplete"},
    }
    report = lint_supervisor_plan(plan)
    codes = {item.code for item in report.findings}
    assert "goal_title_missing" in codes
    assert "goal_acceptance_missing" in codes
    assert "task_title_missing" in codes
    assert "task_acceptance_missing" in codes
    assert "task_outputs_missing" in codes
    assert "task_predicted_files_missing" in codes
    assert "task_validation_commands_missing" in codes
    for field_name in ("mode", "repository_root", "principal_ref"):
        assert f"profile_{field_name}_missing" in codes
    assert PlanLintKind.MISSING.value in report.kinds


def test_unsafe_validation_and_predicted_paths() -> None:
    plan = {
        "plan_id": "plan-unsafe",
        "goals": [_goal("G1")],
        "tasks": [
            _task(
                "T1",
                validation_commands=("rm -rf /",),
                predicted_files=("../etc/passwd",),
            ),
            _task(
                "T2",
                validation_commands=("python -c 'import os; os.system(1)'",),
                predicted_files=("/abs/path.py",),
            ),
            _task(
                "T3",
                validation_commands=("curl https://evil.example",),
                predicted_files=("ok.py",),
            ),
        ],
    }
    report = lint_supervisor_plan(plan)
    codes = {item.code for item in report.findings}
    assert "unsafe_predicted_file" in codes
    assert any(
        code
        in {
            "shell_metacharacters",
            "unsafe_validation_token",
            "validation_command_not_allowlisted",
            "python_validation_must_use_module_form",
        }
        for code in codes
    )
    assert PlanLintKind.UNSAFE.value in report.kinds
    serialized = report.to_json()
    assert "rm -rf" not in serialized or "unsafe" in serialized
    # Body-free: never echo raw shell payloads as success evidence; findings stay coded.
    assert PROMPT_BODY not in serialized


def test_predicted_file_conflicts_and_profile_completeness() -> None:
    plan = {
        "plan_id": "plan-conflict",
        "goals": [_goal("G1")],
        "tasks": [
            _task("T1", predicted_files=("shared/module.py", "only_t1.py")),
            _task("T2", predicted_files=("shared/module.py", "only_t2.py")),
        ],
        "profile": _profile(
            worktree_strategy="isolated",
            principal_ref="",  # conflicting with worktree
            supervisor_argv=["tool", "--token=sk-abcdefghijklmnopqrstuvwxyz"],
        ),
    }
    report = lint_supervisor_plan(plan)
    codes = {item.code for item in report.findings}
    assert "predicted_file_conflict" in codes
    assert "profile_worktree_without_principal" in codes
    assert (
        "profile_argv_secret_bearing" in codes
        or "profile_forbidden_argv_flag" in codes
    )
    assert PlanLintKind.CONFLICTING.value in report.kinds
    serialized = report.to_json()
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert PROMPT_BODY not in serialized


def test_require_profile_and_document_wrapper() -> None:
    document = SupervisorPlanDocument.from_mapping(
        {
            "plan_id": "plan-doc",
            "goals": [_goal("G1")],
            "tasks": [_task("T1")],
        }
    )
    report = lint_supervisor_plan(document, require_profile=True)
    codes = {item.code for item in report.findings}
    assert "profile_missing" in codes
    assert report.profile_present is False


def test_empty_plan_and_malformed_input() -> None:
    report = lint_supervisor_plan({"plan_id": "empty"})
    assert any(item.code == "plan_empty" for item in report.findings)

    with pytest.raises(PlanLintError) as excinfo:
        lint_supervisor_plan("not-a-plan")
    assert PROMPT_BODY not in str(excinfo.value)
    assert "sk-" not in str(excinfo.value)


def test_findings_bound_to_identities_and_sorted() -> None:
    plan = {
        "plan_id": "plan-order",
        "goals": [
            _goal("G2", depends_on=("G-missing",)),
            _goal("G1", depends_on=("G2",)),
        ],
        "tasks": [
            _task("T2", goal_id="G1", depends_on=("T-missing",)),
            _task("T1", goal_id="G1", depends_on=("T2",)),
        ],
    }
    report = lint_supervisor_plan(plan)
    assert report.findings == tuple(
        sorted(
            report.findings,
            key=lambda item: (
                0 if item.severity.value == "error" else 1,
                item.kind.value,
                item.code,
                item.subject_kind.value,
                item.subject_id,
                item.field_name,
                item.related_ids,
                item.message,
            ),
        )
    )
    for finding in report.findings:
        assert finding.subject_id
        assert finding.finding_id
        assert finding.kind in PlanLintKind
    # Profile required field inventory stays explicit for completeness checks.
    assert "profile_name" in REQUIRED_PROFILE_FIELDS


def test_no_mutation_of_nested_structures() -> None:
    plan = _clean_plan()
    original = copy.deepcopy(plan)
    report = lint_supervisor_plan(plan)
    assert report.accepted is True
    assert plan == original
    # Mutating the caller's plan after lint must not alter the sealed report.
    plan["tasks"][0]["task_id"] = "mutated"
    again = lint_supervisor_plan(original)
    assert again.content_id == report.content_id
