"""Strict rendered-card contracts for objective backlog refill."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    ObjectiveFinding,
    ObjectiveTaskRenderProfile,
    admit_objective_task_block,
    generate_objective_todos,
    render_task_block,
    split_terms,
)

STRICT_PROFILE = ObjectiveTaskRenderProfile(
    resource_stage="implementation",
    implementation_timeout_seconds=9_000,
    symbolic_first=True,
    llm_context_budget_bytes=32_000,
    estimated_tokens_default=12_000,
    allowed_statuses=("todo", "completed"),
    require_schedulable=True,
)


def _finding(**overrides: object) -> ObjectiveFinding:
    finding = ObjectiveFinding(
        fingerprint="strict-refill-finding",
        goal_id="STRICT-G010",
        title="Implement strict refill target",
        summary="Close strict objective gap",
        priority="P1",
        track="strict-refill",
        missing_evidence=["strict refill evidence"],
        present_evidence={},
        evidence_methods=[],
        objective_path="docs/objectives.md",
        outputs=["src/refill_target.py"],
        validation="python -m pytest -q test/test_refill_target.py",
        bundle_key="strict/refill",
        parallel_lane="strict-refill",
        predicted_files=["src/refill_target.py"],
        changed_paths=["src/refill_target.py"],
        interfaces=["StrictRefill@1"],
        acceptance_subset=["strict refill evidence"],
        preconditions=["STRICT-G010 is schedulable"],
        effects=["strict refill evidence is produced"],
        evidence_subset=["strict refill evidence"],
        resource_class="cpu-medium",
        estimated_tokens=0,
        embedding_query="strict objective refill",
    )
    return replace(finding, **overrides)


def _render(
    finding: ObjectiveFinding,
    *,
    protected_output_paths: tuple[str, ...] = (),
    discovery_output_path: str = "data/agent_supervisor/discovery",
    evidence_outputs: tuple[str, ...] = (),
) -> str:
    return render_task_block(
        task_id="STRICT-001",
        finding=finding,
        discovery_path=Path("/tmp/strict-001-discovery.md"),
        discovery_output_path=discovery_output_path,
        evidence_outputs=evidence_outputs,
        board_namespace="strict-objective-board-v1",
        protected_output_paths=protected_output_paths,
        task_render_profile=STRICT_PROFILE,
    )


def _strict_seed_board() -> str:
    return """# Strict Objective Taskboard

## STRICT-000 Seed strict refill policy

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Priority: P0
- Goal id: STRICT-G000
- Outputs: docs/strict-seed.md
- Validation: python -m pytest -q test/test_strict_seed.py
- Board namespace: strict-objective-board-v1
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 10000
- Implementation timeout seconds: 6000
- Predicted files: docs/strict-seed.md
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: The strict seed contract remains valid.
"""


def test_strict_profile_renders_complete_canonical_execution_metadata() -> None:
    block = _render(_finding())

    metadata = admit_objective_task_block(
        block,
        task_id="STRICT-001",
        task_render_profile=STRICT_PROFILE,
    )

    assert metadata["status"] == "todo"
    assert metadata["is schedulable"] == "true"
    assert metadata["resource stage"] == "implementation"
    assert metadata["estimated tokens"] == "12000"
    assert metadata["implementation timeout seconds"] == "9000"
    assert metadata["symbolic first"] == "true"
    assert metadata["llm context budget bytes"] == "32000"
    for field in (
        "Resource stage",
        "Implementation timeout seconds",
        "Symbolic first",
        "LLM context budget bytes",
    ):
        assert block.count(f"- {field}:") == 1


def test_protected_control_prefixes_are_read_only_on_every_owned_surface() -> None:
    taskboard = "docs/architecture/strict.todo.md"
    discovery = "data/agent_supervisor/strict/state/discovery"
    taskboard_descendant = f"{taskboard}/forged-child"
    discovery_descendant = f"{discovery}/receipt.json"
    evidence_descendant = f"{discovery}/evidence.json"
    taskboard_sibling = f"{taskboard}.backup"
    finding = _finding(
        outputs=[
            "src/refill_target.py",
            taskboard,
            discovery_descendant,
            taskboard_sibling,
        ],
        predicted_files=[
            "src/refill_target.py",
            taskboard_descendant,
            discovery_descendant,
            taskboard_sibling,
        ],
        changed_paths=[
            "src/refill_target.py",
            taskboard,
            discovery_descendant,
            taskboard_sibling,
        ],
    )

    block = _render(
        finding,
        protected_output_paths=(taskboard,),
        discovery_output_path=discovery,
        evidence_outputs=(evidence_descendant, "test/evidence.json"),
    )
    metadata = admit_objective_task_block(
        block,
        task_id="STRICT-001",
        task_render_profile=STRICT_PROFILE,
        protected_output_paths=(taskboard, discovery),
    )

    expected_owned = ["src/refill_target.py", taskboard_sibling]
    assert split_terms(metadata["outputs"]) == expected_owned
    assert split_terms(metadata["predicted files"]) == expected_owned
    assert split_terms(metadata["changed paths"]) == expected_owned
    assert split_terms(metadata["evidence outputs"]) == ["test/evidence.json"]
    assert set(split_terms(metadata["context paths"])) == {
        taskboard,
        taskboard_descendant,
        discovery_descendant,
        evidence_descendant,
    }


def test_rendered_card_admission_rejects_duplicate_metadata_fields() -> None:
    block = _render(_finding())
    duplicated = block.replace(
        "- Symbolic first: true",
        "- Symbolic first: true\n- SYMBOLIC-FIRST: false",
        1,
    )

    with pytest.raises(ValueError, match="duplicate metadata field 'symbolic first'"):
        admit_objective_task_block(
            duplicated,
            task_id="STRICT-001",
            task_render_profile=STRICT_PROFILE,
        )


def test_rendered_card_admission_rejects_an_injected_second_task_heading() -> None:
    with pytest.raises(ValueError, match="exactly one Markdown task heading"):
        _render(_finding(summary="Close strict objective gap\n## OTHER-999 Injected task"))


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../escape.py",
        "/absolute.py",
        "src/*.py",
        ".git/config",
    ),
)
def test_strict_admission_rejects_unsafe_owned_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError, match="outputs contains unsafe path"):
        _render(_finding(outputs=[unsafe_path]))


def test_legacy_rendering_remains_backward_compatible_without_profile() -> None:
    block = render_task_block(
        task_id="LEGACY-001",
        finding=_finding(),
        discovery_path=Path("/tmp/legacy-discovery.md"),
    )

    assert "- Resource stage:" not in block
    assert "- Implementation timeout seconds:" not in block
    assert "- Symbolic first:" not in block
    assert "- LLM context budget bytes:" not in block
    assert "- Estimated tokens: 0" in block


@pytest.mark.parametrize(
    ("finding", "message"),
    (
        (
            _finding(status="blocked"),
            "status is not admitted",
        ),
        (
            _finding(is_schedulable=False),
            "Is schedulable: true",
        ),
    ),
)
def test_strict_profile_rejects_inadmissible_execution_state_before_render(
    finding: ObjectiveFinding,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _render(finding)


def test_live_generation_infers_strict_profile_before_taskboard_append(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objectives.md"
    todo_path = repo / "todo.md"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "bundles"
    objective_path.write_text("# Objective Heap\n", encoding="utf-8")
    todo_path.write_text(_strict_seed_board(), encoding="utf-8")
    finding = _finding(
        outputs=[
            "src/refill_target.py",
            "todo.md/forged-child",
            "data/agent_supervisor/discovery/forged.json",
        ],
        predicted_files=[
            "src/refill_target.py",
            "todo.md/forged-child",
            "data/agent_supervisor/discovery/forged.json",
        ],
        changed_paths=[
            "src/refill_target.py",
            "todo.md/forged-child",
            "data/agent_supervisor/discovery/forged.json",
        ],
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="STRICT-",
        max_findings=1,
        precomputed_findings=[finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert [record.task_id for record in records] == ["STRICT-001"]
    assert records[0].finding.outputs == ["src/refill_target.py"]
    metadata = admit_objective_task_block(
        records[0].task_block,
        task_id="STRICT-001",
        task_render_profile=ObjectiveTaskRenderProfile(
            resource_stage="implementation",
            implementation_timeout_seconds=6000,
            symbolic_first=True,
            llm_context_budget_bytes=24000,
            estimated_tokens_default=10000,
            allowed_statuses=("todo", "completed"),
            require_schedulable=True,
        ),
        protected_output_paths=(
            "todo.md",
            "data/agent_supervisor/discovery",
        ),
    )
    assert metadata["resource stage"] == "implementation"
    assert metadata["implementation timeout seconds"] == "6000"
    assert metadata["symbolic first"] == "true"
    assert metadata["llm context budget bytes"] == "24000"
    assert metadata["estimated tokens"] == "10000"
    assert split_terms(metadata["outputs"]) == ["src/refill_target.py"]
    assert split_terms(metadata["predicted files"]) == ["src/refill_target.py"]
    assert records[0].task_block in todo_path.read_text(encoding="utf-8")


def test_generate_objective_todos_leaves_board_unchanged_on_profile_rejection(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objectives.md"
    todo_path = repo / "todo.md"
    objective_path.write_text("# Objective Heap\n", encoding="utf-8")
    original = "# Agent Todos\n"
    todo_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="status is not admitted"):
        generate_objective_todos(
            repo_root=repo,
            objective_path=objective_path,
            todo_path=todo_path,
            discovery_dir=repo / "data" / "discovery",
            bundle_dir=repo / "data" / "bundles",
            task_prefix="STRICT-",
            max_findings=1,
            precomputed_findings=[_finding(status="blocked")],
            persist_ast_dataset=False,
            write_todo_vector_index=False,
            task_render_profile=STRICT_PROFILE,
        )

    assert todo_path.read_text(encoding="utf-8") == original
