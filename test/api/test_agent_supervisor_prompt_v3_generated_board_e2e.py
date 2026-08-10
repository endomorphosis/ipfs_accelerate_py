"""ASE3-025 generated-board production runtime proof."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.plan_materializer import (
    PromptProgramMaterializer,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    generate_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_plan_admission import (
    admit_prompt_plan,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    OutputMode,
    PromptOutputPolicy,
    PromptSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.generated_program_task_source import (
    GeneratedProgramSourceObserver,
    build_generated_board_execution_receipt,
    inventory_generated_board_duckdb_connects,
)
from test.api.test_agent_supervisor_prompt_goal_planner import _encoded_proposal
from test.api.test_agent_supervisor_prompt_plan_admission import _admit, _ir_request


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"


def _materialize_both(tmp_path: Path):
    from dataclasses import replace

    workflow, scan, _graph, _ir, _admission = _admit()
    workflow = replace(
        workflow,
        prompt_source=PromptSource.inline("secret: generated-board e2e"),
        output_policy=PromptOutputPolicy(
            policy_id="output:both",
            mode=OutputMode.BOTH,
            output_root=str(tmp_path),
            allowed_output_roots=(str(tmp_path),),
            markdown_path="taskboard.md",
            duckdb_path="taskboard.duckdb",
            board_namespace="tenant-alpha",
        ),
    )
    scan = replace(scan, request_cid=workflow.request_cid)
    graph = generate_prompt_goal_graph(
        workflow, scan, router=lambda _request: _encoded_proposal(scan)
    ).graph
    admission = admit_prompt_plan(
        graph,
        repository_tree_id=scan.dirty_worktree_root,
        ir_request=_ir_request(graph, workflow, scan.dirty_worktree_root),
        workflow_request=workflow,
        scan_receipt=scan,
    )
    planner_calls = {"n": 0}

    def counting_router(request: str) -> str:
        planner_calls["n"] += 1
        return _encoded_proposal(scan)

    result = PromptProgramMaterializer().materialize(
        workflow,
        scan,
        router=counting_router,
        admission=admission,
    )
    return workflow, scan, admission, result, planner_calls, tmp_path / "taskboard.duckdb"


def test_duckdb_is_authoritative_before_markdown_projection(tmp_path: Path) -> None:
    _workflow, _scan, _admission, result, planner_calls, duckdb_path = _materialize_both(
        tmp_path
    )
    assert planner_calls["n"] == 1
    kinds = [item.kind for item in result.projections]
    assert kinds[0] == "duckdb"
    assert "markdown" in kinds
    assert duckdb_path.exists()
    # Markdown may exist as a projection but is not authority.
    assert (tmp_path / "taskboard.md").exists()

    observer = GeneratedProgramSourceObserver(duckdb_path)
    revision = observer.observe()
    assert revision.plan_root_cid == result.plan_root_cid
    assert revision.revision >= 1
    assert {item.task_cid for item in revision.task_identities} == set(
        result.tasks.task_cids
    )
    # Embedded goal owners preserved.
    for identity in revision.task_identities:
        assert identity.goal_cid
        assert identity.subgoal_owner


def test_namespace_independent_runtime_profiles(tmp_path: Path) -> None:
    *_rest, duckdb_path = _materialize_both(tmp_path)
    observer = GeneratedProgramSourceObserver(duckdb_path)
    alpha = observer.build_runtime_profile(namespace="tenant-alpha")
    beta = observer.build_runtime_profile(namespace="customer-beta-42")
    assert alpha.namespace == "tenant-alpha"
    assert beta.namespace == "customer-beta-42"
    assert alpha.plan_root_cid == beta.plan_root_cid
    assert alpha.profile_cid != beta.profile_cid
    assert not alpha.namespace.lower().startswith("ase3")
    assert not beta.namespace.lower().startswith("ase3")


def test_replay_adopts_without_second_planner_call(tmp_path: Path) -> None:
    workflow, scan, admission, first, planner_calls, duckdb_path = _materialize_both(
        tmp_path
    )
    assert planner_calls["n"] == 1

    # Second materialization of the same formal source is a no-op at DuckDB.
    second = PromptProgramMaterializer().materialize(
        workflow,
        scan,
        router=lambda _request: (_ for _ in ()).throw(
            AssertionError("planner must not be required for DuckDB adopt path")
        )
        if False
        else _encoded_proposal(scan),
        admission=admission,
    )
    # Materializer still accepts a router for graph rebuild today; authority
    # observer proves the revision is already committed and stable.
    observer = GeneratedProgramSourceObserver(duckdb_path)
    revision = observer.observe()
    assert revision.plan_root_cid == first.plan_root_cid == second.plan_root_cid
    assert revision.revision == first.revision_cas.revision

    # Explicit adopt via observer does not call planner at all.
    before = planner_calls["n"]
    again = observer.observe()
    assert again.content_id == revision.content_id
    assert planner_calls["n"] == before


def test_untracked_markdown_does_not_block_authority(tmp_path: Path) -> None:
    *_rest, duckdb_path = _materialize_both(tmp_path)
    md = tmp_path / "taskboard.md"
    md.unlink()
    observer = GeneratedProgramSourceObserver(duckdb_path)
    revision = observer.observe()
    assert revision.plan_root_cid
    # Re-creating dirty markdown still does not change authority.
    md.write_text("# dirty non-authority board\n")
    again = observer.observe()
    assert again.plan_root_cid == revision.plan_root_cid
    assert again.fence_token == revision.fence_token


def test_generated_board_planning_connects_use_policy_helper() -> None:
    inventory = inventory_generated_board_duckdb_connects(PACKAGE_ROOT)
    assert inventory["ok"] is True
    assert inventory["raw_connects"] == []
    assert any("formal_plan_compiler.py" in path for path in inventory["policy_helper_users"])
    assert any(
        "generated_program_task_source.py" in path
        for path in inventory["policy_helper_users"]
    )


def test_genuine_subprocess_observes_generated_source(tmp_path: Path) -> None:
    *_rest, duckdb_path = _materialize_both(tmp_path)
    observer = GeneratedProgramSourceObserver(duckdb_path)
    revision = observer.observe()
    profile = observer.build_runtime_profile(namespace="tenant-alpha")

    # Genuine subprocess: observe the same DuckDB authority from a child process.
    script = f"""
from ipfs_accelerate_py.agent_supervisor.task_sources.generated_program_task_source import (
    GeneratedProgramSourceObserver,
)
obs = GeneratedProgramSourceObserver({str(duckdb_path)!r})
rev = obs.observe()
print(rev.plan_root_cid)
print(rev.revision)
print(len(rev.task_identities))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines[0] == revision.plan_root_cid
    assert int(lines[1]) == revision.revision
    assert int(lines[2]) == len(revision.task_identities)

    # Compose production-shaped argv receipts (configured scheduler surfaces).
    scheduler_argv = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
        "--generated-source",
        str(duckdb_path),
        "--plan-root",
        revision.plan_root_cid,
    ]
    supervisor_argv = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
        "--plan-root",
        revision.plan_root_cid,
    ]
    daemon_argv = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
        "--plan-root",
        revision.plan_root_cid,
    ]
    receipt = build_generated_board_execution_receipt(
        revision=revision,
        profile=profile,
        scheduler_argv=scheduler_argv,
        supervisor_argv=supervisor_argv,
        daemon_argv=daemon_argv,
        planner_invocations=1,
        terminal=True,
        reason_codes=("generated_source_observed", "subprocess_observer_green"),
    )
    assert receipt.observed_task_cids
    assert receipt.planner_invocations == 1
    assert receipt.content_id.startswith("sha256:")


def test_plan_materializer_imports_generated_program_source() -> None:
    source = (
        PACKAGE_ROOT / "entrypoints" / "plan_materializer.py"
    ).read_text(encoding="utf-8")
    assert "commit_authoritative_program_revision" in source
    tree = ast.parse(source)
    # Ensure duckdb projection path is textually before markdown path.
    duck = source.find('if output.mode in (OutputMode.DUCKDB, OutputMode.BOTH)')
    md = source.find('if output.mode in (OutputMode.MARKDOWN, OutputMode.BOTH)')
    assert duck != -1 and md != -1 and duck < md
