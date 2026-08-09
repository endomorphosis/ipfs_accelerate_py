from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.plan_materializer import (
    PlanMaterializationAdmissionError,
    PromptProgramMaterializer,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
    parse_markdown_task_source,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import generate_prompt_goal_graph
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_plan_admission import admit_prompt_plan
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    OutputMode,
    PromptOutputPolicy,
    PromptSource,
)
from test.api.test_agent_supervisor_prompt_goal_planner import _encoded_proposal
from test.api.test_agent_supervisor_prompt_plan_admission import _admit, _ir_request


def test_materializer_projects_an_admitted_canonical_plan_without_prompt_body(
    tmp_path: Path,
) -> None:
    workflow, scan, _graph, _ir_request_value, _admission = _admit()
    workflow = replace(workflow, prompt_source=PromptSource.inline("secret: do not persist"))
    scan = replace(scan, request_cid=workflow.request_cid)
    body = workflow.prompt_source.transient_body
    assert body is not None
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
    assert admission.admitted

    result = PromptProgramMaterializer().materialize(
        workflow,
        scan,
        router=lambda _request: _encoded_proposal(scan),
        admission=admission,
        markdown_path=tmp_path / "taskboard.md",
    )

    projection = result.projections[0]
    snapshot = parse_markdown_task_source((tmp_path / "taskboard.md").read_text())
    assert result.goals.root_goal_cid == admission.admitted_graph.root_goal.goal_cid
    assert result.tasks.task_cids == admission.receipt.final_task_cids
    assert projection.kind == "markdown"
    assert snapshot.plan_root == result.plan_root_cid
    assert len(snapshot.tasks) == len(result.tasks.task_cids)
    assert body.decode() not in (tmp_path / "taskboard.md").read_text()
    assert body.decode() not in result.to_dict().__repr__()


def test_rejected_admission_never_creates_a_projection(tmp_path: Path) -> None:
    workflow, scan, _graph, _ir_request, rejected = _admit(security_decision="deny")
    target = tmp_path / "taskboard.md"

    with pytest.raises(PlanMaterializationAdmissionError):
        PromptProgramMaterializer().materialize(
            workflow,
            scan,
            router=lambda _request: _encoded_proposal(scan),
            admission=rejected,
            markdown_path=target,
        )

    assert not target.exists()


def test_both_projections_share_the_admission_published_plan_root(
    tmp_path: Path,
) -> None:
    workflow, scan, _graph, _ir_request_value, _admission = _admit()
    workflow = replace(
        workflow,
        prompt_source=PromptSource.inline("secret: dual projection"),
        output_policy=PromptOutputPolicy(
            policy_id="output:both",
            mode=OutputMode.BOTH,
            output_root=str(tmp_path),
            allowed_output_roots=(str(tmp_path),),
            markdown_path="taskboard.md",
            duckdb_path="taskboard.duckdb",
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

    result = PromptProgramMaterializer().materialize(
        workflow,
        scan,
        router=lambda _request: _encoded_proposal(scan),
        admission=admission,
    )

    assert {item.kind for item in result.projections} == {"markdown", "duckdb"}
    assert {item.plan_root_cid for item in result.projections} == {result.plan_root_cid}
