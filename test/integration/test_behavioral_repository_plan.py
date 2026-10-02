"""Native v2 discovery/proof materials through compiler, planner and critic."""
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import behavioral_repository_plan as adapter
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as finite
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import plan_revision_cid
from test.api.test_finite_integer_plan_preview import preview_arguments, TASK_OFFSET, TASK_TYPE
from test.api.test_finite_integer_codebase import finite_tools
from test.integration.test_behavioral_codebase_match import options
from test.integration.test_terminal_codebase_semantic_index import prepared


@pytest.fixture
def arguments(options):
    owner = options["catalog"].index
    prepared = dict(index=owner, expected_head=options["expected_head"],
        repository=options["repository"], scheduler=None, output=options["output"])
    args = preview_arguments(prepared, options["tool_policy"])
    args.update(catalog=options["catalog"], checked_cache=options["checked_cache"],
        semantic_manifest_cid=options["semantic_manifest_cid"])
    binding = adapter.repository_evidence_binding(owner=args["owner"],
        semantic_manifest_cid=args["semantic_manifest_cid"], tool_policy=args["tool_policy"],
        operation_catalog=args["operation_catalog"])
    args["request"] = replace(args["request"], roots=replace(args["request"].roots,
        configuration_root=plan_revision_cid(binding)))
    return args


def test_v2_snapshot_supplies_real_fact_and_residual_to_native_planning(arguments):
    result = adapter.preview_behavioral_repository_plan(**arguments)
    snapshot = result["repository_proof_snapshot"]
    assert snapshot["schema"] == "repository-proof-planning-snapshot@2"
    assert snapshot["model"] == {"enabled": False, "identity": "explicit-model-off@1"}
    assert result["current_facts_count"] == 1
    assert result["selected_task_ids"] == [TASK_OFFSET]
    assert snapshot["full_declared_task_ids"] == sorted([TASK_OFFSET, TASK_TYPE])
    assert snapshot["residuals"] == [finite.OFFSET_STATEMENT_ID]
    assert len(snapshot["eligible_current_fact_roots"]) == 1
    assert all(snapshot["eligible_current_fact_roots"].values())
    assert snapshot["proof_results"]["checked_cache"]["status"] == "refuted"
    assert snapshot["proof_results"]["source_equivalence_proved"] is False
    assert len(snapshot["code_obligations"]["desired_predicates"]) == 2
    assert result["production_admitted"] is result["completion_authority"] is False
    assert {s["stage"] for s in result["preview"]["stage_results"] if s["passed"]} >= {
        "scan", "query", "evidence", "obligation", "candidate", "critique"}
    output = os.environ.get("RPI_BEHAVIORAL_EVIDENCE")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "native-preview.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def test_wrong_configuration_refused_before_owner_observation(arguments, monkeypatch):
    arguments["request"] = replace(arguments["request"], roots=replace(arguments["request"].roots,
        configuration_root=plan_revision_cid({"forged": True})))
    monkeypatch.setattr(adapter, "match_behavioral_intent", lambda **kwargs: pytest.fail("executed unbound match"))
    with pytest.raises(ValueError, match="configuration"):
        adapter.preview_behavioral_repository_plan(**arguments)


def test_source_change_during_live_policy_observation_refuses_plan(arguments):
    calls = []
    def observe(request):
        calls.append(True)
        if len(calls) == 2:
            (arguments["owner"].repository / "calc.py").write_text(
                "def increment(n: int) -> int:\n    return n + 2\n")
        return request.roots
    arguments["policy_observer"] = observe
    with pytest.raises(ValueError):
        adapter.preview_behavioral_repository_plan(**arguments)
    assert len(calls) >= 2


def test_static_shadowing_and_synthetic_facts_have_no_input_route(arguments):
    for key in ("current_roots", "current_facts", "proof_results", "proof_snapshot", "behavioral_match", "model_prediction"):
        with pytest.raises(TypeError):
            adapter.preview_behavioral_repository_plan(**arguments, **{key: {"authority": "proof:increment"}})


def test_missing_discovery_row_cannot_produce_a_plan(arguments):
    arguments["catalog"]._cx.execute("DELETE FROM intent_codebase.selectors WHERE path='calc.py'")
    with pytest.raises(ValueError, match="selector"):
        adapter.preview_behavioral_repository_plan(**arguments)
