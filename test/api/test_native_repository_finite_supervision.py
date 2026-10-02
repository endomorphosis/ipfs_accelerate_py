"""Cold equivalence compares exact semantics, independent of execution receipts."""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision import finite_successor_semantics


def _match():
    return dict(status="observed", source_cid="source:a", domain_cid="domain:1", domain_inputs=[0],
        eligible_clause_ids=["type", "offset"], residual_clause_ids=[], finite_counterexamples=[],
        query={"source_text": "complete instruction", "contract": {"offset": 2}, "assumptions": []},
        head={"snapshot_cid": "snapshot:a", "generation": 2},
        clause_results=[dict(predicate_id="generation2:offset", statement_id="offset",
                             status="bounded_observed_satisfied", scope="finite", reasons=[])],
        observation=dict(source_sha256="sha256:a", domain_inputs=[0], scope="finite", profile="offset@1",
            observations=[dict(input=0, output=2)], type_clause_satisfied=True, offset_clause_satisfied=True,
            kernel_checked_model_table=True, runtime_observation_coverage_complete=True,
            proof_authority=False, tool_policy_cid="tools:a", result_cid="receipt:a", output="run:a",
            head={"generation": 2}, artifacts={"path": "run:a"}, python_process={"pid": 1},
            lean_certificate={"elapsed": 1}))


def test_cold_generation_and_execution_receipts_remain_distinct():
    incremental = _match()
    cold = deepcopy(incremental)
    cold["head"]["generation"] = 1
    cold["clause_results"][0]["predicate_id"] = "generation1:offset"
    for key in ("result_cid", "head", "artifacts", "python_process", "lean_certificate", "output"):
        cold["observation"][key] = {"distinct_run": True}
    assert cold != incremental
    assert finite_successor_semantics(cold) == finite_successor_semantics(incremental)


@pytest.mark.parametrize("field", ["source", "snapshot", "domain", "query", "outcome", "scope", "proof", "tools", "clause"])
def test_equal_counts_cannot_hide_different_source_requirements_or_observations(field):
    incremental = _match()
    cold = deepcopy(incremental)
    if field == "source": cold["source_cid"] = "source:b"
    elif field == "snapshot": cold["head"]["snapshot_cid"] = "snapshot:b"
    elif field == "domain": cold["domain_inputs"] = [1]
    elif field == "query": cold["query"]["assumptions"] = ["new assumption"]
    elif field == "outcome": cold["observation"]["observations"][0]["output"] = 3
    elif field == "scope": cold["observation"]["scope"] = "universal"
    elif field == "proof": cold["observation"]["kernel_checked_model_table"] = False
    elif field == "tools": cold["observation"]["tool_policy_cid"] = "tools:b"
    elif field == "clause": cold["clause_results"][0]["status"] = "unsupported"
    assert finite_successor_semantics(cold) != finite_successor_semantics(incremental)
