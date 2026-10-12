"""Refresh cost must distinguish new execution, replay and unobserved work."""
from copy import deepcopy
import signal
import time
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_numerical_refresh_accounting as accounting


def execution(reused=False):
    return dict(schema="supervisor-learned-numerical-refresh-execution@1",
        scope="current_successor_refresh_call", counters_known=True, reused=reused,
        native_worker_executed=not reused, inference_executed=not reused,
        model_loads=0 if reused else 1, head_inference_calls=0 if reused else 1,
        inferred_units=0 if reused else 3, embedding_production_calls=0 if reused else 1,
        embedded_texts=0 if reused else 2)


def row(reused=False):
    return dict(status="refreshed", retrieval_status="current",
        learned_numerical_refresh_execution=execution(reused),
        # The original artifact legitimately retains execution even on reload.
        learned_numerical_refresh={"inference_execution": execution(False)})


def report():
    return dict(arm="full", task_state={"status": "completed"}, stop={"status": "succeeded"},
        remaining_processes=0, phases={}, initial_context={"numerical_selection_384": {"authored": True}})


def test_cached_artifact_does_not_double_count_original_inference():
    value = accounting.summarize([row(False), row(True)])
    assert value["all_refreshes_receipted"] is True
    assert value["observed_refreshes"] == 2
    assert value["totals"] == dict(native_worker_calls=1, model_loads=1, head_inference_calls=1,
        inferred_units=3, embedding_production_calls=1, embedded_texts=2)
    assert accounting.project(value) == value


def test_missing_failed_attempt_preserves_known_work_but_never_invents_total():
    value = accounting.summarize([row(), {"status": "unavailable"}])
    assert value["all_refreshes_receipted"] is False and value["unobserved_refreshes"] == 1
    assert all(count is None for count in value["totals"].values())
    assert value["known_subtotals"]["head_inference_calls"] == 1
    assert accounting.project(value) == value


@pytest.mark.parametrize("field,value", [("schema", "unknown"), ("scope", "historical"),
    ("counters_known", False), ("head_inference_calls", True), ("model_loads", -1),
    ("embedded_texts", 129), ("embedded_texts", 4), ("reused", True), ("inference_executed", False),
    ("native_worker_executed", False), ("embedding_production_calls", 0)])
def test_invalid_execution_cannot_support_complete_cost(field, value):
    item = row()
    item["learned_numerical_refresh_execution"][field] = value
    observed = accounting.summarize([item])
    assert observed["all_refreshes_receipted"] is False
    assert all(count is None for count in observed["totals"].values())


@pytest.mark.parametrize("mode", ["unsupported", "token_deferred"])
def test_abstentions_account_for_worker_without_claiming_decoding(mode):
    item = execution()
    item.update(inference_executed=False, head_inference_calls=0, inferred_units=0,
        model_loads=int(mode == "token_deferred"), embedding_production_calls=int(mode == "token_deferred"),
        embedded_texts=0)
    observed = accounting.summarize([{"learned_numerical_refresh_execution": item}])
    assert observed["all_refreshes_receipted"] is True
    assert observed["totals"]["native_worker_calls"] == 1
    assert observed["totals"]["head_inference_calls"] == observed["totals"]["embedded_texts"] == 0


def test_comparison_keeps_learned_cost_separate_from_retrieval_and_bounds_unknowns():
    value = accounting.summarize([row()])
    projected = comparison._refresh_observation({"post_publication_context": {
        "status": "refreshed", "embedding_accounting": {"all_refreshes_receipted": False},
        "learned_numerical_accounting": value}})
    assert projected["embedding_calls"] is None
    assert projected["learned_numerical_accounting"]["totals"]["model_loads"] == 1
    altered = deepcopy(value)
    altered["unobserved_refreshes"] = 1
    assert accounting.project(altered)["all_refreshes_receipted"] is False
    assert accounting.project(altered)["totals"]["head_inference_calls"] is None


@pytest.mark.parametrize("coherent", [False, True])
def test_comparison_refuses_inconsistent_or_impossible_complete_totals(coherent):
    value = accounting.summarize([row()])
    value["totals"]["model_loads"] = 0
    if coherent:
        value["known_subtotals"]["model_loads"] = 0
    projected = accounting.project(value)
    assert projected["all_refreshes_receipted"] is False
    assert projected["totals"]["model_loads"] is None


@pytest.mark.parametrize("failure", [False, True])
def test_driver_retains_unknown_cost_on_failure_and_current_cost_on_success(monkeypatch, failure):
    monkeypatch.setattr(signal, "setitimer", lambda *args: None)
    def refresh():
        if failure:
            raise RuntimeError("no validated model-work receipt")
        return [row()]
    observed = report()
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=refresh), observed,
        deadline=time.monotonic() + 30)
    actual = observed["post_publication_context"]["learned_numerical_accounting"]
    assert actual["all_refreshes_receipted"] is not failure
    assert actual["totals"]["head_inference_calls"] == (None if failure else 1)
    assert observed["task_state"]["status"] == "completed"


def test_exhausted_work_does_not_claim_zero_inference(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.)
    observed = report()
    driver._refresh_completed_context(SimpleNamespace(refresh_after_stop=lambda: pytest.fail("expired work ran")),
        observed, deadline=150., work_deadline=99.)
    assert observed["post_publication_context"]["learned_numerical_accounting"]["totals"]["model_loads"] is None
