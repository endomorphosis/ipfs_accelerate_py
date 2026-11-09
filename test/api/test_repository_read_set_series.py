"""Cumulative allowance over genuine native read sets/router and authored backend."""
from hashlib import sha256
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_read_set_series as subject
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
from test.api.test_repository_read_set_inspect_cli import backend, case, prepared_case  # noqa: F401
from test.api.test_repository_read_set_cli import native_index, REQUIREMENTS, PRIVATE  # noqa: F401


@pytest.fixture
def active(case, native_index, tmp_path):
    with RepositorySourceSession(case["root"], tmp_path / "series-state", representation="original") as session:
        yield case, session


def options(active, *, kind="estimate", amount=None, **changes):
    case, session = active
    result = dict(session=session, generation=session.generation,
        queries=["invoice_total"] * 3, required_paths=case["required_paths"], required_texts=[REQUIREMENTS],
        task_id=case["task_id"], provider="grok_cli", model="grok-4.7", workspace=case["root"],
        token_counter=PromptTokenCounter(provider="grok_cli", model="grok-4.7", tokenizer_id="authored",
            tokenizer_revision="fixture@1", special_token_policy="literal", count_kind=kind,
            compatibility_basis="authored-counter-only", count_tokens=(lambda text: len(text.encode()))
                if amount is None else lambda text: amount),
        max_total_input_tokens=100000, max_input_tokens=32768)
    result.update(changes)
    return result


def test_identical_full_prompt_and_envelope_hash_count_once_per_distinct_backend_call(active, backend):
    observed = []
    result = subject.inspect_repository_read_set_series(**options(active, queries=["invoice_total"] * 2),
        receipt_observer=observed.append)
    r = result.receipt
    assert len(backend.calls) == len(result.selected_prompts) == len(result.response_texts) == 2
    assert result.selected_prompts[0] == result.selected_prompts[1]
    assert [row["prompt"] for row in backend.calls] == list(result.selected_prompts)
    assert r["cumulative_input_tokens"] is None
    assert r["cumulative_estimated_input_tokens"] == 2 * len(result.selected_prompts[0].encode())
    first, second = [row["inspection"] for row in r["attempts"]]
    assert first["invocation_id"] != second["invocation_id"]
    assert first["native_rollout_usage"]["usage_observation_sha256"] == second["native_rollout_usage"]["usage_observation_sha256"]
    assert r["scoped_native_total_tokens"] == 68 and r["backend_dispatches"] == 2
    assert r["scoped_native_observation"]["invocation_envelope_count"] == 2
    assert observed == [r]
    assert first["inspection_projection_only"] is True and "read_set" not in first
    assert r["task_acceptance_verified"] is r["billing_total_verified"] is r["native_cumulative_cap_enforced"] is False
    assert r["whole_task_usage_complete"] is r["total_session_token_savings"] is None
    assert "MANDATORY_ORIGINAL_COMMENT" not in result.receipt_json
    assert REQUIREMENTS not in result.receipt_json and result.selected_prompts[0] not in repr(result)
    copied = result.receipt
    copied["attempts"].clear()
    assert len(result.receipt["attempts"]) == 2


@pytest.mark.parametrize("kind", ["tokenizer", "estimate"])
def test_remaining_allowance_prevents_third_dispatch_without_truncating_required_core(active, backend, kind):
    r = subject.inspect_repository_read_set_series(**options(active, kind=kind, amount=10,
        max_total_input_tokens=25, max_input_tokens=20)).receipt
    assert len(backend.calls) == 2
    assert r["status"] == "stopped" and r["stop_reason"] == "local_allowance_exhausted"
    assert [row["input_allowance"] for row in r["attempts"]] == [20, 15, 5]
    assert [row["local_debit"] for row in r["attempts"]] == [10, 10, 0]
    field = "cumulative_estimated_input_tokens" if kind == "estimate" else "cumulative_input_tokens"
    assert r[field] == 20 and r["remaining_local_input_allowance"] == 5
    assert r["attempts"][2]["inspection"] is None and r["attempts"][2]["backend_dispatches"] == 0
    assert r["attempts"][2]["status"] == "refused"


def test_first_required_core_cannot_fit_and_never_calls_backend(active, backend):
    result = subject.inspect_repository_read_set_series(**options(active, amount=10, max_total_input_tokens=5))
    assert backend.calls == [] and result.selected_prompts == result.response_texts == ()
    assert result.receipt["status"] == "stopped" and result.receipt["backend_dispatches"] == 0


@pytest.mark.parametrize("threshold,calls", [(34, 1), (60, 2)])
def test_observed_native_threshold_stops_next_after_first_may_overshoot(active, backend, threshold, calls):
    r = subject.inspect_repository_read_set_series(**options(active, amount=10,
        max_observed_native_tokens_before_next_dispatch=threshold)).receipt
    assert len(backend.calls) == calls
    assert r["status"] == "stopped" and r["stop_reason"] == "native_threshold_reached"
    assert r["scoped_native_total_tokens"] == 34 * calls
    assert r["attempts"][calls]["backend_dispatches"] == 0


def test_unknown_native_completeness_stops_without_numeric_threshold(active, backend, monkeypatch):
    original = backend.generate
    def incomplete(*args, **kwargs):
        response = original(*args, **kwargs)
        value = get_last_cli_observation("grok_cli")
        value.pop("native_grok_usage_complete", None)
        set_last_cli_observation("grok_cli", value)
        return response
    monkeypatch.setattr(backend, "generate", incomplete)
    r = subject.inspect_repository_read_set_series(**options(active, amount=10)).receipt
    assert len(backend.calls) == 1 and r["stop_reason"] == "prior_native_usage_unknown"
    assert r["cumulative_estimated_input_tokens"] == 10
    assert r["scoped_native_total_tokens"] is None
    assert r["scoped_native_observation"]["reported_native_token_subtotal"] == 34


@pytest.mark.parametrize("error", [RuntimeError(PRIVATE), KeyboardInterrupt(), SystemExit(7)])
def test_paid_provider_failure_debits_input_and_preserves_original_exception(active, backend, error):
    backend.error = error
    observed = []
    with pytest.raises(type(error)) as raised:
        subject.inspect_repository_read_set_series(**options(active, amount=10), receipt_observer=observed.append)
    assert raised.value is error and len(backend.calls) == 1
    r = observed[0]
    assert r["status"] == "failed" and r["cumulative_estimated_input_tokens"] == 10
    assert r["remaining_local_input_allowance"] == 99990
    assert r["attempts"][0]["local_debit"] == 10
    assert r["attempts"][0]["inspection"]["native_rollout_usage"]["usage"]["total_tokens"] == 34
    assert PRIVATE not in json.dumps(r)
    assert r["attempts"][1]["status"] == "not_attempted"


def test_source_change_after_backend_retains_paid_local_debit(active, backend):
    backend.after = lambda: (active[0]["root"] / "ledger.py").write_text("# source changed\n")
    observed = []
    with pytest.raises(Exception):
        subject.inspect_repository_read_set_series(**options(active, amount=10), receipt_observer=observed.append)
    assert len(backend.calls) == 1
    assert observed[0]["attempts"][0]["local_debit"] == 10
    assert observed[0]["remaining_local_input_allowance"] == 99990


def test_counter_global_provenance_and_repeatability_failure_stop_after_paid_call(active, backend):
    state = {"amount": 10}
    supplied = options(active)
    supplied["token_counter"] = PromptTokenCounter(provider="grok_cli", model="grok-4.7", tokenizer_id="authored",
        tokenizer_revision="fixture@1", special_token_policy="literal", count_kind="estimate",
        compatibility_basis="fixture", count_tokens=lambda text: state["amount"])
    backend.after = lambda: state.update(amount=11)
    observed = []
    with pytest.raises(ValueError):
        subject.inspect_repository_read_set_series(**supplied, receipt_observer=observed.append)
    assert len(backend.calls) == 1 and observed[0]["attempts"][0]["local_debit"] == 10


def test_caller_populations_are_snapshots_before_backend_initialization(active, backend):
    supplied = options(active, amount=10)
    def mutate():
        supplied["queries"].clear()
        supplied["required_paths"].append("forged.py")
        supplied["required_texts"][0] = "FORGED_TASK"
    backend.initialize = mutate
    result = subject.inspect_repository_read_set_series(**supplied)
    assert len(backend.calls) == 3 and result.receipt["query_count"] == 3
    assert "FORGED_TASK" not in "".join(result.selected_prompts)


def test_initial_guard_failure_has_manifest_without_masking_original(active, backend):
    error, observed = RuntimeError(PRIVATE), []
    def fail(): raise error
    with pytest.raises(RuntimeError) as raised:
        subject.inspect_repository_read_set_series(**options(active), requirements_guard=fail, receipt_observer=observed.append)
    assert raised.value is error and backend.calls == []
    assert observed[0]["backend_dispatches"] == 0
    assert all(row["status"] == "not_attempted" for row in observed[0]["attempts"])


def test_missing_observer_receipt_withholds_reservation_and_never_reports_zero_cost(active, backend, monkeypatch):
    original, observed = subject.inspect_repository_read_set, []
    def missing(**kwargs):
        kwargs["receipt_observer"] = lambda row: None
        return original(**kwargs)
    monkeypatch.setattr(subject, "inspect_repository_read_set", missing)
    with pytest.raises(ValueError):
        subject.inspect_repository_read_set_series(**options(active, amount=10, max_input_tokens=20), receipt_observer=observed.append)
    r = observed[0]
    assert len(backend.calls) == 1 and r["local_accounting_complete"] is False
    assert r["cumulative_estimated_input_tokens"] is None and r["backend_dispatches"] is None
    assert r["remaining_local_input_allowance"] == 99980
    assert r["attempts"][0]["local_debit_status"] == "reservation_withheld"


def test_secondary_accounting_cancellation_cannot_replace_provider_failure_or_zero_cost(active, backend, monkeypatch):
    error, observed = RuntimeError(PRIVATE), []
    backend.error = error
    monkeypatch.setattr(subject.accounting, "observe_invocation", lambda row: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(RuntimeError) as raised:
        subject.inspect_repository_read_set_series(**options(active, amount=10), receipt_observer=observed.append)
    assert raised.value is error and len(backend.calls) == 1
    r = observed[0]
    assert r["attempts"][0]["local_debit"] == 10 and r["remaining_local_input_allowance"] == 99990
    assert r["cumulative_estimated_input_tokens"] is None and r["backend_dispatches"] is None
    assert r["native_usage_known_for_continuing"] is False


@pytest.mark.parametrize("changes", [dict(queries=[]), dict(queries=["x"] * 9), dict(queries=[""]),
    dict(max_total_input_tokens=True), dict(max_input_tokens=0), dict(timeout=1801),
    dict(max_output_tokens=True), dict(max_observed_native_tokens_before_next_dispatch=0),
    dict(provider="codex_cli")])
def test_invalid_series_refuses_before_backend(active, backend, changes):
    with pytest.raises(ValueError):
        subject.inspect_repository_read_set_series(**options(active, **changes))
    assert backend.calls == []


def test_opt_in_reuses_exact_full_input_and_references_one_original_call(active, backend):
    result = subject.inspect_repository_read_set_series(**options(active, reuse_identical_inputs=True))
    r = result.receipt
    assert len(backend.calls) == 1 and len(result.selected_prompts) == len(result.response_texts) == 3
    assert len(set(result.selected_prompts)) == len(set(result.response_texts)) == 1
    first, *reused = r["attempts"]
    inspection = first["inspection"]
    assert [row["status"] for row in r["attempts"]] == ["returned", "reused", "reused"]
    assert r["reused_results"] == 2 and r["backend_dispatches"] == r["known_backend_dispatches"] == 1
    assert r["cumulative_estimated_input_tokens"] == len(result.selected_prompts[0].encode())
    assert r["scoped_native_total_tokens"] == 34
    assert len(r["scoped_native_observation"]["observations"]) == 1
    for index, row in enumerate(reused, 2):
        assert row["backend_dispatches"] == row["local_debit"] == 0
        assert row["local_input_tokens"] is row["local_estimated_input_tokens"] is row["inspection"] is None
        assert row["local_debit_status"] == "reused" and row["response_index"] == index
        reference = row["reuse_reference"]
        assert reference["original_attempt_ordinal"] == reference["original_response_index"] == 1
        assert reference["original_invocation_id"] == inspection["invocation_id"]
        assert reference["model_prompt_sha256"] == sha256(result.selected_prompts[0].encode()).hexdigest()
        assert reference["response_sha256"] == sha256(result.response_texts[0].encode()).hexdigest()
        assert reference["model_prompt_bytes"] == len(result.selected_prompts[0].encode())
        assert reference["response_bytes"] == len(result.response_texts[0].encode())
    assert r["fresh_independent_allocations"] is True and r["prior_response_replayed"] is False
    assert r["result_reuse_persistent"] is r["reused_results_are_new_provider_calls"] is False
    assert r["total_session_token_savings"] is None and r["task_acceptance_verified"] is False
    assert result.selected_prompts[0] not in result.receipt_json and result.response_texts[0] not in repr(result)
    assert REQUIREMENTS not in result.receipt_json and PRIVATE not in result.receipt_json


@pytest.mark.parametrize("stop", ["local", "native", "unknown"])
def test_hits_precede_spending_stop_but_different_full_input_miss_still_stops(active, backend, monkeypatch, stop):
    settings = dict(reuse_identical_inputs=True, amount=10,
        queries=["invoice_total", "invoice_total", "unrelated"])
    if stop == "local":
        settings["max_total_input_tokens"] = 10
    elif stop == "native":
        settings["max_observed_native_tokens_before_next_dispatch"] = 34
    else:
        original = backend.generate
        def incomplete(*args, **kwargs):
            response = original(*args, **kwargs)
            value = get_last_cli_observation("grok_cli")
            value.pop("native_grok_usage_complete", None)
            set_last_cli_observation("grok_cli", value)
            return response
        monkeypatch.setattr(backend, "generate", incomplete)
    result = subject.inspect_repository_read_set_series(**options(active, **settings))
    r = result.receipt
    assert len(backend.calls) == 1 and len(result.response_texts) == 2
    assert [row["status"] for row in r["attempts"]] == ["returned", "reused", "refused"]
    assert r["attempts"][1]["local_debit"] == 0 and r["cumulative_estimated_input_tokens"] == 10
    assert r["stop_reason"] == {"local": "local_allowance_exhausted", "native": "native_threshold_reached",
        "unknown": "prior_native_usage_unknown"}[stop]
    assert r["scoped_native_total_tokens"] == (None if stop == "unknown" else 34)


@pytest.mark.parametrize("stop", ["local", "native", "unknown"])
def test_all_identical_hits_finish_after_spending_gate_is_closed(active, backend, monkeypatch, stop):
    settings = dict(reuse_identical_inputs=True, amount=10)
    if stop == "local": settings["max_total_input_tokens"] = 10
    elif stop == "native": settings["max_observed_native_tokens_before_next_dispatch"] = 34
    else:
        original = backend.generate
        def incomplete(*args, **kwargs):
            response = original(*args, **kwargs)
            value = get_last_cli_observation("grok_cli")
            value.pop("native_grok_usage_complete", None)
            set_last_cli_observation("grok_cli", value)
            return response
        monkeypatch.setattr(backend, "generate", incomplete)
    result = subject.inspect_repository_read_set_series(**options(active, **settings))
    assert len(backend.calls) == 1 and len(result.response_texts) == 3
    assert result.receipt["status"] == "returned" and result.receipt["stop_reason"] is None
    assert result.receipt["reused_results"] == 2
    assert result.receipt["native_usage_known_for_continuing"] is (stop != "unknown")


def test_equal_local_counts_do_not_reuse_changed_full_prompt(active, backend):
    result = subject.inspect_repository_read_set_series(**options(active, amount=10,
        queries=["invoice_total", "unrelated", "invoice_total"], reuse_identical_inputs=True))
    assert len(backend.calls) == 2 and result.selected_prompts[0] != result.selected_prompts[1]
    assert result.selected_prompts[0] == result.selected_prompts[2]
    assert result.receipt["attempts"][2]["reuse_reference"]["original_attempt_ordinal"] == 1
    assert result.receipt["cumulative_estimated_input_tokens"] == 20
    assert result.receipt["scoped_native_total_tokens"] == 68


@pytest.mark.parametrize("drift", ["source", "requirements", "counter"])
def test_reuse_probe_rechecks_series_bindings_before_returning_cached_response(active, backend, monkeypatch, drift):
    state, observed = {"amount": 10}, []
    supplied = options(active, reuse_identical_inputs=True)
    supplied["token_counter"] = PromptTokenCounter(provider="grok_cli", model="grok-4.7", tokenizer_id="authored",
        tokenizer_revision="1", special_token_policy="literal", count_kind="estimate",
        compatibility_basis="fixture", count_tokens=lambda text: state["amount"])
    requirements = active[0]["requirements_file"]
    initial = requirements.read_bytes()
    def guard():
        if requirements.read_bytes() != initial:
            raise ValueError("authored_requirements_changed")
    original = subject.compile_repository_read_set
    def mutate_before_probe(**kwargs):
        if backend.calls:
            if drift == "source": (active[0]["root"] / "ledger.py").write_text("# source changed\n")
            elif drift == "requirements": requirements.write_text("changed requirements\n")
            else: state["amount"] = 11
        return original(**kwargs)
    monkeypatch.setattr(subject, "compile_repository_read_set", mutate_before_probe)
    with pytest.raises(ValueError):
        subject.inspect_repository_read_set_series(**supplied, requirements_guard=guard, receipt_observer=observed.append)
    assert len(backend.calls) == 1 and observed[0]["reused_results"] == 0
    assert observed[0]["attempts"][0]["local_debit"] == 10
    assert observed[0]["cumulative_estimated_input_tokens"] == 10


def test_reuse_does_not_cross_series_and_explicit_false_still_dispatches_every_repeat(active, backend):
    first = subject.inspect_repository_read_set_series(**options(active, reuse_identical_inputs=True))
    second = subject.inspect_repository_read_set_series(**options(active, reuse_identical_inputs=True))
    default = subject.inspect_repository_read_set_series(**options(active, reuse_identical_inputs=False))
    assert len(backend.calls) == 5
    assert first.receipt["attempts"][0]["inspection"]["invocation_id"] != second.receipt["attempts"][0]["inspection"]["invocation_id"]
    assert first.receipt["reused_results"] == second.receipt["reused_results"] == 2
    assert default.receipt["reused_results"] == 0 and default.receipt["backend_dispatches"] == 3


@pytest.mark.parametrize("error", [RuntimeError(PRIVATE), KeyboardInterrupt(), SystemExit(7)])
def test_failed_miss_after_cached_success_preserves_original_failure_and_both_debits(active, backend, monkeypatch, error):
    original, observed = backend.generate, []
    def fail_second(*args, **kwargs):
        if backend.calls:
            backend.error = error
        return original(*args, **kwargs)
    monkeypatch.setattr(backend, "generate", fail_second)
    with pytest.raises(type(error)) as raised:
        subject.inspect_repository_read_set_series(**options(active, amount=10, reuse_identical_inputs=True,
            queries=["invoice_total", "unrelated", "invoice_total"]), receipt_observer=observed.append)
    assert raised.value is error and len(backend.calls) == 2
    assert observed[0]["cumulative_estimated_input_tokens"] == 20 and observed[0]["reused_results"] == 0
    assert observed[0]["attempts"][2]["status"] == "not_attempted"


@pytest.mark.parametrize("value", [None, 0, 1, "yes", [], {}])
def test_invalid_reuse_flag_refuses_before_compiler_or_backend(active, backend, monkeypatch, value):
    monkeypatch.setattr(subject, "compile_repository_read_set", lambda **kwargs: pytest.fail("compiler must not run"))
    with pytest.raises(ValueError, match="series_reuse_flag_invalid"):
        subject.inspect_repository_read_set_series(**options(active, reuse_identical_inputs=value))
    assert backend.calls == []
