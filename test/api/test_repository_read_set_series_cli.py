"""Real native read sets and shared router, with authored backend observations.

These tests exercise complete independent inputs and private retained costs;
the authored backend is not a native provider, model or task acceptance oracle.
"""
import hashlib
import json
import stat

import pytest

from benchmarks.agent_supervisor import repository_read_set_series as subject
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_read_set import GUIDANCE
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter, render_model_prompt
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
from ipfs_accelerate_py.llm_allocation import api_key_slots
from test.api.test_repository_read_set_cli import (
    POLICY, PRIVATE, REQUIREMENTS, TARGET, case as prepared_case, native_index, tokenizer,
)  # noqa: F401

ANSWER = "PRIVATE_AUTHORED_SERIES_RESPONSE"


@pytest.fixture
def case(prepared_case, monkeypatch):
    monkeypatch.chdir(prepared_case["root"])
    result = dict(prepared_case)
    result["queries"] = [result.pop("query"), "invoice_total"]
    result.update(max_total_input_tokens=200000, timeout=5, max_output_tokens=2048)
    return result


@pytest.fixture
def backend(monkeypatch):
    """Keep actual generate_text; replace discovery and external persistence."""
    class AuthoredBackend:
        def __init__(self):
            self.calls, self.action, self.initialize = [], None, None
        def generate(self, prompt, *, model_name=None, **kwargs):
            self.calls.append(dict(prompt=prompt, model=model_name, options=kwargs))
            payload = dict(type="result", subtype="success", stop_reason="end_turn",
                usage_is_incomplete=False, usage=dict(input_tokens=23, cached_input_tokens=7,
                    cache_write_input_tokens=1, output_tokens=4, total_tokens=35))
            observed = native_grok_observation(payload, exit_code=0)
            observed.update(prompt_tokens=23, completion_tokens=4, model_id=model_name,
                session_id=f"authored-native-series-{len(self.calls):03d}", exit_code=0)
            set_last_cli_observation("grok_cli", observed)
            if self.action is not None:
                self.action()
            return ANSWER + str(len(self.calls))
    selected = AuthoredBackend()
    def discover(*args, **kwargs):
        if selected.initialize is not None:
            selected.initialize()
        return selected
    monkeypatch.setattr(llm_router, "get_llm_provider", discover)
    monkeypatch.setattr(llm_router, "get_allocation_session", lambda *args, **kwargs: {})
    monkeypatch.setattr(llm_router, "_record_llm_allocation_observation", lambda **kwargs: None)
    monkeypatch.setattr(llm_router, "_capture_cli_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_router, "_bind_cli_session_kwargs", lambda _provider, values: dict(values))
    monkeypatch.setattr(api_key_slots, "apply_api_key_to_kwargs", lambda provider, kwargs, **rest: dict(kwargs))
    return selected


def arguments(case):
    result = []
    for key, value in case.items():
        if key == "reuse_identical_inputs":
            if value:
                result.append("--reuse-identical-inputs")
        elif key in {"queries", "required_paths"}:
            flag = "--query" if key == "queries" else "--required-path"
            for item in value:
                result.extend([flag, str(item)])
        else:
            result.extend(["--" + key.replace("_", "-"), str(value)])
    return result


def output(case, expected):
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o700
    assert {path.name for path in case["output"].iterdir()} == expected
    for path in case["output"].iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_nlink == 1
    return json.loads((case["output"] / "receipt.json").read_bytes())


def test_actual_series_repeats_required_bytes_in_fresh_contexts_and_publishes_each_call(
        case, native_index, backend, monkeypatch, capsys):
    case.update(scheduler_state=native_index["state"], scheduler_policy_sha256=native_index["policy_sha256"])
    seen, routes = tokenizer(monkeypatch)
    assert subject.main(arguments(case)) == 0
    summary = json.loads(capsys.readouterr().out)
    receipt = output(case, {"receipt.json", "prompt-01.txt", "prompt-02.txt", "response-01.txt", "response-02.txt"})
    series = receipt["series"]
    assert len(backend.calls) == 2 and len(native_index["paths"]) == 1
    assert routes == [("grok_cli", "grok-4.7")]
    assert series["status"] == summary["series_status"] == "returned"
    assert receipt["completed_calls"] == summary["completed_calls"] == 2
    assert series["backend_dispatches"] == 2
    local_counts = []
    for index, call in enumerate(backend.calls, 1):
        raw = (case["output"] / f"prompt-{index:02d}.txt").read_bytes()
        response = (case["output"] / f"response-{index:02d}.txt").read_bytes()
        assert raw == call["prompt"].encode() and response.decode() == ANSWER + str(index)
        _, advisory = render_model_prompt(prompt="", purpose="coding", workspace=case["root"])
        body = json.loads(raw.decode()[len(advisory) + len(GUIDANCE):])
        assert body["required_texts"] == [REQUIREMENTS]
        assert {row["path"]: row["content"].encode() for row in body["sources"]} == {
            "policy.py": POLICY, "ledger.py": TARGET}
        assert ANSWER not in raw.decode() and call["prompt"].count("Supervisor coding workspace contract:") == 1
        assert call["options"]["resume_session_id"] == "" and call["options"]["continue_session"] is False
        assert call["options"]["grok_max_turns"] == 1
        binding = receipt["output_bindings"][index - 1]
        assert binding["prompt_sha256"] == hashlib.sha256(raw).hexdigest()
        assert binding["response_sha256"] == hashlib.sha256(response).hexdigest()
        attempt = series["attempts"][index - 1]
        assert attempt["response_index"] == index and attempt["backend_dispatches"] == 1
        assert attempt["inspection"]["native_rollout_usage"]["usage"]["total_tokens"] == 35
        assert attempt["local_estimated_input_tokens"] == len(raw)
        local_counts.append(len(raw))
    assert backend.calls[0]["options"]["chat_session_id"] != backend.calls[1]["options"]["chat_session_id"]
    assert series["cumulative_input_tokens"] is None
    assert series["cumulative_estimated_input_tokens"] == sum(local_counts)
    assert series["remaining_local_input_allowance"] == case["max_total_input_tokens"] - sum(local_counts)
    assert receipt["independent_fresh_contexts"] is True and receipt["prior_responses_in_next_prompt"] is False
    assert receipt["native_cumulative_cap_enforced"] is receipt["task_acceptance"] is receipt["billing_total_verified"] is False
    encoded = json.dumps(receipt) + json.dumps(summary)
    assert all(marker not in encoded for marker in (ANSWER, PRIVATE, REQUIREMENTS, "SELECTED_ORIGINAL_COMMENT"))
    assert all(not path.parent.exists() for path in native_index["paths"])


def fixed_tokenizer(monkeypatch, value=100):
    def resolve(*, provider, model):
        return PromptTokenCounter(provider=provider, model=model, tokenizer_id="authored:fixed",
            tokenizer_revision="1", special_token_policy="literal", count_tokens=lambda _text: value,
            count_kind="estimate", compatibility_basis="authored-counter"), {"count_kind": "estimate"}
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", resolve)


@pytest.mark.parametrize("stop", ["local", "native"])
def test_orderly_stop_keeps_completed_private_files_and_never_dispatches_next_query(
        case, native_index, backend, monkeypatch, capsys, stop):
    fixed_tokenizer(monkeypatch)
    if stop == "local":
        case["max_total_input_tokens"] = 150
    else:
        case["max_observed_native_tokens_before_next_dispatch"] = 35
    assert subject.main(arguments(case)) == 0
    summary = json.loads(capsys.readouterr().out)
    receipt = output(case, {"receipt.json", "prompt-01.txt", "response-01.txt"})
    assert len(backend.calls) == 1
    assert receipt["status"] == receipt["series"]["status"] == summary["series_status"] == "stopped"
    assert receipt["series"]["stop_reason"] == ("local_allowance_exhausted" if stop == "local" else "native_threshold_reached")
    assert receipt["series"]["cumulative_estimated_input_tokens"] == 100
    assert receipt["series"]["attempts"][1]["backend_dispatches"] == 0


@pytest.mark.parametrize("failure", ["first_provider", "second_provider", "source", "requirements"])
def test_abort_retains_all_attempt_charges_without_any_partial_source_or_response_export(
        case, native_index, backend, monkeypatch, capsys, failure):
    tokenizer(monkeypatch)
    case["queries"] *= 2
    def change():
        if failure == "first_provider" or (failure == "second_provider" and len(backend.calls) == 2):
            raise RuntimeError(PRIVATE)
        if failure == "source":
            (case["root"] / "ledger.py").write_text("# changed source\n")
        if failure == "requirements":
            case["requirements_file"].write_text(PRIVATE)
    backend.action = change
    assert subject.main(arguments(case)) == 2
    summary = json.loads(capsys.readouterr().out)
    receipt = output(case, {"receipt.json"})
    expected = 2 if failure == "second_provider" else 1
    assert len(backend.calls) == expected and receipt["status"] == "failed"
    assert summary["receipt_retained"] is True
    charged = [row for row in receipt["series"]["attempts"] if row["backend_dispatches"] == 1]
    assert len(charged) == expected
    assert all(row["inspection"]["native_rollout_usage"]["usage"]["total_tokens"] == 35 for row in charged)
    assert all(marker not in json.dumps(receipt) + json.dumps(summary) for marker in (PRIVATE, ANSWER, REQUIREMENTS))
    assert all(not path.parent.exists() for path in native_index["paths"])


@pytest.mark.parametrize("error", [RuntimeError(PRIVATE), KeyboardInterrupt(), SystemExit(7)])
def test_callable_preserves_original_error_even_if_failure_publication_and_observer_fail(
        case, native_index, backend, monkeypatch, error):
    tokenizer(monkeypatch)
    def fail():
        raise error
    backend.action = fail
    monkeypatch.setattr(subject, "_publish", lambda *args: (_ for _ in ()).throw(OSError(PRIVATE)))
    def observer(_metadata):
        raise RuntimeError(PRIVATE)
    with pytest.raises(type(error)) as raised:
        subject.run_repository_read_set_series(**case, failure_publication_observer=observer)
    assert raised.value is error and len(backend.calls) == 1
    assert not case["output"].exists()


@pytest.mark.parametrize("change", ["cwd", "route", "model", "empty", "many", "query", "query_type", "query_nul",
    "total_bool", "total_zero", "total_overflow", "native_bool", "timeout", "output_tokens", "existing_output",
    "output_symlink", "requirements_alias", "duplicate_path", "scheduler_pair"])
def test_invalid_series_refuses_before_tokenizer_native_index_or_provider(
        case, native_index, backend, monkeypatch, change):
    seen, routes = tokenizer(monkeypatch)
    if change == "cwd": monkeypatch.chdir(case["root"].parent)
    elif change == "route": case["provider"] = "codex_cli"
    elif change == "model": case["model"] = "unqualified-model"
    elif change == "empty": case["queries"] = []
    elif change == "many": case["queries"] *= 5
    elif change == "query": case["queries"] = ["x" * 4097]
    elif change == "query_type": case["queries"] = "invoice_total"
    elif change == "query_nul": case["queries"] = ["invoice\x00total"]
    elif change == "total_bool": case["max_total_input_tokens"] = True
    elif change == "total_zero": case["max_total_input_tokens"] = 0
    elif change == "total_overflow": case["max_total_input_tokens"] = 1 << 63
    elif change == "native_bool": case["max_observed_native_tokens_before_next_dispatch"] = True
    elif change == "timeout": case["timeout"] = 1801
    elif change == "output_tokens": case["max_output_tokens"] = True
    elif change == "existing_output": case["output"].mkdir()
    elif change == "output_symlink": case["output"].symlink_to(case["root"], target_is_directory=True)
    elif change == "requirements_alias":
        link = case["root"].parent / "linked.txt"
        link.symlink_to(case["requirements_file"])
        case["requirements_file"] = link
    elif change == "duplicate_path": case["required_paths"] *= 2
    elif change == "scheduler_pair": case["scheduler_policy_sha256"] = "0" * 64
    with pytest.raises(Exception):
        subject.run_repository_read_set_series(**case)
    assert seen == routes == backend.calls == native_index["paths"] == []


def test_publication_rollback_removes_only_owned_files_and_preserves_existing_directory(
        case, monkeypatch, capsys):
    case["output"].mkdir()
    sentinel = case["output"] / "keep.txt"
    sentinel.write_text(PRIVATE)
    assert subject.main(arguments(case)) == 2
    assert sentinel.read_text() == PRIVATE
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "refused" and PRIVATE not in json.dumps(summary)


@pytest.mark.parametrize("reuse", [None, False, True], ids=["omitted-fresh", "explicit-false-fresh", "explicit-reuse"])
def test_reuse_is_explicit_and_avoids_backend_calls_without_duplicating_native_charges(
        case, native_index, backend, monkeypatch, capsys, reuse):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_read_set_series as runtime
    original, forwarded = runtime.inspect_repository_read_set_series, []
    def actual_series(**options):
        forwarded.append(options.get("reuse_identical_inputs"))
        return original(**options)
    monkeypatch.setattr(runtime, "inspect_repository_read_set_series", actual_series)
    fixed_tokenizer(monkeypatch)
    case["queries"] = ["invoice_total"] * 3
    if reuse is False:
        summary = subject.run_repository_read_set_series(**case, reuse_identical_inputs=False)
        assert capsys.readouterr().out == ""
    else:
        if reuse is True:
            case["reuse_identical_inputs"] = True
        assert subject.main(arguments(case)) == 0
        summary = json.loads(capsys.readouterr().out)
    expected_calls = 1 if reuse else 3
    receipt = output(case, {"receipt.json", *{f"{kind}-{index:02d}.txt"
        for kind in ("prompt", "response") for index in range(1, 4)}})
    series = receipt["series"]
    assert forwarded == [reuse is True]
    assert len(backend.calls) == expected_calls and len(native_index["paths"]) == 1
    assert series["backend_dispatches"] == expected_calls
    assert series["cumulative_estimated_input_tokens"] == 100 * expected_calls
    assert series["scoped_native_total_tokens"] == 35 * expected_calls
    assert series["scoped_native_observation"]["supplied_invocation_count"] == expected_calls
    assert receipt["completed_calls"] == summary["completed_calls"] == expected_calls
    assert receipt["completed_results"] == summary["completed_results"] == 3
    assert receipt["reused_results"] == summary["reused_results"] == (2 if reuse else 0)
    assert receipt["reuse_identical_inputs_requested"] is (reuse is True)
    assert series["remaining_local_input_allowance"] == case["max_total_input_tokens"] - 100 * expected_calls
    prompts = [(case["output"] / f"prompt-{index:02d}.txt").read_bytes() for index in range(1, 4)]
    responses = [(case["output"] / f"response-{index:02d}.txt").read_bytes() for index in range(1, 4)]
    assert prompts[0] == prompts[1] == prompts[2] == backend.calls[0]["prompt"].encode()
    assert [row["status"] for row in series["attempts"]] == (["returned", "reused", "reused"] if reuse else ["returned"] * 3)
    assert [row["response_index"] for row in series["attempts"]] == [1, 2, 3]
    if reuse:
        assert responses[0] == responses[1] == responses[2] == (ANSWER + "1").encode()
        assert all(row["backend_dispatches"] == 0 and row["inspection"] is None for row in series["attempts"][1:])
        origin = series["attempts"][0]
        for reused in series["attempts"][1:]:
            reference = reused["reuse_reference"]
            assert reference["original_attempt_ordinal"] == reference["original_response_index"] == 1
            assert reference["original_invocation_id"] == origin["inspection"]["invocation_id"]
            assert reference["model_prompt_sha256"] == hashlib.sha256(prompts[0]).hexdigest()
            assert reference["response_sha256"] == hashlib.sha256(responses[0]).hexdigest()
            assert reference["model_prompt_bytes"] == len(prompts[0]) and reference["response_bytes"] == len(responses[0])
            assert reused["local_debit"] == 0 and reused["local_debit_status"] == "reused"
        assert receipt["independent_fresh_contexts"] is False
    else:
        assert responses == [(ANSWER + str(index)).encode() for index in range(1, 4)]
        assert receipt["independent_fresh_contexts"] is True
    for binding, prompt, response in zip(receipt["output_bindings"], prompts, responses, strict=True):
        assert binding["prompt_sha256"] == hashlib.sha256(prompt).hexdigest()
        assert binding["response_sha256"] == hashlib.sha256(response).hexdigest()
    assert receipt["task_acceptance"] is receipt["billing_total_verified"] is receipt["native_cumulative_cap_enforced"] is False
    encoded = json.dumps(receipt) + json.dumps(summary)
    assert all(marker not in encoded for marker in (ANSWER, PRIVATE, REQUIREMENTS, "SELECTED_ORIGINAL_COMMENT"))
    assert all(not path.parent.exists() for path in native_index["paths"])


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_invalid_reuse_option_is_refused_before_tokenizer_or_native_state(
        case, native_index, backend, monkeypatch, value):
    seen, routes = tokenizer(monkeypatch)
    with pytest.raises(ValueError, match="boolean result reuse"):
        subject.run_repository_read_set_series(**case, reuse_identical_inputs=value)
    assert seen == routes == backend.calls == native_index["paths"] == []
    assert not case["output"].exists()


@pytest.mark.parametrize("ceiling", ["local", "native"])
def test_cached_answer_costs_no_new_allowance_but_a_new_input_still_obeys_the_stop(
        case, native_index, backend, monkeypatch, capsys, ceiling):
    fixed_tokenizer(monkeypatch)
    case.update(queries=["invoice_total", "invoice_total", "unrelated"], reuse_identical_inputs=True)
    if ceiling == "local":
        case["max_total_input_tokens"] = 100
    else:
        case["max_observed_native_tokens_before_next_dispatch"] = 35
    assert subject.main(arguments(case)) == 0
    summary = json.loads(capsys.readouterr().out)
    receipt = output(case, {"receipt.json", "prompt-01.txt", "response-01.txt", "prompt-02.txt", "response-02.txt"})
    series = receipt["series"]
    assert len(backend.calls) == 1 and series["backend_dispatches"] == 1
    assert series["status"] == "stopped" and series["stop_reason"] == (
        "local_allowance_exhausted" if ceiling == "local" else "native_threshold_reached")
    assert [row["status"] for row in series["attempts"]] == ["returned", "reused", "refused"]
    assert series["attempts"][1]["local_debit"] == series["attempts"][2]["backend_dispatches"] == 0
    assert series["cumulative_estimated_input_tokens"] == 100 and series["scoped_native_total_tokens"] == 35
    assert summary["completed_calls"] == receipt["completed_calls"] == 1
    assert summary["completed_results"] == receipt["completed_results"] == 2
    assert summary["reused_results"] == receipt["reused_results"] == 1
    assert (case["output"] / "response-01.txt").read_bytes() == (case["output"] / "response-02.txt").read_bytes()


@pytest.mark.parametrize("changed", ["source", "requirements"])
def test_reuse_option_does_not_hide_post_call_drift_or_discard_failed_charges(
        case, native_index, backend, monkeypatch, capsys, changed):
    tokenizer(monkeypatch)
    case.update(queries=["invoice_total"] * 3, reuse_identical_inputs=True)
    def mutate():
        if changed == "source":
            (case["root"] / "ledger.py").write_text("# changed source\n")
        else:
            case["requirements_file"].write_text(PRIVATE)
    backend.action = mutate
    assert subject.main(arguments(case)) == 2
    summary = json.loads(capsys.readouterr().out)
    receipt = output(case, {"receipt.json"})
    assert len(backend.calls) == 1 and summary["receipt_retained"] is True
    assert receipt["status"] == "failed"
    charged = [row for row in receipt["series"]["attempts"] if row["backend_dispatches"] == 1]
    assert len(charged) == 1 and charged[0]["inspection"]["native_rollout_usage"]["usage"]["total_tokens"] == 35
    assert all(row["status"] != "reused" for row in receipt["series"]["attempts"])
    assert all(marker not in json.dumps(receipt) + json.dumps(summary) for marker in (PRIVATE, ANSWER, REQUIREMENTS))
