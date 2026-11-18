"""Actual read-set core and router, with an authored local backend, no models."""
import hashlib
import json
import stat

import pytest

from benchmarks.agent_supervisor import repository_read_set_inspect as subject
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_read_set import GUIDANCE
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
from ipfs_accelerate_py.llm_allocation import api_key_slots
from test.api.test_repository_read_set_cli import (
    POLICY, PRIVATE, REQUIREMENTS, TARGET, case as prepared_case, native_index, tokenizer,
)  # noqa: F401

ANSWER = "AUTHORED_BACKEND_RESPONSE_KEEP_PRIVATE"


@pytest.fixture
def case(prepared_case, monkeypatch):
    monkeypatch.chdir(prepared_case["root"])
    return dict(prepared_case, timeout=1800, max_output_tokens=2048)


@pytest.fixture
def backend(monkeypatch):
    """Preserve generate_text and its real guarded callback dispatch."""
    class AuthoredBackend:
        def __init__(self):
            self.calls, self.error, self.after, self.initialize = [], None, None, None
        def generate(self, prompt, *, model_name=None, **kwargs):
            self.calls.append(dict(prompt=prompt, model=model_name, options=kwargs))
            payload = dict(type="result", subtype="success", stop_reason="end_turn",
                usage_is_incomplete=False, usage=dict(input_tokens=23, cached_input_tokens=7,
                    cache_write_input_tokens=0, output_tokens=4, total_tokens=34))
            observation = native_grok_observation(payload, exit_code=0)
            observation.update(prompt_tokens=23, completion_tokens=4, model_id=model_name,
                session_id="authored-native-session", exit_code=0)
            set_last_cli_observation("grok_cli", observation)
            if self.after is not None:
                self.after()
            if self.error is not None:
                raise self.error
            return ANSWER
    selected = AuthoredBackend()
    def get_provider(*args, **kwargs):
        if selected.initialize is not None:
            selected.initialize()
        return selected
    monkeypatch.setattr(llm_router, "get_llm_provider", get_provider)
    # Suppress external allocation/auth persistence, not the real router or core guard.
    monkeypatch.setattr(llm_router, "get_allocation_session", lambda *args, **kwargs: {})
    monkeypatch.setattr(llm_router, "_record_llm_allocation_observation", lambda **kwargs: None)
    monkeypatch.setattr(llm_router, "_capture_cli_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_key_slots, "apply_api_key_to_kwargs", lambda provider, kwargs, **rest: dict(kwargs))
    return selected


def arguments(case):
    result = []
    for key, value in case.items():
        if key == "required_paths":
            for path in value:
                result.extend(["--required-path", path])
        else:
            result.extend(["--" + key.replace("_", "-"), str(value)])
    return result


def assert_private_output(case, expected):
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o700
    assert {path.name for path in case["output"].iterdir()} == expected
    for path in case["output"].iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_nlink == 1
    return json.loads((case["output"] / "receipt.json").read_bytes())


def test_cli_actual_core_guard_and_router_deliver_one_complete_prompt(
        case, native_index, backend, monkeypatch, capsys):
    case.update(scheduler_state=native_index["state"], scheduler_policy_sha256=native_index["policy_sha256"])
    seen, routes = tokenizer(monkeypatch)
    assert subject.main(arguments(case)) == 0
    stdout = json.loads(capsys.readouterr().out)
    receipt = assert_private_output(case, {"prompt.txt", "response.txt", "receipt.json"})
    prompt = (case["output"] / "prompt.txt").read_bytes()
    response = (case["output"] / "response.txt").read_bytes()
    inspection = receipt["inspection"]
    assert len(backend.calls) == 1
    assert backend.calls[0]["prompt"].encode() == prompt
    assert backend.calls[0]["model"] == "grok-4.7" and response == ANSWER.encode()
    assert prompt.decode() in seen and routes == [("grok_cli", "grok-4.7")]
    _, advisory = render_model_prompt(prompt="", purpose="coding", workspace=case["root"])
    assert prompt.decode().startswith(advisory + GUIDANCE)
    body = json.loads(prompt.decode()[len(advisory) + len(GUIDANCE):])
    assert body["required_texts"] == [REQUIREMENTS]
    assert {row["path"]: row["content"].encode() for row in body["sources"]} == {
        "policy.py": POLICY, "ledger.py": TARGET}
    assert inspection["model_prompt_sha256"] == receipt["prompt_sha256"] == hashlib.sha256(prompt).hexdigest()
    assert inspection["backend_dispatches"] == 1 and inspection["predispatch_read_set_rechecked"] is True
    assert inspection["closing_read_set_rechecked"] is True
    assert inspection["native_rollout_usage"]["usage"]["total_tokens"] == 34
    assert inspection["native_rollout_usage"]["cache_included_in_input"] is False
    assert set(inspection["read_set"]["selected_paths"]) == {"policy.py", "ledger.py"}
    assert inspection["read_set"]["selected_input_tokens"] is None
    assert inspection["read_set"]["selected_estimated_input_tokens"] == len(prompt)
    options = backend.calls[0]["options"]
    assert options["grok_tools"] == "read_file"
    assert options["grok_disallowed_tools"] == "read_file,search_tool,use_tool"
    assert options["grok_max_turns"] == 1 and options["timeout"] == 1800
    assert options["max_tokens"] == options["max_new_tokens"] == 2048
    assert inspection["automatic_retry_allowed"] is inspection["effective_toolset_verified"] is False
    assert receipt["task_acceptance"] is receipt["billing_total_verified"] is False
    assert stdout["response_sha256"] == hashlib.sha256(response).hexdigest()
    encoded = json.dumps(receipt) + json.dumps(stdout)
    assert all(marker not in encoded for marker in (ANSWER, "MANDATORY_ORIGINAL_COMMENT", "SELECTED_ORIGINAL_COMMENT", REQUIREMENTS))
    assert all(not path.parent.exists() for path in native_index["paths"])


@pytest.mark.parametrize("failure", ["backend_error", "requirements_after_return", "source_after_return"])
def test_failed_call_keeps_native_usage_but_no_prompt_response_or_private_error(
        case, native_index, backend, monkeypatch, capsys, failure):
    tokenizer(monkeypatch)
    if failure == "backend_error":
        backend.error = RuntimeError(PRIVATE)
    elif failure == "requirements_after_return":
        backend.after = lambda: case["requirements_file"].write_text(PRIVATE)
    else:
        backend.after = lambda: (case["root"] / "ledger.py").write_text("# changed original\n")
    assert subject.main(arguments(case)) == 2
    stdout = json.loads(capsys.readouterr().out)
    receipt = assert_private_output(case, {"receipt.json"})
    assert len(backend.calls) == 1
    assert receipt["status"] == "failed" and stdout["receipt_retained"] is True
    assert receipt["inspection"]["native_rollout_usage"]["usage"]["total_tokens"] == 34
    assert receipt["inspection"]["backend_dispatches"] == 1
    assert PRIVATE not in json.dumps(receipt) + json.dumps(stdout)
    assert ANSWER not in json.dumps(receipt) + json.dumps(stdout)


@pytest.mark.parametrize("error", [RuntimeError(PRIVATE), KeyboardInterrupt(), SystemExit(7)])
def test_callable_preserves_original_exception_identity_and_attempt_receipt(
        case, native_index, backend, monkeypatch, error):
    tokenizer(monkeypatch)
    backend.error = error
    publications = []
    with pytest.raises(type(error)) as raised:
        subject.run_repository_read_set_inspection(**case, failure_publication_observer=publications.append)
    assert raised.value is error and len(backend.calls) == 1
    assert publications[0]["receipt_retained"] is True
    receipt = assert_private_output(case, {"receipt.json"})
    assert receipt["inspection"]["native_rollout_usage"]["usage"]["total_tokens"] == 34


def test_changed_requirements_at_backend_initialization_refuse_before_callback(
        case, native_index, backend, monkeypatch, capsys):
    tokenizer(monkeypatch)
    backend.initialize = lambda: case["requirements_file"].write_text("changed requirement\n")
    assert subject.main(arguments(case)) == 2
    assert backend.calls == []
    receipt = assert_private_output(case, {"receipt.json"})
    assert receipt["inspection"]["backend_dispatches"] == 0
    assert receipt["inspection"]["native_rollout_usage"] is None
    assert "changed requirement" not in json.dumps(receipt) + capsys.readouterr().out


@pytest.mark.parametrize("change", ["cwd", "route", "model", "timeout", "output_tokens", "existing_output",
    "output_symlink", "requirements_alias", "duplicate_path", "scheduler_pair"])
def test_early_refusal_precedes_tokenizer_index_and_provider(
        case, native_index, backend, monkeypatch, change):
    seen, routes = tokenizer(monkeypatch)
    if change == "cwd": monkeypatch.chdir(case["root"].parent)
    elif change == "route": case["provider"] = "codex_cli"
    elif change == "model": case["model"] = "unqualified-model"
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
        subject.run_repository_read_set_inspection(**case)
    assert routes == seen == backend.calls == native_index["paths"] == []
