"""The ordinary owner's deadline restricts one provider call without authority.

Provider, boundary and descriptor seams below are authored fixtures. They create
no native owner claim, candidate dispatch admission, or external provider call.
"""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


ATTEMPT = "1b" * 16
SECOND = 1_000_000_000


def options(**changes):
    return dict(prompt="Repair the bounded arithmetic fixture.", provider="codex_cli", model="pinned",
        timeout=30, max_output_tokens=128, purpose="coding", owner_worker_attempt_id=ATTEMPT,
        container_boundary=Path("/authored/boundary.json"), container_boundary_sha256="b" * 64,
        owner_provider_deadline_ns=20 * SECOND) | changes


@pytest.fixture
def authored(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer, container_worker_boundary

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    state = dict(now_ns=10 * SECOND, initialization_ns=0, descriptor_ns=0,
        seen=[], events=[], exit_code=0, provider_error=None, boundaries=[], discovery=0)

    def discover():
        state["discovery"] += 1
        return ["codex_cli"]

    def boundary(**kwargs):
        state["boundaries"].append(kwargs)
        return {"authored_boundary": True}

    def initialize(*args, **kwargs):
        state["events"].append("initialization")
        state["now_ns"] += state["initialization_ns"]
        return object()

    @contextmanager
    def final_descriptor(*args, **kwargs):
        state["events"].append("descriptor_entry")
        state["now_ns"] += state["descriptor_ns"]
        try:
            yield
        finally:
            state["events"].append("descriptor_exit")

    def generate(prompt, **kwargs):
        state["events"].append("provider")
        state["seen"].append((prompt, kwargs))
        if state["provider_error"] is not None:
            raise state["provider_error"]
        set_last_cli_observation("codex_cli", dict(exit_code=state["exit_code"],
            prompt_tokens=15, completion_tokens=4))
        return "Authored response; independent validation remains required."

    monkeypatch.setattr(intelligence_index, "discover_available_providers", discover)
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="authored"))
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    monkeypatch.setattr(llm_router, "generate_text", generate)
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kwargs: (None, None))
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", boundary)
    monkeypatch.setattr(runner, "_source_lookup_descriptor_environment", final_descriptor)
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: state["now_ns"])
    return state


@pytest.mark.parametrize("elapsed_ns", [0, 1])
def test_expired_owner_deadline_retains_zero_call_failure_receipt(authored, capsys, elapsed_ns):
    deadline = authored["now_ns"] - elapsed_ns
    with pytest.raises(TimeoutError):
        runner.run(**options(owner_provider_deadline_ns=deadline))
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["owner_provider_deadline_ns"] == deadline
    assert receipt["owner_worker_attempt_id"] == ATTEMPT
    assert receipt["router_calls"] == 0 and receipt["status"] == "failed"
    assert receipt["completion_authority"] is False
    assert authored["seen"] == []
    assert authored["events"] == ["initialization", "descriptor_entry", "descriptor_exit"]


@pytest.mark.parametrize("initialization_ns,descriptor_ns,remaining", [
    (3 * SECOND, 0, 7.0),
    (0, 9 * SECOND + 750_000_000, 0.25),
    (3 * SECOND, 6 * SECOND + 875_000_000, 0.125),
    (0, 10 * SECOND - 1, 0.000000001),
])
def test_deadline_is_recomputed_after_initialization_and_descriptor_entry(
        authored, capsys, initialization_ns, descriptor_ns, remaining):
    authored.update(initialization_ns=initialization_ns, descriptor_ns=descriptor_ns)
    output, receipt = runner.run(**options())
    assert output and receipt["status"] == "provider_returned"
    assert receipt["owner_provider_deadline_ns"] == 20 * SECOND
    assert receipt["router_calls"] == 1
    assert type(receipt["timeout_seconds"]) is float
    assert receipt["timeout_seconds"] == pytest.approx(remaining)
    prompt, provider_args = authored["seen"][0]
    assert type(provider_args["timeout"]) is float
    assert provider_args["timeout"] == pytest.approx(remaining)
    assert provider_args["timeout"] <= remaining
    assert "owner_provider_deadline_ns" not in provider_args
    assert str(20 * SECOND) not in prompt and ATTEMPT not in prompt
    assert authored["events"] == ["initialization", "descriptor_entry", "provider", "descriptor_exit"]
    assert json.loads(capsys.readouterr().out)["timeout_seconds"] == receipt["timeout_seconds"]


@pytest.mark.parametrize("stage", ["initialization_ns", "descriptor_ns"])
def test_time_consumed_before_dispatch_cannot_restart_expired_deadline(authored, capsys, stage):
    authored[stage] = 10 * SECOND
    with pytest.raises(TimeoutError):
        runner.run(**options())
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["owner_provider_deadline_ns"] == 20 * SECOND
    assert receipt["router_calls"] == 0 and receipt["status"] == "failed"
    assert authored["seen"] == []


@pytest.mark.parametrize("deadline", [40 * SECOND, 2**63 - 1])
def test_long_owner_deadline_cannot_expand_requested_provider_timeout(authored, capsys, deadline):
    _, receipt = runner.run(**options(timeout=3, owner_provider_deadline_ns=deadline))
    assert receipt["owner_provider_deadline_ns"] == deadline
    assert receipt["timeout_seconds"] == authored["seen"][0][1]["timeout"] == 3.0
    assert receipt["router_calls"] == 1
    assert json.loads(capsys.readouterr().out)["owner_provider_deadline_ns"] == deadline


@pytest.mark.parametrize("outcome", ["success", "nonzero_exit", "provider_exception"])
def test_positive_deadline_preserves_one_attempt_receipt_for_success_and_failure(authored, capsys, outcome):
    if outcome == "nonzero_exit":
        authored["exit_code"] = 7
        with pytest.raises(RuntimeError, match="observed successful exit"):
            runner.run(**options())
    elif outcome == "provider_exception":
        error = TimeoutError("authored provider timeout")
        authored["provider_error"] = error
        with pytest.raises(TimeoutError, match="authored provider timeout") as raised:
            runner.run(**options())
        assert raised.value is error
    else:
        _, returned = runner.run(**options())
        assert returned["status"] == "provider_returned"
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["owner_provider_deadline_ns"] == 20 * SECOND
    assert receipt["owner_worker_attempt_id"] == ATTEMPT
    assert receipt["router_calls"] == len(authored["seen"]) == 1
    assert receipt["timeout_seconds"] == authored["seen"][0][1]["timeout"] == 10.0
    assert receipt["status"] == ("provider_returned" if outcome == "success" else "failed")
    assert receipt["completion_authority"] is False


@pytest.mark.parametrize("deadline", [True, False, 0, -1, 2**63, 1.5, "20000000000", {}, []])
def test_malformed_deadline_refuses_before_provider_discovery(monkeypatch, deadline):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("malformed deadline reached provider discovery"))
    with pytest.raises(ValueError, match="deadline"):
        runner.run(**options(owner_provider_deadline_ns=deadline))


@pytest.mark.parametrize("changes", [
    dict(owner_worker_attempt_id=None), dict(container_boundary=None),
    dict(container_boundary_sha256=""),
    dict(owner_worker_attempt_id=None, container_boundary=None, container_boundary_sha256=""),
    dict(purpose="planning"), dict(census_only=True), dict(census_only=0),
])
def test_deadline_requires_existing_isolated_coding_owner_identity(monkeypatch, changes):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("unsupported owner deadline route reached provider discovery"))
    with pytest.raises(ValueError, match="deadline|isolated coding"):
        runner.run(**options(**changes))


def test_owner_deadline_cannot_bypass_existing_container_boundary(authored, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary

    def refuse(**kwargs):
        raise ValueError("authored existing boundary refusal")

    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", refuse)
    with pytest.raises(ValueError, match="existing boundary refusal"):
        runner.run(**options())
    assert authored["discovery"] == 0 and authored["seen"] == []


def test_ordinary_invocation_without_deadline_retains_existing_timeout(authored, capsys):
    _, receipt = runner.run(**options(owner_provider_deadline_ns=None))
    assert "owner_provider_deadline_ns" not in receipt
    assert receipt["timeout_seconds"] == authored["seen"][0][1]["timeout"] == 30
    assert receipt["router_calls"] == 1
    assert "owner_provider_deadline_ns" not in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("deadline", [None, 20 * SECOND])
def test_cli_forwards_explicit_owner_deadline_without_creating_a_default(monkeypatch, capsys, deadline):
    seen = []

    def recorded(**kwargs):
        seen.append(kwargs)
        return "Authored CLI response.", {}

    monkeypatch.setattr(runner, "run", recorded)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"Literal owner prompt.")))
    argv = ["router", "--provider", "codex_cli", "--model", "pinned",
        "--container-boundary", "/authored/boundary.json", "--container-boundary-sha256", "b" * 64,
        "--owner-worker-attempt-id", ATTEMPT]
    if deadline is not None:
        argv.extend(["--owner-provider-deadline-ns", str(deadline)])
    monkeypatch.setattr(runner.sys, "argv", argv)
    assert runner.main() == 0
    assert seen[0]["owner_provider_deadline_ns"] == deadline
    assert seen[0]["owner_worker_attempt_id"] == ATTEMPT
    assert seen[0]["prompt"] == "Literal owner prompt."
    assert capsys.readouterr().out.strip() == "Authored CLI response."
