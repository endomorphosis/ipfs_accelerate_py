"""Owner attempt labels join ordinary receipts without granting continuation.

The existing boundary validator and providers are authored seams. No container,
provider call, launch admission, or native retry authority is created here.
"""
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


ATTEMPT = "1a" * 16


def options(**changes):
    return dict(prompt="Repair the bounded arithmetic fixture.", provider="codex_cli", model="pinned",
        timeout=30, max_output_tokens=128, purpose="coding", owner_worker_attempt_id=ATTEMPT,
        container_boundary=Path("/authored/boundary.json"), container_boundary_sha256="a" * 64) | changes


@pytest.fixture
def authored(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer, container_worker_boundary

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    state = dict(seen=[], exit_code=0, boundaries=[], discovery=0)
    def discover():
        state["discovery"] += 1
        return ["codex_cli"]
    def boundary(**kwargs):
        state["boundaries"].append(kwargs)
        return {"authored_boundary": True}
    def generate(prompt, **kwargs):
        state["seen"].append((prompt, kwargs))
        set_last_cli_observation("codex_cli", dict(exit_code=state["exit_code"],
            prompt_tokens=15, completion_tokens=4))
        return "Authored response; independent validation remains required."
    monkeypatch.setattr(intelligence_index, "discover_available_providers", discover)
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="authored"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(llm_router, "generate_text", generate)
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kwargs: (None, None))
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", boundary)
    return state


@pytest.mark.parametrize("exit_code", [0, 7])
def test_supplied_attempt_identity_is_retained_in_original_success_and_failure_receipts(authored, capsys, exit_code):
    authored["exit_code"] = exit_code
    if exit_code:
        with pytest.raises(RuntimeError, match="observed successful exit"):
            runner.run(**options())
    else:
        output, returned = runner.run(**options())
        assert output and returned["owner_worker_attempt_id"] == ATTEMPT
    retained = json.loads(capsys.readouterr().out)
    assert retained["owner_worker_attempt_id"] == ATTEMPT
    assert retained["invocation_id"] != ATTEMPT
    assert retained["status"] == ("failed" if exit_code else "provider_returned")
    assert retained["completion_authority"] is False
    assert retained["usage"]["prompt_tokens"] == 15
    assert len(authored["seen"]) == authored["discovery"] == 1
    assert len(authored["boundaries"]) == 1
    assert authored["boundaries"][0]["expected_sha256"] == "a" * 64
    # The label joins owner evidence; it is not provider session identity,
    # injected model context, or a provider argument authorizing another call.
    text, provider_args = authored["seen"][0]
    assert ATTEMPT not in text and "owner_worker_attempt_id" not in provider_args
    assert provider_args["allocation_session_id"] == retained["invocation_id"]


@pytest.mark.parametrize("isolated", [True, False])
def test_absent_attempt_identity_keeps_ordinary_host_and_container_receipt_shape(authored, isolated):
    overrides = {} if isolated else dict(container_boundary=None, container_boundary_sha256="")
    _, receipt = runner.run(**options(owner_worker_attempt_id=None, **overrides))
    assert "owner_worker_attempt_id" not in receipt
    assert len(authored["seen"]) == 1


@pytest.mark.parametrize("identity", ["", "a" * 31, "a" * 33, "A" * 32, "g" * 32,
    " " + "a" * 31, "a" * 31 + "\n", "é" * 32, True, False, 0, {}, []])
def test_malformed_owner_identity_refuses_before_provider_discovery(monkeypatch, identity):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("invalid owner attempt identity reached provider discovery"))
    with pytest.raises(ValueError, match="exactly 32 lowercase"):
        runner.run(**options(owner_worker_attempt_id=identity))


@pytest.mark.parametrize("changes", [dict(purpose="planning"), dict(census_only=True),
    dict(census_only=None), dict(census_only=0), dict(container_boundary=None),
    dict(container_boundary_sha256=""), dict(container_boundary=None, container_boundary_sha256="")])
def test_attempt_identity_requires_isolated_coding_generation_before_discovery(monkeypatch, changes):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("unsupported owner attempt route reached provider discovery"))
    with pytest.raises(ValueError, match="isolated coding generation"):
        runner.run(**options(**changes))


def test_existing_container_verification_cannot_be_bypassed_by_observation_identity(authored, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    def refused(**kwargs):
        raise ValueError("authored existing boundary refusal")
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", refused)
    with pytest.raises(ValueError, match="existing boundary refusal"):
        runner.run(**options())
    assert authored["seen"] == [] and authored["discovery"] == 0


@pytest.mark.parametrize("purpose", ["coding", "planning"])
def test_attempt_identity_never_authorizes_reserved_candidate_preparation(monkeypatch, purpose):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("unadmitted candidate reached provider discovery"))
    prompt = json.dumps(dict(schema="supervisor-owner-candidate-router-input@1", dispatch_usable=True,
        native_prompt="Retained original requirements."))
    with pytest.raises(ValueError, match="independent dispatch admission"):
        runner.run(**options(prompt=prompt, purpose=purpose))


@pytest.mark.parametrize("identity", [ATTEMPT, None])
def test_cli_passes_explicit_owner_identity_without_inventing_a_default(monkeypatch, capsys, identity):
    seen = []
    def recorded(**kwargs):
        seen.append(kwargs)
        return "Authored CLI response.", {}
    monkeypatch.setattr(runner, "run", recorded)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"Literal owner prompt.")))
    argv = ["router", "--provider", "codex_cli", "--model", "pinned",
        "--container-boundary", "/authored/boundary.json", "--container-boundary-sha256", "a" * 64]
    if identity is not None:
        argv.extend(["--owner-worker-attempt-id", identity])
    monkeypatch.setattr(runner.sys, "argv", argv)
    assert runner.main() == 0
    assert seen[0]["owner_worker_attempt_id"] == identity
    assert seen[0]["prompt"] == "Literal owner prompt."
    assert capsys.readouterr().out.strip() == "Authored CLI response."
