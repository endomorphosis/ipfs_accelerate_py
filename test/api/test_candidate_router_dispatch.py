"""Candidate router behavior with native prepared context and authored dispatch.

Native public requirements, changed-source capsules, formal artifacts and reply
grammar are real. The independently tested handoff admission is supplied by a
test double here; both model providers are replaced and never called live.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_reply_contract as replies
from test.api.test_owner_candidate_round_prompt import (  # noqa: F401
    public_service, selected, scenario, _candidate)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


@pytest.fixture
def candidate_route(public_service, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation
    from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context
    from ipfs_accelerate_py.agent_supervisor.runtime import codex_usage_receipt
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client

    service, public_args, source = public_service
    original = prompts.capture_public_requirements(source_broker=service, **public_args)
    snapshot = _candidate(service)
    prepared = prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)
    prompt = prepared.provider_prompt.decode()
    metadata = snapshot.metadata
    binding = dict(schema="supervisor-candidate-dispatch-observation@1",
        task_id=metadata["task_id"], native_prompt_sha256=sha(original.native_prompt),
        native_prompt_bytes=len(original.native_prompt), provider_input_sha256=sha(prepared.provider_prompt),
        provider_input_bytes=len(prepared.provider_prompt), session_id="c"*32, round_number=2,
        round_context_sha256=sha(snapshot.metadata_bytes),
        workspace_generation=metadata["workspace_generation"],
        candidate_scope_cid=metadata["candidate_scope_cid"],
        candidate_semantic_root_cid=metadata["semantic_root_cid"],
        expires_monotonic_ns=9_000_000_000_000_000_000, remaining_timeout_seconds=30,
        dispatch_deadline_monotonic_ns=time.monotonic_ns() + 90_000_000_000,
        candidate_artifacts=metadata["artifacts"],
        proof_authority=False, completion_authority=False, settlement_authority=False)
    state = dict(provider="codex_cli", model="gpt-6.1-sol", validation_calls=0,
        discovery_calls=0, calls=[], binding=binding, prepared=prepared,
        original=original, source=source, service=service, preflight_calls=0, admitted_prompt=prompt)
    state["reply"] = json.dumps(dict(schema=replies.SCHEMA, status="candidate_ready",
        **{key: binding[key] for key in replies.BINDING_FIELDS}))
    monkeypatch.chdir(service.workspace)
    monkeypatch.setenv("CODEX_HOME", str(service.workspace.parent / "empty-codex"))
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary",
        lambda **kwargs: {"authored_boundary": True})
    monkeypatch.setattr(codex_usage_receipt, "recover_codex_usage", lambda **_: None)
    monkeypatch.setattr(semantic_router_translation, "encode_semantic_router_prompt",
        lambda **_: pytest.fail("candidate route reused baseline semantic encoder"))
    monkeypatch.setattr(source_formalization_context, "validate_source_formalization_context",
        lambda **_: pytest.fail("candidate route reused baseline source formalization"))

    def validate(**kwargs):
        state["validation_calls"] += 1
        assert kwargs["prompt"] == state["admitted_prompt"]
        assert kwargs["repository"] == service.repository
        assert kwargs["workspace"] == service.workspace
        assert kwargs["launch_settings"] == dict(provider=state["provider"], model=state["model"],
            reasoning_effort="high", max_output_tokens=128, budget_profile=None,
            grok_coding_max_turns=128)
        if state.get("validation_error_at") == state["validation_calls"]:
            raise ValueError("authored candidate preflight refusal")
        # Worker validation checks current candidate bytes; it does not replay
        # the owner-private requirement capture or the baseline encoder.
        source_formalization_context._current_sources(service.workspace, metadata["source_hashes"])
        return dict(binding, remaining_timeout_seconds=state.get("remaining", 30))

    monkeypatch.setattr(runner, "_candidate_dispatch_observation", validate)
    def request_summary(path, digest, **kwargs):
        state["preflight_calls"] += 1
        assert state["validation_calls"] == 2
        assert path == state["kwargs"]["candidate_round_descriptor"] and digest == "d"*64
        assert kwargs == dict(artifact="summary", expected_workspace_generation=binding["workspace_generation"],
                              max_bytes=8192, workspace=service.workspace)
        if state.get("preflight_error"):
            return {"ok": False, "reason_code": "unregistered_child"}
        response = dict(ok=True, page=dict(artifact_sha256=binding["candidate_artifacts"]["summary"]["sha256"],
            workspace_generation=binding["workspace_generation"], content="authored summary page"))
        response["page"].update(state.get("preflight_changes", {}))
        if state.get("after_preflight"):
            state["after_preflight"]()
        return response
    monkeypatch.setattr(worker_candidate_round_client, "request_context_page", request_summary)
    def discover():
        state["discovery_calls"] += 1
        assert state["validation_calls"] == 1
        return [state["provider"]]
    monkeypatch.setattr(intelligence_index, "discover_available_providers", discover)
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider=state["provider"], model_name=state["model"], catalog_revision="fixture"))
    def provider(*args, **kwargs):
        if state.get("after_discovery"):
            state["after_discovery"]()
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", provider)

    def generate(model_prompt, **kwargs):
        state["calls"].append((model_prompt, kwargs))
        assert state["validation_calls"] == 2
        assert state["preflight_calls"] == 1
        assert os.environ[runner.CANDIDATE_DESCRIPTOR_ENV] == str(service.boundary.parent / "candidate.json")
        assert os.environ[runner.CANDIDATE_DESCRIPTOR_DIGEST_ENV] == "d"*64
        assert runner.SOURCE_DESCRIPTOR_ENV not in os.environ
        assert kwargs["side_effecting"] is True
        assert kwargs["allow_local_fallback"] is kwargs["allow_cross_provider_fallback"] is False
        observation = {"exit_code": 0, "prompt_tokens": 123, "completion_tokens": 7, "cached_tokens": 19}
        if state["provider"] == "codex_cli":
            raw = canonical(kwargs["codex_output_schema"])
            observation.update(codex_output_schema_sha256=sha(raw), codex_output_schema_bytes=len(raw))
        else:
            from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
            observation.update(native_grok_observation(dict(stopReason="end_turn", num_turns=2,
                usage=dict(input_tokens=123, cache_read_input_tokens=19, output_tokens=7, total_tokens=149)), exit_code=0))
        observation.update(state.get("observation_overrides", {}))
        set_last_cli_observation(state["provider"], observation)
        if state.get("after_generate"):
            state["after_generate"]()
        if state.get("provider_error"):
            raise RuntimeError("authored provider failure with private text")
        return state["reply"]
    monkeypatch.setattr(llm_router, "generate_text", generate)
    state["kwargs"] = dict(prompt=prompt, provider=state["provider"], model=state["model"],
        timeout=60, max_output_tokens=128, semantic_repository=service.repository,
        container_boundary=service.boundary, container_boundary_sha256="b"*64,
        candidate_dispatch_handoff=service.boundary.parent / "dispatch.json",
        candidate_dispatch_handoff_sha256="e"*64,
        candidate_round_descriptor=service.boundary.parent / "candidate.json",
        candidate_round_descriptor_sha256="d"*64)
    return state


def run(state, **changes):
    return runner.run(**(state["kwargs"] | dict(provider=state["provider"], model=state["model"]) | changes))


@pytest.mark.parametrize("provider", ["codex_cli", "grok_cli"])
def test_candidate_dispatch_preserves_native_public_requirements_and_accounts_input(candidate_route, provider):
    state = candidate_route
    state.update(provider=provider, model="grok-4.7" if provider == "grok_cli" else "gpt-6.1-sol")
    output, receipt = run(state)
    assert json.loads(output) == json.loads(state["reply"])
    assert state["discovery_calls"] == 1 and state["validation_calls"] == 2
    assert len(state["calls"]) == receipt["router_calls"] == 1
    actual, options = state["calls"][0]
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_prompt_projection import (
        project_candidate_prompt, restore_candidate_prompt, validate_selection_receipt)
    projection = validate_selection_receipt(receipt["candidate_prompt_projection"])
    view = project_candidate_prompt(state["prepared"].provider_prompt.decode())
    selected = projection["selected_mode"] != "legacy"
    assert (view.provider_prompt if selected else view.original_prompt) in actual
    assert restore_candidate_prompt(view.provider_prompt, projection["view_receipt"]) == view.original_prompt
    assert projection["selected_complete_sha256"] == sha(actual.encode())
    assert projection["selected_complete_bytes"] <= projection["native_complete_bytes"]
    wire = json.loads(state["prepared"].provider_prompt)
    assert wire["native_prompt"].encode() == state["original"].native_prompt
    assert wire["original_public_instruction_block"].encode() == state["original"].public_instruction_block
    assert state["source"].decode() in wire["original_public_instruction_block"]
    assert wire["candidate_binding"]["semantic_root_cid"] != wire["baseline_binding"]["semantic_root_cid"]
    assert "--select --artifact catalog" in actual and "--select --artifact obligations" in actual
    assert actual.index("Supervisor second coding round reply contract") > actual.index("--select --artifact catalog")
    assert receipt["native_prompt_sha256"] == sha(state["original"].native_prompt)
    assert receipt["prompt_sha256"] == sha(state["prepared"].provider_prompt)
    assert receipt["native_prompt_sha256"] != receipt["prompt_sha256"]
    assert receipt["model_prompt_sha256"] == sha(actual.encode())
    assert receipt["semantic_translation"] is None and "source_formalization_context" not in receipt
    assert "semantic_response_translation" not in receipt and "source_view_retrieval" not in receipt
    assert receipt["candidate_reply_contract"]["response_validated"] is True
    assert receipt["candidate_round_dispatch"]["workspace_generation"] == wire["candidate_binding"]["workspace_generation"]
    assert receipt["candidate_retrieval_guidance"]["requests_observed"] is False
    assert receipt["candidate_round_preflight"]["requests"] == state["preflight_calls"] == 1
    assert receipt["candidate_round_preflight"]["provider_input_bytes"] == 0
    assert receipt["candidate_round_preflight"]["response_bytes"] <= 8192
    assert receipt["timeout_seconds"] == options["timeout"] == 30
    assert receipt["status"] == "provider_returned" and receipt["completion_authority"] is False
    assert receipt["usage"]["prompt_tokens"] == 123
    assert runner.CANDIDATE_DESCRIPTOR_ENV not in os.environ
    assert runner.CANDIDATE_DESCRIPTOR_DIGEST_ENV not in os.environ
    if provider == "grok_cli":
        assert options["grok_max_turns"] == 128
        assert "grok_stream_progress" not in options and "grok_progress_callback" not in options
        assert options["response_format"]["json_schema"]["schema"] == replies.candidate_completion_schema(state["binding"])
        assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 149


def test_candidate_projection_is_not_attempted_before_handoff_validation(candidate_route, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_prompt_projection as projection
    state = candidate_route
    state["validation_error_at"] = 1
    monkeypatch.setattr(projection, "project_candidate_prompt",
        lambda *_: pytest.fail("projection preceded candidate admission"))
    with pytest.raises(ValueError, match="preflight refusal"):
        run(state)
    assert not state["calls"] and state["discovery_calls"] == 0


@pytest.mark.parametrize("operation", ["project_candidate_prompt", "select_candidate_prompt"])
def test_unsupported_candidate_projection_retains_original_input(candidate_route, monkeypatch, operation):
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_prompt_projection as projection
    state = candidate_route
    def unsupported(*_):
        raise projection.CandidatePromptProjectionError("authored unsupported representation")
    monkeypatch.setattr(projection, operation, unsupported)
    _, receipt = run(state)
    assert "candidate_prompt_projection" not in receipt
    assert state["prepared"].provider_prompt.decode() in state["calls"][0][0]
    assert receipt["candidate_round_preflight"]["requests"] == 1


def test_projection_reply_overflow_preserves_valid_admitted_original(candidate_route):
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_prompt_projection as projection
    state = candidate_route
    wire = json.loads(state["kwargs"]["prompt"])
    # This boundary fixture has little nested escaping and a large unknown
    # outer field. Its fully preserved alternative is necessarily larger.
    native = canonical(dict(contract_version=1, evidence=[], goal="g", scope="s",
        acceptance="a", authority="u")).decode()
    wire["native_prompt"] = native
    wire["unknown_outer_padding"] = ""
    def render(prompt):
        model, _ = runner.render_model_prompt(prompt=prompt + runner._candidate_round_guidance(state["binding"]),
            purpose="coding", workspace=state["service"].workspace, candidate_transport=True)
        return replies.apply_candidate_reply_contract(model_prompt=model, binding=state["binding"])[0]
    small = canonical(wire).decode()
    view = projection.project_candidate_prompt(small)
    overhead = len(view.provider_prompt.encode()) - len(small.encode())
    assert overhead > 1
    wire["unknown_outer_padding"] = "x" * (replies.MAX_PROMPT_BYTES - len(render(small).encode()) - 1)
    prompt = canonical(wire).decode()
    original_model = render(prompt)
    view = projection.project_candidate_prompt(prompt)
    with pytest.raises(ValueError, match="reply contract exceeds bound"):
        render(view.provider_prompt)
    state["admitted_prompt"] = state["kwargs"]["prompt"] = prompt
    state["binding"].update(native_prompt_sha256=sha(native.encode()), native_prompt_bytes=len(native.encode()),
        provider_input_sha256=sha(prompt.encode()), provider_input_bytes=len(prompt.encode()))
    _, receipt = run(state)
    assert "candidate_prompt_projection" not in receipt
    assert state["calls"][0][0] == original_model
    assert receipt["model_prompt_bytes"] == replies.MAX_PROMPT_BYTES - 1
    assert receipt["candidate_round_preflight"]["requests"] == 1


@pytest.mark.parametrize("failure", [None, "wrong_generation", "cancelled", "exit_failure"])
def test_candidate_grok_options_reach_real_structured_adapter(candidate_route, monkeypatch, capsys, failure):
    """Exercise adapter option validation too; only native CLI execution is authored."""
    from ipfs_accelerate_py import llm_router

    state = candidate_route
    state.update(provider="grok_cli", model="grok-4.7")
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    native_provider = llm_router._get_grok_cli_provider()
    subprocess_run = llm_router.subprocess.run
    calls = []
    reply = json.loads(state["reply"])
    if failure == "wrong_generation":
        reply["workspace_generation"] = "0" * 64

    def native_cli(argv, **kwargs):
        if not argv or argv[0] != "/fixture/grok":
            return subprocess_run(argv, **kwargs)
        calls.append(argv)
        assert argv[argv.index("--output-format") + 1] == "json"
        assert argv[argv.index("--max-turns") + 1] == "128"
        assert "--json-schema" in argv
        return SimpleNamespace(returncode=1 if failure == "exit_failure" else 0, stderr="",
            stdout=json.dumps(dict(type="end", stopReason="cancelled" if failure == "cancelled" else "end_turn",
                structuredOutput=reply, num_turns=2,
                usage=dict(input_tokens=123, cache_read_input_tokens=19, output_tokens=7, total_tokens=149))))

    monkeypatch.setattr(llm_router.subprocess, "run", native_cli)

    def native_adapter(prompt, **kwargs):
        assert kwargs["side_effecting"] is True
        assert kwargs["allow_local_fallback"] is kwargs["allow_cross_provider_fallback"] is False
        return native_provider.generate(prompt, **kwargs)

    monkeypatch.setattr(llm_router, "generate_text", native_adapter)
    if failure is None:
        output, receipt = run(state)
        assert json.loads(output) == reply
        assert receipt["status"] == "provider_returned"
        assert receipt["candidate_reply_contract"]["response_validated"] is True
    else:
        with pytest.raises((ValueError, llm_router.LLMRouterError)):
            run(state)
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
        assert receipt["status"] == "failed"
        assert receipt["candidate_reply_contract"]["response_validated"] is False
    assert len(calls) == receipt["router_calls"] == 1
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 149
    assert receipt["completion_authority"] is False


def test_candidate_reply_binds_input_generation_after_provider_edits(candidate_route):
    state = candidate_route
    path = state["service"].workspace / "answer.py"
    state["after_generate"] = lambda: path.write_text("def repaired(): return 42\n")
    assert run(state)[1]["status"] == "provider_returned"
    assert sha(path.read_bytes()) != json.loads(state["prepared"].provider_prompt)["candidate_binding"]["source_hashes"]["answer.py"]
    assert state["validation_calls"] == 2


@pytest.mark.parametrize("provider", ["codex_cli", "grok_cli"])
@pytest.mark.parametrize("failure", ["baseline", "wrong_generation", "wrong_round", "provider_failure", "exit_failure"])
def test_failed_candidate_attempt_preserves_native_usage(candidate_route, provider, failure, capsys):
    state = candidate_route
    state.update(provider=provider, model="grok-4.7" if provider == "grok_cli" else "gpt-6.1-sol")
    reply = json.loads(state["reply"])
    if failure == "baseline":
        reply = {"schema": "supervisor-semantic-router-reply@1", "response": "must not be accepted"}
    elif failure == "wrong_generation":
        reply["workspace_generation"] = "0"*64
    elif failure == "wrong_round":
        reply["round_number"] = 1
    elif failure == "provider_failure":
        state["provider_error"] = True
    else:
        state["observation_overrides"] = {"exit_code": 7}
    state["reply"] = json.dumps(reply)
    with pytest.raises((ValueError, RuntimeError)):
        run(state)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["status"] == "failed" and receipt["router_calls"] == len(state["calls"]) == 1
    assert receipt["usage"]["prompt_tokens"] == 123
    assert receipt["candidate_reply_contract"]["response_validated"] is False
    assert receipt["candidate_round_dispatch"]["session_id"] == state["binding"]["session_id"]
    assert "private text" not in json.dumps(receipt)
    assert runner.CANDIDATE_DESCRIPTOR_ENV not in os.environ


@pytest.mark.parametrize("stage", [1, 2])
def test_candidate_context_refusal_prevents_provider_call(candidate_route, stage, capsys):
    state = candidate_route
    state["validation_error_at"] = stage
    with pytest.raises(ValueError, match="preflight refusal"):
        run(state)
    assert state["calls"] == []
    assert state["discovery_calls"] == stage - 1
    if stage == 2:
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
        assert receipt["router_calls"] == 0 and receipt["usage"] == {}
        assert receipt["failure_phase"] == "candidate_dispatch_context"


def test_candidate_deadline_is_clipped_again_after_provider_discovery(candidate_route):
    state = candidate_route
    state["after_discovery"] = lambda: state.update(remaining=2)
    assert run(state)[1]["timeout_seconds"] == state["calls"][0][1]["timeout"] == 2


def test_candidate_expired_after_discovery_cannot_create_fresh_budget(candidate_route):
    state = candidate_route
    state["after_discovery"] = lambda: state.update(remaining=0)
    with pytest.raises(ValueError, match="remaining admitted time"):
        run(state)
    assert state["calls"] == []


@pytest.mark.parametrize("failure", ["custody", "artifact", "generation", "oversized"])
def test_candidate_summary_handshake_refuses_before_provider(candidate_route, failure, capsys):
    state = candidate_route
    if failure == "custody":
        state["preflight_error"] = True
    else:
        key = {"artifact": "artifact_sha256", "generation": "workspace_generation", "oversized": "content"}[failure]
        state["preflight_changes"] = {key: "f"*64 if failure != "oversized" else "x"*8192}
    with pytest.raises(ValueError, match="candidate summary"):
        run(state)
    assert state["calls"] == [] and state["preflight_calls"] == 1
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["router_calls"] == 0 and receipt["failure_phase"] == "candidate_round_preflight"
    assert "candidate_round_preflight" not in receipt and receipt["usage"] == {}


def test_candidate_budget_is_rechecked_after_summary_handshake(candidate_route, monkeypatch, capsys):
    state = candidate_route
    native_time = runner.time
    clock = {"ns": native_time.monotonic_ns()}
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=native_time.monotonic,
        monotonic_ns=lambda: clock["ns"]))
    state["after_preflight"] = lambda: clock.update(ns=state["binding"]["dispatch_deadline_monotonic_ns"] + 1)
    with pytest.raises(ValueError, match="after preflight"):
        run(state)
    assert state["calls"] == [] and state["preflight_calls"] == 1
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["candidate_round_preflight"]["requests"] == 1 and receipt["router_calls"] == 0


@pytest.mark.parametrize("changes", [
    {"candidate_dispatch_handoff": None}, {"candidate_round_descriptor_sha256": None},
    {"semantic_repository": None}, {"purpose": "planning"},
    {"source_formalization_handoff": Path("/untrusted/baseline")},
    {"public_instruction_task_cid": "old-public-instruction"},
    {"doctor_residual_task_cid": "old-residual"},
    {"semantic_metadata_view": "common-bindings@1"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"semantic_working_set": "capsule-references@1"}, {"coding_reply_mode": "ordinary-completion@1"},
])
def test_candidate_route_rejects_partial_or_mixed_baseline_selection(candidate_route, changes):
    state = candidate_route
    with pytest.raises(ValueError, match="separate complete isolated"):
        run(state, **changes)
    assert state["validation_calls"] == state["discovery_calls"] == 0 and state["calls"] == []


@pytest.mark.parametrize("schema", [
    "supervisor-owner-candidate-router-input@1", "supervisor-owner-candidate-router-input@99",
    "supervisor-candidate-provider-view@1", "supervisor-candidate-provider-view@99",
])
def test_ordinary_route_still_rejects_preparation_and_inherited_candidate_environment(monkeypatch, schema):
    kwargs = dict(provider="codex_cli", model="gpt-6.1-sol", timeout=1, max_output_tokens=1)
    with pytest.raises(ValueError, match="independent dispatch admission"):
        runner.run(prompt=json.dumps({"schema": schema}), **kwargs)
    monkeypatch.setenv(runner.CANDIDATE_DESCRIPTOR_ENV, "/forged/descriptor")
    with pytest.raises(ValueError, match="explicit owner selection"):
        runner.run(prompt="native instruction", **kwargs)
