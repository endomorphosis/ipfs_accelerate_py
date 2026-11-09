"""Candidate router admission with authored transport and provider boundaries.

Tests use the real issued client factory and a real local Git workspace. The
owner RPC, UID/container boundary and provider are authored seams; this does not
qualify a native owner handoff, a model call, candidate correctness or savings.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as round_client
from test.api.test_router_owner_worker_attempt_identity import ATTEMPT, authored, options  # noqa: F401
from test.api.test_worker_candidate_round_client import descriptor as round_descriptor


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _receipt(capsys):
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")


@pytest.fixture
def candidate(authored, tmp_path, monkeypatch):
    # This fixture authors transport/deployment policy, never native owner
    # authority. The real factory, registry, shape validators and exact-byte
    # binding still issue and check the typed local handle.
    descriptor = round_descriptor()
    payload = dict(schema="supervisor-owner-candidate-router-input@1",
        native_prompt="Complete retained requirement with numerical-formalization-context historical evidence.",
        original_public_instruction_block="Literal public retained requirement; preserve output scope and acceptance.",
        candidate_binding=dict(workspace_generation=descriptor["workspace_generation"]),
        dispatch_usable=False, baseline_evidence_applicable=False,
        proof_authority=False, execution_authority=False, completion_authority=False)
    prompt = client.canonical_json(payload).decode("utf-8")
    descriptor.update(schema=client.DESCRIPTOR_SCHEMA, workspace=str(Path.cwd()),
        boundary_manifest_sha256="a" * 64, admission_id="c" * 32,
        admission_binding_sha256="d" * 64, worker_attempt_id=ATTEMPT,
        candidate_prompt_sha256=_sha(prompt.encode()), candidate_prompt_bytes=len(prompt.encode()),
        original_requirements_sha256="e" * 64,
        original_native_prompt_sha256=descriptor["native_prompt_sha256"],
        public_instruction_block_sha256="f" * 64,
        absolute_deadline_ns=descriptor["expires_monotonic_ns"] + 1000000000,
        provider="codex_cli", model="pinned", timeout=30, max_output_tokens=128, reasoning_effort="high")
    client.validate_descriptor(descriptor)
    path = tmp_path / "authored-candidate-descriptor.json"
    raw = client.canonical_json(descriptor)
    path.write_bytes(raw); path.chmod(0o400)
    digest = _sha(raw)
    state = dict(requests=[], effective_timeout=20, refuse_consume=False, consume_environment=None)

    def load(selected_path, selected_digest, *, workspace=None):
        assert Path(selected_path) == path and selected_digest == digest
        assert workspace is None or Path(workspace) == Path.cwd()
        if workspace is not None and str(Path(workspace)) != descriptor["workspace"]:
            raise ValueError("authored transport preserved its exact workspace check")
        return client.validate_descriptor(deepcopy(descriptor))

    def exchange(selected_path, selected, request):
        assert Path(selected_path) == path
        client._request_binding(request, selected)
        state["requests"].append(deepcopy(request))
        operation = request["query"]["operation"]
        response = dict(schema=client.RESPONSE_SCHEMA, session_id=selected["session_id"],
            request_id=request["request_id"], ok=True, page=None, reason_code=None)
        if operation == "consume":
            state["consume_environment"] = {name: os.environ.get(name) for name in
                (round_client.DESCRIPTOR_ENV, round_client.DIGEST_ENV)}
            if state["refuse_consume"]:
                response.update(ok=False, reason_code="stale_context")
                return client.validate_response(response, request=request, descriptor=selected)
        response["page"] = dict(**{name: selected[name] for name in client.DISPATCH_BINDING_FIELDS},
            schema=client.PAGE_SCHEMA, session_id=selected["session_id"],
            invocation_id=request["query"]["invocation_id"], workspace_generation=selected["workspace_generation"],
            boundary_manifest_sha256=selected["boundary_manifest_sha256"], operation=operation,
            dispatch_consumed=operation == "consume", timeout_seconds=state["effective_timeout"],
            host_native_admission_only=True, serialized_observation_grants_authority=False,
            provider_dispatch_observed=False, provider_attempted=None, proof_authority=False,
            completion_authority=False, retry_authority=False, production_container_dispatch_qualified=False)
        return client.validate_response(response, request=request, descriptor=selected)

    monkeypatch.setattr(client, "load_descriptor", load)
    monkeypatch.setattr(client, "_exchange", exchange)
    prepared = client.prepare_candidate_router_input(descriptor_path=path, descriptor_sha256=digest,
        prompt=prompt, workspace=Path.cwd())
    return dict(input=prepared, prompt=prompt, descriptor=descriptor, state=state)


def _guidance(candidate):
    return "\n\n" + client.AdmittedCandidateRouterInput.guidance(candidate["input"])


def _invoke(case, **changes):
    kwargs = options(prompt=case["prompt"], candidate_round_input=case["input"])
    kwargs.update(changes)
    return runner.run(**kwargs)


@pytest.mark.parametrize("change", [
    dict(purpose="planning"), dict(provider="grok_cli", model="grok-4.7"),
    dict(container_boundary=None), dict(container_boundary_sha256=""),
    dict(owner_worker_attempt_id=None), dict(semantic_repository=Path("/unselected/canonical")),
    dict(world_task_packet={}), dict(world_packet_audit_directory=Path("/unselected/replay")),
    dict(semantic_transport_schema="supervisor-semantic-router-input@2"),
    dict(semantic_metadata_view="common-bindings@1"), dict(semantic_working_set="capsule-references@1"),
    dict(source_view_profile="identity"), dict(source_edit_mode="mapped-span@2"),
    dict(source_view_descriptor=Path("/unselected/source")), dict(source_view_descriptor_sha256="a" * 64),
    dict(source_lookup_descriptor=Path("/unselected/lookup")), dict(source_lookup_descriptor_sha256="a" * 64),
    dict(source_formalization_handoff=Path("/unselected/formalization")),
    dict(source_formalization_handoff_sha256="a" * 64),
    dict(numerical_obligation_descriptor=Path("/unselected/numerical")),
    dict(numerical_obligation_descriptor_sha256="a" * 64),
    dict(doctor_residual_artifact=Path("/unselected/doctor")), dict(doctor_residual_sha256="a" * 64),
    dict(public_instruction_artifact=Path("/unselected/public")), dict(public_instruction_sha256="a" * 64),
    dict(tool_output_policy="artifact-preview@1"), dict(source_view_advisory_policy="source-view-raw@1"),
    dict(repository_source_policy="repository-source-raw@1"), dict(source_input_policy="source-input-raw@1"),
    dict(query_reference_policy="capsule-columns@1"), dict(census_only=True), dict(grok_coding_max_turns=16),
])
def test_candidate_route_refuses_mixed_authorities_before_provider_discovery(candidate, authored, change):
    with pytest.raises((ValueError, TypeError)):
        _invoke(candidate, **change)
    assert authored["seen"] == [] and authored["discovery"] == 0


@pytest.mark.parametrize("unissued", [None, {}, object()])
def test_raw_preparation_and_unissued_objects_refuse_before_provider_discovery(candidate, authored, unissued):
    with pytest.raises(ValueError):
        _invoke(candidate, candidate_round_input=unissued)
    assert authored["seen"] == [] and authored["discovery"] == 0


def test_type_exact_but_unissued_client_cannot_bypass_live_issuance(candidate, authored):
    unissued = object.__new__(client.AdmittedCandidateRouterInput)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        _invoke(candidate, candidate_round_input=unissued)
    assert authored["seen"] == [] and authored["discovery"] == 0


@pytest.mark.parametrize("change", ["whitespace", "requirements", "attempt", "boundary", "model", "timeout", "output_cap", "reasoning"])
def test_exact_candidate_input_and_invocation_binding_are_required_before_discovery(candidate, authored, change):
    overrides = {}
    if change == "whitespace": overrides["prompt"] = candidate["prompt"] + "\n"
    elif change == "requirements": overrides["prompt"] = candidate["prompt"].replace("retained requirement", "changed requirement")
    elif change == "attempt": overrides["owner_worker_attempt_id"] = "b" * 32
    elif change == "boundary": overrides["container_boundary_sha256"] = "b" * 64
    elif change == "model": overrides["model"] = "other-model"
    elif change == "timeout": overrides["timeout"] = 31
    elif change == "output_cap": overrides["max_output_tokens"] = 129
    else: overrides["reasoning_effort"] = "low"
    with pytest.raises((ValueError, TypeError)):
        _invoke(candidate, **overrides)
    assert authored["seen"] == [] and authored["discovery"] == 0


def test_default_route_keeps_reserved_preparation_refusal_and_original_receipt_shape(authored):
    _output, receipt = runner.run(**options())
    expected, _ = runner.render_model_prompt(prompt=options()["prompt"], purpose="coding", workspace=Path.cwd())
    assert authored["seen"][0][0] == expected
    assert "candidate_round_dispatch" not in receipt
    assert len(authored["seen"]) == 1


def test_admitted_candidate_keeps_complete_original_json_requirements_and_reader_guidance(candidate, authored, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer, source_formalization_context
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kwargs:
        pytest.fail("candidate input must not choose an automatic tokenizer"))
    monkeypatch.setattr(source_formalization_context, "validate_source_formalization_context", lambda **kwargs:
        pytest.fail("candidate historical formal evidence cannot enter baseline source validator"))
    output, receipt = _invoke(candidate)
    expected, advisory = runner.render_model_prompt(prompt=candidate["prompt"] + _guidance(candidate),
        purpose="coding", workspace=Path.cwd())
    assert output and authored["seen"][0][0] == expected
    assert candidate["prompt"] in expected and "retained requirement" in expected
    assert receipt["native_prompt_sha256"] == _sha(candidate["prompt"].encode())
    assert receipt["model_prompt_sha256"] == _sha(expected.encode())
    assert receipt["router_prompt_sha256"] == _sha((candidate["prompt"] + _guidance(candidate)).encode())
    assert receipt["workspace_advisory_sha256"] == _sha(advisory.encode())
    assert receipt["semantic_translation"] is receipt["public_instruction"] is None
    assert "source_formalization_context" not in receipt and "world_task_packet" not in receipt
    assert receipt["owner_worker_attempt_id"] == ATTEMPT
    assert receipt["candidate_round_dispatch"]["provider_attempted"] is True
    assert receipt["router_calls"] == 1 and receipt["completion_authority"] is False


def test_explicit_counter_cannot_drop_candidate_reader_guidance(candidate, authored):
    counter = runner.PromptTokenCounter(provider="codex_cli", model="pinned",
        tokenizer_id="authored-byte-counter", tokenizer_revision="test-v1",
        special_token_policy="none", count_tokens=lambda text: len(text.encode()))
    _output, receipt = _invoke(candidate, prompt_token_counter=counter)
    expected, _ = runner.render_model_prompt(prompt=candidate["prompt"] + _guidance(candidate),
        purpose="coding", workspace=Path.cwd())
    assert authored["seen"][0][0] == expected
    assert receipt["model_prompt_sha256"] == _sha(expected.encode())


@pytest.mark.parametrize("exit_code", [0, 7])
def test_final_dispatch_uses_effective_remaining_timeout_and_retains_failed_usage(candidate, authored, capsys, exit_code):
    authored["exit_code"] = exit_code
    candidate["state"]["effective_timeout"] = 4
    if exit_code:
        with pytest.raises(RuntimeError, match="observed successful exit"):
            _invoke(candidate)
    else:
        _invoke(candidate)
    receipt = _receipt(capsys)
    assert len(authored["seen"]) == 1
    assert authored["seen"][0][1]["timeout"] == receipt["timeout_seconds"] == 4
    assert receipt["router_calls"] == 1 and receipt["usage"]["prompt_tokens"] == 15
    assert receipt["status"] == ("failed" if exit_code else "provider_returned")
    assert receipt["candidate_round_dispatch"]["dispatch_consumed"] is True
    assert receipt["candidate_round_dispatch"]["provider_attempted"] is True
    assert receipt["owner_worker_attempt_id"] == ATTEMPT


def test_final_authorization_failure_prints_failed_zero_call_receipt(candidate, authored, capsys):
    candidate["state"]["refuse_consume"] = True
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _invoke(candidate)
    receipt = _receipt(capsys)
    assert authored["discovery"] == 1 and authored["seen"] == []
    assert receipt["status"] == "failed" and receipt["failure_phase"] == "candidate_dispatch"
    assert receipt["router_calls"] == 0
    assert receipt["candidate_round_dispatch"]["provider_attempted"] is False
    assert receipt["candidate_round_dispatch"]["dispatch_requested"] is True
    assert receipt["candidate_round_dispatch"]["dispatch_consumed"] is None
    assert receipt["owner_worker_attempt_id"] == ATTEMPT
    assert receipt["usage"] == {}


def test_provider_initialization_cannot_bypass_late_owner_refusal(candidate, authored, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    def initialize(*args, **kwargs):
        candidate["state"]["refuse_consume"] = True
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _invoke(candidate)
    assert authored["seen"] == [] and _receipt(capsys)["router_calls"] == 0


def test_provider_initialization_cannot_change_the_current_allocated_workspace(candidate, authored, monkeypatch, capsys, tmp_path):
    from ipfs_accelerate_py import llm_router
    other = tmp_path / "other-cwd"
    other.mkdir()
    def initialize(*args, **kwargs):
        monkeypatch.chdir(other)
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises(ValueError, match="workspace"):
        _invoke(candidate)
    assert authored["seen"] == [] and _receipt(capsys)["router_calls"] == 0


@pytest.mark.parametrize("key", [round_client.DESCRIPTOR_ENV, round_client.DIGEST_ENV])
def test_inherited_candidate_endpoint_requires_explicit_typed_route(authored, monkeypatch, key):
    monkeypatch.setenv(key, "unselected-candidate-endpoint")
    with pytest.raises(ValueError):
        runner.run(**options())
    assert authored["seen"] == [] and authored["discovery"] == 0


@pytest.mark.parametrize("refused", [False, True])
def test_final_authorization_sees_only_selected_candidate_endpoint_and_environment_is_restored(
        candidate, authored, monkeypatch, capsys, refused):
    monkeypatch.setenv(round_client.DESCRIPTOR_ENV, "previous-public-descriptor")
    monkeypatch.delenv(round_client.DIGEST_ENV, raising=False)
    candidate["state"]["refuse_consume"] = refused
    if refused:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            _invoke(candidate)
        assert _receipt(capsys)["router_calls"] == 0
    else:
        _invoke(candidate)
    assert candidate["state"]["consume_environment"] == {
        round_client.DESCRIPTOR_ENV: str(candidate["input"].descriptor_path),
        round_client.DIGEST_ENV: candidate["input"].descriptor_sha256}
    assert os.environ[round_client.DESCRIPTOR_ENV] == "previous-public-descriptor"
    assert round_client.DIGEST_ENV not in os.environ


def test_cli_loads_the_selected_real_client_handle_and_forwards_exact_prompt(candidate, monkeypatch, capsys):
    seen = []
    original = client.prepare_candidate_router_input
    def prepared(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)
    def run(**kwargs):
        seen.append(kwargs)
        return "Authored CLI response.", {}
    monkeypatch.setattr(client, "prepare_candidate_router_input", prepared)
    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(candidate["prompt"].encode())))
    monkeypatch.setattr(runner.sys, "argv", ["router", "--provider", "codex_cli", "--model", "pinned",
        "--container-boundary", "/authored/boundary.json", "--container-boundary-sha256", "a" * 64,
        "--owner-worker-attempt-id", ATTEMPT,
        "--candidate-dispatch-descriptor", str(candidate["input"].descriptor_path),
        "--candidate-dispatch-descriptor-sha256", candidate["input"].descriptor_sha256])
    assert runner.main() == 0
    assert seen[0]["prompt"] == seen[1]["prompt"] == candidate["prompt"]
    assert type(seen[1]["candidate_round_input"]) is client.AdmittedCandidateRouterInput
    assert seen[1]["owner_worker_attempt_id"] == ATTEMPT
    assert capsys.readouterr().out.strip() == "Authored CLI response."
