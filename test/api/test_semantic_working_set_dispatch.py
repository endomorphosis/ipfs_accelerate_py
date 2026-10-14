"""Actual router/projection/socket expansion with a capturing offline provider.

Container UID admission is authored; native capsules, Git, registered child,
socket paging, receipt accounting and historical reconstruction are real.
"""
from copy import deepcopy
import hashlib
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime import worker_semantic_evidence_client as evidence
from test.api.test_worker_semantic_evidence import evidence_service, allocated, native, exchange, _bytes  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401


@pytest.fixture
def dispatch(evidence_service, provider, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
    service = evidence_service
    def load(path, digest, *, workspace=None):
        assert path == service.descriptor_path and digest == service.descriptor_sha256
        assert workspace == service.workspace
        return deepcopy(service.descriptor)
    monkeypatch.setattr(source, "load_descriptor", load)
    monkeypatch.setattr(boundary, "verify_container_worker_boundary", lambda **kwargs:
        dict(manifest_sha256="b" * 64, owner_uid=os.getuid(), worker_uid=os.getuid() + 1))
    def page(path, digest, **kwargs):
        assert path == service.descriptor_path and digest == service.descriptor_sha256
        assert kwargs.pop("workspace") == service.workspace
        return exchange(service, {"client": kwargs})
    monkeypatch.setattr(evidence, "request_page", page)
    arguments = dict(prompt=service.prompt, provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=4096, semantic_repository=service.repository,
        container_boundary=service.boundary, container_boundary_sha256="b" * 64,
        source_view_profile="identity@1", source_view_descriptor=service.descriptor_path,
        source_view_descriptor_sha256=service.descriptor_sha256,
        semantic_working_set="capsule-references@1")
    return service, provider, arguments


@pytest.mark.parametrize("reply_mode", ["legacy", "ordinary-completion@1"])
def test_complete_router_input_retains_constraints_and_expands_exact_native_capsule(dispatch, record_property, reply_mode):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_working_set import (
        restore_semantic_working_set,
    )
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import _session
    service, (state, calls), args = dispatch
    encoded = codec.encode_semantic_router_prompt(prompt=service.prompt, repository=service.repository)
    before = json.loads(encoded.provider_prompt)
    expanded = []
    def resolve(reference):
        cursor, chunks = None, []
        while True:
            response = exchange(service, {"client": dict(capsule_index=reference["capsule_index"],
                cursor=cursor, max_bytes=4096)})
            assert response["ok"]
            chunks.append(_bytes(response["page"]))
            cursor = response["page"]["next_cursor"]
            if cursor is None:
                break
        raw = b"".join(chunks)
        assert hashlib.sha256(raw).hexdigest() == reference["capsule_sha256"]
        expanded.append(len(raw))
        return raw
    def during_call():
        actual = calls[-1][0]
        start = actual.index('{"completion_authority"')
        wire, _ = json.JSONDecoder().raw_decode(actual[start:])
        assert wire["native_context"] == before["native_context"]
        assert wire["native_suffix"] == before["native_suffix"]
        for key in before["translated_semantic"]:
            if key != "capsules":
                assert wire["translated_semantic"][key] == before["translated_semantic"][key]
        # Exercise the same endpoint advertised in actual model input.
        resolve(wire["working_set"]["capsule_references"][0])
        if reply_mode != "legacy":
            from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
            raw = source.canonical_json(calls[-1][1]["codex_output_schema"])
            set_last_cli_observation("codex_cli", {**get_last_cli_observation("codex_cli"),
                "codex_output_schema_sha256": hashlib.sha256(raw).hexdigest(),
                "codex_output_schema_bytes": len(raw)})
    state["after_generate"] = during_call
    if reply_mode != "legacy":
        state["reply"] = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
    _, receipt = runner.run(**args, coding_reply_mode=reply_mode)
    selection = receipt["semantic_working_set"]
    actual = calls[-1][0]
    assert len(calls) == 1 and selection["selected_mode"] == "capsule-references@1"
    assert selection["selected_complete_bytes"] == receipt["model_prompt_bytes"] == len(actual.encode())
    assert selection["selected_complete_bytes"] < selection["native_complete_bytes"]
    assert receipt["semantic_working_set_preflight"]["requests"] == 1
    assert receipt["semantic_working_set_preflight"]["provider_input_bytes"] == 0
    assert receipt["router_calls"] == 1
    record_property("native_complete_bytes", selection["native_complete_bytes"])
    record_property("selected_complete_bytes", selection["selected_complete_bytes"])
    record_property("measurement", "authored fixture UTF-8 input bytes; provider tokens unknown")
    start = actual.index('{"completion_authority"')
    wire, _ = json.JSONDecoder().raw_decode(actual[start:])
    restored = restore_semantic_working_set(provider_prompt=source.canonical_json(wire).decode(),
        receipt=selection["view_receipt"], encoded=encoded, descriptor=service.descriptor, resolver=resolve)
    assert restored == encoded.provider_prompt
    receipt["phase"] = "coding"
    retained = audit._receipt(receipt, service.workspace.parent)
    rebuilt, _, checks = audit._model_projection(rendered=service.prompt, receipt=retained,
        repository=service.repository)
    assert rebuilt == actual and all(checks.values()), checks
    service.close()
    raw = service.audit_path.read_bytes()
    counts, _, _ = _session(raw, receipt, {"task_alias": service.descriptor["task_id"]},
        expected_semantic_working_set="capsule-references@1")
    assert counts["semantic_evidence_counts"]["delivered_pages"] >= len(expanded) + 1
    assert counts["semantic_evidence_counts"]["delivered_bytes"] > sum(expanded)
    assert counts["candidate_context_counts"]["delivered_bytes"] == 0
    assert counts["counts"]["delivered_source_bytes"] == 0
    with pytest.raises(ValueError):
        _session(raw, receipt, {"task_alias": service.descriptor["task_id"]})
    altered = deepcopy(receipt)
    altered["semantic_working_set_preflight"]["response_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _session(raw, altered, {"task_alias": service.descriptor["task_id"]},
            expected_semantic_working_set="capsule-references@1")
    for field in ("descriptor_sha256", "descriptor_session_id"):
        altered = deepcopy(receipt)
        binding = altered["semantic_working_set"]["view_receipt"]["binding"]
        binding[field] = "0" * len(binding[field])
        with pytest.raises(ValueError, match="source session"):
            audit._receipt(altered, service.workspace.parent)
    for field in ("requests", "capsule_index", "response_bytes", "provider_input_bytes"):
        altered = deepcopy(receipt)
        altered["semantic_working_set_preflight"][field] = True
        with pytest.raises(ValueError, match="preflight"):
            audit._receipt(altered, service.workspace.parent)


@pytest.mark.parametrize("failure", ["stale_workspace", "stale_repository", "closed_endpoint"])
def test_unavailable_expansion_after_provider_initialization_prevents_generation(dispatch, monkeypatch, capsys, failure):
    from ipfs_accelerate_py import llm_router
    service, (_, calls), args = dispatch
    def initialize(*_args, **_kwargs):
        if failure == "closed_endpoint":
            service.close()
        else:
            root = service.workspace if failure == "stale_workspace" else service.repository
            path = root / "mod.py"
            path.write_bytes(path.read_bytes() + b"\n# authored concurrent edit\n")
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises((ValueError, OSError, AssertionError)):
        runner.run(**args)
    assert calls == []
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    failed = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert failed["router_calls"] == 0 and failed["usage"] == {}


@pytest.mark.parametrize("change", [dict(semantic_working_set="unknown"),
    dict(semantic_metadata_view="common-bindings@1"), dict(purpose="planning"),
    dict(semantic_transport_schema="supervisor-semantic-router-input@2"),
    dict(source_view_descriptor=None), dict(source_view_profile="disabled")])
def test_invalid_profile_combinations_never_call_provider(dispatch, change):
    _, (_, calls), args = dispatch
    with pytest.raises(ValueError):
        runner.run(**(args | change))
    assert calls == []


def test_grok_profile_reaches_shared_router_with_unknown_token_counts(dispatch, monkeypatch):
    from types import SimpleNamespace
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    _, _, args = dispatch
    captured = []
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="grok_cli", model_name="grok-4.7", catalog_revision="authored"))
    def generate(prompt, **kwargs):
        captured.append((prompt, kwargs))
        set_last_cli_observation("grok_cli", {"exit_code": 0})
        return "Authored offline candidate; no live model."
    monkeypatch.setattr(llm_router, "generate_text", generate)
    _, receipt = runner.run(**(args | dict(provider="grok_cli", model="grok-4.7")))
    assert len(captured) == 1
    prompt, options = captured[0]
    assert options["provider"] == "grok_cli" and options["model_name"] == "grok-4.7"
    assert options["allow_local_fallback"] is options["allow_cross_provider_fallback"] is False
    assert options["grok_max_turns"] == 128
    assert receipt["model_prompt_bytes"] == len(prompt.encode())
    assert receipt["semantic_working_set"]["provider_token_usage"] is None
    assert receipt["semantic_working_set"]["complete_run_token_savings_claimed"] is False
    assert receipt["usage"] == {"exit_code": 0}
