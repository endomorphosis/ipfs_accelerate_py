"""Working-set census shares real preparation and authenticated capsule preflight.

The native encoder, Git relationship, source broker and registered child/socket
are exercised. Container UID admission and the earlier comparison provider are
authored; census itself must never discover, allocate or call a provider.
"""
import hashlib
import io
import json
import subprocess
from copy import deepcopy
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import worker_semantic_evidence_client as evidence
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from test.api.test_semantic_metadata_census_dispatch import forbid_provider
from test.api.test_semantic_working_set_dispatch import dispatch  # noqa: F401
from test.api.test_semantic_router_integration import allocated, provider  # noqa: F401
from test.api.test_semantic_router_translation import native as _standard_native
from test.api.test_worker_semantic_evidence import evidence_service  # noqa: F401


@pytest.fixture
def native(request, tmp_path):
    if getattr(request, "param", None) != "minimal":
        return _standard_native.__wrapped__(tmp_path)
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context

    repository = tmp_path / "repository"
    repository.mkdir()
    text = "# Authored module without functions; its module capsule is still indexed.\n"
    (repository / "mod.py").write_text(text)
    output = repository / ".runtime/semantic"
    prepare_semantic_context(repository=repository, paths=["mod.py"],
        required_raw_paths=["mod.py"], objective="Inspect the module.",
        task_id="TASK-1", output=output)
    artifact = output / "worker-context.json"
    references = build_text_context_references(artifact.read_text(),
        reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(), repository_id="repo:test",
        tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="TASK-1",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "TASK-1"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["mod.py"]},
        acceptance={"criteria": ["pending native validation"]}, evidence=references)
    prompt = render_context_capsule(compiled.capsule) + "\nAuthorized guidance remains literal.\n"
    return repository, artifact, prompt, text


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _census(arguments, **options):
    return runner.run(**(arguments | {"working_set_census_only": True} | options))


def _cli(service, *, extra=()):
    return ["router", "--provider", "codex_cli", "--model", "pinned",
        "--semantic-repository", str(service.repository),
        "--container-boundary", str(service.boundary),
        "--container-boundary-sha256", "b" * 64,
        "--source-view-profile", "identity@1",
        "--source-view-descriptor", str(service.descriptor_path),
        "--source-view-descriptor-sha256", service.descriptor_sha256,
        "--semantic-working-set", "capsule-references@1",
        "--semantic-working-set-census-only", *extra]


def _assert_closed(output, receipt, service):
    assert json.loads(output) == receipt
    assert receipt["schema"] == "router-semantic-working-set-census@1"
    assert receipt["observation_only"] is True
    assert receipt["provider_calls"] == receipt["router_calls"] == 0
    assert receipt["provider_dispatch_observed"] is False
    assert receipt["planning_performed_by_census"] is False
    assert receipt["provider_token_usage"] is None
    assert receipt["total_token_savings_measured"] is False
    for field in ("proof_authority", "execution_authority", "completion_authority",
            "settlement_authority", "publication_authority", "source_freshness_authority",
            "required_fact_omission_authority"):
        assert receipt[field] is False
    for private in (str(service.repository), str(service.workspace), str(service.boundary),
            str(service.descriptor_path), str(service.socket_path),
            service.descriptor["session_id"], service.descriptor["task_id"],
            service.prompt, "lengthy_first_function", "Authorized guidance remains literal",
            "pinned"):
        assert private not in output
    assert "model_prompt" not in receipt
    assert "source_view_retrieval" not in receipt
    assert "semantic_working_set" not in receipt  # Full selection includes session identity.
    assert receipt["semantic_prompt_components"]["schema"] == "supervisor-semantic-prompt-component-census@2"


@pytest.mark.parametrize("reply_mode", ["legacy", "ordinary-completion@1"])
def test_census_matches_actual_complete_input_and_authenticates_without_provider(
        dispatch, monkeypatch, capsys, record_property, reply_mode):
    service, (state, calls), args = dispatch
    page_calls = []
    original_page = evidence.request_page
    def page(*page_args, **kwargs):
        page_calls.append(kwargs)
        return original_page(*page_args, **kwargs)
    monkeypatch.setattr(evidence, "request_page", page)
    if reply_mode != "legacy":
        state["reply"] = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
        def schema_observation():
            from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
            raw = source.canonical_json(calls[-1][1]["codex_output_schema"])
            set_last_cli_observation("codex_cli", {**get_last_cli_observation("codex_cli"),
                "codex_output_schema_sha256": hashlib.sha256(raw).hexdigest(),
                "codex_output_schema_bytes": len(raw)})
        state["after_generate"] = schema_observation
    _, actual_receipt = runner.run(**args, coding_reply_mode=reply_mode)
    actual_prompt = calls[-1][0]
    assert len(page_calls) == 1
    capsys.readouterr()
    page_calls.clear()
    before = subprocess.check_output(["git", "status", "--porcelain"], cwd=service.workspace)
    forbid_provider(monkeypatch)
    output, receipt = _census(args, coding_reply_mode=reply_mode)
    assert capsys.readouterr().out == ""
    _assert_closed(output, receipt, service)
    report = receipt["census"]
    selection = actual_receipt["semantic_working_set"]
    assert report["selected_mode"] == selection["selected_mode"] == "capsule-references@1"
    for prefix in ("native", "candidate", "selected"):
        for suffix in ("bytes", "sha256"):
            key = prefix + "_complete_" + suffix
            assert report[key] == selection[key]
    assert report["selected_complete_sha256"] == actual_receipt["model_prompt_sha256"] == _sha(actual_prompt)
    assert report["selected_complete_bytes"] == actual_receipt["model_prompt_bytes"] == len(actual_prompt.encode())
    assert len(page_calls) == 1
    preflight = receipt["semantic_working_set_preflight"]
    assert preflight["schema"] == "supervisor-semantic-working-set-census-preflight@1"
    assert preflight["requests"] == preflight["capsules_probed"] == 1
    assert preflight["provider_input_bytes"] == 0
    assert preflight["session_id_sha256"] == _sha(service.descriptor["session_id"])
    assert preflight["source_freshness_checked"] is True
    assert preflight["complete_capsule_expansion_verified"] is False
    assert preflight["proof_authority"] is False
    assert len(calls) == 1  # Only the earlier authored comparison provider.
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=service.workspace) == before
    record_property("native_complete_bytes", report["native_complete_bytes"])
    record_property("selected_complete_bytes", report["selected_complete_bytes"])
    record_property("measurement", "authored fixture UTF-8 bytes; provider tokens unknown")


@pytest.mark.parametrize("native", ["minimal"], indirect=True)
def test_census_keeps_smaller_native_input_without_contacting_capsule_endpoint(
        dispatch, monkeypatch, capsys, record_property):
    service, (_, calls), args = dispatch
    monkeypatch.setattr(evidence, "request_page", lambda *args, **kwargs:
        pytest.fail("legacy fallback contacted capsule endpoint"))
    _, actual = runner.run(**args)
    actual_prompt = calls[-1][0]
    assert actual["semantic_working_set"]["selected_mode"] == "legacy"
    capsys.readouterr()
    forbid_provider(monkeypatch)
    output, receipt = _census(args)
    assert capsys.readouterr().out == ""
    _assert_closed(output, receipt, service)
    report = receipt["census"]
    assert report["selected_mode"] == "legacy"
    assert report["fallback_reason"] == "complete_input_not_strictly_smaller"
    assert report["candidate_complete_bytes"] > report["native_complete_bytes"]
    assert report["selected_complete_bytes"] == report["native_complete_bytes"] == len(actual_prompt.encode())
    assert report["selected_complete_sha256"] == report["native_complete_sha256"] == _sha(actual_prompt)
    assert receipt["semantic_working_set_preflight"] is None
    assert len(calls) == 1
    record_property("native_complete_bytes", report["native_complete_bytes"])
    record_property("candidate_complete_bytes", report["candidate_complete_bytes"])
    record_property("selected_mode", "legacy")


def test_census_accepts_actual_grok_eight_turn_configuration_without_discovery(dispatch, monkeypatch):
    service, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    output, receipt = _census(args, provider="grok_cli", model="grok-4.7", grok_coding_max_turns=8)
    _assert_closed(output, receipt, service)
    assert receipt["provider"] == "grok_cli"
    assert receipt["model_sha256"] == _sha("grok-4.7")
    assert "grok-4.7" not in output
    assert calls == []


def test_closed_exporter_rejects_untrusted_fields_and_historical_selection(dispatch, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set_census as exporter

    service, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    original = exporter.working_set_census_receipt
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(exporter, "working_set_census_receipt", capture)
    output, receipt = _census(args, coding_reply_mode="ordinary-completion@1")
    _assert_closed(output, receipt, service)
    assert original(**captured) == receipt

    private = "authored-private-export-must-be-refused"
    mutations = [
        ("provider", None, private),
        ("boundary_sha256", None, private),
        ("preflight", "response_sha256", private),
        ("preflight", "capsule_sha256", private),
        ("preflight", "response_bytes", True),
        ("preflight", "response_bytes", 0),
        ("preflight", "response_bytes", 4097),
        ("preflight", "requests", True),
        ("preflight", "private_extra", private),
        ("reply", "native_output_schema_sha256", private),
        ("reply", "native_output_schema_bytes", True),
        ("reply", "native_output_schema_bytes", 0),
        ("components", "private_extra", private),
        ("components", "token_estimate", 0),
    ]
    for field, nested, value in mutations:
        changed = dict(captured)
        if nested is None:
            changed[field] = value
        else:
            changed[field] = deepcopy(captured[field])
            changed[field][nested] = value
        with pytest.raises(ValueError, match="."):
            original(**changed)
    historical = replace(captured["selection"], dispatch_usable=False, historical_replay=True)
    with pytest.raises(ValueError, match="current working-set selection"):
        original(**(captured | {"selection": historical}))
    historical_encoding = replace(captured["encoded"], freshness_checked=False)
    with pytest.raises(ValueError, match="current semantic encoding"):
        original(**(captured | {"encoded": historical_encoding}))
    assert calls == []


@pytest.mark.parametrize("change", [
    {"working_set_census_only": 1}, {"working_set_census_only": "true"},
    {"working_set_census_only": None},
    {"census_only": True},
    {"semantic_working_set": "legacy"},
    {"semantic_working_set": "unknown"},
    {"semantic_metadata_view": "common-bindings@1"},
    {"purpose": "planning"},
    {"semantic_repository": None},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"source_view_descriptor": None},
    {"source_view_descriptor_sha256": None},
    {"source_view_profile": "disabled"},
    {"container_boundary": None},
    {"source_edit_mode": "mapped-span@1"},
    {"source_edit_mode": "mapped-span@2"},
    {"candidate_dispatch_handoff": "authored-unread-handoff"},
])
def test_census_rejects_incompatible_arguments_before_provider(dispatch, monkeypatch, change):
    _, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    with pytest.raises(ValueError):
        _census(args, **change)
    assert calls == []


@pytest.mark.parametrize("failure", ["stale_repository", "stale_workspace", "closed_endpoint"])
def test_census_refuses_stale_or_unreachable_capsules_before_provider(dispatch, monkeypatch, failure):
    service, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    if failure == "closed_endpoint":
        service.close()
    else:
        root = service.repository if failure == "stale_repository" else service.workspace
        path = root / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# authored concurrent edit\n")
    with pytest.raises((ValueError, OSError, AssertionError)):
        _census(args)
    assert calls == []


def test_census_refuses_inherited_database_authority_before_preflight(dispatch, monkeypatch):
    _, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN", "authored-private-owner")
    monkeypatch.setattr(evidence, "request_page", lambda *args, **kwargs:
        pytest.fail("inherited owner authority reached preflight"))
    with pytest.raises(ValueError, match="inherited database authority"):
        _census(args)
    assert calls == []


def test_cli_census_emits_one_closed_observation(dispatch, monkeypatch, capsys):
    service, (_, calls), _ = dispatch
    forbid_provider(monkeypatch)
    monkeypatch.setattr("sys.argv", _cli(service, extra=("--coding-reply-mode", "ordinary-completion@1")))
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(service.prompt.encode())))
    assert runner.main() == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1 and captured.err == ""
    receipt = json.loads(lines[0])
    _assert_closed(lines[0], receipt, service)
    assert receipt["current_semantic_encoder_used"] is True
    assert receipt["credential_isolation_checked"] is True
    assert calls == []


def test_cli_stale_source_failure_exports_only_closed_diagnostic(dispatch, monkeypatch, capsys):
    service, (_, calls), _ = dispatch
    forbid_provider(monkeypatch)
    path = service.repository / "mod.py"
    path.write_bytes(path.read_bytes() + b"\n# authored-private-source-secret\n")
    monkeypatch.setattr("sys.argv", _cli(service))
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(service.prompt.encode())))
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "authored-private-source-secret" not in captured.err
    assert str(service.repository) not in captured.err
    diagnostic = json.loads(captured.err)
    assert diagnostic["schema"] == "router-implementation-error@2"
    assert diagnostic["diagnostic"]["provider_dispatch_observed"] is None
    assert calls == []


@pytest.mark.parametrize("reply_mode", ["legacy", "ordinary-completion@1"])
@pytest.mark.parametrize("census_schema", ["router-semantic-working-set-census@2",
    "router-semantic-working-set-census@3"])
def test_extended_census_measures_independent_inline_input_with_exact_original_wrappers(
        dispatch, monkeypatch, reply_mode, census_schema):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set_census as exporter
    from ipfs_accelerate_py.agent_supervisor.runtime import inline_evidence_projection as inline
    service, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    _, old = _census(args, coding_reply_mode=reply_mode)
    original = exporter.working_set_census_receipt
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(exporter, "working_set_census_receipt", capture)
    output, result = _census(args, coding_reply_mode=reply_mode,
        working_set_census_schema=census_schema)
    assert json.loads(output) == result and result["schema"] == census_schema
    added = {"inline_evidence_census", "semantic_pack_census"}
    if census_schema == exporter.PACK_SCHEMA:
        added.add("semantic_pack_projection_census")
    assert set(result) - set(old) == added
    assert result["census"] == old["census"]
    assert result["semantic_prompt_components"] == old["semantic_prompt_components"]
    observation = result["inline_evidence_census"]
    assert observation["exact_transport_inverse_verified"] is True
    assert observation["dispatch_enabled"] is observation["composed_with_capsule_references"] is False
    selected = inline.validate_selection_receipt(observation["selection"])
    for suffix in ("_bytes", "_sha256"):
        assert selected["native_complete" + suffix] == old["census"]["native_complete" + suffix]
    # This fixture has no supported required artifact groups. The extra view
    # overhead must produce a fallback, not a fabricated compression result.
    assert selected["view_receipt"]["artifact_count"] == 0
    assert selected["selected_mode"] == "legacy"
    view = inline.project_inline_evidence(captured["encoded"])
    assert inline.restore_inline_evidence(view.provider_prompt, view.receipt) == captured["encoded"].provider_prompt
    assert result["semantic_pack_census"]["pack_object"]["object_bytes"] == (
        result["semantic_prompt_components"]["translated_semantic"]["members"]["pack"]["value_bytes"])
    if census_schema == exporter.PACK_SCHEMA:
        from ipfs_accelerate_py.agent_supervisor.runtime import semantic_pack_projection as pack
        measured = result["semantic_pack_projection_census"]
        assert measured["exact_transport_inverse_verified"] is True
        assert measured["dispatch_enabled"] is measured["composed_with_inline_evidence"] is False
        assert measured["composed_with_capsule_references"] is False
        pack_selection = pack.validate_selection_receipt(measured["selection"])
        for suffix in ("_bytes", "_sha256"):
            assert pack_selection["native_complete" + suffix] == old["census"]["native_complete" + suffix]
        pack_view = pack.project_semantic_pack(captured["encoded"])
        assert pack.restore_semantic_pack(pack_view.provider_prompt, pack_view.receipt) == captured["encoded"].provider_prompt
        prefix, suffix = captured["native_model_prompt"].split(captured["encoded"].provider_prompt)
        candidate = prefix + pack_view.provider_prompt + suffix
        assert pack_selection["candidate_complete_sha256"] == _sha(candidate)
        assert pack_selection["candidate_complete_bytes"] == len(candidate.encode())
        expected_mode = pack.MODE if len(candidate.encode()) < len(captured["native_model_prompt"].encode()) else "legacy"
        assert pack_selection["selected_mode"] == expected_mode
        # Additive schema: the entire previous observation remains unchanged.
        previous = original(**(captured | {"census_schema": exporter.EXTENDED_SCHEMA}))
        assert result == (previous | {"schema": exporter.PACK_SCHEMA,
            "semantic_pack_projection_census": measured})
    assert original(**captured) == result
    for value in (None, captured["native_model_prompt"] + " changed"):
        with pytest.raises(ValueError):
            original(**(captured | {"native_model_prompt": value}))
    for private in (str(service.repository), str(service.workspace), service.prompt):
        assert private not in output
    assert calls == [] and result["provider_calls"] == result["router_calls"] == 0


@pytest.mark.parametrize("schema", [None, True, "unknown", "router-semantic-working-set-census@2",
    "router-semantic-working-set-census@3"])
def test_census_schema_cannot_enable_a_generation_route(dispatch, monkeypatch, schema):
    _, (_, calls), args = dispatch
    forbid_provider(monkeypatch)
    with pytest.raises(ValueError):
        runner.run(**args, working_set_census_schema=schema)
    assert calls == []


@pytest.mark.parametrize("schema", ["router-semantic-working-set-census@2", "router-semantic-working-set-census@3"])
def test_cli_extended_census_forwards_explicit_schema(dispatch, monkeypatch, capsys, schema):
    service, (_, calls), _ = dispatch
    forbid_provider(monkeypatch)
    monkeypatch.setattr("sys.argv", _cli(service, extra=("--semantic-working-set-census-schema",
        schema)))
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(service.prompt.encode())))
    assert runner.main() == 0
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert captured.err == "" and receipt["schema"] == schema
    assert receipt["inline_evidence_census"]["dispatch_enabled"] is False
    if schema == "router-semantic-working-set-census@3":
        assert receipt["semantic_pack_projection_census"]["dispatch_enabled"] is False
    assert calls == []
