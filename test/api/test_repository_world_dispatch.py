"""Current repository packets reach the real runner's authored provider boundary.

Native indexing and source checks run locally. Provider calls and token counts
are authored observations and do not establish live Terminal Bench savings.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler, EvidenceValuePolicy
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    prepare_repository_world_packet, repository_world_task_binding,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from test.api.test_repository_context_compiler import (
    CORE, compile_context, compiler, obligation, required,
)
from test.api.test_repository_source_index import scheduler


TASK = "Update calc.add to increment by two; retain subtract and validate the result."


@pytest.fixture
def world_session(tmp_path, scheduler):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "calc.py").write_text(
        "# Authored arithmetic fixture; never imported.\n"
        "def add(value):\n    return value + 1\n\n"
        "def subtract(value):\n    return value - 1\n")
    (source / "notes.txt").write_text("Retained non-code content.\n")
    with RepositorySourceSession(
        source, tmp_path / "session",
        index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler),
    ) as session:
        yield source, session
    assert scheduler.snapshot()["active_lease_count"] == 0


@pytest.fixture
def dispatch_case(world_session, required, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer

    # Exercise the real offline resolver's missing-runtime fallback without
    # depending on the qualification host's optional tokenizer/rank assets.
    def tokenizer_unavailable():
        raise ImportError("authored unavailable optional tokenizer runtime")
    monkeypatch.setattr(automatic_prompt_tokenizer, "_installed_tiktoken", tokenizer_unavailable)

    source, session = world_session
    adapter = session.context_adapter(compiler())
    effects = ("read_source", "write_source", "run_validation")
    paths = ("calc.py",)
    task_binding = repository_world_task_binding(
        task_id="task:arithmetic", task_text=TASK, allowed_effects=effects, write_paths=paths)
    compilation = compile_context(adapter, required,
        authority={**CORE["authority"], "world_task_packet": task_binding})
    packet = prepare_repository_world_packet(
        adapter=adapter, compilation=compilation, generation=1,
        task_id="task:arithmetic", task_text=TASK,
        unresolved_obligations=("Implement the requested bounded arithmetic change.",),
        validation_commands=("python -c 'from calc import add; assert add(1) == 3'",),
        allowed_effects=effects, write_paths=paths,
    )
    monkeypatch.chdir(source)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-native-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", reasoning_effort="high", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    state = {"seen": []}

    def generate(prompt, **kwargs):
        state["seen"].append((prompt, kwargs))
        observation = {"exit_code": 0, "prompt_tokens": 15, "completion_tokens": 4}
        binding = kwargs.get("codex_repository_source_binding")
        if binding is not None:
            repository_source_policy.validate_binding(binding, workspace=source, materialized=True)
            observation.update(codex_repository_source_configured=True,
                codex_repository_source_config_sha256=binding["config_sha256"])
        set_last_cli_observation("codex_cli", observation)
        if state.get("callback") is not None:
            state["callback"](prompt, kwargs)
        return "Authored provider returned; owner validation remains required."

    monkeypatch.setattr(llm_router, "generate_text", generate)
    return source, session, packet, state


def invoke(case, **changes):
    options = dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
                   max_output_tokens=128, world_task_packet=case[2])
    options.update(changes)
    return runner.run(**options)


def richer_world_case(case, *, reference_count=16, optional_items=None):
    """Issue a real packet with enough repeated metadata to exercise selection."""
    source, session, _packet, state = case
    # The authority contains the IDs and coverage of every required reference;
    # its recursive item budget must accommodate this larger authored packet.
    adapter = session.context_adapter(ContextCompiler(
        ContextBudget(max_input_tokens=20_000, reserved_output_tokens=0,
                      reserved_tool_tokens=0, max_items=128,
                      max_item_bytes=32_768, max_text_bytes=16_384),
        tokenizer=lambda value: max(1, len(value.encode("utf-8")) // 16),
        value_policy=EvidenceValuePolicy(max_optional_items=optional_items)))
    effects = ("read_source", "write_source", "run_validation")
    paths = ("calc.py",)
    task_binding = repository_world_task_binding(
        task_id="task:arithmetic", task_text=TASK, allowed_effects=effects, write_paths=paths)
    compilation = compile_context(adapter, tuple(obligation("retained-" + str(i)) for i in range(reference_count)),
        authority={**CORE["authority"], "world_task_packet": task_binding})
    packet = prepare_repository_world_packet(
        adapter=adapter, compilation=compilation, generation=1,
        task_id="task:arithmetic", task_text=TASK,
        unresolved_obligations=("Implement the requested bounded arithmetic change.",),
        validation_commands=("python -c 'from calc import add; assert add(1) == 3'",),
        allowed_effects=effects, write_paths=paths)
    return source, session, packet, state


def counter(*, count_tokens=None, **changes):
    options = dict(provider="codex_cli", model="pinned", tokenizer_id="authored-byte-counter",
                   tokenizer_revision="test-v1", special_token_policy="none",
                   count_tokens=count_tokens or (lambda value: len(value.encode())))
    options.update(changes)
    return runner.PromptTokenCounter(**options)


def invocation_receipt(capsys):
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")


@pytest.mark.parametrize("count_kind", ["tokenizer", "estimate", "automatic"])
def test_default_compaction_displaces_complete_input_losslessly(dispatch_case, count_kind, record_property):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
        project_repository_world_prompt, restore_repository_world_prompt,
    )

    case = richer_world_case(dispatch_case)
    source, _session, packet, state = case
    native, advisory = runner.render_model_prompt(
        prompt=packet.prompt, purpose="coding", workspace=source, world_packet=True)
    view = project_repository_world_prompt(packet.prompt)
    candidate, candidate_advisory = runner.render_model_prompt(
        prompt=view.provider_prompt, purpose="coding", workspace=source, world_packet=True)
    counted = []
    def count(text):
        counted.append(text)
        return len(text.encode())
    options = {} if count_kind == "automatic" else {
        "prompt_token_counter": counter(count_kind=count_kind, count_tokens=count)}
    _output, receipt = invoke(case, **options)
    assert state["seen"][0][0] == candidate
    assert advisory == candidate_advisory
    assert restore_repository_world_prompt(view.provider_prompt, view.receipt) == packet.prompt
    assert len(candidate.encode()) < len(native.encode())
    descriptor = receipt["world_task_packet"]["prompt_compaction"]
    assert descriptor["enabled"] is True and descriptor["selected_mode"] == "compact"
    assert descriptor["projection"] == view.receipt
    selection = descriptor["selection"]
    field = "input_tokens" if count_kind == "tokenizer" else "estimated_input_tokens"
    original = selection["candidates"]["native"]
    selected = selection["candidates"]["semantic_metadata"]
    assert selected[field] < original[field]
    assert selection["selected_representation"] == "semantic_metadata"
    assert receipt["world_task_packet"]["prompt_measurement"]["model_prompt_sha256"] == receipt["model_prompt_sha256"]
    assert receipt["router_prompt_sha256"] == hashlib.sha256(packet.prompt.encode()).hexdigest()
    assert selection["provider_framing_measured"] is False
    assert selection["provider_tokenizer_equivalence_verified"] is False
    assert selection["total_session_token_savings"] is None
    if count_kind != "automatic":
        assert counted == [native, candidate]
    record_property("world_prompt_compaction_observation", json.dumps({
        "schema": "authored-world-prompt-compaction-observation@1", "count_kind": count_kind,
        "native_bytes": len(native.encode()), "selected_bytes": len(candidate.encode()),
        "native_declared_count": original[field], "selected_declared_count": selected[field],
        "exact_native_restoration": True, "provider_execution": "authored_fixture",
        "provider_token_savings_measured": False, "whole_session_savings_established": False,
    }, sort_keys=True))


def test_optout_preserves_complete_native_input(dispatch_case):
    case = richer_world_case(dispatch_case)
    expected, _ = runner.render_model_prompt(
        prompt=case[2].prompt, purpose="coding", workspace=case[0], world_packet=True)
    _output, receipt = invoke(case, compact_world_prompt=False, prompt_token_counter=counter())
    assert case[3]["seen"][0][0] == expected
    assert receipt["world_task_packet"]["prompt_compaction"] == {
        "schema": "supervisor-world-prompt-compaction@1", "enabled": False,
        "projection": None, "selection": None, "selected_mode": "native"}


@pytest.mark.parametrize("mode", ["tie", "failed", "missing"])
def test_compact_candidate_requires_working_strictly_smaller_count(dispatch_case, mode):
    case = richer_world_case(dispatch_case)
    changes = {}
    if mode == "tie":
        changes["prompt_token_counter"] = counter(count_tokens=lambda _text: 100)
    elif mode == "failed":
        def fail(_text):
            raise RuntimeError("private local counter failure")
        changes["prompt_token_counter"] = counter(count_tokens=fail)
    else:
        # This existing explicit helper route has no automatic tokenizer.
        changes["tool_output_policy"] = "artifact-preview@2"
    _output, receipt = invoke(case, **changes)
    descriptor = receipt["world_task_packet"]["prompt_compaction"]
    assert descriptor["selected_mode"] == "native"
    assert descriptor["selection"]["selected_representation"] == "native"
    assert descriptor["selection"]["measurement_status"] == {
        "tie": "measured", "failed": "failed", "missing": "unavailable"}[mode]
    assert case[3]["seen"][0][0].endswith(case[2].prompt)
    assert "private local" not in json.dumps(receipt)


def test_byte_gate_avoids_counting_unhelpful_candidate(dispatch_case):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import project_repository_world_prompt
    # This separately admitted task has no selected reference bodies. No
    # requirement is removed from the original fixture's issued packet.
    case = richer_world_case(dispatch_case, reference_count=0, optional_items=0)
    packet = case[2]
    view = project_repository_world_prompt(packet.prompt)
    assert view.receipt["reference_count"] == 0
    assert len(view.provider_prompt.encode()) >= len(packet.prompt.encode())
    counted = []
    def count(text):
        counted.append(text)
        return len(text.encode())
    _output, receipt = invoke(case, prompt_token_counter=counter(count_tokens=count))
    descriptor = receipt["world_task_packet"]["prompt_compaction"]
    assert descriptor["enabled"] is True and descriptor["selected_mode"] == "native"
    assert descriptor["projection"] == view.receipt and descriptor["selection"] is None
    assert counted == [case[3]["seen"][0][0]]


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_compaction_flag_requires_explicit_boolean(dispatch_case, value):
    with pytest.raises(ValueError, match="compaction must be an explicit Boolean"):
        invoke(dispatch_case, compact_world_prompt=value)
    assert not dispatch_case[3]["seen"]


def test_current_packet_is_actual_model_input_with_content_free_receipt(dispatch_case):
    source, _session, packet, state = dispatch_case
    _output, receipt = invoke(dispatch_case, compact_world_prompt=False)
    assert len(state["seen"]) == 1
    delivered, options = state["seen"][0]
    expected, advisory = runner.render_model_prompt(
        prompt=packet.prompt, purpose="coding", workspace=source, world_packet=True)
    assert delivered == expected
    assert "self-contained decision packet" in advisory
    assert "Original native task prompt follows without modification" not in advisory
    assert options["side_effecting"] is True
    assert options["allow_cross_provider_fallback"] is options["allow_local_fallback"] is False
    assert options["timeout"] == 30 and options["max_tokens"] == 128
    assert receipt["native_prompt_sha256"] == hashlib.sha256(TASK.encode()).hexdigest()
    assert receipt["model_prompt_sha256"] == hashlib.sha256(delivered.encode()).hexdigest()
    assert receipt["router_prompt_sha256"] == hashlib.sha256(packet.prompt.encode()).hexdigest()
    row = receipt["world_task_packet"]
    assert row["dispatch_rechecked"] is row["provider_attempted"] is True
    assert row["provider_retained_parent"] is False
    measured = row["prompt_measurement"]
    assert measured["model_prompt_sha256"] == receipt["model_prompt_sha256"]
    assert measured["model_prompt_bytes"] == len(delivered.encode())
    assert measured["measurement_status"] == "estimated"
    assert measured["measurement_kind"] == "declared_local_estimate"
    assert measured["tokenizer_binding"]["provider"] == "codex_cli"
    assert measured["tokenizer_binding"]["model"] == "pinned"
    assert measured["selected_input_tokens"] is None
    assert measured["selected_estimated_input_tokens"] > 0
    assert measured["provider_framing_measured"] is False
    assert measured["total_session_token_savings"] is None
    assert receipt["prompt_tokenizer_source"]["offline"] is True
    assert receipt["prompt_tokenizer_source"]["backend"] == "utf8-bytes"
    assert receipt["prompt_tokenizer_source"]["provider_usage"] is False
    assert "prompt_selection" not in receipt
    assert TASK not in json.dumps(receipt)
    assert receipt["usage"]["prompt_tokens"] == 15
    assert receipt["completion_authority"] is False


def test_complete_packet_budget_counts_delivered_wrapper_without_native_fallback(dispatch_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    def unexpected_automatic(**kwargs):
        raise AssertionError("explicit counter must take precedence")
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", unexpected_automatic)
    source, _session, packet, state = dispatch_case
    expected, _ = runner.render_model_prompt(
        prompt=packet.prompt, purpose="coding", workspace=source, world_packet=True)
    counted = []
    def count(value):
        counted.append(value)
        return len(value.encode())
    _output, receipt = invoke(dispatch_case, prompt_token_counter=counter(count_tokens=count),
                              max_input_tokens=len(expected.encode()), compact_world_prompt=False)
    assert counted == [expected]
    assert state["seen"][0][0] == expected
    assert "prompt_selection" not in receipt
    row = receipt["world_task_packet"]["prompt_measurement"]
    assert row["measurement_status"] == "measured"
    assert row["selected_content_tokens"] == row["selected_input_tokens"] == len(expected.encode())
    assert row["measurement_kind"] == "declared_local_tokenizer"
    assert row["provider_framing_measured"] is False
    assert row["provider_tool_schemas_measured"] is False
    assert row["provider_internal_history_measured"] is False
    assert row["total_session_token_savings"] is None


@pytest.mark.parametrize("literal_response", [False, True])
def test_automatic_packet_budget_includes_complete_workspace_wrapper(dispatch_case, literal_response):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer

    source, _session, packet, state = dispatch_case
    local, _provenance = automatic_prompt_tokenizer.resolve_prompt_tokenizer(provider="codex_cli", model="pinned")
    complete, _ = runner.render_model_prompt(prompt=packet.prompt, purpose="coding", workspace=source, world_packet=True)
    # A cap that fits only the packet must never permit dispatch of the larger
    # complete model input, including its literal workspace instructions.
    partial_count = local.count(packet.prompt)
    complete_count = local.count(complete)
    assert partial_count < complete_count
    with pytest.raises(ValueError, match="exceeds its token budget"):
        invoke(dispatch_case, codex_cli_literal_response=literal_response, max_input_tokens=partial_count,
               compact_world_prompt=False)
    assert not state["seen"]


@pytest.mark.parametrize("bounded", [False, True])
def test_failed_automatic_counter_retains_existing_budget_refusal(dispatch_case, monkeypatch, bounded):
    from dataclasses import replace
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer

    local, provenance = automatic_prompt_tokenizer.resolve_prompt_tokenizer(provider="codex_cli", model="pinned")
    def unavailable(_value):
        raise RuntimeError("private automatic counter failure")
    failing = replace(local, count_tokens=unavailable)
    routes = []
    def resolve(**route):
        routes.append(route)
        return failing, provenance
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", resolve)
    if bounded:
        with pytest.raises(ValueError, match="requires a working pinned local counter"):
            invoke(dispatch_case, max_input_tokens=100_000)
        assert not dispatch_case[3]["seen"]
    else:
        _output, receipt = invoke(dispatch_case)
        measured = receipt["world_task_packet"]["prompt_measurement"]
        assert measured["measurement_status"] == "failed"
        assert measured["selected_input_tokens"] is None
        assert "private automatic" not in json.dumps(receipt)
        assert dispatch_case[3]["seen"][0][0].endswith(dispatch_case[2].prompt)
    assert routes == [{"provider": "codex_cli", "model": "pinned"}]


def test_packet_budget_includes_native_coding_reply_schema(dispatch_case, capsys):
    from ipfs_accelerate_py.agent_supervisor.runtime import coding_reply_contract
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    source, _session, packet, state = dispatch_case
    body, _ = runner.render_model_prompt(
        prompt=packet.prompt, purpose="coding", workspace=source, world_packet=True)
    expected, _reply = coding_reply_contract.apply_coding_reply_contract(
        model_prompt=body, mode="ordinary-completion@1", purpose="coding", provider="codex_cli")
    schema = json.dumps(coding_reply_contract.coding_completion_schema(), sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False)
    # Schema mismatch is checked only after dispatch; use that authored failure
    # to inspect the complete count while retaining the production validator.
    def mismatched_schema(_prompt, _options):
        set_last_cli_observation("codex_cli", {"exit_code": 0})
    state["callback"] = mismatched_schema
    with pytest.raises(RuntimeError, match="schema invocation metadata"):
        invoke(dispatch_case, coding_reply_mode="ordinary-completion@1",
               prompt_token_counter=counter(), max_input_tokens=len(expected.encode()) + len(schema.encode()),
               compact_world_prompt=False)
    assert state["seen"][0][0] == expected
    measured = invocation_receipt(capsys)["world_task_packet"]["prompt_measurement"]
    assert measured["native_output_schema_tokens"] == len(schema.encode())
    assert measured["selected_input_tokens"] == len(expected.encode()) + len(schema.encode())


def test_hash_bound_owner_tokenizer_measures_actual_packet(dispatch_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    from test.api.test_local_prompt_tokenizer import make_binding
    from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
    artifact, digest, _ = make_binding(tmp_path, provider="codex_cli", model="pinned")
    local = load_owner_prompt_tokenizer(artifact=artifact, expected_sha256=digest,
                                       provider="codex_cli", model="pinned")
    def unexpected_automatic(**kwargs):
        raise AssertionError("explicit tokenizer descriptor must take precedence")
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", unexpected_automatic)
    _output, receipt = invoke(dispatch_case, prompt_tokenizer_artifact=artifact,
                              prompt_tokenizer_sha256=digest, compact_world_prompt=False)
    delivered = dispatch_case[3]["seen"][0][0]
    measured = receipt["world_task_packet"]["prompt_measurement"]
    assert measured["selected_content_tokens"] == local.count(delivered)
    assert receipt["prompt_tokenizer_source"]["descriptor_sha256"] == digest
    assert delivered.endswith(dispatch_case[2].prompt)


def test_unbounded_failed_counter_is_reported_without_changing_packet(dispatch_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    def unexpected_automatic(**kwargs):
        raise AssertionError("failed explicit counter must not fall back automatically")
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", unexpected_automatic)
    def unavailable(_value):
        raise RuntimeError("private tokenizer error body")
    _output, receipt = invoke(dispatch_case, prompt_token_counter=counter(count_tokens=unavailable))
    measured = receipt["world_task_packet"]["prompt_measurement"]
    assert measured["measurement_status"] == "failed"
    assert measured["selected_input_tokens"] is None
    assert "private tokenizer" not in json.dumps(receipt)
    assert dispatch_case[3]["seen"][0][0].endswith(dispatch_case[2].prompt)


@pytest.mark.parametrize("changes", [
    {"max_input_tokens": 100},
    {"max_input_tokens": 1, "prompt_token_counter": counter()},
    {"max_input_tokens": True, "prompt_token_counter": counter()},
    {"max_input_tokens": 100, "prompt_token_counter": counter(model="foreign")},
])
def test_invalid_or_unsatisfied_packet_budget_never_dispatches(dispatch_case, changes):
    with pytest.raises(ValueError):
        invoke(dispatch_case, **changes)
    assert not dispatch_case[3]["seen"]


def test_packet_with_repository_tools_keeps_configuration_and_cleanup(dispatch_case):
    _output, receipt = invoke(dispatch_case, repository_source_policy="repository-source-compact@1")
    binding = receipt["repository_source_policy"]
    assert dispatch_case[3]["seen"][0][1]["codex_repository_source_binding"] is binding
    assert binding["usage"]["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()
    assert receipt["world_task_packet"]["dispatch_rechecked"] is True
    assert receipt["world_task_packet"]["prompt_measurement"] is None
    assert "prompt_tokenizer_source" not in receipt


def test_source_drift_during_provider_initialization_refuses_dispatch(dispatch_case, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    source, _session, _packet, state = dispatch_case
    def drift(*args, **kwargs):
        path = source / "calc.py"
        path.write_text(path.read_text().replace("return value + 1", "return value + 3"))
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", drift)
    with pytest.raises(ValueError):
        invoke(dispatch_case)
    assert not state["seen"]
    receipt = invocation_receipt(capsys)
    assert receipt["failure_phase"] == "world_task_packet_currentness"
    assert receipt["world_task_packet"]["dispatch_rechecked"] is False
    assert receipt["world_task_packet"]["provider_attempted"] is False


def test_exact_task_binding_is_checked_before_provider_dispatch(dispatch_case):
    with pytest.raises(ValueError):
        invoke(dispatch_case, prompt="A different instruction with the same repository.")
    assert not dispatch_case[3]["seen"]


@pytest.mark.parametrize("failure", [TimeoutError("authored timeout"), RuntimeError("429 rate limit exceeded")])
def test_provider_failure_retains_attempt_without_automatic_retry(dispatch_case, failure, capsys):
    def fail(_prompt, _options):
        raise failure
    dispatch_case[3]["callback"] = fail
    with pytest.raises(type(failure)):
        invoke(dispatch_case)
    assert len(dispatch_case[3]["seen"]) == 1
    receipt = invocation_receipt(capsys)
    assert receipt["status"] == "failed" and receipt["failure_phase"] == "provider_invocation"
    assert receipt["world_task_packet"]["provider_attempted"] is True
    assert receipt["world_task_packet"]["dispatch_rechecked"] is True
    assert receipt["router_calls"] == 1
    assert receipt["provider_failure"]["reason_code"] == (
        "timeout" if isinstance(failure, TimeoutError) else "rate_limit")


@pytest.mark.parametrize("changes", [
    {"purpose": "planning"},
    {"provider": "grok_cli", "model": "grok-4.7"},
    {"semantic_repository": Path(".")},
    {"semantic_metadata_view": "common-bindings@1"},
    {"source_input_policy": "source-compact@1"},
    {"query_reference_policy": "query-reference@1"},
    {"source_view_advisory_policy": "source-view-raw@1"},
    {"source_view_profile": "identity@1"},
    {"source_formalization_handoff": Path("handoff.json")},
    {"public_instruction_artifact": Path("instruction.json")},
    {"doctor_residual_artifact": Path("residual.json")},
    {"census_only": True},
])
def test_unqualified_combinations_fail_before_dispatch(dispatch_case, changes):
    with pytest.raises(ValueError, match="independently selected Codex coding route"):
        invoke(dispatch_case, **changes)
    assert not dispatch_case[3]["seen"]


def test_arbitrary_packet_object_cannot_supply_provider_text(dispatch_case):
    with pytest.raises(ValueError, match="exact prepared repository packet"):
        invoke(dispatch_case, world_task_packet=SimpleNamespace(prompt="unvalidated replacement"))
    assert not dispatch_case[3]["seen"]


def test_legacy_route_retains_literal_prompt_and_has_no_packet_receipt(dispatch_case):
    source, _session, _packet, state = dispatch_case
    _output, receipt = invoke(dispatch_case, world_task_packet=None)
    expected, _ = runner.render_model_prompt(prompt=TASK, purpose="coding", workspace=source)
    assert state["seen"][0][0] == expected
    assert "Original native task prompt follows without modification" in expected
    assert "world_task_packet" not in receipt


@pytest.mark.parametrize("changes", [{"purpose": "planning"}, {"semantic_transport": True}])
def test_world_packet_advisory_refuses_incompatible_rendering(tmp_path, changes):
    options = dict(prompt="task", purpose="coding", workspace=tmp_path, world_packet=True)
    options.update(changes)
    with pytest.raises(ValueError):
        runner.render_model_prompt(**options)
