"""Explicit native-wire control retains producer admission and full budgeting."""

import hashlib
import io
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec

pytest_plugins = ("test.api.test_semantic_router_integration",)


def _counter(count=None):
    return runner.PromptTokenCounter(
        provider="codex_cli", model="pinned", tokenizer_id="authored-fixture:utf8-bytes",
        tokenizer_revision="1", special_token_policy="authored fixture only",
        count_kind="estimate", compatibility_basis="authored fixture only",
        count_tokens=count or (lambda text: len(text.encode())),
    )


def _run(repository, prompt, **kwargs):
    options = dict(
        prompt=prompt, provider="codex_cli", model="pinned", timeout=30, max_output_tokens=128,
        semantic_repository=repository, semantic_native_input=True, prompt_token_counter=_counter(),
    )
    return runner.run(**(options | kwargs))


def test_native_control_uses_full_literal_wire_without_constructing_encoding(allocated, provider, monkeypatch):
    repository, workspace, prompt, source = allocated
    state, observed = provider
    state["reply"] = '{"schema":"supervisor-semantic-router-response@1","literal":"s0"}'
    state["after_generate"] = lambda: (workspace / "mod.py").write_text(source + "\n# candidate edit\n")
    counts = []

    def count(text):
        counts.append(text)
        return len(text.encode())

    def forbidden(**kwargs):
        raise AssertionError("native control must not construct any encoded candidate")

    monkeypatch.setattr(codec, "encode_semantic_router_prompt", forbidden)
    monkeypatch.setattr(codec, "pack_semantic_router_prompt", forbidden)
    output, receipt = _run(repository, prompt, prompt_token_counter=_counter(count))
    expected, advisory = runner.render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
    assert observed[0][0] == expected == advisory + prompt
    assert counts == [expected]
    assert output == state["reply"] and (repository / "mod.py").read_text() == source
    assert (workspace / "mod.py").read_text() != source
    assert receipt["semantic_native_input"] is True
    assert receipt["semantic_translation"] is None and receipt["semantic_encoding_candidate"] is None
    assert "semantic_response_translation" not in receipt
    assert receipt["prompt_selection"]["selected_representation"] == "native"
    assert set(receipt["prompt_selection"]["candidates"]) == {"native"}
    assert receipt["router_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert receipt["model_prompt_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert receipt["usage"]["prompt_tokens"] == 123
    assert receipt["completion_authority"] is False


@pytest.mark.parametrize("fits", [True, False], ids=["complete-budget", "one-token-over-budget"])
def test_native_control_preserves_required_suffix_order_and_full_count(allocated, provider, monkeypatch, fits):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_context, router_public_instruction

    repository, workspace, prompt, _ = allocated
    _, observed = provider
    loaded = []

    def residual(**kwargs):
        loaded.append(("residual", kwargs))
        return "\nLITERAL Doctor suffix s0", {"schema": "authored-residual"}

    def instruction(**kwargs):
        loaded.append(("instruction", kwargs))
        return "\nLITERAL public suffix \u03bb", {"schema": "authored-instruction"}

    monkeypatch.setattr(doctor_residual_context, "load_doctor_residual_advisory", residual)
    monkeypatch.setattr(router_public_instruction, "load_public_instruction", instruction)
    prepared = prompt + "\nLITERAL Doctor suffix s0\nLITERAL public suffix \u03bb"
    expected, _ = runner.render_model_prompt(prompt=prepared, purpose="coding", workspace=workspace)
    options = dict(
        doctor_residual_artifact=Path("/authored/residual.json"), doctor_residual_sha256="a" * 64,
        doctor_residual_task_cid="task", public_instruction_artifact=Path("/authored/public.json"),
        public_instruction_sha256="b" * 64, public_instruction_task_cid="task",
        max_input_tokens=len(expected.encode()) - (0 if fits else 1),
    )
    if fits:
        _, receipt = _run(repository, prompt, **options)
        assert observed[0][0] == expected
        assert receipt["prompt_selection"]["selected_estimated_content_tokens"] == len(expected.encode())
        assert receipt["doctor_residual_context"] == {"schema": "authored-residual"}
        assert receipt["public_instruction"] == {"schema": "authored-instruction"}
    else:
        with pytest.raises(ValueError, match="token budget"):
            _run(repository, prompt, **options)
        assert observed == []
    assert [name for name, _ in loaded] == ["residual", "instruction"]
    assert all(arguments["prompt"] == prompt and arguments["repository"] == repository
               and arguments["workspace"] == workspace for _, arguments in loaded)


@pytest.mark.parametrize("refusal", ["stale", "foreign", "canonical-workspace"])
def test_native_control_keeps_source_and_allocated_repository_checks(allocated, provider, monkeypatch, refusal):
    repository, _, prompt, source = allocated
    _, observed = provider
    if refusal == "stale":
        (repository / "mod.py").write_text(source + "\n# stale canonical source\n")
        match = "stale"
    elif refusal == "foreign":
        repository = repository.parent / "foreign"
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        match = "differs from allocated worktree"
    else:
        monkeypatch.chdir(repository)
        match = "separate allocated worktree"
    with pytest.raises(ValueError, match=match):
        _run(repository, prompt)
    assert observed == []


def test_native_control_rechecks_current_source_after_count_before_provider(allocated, provider):
    repository, _, prompt, source = allocated
    _, observed = provider

    def count(text):
        (repository / "mod.py").write_text(source + "\n# changed during measurement\n")
        return len(text.encode())

    with pytest.raises(ValueError, match="stale"):
        _run(repository, prompt, prompt_token_counter=_counter(count))
    assert observed == []


def test_native_control_rejects_source_change_during_provider_and_keeps_usage(allocated, provider, capsys):
    repository, _, prompt, source = allocated
    state, observed = provider
    state["after_generate"] = lambda: (repository / "mod.py").write_text(source + "\n# changed during provider\n")
    with pytest.raises(ValueError, match="stale"):
        _run(repository, prompt)
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert len(observed) == 1 and receipt["status"] == "failed"
    assert receipt["semantic_native_input"] is True
    assert receipt["usage"]["prompt_tokens"] == 123 and receipt["usage"]["completion_tokens"] == 7


def test_native_control_rejects_projected_parent_instead_of_calling_it_full_native(allocated, provider):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import project_semantic_router_prompt

    repository, _, prompt, _ = allocated
    _, observed = provider
    projection = project_semantic_router_prompt(prompt=prompt, repository=repository)
    with pytest.raises(ValueError, match="complete native worker artifact"):
        _run(repository, projection.projected_prompt)
    assert observed == []


@pytest.mark.parametrize("override", [
    {"semantic_repository": None}, {"purpose": "planning"},
    {"semantic_context_projection": True}, {"semantic_slot_policy_revision": "closed-typed-identifiers@2"},
    {"semantic_native_input": 1}, {"semantic_native_input": "true"},
], ids=["missing-repository", "planning", "projection", "codec-v2", "integer", "string"])
def test_native_control_requires_explicit_exclusive_bound_coding(override):
    options = dict(
        prompt="authored unused input", provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=128, semantic_repository=Path("/unused-repository"), semantic_native_input=True,
        prompt_token_counter=_counter(),
    )
    with pytest.raises(ValueError, match="semantic native input"):
        runner.run(**(options | override))


def test_native_control_default_retains_existing_codec_and_receipt_shape(allocated, provider):
    repository, _, prompt, _ = allocated
    _, receipt = _run(repository, prompt, semantic_native_input=False)
    assert "semantic_native_input" not in receipt
    assert receipt["semantic_encoding_candidate"] is not None


@pytest.mark.parametrize("selected", [False, True], ids=["default", "native-input"])
def test_native_control_cli_forwards_only_explicit_selection(monkeypatch, capsys, selected):
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return "literal result", {}

    monkeypatch.setattr(runner, "run", execute)
    monkeypatch.setattr(runner.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"authored original prompt\n")))
    monkeypatch.setattr(runner.sys, "argv", ["router", "--model", "pinned", "--semantic-repository", "/source",
                                           *(["--semantic-native-input"] if selected else [])])
    assert runner.main() == 0
    assert calls[0]["semantic_native_input"] is selected
    assert calls[0]["semantic_repository"] == Path("/source")
    assert calls[0]["prompt"] == "authored original prompt\n"
    assert capsys.readouterr().out == "literal result\n"
