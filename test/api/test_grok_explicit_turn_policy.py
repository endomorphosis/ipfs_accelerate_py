"""Explicit turn authority at the real Grok argv boundary, without models.

These authored CLI fixtures prove selected arguments, refusals and cleanup.
They do not qualify Grok's own enforcement, billing, or a native token cap.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import (
    get_last_cli_observation, set_last_cli_observation,
)
from ipfs_accelerate_py.cli_runtime.grok_turn_policy import (
    bind_explicit_grok_max_turns, validate_grok_max_turns,
)

MAX_TURNS = (1 << 32) - 1
PRIVATE = "private-explicit-turn-policy-marker"


@pytest.mark.parametrize("value", [1, 2, 128, 100_000, MAX_TURNS])
def test_explicit_turn_validator_preserves_supported_integer_value(value):
    assert validate_grok_max_turns(value) == value
    assert type(validate_grok_max_turns(value)) is int


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.0, "1", MAX_TURNS + 1,
    pytest.param(1 << 20000, id="very_large_integer")])
def test_explicit_turn_validator_rejects_implicit_coercion_and_invalid_bounds(value):
    with pytest.raises(ValueError):
        validate_grok_max_turns(value)
    with pytest.raises(ValueError):
        bind_explicit_grok_max_turns(["grok"], value)


@pytest.mark.parametrize("sequence", [list, tuple])
@pytest.mark.parametrize("turns", [1, 128, MAX_TURNS])
def test_absent_turn_option_is_appended_to_an_independent_argv_copy(sequence, turns):
    original = sequence(["grok", "--model", "grok-4.7", "--tools", "read_file"])
    held = deepcopy(original)
    result = bind_explicit_grok_max_turns(original, turns)
    assert type(result) is list
    assert result == list(original) + ["--max-turns", str(turns)]
    assert original == held
    result.append("--authored-unrelated-option")
    assert original == held


@pytest.mark.parametrize("existing", [["--max-turns", "2"], ["--max-turns=2"]])
def test_one_canonical_matching_option_preserves_its_original_spelling(existing):
    command = ["grok", "--model", "grok-4.7", *existing, "--no-memory"]
    result = bind_explicit_grok_max_turns(command, 2)
    assert result == command and result is not command


@pytest.mark.parametrize("value", ["", "0", "01", "+1", "-1", "1.0", "1e0",
    " 1", "1 ", "1\n", "١", "１", "4294967296", "--no-plan"])
@pytest.mark.parametrize("equals", [False, True])
def test_existing_turn_spelling_must_be_canonical_ascii_positive_decimal(value, equals):
    option = ["--max-turns=" + value] if equals else ["--max-turns", value]
    original = ["grok", *option]
    held = list(original)
    with pytest.raises(ValueError):
        bind_explicit_grok_max_turns(original, 1)
    assert original == held


@pytest.mark.parametrize("suffix", [
    ["--max-turns"], ["--max-turns", "128"], ["--max-turns=128"],
    ["--max-turns", "2", "--max-turns", "2"],
    ["--max-turns=2", "--max-turns=2"],
    ["--max-turns", "2", "--max-turns=2"],
    ["--max-turns", "2", "--max-turns=3"],
    ["--"], ["--", "--max-turns", "2"],
    ["--max-turns", "2", "--"],
])
def test_conflicting_duplicate_missing_or_terminated_options_cannot_override_explicit_bound(suffix):
    command = ["grok", *suffix]
    held = list(command)
    with pytest.raises(ValueError):
        bind_explicit_grok_max_turns(command, 2)
    assert command == held


@pytest.mark.parametrize("command", [None, "grok", [], (), [""], ["grok", None], ["grok", 1]])
def test_turn_binding_requires_a_bounded_nonempty_string_argv(command):
    with pytest.raises(ValueError):
        bind_explicit_grok_max_turns(command, 2)


@pytest.mark.parametrize("damage", ["nul", "surrogate", "large_part", "many_parts", "large_total"])
def test_unrepresentable_or_oversized_custom_argv_is_refused_with_closed_error(damage):
    if damage == "nul":
        argv = ["grok", PRIVATE + "\x00"]
    elif damage == "surrogate":
        argv = ["grok", PRIVATE + "\ud800"]
    elif damage == "large_part":
        argv = ["grok", "x" * 65_537]
    elif damage == "many_parts":
        argv = ["grok"] + ["x"] * 4096
    else:
        argv = ["grok"] + ["x" * 65_536] * 17
    with pytest.raises(ValueError) as captured:
        bind_explicit_grok_max_turns(argv, 2)
    assert PRIVATE not in str(captured.value)


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(llm_router, "_grok_cli_command", lambda: "grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda command: True)
    # The authored boundary has no credential requirement or environment model.
    monkeypatch.setattr(llm_router, "_coalesce_env", lambda *args: "")
    set_last_cli_observation("grok_cli", {})
    value = llm_router._get_grok_cli_provider()
    assert value is not None
    return value


def forbid_native_effects(monkeypatch):
    effects = []
    def forbidden(*args, **kwargs):
        effects.append(True)
        raise AssertionError("invalid explicit turn policy reached native effects")
    monkeypatch.setattr(llm_router.tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(llm_router.subprocess, "run", forbidden)
    monkeypatch.setattr(llm_router.subprocess, "Popen", forbidden)
    monkeypatch.setattr(llm_router, "_run_cli_command", forbidden)
    return effects


@pytest.mark.parametrize("value", [None, True, False, "2", 2.0, 0, -1, MAX_TURNS + 1])
def test_actual_provider_refuses_invalid_explicit_value_before_prompt_or_process(provider, monkeypatch, value):
    effects = forbid_native_effects(monkeypatch)
    with pytest.raises(ValueError):
        provider.generate(PRIVATE, model_name="grok-4.7", grok_max_turns=value)
    assert effects == []


@pytest.mark.parametrize("command", [
    "opaque-wrapper --prompt {prompt}", "sh -c 'grok --max-turns 2'",
    ["grok", "--max-turns", "128"], ["grok", "--max-turns=128"],
    ["grok", "--max-turns", "2", "--max-turns=2"],
    ["grok", "--max-turns="], ["grok", "--max-turns", "02"],
    ["grok", "--max-turns"], ["grok", "--", "--max-turns", "2"],
])
def test_actual_provider_refuses_custom_command_bypass_before_all_native_effects(provider, monkeypatch, command):
    effects = forbid_native_effects(monkeypatch)
    with pytest.raises(ValueError):
        provider.generate(PRIVATE, model_name="grok-4.7", grok_cli_cmd=command,
                          grok_max_turns=2)
    assert effects == []


def success_payload(**changes):
    payload = {"text": "Authored response", "stopReason": "end_turn",
        "sessionId": "authored-native-session", "usage_is_incomplete": False,
        "usage": {"input_tokens": 7, "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1, "output_tokens": 3, "total_tokens": 13}}
    payload.update(changes)
    return payload


@pytest.mark.parametrize("command", [
    ["grok"], ["grok", "--max-turns", "2"], ["grok", "--max-turns=2"],
    "/fixture/grok --max-turns=2",
])
def test_actual_provider_passes_exactly_one_matching_bound_and_preserves_prompt_cleanup(
        provider, monkeypatch, command):
    calls, prompt_files = [], []
    original = deepcopy(command)
    def execute(argv, **kwargs):
        calls.append(list(argv))
        path = Path(argv[argv.index("--prompt-file") + 1])
        prompt_files.append(path)
        assert path.read_text() == PRIVATE
        bounds = [part.partition("=")[2] if part.startswith("--max-turns=") else argv[index + 1]
            for index, part in enumerate(argv) if part == "--max-turns" or part.startswith("--max-turns=")]
        assert bounds == ["2"] and PRIVATE not in argv
        return SimpleNamespace(returncode=0, stdout=json.dumps(success_payload()), stderr="")
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    assert provider.generate(PRIVATE, model_name="grok-4.7", grok_cli_cmd=command,
        grok_max_turns=2) == "Authored response"
    assert len(calls) == 1 and command == original
    assert prompt_files and all(not path.exists() for path in prompt_files)
    observed = get_last_cli_observation("grok_cli")
    assert observed["native_grok_total_tokens"] == 13


@pytest.mark.parametrize("command", ["grok --max-turns 128", "grok --max-turns=128",
    "opaque-wrapper --prompt {prompt}"])
def test_configured_command_has_the_same_explicit_authority_gate_as_per_call_override(
        provider, monkeypatch, command):
    monkeypatch.setattr(llm_router, "_grok_cli_command", lambda: command)
    configured = llm_router._get_grok_cli_provider()
    effects = forbid_native_effects(monkeypatch)
    with pytest.raises(ValueError):
        configured.generate(PRIVATE, model_name="grok-4.7", grok_max_turns=2)
    assert effects == []


def test_omitted_explicit_policy_keeps_legacy_opaque_route_unchanged(provider, monkeypatch):
    calls = []
    def legacy(command, prompt, **kwargs):
        calls.append((command, prompt))
        return "legacy authored response"
    monkeypatch.setattr(llm_router, "_run_cli_command", legacy)
    assert provider.generate(PRIVATE, model_name="grok-4.7",
        grok_cli_cmd="opaque-wrapper --prompt {prompt}") == "legacy authored response"
    assert calls == [("opaque-wrapper --prompt {prompt}", PRIVATE)]


@pytest.mark.parametrize("outcome", [{"stopReason": "max_turn_requests"}, {"type": "max_turns_reached"}])
def test_selected_turn_stop_retains_native_charge_and_never_retries_or_returns_partial_text(
        provider, monkeypatch, outcome):
    calls, prompt_files = [], []
    def execute(argv, **kwargs):
        calls.append(list(argv))
        prompt_files.append(Path(argv[argv.index("--prompt-file") + 1]))
        return SimpleNamespace(returncode=0, stdout=json.dumps(success_payload(**outcome)), stderr="")
    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    with pytest.raises(llm_router.LLMRouterError):
        provider.generate(PRIVATE, model_name="grok-4.7", grok_max_turns=2)
    assert len(calls) == 1 and all(not path.exists() for path in prompt_files)
    observation = get_last_cli_observation("grok_cli")
    assert observation["native_grok_total_tokens"] == 13
    assert observation["native_grok_task_complete"] is False
    assert observation["native_grok_reason_code"] == "max_turns"
    assert PRIVATE not in json.dumps({key: value for key, value in observation.items()
        if key.startswith("native_grok_")})


def test_actual_authored_stream_process_receives_exact_turn_option_and_owned_prompt_is_removed(
        provider, tmp_path):
    """Exercise real process/stream transport; this executable is an authored fixture."""
    script = tmp_path / "authored_cli.py"
    prompt_record, argv_record = tmp_path / "prompt-path", tmp_path / "argv.json"
    final = {"type": "end", "stopReason": "end_turn", "num_turns": 2,
        "usage_is_incomplete": False,
        "usage": {"input_tokens": 7, "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1, "output_tokens": 3, "total_tokens": 13}}
    lines = ["import json, sys", "from pathlib import Path",
        "assert sys.argv.count('--max-turns') == 1",
        "assert sys.argv[sys.argv.index('--max-turns')+1] == '2'",
        "assert sys.argv[sys.argv.index('--output-format')+1] == 'streaming-json'",
        "prompt = Path(sys.argv[sys.argv.index('--prompt-file')+1])",
        f"assert prompt.read_text() == {PRIVATE!r}",
        f"Path({str(prompt_record)!r}).write_text(str(prompt))",
        f"Path({str(argv_record)!r}).write_text(json.dumps(sys.argv))",
        f"print({json.dumps({'type':'text','data':'Authored streamed answer'})!r}, flush=True)",
        f"print({json.dumps(final)!r}, flush=True)"]
    script.write_text("\n".join(lines) + "\n")
    argv = [sys.executable, str(script)]
    output = provider.generate(PRIVATE, model_name="grok-4.7", timeout=5,
        grok_cli_cmd=argv, grok_max_turns=2, grok_stream_progress=True)
    assert output == "Authored streamed answer"
    captured = json.loads(argv_record.read_text())
    assert captured.count("--max-turns") == 1 and PRIVATE not in captured
    assert argv == [sys.executable, str(script)]
    assert not Path(prompt_record.read_text()).exists()
    observed = get_last_cli_observation("grok_cli")
    assert observed["native_grok_total_tokens"] == 13
    assert observed["native_grok_progress"]["response_usable"] is True
    assert observed["native_grok_progress"]["process_exit_observed"] is True
