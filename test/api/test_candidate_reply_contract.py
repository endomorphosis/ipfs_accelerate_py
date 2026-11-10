"""Second round acknowledgment binds its input without claiming its postimage."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_reply_contract as module


@pytest.fixture
def binding():
    return dict(task_id="TASK-1", round_number=2, round_context_sha256="a" * 64,
                workspace_generation="b" * 64)


def ack(binding):
    return dict(schema=module.SCHEMA, status="candidate_ready", **binding)


def test_explicit_override_preserves_complete_native_instructions_and_provides_closed_schema(binding):
    native = 'literal original grammar: {"schema":"supervisor-semantic-router-reply@1"}\nKeep λ and s0.'
    prompt, receipt = module.apply_candidate_reply_contract(model_prompt=native, binding=binding)
    assert prompt.startswith(native)
    assert "supersedes any reply grammar" in prompt[len(native):]
    assert json.dumps(ack(binding), sort_keys=True, ensure_ascii=False, separators=(",", ":")) in prompt
    assert receipt["model_prompt_before_bytes"] == len(native.encode())
    schema = module.candidate_completion_schema(binding)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["round_number"] == dict(type="integer", enum=[2])
    assert set(schema["required"]) == set(ack(binding))
    text = json.dumps(ack(binding))
    assert module.validate_candidate_completion_reply(text, binding=binding) == text
    assert receipt["proof_authority"] is receipt["completion_authority"] is False


@pytest.mark.parametrize("change", ["generation", "round", "task", "context", "extra", "duplicate",
    "nonfinite", "prose", "markdown", "baseline", "residual", "oversize", "float_round", "missing"])
def test_refuses_unbound_or_reserved_model_output(binding, change):
    value = ack(binding)
    if change in {"generation", "round", "task", "context"}:
        key = {"generation": "workspace_generation", "round": "round_number",
               "task": "task_id", "context": "round_context_sha256"}[change]
        value[key] = 3 if change == "round" else "foreign"
    elif change == "extra":
        value["proof_authority"] = True
    elif change == "float_round":
        value["round_number"] = 2.0
    elif change == "missing":
        value.pop("workspace_generation")
    elif change == "baseline":
        value = dict(schema="supervisor-semantic-router-reply@1", response=ack(binding))
    elif change == "residual":
        value = dict(candidate_only=True, validated=False, structured_payload=ack(binding))
    text = json.dumps(value)
    if change == "duplicate":
        text = '{"status":"candidate_ready",' + text[1:]
    elif change == "nonfinite":
        text = text.replace('"candidate_ready"', "NaN")
    elif change == "prose":
        text += "\nFinished."
    elif change == "markdown":
        text = "```json\n" + text + "\n```"
    elif change == "oversize":
        text += " " * module.MAX_REPLY_BYTES
    with pytest.raises(ValueError):
        module.validate_candidate_completion_reply(text, binding=binding)


def test_binding_and_total_prompt_are_bounded(binding):
    with pytest.raises(ValueError):
        module.candidate_completion_schema({**binding, "round_number": True})
    with pytest.raises(ValueError):
        module.apply_candidate_reply_contract(model_prompt="x" * module.MAX_PROMPT_BYTES, binding=binding)


def test_ack_is_generation_reference_not_a_postedit_workspace_gate(binding, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("input candidate")
    raw = json.dumps(ack(binding))
    source.write_text("provider deliberately edits candidate")
    assert module.validate_candidate_completion_reply(raw, binding=binding) == raw
