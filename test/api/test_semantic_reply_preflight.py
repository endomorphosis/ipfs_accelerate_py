"""Cheap reply refusal retains fresh verification before native alias expansion.

Producer artifacts and source checks execute normally for valid candidates.
These are protocol fixtures, never provider, semantic or completion evidence.
"""

from __future__ import annotations

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence import structured_decoding
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from test.api.test_semantic_router_translation import (  # noqa: F401 -- register fixture
    _reply,
    native,
)


@pytest.mark.parametrize("mutation", [
    "extra_field", "missing_response", "table", "task", "root", "scope", "body_type",
    "candidate_false", "candidate_zero", "candidate_missing",
])
def test_basic_reply_refusal_precedes_producer_reencoding(native, monkeypatch, mutation):  # noqa: F811
    repository, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, _ = _reply(encoded)
    if mutation == "extra_field":
        reply["unexpected"] = "forbidden"
    elif mutation == "missing_response":
        del reply["response"]
    elif mutation in {"table", "task", "root", "scope"}:
        key = {"table": "translation_cid", "task": "task_id",
               "root": "semantic_root_cid", "scope": "scope_cid"}[mutation]
        reply[key] = "foreign"
    elif mutation == "body_type":
        reply["response"] = []
    elif mutation == "candidate_missing":
        del reply["response"]["candidate_only"]
    else:
        reply["response"]["candidate_only"] = False if mutation == "candidate_false" else 0
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("an invalid basic reply binding must not repeat producer work")

    monkeypatch.setattr(codec, "encode_semantic_router_prompt", forbidden)
    monkeypatch.setattr(structured_decoding, "decode_structured_output", forbidden)
    with pytest.raises(codec.SemanticTranslationError, match="binding differs|native candidate"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=repository)
    assert calls == []


def test_valid_reply_still_reencodes_current_producer_before_native_decoding(native, monkeypatch):  # noqa: F811
    repository, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, symbol = _reply(encoded)
    actual_encode = codec.encode_semantic_router_prompt
    actual_decode = structured_decoding.decode_structured_output
    events = []

    def encode(**kwargs):
        assert kwargs == {"prompt": prompt, "repository": repository}
        events.append("fresh_verification_started")
        result = actual_encode(**kwargs)
        events.append("fresh_verification_finished")
        return result

    def decode(*args, **kwargs):
        events.append("native_grammar_entered")
        return actual_decode(*args, **kwargs)

    monkeypatch.setattr(codec, "encode_semantic_router_prompt", encode)
    monkeypatch.setattr(structured_decoding, "decode_structured_output", decode)
    decoded = codec.decode_semantic_router_response(
        response=json.dumps(reply), encoded=encoded, repository=repository)
    assert events == ["fresh_verification_started", "fresh_verification_finished", "native_grammar_entered"]
    assert json.loads(decoded.text)["structured_payload"]["symbol_ids"] == [symbol]
    assert decoded.receipt["freshness_checked"] is True
    assert decoded.receipt["native_grammar_validated"] is True
    assert decoded.receipt["completion_authority"] is False


def test_stale_valid_reply_refuses_before_alias_expansion_or_native_grammar(native, monkeypatch):  # noqa: F811
    repository, _, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    reply, _ = _reply(encoded)
    response = json.dumps(reply)
    original_aliases = reply["response"]["structured_payload"]["symbol_ids"]
    (repository / "mod.py").write_text(source + "# source changed after dispatch\n")
    actual_encode, actual_parse = codec.encode_semantic_router_prompt, codec._parse
    parsed_responses, events = [], []

    def parse(text):
        result = actual_parse(text)
        if text == response:
            parsed_responses.append(result)
        return result

    def encode(**kwargs):
        events.append("fresh_verification_started")
        return actual_encode(**kwargs)

    def forbidden(*args, **kwargs):
        events.append("native_grammar_entered")
        raise AssertionError("stale reply must not enter native grammar")

    monkeypatch.setattr(codec, "_parse", parse)
    monkeypatch.setattr(codec, "encode_semantic_router_prompt", encode)
    monkeypatch.setattr(structured_decoding, "decode_structured_output", forbidden)
    with pytest.raises(codec.SemanticTranslationError, match="stale"):
        codec.decode_semantic_router_response(response=response, encoded=encoded, repository=repository)
    assert events == ["fresh_verification_started"]
    assert len(parsed_responses) == 1
    assert parsed_responses[0]["response"]["structured_payload"]["symbol_ids"] == original_aliases
