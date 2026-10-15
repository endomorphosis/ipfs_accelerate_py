"""Reviewed cached tokenizers and labelled estimates, without remote discovery.

The cached-rank cases require the qualification environment's existing reviewed
o200k asset. Neither an installed mapping nor a proxy establishes remote model
compatibility or provider usage.
"""

import base64
import hashlib
import importlib
import socket
import urllib.request
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer as automatic
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES,
    PromptTokenCounter,
    select_model_prompt,
)

O200K_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
BINDING_FIELDS = {"provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"}


@pytest.fixture(autouse=True)
def no_download_or_registry(monkeypatch):
    attempted = []

    def forbidden(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("automatic tokenizer must use authenticated local bytes")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    try:
        module = importlib.import_module("tiktoken")
        loader = importlib.import_module("tiktoken.load")
    except ImportError:
        pass
    else:
        monkeypatch.setattr(module, "get_encoding", forbidden)
        monkeypatch.setattr(module, "encoding_for_model", forbidden)
        monkeypatch.setattr(loader, "read_file", forbidden)
        monkeypatch.setattr(loader, "read_file_cached", forbidden)
    yield
    assert attempted == []


@pytest.mark.parametrize("provider,model,match,prefix", [
    ("codex_cli", "gpt-5", "exact", None),
    ("openai", "gpt-5", "exact", None),
    ("codex_cli", "gpt-5-authored-version", "prefix", "gpt-5-"),
    ("openai", "gpt-5-authored-version", "prefix", "gpt-5-"),
])
def test_real_cached_mapping_retains_exact_route_and_mapping_quality(provider, model, match, prefix):
    counter, provenance = automatic.resolve_prompt_tokenizer(provider=provider, model=model)
    assert (counter.provider, counter.model) == (provider, model)
    assert counter.count_kind == "tokenizer"
    assert counter.compatibility_basis == "installed_tiktoken_model_mapping"
    assert provenance["backend"] == "tiktoken"
    assert provenance["backend_version"] == "0.13.0"
    assert provenance["encoding"] == "o200k_base"
    assert provenance["rank_sha256"] == O200K_SHA256
    assert provenance["model_mapping_match"] == match
    assert provenance["model_mapping_prefix"] == prefix
    assert provenance["count_kind"] == counter.count_kind
    assert provenance["compatibility_basis"] == counter.compatibility_basis
    assert provenance["offline"] is True
    assert provenance["remote_tokenizer_compatibility_verified"] is False
    assert provenance["provider_usage"] is provenance["completion_authority"] is False
    assert set(counter.binding()) == BINDING_FIELDS
    assert counter.count("hello world") == 2


def test_real_rank_bytes_cover_multilingual_code_and_special_spellings_as_ordinary_text():
    module, _, _ = automatic._installed_tiktoken()
    ranks, asset_sha256 = automatic._load_cached_ranks("o200k_base")
    assert asset_sha256 == O200K_SHA256
    assert len(ranks) == 199_998
    assert all(bytes([value]) in ranks for value in range(256))
    reference = module.Encoding(
        name="authored-offline-o200k-reference",
        pat_str=automatic._ENCODING_SPECS["o200k_base"].pattern,
        mergeable_ranks=ranks,
        special_tokens={"<|endoftext|>": 199_999},
    )
    counter, _ = automatic.resolve_prompt_tokenizer(provider="codex_cli", model="gpt-5")
    texts = [
        "café 法律 مرحبا 🙂 e\u0301",
        "def greeting(name):\n    return f'你好, {name}!\\n'\n",
        "<|endoftext|> <|fim_prefix|> <|im_start|> literal task requirements",
    ]
    for text in texts:
        assert counter.count(text) == len(reference.encode_ordinary(text))
        assert counter.count(text) == counter.count(text)
    assert counter.count("<|endoftext|>") > 1


@pytest.mark.parametrize("provider,model", [
    ("codex_cli", "gpt-5.6-sol"),
    ("grok_cli", "grok4.7"),
    ("xai", "grok-4.7"),
])
def test_unmapped_routes_use_proxy_without_aliasing_requested_identity(provider, model):
    counter, provenance = automatic.resolve_prompt_tokenizer(provider=provider, model=model)
    assert counter.binding()["provider"] == provider
    assert counter.binding()["model"] == model
    assert counter.count_kind == "estimate"
    assert counter.compatibility_basis == "offline_o200k_base_proxy"
    assert provenance["backend"] == "tiktoken"
    assert provenance["encoding"] == "o200k_base"
    assert provenance["rank_sha256"] == O200K_SHA256
    assert provenance["count_kind"] == "estimate"
    assert provenance["model_mapping_match"] is None
    assert provenance["remote_tokenizer_compatibility_verified"] is False
    assert counter.count("hello world") == 2


@pytest.mark.parametrize("provider", ["grok_cli", "xai"])
def test_foreign_provider_cannot_inherit_openai_mapping_quality(provider):
    counter, provenance = automatic.resolve_prompt_tokenizer(provider=provider, model="gpt-5")
    assert (counter.provider, counter.model) == (provider, "gpt-5")
    assert counter.count_kind == "estimate"
    assert provenance["compatibility_basis"] == "offline_o200k_base_proxy"
    assert provenance["model_mapping_match"] is None


@pytest.mark.parametrize("provider,model", [
    ("codex_cli", "gpt-5.6-sol"), ("grok_cli", "grok4.7"), ("xai", "grok-4.7"),
])
def test_unavailable_backend_uses_deterministic_utf8_estimate_without_other_io(
        monkeypatch, provider, model):
    def unavailable():
        raise ValueError("authored unavailable runtime")

    def forbidden():
        raise AssertionError("missing runtime must not inspect tokenizer caches")

    monkeypatch.setattr(automatic, "_installed_tiktoken", unavailable)
    monkeypatch.setattr(automatic, "_cache_directories", forbidden)
    counter, provenance = automatic.resolve_prompt_tokenizer(provider=provider, model=model)
    assert (counter.provider, counter.model) == (provider, model)
    assert counter.count_kind == "estimate"
    assert counter.compatibility_basis == "utf8_byte_estimate"
    assert provenance["backend"] == "utf8-bytes"
    assert provenance["encoding"] == "utf8-div4"
    assert provenance["rank_sha256"] is None
    assert provenance["remote_tokenizer_compatibility_verified"] is False
    for text in ("", "a", "hello", "你好🙂", "café\nreturn value", "<|endoftext|>"):
        assert counter.count(text) == max(1, (len(text.encode("utf-8")) + 3) // 4)


@pytest.mark.parametrize("cache_state", ["missing", "corrupt"])
def test_absent_or_corrupt_cache_is_not_repaired_and_does_not_gain_tokenizer_quality(
        tmp_path, monkeypatch, cache_state):
    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    asset = tmp_path / hashlib.sha1(url.encode("ascii")).hexdigest()
    raw = b"authored corrupt rank table"
    if cache_state == "corrupt":
        asset.write_bytes(raw)
    monkeypatch.setattr(automatic, "_cache_directories", lambda: (tmp_path,))
    counter, provenance = automatic.resolve_prompt_tokenizer(provider="codex_cli", model="gpt-5")
    assert counter.count_kind == "estimate"
    assert provenance["backend"] == "utf8-bytes"
    assert provenance["rank_sha256"] is None
    assert counter.count("你好🙂") == (len("你好🙂".encode()) + 3) // 4
    assert asset.exists() is (cache_state == "corrupt")
    if asset.exists():
        assert asset.read_bytes() == raw
    assert list(tmp_path.iterdir()) == ([asset] if cache_state == "corrupt" else [])


@pytest.mark.parametrize("provider,model", [
    (None, "gpt-5"), (True, "gpt-5"), ("", "gpt-5"), (" codex_cli", "gpt-5"),
    ("codex_cli", None), ("codex_cli", ""), ("codex_cli", "gpt-5 "),
])
def test_invalid_route_identity_refuses_before_backend_resolution(monkeypatch, provider, model):
    calls = []
    monkeypatch.setattr(automatic, "_installed_tiktoken", lambda: calls.append(True))
    with pytest.raises(ValueError, match="provider and model"):
        automatic.resolve_prompt_tokenizer(provider=provider, model=model)
    assert calls == []


@pytest.mark.parametrize("text", ["x" * (MAX_MODEL_PROMPT_BYTES + 1), "é" * (MAX_MODEL_PROMPT_BYTES // 2 + 1)],
                         ids=("oversized_ascii", "oversized_unicode"))
def test_automatic_counter_never_truncates_oversized_literal_text(text):
    counter, _ = automatic.resolve_prompt_tokenizer(provider="codex_cli", model="gpt-5")
    with pytest.raises(ValueError, match="byte bound"):
        counter.count(text)


def test_resolved_counter_cannot_cross_provider_or_model_bindings(tmp_path):
    counter, _ = automatic.resolve_prompt_tokenizer(provider="grok_cli", model="grok4.7")
    for provider, model in (("codex_cli", "grok4.7"), ("grok_cli", "grok-4.7")):
        with pytest.raises(ValueError, match="pinned provider/model"):
            select_model_prompt(native_prompt="native", semantic_prompt=None, suffix="",
                                purpose="coding", workspace=tmp_path, provider=provider,
                                model=model, token_counter=counter)


def test_counter_quality_is_separate_from_unchanged_five_field_binding():
    counter = PromptTokenCounter(
        provider="codex_cli", model="authored", tokenizer_id="authored:utf8",
        tokenizer_revision="fixture-v1", special_token_policy="fixture-only",
        count_tokens=lambda text: len(text.encode()),
    )
    assert counter.count_kind == "tokenizer"
    assert counter.compatibility_basis == "caller_declaration"
    estimate = replace(counter, count_kind="estimate", compatibility_basis="authored_proxy")
    observed = replace(estimate, count_tokens=lambda _: 7)
    assert observed.binding() == estimate.binding() == counter.binding()
    assert set(observed.binding()) == BINDING_FIELDS
    assert (observed.count_kind, observed.compatibility_basis, observed.count("literal")) == (
        "estimate", "authored_proxy", 7)


def test_rank_parser_accepts_explicit_single_special_token_gap():
    rows = [base64.b64encode(bytes([value])) + b" " + str(value).encode() for value in range(256)]
    raw = b"\n".join([*rows, base64.b64encode(b"ab") + b" 257"]) + b"\n"
    ranks = automatic._parse_ranks(raw, expected_count=257, maximum_rank=257)
    assert len(ranks) == 257 and ranks[b"ab"] == 257
    assert 256 not in ranks.values()
    with pytest.raises(ValueError, match="ranks"):
        automatic._parse_ranks(raw, expected_count=257)


@pytest.mark.parametrize("rank", [255, 258])
def test_sparse_rank_permission_does_not_allow_duplicates_or_out_of_bound_ids(rank):
    rows = [base64.b64encode(bytes([value])) + b" " + str(value).encode() for value in range(256)]
    raw = b"\n".join([*rows, base64.b64encode(b"ab") + b" " + str(rank).encode()])
    with pytest.raises(ValueError, match="ranks"):
        automatic._parse_ranks(raw, expected_count=257, maximum_rank=257)
