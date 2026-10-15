"""Resolve complete-prompt counters from reviewed local assets without network.

Installed model mappings identify a local tokenizer, not a provider's current
remote implementation. Unmapped routes use a labelled tokenizer proxy or a
deterministic UTF-8 estimate. Explicit owner descriptors take precedence at the
caller; their validation failures must never be routed through this fallback.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import os
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .local_prompt_tokenizer import _local_regular_file, _validate_local_read
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter

SCHEMA = "supervisor-automatic-prompt-tokenizer@1"
MAX_RANK_ASSET_BYTES = 8 * 1024 * 1024
VERIFICATION_CHUNK_BYTES = 65_536
ENCODING_SPEC_REVISION = "tiktoken-openai-public@0.13.0"
_SUPPORTED_TIKTOKEN_VERSIONS = frozenset({"0.13.0"})
_MAPPED_PROVIDERS = frozenset({"openai", "codex_cli"})
_PUBLIC_RANK_ROOT = "https://openaipublic.blob.core.windows.net/encodings/"
_R50K_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
_CL100K_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
_O200K_PATTERN = "|".join((
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""", r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""", r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""", r"""\s+""",
))


@dataclass(frozen=True)
class _EncodingSpec:
    rank_name: str
    rank_sha256: str
    rank_count: int
    pattern: str
    maximum_rank: int | None = None


# Specifications and public digests are copied from the reviewed installed
# tiktoken 0.13.0 openai_public.py. Registry constructors are deliberately never
# invoked: their cache misses can fetch and mutate files. Special-token tables
# do not affect encode_ordinary, which treats every literal as ordinary text.
_ENCODING_SPECS = {
    "r50k_base": _EncodingSpec(
        "r50k_base", "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        50_256, _R50K_PATTERN),
    "p50k_base": _EncodingSpec(
        "p50k_base", "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        50_280, _R50K_PATTERN, maximum_rank=50_280),
    "cl100k_base": _EncodingSpec(
        "cl100k_base", "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        100_256, _CL100K_PATTERN),
    "o200k_base": _EncodingSpec(
        "o200k_base", "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        199_998, _O200K_PATTERN),
}
_ENCODING_SPECS["p50k_edit"] = _ENCODING_SPECS["p50k_base"]
_ENCODING_SPECS["o200k_harmony"] = _ENCODING_SPECS["o200k_base"]


class _UnavailableTokenizer(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _installed_tiktoken() -> tuple[Any, Any, str]:
    """Load only installed code and a reviewed version; no encoding registry."""
    try:
        version = metadata.version("tiktoken")
        if version not in _SUPPORTED_TIKTOKEN_VERSIONS:
            raise _UnavailableTokenizer("unsupported_tiktoken_version")
        module = importlib.import_module("tiktoken")
        model_module = importlib.import_module("tiktoken.model")
    except (ImportError, metadata.PackageNotFoundError) as error:
        raise _UnavailableTokenizer("tiktoken_runtime_unavailable") from error
    if getattr(module, "__version__", None) != version:
        raise _UnavailableTokenizer("tiktoken_runtime_version_mismatch")
    return module, model_module, version


def _cache_directories() -> tuple[Path, ...]:
    """Nominate at most three known roots, without scanning or temp-file probes."""
    # tempfile.gettempdir() may probe writable directories by creating files.
    # Use its already selected directory or the ordinary POSIX default choice
    # without probing. A missing nominated cache simply falls back to estimate.
    temporary = (tempfile.tempdir or os.environ.get("TMPDIR") or os.environ.get("TEMP")
                 or os.environ.get("TMP") or "/tmp")
    values = (os.environ.get("TIKTOKEN_CACHE_DIR"), os.environ.get("DATA_GYM_CACHE_DIR"),
              os.path.join(os.fsdecode(temporary), "data-gym-cache"))
    result = []
    for value in values:
        if not value or "\x00" in value or len(os.fsencode(value)) > 4096:
            continue
        path = Path(os.path.abspath(value))
        if path not in result:
            result.append(path)
    return tuple(result)


def _read_rank_asset(path: Path, *, expected_sha256: str) -> bytes:
    """Capture bounded, authenticated bytes with no-follow and race checks."""
    with _local_regular_file(path, maximum=MAX_RANK_ASSET_BYTES) as (file_fd, before):
        digest, chunks, byte_count = hashlib.sha256(), [], 0
        read_limit = min(MAX_RANK_ASSET_BYTES, before.st_size) + 1
        while byte_count < read_limit:
            chunk = os.read(file_fd, min(VERIFICATION_CHUNK_BYTES, read_limit - byte_count))
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
            chunks.append(chunk)
        _validate_local_read(file_fd, before, byte_count=byte_count,
                             maximum=MAX_RANK_ASSET_BYTES, observed_sha256=digest.hexdigest(),
                             expected_sha256=expected_sha256)
        return b"".join(chunks)


def _parse_ranks(raw: bytes, *, expected_count: int,
                 maximum_rank: int | None = None) -> dict[bytes, int]:
    # p50k reserves special ID 50256 inside the merge-rank range, so its
    # authenticated 50280 merge tokens have maximum rank 50280, not 50279.
    maximum_rank = expected_count - 1 if maximum_rank is None else maximum_rank
    if type(maximum_rank) is not int or not expected_count - 1 <= maximum_rank <= expected_count:
        raise ValueError("bounded cached tokenizer rank IDs required")
    ranks, seen = {}, set()
    for line in raw.splitlines():
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or len(ranks) >= expected_count:
            raise ValueError("malformed cached tokenizer ranks")
        token = base64.b64decode(parts[0], validate=True)
        rank = int(parts[1])
        if (not token or token in ranks or rank in seen or not 0 <= rank <= maximum_rank
                or parts[1] != str(rank).encode("ascii")):
            raise ValueError("malformed cached tokenizer ranks")
        ranks[token] = rank
        seen.add(rank)
    if len(ranks) != expected_count or any(bytes([value]) not in ranks for value in range(256)):
        raise ValueError("incomplete cached tokenizer ranks")
    return ranks


def _load_cached_ranks(encoding_name: str) -> tuple[dict[bytes, int], str]:
    """Read only the known cache filename for one reviewed public rank asset."""
    spec = _ENCODING_SPECS.get(encoding_name)
    if spec is None:
        raise _UnavailableTokenizer("unsupported_mapped_encoding")
    url = _PUBLIC_RANK_ROOT + spec.rank_name + ".tiktoken"
    filename = hashlib.sha1(url.encode("ascii")).hexdigest()
    for directory in _cache_directories():
        try:
            raw = _read_rank_asset(directory / filename, expected_sha256=spec.rank_sha256)
            return _parse_ranks(raw, expected_count=spec.rank_count,
                                maximum_rank=spec.maximum_rank), spec.rank_sha256
        except (ValueError, OSError, OverflowError):
            # Never repair/delete corrupt cache files or invoke a downloader.
            continue
    raise _UnavailableTokenizer("cached_tokenizer_unavailable")


def _build_encoding(module: Any, encoding_name: str, ranks: dict[bytes, int]) -> Any:
    return module.Encoding(name=encoding_name, pat_str=_ENCODING_SPECS[encoding_name].pattern,
                           mergeable_ranks=ranks, special_tokens={})


def _model_encoding(model_module: Any, model: str) -> tuple[str | None, str | None, str | None]:
    """Keep exact and prefix matches distinct; do not alias requested IDs."""
    exact = model_module.MODEL_TO_ENCODING
    if model in exact:
        return exact[model], "exact", None
    for prefix, encoding in model_module.MODEL_PREFIX_TO_ENCODING.items():
        if model.startswith(prefix):
            return encoding, "prefix", prefix
    return None, None, None


def _prompt_bytes(prompt: str) -> bytes:
    if type(prompt) is not str:
        raise ValueError("complete literal prompt text required")
    raw = prompt.encode("utf-8")
    if len(raw) > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("complete user prompt exceeds its existing byte bound")
    return raw


def _provenance(*, backend: str, version: str, encoding: str, kind: str,
                basis: str, reason: str, rank_sha256: str | None,
                match: str | None = None, prefix: str | None = None) -> dict:
    return {
        "schema": SCHEMA, "backend": backend, "backend_version": version,
        "encoding": encoding, "encoding_spec_revision": ENCODING_SPEC_REVISION if backend == "tiktoken" else None,
        "rank_sha256": rank_sha256, "count_kind": kind, "compatibility_basis": basis,
        "reason": reason, "model_mapping_match": match, "model_mapping_prefix": prefix,
        "offline": True, "remote_tokenizer_compatibility_verified": False,
        "provider_usage": False, "completion_authority": False,
    }


def _utf8_estimate(*, provider: str, model: str, reason: str) -> tuple[PromptTokenCounter, dict]:
    def count(prompt: str) -> int:
        return max(1, (len(_prompt_bytes(prompt)) + 3) // 4)

    basis = "utf8_byte_estimate"
    counter = PromptTokenCounter(provider=provider, model=model, tokenizer_id="utf8-div4",
        tokenizer_revision="utf8-div4@1", special_token_policy="utf8-literal-text@1",
        count_tokens=count, count_kind="estimate", compatibility_basis=basis)
    return counter, _provenance(backend="utf8-bytes", version="1", encoding="utf8-div4",
        kind="estimate", basis=basis, reason=reason, rank_sha256=None)


def resolve_prompt_tokenizer(*, provider: str, model: str) -> tuple[PromptTokenCounter, dict]:
    """Resolve one offline counter while preserving the exact requested route.

    Only reviewed installed OpenAI-route model mappings get tokenizer quality.
    Prefix mappings remain explicitly labelled and do not prove a remote model
    exists or uses that encoding. Other routes can use o200k_base only as an
    estimate. Every callback counts the entire literal text without truncation;
    empty text has a minimum count of one for the runner's counter contract.
    """
    for value in (provider, model):
        if type(value) is not str or not value.strip() or value != value.strip():
            raise ValueError("explicit provider and model identifiers required")
    try:
        module, model_module, version = _installed_tiktoken()
    except Exception as error:
        reason = error.reason if isinstance(error, _UnavailableTokenizer) else "tiktoken_runtime_unavailable"
        return _utf8_estimate(provider=provider, model=model, reason=reason)
    match = prefix = None
    encoding = None
    reason = "provider_mapping_unavailable"
    if provider in _MAPPED_PROVIDERS:
        try:
            encoding, match, prefix = _model_encoding(model_module, model)
        except Exception:
            reason = "installed_model_mapping_unavailable"
        else:
            reason = "unmapped_model" if encoding is None else "installed_tiktoken_model_mapping"
    choices = [(encoding, "tokenizer")] if encoding in _ENCODING_SPECS else []
    if encoding is not None and not choices:
        reason = "unsupported_mapped_encoding"
    if encoding != "o200k_base" or not choices:
        choices.append(("o200k_base", "estimate"))
    for candidate, kind in choices:
        try:
            ranks, asset_sha256 = _load_cached_ranks(candidate)
            tokenizer = _build_encoding(module, candidate, ranks)
        except Exception:
            if kind == "tokenizer":
                reason = "mapped_encoding_unavailable"
            continue

        def count(prompt: str, _tokenizer=tokenizer) -> int:
            _prompt_bytes(prompt)
            return max(1, len(_tokenizer.encode_ordinary(prompt)))

        basis = "installed_tiktoken_model_mapping" if kind == "tokenizer" else "offline_o200k_base_proxy"
        counter = PromptTokenCounter(provider=provider, model=model, tokenizer_id="tiktoken/" + candidate,
            tokenizer_revision=f"tiktoken/{version}/{candidate}/sha256:{asset_sha256}/ordinary-spec:1",
            special_token_policy="tiktoken.encode_ordinary:literal-text@1", count_tokens=count,
            count_kind=kind, compatibility_basis=basis)
        return counter, _provenance(backend="tiktoken", version=version, encoding=candidate,
            kind=kind, basis=basis, reason=reason, rank_sha256=asset_sha256,
            match=match, prefix=prefix)
    return _utf8_estimate(provider=provider, model=model, reason="cached_tokenizer_unavailable")
