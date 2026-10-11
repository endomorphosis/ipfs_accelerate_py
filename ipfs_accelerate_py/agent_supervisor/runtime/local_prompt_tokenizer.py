"""Load an owner-declared, hash-bound tokenizer from local JSON only.

This adapter measures complete user-message text. Its descriptor is a trusted
launch input, not evidence that the tokenizer is equivalent to a remote model.
It loads no weights, discovers no assets, and provides no network fallback.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA = "supervisor-local-prompt-tokenizer@1"
MAX_DESCRIPTOR_BYTES = 16_384
MAX_TOKENIZER_BYTES = 32 * 1024 * 1024
MAX_PROMPT_BYTES = 256_000
VERIFICATION_CHUNK_BYTES = 65_536
_FIELDS = frozenset({
    "schema", "provider", "model", "tokenizer_id", "tokenizer_revision",
    "special_token_policy", "backend", "tokenizers_version", "asset_path",
    "asset_sha256",
})
_POLICIES = {
    "tokenizers.encode:add_special_tokens=true": True,
    "tokenizers.encode:add_special_tokens=false": False,
}
_BINDING_FIELDS = (
    "provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy",
)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"explicit lowercase SHA-256 required for {label}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"explicit {label} required")
    return value


def _local_path(value: str | Path, *, base: Path | None = None) -> Path:
    text = os.fspath(value)
    if (not isinstance(text, str) or not text or "\x00" in text or ":" in text
            or "\\" in text or text != str(Path(text))
            or any(part in {".", ".."} for part in text.split("/"))):
        raise ValueError("canonical local path required; URLs and traversal are forbidden")
    path = Path(text)
    if not path.is_absolute():
        path = (base if base is not None else Path.cwd()) / path
    return path


@contextmanager
def _local_regular_file(path: Path, *, maximum: int):
    """Open bounded local input with no-follow checks on every path component."""
    directory_fd = file_fd = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        directory_fd = os.open(path.anchor, directory_flags)
        for part in path.parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                          dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= maximum:
            raise ValueError("local tokenizer input must be a bounded nonempty regular file")
        yield file_fd, before
    except OSError as error:
        raise ValueError("local tokenizer input unavailable or contains a symbolic link") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _validate_local_read(file_fd: int, before: os.stat_result, *, byte_count: int,
                         maximum: int, observed_sha256: str, expected_sha256: str) -> None:
    after = os.fstat(file_fd)
    def identity(item):
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if identity(before) != identity(after) or byte_count != before.st_size:
        raise ValueError("local tokenizer input changed during read")
    if byte_count > maximum or observed_sha256 != expected_sha256:
        raise ValueError("local tokenizer input SHA-256 mismatch or byte bound exceeded")


def _bounded_local_bytes(path: Path, *, maximum: int, expected_sha256: str) -> bytes:
    """Capture authenticated bytes for the initial JSON/from_str loader only."""
    with _local_regular_file(path, maximum=maximum) as (file_fd, before):
        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        _validate_local_read(
            file_fd, before, byte_count=len(raw), maximum=maximum,
            observed_sha256=hashlib.sha256(raw).hexdigest(), expected_sha256=expected_sha256)
        return raw


def _verify_local_file(path: Path, *, maximum: int, expected_sha256: str) -> None:
    """Authenticate the whole file with one bounded chunk retained at a time.

    Reverification does not allocate the complete asset. The same opening,
    size, race and hash checks used for initial capture remain in force.
    No file identity, timestamp or digest is used as a cache substitute.
    """
    with _local_regular_file(path, maximum=maximum) as (file_fd, before):
        digest = hashlib.sha256()
        byte_count = 0
        read_limit = min(maximum, before.st_size) + 1
        while byte_count < read_limit:
            chunk = os.read(file_fd, min(VERIFICATION_CHUNK_BYTES, read_limit - byte_count))
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
            del chunk
        _validate_local_read(
            file_fd, before, byte_count=byte_count, maximum=maximum,
            observed_sha256=digest.hexdigest(), expected_sha256=expected_sha256)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite JSON value")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON value")
        return result

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=constant, parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError(f"{label} must contain finite JSON without duplicate fields") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return result


def _runtime(version: str):
    try:
        installed = metadata.version("tokenizers")
        module = importlib.import_module("tokenizers")
    except (ImportError, metadata.PackageNotFoundError) as error:
        raise ValueError("declared local tokenizers runtime is unavailable") from error
    if installed != version or getattr(module, "__version__", None) != version:
        raise ValueError("declared tokenizers runtime version differs from installed runtime")
    return module


@dataclass(frozen=True)
class OwnerPromptTokenizer:
    """Loaded local counter; compatibility remains an owner declaration."""

    provider: str
    model: str
    tokenizer_id: str
    tokenizer_revision: str
    special_token_policy: str
    _artifact: Path = field(repr=False)
    _artifact_sha256: str = field(repr=False)
    _asset: Path = field(repr=False)
    _asset_sha256: str = field(repr=False)
    _tokenizers_version: str = field(repr=False)
    _tokenizer: Any = field(repr=False, compare=False)

    def binding(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _BINDING_FIELDS}

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "supervisor-local-prompt-tokenizer-provenance@1",
            "backend": "tokenizers-json@1",
            "descriptor_path": str(self._artifact),
            "descriptor_sha256": self._artifact_sha256,
            "asset_path": str(self._asset), "asset_sha256": self._asset_sha256,
            "tokenizers_version": self._tokenizers_version,
            "compatibility_basis": "owner_declaration", "completion_authority": False,
        }

    def _verify_binding(self) -> None:
        _verify_local_file(self._artifact, maximum=MAX_DESCRIPTOR_BYTES,
                           expected_sha256=self._artifact_sha256)
        _verify_local_file(self._asset, maximum=MAX_TOKENIZER_BYTES,
                           expected_sha256=self._asset_sha256)
        _runtime(self._tokenizers_version)
        if self._tokenizer.truncation is not None or self._tokenizer.padding is not None:
            raise ValueError("local tokenizer truncation and padding must remain disabled")

    def count(self, prompt: str) -> int:
        if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("complete user prompt exceeds its existing byte bound")
        self._verify_binding()
        try:
            encoded = self._tokenizer.encode(
                prompt, add_special_tokens=_POLICIES[self.special_token_policy])
            result = len(encoded.ids)
        except Exception as error:
            raise ValueError("local tokenizer could not encode the complete user prompt") from error
        self._verify_binding()
        if type(result) is not int or result < 1 or encoded.overflowing:
            raise ValueError("local tokenizer must return a positive untruncated token count")
        return result


def load_owner_prompt_tokenizer(*, artifact: str | Path, expected_sha256: str,
                                provider: str, model: str) -> OwnerPromptTokenizer:
    """Load the exact owner-selected local descriptor and its JSON tokenizer.

    Relative asset paths stay inside the descriptor directory. Absolute asset
    paths are explicit owner choices. Neither hashes nor schema/version checks
    establish that these assets match a provider's unpublished tokenizer.
    """
    expected_sha256 = _sha(expected_sha256, "descriptor")
    provider, model = _text(provider, "provider"), _text(model, "model")
    artifact_path = _local_path(artifact)
    raw = _bounded_local_bytes(artifact_path, maximum=MAX_DESCRIPTOR_BYTES,
                               expected_sha256=expected_sha256)
    descriptor = _json_object(raw, "local tokenizer descriptor")
    if set(descriptor) != _FIELDS or descriptor.get("schema") != SCHEMA:
        raise ValueError("closed local tokenizer descriptor schema required")
    for name in _FIELDS:
        _text(descriptor[name], name)
    if descriptor["provider"] != provider or descriptor["model"] != model:
        raise ValueError("local tokenizer differs from the pinned provider/model")
    if descriptor["backend"] != "tokenizers-json@1":
        raise ValueError("unsupported local tokenizer backend")
    if descriptor["special_token_policy"] not in _POLICIES:
        raise ValueError("explicit supported tokenizer special-token policy required")
    asset_hash = _sha(descriptor["asset_sha256"], "tokenizer asset")
    revision = "tokenizers/" + descriptor["tokenizers_version"] + "/sha256:" + asset_hash
    if descriptor["tokenizer_revision"] != revision:
        raise ValueError("tokenizer revision must bind runtime version and exact asset SHA-256")
    asset_path = _local_path(descriptor["asset_path"], base=artifact_path.parent)
    asset = _bounded_local_bytes(asset_path, maximum=MAX_TOKENIZER_BYTES,
                                 expected_sha256=asset_hash)
    asset_object = _json_object(asset, "local tokenizer asset")
    if ("truncation" not in asset_object or "padding" not in asset_object
            or asset_object["truncation"] is not None or asset_object["padding"] is not None):
        raise ValueError("local tokenizer asset requires explicit null truncation and padding")
    module = _runtime(descriptor["tokenizers_version"])
    try:
        tokenizer = module.Tokenizer.from_str(asset.decode("utf-8"))
    except Exception as error:
        raise ValueError("local tokenizer asset is not supported tokenizer JSON") from error
    loaded = OwnerPromptTokenizer(
        **{name: descriptor[name] for name in _BINDING_FIELDS},
        _artifact=artifact_path, _artifact_sha256=expected_sha256,
        _asset=asset_path, _asset_sha256=asset_hash,
        _tokenizers_version=descriptor["tokenizers_version"], _tokenizer=tokenizer,
    )
    loaded._verify_binding()
    return loaded
