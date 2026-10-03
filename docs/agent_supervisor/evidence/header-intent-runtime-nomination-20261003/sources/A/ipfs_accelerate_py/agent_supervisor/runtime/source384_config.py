"""Closed offline Source384 selection shared by deployment and native consumers.

Selecting these assets does not establish source coverage or proof authority.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

SCHEMA = "terminal-source384-config@1"
HEADER_SCHEMA = "terminal-source384-config@2"
MAX_CONFIG_BYTES = 32768


def validate_source384_config(value):
    """Validate portable JSON and native embedding pins without opening assets."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    fields = {"schema", "mode", "checkpoint_path", "checkpoint_sha256", "embedding_snapshot",
              "embedding_revision", "embedding_assets", "training_steps", "download_calls"}
    if type(value) is dict and value.get("schema") == HEADER_SCHEMA:
        fields.add("header_applicability")
    if type(value) is not dict or set(value) != fields:
        raise ValueError("closed Source384 config required")
    expected = [{"name": name, "sha256": sha, "bytes": size}
                for name, (size, sha) in sorted(embedding._PINNED_ASSETS.items())]
    if (value["schema"] not in {SCHEMA, HEADER_SCHEMA} or value["mode"] != "pinned_parent"
            or type(value["checkpoint_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value["checkpoint_sha256"]) is None
            or value["embedding_revision"] != embedding.PINNED_REVISION
            or type(value["embedding_assets"]) is not list
            or value["embedding_assets"] != expected
            or any(type(row) is not dict or set(row) != {"name", "sha256", "bytes"}
                   or type(row["bytes"]) is not int for row in value["embedding_assets"])
            or type(value["training_steps"]) is not int or value["training_steps"] != 0
            or type(value["download_calls"]) is not int or value["download_calls"] != 0):
        raise ValueError("Source384 mode, numerical activity or native asset pins differ")
    for key in ("checkpoint_path", "embedding_snapshot"):
        item = value[key]
        if (type(item) is not str or not 0 < len(item) <= 4096 or "\x00" in item
                or not Path(item).is_absolute() or str(Path(item)) != item or ".." in Path(item).parts):
            raise ValueError("canonical absolute Source384 asset paths required")
    if value["schema"] == HEADER_SCHEMA:
        from .header_intent_applicability import validate_runtime_profile
        validate_runtime_profile(value["header_applicability"])
    return deepcopy(value)


def _regular_bytes(path, maximum):
    """Read bounded stable regular bytes without blocking on a FIFO or symlink."""
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("canonical Source384 file required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum:
            raise ValueError("bounded independent regular Source384 file required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(raw) > maximum or identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError("Source384 file changed during bounded read")
    return raw


def validate_source384_assets(config):
    """Replay native offline checkpoint compatibility and exact embedding pins."""
    from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured
    from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384_v2 import load_source_program_decoder_384_v2
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    config = validate_source384_config(config)
    checkpoint = Path(config["checkpoint_path"])
    raw = _regular_bytes(checkpoint, structured.MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("Source384 checkpoint bytes differ")
    load_source_program_decoder_384_v2(checkpoint, expected_sha256=config["checkpoint_sha256"])
    snapshot = Path(config["embedding_snapshot"])
    if snapshot.resolve(strict=True) != snapshot:
        raise ValueError("canonical Source384 embedding snapshot required")
    if embedding._snapshot_assets(snapshot)[1] != config["embedding_assets"]:
        raise ValueError("Source384 embedding assets differ")
    if _regular_bytes(checkpoint, structured.MAX_BYTES) != raw:
        raise ValueError("Source384 checkpoint changed during validation")
    return config


def load_source384_config(path):
    """Return the unchanged closed selection after validating its local assets."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate Source384 config field")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite Source384 config value")
    value = json.loads(_regular_bytes(path, MAX_CONFIG_BYTES), object_pairs_hook=unique, parse_constant=nonfinite)
    return validate_source384_assets(value)
