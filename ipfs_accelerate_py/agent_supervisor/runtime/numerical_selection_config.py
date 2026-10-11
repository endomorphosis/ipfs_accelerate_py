"""Explicit offline selection of the separate experimental NumPy decoder."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .source384_config import _regular_bytes, validate_source384_config

SCHEMA = "supervisor-numerical-selection-config@1"
PROFILE = "numpy-unary-selection-384@1"
PUBLISHED_CHECKPOINT_SHA256 = "26ccc7cf8420e0ed6cbbe22603e09be0e3572fb32a1e489e877fc5081f506e03"
MAX_CONFIG_BYTES = 32768


def validate_numerical_selection_config(value):
    fields = {"schema", "profile", "allow_experimental", "checkpoint_path", "checkpoint_sha256",
              "embedding_snapshot", "embedding_revision", "embedding_assets", "training_steps", "download_calls"}
    if (type(value) is not dict or set(value) != fields or value["schema"] != SCHEMA
            or value["profile"] != PROFILE or value["allow_experimental"] is not True):
        raise ValueError("explicit closed experimental numerical selection required")
    # Reuse the portable asset/path contract, never its scalar checkpoint loader.
    common = {k: v for k, v in value.items() if k not in {"schema", "profile", "allow_experimental"}}
    validate_source384_config(dict(schema="terminal-source384-config@1", mode="pinned_parent", **common))
    return deepcopy(value)


def validate_numerical_selection_assets(value):
    from ipfs_datasets_py.logic.formalization.autoencoder.numerical_selection_384 import load_selection_checkpoint, MAX_BYTES
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    value = validate_numerical_selection_config(value)
    if value["checkpoint_sha256"] != PUBLISHED_CHECKPOINT_SHA256:
        raise ValueError("explicit published numerical checkpoint selection required")
    path = Path(value["checkpoint_path"])
    raw = _regular_bytes(path, MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != value["checkpoint_sha256"]:
        raise ValueError("numerical checkpoint file bytes differ")
    load_selection_checkpoint(path, expected_sha256=value["checkpoint_sha256"])
    snapshot = Path(value["embedding_snapshot"])
    if snapshot.resolve(strict=True) != snapshot or embedding._snapshot_assets(snapshot)[1] != value["embedding_assets"]:
        raise ValueError("numerical embedding snapshot differs")
    if _regular_bytes(path, MAX_BYTES) != raw:
        raise ValueError("numerical checkpoint changed during validation")
    return value


def load_numerical_selection_config(path, expected_sha256=None):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate numerical config field")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite numerical config value")
    raw = _regular_bytes(path, MAX_CONFIG_BYTES)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("numerical config differs from selected digest")
    return validate_numerical_selection_assets(json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite))


def published_numerical_selection_config(*, checkpoint_path, embedding_snapshot):
    """Describe already-local release assets; never download or select implicitly."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    return validate_numerical_selection_config(dict(schema=SCHEMA, profile=PROFILE, allow_experimental=True,
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=PUBLISHED_CHECKPOINT_SHA256,
        embedding_snapshot=str(embedding_snapshot), embedding_revision=embedding.PINNED_REVISION,
        embedding_assets=[dict(name=n, bytes=size, sha256=digest)
            for n, (size, digest) in sorted(embedding._PINNED_ASSETS.items())], training_steps=0, download_calls=0))
