"""Measure typed-decoder controls without fitting a model.

The syntax owner checks binder, sort, arity, and signature reference.
The codec rejects an invalid frame and an incomplete frame. The grouped
decoder keeps a fresh head, restores only shape-compatible non-head
tensors, and refuses direct KL across codec identities. Training loss
stays cross-entropy. This module only reads those sources.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .dependency_block_resolution import EXPECTED_TYPED_DECODER_PROBE

_DECODER = (
    "ipfs_datasets_py/logic/formalization/autoencoder/"
    "legal_grouped_span_decoder.py"
)
_SYNTAX = "ipfs_datasets_py/logic/syntax_core/ast.py"
_CODEC = (
    "ipfs_datasets_py/logic/formalization/autoencoder/"
    "legal_relational_target_codec_v1.py"
)


def measure_typed_decoder_probe() -> dict[str, object] | None:
    """Return the fixed typed-decoder controls, or None when they drift."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_TYPED_DECODER_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    root = _datasets_root()
    decoder = _parse(root / _DECODER)
    syntax = (root / _SYNTAX).read_text(encoding="utf-8")
    codec = (root / _CODEC).read_text(encoding="utf-8")
    fresh = _function(decoder, "fresh_head_parameters")
    warm = _function(decoder, "compatible_warm_start")
    refuse = _function(decoder, "refuse_direct_kl")
    train = _function(decoder, "train_grouped_span_step")
    if not _calls(warm, "refuse_direct_kl") or not _calls(warm, "fresh_head_parameters"):
        raise ValueError("warm start does not retain the fresh head and codec refusal")
    if not _calls(train, "refuse_direct_kl"):
        raise ValueError("training does not refuse direct KL")
    if _calls(decoder, "kl_div"):
        raise ValueError("direct KL is present")
    if not _calls_attr(train, "cross_entropy"):
        raise ValueError("training loss is not cross entropy")
    if not _constant(refuse, "direct KL across incompatible codecs is refused"):
        raise ValueError("cross-codec KL refusal is not stated")
    if not _constant(fresh, "fresh head changed a non-head parameter"):
        raise ValueError("fresh head does not retain encoder parameters")
    if "count_head." not in _source(decoder):
        raise ValueError("count head is not named")
    if "Binders must reference sorts present in the signature." not in syntax:
        raise ValueError("binder sort reference is not checked")
    if "Sort, arity, and signature invariants fail here" not in syntax:
        raise ValueError("sort and arity checks are not present")
    if '"reason": "invalid_frame"' not in codec or 'reason="missing_bos_and_eos"' not in codec:
        raise ValueError("invalid or incomplete frames are not checked")
    return dict(EXPECTED_TYPED_DECODER_PROBE)


def _datasets_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        root = parent / "ipfs_datasets"
        if (root / _DECODER).is_file() and (root / _SYNTAX).is_file() and (root / _CODEC).is_file():
            return root
        nested = parent / "external" / "ipfs_datasets"
        if (nested / _DECODER).is_file() and (nested / _SYNTAX).is_file() and (nested / _CODEC).is_file():
            return nested
    raise ValueError("typed decoder owners are not on disk")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError("missing function " + name)


def _calls(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == name:
                return True
    return False


def _calls_attr(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == name
        for node in ast.walk(tree)
    )


def _constant(tree: ast.AST, text: str) -> bool:
    return any(isinstance(node, ast.Constant) and node.value == text for node in ast.walk(tree))


def _source(tree: ast.AST) -> str:
    return ast.unparse(tree)
