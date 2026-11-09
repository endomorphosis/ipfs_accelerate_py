"""Sealed compiled-source drift is restored from the published inventory cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.rescue.sealed_source_restore import (
    restore_sealed_source_drift,
)

_INVENTORY = "tools/lean_refactor_compiled_sources_v1.json"
_PROFILE = "config/lean_refactor_arena_20261024_execution_v2.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tree(root: Path, source_name: str, source: bytes, *, extra: dict | None = None) -> str:
    target = root / source_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    retained = {source_name: _sha(source)}
    if extra:
        retained.update(extra)
    inventory = {
        "schema": "jevops.lean-refactor.compiled-source-inventory.v1",
        "retained": retained,
        "imports": {},
        "baseline_profile_sha256": "ab" * 32,
        "baseline_profile": "baseline",
        "extension_changes": [],
    }
    inventory_path = root / _INVENTORY
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    digest = _sha(inventory_path.read_bytes())
    profile_path = root / _PROFILE
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps({"source_sha256": {_INVENTORY: digest}}),
        encoding="utf-8",
    )
    return digest


def test_drifted_retained_file_is_restored_after_the_cache_is_warm(tmp_path: Path) -> None:
    source_name = "sources/kept.py"
    _write_tree(tmp_path, source_name, b"print(1)\n")
    assert restore_sealed_source_drift(tmp_path)["restored"] == []
    target = tmp_path / source_name
    target.write_bytes(b"print(2)\n")
    result = restore_sealed_source_drift(tmp_path)
    assert result["restored"] == [source_name]
    assert target.read_bytes() == b"print(1)\n"
    assert result["completion_authority"] is False
    assert result["work_complete"] is False


def test_operator_hold_keeps_drifted_bytes(tmp_path: Path) -> None:
    source_name = "sources/kept.py"
    _write_tree(tmp_path, source_name, b"print(1)\n")
    restore_sealed_source_drift(tmp_path)
    target = tmp_path / source_name
    target.write_bytes(b"print(2)\n")
    hold = tmp_path / "data/lean_refactor_arena_20261024/lane/HOLD"
    hold.parent.mkdir(parents=True)
    hold.write_text("held\n", encoding="utf-8")
    result = restore_sealed_source_drift(tmp_path)
    assert result["reason"] == "operator_hold"
    assert result["restored"] == []
    assert target.read_bytes() == b"print(2)\n"


def test_drifted_inventory_is_restored_before_its_children(tmp_path: Path) -> None:
    source_name = "sources/kept.py"
    _write_tree(tmp_path, source_name, b"print(1)\n")
    restore_sealed_source_drift(tmp_path)
    (tmp_path / source_name).write_bytes(b"print(9)\n")
    (tmp_path / _INVENTORY).write_text("{}\n", encoding="utf-8")
    result = restore_sealed_source_drift(tmp_path)
    assert _INVENTORY in result["restored"]
    assert source_name in result["restored"]
    assert (tmp_path / source_name).read_bytes() == b"print(1)\n"
    assert json.loads((tmp_path / _INVENTORY).read_text())["retained"][source_name]


def test_unsealed_inventory_drift_does_not_rewrite_children(tmp_path: Path) -> None:
    source_name = "sources/kept.py"
    _write_tree(tmp_path, source_name, b"print(1)\n")
    (tmp_path / _INVENTORY).write_text("{}\n", encoding="utf-8")
    (tmp_path / source_name).write_bytes(b"print(9)\n")
    result = restore_sealed_source_drift(tmp_path)
    assert result["reason"] == "inventory_cache_miss"
    assert result["restored"] == []
    assert (tmp_path / source_name).read_bytes() == b"print(9)\n"


def test_symlink_and_parent_escape_are_not_replaced(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"safe\n")
    source_name = "sources/kept.py"
    _write_tree(tmp_path, source_name, b"print(1)\n")
    restore_sealed_source_drift(tmp_path)
    target = tmp_path / source_name
    target.unlink()
    target.symlink_to(outside)
    inventory = json.loads((tmp_path / _INVENTORY).read_text())
    inventory["retained"]["../outside.py"] = hashlib.sha256(b"safe\n").hexdigest()
    (tmp_path / _INVENTORY).write_bytes(json.dumps(inventory).encode())
    # The inventory edit is itself drift. Restore the sealed inventory first.
    result = restore_sealed_source_drift(tmp_path)
    assert source_name not in result["restored"]
    assert outside.read_bytes() == b"safe\n"
    assert target.is_symlink()


def test_missing_profile_is_not_a_compiled_lane(tmp_path: Path) -> None:
    result = restore_sealed_source_drift(tmp_path)
    assert result["reason"] == "not_compiled_lane"
    assert result["restored"] == []
