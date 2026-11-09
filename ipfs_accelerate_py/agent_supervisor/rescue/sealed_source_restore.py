"""Put published compiled-source bytes back when a retained file drifts.

The lean-refactor lane validates a closed inventory. A drifted retained file
makes every probe fail before the real gate is visible. This module copies
the published bytes back from a content-addressed cache under the arena data
directory. It does not edit the task board, mark a task complete, or adopt a
new inventory that the execution profile has not sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

_PROFILE = Path("config/lean_refactor_arena_20261024_execution_v2.json")
_INVENTORY = Path("tools/lean_refactor_compiled_sources_v1.json")
_CACHE = Path("data/lean_refactor_arena_20261024/execution-v2/sealed-source-cache")
_HOLD_NAMES = ("HOLD", "OPERATOR_STOP", "watchdog.disabled", "hold", "operator.hold")
_SHA = re.compile(r"[0-9a-f]{64}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _held(repo_root: Path) -> bool:
    candidates = [repo_root / name for name in _HOLD_NAMES]
    lane = repo_root / "data/lean_refactor_arena_20261024/lane"
    candidates.extend(lane / name for name in _HOLD_NAMES)
    return any(path.exists() for path in candidates)


def _read_bytes(path: Path) -> bytes | None:
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _target(repo_root: Path, name: str) -> Path | None:
    path = Path(name)
    if ".." in path.parts or not name or name.endswith("/"):
        return None
    if path.is_absolute():
        return path
    return repo_root / path


def _cache_blob(cache: Path, digest: str) -> Path | None:
    blob = cache / digest
    data = _read_bytes(blob)
    if data is None or _sha256(data) != digest:
        return None
    return blob


def _store(cache: Path, data: bytes) -> None:
    digest = _sha256(data)
    blob = cache / digest
    existing = _read_bytes(blob)
    if existing is not None and _sha256(existing) == digest:
        return
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / (digest + ".partial")
    temporary.write_bytes(data)
    os.replace(temporary, blob)


def _restore(path: Path, data: bytes) -> bool:
    if path.is_symlink():
        return False
    parent = path.parent
    try:
        parent_info = parent.lstat()
    except OSError:
        return False
    if not stat.S_ISDIR(parent_info.st_mode) or parent.is_symlink():
        return False
    temporary = path.with_name(path.name + ".sealrestore")
    temporary.write_bytes(data)
    if path.exists() and not path.is_symlink():
        os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)
    return _read_bytes(path) == data


def restore_sealed_source_drift(repo_root: Path | str) -> dict[str, Any]:
    """Restore retained files that differ from the sealed inventory.

    A matching tree only fills the cache. A drifted inventory file is restored
    only from the profile hash. Child files are restored only after that
    inventory matches the profile. Completion authority stays false.
    """

    root = Path(repo_root)
    result: dict[str, Any] = {
        "restored": [],
        "misses": [],
        "reason": "",
        "warmed": False,
        "work_complete": False,
        "completion_authority": False,
    }
    if _held(root):
        result["reason"] = "operator_hold"
        return result
    profile_path = root / _PROFILE
    inventory_path = root / _INVENTORY
    profile_bytes = _read_bytes(profile_path)
    if profile_bytes is None or _read_bytes(inventory_path) is None:
        result["reason"] = "not_compiled_lane"
        return result
    try:
        profile = json.loads(profile_bytes)
        expected_inventory = profile["source_sha256"][str(_INVENTORY)]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        result["reason"] = "inventory_seal_unreadable"
        return result
    if not isinstance(expected_inventory, str) or _SHA.fullmatch(expected_inventory) is None:
        result["reason"] = "inventory_seal_unreadable"
        return result
    cache = root / _CACHE
    inventory_bytes = _read_bytes(inventory_path)
    if inventory_bytes is None:
        result["reason"] = "inventory_unreadable"
        return result
    if _sha256(inventory_bytes) != expected_inventory:
        blob = _cache_blob(cache, expected_inventory)
        if blob is None:
            result["reason"] = "inventory_cache_miss"
            return result
        if not _restore(inventory_path, blob.read_bytes()):
            result["reason"] = "inventory_restore_failed"
            return result
        result["restored"].append(str(_INVENTORY))
        inventory_bytes = _read_bytes(inventory_path)
        if inventory_bytes is None or _sha256(inventory_bytes) != expected_inventory:
            result["reason"] = "inventory_restore_failed"
            return result
    try:
        inventory = json.loads(inventory_bytes)
        retained = inventory["retained"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        result["reason"] = "inventory_unreadable"
        return result
    if not isinstance(retained, dict):
        result["reason"] = "inventory_unreadable"
        return result
    _store(cache, inventory_bytes)
    for name, expected in retained.items():
        if not isinstance(name, str) or not isinstance(expected, str) or _SHA.fullmatch(expected) is None:
            result["misses"].append(str(name))
            continue
        target = _target(root, name)
        if target is None:
            result["misses"].append(name)
            continue
        current = _read_bytes(target)
        if current is not None and _sha256(current) == expected:
            _store(cache, current)
            continue
        blob = _cache_blob(cache, expected)
        if blob is None or not _restore(target, blob.read_bytes()):
            result["misses"].append(name)
            continue
        result["restored"].append(name)
    result["warmed"] = not result["misses"]
    if result["misses"] and not result["reason"]:
        result["reason"] = "retained_cache_miss"
    return result
