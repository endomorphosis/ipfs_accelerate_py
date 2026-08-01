"""Regression tests for the GitHub cache's lazy CID provider."""

from __future__ import annotations

import builtins
import hashlib
import json
import sys
from types import ModuleType

from ipfs_accelerate_py.github_cli import cache as cache_module


def _reset_lazy_state(monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "CID", None)
    monkeypatch.setattr(cache_module, "multiformats_multihash", None)
    monkeypatch.setattr(cache_module, "HAVE_MULTIFORMATS", False)
    monkeypatch.setattr(
        cache_module,
        "_MULTIFORMATS_IMPORT_ATTEMPTED",
        False,
    )


def test_missing_multiformats_falls_back_once(monkeypatch) -> None:
    _reset_lazy_state(monkeypatch)
    monkeypatch.delitem(sys.modules, "multiformats", raising=False)
    attempts: list[str] = []
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "multiformats" or name.startswith("multiformats."):
            attempts.append(name)
            raise ModuleNotFoundError(
                "optional CID provider unavailable",
                name="multiformats",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    fields = {"updatedAt": "2026-08-01T00:00:00Z", "revision": 7}
    canonical = json.dumps(fields, sort_keys=True).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()

    assert cache_module.GitHubAPICache._compute_validation_hash(fields) == expected
    assert cache_module.GitHubAPICache._compute_validation_hash(fields) == expected
    assert attempts == ["multiformats"]
    assert cache_module.HAVE_MULTIFORMATS is False


def test_preloaded_multiformats_preserves_capability_contract(monkeypatch) -> None:
    _reset_lazy_state(monkeypatch)
    provider = ModuleType("multiformats")

    class FakeCID:
        pass

    fake_multihash = object()
    provider.CID = FakeCID
    provider.multihash = fake_multihash
    monkeypatch.setitem(sys.modules, "multiformats", provider)

    assert cache_module._ensure_multiformats() is True
    assert cache_module.HAVE_MULTIFORMATS is True
    assert cache_module.CID is FakeCID
    assert cache_module.multiformats_multihash is fake_multihash
