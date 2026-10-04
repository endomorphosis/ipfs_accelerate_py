from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.github_cli import cache as cache_module


def test_github_cache_global_p2p_defaults_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CACHE_ENABLE_P2P", raising=False)
    monkeypatch.setattr(cache_module, "_global_cache", None)

    cache = cache_module.get_global_cache(cache_dir=str(tmp_path), enable_persistence=False)

    assert cache.enable_p2p is False


def test_github_cache_explicit_p2p_requires_encryption(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache_module, "HAVE_LIBP2P", True)
    monkeypatch.setattr(cache_module, "HAVE_CRYPTO", True)
    monkeypatch.setattr(
        cache_module.GitHubAPICache,
        "_init_encryption",
        lambda self: (_ for _ in ()).throw(RuntimeError("missing shared secret")),
    )

    def _fail_if_started(self) -> bool:
        raise AssertionError("custom cache p2p must not start without encryption")

    monkeypatch.setattr(cache_module.GitHubAPICache, "_init_p2p", _fail_if_started)

    cache = cache_module.GitHubAPICache(
        cache_dir=str(tmp_path),
        enable_persistence=False,
        enable_p2p=True,
    )

    assert cache.enable_p2p is False


def test_github_cache_refuses_plaintext_p2p_messages(tmp_path) -> None:
    cache = cache_module.GitHubAPICache(
        cache_dir=str(tmp_path),
        enable_persistence=False,
        enable_p2p=False,
    )

    with pytest.raises(RuntimeError, match="refusing plaintext"):
        cache._encrypt_message({"key": "value"})
    assert cache._decrypt_message(b'{"key": "value"}') is None


def test_github_cache_uses_no_legacy_raw_stream_protocols() -> None:
    text = Path(cache_module.__file__).read_text(encoding="utf-8")

    forbidden = [
        "set_stream_handler",
        "new_stream(",
        "stream.read(",
        "stream.write(",
        "new_libp2p_host",
        "peerinfo_from_multiaddr",
    ]

    assert [pattern for pattern in forbidden if pattern in text] == []
