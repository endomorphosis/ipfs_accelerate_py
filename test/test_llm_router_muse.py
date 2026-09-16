"""Offline tests for the Muse Code LLM router provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ipfs_accelerate_py import llm_router


def _clear_muse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os_env_names()):
        monkeypatch.delenv(name, raising=False)


def os_env_names() -> tuple[str, ...]:
    return (
        "IPFS_ACCELERATE_MUSE_PATH",
        "IPFS_ACCELERATE_PY_MUSE_PATH",
        "ipfs_accelerate_py_MUSE_BIN",
        "MUSE_BIN",
        "MUSE_CLI_PATH",
        "IPFS_ACCELERATE_MUSE_DISCOVERY",
        "IPFS_ACCELERATE_PY_MUSE_DISCOVERY",
        "ipfs_accelerate_py_MUSE_DISCOVERY",
        "IPFS_ACCELERATE_MUSE_AUTO_INSTALL",
        "ipfs_accelerate_py_LLM_PROVIDER",
    )


def _install_result(*, available: bool, executable: str = "", reason: str = "") -> MagicMock:
    result = MagicMock()
    result.available = available
    result.executable = executable
    result.version = "Muse Code 0.1.0" if available else ""
    result.reason = reason
    return result


def test_aliases_canonicalize_to_muse_code() -> None:
    for alias in ("muse", "muse-code", "musecode", "muse_cli", "muse-cli", "muse_code"):
        assert llm_router._canonicalize_provider(alias) == "muse_code"
        assert llm_router._is_muse_code_provider_name(alias)


def test_find_muse_cli_prefers_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_muse_env(monkeypatch)
    binary = tmp_path / "muse"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("IPFS_ACCELERATE_MUSE_PATH", str(binary))
    assert llm_router.find_muse_cli() == str(binary)


def test_implicit_discovery_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_muse_env(monkeypatch)
    monkeypatch.setattr(llm_router, "find_muse_cli", lambda: None)
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse.discover_muse",
        lambda **_k: _install_result(available=False, reason="not_installed"),
    )
    assert llm_router._builtin_provider_by_name("muse_code") is None


def test_implicit_discovery_does_not_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_muse_env(monkeypatch)
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse.ensure_muse",
        lambda **_k: pytest.fail("implicit path must not install"),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse.discover_muse",
        lambda **_k: _install_result(available=False, reason="not_installed"),
    )
    monkeypatch.setattr(llm_router, "find_muse_cli", lambda: None)
    assert llm_router._builtin_provider_by_name("muse_code") is None


def test_muse_not_in_unpinned_order_without_discovery_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_muse_env(monkeypatch)
    monkeypatch.setattr(llm_router, "find_muse_cli", lambda: "/fake/muse")
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse.discover_muse",
        lambda **_k: _install_result(available=True, executable="/fake/muse"),
    )
    names = [name for name, _ in llm_router._iter_unpinned_optional_providers()]
    assert "muse_code" not in names
    monkeypatch.setenv("IPFS_ACCELERATE_MUSE_DISCOVERY", "1")
    names_on = [name for name, _ in llm_router._iter_unpinned_optional_providers()]
    assert "muse_code" in names_on


def test_explicit_provider_generate_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_muse_env(monkeypatch)
    adapter = MagicMock()
    adapter.generate.return_value = "muse says hi"
    adapter.version = ""

    def _create(**_kwargs):
        return adapter

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse.discover_muse",
        lambda **_k: _install_result(available=True, executable="/fake/muse"),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.providers.muse.create_muse_provider",
        _create,
    )
    provider = llm_router._get_muse_code_provider(auto_install=False)
    assert provider is not None
    text = provider.generate("hello", model_name="muse-spark-1.2")
    assert text == "muse says hi"
    adapter.generate.assert_called_once()
    _, kwargs = adapter.generate.call_args
    assert kwargs.get("model_name") == "muse-spark-1.2"
    assert kwargs.get("side_effecting") is True


def test_muse_generate_text_is_treated_as_side_effecting() -> None:
    assert llm_router._is_muse_code_provider_name("muse")
    # Bounded generate_text still counts as side-effecting so cache/retry stay off.
    assert llm_router._kwargs_are_side_effecting({"agent": True})


def test_operator_docs_cover_muse_code() -> None:
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "LLM_ROUTER.md").read_text(encoding="utf-8")
    assert "`muse_code`" in doc
    for marker in (
        "IPFS_ACCELERATE_MUSE_DISCOVERY",
        "IPFS_ACCELERATE_MUSE_AUTO_INSTALL",
        "IPFS_ACCELERATE_MUSE_PATH",
        "META_API_KEY",
        "muse exec",
        "dev.meta.ai/install.sh",
        "muse-spark-1.2",
        "--disable-approval",
        "--yolo",
    ):
        assert marker in doc, f"missing documentation: {marker}"
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Muse Code" in readme
    quick = (root / "docs" / "guides" / "QUICKSTART.md").read_text(encoding="utf-8")
    assert "muse_code" in quick
