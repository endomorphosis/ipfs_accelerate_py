"""Regression tests for TaskQueue P2P transport policy."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.p2p_tasks.protocol import (
    legacy_task_protocol_enabled,
    task_p2p_protocol_order,
)


ENV_NAMES = (
    "IPFS_ACCELERATE_PY_TASK_P2P_PROTOCOL",
    "IPFS_DATASETS_PY_TASK_P2P_PROTOCOL",
    "IPFS_ACCELERATE_PY_TASK_P2P_ALLOW_LEGACY_FALLBACK",
    "IPFS_DATASETS_PY_TASK_P2P_ALLOW_LEGACY_FALLBACK",
    "IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_LEGACY_PROTOCOL",
    "IPFS_DATASETS_PY_TASK_P2P_ENABLE_LEGACY_PROTOCOL",
)


def _clear_env(monkeypatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_taskqueue_p2p_defaults_to_mcp_only(monkeypatch) -> None:
    _clear_env(monkeypatch)

    assert task_p2p_protocol_order() == ["mcp"]
    assert legacy_task_protocol_enabled() is False


def test_taskqueue_p2p_auto_enables_explicit_legacy_fallback(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_PROTOCOL", "auto")

    assert task_p2p_protocol_order() == ["mcp", "legacy"]
    assert legacy_task_protocol_enabled() is True


def test_taskqueue_p2p_legacy_registration_requires_operator_intent(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_LEGACY_PROTOCOL", "1")

    assert legacy_task_protocol_enabled() is True


def test_taskqueue_client_has_no_direct_legacy_stream_dials() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    client_path = repo_root / "ipfs_accelerate_py" / "p2p_tasks" / "client.py"
    text = client_path.read_text(encoding="utf-8")

    assert "new_stream(peer_info.peer_id, [PROTOCOL_V1])" not in text
