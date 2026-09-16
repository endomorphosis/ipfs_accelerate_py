"""Offline tests for Muse Code CLI endpoint registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from ipfs_accelerate_py.cli_runtime.endpoints import (
    create_cli_endpoint,
    list_cli_endpoint_tools,
    reset_default_endpoint_registry,
)
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import MuseCodeAdapter


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_default_endpoint_registry()
    yield
    reset_default_endpoint_registry()


def test_muse_aliases_register() -> None:
    tools = list_cli_endpoint_tools()
    names = {t["name"] for t in tools}
    assert "muse" in names
    adapter = create_cli_endpoint("muse_code", "m_alias")
    assert isinstance(adapter, MuseCodeAdapter)
    adapter2 = create_cli_endpoint("muse", "m_main")
    assert isinstance(adapter2, MuseCodeAdapter)
    adapter3 = create_cli_endpoint("muse-cli", "m_cli")
    assert isinstance(adapter3, MuseCodeAdapter)


def test_register_muse_lazy_no_model_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_calls: List[Any] = []

    def tracking_run(*a: Any, **k: Any) -> Any:
        run_calls.append((a, k))
        raise AssertionError("registration must not start muse")

    monkeypatch.setattr("subprocess.run", tracking_run)
    exe = tmp_path / "muse"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    adapter = create_cli_endpoint(
        "muse",
        "muse-ep",
        cli_path=str(exe),
        config={"model": "muse-spark-1.2"},
    )
    assert isinstance(adapter, MuseCodeAdapter)
    assert adapter.is_available() is True
    assert run_calls == []


def test_muse_endpoint_returns_collected_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.cli_runtime.contracts import CLIResult, ExecutionMode
    from ipfs_accelerate_py.cli_runtime.providers.muse import MuseCLIProvider

    captured: dict[str, Any] = {}

    def fake_generate_result(self, request, **kwargs):
        captured["tmpdir"] = (kwargs, getattr(request, "workspace", None))
        return CLIResult(
            text="hello world",
            ok=True,
            mode=ExecutionMode.AGENT,
            side_effecting=True,
            cacheable=False,
            retryable=False,
            had_side_effect_event=True,
            provider_name="muse_code",
            model_name="muse-spark-1.3-contributor",
            metadata={
                "session_id": "sess-ep",
                "run_id": "run-ep",
                "model_id": "muse-spark-1.3-contributor",
                "request_id": "req-ep",
                "response_id": "resp-ep",
                "time_to_first_event_ms": "45",
                "prompt": "should-not-leak",
            },
        )

    monkeypatch.setattr(MuseCLIProvider, "generate_result", fake_generate_result)
    exe = tmp_path / "muse"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    adapter = MuseCodeAdapter("muse-meta", cli_path=str(exe), config={"model": "muse-spark-1.2"})
    payload = adapter.execute("hello world", timeout=30, workspace=str(tmp_path))
    assert payload["success"] is True
    assert payload["text"] == "hello world"
    assert payload["session_id"] == "sess-ep"
    assert payload["request_id"] == "req-ep"
    assert payload["metadata"]["run_id"] == "run-ep"
    assert payload["metadata"]["time_to_first_event_ms"] == "45"
    assert "prompt" not in payload["metadata"]
    assert "raw_response" not in payload
    assert payload["model"] == "muse-spark-1.3-contributor"


def test_muse_adapter_install_docs_use_official_script() -> None:
    adapter = MuseCodeAdapter("docs")
    payload = adapter._install()
    commands = " ".join(
        cmd
        for method in payload["install_methods"]
        for cmd in method.get("commands", [])
    )
    assert "dev.meta.ai/install.sh" in commands
    assert payload["verify_command"] == "muse --version"
