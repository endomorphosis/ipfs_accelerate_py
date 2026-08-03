"""Exact-provider routing regressions for supervisor LLM children."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon import llm as todo_llm


class _FailingProvider:
    def __init__(self, calls: list[tuple[str, str | None]]) -> None:
        self._calls = calls

    def generate(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._calls.append(("primary", model_name))
        raise llm_router.LLMRouterError("primary provider failed")


class _AlternateProvider:
    def __init__(self, calls: list[tuple[str, str | None]]) -> None:
        self._calls = calls

    def generate(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._calls.append(("alternate", model_name))
        return "alternate-result"


def _install_fallback(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str | None]],
) -> None:
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: False)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("grok_cli", alternate)],
    )
    monkeypatch.setattr(llm_router, "_get_accelerate_provider", lambda _deps: None)


def test_exact_provider_disables_model_and_cross_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    _install_fallback(monkeypatch, calls)

    with pytest.raises(llm_router.LLMRouterError, match="primary provider failed"):
        llm_router.generate_text(
            "audit this patch",
            provider="codex_cli",
            provider_instance=_FailingProvider(calls),
            model_name="gpt-5.6-sol",
            allow_local_fallback=False,
            allow_cross_provider_fallback=False,
        )

    assert calls == [("primary", "gpt-5.6-sol")]


def test_codex_usage_limit_is_typed_with_canonical_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_message = (
        "You've hit your usage limit. Try again at Aug 10th, 2026 5:23 AM."
    )
    assert llm_router._extract_codex_next_eligible_at(
        stdout="",
        stderr=reset_message,
        now=datetime(2026, 8, 3, tzinfo=timezone.utc),
        local_timezone=timezone.utc,
    ) == "2026-08-10T05:23:00Z"

    monkeypatch.setattr(llm_router.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=reset_message,
        ),
    )
    provider = llm_router._get_codex_cli_provider()
    assert provider is not None
    with pytest.raises(llm_router.UsageCapacityError) as raised:
        provider.generate("bounded prompt", model_name="gpt-5.6-sol")

    assert raised.value.reason_codes == (
        "usage_limit",
        "capacity_unavailable",
    )
    assert raised.value.next_eligible_at
    assert raised.value.pre_dispatch is False


def test_legacy_remote_fallback_remains_default_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    _install_fallback(monkeypatch, calls)

    result = llm_router.generate_text(
        "ordinary failover",
        provider="codex_cli",
        provider_instance=_FailingProvider(calls),
        model_name="fixture-model",
        allow_local_fallback=False,
    )

    assert result == "alternate-result"
    assert calls == [
        ("primary", "fixture-model"),
        ("primary", None),
        ("alternate", "fixture-model"),
    ]


def test_new_child_fields_preserve_v1_and_positional_compatibility() -> None:
    invocation = todo_llm.LlmRouterInvocation(
        Path("."), "model", "provider", False, 17
    )
    assert invocation.timeout_seconds == 17
    assert invocation.allow_cross_provider_fallback is None

    envelope = todo_llm.LlmChildRequestEnvelope(
        todo_llm.LLM_CHILD_ENVELOPE_SCHEMA,
        todo_llm.LLM_CHILD_ENVELOPE_VERSION,
        todo_llm.LLM_USAGE_MODE_OFF,
        "request",
        1,
        "idempotency",
        "model",
        "provider",
        17,
        18,
        0.0,
        False,
        "catalog-revision",
    )
    assert envelope.catalog_revision == "catalog-revision"
    assert envelope.allow_cross_provider_fallback is True

    legacy_payload = envelope.to_dict()
    legacy_payload.pop("allow_cross_provider_fallback")
    restored = todo_llm.LlmChildRequestEnvelope.from_dict(legacy_payload)
    assert restored.allow_cross_provider_fallback is True

    result = todo_llm.LlmChildResultEnvelope(
        todo_llm.LLM_CHILD_RESULT_SCHEMA,
        todo_llm.LLM_CHILD_ENVELOPE_VERSION,
        todo_llm.LLM_USAGE_MODE_OFF,
        "request",
        1,
        "idempotency",
        "ok",
        (),
        "supervisor",
        "endpoint",
        "execution",
        "provider",
        2,
        2,
        "digest",
        0,
    )
    assert result.next_eligible_at == ""
    legacy_result_payload = result.to_dict()
    legacy_result_payload.pop("next_eligible_at")
    restored_result = todo_llm.LlmChildResultEnvelope.from_dict(
        legacy_result_payload
    )
    assert restored_result.next_eligible_at == ""


def test_exact_provider_cannot_consume_cross_provider_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    deps = llm_router.RouterDeps()
    alternate = _AlternateProvider(calls)
    monkeypatch.setattr(llm_router, "_response_cache_enabled", lambda: True)
    monkeypatch.setattr(
        llm_router,
        "_iter_unpinned_optional_providers",
        lambda: [("grok_cli", alternate)],
    )
    monkeypatch.setattr(llm_router, "_get_accelerate_provider", lambda _deps: None)

    assert (
        llm_router.generate_text(
            "same request",
            provider="codex_cli",
            provider_instance=_FailingProvider(calls),
            model_name="fixture-model",
            deps=deps,
            allow_local_fallback=False,
            allow_cross_provider_fallback=True,
        )
        == "alternate-result"
    )
    with pytest.raises(llm_router.LLMRouterError, match="primary provider failed"):
        llm_router.generate_text(
            "same request",
            provider="codex_cli",
            provider_instance=_FailingProvider(calls),
            model_name="fixture-model",
            deps=deps,
            allow_local_fallback=False,
            allow_cross_provider_fallback=False,
        )

    assert calls[-1] == ("primary", "fixture-model")


def test_supervisor_child_pins_source_and_fallback_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    hostile_root = tmp_path / "hostile-editable"
    hostile_root.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(hostile_root))

    def fake_popen(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = list(command)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["source"] = Path(command[1]).read_text(encoding="utf-8")

        class _Process:
            returncode = 0
            pid = 9191

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return ("ok", "")

            def poll(self) -> int:
                return self.returncode

        return _Process()

    monkeypatch.setattr(todo_llm.subprocess, "Popen", fake_popen)
    config = todo_llm.LlmRouterInvocation(
        repo_root=tmp_path,
        provider="codex_cli",
        model_name="fixture-model",
        allow_local_fallback=False,
        allow_cross_provider_fallback=False,
        timeout_seconds=1,
        timeout_grace_seconds=0,
        python_executable=sys.executable,
        reject_effective_provider_name=None,
    )

    assert todo_llm.call_llm_router("exact review", config) == "ok"
    env = captured["env"]
    assert env["TODO_DAEMON_LLM_ALLOW_CROSS_PROVIDER_FALLBACK"] == "0"
    assert env["PYTHONPATH"] == str(Path(llm_router.__file__).resolve().parents[1])
    source = str(captured["source"])
    assert "from ipfs_accelerate_py import llm_router" in source
    assert "from ipfs_datasets_py import llm_router" not in source
    assert "allow_cross_provider_fallback=" in source
    assert len(captured["command"]) == 2
    assert not Path(captured["command"][1]).exists()
    assert os.path.basename(captured["command"][1]).startswith(
        "todo-daemon-llm-child-"
    )


def test_isolated_exact_codex_failure_never_launches_grok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_calls = tmp_path / "codex-calls.txt"
    grok_calls = tmp_path / "grok-calls.txt"
    fake_codex = tmp_path / "codex"
    fake_grok = tmp_path / "grok"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(codex_calls)!r}).open('a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.stderr.write('fixture codex failure\\n')\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    fake_grok.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib\n"
        f"pathlib.Path({str(grok_calls)!r}).write_text('called', encoding='utf-8')\n"
        "print('{\"text\":\"wrong-provider\",\"stopReason\":\"EndTurn\"}')\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o700)
    fake_grok.chmod(0o700)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path), os.environ["PATH"])))
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_CACHE", "0")
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_RESPONSE_CACHE", "0")

    config = todo_llm.LlmRouterInvocation(
        repo_root=tmp_path,
        provider="codex_cli",
        model_name="gpt-5.6-sol",
        allow_local_fallback=False,
        allow_cross_provider_fallback=False,
        timeout_seconds=10,
        timeout_grace_seconds=1,
        python_executable=sys.executable,
        required_effective_providers=("codex_cli",),
    )

    with pytest.raises(RuntimeError, match="fixture codex failure"):
        todo_llm.call_llm_router("exact review", config)

    assert len(codex_calls.read_text(encoding="utf-8").splitlines()) == 1
    assert "-m gpt-5.6-sol" in codex_calls.read_text(encoding="utf-8")
    assert not grok_calls.exists()


def test_isolated_codex_capacity_failure_is_typed_and_never_launches_grok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_calls = tmp_path / "codex-capacity-calls.txt"
    grok_calls = tmp_path / "grok-capacity-calls.txt"
    fake_codex = tmp_path / "codex"
    fake_grok = tmp_path / "grok"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(codex_calls)!r}).open('a').write('called\\n')\n"
        "sys.stderr.write(\"You've hit your usage limit. "
        "Try again at Aug 10th, 2026 5:23 AM.\\n\")\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    fake_grok.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib\n"
        f"pathlib.Path({str(grok_calls)!r}).write_text('called', encoding='utf-8')\n"
        "print('{\"text\":\"wrong-provider\",\"stopReason\":\"EndTurn\"}')\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o700)
    fake_grok.chmod(0o700)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path), os.environ["PATH"])))
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_CACHE", "0")
    monkeypatch.setenv("IPFS_DATASETS_PY_ROUTER_RESPONSE_CACHE", "0")

    config = todo_llm.LlmRouterInvocation(
        repo_root=tmp_path,
        provider="codex_cli",
        model_name="gpt-5.6-sol",
        allow_local_fallback=False,
        allow_cross_provider_fallback=False,
        timeout_seconds=10,
        timeout_grace_seconds=1,
        python_executable=sys.executable,
        required_effective_providers=("codex_cli",),
    )

    with pytest.raises(todo_llm.LlmChildProviderCapacityError) as raised:
        todo_llm.call_llm_router("exact capacity review", config)

    assert raised.value.provider_id == "codex_cli"
    assert raised.value.reason_codes == (
        "usage_limit",
        "capacity_unavailable",
    )
    assert raised.value.next_eligible_at == "2026-08-10T05:23:00Z"
    result = todo_llm.last_llm_child_result()
    assert result is not None
    assert result.status == "error"
    assert result.next_eligible_at == "2026-08-10T05:23:00Z"
    assert codex_calls.read_text(encoding="utf-8").splitlines() == ["called"]
    assert not grok_calls.exists()
