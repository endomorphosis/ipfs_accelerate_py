"""Tests for ASI-166 todo_daemon LLM child envelope and receipt propagation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_CHILD_ENVELOPE_SCHEMA,
    LLM_CHILD_ENVELOPE_VERSION,
    LLM_CHILD_RESULT_SCHEMA,
    LLM_USAGE_MODE_ENFORCE,
    LLM_USAGE_MODE_OFF,
    LlmChildRequestEnvelope,
    LlmChildResultEnvelope,
    LlmRouterInvocation,
    build_child_request_envelope,
    call_llm_router,
    call_llm_router_with_receipt,
    last_llm_child_result,
    parse_child_result_envelope,
)


def _config(tmp_path: Path, **overrides: Any) -> LlmRouterInvocation:
    fields = dict(
        repo_root=tmp_path,
        model_name="test-model",
        provider="test-provider",
        timeout_seconds=5,
        timeout_grace_seconds=1,
        max_new_tokens=16,
        python_executable=sys.executable,
        reject_effective_provider_name=None,
        required_effective_providers=(),
        usage_mode=LLM_USAGE_MODE_OFF,
    )
    fields.update(overrides)
    return LlmRouterInvocation(**fields)


def test_new_child_options_preserve_legacy_positional_constructors() -> None:
    invocation = LlmRouterInvocation(
        Path("."), "model", "provider", False, 17
    )
    assert invocation.timeout_seconds == 17
    assert invocation.allow_cross_provider_fallback is None

    envelope = LlmChildRequestEnvelope(
        LLM_CHILD_ENVELOPE_SCHEMA,
        LLM_CHILD_ENVELOPE_VERSION,
        LLM_USAGE_MODE_OFF,
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
    assert envelope.allow_cross_provider_fallback is False


def test_request_envelope_is_bounded_versioned_and_prompt_free() -> None:
    config = _config(
        Path("."),
        usage_mode=LLM_USAGE_MODE_ENFORCE,
        request_id="request:1",
        attempt=2,
        idempotency_key="idem:1",
        supervisor_receipt_id="receipt:supervisor:1",
        endpoint_receipt_id="receipt:endpoint:1",
        catalog_revision="catalog:1",
        usage_revision="usage:1",
        lease_id="lease:1",
        fence_id="fence:1",
        deadline_at="2026-07-28T00:00:00Z",
    )
    envelope = build_child_request_envelope(
        config,
        input_digest="abc123",
        result_file="/tmp/result.json",
    )
    payload = envelope.to_dict()
    assert payload["schema"] == LLM_CHILD_ENVELOPE_SCHEMA
    assert payload["contract_version"] == LLM_CHILD_ENVELOPE_VERSION
    assert payload["usage_mode"] == LLM_USAGE_MODE_ENFORCE
    assert payload["request_id"] == "request:1"
    assert payload["supervisor_receipt_id"] == "receipt:supervisor:1"
    assert payload["endpoint_receipt_id"] == "receipt:endpoint:1"
    assert "prompt" not in payload
    assert "messages" not in payload
    assert "output" not in payload
    # Round-trip.
    restored = LlmChildRequestEnvelope.from_dict(payload)
    assert restored.request_id == "request:1"
    assert restored.input_digest == "abc123"
    encoded = envelope.to_json()
    assert len(encoded.encode("utf-8")) < 16_384


def test_result_envelope_rejects_prompt_and_provider_payload_fields() -> None:
    result = LlmChildResultEnvelope(
        usage_mode=LLM_USAGE_MODE_ENFORCE,
        request_id="request:1",
        attempt=1,
        idempotency_key="idem:1",
        status="ok",
        supervisor_receipt_id="receipt:supervisor:1",
        endpoint_receipt_id="receipt:endpoint:1",
        execution_result_id="result:1",
        effective_provider="test-provider",
        text_chars=12,
        exit_code=0,
    )
    payload = result.to_dict()
    assert payload["schema"] == LLM_CHILD_RESULT_SCHEMA
    assert payload["supervisor_receipt_id"]
    assert payload["endpoint_receipt_id"]
    assert "prompt" not in payload
    assert "output" not in payload
    assert "completion" not in payload
    with pytest.raises(RuntimeError, match="forbidden field"):
        parse_child_result_envelope(
            {
                "schema": LLM_CHILD_RESULT_SCHEMA,
                "contract_version": LLM_CHILD_ENVELOPE_VERSION,
                "usage_mode": "off",
                "status": "ok",
                "prompt": "leak",
            }
        )


def test_parse_result_envelope_from_json() -> None:
    raw = json.dumps(
        {
            "schema": LLM_CHILD_RESULT_SCHEMA,
            "contract_version": 1,
            "usage_mode": "enforce",
            "request_id": "request:9",
            "attempt": 3,
            "idempotency_key": "idem:9",
            "status": "ok",
            "reason_codes": ["settled"],
            "supervisor_receipt_id": "sup:1",
            "endpoint_receipt_id": "end:1",
            "execution_result_id": "exec:1",
            "effective_provider": "grok",
            "text_chars": 4,
            "exit_code": 0,
        },
        sort_keys=True,
    )
    parsed = parse_child_result_envelope(raw)
    assert parsed.request_id == "request:9"
    assert parsed.supervisor_receipt_id == "sup:1"
    assert "settled" in parsed.reason_codes


def test_off_mode_call_llm_router_returns_stdout_text(monkeypatch, tmp_path: Path) -> None:
    """Off mode remains behaviorally compatible: text on stdout, optional metadata."""

    def fake_popen(command, **kwargs):
        class Proc:
            def __init__(self) -> None:
                self.returncode = 0
                self.pid = 12345

            def communicate(self, timeout=None):
                # Write result envelope if RESULT_FILE is set (safe metadata only).
                env = kwargs.get("env") or {}
                result_file = env.get("TODO_DAEMON_LLM_RESULT_FILE") or ""
                if result_file:
                    Path(result_file).write_text(
                        json.dumps(
                            {
                                "schema": LLM_CHILD_RESULT_SCHEMA,
                                "contract_version": 1,
                                "usage_mode": "off",
                                "request_id": "",
                                "attempt": 1,
                                "idempotency_key": "",
                                "status": "ok",
                                "reason_codes": [],
                                "supervisor_receipt_id": "",
                                "endpoint_receipt_id": "",
                                "execution_result_id": "",
                                "effective_provider": "test",
                                "text_chars": 11,
                                "exit_code": 0,
                            }
                        ),
                        encoding="utf-8",
                    )
                return ("hello-world", "")

            def poll(self):
                return self.returncode

        return Proc()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.subprocess.Popen",
        fake_popen,
    )
    config = _config(tmp_path, usage_mode=LLM_USAGE_MODE_OFF)
    text = call_llm_router("prompt body must not leak into envelope", config)
    assert text == "hello-world"


def test_enforce_mode_propagates_receipt_ids_without_prompt_leakage(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        env = kwargs.get("env") or {}
        captured["env"] = dict(env)
        # Ensure child code does not embed the raw prompt into the command string.
        captured["command"] = list(command)

        class Proc:
            def __init__(self) -> None:
                self.returncode = 0
                self.pid = 99

            def communicate(self, timeout=None):
                result_file = env.get("TODO_DAEMON_LLM_RESULT_FILE") or ""
                assert result_file
                Path(result_file).write_text(
                    json.dumps(
                        {
                            "schema": LLM_CHILD_RESULT_SCHEMA,
                            "contract_version": 1,
                            "usage_mode": "enforce",
                            "request_id": "request:77",
                            "attempt": 2,
                            "idempotency_key": "idem:77",
                            "status": "ok",
                            "reason_codes": [],
                            "supervisor_receipt_id": "sup:77",
                            "endpoint_receipt_id": "end:77",
                            "execution_result_id": "",
                            "effective_provider": "remote",
                            "text_chars": 5,
                            "exit_code": 0,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                return ("hello", "")

            def poll(self):
                return self.returncode

        return Proc()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.subprocess.Popen",
        fake_popen,
    )
    secret_prompt = "SECRET_PROMPT_BODY_MUST_NOT_APPEAR_IN_ENVELOPE"
    config = _config(
        tmp_path,
        usage_mode=LLM_USAGE_MODE_ENFORCE,
        request_id="request:77",
        attempt=2,
        idempotency_key="idem:77",
        supervisor_receipt_id="sup:77",
        endpoint_receipt_id="end:77",
    )
    text, result = call_llm_router_with_receipt(secret_prompt, config)
    assert text == "hello"
    assert result is not None
    assert result.supervisor_receipt_id == "sup:77"
    assert result.endpoint_receipt_id == "end:77"
    assert result.request_id == "request:77"
    assert secret_prompt not in json.dumps(result.to_dict())
    # Env carries receipt IDs, not the prompt body.
    env = captured["env"]
    assert env["TODO_DAEMON_LLM_SUPERVISOR_RECEIPT_ID"] == "sup:77"
    assert env["TODO_DAEMON_LLM_ENDPOINT_RECEIPT_ID"] == "end:77"
    assert secret_prompt not in json.dumps(env)
    # Child code string must not embed the prompt body.
    assert secret_prompt not in " ".join(str(part) for part in captured["command"])
    # last_llm_child_result tracks the same envelope.
    last = last_llm_child_result()
    assert last is not None
    assert last.supervisor_receipt_id == "sup:77"


def test_child_pins_canonical_accelerator_router_ahead_of_hostile_editable(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    hostile_root = tmp_path / "hostile-editable"
    hostile_root.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(hostile_root))

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["child_code"] = Path(command[1]).read_text(encoding="utf-8")

        class Proc:
            returncode = 0
            pid = 991

            def communicate(self, timeout=None):
                return ("ok", "")

            def poll(self):
                return self.returncode

        return Proc()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.subprocess.Popen",
        fake_popen,
    )
    assert call_llm_router("prompt", _config(tmp_path)) == "ok"

    source_root = str(Path(__file__).resolve().parents[2])
    pythonpath = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath == [source_root]
    assert len(captured["command"]) == 2
    assert Path(captured["command"][1]).name.startswith("todo-daemon-llm-child-")
    child_code = str(captured["child_code"])
    assert "sys.path[:] = [_canonical_source_root]" in child_code
    assert "from ipfs_accelerate_py import llm_router" in child_code
    assert "from ipfs_datasets_py import llm_router" not in child_code


def test_child_failure_surfaces_without_leaking_prompt(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_popen(command, **kwargs):
        class Proc:
            def __init__(self) -> None:
                self.returncode = 2
                self.pid = 7

            def communicate(self, timeout=None):
                return ("", "provider verification failed")

            def poll(self):
                return self.returncode

        return Proc()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.subprocess.Popen",
        fake_popen,
    )
    config = _config(
        tmp_path,
        usage_mode=LLM_USAGE_MODE_ENFORCE,
        request_id="request:err",
        supervisor_receipt_id="sup:err",
    )
    with pytest.raises(RuntimeError, match="exited with code 2"):
        call_llm_router("prompt-that-must-not-leak", config)


def test_unsupported_usage_mode_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path, usage_mode="unlimited")
    with pytest.raises(RuntimeError, match="unsupported LLM usage_mode"):
        call_llm_router("x", config)


def test_timeout_path_still_terminates_child(monkeypatch, tmp_path: Path) -> None:
    terminated = {"called": False}

    def fake_popen(command, **kwargs):
        class Proc:
            def __init__(self) -> None:
                self.returncode = None
                self.pid = 4242

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

            def poll(self):
                return self.returncode

        return Proc()

    def fake_terminate(process, *, grace_seconds=5.0):
        terminated["called"] = True
        process.returncode = -9

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.terminate_process_group",
        fake_terminate,
    )
    config = _config(tmp_path, timeout_seconds=1, timeout_grace_seconds=0)
    with pytest.raises(RuntimeError, match="timed out"):
        call_llm_router("prompt", config)
    assert terminated["called"] is True
