"""Exact Grok Build quota signal must promote to verified router exception."""

from __future__ import annotations

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ProviderRole,
    VerifiedGrokQuotaExhaustion,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_landed_provider_cli import (
    GROK_BUILD_BALANCE_EXHAUSTED_MARKER,
    NativeGrokQuotaExhaustionSignal,
    _native_cli_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    BoundProductionCLIProvider,
)

_GROK_BALANCE_MESSAGE = (
    "API error (status 402 Payment Required): "
    "Grok Build usage balance exhausted"
)


def _json_mode_error(
    *,
    message: str = _GROK_BALANCE_MESSAGE,
    http_status: object = 402,
    inner_extra: dict[str, object] | None = None,
    outer_extra: dict[str, object] | None = None,
) -> bytes:
    inner = {
        "message": message,
        "http_status": http_status,
        **(inner_extra or {}),
    }
    payload = {
        "type": "error",
        "message": "Internal error: " + json.dumps(inner, indent=2),
        **(outer_extra or {}),
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def test_native_cli_failure_classifies_exact_streaming_json_402() -> None:
    event = (
        b'{"type":"error","message":"API error (status 402 Payment Required): '
        b'Grok Build usage balance exhausted"}\n'
    )
    err = _native_cli_failure(
        ["grok", "--output-format", "streaming-json"],
        return_code=1,
        stdout=bytearray(event),
        stderr=bytearray(),
    )
    assert isinstance(err, NativeGrokQuotaExhaustionSignal)
    assert str(err) == GROK_BUILD_BALANCE_EXHAUSTED_MARKER


def test_native_cli_failure_classifies_exact_json_mode_402_envelope() -> None:
    err = _native_cli_failure(
        ["grok", "--output-format", "json"],
        return_code=1,
        stdout=bytearray(_json_mode_error()),
        stderr=bytearray(b"untrusted pretty diagnostic"),
    )

    assert isinstance(err, NativeGrokQuotaExhaustionSignal)
    assert str(err) == GROK_BUILD_BALANCE_EXHAUSTED_MARKER


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param(
            _json_mode_error(outer_extra={"provider": "grok"}),
            id="outer-extra-field",
        ),
        pytest.param(
            _json_mode_error(inner_extra={"retryable": False}),
            id="inner-extra-field",
        ),
        pytest.param(_json_mode_error(http_status=True), id="boolean-status"),
        pytest.param(_json_mode_error(http_status=403), id="wrong-status"),
        pytest.param(
            _json_mode_error(message=_GROK_BALANCE_MESSAGE + " account=secret"),
            id="message-suffix",
        ),
        pytest.param(
            json.dumps(
                {
                    "type": "error",
                    "message": "prefix Internal error: "
                    + json.dumps(
                        {
                            "message": _GROK_BALANCE_MESSAGE,
                            "http_status": 402,
                        }
                    ),
                }
            ).encode("utf-8"),
            id="wrapper-prefix",
        ),
    ],
)
def test_json_mode_quota_classifier_rejects_non_exact_envelopes(
    stdout: bytes,
) -> None:
    err = _native_cli_failure(
        ["grok", "--output-format", "json"],
        return_code=1,
        stdout=bytearray(stdout),
        stderr=bytearray(),
    )

    assert type(err) is RuntimeError
    assert str(err) == "legacy native provider command failed"


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param(_json_mode_error() + b"{}\n", id="trailing-json"),
        pytest.param(b"{}\n" + _json_mode_error(), id="preceding-json"),
        pytest.param(_json_mode_error() * 2, id="duplicate-quota-json"),
    ],
)
def test_json_mode_quota_classifier_requires_one_stdout_object(
    stdout: bytes,
) -> None:
    err = _native_cli_failure(
        ["grok", "--output-format", "json"],
        return_code=1,
        stdout=bytearray(stdout),
        stderr=bytearray(),
    )

    assert type(err) is RuntimeError
    assert str(err) == "legacy native provider command failed"


def test_streaming_quota_classifier_rejects_direct_shape_extra_fields() -> None:
    stdout = json.dumps(
        {
            "type": "error",
            "message": _GROK_BALANCE_MESSAGE,
            "account": "secret",
        }
    ).encode("utf-8")

    err = _native_cli_failure(
        ["grok", "--output-format", "streaming-json"],
        return_code=1,
        stdout=bytearray(stdout),
        stderr=bytearray(),
    )

    assert type(err) is RuntimeError
    assert str(err) == "legacy native provider command failed"


def test_json_mode_quota_classifier_never_trusts_stderr() -> None:
    err = _native_cli_failure(
        ["grok", "--output-format", "json"],
        return_code=1,
        stdout=bytearray(b'{"type":"error","message":"request failed"}\n'),
        stderr=bytearray(_json_mode_error()),
    )

    assert type(err) is RuntimeError
    assert str(err) == "legacy native provider command failed"


def test_production_cli_promotes_native_signal_only_for_grok_implement() -> None:
    def boom(_prompt, _config):
        raise NativeGrokQuotaExhaustionSignal()

    provider = object.__new__(BoundProductionCLIProvider)
    object.__setattr__(provider, "role", ProviderRole.GROK_IMPLEMENT)
    object.__setattr__(provider, "invoker", boom)

    with pytest.raises(VerifiedGrokQuotaExhaustion):
        provider._invoke("prompt", object(), {}, max_response_bytes=1024)

    object.__setattr__(provider, "role", ProviderRole.CODEX_REVIEW)
    with pytest.raises(RuntimeError, match="non-Grok production provider"):
        provider._invoke("prompt", object(), {}, max_response_bytes=1024)
