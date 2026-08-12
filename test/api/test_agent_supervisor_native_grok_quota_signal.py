"""Exact Grok Build quota signal must promote to verified router exception."""

from __future__ import annotations

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


def test_native_cli_failure_classifies_exact_streaming_json_402() -> None:
    event = (
        b'{"type":"error","message":"API error (status 402 Payment Required): '
        b'Grok Build usage balance exhausted"}\n'
    )
    err = _native_cli_failure(
        ["grok", "prompt"],
        return_code=1,
        stdout=bytearray(event),
        stderr=bytearray(),
    )
    assert isinstance(err, NativeGrokQuotaExhaustionSignal)
    assert str(err) == GROK_BUILD_BALANCE_EXHAUSTED_MARKER


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
