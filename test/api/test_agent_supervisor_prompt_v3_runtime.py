from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import RunRegistry
from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import (
    MissingRuntimeHandlerError,
    RuntimeEffectError,
    StandardSupervisorRuntimeFactory,
)


def test_production_runtime_refuses_missing_effect_handlers(tmp_path) -> None:
    with pytest.raises(MissingRuntimeHandlerError) as raised:
        StandardSupervisorRuntimeFactory(registry=RunRegistry(tmp_path), handlers={})

    assert "start" in raised.value.missing
    assert "materialize" in raised.value.missing


def test_effect_adapter_cannot_turn_missing_effect_into_success(tmp_path) -> None:
    handlers = {name: (lambda *_args, **_kwargs: {"receipt_cid": "", "effect_applied": False}) for name in (
        "resolve", "preview", "authorize", "materialize", "start", "adopt", "observe", "steer", "validate", "stop",
    )}
    factory = StandardSupervisorRuntimeFactory(registry=RunRegistry(tmp_path), handlers=handlers)

    with pytest.raises(RuntimeEffectError):
        factory.invoke("start", object(), object())
