"""Native recovery callback registration preserves explicit execution custody."""
from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)

CALLBACKS = ("quack_preprojection_transport_recovery_fn", "deterministic_reconciliation_fn")


def _daemon(tmp_path, **kwargs):
    return DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb", authority_mode="embedded",
        task_source_kind="duckdb", install_schema=False, **kwargs,
    )


def _execution():
    return {"provider_fn": lambda attempt: {}, "effect_fn": lambda attempt, result: {},
            "validation_fn": lambda attempt, result: {}}


def test_native_recovery_callbacks_default_none_and_bind_once(tmp_path):
    daemon = _daemon(tmp_path, require_real_execution=True)
    for name in CALLBACKS:
        assert getattr(daemon, "_" + name) is None
    callbacks = {name: lambda *args: pytest.fail("binding invoked recovery") for name in CALLBACKS}
    daemon.bind_execution_callbacks(**_execution(), **callbacks)
    for name, callback in callbacks.items():
        assert getattr(daemon, "_" + name) is callback
    with pytest.raises(DatabaseImplementationAuthorityError, match="already bound"):
        daemon.bind_execution_callbacks(**_execution())


@pytest.mark.parametrize("name", CALLBACKS)
def test_native_constructor_rejects_noncallable_recovery(tmp_path, name):
    with pytest.raises(TypeError, match="must be callable"):
        _daemon(tmp_path, **{name: object()})


@pytest.mark.parametrize("name", CALLBACKS)
def test_invalid_native_binding_does_not_partially_install_executor(tmp_path, name):
    daemon = _daemon(tmp_path, require_real_execution=True)
    with pytest.raises(TypeError, match="must be callable"):
        daemon.bind_execution_callbacks(**_execution(), **{name: object()})
    assert daemon._provider_fn is None
    assert daemon._effect_fn is None
    assert daemon._validation_fn is None
    for callback in CALLBACKS:
        assert getattr(daemon, "_" + callback) is None


@pytest.mark.parametrize("name", CALLBACKS)
def test_existing_native_recovery_binding_cannot_be_overwritten(tmp_path, name):
    callback = lambda *args: pytest.fail("binding invoked recovery")
    daemon = _daemon(tmp_path, require_real_execution=True, **{name: callback})
    with pytest.raises(DatabaseImplementationAuthorityError, match="already bound"):
        daemon.bind_execution_callbacks(**_execution())
    assert getattr(daemon, "_" + name) is callback
    assert daemon._provider_fn is None


def test_native_recovery_binding_cannot_enable_observer_execution(tmp_path):
    daemon = _daemon(tmp_path, require_real_execution=False)
    with pytest.raises(DatabaseImplementationAuthorityError, match="explicit real-execution"):
        daemon.bind_execution_callbacks(
            **_execution(), **{name: lambda *args: {} for name in CALLBACKS},
        )
    assert daemon._provider_fn is None
    for name in CALLBACKS:
        assert getattr(daemon, "_" + name) is None


@pytest.mark.parametrize("namespace, expected", [("", ""), ("  board:one  ", "board:one")])
def test_native_board_namespace_is_bound_for_checkout_custody(tmp_path, namespace, expected):
    daemon = _daemon(tmp_path, board_namespace=namespace)
    assert daemon.board_namespace == expected
