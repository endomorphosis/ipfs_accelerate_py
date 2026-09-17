"""Tests for p2p task worker hooks (agent-pluggable task handlers)."""

from __future__ import annotations

import os
import sys
import types

import pytest

from ipfs_accelerate_py.p2p_tasks import worker_hooks
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.worker import run_worker


@pytest.fixture()
def registry():
    reg = worker_hooks.HookRegistry()
    yield reg
    reg.clear()


@pytest.fixture(autouse=True)
def _clean_global_registry_and_env(monkeypatch, tmp_path):
    worker_hooks.clear_task_handlers()
    monkeypatch.delenv(worker_hooks.HOOKS_ENV_VAR, raising=False)
    monkeypatch.delenv(worker_hooks.HOOKS_DISABLE_ENV_VAR, raising=False)
    # Keep the worker light: no HF/multimodal probing, no mesh discovery.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_HF", "0")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_MULTIMODAL", "0")
    import ipfs_accelerate_py.p2p_tasks.worker as worker_mod

    old_advertised = worker_mod._hook_advertised_types
    worker_mod._hook_advertised_types = None
    yield
    worker_hooks.clear_task_handlers()
    worker_mod._hook_advertised_types = old_advertised


def _handler(task):
    return {"ok": True, "echo": (task.get("payload") or {}).get("text")}


def test_register_and_snapshot(registry):
    canonical = registry.register("research.summarize", _handler, aliases=("research.summary",))
    assert canonical == "research.summarize"
    snap = registry.snapshot()
    assert set(snap) == {"research.summarize"}
    assert snap["research.summarize"].handler is _handler
    assert snap["research.summarize"].aliases == ("research.summary",)
    assert registry.task_types() == ["research.summarize"]
    assert registry.unregister("research.summarize") is True
    assert registry.unregister("research.summarize") is False


def test_register_rejects_bad_input(registry):
    with pytest.raises(ValueError):
        registry.register("", _handler)
    with pytest.raises(TypeError):
        registry.register("x.y", "not-callable")


def test_load_task_hooks_dict_shape(monkeypatch, registry):
    module = types.ModuleType("hookmod_dict")
    module.TASK_HOOKS = {"agent.ping": _handler}
    monkeypatch.setitem(sys.modules, "hookmod_dict", module)

    count, errors = worker_hooks.load_hook_specs("hookmod_dict", registry=registry)
    assert errors == []
    assert count == 1
    assert registry.task_types() == ["agent.ping"]


def test_load_register_fn_shape(monkeypatch, registry):
    module = types.ModuleType("hookmod_fn")

    def register_task_hooks(reg):
        reg.register("agent.work", _handler, override=True)

    module.register_task_hooks = register_task_hooks
    monkeypatch.setitem(sys.modules, "hookmod_fn", module)

    count, errors = worker_hooks.load_hook_specs("hookmod_fn", registry=registry)
    assert errors == []
    assert count == 1
    assert registry.snapshot()["agent.work"].override is True


def test_load_explicit_attr_tuple_shape(monkeypatch, registry):
    module = types.ModuleType("hookmod_attr")
    module.my_hook = ("agent.custom", _handler)
    monkeypatch.setitem(sys.modules, "hookmod_attr", module)

    count, errors = worker_hooks.load_hook_specs("hookmod_attr:my_hook", registry=registry)
    assert errors == []
    assert count == 1
    assert registry.task_types() == ["agent.custom"]


def test_load_reports_errors_without_raising(monkeypatch, registry):
    count, errors = worker_hooks.load_hook_specs(
        "definitely.not.a.real.module", registry=registry
    )
    assert count == 0
    assert len(errors) == 1
    assert "definitely.not.a.real.module" in errors[0]


def _run_once(tmp_path, **kwargs):
    queue_path = str(tmp_path / "queue.db")
    worker_id = "hook-test-worker"
    q = TaskQueue(queue_path)
    task_id = q.submit(
        task_type="agent.ping",
        model_name="",
        payload={"text": "hello"},
    )
    q.close()
    rc = run_worker(
        queue_path=queue_path,
        worker_id=worker_id,
        poll_interval_s=0.05,
        once=True,
        mesh=False,
        **kwargs,
    )
    assert rc == 0
    q = TaskQueue(queue_path)
    try:
        return q.get(task_id)
    finally:
        q.close()


def test_extra_handlers_fulfill_custom_task(tmp_path):
    seen = {}

    def ping(task):
        seen["task_id"] = task.get("task_id")
        return {"ok": True, "pong": (task.get("payload") or {}).get("text")}

    task = _run_once(tmp_path, extra_handlers={"agent.ping": ping})
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["pong"] == "hello"
    assert seen["task_id"] == task["task_id"]


def test_global_registry_hook_advertised_and_fulfilled(tmp_path, monkeypatch):
    worker_hooks.register_task_handler("agent.ping", _handler)
    assert worker_hooks.registered_task_types() == ["agent.ping"]

    task = _run_once(tmp_path)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["echo"] == "hello"


def test_env_hook_specs_loaded_by_worker(tmp_path, monkeypatch):
    module = types.ModuleType("hookmod_env")
    module.TASK_HOOKS = {"agent.ping": _handler}
    monkeypatch.setitem(sys.modules, "hookmod_env", module)
    monkeypatch.setenv(worker_hooks.HOOKS_ENV_VAR, "hookmod_env")

    task = _run_once(tmp_path)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["echo"] == "hello"


def test_hook_errors_mark_task_failed(tmp_path):
    def boom(task):
        raise RuntimeError("kaboom")

    task = _run_once(tmp_path, extra_handlers={"agent.ping": boom})
    assert task is not None
    assert task["status"] == "failed"
    assert "kaboom" in str(task.get("error") or "")


def test_plain_registry_hook_does_not_shadow_builtin(tmp_path, capsys):
    worker_hooks.register_task_handler("text-generation", _handler)  # override=False
    queue_path = str(tmp_path / "queue2.db")
    q = TaskQueue(queue_path)
    task_id = q.submit(task_type="text-generation", model_name="", payload={"prompt": "hi"})
    q.close()
    rc = run_worker(
        queue_path=queue_path,
        worker_id="hook-test-worker",
        poll_interval_s=0.05,
        once=True,
        mesh=False,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ignoring hook" in out
    q = TaskQueue(queue_path)
    try:
        task = q.get(task_id)
    finally:
        q.close()
    # Builtin text-generation ran (may fail without transformers, but must NOT
    # carry our hook's marker).
    assert (task.get("result") or {}).get("echo") is None


def test_advertised_types_include_hooks(tmp_path):
    import ipfs_accelerate_py.p2p_tasks.worker as worker_mod

    worker_hooks.register_task_handler("agent.ping", _handler)
    task = _run_once(tmp_path, extra_handlers={"agent.other": _handler})
    assert task is not None
    advertised = worker_mod.get_hook_advertised_task_types()
    assert "agent.ping" in advertised
    assert "agent.other" in advertised


def test_advertised_types_empty_when_allowlist_overridden(tmp_path, monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.worker as worker_mod

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES", "text-generation")
    worker_hooks.register_task_handler("agent.ping", _handler)
    _run_once(tmp_path)
    assert worker_mod.get_hook_advertised_task_types() == []


def test_standalone_advertisement_loads_env_hooks(monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.worker as worker_mod

    module = types.ModuleType("hookmod_adv")
    module.TASK_HOOKS = {"agent.ping": _handler}
    monkeypatch.setitem(sys.modules, "hookmod_adv", module)
    monkeypatch.setenv(worker_hooks.HOOKS_ENV_VAR, "hookmod_adv")
    # No run_worker call: simulates a standalone service status request.
    assert worker_mod._hook_advertised_types is None
    assert "agent.ping" in worker_mod.get_hook_advertised_task_types()
