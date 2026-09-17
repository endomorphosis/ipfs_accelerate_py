"""Pluggable task-handler hooks for the p2p task worker.

This module lets external code -- agents, services, plugins -- hook into
:func:`ipfs_accelerate_py.p2p_tasks.worker.run_worker` to fulfill custom task
types. A hooked worker watches the shared task queue (its own queue plus the
queues of discovered peers, i.e. "mesh" mode), claims tasks whose ``task_type``
matches a registered hook, executes them, and sends the result back to the
submitting peer over the existing libp2p / MCP task-queue transport
(:func:`ipfs_accelerate_py.p2p_tasks.client.complete_task`).

Handler signature::

    def my_handler(task: dict) -> dict:
        # task keys: task_id, task_type, model_name, payload, ...
        # payload is the dict the submitting peer provided.
        return {"ok": True, "summary": "..."}

The returned dict becomes the task result delivered to the submitter. Raising
an exception marks the task failed (the worker records the error text).

Registration (programmatic)::

    from ipfs_accelerate_py.p2p_tasks import worker_hooks

    worker_hooks.register_task_handler("research.summarize", my_handler)

Registration (declarative, picked up automatically by the worker)::

    # my_pkg/hooks.py
    TASK_HOOKS = {"research.summarize": my_handler}

    # or, for full control:
    def register_task_hooks(registry):
        registry.register("research.summarize", my_handler, aliases=("research.summary",))

and then either::

    export IPFS_ACCELERATE_PY_TASK_WORKER_HOOKS="my_pkg.hooks"
    python -m ipfs_accelerate_py.p2p_tasks.worker --p2p-service --mesh

or::

    python -m ipfs_accelerate_py.p2p_tasks.worker --p2p-service --mesh \\
        --hooks my_pkg.hooks

A spec may also name an explicit attribute: ``"my_pkg.hooks:register_task_hooks"``
or ``"my_pkg.hooks:TASK_HOOKS"``. The attribute may be:

- a callable ``register_task_hooks(registry)`` (called with this registry),
- a ``dict`` mapping task_type -> handler,
- a ``(task_type, handler)`` tuple,
- a handler function carrying a ``task_types`` / ``TASK_TYPES`` attribute
  (string or iterable of strings).

Hook handlers are advertised as worker capabilities, so mesh claiming
(``claim_next`` against remote peers) includes the hooked task types and the
submitting peer gets the result via the normal ``complete_task`` RPC. Hooks
registered for a task type that already has a builtin worker handler only
replace the builtin when registered with ``override=True`` (or passed via the
``extra_handlers`` argument of ``run_worker``).
"""

from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .task_types import canonical_task_type, normalize_task_types

#: Handler callable: task dict in -> result dict out.
TaskHandler = Callable[[Dict[str, Any]], Dict[str, Any]]

#: Env var listing hook specs, comma-separated ``module`` or ``module:attr``.
HOOKS_ENV_VAR = "IPFS_ACCELERATE_PY_TASK_WORKER_HOOKS"
#: Set to a truthy value to skip env-var hook loading in the worker.
HOOKS_DISABLE_ENV_VAR = "IPFS_ACCELERATE_PY_TASK_WORKER_HOOKS_DISABLE"

_REGISTER_FN_NAME = "register_task_hooks"
_TASK_HOOKS_ATTR = "TASK_HOOKS"


def _truthy(value: object | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class HookEntry:
    """A registered task handler plus its registration options."""

    handler: TaskHandler
    override: bool = False
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    source: str = ""


class HookRegistry:
    """Thread-safe registry mapping canonical task types to hook handlers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, HookEntry] = {}

    def register(
        self,
        task_type: str,
        handler: TaskHandler,
        *,
        aliases: Iterable[str] = (),
        override: bool = False,
        source: str = "",
    ) -> str:
        """Register ``handler`` for ``task_type``.

        Returns the canonical task type. Raises ``ValueError``/``TypeError``
        for invalid inputs. Re-registering a type replaces the previous entry.
        """

        canonical = canonical_task_type(task_type)
        if not canonical:
            raise ValueError("task_type must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")
        norm_aliases = tuple(
            a for a in normalize_task_types(list(aliases or ()), expand_aliases=False) if a
        )
        with self._lock:
            self._entries[canonical] = HookEntry(
                handler=handler,
                override=bool(override),
                aliases=norm_aliases,
                source=str(source or ""),
            )
        return canonical

    def unregister(self, task_type: str) -> bool:
        canonical = canonical_task_type(task_type)
        with self._lock:
            return self._entries.pop(canonical, None) is not None

    def snapshot(self) -> Dict[str, HookEntry]:
        """Return a copy of canonical task type -> HookEntry."""

        with self._lock:
            return dict(self._entries)

    def task_types(self) -> List[str]:
        with self._lock:
            return sorted(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


#: Process-global registry consulted by ``run_worker``.
_default_registry = HookRegistry()


def get_registry() -> HookRegistry:
    return _default_registry


def register_task_handler(
    task_type: str,
    handler: TaskHandler,
    *,
    aliases: Iterable[str] = (),
    override: bool = False,
    source: str = "",
    registry: Optional[HookRegistry] = None,
) -> str:
    """Register a hook handler on the default (or given) registry."""

    return (registry or _default_registry).register(
        task_type, handler, aliases=aliases, override=override, source=source
    )


def unregister_task_handler(
    task_type: str, *, registry: Optional[HookRegistry] = None
) -> bool:
    return (registry or _default_registry).unregister(task_type)


def registered_task_types(*, registry: Optional[HookRegistry] = None) -> List[str]:
    return (registry or _default_registry).task_types()


def clear_task_handlers(*, registry: Optional[HookRegistry] = None) -> None:
    (registry or _default_registry).clear()


def _register_attr_value(
    value: Any, *, spec: str, registry: HookRegistry
) -> int:
    """Register one resolved hook-module attribute. Returns # of handlers."""

    # Callable taking a registry: register_task_hooks(registry).
    if callable(value) and getattr(value, "__name__", "") == _REGISTER_FN_NAME:
        value(registry)
        return len(registry.snapshot())

    # Dict mapping task_type -> handler (TASK_HOOKS style).
    if isinstance(value, Mapping):
        count = 0
        for task_type, handler in value.items():
            registry.register(str(task_type), handler, source=spec)
            count += 1
        return count

    # (task_type, handler) tuple.
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str):
        registry.register(str(value[0]), value[1], source=spec)
        return 1

    # Handler function carrying task_types / TASK_TYPES.
    if callable(value):
        declared = getattr(value, "task_types", None)
        if declared is None:
            declared = getattr(value, "TASK_TYPES", None)
        if isinstance(declared, str):
            declared = [declared]
        types = [str(t) for t in (declared or []) if str(t or "").strip()]
        if not types:
            raise ValueError(
                f"hook spec {spec!r}: handler has no task_types/TASK_TYPES attribute "
                "and is not a (task_type, handler) pair, dict, or register_task_hooks callable"
            )
        for task_type in types:
            registry.register(task_type, value, source=spec)
        return len(types)

    raise TypeError(
        f"hook spec {spec!r}: unsupported attribute type {type(value).__name__}; "
        "expected register_task_hooks callable, dict, (task_type, handler) tuple, "
        "or handler with task_types attribute"
    )


def _load_one_spec(spec: str, *, registry: HookRegistry) -> int:
    """Import one hook spec and register its handlers. Returns # registered."""

    spec = spec.strip()
    if not spec:
        return 0
    module_name, _, attr_name = spec.partition(":")
    module_name = module_name.strip()
    attr_name = attr_name.strip()
    if not module_name:
        raise ValueError(f"invalid hook spec {spec!r}: empty module name")

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(f"hook spec {spec!r}: cannot import module: {exc}") from exc

    if attr_name:
        try:
            value = getattr(module, attr_name)
        except AttributeError as exc:
            raise ImportError(
                f"hook spec {spec!r}: module has no attribute {attr_name!r}"
            ) from exc
        return _register_attr_value(value, spec=spec, registry=registry)

    # Bare module: prefer register_task_hooks(registry), then TASK_HOOKS.
    register_fn = getattr(module, _REGISTER_FN_NAME, None)
    if callable(register_fn):
        return _register_attr_value(register_fn, spec=spec, registry=registry)
    task_hooks = getattr(module, _TASK_HOOKS_ATTR, None)
    if task_hooks is not None:
        return _register_attr_value(task_hooks, spec=spec, registry=registry)
    raise ImportError(
        f"hook spec {spec!r}: module defines neither {_REGISTER_FN_NAME}() "
        f"nor {_TASK_HOOKS_ATTR}; use 'module:attr' to name the hook explicitly"
    )


def load_hook_specs(
    spec: Optional[str] = None, *, registry: Optional[HookRegistry] = None
) -> Tuple[int, List[str]]:
    """Load hook specs (comma-separated ``module`` / ``module:attr``).

    Defaults to :data:`HOOKS_ENV_VAR`. Returns ``(registered_count, errors)``;
    a failing spec is reported in ``errors`` without aborting the rest.
    Loading is idempotent per process only in the sense that re-registering a
    task type replaces the previous entry.
    """

    reg = registry or _default_registry
    raw = spec if spec is not None else os.environ.get(HOOKS_ENV_VAR, "")
    total = 0
    errors: List[str] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            total += _load_one_spec(part, registry=reg)
        except Exception as exc:
            errors.append(f"{part}: {exc}")
    return total, errors


def hooks_enabled() -> bool:
    """False when env-var hook loading is explicitly disabled."""

    return not _truthy(os.environ.get(HOOKS_DISABLE_ENV_VAR))
