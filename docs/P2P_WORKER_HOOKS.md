# P2P Worker Hooks: let agents fulfill tasks from the shared queue

The task worker (`python -m ipfs_accelerate_py.p2p_tasks.worker`) already knows
how to **look into the shared task queue, claim tasks other peers submitted,
execute them, and send results back** over libp2p (`complete_task` RPC) or the
MCP task-queue tools. What it could not do until now was execute *your* task
types: handlers were hardcoded to ML-inference families (`text-generation`,
`embedding`, `tool.call`, `shell`, `docker.*`, ...).

`ipfs_accelerate_py/p2p_tasks/worker_hooks.py` adds the missing piece: a public
hook registry. Register a handler for a custom task type and the worker will:

1. advertise the task type as a capability (so peers route matching tasks here),
2. claim matching tasks from the local queue **and** from discovered peers'
   queues (mesh mode),
3. run your handler,
4. return the result dict to the submitting peer via the existing
   `complete_task` transport (libp2p RPC; the MCP++ `task_queue` wrapper uses
   the same client calls).

## Handler contract

```python
def my_handler(task: dict) -> dict:
    payload = task.get("payload") or {}
    # ... do the work ...
    return {"ok": True, "answer": "..."}
```

`task` also carries `task_id`, `task_type`, `model_name`. Return a JSON-able
dict; it becomes the task result the submitter sees via `task-wait`. Raising an
exception marks the task `failed` with your error text. Handlers may optionally
accept an `accelerate_instance` keyword (detected automatically).

## Registering hooks

**Programmatic** (before calling `run_worker`):

```python
from ipfs_accelerate_py.p2p_tasks import worker_hooks
from ipfs_accelerate_py.p2p_tasks.worker import run_worker

worker_hooks.register_task_handler("research.summarize", my_handler)

run_worker(queue_path="...", worker_id="agent-1", p2p_service=True, mesh=True)
```

or per-call without touching global state:

```python
run_worker(..., extra_handlers={"research.summarize": my_handler})
```

**Declarative** (no code changes at the call site):

```python
# my_agent/hooks.py
TASK_HOOKS = {
    "research.summarize": summarize_handler,
    "code.review": review_handler,
}

# or, for aliases / overrides:
def register_task_hooks(registry):
    registry.register("research.summarize", summarize_handler,
                       aliases=("research.summary",))
```

```bash
export IPFS_ACCELERATE_PY_TASK_WORKER_HOOKS="my_agent.hooks"
python -m ipfs_accelerate_py.p2p_tasks.worker --p2p-service --mesh

# equivalent:
python -m ipfs_accelerate_py.p2p_tasks.worker --p2p-service --mesh \
    --hooks my_agent.hooks
```

A spec may name an explicit attribute: `my_agent.hooks:register_task_hooks`.
Set `IPFS_ACCELERATE_PY_TASK_WORKER_HOOKS_DISABLE=1` to skip env loading.

## Overriding builtins

A hook for a task type that already has a builtin handler (e.g.
`text-generation`) is ignored with a warning unless registered with
`override=True` (or passed via `extra_handlers`, which always wins).

## Security notes

Hook handlers execute arbitrary peer-submitted payloads. Only enable hooks you
trust, and keep the worker's existing trust/session gating (`peer_trust.py`,
session tags) in mind when exposing a worker to untrusted peers. The `shell`
analogy applies: a hook is remote code execution *by design* -- treat the
worker host accordingly.

## See also

- `docs/P2P_TASKQUEUE_DISCOVERY.md` -- finding peers without pre-shared multiaddrs
- `ipfs_accelerate_py/p2p_tasks/worker_hooks.py` -- registry API reference
- `test/unit/test_p2p_worker_hooks.py` -- executable examples
