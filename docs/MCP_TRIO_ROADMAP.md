# Trio MCP Server Roadmap

This repository currently runs the MCP HTTP server via FastAPI/Uvicorn (ASGI), which is **asyncio-first**. Meanwhile, the libp2p TaskQueue implementation used by the P2P task system is **Trio-based**.

That split (asyncio HTTP + Trio libp2p) is workable, but it requires a clear boundary and a consistent bridging strategy.

## Current state (today)

- **HTTP MCP server**: FastAPI/Starlette running under Uvicorn (asyncio event loop).
- **P2P TaskQueue**: libp2p stack that expects to run under Trio.
- **Bridge approach**: MCP P2P tools call Trio-only libp2p client code using a small adapter that:
  - runs inline when already in Trio,
  - otherwise executes the Trio work in a worker thread using `anyio.run(..., backend="trio")`.

This avoids running Trio-only code under asyncio request handlers, which can otherwise fail with runtime-context errors.

## Why “just use AnyIO everywhere” is not enough

AnyIO is a portability layer for *your code*, but it does not automatically make a **Trio-only dependency** safe to call from an **asyncio** event loop.

- If a dependency internally assumes Trio (nurseries, cancel scopes, Trio socket APIs, etc.), it still needs a Trio runtime context.
- FastAPI/Uvicorn typically provides an asyncio runtime context.

AnyIO is still valuable:
- for writing bridge code,
- for moving internal services toward backend-agnostic patterns,
- for future migration work.

## Bridge approach (recommended short-term)

Goal: Keep FastAPI/Uvicorn (asyncio) for the HTTP side, while making all libp2p operations reliably Trio-backed.

Recommended rules:

1. **All MCP tools that touch libp2p/TaskQueue must call a single helper** (e.g. `_run_in_trio(...)`).
2. Avoid starting/stopping Trio services directly inside asyncio request handlers.
3. Prefer returning structured `{ok, error, ...}` responses so the MCP caller can handle partial connectivity.

Benefits:
- Minimal deployment change.
- Keeps the high-level MCP HTTP server stable.
- Makes P2P tooling usable over HTTP without fragile runtime coupling.

Costs:
- Thread hop per tool call when invoked under asyncio.
- Some latency overhead.

## End state (Trio-backed MCP server)

Goal: Run the entire MCP server under Trio so P2P code runs in-process without bridging.

### Option A: Run an ASGI server with Trio support

Use an ASGI server that supports Trio (commonly **Hypercorn**) to run the FastAPI/Starlette app under Trio.

Migration steps:

1. **Server runner**
   - Add a Trio-capable runner (e.g. `hypercorn --worker-class trio ...`).
   - Ensure your deployment/systemd units use that runner instead of `uvicorn`.

2. **Audit asyncio assumptions**
   - Identify any `asyncio`-specific APIs used in MCP server code (tasks, locks, event loop access).
   - Replace with AnyIO equivalents (cancel scopes, task groups) or Trio equivalents.

3. **Lifecycle hooks**
   - Ensure startup/shutdown hooks work under Trio.
   - Confirm background work (timers, periodic tasks) uses AnyIO/Trio-friendly constructs.

4. **Test strategy**
   - Add a small integration test that boots the ASGI app under Trio and calls a Trio-dependent tool.

Risks:
- Some middleware and third-party FastAPI/Starlette plugins implicitly assume asyncio.
- Threadpool usage and sync dependencies may need review.

### Option B: Separate process (recommended for operational isolation)

Run libp2p TaskQueue as its own long-lived Trio process/service, and keep the HTTP MCP server on asyncio.

- MCP tools communicate with the P2P service via an internal RPC channel (loopback HTTP, Unix socket, or in-memory when co-located).

Benefits:
- Strong isolation, clearer failure modes.
- Avoids embedding a complex P2P runtime inside the web server.

Costs:
- Extra service to manage.
- Need a clean, versioned internal RPC API.

## Practical roadmap (incremental)

1. **Bridge everywhere** (now)
   - Ensure every libp2p/TaskQueue client call in MCP tools runs through the Trio bridge helper.

2. **Reduce bridge overhead** (next)
   - Consider a dedicated background Trio worker/service inside the MCP process to reuse a host/network.
   - If adopted, bridge calls into that worker rather than starting fresh work per request.

3. **Prototype Trio-backed ASGI** (later)
   - Add an alternate entrypoint to run the MCP server under Hypercorn+Trio.
   - Gate behind a feature flag or separate systemd unit.

4. **Switch default** (end)
   - When Trio-backed operation is stable and dependencies are validated, make it the default runner.

## Success criteria

- Calling MCP P2P tools over HTTP works reliably on both machines.
- Discovery and dialing behavior is consistent with the CLI/libp2p-only scripts.
- A Trio-backed server can run the same MCP tool set without the bridge helper.
