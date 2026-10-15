# Task-owned MCP tools for Python code views

The [MCP adapter](../../ipfs_accelerate_py/agent_supervisor/runtime/python_code_view_mcp_tools.py)
exposes the [bounded session broker](PYTHON_CODE_VIEW_SESSIONS.md) through the
real hierarchical tool manager. The controller supplies one live broker and an
otherwise empty private manager/registry. Optional edit proposals prepare a
private native candidate; the controller retains the existing native apply and
acceptance stage.

This is explicit library integration. It does not bootstrap a server, attach
tools to the process-global function registry, or enable provider use. The
supported source remains the closed Python arithmetic profile described in
[Python code views](PYTHON_CODE_VIEWS.md). General application files and
unsupported syntax keep their existing native or builtin tool paths.

## APIs and discovery

```python
adapter = PythonCodeViewMCPTools(
    broker=broker, session_id=session_id, visibility_resolver=None,
)
descriptors = adapter.descriptors()
full_names = register_python_code_view_tools(manager, adapter)
prepared = adapter.consume_candidate(candidate_id, visible_sources=visible_sources)
adapter.close()
```

`descriptors()` returns independent static descriptions and closed input
schemas. Every tool call requires the bound `session_id`. The category is
`python_code_view`; registration returns full names such as
`python_code_view.read_view`. Requests cannot supply owner, scope, fence,
private source, visibility maps or prepared patches.

| Tool | Public operation |
| --- | --- |
| `read_view` | Read the current scoped view or an explicitly acknowledged snapshot reference. |
| `acknowledge_legend` | Record explicit caller feedback for this broker's emitted legend/read. |
| `acknowledge_read` | Record explicit caller feedback for a snapshot after its legend acknowledgement. |
| `deliver_native_source` | Deliver actual confined native filesystem source or an acknowledged current reference. |
| `acknowledge_native_source` | Record explicit caller feedback for the exact native delivery/view. |
| `close_session` | Release broker and pending candidate custody. |
| `propose_edit` | Optional: prepare a private expression candidate and return a body-free candidate identity. |

The first six tools cannot write source. `propose_edit` is discoverable only
when the controller supplies an explicit synchronous `visibility_resolver`.
That callback receives a bounded copy of the edit request and returns actual
transport-visible native bytes. Model arguments cannot replace the callback.
It has no private-artifact fallback and performs no automatic acknowledgement.

The candidate response uses schema
`supervisor-python-code-view-mcp-candidate@1`. It contains the full candidate
CID, view CID and slot ID, candidate status, false authority fields and zero
Lean admits. Native source, private maps, native edit and patch bodies remain
controller-side. Error responses use
`supervisor-python-code-view-mcp-error@1` with a bounded error code; sensitive
controller exceptions are not echoed into tool output.

The full candidate CID binds its adapter allocation, broker/session, request,
native edit identity and patch hash. `consume_candidate` is a controller-only
API, absent from tool discovery. It freshly checks owner, source, scope,
delivery and caller visibility, reconstructs the candidate and requires exact
equality with the retained preparation. Successful consumption removes that
candidate once and returns the private `NativeViewPatch`. It does not apply it.

## Private registration and live dispatch

Create a fresh canonical `HierarchicalToolManager(registry=ToolRegistry())`
with `runtime_router=None`. Registration rejects existing categories, loaders,
tools, execution state, routing or a process-global registry. Register once and
keep the manager and registry under exclusive controller ownership.

Registration calls the real manager's public API. That API also publishes
static `ToolMetadata` in the process-global metadata catalog. Only names,
descriptions and static schemas belong there: no bound `_func`, worktree,
session ID, callback, source, visibility or private candidate is published.
Bound handlers live in the dedicated private manager/registry. Closing the
adapter releases its broker, pending candidates and private callback; retained
handlers refuse later calls. It does not delete unrelated global metadata.

Use the dedicated manager's direct dispatch:

```python
response = await manager.dispatch(
    "python_code_view", "read_view",
    {"session_id": session_id, "path": "calc.py"},
)
```

The generic server's `tools_dispatch` can use `__use_result_cache` and return a
cached response without entering the live handler. That route is outside this
adapter's qualified dispatch boundary. The adapter rejects cache-control
arguments, but it cannot repair an upstream cache that bypasses it. Provider
adoption requires an admitted cacheless transport and worker tool boundary
that always reaches this private manager's live dispatch.

Handlers offload blocking broker, Git and resolver work with
`anyio.to_thread.run_sync(..., abandon_on_cancel=False)`. A per-adapter lock
serializes operations. Cancellation does not intentionally abandon a running
preparation thread; the handler exposes no apply operation or independent
writer. Existing worktree locks, leases and source checks still apply.

Broker TTL and retention remain bounded as documented in the session guide.
The adapter additionally retains at most four pending candidates and 8 MiB of
their encoded private artifacts, request and native patch data. These are
encoded-retention limits, not resident-memory or RSS guarantees. Controller
cleanup should call `close()` in a `finally` block, including refused or expired
sessions. Its own transport-visible map has separate controller custody.

## Controller flow and native visibility

The following standalone sketch assumes the controller already owns a real
clean isolated worktree and scope. `trusted_transport.deliver_and_confirm` is
a host-supplied port, not an API implemented by this module. It must actually
deliver the native public tool payload to the model and return explicit
transport confirmation before the controller records visibility and ACKs.
The sketch runs inside the controller's async task.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_mcp_tools import (
    PythonCodeViewMCPTools, register_python_code_view_tools,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_session_broker import (
    PythonCodeViewSessionBroker,
)
from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import HierarchicalToolManager
from ipfs_accelerate_py.mcp_server.tool_registry import ToolRegistry

broker = PythonCodeViewSessionBroker(
    worktree=worktree, scope=scope, session_id=session_id,
    lease_id=worktree.lease_id, fence=worktree.fence,
)
actually_delivered = {}

def visibility_resolver(_request):
    return dict(actually_delivered)

adapter = PythonCodeViewMCPTools(
    broker=broker, session_id=session_id,
    visibility_resolver=visibility_resolver,
)
manager = HierarchicalToolManager(registry=ToolRegistry(), runtime_router=None)
register_python_code_view_tools(manager, adapter)

try:
    view = await manager.dispatch("python_code_view", "read_view", {
        "session_id": session_id, "path": "calc.py",
    })
    if view.get("is_error") or not await trusted_transport.deliver_and_confirm(view):
        raise RuntimeError("view transport has no confirmation")
    view_id = view["state"]["view_id"]
    await manager.dispatch("python_code_view", "acknowledge_legend", {
        "session_id": session_id, "read_id": view["read_id"],
        "legend_id": view["state"]["legend_id"],
    })
    await manager.dispatch("python_code_view", "acknowledge_read", {
        "session_id": session_id, "read_id": view["read_id"], "view_id": view_id,
    })
    native = await manager.dispatch("python_code_view", "deliver_native_source", {
        "session_id": session_id, "view_id": view_id,
    })
    if native.get("is_error") or not await trusted_transport.deliver_and_confirm(native):
        raise RuntimeError("native transport has no confirmation")
    actually_delivered[native["path"]] = native["source"]
    await manager.dispatch("python_code_view", "acknowledge_native_source", {
        "session_id": session_id, "view_id": view_id,
        "delivery_id": native["delivery_id"],
    })

    # wire_request comes from the model's hash-bound expression proposal.
    candidate = await manager.dispatch("python_code_view", "propose_edit", {
        "session_id": session_id, "wire_request": wire_request,
    })
    if candidate.get("is_error"):
        raise RuntimeError(candidate["error_code"])
    prepared = adapter.consume_candidate(
        candidate["candidate_id"], visible_sources=dict(actually_delivered),
    )
    result = broker.apply_edit(
        prepared=prepared, visible_sources=dict(actually_delivered),
        session_id=session_id,
    )  # The standalone controller's one existing native apply stage.
    # Check result.applied, then run the controller's ordinary acceptance gates.
finally:
    adapter.close()
    actually_delivered.clear()
```

The sketch uses source from the actual public native delivery, never
`view.to_dict()["original_source"]`, a fixture original or a filesystem guess.
An acknowledgement describes explicit caller feedback; it is not a proof of
model consumption. No source can be silently omitted from actual transport
while retained private bytes are reported as visible.

`SemanticCompressionHarness` already validates and calls the native
`worktree.apply_patch` stage. When integrating with it, provide the consumed
candidate's native patch to that existing stage under the same worktree,
scope, fence and actual `visible_sources`. Do not call `broker.apply_edit`
first and then ask the harness to apply it again. That handoff bridge is still
pending. A consumed candidate does not bypass harness production evidence or
acceptance gates. The unchanged clean-base limit permits one successful apply;
further mutations need an ordinarily established fresh base and session.

## Integration status and evidence boundary

The owner/harness path supplies a real `IsolatedWorktree` and already owns
validation, native application and acceptance. The sealed Grok/Codex CLI
provider path currently has no admitted tool callback or MCP route for these
session tools. The daemon also has existing raw-path workflows. Connecting
those worker and daemon boundaries requires an explicit controller bridge;
registration alone does not route their reads, proposals or source visibility
through this adapter. No provider allowlist is weakened to manufacture access.

The [adapter tests](../../test/api/test_python_code_view_mcp_tools.py) exercise
the real private manager's discovery and dispatch, actual native delivery and
explicit ACKs, refusal paths, private candidate custody and controller-only
native apply in authored fixtures. Protocol registration and fixture execution
are separate from real provider or Terminal Bench task results. This guide
records no test-pass totals or adoption result before guarded validation.

Exact model tokens, provider-billed usage and task-success effects remain
unknown. Count discovery schemas, requests, responses, acknowledgements,
native expansion, proposals, diagnostics and retries in a real cacheless
provider trial. A registered tool or body-free candidate CID is not evidence
of token savings, reconstruction fidelity or formal logic validity.

No tool response, prepared patch, native application or Python fixture grants
proof, qualification, task completion or Lean admission. Only an actual
`lake build <Lib>` grants the Lean admit. This integration performs no Lake
build, autoencoder training or weight download; the Constitution remains
unformalized.
