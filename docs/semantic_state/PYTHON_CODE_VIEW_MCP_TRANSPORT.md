# Task-private MCP byte transport for Python code views

The [transport](../../ipfs_accelerate_py/agent_supervisor/runtime/python_code_view_mcp_transport.py)
connects the [MCP tools](PYTHON_CODE_VIEW_MCP_TOOLS.md) to an explicit
controller-owned byte channel. Each channel creates its own empty canonical
tool manager and registry and dispatches directly to live handlers. It does
not start a server, configure a CLI, select a provider, or run a model loop.
The closed arithmetic Python grammar and native worktree admission rules in
[Python code views](PYTHON_CODE_VIEWS.md) and
[code-view sessions](PYTHON_CODE_VIEW_SESSIONS.md) still apply.

This implementation is local and uncommitted. It grants no Lean admits or
proof, qualification, execution, source-write, or completion authority.
Actual provider tokens and token savings remain unknown. A byte-channel
exchange is not evidence of production provider integration.

## Controller API

```python
channel = PythonCodeViewMCPTransport(
    broker=broker, session_id=session_id, editable=False,
)
await channel.exchange(raw_frame, send)
handoff = await channel.consume_candidate(candidate_id)  # Controller only.
await channel.aclose()
```

`broker` must be the existing exact task-owned session broker, bound to its
real isolated worktree, patch scope, lease and fence. The session ID must
match. `editable` is an explicit Boolean; its default disables `propose_edit`.
It does not create an alternative owner or writer. There is no protocol
method for consuming or applying a candidate.

`exchange(raw, send)` accepts one complete frame as `bytes`. `send` must be a
trusted asynchronous function supplied by the controller. It receives the
exact outbound frame and must return only after the complete frame has been
written and flushed to the intended task channel. `exchange` returns no
response bytes or private candidate. Valid notifications produce no reply.
Concurrent or reentrant exchanges, candidate consumption and closure are
refused; the caller owns sequential channel use.
Construct and use the channel on one controller thread. The first async call
binds one `asyncio` event loop; later operations on another thread or loop
are refused. Trio is not supported by this transport.

The outbound frame is canonical JSON followed by one LF. Input accepts one
JSON value with an optional final LF; embedded raw LF or CR, batches,
duplicate JSON keys and non-UTF-8 bytes are refused. JSON-RPC request IDs must
be unique within the channel and be nonempty bounded strings or integers
within the exact JSON integer range. Booleans and null are not request IDs.
Request envelopes and method parameters have closed shapes. Error messages
are fixed protocol text; source, requests and private exception text are not
echoed as diagnostics.

## Handshake and methods

The supported MCP protocol version is **`2024-11-05`**. Initialize using a
request with exactly these parameter fields:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"owned-controller","version":"1"}}}
```

The server always returns its supported version, including when the client
offers another bounded version string. This is a server-version fallback,
not support for the offered version. A client must check the returned version
and stop if it cannot support `2024-11-05`. Client capabilities must be the
empty object; extra capability declarations or client-info fields are
refused. The server advertises only `tools` with `listChanged: false`.

Only after the initialize reply has been successfully sent does the channel
enter `initializing`. The client then sends:

```json
{"jsonrpc":"2.0","method":"notifications/initialized"}
```

That notification makes the channel ready. Tool calls before readiness fail.
The remaining request methods are:

| Method | Parameters and result |
| --- | --- |
| `tools/list` | No parameters or `{}`; returns the channel's static tool descriptors. No pagination. |
| `tools/call` | Exactly `name` and `arguments`; returns one text content block containing canonical tool-result JSON and its `isError` status. |

Tool names are fully qualified, such as `python_code_view.read_view`.
Arguments retain the adapter's exact session ID and closed schemas. Tools
include explicit legend, read and native-delivery acknowledgements; optional
`propose_edit` prepares a candidate only. Unsupported notifications are
silent and never dispatch tools. There are no resources, prompts, sampling,
roots, arbitrary dispatch, native apply, or protocol shutdown methods.

## Bounds and failure custody

| Limit | Bound |
| --- | --- |
| Inbound frame | 96 KiB, including an optional final LF |
| Outbound reply | 512 KiB, including its final LF |
| Whole channel | 8 MiB of inbound and outbound frame bytes |
| Inbound messages | 256, including notifications and invalid messages |
| Parsed JSON depth | 32 |
| Parsed JSON value nodes | 16,384 |

These bounds supplement the broker and adapter limits; they do not increase
source size, expression size, encoder context, retention limits or authority.
They are encoded-byte and parser bounds, not RSS guarantees. The host must
also bound its frame reader before allocating unbounded input; passing a
completed frame to this API cannot retroactively bound an upstream buffer.

Budget exhaustion, failed sends, cancellation and internal custody failures
close the channel and release the broker, adapter, candidate retention and
transport visibility. Cancellation is re-raised after shielded cleanup;
blocking adapter cleanup is offloaded. `aclose()` is idempotent when idle. An
active exchange must first finish or be cancelled before explicit closure.
Closing channel custody does not delete, replace or close the native
worktree's lifecycle owner.

## Delivery, acknowledgement and candidate handoff

Native visibility advances only from an actual successful public
`deliver_native_source` tool reply and only after `await send(frame)` returns.
The transport checks the delivery's session and source bounds and keeps at
most four native paths. It does not obtain visibility from a retained private
original, a filesystem guess, a compact view, or model-supplied visibility.
Reused delivery references do not create fresh native bytes.

A successful send establishes a controller-observed write, not that a model
read or understood it. The client must separately call
`acknowledge_native_source` with the matching delivery and view IDs before
proposing an edit. Legend and read acknowledgements are likewise explicit.
Neither ACKs nor private originals substitute for the actual delivered
source in the transport's visibility map.

After an authorized controller selects a returned candidate ID,
`await channel.consume_candidate(candidate_id)` freshly rechecks the owned
source and visibility and consumes that candidate once. It returns a frozen
`PythonCodeViewCandidateHandoff` containing:

| Field | Private controller material |
| --- | --- |
| `prepared` | The rechecked `NativeViewPatch`, including its native patch text. |
| `visible_sources` | An immutable mapping of exact native source strings actually delivered through this channel. |

Successful consumption seals and closes the entire channel and broker,
releasing other candidates and channel visibility. The handoff retains its
immutable visibility snapshot; it is never a protocol response. A failed
consume also closes the channel. The existing native worktree owner remains
available to the controller's single validation/apply stage.

## Real sender example

This example uses an existing `asyncio.StreamWriter` for the intended task
channel. The host supplies the already owned broker, frame reader, candidate
selection and lifecycle. It starts neither a listener nor a provider.

```python
from asyncio import StreamWriter

from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_mcp_transport import (
    PythonCodeViewMCPTransport,
)

channel = PythonCodeViewMCPTransport(
    broker=broker, session_id=session_id, editable=True,
)

async def send(frame: bytes) -> None:
    writer.write(frame)  # writer: StreamWriter, already bound to this task.
    await writer.drain()

try:
    # The host's bounded reader supplies real client frames in order:
    # initialize, initialized notification, then tools/list and tools/call.
    for raw_frame in received_frames:
        await channel.exchange(raw_frame, send)

    # An authorized controller selects an actually returned candidate.
    handoff = await channel.consume_candidate(selected_candidate_id)
finally:
    await channel.aclose()

# Hand handoff.prepared.patch_text and handoff.visible_sources to the
# existing controller's validation/apply stage for the same native owner.
# That stage rechecks current lease, fence, scope, source and patch, applies
# once, then runs its ordinary verification and acceptance gates.
```

`drain()` establishes completion of the stream write according to the host's
I/O contract; it is not a peer-read receipt or model acknowledgement. No
controller should fabricate frames or automatically ACK merely to satisfy
edit prerequisites. If channel consumption fails, no handoff is available.

## Remaining supervisor integration

The supported execution route here is the private manager's direct live
dispatch. The generic server's `tools_dispatch` can return result-cache hits
without entering a handler; it is outside this boundary. The adapter rejects
cache-control arguments, and this transport provides no such controls. Do not
place another result cache in front of the channel.

`SemanticCompressionHarness` creates its real isolated worktree internally,
then invokes its injected provider, then validates and applies its returned
patch. A production bridge still needs explicit trusted hooks to bind this
channel to that exact harness-created worktree and to pass the consumed
native patch and actual visibility snapshot into those existing stages.
`request.visible_sources` does not automatically learn tool deliveries.
Never apply through the broker first and then let the harness apply again.

The existing context materializer must also admit and count static tool
schemas, real conversation frames and native expansions as required for each
actual provider turn. The current byte counters are not provider token
counts. Do not replace the materializer's source/context/token checks with a
transport receipt or precomputed gateway result.

The sealed Codex and Grok CLI routes remain unchanged and have no adopted
callback loop for these tools. Provider adoption requires a separately
admitted tool-capable route and honest provider attribution. Temperature,
context bounds, native apply limits and all existing verification gates
remain in force. These fixtures and this transport grant **zero Lean
admits**; only an actual `lake build <Lib>` can supply the corresponding Lean
admission evidence.
