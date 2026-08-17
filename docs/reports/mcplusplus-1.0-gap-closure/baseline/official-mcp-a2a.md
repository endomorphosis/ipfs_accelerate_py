# Official MCP 2026-07-28 and A2A extension conventions

**Task:** MCPP-010  
**Verified:** 2026-08-15  
**Policy:** Primary sources only. No binding documents were changed by this task.

## 1. Summary of verified facts

| Topic | Verified finding |
| --- | --- |
| Current MCP revision | `2026-07-28` at https://modelcontextprotocol.io/specification/2026-07-28 |
| Current MCP lifecycle | **Not initialize-based.** Stateless, per-request `_meta`. |
| Legacy MCP lifecycle | `initialize` / `notifications/initialized` handshake for revisions `2025-11-25` and earlier |
| A2A extension identifiers | **URIs**, advertised on the Agent Card, activated via `A2A-Extensions` |
| MCP++ A2A execution extension URI (confirmed) | `https://mcplusplus.io/extensions/execution/v1` |
| Working alias (not the wire identifier) | `io.mcplusplus.execution@1` |

## 2. MCP 2026-07-28 — not initialize-based

### 2.1 Primary sources

| Source | URL |
| --- | --- |
| Specification (latest) | https://modelcontextprotocol.io/specification/2026-07-28 |
| Key changes / changelog | https://modelcontextprotocol.io/specification/2026-07-28/changelog |
| Base protocol (statelessness, `_meta`) | https://modelcontextprotocol.io/specification/2026-07-28/basic |
| Versioning and compatibility | https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning |
| Release announcement | https://blog.modelcontextprotocol.io/posts/2026-07-28/ |

### 2.2 Finding: current MCP is not initialize-based

The current official MCP revision is **stateless**. Protocol-level sessions and the `initialize` / `initialized` exchange are **removed** from modern behavior.

Official changelog (major change #2), https://modelcontextprotocol.io/specification/2026-07-28/changelog:

> Make MCP stateless: remove the `initialize`/`notifications/initialized` handshake. Every request now carries its protocol version and client capabilities in `_meta` (`io.modelcontextprotocol/protocolVersion`, `io.modelcontextprotocol/clientCapabilities`). Clients SHOULD identify themselves on each request (`io.modelcontextprotocol/clientInfo`), and servers SHOULD identify themselves in each result's `_meta` (`io.modelcontextprotocol/serverInfo`). Version mismatches return `UnsupportedProtocolVersionError` ([SEP-2575](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2575)).

Official versioning page, https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning:

> There is no negotiation handshake. Every request carries its protocol version, and the server accepts or rejects each request independently

Terminology on the same page:

> **Modern**: protocol versions that convey version, identity, and capabilities as per-request metadata (revision `2026-07-28` and later).  
> **Legacy**: protocol versions that establish a session with an `initialize` handshake (`2025-11-25` and earlier).

Official release blog, https://blog.modelcontextprotocol.io/posts/2026-07-28/:

> With the new spec version, we’ve officially retired the `initialize`/`initialized` exchange along with the `Mcp-Session-Id` header (refer to [SEP-2575](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2575), [SEP-2567](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2567)). Each request now travels on its own, carrying its protocol version, client identity, and client capabilities in `_meta`. If a client wants to learn a server’s capabilities before doing anything else, there’s a new `server/discover` Remote Procedure Call (RPC) for that; however, it is not required.

Base protocol statelessness requirement, https://modelcontextprotocol.io/specification/2026-07-28/basic:

> The Model Context Protocol (MCP) is a **stateless protocol**: all the information needed to process a request is contained in the request itself. A server processes each request independently; no state should be inferred from previous requests, even those on the same connection or stream.

**Implication for MCP++:** treating `initialize` / `initialized` with `protocolVersion` `2024-11-05` (or any pre-`2026-07-28` handshake) as *current* protocol behavior is incorrect. Dual bindings remain valid only when the initialize path is explicitly named **legacy** and the current binding is per-request `_meta` for `2026-07-28`.

### 2.3 Related modern MCP mechanics (for later bindings)

These are recorded from primary text so later tasks (bindings, Tasks, extensions) do not re-guess:

1. **Discovery:** Servers **MUST** implement `server/discover`. Clients **MAY** call it first; it is not a substitute for a mandatory handshake. Source: versioning page and changelog #3.
2. **Per-request required `_meta` keys:** `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` are required on every client request. Source: base protocol `_meta` table.
3. **HTTP headers:** Streamable HTTP requires `Mcp-Method` and `Mcp-Name`; version also rides in `MCP-Protocol-Version`. Source: changelog minor #4 and release blog.
4. **Tasks:** Long-running work moved out of experimental core into the official extension identifier `io.modelcontextprotocol/tasks` (poll via `tasks/get`, client input via `tasks/update`). Source: changelog major #6; versioning extension examples.
5. **MCP extension identifiers** (distinct from A2A): map keys under capabilities `extensions`, **MUST** follow `_meta` key naming rules with a mandatory prefix (reverse-DNS style, e.g. `io.modelcontextprotocol/tasks`). Source: versioning “Extension Negotiation”.
6. **Dual-era servers:** A dual-era server may accept modern per-request `_meta` **or** an `initialize` request that selects legacy semantics. Source: versioning “Backward Compatibility with Initialization-Based Versions”.

## 3. A2A extension identifier convention

### 3.1 Primary sources

| Source | URL |
| --- | --- |
| A2A Protocol Specification (latest index → 1.0.0) | https://a2a-protocol.org/latest/specification/ |
| A2A 1.0.0 specification | https://a2a-protocol.org/v1.0.0/specification |
| Extensions topic | https://a2a-protocol.org/latest/topics/extensions/ |
| Extension and binding governance / URI namespaces | https://a2a-protocol.org/latest/topics/extension-and-binding-governance/ |

Latest released version at verification time: **1.0.0** (linked from https://a2a-protocol.org/latest/specification/).

### 3.2 Official rule: extensions are identified by URI

Quoted from the official Extensions topic, https://a2a-protocol.org/latest/topics/extensions/:

> Extensions allow for extending the A2A protocol with new data, requirements, RPC methods, and state machines. Agents declare their support for specific extensions in their Agent Card, and clients can then opt in to the behavior offered by an extension as part of requests they make to the agent. **Extensions are identified by a URI and defined by their own specification.** Anyone is able to define, publish, and implement an extension.

Agent Card declaration field (`AgentExtension`):

> `uri` — The unique URI identifying the extension.

Service parameter for activation (A2A specification, Standard A2A Service Parameters):

| Name | Description | Example Value |
| --- | --- | --- |
| `A2A-Extensions` | Comma-separated list of extension URIs that the client wants to use for the request | `https://example.com/extensions/geolocation/v1,https://standards.org/extensions/citations/v1` |

Activation behavior (same extensions topic):

> **Client Request**: A client requests extension activation by including the `A2A-Extensions` header in the HTTP request to the agent. The value is a comma-separated list of extension URIs the client intends to activate.

Versioning guidance (same topic):

> Use the extension's URI as the primary version identifier, ideally including a version number (for example, `https://example.com/ext/my-extension/v1`).  
> **Breaking Changes**: A new URI MUST be used when introducing a breaking change to an extension's logic, data structures, or required parameters.

Official namespace for A2A-organization extensions (governance), https://a2a-protocol.org/latest/topics/extension-and-binding-governance/:

> The official URI prefixes are canonical namespace identifiers used to assign globally unique URIs to extensions and custom protocol bindings.  
> Individual URIs under a prefix identify a specific artifact and, where applicable, its version—for example, `https://a2a-protocol.org/extensions/{name}/v1` or `https://a2a-protocol.org/bindings/{name}/v1`.  
> These URIs are identifiers, HTTP access is not expected.

Third-party / project-owned extensions (extensions topic examples) use HTTPS URIs under domains the author controls, e.g. `https://example.com/ext/konami-code/v1`. Official A2A-project extensions use the `https://a2a-protocol.org/extensions/` prefix; MCP++ is not an A2A-core official extension and therefore **must not** claim that prefix.

### 3.3 What is not an official A2A extension identifier

Reverse-DNS tokens **without** a URI scheme (for example `io.mcplusplus.execution@1`) are **not** the official A2A extension identifier form. Official text and examples consistently require a **URI** (typically `https://…`) in:

- `AgentExtension.uri`
- `A2A-Extensions` header values
- Extension metadata keys that namespace under the extension URI

MCP-style reverse-DNS extension IDs (`io.modelcontextprotocol/tasks`) apply to **MCP** capability maps, not to A2A Agent Card / `A2A-Extensions` identifiers.

## 4. MCP++ A2A execution extension URI

### 4.1 Confirmed stable URI

**Confirmed:** `https://mcplusplus.io/extensions/execution/v1`

Rationale against the official A2A rules above:

1. It is a **URI**, matching the identifier type required by A2A.
2. It includes an explicit major version segment (`/v1`), matching the recommended versioning pattern (`…/v1`).
3. It is under a project-controlled HTTPS authority (`mcplusplus.io`), consistent with third-party extension examples (`https://example.com/ext/…/v1`, `https://example.com/extensions/…/v1`).
4. It does **not** use the reserved official A2A org prefix `https://a2a-protocol.org/extensions/`.
5. Breaking changes later **MUST** introduce a new URI (e.g. `/v2`), not silently redefine `/v1`.

No substitute URI is required. The plan default candidate is the verified identifier for Agent Card advertisement and `A2A-Extensions` activation.

### 4.2 DNS / hosting note (non-blocking for the identifier)

On 2026-08-15, a live HTTP probe to `https://mcplusplus.io/` and `https://mcplusplus.io/extensions/execution/v1` failed with DNS resolution failure (`Could not resolve host: mcplusplus.io`). That does **not** force a substitute identifier:

- A2A governance states extension URIs are **identifiers**; for the official namespace, “HTTP access is not expected.”
- The extensions topic still **recommends** hosting the extension specification at the URI when practical, and encourages permanent identifier practices.

Until the domain and specification document are published, implementers treat the URI as the wire and Agent Card constant; document hosting is a follow-on ops item, not a rename trigger.

### 4.3 Alias `io.mcplusplus.execution@1`

| Role | Value |
| --- | --- |
| **Wire / Agent Card / `A2A-Extensions` identifier** | `https://mcplusplus.io/extensions/execution/v1` |
| **Working alias (human / internal only)** | `io.mcplusplus.execution@1` |

`io.mcplusplus.execution@1` remains a **documented alias** for the execution extension: short name in internal docs, issue trackers, and historical working notes. It is **not** a substitute for the URI on the Agent Card or in `A2A-Extensions`. Interoperability text, schemas, and adapters that speak A2A **MUST** use the confirmed HTTPS URI; they **MAY** mention the alias as a non-normative synonym.

Do not invent a second public A2A task lifecycle. A2A already defines Agent Card, Task, Message, Part, Artifact, status, cancel, streaming, and push notifications. The MCP++ execution extension maps MCP++ envelope, state, and receipt objects onto that lifecycle; it does not replace it.

## 5. Cross-protocol contrast (MCP vs A2A extensions)

| | MCP `2026-07-28` extensions | A2A extensions |
| --- | --- | --- |
| Identifier shape | Prefixed key per `_meta` rules (e.g. `io.modelcontextprotocol/tasks`) | **URI** (e.g. `https://example.com/ext/…/v1`) |
| Advertisement | Client/server `capabilities.extensions` map | `AgentCard.capabilities.extensions[].uri` |
| Activation | Capability negotiation via per-request `_meta` / capabilities | `A2A-Extensions` service parameter (HTTP header on HTTP bindings) |
| Task model | Optional Tasks extension (`io.modelcontextprotocol/tasks`) | Core Task object and lifecycle |

MCP++ therefore needs **two distinct spellings** when bridging:

- MCP-side extension / capability keys follow MCP reverse-DNS-with-slash rules.
- A2A-side execution interop uses `https://mcplusplus.io/extensions/execution/v1` (alias `io.mcplusplus.execution@1` only as a human-facing short name).

## 6. Sources consulted (primary)

1. https://modelcontextprotocol.io/specification/2026-07-28  
2. https://modelcontextprotocol.io/specification/2026-07-28/changelog  
3. https://modelcontextprotocol.io/specification/2026-07-28/basic  
4. https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning  
5. https://blog.modelcontextprotocol.io/posts/2026-07-28/  
6. https://a2a-protocol.org/latest/specification/  
7. https://a2a-protocol.org/latest/topics/extensions/  
8. https://a2a-protocol.org/latest/topics/extension-and-binding-governance/  

## 7. Acceptance checklist

| Criterion | Status |
| --- | --- |
| Note states that current MCP is not initialize-based | Yes — §2 |
| Quotes the official A2A extension identifier rule | Yes — §3.2 (URI identification, Agent Card, `A2A-Extensions`) |
| Confirms `https://mcplusplus.io/extensions/execution/v1` or records substitute | **Confirmed** — §4.1 (no substitute) |
| Alias `io.mcplusplus.execution@1` is documented | Yes — §4.3 |
