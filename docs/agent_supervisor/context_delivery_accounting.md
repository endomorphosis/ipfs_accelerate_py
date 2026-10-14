# Context delivery observations

The router now publishes a context observation alongside its existing final
invocation receipt, including failed provider attempts. The observation binds
the rendered user-prompt hash and byte count to native usage; it does not add
anything to the model prompt. Observer failures retain the original provider
outcome and publish an unavailable observation.

The Codex rollout reader extends its existing interaction telemetry with
bounded tool-event fingerprints. Strings use their recorded UTF-8 bytes;
structured payloads use explicitly labelled canonical JSON bytes. Event and
call-identity hashes join existing record ordinals and pairing decisions without
exporting arguments, output text, source paths or unfamiliar tool names. Bounds,
ambiguity and omitted events remain explicit. These fingerprints do not prove
source identity, tool execution or model consumption.

The TerminalBench collector composes existing retained router receipts, native
CLI usage observations and validated source-reader sessions. It opens no private
audit files or databases and invokes no provider. Every supplied invocation and
declared source session remains in its observation population; missing,
ambiguous or malformed evidence stays explicit.

`native_invocations` reports rendered user-prompt measurements separately from
native session usage. Native cumulative snapshots require session-aware
reconciliation, and missing unreceipted-attempt information leaves complete totals
unknown. Tool counters are never added to provider usage or billing totals.
Codex input already includes cached tokens and reasoning is an output subset.
Grok fresh, cached and cache-write input categories remain disjoint. Optional
missing categories stay unknown. Codex fresh input is left unknown because this
contract does not qualify cache-write disjointness. Repeated cumulative snapshots
of one Codex session select a consistent latest observation rather than being
summed; conflicting or regressed snapshots withhold complete counts.

`source_view_retrieval` joins the existing owner observation by the exact UTF-8
invocation-ID hash. Prepared source bytes, bytes from successful owner sends,
candidate pages and discovery pages retain separate count scopes. These are
projections of the previously validated collector; the adapter does not replay
its private audits again. A successful join does not establish that a model
ingested or displayed the response. A successful owner send has no receiver or
model receipt acknowledgement.

`repository_source_mcp` retains prepared reply counters from each selected router
receipt. Those counters include metadata and repeated replies, and have no
delivery acknowledgement. `join_coverage` therefore keeps exact owner-response
to native-output digest matches, ordinary source identity, provider ingestion and
per-tool provider tokens unknown. No observation grants token-savings, billing,
proof or completion authority. Existing invocation and report artifacts own
durability; this stage creates no additional database or service.

The implementation is in
[`context_delivery_accounting.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/context_delivery_accounting.py),
[`native_context_delivery.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/native_context_delivery.py),
and the
[`TerminalBench collector`](../../benchmarks/agent_supervisor/container_coding/terminal_context_delivery_observation.py).
The combined offline qualification passed 492 tests, including unchanged native
usage, unavailable-observer failure preservation, session reconciliation, source
join coverage, counter overflow and full-supervisor planning-failure publication.
No live efficiency advantage is established by these tests.

The next measurement must bind owner responses to actual native tool output,
observe ordinary source reads where independently resolvable, and compare
matched complete sessions. A compact view must replace an identified disclosure
or avoid supported remaining work. Discovery, expansion, history growth and
repairs belong in that comparison alongside official task success.
