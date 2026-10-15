# Bounded sessions for Python code views

The session layer reuses an exact alias and slot legend across reads of the
[bounded Python code view](PYTHON_CODE_VIEWS.md). Its pure codec restores the
entire [WIRE packet](PYTHON_CODE_VIEW_WIRE.md); a task-local broker checks the
current native producer and controls references through explicit caller
acknowledgements. Canonical source stays native Python.

These are explicit library APIs. They do not register a model tool, select a
production prompt representation, or add a DuckDB, Quack or DuckLake owner.
Transport integrity and an acknowledged delivery do not establish model
consumption, semantic equivalence, task acceptance or proof.

## Pure codec

The [session codec](../../ipfs_accelerate_py/agent_supervisor/semantic_refactoring/python_code_view_session.py)
exports:

```python
legend, state, stats = split_python_code_view_session(wire_packet)
wire_packet = restore_python_code_view_session(legend, state)
legend = parse_python_code_view_session_legend(canonical_json_text)
state = parse_python_code_view_session_state(canonical_json_text)
```

The exact schemas are `supervisor-python-code-view-session-legend@1` and
`supervisor-python-code-view-session-state@1`. Both require profile
`python-arithmetic-model-packet-session-columns@1`.

| Object | Exact fields |
| --- | --- |
| Legend | `schema`, `profile`, `legend_id`, `bindings`, `functions`, `slots` |
| State | `schema`, `profile`, `view_id`, `legend_id`, `source`, `hashes` |

`bindings` and `functions` retain the WIRE rows exactly. Each legend slot is
`[slot_id, function_index, kind_index, editable, refusal_reason]`. Each state
contains the literal compact source and the ordered full SHA-256 of every
original expression. It retains the full view CID and full legend CID. No
identifier, hash, literal, semantic original name, ordering or refusal is
abbreviated or discarded.

`legend_id` is the canonical DAG-JSON CID of every legend field except
`legend_id` itself. Restoration recomputes that CID and requires exact equality
with the state's reference. An expression-only edit can reuse the legend when
its bindings, functions, aliases and slot definitions remain exactly unchanged.
Metadata changes require a new legend. Returned dictionaries are independent
copies; modifying a legend under its old identity causes refusal.

The codec checks closed schemas, exact raw types, row widths, integer indices,
source inventory and full canonical CID encodings. Booleans cannot stand in for
indices. Raw JSON parsers reject duplicate fields and noncanonical JSON bytes.
They enforce the existing literal byte, node and depth envelopes; restoration
measures virtual WIRE and MODEL expansion before constructing repeated alias
or slot fields. The engine's source, token, AST and artifact limits stay intact.

This is representation validation. A syntactically valid substituted view CID
or expression hash is not authenticated by restoration. The native adapter
still compares the entire expanded public packet with the real captured view.
Neither a legend nor a state contains private original source or grants source,
execution, proof, qualification or completion authority.

## Task-local broker

The [broker](../../ipfs_accelerate_py/agent_supervisor/runtime/python_code_view_session_broker.py)
is constructed around one existing isolated worktree:

```python
broker = PythonCodeViewSessionBroker(
    worktree=worktree, scope=scope, session_id=session_id,
    lease_id=worktree.lease_id, fence=worktree.fence, ttl_seconds=120,
)
```

It binds the repository root, isolated path, task, attempt, base commit/tree,
lease/fence and exact patch scope. Every operation checks that binding, current
ownership, active native lifecycle and session deadline. Source reads use the
existing confined, no-follow filesystem reader and verify the complete native
preimage, including bytes outside the proposed expression. A broker has its own
identity: another broker with the same caller session label cannot reuse its
delivery or acknowledgement records.

| Method | Purpose |
| --- | --- |
| `read_view(path=..., session_id=..., known_read_id=None)` | Capture and deliver a legend/state snapshot, or return an acknowledged exact-current snapshot reference. |
| `acknowledge_legend(legend_id=..., read_id=..., session_id=...)` | Record explicit caller transport feedback for a legend actually emitted by this broker/read. |
| `acknowledge_read(read_id=..., view_id=..., session_id=...)` | Permit snapshot references after matching legend acknowledgement. |
| `restore_read(response=..., session_id=...)` | Restore and compare the complete packet with the current native captured producer. |
| `deliver_native_source(view_id=..., session_id=..., known_delivery_id=None)` | Read and deliver actual native filesystem source, or return an acknowledged exact-current native reference. |
| `acknowledge_native_source(delivery_id=..., view_id=..., session_id=...)` | Record explicit caller feedback for that broker's delivered native preimage. |
| `prepare_edit(wire_request=..., visible_sources=..., session_id=...)` | Reconstruct and validate the native expression patch under delivery and visibility gates. |
| `apply_edit(prepared=..., visible_sources=..., session_id=...)` | Freshly revalidate the same native candidate and invoke the existing patch executor. |
| `close_session(session_id=...)` | Clear retained handles and encoded state; repeat closure is idempotent. |

Emission alone permits no reference reuse. A fresh read includes its legend
until explicit legend acknowledgement. A snapshot reference additionally needs
explicit read acknowledgement. Reference reads still capture and check the
current source; they do not trust a cached private original as the producer.
Acknowledgements describe caller-reported transport history, never inferred
model reading or understanding.

Native visibility has a separate gate. `deliver_native_source` performs an
actual confined filesystem read and returns its literal body and full hash.
It does not construct delivery evidence from private retained original source.
Preparation and application require both acknowledgement of that exact current
native delivery and caller-provided `visible_sources` containing its exact
native preimage. The mapping passes through to the existing native validator.
A private artifact, compact-only source, delivery without acknowledgement, or
acknowledgement without matching caller visibility cannot satisfy this gate.
The controller must ensure its feedback and visibility mapping represent real
transport to the model.

The engine bridges repository identity into the pinned software-contract
frontend through an explicit content-addressed private repository record ID.
The actual repository binding is retained separately and compared by the
worktree adapter. This allows safe source-view provenance for another
repository without inventing a replacement repository identity. It does not
permit reusing a view, write scope, lease, fence, base or visibility evidence
across repositories.

### Peer whole-repository storage boundary

The local peer branch `codex/repository-source-view-20261012` contains
`repository_source_view.py` at commit
`1aa0b832f1974799c785258172e8527f4add9b9c`. It packages exact original file bytes,
optional compact representations and a pinned manifest for bounded restoration
into a separate destination. Its renderer dependency, `code_source_view.py`,
is absent from this canonical checkout. This guide records an integration
candidate; no peer module has been imported, copied or merged here.

That private exact-original package could later support storage and transfer,
after its manifest, repository/base identity, per-file native preimages and
current worktree identity are explicitly joined and rechecked. Retained package
bytes alone are not current filesystem source or model visibility. Keep writes
through this guide's scoped, fenced native mediator: the peer MCP path writes
directly without these patch-scope, fence, native visibility and pinned-datasets
producer gates. Reusing its storage representation does not authorize reusing
that write path. Peer capture-time AST checks and historical test counts do not
combine with this session's executed qualification or grant a Lean admit.

## Retention and mutation limits

Default lifetime is 120 seconds; the maximum is 300 seconds. The broker retains
at most four views, four legends, sixteen read deliveries and sixteen native
deliveries, subject to an 8 MiB encoded-retention cap. These limits count
canonical encoded private artifacts, legends and delivery records. They are
not Python heap, resident memory or RSS guarantees. Budget refusal occurs
before publication; expiry, closure and invalid owner/scope bindings stop reuse.
There is no persistent session database or background eviction thread.

A successful apply invalidates old views, read references and native delivery
acknowledgements for the affected path. An unchanged legend may survive, and
the broker can capture the post-edit source. This does not make repeated writes
on a dirty base admissible: the existing native executor requires a clean bound
base. Plan one successful apply per clean base. Further mutations require the
controller's ordinary validated base-establishment workflow and a fresh bound
session. The broker does not commit, reset, rebase or weaken the executor's
cleanliness checks. Diagnostic fixture resets are not a production workflow.

Only native reconstructed source reaches the patch executor or fixture
execution. The controller keeps responsibility for cleanup, quarantine and
ordinary acceptance after a candidate is applied. Session closure grants no
completion. This layer adds no atomic filesystem transaction or source
equivalence proof.

## Whole-session measurement

The [broker tests](../../test/api/test_python_code_view_session_broker.py)
compare scripted matched sessions using the same initial task, expression
broker setup, captured legend/state and final native edit. Strategies are full
native reads, acknowledged native references, and acknowledged compact snapshot
references. This isolates read transport; it is not a generic native-editor
benchmark or a model decision-quality experiment.

The comparison counts rendered request and response frames, explicit
acknowledgements, the shared initial expression setup, required native-source
delivery, edit, apply feedback, post-edit read, diagnostic and closure. Required
native expansion is charged even when compact references supply the earlier
reads. Complete transcript bytes and cumulative rendered history-prefix bytes
are separate diagnostics. Prefix sums are not claims about provider request
counts, prompt caching or billing.

The fixtures execute each resulting native edit against independent expected
arithmetic results and compare final native hashes. Canonical repository bytes
remain intact. No real model/provider is called: exact model token counts,
tokenizer compatibility and provider-billed usage remain unknown. Byte savings
on repeated reads do not establish total token savings or improved task
success, and a one-read session may pay more for the additional metadata.

Production use still needs actual tool registration, a compatible tokenizer,
real model sessions, truthful transport feedback and unchanged task acceptance.
All source expansions, searches, diagnostics and retries must be counted in
that trial. Keep native and compact strategies in competition under the same
success criteria.

These Python transport, patch and fixture checks grant no Lean admit. Only an
actual `lake build <Lib>` grants that admit. This layer performs no Lake build,
autoencoder training or weight download; the Constitution remains unformalized.
