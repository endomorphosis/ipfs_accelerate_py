# Self-contained residual requests for repository tasks

An existing sealed `ResidualLlmPacket` can focus optional repository evidence
while retaining the complete admitted task and every required reference. This
is an explicit programmatic option on the existing strict world-packet route;
defaults and provider selection remain unchanged. Preparing a request calls no
provider and claims no token saving, resolved obligation or execution authority.

The subsequent [matched residual selection](world_residual_selection.md) option
compares full and focused evidence with identical residual diagnostics. It
retains the full residual-bearing input when counts fail, tie or increase.
The original failure-free packet is never its fallback.

The repository already has whole-tree source representations. The earlier
[`repository_source_view.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_source_view.py)
packages bounded original and compact trees and can materialize either view.
[`repository_source_revision.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_source_revision.py)
constructs checked successor revisions through explicit symbol replacement,
file creation/replacement, deletion and rename operations. The newer
[reversible source views](reversible_source_views.md) retain committed originals
and exact KEEP/DELETE maps, with guarded compact-span edit mapping.

Representation roundtrip uses retained original bytes and mapping metadata.
It does not supply a general inverse for an arbitrarily edited minified tree,
prove executable equivalence, or turn a compact display into the canonical
execution checkout. Whole-tree packaging therefore already exists; universally
reversible arbitrary minified editing remains a separate problem. Neither
capability guarantees fewer tokenizer tokens or less conversation history.
The [source read/edit measurement](source_view_read_edit_pair.md) and
[compact worker benchmark](compact_worker_benchmark.md) explain the relevant
measurement boundaries and why discovery, expansions and repairs matter.

## Prepare focused evidence through existing owners

[`prepare_repository_residual_context`](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_residual_context.py)
reconstructs the sealed packet's identity and detaches its nested data. It
requires exact task, repository, tree and nonempty policy bindings; write paths,
validation commands and acceptance IDs must match the independently supplied task
populations in their original order. It reuses
`ResidualProviderInvocation(require_path_lease=True).prepare()` for the existing
body-free renderer, path checks and prompt bounds. Its structural path allow-list
does not issue a native lease. Authority-root references remain inert data.

The frozen binding exposes detached `.packet`, `.context`, `.receipt` and
`.record` properties. The model receives the existing rendered nomination
context once, as an additional failure-evidence member. The complete original
sealed packet stays in the private historical recipe, with independent packet,
context and record pins. Original task obligations are not replaced by the
packet's residual obligation IDs. Secret-shaped or source-body material,
unknown noncanonical packet fields and silently transformed residual context
are refused.

For a current, provider-eligible `PreparedRepositoryWorldPacket`, use
[`prepare_repository_world_residual_request`](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_world_residual.py):

```python
focused = prepare_repository_world_residual_request(
    baseline=current_strict_packet,
    residual_packet=sealed_residual_packet,
    query="bounded arithmetic overflow",
    unresolved_question="Which current symbols implement the failing bound?",
    limit=4,
    max_bytes=16_384,
)
candidate = focused.packet
preparation_observation = focused.receipt()
```

The caller supplies the query and question. The helper recompiles through the
baseline's current source session and context compiler. It preserves exact task
text, repository/tree and objective bindings, policy, goal, task authority,
scope, acceptance, allowed effects, write paths, original unresolved obligations,
validation commands, proof obligations and every required reference body and
coverage identity. The existing compiler's retrieval-population metadata can
change with the query. Optional references may be selected, omitted, added or
rebound; the observation identifies those populations without exposing bodies.

This optional selection is advisory. It does not prove omitted evidence
irrelevant, establish source semantics or complete repository coverage, or
authorize dropping required facts. There is no automatic choice based on bytes
or estimated tokens. A candidate can be larger than its baseline; if bounds
cannot accommodate the full required input, preparation refuses. A read-only
observation packet and an already residual-focused packet cannot serve as this
helper's baseline.

## Dispatch, retain and replay

The candidate is an ordinary issued strict packet. Use the existing runner or
the [durable attempt wrapper](world_packet_attempt_retention.md), in the admitted
workspace, with a separately issued journal for the declared candidate slot:

```python
output, invocation, retention = run_repository_world_packet_attempt(
    journal=issued_candidate_journal,
    runner_options=dict(
        prompt=exact_original_task_text,
        provider="codex_cli", model=pinned_model,
        timeout=180, max_output_tokens=4096,
        world_task_packet=candidate,
    ),
)
```

The journal's existing start, receipt and terminal retention rules apply.
Inspect retention failures; they do not imply an uncharged call. Perform output
verification and historical replay after durable capture. This API does not
add a general Harbor configuration or combine incompatible runner transports.

Preparation fences both baseline and candidate through their original live
owners after rendering. Dispatch retains the runner's closing source and intent
checks. These are cooperative checks, not a lock held across worker execution.
The detached residual snapshot cannot change when its original nested input is
later mutated. `validate_repository_residual_context` recomputes its canonical
binding; `replay_repository_residual_context` checks the detached record against
independently supplied task bindings. Whole-packet replay additionally requires
the independently pinned artifact and reconstructs exact component pins, native
packet, prompt and receipt. Historical records cannot restore live issuance,
source freshness, native signatures or dispatch authority.

Every request is self-contained. No retained provider parent or conversational
memory is assumed. A local context delta, world projection, authenticated worker
acknowledgement or matching hash does not establish provider ingestion, formal
proof or completion. Existing world projections remain advisory; this option
creates no new world state, authority root or proof system.

## Count complete input and complete sessions

The residual receipt's byte-based token estimate covers only its sealed context,
excluding the full task and runner wrappers. It is not native usage. The runner
still measures the complete rendered input. The separate
[local-tokenizer metadata selection](metadata_tokenizer_selection.md) documents
complete-prompt comparison and explicit local-proxy qualification; its metadata
flags are not an additional option on the strict world-packet route. Smaller
payload bytes alone establish neither tokenizer equivalence nor workflow savings.

Retain all declared attempts and use the existing
[context-delivery observations](context_delivery_accounting.md) and
[session accounting](world_benchmark_session_accounting.md). Reconcile cumulative
snapshots of one native session once, preserve failed attempts and unknown
coverage, and keep provider-specific cache categories distinct. Compare complete
workflow usage alongside independent task verification before changing defaults.

## October 15 qualification

Qualification passed 617 distinct binding, focus, native context, strict packet,
historical replay, dispatch and attempt-retention tests, with no skipped cases.
Twelve qualification leases were released, and the existing native scheduler
policy remained unchanged. Initial malformed counterexample fixtures, eager
fixture defaults and two failed finite qualification attempts remain in the
private evidence directory; production admission checks were retained.

Two authored finite inputs exercise genuine source indexing, a real repository
MCP child and guarded symbol replacement through the existing router and durable
journal. The provider callback and CLI configuration observation are scripted.
Each candidate repairs seven failing arithmetic cases and passes all 91 checks
over the declared integer domain `[-3, 3]`, preserving helper and support bytes.
Captured router input matches the complete reconstructed prompt; both packets
replay historically, and both old live packets refuse after the source edit.

The complete input comparisons include workspace and tool guidance, full source
and acceptance requirements, and the added residual context:

| Authored baseline retrieval | Optional references before/after | Complete bytes before/after | Local proxy tokens before/after |
| --- | --- | --- | --- |
| Already narrow | 1 / 1 | 14,328 / 16,339 | 4,329 / 4,911 |
| Broader symbol query | 8 / 1 | 24,849 / 16,339 | 8,178 / 4,918 |

Counts use the cached `o200k_base` tokenizer as a local proxy. They establish
neither the provider's tokenizer nor remote framing, ingestion, cache charges,
output quality or whole-workflow savings. The narrow case grows because it
adds diagnostics without removing optional references. No automatic selection
or default change follows from these fixtures. Both use an explicitly
unavailable world projection; no accepted world authority is manufactured.
No live model calls, Harbor trials, training or downloads were launched.

Private scripts, source snapshots, attempt records, failed attempts and the
qualification manifest are retained at
`/home/barberb/lift_coding/artifacts/world-residual-request-20261015-01`.
The qualification manifest SHA-256 is
`73147ad9be58cc01adb352aafd0f7b86a632ddb8a1fca91898432c3633e287d2`.
