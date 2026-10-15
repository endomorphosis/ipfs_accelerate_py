# Semantic context projection and token accounting

This opt-in router path keeps the canonical repository and complete worker
artifact unchanged. It derives a decision view, verifies its parent against the
current native producer and sources, and compares complete rendered alternatives
with the original message. It does not put the repository into a minified state.

## What the initial policy defers

`estimator-telemetry-only@1` defers only `pack.token_estimate` and
`pack.pack.token_totals`. One existing `ContextReference` identifies the grouped
diagnostics and their original parent. Every selected capsule, admission,
required raw source, manifest binding, objective, assumption, exclusion, risk,
obligation, counterexample, reconstruction limitation and refresh record remains
in the required view. Original requirement coverage identifiers are preserved.

The view uses `supervisor-semantic-worker-role-view@1` and declares its exact
original artifact, SHA256, task, semantic root, source scope, repository and
tree. It is a derived representation, not a shortened canonical producer
artifact. Its complete parent remains available in the existing store.

A view with altered source, a missing capsule, changed coverage, an unfamiliar
field or a foreign parent is refused by deterministic regeneration from that
parent. Current producer, source inventory and retained-source checks run at
the router boundary. A same-HEAD dirty source edit invalidates the view.

## Enable comparison at the router

The existing `router_implementation_runner` API and command support:

```text
--semantic-repository /path/to/canonical/repository
--semantic-slot-policy-revision closed-typed-identifiers@2
--semantic-context-projection
```

Use these with the existing allocated-worktree launch and an explicitly
supplied compatible `--prompt-tokenizer-artifact` / `--prompt-tokenizer-sha256`
binding. An optional `--max-input-tokens` retains the existing message ceiling.
These flags provide no asset staging, model compatibility, source admission or
launch permission. They must be part of the caller's ordinary launch command.

The [compact launch guide](COMPACT_CONTEXT_LAUNCH.md) shows the equivalent
Harbor and supervisor options, their signed-command forwarding, and the
planning/provider-free boundaries.

The defaults remain `legacy@1` and projection disabled. Opting into V2 compares
legacy aliases and full-context V2 aliases, plus a lossless reference-column
alternative when eligible. Enabling projection adds the role view with V2
aliases and its eligible packed alternative. The router measures every complete rendered message,
including instructions, workspace guidance, translation tables and suffixes.
It chooses the smallest eligible alternative only when it uses strictly fewer
tokens than native. Ties between semantic alternatives prefer legacy, then
full-context V2, the role view, packed V2, then packed role view. Missing, failed or inconsistent counter
results keep native context; a strict token ceiling requires valid counts.

The invocation receipt distinguishes the requested projection from the chosen
representation: `semantic_context_projection_selected` says whether the role
view was sent, `semantic_slot_policy_selected` records the selected codec, and
`prompt_selection.comparison_candidates` retains measurements for all variants.
Constructing a view does not imply that sending it saves tokens.

Packing is a separate transport revision, preserving the existing V2 table/CID
and every reference value. See the [reference wire guide](REFERENCE_WIRE_PACKING.md)
for exact restoration, bounded expansion and selected-packing telemetry. A role
view can still be larger than its full-context counterpart after packing.

The new codec table has schema `supervisor-semantic-translation-table@2` and
an explicit `closed-typed-identifiers@2` slot policy. It covers additional
known identifier locations. Source code, comments, literal values, names,
paths and prose stay literal. Existing `@1` tables retain their identities and
legacy policy. Unknown revisions refuse; an old table is never silently
interpreted using the new policy.

Structured identifier replies still enter the existing native candidate
grammar. Ordinary prose and code are not globally rewritten. Reversible
representation does not certify program equivalence or task completion.

## Resolve deferred diagnostics

`semantic_context_projection.expand_semantic_context_projection` accepts a
selected reference identity, exact `SemanticProjectionBindings`, explicit byte
and token bounds, a counter, and a caller-owned resolver. Bind the resolver to
the current native loader rather than a bare filesystem read:

```python
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    load_semantic_worker_context,
)

def resolve_original(**binding):
    return load_semantic_worker_context(repository=canonical_repository, **binding)
```

Expansion verifies the current full parent before extracting the selected
diagnostics, measures the returned reference descriptor, and reloads the parent
after counting. Changed parents, stale sources, unknown handles, cancellation
and exceeded bounds refuse. The caller must still measure the complete message
in which the expanded reference will be used.

This API is a bounded resolver seam. It does not register a model tool or a
second database owner. Future projection policies can use the same compiler
and reference interfaces after their required-coverage rules are established.

## Measure the complete task

`context_projection_comparison.measure_context_candidates` compares complete
rendered native, projected and aliased user messages. It records hashes,
bytes, local content-token counts and the supplied counter binding, without
retaining prompt bodies. An authored fixture counter is labeled as a fixture.
This measurement excludes provider instructions and later tool turns.

`context_projection_comparison.compare_context_projection` pairs the same
assigned task, provider/model, source tree and acceptance bindings. Supply all
observed planning, coding, retry and expansion attempts, including failed and
interrupted work. It delegates usage accounting to the existing cumulative
session reconciler, counting a native session once rather than summing its
cumulative snapshots. Cached input and reasoning output remain subsets; local
counts and tool-result attribution are not charged a second time.

Changed bindings, unmatched outcomes, missing or conflicting usage, shared
sessions between arms, empty accepted populations and authored usage fixtures
cannot yield an eligible provider-savings claim. Campaign completeness remains
unknown: a caller's complete-population declaration is not independently
verified by these helpers. Task acceptance, billing and usage authentication
also remain outside their authority.

See [whole-session accounting](TOKEN_USAGE_RECONCILIATION.md) and the
[implementation design](../architecture/terminal_bench_compact_semantic_projection.design.md).
The implementation and offline contract tests alone do not establish reduced
Terminal Bench tokens. A matched run needs the production model/tokenizer
binding, retained usage for all attempts and the same task acceptance checks.

## Offline byte probe

A retained fixture probe used the real worker-context producer, codec and
rendering path, with no provider call or model tokenizer. Its complete rendered
message sizes were:

| Representation | Bytes |
| --- | ---: |
| Native | 57,550 |
| Legacy aliases | 32,694 |
| Full-context V2 aliases | 32,004 |
| Plain role view | 42,740 |
| Role view with V2 aliases | 33,324 |

The role view with aliases was 1,320 bytes larger than full-context V2 aliases.
These transformations therefore do not automatically compound. The plain-view
comparison also changes reference chunk sizes from 1,201 to 6,144 bytes, so its
byte reduction cannot be attributed solely to deferred diagnostics. Every
model-token count remains unknown; these byte sizes are fixture observations,
not provider-token savings or a Terminal Bench result.

A subsequent controlled fixture keeps both original and projected references
at 6,144-byte chunks and retains the same task core, literal suffix, source and
parent verification. Its complete rendered messages were:

| Representation | Bytes |
| --- | ---: |
| Native, matched chunks | 40,492 |
| Plain role view, matched chunks | 43,527 |
| Full-context V2 aliases | 32,006 |
| Role view with V2 aliases | 33,324 |

The initial role view adds 3,035 bytes to native and 1,318 bytes to full-context
V2. Deferring these small diagnostic fields does not recover the view's wrapper
and reference overhead. Keep projection opt-in and select candidates using
complete-message counts; prioritize measured lossless encoding and reference
packing before broadening deferral. Every required capsule and source field
must remain available under the same coverage checks. See the
[controlled validation evidence](../../../../artifacts/supervisor-compact-launch-20261012/README.md).
These measurements use no model tokenizer or provider and grant no token,
task-success or formalization qualification.

The later [reference-packing campaign](../../../../artifacts/supervisor-reference-packing-20261012/README.md)
adds eligible packed alternatives and validates them separately. Its same-case
messages were 32,006 bytes for full V2, 30,122 for packed V2, 33,326 for projected
V2 and 31,442 for packed projected V2. The role view still adds 1,320 bytes after
packing. The 237 fresh passing cases and exact historical replay preserve
coverage and compatibility contracts; provider-token savings remain unknown.
