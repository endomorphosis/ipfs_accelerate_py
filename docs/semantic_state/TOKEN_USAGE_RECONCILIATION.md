# Whole-session token accounting for context compression

The existing router records invocation metadata and native cumulative usage.
Use
[`reconcile_router_cumulative_usage`](../../ipfs_accelerate_py/agent_supervisor/runtime/cumulative_usage_reconciliation.py)
to reconcile those observations before comparing a baseline with a compressed
context. The function is pure: it opens no owner, database, transcript, tokenizer,
provider or network connection and changes no input receipt. It returns
accounting observations, with execution, correctness, completion and billing
authority explicitly false.

## Reconcile retained router receipts

In the canonical pinned workspace process:

```python
from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
    reconcile_router_cumulative_usage,
)

# Retain observation order and the complete invocation population, including
# planning, coding, failed attempts and fresh expansion/retry sessions.
accounting = reconcile_router_cumulative_usage(report["provider_invocations"])
observed_subtotal = accounting["known_provider_token_subtotal"]
final_total = accounting["provider_total_tokens"]
```

The initial supported inputs are `router-implementation-invocation@1` with
`native-codex-cumulative-usage@1`. Supply an ordered list or tuple of at most
10,000 receipts. The caller supplies ordering; the function does not invent
chronology from filenames or choose the largest counter. Do not add reports
from independent batches: the same native session may occur in several batches
and would need consolidation before summing.

Invocation identity is distinct from native thread identity. Identical receipts
do not add usage. Several invocations can observe one cumulative session; its
latest nonconflicting counters are counted once. A genuine fresh retry or
expansion session is charged separately, including failed and timed-out work.
Changed model/workspace bindings, conflicting invocation/source aliases,
counter regressions or contradictory metadata quarantine the affected session.
Conflicts are excluded from known subtotals and prevent a complete total; they
are not assigned zero usage or resolved by taking a maximum.

The provider token definition is **input plus output**. Cached input and
reasoning output are subset diagnostics. Local prompt counts, tool-result
attribution and other component measurements are not added again to native
cumulative usage. Missing categories remain unknown. A partial snapshot never
fills its missing fields from an earlier one. Per-category observation history
detects regressions after fields disappear and reappear.

Output interpretation:

| Field | Meaning |
| --- | --- |
| `known_provider_token_subtotal` | Sum of available input/output observations from nonconflicting supported sessions; missing work remains unknown |
| `provider_total_tokens` | Total only when the supplied receipt population has supported, consistent, complete usage; otherwise `None` |
| `supplied_receipt_usage_complete` | Internal completeness of those supplied observations, not proof the campaign retained every call |
| `campaign_usage_complete` | Always unknown here; requires independent population evidence |
| `sessions[*].status` | `complete`, `partial`, `unavailable` or `conflict` |
| `issues` | Unsupported schemas, missing usage or identities that cannot form a supported observation |

Workspace paths are represented by their SHA256 bindings. Arbitrary prompts,
source text, credentials and other receipt fields are not exported. Keep the
full source receipts in their existing admitted store; this projection does
not supply current source or owner admission.

## Lifecycle completeness and old receipts

[`recover_codex_usage`](../../ipfs_accelerate_py/agent_supervisor/runtime/codex_usage_receipt.py)
now distinguishes any observed completion event from completion after the latest
usage/work boundary. Later turns, starts, aborts, user messages and token counts
invalidate an earlier completion. Malformed or regressing usage prevents a
complete lifecycle, while retaining the latest reported observation.

New receipts add:

- `completion_basis = "native_lifecycle_after_latest_usage@1"`
- `lifecycle_malformed_records`
- `usage_lifecycle_complete`
- `task_complete_event_observed`, retaining the historical any-event diagnostic

The existing `task_complete_observed` now describes the final valid lifecycle
boundary. Lifecycle completion does not establish billing, task acceptance,
native source custody or a Lean admit. Missing input/output categories still
prevent an exact token total even when the lifecycle closes.

Old `@1` receipts lack the stronger marker and may have retained an earlier
completion through later work. Their producer-declared flag remains visible,
but the reconciler keeps strict final completeness unknown. Historical files
are not rewritten or silently upgraded. Other provider schemas need explicit
adapters: the retained Grok format has a different cache convention and remains
unsupported in this initial Codex-specific utility.

## Count database context separately

[`compile_manifest_to_capsule`](../../ipfs_accelerate_py/agent_supervisor/context/database_context.py)
accepts an optional `tokenizer` callable or the existing compiler-supported
tokenizer object. Omission preserves its existing fallback and capsule return
type. Forward the explicitly supplied compatible owner's counter:

```python
capsule = compile_manifest_to_capsule(manifest, tokenizer=owner_tokenizer.count)
```

Member nomination costs and manifest identity remain unchanged. Compiler
`capsule.input_tokens` is conservative accounting: it can exceed tokenization
of the canonical provider input rendered by `render_context_capsule(capsule)`
because descriptor/member hints remain
conservative. It is not necessarily an exact final-message count. The router
must still count its entire rendered message after required instructions and
workspace wrappers, using the correct model binding. Invalid counters,
exceptions and mandatory-context overflow refuse; the explicit counter cannot
silently fall back to an estimate or remove required evidence.

The standard Terminal Bench path also accepts explicit tokenizer inputs through
`FullSupervisorAgent`, `terminal_container_supervisor.run` and its CLI, and
`terminal_doctor_dispatch.implementation_argv`. The optional arguments are
`prompt_tokenizer_artifact`, `prompt_tokenizer_sha256` and `max_input_tokens`;
the CLI spellings are `--prompt-tokenizer-artifact`,
`--prompt-tokenizer-sha256` and `--max-input-tokens`.

Supply an absolute **container-local** descriptor path and its exact lowercase
SHA-256 together when overriding automatic selection. Without a descriptor,
the router chooses an offline model mapping, tokenizer proxy or UTF-8 estimate.
A token budget can use that automatic result; estimated budgets cannot guarantee
provider limits. The [launch guide](COMPACT_CONTEXT_LAUNCH.md#automatic-counting-and-explicit-overrides)
describes quality labels, provenance and receipt versioning. Both planning and
model-routed coding forward the binding or standalone budget. The coding
arguments are assembled before the existing signed launch is created. Direct
Doctor candidate routes retain their provider-free argument contract.
Defaults add no flags. Partial bindings and malformed budgets refuse before
the work starts; the existing child loader still checks provider/model, asset
hash, runtime and fresh integrity around each count.

This is forwarding for already deployed local inputs. It does not package a
host descriptor into an archive, stage its tokenizer JSON, install a tokenizer
runtime, discover a compatible asset or establish equivalence to a remote
model. The lexical deployment profile does not currently include that runtime.
Production owner/tokenizer composition and portable asset staging remain open.
Forwarding does not force semantic selection or supply an unavailable model
tokenizer. Automatic proxy and byte counts remain labelled estimates and are
never added to provider usage. It downloads no assets and raises no context limit.

## Final Terminal Bench reports

The driver adds `provider_usage_accounting` after collecting retained router
receipts and identifying an unreceipted planning or coding attempt. Its
`terminal-provider-usage-accounting@1` wrapper contains the pure
`reconciliation`, `known_unreceipted_provider_attempt` and
`report_provider_total_tokens`. The reconciler's completeness continues to
describe only supplied receipts. An unreceipted-attempt marker makes the
wrapper's report total unknown, without fabricating a session or invocation
identity. Campaign completeness remains unknown. Accounting errors retain only
the exception type and cannot change task completion or suppress the original
report. Harbor independently bounds its accounting-error type and preserves
readable task metadata when usage projection fails.

The Harbor adapter's `measured_usage` uses the same session reconciliation in
place of summing cumulative snapshots per allocation. It preserves the public
input/output/cache/total keys and records partial known subtotals separately.
Exact fields require supported complete native sessions and no known
unreceipted attempt. A missing category remains unknown even when other
provider categories are complete. `provider_calls` counts retained allocation
identities; it is not the number of native sessions or a proof that all calls
were retained. Historical completion flags and unsupported Grok formats cannot
become exact final totals through this adapter.

The runnable `benchmarks/agent_supervisor/container_coding/benchmark_comparison.py`
now rederives supervisor trial usage through these same reconciliation and
semantic-dialogue coverage helpers. Nested expansion and retry receipts count
even when absent from the top-level invocation list. Invocation rows retain
individual snapshots for attribution and must not be added as separate charges.
Presence of an `unreceipted_provider_attempt` key, including a false value,
withholds complete report usage. Missing dialogue/owner evidence also withholds
totals while preserving independently usable observed counters.

Distinct trials sharing a native session retain their trial observations, but
their arm aggregate totals and subtotals remain unknown: the reporter cannot
infer a cross-trial observation order or add the same session twice. Unverified
dollar snapshots remain invocation diagnostics rather than cumulative dollar
totals. The separate native-baseline producer keeps its existing completion
contract. This report does not certify billing, task acceptance, or campaign
completeness.

The pure `compare_context_projection` helper similarly refuses paired token
metrics when a native session belongs to multiple assigned tasks within an arm.
It preserves both task observations and the reconciled session charge once.
Repeated or cumulative observations for one task continue to reconcile normally.
The [focused reporter validation](../../../../artifacts/supervisor-token-accounting-20261013-02/README.md)
records 65 actual passed tests, retained-run replay, exact tested/current revision
scope, and the concurrent broader guard failures; it does not qualify the current
complete supervisor pipeline.

The [compact context design](../architecture/terminal_bench_compact_semantic_projection.design.md)
documents the next representation change and the existing world/retrieval
projections. It specifies versioned representations, exact restoration and
required evidence.

## Validation and limits

The fresh guarded run executed **138 distinct tests**: 48 lifecycle/router,
38 database-context/compiler and 52 reconciliation cases. Normal conftests,
nonempty mandatory source seals and fresh artifact-local catalogs were active.
No results were skipped, reused, deferred or degraded; source hashes stayed
unchanged. Counters, native lifecycle files, launch declarations and mocked
provider/container interactions in these tests are authored fixtures.

The subsequent forwarding/reporting integration recorded **115 distinct passed
test calls** under stable source snapshots, using the same mandatory seals and
fresh catalogs. One stale existing preparation fixture was corrected after its
first failure; the full reporting selection then reran. No production success
criterion changed. A six-case observation exercised the actual report and
Harbor accounting helpers on fixed sanitized historical reports.

Another process subsequently edited the semantic codec/router. The recorded
test results are not relabeled as qualification of that newer code. The final
smoke is explicitly accounting-only, guards unchanged declared sources,
captures imported origins and call paths, and requires both edited modules to
remain unimported and uncalled. Fresh import-closure observations outside the
test freeze remain observational. Current whole-pipeline codec/dispatch
qualification requires its own stable-source validation.

That validation subsequently completed in a separate frozen-source campaign:
**518 distinct test calls** passed across codec/projection contracts (162),
dispatch/counter integration (164) and launcher/accounting boundaries (192).
All three runs used the same 163-path source guard, canonical import origins,
normal conftests, nonempty mandatory seals and fresh catalogs. Every selected
test ran once, with no skipped, reused, predicted, verified, deferred or degraded
results. The [compact launch guide](COMPACT_CONTEXT_LAUNCH.md) describes the
opt-in forwarding now exercised by those tests.

Its matched-chunk probe also shows why representation selection must measure
the complete message: the initial estimator-only role view adds byte overhead,
while full-context V2 has fewer bytes in that fixture. Those tests use authored
counters and provider/launcher fixtures; production token counts remain
unknown. See the [current validation evidence](../../../../artifacts/supervisor-compact-launch-20261012/README.md)
for exact source versions, byte measurements and historical replay scope.

See the [initial measurement evidence](../../../../artifacts/supervisor-token-reduction-20261012/README.md),
[integration evidence](../../../../artifacts/supervisor-token-integration-20261012/README.md)
and [remaining implementation plan](../architecture/terminal_bench_token_reduction.plan.md).
No actual model/tokenizer qualification or token savings follows from these
tests. Only an actual `lake build <Lib>` grants a Lean admission. The US
Constitution remains unformalized.
