# Reduce complete supervisor tokens with native metadata and checked capsules

The objective is fewer total input and output tokens over the entire Terminal
Bench supervisor run, while retaining the planning session and `llm_router`.
Database metadata should let the supervisor reuse established facts, formal
results should identify which obligations remain, and semantic capsules should
give the model the relevant current evidence without repeating its bookkeeping.
Minification is a representation step within that workflow.

The [qualified prior comparison](evidence/terminal-reply-contract-pair-20261007/README.md)
used 246,992 baseline tokens and 327,117 compact-transport tokens. Compact used
80,125 more tokens (32.44%), despite a smaller initial coding input. Coding
accounted for 89.78% of baseline tokens. These observations prioritize coding
context and complete session accounting; they do not establish a causal effect
from one pair. General planning remains enabled.

## Assign each component a precise responsibility

| Component | Role in reducing repeated work | Required boundary |
| --- | --- | --- |
| DuckDB | Store current task/dependency/evidence indexes, accepted native records and capsule locators | A locator or timestamp does not establish source freshness |
| Quack | Read state through the authenticated current native owner | Use a qualified typed operation; the generic gateway remains unqualified |
| DuckLake | Retain observational history for revision/delta comparisons and experiment accounting | Historical facts need current task/source checks before operational reuse |
| Formal systems | Reuse checked receipts and determine remaining supported obligations | Revalidate source, dependencies and assumptions; retain all six native verdicts |
| Semantic capsules | Present current task-specific evidence and residual work compactly | Required code, constraints, scope and uncertainty must remain available |
| Minification | Share identical metadata or encode exact reversible representations | Verify restoration; input bytes are not total provider tokens |
| Embeddings/autoencoders | Rank relevant facts, operators and capsules | Similarity does not authorize omission, proof, execution or completion |

The existing 8D spaCy, 384D and 768D GTE paths can rank metadata and capsules.
A Leanstral hidden-layer representation can be evaluated for that role. The
existing comparison already reused initial indexes with zero new embedding
calls; a new treatment cannot claim to eliminate those already absent calls.
Record inference, refresh and training separately from native LLM tokens.

## First implemented treatment: readable shared metadata

The largest identical measured component in the prior pair was the semantic
context, 31,939 bytes. Retrieval was 4,832 bytes and the Doctor advisory was
2,109 bytes. The Doctor receipts do not expose row-sharing counts, so successor
counts cannot establish the benefit of factoring its rows. The first treatment
therefore operates on the verified semantic context.

[`semantic_metadata_view.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py)
adds the opt-in `common-bindings@1` view. It factors exact identical schema,
version and source-binding metadata into named **inline** common fields for
capsule/admission rows. The original native dictionary remains visible.
It preserves source text, paths, signatures, unknown fields, native core,
requirements, authority, verdicts and suffixes. It reconstructs the complete
original `@1` representation exactly. It refuses the previous `@2` treatment
so the two experiments are not mixed.

The runner constructs both complete coding inputs with the same workspace,
Doctor advisory, public instruction and coding reply instruction. It selects
the view only if the final input is strictly smaller in bytes and under the
explicit `utf8-bytes-ceil-div4@1` proxy. Otherwise it keeps the original input
and records the fallback. This proxy is byte-derived, not the provider's
tokenizer. The actual selected reply contract binds the delivered input; the
undelivered baseline contract is retained separately for reconstruction.
The audit child independently replays these layers from retained native input.

[`semantic_metadata_catalog.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_catalog.py)
adds an owner-prepared catalog bridge. It pins a read-only artifact and compares
independently supplied task/revision/source/context bindings and the exact
native evidence population before native catalog/link registration. DuckLake
projection remains observational. Actual temporary DuckDB registration and
rollback controls were exercised. Active Quack transport and a DuckLake
extension deployment are separate qualifications; this bridge installs neither.
Registration cannot authorize model use or omit a required fact.

The benchmark configuration, Harbor wrapper, driver and runner accept
`semantic_metadata_view=common-bindings@1` explicitly. The original full arm
and `@1` transport are required. The signed implementation route must be the
model router; archive checks bind the current helpers, and preparation drift
refuses execution. Legacy configuration and input shapes remain the default.
Planning and the native validation/publication lifecycle are retained.

Four authored source fixtures show the selection boundary:

| Capsules | Complete original bytes | Candidate bytes | Selected view |
| ---: | ---: | ---: | --- |
| 1 | 11,319 | 11,817 | Original |
| 3 | 18,959 | 18,394 | Common bindings |
| 8 | 38,505 | 35,290 | Common bindings |
| 22 | 93,321 | 82,665 | Common bindings |

These fixtures use one source file; the recorded benchmark's four worker
capsules covered two public source paths. Their common-field populations differ.
The table is offline representation evidence and cannot predict actual
benchmark input size, behavior, rewards or cumulative tokens.

## Next changes, in measurement order

1. **Census the actual prepared coding view.** Let the verified owner emit only
   row counts, shared-field populations, complete input digests/bytes/proxy and
   selected/fallback status. Preserve original source bodies privately; do not
   export them in diagnostic metadata. Do this before spending on a live pilot.
   The [runner census stop point](terminal_metadata_census.md) now constructs both
   complete inputs after current owner checks and stops before provider
   allocation. Administrative source preparation retains explicit authored-plan
   provenance; live planned input remains a separate scope.
2. **Run a separately versioned matched trial.** Keep one planning session and
   one coding session, model/reasoning/resource/setup budgets, original public
   inputs, reply schema and coding transport fixed. Vary this metadata view
   alone. A fallback run does not count as an observed treatment. Preserve both
   native administrative identities and compare reviewed semantic populations.
   Identical controls can still produce different plans. Report that distinction
   and require substantive native input matching for an exact one-factor claim.
3. **Use current formal status capsules to prevent redundant reasoning.** Resolve
   an accepted native receipt through the supported proof-cache gate, including
   render/commit revalidation. Present property, verdict, assumptions, dependency
   digest and residual obligation. Keep proved, disproved, inconclusive,
   unsupported, error and cancelled distinct. Open Doctor work alone cannot
   produce an accepted proof card. Replace repeated status material rather than
   appending a second card.
4. **Qualify a bounded owner lookup for progressive disclosure.** Required facts
   stay inline until the model has an actual qualified expansion interface.
   Typed queries should return an exact bounded population with an explicit
   unresolved frontier, source/task bindings and counted response costs. Avoid
   model-facing arbitrary SQL and assumptions about the generic Quack gateway.
5. **Use checked history deltas at decision boundaries.** Reconstruct a current
   context from an actually delivered parent plus changed facts. Bind parent,
   task, repository/tree, policy, schema and revision. Invalidate changed proof
   dependencies. DuckLake history does not grant authority to rewrite a native
   CLI's internal conversation or discard unresolved work.
6. **Evaluate retrieval separately.** Rank task dependency closures and proof
   obligations with the available autoencoders/embeddings. Measure required-fact
   coverage, missing/stale evidence and expansion cost before claiming omission
   is safe. Keep new training out of the first metadata-only comparison.

## Decide success from the complete run

Count the final cumulative native input plus output tokens for every retained
planning, coding, lookup, continuation and repair attempt. Cached input is
already part of input; reasoning output is already part of output. Do not sum
cumulative usage records as independent calls. Preserve failed attempts and
unknown usage rather than assigning zero.

Report total tokens, planning/coding phase totals, calls/turns, lookup payloads,
repairs, cost and latency alongside the original verifier reward, native
completion/publication and cleanup. Keep embedding/refresh accounting explicit.
Require final input reconstruction and schema custody. Repeat balanced matched
pairs before claiming repeatable savings. This first implementation establishes
representation and integration checks; live token savings remain unmeasured.
