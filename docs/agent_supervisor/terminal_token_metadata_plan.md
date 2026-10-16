# Reduce complete supervisor tokens with native metadata and checked capsules

The objective is fewer total input and output tokens over the entire Terminal
Bench supervisor run, while retaining the planning session and `llm_router`.
Database metadata should let the supervisor reuse established facts, formal
results should identify which obligations remain, and semantic capsules should
give the model the relevant current evidence without repeating its bookkeeping.
Minification is a representation step within that workflow. The semantic world
model should maintain current task knowledge outside the model conversation;
the context compiler should project only what the next decision requires.
The [world-model integration sequence](#semantic-world-model-integration-sequence-2026-10-14)
below distinguishes the existing components from their remaining worker-loop
integration and measurement gates.

## Working decision: optimize the whole reading and decision loop

The October 14 source audit separates three questions that must not be treated
as the same requirement:

1. Can a bounded admitted repository have an exact original representation and
   a compact representation? Yes. The snapshot package exports both trees;
   retained-source inventories and byte-map views also support exact historical
   restoration. These are distinct implementations with different codecs and
   scope limits.
2. Can a worker navigate and edit using those representations? Yes for the
   existing admitted read/expand/guarded-edit contracts. Arbitrary edits to a
   multilingual minified executable checkout have no general inverse. Ordinary
   shell reads still access original source unless a separately qualified
   navigation surface redirects them.
3. Does that lower complete-run tokens? This is still unestablished. A compact
   representation helps only when it replaces model-facing text or avoids
   later work. It can add discovery, guidance, expansion and repair instead.

The next implementation priority is the **existing world-packet and indexed
source workflow in Terminal Bench**, with explicit accounting for which content
it replaces. The existing historical replay qualification is a prerequisite
for that comparison; it does not shrink a packet by itself. Its current
self-contained builder retains every selected reference body. Factor or select
content through a separately qualified context policy, preserving mandatory
facts and available exact expansion.

Use the existing owners to retain full source, graphs, maps, IR bodies and proof
artifacts. Compile a readable decision context from their current task-bound
facts: objective, applicable constraints, relevant source, known outcomes,
uncertainty and remaining obligations. IntentIR supplies requested behavior;
CodebaseIR supplies source correspondence and dependencies; SecurityIR and UI/UX
IR supply applicable properties with explicit unavailable coverage. Learned
CodebaseIR and world predictors can nominate useful material or actions. Their
latent representations need a grounded decoder or retrieval interface to affect
the coding model, and predictions cannot stand in for checked observations.

### Keep the two existing worker source routes distinct

| Route | Current scope and representation | Composition boundary |
| --- | --- | --- |
| `WorkerSourceViews` / `source_view_profile` | At most 64 admitted semantic-manifest files and 1 MiB; retained originals and KEEP/DELETE views, with separately selected mapped-span edits | Real opt-in worker route; currently rejected alongside `world_task_packet` |
| `RepositorySourceSession` / `repository_source_policy` | Default 512 files and 16 MiB, bounded repository snapshot and native index; original/compact file or supported definition reads and guarded canonical edits | Works with world packets and retained basic-context Terminal collection; production worker handoff remains pending |

Neither route intercepts arbitrary terminal tools. Neither establishes that the
model read every path through its compact representation. Reuse the second
route for the immediate world-packet experiment. Broader scope admission,
cross-language mappings and a transparent compact filesystem are separate
extensions; building another snapshot codec would not close the measurement
gap. The [source-view plan](reversible_codebase_views_plan.md#current-worker-routes-and-the-whole-codebase-hypothesis)
records the different restoration and editing contracts.

### Next experiment and acceptance gates

The collector now has a dedicated retained world-input path, described below.
Next connect production worker preparation and dispatch to current native owners,
including the selected formalization and signed scope contracts. Historical
replay must remain separate from new live owner admission. Preserve complete
native invocation accounting; build and probe a runtime archive containing the
selected implementation before trials. Then use these controls:

| Arm | Source access | Decision context | Question answered |
| --- | --- | --- | --- |
| A | Current ordinary tools | Current context policy | Reference behavior and ordinary-read exposure |
| B | Indexed gateway, exact source | Same context policy | Does selective navigation avoid irrelevant reads? |
| C | Same indexed gateway, compact source with exact fallback | Same context policy | Does representation alone add a benefit? |
| D | Same indexed gateway, exact source | Current-world decision context | Does selected state and checked reuse reduce repeated work? |
| E | Same indexed gateway, compact source with exact fallback | Same current-world policy | Does compaction help after context selection? |

Qualify B/C and B/D as separate matched comparisons before evaluating E.
Required task instructions, applicable mandatory constraints, model, source
revision, planning policy and resource controls stay fixed within each pair.
Keep model-driven planning enabled; record realized plans and coding behavior,
which can still vary. Learned ranking/adaptation is a later independent arm.
The current world-packet runner supports the Codex coding route; use that same
pinned route for its controls until another provider adapter is qualified.

For each actual read, retain its generation, operation, representation, exact
fallback or expansion, and complete model-visible response size; count declared
local tokens only when a compatible pinned counter is available. Retain ordinary
tool reads as well where the provider exposes them, and report unknown coverage
otherwise. Repeated reads and responses already supplied inline must remain
visible. Trace bytes and local token counts explain exposure; they must not be
added again to native invocation usage.

The primary result is total native planning, coding and retry input/output
tokens at comparable official task success across preregistered repeated trials.
Respect each provider's final/cumulative and cache accounting, retaining failed,
deferred and incomplete attempts. Also report turns, expansions, repairs,
latency, cost and cold versus warm index/training/check overhead. Existing
resource admission and provider cooldown apply to every arm. A smaller archive,
source file, first prompt or replay artifact is insufficient for promotion.

## Retained measurements
The [October 13 complete-context comparison](context_projection_pair.md) now
measures the existing lossless metadata projection with identical source reads.
It saves 885 of 79,362 one-pass exposed UTF-8 bytes (1.12%); both public program
bodies were already deferred. This authored-hook Docker result has no provider
token count or official reward. A separate completed GPT-6.1-sol native pair
used 313,979 versus 313,980 tokens, with rewards 0 and 1. Thus the smaller input
did not reduce total usage in this pair. Its retained evidence includes the
Grok credential-readiness failure before generation. Both native trials lacked
an IntentIR checkpoint and had zero verified Source384 source candidates.

The [October 12 full coding pair](evidence/terminal-instruction-dedup-pair-20261012/README.md)
passed both official tasks. Instruction deduplication used 309,813 versus
311,776 total native tokens (0.63% fewer), principally cached input; uncached
input fell by 32 and output increased by 245. Agent execution and setup took
longer. This single independently planned pair establishes no repeatable token
or cost advantage. The existing extended formalization budget now works with
pinned setup-cache handling. Actual Source384 coverage remained 0/4 aligned,
and IntentIR failed open without a checkpoint. A separate deterministic
numerical probe linked all four functions, structurally translated two, and
replayed the real DuckLakes; generated Lean compiled, but all 45 source
obligations remain pending. The subsequent
[bounded obligation retrieval](numerical_proof_admission_readiness.md#current-structural-obligation-retrieval)
keeps complete kind counts in the planning summary and exposes exact,
receipt-bound pages of full unresolved rows. It reuses existing reports and
lakes within the same summary bound. Planning remains a single request;
automatic model tool use, wider grammar and token savings still need separate
qualification.
Its [source-bound qualification](evidence/numerical-obligation-retrieval-20261012/README.md)
recovered all 45 pending public-source obligations across 12 bounded pages,
replayed them in a fresh process with the existing DuckLakes, and passed 263
local tests. A subsequent owner-mediated worker service now lets the isolated
coding CLI request bounded pages during its existing `llm_router` session.
Discovery guidance is part of the complete coding input; per-session accounting
is collected before container removal. Its
[provider-free qualification](evidence/numerical-obligation-worker-20261012/README.md) is
separate from a live model's tool choices and complete-run token savings.

For repository-wide source views, separate three requirements: exact original
restoration, preservation of accepted edits, and measured complete-run tokens.
The JSON/metadata codecs restore their transports. The newer repository source
package also captures bounded whole-tree snapshots, retaining original bytes
and exporting original or compact trees. Validated revision operations and the
indexed definition read/edit workflow preserve canonical editing. See the
[current source-view qualification](compact_worker_benchmark.md#follow-up-indexed-definition-reads-in-the-worker-edit-loop).
Arbitrary edits to a multilingual compact checkout still have no general
lossless inverse. A historical read-only Python preview used only removal
hashes; that older limitation should not be attributed to the newer retained
original-source package.
The [repository-wide compact-view plan](reversible_codebase_views_plan.md)
specifies exact source custody, bounded source access, original-source editing
first and an optional mapped-edit profile later. A reversible view is useful
for source fidelity; its guidance, expansion and repair costs still belong in
the total-token comparison. Reuse the current ranked CLI/successor refresh and
retained embeddings rather than restarting their work.
The [retained committed-source inventory](repository_source_inventory.md) now
implements exact historical byte custody through the existing content store.
Its host metadata pages retain complete committed scope and Gitlink frontiers;
they do not replace current required source or qualify a model expansion tool.
The [reversible read-only source views](reversible_source_views.md) additionally
retain complete byte maps for exact unchanged-source restoration. Raw candidate
byte reductions exclude guidance, sidecars and tool costs; edited views refuse.
An opt-in [worker source reader](worker_source_views.md) now connects compact
navigation and exact-source expansion to the isolated `llm_router` session.
Its working-candidate mode refreshes source views after edits and requires the
same candidate generation for continuation or expansion; it does not regenerate
formal evidence or permit old proof reuse.
Its profile is frozen through benchmark preparation and its guidance contributes
to complete input accounting. Required prompt source remains present; this
integration alone establishes no complete-run token saving.

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
| Semantic world model | Maintain current task state, dependencies, observed transitions, reusable outcomes and unresolved questions | Learned predictions remain distinct from observations and independently checked facts |
| Formal systems | Reuse checked receipts and determine remaining supported obligations | Revalidate source, dependencies and assumptions; retain all six native verdicts |
| Semantic capsules | Present current task-specific evidence and residual work compactly | Required code, constraints, scope and uncertainty must remain available |
| Minification | Share identical metadata or encode exact reversible representations | Verify restoration; input bytes are not total provider tokens |
| Embeddings/autoencoders | Rank relevant facts, operators and capsules | Similarity does not authorize omission, proof, execution or completion |

The existing 8D spaCy, 384D and 768D GTE paths can rank metadata and capsules.
A Leanstral hidden-layer representation can be evaluated for that role. The
existing comparison already reused initial indexes with zero new embedding
calls; a new treatment cannot claim to eliminate those already absent calls.
Record inference, refresh and training separately from native LLM tokens.
Reuse the retained embeddings and checkpoints in their existing namespaces:
IR family, schema/version, decoder task, dimension and token profile. LegalIR,
SecurityIR, CodebaseIR and IntentIR remain separate; 8D, 384D and 768D ranking
paths can run in parallel. Their encoder token windows do not identify the
coding provider's tokenizer. This metadata measurement change creates no new
checkpoint, embedding or database owner.

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

### Optional token-counted selection and retained observations

The programmatic selector now accepts `SemanticMetadataTokenCounter`, wrapping
an existing counting callable with its declared provider, model, tokenizer,
tokenizer revision and qualification (`estimate` or `provider_tokenizer`). The
existing `CalibratedTokenEstimator.estimate` method can supply that callable.
The runner's `semantic_metadata_token_counter` argument requires the explicit
common-bindings route and the pinned provider/model to match before counting
or provider allocation. This identity is caller-declared; its digest does not
authenticate tokenizer compatibility.

An injected counter produces `supervisor-semantic-metadata-selection@2`; the
default byte-proxy path retains its original `@1` receipt exactly. Both complete
rendered prompt strings include literal workspace, public instruction, Doctor
and reply instructions. Compact selection requires strictly fewer bytes and
strictly fewer counted tokens. Invalid or failing counts retain the native
input with a sanitized reason and no proxy rescue. A failed callback never
exports its exception text. The receipt binds the complete input digests,
counter identity and selected count, while provider usage, hidden CLI
scaffolding and complete-run savings remain unverified.

The offline census accepts the same counter and hashes its textual identities.
The runner census counts each complete input once and still stops before
provider discovery. Benchmark comparison retains an allowlisted representation
observation beside the invocation's native usage; these counts never charge the
token ledger or substitute for missing usage. Audit replay can check `@2` with
the same independently supplied counter. Without that callable, it reconstructs
the recorded selected bytes but leaves counter/selection verification false.
The subprocess audit has no trusted callable factory and therefore cannot
qualify an injected counter merely from its receipt.

There is currently no qualified Codex/Grok tokenizer factory in benchmark
profiles. The CLI keeps its byte-proxy behavior. Before using this opt-in in a
live trial, qualify a reviewed provider/model/tokenizer/revision factory,
deterministic replay and its exact measurement scope. Build and qualify a new
runtime archive containing the changed helpers; existing frozen archives remain
historical evidence. Then compare complete native sessions. This implementation
alone establishes no live token savings.

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

## Additional token-efficiency controls (2026-10-12)

The opt-in `common-bindings-instruction@1` treatment extends the readable shared
metadata view. The current supervisor carries the public instruction both in
`semantic.raw_sources` and in the separately admitted verbatim requirements
block. This view replaces only the exact duplicated raw-source entry with a
path/hash/byte-bound reference to that block. The complete public instruction
remains inline. Other source text, native task state, Doctor context, workspace
mapping and reply instructions remain available. It requires the independently
admitted public-instruction receipt and matching captured source bytes; a
missing or stale binding refuses before provider discovery.

The original semantic input restores byte-for-byte. Selection compares both
complete rendered inputs, with byte/proxy selection or an explicitly injected
token counter. A candidate that does not shrink both required measurements
retains the original input and records the requested treatment and fallback.
Counter failure also retains the original input. Census stops before provider
discovery and emits measurements and identities, without source bodies. A
retained historical audit reconstructs the actual selected input even after the
code changes or the temporary checkout is removed. A retained token-counter
receipt alone cannot qualify an unavailable counter.

Use `--semantic-metadata-view common-bindings-instruction@1` when preparing a
full-supervisor experiment with `supervisor-semantic-router-input@1` and a rebuilt
runtime archive containing the current policy closure. It is also accepted by
the existing live census route. The default remains `legacy`; preparation,
archive and signed router checks still apply. This treatment does not change
planning or the provider selected through `llm_router`.

Capsule priority can additionally consume the existing retrieval index. Select
`--capsule-selection verified-retrieval-symbol-rank@1` during full-supervisor
benchmark preparation, with an explicit public `--task-profile` and a rebuilt
runtime archive. The default remains `lexical-query-symbol-name@1`; absent
options retain the existing configuration and receipt shapes. The selection is
bound through benchmark controls, Harbor arguments, the container driver,
initial indexes and admitted task context. Configuration drift or a returned
context with a different selection refuses execution. The no-index arm cannot
select ranking. Legacy profiles without an independently signed program/support
partition refuse before indexing or planning; an empty program population also
cannot supply ranked capsules.

The underlying
`prepare_initial_context(..., retrieval_ranked_capsules=True)` profile prepares
retrieval before capsules, then passes its artifact and digest to
`prepare_semantic_context(..., worker_retrieval=...)`. The owner replays the
query, verifies its exact source population, and maps hits to native symbols
using source and line-span identities. Up to 20 verified hits lead the order;
remaining symbols retain lexical priority. Existing required raw sources,
freshness/admission checks, capsule count and byte limits still apply. Rankings
cannot confer proof or permission to omit a required fact. The descriptor and
cold reload retain the exact selected retrieval binding. Historical audit
reconstructs the original ranking from retained sources, AST and query evidence
after code edits; it does not label those historical sources current.

Published successors now authenticate the prior ranking and prepare current
retrieval before compiling new semantic capsules. Unchanged source scope reuses
its verified retrieval; changed source scope requires the pinned native rebuild
policy. Missing, stale or failed retrieval refuses the ranked successor before
semantic preparation. The successor retains the selected query and capsule
bounds, and cold reload independently checks the old-to-new retrieval lineage.
Owner changes during refresh also refuse. This preserves current-source checks
while allowing retained historical replay. It does not establish ranking
quality, proof coverage or reduced token use. Current behavior and qualification
status are recorded in
[the ranked-successor evidence](evidence/terminal-ranked-successor-20261012/README.md).

Comparison output now includes **total native tokens per official pass** and
**reported provider USD per official pass**. Each numerator includes all
supplied retained attempts, including failures. Unknown official rewards,
missing usage/cost or zero passes leave the affected ratio unavailable, with
explicit reasons. Cache inclusion semantics remain visible. These are observed
ratios for the supplied attempts, not verified bills or causal savings.

Qualification measurements and source-bound commands are retained in
[evidence/terminal-token-efficiency-20261012](evidence/terminal-token-efficiency-20261012/README.md).
No live token reduction or new official Terminal Bench reward is established by
these implementation tests. The next comparison should hold task, provider,
model, planning strategy, budget and reply contract fixed, change one treatment
at a time, and retain every attempt and its native usage.

## Next changes, in measurement order

1. **Census the actual prepared coding view.** Let the verified owner emit only
   row counts, shared-field populations, complete input digests/bytes/proxy and
   selected/fallback status. Preserve original source bodies privately; do not
   export them in diagnostic metadata. Do this before spending on a live pilot.
   The [runner census stop point](terminal_metadata_census.md) now constructs both
   complete inputs after current owner checks and stops before provider
   allocation. Administrative source preparation retains explicit authored-plan
   provenance; live planned input remains a separate scope.
   The fresh live Source384 census now measured **59,336 to 58,454 bytes**,
   an 882-byte reduction (1.49%), with exact restoration and zero coding calls.
   Its retained planning session used **25,441 native tokens**. Current source,
   owner/world, schema custody and cleanup checks passed. This completes the
   input census gate; the result remains byte evidence, without complete-run
   savings or benchmark reward. See the [live observation](terminal_metadata_census.md#actual-live-source384-census-2026-10-11).
   A later instruction-deduplication census used the actual public
   `largest-eigenval` sources with an **authored administrative plan** and
   measured **57,423 to 56,452 bytes**, a 971-byte reduction (1.69%), with exact
   restoration and zero real planning or coding calls. It is a separate scope
   from the live Source384 result above. The corrected fresh live instruction
   census now measured **65,222 to 64,254 bytes** (968 bytes, 1.484%), with exact
   restoration and **25,315 planning tokens**, then stopped before coding. Its
   three retained attempts include 50,500 known planning tokens and 1,839.67
   seconds; first-attempt usage and dollar cost remain unknown. The selected
   profile uses lexical capsules and leaves Intent autoencoding disabled.
   Neither byte census establishes complete-run provider-token savings. The
   [closed evidence](evidence/terminal-ranked-successor-20261012/README.md)
   includes the portability and JSON-framing repairs required to finish it.
2. **Use current formal status capsules to prevent redundant reasoning.** Resolve
   an accepted native receipt through the supported proof-cache gate, including
   render/commit revalidation. Present property, verdict, assumptions, dependency
   digest and residual obligation. Keep proved, disproved, inconclusive,
   unsupported, error and cancelled distinct. Open Doctor work alone cannot
   produce an accepted proof card. Replace repeated status material rather than
   appending a second card.
   This is the next implementation priority after the 1.49% live metadata
   reduction. Inventory actual accepted receipts and repeated status populations
   first; unsupported or absent evidence remains explicit. Qualify replacement
   against unchanged obligations and current dependency/source digests with
   offline refusal controls, then census the complete new input. Existing
   conditional spectral lemmas do not certify the whole eigenvalue task or
   qualify an unavailable/slower numerical route for benchmark execution.
   The [source-only inventory](evidence/terminal-live-census-20261011/formal-status-next-treatment-inventory-01.json)
   identifies `DoctorProofCacheGate.lookup`, render/commit revalidation and
   existing proof-context digest handles. The inspected Terminal wiring lacks a
   complete admitted proof-owner/key binding; actual accepted receipt and
   repeated-status coverage remain unknown. Begin with a closed owner audit,
   refuse absent bindings before opening storage, and retain the native input
   when evidence or measured benefit is absent.
   The new [formal-status census](terminal_formal_status_census.md) implements
   this admission boundary. Its Terminal opt-in reports current task/source/
   context bindings separately from the unavailable native proof binding;
   accepted receipt and repeated-status counts remain null, with no proof-owner
   operations. It preserves the native prompt. A programmatic host can supply
   an already admitted Doctor gate, complete keys, independently expected key
   identities and a current-binding probe. That lane checks native lookup and
   render revalidation, including visible revocations, and explicitly reports
   possible bookkeeping writes to the existing owner. It observes the requested
   keys, without an atomic owner snapshot or a reusable proof receipt.
   The scoped header workflow now retains [complete key inputs and restart
   custody](header_proof_key_custody.md) in its existing sealed native owner.
   Rebuilt preparation and fresh native kernel replay qualify native reuse.
   The reviewed [scoped header bridge](header_proof_cache_bridge.md) now admits
   a native local lemma after fresh kernel replay and provides bound render/commit
   checks. A typed `@2` receipt preserves assumptions and actual residual IDs;
   local assurance cannot authorize task completion. The flat completion, claim,
   policy and planning consumers refuse scope loss.
   The explicit `TerminalFormalStatusOwner` host entry now joins an existing
   scoped Doctor analysis and native gate/full keys to the actual signed task.
   It reconstructs the source scope independently, rechecks Terminal/source
   custody around consultations, fences the same owner across context exit and
   clears escaped counts on drift or cleanup errors. The container host forwards
   these trusted objects locally without changing worker input. Authored native
   owner tests qualify this boundary. The [typed header admission path](terminal_header_proof_admissions.md)
   now retains bound handles, current native custody/profiles/implementations/
   executables and independently expected generic receipt IDs. It rechecks both
   observations using private lifecycle control; changing exported report fields
   cannot skip final proof checks. Raw Terminal gate/key inputs refuse scoped
   receipts without this binding. Reconnect the same current owners after restart,
   reimport with fresh native replay, admit the actual task proof inventory, then
   measure accepted coverage and dependency-complete repeats.
   No current Terminal proof coverage or token benefit has been established.
3. **Qualify a bounded owner lookup for progressive disclosure.** Required facts
   stay inline until the model has an actual qualified expansion interface.
   Typed queries should return an exact bounded population with an explicit
   unresolved frontier, source/task bindings and counted response costs. Avoid
   model-facing arbitrary SQL and assumptions about the generic Quack gateway.
   A source-only audit found reusable verified capsule/index readers, sealed
   producer-bound source materialization and bounded symbol-query primitives,
   but initially no capsule/source operation in the Quack registry or qualified
   expansion tool in the coding CLI route. The [bounded source reader](current_source_lookup.md)
   now attaches `source.get` to that existing owner, joins native task custody,
   verifies selected working bytes and complete response costs, and provides a
   host-issued subprocess CLI. The opt-in [worker source reader](worker_source_views.md)
   now provides a separate semantic-source-bound descendant route for compact
   views and exact expansion; it does not forward a typed-owner host grant.
   The numerical endpoint above
   covers pending numerical obligations; required facts remain inline. Add
   read-only `capsule.get` and other residual-obligation queries separately. Bind host-issued
   caller grant, task/revision, repository/source scope, root/context, owner
   generation/fence, request identity and byte/result bounds. Recheck current
   workspace source after edits on every read. Return exact identities,
   freshness/admission caveats, completeness, unresolved frontier and a bound
   continuation; account for the complete encoded response. Qualify actual CLI
   access, stale/foreign/tampered refusals and oversized responses before moving
   any required fact out of the prompt. Add ranked `symbol.query` afterward.
   This qualification reuses existing assets and requires no new training.
4. **Run a separately versioned matched trial.** First census the qualified
   treatment and retain enough measured input reduction to justify the trial.
   Keep one planning session and one coding session, model/reasoning/resource/
   setup budgets, original public inputs, reply schema and coding transport
   fixed. Vary one qualified treatment alone. A fallback run does not count as
   an observed treatment. Preserve both native administrative identities and
   compare reviewed semantic populations. Identical controls can still produce
   different plans. Report that distinction and require substantive native
   input matching for an exact one-factor claim.
5. **Use checked history deltas at decision boundaries.** Reconstruct a current
   context from an actually delivered parent plus changed facts. Bind parent,
   task, repository/tree, policy, schema and revision. Invalidate changed proof
   dependencies. DuckLake history does not grant authority to rewrite a native
   CLI's internal conversation or discard unresolved work.
6. **Evaluate retrieval separately.** Rank task dependency closures and proof
   obligations with the available autoencoders/embeddings. Measure required-fact
   coverage, missing/stale evidence and expansion cost before claiming omission
   is safe. The versioned ranked-capsule CLI and pinned successor refresh now
   support a separately controlled full-supervisor treatment on explicitly
   signed public task profiles. Complete its native lifecycle and rebuilt-archive
   qualification, then census ranked input and run matched lexical/ranked trials
   with the same metadata view. Retain refresh failures and embedding costs as
   part of each attempt. Keep new training out of the first metadata-only
   comparison.

## Semantic world model integration sequence (2026-10-14)

The objective is to reduce repeated reasoning, repeated discovery and repeated
provider context while preserving official task success. Whole-tree source
restoration supports fidelity and editing; a smaller source representation
affects only the source that the model actually receives. Both capabilities
are useful, but neither establishes a complete-run token reduction.

The existing [source comparison](compact_worker_benchmark.md) illustrates the
cost distribution: an inspected 78,597-byte initial coding prompt contained
2,588 bytes of raw source and 40,012 bytes of native context. The 12,008 capsule
bytes were nested within translated semantics and must not be added again.
One compact-source trial reported more native tokens than its baseline despite
smaller broker traffic. Usage completeness and single-trial variability limit
that observation. The newer 62.2% request/reply byte reduction for one authored
definition read likewise measures one operation, not the entire conversation.

### Reuse the current components

| Component inspected | Existing behavior | Integration work still required |
| --- | --- | --- |
| [RepositoryWorldState and deterministic deltas](../../ipfs_accelerate_py/agent_supervisor/procedure_compiler/world_model.py) | Bounded references for repository graphs, source changes, tests/proofs, resources, capabilities and unknown dimensions | Feed actual Terminal Bench actions and observations into successive planning states; current reference search found exports and procedure experiments rather than the worker-loop connection |
| [Transition contracts](../../ipfs_accelerate_py/agent_supervisor/procedure_compiler/transition_model.py) | Predictions, observations, uncertainty and calibration remain distinct | Connect real action outcomes; qualify any learned predictor on held-out transitions |
| [Semantic preparation](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py) | Initial source/capsule preparation and a persisted `context_prepared` record | Maintain ongoing state after reads, edits, tests and failures; a preparation row alone does not do this |
| [Existing ContextCompiler](../../ipfs_accelerate_py/agent_supervisor/context/context_compiler.py) | Bounded compilation, required coverage, retry deltas and prefix contracts | Compile the actual next provider input from current world state and verify its complete token count |
| [DeltaTaskPacket](../../ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py) | Task/tree/plan/evidence identities, impacted symbols, residual obligations, validation requirements and deterministic-first admission; connected to the opt-in world packet route with bounded historical export and replay below | Qualify a production Terminal Bench caller/collector and fresh live admission after restart || [Program-world context adapter](../../ipfs_accelerate_py/agent_supervisor/context/program_world_context.py) | Selects material identities using supplied costs; caches prefix identity bytes | Resolve admitted material bodies and measure final provider text; identity reuse is not evidence of provider prefix retention |
| [World database](../../ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_database.py) | Advisory DuckDB records and observational DuckLake history | Query these records through their current owners to drive selection, reuse and invalidation |
| [Source broker](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_source_mcp.py) | Current query, expansion, definition read, guarded edit and refresh | Attach observations to the task world and account for which model-facing context each operation replaces |

These are source observations, not qualification of a complete world-model
worker. Reuse the existing supervisor, context compiler, owner, model router
and admission mechanisms. Preserve the distinct source, semantic, execution
and completion authorities described by the existing architecture plans.

### State and IR responsibilities

The world model needs both a factual state and an optional predictive layer.
The factual state records what was observed or independently checked, the
dependencies of that evidence, what changed and what remains unknown. A
prediction estimates an action's likely effect, cost or useful next observation;
it cannot update the factual state until an actual observation is admitted.

| Input | Contribution to the current task world |
| --- | --- |
| IntentIR | Requested behavior, constraints and acceptance obligations |
| CodebaseIR | Current symbols, dependencies, supported effects/contracts, source correspondence and unsupported frontiers |
| SecurityIR | Applicable trust boundaries, security properties and associated evidence or unresolved checks |
| UI/UX IR | Applicable interaction flows, user-visible behavior and accessibility constraints |
| Environment observations | Filesystem/toolchain state, commands and test outcomes, available tools, resource pressure and provider availability |
| Formal systems | Scoped checked results, assumptions, counterexamples and obligations requiring further work |

Join these inputs through exact task/source/property identities and versioned
adapters. Record each family's coverage as observed, unavailable, unsupported
or explicitly out of task scope. Structural Codebase capsules do not establish
that every other IR family participated. Current semantic payload observations
and optional result-side advice do not yet establish that combined coverage.

Facts retain source generation, task/intent revision, dependency identities,
evidence references, assumptions and freshness. A proof cache key additionally
binds the property, logic/encoding, checker and environment relevant to that
proof. A similar embedding, matching symbol name or successful historical test
does not discharge a current obligation. Keep complete evidence in the owner
and storage layers; expose a readable scoped result and retrieval handle when
that is sufficient for the model's decision.

### Implementation phases and acceptance gates

1. **Join current observations into one task world.** Project already admitted
   source, plan, environment, evidence and obligation records into the existing
   `RepositoryWorldState`. Bind the world projection to its authoritative
   snapshot and task revision. Add bounded action observation hooks around
   repository operations and completed command/test outcomes. Record proposed,
   executed, observed and independently checked outcomes separately. Acceptance:
   an authored worker edit and test produce a successor world with exact source
   and result bindings; restart reconstructs the same current knowledge.

2. **Compile a small, self-contained decision context.** Adapt the world state
   into `DeltaTaskPacketRequest` and the existing `ContextCompiler`. Include
   the current objective, permitted effects/files, relevant established facts,
   unresolved obligations, a necessary code slice, material uncertainty and
   required validation. Retrieve optional capsule detail on demand. Count the
   complete rendered provider input, including guidance, tool schemas and
   reference overhead. If required material cannot fit, report overflow or
   decompose the work through the existing planner. Acceptance: a fresh worker
   can choose and perform the next action without an unstated prior transcript,
   and every required item is included or available through an admitted rule.

3. **Connect a real provider decision boundary.** The current daemon explicitly
   renders `result.reconstructed_capsule` for a stateless retry and records
   `stateless_retry_full_reconstruction` with `provider_retained_parent=False`.
   Keep that fallback until the replacement is qualified. For stateless calls,
   send a newly compiled current-state context with enough meaning to stand
   alone. For retained sessions, send deltas only after verifying the session
   retains the exact parent and compatible task/policy/source bindings. An
   opaque CID cannot teach an otherwise stateless model its missing contents.
   Acceptance: actual dispatch receipts bind the exact delivered bytes, parent
   disposition and measured usage. Do not claim control over a native coding
   CLI's internal history until its provider adapter exposes and qualifies that
   boundary. Preserve the existing planning session and router initially.

4. **Reuse outcomes and suppress repeated unsuccessful work.** Connect the
   existing deterministic-first packet decision to current proof/test/procedure
   reuse and residual obligations. Store a bounded failure signature and its
   dependencies so an unchanged failed action is not repeatedly proposed.
   Invalidate affected facts after edits, including dependent imports/contracts;
   unknown dependency effects require conservative invalidation. Direct shell
   changes require source reconciliation. Acceptance: valid reuse avoids the
   corresponding reasoning or check without dropping validation requirements;
   source/environment/assumption drift prevents stale reuse and permits a new
   attempt only with changed evidence or an explicit existing escalation path.

5. **Add learned CodebaseIR and world-transition ranking as a separate arm.**
   Reuse source-bound embeddings/checkpoints first. Train only admitted changed
   units under a bounded adaptation budget when the experiment can measure its
   benefit. Learn retrieval order, likely affected symbols, likely failure modes
   or which observation resolves a remaining question. Pin the source population,
   objective, encoder, decoder, tokenizer and checkpoint lineage; use held-out
   questions/transitions and a model-disabled control. Acceptance: measured
   selection quality and complete-run cost improve without losing required
   coverage. Reconstruction quality alone is not formal correctness. A
   repository-trained latent is not directly understood by a hosted coding LLM;
   a grounded decoder, readable capsule or retrieval tool must bridge it.

6. **Run the matched Terminal Bench experiment and promote measured wins.**
   First use paired arms differing only in the current-world decision context,
   keeping the qualified indexed source view, capsule budget, planning policy,
   model and resource limits fixed. Separately compare original versus compact
   source reads and deterministic versus learned selection. Report cold scan/
   training and warm reuse independently. Register repetitions and keep every
   failed, deferred or missing attempt. Acceptance: complete native accounting
   and official reward establish the preregistered success/token/cost criteria
   across repeated task-matched runs. Do not promote from source bytes, capsule
   counts, storage size or authored test counts alone.

### Reversible codebase views within this loop

The supported direction is retained original snapshot to compact view plus
mapping, followed by guarded edits against canonical source and a successor
snapshot. Exact restoration uses retained bytes: comments and formatting
cannot be recovered from a representation that discarded them without a
sidecar or another exact source copy. Keep execution, tests, paths and public
interfaces grounded in canonical source. Exporting a whole compact tree is an
optional navigation facility; it does not mean the model receives that tree.

Broader mapped edits need separate language-specific qualification for aliases,
imports, decorators, reflection, generated files and unsupported syntax. Use
short worker handles only with a generation-bound owner mapping and usable
expansion; include the mapping's delivery cost. Do not assume that shorter
identifiers, fewer bytes or a smaller archive imply fewer tokenizer tokens or
less reasoning. The existing guarded definition workflow supplies an immediate
source access path while these broader profiles remain bounded.

### Resource and failure behavior

Share the current resource admission and provider cooldown mechanisms. Bound
world-state materialization, graph queries, retained generations, capsule pages,
training batches and proof jobs. Under external memory/CPU/disk pressure, pause
optional adaptation and prefetch, release reconstructible projections and
reduce concurrent work. Persist enough admitted state to resume safely. A
resource limit must not silently remove required evidence or convert unknown
work to completion. Rate-limit deferral preserves the task/packet identity and
retry deadline; it must not start repeated replacement provider sessions.

Required integration tests cover cold reconstruction; forgotten or foreign
provider parents; omitted required constraints; stale cross-file dependencies;
unsupported dynamic behavior; changed proof assumptions; corrupt checkpoints;
out-of-scope edits; exact original fallback; identical-failure suppression;
resource-pressure backoff; rate-limit recovery; and interruption after a side
effect but before its receipt. Qualify one complete action-observation-context
cycle before expanding the loop to more IR families or learned predictions.

### First implemented boundary

The opt-in [repository world packet adapter](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet.py)
now connects an issued source context to existing delta-packet admission and
the coding runner's actual provider boundary. It preserves every selected
context body, the exact task and explicit residual/validation requirements,
rejects omitted or transformed content, and rechecks source freshness before
dispatch. The task text, task ID and effect/path scope must be bound into the
context when it is compiled; preparation rejects a different instruction or
scope attached afterward. A supplied world projection must have compatible task/source bindings;
the adapter does not create or independently refresh its authoritative owner.
Missing world coverage is explicit. The worker receives a self-contained packet
without relying on provider parent retention.

This is a first slice of phases 2 and 3 with a local authored cycle for phase 1.
At this first boundary, world-owner observation ingestion and historical
reconstruction were still pending; the subsequent slices below address them.
Combined IR adapters, deterministic result reuse, learned prediction and the
matched Terminal Bench experiment remain incomplete. Its in-process API remains
opt-in. The dedicated collector integration below now admits retained basic
context; the production dispatch handoff remains pending. Complete-run savings
remain unknown.
[Local qualification](evidence/world-packet-20261014/qualification.json) records
277 passing tests: 68 new packet/dispatch/cycle cases and 209 existing regression
cases. The cycle uses real native indexing, a real child MCP read and guarded
edit, a real Python command check, source refresh and a successor dispatch.
Both provider responses and world-owner declarations are authored. The stale
first packet is refused after the edit. Packet bodies were 11,876 and 13,577
bytes; these are wire-size observations without a matched token-savings baseline.
Tests also cover late source drift, task/scope mismatches, omitted requirements,
token-budget enforcement, timeout/rate-limit failure receipts and tool cleanup.
Tests ran serially with one native CPU slot and offline model assets; the report
pins dependency commits and records source and JUnit hashes.

### Native intent and validation observation bridge

The next boundary uses the existing `capture_intent_world_snapshot` and
`observe_local_intent_requirements` owners. A source-bound required reference
joins the selected native task specification, signed public-check observations,
applicable requirements and unavailable world components. The task context
compiler and packet builder retain that reference; the prepared packet repeats
the live owner check immediately before dispatch. This prevents a newer failed
check, changed task revision or changed admission from reusing an old passing
observation even when the source bytes match.

Validation execution remains in `run_local_task_validations`, which records
signed command results through the native intent repository. The bridge reads
and verifies those results; it adds no command runner or catalog. A generic
caller-supplied validation result is not promoted to signed check evidence.
Source-session tree hashes and the validation owner's inventory identity remain
distinct and are checked against the same workspace. Missing accepted tree,
policy, contract and other world authorities remain unavailable instead of
being replaced by source-session hashes.

The first capture API supports independently owned local intent databases.
The native Quack owner and historical replay extensions are described below;
live restart rebinding, joined IR families and matched live Terminal Bench
experiments remain outstanding.
[Native intent qualification](evidence/intent-world-20261014/qualification.json)
records 193 passing tests: 31 focused cases and 162 regressions. The end-to-end
cycle uses a real native intent database, three signed validation commands,
a child MCP query/read/guarded edit, two source generations and two authored
provider dispatches. It refuses the stale source packet and a formerly passing
packet after a newer failed observation, even after exact source restoration.
The task remains in progress; check evidence does not grant completion authority.
Tests also cover dropped or changed required evidence, forged observations,
admission mutation, mixed owner snapshots, resource denial, cancellation and
capture-size limits. Runs were serial with one native CPU slot and offline
model assets. The report records dependency pins, JUnit/source hashes, corrected
fixture attempts and a documentation-only source correction. Recapture overhead
and complete live-provider token savings remain unmeasured.

### Native Quack owner capture

`capture_repository_owner_intent_world` extends the required-reference path to
an actual live `QuackStateServer`. It reuses the native signed requirement
observer and captures the world snapshot in a transaction on the owner's
exclusive connection. It does not reopen the database file. Original admission
signer identity stays distinct from current server custody, and only allowlisted
custody fields reach worker context and receipts. Private endpoints, paths and
credentials stay with the owner.

The issued observation retains the exact owner, connection and transport.
Preparation and immediate pre-dispatch checks reject changed evidence or owner
custody, including a stopped or restarted server. Resource admission precedes
native reads, the world projection acquires its reservation before the owner
lock, and transaction exceptions leave the connection usable. The existing
observer retains its own transaction, with deadline-bounded lock acquisition;
deadlines remain cooperative. Native task data stays unchanged, while the reused
readiness check may restore the owner's private attach-token file.
No new command runner, metadata catalog or execution/completion authority is
introduced. This qualifies an in-process native owner integration. Historical
reconstruction is covered by the next slice; resumed dispatch still requires
new live source and owner observations.

[Native Quack qualification](evidence/quack-world-20261014/qualification.json)
records 230 passing tests: 17 native bridge cases and 213 existing regressions.
The cycle starts the installed Quack listener, queries it over TCP, admits a
typed native attempt and runs three signed validation commands. A real child
MCP edits the source between two authored provider dispatches. The daemon's
existing environment scrub removes database authority at the provider boundary.
Source staleness and a newer failed observation each prevent dispatch, including
after exact source restoration. Native tests also cover owner stop/restart,
timed lock contention, mixed snapshots, resource denial/cancellation, rollback
and preservation of an existing caller transaction. No test was skipped.
The report retains corrected fixture attempts, exact source and JUnit hashes,
dependency pins and serial resource bounds. Provider usage, capture overhead and
complete-session token savings remain unmeasured.

### Historical packet and complete-input replay

The replay boundary retains a bounded canonical artifact before the worker
edits source. It includes the compiled capsule, compilation receipt, exact task
and scope recipe, full native packet, emitted prompt and packet receipt.
Reconstruction calls the same native packet builder as live preparation and
compares every generated record. Claimed item digests alone cannot substitute
for rebuilding their payloads.

A second artifact joins that packet to the runner's recorded input bindings.
It reconstructs workspace guidance, tool policies and the selected coding reply
contract, then checks the final user-input hash and separate native output-schema
identity. Both APIs require a caller-retained SHA-256. They return historical
values that cannot enter the live runner or restore invocation-local issuance.
After a restart, resumed execution must recapture current native owners and
recompile a new packet.

The artifact contains full task/context content; diagnostics remain content-free.
Storage and independent digest retention belong to the caller. No new metadata
catalog, provider route or file loader is introduced. Input size, JSON depth and
node bounds apply before reconstruction. Replay does not open the old source,
intent database or Quack owner, reverify original signatures, recount tokenizer
results or prove provider delivery. It checks the structural consistency of
retained measurements while preserving their recorded scope.

This supplies the reconstruction prerequisite for the dedicated Terminal Bench
collector branch below. A world marker alone remains insufficient: collection
requires a pinned invocation artifact and the retained Terminal context join.
Production dispatch, live restart rebinding and matched complete-session token
measurements remain pending. Replay alone establishes no token saving and grants
no proof, execution or completion authority.

[Historical replay qualification](evidence/world-replay-20261014/qualification.json)
records 289 passing tests: 92 focused replay cases and 197 regressions, with no
failures or skips. The end-to-end case starts the real Quack listener, records
a signed failed command check, freezes its source-bound packet, then performs
a real child MCP query/read/guarded edit during an authored provider dispatch.
The edit prevents reuse and re-export of the stale live packet. After the
owners close and the original workspace and owner files become unavailable,
a fresh Python process reconstructs the exact rendered model input with native
credentials removed and fixture reads, database opens, sockets, subprocesses
and provider entry points blocked during replay.

Tests also cover canonical JSON bounds, external pins, repinned inconsistencies,
required evidence, immutable historical results, helper guidance, separate reply
schema identity, measured and failed counter records, and live packet/audit
regressions. Runs were serial within this session with one native CPU slot and
offline model assets. Source bytes stayed unchanged through both runs; the
report retains commands, dependency pins and source/JUnit hashes. Replay latency,
operating-system memory stress and live-provider token savings were not measured.

### Private invocation capture and Terminal collector join

The runner now accepts an optional `world_packet_audit_directory` for an issued
world packet. It freezes the input before provider initialization and retains
the final receipt join after helper cleanup. This survives source edits and
closed source owners. The store requires private owner permissions, rejects
symlink components and directory replacement, writes exclusively, and has
per-file, aggregate and inventory limits plus a bounded lock wait. Capture
failure cannot mask a provider result or trigger another provider attempt.
The independent public receipt carries only IDs, lengths and digests; the
private artifact contains task and source context.

The dedicated [world collector](../../benchmarks/agent_supervisor/container_coding/terminal_world_packet_audit.py)
loads `state/world-invocations/<invocation_id>.json`, using the independent
receipt's digest and bounded nofollow reads. It reconstructs actual rendered
input and reply-schema identity, joins the public task to retained preparation,
and preserves the existing required semantic, retrieval and intent-world
context checks. Explicit unknown context schemas cannot enter the legacy
compatibility branch. Mixed world and legacy populations must each qualify;
missing, changed or duplicated observations prevent a complete-input claim.
Installed helper-source digests remain checked. The store permits 16 files and
16 MiB; the collector permits 16 receipts and 12 MiB of declared artifact bodies
within its existing 30-second child deadline.

This is a historical audit path for basic required contexts. Selected source
formalization bundles and signed Terminal scopes are explicitly rejected until
their own preparation and reconstruction are connected. No production Terminal
CLI option recreates a live packet from these bytes. A current owner capture,
recompilation and dispatch handoff, followed by a rebuilt runtime archive, are
still needed for the live experiment. Even a prepared input that was never
dispatched can reconstruct correctly: provider delivery, actual repository tool
adoption, official task success and complete token usage remain independent
observations.

[Collector qualification](evidence/world-collector-20261014/qualification.json)
records 783 passing tests in the final runs: 119 focused cases and 664
regressions, with no failures or skips. The end-to-end fixture queries and
expands a real native capsule through child MCP, observes a smaller compact file
read and an exact symbol read, commits a guarded canonical edit, closes its
source session and makes the original workspace unavailable. The bounded
collector then reconstructs the exact historical input and joins the independent
repository-tool observations. Provider responses and token counts are authored;
this does not establish live task success or savings.

The tests cover malformed pins and receipts, missing required context, unknown
schemas, mixed populations, symlink/FIFO refusal, byte/count limits, lock
contention, failed provider attempts and recovery after zero-byte or partial
writes. The report retains the initial fixture failure and corrected result,
commands, dependency pins, source/JUnit hashes and a syntax-preserving test
format adjustment after the focused run. Runtime sources stayed unchanged
through qualification. Runs were serial with one native CPU slot and offline
model assets; operating-system memory stress and live token savings were not
measured.
## Decide success from the complete run

Count the final cumulative native input plus output tokens for every retained
planning, coding, lookup, continuation and repair attempt. Use the existing
native accounting rules: Codex cached input is already a subset of input;
Grok's normalized final-envelope input excludes separately reported cache
reads/writes. Preserve that distinction and each receipt's completeness scope.
Reasoning output is already part of output. Do not sum
cumulative usage records as independent calls. Preserve failed attempts and
unknown usage rather than assigning zero.

Report total tokens, planning/coding phase totals, calls/turns, lookup payloads,
repairs, cost and latency alongside the original verifier reward, native
completion/publication and cleanup. Keep embedding/refresh accounting explicit.
Require final input reconstruction and schema custody. Repeat balanced matched
pairs before claiming repeatable savings. This first implementation establishes
representation and integration checks. The October 12 instruction pair above
records a small observed native-token difference; repeatable savings remain
unestablished. Reversible source-view input and complete-run benefits are still
unmeasured.
## October 14: preserve unfinished coding work before shortening sessions

Inspection of the native continuation path found two prerequisites for bounded
coding sessions. The benchmark currently permits one implementation attempt,
and a Grok coding invocation can run 128 native turns. An ordinary nonzero
provider exit previously left a dirty nonpooled checkout, or released a pooled
checkout, without creating the committed candidate reference that the retry
seeding path reads. Simply increasing the attempt count could therefore repeat
work. This is a possible retry cost mechanism, not a measured explanation of
the historical 26-turn run, which completed within its original attempt.

Failed-provider retention now uses the existing native rescue-commit machinery.
It keeps the provider failure and records that validation was not attempted;
it does not mark the task complete. Before releasing a changed checkout, the
owner must preserve the candidate, verify its rescue reference and record the
retention event. Failed preservation retains the physical checkout. Existing
capacity, protected-path and uncertain-process custody paths keep their
separate handling. A retained commit is not a proof or permission to replay:
the existing same-task accepted-proposal, path-scope and fresh patch checks
still control retry seeding.

The generic nonstreaming Grok adapter also rejects explicit native incomplete
outcomes even when the process exits zero and includes text. This closes a
gap outside the supervisor's already stricter streaming coding path. Native
usage remains available on failure; conflicting stop aliases are not rewritten
into fabricated successful observations. Literal stop-related words inside
model text are not control messages.

Offline qualification uses authored child executables and real Git retention,
with cold retry-context checks both without indexes and with the real semantic
index producer. It verifies that unresolved acceptance and validation
requirements survive a restart, and that a worker checkpoint saying
`complete` cannot discharge them. The canonical source/index still describe
the admitted baseline; preserved candidate changes remain distinct evidence.
These checks do not run a paid model, demonstrate a second coding dispatch or
establish a new Terminal-Bench score or token saving.

The remaining bounded-session integration must:

1. Admit an explicit initial-call and repair-call budget under one cumulative
   deadline, using the existing same-attempt validation/Doctor repair pipeline.
   Keep timeout, turn exhaustion and unknown outcomes distinct from completion.
2. Obtain a fresh owner-admitted prompt and command after a failed validation.
   Sealed commands cannot be reused with an altered prompt. Retain the full
   required context because provider retention of a local context delta is
   unverified.
3. Bind the edited working candidate to a new source generation and refresh
   its index/proof frontier. Baseline `capsule-references@1` retrieval becomes
   stale after edits; it cannot stand in for candidate evidence.
4. Measure every planning, coding and repair invocation, retrieval expansion,
   indexing refresh and failed attempt, preserving unknown usage and the
   provider's cache accounting. Compare total-run cost and verifier reward in
   fresh matched indexed/no-index/native trials once provider access is ready.

This change does not shorten the default 128-turn coding call or enable
automatic multi-round benchmark sessions. It fixes retention and outcome
contracts needed before that experiment can be interpreted reliably.

Qualification records and source hashes are retained in
[qualification.json](evidence/coding-checkpoints-20261014/qualification.json).
The main run passed 260 checks and exposed one test assertion that incorrectly
searched across semantic chunk boundaries. After correcting that assertion to
reassemble and verify the exact source and hash, both cold-context cases passed.
The latest results therefore cover 261 distinct passing checks across the two
runs, with unchanged production sources; this was not one all-green invocation.
Broader regression checks found four failures that also reproduce on the clean
starting commit: an obsolete deterministic-writer API call, a checkpoint-path
omission assertion that conflicts with full context reconstruction, and two
capacity-custody expectations. The incremental-runtime test module also fails
collection on that baseline because `guarded_worktree_pool_mutation` is absent.
These are recorded separately from the focused checkpoint/router qualification;
this is not a claim that the entire supervisor test suite passes.


## October 14: exact capsule provenance retrieval

The opt-in `semantic_working_set="capsule-references@1"` now connects an
exact baseline capsule reader to the actual coding router. It requires the
full indexed arm, enabled source views, semantic transport `@1`, and the
legacy metadata view. Select it with `--semantic-working-set
capsule-references@1` in the full-supervisor benchmark preparation command;
retain the same source-view profile in both comparison arms. The frozen
configuration, installed worker arguments and declared controls bind the
selection. Provider-free Doctor routes continue through their existing path.

This first profile defers a closed set of capsule provenance fields, including
producer versions and dependency version/fact/link identifiers. It preserves
behavioral and unknown capsule fields, contracts, effects, exceptions, proof
obligation references, raw source, the native task core, world/formalization
evidence, public requirements and the existing Doctor advisory. The original
semantic transport remains available to the owner and response decoder.
Every deferred row has an index and hash for its exact native capsule bytes;
candidate regeneration cannot substitute for this retrieval.

The distinction from existing Doctor integration matters:
`prepare_doctor_residual_context` already compiles a PlannerDoctor capsule
from actual admitted tasks and residuals. Its router advisory is appended to
the full native context. The new profile reduces selected provenance overhead;
it does not replace the complete task with that capsule or invent accepted
proofs. Required-core identifiers alone do not establish that constraint text
has reached the model.

Before model generation, the router performs one bounded request through the
existing source socket. The exact capsule hash and current canonical/allocated
source checks must pass. This probe consumes the existing retrieval budget,
appears in session accounting and adds no content to model input. A missing,
foreign, stale or closed endpoint refuses generation. After edits, baseline
retrieval refuses; current candidate analysis is a different operation.

Selection compares complete UTF-8 input bytes, including workspace, source
guidance, public/Doctor/reply wrappers and expansion instructions. A larger
candidate retains the original prompt. The receipt separately binds the
original and selected inputs, broker session and preflight response. Historical
audit reconstructs the delivered representation without claiming current
freshness. Evidence response bytes are counted separately from source bytes
and candidate-analysis bytes. Provider token counters remain unknown here.

The offline qualification and source pins are recorded in
[qualification.json](evidence/semantic-working-set-20261014/qualification.json).
It uses authored software, the real semantic producer, Git worktrees, a
registered child and socket, and a capturing `llm_router` provider. Container
UID admission is authored; this is not a new Docker or Terminal-Bench reward
run. Both literal and structured coding replies are checked, along with the
Grok route, generated worker/driver wiring, stale-source refusal, exact inverse,
accounting and audit reconstruction.

The final consolidated run passed 199 tests with unchanged source pins.
The authored fixture's complete initial router input fell from 34,580 to
32,977 bytes (4.64%); with the structured reply wrapper it fell from 35,086 to
33,482 bytes (4.57%). Expansion responses are additional traffic. These
measurements establish initial byte reduction and exact restoration, not net
token, latency, cost or benchmark-score improvements. The linked receipt also
records separate regression runs; their test populations overlap.

The remaining large cost is still native conversation history: Grok coding can
run up to 128 turns inside one router invocation. This profile does not control
or truncate that history. A subsequent integration must persist unresolved
work at supervisor decision boundaries, rebuild the required current context,
and measure every call, tool expansion and retry. World/formalization payloads
need their own exact readers and coverage checks before further deferral. A
live matched comparison remains necessary; the preceding live treatment was
authentication-blocked and supplies no token-savings result for this profile.

## October 14: bound coding calls without resetting the repair budget

Grok coding now has an explicit `--grok-coding-max-turns` experiment control.
It accepts integers from 1 through 128; the omitted/default value remains 128,
and planning retains its separate two-turn bound. Select the control during
benchmark `prepare` with the Grok provider profile. It is frozen through the
configuration, declared-control hash, runtime archive, Harbor adapter, container
driver and isolated worker into `llm_router`. Both full and no-index arms can
select it. Codex and metadata-only routes reject a nondefault value. Doctor
candidate routes retain their provider-free behavior.

This is a single-call limit, not an automatic continuation policy. Reaching
the native turn limit remains a failed provider outcome, including when the
CLI exits zero with partial text. Native usage observations remain attached
to the failure, and the prior rescue-commit work preserves unfinished source.
The collector distinguishes the configured value, runtime declaration and
observed coding-call policy. Missing invocation evidence remains unavailable;
neither a declaration nor fewer turns establishes lower total cost at equal
verifier reward.

A provider-free Doctor route can still produce a complete benchmark receipt
when its explicit nonapplication record matches the frozen turn selection and
there is no coding or unreceipted invocation. Its coding-policy observation
stays unavailable because no coding call exercised the bound. Empty receipts
alone cannot establish this exception, and any planning usage is retained.

The legacy inline repair path previously granted a fresh provider timeout
after the first child and its validation. It now receives one absolute
monotonic deadline captured at the initial child dispatch. Validation, prompt
and environment preparation, and event persistence consume that remaining
window. Missing, invalid or exhausted deadlines block another provider call.
Initial indexing and setup remain outside this provider-call window and under
the existing outer benchmark deadline. Deterministic materialization, staging
and revalidation remain available when the provider budget is exhausted.
Prepared and actually started children have separate events.

The process runner also receives that absolute deadline. It refuses an expired
lease before launch, deducts child-start event persistence from the subsequent
wait, and caps progress-based waits at the same deadline. If a synchronous
birth callback returns after expiry, the runner fences the child and reports
timeout. Cleanup retains its own grace period; the deadline cannot interrupt
a callback that has not returned.

Database-bound inline model repair is explicitly refused until the owner can
admit a new round and account for its child custody. Previously this path could
call the subprocess runner without the first launch's owner gate, child-birth
binding and custody tracking. Scoped sealed-provider commands retain their
existing refusal as well. These refusals do not prohibit deterministic repair
or turn a failed attempt into completion.

Tracing the Terminal wrapper clarifies the remaining integration: it already
seals fresh stdin and creates new broker descriptors on each launch. Its
semantic producer and source broker still require the allocated source to
match the admitted baseline. Reissuing the original context after edits
therefore fails the source checks; reusing old descriptor fields is not the
solution. Ordinary canonical refresh also reads `self.repo_root`, rather than
the working candidate.

The next native-round contract must bind all of the following:

1. The existing task, attempt, claim and workspace fence, plus a distinct round
   identity, the prior validation outcome and remaining cumulative budget.
2. An owner-captured candidate generation alongside the independently checked
   canonical baseline. The existing candidate builder can regenerate native
   capsules and numerical grammar artifacts from those captured bytes. Its
   output is advisory: pending obligations and unsupported constructs remain
   visible, and baseline proofs do not become candidate proofs. The current
   population is bounded to 64 admitted files and 1 MiB; added or deleted paths
   require an explicit scope decision.
3. A fresh typed context, formalization handoff and broker binding for those
   candidate bytes, retaining the complete task requirements, unresolved
   constraints and failure evidence. The baseline-only exact capsule reader
   must continue refusing edited source.
4. The actual next child birth, waited outcome and quiescence, through the
   native owner launch gate. The existing single-child receipt cannot stand in
   for multiple children. Deterministic candidate review should run before
   admitting another paid call, including after an exhausted call; exhaustion
   itself remains neither validation nor completion evidence.

Offline qualification includes an authored Grok executable that receives the
16-turn flag, edits a real Git worktree and emits a native turn-exhaustion end
event. The real router rejects it, the database supervisor settles the failed
child, and the unfinished commit and usage survive. Container isolation is
authored in that fixture; it is not a Docker or live-model benchmark. Native
database tests also exercise successful first children whose output fails
validation, verifying that no unbound second child launches.

Source pins and final test results are recorded in
[qualification.json](evidence/bounded-rounds-20261014/qualification.json).
The final frozen-source run passed all 641 checks across 19 modules, with no
skips or live provider calls. It includes process cleanup, router/worker
dispatch, benchmark controls, deterministic rescue and native failed-checkpoint
retention. The exact datasets dependency was restored in a new isolated
checkout after its previous path disappeared; its commit and import origin
are included in the qualification record.
Automatic native multi-round sessions, a refreshed accepted proof index for
candidate edits, and a matched live token/reward result remain outstanding.


## October 14: refresh candidate context after a failed worker

The installed owner monitor now rebuilds advisory candidate context after a
coding worker exits with a positive failure code, isolated-worker cleanup
succeeds, and all brokers close. It reuses the retained source admission and
the native datasets candidate builder to produce fresh semantic capsules and
formal projections from the edited working tree. The canonical baseline and
its original semantic translation remain independently checked. Changes do
not become baseline proofs, and the baseline-only reader continues rejecting
edited source.

The resulting snapshot binds the task, original prompt digest, prior child
and source session, descriptor digest, working generation, source hashes,
program-file selection and native producer hashes. Source and producer
revalidation bracket generation and checkpoint writing. Native generation
invokes no model or prover. Unresolved obligations and unsupported constructs
remain explicit; generated Lean is marked prepared, not built or verified.
The population remains the original admitted files. Added paths receive no
coverage, and missing admitted files refuse refresh rather than silently
shrinking that population.

The owner retains three artifacts (summary, capsules and formal projections)
in a private directory, with a bounded manifest that contains bindings and
hashes but no native prompt. Creation is exclusive: another observation for
the same session cannot replace an existing checkpoint. Ordinary source drift,
deadline expiry and write failures remove only files created by that call.
The manifest is written last, but a hard process crash can leave incomplete
files; this is not crash-atomic publication or automatic recovery. The
collector rejects incomplete evidence.

Refresh uses the remaining allowance measured from owner entry, including
ownership setup and formalization. It does not reset a timeout after failure
or change the existing provider-call timeout. Native generation checks this
deadline cooperatively. Cleanup remains protected from interruption, then
interrupt handlers are restored before analysis. Refresh refusal or
interruption preserves the failed worker outcome and emits bounded diagnostic
metadata. Negative signal exits, unretired workers and failed cleanup do not
trigger refresh.

The benchmark report adds `owner_candidate_refresh_observation`. It selects
only manifest names bound to source-session receipts from the current trial
and verifies private file modes, stable reads, hashes, sizes and task/prompt
bindings. A provider-success receipt can still have a prepared checkpoint
when its enclosing worker later fails. Missing evidence for a failed selected
session stays unavailable; it is not counted as zero. Reported artifact bytes
measure retained files, not model input, delivered context or saved tokens.
Checkpoint `elapsed_seconds` runs through artifact writing and prepublication
validation; it excludes the final manifest write and final revalidation. Use
the outer trial duration for total cost, including that remaining work.

This closes candidate preparation and retention for failed worker exits.
It does not yet admit a second coding child, send these artifacts to a model,
publish an accepted proof index, or handle a successful child followed by
validation failure. Continuation still needs an owner-admitted candidate-aware
prompt, source descriptor and child custody under the same task, attempt and
remaining budget. The existing baseline descriptor must not be reused.

The end-to-end authored fixture runs the installed monitor, a real editing
child, actual broker retirement, the native datasets grammar, private writer
and report collector. An `argmax` to `argmin` edit changes the emitted Lean
binding from `SelectionMax` to `SelectionMin`, changes the capsule signature
and source hash, preserves canonical source and leaves obligations pending.
Container UID admission, the isolated cleanup executable and router receipt
are authored; this does not qualify Docker isolation, provider delivery or
full live supervision. Both failed and provider-returned receipt cases are
covered. Source pins, test results and limitations are retained in
[qualification.json](evidence/owner-rounds-20261014/qualification.json).

The final frozen-source qualification passed all 527 checks across 18 modules,
with no skips or live provider calls. It includes candidate source/admission,
checkpoint rollback, cancellation, native failed-provider retention, worker
wiring and benchmark observations. The datasets and kit commits and actual
import origins stayed pinned and clean. An early fixture socket-path error
and an overly broad Lean assertion are retained in the run history alongside
their corrected results; they are not hidden as successful runs.

## October 15: account for every coding child before enabling continuation

The next execution dependency is complete attempt accounting. The previous
collector only flagged a missing final coding receipt when no coding receipts
survived. One recorded invocation could therefore hide a later child with no
final receipt. That would make a multi-round token comparison unreliable.

Ordinary isolated coding now receives an owner-generated attempt ID. Before
launch, the owner writes an exclusive private preparation event bound to the
sealed native prompt, allocated workspace, container boundary, original absolute
deadline and owner process birth. A second immutable event records the actual
`Popen` child birth. Retirement records the observed wait, original child exit,
isolated cleanup and broker closure, including failures. The events form a hash
chain; files are owner-only, read-only, bounded and never adopted or overwritten.
IDs and event files are accounting observations and grant no launch permission.

The installed worker forwards the ID to the existing router, which retains it
on successful and failed final receipts without putting it in model input. The
Terminal collector matches every child to exactly one distinct invocation with
the same ID, native prompt digest and workspace identity. Identical repeated
receipts are harmless. Conflicting receipts remain retained separately and make
coverage unknown; they cannot disappear through invocation-ID deduplication.
Planning, provider-free candidates, metadata census, preflight and semantic
decision dialogues retain their existing accounting routes.

The report's `owner_worker_attempt_census` observes child/receipt coverage,
separately from native token completeness. A missing later receipt, unfinished
retirement, malformed event, prompt/workspace mismatch or uncertain child birth sets
a nonempty `unreceipted_provider_attempt` marker with unknown usage. Existing
Harbor and comparison projections then withhold complete token totals while
preserving known receipts and charges. Observed receipt counts are not a complete
physical provider-call census, and matching all child receipts does not establish
native usage completeness, billing totals or saved tokens.

A process-creation error before `Popen` assignment leaves preparation-only
evidence: absence of a recorded PID cannot prove no child ever started. A live
writer can record an explicit unstarted attempt only when its caller knows no
launch was attempted. Retirement writing remains possible after deadline expiry;
write failure preserves the worker outcome and leaves incomplete evidence.
Archive compatibility now pins both the writer and collector, alongside the
existing deployment, router and supervisor pins.

This increment prepares accounting for a second coding round. It does not
enable that round. The remaining launch work must retain the original database
attempt, task claim, resource reservation and deadline, obtain a fresh typed
candidate admission on that owner scope, and record a separate child birth.
The existing finite-proof dispatch broker and database callback are one-use;
replaying either or removing the candidate router refusal would bypass their
contract. Candidate artifacts and serialized checkpoints cannot supply that
admission. A successful coding child followed by validation failure also needs
its own qualification because the current snapshot factory accepts positive
failed-child exits only.

Qualification uses real local children through the production monitor and
writer, plus authored container admission, cleanup commands and provider
receipts. It includes two separate children, missing second receipts, failed
registration/cleanup/closure, failed birth observations, private-file drift and
archive compatibility. These tests invoke no model and do not qualify Docker
identity isolation or live candidate delivery. Results, dependency revisions and
retained failed runs are recorded in
[qualification.json](evidence/owner-worker-attempt-census-20261015/qualification.json).

## October 15: host-native admission for one candidate child

The next connection is a host-owner admission that joins the existing native
task and allocation handles to the failed-child snapshot, complete candidate
prompt, original deadline, supervisor run lease and both worker census handles.
It admits at most one candidate child per retained original source worker.
Serialized observations, checkpoints and read descriptors remain insufficient.

`NativeSourceWorktreeAllocation.execute_with_current_claim(...)` now provides
the required native boundary. The real process-serialized coordinator holds its
task-claim transaction across the owner's callback, with COOR before SERVER lock
order. The surrounding checks retain the full native task body, claim tuple,
active daemon grant, Portal projection, ACTIVE worktree lifecycle, linked Git
controls and monotonic lifecycle renewals. Reentry, a borrowed SQL transaction,
or changed authority refuses the callback. This reusable boundary reads
allocation controls; it does not turn source contents into proofs or authorize
a coding command by itself.

The new `capture_native_candidate_round_admission(...)` factory independently
checks a successfully published prior retirement event, the actual waited
positive failed child, exact native allocation/task identity, current candidate
generation and producers, original requirements and new prepared worker event.
The immutable candidate event must bind the exact complete candidate prompt and
the same absolute deadline as its predecessor. The run lease is read from its
existing native coordinator with the original owner and fencing identity.
It is a `supervisor_run` lease, separately from quantitative CPU, memory and
child-slot reservations; this increment does not establish those reservations.

The issued handle consumes its birth admission once, holds the real native
claim fence through actual `Popen` registration, and records the new census
birth before external broker registration. A postbirth failure retains the
actual process and possible effects; neither a failed acknowledgement nor a
missing receipt becomes a refund or permission to launch another child.
Source verification precedes birth. Afterward, the final pre-input check
revalidates the candidate prompt/generation after coordinator closing callbacks
and closes with native control and expiry observations. These are bounded
sequential checks, not a filesystem or distributed transaction.

`run_native_candidate_round_worker(...)` supplies the host-owner execution
helper. It opens an empty stdin pipe, creates and registers the child through
that admission, and releases the exact candidate bytes only after the native
boundary and final checks return. Its wait uses the original remaining deadline.
Failed registration, expiry, interrupted execution and failed cleanup preserve
the child handle and census evidence. The caller supplies its existing
descendant cleanup operation and owner output streams; the helper refuses an
environment carrying database authority and unbounded output-pipe capture.
Successful broker retirement also requires the reader thread to have exited
and its descriptor and socket to have been removed.

This qualifies a host-owner boundary with real native claims, grants,
allocation/lifecycle records, run leases, candidate producers, private events
and local children. The fixture authors Quack transport, deployment admission
and cleanup for its inert same-UID children. It invokes no provider, executes
no target code and leaves the native task in progress. The original candidate
prompt remains `dispatch_usable=False`; the router still refuses it without a
separate admitted input route.

The deployed Terminal path still needs its authenticated daemon-to-wrapper
handoff, original absolute provider deadline, explicit two-child policy and
typed candidate router input. Ordinary single-lane execution must select shared
coordinator serialization before opening its sidecar. Its group-writable
container workspace also needs a separately qualified allocation contract: the
existing source allocation handle continues rejecting those modes. None of
these deployment properties is inferred from this host admission or enabled by
default. Matched native token/reward trials remain a later measurement gate.

Qualification results and explicit limitations are retained in
[qualification.json](evidence/native-candidate-admission-20261015/qualification.json).

## October 15: candidate dispatch through the existing router

The host candidate launcher now accepts a `CandidateDispatchBroker` bound to
the exact live `NativeCandidateRoundAdmission`. This extends the existing
candidate read service with a distinct version-two descriptor and a closed
check/consume operation. A version-one read descriptor still supplies no
dispatch permission. Both endpoints retain the original snapshot generation;
neither refreshes a changed candidate or publishes an accepted index.

The dispatch service retains the complete candidate prompt hash and length,
original requirements binding, context, owner attempt, original deadline, and
explicit owner-selected Codex model, timeout, reasoning effort and output cap.
It accepts requests only from the registered child's kernel ancestry through
the existing UID, process-birth and Unix peer checks. The server replays the
actual native admission, including its task claim, grant, run lease and source
generation. It consumes the provider permit once before closing checks and
response delivery. A failed check after consumption or a lost response leaves
that permit spent; retrying the RPC cannot create a second invocation.

`prepare_candidate_router_input(...)` creates a worker-local typed input from
that protected descriptor and an authenticated owner exchange, retaining the
exact stdin bytes. Serialized observations and candidate JSON are insufficient.
`router_implementation_runner.run(candidate_round_input=...)` accepts this
separate route only for isolated Codex coding with the matching worker attempt
and policy. The ordinary raw-candidate refusal remains. The route preserves
the complete original requirements, treats retained baseline evidence as
historical, and adds the candidate read client's guidance. It excludes baseline
translation, public-source currentness checks and unrelated projection routes.
An explicitly selected tokenizer may measure the complete rendered prompt;
the automatic ordinary-prompt tokenizer is not selected for this route.

The runner consumes the live permit immediately before its existing provider
call, after provider initialization, prompt measurement and descriptor-context
entry. The actual timeout is capped by the original remaining deadline. Its
ordinary invocation receipt and worker-attempt label retain usage and failures.
An authorization request whose reply is lost records consumption as unknown,
with zero router calls; it does not prove that the owner permit stayed unused.
The dispatch observation itself cannot attest model execution, proof acceptance
or completion. Reader transport is retired through the existing launcher.

The explicit host CLI uses `--candidate-dispatch-descriptor` and
`--candidate-dispatch-descriptor-sha256` together with the existing owner-attempt
and isolated-boundary arguments. No installed Terminal wrapper or default retry
policy enables these arguments automatically. The live native owner and the
original worker census still have to share a genuine ownership scope; copying
their identifiers into the separate deployed wrapper cannot establish it.
The writable-allocation delegation, quantitative resource reservation and
production absolute-deadline handoff described above remain required before
production use. Matched TerminalBench token/reward trials remain outstanding.

Qualification distinguishes actual native state, local processes and Unix
transport from authored same-UID deployment admission and provider stubs. No
external provider or target program is invoked. Results and limitations are in
[qualification.json](evidence/native-candidate-router-20261015/qualification.json).

## October 15: quantitative capacity through candidate retirement

`reserve_native_candidate_round_resources(...)` now reserves a genuine native
child `ResourceLease` below the caller's existing parent. It uses that same
datasets scheduler, policy and authenticated capability; it creates no scheduler
or new root reservation. The child fits the parent's CPU, RAM and child-slot
capacity and retains the candidate worker attempt, repository, workspace and
original absolute deadline. Detached IDs, tokens or observations cannot replace
the owner-issued resource handle. Public metadata excludes capability keys.

The optional `resources=` argument on native candidate admission joins this
handle to the exact new worker attempt. Native owner checks authenticate the
parent and child keys against locked scheduler state, retain immutable capacity
and ancestry, and observe cancellation, release, expiry and policy drift. These
checks accompany task/run/source checks before birth and input delivery; a
resource-only closing check follows late native observers. The candidate
dispatch broker therefore also refuses delivery if the selected resource scope
has been cancelled. The existing host route without this explicit selection
retains its unreserved status.

The launcher applies the admitted CPU thread environment to its literal child
and refuses conflicting declarations. It polls resource cancellation in bounded
`communicate` waits, supplies the prompt exactly once, and continues subsequent
waits without another input write. Every wait stays within the original deadline.
Neither cancellation nor a later refusal authorizes another candidate child.

Resource retirement occurs after actual child reap, the caller's descendant
cleanup, complete broker closure and the immutable worker census. It publishes
a separate private quiescence record before keyed native release, refuses live
native consumers beneath the leaf, and verifies actual leaf removal rather than
treating a release acknowledgement as freed capacity. A scope that never
attempted birth can be aborted explicitly; its gate is consumed before release.

Failed cleanup, missing census publication, uncertain birth and failed resource
publication retain strong owner custody. For a known child,
`recover_retirement(cleanup=...)` runs the existing owner cleanup operation,
checks actual child and broker quiescence again, and writes a separate immutable
resource record. Historical census flags remain unchanged. Partial publications
remain in place; bounded nonce filenames let a fresh recovery publish its own
witness without adopting or overwriting an uncertain file. Reentrant retirement
is refused, including a suppressed recursive attempt inside cleanup. Unknown
births without an actual process handle remain unresolved.
If construction fails before any launch is admitted, the factory aborts its
proven-unstarted sublease. A failed abort preserves the original error and
attaches the live owner handle for explicit cleanup.

This is cooperative admission accounting, not CPU/RAM cgroup enforcement. The
declared child budget must cover the selected command's subprocess needs.
Strong pending custody protects a live owner process; it does not qualify
supervisor-death recovery while external children survive. The deployed wrapper
ownership crossing, original production deadline, writable-allocation delegation
and explicit two-child policy still require their separate integration. This
increment enables none of those deployment defaults and measures no token saving.

Qualification passes the existing validation parent's real capability to the
owner test process through a sealed private descriptor. Tests acquire genuine
nested capacity beneath it, preserve the scheduler policy and original root
charge, and keep that descriptor out of worker inheritance and model input.
They exercise native authority, actual local children, cancellation during
waiting, failed and partial publication, explicit recovery, and the existing
candidate route. Deployment UID/path admission remains authored, and no provider
model or target code runs. Results and explicit limits are retained in
[qualification.json](evidence/native-candidate-resources-20261015/qualification.json).

## October 16: retain the provider deadline across the owner wrapper

The daemon already creates an absolute provider deadline before its streamed
launch. The exact `/opt/ipfs-supervisor/bin/router-worker` command now receives
that restriction through an anonymous sealed descriptor. The payload binds the
direct parent PID and birth, UID, boot, literal command, target workspace inode
and process/mount namespaces, plus the time namespace where available. It carries
no native task, resource, dispatch,
retry or completion authority. The daemon may remain in its canonical repository
while the child starts in the allocated worktree.

The generated owner entry consumes the descriptor before changing workspace
permissions, removes its private environment markers and closes it on acceptance
or refusal. A partial pair, changed bytes or identity, missing seals, changed
workspace, different direct parent or expired deadline refuses the handoff.
Existing launches without a descriptor retain their previous contract.

The owner uses the earlier of that inherited deadline and its valid requested
local timeout cap. An invalid local timeout cannot erase the inherited limit.
Workspace setup and prompt capture consume that budget. The effective deadline
also bounds actual isolated child birth and wait and failed-worker candidate
refresh, and is bound into the worker census. Retirement publication remains
possible after expiry. A late birth is reaped through existing cleanup and remains
an observed birth; an expired or interrupted wait keeps its census uncertainty.
Cleanup retains its independent grace period rather than claiming that expiry
has already stopped descendants.

For ordinary coding, the owner generates `--owner-provider-deadline-ns` and
rejects caller-supplied versions at its inlet. The isolated entry forwards only
this integer restriction and the existing census identity. The router retains
the deadline in its invocation receipt and recalculates remaining time after
provider initialization and descriptor-context entry, immediately before its
one provider attempt. Positive fractional timeouts are preserved; expiry keeps
zero router calls. The value supplies no dispatch authority and does not enter
the model prompt. The outer owner wait still bounds subsequent provider-adapter
preparation and execution. Planning, decision dialogues and provider-free paths
receive the owner wait limit, without this ordinary-coding router argument.

Qualification exercises real sealed descriptors, direct children, a foreign
grandchild refusal, differing daemon/worker directories, actual monitor waits
and reaps, and the streamed launcher path. Deployment paths and cleanup remain
authored host seams; router provider observations remain stubs. The generated
scripts have not been installed or qualified in Docker by this increment.
Shared native ownership, writable allocation delegation, quantitative production
resources and the explicit second-child policy remain separate work. This
change enables no candidate retry and measures no token saving. Results are in
[qualification.json](evidence/owner-provider-deadline-20261015/qualification.json).

## October 16: compose the native second-child owner lifecycle

`prepare_native_candidate_continuation(...)` now combines the existing live
owner components into one reusable entrypoint. Callers must already hold the
exact native allocation, independent run coordinator and lease, quantitative
resource parent, closed first source broker, original requirements and retired
first worker census in the same owner process. This API reconstructs none of
those handles from a checkpoint or deadline observation.

The caller must explicitly select `max_worker_children=2` and the provider,
model, timeout, reasoning effort, output cap and candidate resource demand.
This bounds the composed lifecycle to the observed first worker and one
candidate child. It does not count arbitrary subprocesses elsewhere in the
owner; their demand remains part of quantitative resource admission.

Preparation first verifies the actual positive failed exit and immutable
wait/cleanup/broker retirement evidence, exact original requirements and
deadline. Existing task/grant, independent run and keyed native parent checks
precede fresh candidate generation. The composer then prepares the candidate
snapshot and complete prompt, seals its input, creates a fresh worker census,
reserves genuine child capacity, captures native admission and starts the
authenticated candidate dispatch broker. Original authority material is
retained across preparation callbacks and rechecked before later handoffs.

`session.run(...)` consumes its one attempt before command validation. It
appends the actual owner-generated dispatch bindings and census identity,
rejects conflicting caller-supplied binding flags and delegates birth, prompt
delivery, cancellation, wait and retirement to the existing native worker.
A preparation failure consumes preparation for that original source broker;
neither a missing acknowledgement nor an exception authorizes another attempt.
Successful child exit remains a provider outcome, without task completion or
proof acceptance.

Errors after issuance retain the exact session when available. Uncertain
native capacity retains strong custody based on the resource kernel's verified
retirement state, rather than a local release acknowledgement. Preallocated
broker objects remain reachable through partial construction failures.
`abort_unstarted()` closes the retained service before delegating unstarted
resource retirement; `recover_retirement(cleanup=...)` uses the existing
known-child recovery and preserves historical census flags. The caller retains
ownership of its original allocation, run lease and parent reservation.

This closes manual orchestration of the host round-two path. The deployed
daemon and wrapper still have separate live ownership scopes, and the ordinary
native allocation contract continues rejecting group-writable workspaces.
Their authenticated writable delegation remains separate work; this composer
enables no installed retry default, transfers no native authority and measures
no token saving. Qualification and fixture limits are recorded in
[qualification.json](evidence/native-candidate-continuation-20261016/qualification.json).

## October 16: capture a separately provisioned writable allocation

`capture_native_delegated_source_worktree_allocation(...)` now captures an
already provisioned writable worktree through the genuine native task admission.
It issues the separate exact `NativeDelegatedSourceWorktreeAllocation` type.
The ordinary `NativeSourceWorktreeAllocation` continues requiring its workspace
to remain protected from group and other writes.

The delegated root must have exact mode `03770`, owner UID and owner GID, with
the linked `.git` marker an owner-controlled, single-link regular file at `0440`.
Opened-descriptor checks retain root and marker device/inode, mount ID and mount
namespace, and refuse access/default POSIX ACLs. Directory contents may change;
the root custody and immutable marker witness may not. All canonical Git,
registered worktree, Portal, lifecycle/index and attempt controls keep the
existing strict native verification in a shared private kernel.

Capture retains the actual first `OwnerWorkerAttempt`, its original prepared
event and protected file witness, owner birth, boundary, workspace identity and
absolute deadline. The workspace and host worktree root must be in the boundary's
authenticated scope. The first attempt must remain unspent through the closing
check after native constructor callbacks. Later observations allow that same
attempt to advance through actual birth and retirement without rebinding it.

Candidate admission and the continuation composer explicitly accept this new
type and require its exact original first attempt, source-broker boundary and
selected worker UID, workspace and deadline. Their native fences reuse the
shared kernel. Delegation checks inspect retained controls and census evidence;
they do not replay candidate source after a child has been born. Source binding
retains the original publication and committed-inventory checks. The separate
`SourceLookupWorkerBroker` launch lane remains restricted to the ordinary type.

This API performs no chmod and certifies no permission transition. Its metadata
states `permission_transition_verified=False`,
`current_workspace_policy_verified=True` and `worker_isolation_qualified=False`.
It does not qualify nested-tree permission safety, continuously prevent worker
links, change group membership or reconstruct a native handle from metadata.
Owner reads and candidate generation retain their existing source checks.

Qualification uses actual host chmod, native authority, immutable controls,
local first/second children and nested resource leases. Boundary/UID admission
remains an authored host seam. The generated production permission loop, actual
separate worker UID, Docker deployment and daemon-to-wrapper ownership handoff
remain unqualified. No installed retry default or token-savings claim is enabled.
Initial fixture provisioning and renewable-parent assertion failures are retained
with the successful reruns in
[qualification.json](evidence/native-delegated-allocation-20261016/qualification.json).
