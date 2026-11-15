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
| `RepositorySourceSession` / `repository_source_policy` | Default 512 files and 16 MiB, bounded repository snapshot and native index; original/compact file or supported definition reads and guarded canonical edits | Works with world packets and retained basic-context Terminal collection, now including ordinary nomination joins; production worker handoff remains pending |

Neither route intercepts arbitrary terminal tools. Neither establishes that the
model read every path through its compact representation. Reuse the second
route for the immediate world-packet experiment. Broader scope admission,
cross-language mappings and a transparent compact filesystem are separate
extensions; building another snapshot codec would not close the measurement
gap. The [source-view plan](reversible_codebase_views_plan.md#current-worker-routes-and-the-whole-codebase-hypothesis)
records the different restoration and editing contracts.

### Next experiment and acceptance gates

The collector now has a dedicated retained world-input path with ordinary
nomination joins, described below. A separate compiler preserves a complete
pinned initial daemon parent while issuing a fresh source packet. Next connect
production worker preparation and dispatch through authenticated current-owner
admission and fresh issuance
against the worker's actual source session. Formalization and signed scope
contracts need their own integration. Historical replay must remain separate
from new live owner admission. Preserve complete
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
| [DeltaTaskPacket](../../ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py) | Task/tree/plan/evidence identities, impacted symbols, residual obligations, validation requirements and deterministic-first admission; connected to the opt-in world packet route with bounded historical export and Terminal collection below | Qualify the production owner-to-worker handoff and fresh live admission after restart |
| [Program-world context adapter](../../ipfs_accelerate_py/agent_supervisor/context/program_world_context.py) | Selects material identities using supplied costs; caches prefix identity bytes | Resolve admitted material bodies and measure final provider text; identity reuse is not evidence of provider prefix retention |
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

### Ordinary nomination bundles from native Terminal preparation

Real Terminal preparation publishes a nomination bundle even when it selects
no advanced formalization family. The
[ordinary nomination audit](../../benchmarks/agent_supervisor/container_coding/terminal_world_nomination_audit.py)
now admits that case by checking the independently retained bundle digest,
exact task CID and alias, and all three required context paths and digests.
It requires exact nominated semantic text, reconstructs the native retrieval
projection from the retained index/query and SHA-256/CID-bound source blocks,
and reconstructs the minified intent-world projection. These checks preserve
the existing compiled context and public-task checks.

Only bounded nofollow reads reach retained artifacts: at most 2,000,000 bytes
per file and 8 MiB per receipt. Native loaders operate on private staged copies
of those pinned bytes. The original worker checkout and original source files
are unnecessary; the canonical preparation repository, retained bundle, context
artifacts and source blocks must remain available. Full semantic-producer
reconstruction is explicitly unperformed. Empty retrieval, refreshed or rebound
nominations, selected formalization families and signed Terminal scopes remain
unsupported by this path.

The [nomination qualification record](evidence/world-nominations-20261014/qualification.json)
records **659 passing tests**: 61 focused cases and 598 compatibility regressions,
with no failures or skips in the final runs and identical source manifests.
The end-to-end fixture uses real signed preparation, native indexes, daemon
compilation, packet capture and the bounded audit child; its planner proposal
and provider response are authored. Fault cases cover changed nominations,
repacked context, symlinks, FIFOs, byte limits, referenced snapshots, staged-file
isolation and cleanup. The report retains the earlier fixture failures.
Historical projection checks do not reverify
signatures, source or owner currentness, token counts or provider delivery, and
grant no proof, execution or completion authority. The next production step is
an authenticated owner-to-worker bridge that issues a fresh packet against the
worker's actual source session and checks current owner state before dispatch.
Replayed artifacts cannot supply that admission. Matched complete-run token
measurements remain pending.

### Preserve the full daemon parent during fresh source compilation

The [full-parent compiler](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_world_parent.py)
adds a bounded in-process preparation API, `compile_implementation_world_packet`.
Its inputs include an existing source-session adapter, the complete daemon
capsule and independently supplied expected CID, the exact rendered parent
prompt, and separately pinned public task text. It requires the prompt to equal
the native capsule render byte for byte. This excludes appended feedback or
retry instructions until they have their own preservation contract. The current
profile accepts only initial implementation attempts without parent/delta
bindings, omissions or deferred references.

The compiler retains the original goal, authority values, scope, acceptance,
objective and policy identities, and all selected evidence, including optional
references. Original caller, stage and parent identities remain explicit in the
added provenance. It rejects any selected reference dropped during recompilation.
Every selected reference must explicitly match the parent's repository and tree;
only those outer bindings change to the fresh source owner. Reference bodies,
nested semantic/world identities, metadata and paths retain their original
values. Write paths and protected paths must agree between the scope and edit
policy, with no overlap. The retained timeout policy and validation commands
remain instructions; preparation executes neither.

The new compilation and packet must fit the parent's declared token and byte
ceilings, including any declared provider-window reserves. Token checks use the
current compiler's counter and record its identity. They do not replay the
parent's tokenizer or establish the final provider token count. The runner owns
final accounting of its rendered advisory and output schema, which requires a
separately selected provider-bound counter. Preserving the whole parent may
increase input size; this API establishes context preservation, with token
savings still unmeasured.

Fresh source checks establish the new session's current local source state.
They do not establish currentness or correctness of inherited semantic, retrieval
or world evidence. Matching the parent and task pins does not verify signatures,
native task claims, allocations or the live intent owner. Historical packet
reconstruction cannot provide that authority. Source-formalization references
and live intent-observation references are excluded from this first profile.
The [qualification record](evidence/world-parent-20261014/qualification.json)
records the preservation, refusal, fresh-process and native Terminal fixture
checks and their limits.

The production path still needs authenticated owner-to-worker admission, exact
task/allocation custody, the real UID boundary and a bounded transport into the
owner's private invocation store. It also needs CLI selection and a rebuilt,
qualified runtime archive. Current owner checks before dispatch and matched
live-provider runs with complete native token accounting remain separate gates.

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
