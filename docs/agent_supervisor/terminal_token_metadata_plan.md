# Reduce complete supervisor tokens with native metadata and checked capsules

The objective is fewer total input and output tokens over the entire Terminal
Bench supervisor run, while retaining the planning session and `llm_router`.
Database metadata should let the supervisor reuse established facts, formal
results should identify which obligations remain, and semantic capsules should
give the model the relevant current evidence without repeating its bookkeeping.
Minification is a representation step within that workflow.

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
The current JSON/metadata codecs restore their transports; they do not transform
the executable repository to and from an editable minified state. A historical
read-only Python preview exists in retained snapshots, but its sidecar stores
removal hashes rather than removed bytes and has no inverse/edit mapper.
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
