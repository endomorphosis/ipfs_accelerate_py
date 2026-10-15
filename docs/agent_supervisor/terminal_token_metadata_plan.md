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


## October 14: preserve original requirements and read a fresh candidate generation

The sealed source-broker factory now captures the original requirements before
registering the first child. Native-only launches explicitly retain that
selection; public-instruction launches replay the existing native loader and
retain its complete admitted block. That separate block can contain the full
Terminal Bench instruction, IntentIR prerequisites/prohibitions and inventory
advisories. Retaining stdin alone would omit it. Partial public-instruction
arguments refuse, and capture failure retires the newly opened service.

`owner_candidate_round_prompt` prepares an immutable, owner-local candidate
envelope from the live failed-worker snapshot and the retained requirement
handle. The envelope preserves the complete native prompt, including its
suffix, and the complete public block. It names baseline identities separately
from the candidate generation, source hashes, producer hashes and artifact
references. Its summary retains full coverage, pending-obligation counts and
unsupported frontiers. The complete artifacts stay available separately;
there is no automatic insertion of up to 16 MiB into model input. Oversized
summaries or combined prompts refuse rather than truncate required text.

Public requirements are checked against their original signed artifact,
canonical commit and sources. Historical reconstruction only checks the
previously admitted bytes; it never admits the edited working tree. The live
snapshot independently rechecks candidate source and producer freshness.
There is no deserializer that turns a checkpoint dictionary into one of these
owner handles.

Doctor residual launches have a separate context containing acceptance and
security constraints, open obligations and their own call/round budget.
Those are not captured by this new prompt preparation. The original source
service and Doctor route remain available, but candidate requirement capture
is explicitly unavailable for that selection. A native-only recapture cannot
erase that refusal. Separately, the ordinary router refuses the reserved
candidate-envelope family even when passed as a raw prompt, preventing an
advisory preparation from silently becoming a model dispatch.

`WorkerCandidateRoundBroker` creates a separate read session from an exact live
`OwnerCandidateRoundContext`. Its new descriptor retains the baseline binding
and adds candidate generation, snapshot digest, prior session/descriptor,
round identity and candidate semantic/scope IDs. Its socket, request ledger,
child registration and cursor admission are fresh. Session expiry is clipped
to the original deadline. A source cursor from the earlier session or another
candidate reader cannot be used here, including when both readers retain the
same immutable source snapshot.

The standalone `worker_candidate_round_client` runs with Python `-I` and uses
the existing authenticated socket and wire validators. It permits only
explicitly generation-pinned working-source and candidate-artifact reads.
Baseline discovery/read, baseline capsule evidence, mapped edits and numerical
baseline obligations are excluded. Ordinary source clients reject this new
descriptor. Source changes invalidate the session instead of silently moving
it to a later generation. Existing native candidate artifacts are reused,
including their unresolved proof obligations; no model or theorem prover is
called to serve a page.

The native pipeline qualification uses sealed initial stdin, a real first
child that edits source and fails, then a distinct registered Python `-I`
reader. That second process loads the real descriptor, exchanges socket
frames and pages the exact retained source, summary, capsules and formal
artifacts. It observes `SelectionMin` in the emitted Lean binding after an
`argmax` to `argmin` edit, preserves canonical source and leaves Lean
`prepared_not_built`. Page budgets and continuations use the actual transport
limits. PID ancestry, hashes, modes and source freshness are checked; only
the local host's missing cross-UID/path boundary is authored. This is not a
Docker or live-model run.

These APIs prepare a prompt and admit advisory reads to an independently
registered child. They do not authorize or automatically launch another coding
worker. The prepared prompt remains `dispatch_usable=False`. Remaining work is
the one-use owner launch admission, a candidate-specific formalization handoff,
a candidate reply grammar and round-to-invocation accounting under the existing
task/attempt and cumulative budget. Doctor, finite-proof and numerical contexts
need their own compatible candidate contracts before inclusion. There is no
accepted candidate proof-index publication or new token-saving measurement.

Source pins, results and earlier corrected qualification failures are recorded
in [qualification.json](evidence/candidate-admission-20261014/qualification.json).

The final frozen-source qualification passed all 669 tests across 19 modules
in 537.11 seconds, with no failures or skips. It covers candidate preparation,
the real-process reader, existing source and candidate clients, failed-worker
snapshots/checkpoints, public instructions, router contracts and worker wiring.
Source hashes, pinned datasets/kit commits and their actual import origins
remained unchanged. The run made no live model calls and did not run Docker,
training, Lean or a new Terminal Bench comparison.

## October 14: admit one second coding round through the shared router

The container owner now supports an explicit `--candidate-rounds 2` selection;
absence retains one round. `terminal_doctor_dispatch.implementation_argv` can
sign this selection for ordinary semantic coding with source retrieval. The
owner strips this control before preparing the first worker, captures its
provider/model/settings and complete task requirements before launch, and
keeps the original absolute deadline. This low-level option is not yet exposed
as a frozen full-suite benchmark configuration.

`owner_candidate_dispatch` binds a successful fixed worker cleanup and a waited
failed child to one continuation claim. It rebuilds candidate artifacts through
the existing native symbolic pipeline, prepares the bounded prompt, opens a
fresh read session and publishes a protected candidate handoff. The second
child receives sealed stdin and only the remaining time. Capture, preparation
and start are separate checks; repeated preparation/start cannot create another
child. Replaying an old retirement refuses before worker-wide cleanup, and a
waited second child still requires descendant cleanup before closure.

`candidate_round_monitor` owns both waits and retirements. Initial success,
signal termination, cancellation, exhausted time, registration failure or
cleanup failure never starts a continuation. Only an ordinary positive failed
first exit can consume the second round. A second failure is terminal. The
existing cleanup allowance contains processes; it does not replenish model
time. Provider output settings remain per invocation and are not represented
as an enforced cumulative token cap.

The worker and `llm_router` validate the separate candidate handoff before
provider discovery. It binds the original baseline and public requirements,
current candidate sources, native artifact pins, owner admission, settings and
deadline. A bounded summary request immediately before generation proves the
registered child's live broker access and current owner-side source/producer
checks. That preflight is observed separately and adds no artifact text to the
model input. Final input contains the complete original requirements, a
bounded candidate summary and paged-retrieval guidance.

The new closed reply grammar acknowledges candidate edits using the exact
round and input generation. It does not reuse baseline symbol aliases or claim
proof/completion. Post-edit acknowledgment checks retain those input identities
without requiring the edited source to equal its preimage. Further reads of
the frozen candidate context correctly refuse once that source changes.
Independent supervisor validation remains responsible for candidate effects.

Both pinned Grok and Codex paths use the existing shared router and receipt
accounting. Each invocation records its full provider-input digest/size and
usage separately, while retaining the original native-prompt identity. A
separate owner lifecycle observation reports round outcomes without task text
or provider output. Candidate-aware indexed-context report reconciliation and
frozen full-suite controls remain necessary before a new benchmark campaign;
baseline-only fields must not be interpreted as proof of absent candidate
retrieval or zero usage.

The concrete reporting follow-up is `terminal_context_audit._receipt`
and its input reconstruction, whose current contract equates stdin with the
original native prompt, plus `terminal_source_view_observation.collect`, which
selects only `source_view_retrieval` receipts. The generic log collector already
retains distinct router invocation IDs. Candidate-aware replay also needs
owner-private retention of the pre-dispatch advisory artifacts and observation
bindings before cleanup, rather than reconstructing them from the twice-edited
workspace. Such retained bytes would be audit evidence only, never a way to
reload dispatch admission. Full benchmark exposure then needs matching signed
archive/config controls and provider-attempt accounting checks for both rounds.

Qualification includes actual failed/second Python children, native source and
formal artifact generation, sealed stdin, protected handoff validation, real
socket pagination and router invocation/response checks. The local fixture
authors its missing container UID/path isolation and replaces providers with
test doubles. The Grok case retains a separately signed public instruction.
No Docker, live model, theorem prover or training run was performed here.
Doctor residual, finite-proof, numerical and mapped-edit continuations remain
excluded. Candidate alias-table rebuilding, accepted proof-index publication
and measured token savings remain outstanding.

Results, frozen source/dependency pins and corrected developmental failures are
recorded in [qualification.json](evidence/candidate-dispatch-20261014/qualification.json).

The final frozen-source run passed all 713 tests across 24 modules in 857.40
seconds, with no failures or skips. Source hashes and the pinned datasets/kit
commits and import origins remained unchanged. It includes the real-process
candidate router paths, mandatory broker preflight, both provider test doubles,
signed public requirements, owner cleanup/cancellation, repeated-claim refusal,
older-archive default behavior and the affected existing router/container tests.

## October 14: retain and reconcile second-round benchmark evidence

Second-round preparation now retains an owner-private historical package before
launch: the exact original requirements, candidate input, descriptor, handoff,
source snapshot and native summary/capsule/formal artifacts. The manifest is
published last, each payload is digest-bound, and successful packages survive
live handoff cleanup and subsequent workspace edits. Failed publication removes
only files created by that attempt. These bytes have no dispatch reload API and
do not establish semantic equivalence, accepted proofs or task completion.

The context audit distinguishes the original native task from candidate stdin
and reconstructs the complete candidate provider input from the retained
package, fixed retrieval guidance, workspace advisory and candidate reply
grammar. It checks file custody, source/artifact digests and the dispatched
session/generation. A missing or damaged package remains unverified. Failure
for one invocation does not suppress another invocation's independent audit.
It neither reruns the autoencoder nor assumes that twice-edited files represent
the context the second worker received.

A separate candidate retrieval observer joins the closed owner session audit
to the router's descriptor and mandatory preflight. It accounts for source
bytes, candidate artifact bytes, response bytes and preflight separately.
Baseline retrieval reporting explicitly identifies excluded candidate sessions;
it cannot report complete session coverage from the first round alone. These
byte counts are not tokenizer counts, provider usage or measured token savings.

Owner lifecycle retention and attempt reconciliation prevent an observed first
receipt from hiding a possible unreceipted second attempt. Each started or
possibly started coding child needs its own matching router receipt. Missing,
ambiguous or malformed lifecycle evidence keeps aggregate usage unknown through
the existing unreceipted-attempt reporting path. A surviving candidate archive
also indicates a possible second attempt when owner interruption prevents final
lifecycle retention, even if configuration, log markers and a second receipt are
absent. Unmatched sibling archives keep accounting unknown. Provider usage remains a
separate observation, including failed invocations and cached-input accounting.

Full-suite admission still has a concrete blocker. Every full-arm task enters
Doctor, and the current residual budget is one call, one round and 4096 tokens.
The ordinary candidate continuation intentionally does not capture those
residual obligations or that budget. The no-index arm has no required semantic
source broker. Benchmark configuration, preparation, frozen execution and Harbor
now reject an unsupported two-round selection before setup or model work;
default one-round configurations and historical hashes remain unchanged.
Enabling a full-suite two-round treatment requires explicit Doctor budget
admission and generation-aware residual rebinding. Removing Doctor constraints
or simply forwarding the new owner flag would not provide that behavior.

Qualification uses native artifact generation, actual failed/second child
processes, real socket preflight/pagination and retained-input reconstruction
with authored container isolation and provider doubles. No Docker campaign,
live provider, training or theorem-prover run was performed for this change.

The frozen 19-module regression passed all 705 tests. After the final interrupted
owner accounting correction, all 265 targeted tests across seven modules passed
against the final source state. The full regression was not repeated after that
small correction. Source and dependency pins stayed unchanged within each run;
the final evidence verifies all source pins and records the three changed files
between runs. See [qualification.json](evidence/candidate-reporting-20261014/qualification.json)
for results, source hashes, pinned import origins and retained developmental
failures. These results establish local contract coverage, not a new live
Terminal Bench score or token-saving measurement.

## October 14: preserve Doctor constraints and expose the actual budget boundary

Tracing the full-arm continuation found that `ResidualLlmBudget` was serialized
but not validated or charged by the coding route. The shared proposal admission
code charges response tokens and caller-supplied cost units; the native coding
CLI can make multiple internal model requests. Neither current CLI adapter maps
the router's requested output tokens into a native hard token cap. Consequently
the declared one-call/4096-token/one-round/one-cost-unit tuple cannot establish a
known remaining coding budget or authorize a second coding worker.

New Doctor artifacts use version 2 and validate every limit as an exact positive
integer, with closed fields and matching duplicated native capsule counters.
The retained artifact and router receipt distinguish declared limits from
unavailable native enforcement and cost conversion. Only the compact limits
enter the model advisory, adding 104 UTF-8 bytes in the native fixture; verbose
capability observations remain outside model input. Valid version 1 artifacts
still reconstruct their original advisory and receipt formats. Both versions
receive strict budget validation, and the input audit checks version 2 budget
and capability bindings against the retained artifact.

The source broker now captures actual Doctor residual requirements before the
first child. A separate typed candidate preparation preserves the exact original
advisory and obligations while binding it to the candidate generation, native
artifact roots and current source hashes. Doctor inventory outside the candidate
inventory must remain unchanged, including executable mode. The original
canonical sources must still match their admitted baseline. This preparation
does not rerun a prover or discharge an obligation, and it has no serialized
reload or dispatch API. Ordinary candidate prompt preparation refuses captured
Doctor requirements until continuation budget enforcement is available.

The ordinary Doctor router also rechecks the entire residual artifact and source
inventory after provider initialization and retrieval preflight, immediately
before the model call. A change during setup fails before provider invocation;
legitimate candidate edits made during the model call remain permitted.

Full-benchmark two-round selection remains unavailable. A concrete next step
with the current providers is a separately admitted coding policy limiting two
owner attempts, two outer router invocations and the original shared deadline,
while explicitly treating token and dollar usage as observations. This would
leave the separate residual-proposal budget unchanged. It would not claim to
limit CLI-internal model calls or tool turns. Missing usage would remain unknown
accounting, rather than create a fresh budget or imply savings.

That policy must be captured from frozen configuration and signed launch
arguments. Every possible child attempt, including an interrupted or unreceipted
one, must consume its reservation. Actual first-child retirement and the fresh
candidate generation must bind the second admission. Doctor preparation must
then be bound into dispatch, immutable retention and historical input replay
before forwarding the selection through the benchmark harness. Qualification
must cover policy omission/substitution, third-attempt and replay refusal,
deadline expiry, complete Doctor source drift and two real child/router paths
carrying unchanged obligations and proposal limits. The current change does not
raise the one-round default, spend additional provider calls, or establish token
savings.

The final frozen regression passed all 718 tests across 24 modules in 1022.78
seconds, with no failures or skips. All 33 Python source pins and the pinned
datasets/kit commits and import origins remained unchanged. Native Doctor
capture, candidate generation, late source drift, both ordinary candidate router
paths, historical input reconstruction, metadata census and benchmark reporting
are covered. [Qualification evidence](evidence/doctor-continuation-20261014/qualification.json)
also retains the corrected developmental fixture failures. Provider calls use
test doubles; this is not a live Terminal Bench campaign or a token score.

## October 14: admit and retain two Doctor coding attempts

The explicit `doctor-coding-two-attempts@1` policy now connects Doctor capture,
candidate generation, owner dispatch, worker validation and benchmark reporting.
It replaces the full-arm two-round rejection described in the preceding section
when the policy is explicitly selected. The one-attempt default is unchanged.

Select the treatment during benchmark preparation with
`--candidate-rounds 2 --doctor-coding-policy doctor-coding-two-attempts@1`, together
with `--arm full`, an enabled source view such as `--source-view-profile identity@1`,
and `--source-edit-mode disabled`. Preparation freezes the pair in configuration
and comparison controls, and requires a new runtime archive containing the
qualified implementation. Execution forwards the frozen selection into the signed
implementation command; collection reads that same selection. No-index, mapped-edit
and separate numerical-obligation broker continuations require their own supported
runtime contracts and remain
refused before setup. Keep `--numerical-formalization-profile` unset for this
two-attempt treatment; one-attempt numerical profiles remain supported.
Provider-free Doctor repairs still use their deterministic materialization route.

The policy reserves at most two owner child attempts and two outer coding-router
invocations against the original absolute deadline. Reservation precedes process
creation, so an interrupted or unreceipted launch cannot reclaim its attempt.
The second dispatch requires the actual failed child's wait and cleanup, a fresh
candidate generation, and the original Doctor artifact, advice, obligations and
proposal budget. It provides no third-attempt or serialized admission reload API.
Tokens, cost and CLI-internal calls remain observations; the unchanged
`ResidualLlmBudget(1, 4096, 1, 1)` continues to describe residual-proposal admission.

The model receives the exact Doctor advisory and a compact hash/length reference
to the candidate context. Full Doctor metadata and source inventories remain in
the protected owner handoff and archive. In the native fixture the full metadata
is 3,606 UTF-8 bytes and its reference is 147 bytes: 3,459 metadata bytes are
omitted from the candidate prompt while the advisory is preserved. This is an
input-byte observation, not a measured reduction in total provider tokens.
Before a provider call, the worker checks
the canonical baseline and the entire Doctor inventory in its workspace, including
files outside the indexed candidate and their executable modes. A new source hash
is a generation binding, not a proof that the original obligation was discharged.

The immutable archive retains the original Doctor artifact, advice, native receipt
and candidate context before the second child can launch. Historical input replay
joins those bytes to the independently dispatched handoff after runtime cleanup or
later source edits. Attempt accounting joins both child sessions to distinct router
receipts; missing usage remains unknown. A selected two-attempt treatment is marked
inapplicable to coding only when an observed provider-free materialization and the
absence of private coding-attempt evidence support that conclusion.

Final frozen qualification passed all 1,407 tests across 40 modules: 489 runtime
tests and 918 benchmark tests, with no failures or skips. All 60 Python source
pins, the datasets/kit commits and import origins, and the installed dependency
metadata remained unchanged. The two suites ran independently in parallel;
runtime qualification took 1,769.16 seconds and benchmark qualification took
282.72 seconds. Pytest reported one XML-format warning for `record_property`;
the metadata-byte measurements were present in the XML and verified when
recording the evidence. [Qualification evidence](evidence/doctor-dispatch-20261014/qualification.json)
retains source and report hashes, dependency installation provenance, and the
developmental failures that were corrected. Provider responses use test doubles;
this establishes neither a live Terminal Bench score nor provider token savings.

## October 14: first live Doctor pair and failure accounting

The [first fresh live pair](evidence/doctor-live-pair-20261014/README.md) now
exercises the full indexed supervisor with the two-attempt Doctor policy on
`largest-eigenval`. Both trials used the same frozen runtime and Grok 4.7 router
profile. The native coding turn limit was the only treatment change: default
128 versus explicit 8, with output paths necessarily differing. Both official
rewards were zero: correctness and dominance passed, speed failed.

The default-limit run used one coding child and 27 internal coding turns. Its
observed planning-plus-coding subtotal was 1,764,408 tokens, including 1,561,088
cached input tokens. Native completeness and billing totals are still unknown.
The capped run started two coding children, but only the first produced a router
receipt and neither had usable final coding usage. Its 23,347 observed tokens
cover planning alone. These results establish neither savings nor a successful
two-model-call continuation. The [pair summary](evidence/doctor-live-pair-20261014/pair-summary.json)
keeps coding usage unknown rather than treating the missing observations as zero.

The runs exposed a Doctor v2 audit serialization defect: the collector dropped
three original router-budget fields needed by the independent receipt validator.
The fix forwards those fields exactly, leaving absent or forged values invalid.
Both historical input audits remain unknown. The capped Grok stream also exposed
a `max_turns_reached` followed by `end` sequence which the old parser treated as
tainted, discarding its final payload. The original end payload is unavailable;
the retained progress counters cannot establish a recovered coding total.

Finally, the bounded child-error collector reported an ambiguous diagnostic,
preventing identification of the second child's exception after container cleanup.
The original row population is unknown: multiple valid errors and mixed malformed
data can both produce that status. Closed diagnostic retention needs to preserve
multiple valid errors without assigning unverified log messages to an owner
attempt. A fresh runtime and trial are needed
to validate the repaired observation paths live.

The public-validation design in the live-pair report is the next functional
gap. Completion currently checks output structure, while continuation reacts to
child process failure. The public speed requirement must become a signed,
executable acceptance check, with a typed candidate-validation observation usable
between attempts and independent canonical validation after publication. The
child's actual exit status, original deadline and maximum of two attempts must
remain intact. This work is still pending; the current symbolic layers have not
discharged the task's performance obligation.

The accompanying [repair qualification](evidence/doctor-live-fixes-20261014/README.md)
passed 426 distinct tests across 12 modules. Doctor v2 collector fields are now
preserved; Grok terminal/end records retain a consistent native outcome and final
usage when present; and bounded distinct child errors remain available as
unattributed observations. Contradictory or malformed data still fails closed.
All recorded current source pins match their qualified runs. Tests also confirm
explicit eight-turn propagation through two native child/router paths and the
installed worker parser, using provider/UID doubles. The live second-child cause
remains unknown and requires fresh container evidence.

The [fresh repaired diagnostic run](evidence/doctor-repaired-live-20261014/README.md)
now verifies the sole observed coding input and retains previously lost closed
child-error observations. Official reward remains zero. Both coding children
started and exited 1, but only one coding router receipt was retained; the
22,214-token subtotal covers planning alone. The new candidate error reaches
the handoff lifetime/current-clock/original-deadline predicate in the frozen
runtime. Its exact failed conjunct remains unknown because timestamps were not
exported. Grok's terminal/end sequence still taints the stream, so final coding
usage remains unavailable. These results qualify the audit and diagnostic
improvements live, while narrowing the remaining continuation and accounting
gaps. They do not establish savings or a completed benchmark task.

The subsequent [lifetime and rejection-diagnostic qualification](evidence/doctor-final-gaps-20261014/README.md)
passed 393 distinct cases with all 24 recorded source hashes and dependency pins
unchanged. An authored second-worker test reproduced the ordinary 90-second
handoff expiry after a 95-second guard-clock advance. Typed Doctor handoffs now
remain usable within a 1,800-second ceiling and the unchanged original deadline
and descriptor; ordinary handoffs retain 90 seconds. Grok progress now records
closed rejection reasons and field types without changing terminal acceptance
or exporting provider text. Historical progress schemas remain supported, and
unknown native usage remains unknown. No new live run has exercised these final
changes. Public behavioral/performance validation and live final-usage recovery
remain open; the earlier live scores and token observations are unchanged.

## October 15: validate public obligations between Doctor attempts

The new explicit `doctor-coding-public-validation-two-attempts@1` policy connects
signed public checks to Doctor continuation. Its
`terminal-public-task-profile@4` declaration binds the reviewed `largest-eigenval`
instruction and evaluator to structural, eigenpair and performance checks. Native
planning admission verifies all three checks and acceptance criteria, and source
partitioning attaches evidence to each. Existing profiles remain available.

After the actual coding worker retires, the owner runs those commands in the
isolated validation worker. A successful worker exit can now lead to a second
attempt when a typed public observation establishes a failed requirement. The
actual exit code remains separate from check outcomes. Inconclusive timing alone
does not grant a retry. Both attempts share the original deadline and proposal
allocation, and canonical validation still runs independently after publication.
These are empirical checks, not discharged formal proof obligations.

Doctor receives failed obligations and five per-size timing summaries; all
15 measurements and the full source bindings remain in protected audit evidence.
The second dispatch, retained input replay and attempt accounting join that
feedback to the exact candidate generation and prior child. Late setup, replay
and source checks cannot renew an expired deadline. Check output is bounded, and
worker cleanup must complete before any observation can support continuation.

[Qualification and the exact profile declaration](evidence/doctor-public-validation-20261014/README.md)
cover 816 distinct passing tests, including 28 pure artifact-reader cases. The
full continuation run's one test-order failure is retained and superseded only
for its three-case native module by a passing replay. Production source and
dependency pins stayed unchanged; the single test-file replacement is recorded.
Provider, host isolation and native continuation check results use authored
boundaries in these tests. Separate generated-checker tests execute real NumPy
behavioral cases. This does not establish a new live score or token reduction.

The fresh diagnostic recipe selects the new profile and policy with Grok 4.7
through `llm_router`, eight native coding turns per attempt and the existing
1,800-second shared coding deadline. A metadata-only container probe observed the
selected model with no generation calls or host credential mutation. A newly
committed archive, verified preparation and qualified reader are required before
execution. The next live result must keep official reward, public-check outcomes
and incomplete native token accounting separate from one another.

## October 15: separate acceptance criteria from task allocation

The first public-validation live trial stopped during model planning, with
official reward **0**. Its **26,617 observed tokens cover planning only**; no
coding receipt or public-validation continuation was recorded. The native CLI
returned successfully, but the parser rejected `goals[TB-GOAL].acceptance`.
Matching the retained error digest against the reviewed static error formatter
identified the cause without reading provider text: goal acceptance arrays were
limited by `max_tasks=1`, while the signed public task requires three criteria.

The parser and proposal schema now give both goal and task acceptance arrays an
independent 256-record limit, matching the existing task-acceptance bound. The
one-task allocation, byte/depth budgets, evidence checks and independent signed
admission remain enforced. The earlier native continuation fixture used declared
contract planning and therefore did not exercise this model-parser path.

[Qualification](evidence/planner-acceptance-20261015/qualification.json) records
155 passing cases, including authored router responses through actual model-plan
parsing, signed admission and DuckDB storage. Both goals and the single task
retain all three criteria; changed or omitted task criteria and altered goal
criteria are refused. Parser tests cover the record boundary, overflow, duplicate
keys, unknown evidence and unchanged task allocation. Two development assertion
failures and a concurrent change to a nonselected test file are retained and
explained in the evidence. Selected production/test bytes and dependency pins
match their successful qualification runs.

The [closed failure diagnosis](evidence/planner-acceptance-20261015/prior-planning-failure-diagnosis.json)
and [safe live readout](evidence/planner-acceptance-20261015/prior-readout-final.json)
preserve the failed trial. Container cleanup was observed. A fresh diagnostic
will use the corrected parser with the same live controls; the previous result
does not establish successful task completion or token savings.

## October 15: qualify the corrected plan and repair Grok coding routes

The corrected model-planning diagnostic now admits two goals and one task,
including all three public criteria. It records two distinct coding router
receipts with matched continuation accounting. Public structure and eigenpair
checks pass after each attempt, while timing remains inconclusive. Official
reward remains **0**. The **26,977 observed tokens cover planning only**; both
coding usage totals remain unknown. The model-planning phase took 120.576 seconds
and the native supervisor run took 521.323 seconds.

The closed diagnostics exposed two further integration defects. First, the
native eight-turn limit emits `max_turns_reached` followed by an `end` trailer
whose `stopReason` is `cancelled`. Requiring identical reasons discarded its
final usage envelope. The parser now accepts only that exact additional pair,
preserves the first `max_turns` failure, keeps the actual cancellation reason,
and observes counters from the untouched final envelope. Timeout still takes
precedence. Shape, identity, duplicate-trailer and conflicting-spend checks
remain enforced, and per-event counters are never summed.

Second, candidate dispatch requested both streaming progress and a native JSON
schema. The shared Grok adapter rejects that combination before launching the
CLI. Schema-bearing candidate attempts now use its supported JSON route;
schema-free coding retains streaming progress. New tests reach the actual
provider adapter with authored CLI responses, so a mock cannot hide this option
conflict again. Invalid candidate generation, cancellation and nonzero exits
remain failures, with their observed usage retained.

[Qualification](evidence/grok-coding-route-20261015/qualification.json) records
415 distinct passing cases. The initial three failing trailer regressions are
retained. One test comment/import cleanup supersedes its earlier capture-module
run; all selected current source hashes match. These are offline qualifications,
not a new live score or retrospective recovery of discarded counters. The
[retained model run](evidence/grok-coding-route-20261015/model-run-readout-final.json)
continues to report missing coding usage honestly.

## October 15: live router qualification and measured prompt overhead

The fresh declared-contract trial used runtime commit
`9d6ef76c8e2bb6916b68cad3563c6ec530b32736` through `llm_router`. Both repairs
reached their live native boundaries. The first eight-turn attempt still failed
with actual exit 1, but its stream remained untainted and retained one final
usage envelope: **269,557 observed tokens**. The structured second attempt now
launched Grok, returned exit 0 after one native turn, and retained **41,127
observed tokens**. Their subtotal is **310,684**, including **197,248 cached
input tokens**. The 9,270 reasoning tokens are reported separately
and are not added again to the native subtotal. Native completeness and billing verification
remain unknown; this subtotal is not a verified whole-run billing total.

Official reward remains **0**, and the supervisor reports the task incomplete.
Both rounds pass public structural and eigenpair checks; performance remains
inconclusive because of unstable timing. The second round retains the same
workspace generation. Successful provider return therefore establishes that the
route works, not that a repair or successful task completion occurred. Both
started children have distinct joined router receipts, and the two model-input
bindings agree with in-container accounting. Independent owner replay was not
performed. The trial container was removed; no services were changed.

The preceding model-planning trial observed 26,977 planning tokens and took
120.576 seconds in that phase. The otherwise matched declared-contract trial
reported zero planning provider/model calls and took 4.666 seconds; the fresh
router trial took 4.392 seconds with the same zero-call route. This route copies
an independently admitted signed task structure into two goals and one task.
It does not interpret the instruction or prove source semantics. All three
trials scored zero, and the earlier coding totals remain unknown. These single
runs establish neither a causal overall token reduction nor a speedup estimate.
Some offline tests overlapped the preceding declared trial's wall time.

The pure report reader needed a lazy-import repair: its custody validator was
already hash-pinned, but the loader removed its import path before the first
non-null custody observation. A separate qualified bundle preloads the pinned
module closure, verifies its imports, and restores the search path. Its 43
passing checks include fresh-process custody observations, invalid bindings and
unchanged prior readouts. The original reader and failed export are preserved.
The custody observation itself records passed checks and is not a new runtime
failure.

A separately qualified, observation-only component reader passes 15 checks and
joins closed byte partitions to the retained receipt hashes. It exports no
prompt/source bodies and does not infer token counts. The first-input partitions
are identical in the two declared trials:

| Input component | UTF-8 bytes | Scope |
| --- | ---: | --- |
| Original native envelope | 67,378 | Whole prompt before translation |
| Evidence array | 58,932 | Nested within that envelope, about 87.5% |
| Translated transport | 60,826 | Whole transport before later additions |
| Complete first model input | 67,380 | Actual final router input |
| Complete second model input | 91,839 | Fresh router trial; prior trial 91,841 |

Rows overlap and must not be summed. Reversible translation reduced the measured
transport by 6,552 bytes, while subsequent additions brought the final input
back to approximately its original size. Capsule/admission values account for
7,550 bytes within the 28,751-byte translated semantic object; other semantic
fields and retained nonsemantic evidence also contribute substantially. Six
semantic evidence rows are serialization chunks, not six capsules. Zero source
bytes delivered by the broker does not mean zero provider source reads: native
stream observations separately record file/tool activity.

The next contract gap is candidate prompt projection. Its current renderer
JSON-embeds the entire original native envelope, including historical indexed
evidence, then appends current context and feedback. Preserve exact original
requirements and full evidence owner-side. Add a versioned, independently
admitted provider projection that retains the literal public instruction and
all requirement/goal/scope/acceptance/authority and unknown fields, while
replacing only explicitly classified baseline advisory material with bounded
digest references. Bind original and projected hashes in dispatch and replay;
keep current source checks and proof obligations authoritative owner-side.
Existing one-capsule selection can bound part of the payload, but its benefit
needs a measured ablation and the public two-attempt combination still needs
qualification. Neither broader projection nor that ablation is implemented or
claimed by this live result.

[Closed results, reader qualification, and component measurements](evidence/grok-live-token-census-20261015/manifest.json)
retain the new observations without rewriting earlier missing token totals.

## October 15: lossless candidate provider representation

The next implementation takes a narrower step than removing indexed evidence.
Semantic and source-formalization rows can carry required invariant evidence;
their kind alone does not authorize replacing their bodies with references.
The new `native-context-object@1` provider representation instead removes one
layer of JSON string escaping from the second coding attempt. It carries the
parsed native context and its exact literal suffix, plus every other original
candidate field. Evidence bodies, unknown fields, public instructions, Doctor
constraints and public-validation feedback remain inline. Its inverse restores
the exact admitted owner input, including each evidence item's original status.

The router applies this representation after candidate handoff admission and
uses it only when the complete model input is strictly smaller in UTF-8 bytes.
Workspace instructions, retrieval guidance and final reply grammar are included
in that comparison and must remain identical. Unsupported representations and
oversized alternatives retain the valid original input. The original owner
envelope and its admission hash stay unchanged; a versioned receipt binds that
input to the selected router/model input. Historical replay reconstructs the
choice from retained owner bytes. Both candidate envelope families are refused
on ordinary routes, and frozen runtime archives bind the new codec.

The first native integration test exposed a serialization mismatch: public
checks produce finite decimal timing measurements, while the formal-IR JSON
serializer rejects every float. The codec now follows the owner envelope's
canonical JSON contract, retaining finite numbers while rejecting nonfinite
values, duplicate keys, noncanonical input and excessive size or structure.
A boundary regression also preserves a valid 255,999-byte legacy model input
when the alternative would exceed the final reply contract's limit. Review
additionally caught provider guidance that could upgrade optional evidence to
required evidence; the representation preserves the original requirement status.

This changes serialization for the conditional second coding call. It does not
minify files on disk or establish source semantics, omission authority or token
savings. The previous first call accounted for 269,557 of 310,684 observed
tokens, so improving that path remains a separate priority. Broader selective
evidence delivery needs an explicit disclosure contract and a measured ablation;
the existing required evidence cannot simply disappear from the provider view.

The broader regression snapshot passed 294 checks. After the guidance-only
correction, 117 focused checks passed on the final bytes: 104 codec cases,
12 provider/router/archive cases and one full indexed public+Doctor continuation.
Source and dependency pins remained unchanged within every qualification run.
The final native fixture reduced the complete second input from 46,230 to
42,785 bytes (3,445 bytes, about 7.45%). Both real router children, continuation
admission, retained input audit and replay after cleanup/later source edits run
natively; provider responses and public-check execution are authored fixtures.
Five forged receipt variants are refused. These are representation measurements,
with zero live provider calls and no inferred token savings.

The fresh Docker trial used commit `cef1bb3764aa2ea9ed453a2feec5c240cca2dc54`
through `llm_router`, with the prior Grok profile, eight-turn cap, two-attempt
policy, declared planning and resource allocation. Its selected second input
fell from a reconstructed legacy 91,842 bytes to 83,991 bytes: 7,851 bytes,
or 8.55%. The two input observations agree with in-container accounting.
The closed reader validates selection arithmetic and receipt joins; it does not
independently replay the live owner's private retained input.

| Observed native usage | Previous trial | Projection trial | Change |
| --- | ---: | ---: | ---: |
| First coding invocation | 269,557 | 312,698 | +43,141 |
| Second coding invocation | 41,127 | 38,573 | −2,554 |
| Observed subtotal | 310,684 | 351,271 | +40,587 |
| Official reward | 0 | 0 | 0 |

The second invocation used 6.21% fewer observed tokens in this pair, while the
subtotal increased 13.06%. Neither observation establishes a causal effect from
one run per revision. These native subtotals include cached input; reasoning
output is reported separately and is not added again. Full usage completeness
and billing remain unverified. The first prompt's byte length stayed at 67,380;
its byte identity was not independently compared. Both first invocations hit
the eight-turn cap with exit 1 and an untainted final usage envelope. The new
first invocation generated 12,632 output tokens versus 6,538 before, with higher
subsequent input/cache counts. The closed stream observations show ten tool
calls in each run but do not identify the tool-result histories responsible for
that difference. A smaller second prompt therefore did not reduce this run's
overall observed usage.

The second invocation returned exit 0 after one turn, while retaining the same
workspace generation as the first round. Both rounds passed public structural
and eigenpair checks, but public performance remained `timing_unstable` and
inconclusive. The task is incomplete and official reward remains zero. Native
supervision took 675.34 seconds (previously 461.32); whole execution took 907.67
seconds (previously 816.65). Our offline qualification tests had finished before
the new live execution. The exact trial container is absent after collection, retained
artifacts are preserved, and no services were changed.

The next bounded first-input experiment is the existing
`capsule-references@1` working-set profile against `legacy` on the same qualified
revision. It defers only allowlisted capsule provenance using authenticated,
hash-bound retrieval. The previous closed census measured just 6,395 bytes of
capsule values within the 67,380-byte complete first input, before reference and
instruction overhead. Raw source, other semantic fields and required formal
evidence remain outside that profile. The existing census-only path does not
support working-set mode; qualify a capturing-provider measurement or an
additive working-set census before that ablation. Larger reductions require
finer field accounting and an explicit owner-selected disclosure contract with
authenticated artifact paging. Existing capsule retrieval cannot authorize
dropping arbitrary required world-model or formalization evidence.

[Qualification, closed live results, and next-step source analysis](evidence/candidate-prompt-projection-20261015/manifest.json)
retain both development failures and the final evidence without replacing prior
results or upgrading missing usage into a verified total.

## First-input working-set census qualification (2026-10-15)

The additive router census and installed worker flag are implemented in
`5c1ab44ede`. They use the normal complete-input selector and authenticated
capsule preflight, returning before provider discovery or allocation. A closed
component census `@2` separates additional known semantic fields and
intent/world/retrieval evidence while retaining the old default `@1` output.

The [qualified measurements](terminal_working_set_census.md) show the limit of
this representation: the native indexed public/Doctor fixture would grow from
23,202 to 23,249 bytes, so it correctly retains the original input. Another
dispatch fixture saves 1,603 bytes. Complete capsule retrieval, exact inverse,
required-field preservation, stale-source refusal and census/input identity
passed. The final 235 tests passed with unchanged source and dependency pins;
no live provider calls were made.

The remaining first-input pilot prerequisite is a declared-plan census profile
in the Terminal Bench owner driver. Its current metadata census requires an
actual model-generated planning invocation and cannot truthfully represent a
zero-call declared plan. Preserve that contract while adding the explicit
profile, then collect current public-task component measurements and count
retrieval exposure in any subsequent live comparison. No new benchmark reward
or total token saving is established by these fixture results.

## Fresh declared-plan working-set census (2026-10-15)

The owner prerequisite above is complete at `92b540f004`. The integration passed
267 runtime tests, with 19 additional launcher/exporter tests. A fresh public
`largest-eigenval` Docker run passed full indexed preparation, pinned Source384
inference, retained declared-plan/admission checks, native worker dispatch,
complete-input owner replay and cleanup. No model generation or official
verification ran. [Closed evidence and component sizes](evidence/working-set-owner-census-20261015/README.md)
are retained with source/dependency/asset pins and hashes.

The complete input shrank from **67,201 to 67,120 bytes: only 81 bytes (0.12%)**.
The 24 deferred fields across two capsules are provenance; behavioral fields and
raw source remain inline. Most bytes are in the remaining native context
(30,343) and translated semantic context (28,751). This result does not support
running a costly token-efficiency ablation of the existing capsule-reference
profile alone. It is also not a comparison against native Codex or a new score.

The next bounded projection should keep required evidence inline but encode
its canonical JSON artifacts as typed objects with exact original-chunk
reconstruction. The remaining intent/world, formalization and retrieval rows
contain 1,373 bytes of JSON-string overhead: a gross ceiling of about 2.04% of
this complete input before replacement metadata. Reuse existing artifact/chunk
bindings; intent/world spans two chunks, so independently parsing each summary
would be incorrect. Preserve unknown or noncanonical summaries unchanged and
select a candidate only after complete-input comparison. This is a design
candidate, not an implemented or measured saving.

In parallel, measure repeated content inside the 6,266-byte semantic pack before
attempting deduplication. Its size alone does not establish redundancy. Larger
reductions still require explicit source-bound selection of required evidence
and accounted retrieval; arbitrary deletion and symbol renaming alone cannot
establish reduced full-run token usage.

## Reversible inline evidence and pack accounting (2026-10-15)

The next projection above is implemented at `78c1494bfb` and qualified in a
fresh public Docker census. Required intent/world, formalization and retrieval
artifacts remain inline as typed objects; their original UTF-8 chunks and the
complete original semantic transport reconstruct exactly. Unknown fields and
ineligible bodies stay literal. The independent complete-input selector keeps
the original whenever the candidate is larger. The new candidate is available
only for measurement; ordinary generation remains unchanged.

The extended `router-semantic-working-set-census@2` measures both candidates
against the same original input. Direct router `@1` output is preserved, while
the explicit container-owner census selects `@2`. The registered child and
owner independently reproduce the full observation after source cleanup.

| Complete first coding input | Bytes | Reduction |
| --- | ---: | ---: |
| Original | 67,201 | — |
| Capsule references | 67,120 | 81 (0.12%) |
| Inline evidence objects | 66,452 | 749 (1.11%) |

These are separate representations of one prepared input, not a composition or
an ablation against native Codex. Of 1,373 bytes of gross JSON-string overhead,
624 bytes are replaced by inline structure and guidance. No model generation
or official verification ran, so there is no new token score or task reward.
The indexed profile consumed its pinned Source384 checkpoint; intent autoencoder
inference was disabled. This does not qualify all IR checkpoints or prove the
formalized behavior correct.

The new pack census measures 6,266 bytes, including a 3,298-byte reference array
and 1,918-byte inner contract. Fifteen known identifier slots cover five native
identities and 763 serialized value bytes; equal IDs do not make their roles
redundant. The next source-based design factors duplicate token estimates and
identical reference envelope defaults with an exact inverse, preserving roles,
requiredness and caveats. This is not yet implemented or measured. The larger
history/tool-response cost also remains to be measured before claiming total
token savings or enabling a paid ablation of these small byte reductions.

All 178 runtime tests and 33 launcher/exporter tests passed with unchanged
selected pins. The fresh run completed owner replay, worktree/worker cleanup,
source-session accounting and two exact container-absence checks. Retained
[closed evidence and next pack design](evidence/inline-evidence-census-20261015/README.md)
include the final freeze check; private prompts and owner archives are excluded.
