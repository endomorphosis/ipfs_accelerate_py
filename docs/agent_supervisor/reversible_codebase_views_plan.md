# Repository-wide compact views, source restoration and total tokens

Status: reviewed improvement plan, 2026-10-12. The source audit starts from
accelerator `1f6d998a92db4024f35b57bbdb183a6626a7fa31` and incorporates parallel
work published through `ffc6643826e546e5a0b3436b766b22f39f243b71`.
The contracts and stages below are proposed unless explicitly marked current.

Stage 1 now has a [retained committed-source inventory API](repository_source_inventory.md).
It reuses the datasets complete chunked producer and caller-owned CAS, then
replays exact historical bytes and Git modes after the original checkout is
unavailable. It preserves explicit Gitlink references and provides bounded host
metadata pages. It creates no filesystem checkout, compact source transform,
current-source authority or coding-model expansion tool. Those remain later
stages; byte fidelity alone establishes no token savings.

The [reversible read-only source views](reversible_source_views.md) now add
complete KEEP/DELETE byte maps and repository-wide view inventories in the same
CAS. Default Python views preserve comments; comment removal is a distinct
opt-in profile. Unsupported source remains exact and Gitlink frontiers stay
explicit. Deterministic replay restores unchanged source and refuses edited
views. Mapped edits and token savings remain separate stages. The opt-in
[worker source reader](worker_source_views.md) now supplies task-scoped compact
views and exact expansion using the same codec and isolated child lifecycle.

The [current-source lookup](current_source_lookup.md) now attaches a bounded
original-source operation to the existing typed owner. It separately checks
native task custody, historical CAS bytes and clean selected working-file
equality, with grant-bound continuation and full response byte accounting.
A host-issued subprocess CLI exercises that operation. Its typed grant remains
separate from the coding-worker reader, which admits the already verified
semantic producer scope without copying host credentials. Required facts remain
inline. The semantic reader now supports a separate working-candidate snapshot
after edits, with generation-bound expansion and stale-continuation refusal.
The [candidate context reader](worker_candidate_context.md) separately rebuilds
session-local capsules and pending formulas; accepted formal-index publication
remains open. It also exposes a compact catalog and exact path/symbol/function/
obligation-kind selections. Returned native rows retain their identities and
explicit filtered-out counts. This is selective disclosure with narrower
coverage, not lossless compression of a full formal artifact.

The [host-only native task guard](current_source_task_admission.md) now captures
actual one-attempt claim, daemon grant and full-body custody through the
existing coordinator, and can refence the native source broker. It does not
activate the ordinary Terminal remote allocation, compiled-context or actual
coding-child handoff. Native restart and working-overlay admission remain
separate from the semantic candidate-refresh channel.

The proposed design maintains a compact view of an entire admitted repository
while retaining exact original source in its existing content store. Deliver
only the current task's relevant views, fetch exact source when needed, and
apply accepted edits to the original bytes. This can reduce repeated source
and reasoning in the model's conversation. It needs both source/edit fidelity
and complete-run measurement; an inverse alone establishes neither token
savings nor correct execution of transformed source.

## What the audit found

| Work | Current meaning | Remaining boundary |
| --- | --- | --- |
| [`semantic_context_runtime.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py) | Bounded source snapshots, source hashes, capsules and compact JSON | JSON restoration is not a whole-repository source rewrite |
| [`semantic_metadata_view.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py) | Readable shared metadata with exact transport restoration | Edited transports fail retained identity checks; there is no source edit mapper |
| [`semantic_instruction_view.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_instruction_view.py) | Reference one exact duplicated instruction already present verbatim | Preserves required instruction text; does not minify executable code |
| Historical `code_view_minification.py` | Read-only Python view removing ordinary comments and trailing trivia | Absent from this published runtime; sidecar has offsets/hashes, not removed bytes; no inverse or edit mapper |
| [`doctor_contract_candidate_runner.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/doctor_contract_candidate_runner.py) | Exact-byte candidate materialization with preimage checks and allocated-worktree controls | Reusable write boundary, without compact-view edit translation |
| [`published_task_context.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py) | Current ranked retrieval refreshed before successor capsules | Current source lineage is available; ranking quality and token benefit remain unmeasured |

The historical helper has SHA-256
`0c3ed9348189ed0bf5f45a1c11c24fc3a6acff9c45d9400c853181fe58e3bebd` in retained
October 7 and October 11 maintenance snapshots. It preserves physical lines,
names, imports, indentation, meaningful token spellings, strings/docstrings,
and conservative protected comments. It accepts strict UTF-8 Python only.
It labels its output read-only and requires exact source before editing.
Its removed-comment digests cannot recover the removed bytes. Its AST/token
checks establish structural properties, not general semantic equivalence.
The [audit evidence](evidence/reversible-codebase-views-20261012/README.md)
retains its exact historical bytes and a small provider-free observation.

An untracked architecture plan in the canonical dirty checkout contains related
counter/projection proposals. Its existence does not put those APIs in main.
The audit retains that file's hash and treats it as design input. Foreign
uncommitted work and original model assets are preserved. This plan reuses the
published metadata, retrieval and source-materialization seams rather than
installing a competing context compiler or database owner.

## Why reversibility and token savings need separate checks

Let `S` be original repository bytes, `V` a compact view, and `R` host-retained
restoration data. The unchanged-source contract is `restore(V, R) = S`, with
byte equality and independently checked file identities. `R` can include the
original source blobs; it need not travel to the model. If both `V` and all
removed source are repeated in the prompt, their wrappers can increase input.

An edited view introduces a different contract:
`materialize_edit(S, V, mapping, edit) = S'`. Restoring an old snapshot after an
edit discards that edit. An inverse for an unchanged view cannot safely infer
where arbitrary rewritten compact text belongs in the original source.

Bytes, local tokenizer counts and complete native session usage are different
measurements. Shorter identifiers, dense punctuation and removal of whitespace
have no accepted provider-token benefit until the complete rendered input is
counted with a compatible provider/model tokenizer. Count view guidance,
aliases, manifests actually sent, tool definitions, reply instructions and
expansion responses. A GTE encoder's tokenizer or Python's lexical tokenizer
does not qualify the coding provider's count.

The [qualified earlier pair](evidence/terminal-reply-contract-pair-20261007/README.md)
used 246,992 baseline versus 327,117 compact-transport total native tokens.
Both received official reward 1; compact used 80,125 more tokens (32.44%).
Initial coding input was smaller, but generated plans also differed. One pair
does not identify a causal effect or show that all compact representations fail.

The [latest live instruction census](evidence/terminal-ranked-successor-20261012/README.md#current-measurements)
measured 65,222 to 64,254 input bytes, 968 fewer (1.484%), and 25,315 planning
tokens. It made zero coding calls and ran no official verifier. It therefore
does not establish total-token savings. Across its retained three attempts,
50,500 planning tokens are known; first-attempt usage remains unknown.

## Use the existing stores and logic systems together

| Component | Proposed contribution | Information the model needs |
| --- | --- | --- |
| DuckDB metadata/indexes | Locate current paths, source spans, dependency closures, views and accepted proof references | Bounded task-specific results and an explicit unresolved frontier |
| Existing Quack owner | Serve closed `source.get`, `capsule.get` and later `symbol.query` operations | Current exact snippets or admitted capsules, with counted responses |
| DuckLake | Retain revision transitions, delivered context identities and experiment observations | Relevant changed facts at decision boundaries |
| Formal systems/proof cache | Avoid re-deriving currently checked supported obligations | Property, native verdict, assumptions, dependencies and residual obligations |
| Semantic capsules | Summarize current dependency evidence and remaining work | Required task facts and needed source remain available |
| Compact source views | Navigate many files without sending irrelevant bodies or repeated trivia | Readable code, original source locations and expansion/edit rules |
| Existing autoencoders | Rank spans, dependencies and obligations | Candidates, with current-source checks before use |

Metadata locators, historical rows, latent vectors and generated formulas are
not proof receipts. Keep the native proved, disproved, inconclusive,
unsupported, error and cancelled verdicts distinct. Existing scoped header
admissions and [sealed inventories](header_proof_inventory.md) retain local
assumptions and residuals. They do not certify an entire repository or the
numerical `largest-eigenval` task. Replace duplicate status text only after
current native lookup/render checks and a measured input benefit.

Whole-repository inventory means the complete **admitted public scope**, not
one unbounded prompt or silently skipped files. Enumerate it in bounded pages
and record covered, excluded, unsupported, unavailable and pending entries.
Preserve exact bytes for unsupported languages, binary files and non-UTF-8
files. Retain paths, file kinds, executable modes, newline/encoding bytes and
symlink identities under the existing repository policy. Do not treat generated
files, external dependencies or hidden evaluator material as ordinary indexed
public source without that scope's admission.

Store source/view mappings beside the existing content/index records. Proposed
minimum identities are repository/task scope, source tree/revision and blob,
relative path/file kind/mode, view digest, transform/schema version, retained
span mapping, hidden-region references and refresh parent. Token observations
also bind provider/model/tokenizer/revision, complete input digest and counting
scope. A content hash associates bytes; owner admission and currentness remain
separate checks. A missing blob or stale mapping requires expansion/rebuild or
refusal, never an approximate inverse.

LegalIR, SecurityIR, CodebaseIR and IntentIR keep their separate inventories,
DuckDB/DuckLake namespaces and existing Hugging Face/checkpoint identities.
Within each, retain family, schema/version, decoder task, 8D/384D/768D dimension
and token profile. Reuse prior embeddings and checkpoints. Source-view fidelity
is an exact-byte storage/codec responsibility; a semantic decoder's training
loss cannot certify original comments, formatting or arbitrary source edits.
No new training is needed for the first source-view experiment.

## Provide repository-wide views with explicit edit handling

1. **Capture the baseline.** Acquire the admitted source inventory through the
   existing producer. Retain exact original blobs and repository metadata.
   Bind a stable manifest and explicit unsupported/excluded frontier. Persist
   references through the current owner rather than a second database service.
2. **Build read-only views.** Start with a conservative versioned Python display
   transform derived from the audited historical helper. Retain original blob
   references and piecewise view-to-original byte mappings, including hidden
   regions. Keep names, signatures, strings/docstrings and relevant explanatory
   comments. A keyword whitelist alone cannot establish which comments matter
   to a task. Omit unsupported transformations and retain the exact source view.
3. **Expose bounded expansion.** Qualify actual model-tool access to exact source
   and capsules through the existing owner. Bind task/scope/root, owner
   generation, request identity, byte/result limits and continuations. Recheck
   current source after edits; return completeness and unresolved frontier.
   Required source remains inline until this interface works through the real
   coding route. An unregistered resolver handle cannot replace required text.
4. **Expand before edits initially.** The first profile edits exact original
   source in the ordinary workspace. Preview navigation may be compact, while
   normal bounded reads and patches preserve the existing validation lifecycle.
   This tests whether navigation saves enough tokens to pay for exact-source
   expansion, without requiring arbitrary compact-text inversion.
5. **Add a separately versioned mapped-edit profile.** Accept explicit span edits
   with original/view identities, original preimage bytes and replacement bytes.
   Map only an unambiguous retained region. Preserve every untouched original
   byte. Expand source for edits spanning hidden trivia, stale or ambiguous
   anchors, or symbol/import rewrites until those operations have their own
   reviewed mapping rules. File moves/deletes and symlink changes require their
   own supported native operations; exact-source expansion alone cannot enable
   them. Conflicting concurrent edits refuse.
   Arbitrary replacement of a minified file is not an accepted mapped edit.
6. **Materialize and validate the proposal.** Construct `S'`, preflight all old
   file bytes and permitted paths, and use the existing exact-byte candidate
   materializer in an allocated workspace. Its current regular-file modify/create
   lane accepts at most eight outputs; moves, deletes and symlink edits remain
   unsupported by this lane. Preserve admitted modes and file kinds.
   Retain baseline, accepted patch and resulting bytes so reversal can restore
   the baseline while forward replay preserves the edit. Apply ordinary focused
   tests, native proof revalidation and official acceptance to `S'`.
7. **Refresh against the appropriate source state.** The coding worker's
   [source reader](worker_source_views.md#read-working-files-after-edits) now
   provides a session-local candidate refresh boundary for source views.
   Mid-coding formal indexes still need a separate rebuild boundary. Bind
   the working source revision, label it candidate, and refuse prior proof reuse
   after dependency drift. The existing ranked successor API requires accepted
   publication and a completed native task; it is not that mid-coding API.
   After acceptance, use its pinned rebuild/lineage rules to regenerate affected
   retrieval and capsules from accepted source. Invalidate changed proof
   dependencies. Retain old blobs/views for historical replay without labeling
   them current.

This supports a repository-wide dual representation: exact source for execution
and accepted edits, compact views for navigation and context. Running tests
against rewritten minified files is a separate research profile. It would need
language/runtime validation for source introspection, tracebacks, line mappings,
tool directives, licenses, public names and dynamic imports. AST equality or
successful source restoration alone does not qualify that execution profile.

## Qualification and experiment order

The [working candidate context reader](worker_candidate_context.md) now rebuilds
native capsules and pending numerical formulas for a pinned edited snapshot.
It shares the isolated worker's read-only source session and accounts artifact
bytes separately. This advances the mid-coding refresh portion of stages 2–3;
accepted DuckLake/proof publication, mapped edits and complete-session token
savings remain separate qualification work. Full formal artifacts can be much
larger than source, so the benchmark must charge summaries, expansions and
refreshes actually requested by the worker.

The [selected-reader Docker probe](evidence/candidate-context-selection-20261012/README.md)
completely retrieved one catalog, one capsule
and one obligation in 19,858 wire bytes. Its 4,545-byte obligation selection came
from a 476,030-byte full formal artifact, while the source file itself was only
642 bytes. Thus a smaller formal retrieval does not establish an advantage over
reading source. Keep whole-session usage and official task results as the gate
for preferring this treatment in a benchmark.

| Stage | Concrete deliverable | Acceptance before advancing |
| --- | --- | --- |
| 0: reconcile current work | This source audit and plan, incorporating ranked CLI/successor refresh and the latest census | Historical helper and foreign untracked plan remain clearly identified |
| 1: source inventory/restoration | Bounded repository manifest, exact blobs, view mappings and explicit coverage frontier | Cold restore is byte-identical for supported and passthrough entries; missing/tampered/stale blobs refuse |
| 2: read-only disclosure | Qualified source/capsule operations through the actual coding interface | Current-source/owner checks, bounds, continuations and complete response accounting; unchanged required coverage |
| 3: original-source editing | Preview navigation plus expansion and normal patch lifecycle | Original tests and accepted edit bytes survive refresh; unrelated source/trivia remains exact |
| 4: optional mapped edits | Explicit preimage-bound edits restricted to retained regions | Untouched bytes preserved; hidden/crossing/stale/ambiguous edits refuse or expand; forward/backward replay checked by actual bytes |
| 5: provider-bound census | Count complete candidate and baseline inputs with the same qualified counter | Strict improvement after every wrapper; counter failure/larger candidates retain original representation |
| 6: matched complete sessions | Separately versioned treatments on representative tasks | Complete native usage and original official rewards, including all failures/expansions/repairs |

Restoration controls should include CRLF/LF, no final newline, Unicode, encoding
declarations, tabs, multiline strings, f-strings, docstrings, protected comments,
binary/non-UTF-8 passthrough, executable modes, allowed symlinks and empty files.
Edit controls should include insert/delete/replace, duplicate anchors, hidden
comment boundaries, concurrent source drift, out-of-scope paths, partial writes,
restart and actual forward/inverse application. A diff containing expected hash
strings is insufficient. Use meaningful byte/preimage assertions and ordinary
runtime tests; keep AST checks distinct from correctness claims.

First complete the already prepared lexical baseline versus instruction
deduplication comparison and, separately, lexical versus ranked capsules. The
[archive04 recipes](evidence/terminal-ranked-successor-20261012/archive-04/matched-comparison-recipes.json)
are preparations, not new deployment or live results. Their admitted default
Intent handling differs from the census's disabled Intent profile; preserve
that distinction instead of treating the census as a matched baseline.

Then compare exact source versus compact read-only navigation with expansion,
holding metadata/capsule treatment constant. Evaluate mapped edits afterward,
and formal-status reuse and history deltas as separate treatments. Combine
treatments only after their individual behavior and costs are understood.
Include multi-file navigation, low-comment code, documentation-sensitive changes,
generated/unsupported files and tasks that force exact-source expansion.

Charge final cumulative native input plus output for every retained planning,
coding, lookup, continuation and repair session. Do not sum cumulative snapshots
or double-count cached input, reasoning output or tool text already included in
provider input. Preserve the route's cache accounting semantics and missing
usage as unknown. Local transform/index/inference work is separate compute;
refreshes requiring an LLM contribute their native usage. Database byte savings
and local counts remain separate from total provider tokens and billing.

Report official success rate, all-attempt tokens per official pass, per-task
totals, expansions, turns, repairs, input/output split, latency and available
reported cost. Hold public task inputs, provider/model, reasoning, planning,
reply contract, runtime/resource budgets and setup policy fixed. Retain failed
attempts, fallback selection and changed plans. Use repeated balanced matched
pairs before a repeatable savings claim. If expansion or repair erases the
benefit, keep exact source as the selected view for that task/profile.
