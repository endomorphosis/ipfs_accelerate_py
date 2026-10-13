# Repository-wide compact views, source restoration and total tokens

Status: reviewed improvement plan, 2026-10-13. The source audit starts from
accelerator `1f6d998a92db4024f35b57bbdb183a6626a7fa31` and incorporates parallel
work published through `f86f955eb4bb63613a2c3aaffa968e87382044f1`, plus the
host-side compact-edit mapper and isolated application qualified below.
The contracts and stages below are proposed unless explicitly marked current.

The October 13 [reading-workflow audit](evidence/context-reading-audit-20261013/README.md)
reconciles accelerator `cdca08b03c9ca015b7ba8799b9161de6ac20ce4d` and the parent
datasets pin `b95938c2a816075466f8dc84f339cf622414030d`, including parallel
compact-edit and LegalIR permission-retention publications. It distinguishes
source already omitted during preparation from later reads, and identifies
shared view policy, discovery transport and read tracing as the next gaps at
that revision. Shared policy and client discovery are now implemented below;
complete inventory-to-task admission and general read tracing remain open.

The subsequent [matched Docker read/edit experiment](source_view_read_edit_pair.md)
implements strict comparison of complete scripted traces. Exact and compact
views produce the same source edit; one-pass prompt plus default CLI exposure
falls from 79,362 to 79,229 bytes (0.17%). This measures neither model tokens nor
IR/proof reuse. It leaves general read accounting, context substitution and
matched live-model qualification open.

The [complete-context follow-up](context_projection_pair.md) found that both
public program bodies were already deferred in this fixture. Reusing the
existing lossless metadata projection with the exact source reader reduces
scripted exposure from 79,362 to 78,477 bytes (1.12%). Its strict replay and
289 passing regression tests qualify the byte comparison. Its subsequent
GPT-6.1-sol native pair used 313,979 versus 313,980 total tokens, with official
rewards 0 and 1 respectively. There is no total-token reduction or causal
advantage established by this single pair. IntentIR lacked a checkpoint and all
four Source384 candidates remained unverified. No additional source-omission
profile was introduced.

The October 12 follow-up audit includes publications through
`48f442b4ca468b8cf2e858bca72a676812344302`. Unchanged repository restoration
already works. The next token-efficiency priority is to identify which source
reads and prompt content a representation actually replaces, then measure the
complete session. Editable compact workspaces and uniform UI/UX IR model
routing remain distinct proposed capabilities. The [follow-up audit and
documentation checks](evidence/compact-context-plan-20261012/README.md) retain
the source/research scope of this plan change.

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
views. A separate [mapped-edit adapter](reversible_source_views.md#apply-through-the-existing-isolated-worktree-owner)
now translates a supported compact span into original bytes and applies it in a
fresh fenced candidate. Model-tool integration and token savings remain separate
stages. The opt-in
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

The separate [native worktree allocation guard](current_source_worktree_allocation.md)
joins that task handle to the sealed Portal execution projection, managed
worktree lifecycle and Git registration. It advances the allocation prerequisite
for stage 2. Runtime bootstrap, compiled context, remote coding-child admission
and complete-session token measurements remain separate production handoffs.

The proposed design maintains a compact view of an entire admitted repository
while retaining exact original source in its existing content store. Deliver
only the current task's relevant views, fetch exact source when needed, and
apply accepted edits to the original bytes. This can reduce repeated source
and reasoning in the model's conversation. It needs both source/edit fidelity
and complete-run measurement; an inverse alone establishes neither token
savings nor correct execution of transformed source.

## What moving the whole codebase to and from a compact state means

The goal is less total model work at equal official task success. Keep three
distinct representations of the same admitted source revision:

1. **Exact source and compact views.** Existing source CAS custody, complete
   admitted inventories and deterministic maps preserve bytes. An unchanged
   compact view can already restore original bytes across the retained
   repository population. This host API returns records rather than creating
   an executable checkout; Gitlinks remain explicit frontiers. The live worker
   instead serves its bounded semantic source scope. The repository-wide
   inventory is not yet its ordinary source of views.
2. **IR and checked state.** ASTs, dependency graphs, learned nominations,
   constraints and native proof receipts help find relevant code and decide
   remaining work. These are derived records with different fidelity and
   authority from source bytes. A 384-dimensional embedding, reconstructed
   feature vector or generated formula does not provide an exact source inverse.
3. **Model context.** Send the task requirements, needed code and applicable
   constraints/results. Keep the complete indexes, translation maps, vectors,
   checkpoint weights and proof artifacts in the existing host stores. Use
   qualified retrieval for expansion; an opaque reference alone does not give
   the model its referent's meaning.

The [cold restoration qualification](evidence/reversible-source-views-20261012/README.md)
restores bytes after the original checkout is deleted. Restoration uses the
original blobs retained in CAS; compact text alone cannot recover discarded
comments or formatting. The current transform preserves paths, identifiers,
imports, strings, indentation and physical lines. Its default removes trailing
trivia; additional comment removal is opt-in.

The model-facing [worker reader](worker_source_views.md#select-and-freeze-the-treatment)
has a different population: at most 64 admitted semantic-manifest files and
1 MiB total. Required source remains inline, and the reader does not rewrite
the executable checkout. Its working refresh covers that same population;
new files remain unadmitted and renamed/deleted files refuse. Repository-wide
storage therefore does not yet mean repository-wide worker navigation or
replacement of the original conversation content.

| Contract | Current capability | Next acceptance condition |
| --- | --- | --- |
| Restore an unchanged snapshot | Exact bytes from retained original blobs and deterministic view maps | Already qualified within the admitted inventory; retain explicit coverage frontiers |
| Navigate a complete admitted repository | Complete stored inventory; path/pin discovery within the bounded coding-reader population | Page repository-wide path/symbol discovery into task-specific read scopes without sending every body or removing response bounds |
| Preserve edits made while using compact views | Host-only preimage-bound mapping and one-file application in a clean fenced candidate; expanded-source editing and working refresh remain available | Qualify the actual worker write tool, repeated candidate edits and explicit population deltas for supported create/move/delete operations |
| Execute a physically minified checkout | No such profile | Separate language/runtime/import/reflection/tooling validation and original-task acceptance |
| Reduce total provider tokens | No qualified compact-source session benefit | Show displaced content and repeated reads, then compare complete native usage at equal official success |

There is a concrete additive cost in the current source-view treatment.
[`_worker_payload`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py)
retains required raw source in `raw_sources`. Initial Terminal preparation also
supplies `worker_query`, selecting a bounded capsule projection. In that path,
program bodies outside the required set are generally represented by fetch
identities rather than copied wholesale; nondeferred support source remains
inline. Without that projection, the producer can include additional raw source.
The baseline therefore must be inspected per task: it does not automatically
contain the entire codebase. Minifying a body already absent from the prompt
cannot reduce that initial disclosure.
[`router_implementation_runner.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py)
then appends source-reader guidance when a source-view profile is enabled; that
option does not replace the required inline source. A model can pay for original
source, a compact reread, exact expansion, and their wrappers. This mechanism
needs measurement; it is not evidence that it caused an earlier token regression.
The earlier compact-transport experiment changed metadata representation, not
an editable minified source tree.

The stronger treatment must explicitly replace eligible duplicated inline source
with a smaller readable view or retrievable source reference, after qualifying
the actual worker's ability to resolve it. Preserve the public instruction and
applicable constraints. Required facts remain inline when retrieval is unavailable
or unqualified. Charge discovery and later expansions. This is a new opt-in
disclosure policy, not an automatic consequence of selecting a codec.

For source already deferred, the opportunity is smaller or fewer later tool
results. Ordinary file, search and terminal tools still read the allocated exact
checkout. Selecting the compact helper does not redirect those tools. A compact
read followed by an ordinary read or exact expansion can add text; it can also
help by avoiding other files. Native complete-run usage and matched task outcomes
must decide which effect dominates.

The worker-facing round trip still needs **compact edit to original-source
patch** submission. The host-only mapper and isolated application now implement
the first exact generation/preimage-bound step, without asking an autoencoder or
LLM to recover deleted bytes. Ordinary restoration still refuses modified compact
bytes. The mapper accepts edits wholly within a retained KEEP region, refuses
ambiguous hidden boundaries, and rebuilds the resulting source view to reproduce
the requested compact edit. The owner adapter applies the original patch through
the existing fenced writer and captures a new source generation. Semantic source
and candidate formal cursors refuse prior-generation reuse. Accepted proof/index
publication remains separate. This first write adapter requires a clean candidate;
it does not enable repeated edits to a dirty overlay or a model-facing write tool.
Its isolated-worktree ownership checks are separate from the native task/Portal
allocation guard above. Neither contract supplies an atomic filesystem writer
lease; a failed post-application check can leave a candidate requiring cleanup.
Repository-wide identifier renaming and execution of minified files require
additional language-specific rules; neither is implemented by the current codec.

A physical shadow checkout may be useful for familiar navigation tools, but it
must be derived from the same inventory and mappings. A read-only export is a
different deliverable from an editable virtual filesystem. Arbitrary shell,
editor and build-tool writes cannot be translated merely by saving an old
snapshot. Writes outside the qualified mapped-edit contract still require
expanded originals and their normal edit lifecycle. Running tests against those
originals also avoids assuming that a renamed/minified execution environment is
equivalent.

Whole-repository reversible minification is neither necessary nor sufficient
for token savings. Selective exact reads can avoid irrelevant or repeated text
while the operational source stays ordinary. Conversely, a reversible view can
cost more through guidance, aliases, expansion and repair. The immediate
experiment should isolate context displacement and retrieval behavior before
adding a more aggressive source transform.

## Make repository views one consistent reading workflow

The [shared-policy qualification](evidence/shared-source-view-policy-20261013/README.md)
retains 798 passing selected regressions, including 52 new policy cases, without
changing codec bytes or existing model assets. The subsequent
[discovery and policy transport](evidence/source-view-discovery-transport-20261013/README.md)
retains 1,603 passing selected tests, including 79 new transport cases, and adds an actual
client operation for admitted paths carrying an owner-selected shared policy
through the source socket. Shared-policy benchmark selection is implemented
below; repository-wide admission remains a separate gap.

The legacy complete inventory and coding reader share a codec but select views
differently: inventory handles `.py`/`.pyi` and caller-protected lines, while
the original coding reader handles `.py` without that caller policy. The new
[explicit shared policy](reversible_source_views.md#share-an-explicit-policy-across-owner-apis)
now converges inventory, host reader and edit adapter on one resolver. New
versioned roots/admissions bind configuration and resolver/codec identity to
the admitted path/source generation; the mapper consumes the checked owner
projection. Old defaults, codec manifests, roots and checkpoints retain their
identities. Shared pages now use descriptor version 2 and request version 3;
the owner fixes the policy, and the client selects that pinned protocol. Legacy
requests refuse shared pages. The benchmark can now freeze a shared policy
recipe during preparation, retaining configuration and resolver/codec digests
and joining actual descriptor pins during collection. Omitting the recipe keeps
legacy defaults. See the [benchmark policy workflow](worker_source_views.md#freeze-a-shared-policy-in-a-benchmark-run)
and its [separate qualification evidence](evidence/benchmark-shared-source-policy-20261013/README.md).
A display policy does not admit paths or export a filesystem.

[`WorkerSourceViews.discovery()`](../../ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py)
produces bounded host path/pin pages. The actual
[worker client](../../ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py)
now offers `--discover` alongside `--path`, validates the closed metadata shape
and joins shared pages to descriptor policy/scope pins. Discovery contains no
source bodies, verifies deterministic cache projections, and charges its wire
bytes while counting rows separately from source slices. It remains baseline
only within the existing admitted scope. The new
[historical scope nomination](repository_source_scope.md) joins complete inventory
metadata to a reviewed bounded subset and retains every remaining frontier.
It supplies detached semantic arguments, not current-source or native task
admission. The [signed Terminal context selection](signed_terminal_source_scope.md)
now versions one Python task's bounded semantic/vector subset beside its complete
signed inventory. Its join to the native nomination reader and actual coding
child remains separate. Discovering a path does not authorize a new source read
or expand the current 64-file limit.

Present discovery, symbol/span selection, compact reads, exact expansion and
candidate capsules as one model-facing workflow. Retain the distinct native
source and semantic-reader admission rules behind it. An interface adapter must
not turn a semantic identifier into native task custody. Keep mapping sidecars,
full indexes and proof artifacts host-side; return readable code or selected
facts and their bounded identities. Ordinary exact reads remain available and
must be observed in the comparison rather than assumed to use the compact route.

A complete read-only filesystem export is a possible later consumer of this
same resolver. Test its population, paths, modes, source/view identities and
exact inverse against the retained inventory. Such an export would make the
compact repository inspectable with familiar tools. It would still not be an
editable or executable minified workspace. Worker edit submission, repeated
dirty-overlay edits and create/move/delete support retain the mapped-edit and
native write lifecycle prerequisites described below.

## Observe what enters the coding conversation

The current usage receipts retain total native tokens and aggregate tool counts.
The source broker separately records request/response identities and byte
accounting. They do not yet provide a joined ledger of ordinary source reads,
compact reads, search excerpts and exact expansions. A lower count of helper
calls or fewer stored bytes cannot establish which model context was avoided.

Propose an owner-local read observation joined to run, attempt, session and
native tool call identity: route, request/result digest, actual displayed-text
digest and bytes, source generation, representation and exact path/span when
independently resolvable.
Keep protocol wire bytes distinct from displayed client text. Record initial
inline disclosure as well as subsequent tools. Classify arbitrary terminal
output or missing provider detail as unknown; a command mentioning a path does
not prove which source bytes were read or delivered. Publish bounded hashes,
counts and coverage rather than raw private code or tool output.

Use this ledger to diagnose repeated exposure and compare the same permitted
read/expand/edit trace through exact and compact views. It cannot authenticate
provider delivery by itself or observe a counterfactual avoided read. Those
claims need the actual route and matched complete sessions. A host-side seen set
also does not remove earlier source text from the provider's conversation.
Any later history-compaction or session-checkpoint treatment needs its own
supported handoff, retained task facts, source/proof dependencies and measurement.
Do not assume DuckLake persistence performs that context replacement.

## What the audit found

| Work | Current meaning | Remaining boundary |
| --- | --- | --- |
| [`semantic_context_runtime.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py) | Bounded source snapshots, source hashes, capsules and compact JSON | JSON restoration is not a whole-repository source rewrite |
| [`semantic_metadata_view.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py) | Readable shared metadata with exact transport restoration | Edited transports fail retained identity checks; this transport codec does not map source edits |
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

DuckLake snapshots can identify consistent database history and support queries
at a retained version or timestamp, as documented in the upstream
[snapshot](https://ducklake.select/docs/stable/duckdb/usage/snapshots) and
[time-travel](https://ducklake.select/docs/stable/duckdb/usage/time_travel)
interfaces. The supervisor still has to bind a database snapshot to admitted
source generations and current proof dependencies. Database versioning alone
does not provide source freshness, reasoning state or model-context compression.

## Route IRs to current code and disclose only useful results

The IR families should contribute different information to that shared source
and dependency graph, rather than each appending a full representation of it:

| Family | Intended contribution to the task | Current boundary to preserve |
| --- | --- | --- |
| IntentIR | Convert instruction requirements into goals, invariants, postconditions and completion obligations, grounded to code symbols | The Terminal path accepts advisory inference; separately admitted requirement contracts drive symbolic planning. A decoded instruction is not automatically a verified plan contract |
| CodebaseIR and target-codebase learner | Describe current symbols, calls and dependencies; rank code and candidate obligations against intent | The structural index and target-trained feature reconstruction/ranker are distinct. Training on a repository does not produce a lossless source decoder or complete behavioral theory |
| SecurityIR | Supply source-grounded security properties, findings and supported repair obligations | Learned candidates require applicable native grammar, current source and independent validation; reconstruction accuracy does not discharge an obligation |
| UI/UX IR | Supply interaction, accessibility and UI state-transition constraints when the task has UI behavior | Generic `ui_ir` hooks do not establish that a UI checkpoint ran on the benchmark or that arbitrary code has a UI contract |
| LegalIR | Supply applicable policy/legal constraints and reusable shared training machinery | Forking LegalIR initialization weights is not applying legal constraints or proving software behavior; domain and checkpoint identity stay separate |

In the audited Terminal preparation/runtime paths, no LegalIR or UI/UX checkpoint
inference invocation was found. Their shared datasets entrypoints and generic
supervisor adapters exist, but that does not establish task-specific inference.
The newer [Codebase feature context](../../ipfs_accelerate_py/agent_supervisor/planning/codebase_feature_context.py)
also explicitly declares no formal decoder and uses an 8D profile. Keep it
distinct from legacy target-codebase adaptation and the shared Source384 path;
[initial preparation](../../benchmarks/agent_supervisor/container_coding/terminal_initial_context.py)
makes the latter two profiles mutually exclusive. The current candidate capsule
reader's numerical formulas come from deterministic datasets grammar, not a
checkpoint call made by that reader.

These are proposed routing roles, not a claim that each learned decoder is
ready for arbitrary Terminal Bench tasks. UI/UX IR semantics, a shared
autoencoder family wrapper, a 384D candidate decoder and checkpoint publication
receipts already exist and must be reused. Uniform fifteen-cell routing and
successful task-specific reconstruction still require their own inventory and
evaluation evidence; publication or registration alone does not qualify a
decoder. Do not generate every family for every file.

The intended join key includes admitted repository/source revision, file/symbol,
requirement identity, IR family/schema, checkpoint/decoder profile, projection
family and assumptions. A proof-cache entry additionally binds its exact theorem,
source dependencies, prover/version and native verdict. Changed dependencies
invalidate reuse even if a learned vector remains similar. The planner should
consume supported checked results and explicit residual obligations; the model
should see the small portion needed for its current task. Successful `lake build`
establishes only what the submitted Lean artifact actually checks, not automatic
agreement between learned formulas, user intent and original program behavior.

Target-codebase adaptation belongs before or alongside this join as a local
candidate-selection experiment. Reuse the shared datasets training/inference
modules and a separately versioned target checkpoint; preserve the existing
family weights. Pin the repository revision, training split, encoder,
dimension/window and decoder task. Freeze the adaptation recipe across paired
benchmark arms and use only admitted public task inputs. Report cold-start and
reused-index/model training, indexing and inference time/resources separately
from provider tokens, and charge any LLM-generated labels or repairs to the
native token ledger. Measure held-out requirement/source alignment, semantic
reconstruction, selection and task outcomes rather than treating training
reconstruction as source correctness.

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

LegalIR, SecurityIR, CodebaseIR, IntentIR and UI/UX IR keep separate inventories,
DuckDB/DuckLake namespaces and existing Hugging Face/checkpoint identities.
Within each, retain family, schema/version, decoder task, 8D/384D/768D dimension
and token profile. The existing four-family/twelve-cell inventory is not proof
that all five families have three ready trained profiles. Extend routing and
inventory versions additively for UI/UX IR, preserving previous manifests and
explicitly declaring missing checkpoints or unsupported decoder tasks. Reuse
prior embeddings and checkpoints when their producer, source and profile pins
match. Source-view fidelity is an exact-byte storage/codec responsibility;
a semantic decoder's training
loss cannot certify original comments, formatting or arbitrary source edits.
No new training is needed for the first source-view experiment.

An 8D/384D/768D latent is an internal model representation, not that number of
provider tokens. The coding model needs a supported decoded capsule or tool
result; serializing a vector does not teach it the decoder. GTE window/tokenizer
limits describe the encoder and cannot measure the coding provider's usage.

Retrieve a small useful statement and its evidence reference rather than
automatically printing a complete formal derivation. The selected-reader
observation below illustrates why even a filtered obligation can exceed the
source body it describes.

## Require evidence of displaced context before claiming compression

The [metadata displacement observer](representation_displacement.md) now joins
the existing opt-in metadata/instruction projection and selection to one exact
prepared input. It records UTF-8 transport spans, unchanged wrappers and the
selected replacement or fallback. Actual delivery, source/capsule/IR decisions,
required-fact coverage and avoided future reads remain separate qualification
work; this bounded first step does not implement the general receipt below.
Its default counter remains a byte proxy; declared upstream token counts are
not independently authenticated by the observer. Source-only sessions without
the metadata treatment do not yet receive this observation.

Add a versioned representation-decision receipt at the existing context compiler
and reader boundaries. For each proposed replacement, retain the task/source
generation, IR family/schema/checkpoint if used, exact baseline and candidate
rendered identities, required-fact coverage, provider/model/counter identity,
and the original spans or repeated reads expected to be avoided. Keep restoration
maps, full indexes and translation dictionaries host-side unless requested.
Selecting a new tool while retaining all baseline text is an additive treatment;
record that fact rather than claiming the text was displaced. A declared
displacement remains a proposal until the delivered input confirms which spans
were replaced. Avoided future reads are counterfactual; matched complete runs
are needed to assess repeated-read behavior and net benefit.

Use the existing complete-input counter/fallback pattern. Choose a replacement
only when required coverage and the real coding-route expansion lifecycle are
qualified, and the full rendered candidate is smaller with the same compatible
counter. Counter failure, missing coverage, unsupported source or a larger
candidate retains exact source. A smaller initial prompt does not guarantee a
smaller run: separately measure tool-guidance overhead, discovery, expansions,
generated output and repairs. Do not silently remove required source while the
actual native coding-child handoff remains unfinished.

The run total is the sum of final native input plus output over actual planning,
coding and any additional LLM sessions. Tool text and reasoning already included
by the provider are not added again; cached input is a subset, not an additional
total. Non-provider database, codec, verifier and local-model work has its own
time/compute ledger. Token benefit requires that avoided input and reasoning
exceed added guidance, retrieval, expansion and repair across the complete run.

Prioritize three independent steps: qualify repository-wide discovery feeding
bounded task-specific scopes; extend the displacement receipt and complete
input selection gate; then run the same-channel exact-versus-compact treatment
with official verification. Optional mapped edits and aggressive executable
minification should be tested as separate later treatments; the bounded host
mapper is already qualified. The existing native bootstrap/context/
actual-child and source-CAS integration prerequisites remain necessary for
operational deployment; the allocation guard alone does not activate them.

The immediate implementation order within those steps is:

| Increment | Acceptance |
| --- | --- |
| Shared view policy | Inventory, worker and mapper agree for the same source, `.py`/`.pyi` classification, protected lines and codec; old versions retain their identities |
| Reader discovery | Actual coding client pages admitted paths without bodies; repository metadata feeds explicit bounded admission rather than silently widening access |
| Frozen benchmark policy | Preparation pins one recipe and resolver/codec identity; coding owner and collected session match it; absent recipes retain legacy shapes |
| Read observations | Initial disclosure, ordinary tools and indexed reads have joined call/result identities or explicit unknowns; byte counts stay separate from native tokens |
| Fixed-trace census | Same task, disclosure and read trace through identity/compact modes; charge guidance, display wrappers, expansions and repairs |
| Matched live comparison | Same route, provider/model and task controls; repeated official outcomes and complete all-attempt usage |
| IR/proof reuse ablations | Add one relevant family or checked fact at a time, preserving source/model/dependency identity and cold versus reused setup costs |

Shared owner policy is implemented. The discovery/transport increment adds
client paging within the admitted population and carries an owner-selected
policy. Benchmark preparation now freezes that policy through the coding owner
and checks the collected session against it. Historical inventory nominations
now retain a complete frontier beside a bounded reviewed subset. The native
nomination-to-current-source/task-owner join and signed Python task-profile
selection are implemented as separate boundaries. Their production allocation
and coding-child join, data/multitask scope, and explicit expansion lifecycle
remain open. The existing parallel repository-session/compiler/planning stack
needs explicit schema and CLI reconciliation before integration. General
read observations and subsequent comparisons remain proposed. The earlier
compact-edit bundle retains its 729-test and 11 authored numerical-fixture
results; this phase records its own selected regression run separately, without
a live token measurement. No new training, decoder, embeddings or database owner
are needed to establish the first four increments.

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
5. **Expose the separately versioned mapped-edit profile to the worker.** The
   qualified host mapper accepts explicit span edits with original/view/manifest
   identities, compact-view preimage bytes and replacement bytes.
   Map only an unambiguous retained region. Preserve every untouched original
   byte. Expand source for edits spanning hidden trivia, stale or ambiguous
   anchors, or cross-file symbol/import rewrites until those operations have their own
   reviewed mapping rules. File moves/deletes and symlink changes require their
   own supported native operations; exact-source expansion alone cannot enable
   them. Conflicting concurrent edits refuse.
   Arbitrary replacement of a minified file is not an accepted mapped edit.
   The actual model-facing write tool and repeated dirty-overlay edits remain
   unimplemented; the read socket provides no mutation capability.
6. **Materialize and validate the proposal.** Construct `S'`, preflight all old
   file bytes and permitted paths, and use the existing exact-byte candidate
   materializer in an allocated workspace. The existing general regular-file
   modify/create lane accepts at most eight outputs; the new mapped-edit owner
   adapter is narrower: one existing UTF-8 file in a clean fenced candidate,
   with exact staged bytes and whole-admitted-scope generation checks. Moves,
   deletes and symlink edits remain unsupported by these lanes. Preserve
   admitted modes and file kinds.
   Retain baseline, accepted patch and resulting bytes so reversal can restore
   the baseline while forward replay preserves the edit. Apply ordinary focused
   tests, native proof revalidation and official acceptance to `S'`.
7. **Refresh against the appropriate source state.** The coding worker's
   [source reader](worker_source_views.md#read-working-files-after-edits) now
   provides a session-local candidate refresh boundary for source views.
   The candidate context reader rebuilds advisory capsules and numerical formulas;
   accepted formal indexes still need a separate publication boundary. Bind
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
accepted DuckLake/proof publication, model-facing mapped edits and complete-session
token savings remain separate qualification work. Full formal artifacts can be much
larger than source, so the benchmark must charge summaries, expansions and
refreshes actually requested by the worker.

The [mapped-edit qualification](evidence/compact-source-edit-20261012/README.md)
retains 339 passing regressions and a public-source fixture with 11 authored
numerical cases. It exercises a prescribed local edit through the host adapter,
not autonomous model editing, the official task verifier or a token comparison.

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
| 4: optional mapped edits | Host mapper plus one-file application through the existing clean fenced candidate; worker tool remains open | Untouched bytes preserved; hidden/crossing/stale/ambiguous edits refuse; reprojection and actual staged bytes checked; candidate generation refreshed |
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

The instruction-deduplication [live pair](evidence/terminal-instruction-dedup-pair-20261012/README.md)
has now completed: 311,776 baseline versus 309,813 treatment native tokens,
with both official rewards equal to one. Its 0.63% difference, mostly cached
input, comes from one independently planned pair and establishes no repeatable
or causal saving. Lexical versus ranked capsules and exact versus compact
source remain distinct experiments. The
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

Before a live pair, replay one fixed permitted read/expand/edit/refresh trace
through `identity@1` and the compact profile. Count the complete rendered initial
prompt and actual default CLI text displayed to the model, including failures,
repeated reads and guidance. Socket response JSON and model-displayed JSON are
different: the selected-obligation probe measured 6,319 versus 5,520 bytes.
Use a qualified provider-compatible counter when available; otherwise label the
result a byte census. The metadata selector's optional counter does not currently
provide a qualified CLI tokenizer factory or gate all source-tool responses.

Separate the first live experiments into these contrasts, reusing the existing
comparison machinery and actual `llm_router` route:

| Contrast | Controls held fixed | Question |
| --- | --- | --- |
| Existing exact-source reader versus compact reader | Same initial source, plan, IR/capsule selection and read policy | Does the currently additive compact option help at all? |
| Inline source versus admitted on-demand source | Identity source codec, same relevant information available, task and planning treatment | Does removing duplicate initial disclosure repay discovery/expansion costs? |
| On-demand exact source versus on-demand compact views | Same retrieval interface, selected scope, formal/capsule treatment and exact-source edit policy | Does source minification itself help? |
| Expand-before-edit versus supported mapped compact edits | Same compact view and admitted edit scope | Does avoiding expansion reduce complete task tokens without harming correctness? |

These are proposed contrasts, not completed runs. A fixed trace diagnoses the
representation overhead; native-model runs measure changed tool choices and
reasoning. Use matched native Codex and no-index supervisor baselines for the
broader architecture comparison, then ablate indexed features within the same
supervisor/provider/model configuration. A full-system advantage cannot be
attributed to minification when planning, model or other IR features also change.

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

## Native nomination handoff, 2026-10-13

The [live task-owner handoff](current_source_task_admission.md#bind-a-pinned-historical-nomination)
now replays the historical nomination with independent CAS/inventory/request/limit
and scope pins, then derives an exact selected path binding through the original
native source reader. Task claims, owner grants and current originals remain
independent checks. Publication failures retain bind-once state; a detached
observation does not become authority.

This closes the host nomination-to-reader join. The opt-in signed Terminal
context selection described below now coordinates preparation, index/partition,
loaders and rebind for a single Python task. Data/multitask scopes and actual
worker/allocation integration remain open. Preserve the complete signed task
inventory. Keep the signed selection declarative: putting its postcommit
inventory CID into the indexed task profile creates a content cycle. Join those
identities in a later independently checked receipt instead.

The existing parallel repository sessions/compiler/planning stack remains the
implementation to reconcile. Its policy helper/schema collision is unchanged;
this increment creates no competing sessions, decoder checkpoints or databases.
Matched full-session token/task measurements remain necessary after native
worker adoption. The [qualification bundle](evidence/native-nominated-source-scope-20261013/README.md)
records the tested boundary and its limits.

## Reuse the fresh live baseline, 2026-10-13

The parallel [live comparison](https://github.com/endomorphosis/ipfs_accelerate_py/blob/4ea2271709bf697f175a198d3df02ac1e1a38205/docs/agent_supervisor/live_indexed_baseline.md)
published three fresh successful `largest-eigenval` trials. Native Codex reported
113,883 tokens, no-index supervision 358,492 and indexed supervision 333,973.
Indexed supervision used 6.84% fewer reported native tokens than no-index, while
still using 2.93 times the native harness's tokens. This single sequential pilot
is a useful baseline, not a demonstrated overall token advantage over native
Codex. Auxiliary refresh accounting is incomplete; these are reported native
session counters rather than exhaustive pipeline cost or billing.

That indexed arm reused four symbols and four worker capsules, hydrated three
DuckLake catalogs and delivered eight numerical planning hints. It made zero
optional numerical retrieval calls, all 105 obligations remained unresolved,
Source384's four candidates were unsupported, and IntentIR had no selected
checkpoint. Source views were disabled. These observations support testing
actual selective-tool adoption, decoder coverage and successor refresh; they do
not measure this nomination handoff or reversible minification.

The same publication repairs queued-completion output observation after the
frozen live runs. It does not recover those runs' missing output hashes. Retain
its original benchmark runtime pins separately from the current qualification
and later merged publication. The next live comparison should preserve all
three harness arms, repeat matched tasks, count planning/coding/refresh/retrieval
and failed attempts, verify final output custody, and record whether nominated
source scopes and optional retrieval tools were actually consumed. Introduce one
representation change at a time before attributing any token difference to it.

Nominated scope discovery also needs an explicit expansion policy: this native
reader remains bind-once for one admitted attempt, and unselected frontier entries
cannot be fetched through it. Qualify a new scope/task-generation admission and
retire stale capabilities before widening a read set. Charge any new planning,
admission and reconstruction work to the complete-session experiment; silently
mutating the existing path scope would invalidate its original custody contract.

## Signed Terminal context selection, 2026-10-13

The opt-in [profile and consumer path](signed_terminal_source_scope.md) now signs
one task's semantic/program selection inside the existing profile support file.
It preserves complete signed inputs, original task effects and unselected source
identities. Initial build, cold reload, direct admitted context, admitted cache
reuse and warm native-owner capture consume the same selection. Bounded planning
coverage reports selected/full/frontier identities without full source metadata
or frontier rows in that projection. Complete signed task path declarations
remain in planning input. Compiled worker input and historical audit
retain the selected hashes and roles; they are separate from a launched worker's
source capability or proof of program semantics.

A host regression uses 70 ordinary Python originals: all 73 inputs remain signed
including three generated supports, while five files enter semantic context and
two enter the vector index. This demonstrates selection and exact cache reuse,
not a measured reduction in a live model's complete-session tokens. Current
replay checks omitted originals too. Selection stays bounded at 64 total files
and 1,000,000 original per-path bytes including supports; complete acquisition,
signature replay and source hashing are still host work. The
[qualification evidence](evidence/signed-terminal-source-scope-20261013/README.md)
records execution and compatibility limits.

The next integration steps are:

1. Reconcile the existing repository compiler/planning stack's policy schemas and
   CLI, then let its reviewed metadata proposal feed this declarative profile.
   Independently bind the actual historical inventory and nomination after the
   profile-containing commit; a proposal alone has no native source authority.
2. Join the authenticated allocation, original owner claim, compiled prompt and
   actual coding child's birth in the production launcher. Preserve the native
   broker's independent 32-path bound and bind-once scope; use new admission and
   stale-capability retirement for expansion or edits.
3. Qualify selection for data and multitask profiles and for each decoder task's
   own population contract. The current scoped mode refuses local training,
   SecurityIR, Source384 and optional numerical/header context combinations and
   explicitly abstains from complete Doctor repair. Reuse existing family,
   schema/version/task and 8D/384D/768D checkpoints and embedding inventories;
   do not reinterpret a selected-source regression as reconstruction training.
4. Observe actual initial disclosure, source lookups, expansions and refreshes;
   rebuild and pin a fresh deployment archive containing the new scope module,
   then run the fixed-trace census and repeated matched live comparisons above.
   Charge planning, coding, admission, reconstruction, failed attempts and
   auxiliary refreshes. Compare against the fresh native/no-index/indexed
   baseline before attributing a difference to capsules or minification.

No checkpoint, embedding population, model-manager registration, DuckDB/DuckLake
inventory or foreign worktree is rewritten by this context-selection increment.
It supplies a bounded context path for later controlled experiments.
