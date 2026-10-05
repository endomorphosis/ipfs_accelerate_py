# Terminal-Bench symbolic capability assessment

`benchmarks/agent_supervisor/container_coding/terminal_symbolic_capabilities.py`
produces a bounded, body-free `terminal-symbolic-capabilities@1` observation.
Call `assess_terminal_symbolic_capabilities` **after**
`verify_local_benchmark_admission`, passing the original signed envelope at
`admission["manifest"]`, the exact member task specification from the verified
payload, task CID, actual native Doctor result, and selected `lean`/`z3` paths.
The native manifest CID hashes the complete envelope, including its binding;
the verified payload alone has a different identity. The helper does not verify
signatures or proof receipts again;
model suggestions are not valid inputs to this owner-side boundary.

The report separates complete source-hash binding from semantic coverage;
independently bound program inputs from harness support; structural validations
from reviewed local behavioral contracts; declared output effects from the
generic operator's supported effects; executable presence from native proof
observations; and declared task count from actual planning or parallel execution.
Unknown diagnostic text is counted without copying it. A report grants no proof,
execution, publication, or completion authority and makes no solvability claim.
The prover probe uses filesystem metadata only. It never runs or reads a binary.
A validation named as a structural check remains explicitly unverified by this
assessment: its name cannot certify arbitrary command arguments. A requested
header profile is reported separately from the actual selected Doctor workflow.

The optional `doctor-terminal-source-partition@1` must account for every signed
input exactly once. Its instruction, task profile, and generated structural
smoke retain their hashes and explicit roles. All original task input files,
including unsupported data, remain in the program-input population. An empty
population stays empty; instruction text does not become a software symbol.
No partition in a legacy result means unknown partition coverage.

## Harness integration

The full arm attaches this observation as
`report["doctor_dispatch"]["symbolic_capabilities"]` for both the generic Doctor and the
explicit header-contract route. The no-index arm continues to disable Doctor.
The assessment binds the same signed-envelope CID used by native admission;
the separate payload CID is also retained. Requested header profiles and the
actually selected workflow are reported separately. A validation named
`public-structural-smoke` is only a name here, not evidence that it ran or proved
behavior.

Previously, the generic Doctor required every signed input to be Python AST
source. The harness always adds an instruction Markdown file and a JSON task
profile, so even a supported Python repair fell into the unsupported-inventory
route. `runtime/doctor_source_partition.py` now independently replays the signed
admission and exact canonical profile, instruction binding, task specification,
and generated smoke. The shared `runtime/terminal_source_partition.py` checker
recomputes the selected planning evidence used by the real preparation entrypoint,
including its acceptance-evidence CID. Only those three generated files receive support roles.
Every actual input remains in the program population, including JSON, Markdown,
and other unsupported inputs; their unresolved coverage still blocks mutation.
All signed file hashes remain in the transaction ledger. Replays before proof
and candidate mutation reject stale support or program inputs.

The canonical profile producer lives in
`ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile`; the benchmark
module reexports it for compatibility. Generated validation now runs with
`python3 -I -B` so candidate modules and `PYTHONPATH` cannot replace the smoke's
standard-library imports. Re-prepare trials after this change: old signed task
specifications with the former argv are not silently rewritten or grandfathered
into the support partition. Generated smoke is read with an exact producer-derived
byte bound, including profiles whose escaped paths expand in its Python literal.
The profile producer now checks native cardinality limits before preparation
mutates the workspace: at most 32 created outputs, at most 256 sources after
creation (including the three support files), and the existing 128-source limit
for modify-only tasks. Up to 64 total outputs remain allowed within those limits.
Large declarations can still exceed the separate pending-receipt summary bound;
an exact early sizing check remains outstanding.

Two generic closed operators are available on eligible profiles: a local Python
keyword rename and an imported alias call repair. For example, an existing
`from helpers import transform as normalize` can justify changing the unresolved
`transform(value)` call to `normalize(value)`. The latter route requires a signed,
flat local donor, an unambiguous alias, inert function bodies and arguments,
compatible signatures, no shadowing, no unresolved calls or call cycles, and no
import-time effects. The grammar admits at most 16 bindings per module and
64-character identifiers. Separate native theorem and prover-argument limits
return an explicit residual on overflow, before proof state is created.
It does not insert dependencies or guess missing imports.
The source-derived finite binding map is projected to real Lean/Z3 checks for
alias resolution, the original name's absence, and argument preservation. Local
module resolution is an explicit assumption, not a theorem about Python's loader.
The report identifies the actually selected operator using a closed map.
The theorem remains about the reviewed local transformation; AST placement,
signature binding, source hashes, dependency impact and transaction validation
are separate gates. Autoencoder formulas and index results remain nominations,
not independent behavioral specifications or proofs of the whole task.

## Native execution diagnostics

`terminal-native-progress@1` records bounded task-revision transitions, hashed
attempt/claim identifiers and closed rejection codes when available, heartbeat
progress, START/STOP status, and router timeout observations. It retains at most
32 transitions plus the latest sample. Missing settlement details remain
unknown. An unattributed `TimeoutError` is not reported as a proven watchdog
expiry or the cause of the earlier eigenvalue failure.

The benchmark explicitly signs its implementation watchdog instead of inheriting
the daemon's 1,800-second default. The outer work deadline and reserved cleanup
budget still apply. Other runtime callers retain their existing default unless
they supply `implementation_timeout_seconds`.

Two advancing heartbeats from the same generation, with an unchanged task
revision and no active or claimed work, can stop a disposable run after native
attempt exhaustion or quarantine. Ordinary idle or temporarily unavailable
completion evidence keeps waiting. STOP and worker cleanup still run; this
observation cannot release claims, settle callbacks, retry providers, or mark a
task complete.

## Prioritized gaps and reusable owners

The earlier local qualification on 2026-10-04 records **316 distinct tests passing in their
latest focused runs**, with no remaining failures or skips in that selection.
The [qualification evidence](evidence/terminal-symbolic-harness-20261004/qualification.json)
retains source hashes, exact commands, per-run counts and log/XML digests,
including earlier fixture failures corrected by subsequent targeted runs.
It covers actual Lean/Z3 candidate repairs, source/contract drift and import
shadowing controls, native runtime construction, lifecycle doubles, and frozen
checkpoint inference with initial-index replay. The AST-seal plugin remained
enabled. This is local qualification, not a new Terminal-Bench score or full
live supervisor execution.

The [follow-up qualification](evidence/terminal-symbolic-followup-20261004/qualification.json)
records **352 distinct passing tests, zero skips** across focused runs (374
executions including repeated cases). It includes five passing integration
checks on code commit `e6cd2df9519f92d2e1c9eae6d01964de31d4079d` after incorporating
concurrent upstream changes. Commands, source pins, log/XML digests, and the
earlier corrected failures are retained. The AST-seal plugin remained enabled.

The [source-currentness qualification](evidence/supervisor-source-currentness-20261004/qualification.json)
records **76 passing tests, zero failures or skips** on the subsequent cache and
security-advisor fixes. Automatic source-change invalidation now includes inputs
declared only by proof receipts, invalidates their owning obligations before
propagating to dependent proofs, and handles whole-file dependencies even when
the file has no parsed scopes. Twelve regressions cover changes, deletions and
renames across file, symbol and scope-ID dependencies, persisted-index replay,
bounded reason chains and preservation of unrelated evidence.

The security advisor captures validated source rows before checkpoint loading
or inference callbacks. Six callback-mutation regressions bind inference,
source hashes and optional Lake inputs to the same original bytes. This run uses
runtime and Lake doubles with actual bounded native source qualification; it
does not execute checkpoint inference or a Lake subprocess. The existing
advisory authority limits and family-specific checkpoint selection remain in
place. These fixes do not close the broader planning gaps listed below.

The [profile and context recovery evidence](evidence/supervisor-profile-recovery-20261004/qualification.json)
records **254 distinct passing tests, zero failures or skips** across the
subsequent integration repair, including actual cached-checkpoint inference and
four final preparation-to-Doctor checks, two with real Lean/Z3 repairs. AST sealing
remained enabled; exact commands, source/asset pins and run boundaries are
retained. The audited `main` retained generic
profile tests while its preparation entry point rejected `task_profile`, and
native admission passed applicability arguments missing from the symbolic
planner and requirement adapter. The repair restores those retained interfaces,
exact signed profile replay, Source384 selection and initial-context transport,
and the closing checks on warm rebind. It preserves the repository-preview and
source-unit advisory routes added by other work.

The recovery reuses implementation from `e6cd2df95` and the existing Security384
checkpoint and cached GTE-small snapshot. A new regression rejects changes to
original source bindings during declaration before a prepared task can be
published. Empty repositories can prepare an explicit signed create task, but
source indexing still refuses an absent program population; zero-symbol source
files also remain outside the current vector qualification contract. The subsequent
`bc4e6fdf2` repair restores resource-profile wiring, IntentAction384 selection and
header-v3 orchestration. Its separate contract qualification below retains the
source revisions and limits; those integrations are no longer recovery gaps.

The repair integration uses an explicitly authored IntentIR contract and lexical
vector-index ablation. It traverses actual prompt preparation, initial indexing,
symbolic planning, task-bound context, Doctor proof and candidate commit, worker
materialization and structural smoke for both operators. It makes no trained
intent-interpretation, task-completion or full daemon START claim.

Separately, genuine CPU GTE/Source384 inference through initial-context preparation
loaded the pinned checkpoint once, inferred one program file, and retained all
four signed inputs including three support files. Replay did not load the model
again. Its decoded candidate remained `fail_open_source_contract_unsupported`;
checkpoint consumption does not establish source correctness. Tests used datasets
revision `d5aaf256618c53b009695e8d5618436fba1c0419`. All 187 observed Source384
pin-import paths have identical Git blobs at the newer preserved publication
revision `aa586a9b91b583a2ac5d867bd107ba0c5a4762d5`; this comparison is not a claim
that the full tests ran on that newer revision. No paid model calls, new Docker
START or new Terminal-Bench score were produced by this qualification.

| Priority | Missing capability | Reuse before adding another subsystem |
| --- | --- | --- |
| P0 | Broad operator dispatch beyond the two closed Python repairs | `runtime/doctor_task_workflow.py`, `planning/repair_operator_registry.py`, `planning/deterministic_doctor_transforms.py`, `planning/program_repair_synthesis.py`. Each new route needs independent preconditions, exact source lowering, native proof reconstruction, impact and publication gates. Registry presence alone does not make it executable. |
| P0 | Symbolic decomposition in generic benchmark profiles | The shared requirement adapter, obligation compiler, symbolic planner, critic and formal-plan validator already handle reviewed multiple operations. Generic preparation and several execution wrappers still require one task. Add the bounded reviewed profile, then per-task context and explicit native execution scope described below. |
| P1 | Independent behavioral contracts for newly created code | `analysis/required_behavior_synthesis.py`, `proof/missing_input_synthesis.py`, `analysis/tactician_guided_behavior_synthesis.py`. Preserve source precedence and unresolved clauses; inferred formulas are nominations until independently grounded. |
| P1 | Authenticated live-provider qualification of optional independent review | Operator policy transport and the native reviewed-effect, merge, and restart path are connected below. Qualify authenticated Grok implementation and independent Codex review with real provider usage receipts; the local qualification uses authored responses and adds no live provider or benchmark score claim. |
| P1 | Source/data and language coverage | `analysis/program_ast_adapters.py`, exhaustive corpus inventory, scoped Doctor diagnostics. Preserve all hashes and unresolved frontiers. Parsing, tokenization and vector hits do not establish executable semantics. |
| P2 | Build/install/training and generated-artifact stages | Native task admission, validation and capability machinery. Bind commands, dependencies, environment identities, budgets and produced artifacts. Static file publication does not describe these workflows. |
| P2 | Git state, services/VMs and alternate input roots | Existing source/task admission and process supervision. Add explicit effects and immutable capture profiles rather than broadening filesystem scope implicitly. |

The prior suite readiness assessment grouped 35 tasks under source/data/language
review, 15 under build/install/training, 10 under services/VMs, and 8 under empty
source. These are engineering categories, not predicted passes or newly
qualified tasks. See
[the complete readiness evidence](evidence/terminal-suite-pilot-20261004/broader-readiness.json).

Validation must retain stale-source, ambiguous-contract, unsupported-language,
empty-source, and absent-output controls. A real prover run should reject
unreviewed axioms and `sorry`; successful Lean elaboration or `lake build` still
requires independent source-to-contract binding before supporting a software
correctness claim. This helper emits no theorem template and executes no prover.

## Source384 program scope

The native scanner also captures the generated Python smoke check. Scoped
Source384 inference now consumes only independently verified Python program
inputs. The complete scanner inventory and all support hashes remain in its
receipt. A retained signed `source-selection.json` binds the support roles;
`source384-terminal-program-scope@1` binds that selection to the current source
population. Neither grants proof, execution, formalization or completion authority.

Replay rejects changed support, selection provenance, scope or source bytes.
Successor inference reuses the immutable support declaration while binding the
new program bytes; it does not pretend the original program hashes are still
current. Newly created outputs remain outside that old scope pending independent
successor admission. Empty or non-Python program populations explicitly abstain
before checkpoint consumption. General empty retrieval is supported by the
separate native contract below. Selecting Source384 or legacy decoder training
on that retrieval lane explicitly abstains before checkpoint consumption;
independent decoder abstention transport remains future work.

## Empty-source and zero-symbol context (2026-10-05)

The native empty-retrieval lane distinguishes no declared program inputs from
successfully parsed inputs with zero qualified code symbols. It authenticates
the signed public task partition through initial capture, saved replay,
planning, admitted preparation, warm owner rebind, worker context, historical
input audit and publication refresh. Existing lexical and learned vector
producers retain their positive-population contracts.

The implementation preserves these bindings:

1. Reuse `runtime/terminal_source_partition.py` to independently reproduce the
   signed instruction, profile and generated smoke. Keep their exact bytes in
   the complete manifest. Its explicit `program_paths` alone drive semantic
   scans and code retrieval; the three support files retain their support roles.
2. `supervisor-empty-code-retrieval@1` records explicit `no_program_inputs` or
   `zero_qualified_symbols` disposition. The latter requires a complete,
   successful native scan of every declared program input. Unsupported language,
   parser failure, truncation or missing coverage remains unavailable.
3. Bind exact source hashes, partition, task and query to content-addressed
   `source_population_cid`, `query_id` and `result_id`. Set `index_id` to null,
   retain no vector configuration, vector rows or retrieval hits, and record
   zero embedding calls. Native vector configuration requires a positive
   dimension; a dummy lexical dimension would invent an embedding policy.
   Proof, execution, completion, omission and equivalence authority remain false.
4. Capture an actual cold native semantic root using only the program subset.
   Retain required raw support text and freshness of the entire signed manifest.
   No program inputs produces zero semantic symbols and capsules. A comment-only
   Python file retains its genuine native module fact/capsule, while qualified
   retrieval symbols remain zero. In the explicit subset route, support files
   never become task-program facts.
   Explicit semantic subsets use schema `@2`; the default `@1` route retains
   its existing behavior. Verified positive-population generic `@1` contexts
   remain reusable with their original vector and semantic bytes; only empty
   retrieval requires the explicit `@2` partition. Doctor independently reproduces
   the signed partition before accepting an explicit subset for repair eligibility.
5. Authenticate the same artifact union throughout creation, saved replay,
   planner admission, admitted reuse, warm rebind, worker context, final audit
   and successor publication. Keep the existing `code-retrieval-context`
   evidence kind, so the three required context kinds remain available.
6. Refresh only the authenticated original input scope. Newly created outputs
   remain explicitly outside that scope until a separate authenticated scope
   transition admits them. An empty observation cannot authorize output effects,
   declare task completion or justify skipping verification.
   A newly introduced qualified symbol makes retrieval unavailable until an
   independently admitted vector-policy transition; refresh neither manufactures
   vectors nor silently changes embedding models. Native STOP-triggered refresh
   and cache reload preserve this behavior and record zero retrieval model use.

| Owner | Implemented responsibility |
| --- | --- |
| `terminal_task_profile.py`, `terminal_initial_context.py`, `terminal_indexed_preparation.py` | Admit an explicit empty program population after signed partition replay; preserve complete support/source inventories in both initial and direct admitted preparation. |
| `runtime/semantic_context_runtime.py` | Capture, replay and refresh an explicit program subset while retaining the full manifest and required raw support. |
| `runtime/code_retrieval_context.py`, `runtime/empty_code_retrieval.py` | Validate the closed empty/vector artifact union, exact query/task/partition identity, complete native scan and current source bytes. Reject an empty marker over code containing qualified symbols; pure historical validation authenticates original population IDs without claiming fresh source access. |
| `terminal_context_rebind.py`, `terminal_context_audit.py` | Bind the population CID when vector identity is null; authenticate the empty schema before accepting null identity. |
| `terminal_container_supervisor.py`, `entrypoints/admitted_benchmark_runtime.py` | Select an explicit empty publication policy rather than a lexical or learned vector policy. |
| `runtime/published_task_context.py` | Refresh empty observations without assuming a native vector snapshot/query; retain the original scope and independently bind successor source changes. |

Both signed create-only tasks with zero program inputs and comment-only Python
inputs traverse native preparation, authored planning fixtures, admitted world
capture, warm rebind, worker prompt reconstruction and historical audits. Tests
also cover real native publication/STOP/refresh, empty-to-symbol transitions,
support drift, rehashed scope/identity/count/type tampering and false authority.
No model output is relabeled as proof or benchmark correctness. This change
creates no decoder checkpoint and preserves separate IR families, schemas,
tasks and 8D/384D/768D paths.

The [empty-context qualification](evidence/supervisor-empty-context-20261005/qualification.json)
records 464 distinct passing cases with zero skips, fresh AST seals and unchanged
production/test source pins across the combined run and native Lean correction.
The combined run passed 462 cases and failed two after selecting Lake for a
fixture that invokes Lean directly; all nine cases in that file pass with the
installed native Lean 4.33.1 executable. Both logs and XML remain retained.
Eleven existing cached MiniLM learned-retrieval regressions pass offline; they
qualify the ordinary vector lane and do not establish GTE decoder quality.
Retained MiniLM, GTE-small and Intent384 weights have identical before/after
hashes. The qualification produces no training, downloads or benchmark score.


## Runtime contract repairs and current model defaults (2026-10-04)

The [contract qualification](evidence/terminal-supervisor-contracts-20261004/qualification.json)
records **276 distinct passing tests, zero skips**, on accelerate source commit
`bc4e6fdf20462fc31e6b1ba19207b2dc3844cc20` and datasets
`987cf856b2b902aa68c4587bb492b19b932b5d30`. The final selection contains
179 model-route/benchmark controls, 87 Source384/currentness checks and ten native
cache/lifecycle checks. Production source hashes stayed unchanged during these
runs. The native replay starts and stops the real supervisor, executes an authored
code repair with owner validation, and leaves no owned child processes behind.

Earlier focused qualifications retain their own source hashes and revisions:
534 datasets compatibility cases, 123 native lifecycle/deadline cases, 155
preflight/persistence cases, and actual Lean/Z3 Doctor repair integrations. Counts
across stages overlap and must not be added as distinct final-revision coverage.
The repairs restore bounded checkpoint reads, verified embedding lifetime,
resource budgets, typed completion/owner contracts, shared preflight storage,
staged-new-file patch rendering, and custody after failed native START.

Fresh general supervisor routes select `grok-4.7` with guarded `gpt-6.1-sol`
fallback. Fresh signed authorization, runner commands, probes and recovery
protection agree on that pair. Historical signatures retain their exact old
model tuple; mixed signed profile/route pairs are rejected. Recovery continues
to protect retained old-model attempts. Explicit container comparisons pin
`codex_cli` / `gpt-6.1-sol` in both supervisor and native Codex arms; they do not
exercise Grok dispatch or fallback.

This qualification made no paid provider calls and supplies no new Terminal-Bench
reward or token score. The installed Grok CLI listed 4.7 but reported unauthenticated;
live model dispatch remains unverified. Real CPU Source384 checkpoint inference
passes, while unsupported decoded source contracts still abstain. Seven existing
[optional independent-review integration tests](evidence/terminal-supervisor-contracts-20261004/models-independent-review-gap.json)
remain unresolved: the daemon lacks policy CLI/initialization and the joined
reviewed-effect path. Retired SQL-inbox and HMAC-profile fixtures also remain
outside this qualification. GitHub hosted datasets checks could not start because
the account was locked for billing; local test results are independent of CI.

## Intent replay input currentness (2026-10-05)

The [replay qualification](evidence/supervisor-intent-replay-20261005/qualification.json)
records **242 distinct passing tests, zero errors or skips**, in the final
complete selection on unchanged production/test and asset pins. There are
44 new cases. Thirteen new planner regressions all fail against original
runtime commit `26812b9267e765ed5792552befea2710bc04b3c6`; the original source-drift
case reaches its authored provider, whereas the corrected path stops before
provider dispatch. The separate 20-case effect-adapter baseline records 18
failures and two passing controls. Baseline review corrected one test fixture
that raised an unrelated `KeyError` before mutating an actual declared input
domain; the corrected controls require the intended `ValueError`. These
controls use no paid model calls.

Intent384 preparation and saved validation now retain original canonical bytes
and pass detached reports to numerical verification. Mutating a verifier's
input, returning that same mutated object, rehashing it, or replacing saved
advice through a caller reference cannot change accepted planner contract slots.
Saved selection retains its original configuration and advice bytes through
both load and summary replay. Optional failures deliver the exact original
planner request without the candidate. Signed source or preparation drift
prevents a provider call, even without initial indexed context or a requirement
contract.

The Intent/code effect adapter validates and captures the complete source and
prediction join before external Intent replay. Original Intent advice, supplied
Security advice, source rows and selected configuration remain bound throughout
callbacks and publication. Generated association verification also compares
detached inputs against a fixed original identity. The Security checkpoint and
advice digest retain supplied-inference provenance; this consumer claims no
independent Security inference. Unsupported, refuted and no-enabled-input
dispositions preserve their existing meaning and false authority flags.

The qualification consumes the retained IntentAction384 checkpoint with SHA256
`4f3fd17ea2d908fe36c57a444517f3cc0a983cab26f0f67f2e32371e25def3cd`;
its 384,106 bytes remain unchanged. Six live native Lake checks preserve
satisfied, refuted and no-enabled-input outcomes. The first run selected an
elan shim, which the native owner correctly rejected before executing a process.
Those six checks pass after selecting the installed native Lean 4.33.1/Lake
binary; both runs and the exact refusal diagnostic remain in the evidence.
AST sealing stayed enabled with fresh private catalogs and no completion
authority. No decoder training, downloads or new benchmark score were produced.
That qualification preceded the empty-source implementation above and the
independent-review integration described below.

## Operator-selected independent review (2026-10-05)

The [independent-review qualification](evidence/supervisor-independent-review-20261005/qualification.json)
records the final source-pinned local checks, original seven failing integration
cases, development failures and their corrections. Those seven cases now pass.
The final local selection passes 328 tests with unchanged source, datasets and
retained model hashes, fresh AST catalogs and no skipped selected cases.
Fifteen existing cases remain outside this qualification: two security tests
exercise retired handoff APIs, and thirteen runner tests have existing API or
behavior mismatches. Every omitted case also fails on unchanged `fa51b32`
source. Their exact inventory, complete failure logs and baseline comparison
remain in the evidence; this does not establish those behaviors as correct.
The current DuckDB-to-Portal binding is tested with both policy selections.
New native integration controls exercise the current owners. Missing historical
status exports and unrelated process-authority exports are recorded separately,
without restoring those retired interfaces.

An operator explicitly selects `grok-implement-codex-independent-review`.
The supervisor, native daemon entrypoint and DuckDB-to-Portal execution factory
carry the same policy, context budget, timeout and signing-key path. An optional
launch-receipt path and exact CID travel together and are verified against the
existing four-root launch authority. Adoption requires canonical flag names and
rejects abbreviated, missing, duplicated or changed operator fields. Task
metadata cannot select the policy or its trust roots. CLI context defaults
remain 24,576 tokens and 300 seconds; this packet budget is separate from the
GTE autoencoder token windows. Model defaults remain
Grok 4.7 for implementation and GPT-6.1-Sol with medium reasoning for review.

The joined execution path derives a bounded source packet from the exact task
and Git baseline. It rechecks task, source, provider, policy and launch pins
across callbacks. An actual native checkout mutation lease and registered
repository/worktree identity own the writer. Only the exact independently
approved Grok proposal may write; missing review, quota exhaustion, reviewer
replacement and forged provider provenance grant no write on this policy.
The existing generic router's explicitly non-authoritative capacity recovery
behavior is preserved separately.

Replacement files retain no-follow directory descriptors and compare original
bytes and modes before writing. Unified patches first reconstruct intended
postimages in an isolated temporary Git index/object store with native capture,
time and size bounds. The same reconstruction checks finalized effect bindings.
Native patch application uses a child-only `0022` file creation mask to match
the preview's `0644`/`0755` modes without changing the daemon's process mask.
Compensation restores only the writer's own expected bytes and modes; external
edits remain untouched. Losing the actual lease prevents further or compensating
writes and retains the candidate for native recovery. A patch that installs
bytes but reports failure is compensated from its pre-registered intended
postimages.

Native validation, candidate binding and Git commit precede effect finalization
and Ed25519 attestation. Failed validation emits neither a candidate commit nor
an attestation. The durable merge request carries all four review records bound
by the attestation. Another lane or a restarted daemon verifies these carriers
against its operator-pinned policy and shared public key before integration and completion.
Recovery constructors inherit operator settings from their owner. Carriers
cannot install a different policy or signer.

Completion requires the reviewed implementation's ancestry and exact reviewed
blob bytes and Git modes at the selected current merge target. Choosing an old
valid commit cannot hide a changed current output. Unrelated descendants and
the native task-board status commit may preserve that provider-review gate.
Pending completion intents revalidate the current task and signed material
before queue or decision publication. Independent review remains one gate;
native validation, proof, source and publication owners retain their authority.

`production-task-contract@2` binds task identity, requirements, dependencies,
effect scope and every metadata entry except the case/whitespace-normalized
workflow `status`. The native ready-to-completed
projection therefore preserves the reviewed task meaning and canonical revision.
Acceptance, validation, provider metadata and effect scope remain bound. Earlier
raw-status contract CIDs fail closed; they are not silently reinterpreted as
this version. Existing IR families, dimensions, decoder tasks, checkpoint bytes
and embedding assets are unchanged.

The qualification uses authored bounded provider responses and child envelopes
with actual native Git/worktree, validation, lease, queue and Ed25519 owners.
It makes no paid model calls and trains or downloads no models. AST sealing
stays enabled with fresh private catalogs and false completion authority.
These observations add no Terminal-Bench reward, proof of arbitrary generated
behavior or whole-repository authority. Authenticated live provider dispatch,
broader operator/decomposition coverage and independent behavioral contracts
remain separate work.

## Router execution, checkpoint currentness and persistence (2026-10-05)

The benchmark planning and coding workers use `llm_router.generate_text` with
an explicit side-effecting request. Each invocation now requires a fresh native
process and usage receipt; response caching cannot substitute old text for
workspace effects. The Codex provider, cache identity and catalog agree on
`gpt-6.1-sol` and all six supported model environment aliases. The Grok provider
explicitly supplies `grok-4.7` when no model override is configured. The ordinary
Grok supervisor runner retains router-owned command, authorization and guarded
fallback decisions. Ordinary automatic or explicitly pinned Codex daemon
invocations still construct native CLI arguments directly after router selection;
that remaining consolidation boundary is separate from benchmark dispatch.

A failed isolated constructor releases its exact fenced lease and closes its
coordinator only when no native child custody exists. Security source advice
rechecks the selected checkpoint bytes after inference and optional Lake work;
drift discards dependent candidates and state while planning continues. Shared
preflight storage republishes a verified backup manifest during restart. Its
close deadline now covers thread and process lock contention, and an incomplete
close leaves the daemon's store handle available for retry. Candidate patches
use literal Git paths and preserve CRLF bytes across staged, untracked and mixed
changes, with external diff and text-conversion hooks disabled.

The [frozen final qualification](evidence/supervisor-router-lifecycle-20261005/final.json)
passes **412 distinct tests** against datasets commit
`987cf856b2b902aa68c4587bb492b19b932b5d30`: 235 router/checkpoint cases and
177 lifecycle/preflight/replay cases. Source and test hashes remain unchanged
during both runs. Coverage includes real cached Security384/Intent384 CPU
inference, native Lake checks, signed START/observe/STOP, an authored symbolic
repair with refreshed indexes, actual lock contention and Git patch replay.
No test-owned native process remained after the run. Earlier checkpoint,
lifecycle and preflight counts overlap this final coverage and must not be added.

The [baseline router audit](evidence/supervisor-router-lifecycle-20261005/router-baseline.json)
reproduces eight broader failures and a historical authorization collection
error on the unchanged base. The
[preflight evidence](evidence/supervisor-router-lifecycle-20261005/preflight.json)
records 48 unresolved archived plan-bound recovery cases and three separate
prior-seed contract/guidance failures; ordinary shared-store recovery is covered,
but sealed recovery-snapshot integration remains incomplete. Operator-selected
independent-review daemon wiring is qualified separately above. This router
qualification uses no paid providers and produces no new Terminal-Bench reward
or token score.

## Typed restart recovery and shared Codex commands (2026-10-05)

Ordinary Codex daemon invocations now use the command builder owned by
`llm_router`, including its six model environment aliases. Explicit supervisor
and signed fallback overrides retain precedence. The daemon retains native
streaming, cancellation and process custody; sharing argument construction does
not imply generic `generate_text` usage accounting. The benchmark worker retains
its existing side-effecting router dispatch. Defaults remain Grok `grok-4.7`
with the authorized Codex `gpt-6.1-sol` fallback.

Retry seed guidance now survives restart in one bounded artifact bound to the
exact task revision, namespace and attempt. Recovery and prompt replay share its
closed decoder, reject duplicate keys and consume the advice only after prompt
budget acceptance. The artifact stays outside candidate worktrees and carries
no proof, execution or completion authority.

Typed preflight recovery now uses current Portal projections and canonical
artifact-store contracts. Parent snapshots and child accepted-tree validation
join the same exact slice, lane, task and immutable bytes. Reassignment uses the
actual producer's prefix contract. Retained donor state is recognized only from
verified launch, fence and never-attempted history; it supplies custody checks,
without restoring donor execution or merge rights. Ordinary live sibling state
is likewise custody-checked. Unknown files, changed selected evidence, unsafe
permissions and redirected paths remain rejected. Git status collection enforces
byte and record limits while reading, and stable artifact and authority readers
cannot block on a regular-file-to-FIFO substitution.

The [merged final qualification](evidence/supervisor-recovery-routing-20261005/final.json)
records **315 distinct passing tests**, with no failures, errors or skips:
298 joined contract tests, 14 native lifecycle tests and three real Security384
checkpoint/Lake checks. The source and test hashes stayed unchanged against
datasets commit `987cf856b2b902aa68c4587bb492b19b932b5d30`. Every passing case has
an AST seal in a fresh local catalog. Native coverage includes signed
START/observe/STOP, failed-constructor cleanup and an authored symbolic repair
with refreshed indexes. No test-owned process remained in the audited test roots.
The joined suite preserves the upstream independent-review and staged-patch
contracts merged before qualification.

The [focused recovery evidence](evidence/supervisor-recovery-routing-20261005/preflight.json)
records the original failures and 86 passing recovery cases, including an actual
fenced reassignment through snapshot and accepted-tree validation. These cases,
the [donor checks](evidence/supervisor-recovery-routing-20261005/retained-donor.json),
[seed replay checks](evidence/supervisor-recovery-routing-20261005/seed.json) and
[bounded status checks](evidence/supervisor-recovery-routing-20261005/bounded-status.json)
overlap the final selection and must not be added to its count.

At that qualification, prior-attempt MODIFY seeding into existing outputs was
unsupported by preflight v2; the next section records its implementation.
Direct-lane advisory guidance without a trusted exact task projection remains
refused. This focused qualification does not establish that
every archived legacy suite passes. It uses no paid providers and produces no
new Terminal-Bench reward or token score.

## Existing-output retry replay (2026-10-05)

An accepted prior attempt can now replay a MODIFY to an existing declared
scoped output through both dependency preflights and reach provider command
construction. The existing v2 present-target configuration supplies the baseline
contract. A new version-2 prior-seed handoff carries exact before/after SHA256 and
Git blob identities; ADD-only handoffs retain their version-1 shape. Accepted
proposal source, replay gate, task identity, board namespace and baseline receipt
must agree. Missing or foreign source-event namespaces cannot mint this new
MODIFY handoff.

Preflight checks the actual baseline commit/tree and selected file blob,
including baseline-pinned nested gitlinks. It rechecks root and child HEADs and
rereads the candidate after Git verification. Git reads disable caller overrides,
replacement objects and lazy fetch, with output, depth and elapsed-time limits.
CRLF, Unicode and missing final newlines retain their exact bytes. Renames,
binary entries, source/blob mismatches and unchanged-content mode-only entries
cannot supply MODIFY attestations. An unchanged validation target can still use
its baseline contract when the prior attempt changed another declared output.
These digests bind a private daemon handoff; they are not independent proof or
completion authority.

The [frozen final qualification](evidence/supervisor-modify-seed-20261005/final.json)
records **404 distinct passing tests**: 390 joined checks and 14 native lifecycle
checks, with no failures, errors or skips in that selection. One known baseline
failure is explicitly deselected below. Source/test hashes remain unchanged
against datasets commit `987cf856b2b902aa68c4587bb492b19b932b5d30`, and all passing
cases have fresh AST seals with false completion authority. The native checks
include signed START/observe/STOP, constructor cleanup and an authored symbolic
repair with refreshed indexes. No test-owned process remained in the audited
roots.

The [end-to-end evidence](evidence/supervisor-modify-seed-20261005/e2e.json)
reproduces the original present-target digest refusal on pristine commit
`39f88fee698cbb85af167ac49a3ec9fe6a0de799`. The corrected case uses real Git,
accepted proposal gates, prompt compilation and the ephemeral implementation
owner, with a deterministic dependency-environment probe and a stop at provider
command construction. Candidate drift prevents that dispatch. The
[contract evidence](evidence/supervisor-modify-seed-20261005/contract.json)
also records malformed/resealed carriers, baseline and namespace mismatches,
nested checkout drift, and resource bounds. Focused counts overlap the final
selection and must not be added to it.

The existing `test_v2_auto_repairs_undeclared_same_board_pytest_task` expectation
still fails on the pristine base: detection-only preflight does not synthesize
contracts for undeclared tasks. Archived v3/v4/v5 configuration suites import
schema constants absent from that base; their collection errors are retained in
the evidence. They are separate from the new version-2 prior-seed handoff.
Direct-lane guidance without an exact trusted task projection remains
unsupported. This qualification uses no paid models and produces no new
Terminal-Bench reward or token score.

## Supervisor restart, preflight and staged patch recovery (2026-10-05)

Supervisor provider execution continues through the canonical `llm_router`.
Routing regression coverage distinguishes same-provider model retries from
cross-provider fallback and verifies the shared Codex command owner, isolated
worker dispatch and proposal subprocess boundary.

An interrupted START or RESTART resumes only the recorded launched root. The
root may be reparented after its supervisor exits, and its descendants may
finish naturally; PID start time, boot, run, launch profile and fence remain
checked. RESTART verifies that the recorded old process identities are gone
without mistaking the surviving new process for an unfenced old tree. A resumed
owner observes a fresh health window, and root replacement during that window
is rejected. Malformed saga records are rejected before process effects.

The daemon checkpoint API now writes a bounded, versioned JSON envelope with a
record digest, atomic replacement, and file and directory fsync. Reload verifies
the envelope and current attempt, plan, tree and fence bindings. In-memory
legacy callers retain the typed stale-stop interface; persisted checkpoints use
the new envelope rather than Python repr.

Dependency-preflight persistence uses a one-second lock-acquisition deadline
across thread and process locks. Contention publishes the existing failed
infrastructure receipt and defers dispatch. The store remains retryable, and a
subsequent successful attempt persists and rereads the canonical receipt before
returning. Other artifact-store consumers retain their existing behavior unless
they configure a lock deadline.

The shared Git patch collector captures final contents against HEAD, covering
staged and unstaged edits together. New files use nonmutating no-index diffs;
collection preserves the real index, HEAD and working files. Literal filenames,
binary patches and CRLF/CR text replay are covered, including the legal-parser
adapter's full patch output. Failed reads and undecodable text stop collection
before a partial or altered patch is returned.

The [follow-up qualification](evidence/supervisor-lifecycle-followup-20261005/final.json)
records **534 distinct passing tests**, frozen source bindings and independent
review. Eight broader validation failures reproduced on the unchanged baseline
and are excluded from this selection; both runs are retained in the evidence.
No new Ruff diagnostics were introduced. This qualification does not produce
new model weights, live-provider results or benchmark scores.

## Ordinary child ownership and bounded decomposition (2026-10-05)

The [ordinary child qualification](evidence/supervisor-ordinary-identity-20261005/qualification.json)
records the observed adoption and shutdown gaps, their native controls, and the
final source-pinned checks. Ordinary non-plan-bound supervisor loops configure
the existing child identity owner with exact launch arguments and owner scope.
Native adoption also verifies the observed dedicated session and process group.
Appended resource or execution options, altered bootstrap arguments, birth drift
and foreign scope cannot satisfy that owner. Exact live legacy migration remains
available through its existing native argument and birth checks. Cleanup does
not create an identity from process-list matches.

Supervisor validation-worker configuration now carries the same explicit
integer through CLI, configuration and child arguments, bounded to 1–256.
Cleanup uses the recorded owner fence. Ambiguous ownership retains custody and
reports blocked cleanup. Completed-task release requires quiescence before
clearing active state, and signal shutdown preserves unresolved child state.
Native process tests use only their own benign sessions; no model providers,
training or decoder-weight changes are involved.

The merged final selection passes **649 distinct tests**, with no failures,
errors or skips. Every passing case has an AST seal in a fresh catalog;
source, dataset and retained-asset hashes stayed unchanged during the run.
The 55 new adoption, CLI and cleanup controls are included in that total.
The 29 failures from the broader unchanged-base owner diagnostic remain
separate: 19 reference absent APIs and 10 reflect legacy behavior or fixtures.
The selection retains 29 previously passing owner controls; two synthetic
adoption fixtures were updated to model the newly required native session
observations. No retired APIs were restored. Actual kernel PID reuse was not
forced, and the existing process fence does not provide cgroup containment
for workers that escaped before its first census. No executable test-owned
process was observed in the final audited fixture roots.

After publication, another session integrated closed router failure diagnostics
while retaining the ordinary ownership and native test bytes. The
[current-main integration](evidence/supervisor-ordinary-identity-20261005/publication-integration.json)
records **199 passing checks** across two disjoint selections on `e804005fc`,
with fresh AST seals, unchanged run pins and retained assets, and empty native
process audits. These checks overlap the earlier 649-case qualification and
are recorded separately. Independent review found no remaining blocker in the
three incoming router changes. GitHub's documentation job for the original
publication did not start because the account was locked for a billing issue;
the local documentation gates passed.

The [decomposition survey](evidence/supervisor-ordinary-identity-20261005/decomposition-survey.json)
qualified 127 existing checks on the unchanged `39f88fee` baseline: 38 shared
requirement/ordered-planner checks and 89 generic wrapper checks. Those counts
are separate from the child qualification. The shared planner already projects
two independently reviewed tasks through the actual obligation compiler, critic,
formal compiler and native dependency rows. The generic wrapper still signs one
specification, fixes its task budget to one and reconstructs one task during
replay. Ten related context, dispatch, execution or indexed-qualification guards
also retain singleton requirements. Increasing the budget alone would leave
those joins incomplete.

The next planning increment is an explicit `terminal-public-task-profile@2`
with at most 16 reviewed task/operation bindings. Bind the original instruction
hash and requirement-contract CID, unique task identities, disjoint output
ownership with exact union coverage, operation and requirement dependencies,
and each task's validation and acceptance keys. Preserve the current profile
bytes and default path. Preparation must sign all canonical specifications and
reconstruct them exactly during replay before using the existing symbolic
selection, signed admission and transactional materialization. This increment
would qualify administrative scheduling; context and execution refusals remain
until their own joins are qualified.

The following increment binds each task to its source revision, dependency wave
and native world context, including verified source successors after earlier
changes. Each context retains its selected IR family, schema/version, decoder
task, dimension, token/span budget and frozen asset identities. CodebaseIR,
SecurityIR, LegalIR and IntentIR keep separate inventories and checkpoints,
with parallel 8D, 384D and 768D selections. Bind each context to its selected
DuckDB/DuckLake database and Hugging Face artifact revision, preserving the
embeddings and decoder assets already used for that selection. The execution
increment then uses an explicit multi-task native scope, isolated worktrees and
leases, independent
validation/review, merge currentness and durable completion. Its controls must
cover parallel independent tasks, dependency-aware readiness and cold restart
before claiming parallel execution. These profile and execution increments
remain planned; this child-ownership change does not implement them.

## Terminal-Bench finishing gates (2026-10-05)

The [initial readiness assessment](evidence/terminal-bench-finish-readiness-20261005/assessment.json)
records **160 passing tests**, zero failures or skips, and unchanged source
bindings. This includes native empty-program owner/context cases and the shared
router's closed failure diagnostics. The earlier 156-pass/four-failure run is
retained: its interpreter used DuckDB 1.4.3 and a Quack build without the required
serve/query functions. The existing interpreter with DuckDB 1.5.5 passes the
unchanged capability gate and all 160 cases, matching the container dependency
pin. The first diagnostic code alone does not identify the substantive failure;
the complete capability report does.

The latest completed eigenvalue trial at the initial audit failed in planning before native START:
the isolated router exited unsuccessfully and returned no response. Its official
reward remains zero. Two earlier Codex CLI 0.158 health checks failed while a
0.160 host check succeeded; this observation does not prove version causality or
establish health inside the task container. Canonical production calls continue
through llm_router. The reviewed semantic-response diagnostics commit is
integrated here. The subsequently frozen provider and cache upgrade is also
integrated; its completed recovery result is recorded below.

Trial 03 satisfies the selected-task provider and immutable-runtime gates with
a fresh archive, signed preparation, rebuilt CLI/cache policy and provider
accounting inside the original disposable container. Future trials must preserve
these bindings. Earlier prepared archives and trials cannot establish execution
of the new lifecycle/checkpoint, staged-patch or provider changes.

The eight documented default-host validation failures do not explain this
planning failure. The full container arm installs its own signed,
privilege-dropping candidate runner. Signed empty-program retrieval is also
implemented; selecting Source384, security inference or training on that empty
population still requires independently bound decoder abstention. That
conditional gate must not be described as a blanket lack of empty-source support.

The historical ledger leaves 85 of 89 task names without trials under its
methodology. Its two selected-task passes use an older model/profile and are
not a completed suite score. Exact file effects under /app still need broader
public profiles for builds, installations, services, VMs, Git transitions and
other roots. Static category counts are review guidance, not execution verdicts.
The development 5 CPU/16 GiB/840-second work profile is distinct from native task
budgets: 87 of 89 public agent limits exceed 840 seconds. A comparable complete
evaluation needs a pinned per-task resource/time policy and matched fresh arms.

Grok container transport, the inactive Leanstral service and broader symbolic
operator coverage remain improvements; the selected Codex container arm does
not depend on their completion. GitHub's account billing lock prevents hosted
CI from starting but does not prevent local Harbor execution. This readiness
audit made no provider calls or container launches; trial 03 was run by its
existing independent owner.

The [merged qualification](evidence/terminal-bench-finish-readiness-20261005/merged-qualification.json)
records **215 passing tests**, zero failures or skips, after incorporating the
independently qualified ordinary-supervisor custody changes. All 2,870 source
bindings match that tested commit and remained unchanged during qualification.
Earlier selections overlap this run and are not added to its count. A separate owner started
fresh eigenvalue trial 03 with CLI 0.160.0 and a new archive; the retained
[observation](evidence/terminal-bench-finish-readiness-20261005/trial-03-observation.json)
was captured before a completed receipt or official reward was available.
The later [completed recovery evidence](evidence/terminal-supervisor-blockers-20261005/README.md)
records reward 1.0, native completion and clean shutdown, actual Source384
inference, and 291432 observed tokens. The CLI/cache upgrade is now frozen and
qualified in that Docker trial; the remaining suite gates above still apply.

The [completed assessment](evidence/terminal-bench-finish-readiness-20261005/completed-assessment.json)
records the remaining implementation and evaluation priorities. The
[completed-trial review](evidence/terminal-bench-finish-readiness-20261005/completed-trial-review.json)
independently binds the canonical receipt and official result: reward **1.0**,
supervisor completion, zero remaining processes and cleanup exit code zero.
Planning and coding both returned through llm_router with `gpt-6.1-sol`, high
reasoning and CLI `0.160.0`. The published runtime matches the successful
archive, and its later qualification records **560 passed and seven skipped**,
with no failures. These selections overlap and their counts are not summed.
The historical pending observation and failed trial 02 remain retained; the
new pass closes those selected-task readiness gates, while 85 untried task names,
conditional empty-population decoder abstention and matched native-budget
coverage still prevent a complete suite result.

## Regular Codex eigenvalue baseline (2026-10-05)

The [regular Codex baseline](evidence/terminal-codex-eigenvalue-baseline-20261005/README.md)
finished with official reward **0.0**, **161,750 observed tokens** and 288.11
agent seconds. The retained indexed supervisor trial has reward **1.0**,
291,432 tokens and 336.36 agent seconds. The native failed-trial cost is retained
with completed usage, independently corroborated by Harbor's job counters.
Both select the same model, CLI, reasoning and common outer resource limits.
The supervisor has tighter internal work/provider-call limits, and setup, tools
and cache treatment differ. Serialized retry-list order also differs while both
policies disable retries; the strict control mismatch remains visible. This
one-trial observation does not establish an efficiency or reliability advantage
or change the full-suite coverage denominator.
