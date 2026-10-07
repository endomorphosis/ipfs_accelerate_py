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
smoke retain their hashes and explicit roles. Under the original public profile,
all original task input files, including unsupported data, remain in the
program-input population. The explicit public profile `@2` additionally permits
bounded, validated JSON, JSONL, XML and calendar inputs with a `task_data` role.
Their signed hashes remain in the complete ledger; the bound canonical profile
allows independent replay of the code/data classification. An empty program
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
including its acceptance-evidence CID. The original public profile assigns support
roles only to those three generated files. The explicit `@2` profile also
classifies independently validated, declared data inputs; undeclared data and
other unsupported inputs remain in the program population. Data classification
does not establish a behavioral contract for consumers: generic Doctor dispatch
retains `doctor_task_data_contract_unavailable`, and direct repair composition
refuses that population. These unresolved obligations still block symbolic mutation.
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
`transform(value)` call to `normalize(value)`. The latter route requires a signed
local donor, an unambiguous alias, closed function bodies and inert arguments,
compatible signatures, no shadowing, no unresolved calls or call cycles, and no
import-time effects. Module and function docstrings, immutable literal defaults
(including signed numeric literals), positional-only parameters, and arithmetic
or conditional return expressions are supported. Every function still has one
return statement after its optional docstring. Call arguments retain the narrower
name/constant grammar; annotations, decorators, mutable or computed defaults,
variadic parameters, dynamic access, rebinding and nested calls remain unsupported.
The grammar admits at most 16 bindings per module, 32 parameters per function,
and 64-character identifiers. Separate native theorem and prover-argument limits
return an explicit residual on overflow, before proof state is created.
It does not insert dependencies or guess missing imports.
The [package operator](PACKAGE_ALIAS_CONTRACTS.md) additionally resolves absolute
and relative imports through fully captured regular packages with inert
initializers. Namespace packages, reexports and dynamic imports retain residuals.
The source-derived finite binding map is projected to real Lean/Z3 checks for
alias resolution, the original name's absence, and argument preservation. The
finite signature projection additionally checks positional capacity, permitted
keyword names, required-parameter coverage and absence of duplicate bindings.
Both the native Python signature replay and the independently executed Lean/Z3
checks must succeed. Arithmetic return behavior and overloaded Python operators
are unchanged by the identifier repair; their behavior is not proved. Local
module resolution is an explicit assumption, not a theorem about Python's loader.
The report identifies the actually selected operator using a closed map.
The theorem remains about the reviewed local transformation; AST placement,
signature binding, source hashes, dependency impact and transaction validation
are separate gates. Autoencoder formulas and index results remain nominations,
not independent behavioral specifications or proofs of the whole task.

The [expanded authored workflow tests](../../benchmarks/agent_supervisor/container_coding/test_terminal_expanded_symbolic_repair.py)
exercise actual public preparation, reused indexes, an authored IntentIR symbolic
plan, native Lean/Z3 proof, isolated Doctor transaction, candidate handoff and
worker materialization without a model call. Separate authored execution checks
exercise defaults and positional-only calls. Negative controls retain residuals
or refuse publication for missing parameters, failed provers and stale donors;
the [projection controls](../../test/api/test_doctor_alias_binding_coverage.py)
also make real Lean reject an incorrect binding and real Z3 return `sat` for its
falsified counterpart. The focused AST-sealed qualification passed 136 tests;
18 overlapping partition/data checks passed again after the final partition
correction. These are authored integration results, not Terminal-Bench rewards,
learned intent-interpretation accuracy or whole-program correctness results.
See the [public suite profiles](../../benchmarks/agent_supervisor/container_coding/TERMINAL_SUITE_PROFILES.md)
for input-format coverage and the remaining task-specific validation obligations.

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
| P0 | Symbolic decomposition in generic benchmark profiles | Reviewed bounded multi-task preparation now reuses the shared compiler, planner, critic, signed admission and native storage. Per-task source/context bindings and an explicit execution scope remain pending; the new profile refuses native launches until those joins are qualified. |
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

The reviewed planning increment now uses `terminal-public-task-profile@3`
with 2–16 reviewed task/operation bindings. Version 2 supports structured-data
declarations and retains the single-task contract, so the multi-task schema
has its own version. Preparation binds the original instruction hash and exact
requirement-contract CID, unique task identities, disjoint output ownership
with exact union coverage, operation and requirement dependencies, and each
task's validation and acceptance keys. It signs all canonical specifications
and reconstructs them during replay before existing symbolic selection,
signed admission and transactional materialization. The existing version 1
profile, specification and smoke bytes keep their default path. The native
qualification and remaining execution boundary are recorded below.

The ready-root context increment below binds each selected task to the original
signed source revision and its current native world. Dependent tasks still need
verified source successors after earlier changes. Each context retains its selected IR family, schema/version, decoder
task, dimension, token/span budget and frozen asset identities. CodebaseIR,
SecurityIR, LegalIR and IntentIR keep separate inventories and checkpoints,
with parallel 8D, 384D and 768D selections. Bind each context to its selected
DuckDB/DuckLake database and Hugging Face artifact revision, preserving the
embeddings and decoder assets already used for that selection. The execution
increment then uses an explicit multi-task native scope, isolated worktrees and
leases, independent
validation/review, merge currentness and durable completion. Its controls must
cover parallel independent tasks, dependency-aware readiness and cold restart
before claiming parallel execution. The explicit ready-root context route is implemented below. Decoder runtime
admission, dependent source successors and reviewed-profile execution remain
planned; reviewed preparation alone does not implement them.

## Reviewed multi-task preparation and native binding (2026-10-05)

The [multi-task qualification](evidence/supervisor-multitask-profile-20261005/qualification.json)
records the exact profile, preparation, admission and native owner checks.
The final offline run passed 510 checks with zero failures or skips, including
93 new profile/native controls and all 183 selected unchanged-baseline nodes.
All 510 cases have fresh local AST seals without completion authority; 63
manual source/test bindings, the pinned datasets checkout and four retained
embedding/decoder assets stayed unchanged during the run. The focused and
baseline runs overlap this final selection and are not added to its total.

Call `prepare(..., task_profile=profile, intent_requirement_contract=path)`
with an explicitly authored profile and complete
`intent-plan-requirement-contract@2`. Each profile task contains only its task
and operation IDs, owned output paths, dependency task IDs, validation key and
acceptance key. The producer supplies the scope, fixed structural validation
command, policy identity and selected source evidence. Arbitrary commands or
authority fields cannot be injected through the profile.

The join requires atomic mandatory requirements with exact native matchers;
unresolved source units and unsupported compounds refuse. Task dependencies
equal the operation ordering induced by requirement dependencies. Every
declared output, including modified files, has one task owner. Preparation
records a versioned list of all specifications and the exact task-count
budget, then uses the existing symbolic planner, critic, formal compiler,
signed admission and native owner transaction. It makes no provider call.

The initial diagnostic showed a real owner-binding gap: a separately signed
requirement contract with a changed review reference could differ from the
immutable profile CID, enter storage as three tasks and pass native Quack
observation. Native manifest verification now independently reconstructs the
profile, original instruction, requirement CID, complete task specifications,
selected evidence and generated smoke bytes. Reserved profile JSON must be
unambiguous; malformed declarations and downgrading a multi-task declaration
to a singleton schema refuse. Admission rechecks source currentness after
planning replay, and materialization rechecks it before the native transaction
commits.

The native fixture has two independent root tasks and a third task depending
on both. The typed owner exposes only the roots as initially ready and retains
both dependency edges. These observations qualify administrative scheduling,
not task execution or source interpretation quality. The legacy initial/context
preparation routes refuse this profile; the explicit ready-root route below
prepares advisory contexts. `AdmittedBenchmarkRuntime.create` refuses
both observation and implementation launches before allocating runtime state;
unsigned preparation metadata cannot lift that boundary. Existing generic
multi-task admissions keep their separate execution rules.

Each fixed smoke selector checks only its task's owned outputs, bounded to
1 MB per file and 4 MB per selected task. It parses Python output syntax and
never executes candidate code. Semantic correctness and an aggregate
multi-task completion bound still need separate validators. Existing manifest,
scan and serialized-input ceilings remain in force, so structural validity of
a 16-task profile does not guarantee admission of an oversized declaration.
Full decoder runtime admission still must retain separate IR family,
schema/version, decoder task, 8D/384D/768D geometry, verified token/span budget,
DuckDB/DuckLake inventory and pinned embedding/checkpoint identities. The
ready-root route below preserves exact catalog nominations while those runtime
checks remain unqualified.

The remaining context and execution work has three acceptance gates:

- Extend current source/task/requirement/world bindings with dependency waves
  and verified predecessor publications. Join each exact IR family,
  schema/version, decoder task and dimension geometry to verified token/span
  budgets, inventory database identity and immutable Hugging Face/checkpoint
  revisions through actual runtime admission. Preserve retained asset hashes.
- Require a dependent task to use verified source successors after prerequisites
  publish. Missing assets, mismatched family/decoder or stale source/dependency
  receipts must refuse runtime context admission; unavailable selections cannot
  silently choose a different IR or dimension. Root context nomination alone
  cannot lift those gates.
- Retain the ready-root/cache/cold-reopen controls below and qualify dependency
  waves and larger admitted populations. Qualify isolated worktree leases,
  validation/review, merge currentness and durable completion before enabling
  reviewed-profile execution. Autoformalized candidates remain distinct from
  checked proofs in the proof index.

## Ready-root contexts and exact retained IR nominations (2026-10-05)

The [frozen context qualification](evidence/supervisor-task-context-20261005/qualification.json)
passed **445 checks** with zero failures, errors or skips, including all 125
unchanged baseline nodes, 67 new native context controls and 40 exact catalog
controls. All cases have fresh AST seals without completion authority. The 65
manual source/test pins, datasets checkout and thirteen existing retained files
stayed unchanged. Baseline and development runs overlap and are not added.

`terminal_multitask_context.prepare_ready_task_contexts` is the explicit
administrative route for 1–16 selected ready roots in an admitted reviewed
multi-task profile. It verifies the complete signed native task population,
physical output/acceptance/validation rows and retained planning receipt, then
asks the actual owner for readiness. A task with an active blocker cannot enter
because its stored status says ready. Every selected task must be an independent
root; marking prerequisites complete does not admit a dependent task yet.

Nonempty programs require supplied existing native `CodeVectorIndexSnapshot`
and `CodeVectorSearchResult` objects covering every signed program input.
The route replays their numerical and source bindings and reuses their vector
bytes for each root. It builds separate task-alias-bound semantic, retrieval
and live intent-world envelopes without relabelling an earlier artifact. Source
ASTs and capsules are rebuilt by the existing semantic producer. An actually
empty program uses the native empty-population observation. No embedding,
decoder load, training or provider call occurs in this route; numerical replay
alone does not establish query-vector semantic alignment or reconstruction
quality.

`load_ready_task_contexts` cold-reopens the canonical receipt using a trusted
expected byte digest and independently checks the current signed source,
profile/admission, native owner identity, readiness and complete pending
contracts. It then loads the real semantic, retrieval and live world owners,
requiring the exact signed program scope/query, per-root world projection and
original retrieval identities. The task bundle must be the bounded plain
nomination format. Source384 runtime bundle variants cannot enter this route.
Repository runtime paths, regular-file bounds and typed closed metadata are
checked before generic consumer calls. Any native intent event conservatively
stales the wave; these are cooperative currentness checks, not an atomic
filesystem/database transaction.

Optional `ir_catalog_path` and `ir_selections` bind each selected task CID to
explicit persisted ModelManager catalog nominations. The resolver preserves
all ten native selectors: family, dimension, dimension role, schema version,
decoder task, profile ID, format ID, asset role, record ID and checkpoint ID.
Different heads can coexist within one family/dimension; competing assets for
one complete namespace refuse. All selected nominations must belong to one
unchanged logical catalog generation. Unknown fields remain unknown and a
missing exact choice refuses. These metadata observations authenticate neither
checkpoint bytes nor Hugging Face payloads and grant no decoder runtime,
training, proof, execution or completion authority.

The [retained catalog survey](evidence/supervisor-task-context-20261005/live-model-manager-catalog-survey.json)
read one explicit existing ModelManager DuckDB store without constructing the
manager or changing the store. It contains 668 IR declarations, including
detached components and an existing UI/UX family. Only two declarations have
all four schema-version/decoder-task/profile/format fields populated. The core
family/dimension counts in that store are:

| Family | 8D latent declarations | 384D input-embedding declarations | 768D input-embedding declarations |
| --- | ---: | ---: | ---: |
| CodebaseIR | 7 | 2 | 0 |
| SecurityIR | 18 | 47 | 0 |
| LegalIR | 16 | 90 | 10 |
| IntentIR | 18 | 19 | 0 |

LegalIR also has one separate 8D input-embedding declaration; its geometry is
not interchangeable with the 8D latent column. These counts describe this
store's metadata and do not count qualified decoders or survey every host
inventory. [Nine exact existing nominations](evidence/supervisor-task-context-20261005/retained-ir-catalog-nominations.json)
resolve across the four 8D/384D pairs and LegalIR 768D. Exact CodebaseIR,
SecurityIR and IntentIR 768D nominations refuse in this store. A separate
[filesystem witness](evidence/supervisor-task-context-20261005/retained-checkpoint-file-witnesses.json)
matched all nine selected checkpoint files to their registered byte hashes
without loading them. These witnesses do not change the resolver's false
checkpoint/runtime authority flags or select a quality winner.

The next work should recover missing format identities from original validated
checkpoint manifests, retaining separate registrations for semantic IR,
source-text reconstruction and each formal-logic output task. It must preserve
family-specific 8D/384D/768D inventories and DuckDB/DuckLake identities, cached
embedding geometry/encoder revisions and immutable public Hugging Face
revisions. Runtime admission must join those exact declarations to authenticated
checkpoint bytes, the actual family/decoder ABI, verified token/span ceilings,
inventory and runtime receipts. The context route's 8192-byte query bound is
not an encoder token-budget claim. Reuse the existing teachers, embeddings,
heads and held-out reconstruction splits when qualifying distillation; do not
infer an 8192-token capability from the selected dimension or seed new training
from random weights.

After runtime selections qualify, bind dependent contexts to checked
predecessor publications and the resulting source successor, then qualify
parallel isolated worktree leases, independent validation/review, merge
currentness and durable restart/completion. Keep generated formal candidates
separate from checked proofs in the proof index. The reviewed profile's legacy
context and native launch guards remain in place until those gates pass.

## Retained checkpoint authentication and decoder recovery (2026-10-06)

The [frozen qualification](evidence/supervisor-decoder-contract-20261006/qualification.json)
passed **512 checks** with zero failures, errors or skips, including all 445
preceding ready-root regression nodes, 41 new byte-authentication controls and 26
new native context controls. All cases have fresh AST seals without completion
authority. The 68 manual source/test pins, datasets checkout and16 retained files
stayed unchanged. Focused/development runs overlap and are not added.

`authenticate_task_ir_checkpoints` joins exact persisted ModelManager selections
to the original registered checkpoint byte pins. It validates every selected
path, regular single-link file and byte budget before opening any checkpoint,
then streams SHA256 through read-only descriptors and checks file identities,
a fresh complete catalog generation and all closing descriptor/path witnesses.
The per-file ceiling is 512 MiB; the conservative selected population ceiling
is 1 GiB. Checkpoint JSON, tensors and models are not deserialized. This closes
a byte-authentication gap while leaving shape/ABI, encoder, token/span,
inventory, reconstruction quality and decoder runtime admission separate.

The ready-root producer has an explicit `authenticate_ir_checkpoints=True`
option using exact per-task catalog nominations. Its version-2 receipt carries
per-task checkpoint observations and marks only its own checkpoint-byte
observation true. The embedded catalog resolutions retain their original false
checkpoint/runtime authority fields. Source/admission/native-owner fences run
again after checkpoint reads, and a batch fence rechecks earlier tasks' file
identities after later task reads. Cold reopening repeats byte authentication;
`require_checkpoint_authentication=True` refuses a downgrade to metadata-only
receipts. Default version-1 receipt fields and behavior remain unchanged. Both
versions keep decoder runtime, proof, execution and completion authority false.
These endpoint observations are cooperative currentness checks, not an atomic
filesystem/catalog/native-owner transaction or a lease.

The [actual retained replay](evidence/supervisor-decoder-contract-20261006/retained-checkpoint-authentication.json)
authenticated **eight existing checkpoint files**, 105,868,568 bytes, from the
selected existing ModelManager store, with unchanged store/WAL stat witnesses.
The original nine-selection batch refuses the borrowed Codebase384 checkpoint's
`test_real_fit_registry_replay_current` ancestor alias. Its bytes still match
the earlier filesystem witness, but an alias is not silently substituted with
the current target. Recover a separately reviewed canonical asset binding with
the original identity/provenance retained. None of these byte observations
loads a model or authenticates a remote Hugging Face payload.

The [retained contract survey](evidence/supervisor-decoder-contract-20261006/retained-decoder-contract-survey.json)
found substantial differences within the nominal family/dimension matrix:

| Retained selection | Actual contract | Decoder admission gap |
| --- | --- | --- |
| Core-family 8D latent checkpoints | Structural feature/projection reconstruction; source-text decoder untrained | Preserve feature codecs and geometry; author separate supported output-task contracts |
| Codebase384 | SecurityIR payload inside a Codebase source-generation envelope; native Codebase decoder incomplete | Preserve owner/payload distinction and resolve the original locator alias explicitly |
| Security384 | Source to serialized program-expression fragment, `program-ir/v1` | Existing exact profile is recoverable; full native document and source-text decoding unsupported |
| Intent384 | Source to serialized rich-intent AST fragment, `intent-rich-grammar/v1` | Existing exact profile is recoverable; full native document and source-text decoding unsupported |
| Legal384 pilot | One-rule deontic formula, `CanonicalRoundTripIR@1`, 64 source/target codec tokens | This head does not reconstruct the originating legal prose |
| Legal768 interfaces | Frozen inherited 8D/384D bodies and trained connecting interfaces | Encoder 8192 profile differs from inherited output limits; source fidelity and distillation remain unqualified |

The clean committed native profile owner is pinned separately at datasets
`73db2c8f3edb9fbdfc5decb0e8f12bb4e86e10f3`; it is not mixed into the supervisor's
`987cf856b2b902aa68c4587bb492b19b932b5d30` runtime checkout. Its original inventory
supports exactly the two Intent/Security384 fragment profiles. Rebuilding the
original saved inventory preserves both format/profile/checkpoint-record IDs
and all twelve lanes, but native route selection initially refuses changed
source-file locations and the corresponding concrete contract digests. The
[explicit custody rebind](evidence/supervisor-decoder-contract-20261006/original-profile-custody-rebind-receipt.json)
writes a new detached inventory and resolves both original exact routes through
the actual public owner. It preserves the original inventory and all false
runtime/quality authorities. The other seven nominated legacy assets still lack
supported native format contracts; their null selectors cannot be filled from
filenames, serialization schemas or family/dimension alone.

The survey also recovered the stronger semantic reconstruction path:
historical contextual384 and contextual768 replay each reports **48/48 exact
canonical IRs, 180/180 rules and 720/720 actor/action/modality/object fields**.
Their complete retained states remain present and byte-bound. The historical
replay used paragraph vectors plus explicit cached source-clause vectors,
clause masks and the original codec/preprocessing; this survey does not freshly
witness those contextual cache files. This is historical replay
on its specified panel, not a fresh holdout or original legal-text reconstruction.
Both contextual decoders retain 512-token output limits; the new
[custody recovery](contextual_ir_checkpoint_recovery.md) distinguishes their
actual source-producer scopes. The result does not qualify an
8192-token span or single-vector reconstruction. Neither selected-state SHA
appears in that survey's 668-binding selected ModelManager store. Remote Hub
publication of those exact complete states is unestablished in this evidence.

A [fresh publication survey](evidence/supervisor-decoder-contract-20261006/native-profile-publication-survey.json)
confirms that the native profile owner file exists in clean committed `73db2c8f`
but is absent from the surveyed datasets `origin/main` tree at `5171a632`.
Commit ancestry alone does not establish that a runtime checkout contains the
owner. Recover its complete metadata-owner closure against current main and
qualify codec/profile identity and API compatibility before publishing that
integration; do not import a partial old closure into the supervisor's pinned
runtime.

The next recovery plan prioritizes those existing contextual states before
further fitting. Create reviewed append-only format/input-contract records for
both complete wrappers, retaining their exact donors, paragraph/clause cached
vectors, masks, ordered codecs, frozen held-out splits and original byte hashes.
Recover the actual IR output schema independently of checkpoint serialization;
keep semantic IR generation, originating legal-text reconstruction and each
FOL/TDFOL output as separate decoder tasks. Then use the existing publisher and
ModelManager importer with immutable public Hub receipts and actual persisted
readback. Preserve earlier unknown-profile records and separate family/8D/
384D/768D inventories, geometry and DuckDB/DuckLake databases.

For each output task, report canonical semantic equality, field/rule coverage,
syntax validity and verbatim/normalized source-text equality separately.
Source reconstruction must evaluate `legal text -> legal_ir -> legal text`;
deterministic decompilation of a canonical formula does not recover original
wording. Define whether the text task requires verbatim bytes or equivalent
legal meaning before choosing its objective. A semantic IR that collapses
wording, punctuation, references or ordering must retain source-form anchors
or an explicit residual when lossless wording is required. Count those inputs
in the representation and report a separate ablation without them; cached
clause vectors and raw-source sidecars cannot be treated as single-vector
reconstruction. Any clause/raw-source inputs retained by the IR need an explicit
input contract and an ablation against embedding-only decoding. Reuse the successful
8D/384D teachers and heads to initialize 768D interfaces, then measure semantic
and source fidelity on the same held-out corpus before increasing span budgets.
An 8192-token encoder declaration cannot enlarge the 64/512-token decoder
output budgets or change the inherited vocabularies.
Missing family runtimes or format contracts refuse rather than substitute a
borrowed payload. After those decoder/inventory gates pass, bind dependent
contexts to verified predecessor source successors and qualify parallel leases,
review, merge currentness and durable completion before lifting execution gates.

### Original contextual state registration (2026-10-06)

The [contextual checkpoint recovery](contextual_ir_checkpoint_recovery.md)
closes the availability gaps identified above. Datasets main at `3b3b9944`
restores the complete native metadata owner; a fresh original-asset rebind
preserves all twelve lanes and both original fragment profile identities.
Freshly downloaded complete contextual384/768 states match their original
hashes at the already published immutable Hugging Face revision. Genuine
ModelManager registration and cold reload add both separate selected-state
records, moving the selected store from 668 to 670 records while preserving all
existing records, activity timestamps, schema and indexes exactly.

The original contextual cache, transforms, mask producer and ordered codec are
now freshly witnessed. Exact supervisor nominations authenticate both registered
checkpoint byte pins. Native IR schema/profile/format identities remain unknown.
The later [cached contextual replay and improvement plan](contextual_ir_checkpoint_recovery.md#next-implementation-gates)
adds an explicit datasets runtime using the original full states and caches.
Fresh replay gets 48/48 ordered exact IRs at each width, while the existing
source-withheld text renderer gets 0/48 originating-prose byte or normalized-text
matches at either width. Its 48 text outputs per width are all present. The
454 relevant controls pass; all reference qualifiers remain empty. This closes
the cached numerical replay gap while retaining separate prose-training,
fresh-holdout, native route, 8192-token, distillation and proof-index gates.

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

## Eigenvalue failure and symbolic planning reduction (2026-10-05)

The [follow-up diagnosis and native symbolic qualification](evidence/terminal-eigenvalue-symbolic-diagnosis-20261005/README.md)
identify regular Codex's sole observed failure as the size-9 speed inequality:
18.86 microseconds versus 15.76, about 19.64% too slow. Its official eigenpair
and dominance families passed. A custom size-9 probe had already reported a
speed win; the exact timing mechanism remains unresolved. The public evaluator
itself covers only even sizes, which is a validation limitation rather than a
complete explanation of the failure.

An agent-authored, source-bound version-2 requirement contract selects the
existing native symbolic planner with zero planning-provider calls. Eight
candidate atoms are covered, replayed, admitted and stored as one coding task.
This is administrative coverage; source semantics, numerical proof and
completion authority remain false. It projects the historical supervisor's
two router sessions to one, eliminating a 23,766-token planning component
(8.15% of the historical total), without measuring a new coding outcome or
official reward. Regular Codex already has one session. Further call reduction
needs a reviewed general numeric operator, compatible float/complex semantics
and independently measured public correctness and timing gates.

## Expanded profiles and Grok container qualification (2026-10-05)

The [expansion evidence](evidence/terminal-expansion-20261005/README.md) records
signed preparation and real index hydration for public XML, JSONL and calendar
task inputs, separate Source384 inference for the Python populations, and
broader finite alias/signature repairs checked with Lean and Z3. Structured-data
semantics and numerical behavior remain explicit proof obligations. Publication
refresh preserves exact data references without claiming that the model fetched
every retained capsule or source byte.

The pinned Grok route uses `llm_router` inside the isolated worker. Native
qualification found that an empty Grok tool allowlist enabled its default tools;
the corrected planning filter independently exhibited zero tools in both a
readiness probe and a full task trial. Coding exhibited its six requested tools.
The two latest MuJoCo plans still failed the strict JSON response contract.

The eigenvalue symbolic contract was also exercised in a full Grok trial: two
goals and one task were admitted with zero planning-provider calls, followed by
successful native START and coding dispatch. Coding exceeded its 300-second
limit. Replacement bootstrap failures left the task in progress, and runtime
closure refused launched-child custody despite STOP reporting zero tracked
processes. Official reward was zero and token totals were unavailable. This
qualifies additional route boundaries, not successful Grok task completion or
an efficiency advantage. The raw model/verifier bodies are excluded from the
published metadata.

## Completed Codex trial with symbolic planning (2026-10-06)

The [fresh matched-archive trial](evidence/terminal-eigenvalue-symbolic-trial-20261006/README.md)
received official reward **1.0**, covered eight authored requirement atoms with
**zero planning-provider calls**, and completed through one canonical
`llm_router` coding session. Complete observed usage was **255,951 tokens**,
including cached input, versus **291,432** in the historical successful direct
supervisor. Agent execution was **182.40 seconds**, versus **336.36**. Native
STOP, cleanup exit zero and zero remaining processes were recorded, and the
exact owned container was removed.

The demonstrated call reduction is two router sessions to one. The observed
35,481-token reduction (12.17%) comprises the removed 23,766-token planning
session plus an 11,715-token coding difference. Extra coding and time changes
remain single-trial observations: historical task/container fingerprints,
package resolution, host load, provider state and cache history differ. The
same runtime archive, checkpoint, model, public task and limits were retained;
strict retry-list serialization equality remains false although both policies
disable retries. The source-bound symbolic plan remains administrative coverage,
with no source-semantic or universal float/complex proof authority. Source384
advice has unsupported source-contract nominations and is not a numeric proof.

## Context token economy with LLM planning retained (2026-10-06)

The user clarified that removing the planning session is not the intended
optimization. The [corrected context improvement plan](evidence/supervisor-context-token-economy-20261006/README.md)
keeps both planning and coding through `llm_router`, and uses catalog metadata,
formal constraints, selected semantic capsules, reversible minification,
bounded lookup and context deltas to reduce total tokens. The earlier symbolic
trial remains an explicit historical ablation; the normal direct route was
never globally disabled.

Independent read-only audits show that the current coding transport already
removes 7,443 bytes through JSON unpacking and typed identifier aliases. Its
appended metadata matters: the historical direct input is 59,301 bytes after
all additions, while the symbolic ablation is 75,724 bytes, including a
20,076-byte instruction/IntentIR block. Current catalog locators are not yet a
bounded model lookup tool, and existing decision/delta compilers are not fully
integrated with the terminal provider/session path. The plan prioritizes full
input/per-turn accounting, checked decision views, shared metadata tables,
compact proof/status cards and progressive disclosure while preserving source,
scope, unknown obligations and normal native validation. It claims no measured
token saving for an unrun planning-preserved context experiment.

## Opt-in coding transport with controller-held identifiers (2026-10-06)

The [first context-reduction implementation](evidence/terminal-context-dictionary-20261006/README.md)
adds `supervisor-semantic-router-input@2`. The controller retains the immutable
identifier dictionary while the coding model receives short typed references,
literal source and native constraints. Exact reconstruction, structured reply
validation and freshness checks remain enforced. The selection is explicit and
bound through benchmark preparation, configuration, container launch and
receipts; old archives are rejected before model dispatch. Default `@1` bytes
and historical replay remain qualified.

The primary Terminal Bench comparison must retain LLM planning and coding on
the same newly pinned archive. Public-task offline prompt/tokenizer sizing
qualifies this representation experiment, while total provider-token savings
and official reward require a fresh matched comparison. Catalog lookups,
compact formal status and terminal context deltas remain the next integration
steps; the initial coding-input transport does not control native CLI history.

## Grok planning, resource admission and lifecycle recovery (2026-10-06)

The later [worker-custody qualification](evidence/grok-worker-custody-20261006/README.md)
reproduces and repairs descendant cleanup inside the isolated router worker,
requires rebuilt capability `@2` archives for every supervisor profile, and
retains bounded failure-gate observations. Fresh full-indexed trial12 still
timed out with reward 0.0, but its callback settled, the task became blocked,
and START/STOP/runtime closure completed with no remaining tracked processes.
Planning used 22,072 observed native tokens; coding and aggregate usage remain
unknown. The evidence also records concrete proposed data-contract and
package-aware symbolic coverage extensions, without claiming they are implemented.

The [Grok recovery evidence](evidence/grok-recovery-20261006/README.md) records
canonical structured planning through `llm_router`, a signed single-task schema
constraint and explicit XML, NDJSON and calendar output labels. Native Grok
proposals now pass the original schema, graph checks and independent task
admission. The added labels qualify transport and scope; they do not supply
symbolic repair operators for those data formats.

The combined runtime at `020e6adccc9ad56b1df0ee2b5984ce0adcb39d89` passed
282 targeted tests with no failures or skips. Native failed-provider callbacks
can now settle after independently verified child custody and durable state;
uncertain callbacks retain their claims. Worktree activation, pooled moves and
cleanup compare the original captured record under the store locks. Interrupted
START retains its launched root and can perform cleanup through the original
authorization before a normal STOP. Cleanup repair preserves the failed START
result and grants no completion or retry authority. Separate bounded report
fields distinguish that repair from STOP, runtime closure and worker cleanup.
These checks qualify runtime contracts, not benchmark task correctness. Earlier
failed tests and unsupported legacy APIs remain visible in the linked evidence.

The fifth fresh `tune-mjcf` trial reached coding after the explicit 20 GiB
profile cleared the earlier memory gate. Coding reached its 300-second timeout;
its final usage envelope was absent, and the unresolved callback then held the
claim until the work deadline. Official reward was zero. The sixth trial
selected a separately qualified 600-second coding cap but failed during native
START on a process-identity mismatch. STOP conflicted with the unfinished
transition, and native runtime closure retained two tracked processes. Container
teardown completed. This remains a native custody failure and does not establish
600-second provider execution. Planning reported 22,963 native tokens;
aggregate usage remains unknown.

The seventh fresh trial at the combined runtime completed native START, STOP
and runtime closure, with zero remaining processes and successful worker
cleanup. Its task became blocked and official reward remained zero. Planning
reported 21,316 native tokens; no coding invocation receipt survived, so coding
dispatch, coding usage and aggregate usage remain unknown. The retained task
receipt identifies a terminal Portal bridge failure but does not retain its
underlying error or establish which failure-settlement path was exercised.

Trial seven indexed four symbols into eleven semantic capsules, reused the
metadata/vector context and performed fresh Source384 inference. Retrieval
used lexical TF-IDF vectors in DuckDB with the DuckLake metadata projection;
the separately pinned GTE-small embedding model and security checkpoint served
Source384 inference. Neural retrieval was not selected. The generic Doctor
still reports unsupported structured-data and numerical obligations. These
trials establish neither a task proof, a full-suite score nor an efficiency
advantage. Exact source pins, earlier failures, cleanup evidence and overlapping
targeted qualifications are retained in the linked record.

The later diagnostic integration at `585cf2adc` passed 127 targeted tests on a
clean checkout. Router failures before the invocation receipt now identify a
bounded source phase. The native daemon retains a task/attempt-bound observation
outside its optional JSON event mirrors, and the final report reads it after
cleanup. A signed native integration first exposed the absent event mirror and
then verified the corrected path through START, terminal failure, STOP and
closure. These observations cannot authorize settlement or retry, establish
provider dispatch, or supply missing token totals.

Trial eight exercised that diagnostic path: the native child exited with code
one and the failed callback settled, followed by successful START/STOP closure,
zero remaining processes and worker cleanup. Official reward was zero. Planning
used 21,343 native tokens; no coding invocation receipt or router diagnostic was
retained. The bounded monitor observed no cgroup OOM event during its window.

An offline Docker replay subsequently reproduced a concrete configuration
blocker in the archived worker: its 300-second argument limit rejected the
selected 600-second cap before the router could run. The worker now shares the
router's 600-second ceiling, while older routers retain their original worker
limit. All six owner-to-worker Docker compatibility cases passed with networking
disconnected and no provider calls. Accepted arguments deliberately reached an
invalid-model preflight error; they were not successful coding runs. The replay
used the archived source with a rebuilt image, and cannot recover trial eight's
missing stderr. Extended-budget archives now bind the worker and router source
hashes and generated entrypoint, so unsupported archives are refused before paid
planning. Existing signed profiles still determine the selected invocation cap.
The merged timeout, archive capability, retrieval and budget changes passed
197 targeted tests on clean source `b966623d3`, with no failures or skips.

Separately, retrieval configuration now derives its model revision from the
selected archive instead of an older hardcoded revision. Stale active pins are
rejected before deployment or dispatch. That change passed 52 development tests;
an offline CPU GTE-small probe at revision `17e1f347…` produced 384-dimensional
vectors for two authored symbols, reopened the persisted DuckDB index and
verified its DuckLake metadata projection. It used no LLM calls, downloads or
training. This small symbol-name retrieval check does not establish full-task
semantic coverage, a benchmark memory bound or a performance advantage.

The ninth attempt selected the learned retrieval snapshot and passed archive
and preparation checks, but container setup stopped when the pinned Python
runtime download returned HTTP 500 after three retries. Indexing, planning and
native START did not begin. No verifier reward exists for that attempt; its
score is unknown, not zero. The container was removed before the independent
monitor received its identity, so no live resource samples are available.

The tenth attempt reused the exact ninth archive after the public Python asset
recovered. Actual container indexing selected learned GTE-small retrieval:
four symbols and eleven capsules in 17.40 seconds, including fresh Source384
inference in 9.74 seconds. Indexing made no LLM calls and performed no training.
The admitted context reused three capsules without new embedding calls. Planning
passed independent admission with two goals and one task, and native START
succeeded. Grok coding ran for 534.12 seconds under the corrected 600-second cap.
The original verifier awarded **1.0**, and the native task reached completed
revision four. Native usage was **1,212,732 tokens**: 136,091 input excluding
cache, 1,037,440 cached input and 39,201 output. Reasoning tokens are included
in output. Billing totals and a matched performance advantage are unverified.

This is task success with incomplete supervisor shutdown. No native STOP
receipt was retained, runtime closure refused a live launched child, and the
driver therefore reported `task_completed=false`. Worker cleanup returned zero
and the exact container was removed. The continuous monitor observed a daemon
replacement near shutdown and no OOM events; it does not establish the cause of
the missing STOP. The retained close error can mask an earlier STOP exception.
Further shutdown fixes must be qualified separately from this historical run.

A subsequent signed native regression reproduced a shutdown defect: the
interrupted-START repair hook revalidated the original START context while
reading its committed transaction. Accepted publication had correctly made
that context stale for another START, but the lookup prevented STOP as well.
The correction checks the exact prior transaction under a current signed STOP
grant and live lease. Actual interrupted-START repair still requires its
original permit and revision checks. The diagnostic path separately records
bounded STOP and close failures, including the phase of post-STOP observations.
Trial ten's original masked STOP exception remains unavailable; this
reproduction does not retroactively establish that exception's identity.
The merged correction and diagnostics at `0e2d9a8c6` passed 120 targeted tests
on a clean checkout, including signed native publication, STOP and interrupted
START recovery, with no failures or skips.

The eleventh fresh trial used that corrected source and actual CPU GTE-small
retrieval, indexing four symbols into eleven capsules in 18.19 seconds. Fresh
Source384 checkpoint inference took 10.60 seconds; the admitted context reused
three capsules without new embedding calls. Planning again independently
admitted two goals and one task. Coding reached its 600-second timeout after
600.25 seconds, and the original verifier awarded **0.0**. Planning used
**22,806 observed native tokens**; coding and aggregate usage remain unknown
because no final coding usage envelope was retained.

Native START and STOP succeeded, runtime closure succeeded, zero tracked
processes remained, worker cleanup returned zero and the exact container was
removed. No bounded shutdown failure was recorded. The task nevertheless
remained in progress at revision three: its callback retained an unknown outcome
because a complete native child-custody receipt was unavailable. Successful STOP
does not settle that callback. This qualifies live cleanup after a timeout,
while a live task pass followed by clean shutdown remains unverified. The
missing custody join and general numerical repair coverage remain follow-up
work; the retained diagnostics do not identify which custody check failed.

Publication then merged the concurrent native Codex planning-schema and
contextual decoder documentation updates. The merged source at `0a8a1b3e8`
passed 213 focused router, schema and worker-capability tests with no failures
or skips. This separate offline qualification does not change the eleventh
trial's source pin or rerun that benchmark. Its test count overlaps earlier
qualifications and is not an additional independent coverage total.

The broader regression suite still has three independently recorded contract
gaps: projection-identity fixtures reference a removed helper, older protected
path recovery expects a different retry protocol, and Docker candidate cleanup
has missing producer/consumer interfaces. These remain unqualified; the focused
passes do not establish whole-repository correctness. General numerical and
structured-data symbolic repair coverage also remains incomplete.


## Reviewed finite scheduling (2026-10-07)

The [integer scheduling contract](FINITE_SCHEDULE_CONTRACTS.md) adds an opt-in
signed @5 route for one JSON output. Shared datasets code compiles explicit
constraints to bounded QF_LIA and independently replays each SAT witness.
The supervisor hydrates its finite-check index and retains staged native
validation and publication. UNSAT and indeterminate solver outcomes remain
residuals. The [qualification record](evidence/finite-schedule-20261007/README.md)
distinguishes authored public indexing, native semantic validation and cleanup
from unsupported calendar semantics, kernel proofs and benchmark rewards.

## Reviewed spectral candidate (2026-10-07)

The [dominant eigenpair operator](spectral_symbolic_operator.md) computes with a
fixed numerical recipe and checks conditional adapter lemmas without provider
calls. Explicit task/source and target-runtime bindings permit an inert native
staging candidate. Finite numerical qualification is separate from performance:
the retained local timing attempts failed consistent improvement over NumPy,
so explicit Doctor selection remains residual and does not promote this
candidate for the optimization benchmark. General planning remains unchanged.

## Complete token reduction with metadata (2026-10-07)

The [metadata and capsule plan](terminal_token_metadata_plan.md) retains planning
and `llm_router` while reducing repeated context and reasoning. The first opt-in
implementation shares identical inline semantic metadata, verifies exact
restoration, and selects it only when the complete coding input shrinks under
the recorded byte proxy. Native catalog registration and independent input
reconstruction are qualified offline. Total provider token savings require a
separate matched benchmark; smaller input bytes do not establish that result.
