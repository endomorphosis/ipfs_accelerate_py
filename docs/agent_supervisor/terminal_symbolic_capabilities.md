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
| P0 | Symbolic decomposition in generic benchmark profiles | `planning/intent_requirement_adapter.py`, `planning/intent_symbolic_planning.py`, obligation compiler, critic and formal-plan validator. Current generic constraints force one task; add reviewed requirement/operation bindings and exact dependency/effect coverage. |
| P1 | Honest empty-source and zero-symbol context | Existing manifest, corpus inventory and native empty-intent-world capture. Extend initial context, replay, audits and successor refresh together. Do not invent code symbols or remove a guard without a replacement schema. |
| P1 | Independent behavioral contracts for newly created code | `analysis/required_behavior_synthesis.py`, `proof/missing_input_synthesis.py`, `analysis/tactician_guided_behavior_synthesis.py`. Preserve source precedence and unresolved clauses; inferred formulas are nominations until independently grounded. |
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
before checkpoint consumption. A general source-empty context remains unfinished.
