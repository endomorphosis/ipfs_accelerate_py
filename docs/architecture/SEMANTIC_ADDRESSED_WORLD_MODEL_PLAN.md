# Semantic-Addressed Neuro-Symbolic Program World Model

Status: operator-owned R2 program control

Plan revision: `SAWM-PLAN-R2`

Board namespace: `semantic-addressed-world-model-v1`

Root goal: `SAWM-G000`

Task prefix: `SAWM-`

Goal prefix: `SAWM-G`

Source observation date: 2026-08-28 UTC

## 1. Purpose and release rule

This plan governs the audit, reconciliation, extension, implementation, qualification, and progressive self-hosting of a semantic-addressed neuro-symbolic program world model across the current authoritative snapshots of `ipfs_accelerate_py`, `ipfs_datasets_py`, and `ipfs_kit_py`.

The product objective is:

```text
minimize general-LLM invocations and model context
subject to:
    no reduction in proof, validation, authority, freshness,
    safety, test, scope, or completion guarantees
```

The existing `ipfs_accelerate_py.agent_supervisor` is the execution authority. This control program does not create a second agent framework, queue, merge system, task authority, context compiler, model router, proof system, or database control plane. Ordinary implementation work begins only after this plan, its objective graph, executable board, source/dependency seal, materializer, validator, and scheduler are protected and materialized into the current DuckDB/Quack task authority.

No Markdown status is completion evidence. No generated report, class name, historical branch, fixture, model output, or planning receipt is authority for a capability. A task reaches an accepted terminal only through the current supervisor's independent validation, proof, merge, event, fence, and completion authorities.

## 2. Frozen source basis and dirty-work preservation

The program is bound to a clean, isolated exact-source worktree. The snapshot is not a claim that the original operator checkouts were clean.

| Authority | Isolated root | Branch | HEAD | Tree | Package |
| --- | --- | --- | --- | --- | --- |
| accelerator and supervisor | `/home/barberb/lift_coding/.worktrees/semantic-addressed-world-model-r2` | `agent/semantic-addressed-world-model-v1` | `54bc66e9512079ef9ddc04152bc5517cc03292d6` | `81680791ef2df9ad7cfb01dea428ed691134d5e0` | `0.0.45` |
| datasets semantic/formal authority | `ipfs_datasets_py` gitlink in the isolated root | detached exact gitlink (commit reachable from `remotes/origin/agent/sawm-r2-datasets-snapshot`) | `58e5455a600d9b88e311842541d6612649d5b8cb` | `964b3e949b80191fe1b8e59af62df08f38857102` | `0.2.0` |
| kit storage/durable graph authority | `ipfs_kit_py` gitlink in the isolated root | detached exact gitlink | `6196017ca3df016c7159dce43af60f2a0d96a9ae` | `93070c709af29095fdff11f3e2698543449c08ef` | `0.3.0` |

All three isolated repositories were observed clean before R2 control files were authored. Python was CPython 3.12.3 on Linux aarch64. Runtime dependency availability, external provers, model checkpoints, GPUs, Quack liveness, and DuckLake liveness are not inferred from paths or imports; launch preflight must probe them without installation or network side effects.

The original authority candidates were preserved in place:

- `/home/barberb/lift_coding/external/ipfs_accelerate` was at `5d79bbf8bb61e027fbda76920580cb9d4fd919c1`, tree `c5ddd043de58349aafac0f9af03c1b018313623d`, branch `recovery/pre-reboot-agent-supervisor-20260822`. It had tracked changes to `runtime/quack_state_server.py` and `task_sources/duckdb_state.py`, untracked source files `analysis/mcp_contract_catalog.py` and `analysis/mcp_invocation_trace.py`, plus untracked runtime lock/audit artifacts. None was discarded or silently used as clean evidence.
- `/home/barberb/lift_coding/external/ipfs_datasets` was at `ac82107e246b30e35a2bbdcf75e01370d22350c6`, tree `2b3d892dd1c31fb6b8a3eebdb88616d411c49a47`, branch `main`. It had a dirty `.tools/ipfs_kit_py` gitlink, a tracked `logic_processor.py` change, and untracked UI/UX IR, MCP transport, fixture, and test sources. None was discarded or silently admitted.
- `/home/barberb/lift_coding/external/ipfs_kit` was clean at the same `6196017...` commit and `93070c7...` tree used by the isolated snapshot.
- The enclosing `/home/barberb/lift_coding` superproject was dirty at `8601408d6406681a14a9488d31d1cd9a16164649`, tree `e35dfaebbd229d7e7de988b2e0345444d9284221`, branch `chore/fmt-check-main`. Its modifications remain outside this isolated implementation authority.

Ambient clones under `/home/barberb`, HACC, caches, state directories, and other worktrees are `duplicate_non_authoritative` or `historical_only` unless a later operator-owned source migration explicitly seals and admits them.

## 3. Prior-program reconciliation

Before creating R2 controls, the operator search covered the isolated tree, the original authority candidates, the enclosing superproject, sibling worktrees, `/home/barberb/.local/state/ipfs_accelerate_py`, path names, textual task/event projections, and candidate DuckDB state locations for `SAWM-`, `SAWM-G`, `SAWM-PLAN-R2`, and `semantic-addressed-world-model-v1`. No earlier SAWM plan, board, accepted task projection, or materialized SAWM database program was found.

That finding is a planning observation, not an accepted database receipt. The R2 materializer must repeat an authoritative task-store query immediately before insertion. If it finds an existing accepted SAWM record, it must stop and enter append-only reconciliation: preserve definitions, revisions, CIDs, evidence, events, and world roots; inventory current-tree satisfaction; append successors rather than rewrite history; and emit an independently admitted migration/supersession receipt. The bootstrap planning record has no fabricated CID and grants no completion.

## 4. Invariants and authority model

The system keeps four concepts separate:

1. **Exact identity.** A CID identifies canonical bytes under an explicit codec/profile. It does not prove truth, equivalence, freshness, authorization, safe reuse, or completion.
2. **Semantic projection.** An embedding or structural feature vector proposes similarity under an exact model and preprocessing profile. Projection changes never rewrite semantic-object identity.
3. **Typed semantic relation.** Equality, refinement, entailment, contradiction, compatibility, alpha/structural/logical/behavioral/observational equivalence, intent, and transition-behavior claims retain scope, assumptions, environment, policy/theory, evidence, invalidators, and authority state.
4. **Authority and admission.** A separate current authority decides whether a state, relation, proof, procedure, transition, prediction, or cache result can influence planning or execution.

Normative invariant:

```text
neural retrieval proposes;
exact resolution identifies;
symbolic analysis constrains;
proof or observation validates;
policy and current authority admit;
the supervisor decides and records.
```

No producer may approve its own context, procedure, prediction, proof, transition, world root, or completion. Contradictory admitted premises yield conflict and abstention, never ex-falso repair authority. Unknown dynamic behavior widens an explicit frontier and is never treated as absence.

## 5. Repository ownership

### 5.1 `ipfs_datasets_py`

Datasets is the semantic and formal authority. Extend its landed `ir_core`, software-contract semantic index/state, software verification, Tactician, Hammer, formalization, logic-family, proof-corpus, and domain IR contracts. It owns canonical semantic schemas, canonicalization profiles, semantic/domain identity, relation semantics, static program-graph meaning, execution-state and trace semantics, solver-neutral claims and obligations, corpus/split/rights/leakage semantics, domain adapters, freshness/invalidation facts, and proof/test selection facts.

It does not own operational scheduling, provider/model invocation, mutable vector-index files, root-pointer CAS, WAL, worktree mutation, operational transition acceptance, or supervisor completion. Existing `@1` identity payloads are immutable; new semantics use adjacent versioned contracts or explicit adapters.

### 5.2 `ipfs_kit_py`

Kit is the verified storage, retrieval, and durable graph authority. Extend its landed CID/IPLD/CAR, durable coordination/root adapter, GraphRAG, semantic-governor store, VFS, WAL/recovery, proof-certificate, cache, replication, and concurrency surfaces. It owns byte-to-CID verification, immutable block persistence, projection records and rebuildable indexes, immutable graph/events/transitions, store-before-reference, generation-bearing root CAS, recovery, replication/pinning, post-commit VFS/outbox receipts, and corruption detection.

Kit never decides semantic equivalence, proof validity, prediction truth, cache reuse, task completion, or model admission.

### 5.3 `ipfs_accelerate_py`

Accelerate is the operational intelligence consumer. Extend the current `ContextCompiler`, `DecisionRuntime`, semantic compression harness/governor, incremental verification and proof sealing, Tactician/Hammer orchestration, deterministic repair, planning, provider/resource scheduling, worktree/lease/fence/merge, adversarial assurance, event/outbox, and DuckDB/Quack control authorities.

It owns task/action selection, context construction, exact/procedural/symbolic/retrieval/model routing, execution, operational transition compilation, world-root publication, rollout, model serving, benchmark accounting, and final operational acceptance. It consumes datasets identities without redefining them and kit candidates without treating ANN output as authority.

The exact named `RepositoryWorldModel`, `ProofCarryingProcedureCompiler`, `VerifiedResidualIntelligenceFoundry`, `AutonomousMetaController`, and `CausalAbstractionSupervisorFederation` were not found as current exported snapshot classes. Similarly named ambient worktrees are historical only. Tasks must extend current landed generic contracts and services or add narrow versioned adapters; they must not copy code or authority from those worktrees by name.

## 6. Identity and graph architecture

The program preserves distinct identities for source, domain state, semantic object, canonical/static graph, graph snapshot/delta, raw and abstract execution state, stack frame, event, trace, projection/index manifest, relation claim, proof obligation/result, procedure/certificate, transition query/prediction/observation/admission/accepted transition, world snapshot/root, model/tokenizer/preprocessor/environment/policy.

Semantic-object CIDs never derive from embedding scores. Model, tokenizer, metric, quantization, dimension, preprocessing, or vector-byte changes alter projection identity only. Irrelevant wall-clock observations are excluded from semantic identity. Raw state and abstract state always have separate CIDs, and the versioned abstraction profile participates in abstract-state identity.

The physical IPLD/Merkle block graph is immutable and acyclic. Logical program graphs may contain recursion, cyclic imports, bidirectional relations, state-machine cycles, contradictory claims, and repeated states. Logical cycles are represented by immutable node, edge, relation-set, event, and snapshot blocks. HNSW or other cyclic ANN files are rebuildable projection artifacts bound to immutable projection records and an index manifest; they are not the canonical graph.

## 7. Contract families

R2 first audits and reuses landed contracts, then implements only missing versioned contracts:

- static world: `ProgramGraphNode@1`, `ProgramGraphEdge@1`, `ProgramGraphSnapshot@1`, `ProgramGraphDelta@1`, `ProgramGraphIndexManifest@1`, `CallsiteRecord@1`, `FunctionSymbolRecord@1`, `ContractStateRecord@1`, `ProofObligationGraph@1`, `StaticSuccessorSet@1`, `DynamicFrontierRecord@1`;
- dynamic world: `StackFrameState@1`, `ProgramExecutionState@1`, `AbstractProgramState@1`, `ProgramEvent@1`, `ExecutionTrace@1`, `ExecutionTraceSegment@1`, `ExecutionObservation@1`, `StateAbstractionProfile@1`, `StateAbstractionReceipt@1`;
- program transition: query, candidate, proposal-only prediction, observation, admission, receipt, model profile, and calibration contracts;
- repair: bounded operator, operator candidate, graph-delta proposal, patch-sketch IR, repair query/observation/admission contracts.

The initial implementation profile is Python. Contracts remain language-neutral, but JavaScript/TypeScript, Rust, C/C++, Java, and tightly bounded shell observation remain typed unavailable until separately implemented and qualified.

## 8. Mandatory inference cascade

Every decision records selection, skip, unavailable, rejection, and escalation reasons for this ordered cascade:

```text
1  current admitted answer or exact task result
2  exact raw-state transition
3  exact abstract-state transition
4  exact proof/test/receipt cache
5  verified procedure
6  deterministic static analysis
7  abstract interpretation, type/effect/contract/SMT
8  Tactician/Hammer proof or countermodel search
9  e-graph/equality-saturation normalization
10 deterministic repair or bounded CEGIS
11 lexical, graph, and vector analogues
12 deterministic/statistical ranker
13 local structured specialist
14 local general model
15 remote standard model
16 remote strong model
17 human review
```

A model call is forbidden without a named decision-relevant unresolved question, when an authoritative deterministic answer exists, when identical evidence already failed without a changed strategy, when privacy or validation reserves forbid it, when the model cannot supply the needed authority class, or when an empty candidate set reflects incomplete analysis rather than proved impossibility.

Specialists are separate, calibrated, abstaining problem families: existing-symbol call-target ranking, structured next-event/state delta prediction, set-valued inverse trace reconstruction, and bounded repair-operator/graph-delta prediction. Full graph-sequence/TAGSeq-style models remain experiments until simpler deterministic, lexical, vector, linear, GNN, and graph-transformer baselines are frozen and a measured held-out benefit clears safety and calibration gates.

## 9. Memory, data, privacy, and projections

The world exposes typed declarative, episodic, procedural, associative, predictive, and negative memories. Associative and predictive memories are proposal-only. Prediction becomes accepted world state only after independent observation or validation. Negative memory is scope/freshness bound and suppresses repeated waste without inventing permanence.

Training data must pass current corpus rights, privacy, lineage, and leakage authorities. Initial sources may be first-party hermetic traces, instrumented tests, admitted static/dynamic edges, synthetic programs, adversarial mutants, mechanical counterexamples, admitted public data, and explicitly rights-cleared reviewed examples. Private chain-of-thought, credentials, hidden tests, unadmitted production traces, tenant-private data, private witnesses, arbitrary transcripts, and unknown-rights source are forbidden. If no admitted corpus/checkpoint exists, the learned path returns `training_unavailable`; contracts, instrumentation, baselines, benchmarks, and runtime fallback still proceed.

Every projection binds its semantic/state CID, projection kind, model/tokenizer/preprocessing/normalization identities, dimension, metric, dtype, byte order, optional quantization identity, vector CID, privacy class, availability policy, and projection CID. NaN, infinity, dimension mismatch, unspecified byte order, unpinned models/preprocessors, score-derived identity, and unresolved mutable document IDs fail closed. ANN results explicitly carry `authoritative=false`, exact resolution data, manifest/profile identity, and limitations.

## 10. Symbolic repair, procedure extraction, and context

The repair route is: current static facts; complete resolved impact closure plus explicit unknown frontier; finite positive/negative obligations; content-addressed premise corpus; Tactician decomposition; Hammer proof/countermodel candidates; native reconstruction/replay; admitted proof/counterexample/abstention; unique analytical repair; bounded CEGIS/e-graph/operator candidate; and only then an unresolved-residual context capsule for an untrusted model patch. Every accepted edit is re-indexed, re-planned, re-proved, replayed, and validated to a fixed point.

After an accepted task, the supervisor normalizes the trajectory, matches procedure families, anti-unifies repeated successful routes, validates preconditions/invariants/effects/rollback/generalization, tests held-out/adversarial cases, and promotes only through independent procedure authority. A typed hole is solved deterministically or by the smallest qualified specialist before a general model. Identical failed attempts cannot trigger another model call without new evidence or strategy.

The current `ContextCompiler` is extended, not replaced. Context ordering is goal/acceptance; exact current bindings; verified procedure prefix; unresolved questions; affected slice; contracts/obligations; counterexamples; selected tests/proofs; exact prior transitions; context-only analogues; and raw-source fallback. Its receipt explains every inclusion and omission, authority/freshness/source identity, decision relevance, token budget, reusable prefix, and residual question. Embeddings never suppress required source, tests, proof, policy, or authority material.

## 11. Rollout state machine

Rollout is closed and monotone:

```text
bootstrap -> shadow_write -> shadow_read -> guarded -> required
```

- `bootstrap`: build only contracts, stores, adapters, controls, and minimal wiring; existing behavior stays authoritative.
- `shadow_write`: emit candidate artifacts and roots without influencing work.
- `shadow_read`: query hypothetical exact/procedural/symbolic/neural decisions without influencing execution.
- `guarded`: exact current hits, current verified procedures, and proof-backed scoped relations may influence planning; neural output only ranks/adds context and cannot suppress source or validation.
- `required`: dispatch requires a verified pre-world and completion requires an independently verified accepted transition and generation-bearing post-root.

Required-mode tasks bind pre/post root CIDs and generations, current semantic/program state, goal/task revision, context receipt, reuse decision, predictions considered, proof/test/validation receipts, accepted execution transition, and `dogfood_mode=required`. Any unavailable or corrupt binding fails closed.

## 12. Goal hierarchy

```text
SAWM-G000  Deliver and self-host the semantic-addressed neuro-symbolic program world model
|-- SAWM-G010  Freeze authorities, identities, and controls
|   |-- SAWM-G011  Seal repositories and capability authorities
|   |-- SAWM-G012  Reconcile landed systems and exact gaps
|   `-- SAWM-G013  Establish identity, relation, and authority contracts
|-- SAWM-G020  Build canonical static and dynamic semantics
|   |-- SAWM-G021  Typed static program graph
|   |-- SAWM-G022  Runtime state, event, and trace semantics
|   `-- SAWM-G023  Abstraction, equivalence, and invalidation
|-- SAWM-G030  Build verified Merkle-DAG storage and retrieval
|   |-- SAWM-G031  Semantic objects and model-pinned projections
|   |-- SAWM-G032  Graph snapshots, transitions, and world roots
|   `-- SAWM-G033  Recovery, VFS, replication, and concurrency
|-- SAWM-G040  Increase symbolic autonomy and reusable synthesis
|   |-- SAWM-G041  Exact and proof-backed reuse
|   |-- SAWM-G042  Tactician, Hammer, e-graphs, and CEGIS
|   `-- SAWM-G043  Accepted trajectories into procedures
|-- SAWM-G050  Add calibrated residual transition intelligence
|   |-- SAWM-G051  Rights-admitted corpora and benchmarks
|   |-- SAWM-G052  Next-call edge completion
|   |-- SAWM-G053  Next-event, inverse-trace, and repair prediction
|   `-- SAWM-G054  Graph-sequence and structured model experiments
|-- SAWM-G060  Integrate supervisor planning and context minimization
|   |-- SAWM-G061  Context, memory, routing, and cognitive budgets
|   |-- SAWM-G062  Progressive self-hosting and required dogfooding
|   `-- SAWM-G063  Causal federation and distributed execution
`-- SAWM-G070  Qualify safety, efficiency, self-hosting, and release
    |-- SAWM-G071  Adversarial, corruption, recovery, and security gates
    `-- SAWM-G072  Benchmark, capstone, and release evidence
```

## 13. Executable task program

Every task record carries its goal/parent, closed status, completion mode, priority, track, dependencies, bundle, lane, resources, timeout, provider role, repository owner, exact inputs, outputs, predicted paths/symbols, public interfaces, preconditions, effects, current validation command, evidence, acceptance criteria, conflict policy, context budget, no-model route, bounded fallback, rollout mode, protected paths, and limitations. Unknown write overlap serializes. Shared exports, gitlinks, schema catalogs, CLI registration, scheduler controls, and release reports have explicit integration owners.

| Wave | Tasks | Purpose and principal owner |
| --- | --- | --- |
| W0 | `SAWM-000` | Operator-only R2 inventory, controls, seals, materialization, validation, scheduler dry-run |
| W1 | `SAWM-001` | Current authority and overlap matrix across all three repositories |
| W2 | `SAWM-002`–`004` | Datasets identity/canonicalization, scoped relations, and domain adapters |
| W3 | `SAWM-005`, `007`, `011`, `015` | Static/dynamic contracts, kit verified storage, accelerator narrow adapters |
| W4 | `SAWM-006`, `008`, `010`, `012` | Static construction, hermetic Python tracing, transition contracts, exact projection resolver |
| W5 | `SAWM-009`, `013`, `017` | Sound abstraction profiles, immutable graph/transition/root persistence, symbolic successors |
| W6 | `SAWM-014`, `016`, `018`, `023` | Kit CAS/recovery/VFS, exact reuse gates, Tactician/Hammer, admitted trace corpus |
| W7 | `SAWM-019`, `020`, `024` | Sound normalization, procedure compilation integration, next-call benchmark |
| W8 | `SAWM-021`, `022`, `025` | Deterministic operators/CEGIS, context capsules, smallest qualified call ranker |
| W9 | `SAWM-026`–`029` | Next-event, inverse trace, repair prediction, fair graph-model ablations |
| W10 | `SAWM-030` | Existing foundry-style corpus/training/calibration authority integration; typed unavailable if absent |
| W11 | `SAWM-031`–`033` | Serving/hardware, cognitive scheduler integration, causal event/outbox integration |
| W12 | `SAWM-034` | `shadow_write` activation and parity evidence |
| W13 | `SAWM-035` | `shadow_read` activation and false/missed-candidate evidence |
| W14 | `SAWM-036` | `guarded` qualification |
| W15 | `SAWM-037` | `required` enforcement with independent root/transition completion gates |
| W16 | `SAWM-038`, `039` | Multi-view projections and typed service/CLI/MCP diagnostics |
| W17 | `SAWM-040` | End-to-end positive and negative acceptance matrix |
| W18 | `SAWM-041` | Frozen paired ablation and efficiency benchmark |
| W19 | `SAWM-042` | Adversarial assurance and security qualification |
| W20 | `SAWM-043` | Supervisor-owned required-mode bounded self-hosting capstone |
| W21 | `SAWM-044` | Current-tree release/migration/limitation report and transitive root verification |

Task-specific intent is normative as follows: `002` preserves all landed `@1` identities; `003` keeps similarity non-authoritative; `005` and `006` retain logical cycles and unknown frontiers; `007`–`009` keep raw/abstract state distinct and abstraction fail-closed; `010` makes all predictions proposals; `011`–`014` compose kit storage rather than adding an engine; `015`–`022` extend landed accelerator services; `023`–`031` cannot fabricate corpora/checkpoints/hardware; `032` forbids model calls that cannot alter an admissible decision; `033` wakes only affected supervisors; `034`–`037` never skip rollout gates; `039` adds no service/dashboard/second MCP server; `040`–`044` retain failures and unavailable results in denominators and may not lower a safety floor.

## 14. Validation and safety floors

Focused suites must cover deterministic/strict identity, Merkle acyclicity and logical cycles, advisory retrieval and exact resolution, raw/abstract state and trace truth, calibrated call/event/inverse/repair predictions, procedure independence and freshness, VFS/WAL/root concurrency and recovery, cold-import safety, privacy, path confinement, rights admission, cancellation, and simulated-versus-live separation.

Required zero floors are:

```text
false completions
unauthorized mutations
similarity-only exact reuse
stale authoritative reuse
simulated-as-live admissions
model-created proof, authority, or completion
hidden validation reductions
escaped critical seeded defects
unsafe abstraction reuse
root overwrite conflicts
```

Tests have no ordinary network, installer, socket, repository scan, database-open, model-load, watcher, subprocess, or environment-mutation import side effect. Authoritative records never use pickle or arbitrary-code deserialization. Credentials remain environment-only and never enter prompts, argv, logs, receipts, taskboards, corpora, or model artifacts.

## 15. Benchmark and promotion rule

Freeze a paired benchmark before tuning. Its ladder is general-LLM baseline; current supervisor; exact CID/receipt reuse; procedures; static/symbolic pruning; Tactician/Hammer/synthesis; graph/vector retrieval; simple call ranker; structured event specialist; inverse/repair specialists; experimental graph encoder; and complete guarded/required consumer.

Report accepted-task and accepted-criterion denominators; all LLM/specialist calls and tokens; prefix reuse; avoided-call reasons; human/provider/CPU/GPU/validation/proof cost; failures/escalations; exact/procedural/proof/context retrieval; stale/rejected candidates; latency; call-target top-k/MRR/coverage/calibration/OOD; event/state agreement; inverse valid-set/reproduction; and repair validity/type/test/proof/acceptance/regression.

Intended efficiency targets are at least 30% lower median model input per accepted criterion, 25% fewer total model calls per accepted task, 40% lower retry input, 80% of deterministic questions without a model, and 50% of eligible low-risk work without a large remote model, with no safety, proof, test, evidence, mutation, or acceptance loss. Missing a target is non-promotion, never permission to weaken a gate.

## 16. End-to-end acceptance and self-hosting

The acceptance scenario binds exact source/environment/policy; constructs datasets-authoritative static and semantic state; stores kit-authoritative blocks/projections/index manifest; traces hermetic tests; submits a goal; performs exact/procedure/proof/static/symbolic/retrieval/specialist stages; resolves current CIDs; compiles minimal context; executes in a fenced worktree; captures VFS/outbox evidence; rescans; validates; compares prediction to observation; appends the admitted transition; CAS-publishes a new root; restarts; runs a related second goal; demonstrates valid reuse and avoided/reduced LLM context; extracts a procedure when qualified; and verifies no stale, simulated, similarity-only, or self-authorized result entered the root.

Negative variants include corrupt blocks, stale source/environment, similarity traps, unsafe abstraction, unavailable ANN/checkpoint, OOD models, wrong targets/predecessors, failed proof/replay, root CAS conflict, cancellation, poisoned traces, forged receipts, privacy breach, and repeated identical failure.

`SAWM-043` is implemented and completed only by the launched supervisor in required mode. The top-level operator session cannot implement or manually complete it.

## 17. Materialization, launch, and terminal conditions

The operator sequence is:

1. validate the exact dependency/source seal;
2. validate the architecture, objectives, executable board, dependency DAG, output ownership, protected paths, rollout order, evidence, and authority rules;
3. query and materialize goals/tasks append-only into current DuckDB/Quack authority;
4. run configured-board scheduler preflight;
5. run `launch --implement --dry-run`;
6. launch the implementation supervisor;
7. monitor leases, fences, providers, resources, worktrees, validation, proof, merge, events, tasks, and goals;
8. make only bounded repairs to genuine control-plane launch blockers, then resume;
9. preserve typed blockers and every failed attempt.

The first sealed M5 implementation launch is retained as immutable evidence,
not erased or retried in place. It admitted the control capsule, claimed
`SAWM-001`, created and removed its fenced worktree, and then settled the task
to a typed pre-provider `blocked` state with zero provider calls, effects,
commits, merges, or completion changes. M6 is an append-only operator
successor: it binds the already present root-owned, read-only validation
closure; reconciles the stale root `urllib3` constraint with the current 2.x
package authority; appends operational validation-command revisions that use
the sealed bare `python` launcher while preserving every historical task
definition; and only then performs an explicit operator CAS requeue. M5's
canonical database and execution, coordination, portal, owner, and log
sidecars remain frozen and are never used as implementation source.

The first authenticated M6 live preflight then failed closed before provider
probing, task claim, or task mutation because its operator comparator evaluated
the immutable Markdown validation commands instead of M6's exact admitted
operational validation view. M6 was stopped and frozen with its database,
event prefix, projection, owner identity, stopped-status record, and migration
receipt rehashed. M7 is a source-only append-only successor: it changes no task
revision, status, definition, validation, completion, effect, commit, or merge
authority; appends only the plan source-binding revision and its operator
migration evidence; and makes authenticated live comparison reuse the closed
M6 operational-contract reconstruction. A fresh generation-9 owner on the M7
store is required before launch resumes. Direct configured-scheduler launch is
not an acceptable substitute because it bypasses the authenticated Quack,
mutation-binding, provider-token retirement, and live task-contract gates.

The first committed M7 materialization attempt is also preserved. It failed
closed before target publication while verifying the frozen M6 owner and
completion rows: the live-capable query adapter returned `DuckDBRow` records,
whose direct tuple comparison is false even though their positional values
matched the sealed M6 evidence exactly. M6 bytes remained unchanged, no M7
store was published, and no provider, task, effect, commit, merge, or
completion authority was reached. The bounded successor repair normalizes
only those closed row projections by ordinal and records the typed failed
attempt in M7's migration evidence before retrying materialization.

M7 then materialized and reached a live, authenticated generation-9 Quack
owner. Its first authenticated preflight verified the exact snapshot, all 45
task contracts, and the frozen semantic-authority digest, but failed closed
while checking the second goal: only the root goal has an `objective_id` in
the canonical population, while the comparator incorrectly indexed that
optional key for every child goal. The failure occurred before provider
probing, task claim, task mutation, effects, commits, merges, or completion.
Generation 9 was stopped and frozen. M8 is the append-only source successor:
it retains M7 and its receipt as immutable history, appends only plan revision
9 and typed operator migration evidence, binds the optional child-goal value
as the existing empty string, and moves execution to a fresh generation-10
owner. Changed controls are forbidden from restarting against generation 9.

### 17.1 M9 append-only live-runtime recovery

The first M8 implementation launch is retained as immutable live evidence. It
claimed `SAWM-001` from `todo` revision 7 to `in_progress` revision 8, then
settled it as `blocked` revision 9 after the database portal bridge classified
the implementation supervisor's stable
`external_protected_checkout_recovery_required` no-write result as a terminal
portal error. The attempt produced no provider invocation, effect claim,
implementation commit, merge attempt, or accepted-completion change. Its claim,
lease, attempt, failure payload, settlement event, and full failure receipt are
preserved. The Grok capability probe reported a hard-quota class, but the
independent quota verifier did not run, so the quota-only Codex fallback was
not admitted or dispatched.

Generation 10 was stopped before audit. The required offline DuckDB checkpoint
changed the physical control-file digest from its pre-audit digest but did not
change the event prefix, event watermark 174, task or goal meaning, projection,
semantic-authority digest, frozen-base digest, or append-surface digest. Both
physical digests and the unchanged semantic projections are sealed; neither
physical equality with the pre-checkpoint file nor a fabricated clean-state
claim may replace that evidence.

M9 authorizes exactly two bounded source repairs. First, the existing
implementation daemon, runner, and managed-supervisor command propagate the
board namespace into merge and protected-checkout maintenance locking. Equal
namespaces contend on the same board-scoped locks, different namespaces do not
share those locks, and an explicitly empty namespace retains the historical
global-lock behavior for legacy callers. Second, the existing portal bridge
defers only the closed, exact supervisor result whose reason is
`external_protected_checkout_recovery_required`, whose owner is
`implementation_supervisor`, and whose write count, projection delta, merge
reconciliation, and implementation result prove that no implementation write
occurred. Missing or altered owner, recovery, write, delta, or merge fields
remain terminal; this repair does not weaken owner tags or make general portal
failures retryable.

Unlike the earlier source-only successors, M9 must copy the frozen M8
`control.coordination.duckdb` together with the frozen control store because
the failed-completion barrier is current coordinator authority. The copied
coordinator is rearmed once using the complete sealed failure receipt and the
exact successful control CAS observation. Rearm removes only that mutable
failed-completion barrier, makes the task schedulable, and appends one
coordination event. It preserves the failed attempt, claim, lease, settlement,
failure receipt, and all prior coordination events. The execution database is
historical attempt evidence and is never copied. This is a narrowly sealed
exception for live-failure recovery, not a general permission to migrate
execution or coordination sidecars.

The provider route is unchanged: Grok 4.6 remains primary and Codex remains a
fallback only after independently verified primary quota exhaustion. Provider
output remains proposal-only and never establishes task completion. A
configuration-only Codex-primary route, an unverified quota fallback, or a
model/provider name not admitted by the current capability probe is outside
M9 authority.

M9 targets generation 11 and the exact pair:

```text
control       data/agent_supervisor/semantic_addressed_world_model/run-r2-m9/control.duckdb
coordination  data/agent_supervisor/semantic_addressed_world_model/run-r2-m9/control.coordination.duckdb
plan revision 10
event cursor  177
projection    baguqeeraebsdnrj7pvgd6yp26yr6ob7ocbrfzjwg57xfjnr6sn4xfkuvkq2q
SAWM-001      retrying revision 10
coord events  36
coord root    sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e
```

The coordination projection is deterministic, while the newly appended
coordination event identifier makes its physical database digest a publication
result. The final migration receipt records that actual digest and size; it is
never predicted or fabricated in advance.

The two stores form one publication bundle. Materialization must verify both
staged stores, remove every mutable staging alias, prove that no M9 execution
sidecar exists, hardlink-publish both stores under the pair lock, reverify the
published bytes and projections, and publish
`sawm/non-authoritative-migration-receipt@7` last. That receipt is the final
pair commit marker: it binds both physical digests, both logical projections,
the three control events, coordinator rearm event, failure-lineage identities,
and current source binding, but does not itself become task or completion
authority. A single store without its partner, an orphan marker, a pending
receipt alias, a changed pair, or a receipt published before staging cleanup
fails closed. The operator's materializer/check path must verify this final
marker before generation 11 may start and every live preflight must rebind the
marker without directly opening the Quack-owned control database.

DuckDB plus a live Quack exclusive state owner is the authoritative multi-writer control path. DuckLake is optional non-authoritative history/projection storage and cannot substitute for DuckDB/Quack. This plan does not assert either service live; preflight must prove current capability and fail closed.

The strongest honest terminal is one of: all mandatory work independently accepted with final transitive root verification; an explicitly permitted typed external-capability terminal; or an observed control-plane blocker that cannot be safely repaired within bootstrap authority. There is no background-completion claim.

Release requires terminal task/goal validation, source/dependency seal validation, required-mode context and transition receipts, self-hosted capstone evidence, transitive final world-root verification, unchanged existing identities, and all safety floors at zero. The final human- and machine-readable report must retain repository/source/dirty-state/migration facts, task revisions and merge outcomes, interface/authority/identity coverage, corpus/model/toolchain identities and typed unavailability, tests/proofs/replay/recovery, rollout receipts, benchmark denominators, token/call metrics, capstone roots, limitations, blockers, compatibility/migration guidance, and rollback target.

## 18. M10 live-projection comparator successor

M9 materialized generation 11, preserved the failed M8 attempt, rearmed
`SAWM-001` exactly once to `retrying` revision 10, and produced a valid final
pair marker. The generation-11 Quack owner then started and authenticated.
The first live operator preflight stopped before any provider probe, task
claim, effect, implementation commit, or merge with the exact typed error
`OperatorError: live Quack task authority conflict: frozen M6 task
status/revision projection differs`. The operator had incorrectly reused the
closed M6 task-head comparator, which expects `SAWM-001` at `todo` revision 7,
against the valid M9 head at `retrying` revision 10. The immediately following
check also selected M9's predecessor-M8 semantic digest rather than M9's
current semantic digest. Those are control-plane comparator defects; they are
not implementation failures and do not authorize ordinary program work here.

M10 is an append-only, source-only repair of those two live comparisons. Its
exact source scope is the scheduler, dependency seal, this plan, migration
inventory, materializer, operator, two validators, and semantic-board test.
It changes no accepted task definition, completion, task status, task
revision, provider route, coordination event, implementation source, effect,
or merge result. It preserves the M9 materialization receipt's original
publication digest as historical evidence while anchoring its copy to the
stopped post-live M9 database bytes.

M10 targets generation 12 and the exact pair:

```text
control       data/agent_supervisor/semantic_addressed_world_model/run-r2-m10/control.duckdb
coordination  data/agent_supervisor/semantic_addressed_world_model/run-r2-m10/control.coordination.duckdb
plan revision 11
event cursor  179
projection    baguqeerareq2bngq3hffyk5vidym2ukeleg5gehpaxhqvvdjayn7ucxplcaq
SAWM-001      retrying revision 10
coord events  36
coord root    sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e
semantic      sha256:a4903791c91cc2e9c3337f2abdfd6af78389f7036d54cb7f8e43246cd4f0c023
```

Materialization copies the exact stopped M9 pair, appends only plan revision
11 and one operator evidence event, and leaves task, task-revision,
completion-receipt, and coordination projections unchanged. The coordination
database is copied byte-for-byte; a second task or coordination rearm is
forbidden. Publication uses the board-scoped pair lock, no-follow stable-file
verification, store-before-reference ordering, no execution sidecar, removal
of mutable staging aliases, published-byte revalidation, and a
`sawm/non-authoritative-migration-receipt@8` final pair marker written last.
Key presence selects M10 before every predecessor and malformed M10 authority
fails closed rather than falling back.

The repaired live comparator must verify all immutable task contracts and the
single historical completion receipt while expecting exactly
`SAWM-000=completed@2`, `SAWM-001=retrying@10`, and
`SAWM-002..SAWM-044=todo@2`. It compares live semantic authority to M10's
explicit target digest above. It does not weaken definition, receipt, proof,
validation, provider, or completion authority. The Grok-primary route and the
independently verified quota condition on the Codex fallback remain unchanged.

## 19. M11 provider-launch repair and exact retry successor

M10 materialized generation 12 and its receipt-last pair marker verified. The
live owner authenticated, preflight passed, and the scheduler dispatched
`SAWM-001`. The task advanced from `retrying` revision 10 to `in_progress`
revision 11 under the exact claim and then settled `blocked` revision 12 with
settlement
`baguqeera7cwinhpjgl2etuitwiuhs4ts6lix5txyb2pjtsbfz2npfyryznma`.
The terminal portal payload reports return code 127 after the Grok container
lifecycle timed out. It records `provider_invocation_count=0` and
`effect_claim_count=0`; no implementation commit, merge, accepted definition,
accepted completion, or worker self-approval occurred. The retry policy
correctly denied automatic retry, and the claim, lease, attempt, task
revisions 10 through 12, failure payload, settlement, and complete failure
receipt remain immutable history. That canonical zero invocation counter must
not be misread as proof that no provider process ran: the pre-created
container's attached Grok execution did start, but it produced no admitted
provider result or repository effect before the lifecycle failure settled.

Read-only diagnosis found one bounded integration defect in the authenticated
Grok Docker route. The caller first invoked
`_create_grok_container_and_build_start_command`, then passed the returned
`docker start` command to
`_run_created_grok_container_with_typed_failure_capture`, whose contract owns
and executes `docker create` before exact cidfile/identity verification and an
attached start. That made the typed route process the attached start as a
creation-phase command under the 120-second creation timeout. It did not issue
a second `docker create`; it ran `docker start --attach` in the wrong phase.
M11 changes the runner so the typed fallback-capable route owns exactly one
create/identity/start sequence; the historical bounded-stderr route retains
its pre-create behavior. The focused gate proves a create timeout is a
pre-effect terminal, is attempted once, never starts either provider, and
never authorizes cross-provider fallback.

The exact M11 source/control capsule contains the nine M10 control paths plus
only these two implementation repair paths:

```text
ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py
test/api/test_agent_supervisor_grok_quota_terra_gate.py
```

M11 is not a fallback-policy change. Grok 4.6 remains primary; Codex remains
eligible only after independently verified Grok quota exhaustion; the current
capability probe is still mandatory; and provider output remains proposal-only
and cannot establish completion. The failed M10 attempt created neither
an admitted provider-result authority nor a model-invocation receipt, although
the attached Grok process did execute as described above. M11 is therefore a
zero-admitted-invocation transient launch repair followed by one
operator-authorized rearm, not a replay, provider substitution, or fabricated
completion.

The exact stopped M10 control authority is cursor 181, projection
`baguqeeramqb56wqvzegrmmuchnldydvhtu3o3takpwgzh2lyd55k4lxtzboa`, and semantic
digest
`sha256:7dcdccefb3a54d4604716427bed164e36032429477612d9c92f3bf75fc4f9d46`.
The stopped control file is SHA-256
`c38a6477e4e87aecea1c50f68d75e7c5c7cbe1ac0f50f543a9a8c2ace55c2467`;
its original M10 publication digest
`f0106a98b390e7933ff55f3f0f5326dbb8a6951adaa84721c13beff65e0a01ec`
remains separate history. The coordination base digest is
`e69a4c22a1a1a8bd38ccf210803a02ecb040d0ee9d4cc776165a4570425d6862`,
its required WAL digest is
`938dbc7028ec438889019c352b5a2529cc8c0f0fd18a11f0eb157a6772fc5321`,
and deterministic WAL replay yields predecessor coordination root
`sha256:25b8bd03cfe9a684a2626a6c5e6c955b295ba6073248c7a1cf2cd83d7550576d`
at 55 events. The required offline checkpoint does not rewrite task or event
meaning. Independent rehearsals produced the same logical replay root and
count but different post-replay DuckDB file layout bytes. Therefore M11 seals
the exact predecessor base and WAL bytes plus the logical replay root/count;
it deliberately omits any post-replay physical digest or size. The earlier
draft authority CID
`sha256:f9c059facc33c5a7aa61d137eb5175e1412cf074d715bd3a02cabefd2494f7ee`
is superseded before publication and is not launch authority. M11
copies the exact stopped control store and the complete coordination database
including its WAL-replayed failed-completion barrier. It does not copy the
execution sidecar. Under the board-scoped pair lock it appends plan revision
12, one typed repair-evidence event, and one exact task CAS from
`blocked@12` to `retrying@13`; the coordinator is rearmed once with the same
settlement and preserves all 55 predecessor coordination events before
appending event 56.

M11 targets generation 13 and the exact pair:

```text
control       data/agent_supervisor/semantic_addressed_world_model/run-r2-m11/control.duckdb
coordination  data/agent_supervisor/semantic_addressed_world_model/run-r2-m11/control.coordination.duckdb
Quack port    45255
plan revision 12
event cursor  184
projection    baguqeeratdygaminyfax543hh5bik3kj5admm5filgtu37a3jfnyc6snwnxa
SAWM-001      retrying revision 13
coord events  56
coord root    sha256:3b9ce361244387492da0156888cf6cb7377a020ca1f988b37509050975205ae7
semantic      sha256:d9a1f5bd53884346a847cb9e440b5e8e37c7e4e7a87ea52e173d719f2933821c
```

Key presence selects
`live_provider_retry_successor_materialization` before M10 and every earlier
authority. A null, incomplete, or malformed M11 declaration fails closed and
cannot reactivate a predecessor. The operator's current-head comparator must
verify `SAWM-000=completed@2`, `SAWM-001=retrying@13`, every remaining task at
`todo@2`, the one accepted completion receipt, and the exact four-entry
`SAWM-001` revision history: M9 rearm at 10, M10 claim at 11, preserved failure
at 12, and M11 rearm at 13. It compares the current semantic digest to M11's
target, never to the M8 or pre-failure M10 digest.

Materialization uses store-before-reference ordering, stable no-follow byte
checks, the board-scoped pair lock, staging-alias removal, no execution
sidecar, and published-pair revalidation. It writes
`sawm/non-authoritative-migration-receipt@9` last. That receipt is the sole
final pair commit marker and binds the authority CID, both store paths and
physical digests, cursor 184, projection, semantic digest, coordination event
count/root, exact failure and rearm lineage, source capsule, and zero-authority
claims above. A partial pair, pre-existing/orphan marker, pending alias,
changed source capsule, missing failure history, second rearm, changed provider
route, or receipt written before cleanup fails closed. Offline start and every
live preflight must verify the materializer/check result and the exact final
marker before generation 13 can launch. This control update does not itself
publish or launch M11.

## 20. M12 declared-output projection repair and exact retry successor

The stopped M11 run reached the implementation-output admission boundary for
`SAWM-001`. Both the initial provider execution and the automatic inline rescue
created the requested repository files, but the disposable Portal projection
declared their content-addressed effect identities as output paths. The scope
gate consequently restored the real files as out of scope, the ignored JSON
receipt was not staged, and both proposal paths terminated with an empty
patch. This is a generic database-to-Portal projection defect: the outer
IntentRepository `path` is the durable effect identity, while the closed
canonical nested `effect.declared_path` is the exact repository-relative
output path.

M12 changes only that adapter boundary. A closed nested record with exactly
`effect_id`, `declared_path`, and `effect=declared_output` may project its
canonical repository-relative declared path only when the outer and nested
effect identities agree. Malformed, open, unsafe, noncanonical, direct, or
conflicting declarations fail closed. Ordinary legacy outer `path` records
retain their existing behavior. The provider envelope, proposal gate,
acceptance authority, and completion authority are unchanged.

The exact M12 source/control capsule is:

```text
config/agent_supervisor_semantic_addressed_world_model_scheduler.json
config/semantic_addressed_world_model_dependencies.seal.json
docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md
docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json
ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py
scripts/materialize_semantic_addressed_world_model_program.py
scripts/ops/agent_supervisor/semantic_addressed_world_model.py
scripts/validate_semantic_addressed_world_model_board.py
scripts/validate_semantic_addressed_world_model_dependencies.py
test/api/semantic_world/test_semantic_addressed_world_model_board.py
test/api/test_agent_supervisor_database_portal_bridge.py
```

M12 does not change provider policy. Grok 4.6 remains primary; Codex remains
eligible only after independently verified Grok quota exhaustion; the current
capability probe remains mandatory; and provider results remain proposal-only.
Read-only audit observed two provider executions, 92 model calls, 11,202,083
tokens, and USD 1.23726850 across the initial and automatic-rescue routes. The
canonical failure settlement records `provider_invocation_count=0`, so M12
explicitly records `provider_execution_accounting_mismatch=true` rather than
rewriting either observation. Neither route produced an admitted effect claim,
implementation commit, merge attempt, validation run, accepted definition,
accepted completion, or worker self-approval.

The exact stopped M11 control base is SHA-256
`d32a30bf320a07b2ebf9ebd1ee66012651346b51247a2fbec983a91772f4ee48`
at 45,101,056 bytes. Its M11 publication-marker control hash
`5accfe02488f9216b3f110a35f672cce98b24762cfeca1d8918cc1c182d7a3d7`
remains separate immutable history. The stopped coordination base is SHA-256
`0c7335f3e9545859ad2da7ebc384eedee2302fd9a69b38e8e4df459b4930a0d5`
at 12,857,344 bytes; its publication-marker hash
`c7d2fdce85eb9f7003feff0205a7dd3fbccfeffb5c3da7c821c0de658dc06af0`
also remains separate. Both stopped bases have no WAL. Their logical authority
is cursor 186, projection
`baguqeeraj42oauw2k2dqekni2dsdgh6sfgbewkqpyintvry4sfzxmiu3pkfa`,
semantic digest
`sha256:44bf091895a4bc1fb2c5f0ecb95949dd01a4c5bcd1d5b4c909b49a3806978086`,
frozen-base digest
`sha256:23e1e957dea4674abbfd0b6278872aef3c7be51f4d2bfcd6bc06c758b900c732`,
append-surface digest
`sha256:9f8670c078b905833a5f0cb0cc8fba00732ca60989f23862b12d1770da135660`,
and coordination root
`sha256:b27edbdc376ccab7a167a65e13967738a30f79b42affebeb1c94710db869d963`
at 223 events.

M12 copies the exact control and coordination bases under the board-scoped
pair lock, verifies both WAL absences, and copies neither the execution nor
read-replica sidecar. It preserves the M11 publication receipt and all live
claim/failure history, then appends plan revision 13, one typed repair-evidence
event, and one task CAS from `blocked@15` to `retrying@16`. The coordinator is
rearmed once against settlement
`baguqeerafg6ci2g5mrq7ilfpa2qvinghhjqq2hdzmfprut4sbdiqosoxi5pq`,
preserving the 223 predecessor events before appending event 224. No other task
or coordination status changes.

M12 targets generation 14 and the exact pair:

```text
control       data/agent_supervisor/semantic_addressed_world_model/run-r2-m12/control.duckdb
coordination  data/agent_supervisor/semantic_addressed_world_model/run-r2-m12/control.coordination.duckdb
Quack port    45256
plan revision 13
event cursor  189
projection    baguqeeragb5uufggmw6glss2ttbubyb2aixhfio6cmmit6w4njiuozj3csmq
SAWM-001      retrying revision 16
coord events  224
coord root    sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f
semantic      sha256:239db939a5e7af260f334b325c4ec075a2faf18f01b683448625647e0962d36c
frozen base   sha256:12b25f50a7c5d3b1020b7c1f86412fa863a6c00027ee780ecd1f78c88dfa7a95
```

Key presence selects `declared_output_retry_successor_materialization` before
M11 and every earlier authority. A null, incomplete, or malformed M12 object
fails closed and cannot reactivate a predecessor. Scheduler and inventory must
carry the same closed object, whose canonical authority CID is
`sha256:786dde1f1728b907c3e28e5a09c746842c0a4a5ac0333f3ccb25f5290e8227a8`.
Materialization must preserve the exact attempt, claim, lease, fencing,
settlement, failure, provider-observation, and task-revision lineage and write
the final pair marker last. Offline start and every live preflight must verify
the materialized pair and final marker before generation 14 can launch. This
control update does not itself publish or launch M12.

## 21. M13 Quack initial-refresh lifecycle repair

M12 materialized its exact pair and receipt-last publication marker, but the
first Quack start did not reach a live scheduler.  The owner start path copied
the canonical database to its read-only replica and the initial
`quack_serve` bind on port 45256 failed while the endpoint was still observed
in transient prior-socket state.  The control database's maximum generation
remained 13, proving that this failure preceded generation-14 identity
publication.  A later start attempt correctly failed closed because the first
attempt had left a read-replica sidecar beside the otherwise stopped M12
authority.  Neither failure dispatched a task, invoked a provider, changed a
task or coordination projection, created an implementation commit, attempted
a merge, or granted completion.

Static inspection also identified a latent adjacent lifecycle hazard: after a
successful initial bind, the initial transport probe would open a client
connection before identity publication required an immediate stop/rebind on
the same endpoint.  M13 addresses both facts.  `_SawmQuackTransport.refresh`
accepts an explicit probe decision.  The transport's initial refresh is
unprobed; the post-identity refresh and every later mutation refresh remain
probed.  An unprobed refresh is never considered live.  Port `24056` is used
for the successor so the failed M12 endpoint and its transient socket state
are not silently reused.

The exact M13 source/control capsule is:

```text
config/agent_supervisor_semantic_addressed_world_model_scheduler.json
config/semantic_addressed_world_model_dependencies.seal.json
docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md
docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json
scripts/materialize_semantic_addressed_world_model_program.py
scripts/ops/agent_supervisor/semantic_addressed_world_model.py
scripts/validate_semantic_addressed_world_model_board.py
scripts/validate_semantic_addressed_world_model_dependencies.py
test/api/semantic_world/test_semantic_addressed_world_model_board.py
```

The stopped post-failure M12 control database is SHA-256
`9b09d7dcb2079963a20c6e22fa0a5ff1e0e6833a2a8ae027ac36a973adc037f2`
at 45,101,056 bytes.  Its distinct-inode read replica has the same bytes and
size; it is recorded as failed-attempt evidence and is forbidden as a
materialization source or copied sidecar.  The M12 coordination database is
SHA-256
`113f5b629317effeada5b8b81a0f8b8aaf4322ab36b6c111d830c5d9095eea55`
at 12,857,344 bytes.  Both WALs are absent, the owner-status file is absent,
and the owner directory is empty.  The immutable M12 publication receipt is
SHA-256
`fd736f7cf349efee968133fd970e33dfc1ff7f8ac8dbf22c98618ce2a0ac0185`
with receipt CID
`sha256:d17b7cea163f6834ccd7b42b30b1e19109550802d83eb9806ebab11e9d55e196`.
Its publication control digest remains distinct from the stopped post-start
control digest above.

M13 stages copies of only the exact stopped canonical control and coordination
databases outside the board-scoped pair lock.  After staging, it revalidates
the committed source, source delta, and predecessor bytes before and inside a
bounded publication lock; only then are the pair hardlinks admitted and the
directory synchronized.  It copies neither the read replica nor any execution
sidecar, verifies the absent WAL and owner artifacts, and appends exactly one
plan event and one repair-evidence event.  It performs no task CAS and no
coordination mutation: `SAWM-001` remains `retrying@16`,
the coordination database remains at 224 events with projection digest
`sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f`,
and all accepted definition/completion evidence remains unchanged.

M13 targets the following runtime binding:

```text
control       data/agent_supervisor/semantic_addressed_world_model/run-r2-m13/control.duckdb
coordination  data/agent_supervisor/semantic_addressed_world_model/run-r2-m13/control.coordination.duckdb
Quack port    24056
generation    14
plan revision 14
event cursor  191
projection    baguqeeraqpkofyd3pnjnaqiwwefz7pkubim7kaklbp65puqu47ckb2vdmtxa
SAWM-001      retrying revision 16
coord events  224
coord root    sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f
semantic      sha256:239db939a5e7af260f334b325c4ec075a2faf18f01b683448625647e0962d36c
frozen base   sha256:12b25f50a7c5d3b1020b7c1f86412fa863a6c00027ee780ecd1f78c88dfa7a95
prior append surface sha256:c5eebae5486bf281ebfaa03f369eb114ceef707d446afe45a929a27d250363aa
```

Generation 14 is retained because the stopped M12 database has maximum owner
generation 13 and the M13 source-only append creates no owner-generation row;
the existing state-server algorithm therefore selects 14 on the next start.
The target projection above was computed by a deterministic isolated rehearsal
of the two-event append and is prebound by the M13 authority; publication must
recompute it before the receipt can seal the pair.  The listed append digest is
explicitly the prior digest.  The target append digest depends on the sealed
source binding, is recomputed from the staged bytes, and appears only in the
final pair receipt.  Key presence selects
`quack_refresh_successor_materialization` before M12 and every predecessor.
A null, partial, malformed, sidecar-sourced, task-mutating, coordination-
mutating, or unprobed-live successor fails closed.

## 22. M14 stale-owner restart successor (controlled, not materialized)

M14 is the presence-first successor to the stopped M13 owner recovery.  It
binds the stopped canonical M13 control pair, recovery receipt, status
projection, absent owner marker/WAL/listener/PID artifacts, and the exact
protected c31-to-successor source delta.  The successor must also descend from
the exact post-stop source `e21ff3a799138f3da4e492abdb34c7a4496c3da5`
with tree `8389435cf9a79c9561477ab7f11e5491a6659164`; ancestry from the
older c31 launch source alone is insufficient.  It targets `run-r2-m14`,
generation 15, Quack port 24057, plan revision 15, and only events 198/199.
Generation 15 follows from the stopped M13 database's exact maximum owner
generation 14; the M14 source-only append creates no owner-generation row.

The sealed `sha256:6a1450549a555f556ac08f6e152bca0ebfdea54d42babb6db3f93c062eac2e72`
append-surface digest is explicitly the stopped M13 *prior* digest.  M14's
target append digest includes the exact revision-15 plan, evidence node, and
event envelopes, depends on the sealed successor source binding, and therefore
is reconstructed from the verified published database and recorded only in
the final pair receipt.  A projection CID is not evidence for those bodies.

The control and coordination pair must be copied privately on the target
filesystem, revalidated on disposable copies, admitted as inode-bound
hardlinks under one bounded pair lock, and receive its non-authoritative
receipt only after every target sidecar and staging alias is absent.  A valid
pair left before its receipt is recovered by exact re-verification and the
pending-receipt hardlink protocol; partial pairs, ambiguous pending receipts,
and forged receipts fail closed.  No M14 target artifact is created by this
source-control change.

## 23. Current limitations at seal time

- R2 program-world-specific contracts, trace corpus, prediction specialists, calibrated checkpoints, required-mode roots, capstone evidence, and release benchmarks are not present at bootstrap and cannot be claimed by this document.
- Several desired accelerator authorities exist only as related current primitives or ambient historical worktrees, not as the exact named landed services. Their tasks begin with interface reconciliation and versioned extension.
- External theorem provers, model checkpoints, ANN backends, provider routes, hardware, Quack transport, and DuckLake execution remain runtime-probed capabilities; absence is typed and never simulated.
- Python is the only initial execution/tracing profile. Other languages remain unavailable until separately qualified.
- This plan and the inventory migration observation are operator controls, not task acceptance receipts, semantic CIDs, proofs, benchmarks, or world roots.
