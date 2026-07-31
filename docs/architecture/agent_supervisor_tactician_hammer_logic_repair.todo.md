# Agent Supervisor Tactician-Hammer Logic Repair Taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `LPR-`.

Companion artifacts:

- objective heap:
  `docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md`
- architecture:
  `docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md`

This is a successor projection over completed `RPR-000` through `RPR-047`.
Those landed interfaces are capability-probed preconditions, not unknown local
task dependencies.

Control-plane extension note: `LPR-000` records the original sealed 21-task,
seven-goal bootstrap acceptance snapshot. The append-only `LPR-G070` extension
adds `LPR-021` through `LPR-028` without rewriting any existing task identity;
the append-only `LPR-G080` through `LPR-G110` deterministic-doctor extension
adds `LPR-029` through `LPR-042` while preserving all prior task identities.
The current validator therefore requires 43 tasks, 12 goals, and `LPR-042` as
the unique terminal task.

Normative execution order:

```text
static facts and impact closure
  -> finite goals and independent premise corpus
  -> domain-neutral Tactician plan
  -> bounded non-authoritative hypotheses
  -> strict typed obligation lowering
  -> Hammer candidate/counterexample
  -> native kernel reconstruction
  -> prediction admission or abstention
  -> analytical transform
  -> bounded context-rich llm_router proposal when necessary
  -> exact atomic transaction
  -> candidate-tree logic and program fixed point
```

The deterministic-doctor path is an independent no-model specialization:

```text
frozen repository/AST/graph/cache/index CIDs
  -> typed diagnostics and complete impact frontier
  -> exact-first plus advisory KG/vector/embedding candidate retrieval
  -> datasets Tactician plan
  -> cache-first Hammer/CEGIS and native reconstruction
  -> one closed typed analytical operator or abstention
  -> isolated candidate worktree and atomic all-caller SCC transaction
  -> re-index, invalidate, re-prove, and reach a fixed point
```

Its report-only mode is the default. Deterministic repair forbids every LLM or
model-provider invocation and never falls back to one; unavailable,
ambiguous, unsupported, stale, or incompletely closed work produces a typed
abstention or approval-required receipt with zero source writes.

Tactician, vector/KG retrieval, learned ranking, solver candidates, runtime
witnesses, ordinary tests, and LLM proposals do not independently grant
semantic or mutation authority. Reviewed conformance tests may define
expectations under existing precedence, but do not prove them by passing.

## LPR-000 Bootstrap and seal the Tactician-Hammer logic-repair control plane

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: control
- Depends on:
- Goal id: LPR-G000
- Outputs: docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md, docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md, docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md, config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json, scripts/validate_tactician_hammer_logic_repair_board.py, scripts/tactician_hammer_logic_repair_supervisor.sh, test/api/test_agent_supervisor_tactician_hammer_logic_repair_bootstrap.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_logic_repair_bootstrap.py && python scripts/validate_tactician_hammer_logic_repair_board.py --check-all && scripts/tactician_hammer_logic_repair_supervisor.sh doctor
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/control
- Parallel lane: control
- Resource class: cpu-small
- Resource stage: analysis
- Token class: small
- Estimated tokens: 16000
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md, docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md, docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md, config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json, scripts/validate_tactician_hammer_logic_repair_board.py, scripts/tactician_hammer_logic_repair_supervisor.sh, test/api/test_agent_supervisor_tactician_hammer_logic_repair_bootstrap.py
- AST symbols: ProgramLogicGoal, LogicGap, TacticianSearchPlan, LogicPredictionReceipt
- Interfaces: TacticianHammerLogicRepairPlan@1
- Allow concurrent with:
- Conflict policy: All six control artifacts are protected after this committed bootstrap; implementation workers and terminal rollout tasks must not edit them.
- Preconditions: RPR-000 through RPR-047 are complete; accelerator HEAD and the exact initialized ipfs_datasets_py gitlink are recorded by launch doctor.
- Effects: A parseable objective/goal/subgoal/task program has a protected scheduler, full DAG validator, and idempotent dedicated launcher; after this task completes, four file-disjoint foundation tasks are ready.
- Evidence subset: architecture, objective hierarchy, task dependency DAG, ownership, validations, acceptance, rollout
- Acceptance: All six control artifacts exist and are committed; validator proves 21 unique tasks, all seven goals, every local dependency and goal reference, acyclicity, and that exactly LPR-001 through LPR-004 become ready after LPR-000; scheduler disables objective/codebase refill, protects all six artifacts, binds the exact accelerator branch and initialized datasets gitlink, configures four strict file-disjoint lanes, shadow mode and all zero safety floors; launcher doctor/start/status/restart/stop is idempotent in fake-process tests, stores no secret in argv/logs and never kills unrelated processes; no vector, Tactician, Hammer candidate/countermodel, ordinary test, or LLM output is described as independent authority.
- Embedding query: seal code logic prediction tactician hammer supervisor objective goal subgoal taskboard

## LPR-001 Define bounded program-logic prediction contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prediction-contracts
- Depends on: LPR-000
- Goal id: LPR-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_prediction_contracts.py, test/api/test_agent_supervisor_program_logic_prediction_contracts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_logic_prediction_contracts.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/contracts
- Parallel lane: lpr-contracts
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_prediction_contracts.py, test/api/test_agent_supervisor_program_logic_prediction_contracts.py
- AST symbols: ProgramLogicGoal, LogicGap, TacticianSearchPlan, LogicHypothesis, LogicSubgoal, LogicPredictionReceipt, ProgramLogicNativeGoalBinding, CountermodelValidationReceipt, LogicGuidedRepairPacket, LogicFixedPointEvidenceAttachment
- Interfaces: RepositorySnapshot, ProgramContract@1, PropagationAuthorityRoots, ProofReceipt, ContextCapsule
- Allow concurrent with: LPR-002, LPR-003, LPR-004
- Conflict policy: Own only the new LPR contract module and test; import existing canonical identity, RPR contract, proof, and context types without editing or redefining them.
- Preconditions: Existing RPR contracts, exact authority roots, canonical JSON/identity helpers, proof receipts, and context capsule types are importable.
- Effects: Every later stage exchanges immutable, finite, body-free, content-addressed records with explicit authority, completeness, unsupported state, and invalidation.
- Evidence subset: schemas, canonical serialization, bounds, content IDs, state machines, authority lattice, forged/stale rejection
- Acceptance: Define closed goal/gap/source/subgoal/hypothesis/prediction/native-goal/countermodel/context-overlay/fixed-point-attachment dispositions; bind objective/trace/change/consumer plus forest/tree/overlay/graph/index/corpus/model/translator/toolchain/policy/environment roots; ProgramLogicNativeGoalBinding carries exact GoalSnapshot/native source/kernel and semantic round-trip receipt; CountermodelValidationReceipt separates raw diagnostic from replayed rejection evidence; LogicGuidedRepairPacket is explicitly a context overlay over existing RPR packet/plan/lease authority; LogicFixedPointEvidenceAttachment extends rather than replaces existing completion; record source authority separately from proof status and scores; reject bodies, secrets, non-finite or unbounded structures, cycles, forged ids, inconsistent states, semantic authority on Tactician/vector/KG/LLM nominations, solver-only verified/refuted claims, and write scope without an existing admitted RPR plan/lease; memory/resource/type facets remain distinct.
- Embedding query: typed program logic goal gap tactician plan hypothesis prediction receipt bounded authority

## LPR-002 Probe exact Tactician, Hammer, and static-analysis capabilities

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: capabilities
- Depends on: LPR-000
- Goal id: LPR-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/tactician_hammer_capabilities.py, test/api/test_agent_supervisor_tactician_hammer_capabilities.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_capabilities.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/capabilities
- Parallel lane: lpr-capabilities
- Resource class: cpu-small
- Resource stage: analysis
- Token class: medium
- Estimated tokens: 16000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/tactician_hammer_capabilities.py, test/api/test_agent_supervisor_tactician_hammer_capabilities.py
- AST symbols: TacticianHammerCapability, TacticianHammerCapabilityReport, probe_tactician_hammer_capabilities
- Interfaces: ContractRepairCapabilityReport, ChangePropagationCapabilityReport, IpfsDatasetsLogicProvider, ProgramGraph, ProgramCallResolver
- Allow concurrent with: LPR-001, LPR-003, LPR-004
- Conflict policy: Own only the new capability adapter/test; do not install dependencies, execute proof search, import optional providers eagerly, or edit existing provider implementations.
- Preconditions: The accelerator checkout and initialized datasets gitlink are discoverable without loading the current unsafe Hammer import path; the probe is schema-agnostic when the generic Tactician has not landed yet and can be refreshed after LPR-003.
- Effects: Runtime can prove which exact static-analysis, generic Tactician, corpus, selector, translation, portfolio, reconstruction, kernel, vector/KG, and llm_router capabilities are usable.
- Evidence subset: exact module paths, git revisions, schema/API versions, supported semantics, executables, environment locks, lazy/cold imports
- Acceptance: Without importing an unsafe optional provider, probe exact domain-neutral Tactician descriptors when present and otherwise report typed unavailable pending LPR-003; retain legal-adapter-only disposition; probe Hammer corpus and deterministic/learned selector descriptors, translation targets, Z3/CVC5/Vampire/E executables, Lean/Coq/Isabelle reconstruction, receipt interfaces, AST/call/dataflow/type/effect analyzers, vector/KG providers and llm_router; package presence alone is insufficient; current process-global HOME/sys.prefix lazy-load behavior is explicitly reported as import_isolation_unsafe until LPR-012 hardens it; missing/partial/incompatible/timeout and platform resource-enforcement strength are typed; learned/model/native/network/install features remain off unless explicitly admitted; policy-declared network denial is not represented as OS isolation and path/version environment locks are not represented as signed binary integrity.
- Embedding query: capability probe generic tactician hammer corpus selector translator kernel static analysis lazy

## LPR-003 Add a domain-neutral versioned datasets Logic Tactician

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-tactician
- Depends on: LPR-000
- Goal id: LPR-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/tactician, ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/submodule_registry.py, ipfs_datasets_py/tests/unit/logic/tactician, ipfs_datasets_py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/tactician
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/datasets-tactician
- Parallel lane: lpr-datasets-tactician
- Resource class: cpu-medium
- Resource stage: analysis
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/tactician/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/tactician/models.py, ipfs_datasets_py/ipfs_datasets_py/logic/tactician/planner.py, ipfs_datasets_py/ipfs_datasets_py/logic/tactician/policy.py, ipfs_datasets_py/ipfs_datasets_py/logic/tactician/receipts.py, ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/submodule_registry.py, ipfs_datasets_py/tests/unit/logic/tactician, ipfs_datasets_py
- AST symbols: TacticianGoal, TacticianSource, TacticianRoute, TacticianSubgoal, TacticianPlan, TacticianPolicy, TacticianReceipt, LogicTactician
- Interfaces: ipfs_datasets_py.logic.tactician@1, ProofTactician domain adapter, CorpusManifest
- Allow concurrent with: LPR-001, LPR-002, LPR-004
- Conflict policy: Cross-repository manual-review task: own the new datasets logic/tactician package, lazy namespace/registry entries and tests, then advance only the parent ipfs_datasets_py gitlink; any legal ProofTactician adapter preserves legal behavior and cannot import legal source categories into the generic models.
- Preconditions: The datasets submodule is initialized, clean, non-detached and on an isolated reviewed feature branch; its exact base SHA and parent gitlink are recorded; existing logic/hammers typed-record conventions are reusable.
- Effects: Datasets exposes a finite, deterministic, content-addressed, domain-neutral proof-search planner that program and legal adapters can populate with their own source types.
- Evidence subset: schema/version, canonical IDs, source routes, exclusions, subgoal DAG, stop/escalation policy, bounds, deterministic replay, domain adapters
- Acceptance: Models validate exact opaque goal/source/corpus/config roots and reject cycles, duplicate identities, unbounded fields and authority promotion; planner orders caller-provided source classes under explicit policy, records selected/excluded routes and proof gaps, emits a finite acyclic decomposition and stop/abstain conditions, performs no proof/write/network by itself, and is byte-stable on replay; optional learned/LLM guidance only reorders/nominates under pinned identity with deterministic fallback; logic/__init__.py exposes the interface lazily and submodule_registry records it without eager optional imports; legal ProofTactician compatibility is tested as an adapter, not the generic semantics; a reviewed commit lands in the nested datasets repository, the parent gitlink advances to exactly that commit, both receipts are recorded, rollback restores the prior gitlink, and no dirty/detached nested state remains.
- Embedding query: generic datasets logic tactician proof search plan source route subgoal deterministic receipt

## LPR-004 Build the adversarial live logic-repair fixture corpus

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: fixtures
- Depends on: LPR-000
- Goal id: LPR-G010
- Outputs: test/fixtures/agent_supervisor/tactician_hammer_logic_repair, test/api/test_agent_supervisor_tactician_hammer_logic_repair_fixtures.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_logic_repair_fixtures.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/fixtures
- Parallel lane: lpr-fixtures
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: test/fixtures/agent_supervisor/tactician_hammer_logic_repair, test/api/test_agent_supervisor_tactician_hammer_logic_repair_fixtures.py
- AST symbols: LogicRepairFixture, LogicRepairFixtureManifest, ExpectedLogicGoal, ExpectedRepairDisposition
- Interfaces: fixture manifest schema, existing RPR fixture schemas
- Allow concurrent with: LPR-001, LPR-002, LPR-003
- Conflict policy: Own only the new hermetic fixture directory and validation test; do not place production logic, credentials, network calls, or executable untrusted payloads in fixtures.
- Preconditions: Existing broken-contract and change-propagation fixtures may be referenced by content ID rather than copied.
- Effects: Every stage shares exact positive, negative, ambiguous, stale, poisoned, unsupported, budget, model, transaction, and fixed-point expectations.
- Evidence subset: two-to-three arguments, rename/move, threading, adapters, new support types, schemas, poisoned evidence, contradictions, native frontier, model escape, second order
- Acceptance: Manifest covers unique local value, upstream threading, deterministic constructor, multiple callers, rename equivalence, immutable and stateful support types, schema/async/error migrations, analytical and model-required paths; adversarial cases cover same-typed wrong value, vector/KG/comment poisoning, self-authored expectation, contradictory/circular premises, raw/malformed countermodel, stale/forged proof, wrong-theorem/native-statement drift, dynamic/reflection/generated/FFI/lifetime/concurrency, timeout/cancellation, path/prompt escape, partial SCC rollback, passing tests with missed caller, ordinary generic-provider f(a,b)->f(a,b,c) overlay, and second-order logic gaps; expected goals, subgoals, proof/validated-refutation/abstention, edit set and fixed point are content-identified.
- Embedding query: adversarial code logic repair fixture missing argument all callers tactician hammer prompt fixed point

## LPR-005 Build an independent content-addressed program premise corpus

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: premise-corpus
- Depends on: LPR-001, LPR-002, LPR-004
- Goal id: LPR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_premise_corpus.py, test/api/test_agent_supervisor_program_logic_premise_corpus.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_logic_premise_corpus.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/premise-corpus
- Parallel lane: lpr-premises
- Resource class: cpu-large
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_premise_corpus.py, test/api/test_agent_supervisor_program_logic_premise_corpus.py
- AST symbols: ProgramLogicPremise, ProgramLogicPremiseCorpus, ProgramLogicPremiseCorpusBuilder, PremiseAuthority, PremiseConflictReceipt
- Interfaces: ProgramLogicPremiseCorpus, lazy CorpusManifest projection, ProgramContract@1, RequiredBehaviorContract@1, ProgramDependencyGraph, ValueProvenanceGraph
- Allow concurrent with: LPR-006
- Conflict policy: Own a supervisor-native corpus/test in analysis without importing optional datasets/Hammer modules; only the lazy integration layer may project it into CorpusManifest; consume existing specs/contracts/tests/graphs through bounded references and do not edit Hammer corpus internals.
- Preconditions: LPR records/capabilities and adversarial source-precedence fixtures are stable; production Hammer corpus schema is capability-admitted.
- Effects: Tactician and Hammer receive a reproducible theorem corpus whose source authority, graph relations, exclusions, assumptions and invalidators are explicit.
- Evidence subset: reviewed IDL/schema/stub/spec/manifest/test, static type/effect/dataflow facts, graph facts, implementation hypotheses, conflict/cycle detection, licenses/redaction
- Acceptance: Project reviewed contracts/specs/conformance tests as explicit expectation classes; project candidate implementation, comments, runtime, history, vector/KG and model material as non-authoritative hypotheses; bind exact roots, symbol/type/effect/import features, dependency edges, source/span digests, translation refs, license and redaction policy; exclude bodies/secrets/unlowered directives; reject duplicate/forged identity, self-validation and derivation cycles; structural conflicts are distinguished from arbitrary logical consistency; suspected authoritative contradiction emits bounded consistency obligations, and only an independently replayed unsat core/native conflict proof creates a conflict receipt while unknown consistency abstains; analysis remains cold-importable when datasets is missing; incremental rebuild equals clean rebuild including tombstones.
- Embedding query: program theorem premise corpus authoritative contract specs dataflow hammer content address conflict

## LPR-006 Compile task and repair evidence into finite program-logic goals

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: goal-compilation
- Depends on: LPR-001, LPR-004
- Goal id: LPR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_goal_compiler.py, test/api/test_agent_supervisor_program_logic_goal_compiler.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_logic_goal_compiler.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/goal-compiler
- Parallel lane: lpr-goals
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_goal_compiler.py, test/api/test_agent_supervisor_program_logic_goal_compiler.py
- AST symbols: ProgramLogicGoalCompiler, ProgramLogicGoalCompilation, GoalFamily, GoalSourceBinding
- Interfaces: BrokenContractTrace@1, CallRequirementContract@1, ProgramContractDelta@1, ConsumerMigrationObligation@1, MissingInputRequirement@1, RequiredBehaviorContract@1, MemorySafetyFacet@1
- Allow concurrent with: LPR-005
- Conflict policy: Own only the new goal compiler/test; consume exact RPR artifacts and never convert free-form objective text or candidate observations directly into axioms.
- Preconditions: LPR goal records and the fixture corpus are stable; RPR trace/change/consumer/value/behavior records remain version-compatible.
- Effects: Each task, broken trace, and contract delta yields a finite positive/negative obligation inventory with explicit unsupported residuals.
- Evidence subset: inputs/outputs, information provenance, errors/effects, auth/capabilities, state/lifecycle, schema/serialization, placement, resource, memory facets
- Acceptance: Compile caller acceptance/value sufficiency, receiver output refinement, totality/nullability/range, allowed errors/effects/auth/resources, temporal/state/concurrency, schema/constructor/serialization/registration, placement and supported ownership/lifetime goals; retain source precedence, actual/expected facts, counterexample targets, assumptions, roots and bounds; one required goal exists per resolved consumer/facet; dynamic/native/unsupported semantics remain explicit; conflicting intent creates a diagnostic; prose can nominate a goal but cannot satisfy it.
- Embedding query: compile task intent broken trace contract delta into finite program logic goals counterexamples

## LPR-007 Slice logic gaps and required information conservatively

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: gap-slicing
- Depends on: LPR-001, LPR-005, LPR-006
- Goal id: LPR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/logic_gap_slicer.py, test/api/test_agent_supervisor_logic_gap_slicer.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_logic_gap_slicer.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/gap-slicer
- Parallel lane: lpr-gap-slicer
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/logic_gap_slicer.py, test/api/test_agent_supervisor_logic_gap_slicer.py
- AST symbols: LogicGapSlicer, LogicGapSlice, InformationDemand, StaticSliceCompleteness
- Interfaces: ProgramLogicGoal@1, ProgramLogicPremiseCorpus, ProgramDependencyGraph, ValueProvenanceGraph, ImpactClosureReceipt@1
- Allow concurrent with:
- Conflict policy: Own the new slicer/test; query existing graphs and corpora without manufacturing missing edges, definitions, dominance, or completeness.
- Preconditions: Exact goals and independent premise corpus are available under matching roots.
- Effects: Each unresolved clause carries the minimal bounded static premise/dataflow/graph slice and an explicit information demand for Tactician routing.
- Evidence subset: backward slice, reaching definitions, dominance, path condition, graph boundary, consumer closure, unsupported constructs, bounds
- Acceptance: Produce a dependency-complete minimal slice for every supported goal, including selected and excluded facts, reaching/path/dominance requirements and caller/constructor/schema boundaries; record unknown frontier, exclusions, analyzer coverage and required next source types; reject cross-root facts and forged completeness; cycles are finite SCC references; a bound or unsupported syntax yields incomplete/unsupported, never solved; slices contain references rather than source bodies.
- Embedding query: logic gap static slice information demand reaching definitions dominance impact frontier

## LPR-008 Adapt the generic datasets Tactician for program repair

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: tactician-provider
- Depends on: LPR-002, LPR-003, LPR-005, LPR-006
- Goal id: LPR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_tactician_provider.py, test/api/test_agent_supervisor_ipfs_datasets_tactician_provider.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_ipfs_datasets_tactician_provider.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/tactician-provider
- Parallel lane: lpr-tactician-provider
- Resource class: cpu-medium
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_tactician_provider.py, test/api/test_agent_supervisor_ipfs_datasets_tactician_provider.py
- AST symbols: IpfsDatasetsTacticianProvider, CodeTacticianPolicy, CodeTacticianRequest, CodeTacticianResponse
- Interfaces: ipfs_datasets_py.logic.tactician@1, ProgramLogicGoal@1, ProgramLogicPremiseCorpus, TacticianSearchPlan@1
- Allow concurrent with:
- Conflict policy: Own the new lazy provider/test; do not import legal source semantics, invoke Hammer, call an LLM, execute untrusted code, or choose paths.
- Preconditions: Capability probe admits the exact generic Tactician schema and goals/corpus share exact roots.
- Effects: Program source categories and information demands become bounded generic Tactician requests with reproducible search/decomposition plan receipts.
- Evidence subset: exact goals/corpus/index refs, source routes, queries, exclusions, subgoals, budgets, stop/escalation, lazy loading
- Acceptance: Route authoritative contracts, type/effect facts, value provenance, program graph, schema/protocol, specs/tests, history, theorem corpus, vector/KG analogues, runtime witnesses and model hypotheses under explicit precedence; local authoritative routes precede approximate/model routes; execute actual provider queries only through bounded referenced adapters; record every query/result ID and exclusion; reject stale/cross-root/malformed/free-form authority, unbounded plans and unsupported source types; response is deterministic and semantic_authority=false; unavailable provider returns typed abstention.
- Embedding query: program code tactician adapter source routes theorem vector graph proof gaps

## LPR-009 Nominate bounded program-logic hypotheses

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: hypothesis-retrieval
- Depends on: LPR-004, LPR-007, LPR-008
- Goal id: LPR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_hypothesis_retrieval.py, test/api/test_agent_supervisor_program_logic_hypothesis_retrieval.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_logic_hypothesis_retrieval.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/hypotheses
- Parallel lane: lpr-hypotheses
- Resource class: cpu-medium
- Resource stage: retrieval
- Token class: large
- Estimated tokens: 28000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_hypothesis_retrieval.py, test/api/test_agent_supervisor_program_logic_hypothesis_retrieval.py
- AST symbols: ProgramLogicHypothesisRetriever, LogicHypothesisNomination, HypothesisSignal, HypothesisCandidateSet
- Interfaces: LogicGap@1, TacticianSearchPlan@1, ProgramLogicPremiseCorpus, MissingInputCandidateRetriever, CodeSymbolVectorIndex, ChangeValueVectorIndex
- Allow concurrent with:
- Conflict policy: Own the new retriever/test; retrieval cannot lower/prove goals, select an edit target, or promote semantic authority.
- Preconditions: Exact gap slices and a current Tactician plan exist; all retrieval providers expose roots and finite bounds.
- Effects: Deterministic templates, dataflow, graph, schema, history, vector, test/spec analogues and Tactician subgoals yield a finite candidate set with counterexample targets.
- Evidence subset: source signal refs, graph/dataflow coverage, corpus premise IDs, query/result IDs, scores, exclusions, counterexample targets
- Acceptance: Union exact analytical constructions, existing values/constructors/adapters, theorem premises, graph neighborhoods, lineage, lexical/vector and optional learned/model nominations; every item carries semantic_authority=false until proof admission; hard compatibility and information-content facts are separate from ranking scores; same name/type/similarity cannot establish sufficiency; poisoned/stale/cross-root rows are rejected or retained only with rejection reasons; candidate count, text and evidence are bounded; ambiguity and no-candidate remain explicit.
- Embedding query: retrieve program logic hypothesis analytical dataflow graph vector history theorem counterexample

## LPR-010 Gate Tactician plans against axiom smuggling and stale evidence

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: tactician-security
- Depends on: LPR-001, LPR-008, LPR-009
- Goal id: LPR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/tactician_plan_gate.py, test/api/test_agent_supervisor_tactician_plan_gate.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_plan_gate.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/tactician-security
- Parallel lane: lpr-tactician-security
- Resource class: cpu-medium
- Resource stage: validation
- Token class: large
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/tactician_plan_gate.py, test/api/test_agent_supervisor_tactician_plan_gate.py
- AST symbols: TacticianPlanGate, TacticianPlanGateReceipt, TacticianPlanRejectionReason
- Interfaces: TacticianSearchPlan@1, LogicHypothesis@1, ProgramLogicPremiseCorpus, LogicGapSlice
- Allow concurrent with:
- Conflict policy: Own the plan-security gate/test; do not run solvers or mutate plan/corpus inputs during validation.
- Preconditions: Tactician plans and nominated hypotheses are complete immutable records under the same roots.
- Effects: Only current, acyclic, source-authorized, complete and policy-bounded plans may be lowered to proof obligations.
- Evidence subset: roots, goal coverage, source authority, provenance DAG, consistency, cycle detection, prompt/secret redaction, budgets
- Acceptance: Recompute goal/plan/candidate/corpus identities; require one disposition for every original goal and residual; reject changed roots, omitted facets, cycles, duplicated subgoal identity, self-authoring candidate premises, unauthorized sources, prompt directives treated as policy, secret/body leakage, forged exclusions, budget escalation, and semantic/write authority flags; structural conflicts abstain directly; suspected logical contradiction emits a consistency subgoal and permits only its proof plan to proceed, while semantic prediction admission remains blocked until LPR-012 supplies a separately validated consistency/unsat-core or native conflict receipt; unknown consistency abstains; learned/vector/model scores cannot override a hard failure.
- Embedding query: tactician plan security gate axiom smuggling contradiction cycle stale prompt injection

## LPR-011 Lower admitted tactic plans to exact existing proof obligations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: obligation-lowering
- Depends on: LPR-007, LPR-009, LPR-010
- Goal id: LPR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_obligations.py, test/api/test_agent_supervisor_tactician_hammer_obligations.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_obligations.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/obligations
- Parallel lane: lpr-lowering
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_obligations.py, test/api/test_agent_supervisor_tactician_hammer_obligations.py
- AST symbols: TacticianHammerObligationCompiler, TacticianHammerObligationCompilation, ProgramLogicNativeGoalCompiler, ProgramLogicNativeGoalBinding, LoweringResidual
- Interfaces: CodeProofObligation, ProofObligation, ContractRepairObligationCompilation, ChangePropagationObligation, LogicIR@1, GoalSnapshot, HammerRequest
- Allow concurrent with:
- Conflict policy: Own the LPR lowering adapter/test; reuse existing obligation contracts and translator capabilities without redefining or weakening them.
- Preconditions: A current Tactician plan gate receipt, exact hypotheses, corpus and static slices are available.
- Effects: Each selected hypothesis/subgoal becomes an exact existing supervisor obligation plus a semantically equivalent native reconstruction input with explicit Hammer premise and translation-map identities, or an unsupported residual.
- Evidence subset: goal/hypothesis/premise IDs, translation family/map, assumptions, facet separation, unsupported semantics, exact roots
- Acceptance: Lower input/information/output/totality/error/effect/auth/resource/state/schema/placement and supported ownership/lifetime clauses separately; bind every premise, source authority, assumption, tree/corpus/translator/toolchain/policy/environment identity; use existing ProofObligation/ContractRepairObligationCompilation and change-propagation obligations where applicable; emit no raw natural-language axiom; compile exact GoalSnapshot, single-goal Lean/Coq/Isabelle theorem source, proof hole, kernel_id and toolchain binding; independently round-trip the native statement to prove it is the same LogicIR claim; reject wrong theorem, changed assumptions/imports, source drift, omitted facets, inconsistent assumptions and cross-root premises; higher-order/dependent/dynamic/native/concurrency/lifetime semantics unsupported by the admitted translator remain typed residuals; deterministic compilation is byte-stable.
- Embedding query: tactician plan lower code contract change propagation logic obligation hammer translation

## LPR-012 Coordinate bounded production Hammer proof and reconstruction

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: hammer-coordination
- Depends on: LPR-002, LPR-005, LPR-011
- Goal id: LPR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_coordinator.py, ipfs_accelerate_py/agent_supervisor/validation/hammer_native_execution_gate.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_tactician_hammer_coordinator.py, test/api/test_agent_supervisor_hammer_native_execution_gate.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_coordinator.py test/api/test_agent_supervisor_hammer_native_execution_gate.py test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py test/api/test_agent_supervisor_ipfs_datasets_logic_conformance.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/hammer-coordinator
- Parallel lane: lpr-hammer
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 36000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_coordinator.py, ipfs_accelerate_py/agent_supervisor/validation/hammer_native_execution_gate.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_tactician_hammer_coordinator.py, test/api/test_agent_supervisor_hammer_native_execution_gate.py, test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py
- AST symbols: TacticianHammerCoordinator, HammerCoordinationReceipt, CountermodelValidator, NativeExecutionAuthorizationGate, ResourceEnforcementReport, IsolatedHammerLoader, HammerReceiptBinding
- Interfaces: IpfsDatasetsLogicProvider, ProgramLogicNativeGoalBinding@1, CountermodelValidationReceipt@1, HammerPolicy, CorpusManifest, SolverPortfolio, HammerReceipt, ReconstructionRecord, EnvironmentLockRecord
- Allow concurrent with:
- Conflict policy: Own the new coordinator/test and a focused existing Hammer-provider hardening; preserve public adapter behavior and do not route authority through logic.integration.reasoning.HammerPipeline.
- Preconditions: Exact lowering artifacts and corpus are available; capability probe admits production logic.hammers and a pinned environment; provider changes run after no concurrent task owns the file.
- Effects: After an explicit native-execution permit, the supervisor can select premises, run bounded solver candidates, normalize evidence, reconstruct exact native proofs, validate countermodels, map every Hammer outcome and persist complete state-bound receipts without process-global import races.
- Evidence subset: effective policy intersection, deterministic/learned selection, translation map, process budgets, candidate/counterexample, reconstruction, environment, full receipt
- Acceptance: Extend rather than bypass IpfsDatasetsLogicProvider; replace or isolate temporary HOME/sys.prefix mutation so unrelated threads cannot observe it; pass exact translation maps and ProgramLogicNativeGoalBinding into normalization/reconstruction; assemble/persist a complete Hammer receipt; a supervisor-owned gate defaults solver/frontend/kernel execution to disabled and requires exact operation permit, environment and policy; intersect solver/process/time/CPU/memory/native/model policies while reporting that network=false is metadata unless an OS isolation receipt exists; report POSIX/non-POSIX enforcement strength and block autonomous lanes when required CPU/memory bounds are unenforced; require reviewed executable digest or isolated execution receipt where supply-chain integrity is policy-required; deterministic selector is default and learned selection is opt-in/pinned/ranking-only; map verified/candidate/counterexample/timeout/unsupported/unavailable/policy-denied/unknown/stale/error exactly; only a matching native kernel reconstruction may prove a claim; raw countermodels remain diagnostic until deterministic replay against the original LogicIR semantics or proof of negation; stale/cross-root/corpus/environment results, malformed traces, timeouts, denials and unavailable kernels are non-conclusive; cancellation leaves no child process/temp leak.
- Embedding query: production hammer coordinator premise selection solver portfolio provenance kernel reconstruction receipt isolation

## LPR-013 Implement bounded counterexample-guided tactic refinement

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: logic-refinement
- Depends on: LPR-008, LPR-012
- Goal id: LPR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/logic_prediction_cegis.py, test/api/test_agent_supervisor_logic_prediction_cegis.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_logic_prediction_cegis.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/refinement
- Parallel lane: lpr-cegis
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/logic_prediction_cegis.py, test/api/test_agent_supervisor_logic_prediction_cegis.py
- AST symbols: LogicPredictionCEGIS, LogicRefinementState, LogicRefinementRound, LogicRefinementReceipt
- Interfaces: TacticianSearchPlan@1, HammerCoordinationReceipt, LogicGap@1, LogicHypothesis@1
- Allow concurrent with:
- Conflict policy: Own the refinement state machine/test; do not edit original goals, source authority, provider policies, or code.
- Preconditions: The Tactician adapter and Hammer coordinator emit canonical current receipts, raw diagnostic countermodels, independently validated countermodel/proof-of-negation receipts, and residuals as distinct evidence classes.
- Effects: Valid counterexamples, missing premises and unsupported constructs monotonically narrow hypotheses or decompose remaining goals under strict bounds.
- Evidence subset: prior state, counterexample binding, added/excluded premises, subgoal refinement, residual gaps, repeated-state/cycle/budget termination
- Acceptance: Raw solver countermodels may guide diagnostic retrieval but cannot eliminate a hypothesis or influence admission; only CountermodelValidationReceipt with deterministic replay against the originating LogicIR semantics or a kernel proof of negation may narrow/reject; feed explicit residuals back to Tactician; preserve every original goal/facet and prove that subgoal conjunction refines it; never weaken/delete a goal, promote a model decomposition, or add an unauthorized premise; state identity is monotonic; repeated state and cycles terminate; enforce maximum rounds/goals/subgoals/premises/counterexamples/wall/CPU/memory/context; cancellation/timeout/bound exhaustion returns inconclusive with residual gaps; deterministic replay is identity-equivalent.
- Embedding query: counterexample guided inductive synthesis tactician hammer refine program logic bounded

## LPR-014 Admit only reconstructed and unique logic predictions

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prediction-admission
- Depends on: LPR-012, LPR-013
- Goal id: LPR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/logic_prediction_admission.py, test/api/test_agent_supervisor_logic_prediction_admission.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_logic_prediction_admission.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/prediction-admission
- Parallel lane: lpr-admission
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/logic_prediction_admission.py, test/api/test_agent_supervisor_logic_prediction_admission.py
- AST symbols: LogicPredictionAdmission, LogicPredictionDecision, LogicPredictionRejectionReason
- Interfaces: LogicPredictionReceipt@1, HammerCoordinationReceipt, CountermodelValidationReceipt@1, LogicRefinementReceipt, ProofReceipt, RequiredBehaviorContract@1
- Allow concurrent with:
- Conflict policy: Own the prediction admission module/test; do not choose repair paths or mutate existing behavior/value records.
- Preconditions: Final refinement states include every goal and exact current Hammer receipts.
- Effects: A derived behavior clause, value mapping, construction route or placement becomes eligible for existing RPR synthesis only when independently authoritative and reconstructed.
- Evidence subset: source authority, exact roots, consistency, reconstruction, counterexamples, uniqueness, residuals, invalidators
- Acceptance: Recompute all identities and require authoritative independent premises, matching translation/native-goal/environment and native kernel acceptance; disallow ex-falso derivation when bounded consistency is invalid or unknown; preserve assumptions/unsupported facets; reject only an independently validated countermodel/proof of negation, never a raw solver claim; reject residual mandatory gap, stale state, solver-only proof/refutation, learned/vector/model authority or higher-precedence contract conflict; automatic value/construction/placement requires exactly one eligible consequence under deterministic tie rules; zero/multiple candidates abstain; emitted receipt contains no write authority.
- Embedding query: admit reconstructed program logic prediction unique value behavior placement abstain

## LPR-015 Bridge predictions into existing behavior and value synthesis

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rpr-bridge
- Depends on: LPR-014
- Goal id: LPR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/tactician_guided_behavior_synthesis.py, test/api/test_agent_supervisor_tactician_guided_behavior_synthesis.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_guided_behavior_synthesis.py test/api/test_agent_supervisor_required_behavior_synthesis.py test/api/test_agent_supervisor_missing_input_synthesis.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/rpr-bridge
- Parallel lane: lpr-rpr-bridge
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/tactician_guided_behavior_synthesis.py, test/api/test_agent_supervisor_tactician_guided_behavior_synthesis.py
- AST symbols: TacticianGuidedBehaviorSynthesizer, TacticianBehaviorSynthesisReceipt, ContractRepairPredictionBridge, PredictionEvidenceBinding
- Interfaces: ContractRepairProver, CandidateProofBundle, ContractRepairReranker, RequiredBehaviorSynthesizer, MissingInputSynthesizer, ChangePropagationObligationCompiler, LogicPredictionReceipt@1
- Allow concurrent with:
- Conflict policy: Own the composition adapter/test; do not fork or weaken RequiredBehaviorContract, MissingInputSynthesisReceipt, source precedence, or proof obligations.
- Preconditions: Logic prediction admission emits current reconstructed, source-traceable consequences.
- Effects: Candidate-specific substitution/equivalence/placement predictions project into existing CandidateProofBundle, while proven implied behavior/value consequences close existing RPR propagation gaps and preserve source precedence and unsupported facets.
- Evidence subset: prediction/source/proof bindings, behavior clause families, value mappings, conflicts, residual gaps, existing receipt composition
- Acceptance: Map candidate-specific reconstructed substitution/equivalence/placement results into the exact CandidateProofBundle/ContractRepairProver evidence consumed by ContractRepairReranker; compose propagation consequences with existing RequiredBehaviorSynthesizer, ChangePropagationObligationCompiler and MissingInputSynthesizer; proof status is orthogonal to source authority and a consequence inherits the weakest effective precedence of its independent premises rather than inventing a new closed-enum rank; never overwrite explicit conflict, higher-precedence source, unsupported memory/lifetime/native/concurrency facet or consumer-specific requirement; one admitted consequence maps to exact existing clause/value/placement references; stale/ambiguous proof remains a nomination; outputs are canonical and accepted unchanged by existing repair target and propagation plan admission.
- Embedding query: tactician guided required behavior synthesis missing input proof bridge existing rpr

## LPR-016 Materialize analytical-first, context-rich repair packets

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: logic-repair-packets
- Depends on: LPR-015
- Goal id: LPR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/logic_guided_repair_packet.py, ipfs_accelerate_py/agent_supervisor/context/logic_repair_context.py, test/api/test_agent_supervisor_logic_guided_repair_packet.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_logic_guided_repair_packet.py test/api/test_agent_supervisor_change_propagation_provider_router.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/packets
- Parallel lane: lpr-packets
- Resource class: cpu-medium
- Resource stage: materialize
- Token class: large
- Estimated tokens: 32000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/logic_guided_repair_packet.py, ipfs_accelerate_py/agent_supervisor/context/logic_repair_context.py, test/api/test_agent_supervisor_logic_guided_repair_packet.py
- AST symbols: LogicGuidedRepairPacketMaterializer, LogicRepairContextBuilder, LogicRepairExpansionHandle
- Interfaces: canonical LogicGuidedRepairPacket@1 from LPR-001, ChangePropagationEditPacket@1, ContractRepairEditPacket@2, ContextCapsule, AnalyticalChangeTransformer, ChangePropagationProviderRouter, WriterLease
- Allow concurrent with:
- Conflict policy: Own new context/materialization adapters and test; import/materialize/return the LPR-001 canonical LogicGuidedRepairPacket without redefining it, project it through existing RPR packets/provider routers, and never create a third write-authority packet.
- Preconditions: Existing RPR plan admission has selected exact steps/paths and the Tactician-guided behavior bridge is complete.
- Effects: Unique analytical repairs bypass models; behavior-complete syntax/implementation gaps receive a small exact provider capsule with high-value static/proof context.
- Evidence subset: objective/delta/consumer/SCC, behavior/value predictions, proof/counterexample refs, exact scope/hashes/lease, forbidden changes, postconditions, validations, expansion budgets
- Acceptance: Attempt deterministic AnalyticalChangeTransformer first and prove that success invokes no provider; exact target/atomic plan admission precedes all packet/provider work; the LPR context overlay binds an existing ChangePropagationEditPacket@1 or ContractRepairEditPacket@2 plus exact plan/step/SCC, current prediction receipts, chosen values/construction, admitted behavior, validated countermodels, read/write spans and before hashes, forbidden paths/semantics, type/effect/resource/test/fixed-point validations, provider/model/config and writer lease; existing router type checks and packet write authority remain intact; source/comments/issues are delimited as untrusted data and secrets/bodies are redacted; expansion handles are typed/bounded; the model cannot choose meaning, source, owner, dependency, caller set, target or path; malformed/refused/timeout/scope-escape proposal creates no write.
- Embedding query: analytical first logic guided repair packet llm router context counterexamples exact paths

## LPR-017 Integrate a live logic-repair controller into the RPR pipelines

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: live-integration
- Depends on: LPR-016
- Goal id: LPR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/live_logic_repair_controller.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_pre_provider_gate.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_live_logic_repair_integration.py, test/api/test_agent_supervisor_logic_repair_proposal_overlay.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_live_logic_repair_integration.py test/api/test_agent_supervisor_logic_repair_proposal_overlay.py test/api/test_agent_supervisor_contract_repair_integration.py test/api/test_agent_supervisor_change_propagation_integration.py test/api/test_agent_supervisor_change_propagation_pre_provider_gate.py test/api/test_agent_supervisor_proposal_validation.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/live-integration
- Parallel lane: lpr-integration-serialized
- Resource class: cpu-large
- Resource stage: materialize
- Token class: large
- Estimated tokens: 42000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/live_logic_repair_controller.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/objectives/contract_mismatch_refinery.py, ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_pre_provider_gate.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_live_logic_repair_integration.py, test/api/test_agent_supervisor_logic_repair_proposal_overlay.py
- AST symbols: LiveLogicRepairController, LiveLogicRepairRequest, LiveLogicRepairResult, CandidateOverlayContractDeltaGate, AnalysisPipeline, ChangePropagationPipeline, ProposalValidator, PortalImplementationDaemon
- Interfaces: completed RPR-047 pipeline interfaces, ProgramLogicGoalCompiler, IpfsDatasetsTacticianProvider, TacticianHammerCoordinator, LogicPredictionAdmission, LogicGuidedRepairPacket
- Allow concurrent with:
- Conflict policy: This is the single serialized shared-pipeline and provider-result-intake cutover; the edge controller lives in todo_daemon and injects pure analysis/proof/planning/validation callbacks rather than reversing the package DAG; all new-file adapters must be complete first, existing @1/@2 paths remain compatible/default-off, and no planning/control artifact is edited.
- Preconditions: Completed RPR-047 interfaces are capability-admitted; LPR contracts through packet generation are stable; shared files have no concurrent owner.
- Effects: Explicit repair requests and ordinary implementation-provider diffs invoke the full static facts-to-admitted-plan chain; callable changes are analyzed in a read-only overlay before any source-tree mutation.
- Evidence subset: stage order, exact roots, trace/change mode, all-consumer closure, goal/corpus/tactic/proof/prediction, analytical/provider route, pre-provider freshness, legacy compatibility
- Acceptance: Add a versioned feature-gated edge path that invokes trace/contracts/retrieval or delta/graph/impact/consumer/value/behavior analysis, then goal/corpus/Tactician/hypothesis/gate/lowering/Hammer/refinement/admission before existing target/atomic-plan admission; bridge contract predictions into CandidateProofBundle; intercept every ordinary provider proposal before patch application, materialize it as a read-only candidate overlay, compute base-to-proposal callable contract delta and complete impact/consumer frontier, and reject/defer or expand/re-admit any signature change whose existing write set omits a caller; include an exact generic-LLM f(a,b)->f(a,b,c) regression; every resolved caller is dispositioned before mutation and required unknown frontier abstains; analytical success makes no model call; model calls use only context overlays projected through LPR-016 existing packets; pre-provider/proposal gates revalidate all new roots/receipts and existing scope/lease; no direct write bypass exists; imports remain lazy/cold; old artifact-supplied RPR and generic proposal flows remain unchanged when the flag is off.
- Embedding query: live end to end contract repair change propagation tactician hammer all callers integration

## LPR-018 Re-plan and re-prove logic to a post-edit fixed point

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: fixed-point-validation
- Depends on: LPR-017
- Goal id: LPR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/logic_repair_fixed_point.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py, ipfs_accelerate_py/agent_supervisor/validation/contract_repair_validation.py, ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py, ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_logic_repair_fixed_point.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_logic_repair_fixed_point.py test/api/test_agent_supervisor_change_propagation_validation.py test/api/test_agent_supervisor_contract_repair_validation.py test/api/test_agent_supervisor_change_propagation_transaction.py test/api/test_agent_supervisor_change_propagation_integration.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/fixed-point
- Parallel lane: lpr-validation-serialized
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 40000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/logic_repair_fixed_point.py, ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py, ipfs_accelerate_py/agent_supervisor/validation/contract_repair_validation.py, ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py, ipfs_accelerate_py/agent_supervisor/analysis/change_propagation_pipeline.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_logic_repair_fixed_point.py
- AST symbols: LogicRepairFixedPointValidator, LogicRepairIterationReceipt, LogicFixedPointEvidenceAttachment, PropagationFinalizeReceipt, CompensatingRollbackReceipt
- Interfaces: ChangePropagationValidator, ContractRepairValidator, ChangePropagationTransaction, PropagationCompletionReceipt@1, canonical LogicFixedPointEvidenceAttachment@1 from LPR-001
- Allow concurrent with:
- Conflict policy: Serialized post-integration validation/pipeline cutover; import/materialize the LPR-001 logic evidence attachment instead of defining a new completion receipt, reuse existing checkpoint/SCC rollback and completion contracts, and do not weaken any legacy fixed-point condition.
- Preconditions: Live pipeline emits exact plan/prediction/packet/transaction bindings and existing RPR transaction/validator suites pass.
- Effects: The actual change-propagation and broken-contract completion paths rebuild program and logic artifacts each round; provisional transaction commits are finalized only at a joint fixed point and otherwise receive a compensating rollback.
- Evidence subset: checkpoint, rebuilt indexes/graphs/corpus, redelta/reclosure, regenerated goals/tactic plans, renewed reconstructions, second-order impacts, static/tool/test gates, rollback
- Acceptance: Wire LogicRepairFixedPointValidator into the actual ChangePropagationPipeline/daemon completion wrapper and either route broken-contract repair through the atomic propagation transaction after target admission or extend ContractRepairValidator equivalently; after each provisional transaction rebuild repository/AST/vector/KG/call/dependency/schema/value graphs and tombstones; recompute delta/closure/consumer ledger; regenerate corpus/goals/gaps/Tactician plan and Hammer/native-goal/countermodel receipts for changed or introduced clauses; revalidate every original and newly resolved caller, second-order change, chosen value/behavior/placement and policy tool; require no unresolved mandatory consumer, open required frontier, unplanned breaking delta, new required logic gap, stale prediction or failed validation; attach exact per-iteration logic evidence to the existing completion receipt; finalize only after success; bound exhaustion/incomplete/failure after provisional commit triggers compensating rollback to the checkpoint; partial SCC/packet completion can never close the task.
- Embedding query: post edit reindex replan tactician rehammer all callers logic program fixed point rollback

## LPR-019 Benchmark adversarial logic prediction and all-caller repair

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark
- Depends on: LPR-004, LPR-018
- Goal id: LPR-G060
- Outputs: test/api/test_agent_supervisor_tactician_hammer_logic_repair_benchmark.py, scripts/benchmark_tactician_hammer_logic_repair.py, data/agent_supervisor/tactician_hammer_logic_repair/benchmark/.gitkeep
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_logic_repair_benchmark.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/benchmark
- Parallel lane: lpr-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: test/api/test_agent_supervisor_tactician_hammer_logic_repair_benchmark.py, scripts/benchmark_tactician_hammer_logic_repair.py, data/agent_supervisor/tactician_hammer_logic_repair/benchmark/.gitkeep
- AST symbols: LogicRepairBenchmark, LogicRepairBenchmarkMetrics, LogicRepairFailureStage
- Interfaces: LogicRepairFixtureManifest, LogicPredictionReceipt@1, CountermodelValidationReceipt@1, PropagationCompletionReceipt@1, LogicFixedPointEvidenceAttachment@1
- Allow concurrent with:
- Conflict policy: Own the benchmark runner/test/output placeholder; runtime reports stay untracked unless separately reviewed.
- Preconditions: The adversarial corpus and live atomic fixed-point pipeline are stable.
- Effects: Release policy receives stage-specific safety, efficacy, cost and abstention evidence rather than aggregate pass/fail.
- Evidence subset: goal/premise recall, Tactician plan, hypothesis, lowering, solver, reconstruction, prediction, analytical/model edit, caller closure, transaction, fixed point, resource cost
- Acceptance: Deterministically run every fixture twice with exact roots and identity-equivalent receipts; distinguish static/impact/goal/corpus/Tactician/retrieval/lowering/solver/raw-countermodel/countermodel-validation/native-goal/reconstruction/admission/analytical/provider/transaction/fixed-point failures; enforce exactly zero missed resolved caller, unreconstructed or raw-countermodel admission, unauthorized axiom, invented behavior, wrong value/source/placement, stale root/corpus/receipt, failed-obligation override, LLM scope/semantic escape, partial transaction and false completion; include ordinary generic-provider signature-change overlay and explicit LPR cases; report goal/subgoal and hypothesis precision/recall, premise recall@k, first-plan closure, lowering/reconstruction/validated-countermodel/abstention/analytical/model/all-caller rates, platform enforcement, iterations, p50/p95 time/CPU/memory/context/tokens and cache/invalidation accuracy without making metrics authority.
- Embedding query: benchmark program logic prediction tactician hammer all caller repair safety efficacy cost

## LPR-020 Add rollout controls, operations, validation, and supervisor launch

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rollout-operations
- Depends on: LPR-019
- Goal id: LPR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py, scripts/validate_tactician_hammer_logic_repair.py, docs/guides/TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md, test/api/test_agent_supervisor_tactician_hammer_logic_repair_end_to_end.py, test/api/test_agent_supervisor_tactician_hammer_logic_repair_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_tactician_hammer_logic_repair_rollout.py test/api/test_agent_supervisor_tactician_hammer_logic_repair_end_to_end.py && python scripts/validate_tactician_hammer_logic_repair.py --check-all
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/rollout
- Parallel lane: lpr-rollout
- Resource class: cpu-large
- Resource stage: rollout
- Token class: large
- Estimated tokens: 38000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py, scripts/validate_tactician_hammer_logic_repair.py, docs/guides/TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md, test/api/test_agent_supervisor_tactician_hammer_logic_repair_end_to_end.py, test/api/test_agent_supervisor_tactician_hammer_logic_repair_rollout.py
- AST symbols: LogicRepairRolloutPolicy, LogicRepairMetrics, LogicRepairRollbackGate, LogicRepairOperationsValidator
- Interfaces: LogicRepairBenchmarkMetrics, LiveLogicRepairController, PropagationCompletionReceipt@1, LogicFixedPointEvidenceAttachment@1, SupervisorControlService, LPR-000 protected bootstrap
- Allow concurrent with:
- Conflict policy: Own terminal LPR rollout/release-validator/guide/tests; consume but do not edit the LPR-000 protected plan/objective/taskboard/scheduler/launcher/board-validator, preserve existing RPR operations, and default to shadow with no autonomous model-authored change.
- Preconditions: Benchmark meets all zero safety floors; exact accelerator/datasets bindings and generic Tactician/Hammer capabilities are healthy.
- Effects: Operators can doctor, replay, inspect, start, status, restart, stop, shadow, assist, narrowly automate deterministic repairs, approval-gate model proposals, and roll back safely.
- Evidence subset: dependency/goal DAG, exact bindings, capabilities, provider/process health, metrics/floors, feature flags, lifecycle, rollback, end-to-end fixtures
- Acceptance: Release validator composes the protected bootstrap board/DAG doctor with exact two-repository gitlink/module/schema/tool/environment bindings, import-isolation and native-execution permits, platform resource/network isolation strength, capability health and benchmark floors; it verifies protected paths, strict four-lane sharding, isolated state/worktrees, one merge queue, bounded retries and one refill owner; the unchanged bootstrap launcher doctor/start/status/restart/stop remains idempotent and never kills unrelated processes or stores secrets in argv/logs; stages are doctor/replay, shadow default, assist, deterministic narrow-auto and approval-gated behavior-complete model edit; independent flags disable prediction/learned ranking/Hammer/refinement/LLM/auto; stateful/public-schema/API/dynamic/generated/native/cross-root/new-dependency work remains approval-required; any nonzero floor, drift, reconstruction/countermodel-validation loss, inconsistency, transaction, isolation or budget regression rolls back; seeded explicit and ordinary-proposal two-to-three-argument plus complex support-type cases update all resolved callers and reach an existing completion receipt with a current logic fixed-point attachment or abstain; a healthy isolated supervisor drains the board without dependency, provider, protected-path, merge or lifecycle blockage.
- Embedding query: logic repair rollout doctor replay shadow assist narrow auto supervisor launch rollback end to end

## LPR-021 Pin the VFS source baseline and generalize repository surface inventory

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-020
- Goal id: LPR-G070
- Outputs: config/agent_supervisor_vfs_generalization_sources.lock.json, docs/architecture/agent_supervisor/VFS_ASSURANCE_GENERALIZATION_MAP.md, ipfs_accelerate_py/agent_supervisor/analysis/repository_surface_inventory.py, test/api/test_agent_supervisor_repository_surface_inventory.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_repository_surface_inventory.py && python scripts/validate_tactician_hammer_logic_repair_board.py --check-all
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-inventory
- Parallel lane: lpr-general-inventory
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 36000
- Implementation timeout seconds: 7200
- Predicted files: config/agent_supervisor_vfs_generalization_sources.lock.json, docs/architecture/agent_supervisor/VFS_ASSURANCE_GENERALIZATION_MAP.md, ipfs_accelerate_py/agent_supervisor/analysis/repository_surface_inventory.py, test/api/test_agent_supervisor_repository_surface_inventory.py
- AST symbols: SurfaceInventoryPolicy, SurfaceSignal, SurfaceKindSpec, Definition, SurfaceEvidence, SurfaceContradiction, InventoryDiagnostic, SurfaceRecord, InventoryCompleteness, RepositorySurfaceInventory, discover_surface_paths, inventory_repository_surfaces, assert_inventory_complete, publish_surface_inventory
- Interfaces: ProgramGraph, RepositorySnapshot, RepositoryForest, canonical content identity, source-lock manifest
- Allow concurrent with:
- Conflict policy: Read only the exact source-lock Git blobs and planned VFS board contracts; never merge or cherry-pick the broad source revision, copy a root vfs_* module, import scanned code, or embed repository bodies in the lock/map.
- Preconditions: The complete LPR release is stable; Git commit 0cc04ebb640c4c981cf4650016e096a73ab0e8c0 is reachable from the namespaced source-lock ref or exact origin object fetch, and its seven declared module/test blobs are available.
- Effects: Workers receive a content-addressed old-to-new API/schema/caller map and a reusable static repository-surface inventory whose signals, suffixes, taxonomy, scan roots, classification policy, bounds and schema identity are supplied by a profile.
- Evidence subset: exact Git revision and blob identities, seven public exports, schemas, entry points, callers/tests, planned paths, package ownership, non-VFS fixture
- Acceptance: Lock and verify source modules vfs_surface_inventory.py=76f34e1b9320e4bbc15706e4895c02af805af5e0, vfs_contract_pack.py=9acc4ceba42b8767f5b4e4b6ce7d4bc55893bcf2, vfs_differential_harness.py=8a6c8af69b6cbcb76a2b79a51f406d13e10947ce, vfs_mcp_contract_checker.py=26144a7b78c1bbbb94edc67ab13e2eab03850924, vfs_symbolic_benchmark.py=90023a09e9eb01ee454718f60fe758e33434c56b, vfs_symbolic_pilot.py=483ecaf622caa3c91d80d9710b63b1fd36fb8f90 and vfs_symbolic_rollout.py=6a1ef7b87172aa413f81b37f0ba36954af774d40 plus their test blobs; record target-present, source-only and planned-only states without conflation; inventory uses bounded byte/static-AST scans with no code import, reports completeness/unknowns and is deterministic under reordered inputs; VFS-equivalent profile results match the locked source contract while a non-VFS fixture proves parameterization; generic code contains no VFS/IPFS/fsspec/SwissKnife literals, fixed aliases, board IDs or implicit provider imports.
- Embedding query: pin vfs source blobs generalize static repository surface inventory profile classification

## LPR-022 Generalize the evidence-bound program contract profile

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-021
- Goal id: LPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_contract_profile.py, test/api/test_agent_supervisor_program_contract_profile.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_contract_profile.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-contracts
- Parallel lane: lpr-general-contracts
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 34000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/program_contract_profile.py, test/api/test_agent_supervisor_program_contract_profile.py
- AST symbols: ContractVocabulary, ExpectationState, OperationSupport, SourceContract, InvariantContract, OperationContract, SurfaceOperationContract, PublicSurfaceContract, ExpectationIssue, CanonicalVector, FacadeExample, ProgramContractProfile, ProgramContractProfileCompiler, assert_contract_profile_complete, publish_contract_profile
- Interfaces: ProgramContract@1, ProgramContractBundle, SenderRequirement, ReceiverGuarantee, RequiredBehaviorContract@1, source-precedence policy
- Allow concurrent with: LPR-025
- Conflict policy: Own the generic contract-profile module/test only; preserve existing program contract and source-authority types, and move all closed VFS operations, invariants, errors, surface names, defaults and goal IDs into later profile data rather than a parallel contract language.
- Preconditions: The source lock/API map and reusable inventory records are stable.
- Effects: Any bounded domain profile can compile evidence-bound operation, invariant, surface, vector, facade and unresolved-expectation records into existing program contracts.
- Evidence subset: vocabulary/profile schema, source authority, operation/invariant/interface bindings, error/effect/resource semantics, canonical vectors, conflicts, identity compatibility
- Acceptance: Compile immutable bounded content-identified vocabularies and profiles; require every expectation and canonical vector to cite independent source evidence and exact semantics; preserve unresolved/conflicting expectations instead of selecting the popular implementation; reject duplicate/unknown vocabulary entries, unbounded fields, self-authority, forged IDs and schema drift; no generic branch or default contains VFS/IPFS/fsspec/SwissKnife terms; an unrelated RPC/key-value profile exercises every generic record; the VFS profile projection can preserve the locked operation/invariant/schema identities without making module paths semantic identity.
- Embedding query: generic program behavior contract profile operations invariants surfaces vectors source authority

## LPR-023 Generalize the hermetic differential contract harness

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-022
- Goal id: LPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/differential_contract_harness.py, test/api/test_agent_supervisor_differential_contract_harness.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_differential_contract_harness.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-differential
- Parallel lane: lpr-general-differential
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 34000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/differential_contract_harness.py, test/api/test_agent_supervisor_differential_contract_harness.py
- AST symbols: ContractTraceProvider, FixtureAdapter, ResultNormalizer, ErrorClassifier, DriftClassifier, SurfaceAdapter, CallableSurfaceAdapter, CanonicalOperationTrace, DifferentialWitness, run_differential_contract_harness, write_differential_witness
- Interfaces: ProgramContractProfile, CanonicalVector, ExecutionPermit, RuntimeIdentity, CleanupReceipt
- Allow concurrent with: LPR-024, LPR-025, LPR-026
- Conflict policy: Own the generic validation harness/test; inject domain fixtures, normalizers, error/invariant maps and drift taxonomy, use temporary roots and bounded subprocesses, and never touch user state or claim that declared network denial is OS isolation.
- Preconditions: Generic contract-profile semantics and locked differential behavior are stable.
- Effects: Different implementations of any profiled interface can execute the same finite traces and emit reproducible typed observations, drift findings and cleanup receipts.
- Evidence subset: finite traces, adapters, normalization, error/effect identity, isolation, network/resource bounds, cleanup, unavailable/unknown states, replay
- Acceptance: Preserve deterministic canonical JSON/content IDs, sync/async adapter handling, exact implementation/runtime/toolchain identities, contract-approved normalization only, explicit unavailable/mock/unknown outcomes, network denial, temporary fixture isolation and cleanup; parameterize all VFS-specific tree fixtures, invariants, errors, schemas and drift types; reject path escape, adapter identity drift, silent success, incomplete cleanup, timeout and over-budget execution; VFS canonical vectors are equivalent and an in-memory non-VFS adapter detects seeded true drift without false mismatches.
- Embedding query: generic hermetic differential contract harness trace adapter normalization drift cleanup

## LPR-024 Generalize interface, manifest, SDK, MCP, and transport parity

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-022
- Goal id: LPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/interface_contract_parity.py, test/api/test_agent_supervisor_interface_contract_parity.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_interface_contract_parity.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-parity
- Parallel lane: lpr-general-parity
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 36000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/interface_contract_parity.py, test/api/test_agent_supervisor_interface_contract_parity.py
- AST symbols: ToolSelectionPolicy, ParitySurfaceSpec, ContractProfileAdapter, SurfaceView, ParityWitness, ParityFinding, ToolParityResult, InterfaceParityReport, InterfaceContractParityAnalyzer, discover_tool_names, build_surface_views, compare_tool_surfaces, check_interface_parity
- Interfaces: McpContractAnalyzer, MCPlusPlusCallPath, RuntimeWitnessReceipt, ProgramContractProfile, manifest and SDK schema projections
- Allow concurrent with: LPR-023, LPR-025, LPR-026
- Conflict policy: Reuse canonical MCP/MCP++ resolver, invocation-trace and runtime-witness records; this task owns only generic parity analysis and cannot create a second transport authority model or hard-code VFS tool aliases and SwissKnife paths.
- Preconditions: Generic contract-profile surface mappings are stable and the locked parity checker API is mapped.
- Effects: Python, CLI, MCP, MCP++, HTTP, libp2p, generated manifest and SDK surfaces can be compared against resolved implementation paths under a caller-provided profile.
- Evidence subset: signatures, registrations, schemas, aliases, errors, call paths, manifests, SDKs, runtime witnesses, mock/fallback/bypass, bounded minimal witnesses
- Acceptance: Resolve exact call paths before declaring parity; same text/name alone is insufficient; report stale generated artifacts, missing/extra/unreachable tools, schema/alias/error drift, local bypass, mock/fallback dispatch and ambiguous paths with minimal witnesses; inject surface kinds, tool selection/alias mapping and contract-pack projection; reject unbounded or forged artifacts and keep unresolved paths explicit; generic core contains no vfs.* aliases, VFS operations, fixed connector names or source paths; math.add and a generic connector pass while a seeded unresolved or wrong-target path fails.
- Embedding query: generic interface contract parity manifest sdk mcp transport resolved call path

## LPR-025 Generalize symbolic-efficiency benchmark and replay gates

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-021
- Goal id: LPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/symbolic_efficiency_benchmark.py, test/api/test_agent_supervisor_symbolic_efficiency_benchmark.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_symbolic_efficiency_benchmark.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-benchmark
- Parallel lane: lpr-general-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/symbolic_efficiency_benchmark.py, test/api/test_agent_supervisor_symbolic_efficiency_benchmark.py
- AST symbols: SymbolicBenchmarkPolicy, BenchmarkProfile, SymbolicBenchmarkObservation, SymbolicBenchmarkPopulation, BenchmarkGate, SymbolicEfficiencyBenchmarkReport, evaluate_symbolic_efficiency, build_symbolic_efficiency_report, verify_symbolic_efficiency_report
- Interfaces: repository/AST/graph/contract/proof cache metrics, finding/task metrics, provider packet metrics, ResourceMeasurement
- Allow concurrent with: LPR-022
- Conflict policy: Own the generic benchmark/test; make required stages, scan modes, cache/finding populations, schemas and thresholds policy/profile data, retain reports as non-authoritative measurements, and perform no provider/network call.
- Preconditions: The locked benchmark schemas, gates and sample identities are mapped by LPR-021.
- Effects: Any assurance profile can measure cold/warm/delta behavior, reuse, invalidation, finding quality, packet reduction and bounded resources through the same deterministic evaluator.
- Evidence subset: frozen profile/population, sample sufficiency, stage and cache coverage, seeded truth, time/resources, packet bytes/tokens, deterministic replay, non-authority
- Acceptance: Preserve finite canonical records, rational/integer arithmetic, identity-equivalent replay, sample-sufficiency gates, no promotion/completion authority and zero LLM calls in deterministic stages; inject every VFS schema/stage/mode/threshold rather than rename it in place; byte-equivalent locked VFS observations yield equivalent decisions; an unrelated profile exercises custom stages and bounds; malformed, stale, mixed-profile, under-sampled, resource-exceeding and forged-report inputs fail closed.
- Embedding query: generic symbolic efficiency benchmark deterministic replay profile cache finding resources

## LPR-026 Generalize frozen multi-repository assurance pilot orchestration

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-022
- Goal id: LPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/symbolic_assurance_pilot.py, test/api/test_agent_supervisor_symbolic_assurance_pilot.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_symbolic_assurance_pilot.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-pilot
- Parallel lane: lpr-general-pilot
- Resource class: cpu-large
- Resource stage: runtime
- Token class: large
- Estimated tokens: 38000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/symbolic_assurance_pilot.py, test/api/test_agent_supervisor_symbolic_assurance_pilot.py
- AST symbols: PilotProgramProfile, RepositoryAdmissionPolicy, PilotConfig, PilotStage, StageReceipt, PilotArtifactSet, SymbolicAssurancePilotReport, execute_pilot, verify_pilot_report, verify_pilot, render_findings_board_document
- Interfaces: RepositoryForest, RepositorySurfaceInventory, ProgramGraph, ProgramContractProfile, proof/cache/finding/taskboard stage callbacks
- Allow concurrent with: LPR-023, LPR-024, LPR-025
- Conflict policy: Own the generic runtime pilot/test; inject repositories, path admission, stages, schemas, task/goal namespace and artifact locations, expose no CLI, and perform zero provider calls or source mutation during dry-run/verify.
- Preconditions: Generic inventory and contract-profile interfaces plus the locked pilot stage contract are stable.
- Effects: A bounded multi-repository assurance program can freeze roots, run injected deterministic stages, publish artifacts/findings, and verify replay without knowing a product, checkout alias, board or provider.
- Evidence subset: repository bindings, admitted closure, stage DAG/receipts, artifact identities, findings board, dry-run/verify, provider absence, source immutability, rollback
- Acceptance: PilotConfig is tuple/profile-driven rather than fixed accelerator/SwissKnife/kit/datasets fields; inject forest builder, path admission, stage runners, schema/goal/task/board identities and artifact destinations; generic code has no VFS regex, repository aliases, environment-variable names, fixture construction or argparse/main; preserve atomic bounded writes, exact repository descriptors, stage failure/abstention, deterministic report verification and no provider surface; reject changed trees, incomplete inventory, stale/corrupt artifacts, duplicate stages, unsafe output paths, provider access and source mutation; a non-VFS two-repository fixture completes through the same orchestrator.
- Embedding query: generic frozen multi repository symbolic assurance pilot profile stage receipts verify

## LPR-027 Generalize rollout control and add the thin IPFS Kit VFS job adapter

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-023, LPR-024, LPR-025, LPR-026
- Goal id: LPR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/control/symbolic_assurance_rollout.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_kit_vfs_assurance.py, config/ipfs_kit_vfs_symbolic_assurance.json, scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py, test/api/test_agent_supervisor_symbolic_assurance_rollout.py, test/api/test_agent_supervisor_ipfs_kit_vfs_assurance_profile.py, test/api/test_agent_supervisor_ipfs_kit_vfs_assurance_ops.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_symbolic_assurance_rollout.py test/api/test_agent_supervisor_ipfs_kit_vfs_assurance_profile.py test/api/test_agent_supervisor_ipfs_kit_vfs_assurance_ops.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-vfs-adapter
- Parallel lane: lpr-vfs-adapter
- Resource class: cpu-large
- Resource stage: integration
- Token class: large
- Estimated tokens: 44000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/symbolic_assurance_rollout.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_kit_vfs_assurance.py, config/ipfs_kit_vfs_symbolic_assurance.json, scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py, test/api/test_agent_supervisor_symbolic_assurance_rollout.py, test/api/test_agent_supervisor_ipfs_kit_vfs_assurance_profile.py, test/api/test_agent_supervisor_ipfs_kit_vfs_assurance_ops.py
- AST symbols: AssuranceRolloutMode, ControlAction, ControlSurface, GateDefinition, GateObservation, AdversarialGateReport, AssuranceRolloutBinding, AssuranceRolloutPolicy, AssuranceRolloutDecision, SymbolicAssurancePublicAPI, evaluate_adversarial_gates, evaluate_symbolic_assurance_rollout, build_ipfs_kit_vfs_assurance_profile
- Interfaces: generic inventory/contract/differential/parity/benchmark/pilot engines, SupervisorControlService, lazy integration registry, bounded config profile, ops main(argv)
- Allow concurrent with:
- Conflict policy: Generic control owns rollout decisions; integrations owns only lazy VFS factories; config owns declarative VFS vocabulary/policy; the ops script only validates/selects/delegates and may not contain scanning, graph, proof, parity, benchmark, rollout, repair, provider or mutation logic.
- Preconditions: All generic VFS-source-derived engines are stable and their locked semantic contracts are mapped.
- Effects: The original IPFS Kit VFS assurance job is one typed profile over reusable engines, operable through a cold-import-safe scripts/ops entry point with inventory, contracts, differential, parity, benchmark, pilot, rollout and verify subcommands.
- Evidence subset: rollout modes/gates, shadow and mutation authority, bounded projections, VFS factories/config, lazy import, CLI behavior, original schemas/receipts, non-VFS profile
- Acceptance: Rollout gates/modes/control surfaces are profile definitions with shadow default and automatic mutation disabled; preserve original VFS schemas, operation/invariant/error mappings, canonical vectors, authority flags, status/findings/receipts projections and CLI exit semantics through the locked profile; config is immutable/bounded/content-identified with a closed adapter registry and safe relative roots; integration imports optional providers lazily; wrapper --help and cold import start no process, open no database, access no network/storage and import no optional provider; wrapper contains only argument/config/bootstrap/delegation code; generic control contains no VFS/IPFS/fsspec/SwissKnife constants; a non-VFS profile traverses the same public API.
- Embedding query: generic assurance rollout thin scripts ops ipfs kit vfs typed lazy profile

## LPR-028 Prove equivalence, propagate every caller, and enforce the root-layout cutover

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: assurance-generalization
- Depends on: LPR-027
- Goal id: LPR-G070
- Outputs: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, scripts/ops/agent_supervisor/README.md, test/api/test_agent_supervisor_vfs_generalization_equivalence.py, test/api/test_agent_supervisor_vfs_root_layout_guard.py, test/api/test_agent_supervisor_assurance_two_profile_end_to_end.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_vfs_generalization_equivalence.py test/api/test_agent_supervisor_vfs_root_layout_guard.py test/api/test_agent_supervisor_assurance_two_profile_end_to_end.py && python scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py && python scripts/validate_tactician_hammer_logic_repair_board.py --check-all
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/generalization-cutover
- Parallel lane: lpr-general-cutover
- Resource class: cpu-large
- Resource stage: rollout
- Token class: large
- Estimated tokens: 46000
- Implementation timeout seconds: 10800
- Predicted files: docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md, docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md, scripts/ops/agent_supervisor/README.md, test/api/test_agent_supervisor_vfs_generalization_equivalence.py, test/api/test_agent_supervisor_vfs_root_layout_guard.py, test/api/test_agent_supervisor_assurance_two_profile_end_to_end.py
- AST symbols: VfsGeneralizationEquivalenceReceipt, VfsCallerMigrationReceipt, VfsRootLayoutGuard, AssuranceTwoProfileConformance
- Interfaces: ProgramContractDelta@1, ProgramDependencyGraph, ConsumerMigrationObligation@1, TacticianSearchPlan@1, LogicPredictionReceipt@1, ChangePropagationPlan@1, PropagationCompletionReceipt@1
- Allow concurrent with:
- Conflict policy: Serialize the final migration; confirm the dedicated VFS supervisor has no live owned process before editing its plan/objective/taskboard, preserve completed VFS task history, update only still-open output/caller projections, and never add root compatibility shims or silently delete a public contract.
- Preconditions: The generic engines, VFS profile/adapter and thin ops entry point pass their local suites; the exact source lock and current target tree are fresh.
- Effects: Every realized or planned VFS root implementation/import is redirected to semantic domain packages plus the VFS profile/ops adapter, with proof-bearing caller dispositions and a future-proof placement gate.
- Evidence subset: base/source/generic contract delta, AST/import/string-import/entry-point impact closure, Tactician plan, Hammer reconstruction or abstention, public schema/identity parity, migrated tests/docs/boards, non-VFS reuse, root scan, fixed point
- Acceptance: Compute the locked-source-to-generic ProgramContractDelta and complete resolved impact closure before edits; use Tactician/Hammer to prove delegation/profile equivalence for supported clauses and retain explicit unsupported/approval-required dispositions otherwise; atomically migrate every import, string import, export, entry point, test, documentation link and open VFS-board output; preserve canonical VFS vectors, errors, side effects, schemas, CIDs, receipts and CLI results or record reviewed versioned migration; run the migrated locked VFS corpus and a hermetic non-VFS inventory-to-rollout fixture through identical generic engine modules; assert recursively that the agent_supervisor root contains no vfs_*.py file and no import references agent_supervisor.vfs_*; reject VFS/IPFS/fsspec/SwissKnife/board-specific branches in generic modules and substantive logic in the ops wrapper; cold import remains side-effect free; re-index, re-resolve and re-prove to an existing completion receipt plus current logic fixed-point attachment; any missed caller, stale blob, unresolved required frontier, unproved semantic drift, layout regression or partial migration rolls back or abstains.
- Embedding query: prove vfs generalization equivalence migrate all callers no root modules two profile fixed point

## LPR-029 Define deterministic-doctor contracts, policy, and receipts

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-foundations
- Depends on: LPR-020
- Goal id: LPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_policy.py, test/api/test_agent_supervisor_deterministic_doctor_contracts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_contracts.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-contracts
- Parallel lane: lpr-doctor-contracts
- Resource class: cpu-small
- Resource stage: analysis
- Token class: large
- Estimated tokens: 30000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_policy.py, test/api/test_agent_supervisor_deterministic_doctor_contracts.py
- AST symbols: DoctorAuthorityRoots, DoctorEvidenceSnapshot, DeterministicDoctorFinding, DoctorRepairDisposition, DoctorRepairOperatorSpec, DeterministicDoctorPlan, DoctorProofCacheAuditReceipt, DeterministicDoctorRunReceipt, DeterministicDoctorPolicy
- Interfaces: RepositorySnapshot, RepositoryForest, ContentIdentityBridge@1, ProgramLogicGoal@1, LogicPredictionReceipt@1, ChangePropagationPlan@1, PropagationCompletionReceipt@1
- Allow concurrent with: LPR-021, LPR-030, LPR-031, LPR-032
- Conflict policy: Own only new deterministic-doctor contracts/policy/tests; reuse canonical identity, authority, proof, packet, transaction, and completion records by reference, leave the lifecycle doctor and ProgrammaticRecoveryController read-only with respect to source semantics, and define no model route.
- Preconditions: LPR release contracts, canonical CIDs, exact authority roots, proof receipts, edit plans, leases, transactions, and completion attachments are stable.
- Effects: Every doctor stage exchanges immutable bounded body-free content-addressed records with closed dispositions, invalidators, resource bounds, explicit authority, and a hard zero-LLM/model-provider-call invariant.
- Evidence subset: modes, authority roots, canonical preimages/CIDs, states, bounds, trusted base, approval classes, invalidation, no-model invariant
- Acceptance: Define inspect/explain/plan/repair/replay/rollback with inspect/explain/plan read-only and report-only default; bind forest/tree/overlay/file/AST/graph/corpus/index/model/cache/operator/translator/solver/kernel/toolchain/policy/sandbox/environment/lease roots; distinguish observed facts from expected behavior and nomination from proof/write authority; enumerate supported, abstain, approval-required, rolled-back and quarantined states; reject forged CIDs, bodies/secrets, cycles, unbounded data, partial plans, open required frontiers, semantic authority on KG/vector/embedding/Tactician/Hammer candidates/cache metadata, LLM or remote model-provider invocation in deterministic mode, and repair without an existing admitted plan/lease/checkpoint/rollback; explicitly protect doctor/proof/identity/transaction trusted-base paths and public/native/stateful/cross-root/new-dependency classes.
- Embedding query: deterministic doctor contracts policy receipts cid no llm report only repair authority

## LPR-030 Compile a real-checkout AST and contract diagnostic snapshot

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-foundations
- Depends on: LPR-020
- Goal id: LPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/doctor_repository_diagnostics.py, test/api/test_agent_supervisor_doctor_repository_diagnostics.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_doctor_repository_diagnostics.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-diagnostics
- Parallel lane: lpr-doctor-diagnostics
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 44000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/doctor_repository_diagnostics.py, test/api/test_agent_supervisor_doctor_repository_diagnostics.py
- AST symbols: DoctorSnapshotPolicy, DoctorDiagnosticInput, DoctorDiagnosticFinding, DoctorEvidenceCompiler, compile_doctor_evidence_snapshot, diagnose_repository
- Interfaces: build_repository_snapshot, RepositoryIndexer, build_analysis_ast_index, adapt_program_source, ProgramGraph, ProgramDependencyGraph, ProgramCallResolver, BrokenContractTrace@1, ProgramContractDelta@1, SenderReceiverContractCompiler, ChangeConsumerInventory, ValueProvenanceCompiler
- Allow concurrent with: LPR-021, LPR-029, LPR-031, LPR-032
- Conflict policy: Compose existing snapshot/corpus/AST/graph/trace/contract/value interfaces without importing or executing target code, inventing expected behavior, or treating parser/analyzer diagnostics as repair authority.
- Preconditions: LPR-020 exact source/capability bindings and the reusable RPR static-analysis interfaces are current.
- Effects: A bounded real checkout becomes a frozen evidence snapshot plus typed syntax/import/name/call-arity/type/contract/value/dataflow/error/effect/resource/state/schema/memory findings, consumer candidates and explicit unknown frontiers.
- Evidence subset: forest/tree/overlay/file roots, parsers, AST symbols/imports/calls/references, contract delta, reaching values, effects/resources/memory, diagnostics, completeness
- Acceptance: Parse Python plus every already-supported adapter language as inert bytes; bind parser/config/toolchain and all derived index roots; deterministically join broken traces and structured validation failures to current AST facts; preserve expectation source/precedence separately from observations; query imports, exports, aliases, wrappers, entry points and call sites; expose Python-only or unsupported CFG/reflection/exception/native/FFI/concurrency/interprocedural limitations as open frontiers; issue canonical finding CIDs; incremental invalidation and a clean rebuild are identity-equivalent; malformed, oversized, symlink-escaped, stale or mixed-root input fails closed with zero source writes and zero provider calls.
- Embedding query: real checkout deterministic doctor repository ast diagnostics broken contracts callers value provenance

## LPR-031 Retrieve refactor and value candidates through exact graphs and advisory KG/vector embeddings

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-foundations
- Depends on: LPR-020
- Goal id: LPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/doctor_repair_candidate_retrieval.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_embedding_provider.py, test/api/test_agent_supervisor_doctor_repair_candidate_retrieval.py, test/api/test_agent_supervisor_ipfs_datasets_embedding_provider.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_doctor_repair_candidate_retrieval.py test/api/test_agent_supervisor_ipfs_datasets_embedding_provider.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-retrieval
- Parallel lane: lpr-doctor-retrieval
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 42000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/doctor_repair_candidate_retrieval.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_embedding_provider.py, test/api/test_agent_supervisor_doctor_repair_candidate_retrieval.py, test/api/test_agent_supervisor_ipfs_datasets_embedding_provider.py
- AST symbols: DoctorCandidateQuery, DoctorCandidateEvidence, DoctorCandidateSet, DoctorRepairCandidateRetriever, DatasetsEmbeddingCapability, PinnedEmbeddingPolicy, IpfsDatasetsEmbeddingProvider
- Interfaces: CodeEvidenceGraph, SemanticDependencyGraph, SymbolicContractGraph, ExactDatasetsGraphRAGAdapter, build_code_symbol_vector_index, build_change_value_vector_index, retrieve_proof_directed, ContractRepairCandidateRetriever, MissingInputCandidateRetriever
- Allow concurrent with: LPR-021, LPR-029, LPR-030, LPR-032
- Conflict policy: Exact static/history routes precede GraphRAG/vector routes; all KG/history/vector/embedding outputs are bounded nominations with semantic_authority=false, no remote fallback, no target/write selection, and no optional eager import.
- Preconditions: Content-addressed code-symbol/value indexes, semantic/evidence graphs, exact datasets GraphRAG capability probes, and proof-directed retrieval interfaces are importable.
- Effects: Findings can nominate renamed/moved definitions, exact structural equivalents, constructors/factories/adapters, reaching values and analogous repairs with complete query/result provenance and rejection reasons.
- Evidence subset: AST/graph/history exact match, KG neighborhoods, code/value vectors, embedding model/config/index roots, tombstones, compatibility filters, ambiguity
- Acceptance: Use exact symbol/contract/value/lineage/graph routes before lexical/KG/vector similarity; pin embedding provider/model artifact/revision/dimension/chunker/normalizer/distance and corpus/index roots; a deterministic canary rejects missing-dependency success shims, constant vectors, NaN/Inf, dimension or config drift and disables only the optional vector lane; never use an unpinned remote embedding; reject stale/cross-tree/generated/read-only/poisoned/forged candidates before scoring; carry candidate CIDs, source authority, hard compatibility and information-content facts separately from scores; deterministic ties, no candidate or multiple equally eligible candidates remain explicit and cannot authorize semantics, values, placements, targets or writes.
- Embedding query: code refactor rename move missing value graph rag knowledge graph embeddings vector cid candidate retrieval

## LPR-032 Federate exact proof caches and lazy datasets Logic capabilities

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-foundations
- Depends on: LPR-020
- Goal id: LPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_doctor_logic.py, test/api/test_agent_supervisor_doctor_proof_cache.py, test/api/test_agent_supervisor_ipfs_datasets_doctor_logic.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_doctor_proof_cache.py test/api/test_agent_supervisor_ipfs_datasets_doctor_logic.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-cache-logic
- Parallel lane: lpr-doctor-cache-logic
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 40000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_doctor_logic.py, test/api/test_agent_supervisor_doctor_proof_cache.py, test/api/test_agent_supervisor_ipfs_datasets_doctor_logic.py
- AST symbols: DoctorProofCacheBinding, DoctorProofCacheGate, DoctorCacheAuditReceipt, DatasetsDoctorLogicCapability, IpfsDatasetsDoctorLogic
- Interfaces: FormalVerificationCache, ProofCacheKey, ProverEvidenceStore, RuntimeCAS, ContentIdentityBridge@1, ProofCorpusStore, AttestedProofVerifier, IpfsDatasetsTacticianProvider, IsolatedHammerLoader
- Allow concurrent with: LPR-021, LPR-029, LPR-030, LPR-031
- Conflict policy: Wrap rather than replace existing caches/CAS and lazy logic providers; the supervisor formal cache is proof-authority storage, datasets Hammer cache is provider-local acceleration, attested corpus is nomination/applicability input, and legacy IPFS cache is transport-only until full verification.
- Preconditions: Canonical identity/CAS, formal verification cache, proof evidence store, datasets Tactician/Hammer capability adapters and source-bound schemas are stable.
- Effects: Doctor stages can lazily query exact logic capabilities and reuse only current fully bound reconstructed proofs, with single-flight, dependency invalidation, quarantine and replay receipts.
- Evidence subset: canonical preimage/CID, cache namespace/key, obligation/premises, translator/solver/kernel/toolchain/registry/policy/resource/environment/tree roots, trust, invalidation
- Acceptance: Require CIDv1/profile/preimage conformance and keys binding forest/tree/overlay/AST/graph/corpus/goal/premises/translation/solver/kernel/toolchain/registry/policy/budget/sandbox/environment/candidate tree; reject digest-like pseudo-CIDs, double hashing, alias/profile mismatch, partial/solver-only/raw-countermodel/expired/stale/corrupt entries; revalidate and reconstruct positive hits immediately before render and commit; negative hits/timeouts remain diagnostic; semantic-root changes invalidate descendants and tombstone indexes; equivocation quarantines; all datasets logic/solver/cache modules load lazily through exact capability probes with no install, network, target import or process-global unsafe mutation.
- Embedding query: deterministic doctor proof cache cid cas datasets logic tactician hammer lazy reconstruction invalidation

## LPR-033 Register allowlisted deterministic AST repair transforms

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-foundations
- Depends on: LPR-029, LPR-030
- Goal id: LPR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/test_agent_supervisor_deterministic_doctor_transforms.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_transforms.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-transforms
- Parallel lane: lpr-doctor-transforms
- Resource class: cpu-large
- Resource stage: planning
- Token class: large
- Estimated tokens: 38000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/test_agent_supervisor_deterministic_doctor_transforms.py
- AST symbols: DoctorRepairOperatorRegistry, DoctorRepairOperatorSpec, DoctorOperatorProposal, DoctorOperatorReceipt, build_default_doctor_operator_registry
- Interfaces: AnalyticalChangeTransformer@1, AnalyticalTransformRequest, ContractRepairEditPacket@2, ChangePropagationEditPacket@1, ProgramContractDelta@1
- Allow concurrent with: LPR-034
- Conflict policy: Adapt existing analytical transforms into a closed immutable registry; no arbitrary text/template/shell/plugin/model-generated operator, free semantics, new dependency, public/native/stateful invention, or edit to the doctor/proof/identity/transaction trusted base.
- Preconditions: Doctor contracts/policy and real-checkout source-span/AST findings are stable.
- Effects: Exact rename, import/export/re-export/registration, add/rename/reorder/thread argument, constructor/factory, finite total adapter, authorized schema/serializer/fixture/manifest mapping and verified-artifact restoration become typed proposals with proof templates.
- Evidence subset: operator registry identity, AST/span/before hashes, types/values/placements, pre/post/frame conditions, read/write paths, idempotency, inverse/compensation
- Acceptance: Every operator has closed input/output types, supported AST shapes, preconditions, semantic postconditions, frame conditions, exact read/write sets, value-source requirements, deterministic byte-stable renderer, idempotency and inverse/compensation; produce only body-free proposals until proof admission; reject splats, ambiguous overloads, reflection, monkey patches, stale spans, incomplete mappings, unproved values, generated/native/FFI/unsafe/concurrency targets, forbidden paths, cross-root writes and complex new behavior; repeating a transform is a no-op or deterministic rejection.
- Embedding query: deterministic ast codemod repair operators rename import argument threading adapter schema no llm

## LPR-034 Compile findings into independent goals and gated Tactician plans

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-proof-synthesis
- Depends on: LPR-029, LPR-030, LPR-031
- Goal id: LPR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_tactician.py, test/api/test_agent_supervisor_deterministic_doctor_tactician.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_tactician.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-tactician
- Parallel lane: lpr-doctor-tactician
- Resource class: cpu-medium
- Resource stage: planning
- Token class: large
- Estimated tokens: 36000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_tactician.py, test/api/test_agent_supervisor_deterministic_doctor_tactician.py
- AST symbols: DoctorRepairGoalCompiler, DeterministicDoctorTactician, DoctorTacticianPlanReceipt
- Interfaces: ProgramLogicPremiseCorpusBuilder, ProgramLogicGoalCompiler, LogicGapSlicer, IpfsDatasetsTacticianProvider, TacticianPlanGate, ProofDirectedRetrievalReceipt
- Allow concurrent with: LPR-033
- Conflict policy: Compile independent expectations and observed facts before calling the domain-neutral datasets Tactician; the Tactician plans finite proof/synthesis search only and receives no learned/model route in deterministic mode.
- Preconditions: Doctor contracts, diagnostics and hybrid candidate evidence are current and mutually root-bound.
- Effects: Each supported finding becomes a bounded premise corpus, complete typed goal/frontier inventory and acyclic exact-first proof/operator search plan with selected/excluded source routes and budgets.
- Evidence subset: expectation authority, observations, facets, premises, goals, candidates, Tactician routes/subgoals, exclusions, completeness, budgets
- Acceptance: Preserve required input/output/error/effect/auth/resource/state/schema/placement/information/memory facets and every unknown frontier; prevent candidate code, cache metadata, tests by mere success, KG/vector/embedding scores or Tactician output from authoring their own expectation; order authoritative exact local routes before approximate nominations; reject cycles, axiom smuggling, self-validation, prompt directives, changed roots, missing consumers/facets, forged IDs, unbounded routes and score-based authority; emit deterministic plans with semantic_authority=false and no LLM/model-provider route, or a typed abstention.
- Embedding query: deterministic doctor findings premises goals datasets tactician plan exact first no model

## LPR-035 Verify candidate repairs through cache-first isolated Hammer and CEGIS

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-proof-synthesis
- Depends on: LPR-032, LPR-034
- Goal id: LPR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py, test/api/test_agent_supervisor_deterministic_doctor_hammer.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_hammer.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-hammer
- Parallel lane: lpr-doctor-hammer
- Resource class: cpu-proof-solver
- Resource stage: proof
- Token class: large
- Estimated tokens: 46000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py, test/api/test_agent_supervisor_deterministic_doctor_hammer.py
- AST symbols: DoctorRepairObligationCompiler, DeterministicDoctorHammer, DoctorRepairProofReceipt
- Interfaces: TacticianPlanGate, TacticianHammerObligationCompiler, TacticianHammerCoordinator, LogicPredictionCEGIS, LogicPredictionAdmission, DoctorProofCacheGate, CountermodelValidator, NativeReconstructionReceipt
- Allow concurrent with:
- Conflict policy: Reuse the production isolated Hammer route and formal cache; never use the unchecked legacy HammerPipeline, mutate process-global import state concurrently, accept a solver label/raw countermodel, or fall through to a model.
- Preconditions: Exact logic capabilities and proof-cache gate are healthy; the finding has an admitted Tactician plan with current corpus/candidate roots.
- Effects: Candidate operator/value/placement combinations are lowered, cache-checked, resource-bounded, proved/refuted/refined and admitted only through current native reconstruction or independently replayed countermodels.
- Evidence subset: obligations, premises, cache audit, native goals, translation round trip, solver portfolio, countermodels, reconstruction, kernel, CEGIS residuals, uniqueness
- Acceptance: Revalidate every cache binding; require an explicit native-execution permit and adequate subprocess/platform isolation; bind exact obligation/premise/translator/solver/kernel/toolchain/policy/resource/environment identities; reconstruct the matching theorem in the pinned kernel; independently replay a countermodel or proof of negation before refutation; CEGIS is finite, monotonic and repetition-bounded; exactly one complete eligible consequence may proceed; unavailable isolation/provider/kernel, unsupported lowering, inconsistency, ambiguity, stale roots, timeout or bound exhaustion abstains with zero writes and zero LLM or remote model-provider calls.
- Embedding query: deterministic doctor hammer proof cache cegis native reconstruction countermodel unique repair no llm

## LPR-036 Materialize only proof-admitted deterministic repair overlays

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-proof-synthesis
- Depends on: LPR-033, LPR-035
- Goal id: LPR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, test/api/test_agent_supervisor_deterministic_doctor_synthesis.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_synthesis.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-synthesis
- Parallel lane: lpr-doctor-synthesis
- Resource class: cpu-large
- Resource stage: planning
- Token class: large
- Estimated tokens: 38000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, test/api/test_agent_supervisor_deterministic_doctor_synthesis.py
- AST symbols: DeterministicDoctorSynthesizer, DoctorAnalyticalOverlay, DoctorSynthesisReceipt
- Interfaces: DoctorRepairOperatorRegistry, AnalyticalChangeTransformer@1, LogicPredictionAdmission, LogicGuidedRepairPacketMaterializer, ContractRepairEditPacket@2, ChangePropagationEditPacket@1
- Allow concurrent with:
- Conflict policy: Materialize the admitted closed operator through the existing analytical path; do not import or call llm_router/provider modules, invent text or behavior, select new targets/paths/dependencies, or grant an overlay write authority.
- Preconditions: One current reconstructed repair consequence and matching registered operator are uniquely admitted.
- Effects: The exact proved substitution becomes a deterministic source-span/before-hash-bound candidate overlay and simulation receipt, ready for impact planning but incapable of direct mutation.
- Evidence subset: finding/plan/proof/operator CIDs, target/value/placement, exact spans/hashes, deterministic render, syntax simulation, forbidden paths, postconditions
- Acceptance: Recompute every input identity and operator precondition; render only the unique admitted target/value/placement with exact before hashes and bounded paths; prove output patch CID and byte-equivalent replay; parse/simulate without writing the target; reject changed spans, unsupported AST shape, extra files/imports/dependencies, semantics outside the admitted consequence, non-idempotency or provider/model import/call; monkeypatched LLM routes that raise remain untouched; a failed render yields a typed abstention and no partial overlay.
- Embedding query: proof admitted deterministic repair synthesis analytical overlay exact spans no llm

## LPR-037 Close all callers and compile one atomic deterministic repair plan

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-transaction
- Depends on: LPR-030, LPR-036
- Goal id: LPR-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_impact.py, test/api/test_agent_supervisor_deterministic_doctor_impact.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_impact.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-impact
- Parallel lane: lpr-doctor-impact
- Resource class: cpu-large
- Resource stage: analysis
- Token class: large
- Estimated tokens: 40000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_impact.py, test/api/test_agent_supervisor_deterministic_doctor_impact.py
- AST symbols: DeterministicDoctorImpactAnalyzer, DoctorConsumerDisposition, DoctorImpactClosureReceipt, compile_deterministic_doctor_plan
- Interfaces: ContractChangeImpactAnalyzer, ProgramDependencyGraph, ProgramCallResolver, ChangeConsumerInventory, ChangePropagationObligationCompiler, ChangePropagationPlanner, ChangePropagationPlan@1
- Allow concurrent with:
- Conflict policy: Reuse current impact/propagation authority and preserve unresolved frontiers; never infer completeness from a passing test, vector/KG neighborhood, static subset, or absence of observed failures.
- Preconditions: Current real-checkout diagnostic snapshot and proof-admitted analytical overlay are available.
- Effects: Direct/transitive calls, aliases, methods, wrappers, imports/exports, registrations, entry points, schemas, generated declarations and second-order effects receive explicit dispositions and atomic SCC grouping.
- Evidence subset: base/candidate contract delta, AST/import/call/dependency closure, callers/consumers, dynamic/generated/native frontier, SCCs, obligations, dispositions, read/write set
- Acceptance: Rebuild the candidate ProgramContractDelta and resolve the bounded current call/import/dependency graph; require one migrated/proved-compatible/unaffected/approval/unsupported disposition per resolved consumer; discover second-order consumers introduced by the overlay; represent reflection, unknown dispatch, generated code, native/FFI and unsupported interprocedural paths as required open frontiers; mutation requires complete closure, current CIDs, no forbidden path and one atomic plan covering all necessary SCC steps; missed/duplicate/stale consumers, circular ownership, plan gaps or an open required frontier abstain before any write.
- Embedding query: deterministic doctor all caller consumer impact closure scc atomic repair plan

## LPR-038 Apply in isolation and re-prove to a deterministic fixed point

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-transaction
- Depends on: LPR-032, LPR-037
- Goal id: LPR-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transaction.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_fixed_point.py, test/api/test_agent_supervisor_deterministic_doctor_transaction.py, test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_transaction.py test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-transaction
- Parallel lane: lpr-doctor-transaction
- Resource class: cpu-large
- Resource stage: integration
- Token class: large
- Estimated tokens: 52000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transaction.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_fixed_point.py, test/api/test_agent_supervisor_deterministic_doctor_transaction.py, test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py
- AST symbols: DoctorSandboxPolicy, DoctorSandboxCapability, DeterministicDoctorTransaction, DeterministicDoctorFixedPointValidator, DoctorCandidateTreeReceipt, DoctorFixedPointReceipt
- Interfaces: ChangePropagationTransaction, ChangePropagationValidator, WriterLease, CheckoutLock, MergeQueue, RuntimeCAS, DoctorProofCacheGate, PropagationCompletionReceipt@1, LogicFixedPointEvidenceAttachment@1
- Allow concurrent with:
- Conflict policy: Apply only inside a disposable exact-base worktree and integrate through existing lease/locks/merge queue/ref-CAS; never import target code into the doctor, inherit secrets, write a dirty user tree directly, weaken sandbox claims, or bypass compensating rollback.
- Preconditions: A complete current deterministic repair plan covers every required consumer/SCC and the cache/proof bindings pass pre-render validation.
- Effects: The complete patch is rendered and validated under bounded isolation, integrated atomically, indexes/caches are invalidated, and diagnosis/impact/Tactician/Hammer/program validation iterate to a current fixed point or roll back.
- Evidence subset: base/candidate/committed CIDs, sandbox enforcement, allowlisted commands, resources, lease/checkpoint/before hashes, SCC transaction, validations, cache invalidation, fixed point, rollback
- Acceptance: Confine paths and deny inherited secrets/network/process escape with enforced platform evidence before target execution; treat symlink/hardlink/submodule/device/path races as hostile; weak isolation permits pure static replay only and forces abstention for execution-dependent repair; acquire checkout lock/writer lease/checkpoint and revalidate roots/cache immediately before commit; apply the entire SCC or nothing through merge ref compare-and-swap; reparse/type/static/differential/proof and memory/effect/resource checks, rebuild AST/graphs/KG/vector tombstones, invalidate dependent CAS/cache, redelta/reclose/replan/reprove until no original/second-order finding remains; bounds/oscillation/drift/failure compensates, rollback failure quarantines, and neither state may claim completion or call a model.
- Embedding query: deterministic doctor sandbox worktree transaction rollback reindex cache invalidation fixed point

## LPR-039 Expose deterministic doctor operations through a thin service and CLI

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-transaction
- Depends on: LPR-038
- Goal id: LPR-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_service.py, scripts/ops/agent_supervisor/deterministic_doctor.py, test/api/test_agent_supervisor_deterministic_doctor_service.py, test/api/test_agent_supervisor_deterministic_doctor_ops.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_service.py test/api/test_agent_supervisor_deterministic_doctor_ops.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-operations
- Parallel lane: lpr-doctor-operations
- Resource class: cpu-medium
- Resource stage: control
- Token class: large
- Estimated tokens: 38000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_service.py, scripts/ops/agent_supervisor/deterministic_doctor.py, test/api/test_agent_supervisor_deterministic_doctor_service.py, test/api/test_agent_supervisor_deterministic_doctor_ops.py
- AST symbols: DeterministicDoctorService, DoctorOperation, DoctorOperationRequest, DoctorOperationResult, main
- Interfaces: deterministic-doctor diagnose/plan/synthesis/impact/transaction/fixed-point stages, SupervisorControlService, RuntimeCAS
- Allow concurrent with:
- Conflict policy: The service owns orchestration and the ops script owns only argument/config/bootstrap/delegation; preserve the shell lifecycle doctor's read-only health semantics and do not put analysis/proof/edit/transaction logic or optional eager imports in the wrapper.
- Preconditions: The deterministic transaction/fixed-point controller is stable and all operations emit canonical receipts.
- Effects: Operators can inspect, explain, plan, repair, replay, verify, status and rollback an incident by CID with bounded machine-readable output and no silent provider/model escalation.
- Evidence subset: CLI modes, policy/feature gates, incident idempotency, exact roots, operation receipts, exit codes, cold import, provider/network/process absence, rollback
- Acceptance: Cold import and --help start no process, import no optional datasets/prover/embedding provider, open no database and access no network; inspect/explain/plan are always read-only; repair requires explicit operation plus enabled policy, exact clean target, writer lease and eligible plan; replay is identity-equivalent and incident-CID idempotent; unsupported/unavailable capabilities return actionable abstentions without making startup unhealthy; no secret/body enters argv/logs; wrapper contains no analysis, proof, rendering, transaction or mutation logic; an intercepted llm_router/LLM/model-provider call fails the test and no automatic fallback exists.
- Embedding query: deterministic doctor service thin ops cli inspect explain plan repair replay rollback no llm

## LPR-040 Benchmark adversarial no-LLM diagnosis and repair

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-release
- Depends on: LPR-030, LPR-031, LPR-032, LPR-039
- Goal id: LPR-G110
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_benchmark.py, test/fixtures/agent_supervisor/deterministic_doctor, test/api/test_agent_supervisor_deterministic_doctor_benchmark.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_benchmark.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-benchmark
- Parallel lane: lpr-doctor-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Token class: large
- Estimated tokens: 46000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_benchmark.py, test/fixtures/agent_supervisor/deterministic_doctor, test/api/test_agent_supervisor_deterministic_doctor_benchmark.py
- AST symbols: DeterministicDoctorFixture, DeterministicDoctorBenchmarkPolicy, DeterministicDoctorMetrics, DeterministicDoctorBenchmarkReport
- Interfaces: DeterministicDoctorService, DoctorEvidenceSnapshot@1, DeterministicDoctorRunReceipt@1, ResourceMeasurement
- Allow concurrent with:
- Conflict policy: Own a hermetic content-addressed fixture corpus and measurement-only evaluator; seeded truth may define expected outcomes but metrics, passing tests, retrieval ranks and observed implementations do not grant production authority.
- Preconditions: End-to-end deterministic-doctor operations work with injected exact static/logic/cache/sandbox providers.
- Effects: Repeatable stage-specific metrics measure diagnostic, candidate, proof, transform, impact, transaction, rollback, abstention and fixed-point safety/efficacy with every model route patched to raise.
- Evidence subset: seeded truth, exact roots/CIDs, positive/adversarial cases, stage receipts, repeated replay, resources, safety floors, LLM/model-provider invocation counters
- Acceptance: Include renamed/moved symbol, import/export/registration, two-to-three-argument changes across direct/alias/wrapper/method callers, constructor/factory/context threading, finite adapter/schema/serializer/manifest/artifact repairs; include same-type wrong value, vector collision, KG omission, constant embedding fallback, stale/corrupt/forged CID/cache, solver lie/countermodel, incomplete AST/impact/SCC, dynamic/generated/native/FFI/public-schema/cross-root/new-dependency/complex behavior, sandbox escape, crash/rollback and oscillation; run every fixture twice with identity-equivalent receipts; require correct repair or typed abstention, zero missed mandatory caller, authority promotion, stale proof/CID, out-of-scope/sandbox write, partial transaction, rollback failure, nondeterministic render, false fixed point and llm_router/LLM/model-provider calls.
- Embedding query: adversarial deterministic doctor benchmark rename missing argument graph vector proof cache no llm safety

## LPR-041 Add report-only through narrow-auto rollout controls and operator validation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-release
- Depends on: LPR-040
- Goal id: LPR-G110
- Outputs: ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_rollout.py, config/agent_supervisor_deterministic_doctor.json, scripts/ops/agent_supervisor/validate_deterministic_doctor.py, docs/guides/DETERMINISTIC_DOCTOR_GUIDE.md, test/api/test_agent_supervisor_deterministic_doctor_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_rollout.py && python scripts/ops/agent_supervisor/validate_deterministic_doctor.py
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-rollout
- Parallel lane: lpr-doctor-rollout
- Resource class: cpu-large
- Resource stage: rollout
- Token class: large
- Estimated tokens: 40000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_rollout.py, config/agent_supervisor_deterministic_doctor.json, scripts/ops/agent_supervisor/validate_deterministic_doctor.py, docs/guides/DETERMINISTIC_DOCTOR_GUIDE.md, test/api/test_agent_supervisor_deterministic_doctor_rollout.py
- AST symbols: DeterministicDoctorMode, DeterministicDoctorRolloutPolicy, DeterministicDoctorRolloutDecision, DeterministicDoctorRollbackGate, DeterministicDoctorOperationsValidator
- Interfaces: DeterministicDoctorMetrics, DeterministicDoctorService, SupervisorControlService, feature-gate and rollback receipts
- Allow concurrent with:
- Conflict policy: Own doctor rollout/config/validator/guide/test only; report-only remains the default, narrow auto remains off until explicit promotion, optional retrieval/prover failures cause degradation/abstention rather than supervisor blockage, and model flags are orthogonal and forbidden in deterministic mode.
- Preconditions: The adversarial benchmark meets every zero safety floor under repeated replay.
- Effects: Operators can independently stage report-only, plan, sandbox-auto and deterministic narrow-auto, inspect capabilities/receipts/metrics, use a kill switch and roll back one stage on any regression.
- Evidence subset: modes, gates, exact config CID, capabilities, safety floors, resource limits, feature flags, kill switch, lifecycle, rollback, operator actions
- Acceptance: Immutable bounded config defaults to report-only with deterministic narrow-auto false, remote embeddings/network/LLM/remote model-provider calls false, and exact-root/proof-cache-revalidation/native-reconstruction/all-callers/sandbox/lease/atomic/fixed-point gates true; define limits for findings/candidates/queries/operators/plan steps/iterations/files/bytes/processes/time/CPU/memory; promotion is manual and monotonic; any nonzero safety floor, root/schema/capability drift, embedding canary failure, reconstruction/isolation loss, transaction/rollback failure or material resource regression rolls back or disables auto; ordinary lifecycle doctor remains read-only/idempotent; optional provider absence is actionable but does not block report-only startup.
- Embedding query: deterministic doctor rollout report only plan sandbox auto narrow auto kill switch rollback operations

## LPR-042 Release the joined VFS and deterministic-doctor fixed point

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-doctor-release
- Depends on: LPR-028, LPR-041
- Goal id: LPR-G110
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_release.py, test/api/test_agent_supervisor_deterministic_doctor_end_to_end.py, test/api/test_agent_supervisor_deterministic_doctor_replay.py, docs/architecture/DETERMINISTIC_DOCTOR_RELEASE.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_deterministic_doctor_end_to_end.py test/api/test_agent_supervisor_deterministic_doctor_replay.py && python scripts/ops/agent_supervisor/validate_deterministic_doctor.py && python scripts/validate_tactician_hammer_logic_repair_board.py --check-all
- Board namespace: agent-supervisor-tactician-hammer-logic-repair-v1
- Bundle: agent-supervisor/tactician-hammer-logic-repair/doctor-release
- Parallel lane: lpr-doctor-release
- Resource class: cpu-large
- Resource stage: rollout
- Token class: large
- Estimated tokens: 48000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_release.py, test/api/test_agent_supervisor_deterministic_doctor_end_to_end.py, test/api/test_agent_supervisor_deterministic_doctor_replay.py, docs/architecture/DETERMINISTIC_DOCTOR_RELEASE.md
- AST symbols: DeterministicDoctorReleasePolicy, DeterministicDoctorReleaseReceipt, validate_deterministic_doctor_release
- Interfaces: VfsGeneralizationEquivalenceReceipt, AssuranceTwoProfileConformance, DeterministicDoctorRunReceipt@1, DeterministicDoctorMetrics, PropagationCompletionReceipt@1, LogicFixedPointEvidenceAttachment@1
- Allow concurrent with:
- Conflict policy: Serialize joined terminal validation; consume but do not rewrite completed task history or protected control artifacts, preserve report-only default and VFS placement/equivalence gates, and never weaken a zero floor to obtain release.
- Preconditions: LPR-028 proves the generic/VFS cutover and LPR-041 proves deterministic-doctor benchmark/rollout readiness on the same current target tree.
- Effects: One content-addressed release receipt joins both branches and proves the supervisor can diagnose, prove, analytically repair or abstain, transact, replay, roll back and drain the complete board without an LLM or control-plane blocker.
- Evidence subset: board/goal DAG, preserved task CIDs, VFS equivalence/layout, exact roots/capabilities, doctor fixtures/metrics/floors, cold imports, provider absence, transaction/rollback, fixed point, supervisor drain
- Acceptance: Validate all 43 canonical tasks, 12 goals and LPR-042 as unique terminal; preserve semantic CIDs of LPR-000 through LPR-028; run the migrated VFS and non-VFS profiles plus deterministic-doctor positive/adversarial real-checkout fixtures twice with identical CIDs/receipts; prove optional provider absence and cold imports are safe; prove report-only makes no write and an eligible no-model repair reaches complete all-caller atomic fixed point; prove every ambiguous/unsupported case abstains with a clean tree and rollback restores exact roots; require zero LLM or remote model-provider invocation, KG/vector/embedding authority promotion, stale/forged cache/CID admission, missed caller/open-frontier mutation, sandbox/path escape, partial transaction, rollback failure, nondeterminism and false completion; confirm the healthy four-lane supervisor can drain the joined DAG without dependency, provider, protected-path, merge or lifecycle blockage.
- Embedding query: joined release vfs deterministic doctor no llm canonical cid replay rollback fixed point supervisor drain
