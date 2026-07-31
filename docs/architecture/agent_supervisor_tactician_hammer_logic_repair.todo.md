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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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
