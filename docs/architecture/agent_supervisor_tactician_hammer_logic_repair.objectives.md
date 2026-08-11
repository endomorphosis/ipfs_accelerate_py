# Agent Supervisor Tactician-Hammer Logic Repair Objective Heap

This is the durable source of intent for the Tactician-Hammer logic-repair
program. Child goals are the executable subgoals of `LPR-G000`. The task
projection is `agent_supervisor_tactician_hammer_logic_repair.todo.md`; the
normative design is
`AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md`.

Program invariants:

- Reuse the completed RPR trace, contract, graph, impact, value, behavior,
  proof, analytical-transform, plan, packet, transaction, and fixed-point
  interfaces; do not fork them.
- The current legal `ProofTactician` is a domain adapter, not code authority.
  Program repair requires a domain-neutral, versioned logic Tactician.
- Tactician, vectors, knowledge graphs, learned ranking, history, runtime
  witnesses, ordinary tests, and LLMs nominate or explain; they do not prove
  behavior or authorize writes. A reviewed conformance test may define an
  expectation under existing precedence, but does not prove it by passing.
- Candidate implementation observations cannot define the expected behavior
  used to admit that candidate.
- Hammer solver output is untrusted until the exact native proof is
  reconstructed and accepted by the pinned kernel.
- Contradictory, circular, self-referential, incomplete, unsupported, stale,
  timed-out, or unreconstructed evidence fails closed.
- Every resolved affected consumer receives one explicit disposition; an open
  required dynamic/generated/native frontier blocks autonomous mutation.
- Analytical synthesis precedes `llm_router`; models receive exact admitted
  semantics and paths and return proposal-only diffs.
- The deterministic-doctor mode is a separate analytical path: it may use
  datasets Tactician/Hammer and advisory KG/vector/embeddings, but invokes no
  LLM/model provider and never falls back to one.
- Type/resource evidence is not general memory-safety evidence; unsupported
  ownership, lifetime, unsafe, allocator, concurrency, or FFI claims remain
  explicit.
- Writes are checkpointed and transactional; completion is a candidate-tree
  fixed point that includes renewed Tactician/Hammer analysis.

## LPR-G000 Deliver live proof-guided logic prediction and transitive repair

- Status: blocked
- Review only: true
- Parent:
- Priority: P0
- Track: tactician-hammer-logic-repair
- Bundle: agent-supervisor/tactician-hammer-logic-repair/control
- Parallel lane: control
- Conflict policy: Root is review and completion aggregation only; implementation belongs to child subgoals.
- Resource class: cpu-medium
- Token class: small
- Goal: Starting from task intent, a broken trace, or an intentional contract delta, derive the finite logic the program must satisfy, plan evidence and proof search with a domain-neutral Tactician, prove supported hypotheses through kernel reconstruction or reject them through independently validated countermodels/proofs of negation, repair every resolved caller analytically where possible, otherwise issue a bounded context-rich llm_router proposal only on the explicitly model-enabled path, and complete only after atomic candidate-tree fixed-point validation. On the separate deterministic-doctor path, freeze current AST/graph/cache/index/content roots and use only closed proof-admitted analytical operators, with no LLM/model-provider invocation or fallback.
- Subgoals: LPR-G010, LPR-G020, LPR-G030, LPR-G040, LPR-G050, LPR-G060, LPR-G070, LPR-G080, LPR-G090, LPR-G100, LPR-G110
- Evidence: LPR-G010, LPR-G020, LPR-G030, LPR-G040, LPR-G050, LPR-G060, LPR-G070, LPR-G080, LPR-G090, LPR-G100, LPR-G110
- Outputs:
- Validation: Parse and validate the LPR taskboard, run the LPR-020 logic-repair release validator, run the LPR-028 VFS-generalization equivalence and placement validator, and finish with the terminal LPR-042 deterministic-doctor joined release validator.
- Acceptance: Every child subgoal has current-tree evidence; the all-caller two-to-three-argument fixture and complex support-type fixture either complete with reconstructed proof and a fixed point or conservatively abstain; deterministic-doctor fixtures replay with identical CIDs, use exact-first AST/graph/cache evidence plus advisory KG/vector/embeddings, invoke no LLM/model provider, repair only a unique closed proved operator, and leave a clean tree on abstention; unauthorized axiom, advisory-source authority promotion, stale proof/CID, missed caller/open-frontier mutation, invented behavior, wrong value/source/placement, sandbox/path escape, model scope escape, partial transaction, rollback failure, nondeterminism, and false fixed-point completion rates are zero.
- Gap task: Aggregate child evidence and release gates only; do not perform a cross-program implementation edit.
- Refinement: Prefer explicit unsupported and approval-required states over unearned automation.
- Embedding query: live code repair program logic tactician hammer static analysis synthesis all callers deterministic doctor content identity proof cache knowledge graph embeddings fixed point
- AST query: ProgramLogicGoal LogicGap TacticianSearchPlan LogicHypothesis LogicPredictionReceipt LogicGuidedRepairPacket DoctorEvidenceSnapshot DeterministicDoctorPlan DeterministicDoctorRunReceipt

## LPR-G010 Establish trust contracts, exact capabilities, and a generic Tactician

- Status: active
- Parent: LPR-G000
- Priority: P0
- Track: foundations
- Bundle: agent-supervisor/tactician-hammer-logic-repair/foundations
- Parallel lane: lpr-foundations
- Conflict policy: Own only LPR records, capability adapters, the new upstream generic Tactician package, and fixtures; do not alter RPR authority records or treat the legal Tactician as a code interface.
- Resource class: cpu-medium
- Token class: large
- Goal: Define bounded content-addressed prediction records, probe exact Tactician/Hammer/static-analysis capabilities lazily, provide a domain-neutral Tactician API, and establish a shared adversarial fixture corpus.
- Evidence: LPR-001, LPR-002, LPR-003, LPR-004
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_prediction_contracts.py, ipfs_accelerate_py/agent_supervisor/integrations/tactician_hammer_capabilities.py, ipfs_datasets_py/ipfs_datasets_py/logic/tactician, test/fixtures/agent_supervisor/tactician_hammer_logic_repair
- Validation: Run the four task-local foundation suites plus cold-import and fixture-manifest validation.
- Acceptance: Records bind exact tree/overlay/graph/index/corpus/model/translator/toolchain/policy/environment roots and reject forged/unbounded/body-bearing states; capability probes infer nothing from package presence and install nothing; generic Tactician planning is deterministic, finite, domain-neutral and advisory; the datasets implementation lands as a reviewed nested-repository commit plus exact parent gitlink update; fixtures cover positive, ambiguous, poisoned, stale, unsupported, resource, prompt, transaction, and second-order cases.
- Gap task: Build the trust and capability substrate before any live proof or model integration.
- Refinement: The new Tactician plans proof search; it does not prove the plan or select a write path.
- Embedding query: versioned code tactician contracts capabilities hammer corpus policy adversarial fixtures
- AST query: ProgramLogicGoal LogicGap TacticianSearchPlan LogicHypothesis LogicPredictionReceipt TacticianHammerCapabilityReport

## LPR-G020 Compile independent premises and finite program-logic goals

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G010
- Priority: P0
- Track: static-logic-analysis
- Bundle: agent-supervisor/tactician-hammer-logic-repair/static-logic
- Parallel lane: lpr-static-logic
- Conflict policy: Reuse RPR contracts/graphs/value provenance by reference; own only program premise, goal, and gap adapters.
- Resource class: cpu-large
- Token class: large
- Goal: From objective, trace, contract delta, impact closure, consumer obligation, value provenance, and behavior gaps, compile an independent content-addressed theorem corpus plus a finite positive/negative logic-goal inventory and minimal static information-demand slices.
- Evidence: LPR-005, LPR-006, LPR-007
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/program_logic_premise_corpus.py, ipfs_accelerate_py/agent_supervisor/analysis/program_logic_goal_compiler.py, ipfs_accelerate_py/agent_supervisor/analysis/logic_gap_slicer.py
- Validation: Run premise-corpus, goal-compiler, and gap-slicer suites against positive and adversarial fixtures.
- Acceptance: Reviewed contracts/specs/tests retain explicit precedence; candidate code/comments/vector/KG/runtime/model evidence remains non-authoritative; contradictory or self-referential premises abstain; every input/output/error/effect/auth/resource/state/schema/placement/information facet becomes a typed goal or explicit unsupported residual; bounded/incomplete slices never claim completeness.
- Gap task: Turn code evidence into exact questions before asking either Tactician or Hammer.
- Refinement: Natural-language task text proposes goals but never enters the premise corpus as an axiom.
- Embedding query: static program facts independent theorem corpus logic goals gap slicing dataflow contracts
- AST query: ProgramLogicPremiseCorpusBuilder ProgramLogicGoalCompiler LogicGapSlicer ProgramLogicGoal LogicGap

## LPR-G030 Plan proof search and nominate safe logic hypotheses

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G010, LPR-G020
- Priority: P0
- Track: tactician-planning
- Bundle: agent-supervisor/tactician-hammer-logic-repair/tactician
- Parallel lane: lpr-tactician
- Conflict policy: Own the supervisor Tactician adapter, hypothesis retrieval, and plan gate; do not lower proofs or mutate code in this subgoal.
- Resource class: cpu-medium
- Token class: large
- Goal: Translate exact program goals and corpus references into a bounded Code Tactician plan, union deterministic/static/graph/vector/history/spec hypotheses, and admit only acyclic, source-authorized, root-current proof plans.
- Evidence: LPR-008, LPR-009, LPR-010
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_tactician_provider.py, ipfs_accelerate_py/agent_supervisor/analysis/program_logic_hypothesis_retrieval.py, ipfs_accelerate_py/agent_supervisor/validation/tactician_plan_gate.py
- Validation: Run provider, retrieval, and security-gate suites with deterministic replay, stale roots, poisoned retrieval, prompt injection, cycles, and contradictory premise cases.
- Acceptance: Plans order exact local sources before approximate routes, record exclusions/stops/budgets, preserve all residual goals, and carry semantic_authority=false; hard source/root/completeness/consistency gates precede ranking; same type/name/similarity never proves information sufficiency; invalid or ambiguous plans abstain.
- Gap task: Produce a safe proof-search plan and bounded alternatives, not a guessed program implementation.
- Refinement: Learned ranking is opt-in, digest-pinned, nomination-only, and falls back deterministically.
- Embedding query: code tactician proof search route premise selection vector graph hypothesis security gate
- AST query: IpfsDatasetsTacticianProvider ProgramLogicHypothesisRetriever TacticianPlanGate TacticianSearchPlan LogicHypothesis

## LPR-G040 Prove, refute, refine, and admit program-logic predictions

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G030
- Priority: P0
- Track: hammer-proof
- Bundle: agent-supervisor/tactician-hammer-logic-repair/hammer
- Parallel lane: lpr-hammer
- Conflict policy: Extend the existing production Hammer provider and proof contracts; never route authority through the legacy unchecked HammerPipeline.
- Resource class: cpu-proof-solver
- Token class: large
- Goal: Strictly lower admitted tactic plans to existing supervisor obligations and semantically identical native goals, run a separately authorized resource-bounded production Hammer portfolio, independently reconstruct native proofs, validate countermodels before they can reject a hypothesis, refine residual gaps monotonically, and admit only current uniquely supported consequences.
- Evidence: LPR-011, LPR-012, LPR-013, LPR-014
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_obligations.py, ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_coordinator.py, ipfs_accelerate_py/agent_supervisor/validation/hammer_native_execution_gate.py, ipfs_accelerate_py/agent_supervisor/proof/logic_prediction_cegis.py, ipfs_accelerate_py/agent_supervisor/planning/logic_prediction_admission.py
- Validation: Run lowering, coordinator, refinement, and admission suites with unsupported semantics, solver lies, wrong-theorem reconstruction, stale environment, timeout, contradiction, cycle, and bound-exhaustion cases.
- Acceptance: Exact premise, native-goal and translation-map identities reach Hammer; native execution requires an explicit permit and adequate platform enforcement; process-global lazy-import state is concurrency-safe or isolated; every attempt has a complete receipt; solver candidates and raw countermodels cannot claim verified; only matching kernel acceptance or independent countermodel replay/proof of negation affects admission; refinement never weakens/deletes an original goal and terminates on repeats/bounds; ambiguous, inconsistent, unreconstructed, unsupported, policy-denied, unavailable, timed-out, or stale results abstain.
- Gap task: Convert advisory plans into checked consequences without transferring authority from retrieval or solver scores.
- Refinement: Hammer proves the encoded obligation only; static/type/runtime/native validations remain separate.
- Embedding query: lower program obligation hammer portfolio counterexample reconstruction kernel prediction admission
- AST query: TacticianHammerObligationCompiler TacticianHammerCoordinator LogicPredictionCEGIS LogicPredictionAdmission

## LPR-G050 Integrate admitted logic into analytical and contextual transitive repair

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G040
- Priority: P0
- Track: repair-integration
- Bundle: agent-supervisor/tactician-hammer-logic-repair/integration
- Parallel lane: lpr-repair-integration
- Conflict policy: New-file bridges precede one serialized shared-pipeline cutover; preserve existing RPR defaults, packets, authority gates, transactions, and completion receipts.
- Resource class: cpu-large
- Token class: large
- Goal: Feed reconstructed consequences into existing contract-repair ranking and behavior/value synthesis, render deterministic repairs first, issue exact context overlays through existing RPR packets only for behavior-complete syntax gaps, intercept ordinary model diffs as read-only candidate overlays before mutation, wire live trace/change orchestration at the daemon edge, and require Tactician/Hammer reanalysis plus compensating rollback at the existing fixed-point gate.
- Evidence: LPR-015, LPR-016, LPR-017, LPR-018
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/tactician_guided_behavior_synthesis.py, ipfs_accelerate_py/agent_supervisor/proof/logic_guided_repair_packet.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/live_logic_repair_controller.py, proposal-overlay intake, live pipeline integration, fixed-point logic revalidation
- Validation: Run bridge, packet, shared integration, transaction, rollback, legacy RPR compatibility, and post-edit fixed-point suites.
- Acceptance: Predictions compose with rather than replace ContractRepairProver/CandidateProofBundle, RequiredBehaviorSynthesizer and MissingInputSynthesizer; proof remains orthogonal to and inherits the weakest independent source precedence; analytical transforms make no model call; context overlays preserve existing packet write authority and fix semantics, values, callers, paths, validated countermodels, postconditions and validations; every ordinary or explicit model proposal is analyzed as an overlay and dispositions every resolved caller before mutation; no bypass around plan admission, pre-provider gate, writer lease, checkpoint, SCC transaction, post-validation compensating rollback, or renewed fixed-point proof exists.
- Gap task: Make the proof machinery live and useful while retaining exact RPR write authority.
- Refinement: A model implements an admitted decision; it cannot discover meaning by editing first.
- Embedding query: tactician guided behavior analytical repair llm context all callers transaction fixed point
- AST query: TacticianGuidedBehaviorSynthesizer LogicGuidedRepairPacket LiveLogicRepairController LogicRepairFixedPointValidator

## LPR-G060 Benchmark efficacy and operate a reversible staged rollout

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G050
- Priority: P0
- Track: validation-rollout
- Bundle: agent-supervisor/tactician-hammer-logic-repair/rollout
- Parallel lane: lpr-rollout
- Conflict policy: Own LPR benchmark, validator, scheduler, launcher, guide, metrics, and rollout policy; shadow remains default and generated runtime reports are not committed by workers.
- Resource class: cpu-large
- Token class: large
- Goal: Measure stage-specific logic-prediction and all-caller repair safety/efficacy, expose doctor/replay/status/rollback controls, and release only through shadow, assist, deterministic narrow-auto, and approval-gated contextual-model stages.
- Evidence: LPR-019, LPR-020
- Outputs: benchmark and operations tests/scripts, rollout policy, scheduler, launcher, operator guide
- Validation: Run every adversarial fixture twice for identity-equivalent receipts, enforce zero safety floors, exercise launch/status/restart/stop and rollback using fake providers/processes, and run a seeded end-to-end live repair.
- Acceptance: Stage metrics distinguish goal, premise, Tactician, lowering, solver, reconstruction, prediction, analytical, model, transaction, and fixed-point failures; all ten safety floors are zero; capability or budget regression rolls back; dynamic/generated/native/public-schema/stateful/cross-root changes remain approval-required; a healthy isolated four-lane supervisor can drain the board without unknown dependencies, protected-path conflict, provider blockage, or lifecycle leak.
- Gap task: Establish evidence for efficacy and safety before enabling any autonomous path.
- Refinement: High proof-search recall is useful only under zero unauthorized-admission and false-completion floors.
- Embedding query: logic repair adversarial benchmark metrics shadow assist narrow auto rollback supervisor operations
- AST query: LogicRepairBenchmark LogicRepairMetrics LogicRepairRolloutPolicy LogicRepairOperationsValidator

## LPR-G070 Generalize assurance engines and isolate the IPFS Kit VFS job

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G060
- Priority: P0
- Track: assurance-generalization
- Bundle: agent-supervisor/tactician-hammer-logic-repair/assurance-generalization
- Parallel lane: lpr-assurance-generalization
- Conflict policy: Port only from the exact source-lock blobs; place reusable behavior in semantic domain packages, keep VFS vocabulary and adapters in one lazy integration/profile, keep the ops entry point thin, and never introduce substantive root-level vfs_* modules or merge the broad source snapshot.
- Resource class: cpu-large
- Token class: large
- Goal: Refactor the seven VFS-prefixed supervisor implementations from a reviewed source snapshot into profile-driven repository inventory, behavior-contract, differential, interface-parity, benchmark, pilot, and rollout engines that work for non-VFS programs, then expose the original IPFS Kit VFS job through one bounded declarative profile and a thin scripts/ops entry point.
- Evidence: LPR-021, LPR-022, LPR-023, LPR-024, LPR-025, LPR-026, LPR-027, LPR-028
- Outputs: source-lock and public-contract map, generic analysis/validation/runtime/control engines, lazy IPFS Kit VFS integration, bounded VFS profile, thin operations entry point, caller migration, cross-profile equivalence and root-layout guards
- Validation: Verify every source blob against the pinned Git revision; run focused generic-engine suites, the migrated VFS corpus, a non-VFS end-to-end profile, cold-import/ops smoke tests, Tactician/Hammer delegation proof or explicit abstention, impact-closure validation, and the no-root-vfs placement gate.
- Acceptance: The same generic engine code runs the VFS and a hermetic non-VFS profile; VFS operations, invariants, schemas, errors, effects, canonical identities, authority flags, CLI behavior and receipts remain equivalent; all resolved imports/callers are migrated atomically; the agent_supervisor root contains no vfs_*.py implementation or compatibility stub; generic modules contain no VFS, IPFS, fsspec, SwissKnife, board-ID or fixed-checkout branches; the VFS wrapper only validates configuration, lazy-loads the integration, and delegates; unresolved semantic, dynamic, native or public-API differences abstain rather than being guessed.
- Gap task: Convert one large instance-specific assurance program into reusable machinery without losing the exact behavior of its original job.
- Refinement: Generalization is accepted by proof-backed contract equivalence and a second profile, not by renaming symbols or moving code alone.
- Embedding query: generalize vfs assurance modules semantic packages thin ops profile contract equivalence callers
- AST query: RepositorySurfaceInventory ProgramContractProfile DifferentialContractHarness InterfaceContractParityAnalyzer SymbolicEfficiencyBenchmark SymbolicAssurancePilot SymbolicAssuranceRolloutPolicy

## LPR-G080 Build content-addressed deterministic-doctor evidence and repair primitives

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G060
- Priority: P0
- Track: deterministic-doctor-foundations
- Bundle: agent-supervisor/tactician-hammer-logic-repair/deterministic-doctor-foundations
- Parallel lane: lpr-doctor-foundations
- Conflict policy: Compose existing repository/AST/graph/vector/CAS/proof/analytical-transform interfaces by reference; do not create a second identity, cache, contract, edit, or authority system, import target code, or change the lifecycle recovery controller.
- Resource class: cpu-large
- Token class: large
- Goal: Freeze a complete content-addressed view of a real checkout, compile broken-code diagnostics and repair obligations from AST/contract/type/value/effect/resource/memory evidence, retrieve refactor and missing-value candidates through exact graphs plus advisory KG/vector/embeddings, safely federate proof caches, and expose only closed typed analytical repair operators.
- Evidence: LPR-029, LPR-030, LPR-031, LPR-032, LPR-033
- Outputs: deterministic-doctor contracts and policy, repository diagnostic snapshot, hybrid repair-candidate retriever, exact proof-cache/datasets-logic bridge, closed analytical AST transform registry
- Validation: Run contract, real-checkout diagnostics, KG/vector/embedding, proof-cache/CID, transform, cold-import, clean-rebuild-equivalence, stale/poison, and no-target-import suites.
- Acceptance: Every artifact binds canonical forest/tree/overlay/file/AST/graph/corpus/index/model/cache/toolchain/policy/environment CIDs; target source is parsed as data; exact static routes precede approximate retrieval; KG/GraphRAG/history/vector/embedding/cache candidates carry semantic_authority=false; pinned embedding canaries reject constant fallback, non-finite or dimension-drifted output and disable only the optional lane; positive proof-cache reuse requires the full current key and reconstruction while negative/timeout/provider-local/legacy entries remain diagnostic; operators have closed typed pre/post/frame/value/write/inverse contracts; unsupported or ambiguous evidence abstains and no provider/model call or source write occurs.
- Gap task: Turn the existing disconnected analysis, retrieval, cache, and codemod primitives into one frozen fail-closed substrate for a semantic doctor.
- Refinement: Better retrieval changes recall and search cost, never the standard of proof or write authority.
- Embedding query: deterministic code doctor frozen checkout ast graph knowledge graph embeddings proof cache cid analytical transforms
- AST query: DoctorEvidenceSnapshot DeterministicDoctorFinding DoctorProofCacheAuditReceipt DoctorRepairOperatorRegistry

## LPR-G090 Plan, prove, and synthesize repairs without a model

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G080
- Priority: P0
- Track: deterministic-doctor-proof-synthesis
- Bundle: agent-supervisor/tactician-hammer-logic-repair/deterministic-doctor-proof
- Parallel lane: lpr-doctor-proof
- Conflict policy: Reuse the admitted datasets Tactician provider, plan gate, production Hammer coordinator, CEGIS, native reconstruction, prediction admission, and analytical transform paths; no LLM/model route, free-form patch generator, unchecked solver, or cache-derived authority is permitted.
- Resource class: cpu-proof-solver
- Token class: large
- Goal: Convert deterministic findings and bounded candidates into independent goals and premises, let the domain-neutral datasets Tactician plan exact-first proof search, use cache-first resource-bounded Hammer/CEGIS plus native reconstruction to identify a unique repair, and materialize only a proof-admitted deterministic overlay.
- Evidence: LPR-034, LPR-035, LPR-036
- Outputs: deterministic-doctor Tactician planner, isolated cache-first Hammer/CEGIS verifier, proof-admitted deterministic synthesis/materialization
- Validation: Run axiom-smuggling, poisoned-score, stale-cache, solver-lie, wrong-theorem, countermodel-replay, ambiguity, timeout, resource, import-isolation, no-model-import, and deterministic-render suites.
- Acceptance: Tactician preserves every goal/frontier and cannot promote a source; Hammer runs only behind an explicit native-execution permit in an isolated worker, and its proof/countermodel affects admission only after matching reconstruction or independent replay; cache hits are revalidated; exactly one complete target/value/placement/operator consequence may render; zero/multiple/unsupported/unavailable/timed-out results abstain; generated overlays are byte-stable, body-free, before-hash-bound and import/call no LLM route.
- Gap task: Make deterministic synthesis analytical and proof-directed rather than nearest-neighbor or model-authored.
- Refinement: An unavailable prover lowers automation coverage; it never licenses a heuristic edit or model fallback.
- Embedding query: no llm deterministic repair tactician hammer cegis kernel reconstruction unique analytical operator
- AST query: DeterministicDoctorTactician DoctorRepairObligationCompiler DeterministicDoctorHammer DeterministicDoctorSynthesis

## LPR-G100 Close impacts and execute admitted repairs atomically

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G090
- Priority: P0
- Track: deterministic-doctor-transaction
- Bundle: agent-supervisor/tactician-hammer-logic-repair/deterministic-doctor-transaction
- Parallel lane: lpr-doctor-transaction
- Conflict policy: Reuse the current impact analyzer, propagation planner, writer lease, checkout/merge locks, SCC transaction, CAS invalidation, completion receipt, and compensating rollback; never write the user's dirty tree directly or put semantic editing into lifecycle recovery.
- Resource class: cpu-large
- Token class: large
- Goal: Close every direct, transitive, alias, wrapper, import/export/entry-point, schema, and second-order consumer; validate the uniquely admitted repair in a confined disposable worktree; transact the whole SCC atomically; re-index, invalidate, re-diagnose, and re-prove to a fixed point; and expose bounded deterministic-doctor operations through a thin service and CLI.
- Evidence: LPR-037, LPR-038, LPR-039
- Outputs: deterministic impact closer, sandboxed transaction/fixed-point validator, deterministic-doctor service and thin scripts/ops facade
- Validation: Run complete/incomplete impact, dynamic/generated/native frontier, symlink/path escape, network/secret/process isolation, lease/hash race, crash/recovery, partial-SCC, rollback, oscillation, cold-import, replay, and idempotent CLI suites.
- Acceptance: Every resolved consumer has one disposition and an open required frontier blocks mutation; target code is never imported into the doctor; execution-dependent repair requires enforced filesystem/network/process/resource isolation; exact before hashes, lease, checkpoint and complete SCC precede writes; candidate and committed roots are rebuilt and re-proved; any drift/failure compensates or quarantines; inspect/explain/plan remain read-only, repair is explicitly policy-gated, receipts are queryable by CID, and no operation silently invokes a model.
- Gap task: Convert a proved analytical overlay into a safe real-checkout repair without partial writes or false completion.
- Refinement: Pure static planning may remain available on weak platforms, but execution-dependent automatic repair must abstain when isolation cannot be enforced.
- Embedding query: deterministic doctor all caller closure sandbox atomic transaction rollback fixed point thin cli
- AST query: DeterministicDoctorImpact DeterministicDoctorTransaction DeterministicDoctorFixedPoint DoctorService DoctorOperationResult

## LPR-G110 Benchmark and release the no-LLM deterministic doctor

- Status: active
- Parent: LPR-G000
- Depends on: LPR-G070, LPR-G100
- Priority: P0
- Track: deterministic-doctor-release
- Bundle: agent-supervisor/tactician-hammer-logic-repair/deterministic-doctor-release
- Parallel lane: lpr-doctor-release
- Conflict policy: Own doctor fixtures, metrics, rollout, operator documentation and joined terminal validation; preserve the existing lifecycle doctor and protected control plane, keep report-only as default, and do not require optional retrieval/prover availability for supervisor startup.
- Resource class: cpu-large
- Token class: large
- Goal: Measure deterministic diagnosis, retrieval, proof, cache, transform, impact, transaction, rollback, and fixed-point behavior on real-checkout adversarial fixtures; stage report-only through sandbox and narrow auto; and release only when the VFS and deterministic-doctor branches join with every no-LLM safety floor at zero.
- Evidence: LPR-040, LPR-041, LPR-042
- Outputs: adversarial deterministic-doctor benchmark corpus and metrics, staged rollout controls and operator guide, joined release validator and end-to-end replay suite
- Validation: Run every positive and adversarial fixture twice, validate exact CIDs/receipts and zero model calls, exercise report/plan/sandbox/narrow-auto/rollback, then run VFS plus deterministic-doctor terminal validation with cold imports and absent optional providers.
- Acceptance: Fixtures cover rename/move, imports/exports/registration, two-to-three-argument propagation across aliases/wrappers/methods, constructor/context threading, finite adapters/schema/artifacts, and ambiguous/stale/poisoned/forged/dynamic/generated/native/public/cross-root cases; correct repair or abstention is deterministic; model routes patched to raise are never invoked; missed caller, authority promotion, stale proof/CID, path/sandbox escape, partial mutation, rollback failure, nondeterminism and false fixed-point rates are zero; rollout defaults to report-only with narrow auto off and independent kill switch; LPR-042 proves both branches and is the unique terminal.
- Gap task: Establish measurable efficacy and fail-closed safety before the deterministic doctor may repair a checkout automatically.
- Refinement: Correct abstention is part of efficacy; approximate recall and cache hit rate never override a zero safety floor.
- Embedding query: deterministic doctor benchmark no llm report plan sandbox narrow auto rollback joined release
- AST query: DeterministicDoctorBenchmark DeterministicDoctorMetrics DeterministicDoctorRollout DeterministicDoctorRelease
