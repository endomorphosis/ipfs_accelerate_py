# Proof-Carrying Procedure Compiler Objective Heap

This is the sealed bootstrap projection for
`agent-supervisor-proof-carrying-procedure-compiler-v1`. After materialization,
the DuckDB intent repository is status authority; this document remains a
deterministic human-readable export. Status never establishes completion.

## PCPC-G000 Compile repeated validated reasoning into safe reusable procedures

- Status: active
- Review only: true
- Parent:
- Depends on:
- Fib priority: 1
- Track: program-control
- Priority: P0
- Bundle: agent-supervisor/procedure-compiler/root
- Parallel lane: pcpc-control
- Resource class: cpu-large
- Goal: Implement ProofCarryingProcedureCompiler so recurring supervisor work can execute through verified bounded procedures and typed residual holes without expanding authority or reducing validation.
- Subgoals: PCPC-G010, PCPC-G020, PCPC-G030, PCPC-G040
- Evidence: PCPC-G010, PCPC-G020, PCPC-G030, PCPC-G040, PCPC-031
- Evidence criteria: Every child goal has current-tree admitted evidence; safety and correctness floors pass; the final release report binds the actual merged tree and exact rollback target.
- Evidence source policy: Current Git identities, typed DuckDB records, external authorization/effect observations, admitted test/proof/adversarial receipts, held-out evaluation, and post-merge qualification are authoritative; procedure assertions, CIDs alone, models, prose, task status, and DuckLake projections are not.
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_PROCEDURE_COMPILER_PLAN.md, ipfs_accelerate_py/agent_supervisor/procedure_compiler, test/api/procedure_compiler, benchmarks/agent_supervisor/procedure_compiler
- Predicted files: ipfs_accelerate_py/agent_supervisor/procedure_compiler, test/api/procedure_compiler, docs/architecture, scripts, benchmarks/agent_supervisor/procedure_compiler
- Interfaces: ProofCarryingProcedureCompiler@1, ProcedureIR@1, ProcedureInterpreter@1, ProcedureRegistry@1
- Validation: python scripts/validate_agent_supervisor_procedure_compiler_board.py --check-all && python -m pytest -q test/api/procedure_compiler
- Acceptance: PCPC-000 through PCPC-031 have exact current-tree evidence from declared producers; no safety floor is violated; remaining caveats and blockers are typed and no blocked feature is represented as available.
- Gap task: Aggregate child evidence and obtain independent release disposition; root review implements no subsystem behavior.
- Refinement: Successful reuse improves cost but never substitutes for authority, evidence freshness, required validation, proof, review, or actual post-merge observation.
- Embedding query: proof carrying procedure compiler deterministic recurring task reasoning amortization
- AST query: ProofCarryingProcedureCompiler ProcedureInterpreter ProcedureRegistry AdaptivePlanner SupervisorControlService
- Conflict policy: Child goals own implementation; PCPC-031 alone owns the joined qualification and residual-gap report.

## PCPC-G010 Establish bounded procedure contracts and deterministic runtime

- Status: active
- Review only: false
- Parent: PCPC-G000
- Depends on:
- Fib priority: 1
- Track: p0-runtime
- Priority: P0
- Bundle: agent-supervisor/procedure-compiler/p0
- Parallel lane: pcpc-p0
- Resource class: cpu-large
- Goal: Seal the exact baseline, inventory reusable authorities, define every closed artifact, validate ProcedureIR, execute it deterministically through trusted adapters, and project/calibrate repository transitions.
- Evidence: PCPC-000, PCPC-001, PCPC-002, PCPC-003, PCPC-004, PCPC-005, PCPC-006, PCPC-007, PCPC-008
- Evidence criteria: Canonical round trips, unknown-field and large-artifact rejection, bounded control flow, certificate/authority/effect/scope/confirmation refusal, concurrent invocation exclusion, idempotent replay, checkpoint recovery, rollback, idle stability, world delta, and calibration tests pass on the exact current tree.
- Evidence source policy: Git object identities, canonical CID implementation, existing authority interfaces, hermetic independent test producers, and typed prerequisite statuses are primary; similarly named files, plans, and generated receipts are not.
- Outputs: ipfs_accelerate_py/agent_supervisor/procedure_compiler/contracts.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/procedure_ir.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/interpreter.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/world_model.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/transition_model.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/procedure_compiler, test/api/procedure_compiler, docs/architecture/procedure_compiler_inventory
- Interfaces: ProcedureCompilerContracts@1, ProcedureIR@1, ProcedureInterpreter@1, RepositoryWorldModel@1, TransitionModel@1
- Validation: python -m pytest -q test/api/procedure_compiler/test_contracts.py test/api/procedure_compiler/test_procedure_ir.py test/api/procedure_compiler/test_interpreter.py test/api/procedure_compiler/test_world_model.py test/api/procedure_compiler/test_transition_model.py
- Acceptance: P0 is import-safe and hermetic; no arbitrary execution is expressible; external admission is required; world predictions cannot overclaim; current-tree qualification exists before database tasks advance.
- Gap task: Supply only narrow typed compatibility ports for prerequisite caveats; do not rebuild missing successor-generation systems.
- Refinement: P0 certificate admission is fail-closed until PCPC-017; only explicitly labeled read-only/test shadow execution may use test admissions.
- Embedding query: procedure IR interpreter world state transition calibration bounded contracts
- AST query: CanonicalContract ControlBounds SupervisorWorldSnapshot RepositorySnapshot DatabaseTaskSource
- Conflict policy: Contracts own shared wire shapes; interpreter and world/transition modules develop in parallel after those shapes stabilize.

## PCPC-G020 Mine, generalize, synthesize, and certify procedures

- Status: active
- Review only: false
- Parent: PCPC-G000
- Depends on: PCPC-G010
- Fib priority: 2
- Track: mining-synthesis
- Priority: P0
- Bundle: agent-supervisor/procedure-compiler/synthesis
- Parallel lane: pcpc-synthesis
- Resource class: cpu-large
- Goal: Normalize admitted trajectories, discover safe task families, anti-unify structures, mine non-vacuous candidate specifications, run bounded CEGIS/adversarial refinement, and issue independently verified certificates.
- Evidence: PCPC-009, PCPC-010, PCPC-011, PCPC-012, PCPC-013, PCPC-014, PCPC-015, PCPC-016, PCPC-017
- Evidence criteria: Held-out, negative, boundary, vacuity, mutant, replay, and budget-exhaustion fixtures produce the required accepted/refused outcomes with no validation loss or unsafe family merge.
- Evidence source policy: Independently admitted episodes, current types/contracts/tests/proofs/runtime checks, rejected traces, failure signatures, and adversarial counterexamples are primary; frequency, embeddings, model confidence, and documentation not marked authoritative are nomination only.
- Outputs: ipfs_accelerate_py/agent_supervisor/procedure_compiler/trajectory.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/task_family.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/anti_unification.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/specification_mining.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/cegis.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/verifier.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/certificate.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/procedure_compiler, test/api/procedure_compiler, benchmarks/agent_supervisor/procedure_compiler
- Interfaces: TrajectoryNormalizer@1, TaskFamilyDiscovery@1, ProcedureAntiUnifier@1, ProcedureCegis@1, ProcedureCertificateVerifier@1
- Validation: python -m pytest -q test/api/procedure_compiler -k 'trajectory or family or boundary or anti_unification or specification or vacuity or cegis or certificate'
- Acceptance: Unsafe generalizations and boundary near-matches are rejected; counterexamples persist; synthesis terminates within declared bounds; certificates bind all required current identities and evidence.
- Gap task: Adapt the existing assurance API and sealer leaves; resolve only the imports needed for their canonical public behavior.
- Refinement: A model may propose a declarative sketch but cannot certify or promote it; passing observed traces never proves completeness.
- Embedding query: validated trajectory task family anti unification specification mining cegis procedure certificate
- AST query: AssuranceCampaignApi IncrementalVerificationPlanner IncrementalProofSealer FormalAssuranceCegis
- Conflict policy: Trajectory/family and specification work are file-disjoint; CEGIS joins both; certificate verification serializes their public contracts.

## PCPC-G030 Integrate verified procedures into planning and bounded execution

- Status: active
- Review only: false
- Parent: PCPC-G000
- Depends on: PCPC-G020
- Fib priority: 3
- Track: planning-execution
- Priority: P1
- Bundle: agent-supervisor/procedure-compiler/integration
- Parallel lane: pcpc-integration
- Resource class: cpu-io-large
- Goal: Store and lifecycle procedures, expose them as compatible planning operators, resolve typed holes economically, synthesize bounded deterministic tools, run safe experiments/transfer checks, and guide isolated repair.
- Evidence: PCPC-018, PCPC-019, PCPC-020, PCPC-021, PCPC-022, PCPC-023, PCPC-024, PCPC-025, PCPC-026, PCPC-027
- Evidence criteria: Registry CAS/rollback/drift, planner matching/composition, hole validation/distillation, translation equivalence, transfer refusal, and autonomous repair scope/validation tests pass with no authority escalation.
- Evidence source policy: Promoted current certificates, registry CAS, existing planner/worktree/lease/merge authorities, held-out results, independent observations, and exact route policy are authoritative; match score, local-model confidence, and experiment suggestion are not.
- Outputs: ipfs_accelerate_py/agent_supervisor/procedure_compiler/registry.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/planner_adapter.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/hole_resolution.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/distillation.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/tool_synthesis.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/experiments.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/transfer.py, ipfs_accelerate_py/agent_supervisor/procedure_compiler/recovery.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/procedure_compiler, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/autonomous_repair, test/api/procedure_compiler
- Interfaces: ProcedureRegistry@1, ProcedurePlannerAdapter@1, HoleResolver@1, DeterministicToolDsl@1, ProcedureTransferGate@1
- Validation: python -m pytest -q test/api/procedure_compiler -k 'registry or planner or hole or distillation or tool or experiment or transfer or drift or repair or recovery'
- Acceptance: Planner order and composition rules are exact; typed holes remain candidates; unsafe transfer and generated tools are refused; repair stays in declared scope and cannot merge above policy ceiling.
- Gap task: Resolve the committed AdaptivePlanner import incompatibility (`HAMMER_TRACE_SCHEMA` is undefined in `multi_prover_router.py`) through a separately reviewed narrow port before PCPC-019 can be admitted.
- Refinement: Until planner compatibility qualifies, registry/hole/tool/transfer work continues independently and planner integration remains typed unavailable.
- Embedding query: procedure registry hierarchical planner typed holes distillation generated tools transfer repair recovery
- AST query: AdaptivePlanner AutonomousRepairEngine WorktreeLifecycleStore LeaseCoordinator MergeTrain
- Conflict policy: Registry owns lifecycle state; planner adapter owns no planner authority; repair integration lands only after certificate and boundary gates.

## PCPC-G040 Expose, benchmark, qualify, and release the compiler

- Status: active
- Review only: false
- Parent: PCPC-G000
- Depends on: PCPC-G030
- Fib priority: 5
- Track: product-qualification
- Priority: P1
- Bundle: agent-supervisor/procedure-compiler/release
- Parallel lane: pcpc-release
- Resource class: cpu-large
- Goal: Add typed Python/control/CLI/MCP surfaces, freeze the complete benchmark, enforce promotion/release metrics and safety gates, and produce exact current-tree human/machine qualification reports.
- Evidence: PCPC-028, PCPC-029, PCPC-030, PCPC-031
- Evidence criteria: Surface parity, authorization/idempotency/dry-run/fencing, frozen partition disjointness, all non-compensable gates, denominator-preserving metrics, and post-merge requalification pass.
- Evidence source policy: Direct typed service results, CLI/MCP parity tests, frozen manifests, actual provider/token/cost receipts, current proof/test/adversarial evidence, and authorized release review are primary; CLI text parsing, benchmark self-report, board status, and partial-success denominators are not.
- Outputs: ipfs_accelerate_py/agent_supervisor/procedure_compiler/cli.py, ipfs_accelerate_py/agent_supervisor/control, benchmarks/agent_supervisor/procedure_compiler, docs/architecture/procedure_compiler_inventory, final qualification reports
- Predicted files: ipfs_accelerate_py/agent_supervisor/procedure_compiler, ipfs_accelerate_py/agent_supervisor/control, ipfs_accelerate_py/cli_entry.py, benchmarks/agent_supervisor/procedure_compiler, test/api/procedure_compiler, docs/architecture
- Interfaces: ProcedureControlService@1, ProcedureCLI@1, ProcedureMCPSurface@1, ProcedureCompilerReleaseReceipt@1
- Required public reads: procedures.capabilities, procedures.list, procedures.get, procedures.explain, procedures.match, procedures.registry_status, procedures.task_families, procedures.counterexamples, procedures.drift, procedures.metrics, procedures.shadow_results, procedures.synthesis_status, procedures.world_model_status
- Required public mutations: procedures.synthesize, procedures.evaluate, procedures.promote, procedures.rollback, procedures.revoke, procedures.quarantine, procedures.run_shadow, procedures.cancel, procedures.request_review
- Required numeric gates: >=50% lower median planning tokens; >=40% lower total model input tokens; >=60% fewer remote-model calls; >=70% lower retry tokens; >=60% recurring tasks without a remote model; >=80% deterministic repairs without any model; >=30% accepted benchmark work through procedures; >=25% fewer human interventions; 100% postcondition coverage/validation retention/boundary rejection; 0 unsafe transfer.
- Validation: python -m pytest -q test/api/procedure_compiler && python scripts/validate_agent_supervisor_procedure_compiler_board.py --check-all
- Acceptance: Every exact `procedures.*` operation above has typed service parity; benchmark partitions are frozen/disjoint; all measured gates enforce the exact numeric thresholds with complete denominators; the release report states promotion eligibility and blockers honestly.
- Gap task: Obtain the missing qualified autonomous-meta-controller comparison baseline or mark every dependent token/autonomy gate unevaluable without promoting.
- Refinement: A partial successful benchmark cannot establish savings or release; live DuckLake projection health remains a separate non-authoritative result.
- Embedding query: procedure compiler control service cli mcp benchmark promotion release qualification
- AST query: SupervisorControlService cli_entry MCP procedure_compiler metrics release
- Conflict policy: Public surface changes serialize; benchmark and metrics work are parallel; PCPC-031 alone joins and reports the final disposition.
