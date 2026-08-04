# Agent Supervisor Control-Plane Planner/Doctor V2 Taskboard (CPD)

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix
`## CPD-`. Companion objective heap:
`agent_supervisor_control_plane_planner_doctor_v2.objectives.md`. Normative
plan: `AGENT_SUPERVISOR_CONTROL_PLANE_PLANNER_DOCTOR_V2_PLAN.md`. Scheduler
profile: `config/agent_supervisor_control_plane_planner_doctor_v2_scheduler.json`.

## Execution doctrine

- This is a narrow integration and qualification successor to PDR/ASI. Import
  their current-tree receipts and modules; do not duplicate their analyzers.
- The plan, objective heap, seed board, scheduler/promotion policy, imported
  authority policies, holdout, and oracle are protected operator inputs.
- Generated successor work goes to a separate DuckDB/CAS source and never
  appends to this seed board.
- Preview and diagnosis are read-only. Every effect needs current roots,
  separate permit, lease, fence, idempotency, expected effects, checkpoint,
  rollback, and independent observed-effects validation.
- Retrieval/model/cache metadata cannot grant proof, mutation, completion,
  rollout, or promotion authority.
- Grok 4.5 is primary. Terra medium is a hard-quota-authorized fallback only.
- Completed task status is not objective-completion authority.

## Parallel lanes

| Lane | Ownership |
| --- | --- |
| `cpd-foundation` | baseline and imported policy authority |
| `cpd-contracts` | prompt artifacts, bundle, launch and snapshot contracts |
| `cpd-planner` | intent/task/candidate/parallel/launch compilation and admission |
| `cpd-control` | bootstrap service, projections, transports and lifecycle |
| `cpd-doctor` | reasoning invocation, mutation context and diagnosis |
| `cpd-repair` | repair portfolios and fixed point |
| `cpd-proof` | CID/Merkle/signature/ZKP lineage |
| `cpd-benchmark` | live paired runner and quality oracle |
| `cpd-telemetry` | clock/token/process/GPU/resource attribution |
| `cpd-self-improvement` | epochs, refill and rollout |
| `cpd-release` | E2E, chaos, operations and terminal release |

---

## CPD-000 Seal the CPD successor plan, objectives, taskboard, and scheduler profile

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bootstrap
- Depends on:
- Goal id: CPD-G000
- Outputs: docs/architecture/AGENT_SUPERVISOR_CONTROL_PLANE_PLANNER_DOCTOR_V2_PLAN.md, docs/architecture/agent_supervisor_control_plane_planner_doctor_v2.objectives.md, docs/architecture/agent_supervisor_control_plane_planner_doctor_v2.todo.md, config/agent_supervisor_control_plane_planner_doctor_v2_scheduler.json
- Validation: python -m json.tool config/agent_supervisor_control_plane_planner_doctor_v2_scheduler.json
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/bootstrap
- Parallel lane: cpd-foundation
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/AGENT_SUPERVISOR_CONTROL_PLANE_PLANNER_DOCTOR_V2_PLAN.md, docs/architecture/agent_supervisor_control_plane_planner_doctor_v2.objectives.md, docs/architecture/agent_supervisor_control_plane_planner_doctor_v2.todo.md, config/agent_supervisor_control_plane_planner_doctor_v2_scheduler.json, .gitignore
- Interfaces: CPDPlan@1, CPDObjectiveHeap@1, CPDTaskboard@1, CPDSchedulerConfig@1
- Conflict policy: These seed artifacts become protected; automatic work cannot edit them.
- Preconditions: PDR and ASI artifacts are present for read-only audit.
- Effects: Creates one machine-ingestible successor namespace without changing the protected PDR board.
- Evidence subset: plan/task/goal/config path identity and parser conformance
- Acceptance: Artifacts agree on namespace, prefixes, paths, dependencies, authority defaults, provider policy, resource bounds, and protected paths; task IDs are unique and the scheduler loader accepts the profile.

## CPD-001 Revalidate current PDR/ASI receipts and reproduce the raw-prompt bootstrap gap

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline
- Depends on: CPD-000
- Goal id: CPD-G010
- Outputs: docs/architecture/agent_supervisor_control_plane_planner_doctor_baseline.md, test/api/test_agent_supervisor_control_plane_bootstrap_baseline.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_bootstrap_baseline.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/baseline
- Parallel lane: cpd-foundation
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 5400
- Predicted files: docs/architecture/agent_supervisor_control_plane_planner_doctor_baseline.md, test/api/test_agent_supervisor_control_plane_bootstrap_baseline.py
- Interfaces: CPDCapabilityImportReceipt@1, PromptBootstrapGapReceipt@1
- Conflict policy: Read PDR/ASI artifacts and public services only; do not change their status or evidence.
- Preconditions: CPD seed artifacts are sealed.
- Effects: Establishes the exact current-tree shipped/wired/missing inventory and a failing-before-fix conformance fixture.
- Evidence subset: PDR/ASI artifact CIDs, default handler construction, Python/CLI/MCP raw-prompt result, inert `--start`, launch-profile behavior
- Acceptance: Reproduce that raw `workflow-preview` returns a sparse `plan_request_present=false` proposal; prove the closed catalog cannot carry the required plan request; distinguish the separate working PromptSupervisorService from default control wiring; inventory all imported symbolic, Doctor, repair, projection, lifecycle, benchmark and provider-route capabilities with current CIDs.

## CPD-002 Bind imported authority, threat, benchmark, holdout, and provider policies

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority
- Depends on: CPD-000
- Goal id: CPD-G010
- Outputs: config/agent_supervisor_control_plane_policy_imports.json, test/api/test_agent_supervisor_control_plane_policy_imports.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_policy_imports.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/policy-imports
- Parallel lane: cpd-foundation
- Resource class: cpu-small
- Resource stage: validation
- Implementation timeout seconds: 5400
- Predicted files: config/agent_supervisor_control_plane_policy_imports.json, test/api/test_agent_supervisor_control_plane_policy_imports.py
- Interfaces: CPDPolicyImportManifest@1
- Conflict policy: Import operator-owned PDR policies by CID; never copy or weaken them.
- Preconditions: CPD seed plan states the non-compensable authority boundary.
- Effects: Pins policy roots and defines which separate permits are required at preview/apply/start/mutation/benchmark/rollout/promotion.
- Evidence subset: authority policy, threat model, benchmark manifest/seal, holdout roots, provider failure policy, protected path population
- Acceptance: Manifest pins and verifies the existing operator-sealed roots for the repository-self corpus and policies without granting new authority; candidate scope excludes all policy/oracle inputs; Grok 4.5 is primary and Terra medium is eligible only under a verified durable hard-quota receipt; any missing, stale or changed operator seal blocks rather than being regenerated.

## CPD-010 Implement the authorized content-addressed prompt artifact resolver

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prompt-contracts
- Depends on: CPD-000
- Goal id: CPD-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/prompt_artifact_resolver.py, test/api/test_agent_supervisor_prompt_artifact_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_artifact_resolver.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/prompt-artifacts
- Parallel lane: cpd-contracts
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/prompt_artifact_resolver.py, test/api/test_agent_supervisor_prompt_artifact_resolver.py
- Interfaces: PromptIntentEnvelope@1, PromptArtifactResolver@1, PromptArtifactReceipt@1
- Conflict policy: Extend PromptSource/PromptWorkflow contracts through checked adapters; do not put bodies into OperationRequest or receipts.
- Preconditions: CPD prompt/body authority invariants are fixed.
- Effects: Atomically stores inline/file/stdin prompt bytes in an allowlisted artifact store and resolves bounded verified bytes from body-free handles.
- Evidence subset: prompt CID, artifact handle, media type, byte count, provenance, redaction/secret-scan receipt, resolver policy root
- Acceptance: Exact CID verification and UTF-8/size bounds hold; CLI stdin/file can persist an ephemeral authorized artifact; MCP cannot resolve arbitrary paths; symlink/path escape, tampering, secret logging, cross-caller replay, unavailable store and stale handle fail closed; import is provider/network free.

## CPD-011 Define the control-plane bundle, launch spec, revision, and receipt contracts

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bundle-contracts
- Depends on: CPD-000
- Goal id: CPD-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/control_plane_bundle_contracts.py, test/api/test_agent_supervisor_control_plane_bundle_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_bundle_contracts.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/bundle-contracts
- Parallel lane: cpd-contracts
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/control_plane_bundle_contracts.py, test/api/test_agent_supervisor_control_plane_bundle_contracts.py
- Interfaces: ControlPlaneBundle@1, SupervisorLaunchSpec@1, ControlPlaneRevision@1, ControlPlanePreviewReceipt@1, ControlPlaneMaterializationReceipt@1, SupervisorStartReceipt@1
- Conflict policy: Reference existing PromptGoalGraph, PlanRevision, ParallelExecutionPlan, LifecycleProfile and task-source records by checked identities rather than cloning them.
- Preconditions: CPD contract inventory is available in the plan.
- Effects: Defines one closed immutable manifest for all initial control-plane and launch inputs with separate mutable status projections.
- Evidence subset: goals/tasks/obligations/evidence/parallel/projection/scheduler/provider/launch roots, expected effects, required permits, CID/Merkle identity
- Acceptance: Canonical round trips and stable content IDs hold; unknown/duplicate fields, excessive counts/depth/bytes, cycles, body/secret values, shell command strings, invalid paths/env names/resources/providers, mutable status in semantic CIDs, tampering and replay fail; argv are bounded arrays and secret values are opaque handles only.

## CPD-012 Compose the current repository/capability snapshot for control-plane creation

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: snapshot
- Depends on: CPD-001, CPD-010, CPD-011, CPD-013
- Goal id: CPD-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_snapshot_factory.py, test/api/test_agent_supervisor_control_plane_snapshot_factory.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_snapshot_factory.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/snapshot
- Parallel lane: cpd-contracts
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_snapshot_factory.py, test/api/test_agent_supervisor_control_plane_snapshot_factory.py
- Interfaces: ControlPlaneSnapshotFactory@1, ControlPlaneCapabilityImportReceipt@1
- Conflict policy: Compose RepositoryReasoningSnapshot and PlanningAnalysisFactory; no second repository index or graph.
- Preconditions: Current-tree capability import and canonical bundle contracts exist.
- Effects: Builds the exact superproject/submodule/dirty/task/policy/toolchain/provider/index view used by prompt compilation.
- Evidence subset: tracked/staged/modified/deleted/renamed/admitted-untracked paths, recursive gitlinks, AST/program/contract/value/KG/BM25/vector/proof/cache handles and health
- Acceptance: Real checkouts are enumerated without importing target code; required capability loss abstains and optional loss becomes debt; stale/unstable/wrong-tree/secret/symlink escape and poisoned or dimension-drift indexes fail or degrade explicitly; current roots bind every later receipt.

## CPD-013 Make repository-root and capability assurance independently truthful

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-authority
- Depends on: CPD-001, CPD-011
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_strategy_registry.py, test/api/test_agent_supervisor_control_plane_root_capability_authority.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_root_capability_authority.py test/api/test_agent_supervisor_planning_analysis_factory.py test/api/test_agent_supervisor_analysis_strategy_registry.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/evidence-authority
- Parallel lane: cpd-doctor
- Resource class: cpu-medium
- Resource stage: validation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_strategy_registry.py, test/api/test_agent_supervisor_control_plane_root_capability_authority.py, test/api/test_agent_supervisor_planning_analysis_factory.py, test/api/test_agent_supervisor_analysis_strategy_registry.py
- Interfaces: IndependentRootBinding@1, ExecutedCapabilityReceipt@1
- Conflict policy: Correct existing authority logic in place; do not change roots to accommodate fixtures or infer execution from import/lazy/declaration state.
- Preconditions: The baseline reproduces current root/capability behavior and canonical bundle roots are defined.
- Effects: Independently recomputes the live repository forest before comparison and derives assurance only from executed, schema-conformant, exact-root provider receipts.
- Evidence subset: independently observed tree/repository/dirty/gitlink roots, provider execution input/output/version/health/bounds and certificate identity
- Acceptance: Wrong expected tree, repository ID, dirty root, gitlink, parser/toolchain or provider result is rejected adversarially; repository mismatch is never ignored; LAZY/boolean/declarative capability states cannot satisfy an obligation or inherit `max_assurance`; placeholder receipts leave explicit frontiers; overclaims are zero and stale-root rejection is 100% in the protected cases.

## CPD-014 Compose the live AST-to-semantic-graph and flow-analysis pipeline

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: semantic-graphs
- Depends on: CPD-012, CPD-013
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/semantic_graph_pipeline.py, test/api/test_agent_supervisor_semantic_graph_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_semantic_graph_pipeline.py test/api/test_agent_supervisor_program_graph.py test/api/test_agent_supervisor_value_provenance_graph.py test/api/test_agent_supervisor_program_dependency_graph.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/semantic-graphs
- Parallel lane: cpd-doctor
- Resource class: cpu-large
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/semantic_graph_pipeline.py, test/api/test_agent_supervisor_semantic_graph_pipeline.py
- Interfaces: SemanticGraphPipeline@1, SemanticGraphCoverageReceipt@1
- Conflict policy: Compose existing AST adapters, ProgramGraph, ValueProvenanceGraph, ProgramDependencyGraph and flow/security graphs; no lexical substitute may claim semantic graph assurance.
- Preconditions: Exact snapshot and truthful capability receipts are available.
- Effects: Builds current-root AST/symbol/type/call/CFG/reaching-def/def-use/value-provenance/dependency/contract/flow graph handles and explicit unsupported-language/open-world frontiers.
- Evidence subset: source/parser CIDs, nodes/edges/provenance, call/def-use/taint reachability, dynamic dispatch assumptions, graph completeness and cache dependencies
- Acceptance: Default production construction builds real graphs rather than requiring caller injection; planted call/def-use/taint/contract paths are found with measured precision/recall; unsupported Go/Rust/C/C++ or unresolved dynamic behavior is a typed frontier until a certified adapter exists; lexical overlap cannot satisfy AST-impact obligations; incremental rebuild invalidates affected dependents.

## CPD-015 Build true BM25, vector, knowledge-graph, and exact-fact evidence composition

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: retrieval
- Depends on: CPD-012, CPD-013
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_evidence_retrieval.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_retrieval.py, test/api/test_agent_supervisor_control_plane_evidence_retrieval.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_evidence_retrieval.py test/api/test_agent_supervisor_analysis_retrieval.py test/api/test_agent_supervisor_planning_evidence_bundle.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/retrieval
- Parallel lane: cpd-doctor
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_evidence_retrieval.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_retrieval.py, test/api/test_agent_supervisor_control_plane_evidence_retrieval.py
- Interfaces: ControlPlaneEvidenceRetrieval@1, BM25IndexReceipt@1, HybridEvidenceCoverageReceipt@1
- Conflict policy: Extend the existing retrieval/evidence bundle; rankings nominate evidence and never override exact graph/proof facts.
- Preconditions: Exact snapshot and truthful capability receipt rules exist.
- Effects: Implements document-frequency/document-length BM25, typed KG traversal, root-bound vector search, exact AST/dependency/contract/proof facts, provenance/history and health-aware fusion in the default production factory.
- Evidence subset: corpus/index/config/model/dimension roots, BM25 statistics, vector health, graph paths, exact fact handles, fusion explanation and required-slot coverage
- Acceptance: BM25 is not Jaccard/coverage masquerading as BM25; Recall@k/MRR/nDCG and graph-hop precision are measured; poisoned/stale/constant/non-finite/dimension-drift indexes are disabled; absent injected adapters no longer silently remove live evidence; exact dependencies cannot be outranked away; retrieval reduces context/tokens without lowering required evidence coverage.

## CPD-016 Require executable formal portfolios and reconstructable proof authority

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: formal-authority
- Depends on: CPD-002, CPD-013
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/control_plane_formal_portfolio.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, test/api/test_agent_supervisor_control_plane_formal_portfolio.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_formal_portfolio.py test/api/test_agent_supervisor_multi_prover_router.py test/api/test_agent_supervisor_deterministic_doctor_proof_authority.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/formal-portfolio
- Parallel lane: cpd-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/control_plane_formal_portfolio.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, test/api/test_agent_supervisor_control_plane_formal_portfolio.py, test/api/test_agent_supervisor_multi_prover_router.py
- Interfaces: ControlPlaneFormalPortfolio@1, AuthoritativeProofExecutionReceipt@1
- Conflict policy: Compose ProverMatrixRegistry, DeterministicDoctorHammer, state/protocol/hyperproperty tools and proof kernels; bare runners never become authoritative.
- Preconditions: Imported authority policy and truthful capability receipts exist.
- Effects: Routes obligations through pinned Datalog/CHC/SAT/SMT/MaxSAT/CEGAR/PDR, TLC/Apalache, protocol/hyperproperty and kernel providers with certificates, disagreement handling and reconstruction.
- Evidence subset: executable/provider/toolchain/config CIDs, encoded property/assumptions/bounds, raw result/certificate/counterexample, matrix capability and independent reconstruction receipt
- Acceptance: Capability evidence and matrix pinning are mandatory; bare or arbitrary runner output cannot become PROVED/VERIFIED; required certificate/kernel replay and translation validation pass; disagreement, timeout, unsupported theory, bounded-only result or missing provider is explicit unknown/debt; proof coverage, reconstruction rate, solver time and unknown rate are measured.

## CPD-017 Derive independent IntentIR and CodeIR streams and fail closed on security gaps

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: security-authority
- Depends on: CPD-002, CPD-012, CPD-013
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/independent_intent_code_security.py, ipfs_accelerate_py/agent_supervisor/security_contract_analysis.py, test/api/test_agent_supervisor_independent_intent_code_security.py
- Validation: python -m pytest test/api/test_agent_supervisor_independent_intent_code_security.py test/api/test_agent_supervisor_security_contract_analysis.py test/api/test_agent_supervisor_cve_security_gate.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/security
- Parallel lane: cpd-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/independent_intent_code_security.py, ipfs_accelerate_py/agent_supervisor/security_contract_analysis.py, test/api/test_agent_supervisor_independent_intent_code_security.py
- Interfaces: IndependentIntentCodeSecurityGate@1, IntentCodeRelationalProof@1
- Conflict policy: Reuse IntentIR/SecurityIR/FlowGraph/CVE gate; never substitute code effects for missing intent or auto-union coverage IDs.
- Preconditions: Exact prompt intent, source snapshot and operator security policy roots exist independently.
- Effects: Derives IntentIR from the prompt/policy path and CodeIR/flows from AST/semantic graphs, then checks forbidden effects, authorization, taint, noninterference/hyperproperties and supply-chain constraints relationally.
- Evidence subset: independent translator/source roots, intent and code predicates/effects/flows, translation coverage, Datalog/SMT/model-check/proof results and counterexamples
- Acceptance: Missing either stream, translator coverage, flow/hyperproperty evidence or independently derived coverage abstains/rejects; identical/circular provenance is detected; planted forbidden intent/code mismatches have 100% protected-case recall; caller/candidate flags cannot mint coverage; CVE/SecurityIR gate is invoked in live plan, mutation and fixed-point paths.

## CPD-020 Compile prompt intent into bounded goals and subgoals

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: intent-planning
- Depends on: CPD-001, CPD-010, CPD-011
- Goal id: CPD-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/prompt_intent_compiler.py, test/api/test_agent_supervisor_prompt_intent_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_intent_compiler.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/intent
- Parallel lane: cpd-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/prompt_intent_compiler.py, test/api/test_agent_supervisor_prompt_intent_compiler.py
- Interfaces: PromptIntentCompiler@1, NormalizedIntent@1, GoalHierarchyProposal@1
- Conflict policy: Adapt PromptGoalPlanner and PDR query planner; deterministic parsing/templates precede model use.
- Preconditions: Verified prompt artifact and bundle contracts exist.
- Effects: Converts prompt plus repository evidence into outcomes, constraints, non-goals, assumptions, questions, AND/OR goal hierarchy and evidence requirements.
- Evidence subset: prompt CID, normalized clauses, repository capabilities/contracts, policy/IR roots, uncertainty and provenance
- Acceptance: Every material prompt clause maps to a goal, constraint, non-goal, question or explicit rejection; goals have parentage, completion queries, proof/evidence requirements and bounds; ambiguous authority-changing choices block; deterministic output is stable; residual LLM calls are capped, schema-strict and proposal-only.

## CPD-021 Compile executable tasks, obligations, acceptance, and minimal evidence needs

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: task-planning
- Depends on: CPD-012, CPD-020, CPD-040
- Goal id: CPD-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/control_plane_task_compiler.py, test/api/test_agent_supervisor_control_plane_task_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_task_compiler.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/tasks
- Parallel lane: cpd-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/control_plane_task_compiler.py, test/api/test_agent_supervisor_control_plane_task_compiler.py
- Interfaces: ControlPlaneTaskCompiler@1, ControlPlaneTaskRecord@1, TaskObligationBinding@1
- Conflict policy: Reuse PromptTaskRecordV2, ObligationGraphCompiler and minimal context contracts via versioned bridges.
- Preconditions: Goal hierarchy, exact snapshot and reasoning strategy invocation are available.
- Effects: Produces tasks with semantic CIDs, dependencies, scope, outputs/read-write sets, acceptance, validation argv, resources, providers, conflicts, proof/security gates and evidence queries.
- Evidence subset: goal clauses, AST/KG dependency and impact slices, contracts/tests/build graph, formal/security properties, capability/budget roots
- Acceptance: Producer and acceptance closure is complete; no dangling/cyclic dependency, unbounded command, path escape, undeclared effect, impossible resource, fake lane, self-validating test, hidden-oracle access or uncovered required obligation is admitted; context requirements are path/evidence bounded.

## CPD-022 Generate, critique, and replan a bounded control-plane portfolio

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: candidate-planning
- Depends on: CPD-020, CPD-021
- Goal id: CPD-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/control_plane_candidate_planner.py, test/api/test_agent_supervisor_control_plane_candidate_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_candidate_planner.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/candidates
- Parallel lane: cpd-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/control_plane_candidate_planner.py, test/api/test_agent_supervisor_control_plane_candidate_planner.py
- Interfaces: ControlPlaneCandidatePlanner@1, ControlPlaneCriticReceipt@1
- Conflict policy: Compose PDR SymbolicCandidatePlanner and PlanCritic; do not accept model prose or confidence as evidence.
- Preconditions: Candidate task/obligation compiler exists.
- Effects: Creates deterministic alternatives, optional bounded residual LLM alternatives, counterexamples and repaired revisions, then selects or abstains.
- Evidence subset: constraint satisfaction, evidence coverage, satisfiability, dependency/conflict/resource feasibility, critical path, context/token cost, proof/security debt
- Acceptance: Candidate count and repair rounds are enforced; deterministic candidates run first; critic catches planted cycles/conflicts/missing checks/infeasible resources/unsafe effects; counterexamples change or reject the plan; exhaustion yields a typed blocker; selection is reproducible from bound inputs.

## CPD-023 Compile the parallel execution plan and exact supervisor launch specification

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: launch-planning
- Depends on: CPD-011, CPD-021, CPD-022
- Goal id: CPD-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/supervisor_launch_compiler.py, test/api/test_agent_supervisor_launch_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_launch_compiler.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/launch-compiler
- Parallel lane: cpd-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/supervisor_launch_compiler.py, test/api/test_agent_supervisor_launch_compiler.py
- Interfaces: SupervisorLaunchCompiler@1, SupervisorLaunchSpec@1, LaunchValidationReceipt@1
- Conflict policy: Bridge ParallelExecutionPlan, scheduler-config loader, LifecycleProfile and multi-supervisor argv builders; do not invent a process runner.
- Preconditions: Candidate tasks, resources, providers, conflicts and launch contracts exist.
- Effects: Computes ready waves and compiles shell-free argv, cwd/roots, task sources, lanes, timeouts, validation workers, submodules, env-name/secret handles, CUDA allocation, health/restart/kill switch and expected process identity.
- Evidence subset: dependency/conflict graph, resource inventory, scheduler parser, lifecycle parser, provider failure policy, GPU health/capacity, protected paths
- Acceptance: Round-trip through the real parsers succeeds; argv injection, broad paths, secret values, infeasible CPU/RAM/disk/GPU/process limits, false lane width and mutable policy fields fail; Grok 4.5 is primary and Terra medium appears only as a hard-quota-gated route; no effect occurs during compilation.

## CPD-024 Independently admit the complete control-plane bundle

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-admission
- Depends on: CPD-002, CPD-012, CPD-021, CPD-022
- Goal id: CPD-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/control_plane_admission.py, test/api/test_agent_supervisor_control_plane_admission.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_admission.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/admission
- Parallel lane: cpd-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/control_plane_admission.py, test/api/test_agent_supervisor_control_plane_admission.py
- Interfaces: ControlPlaneAdmissionService@1, ControlPlaneAdmissionRequest@1, ControlPlaneAdmissionReceipt@1
- Conflict policy: Compose PDR plan admission, formal plan, IR, proof, security and validation gates; reconstruct evidence independently.
- Preconditions: Current snapshot and candidate control-plane portfolio exist.
- Effects: Checks the selected goal/task/obligation/parallel/projection/scheduler/provider/launch graph and either seals a proposal CID or emits typed findings.
- Evidence subset: current roots, evidence coverage, formal-plan conformance, intent/security/legal IR, proof/test obligations, resource/conflict/launch constraints, authority policy
- Acceptance: Caller booleans, expected results, candidate proofs and generated tests cannot certify themselves; stale caches rederive assurance; required missing capability or frontier rejects; all roots and assumptions are bound; admission remains read-only and cannot grant apply/start authority.

## CPD-030 Build the production raw-prompt control-plane bootstrap service

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bootstrap-service
- Depends on: CPD-010, CPD-012, CPD-022, CPD-023, CPD-024
- Goal id: CPD-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/control_plane_bootstrap_service.py, test/api/test_agent_supervisor_control_plane_bootstrap_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_bootstrap_service.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/bootstrap-service
- Parallel lane: cpd-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/control_plane_bootstrap_service.py, test/api/test_agent_supervisor_control_plane_bootstrap_service.py
- Interfaces: ControlPlaneBootstrapService@1, PromptBootstrapRequest@1
- Conflict policy: Unify PlanSupervisorService, PlanCreateService and PromptSupervisorService behind one adapter; preserve their public compatibility aliases.
- Preconditions: Prompt resolver, compiler, launch spec and independent admission are production wired.
- Effects: `preview` resolves a raw prompt artifact, builds typed workflow/create requests, runs the complete symbolic pipeline and returns an admitted bundle without effects.
- Evidence subset: prompt/request/snapshot/query/obligation/candidate/critic/admission/bundle receipts
- Acceptance: Default construction uses repository allowlists and PlanningAnalysisFactory; raw prompt cannot return sparse success; deterministic preview is idempotent and provider-lazy; typed plan requests remain supported; stale roots, missing prompt artifact, rejected plan and unavailable required capability yield closed results with resumable references.

## CPD-031 Materialize admitted bundles atomically into equivalent task sources

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: materialization
- Depends on: CPD-030
- Goal id: CPD-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_materializer.py, test/api/test_agent_supervisor_control_plane_materializer.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_materializer.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/materialization
- Parallel lane: cpd-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_materializer.py, test/api/test_agent_supervisor_control_plane_materializer.py
- Interfaces: ControlPlaneMaterializer@1, ControlPlaneMaterializationReceipt@1
- Conflict policy: Compose existing Markdown/DuckDB/CAS and PlanRevisionStore transactions; task sources remain projections.
- Preconditions: An admitted bundle and separate apply permit, lease, fence and expected effects exist.
- Effects: Writes plan/objective/task/scheduler projections and durable body-free continuation receipts with compare-and-swap and rollback.
- Evidence subset: admitted bundle CID, output policy, before/after projection roots, task CIDs, event cursor, permit/lease/fence/idempotency, rollback checkpoint
- Acceptance: Markdown and DuckDB expose identical semantic task CIDs and readiness; partial writes are resumable or exactly rolled back; changed output paths, stale bundle, conflicting idempotency, lease/fence loss, protected path, body/secret persistence and projection divergence fail closed.

## CPD-032 Expose a stable durable Python control-plane API

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: python-api
- Depends on: CPD-030
- Goal id: CPD-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_control_plane_public_api.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_public_api.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/python-api
- Parallel lane: cpd-control
- Resource class: cpu-small
- Resource stage: implementation
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_control_plane_public_api.py
- Interfaces: ControlPlaneBootstrapClient@1
- Conflict policy: Re-export only stable contracts/facade; preserve cold import and existing package API.
- Preconditions: Canonical bootstrap service exists.
- Effects: Publishes preview/apply/start/bootstrap and receipt-resume methods using one durable artifact/receipt store abstraction.
- Evidence subset: public signatures, lazy import graph, cross-process receipt reload, canonical encoding
- Acceptance: A new process can resume from receipt/CAS handles without raw prompt bodies or process-local state; exact replay performs no provider call or duplicate write/process; import loads no model/network/solver/DB clients; type and exception behavior is documented and tested.

## CPD-033 Wire end-to-end raw-prompt preview/apply/start through the CLI

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: cli
- Depends on: CPD-031, CPD-032
- Goal id: CPD-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/control/control_cli.py, ipfs_accelerate_py/cli.py, test/api/test_agent_supervisor_control_plane_cli.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_cli.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/cli
- Parallel lane: cpd-control
- Resource class: cpu-small
- Resource stage: implementation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/control_cli.py, ipfs_accelerate_py/cli.py, test/api/test_agent_supervisor_control_plane_cli.py
- Interfaces: AgentControlPlaneCLI@1
- Conflict policy: CLI remains a thin canonical adapter and never constructs shell strings or owns authority policy.
- Preconditions: Durable public service and materialization are available.
- Effects: Resolves inline/file/stdin prompt artifacts, emits full preview JSON, accepts explicit apply/start permits, and makes `--start` execute the resumable start stage or removes the misleading flag.
- Evidence subset: canonical request/result parity, prompt artifact receipt, exit codes, stdout/stderr redaction, apply/start effects
- Acceptance: Raw CLI prompt yields a full bundle; secret prompt text is absent from logs/receipts and file/stdin are preferred; preview is forced dry-run; apply/start cannot be conflated; typed JSON remains available; injection, ambiguous sources, arbitrary paths, missing permits, stale roots, partial saga and restart behave deterministically.

## CPD-034 Wire exact lazy MCP parity with server-configured prompt artifacts

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp
- Depends on: CPD-031, CPD-032
- Goal id: CPD-G040
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_control_plane_mcp.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_mcp.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/mcp
- Parallel lane: cpd-control
- Resource class: cpu-small
- Resource stage: implementation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_control_plane_mcp.py
- Interfaces: AgentSupervisorControlPlaneMCPTools@1
- Conflict policy: Generate tools from the shared operation catalog; lazy registration and domain-service parity are mandatory.
- Preconditions: Durable public service and materialization are available.
- Effects: Adds or versions prompt bootstrap preview/apply/start tools using artifact handles and canonical OperationRequest/Result schemas.
- Evidence subset: tool discovery schema, exact Python result CID, lazy import graph, server artifact allowlist and caller authority
- Acceptance: MCP and Python produce byte-equivalent canonical results for the same request; tool discovery starts no providers/processes; server paths and inline secrets are rejected; unknown fields/bounds/authority/replay fail; apply/start remain separate tools and permits.

## CPD-035 Start and supervise a real daemon from an admitted launch spec

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: lifecycle
- Depends on: CPD-023, CPD-031, CPD-032
- Goal id: CPD-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/control/control_plane_lifecycle.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, test/api/test_agent_supervisor_control_plane_lifecycle.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_lifecycle.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/lifecycle
- Parallel lane: cpd-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/control_plane_lifecycle.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, test/api/test_agent_supervisor_control_plane_lifecycle.py
- Interfaces: ControlPlaneLifecycleService@1, SupervisorStartReceipt@1
- Conflict policy: Invoke existing LifecycleOrchestrator/multi-supervisor runner; do not record the caller PID as a synthetic daemon.
- Preconditions: Materialized task source, admitted launch spec and distinct start permit exist.
- Effects: Spawns, owns and observes the real supervisor child; records argv/env-name/process-birth/GPU/profile roots; supports idempotent health, drain, stop, restart and rollback.
- Evidence subset: launch/bundle/task-source roots, permit/lease/fence, actual process tree, PID birth identity, health/readiness/log paths, observed CUDA allocation and effects
- Acceptance: Real daemon reaches ready state and consumes the exact task source; duplicate start is idempotent; PID reuse/cross-run control/stale spec/env secret leakage fail; process death and partial start recover or roll back; stop/kill switch terminates the owned process tree; Grok/Terra routing matches the admitted policy.

## CPD-040 Certify and invoke the shared deterministic reasoning strategy portfolio

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: reasoning-kernel
- Depends on: CPD-002, CPD-014, CPD-015, CPD-016, CPD-017, CPD-046
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_reasoning_kernel.py, test/api/test_agent_supervisor_control_plane_reasoning_kernel.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_reasoning_kernel.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/reasoning-kernel
- Parallel lane: cpd-doctor
- Resource class: cpu-proof-solver
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_reasoning_kernel.py, test/api/test_agent_supervisor_control_plane_reasoning_kernel.py
- Interfaces: ControlPlaneReasoningKernel@1, ReasoningExecutionReceipt@1
- Conflict policy: Invoke AnalysisStrategyRegistry/OperationRegistry providers through checked adapters; do not reimplement their algorithms.
- Preconditions: Current PDR capability inventory, truthful execution receipts, live semantic graphs/retrieval/formal/security adapters, cross-stage caches and CPD authority imports exist.
- Effects: Routes typed properties through AST/CFG/SSA/PDG/call/alias/effect/taint/abstract interpretation, Datalog/CHC/SAT/SMT/model checking, proof/test/fuzz/concolic/mutation/differential, supply-chain, IntentIR/SecurityIR and retrieval strategies.
- Evidence subset: provider capability/version/health/soundness/bounds, inputs/outputs, cache keys, coverage and open frontier
- Acceptance: Tests prove methods are actually executed rather than merely enumerated; independently recomputed roots and execution receipts bind every result; required unavailable providers abstain and optional loss creates debt; learned/retrieval signals remain nomination-only; cold import loads no network/model clients; exact cache hits rederive assurance and invalidation follows semantic dependencies.

## CPD-041 Plan mutation impact and compile minimal implementation context

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mutation-planning
- Depends on: CPD-012, CPD-020, CPD-040
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/mutation_context_planner.py, test/api/test_agent_supervisor_mutation_context_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_mutation_context_planner.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/mutation-context
- Parallel lane: cpd-doctor
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/mutation_context_planner.py, test/api/test_agent_supervisor_mutation_context_planner.py
- Interfaces: MutationPlan@1, MinimalImplementationContext@1, ImpactClosureReceipt@1
- Conflict policy: Reuse query/evidence/obligation/parallel compilers and program graphs; source bodies are fetched only for admitted context slices.
- Preconditions: Exact snapshot, normalized goal and certified reasoning kernel exist.
- Effects: Computes write targets, affected SCC/callers/consumers/build/tests/policies, pre/post obligations, validation DAG and a bounded evidence-ranked context packet.
- Evidence subset: declarations/contracts, AST/CFG/data/call slices, counterexamples, tests/proofs, history and target file CIDs
- Acceptance: Context contains only necessary evidence with inclusion reasons and byte/token caps; every impacted contract/consumer/test has validation or explicit frontier; poisoned retrieval cannot exclude exact dependencies; broad or unproved impact closure abstains; no LLM is needed for deterministic context compilation.

## CPD-042 Bridge live Doctor diagnoses into shared obligations and plan revisions

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-bridge
- Depends on: CPD-040, CPD-041
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_doctor_bridge.py, test/api/test_agent_supervisor_control_plane_doctor_bridge.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_doctor_bridge.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/doctor-bridge
- Parallel lane: cpd-doctor
- Resource class: cpu-large
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_doctor_bridge.py, test/api/test_agent_supervisor_control_plane_doctor_bridge.py
- Interfaces: ControlPlaneDoctorBridge@1, DiagnosisPlanRevisionProposal@1
- Conflict policy: Compose DeterministicDoctorRuntime, causal localization and DiagnosisObligationAdapter; preserve report-only defaults.
- Preconditions: Reasoning kernel and mutation impact planner are available.
- Effects: Runs Doctor on a real checkout, localizes contract/intent/security mismatches and emits append-only obligation/revision proposals with causal and frontier evidence.
- Evidence subset: observed findings, complete/open frontier, causal slice, counterexample, impacted graph, policy/IR roots and proposal CID
- Acceptance: No caller-supplied finding/proof/result becomes authority; no LLM/network is used; duplicate/root-mismatched/stale findings fail; diagnosis distinguishes cause from symptom and complete from open frontier; fixed-point-closed input emits no work; proposals never apply changes or mutate the seed board.

## CPD-043 Generate and independently rank deterministic-first repair portfolios

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repair-synthesis
- Depends on: CPD-022, CPD-042
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/control_plane_repair_portfolio.py, test/api/test_agent_supervisor_control_plane_repair_portfolio.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_repair_portfolio.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/repair-portfolio
- Parallel lane: cpd-repair
- Resource class: cpu-large
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/control_plane_repair_portfolio.py, test/api/test_agent_supervisor_control_plane_repair_portfolio.py
- Interfaces: ControlPlaneRepairPortfolio@1, RepairCandidateAdmissionReceipt@1
- Conflict policy: Compose RepairOperatorRegistry, ProgramRepairSynthesis and RepairCandidatePortfolio; LLM output remains an untrusted candidate.
- Preconditions: Typed diagnosis obligation and mutation plan exist.
- Effects: Tries mechanical/semantic patch, solver/proof-derived, e-graph, CEGIS/template and bounded search candidates before a residual-only LLM candidate; evaluates all through independent gates.
- Evidence subset: candidate diff CID, operator/provenance, solved obligations, counterexamples, targeted/impact validation, semantic minimality and resource cost
- Acceptance: Candidate budgets and ordering hold; fabricated tests/proofs fail; same-root/no-byte-change candidates fail; LLM sees only the admitted minimal context and cannot request new authority; ranking cannot trade a safety violation for lower cost; no admitted candidate yields an explicit residual task.

## CPD-044 Apply repairs in isolation and prove a live intent/security fixed point

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repair-fixed-point
- Depends on: CPD-035, CPD-041, CPD-043, CPD-045, CPD-046
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/control_plane_mutation_fixed_point.py, test/api/test_agent_supervisor_control_plane_mutation_fixed_point.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_mutation_fixed_point.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/fixed-point
- Parallel lane: cpd-repair
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/control_plane_mutation_fixed_point.py, test/api/test_agent_supervisor_control_plane_mutation_fixed_point.py
- Interfaces: ControlPlaneMutationFixedPoint@1, MutationTransactionReceipt@1
- Conflict policy: Reuse DoctorWorktreeAdapter, live fixed-point, ref-CAS, leases/fences, merge train and SecurityIR/IntentIR gates.
- Preconditions: Admitted repair candidate, impact closure and mutation permit exist.
- Effects: Mutates real bytes only in an isolated worktree/VFS, runs targeted and full impacted validation, refreshes graphs/proofs/security, commits or exactly rolls back, and emits residuals.
- Evidence subset: before/after bytes/CIDs/forest roots, lease/fence/checkpoint/ref-CAS, validation/proof/security outcomes, affected SCC/callers/consumers, rollback and fixed-point receipts
- Acceptance: Changed bytes and roots are reread; expected/observed effects match; all-caller/consumer closure, tests, contracts, formal checks, IntentIR/SecurityIR and supply-chain gates pass; second-order defects iterate within budget; timeout/lease loss/merge conflict/regression/unknown frontier rolls back exactly; the seed/control policies are never writable.

## CPD-045 Replace deferred Doctor stages with a typed live diagnose-to-transaction loop

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-loop
- Depends on: CPD-040, CPD-042
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_doctor_loop.py, ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py, test/api/test_agent_supervisor_control_plane_doctor_loop.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_doctor_loop.py test/api/test_agent_supervisor_deterministic_doctor_runtime.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/doctor-loop
- Parallel lane: cpd-doctor
- Resource class: cpu-large
- Resource stage: implementation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_doctor_loop.py, ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py, test/api/test_agent_supervisor_control_plane_doctor_loop.py, test/api/test_agent_supervisor_deterministic_doctor_runtime.py
- Interfaces: ControlPlaneDoctorLoop@1, DoctorStageExecutionReceipt@1
- Conflict policy: Replace production deferral with typed composition of existing query/tactician/proof/synthesis/impact/transaction services; retain report-only default and authority separation.
- Preconditions: Live reasoning kernel and diagnosis-to-obligation bridge exist.
- Effects: Executes diagnose→retrieve→plan→prove/counterexample→synthesis preview→impact→transaction proposal→fixed-point request, with typed stage outputs and bounded residuals.
- Evidence subset: real checkout snapshot, stage capability/input/output/status/timing/cache roots, obligation transitions, proposal/permit requirements and open frontier
- Acceptance: Normal production inputs no longer unconditionally return `plan_inputs_deferred`, typed-stage deferral or transaction deferral; each available stage executes and each unavailable required stage abstains explicitly; report-only performs no mutation; transaction execution still requires separate permit/worktree/lease/fence; restart resumes from body-free receipts; deterministic mode performs zero model/network calls.

## CPD-046 Wire the reasoning, proof, graph, retrieval, and artifact caches into production

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: reasoning-cache
- Depends on: CPD-014, CPD-015, CPD-016, CPD-017
- Goal id: CPD-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_reasoning_cache.py, test/api/test_agent_supervisor_control_plane_reasoning_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_reasoning_cache.py test/api/test_agent_supervisor_reasoning_cache.py test/api/test_agent_supervisor_program_analysis_cache.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/reasoning-cache
- Parallel lane: cpd-doctor
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/control_plane_reasoning_cache.py, test/api/test_agent_supervisor_control_plane_reasoning_cache.py
- Interfaces: ControlPlaneReasoningCache@1, CrossStageInvalidationReceipt@1
- Conflict policy: Compose ReasoningCacheCoordinator, ProgramAnalysisCache, repository AST/CAS and proof cache; no competing cache identity or assurance tier.
- Preconditions: Semantic graph, hybrid retrieval, formal portfolio and independent security adapters expose exact semantic inputs and dependency roots.
- Effects: Adds a production cache composition root, single flight, exact/delta lookup, dependency invalidation, restart persistence, health/quarantine and assurance reconstruction.
- Evidence subset: repository/dirty/query/property/parser/index/translator/toolchain/provider/policy/assumption/bound/dependency roots, hit/miss/invalidation/provenance and reconstructed assurance
- Acceptance: Exact reuse avoids duplicate execution; changed AST/call/data/contract/build/test/policy/submodule/provider inputs invalidate all affected entries and no unaffected entries in protected cases; stale/tampered/poisoned/cross-run entries quarantine; hits never upgrade assurance; cold/delta/restart latency, RSS/disk, invalidation recall and duplicate-compute rate are measured.

## CPD-050 Bind prompt-to-promotion receipt lineage with CIDs, Merkle proofs, and signatures

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: attestation
- Depends on: CPD-001, CPD-011
- Goal id: CPD-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/control_plane_attestation.py, test/api/test_agent_supervisor_control_plane_attestation.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_attestation.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/attestation
- Parallel lane: cpd-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/control_plane_attestation.py, test/api/test_agent_supervisor_control_plane_attestation.py
- Interfaces: ControlPlaneAttestation@1, ControlPlaneMerkleProof@1
- Conflict policy: Reuse existing multiformats/multihash/CID and proof attestation interfaces; no new assurance lattice.
- Preconditions: Canonical bundle and receipt identities exist.
- Effects: Links prompt/snapshot/plan/materialization/launch/mutation/validation/benchmark/rollout/promotion receipts in a verifiable append-only manifest.
- Evidence subset: canonical preimages, parent links, Merkle paths, issuer key identifiers, timestamps/order from trusted source and revocation state
- Acceptance: Tampering, omitted/reordered stage, invalid preimage/path/signature, cross-repository/run/policy replay and revoked key fail; bodies/secrets/private witnesses are absent; lineage establishes identity and issuer only, not semantic truth.

## CPD-051 Define and gate an optional narrow ZKP claim

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P2
- Track: zkp
- Depends on: CPD-002, CPD-050
- Goal id: CPD-G060
- Outputs: docs/architecture/agent_supervisor_control_plane_zkp_threat_model.md, test/api/test_agent_supervisor_control_plane_zkp.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_zkp.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/zkp
- Parallel lane: cpd-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: docs/architecture/agent_supervisor_control_plane_zkp_threat_model.md, test/api/test_agent_supervisor_control_plane_zkp.py
- Interfaces: ControlPlaneZKClaim@1
- Conflict policy: Reuse program-analysis ZKP only if its fixed circuit matches an approved claim; otherwise document that ZKP remains disabled.
- Preconditions: Cryptographic lineage exists and an operator can identify a privacy or fixed-computation need.
- Effects: Specifies witness/public inputs/circuit/verifier/setup/trust/replay/privacy limits and real-versus-simulated assurance.
- Evidence subset: operator-approved threat model, circuit/verifier/toolchain CIDs, test vectors and privacy analysis
- Acceptance: Claim is narrow and falsifiable; simulation is labeled and cannot attest; only an independently authenticated and cryptographically verified `ProgramZkpVerificationReceipt` can support `ATTESTED`, while caller `production_eligible` or mode/status flags cannot; invalid witness/public input/replay/version/verifier mismatch fail; no semantic correctness, inventory completeness or translator soundness is inferred; operator manual review is required, and disabling ZKP is an acceptable result when no justified claim exists.

## CPD-060 Execute the live paired repository-self benchmark

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark-runner
- Depends on: CPD-002, CPD-033, CPD-034, CPD-035, CPD-044
- Goal id: CPD-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/control_plane_live_benchmark.py, test/api/test_agent_supervisor_control_plane_live_benchmark.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_live_benchmark.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/live-benchmark
- Parallel lane: cpd-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/control_plane_live_benchmark.py, test/api/test_agent_supervisor_control_plane_live_benchmark.py
- Interfaces: ControlPlaneLiveBenchmark@1, PairedControlPlaneRun@1
- Conflict policy: Import the sealed PDR repository corpus, partitions, mutations, budgets and denominator rules; candidates cannot edit them.
- Preconditions: Public raw-prompt transports, real lifecycle and live fixed-point repair work.
- Effects: Runs baseline, deterministic, hybrid and ablation configurations on exact paired roots under cold/exact/delta/restart caches and concurrency 1/2/4/6-or-width.
- Evidence subset: live prompt/create/materialize/start/dispatch/Doctor/mutation receipts, clean roots, corpus case/mutation/toolchain/provider/hardware/budget/cache/concurrency identity
- Acceptance: All measured outcomes come from real services/processes and observed effects; PlanCreate uses the production analysis factory, PlanSteer executes rather than class-probes, and Doctor runs its typed stages; predictions cannot be derived from seed recipes/markers; paired inputs and denominators match; development and provenance-family holdout stay separate; skipped/synthetic/unpaired runs cannot qualify; failures/timeouts/abstentions remain in denominators.

## CPD-061 Attribute wall time, tokens, process-tree resources, GPU, quota, and cost

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: telemetry
- Depends on: CPD-035
- Goal id: CPD-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_benchmark_telemetry.py, test/api/test_agent_supervisor_control_plane_benchmark_telemetry.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_benchmark_telemetry.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/telemetry
- Parallel lane: cpd-telemetry
- Resource class: cpu-medium
- Resource stage: validation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_benchmark_telemetry.py, test/api/test_agent_supervisor_control_plane_benchmark_telemetry.py
- Interfaces: ControlPlaneBenchmarkTelemetry@1, ControlPlaneResourceReceipt@1
- Conflict policy: Extend PDR BenchmarkTelemetry; never synthesize missing provider or OS counters.
- Preconditions: Real lifecycle exposes owned process-tree identity.
- Effects: Attributes makespan/critical path/queue/occupancy, provider tokens/cache/retries/cancellations, CPU/RSS/I/O/network/disk/process/GPU/VRAM/quota/cost to exact stages and tasks.
- Evidence subset: monotonic clocks, process birth/tree, cgroup/proc/GPU/provider receipts, tokenizer/model identity, cache and task/bundle correlations
- Acceptance: Children and cancelled/retried work are included exactly once; missing or inaccessible counters are unavailable; counter reset/PID reuse/non-finite/negative/impossible values fail validation; CUDA assignment and provider hard-quota fallback are observable; overhead is bounded and reported.

## CPD-062 Build the independent protected solution-quality oracle

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: quality-oracle
- Depends on: CPD-060, CPD-061
- Goal id: CPD-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/control_plane_quality_oracle.py, test/api/test_agent_supervisor_control_plane_quality_oracle.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_quality_oracle.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/quality-oracle
- Parallel lane: cpd-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/control_plane_quality_oracle.py, test/api/test_agent_supervisor_control_plane_quality_oracle.py
- Interfaces: ControlPlaneQualityOracle@1, ControlPlaneQualityReceipt@1
- Conflict policy: Oracle bodies and held-out membership are operator-owned and outside candidate read/write scope.
- Preconditions: Live paired benchmark produces complete receipts and telemetry.
- Effects: Scores plan coverage/dependencies/conflicts/resources/context, diagnosis/localization/repair, API/security/proof/validation, regression/minimality/rollback and end-to-end solution quality independently.
- Evidence subset: protected expected properties and independent observations, never candidate `expected` fields
- Acceptance: Protected tests/proofs/fuzz/mutation/security checks execute outside the candidate worktree and sign observed receipts; caller-supplied candidate or truth IDs and candidate predictions cannot score themselves, and absent gold never falls back to candidate output; precision/recall and abstention are denominator-correct; hidden cases cannot be inferred from public fixtures or artifacts; safety and quality floors are non-compensable; oracle replay binds exact roots and rejects tampering; an independent operator seals the oracle before manual completion.

## CPD-070 Implement bounded unattended Planner/Doctor benchmark epochs

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: epoch
- Depends on: CPD-044, CPD-060, CPD-062
- Goal id: CPD-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/control_plane_epoch.py, test/api/test_agent_supervisor_control_plane_epoch.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_epoch.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/epoch
- Parallel lane: cpd-self-improvement
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/control_plane_epoch.py, test/api/test_agent_supervisor_control_plane_epoch.py
- Interfaces: ControlPlaneImprovementEpoch@1, ControlPlaneEpochReceipt@1
- Conflict policy: Controller consumes immutable policies and invokes existing services; it cannot grant its own mutation/benchmark/rollout authority.
- Preconditions: Live mutation loop, paired benchmark and protected quality oracle exist.
- Effects: Runs baseline→propose→shadow→evaluate→reject/retain→canary→recheck→promote/rollback with bounded candidates, calls, repairs, time and resources.
- Evidence subset: exact baseline/challenger roots and policies, stage receipts, budgets, quality/safety/Pareto decision and rollback state
- Acceptance: Epoch and stage caps, quality non-inferiority, zero safety violations, paired denominators, stop conditions and exact rollback are enforced; crash/restart resumes idempotently; missing authority/evidence/telemetry stops; automatic rollout remains disabled at bootstrap.

## CPD-071 Compile Doctor and benchmark residuals into a bounded derived task source

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: refill
- Depends on: CPD-070
- Goal id: CPD-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives/control_plane_residual_refill.py, test/api/test_agent_supervisor_control_plane_residual_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_residual_refill.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/refill
- Parallel lane: cpd-self-improvement
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/objectives/control_plane_residual_refill.py, test/api/test_agent_supervisor_control_plane_residual_refill.py
- Interfaces: ControlPlaneResidualRefill@1, DerivedGoalTaskSource@1
- Conflict policy: Write only the configured derived DuckDB/CAS source; never the CPD/PDR/ASI seed boards or goal heaps.
- Preconditions: An epoch emits typed residuals with current evidence and causal identity.
- Effects: Deduplicates residuals, maps them to append-only goals/tasks/dependencies/context, enforces caps/depth/retry/cooldown and submits them for normal admission.
- Evidence subset: residual/counterexample/root/repetition/novelty/provenance, producing epoch and obligation closure
- Acceptance: Maximum 8 goals, 24 tasks, 48 open, depth 3 and retry 2 are enforced; unchanged failures cool down; fixed-point-closed residuals emit no work; task semantics have stable CIDs; generated tasks cannot edit protected inputs or self-authorize; overflow is explicit debt.

## CPD-072 Gate canary and automatic rollout with quality-safe Pareto and anti-gaming rules

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rollout
- Depends on: CPD-071
- Goal id: CPD-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/control_plane_rollout.py, test/api/test_agent_supervisor_control_plane_rollout.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_rollout.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/rollout
- Parallel lane: cpd-self-improvement
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/control_plane_rollout.py, test/api/test_agent_supervisor_control_plane_rollout.py
- Interfaces: ControlPlaneRolloutDecision@1, ControlPlanePromotionReceipt@1
- Conflict policy: Promotion policy and oracle remain protected; candidate code cannot alter weights, denominators, floors, stages or rollback triggers.
- Preconditions: Bounded epochs and derived refill pass live paired holdout evaluation.
- Effects: Evaluates off/observe/shadow/assist/canary/automatic progression, non-inferiority, Pareto dominance, anti-gaming, freshness and rollback readiness.
- Evidence subset: current-tree paired metrics, safety/quality floors, attribution confidence, canary health, rollback drill and independent authority
- Acceptance: No scalar aggregate compensates for safety/quality regression; missing/unpaired/stale/skipped/synthetic evidence cannot promote; suspicious metric suppression or scope reduction rejects; rollback is prevalidated; automatic remains off until an independent operator signs the current promotion receipt.

## CPD-080 Prove prompt-to-daemon-to-repair parity under E2E, adversarial, and chaos cases

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: e2e-chaos
- Depends on: CPD-033, CPD-034, CPD-035, CPD-050, CPD-072
- Goal id: CPD-G090
- Outputs: test/e2e/test_agent_supervisor_control_plane_loop.py, test/api/test_agent_supervisor_control_plane_chaos.py
- Validation: python -m pytest test/e2e/test_agent_supervisor_control_plane_loop.py test/api/test_agent_supervisor_control_plane_chaos.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/e2e-chaos
- Parallel lane: cpd-release
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: test/e2e/test_agent_supervisor_control_plane_loop.py, test/api/test_agent_supervisor_control_plane_chaos.py
- Interfaces: ControlPlaneE2EQualification@1
- Conflict policy: Tests may use isolated fixtures/worktrees only and cannot weaken production validation.
- Preconditions: Public transports, lifecycle, repair, attestation and rollout gates are complete.
- Effects: Exercises real raw prompt→goals/subgoals/tasks→Markdown/DuckDB→launch spec→daemon→dispatch→Doctor→repair→benchmark→refill across Python/CLI/MCP and faults.
- Evidence subset: canonical result parity, actual processes/effects, restarts, attack/fault matrix and rollback receipts
- Acceptance: Same input yields same semantic bundle/task CIDs across transports; stale/tampered prompt and roots, prompt injection, path/argv injection, partial projection/start, process/PID loss, provider hard quota/transient failure, solver/index/cache loss/corruption, GPU pressure, timeout, merge conflict, rollback and kill switch fail safely and resume or stop deterministically.

## CPD-081 Deliver the protected launcher, lifecycle controls, kill switch, and runbook

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operations
- Depends on: CPD-080
- Goal id: CPD-G090
- Outputs: scripts/ops/agent_supervisor/control_plane_planner_doctor.py, docs/guides/CONTROL_PLANE_PLANNER_DOCTOR_GUIDE.md, test/api/test_agent_supervisor_control_plane_operations.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_operations.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/operations
- Parallel lane: cpd-release
- Resource class: cpu-medium
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: scripts/ops/agent_supervisor/control_plane_planner_doctor.py, docs/guides/CONTROL_PLANE_PLANNER_DOCTOR_GUIDE.md, test/api/test_agent_supervisor_control_plane_operations.py
- Interfaces: ControlPlaneOperations@1
- Conflict policy: Operator launcher consumes sealed bundle/profile and cannot enable mutation/refill/automatic rollout by configuration alone.
- Preconditions: E2E/chaos qualification passes on the current tree.
- Effects: Provides validate/preview/apply/start/status/health/drain/stop/restart/kill/rollback/benchmark commands and a reproducible operational runbook.
- Evidence subset: parser-validated commands, current roots, permits, process ownership, logs/receipts, resource/provider/GPU status, incident and rollback drills
- Acceptance: Safe defaults are preview/report-only/shadow; commands are noninteractive and shell-injection safe; health verifies real child identity; kill switch stops the owned tree; secrets are never printed; CUDA and Grok→hard-quota-only Terra policy are visible; an independent operator successfully performs start, restart, fault, kill and rollback drills before manual completion.

## CPD-082 Issue the independently replayed terminal CPD release receipt

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release
- Depends on: CPD-051, CPD-062, CPD-072, CPD-081
- Goal id: CPD-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/control_plane_planner_doctor_release.py, docs/architecture/CONTROL_PLANE_PLANNER_DOCTOR_V2_RELEASE.md, test/api/test_agent_supervisor_control_plane_planner_doctor_release.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_planner_doctor_release.py -q
- Board namespace: agent-supervisor-control-plane-planner-doctor-v2
- Bundle: agent-supervisor/control-plane-planner-doctor/release
- Parallel lane: cpd-release
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/control_plane_planner_doctor_release.py, docs/architecture/CONTROL_PLANE_PLANNER_DOCTOR_V2_RELEASE.md, test/api/test_agent_supervisor_control_plane_planner_doctor_release.py
- Interfaces: ControlPlanePlannerDoctorRelease@1
- Conflict policy: Release aggregates and independently replays constituent evidence; it cannot repair, waive, simulate or self-certify a failure.
- Preconditions: All child goals have current evidence or an explicitly accepted optional-ZKP-disabled result; operations and rollout manual reviews are sealed.
- Effects: Recomputes artifact/task/goal/policy/root identities, reruns terminal conformance and holdout gates, verifies authority/promotions and emits the terminal release proposal for operator seal.
- Evidence subset: every CPD task output and validation, PDR/ASI import CIDs, current forest, transport/E2E/chaos, live benchmark/oracle/telemetry, security/proof/fixed-point, operations, promotion and rollback receipts
- Acceptance: No pending required task, stale root, missing/manual-unsealed authority, skipped/synthetic gate, safety violation, quality regression, unexplained unavailable telemetry, projection/transport mismatch, process-control failure or rollback failure remains; independent replay matches; an external operator signs the terminal receipt, after which automatic mode may be separately considered but is not enabled by release itself.
