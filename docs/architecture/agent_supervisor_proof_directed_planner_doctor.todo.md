# Agent Supervisor Proof-Directed Planner and Doctor Taskboard (PDR)

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix
`## PDR-`. Companion objective heap:
`agent_supervisor_proof_directed_planner_doctor.objectives.md`. Normative plan:
`AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md`. Scheduler profile:
`config/agent_supervisor_proof_directed_planner_doctor_scheduler.json`.

## Execution doctrine

- The plan, objective heap, this seed board, scheduler/promotion policy, and
  holdout/oracle artifacts are protected operator inputs.
- Automatically generated successor work goes to the separate derived
  DuckDB/CAS task source; agents never append to this seed board.
- Retrieval, KG, vector, embedding, model, synthetic benchmark, and cache
  metadata are not proof, mutation, promotion, or completion authority.
- Deterministic Doctor mode never calls an LLM or remote provider.
- Preview/diagnosis is read-only. Apply/repair needs current roots, permit,
  lease, fence, expected effects, checkpoint, rollback, and independent
  current-tree validation.
- A completed task row is not objective-completion authority.

## Initial parallel lanes

| Lane | Primary ownership |
| --- | --- |
| `pdr-foundation` | baseline, threat model, benchmark contract |
| `pdr-evidence` | canonical snapshots, graph/query evidence, cache |
| `pdr-planner` | plan contracts, query/obligation/candidate/critic |
| `pdr-proof` | proof authority, IR/security admission, attestation |
| `pdr-control` | create/steer service, storage, transports, runtime |
| `pdr-doctor` | production Doctor composition and diagnosis |
| `pdr-repair` | synthesis, transaction, fixed point, candidate validation |
| `pdr-benchmark` | live paired execution and quality oracles |
| `pdr-telemetry` | process/provider/token/resource attribution |
| `pdr-self-improvement` | epoch, derived refill, rollout |
| `pdr-release` | chaos, operator controls, terminal receipt |

---

## PDR-000 Seal PDR plan, objective heap, seed board, and scheduler profile

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bootstrap
- Depends on:
- Goal id: PDR-G000
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md, docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md, docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md, config/agent_supervisor_proof_directed_planner_doctor_scheduler.json, docs/architecture/agent_supervisor/PROGRAMS.md
- Validation: python -m json.tool config/agent_supervisor_proof_directed_planner_doctor_scheduler.json >/dev/null && rg -q 'PDR-' docs/architecture/agent_supervisor/PROGRAMS.md
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/bootstrap
- Parallel lane: pdr-foundation
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md, docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md, docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md, config/agent_supervisor_proof_directed_planner_doctor_scheduler.json, docs/architecture/agent_supervisor/PROGRAMS.md
- Interfaces: PDRSeedPlan@1
- Conflict policy: PDR bootstrap artifacts only; do not edit any foreign board or generated completion projection.
- Preconditions: Audited against origin/main at f25e5719cb738a50fb96bac4bea3f66ebca9800b.
- Effects: Establishes immutable scheduling inputs and a collision-free PDR namespace.
- Evidence subset: pdr seed plan and parser validation
- Acceptance: All four program artifacts plus the PDR glossary row exist, JSON is valid, every task uses one unique `## PDR-###` heading, every dependency and goal reference resolves, the graph is acyclic, `PDR-000` is the only completed bootstrap task, and the plan states that automatic refill uses a separate derived task source.

## PDR-001 Freeze the mainline capability and residual-gap inventory

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundation
- Depends on: PDR-000
- Goal id: PDR-G010
- Outputs: docs/architecture/agent_supervisor_planner_doctor_baseline.md, ipfs_accelerate_py/agent_supervisor/analysis/planner_doctor_capability_inventory.py, test/api/test_agent_supervisor_planner_doctor_capability_inventory.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_capability_inventory.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/foundation
- Parallel lane: pdr-foundation
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/agent_supervisor_planner_doctor_baseline.md, ipfs_accelerate_py/agent_supervisor/analysis/planner_doctor_capability_inventory.py, test/api/test_agent_supervisor_planner_doctor_capability_inventory.py
- Interfaces: PlannerDoctorCapabilityInventory@1
- Conflict policy: Read existing ASI/CBP/RPR/LPR/formal-verification artifacts; do not change them.
- Preconditions: PDR seed plan is sealed.
- Effects: Produces an exact, replayable shipped-vs-gap inventory and capability root.
- Evidence subset: mainline module, task, objective, test, config, and capability records
- Acceptance: Inventory binds current commit/tree/recursive gitlinks and records shipped interfaces, default construction behavior, objective/task statuses, tests, configs, optional tool health, and missing live wiring separately. It explicitly captures unset prompt analysis/admission factories, missing create/steer/parallel contracts, missing Doctor backends, incompatible snapshots, proof/transaction/fixed-point trust gaps, synthetic benchmark producers, disabled refill, and the cold-import regression. Package presence alone never means capability.

## PDR-002 Seal authority, threat model, mutation boundary, and safety floors

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: trust
- Depends on: PDR-000
- Goal id: PDR-G010
- Outputs: docs/architecture/agent_supervisor_planner_doctor_threat_model.md, config/agent_supervisor_planner_doctor_authority_policy.json, config/agent_supervisor_planner_doctor_authority_policy.seal.json, test/api/test_agent_supervisor_planner_doctor_authority_policy.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_authority_policy.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/foundation
- Parallel lane: pdr-proof
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 5400
- Predicted files: docs/architecture/agent_supervisor_planner_doctor_threat_model.md, config/agent_supervisor_planner_doctor_authority_policy.json, config/agent_supervisor_planner_doctor_authority_policy.seal.json, test/api/test_agent_supervisor_planner_doctor_authority_policy.py
- Interfaces: PlannerDoctorAuthorityPolicy@1, PlannerDoctorThreatModel@1
- Conflict policy: Own new PDR policy only; do not weaken existing control, proof, Doctor, or completion policies.
- Preconditions: Existing assurance lattice, control permits, Doctor rollout policy, proof cache, IR adapters, and self-improvement rollout contracts are inventoried.
- Effects: Creates the immutable non-compensable policy root used by every later admission and benchmark.
- Evidence subset: authority ladder, adversary model, protected anchors, forbidden transitions
- Acceptance: Policy distinguishes nomination, observation, bounded check, kernel proof, and attestation; forbids candidate/self/synthetic authority, model writes, benchmark/oracle mutation, stale replay, unproved security, partial transaction, false fixed point, and task-count completion; keeps deterministic mode model/network-free; declares exact zero safety floors, preview/apply separation, kill switch, manual escalation, and current-tree revalidation.

## PDR-003 Preregister live paired benchmark, protected holdouts, and budgets

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark-contract
- Depends on: PDR-000
- Goal id: PDR-G010
- Outputs: config/agent_supervisor_planner_doctor_benchmark.json, config/agent_supervisor_planner_doctor_benchmark.seal.json, test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json, docs/architecture/agent_supervisor_planner_doctor_benchmark.md, test/api/test_agent_supervisor_planner_doctor_benchmark_contract.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_benchmark_contract.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/benchmark-contract
- Parallel lane: pdr-benchmark
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 5400
- Predicted files: config/agent_supervisor_planner_doctor_benchmark.json, config/agent_supervisor_planner_doctor_benchmark.seal.json, test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json, docs/architecture/agent_supervisor_planner_doctor_benchmark.md, test/api/test_agent_supervisor_planner_doctor_benchmark_contract.py
- Interfaces: PlannerDoctorBenchmarkManifest@1, PlannerDoctorBenchmarkPolicy@1
- Conflict policy: Holdout manifest and benchmark policy become protected after review; candidate code cannot edit or read hidden oracle bodies.
- Preconditions: PDR seed plan describes the paired benchmark dimensions and promotion boundary.
- Effects: Freezes cases, paired roots, metrics, denominators, cache/concurrency strata, budgets, oracles, ablations, and stop conditions before benchmark code exists.
- Evidence subset: benchmark manifest and policy roots
- Acceptance: Manifest separates development and held-out roots; fixes prompt/mutation/toolchain/provider/tokenizer/hardware/cache/worker inputs; sweeps 1/2/4/configured-maximum lanes (six at bootstrap); measures clock/tokens/process-tree resources/GPU/quality; labels missing telemetry unavailable; prohibits synthetic/skipped promotion; protects oracles and denominators; requires quality non-inferiority and zero safety floors before Pareto comparison.

## PDR-010 Unify repository forest, Doctor snapshot, finding, and root contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-contracts
- Depends on: PDR-001, PDR-002
- Goal id: PDR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/repository_reasoning_snapshot.py, ipfs_accelerate_py/agent_supervisor/analysis/doctor_contract_adapters.py, test/api/test_agent_supervisor_repository_reasoning_snapshot.py, test/api/test_agent_supervisor_doctor_contract_adapters.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_reasoning_snapshot.py test/api/test_agent_supervisor_doctor_contract_adapters.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/evidence-contracts
- Parallel lane: pdr-evidence
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/repository_reasoning_snapshot.py, ipfs_accelerate_py/agent_supervisor/analysis/doctor_contract_adapters.py, test/api/test_agent_supervisor_repository_reasoning_snapshot.py, test/api/test_agent_supervisor_doctor_contract_adapters.py
- Interfaces: RepositoryReasoningSnapshot@1, DiagnosisObligationBridge@1
- Conflict policy: Add checked bridges over existing prompt and Doctor records; do not silently alias incompatible schemas or create a second repository root.
- Preconditions: Exact existing snapshot/finding/root schemas are inventoried.
- Effects: One canonical content-addressed snapshot can serve Planner and Doctor while preserving source schemas and evidence tiers.
- Evidence subset: repository forest, dirty overlay, recursive gitlinks, task source, parser/index/toolchain/capability/policy roots
- Acceptance: Snapshot covers tracked/staged/modified/deleted/renamed/admitted-untracked paths, recursive submodules, exclusions, instability, truncation, task-source revision/status/evidence/event cursor, and all tool/policy roots. Both Doctor snapshot/finding families round-trip through explicit adapters; issue/root/CID mismatches, duplicate/unknown fields, body/secret material, tampering, and cross-repository replay fail closed.

## PDR-011 Build the production repository-analysis and admission factory

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-runtime
- Depends on: PDR-001, PDR-010
- Goal id: PDR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py, ipfs_accelerate_py/agent_supervisor/analysis/repository_indexer.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_directory_scanner.py, test/api/test_agent_supervisor_planning_analysis_factory.py
- Validation: python -m pytest test/api/test_agent_supervisor_planning_analysis_factory.py test/api/test_agent_supervisor_prompt_directory_scanner.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/evidence-runtime
- Parallel lane: pdr-evidence
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py, ipfs_accelerate_py/agent_supervisor/analysis/repository_indexer.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_directory_scanner.py, test/api/test_agent_supervisor_planning_analysis_factory.py
- Interfaces: PlanningAnalysisFactory@1, RepositoryIndexer
- Conflict policy: Reuse `AnalysisOperationRegistry`, repository/program/contract/value graphs, and prompt scan contracts; no direct planner import of optional datasets providers.
- Preconditions: Canonical snapshot and adapter contracts are available.
- Effects: Normal service construction can enumerate a real checkout, produce the exact evidence view, and create an independent admission request factory.
- Evidence subset: exhaustive source inventory, AST/program/contract/value graph handles, capability and admission roots
- Acceptance: Factory scans an allowlisted checkout and recursive configured submodules without importing target code; includes tests/config/build/schema/docs/policies and dirty overlay; wires default prompt `optional_analysis` and `admission_request_factory`; records CFG/dataflow/native/generated/concurrency open frontiers; lazy optional-provider loss degrades or abstains; wrong-tree, unstable, secret, symlink and path escape cases fail.

## PDR-012 Add the deterministic analysis and formal-method strategy registry

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: capability-routing
- Depends on: PDR-001, PDR-002
- Goal id: PDR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/analysis_strategy_registry.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py, test/api/test_agent_supervisor_analysis_strategy_registry.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_strategy_registry.py test/api/test_agent_supervisor_analysis_operation_registry.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/capabilities
- Parallel lane: pdr-evidence
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/analysis_strategy_registry.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py, test/api/test_agent_supervisor_analysis_strategy_registry.py
- Interfaces: AnalysisStrategyRegistry@1, AnalysisCapabilityReceipt@1
- Conflict policy: Extend the existing registry with property-to-strategy routing and lazy provider adapters; do not infer support from imports.
- Preconditions: Capability inventory and authority policy are sealed.
- Effects: Planner and Doctor can select the least-cost sufficient deterministic method with explicit capability/fallback receipts.
- Evidence subset: AST/Tree-sitter, CFG/SSA/PDG, call/points-to/dataflow/taint/effect/typestate/abstract interpretation, symbolic/concolic/WP/separation, SAT/SMT/CHC/Datalog/CEGAR/CEGIS, temporal/hyperproperty/protocol, test/fuzz/mutation/differential/metamorphic, runtime-invariant, supply-chain and kernel-proof strategies
- Acceptance: Closed property classes map to bounded strategies, required assurance, provider capabilities, input/output schemas, cache rules, budgets, and fallback/abstention behavior. Required unavailable methods abstain; optional methods add debt. Retrieval and learned ranking remain nomination-only. Discovery is cold/lazy and provider health/version/config is bound to results.

## PDR-013 Compile live hybrid retrieval and evidence coverage

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: retrieval-coverage
- Depends on: PDR-011, PDR-012
- Goal id: PDR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/planning_evidence_bundle.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_retrieval.py, test/api/test_agent_supervisor_planning_evidence_bundle.py
- Validation: python -m pytest test/api/test_agent_supervisor_planning_evidence_bundle.py test/api/test_agent_supervisor_analysis_retrieval.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/retrieval
- Parallel lane: pdr-evidence
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/planning_evidence_bundle.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_retrieval.py, test/api/test_agent_supervisor_planning_evidence_bundle.py
- Interfaces: PlanningEvidenceBundle@1, EvidenceCoverageReceipt@1
- Conflict policy: Fuse existing lexical/vector/GraphRAG/AST/dependency/proof signals; do not duplicate indexes or promote ranking.
- Preconditions: Production analysis factory and strategy registry exist.
- Effects: Creates bounded body-free evidence bundles and proves coverage/debt for planning and diagnosis.
- Evidence subset: exact graph facts, BM25/vector/KG nominations, proof/counterexample handles, coverage slots, truncation and health
- Acceptance: Live adapters query AST, BM25, KG/GraphRAG, vectors/embeddings when healthy, lineage/history, contracts, value provenance, tests and proofs; every result carries authority/provenance/current-root/capability/cache labels and ranking explanation; missing required slots reject or schedule a query; poisoned/stale/constant/non-finite/dimension-drift indexes are disabled; source bodies and prompt instructions remain inert data.

## PDR-014 Coordinate content-addressed analysis/proof/artifact caches and invalidation

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: cache
- Depends on: PDR-010, PDR-011, PDR-012
- Goal id: PDR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/reasoning_cache.py, test/api/test_agent_supervisor_reasoning_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_reasoning_cache.py test/api/test_agent_supervisor_formal_verification_cache.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/cache
- Parallel lane: pdr-evidence
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/reasoning_cache.py, test/api/test_agent_supervisor_reasoning_cache.py
- Interfaces: ReasoningCacheCoordinator@1, TrustAwareProofCache
- Conflict policy: Coordinate existing analysis cache, proof cache, artifact CAS, and single-flight; never add a parallel proof trust root.
- Preconditions: Exact snapshot and capability identities exist.
- Effects: Exact concurrent computations collapse and dependency-local changes invalidate only affected evidence.
- Evidence subset: semantic computation keys, cache lookup/rejection, dependency invalidation and single-flight receipts
- Acceptance: Keys bind operation/property, forest/scope, premises/assumptions, parser/index/translator/toolchain/capability, policy/IR/catalog, assurance and bounds. Hits reload and verify source receipts and re-derive assurance. Wrong tree/tool/policy/schema, poisoned/partial/forged entries, undeclared dependencies, private material and cross-run replay fail with reason codes. Cache miss is not refutation.

## PDR-015 Restore truly cold Planner/Doctor imports and capability discovery

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: import-hygiene
- Depends on: PDR-001, PDR-002
- Goal id: PDR-G020
- Outputs: ipfs_accelerate_py/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_doctor_cold_import.py, test/api/test_agent_supervisor_planner_cold_import.py
- Validation: python -m pytest test/api/test_agent_supervisor_doctor_cold_import.py test/api/test_agent_supervisor_planner_cold_import.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/import-hygiene
- Parallel lane: pdr-control
- Resource class: cpu-small
- Resource stage: validation
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_doctor_cold_import.py, test/api/test_agent_supervisor_planner_cold_import.py
- Interfaces: AgentSupervisorColdDiscovery@1
- Conflict policy: Serialize edits to package-root import surfaces; preserve reviewed public exports and compatibility aliases.
- Preconditions: Reproduce the current `requests` cold-import failure with `IPFS_ACCEL_SKIP_CORE=1`.
- Effects: Planner/Doctor discovery and help can run without optional providers, network clients, storage initialization, subprocesses, or repository mutation.
- Evidence subset: subprocess import module set, latency, RSS, side effects, capability report
- Acceptance: Fresh-process tests import service/contracts/discovery/help without loading requests/httpx/aiohttp/urllib3, llm_router, model SDKs, torch/transformers, neo4j, DuckDB or optional datasets providers; no network/process/database/storage initialization occurs; strict latency/RSS/module-count budgets are recorded; accessing an optional capability remains lazy and reports unavailable rather than failing root import.

## PDR-020 Define create/steer, plan-delta/revision, and task v2 contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-contracts
- Depends on: PDR-001, PDR-002
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py, test/api/test_agent_supervisor_plan_revision_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_revision_contracts.py test/api/test_agent_supervisor_prompt_workflow_contracts.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/plan-contracts
- Parallel lane: pdr-planner
- Resource class: cpu-small
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py, test/api/test_agent_supervisor_plan_revision_contracts.py
- Interfaces: PlanCreateRequest@1, PlanSteerRequest@1, PlanDelta@1, PlanRevision@1, PromptGoalRecord@2, PromptTaskRecord@2
- Conflict policy: Add semantic v2 contracts and a conservative v1 adapter; preserve current prompt contract identity and compatibility.
- Preconditions: Baseline inventory and authority policy are sealed.
- Effects: Create and steer have immutable ancestry, lifecycle-safe deltas, and complete planning/execution metadata.
- Evidence subset: canonical round-trip, ancestry, supersession, conflict/resource/lifecycle/validation/evidence fields
- Acceptance: Strict provider-free records bind exact repository/task/policy/IR/catalog/capability roots, create/steer origin, parent revision, status/claimed/accepted populations, closed delta operations, goals/subgoals/tasks, outputs/effects, validation DAG, completion rules, conflicts, resources, providers, leases, retries, worktrees and merge strategy. Completed/accepted/claimed history cannot be edited or deleted; unknown fields, secrets, floats, path/CID errors, stale roots and identity tampering fail. V1 reads with conservative non-parallel defaults.

## PDR-021 Compile deterministic reasoning queries and evidence coverage

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: query-planning
- Depends on: PDR-013, PDR-014, PDR-020
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/plan_analysis_query_planner.py, test/api/test_agent_supervisor_plan_analysis_query_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_analysis_query_planner.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/query-planner
- Parallel lane: pdr-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/plan_analysis_query_planner.py, test/api/test_agent_supervisor_plan_analysis_query_planner.py
- Interfaces: ReasoningQueryPlan@1, EvidenceCoverageReceipt@1
- Conflict policy: Query only through `AnalysisOperationRegistry` and the strategy registry; no direct optional-provider coupling.
- Preconditions: Live evidence bundle, caches, and plan contracts exist.
- Effects: Required evidence is gathered or exposed as a blocker before candidate generation.
- Evidence subset: request concepts, changed scope, risk, evidence slots, query capabilities/budgets/cache/fallback
- Acceptance: Fixed rules select required symbol/impact, GraphRAG nomination, premise, contradiction, logic translation, proof/counterexample and security queries from create/steer/diagnosis inputs. Each query binds why/which slot, exact scope, provider capability, maximum bytes/items/time/cost, cache reuse and failure semantics. Optional model suggestions cannot suppress required queries or select credentials/endpoints. Coverage reruns after proposal; code/policy/security claims cannot rely on prompt evidence alone.

## PDR-022 Compile desired and observed behavior into an AND/OR obligation graph

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: obligation-planning
- Depends on: PDR-011, PDR-012, PDR-020, PDR-021
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/obligation_graph_compiler.py, test/api/test_agent_supervisor_obligation_graph_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_obligation_graph_compiler.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/obligations
- Parallel lane: pdr-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/obligation_graph_compiler.py, test/api/test_agent_supervisor_obligation_graph_compiler.py
- Interfaces: ObligationGraph@1, CodeProofObligation, FormalWorkPlan
- Conflict policy: Reuse formal planning contracts, property catalog, logic goals and code-proof obligations; no natural-language proof templates.
- Preconditions: V2 plan contracts and complete required query plan are available.
- Effects: Planner desired state and Doctor observed mismatch share one formal gap representation.
- Evidence subset: predicates, assumptions, pre/postconditions, AND/OR refinement, producers, proof/validation requirements and invalidators
- Acceptance: Compiler performs bounded backward chaining/abductive gap analysis from typed intent/current facts; creates goals, subgoals and producer obligations with dependency/provenance; distinguishes AND requirements from alternative OR strategies; binds assumptions and invalidation selectors; detects cycles, contradictions, uncovered leaves and inconsistent premises; unsupported semantics remain review/unknown; every task candidate must close a named leaf obligation.

## PDR-023 Generate and hard-gate a deterministic-first candidate portfolio

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: candidate-planning
- Depends on: PDR-022
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/symbolic_candidate_planner.py, ipfs_accelerate_py/agent_supervisor/planning/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py, test/api/test_agent_supervisor_symbolic_candidate_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_symbolic_candidate_planner.py test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_prompt_goal_planner.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/candidate-planner
- Parallel lane: pdr-planner
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/symbolic_candidate_planner.py, ipfs_accelerate_py/agent_supervisor/planning/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py, test/api/test_agent_supervisor_symbolic_candidate_planner.py
- Interfaces: SymbolicCandidatePlanner@1, AdaptivePlanSelection, PromptPlanningPolicy
- Conflict policy: Compose existing adaptive evaluator, AND/OR planning, bundle optimizer and failure memory; provider output remains proposal-tier.
- Preconditions: Obligation graph compiler is live.
- Effects: `candidate_count` controls a real bounded portfolio rather than an unused field.
- Evidence subset: deterministic template/HTN/partial-order/constraint candidates, proof feasibility, information gain, cost and hard-gate receipts
- Acceptance: Always produce a codebase-derived deterministic baseline; generate up to configured candidates using reviewed templates, backward chaining, partial-order scheduling, constraint solving, failure memory, proof feasibility and expected information gain; optionally admit bounded model proposals over the same frozen inputs; authority/scope/safety/proof failures are non-compensable; selected/rejected snapshots and provider usage are content-addressed; generic repository-wide one-task fallback is eliminated.

## PDR-024 Add deterministic plan critique, unsat cores, counterexamples, and replan

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-critique
- Depends on: PDR-021, PDR-022, PDR-023
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/plan_critic.py, ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/planning/plan_failure_memory.py, test/api/test_agent_supervisor_plan_critic.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_critic.py test/api/test_agent_supervisor_formal_replanner.py test/api/test_agent_supervisor_plan_failure_memory.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/plan-critic
- Parallel lane: pdr-planner
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/plan_critic.py, ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/planning/plan_failure_memory.py, test/api/test_agent_supervisor_plan_critic.py
- Interfaces: PlanCritique@1, RepairTransition, PlanFailureMemory
- Conflict policy: Extend existing replanner/failure memory with typed critique; never let a model alter scanner/policy/admission findings.
- Preconditions: Candidate portfolio and obligation graph exist.
- Effects: Counterexamples and conflicts trigger minimal queries/repairs instead of full-context retries.
- Evidence subset: schema/graph/coverage/impact/logic/security/proof/conflict/resource/lifecycle defects, minimal unsat cores and counterexamples
- Acceptance: Critic independently recomputes identities and finds cycles, orphans, uncovered goals, contradictions, unsatisfied assumptions, missing consumers, invalid effects, policy/IR/security/proof failures, false parallelism and infeasible resources. It emits bounded minimal cores, typed counterexamples and exact repairable record IDs. Replanning uses new evidence only, respects retry/backoff budgets, preserves accepted history and terminates on unchanged failures.

## PDR-025 Compile proof-directed minimal context and residual-only LLM repair

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-synthesis
- Depends on: PDR-013, PDR-014, PDR-022, PDR-024
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py, ipfs_accelerate_py/agent_supervisor/proof/proof_directed_retrieval.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py, test/api/test_agent_supervisor_planner_doctor_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_context.py test/api/test_agent_supervisor_proof_directed_retrieval.py test/api/test_agent_supervisor_prompt_goal_planner.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/context
- Parallel lane: pdr-planner
- Resource class: provider-llm
- Resource stage: implementation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py, ipfs_accelerate_py/agent_supervisor/proof/proof_directed_retrieval.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py, test/api/test_agent_supervisor_planner_doctor_context.py
- Interfaces: PlannerDoctorContextCapsule@1, PromptGoalPlanningReceipt
- Conflict policy: Reuse context compiler and proof-directed retrieval; never include raw repository dumps, secrets, full proof bodies, or model authority.
- Preconditions: Critique IDs, evidence coverage and proof obligations exist.
- Effects: The LLM is avoided when deterministic closure exists and otherwise sees only the exact residual.
- Evidence subset: task/acceptance IDs, open obligations, assumptions, counterexamples, causal/AST slice, allowed paths/effects/tests, satisfied proof handles and expansion CIDs
- Acceptance: Required core cannot drop intent, security, acceptance, open obligations, assumptions, impact coverage, counterexamples, allowed paths/effects or validation. Satisfied evidence is represented by digest/handle. The model may replace only rejected proposal records or fill behavior-fixed syntax; maximum calls/tokens/rounds/cost are enforced; prompt/repository instructions are inert; malformed/scope-widening/authority/completion output fails; retry sends proof/evidence delta rather than full context.

## PDR-026 Compile replayable conflict/resource/lease/merge parallel execution plans

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: parallel-planning
- Depends on: PDR-011, PDR-012, PDR-020, PDR-021
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py, test/api/test_agent_supervisor_parallel_plan_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_parallel_plan_compiler.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/parallel-plan
- Parallel lane: pdr-control
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py, test/api/test_agent_supervisor_parallel_plan_compiler.py
- Interfaces: ParallelPlanCompiler@1, ParallelExecutionPlan@1
- Conflict policy: Reuse conflict graph, bundle optimizer, resource scheduler, worktree lifecycle and merge contracts; lane labels remain hints.
- Preconditions: V2 task contracts and current repository/resource/provider snapshots exist.
- Effects: Plan admission can prove actual safe width and executable waves.
- Evidence subset: DAG/critical path, path/symbol/interface/submodule/generated/protected conflicts, resources/providers, affinity, leases, worktrees, merge trains and post-merge validation
- Acceptance: Compiler proves leaf producer closure, computes ready waves/critical path, complete conflict surfaces, resource/provider/token/cost/context feasibility, shard/affinity/exclusive/worktree/lease/fence assignments, merge order and rollback boundaries. It emits requested/graph/conflict/resource/admitted width and deterministic replay. It rejects output collisions, protected bottlenecks, overlapping submodules, stale capacity, impossible deadlines and fake lane labels; serial/degraded/review-only outcomes are typed.

## PDR-027 Independently admit plans through formal, IR, security, proof, and authority gates

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-admission
- Depends on: PDR-022, PDR-024, PDR-026
- Goal id: PDR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/plan_admission_service.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_plan_admission.py, ipfs_accelerate_py/agent_supervisor/proof/ir_constraint_compiler.py, test/api/test_agent_supervisor_plan_admission_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_admission_service.py test/api/test_agent_supervisor_prompt_plan_admission.py test/api/test_agent_supervisor_ir_constraint_compiler.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/plan-admission
- Parallel lane: pdr-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/plan_admission_service.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_plan_admission.py, ipfs_accelerate_py/agent_supervisor/proof/ir_constraint_compiler.py, test/api/test_agent_supervisor_plan_admission_service.py
- Interfaces: PlanAdmissionService@1, PlanAdmissionRequest, PlanAdmissionReceipt
- Conflict policy: One independent admission pipeline; no hard-domain failure can be offset by quality/cost scores.
- Preconditions: Obligation graph, critique and parallel execution plan exist.
- Effects: Only evidence-covered, policy-safe, formally valid, proof-feasible plans can be materialized.
- Evidence subset: canonical/schema/graph, goal quality, IntentIR/LegalIR/SecurityIR/program effects, authorization, proof requirements, validation policy, conflict/resource execution plan
- Acceptance: Service constructs its own exact `PlanAdmissionRequest`, never trusts provider admission claims, and checks every stage in fixed order. Unknown mandatory applicability/security/authority/effect/proof fails. Security forbidden logic is checked against intent and code effects. Every admitted receipt binds candidate, evidence, formal plan, IR roots, proof obligations, execution plan, policies and current tree; tampering/replay fails; default prompt service no longer returns `IR_BINDING_MISMATCH` merely because its factory is absent.

## PDR-028 Build revision-bound steer preview and preserve live task history

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-steer
- Depends on: PDR-020, PDR-021, PDR-022, PDR-024, PDR-026, PDR-027
- Goal id: PDR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py, test/api/test_agent_supervisor_plan_steer_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_steer_service.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/plan-steer
- Parallel lane: pdr-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py, test/api/test_agent_supervisor_plan_steer_service.py
- Interfaces: PlanSteerService@1, PlanSteerRequest@1, PlanDelta@1
- Conflict policy: Own the steer-specific service module and tests; shared facade integration belongs to PDR-032.
- Preconditions: Plan revision/delta, query/obligation/critique/admission and parallel-plan components exist.
- Effects: `preview_steer` produces an append-only, lifecycle-safe candidate revision without writing the task source.
- Evidence subset: exact base plan/task/run/status/accepted evidence/event cursor/current tree, impact queries, immutable/deferred delta and full resulting-plan admission
- Acceptance: Steer loads and integrity-checks exact plan/task/run/lease/worktree/merge state; partitions completed/accepted/claimed/running/settling/unstarted/blocked/superseded populations; scans current tree and impact; generates a closed delta; validates the complete resulting plan. Stale base/root/revision/cursor/claimed/lease/fence/policy fails. Running work is never edited; successors, deferred supersession and separate lifecycle requests are explicit. Preview is body-free/read-only/restart-serializable.

## PDR-030 Build first-class create-plan preview with production wiring

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-service
- Depends on: PDR-020, PDR-021, PDR-023, PDR-025, PDR-026, PDR-027
- Goal id: PDR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/plan_create_service.py, test/api/test_agent_supervisor_plan_create_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_create_service.py test/api/test_agent_supervisor_prompt_workflow_e2e.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/plan-service
- Parallel lane: pdr-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/plan_create_service.py, test/api/test_agent_supervisor_plan_create_service.py
- Interfaces: PlanCreateService@1
- Conflict policy: Own the create-specific service module and tests; shared facade/workflow alias integration belongs to PDR-032.
- Preconditions: Planner kernel and admission are complete.
- Effects: `preview_create` runs the full live planning pipeline without writes.
- Evidence subset: exact request/snapshot/query/evidence/obligation/candidate/critique/admission/execution-plan receipts
- Acceptance: Default service factory wires production analysis and admission; create scans current scope and runs query, evidence, obligation, candidate, critique, admission and parallel-plan stages; preview is body-free, read-only and restart-serializable; stale root/policy fails rather than regenerating silently; deterministic and model-assisted modes share exact inputs and bounds; existing workflow preview remains a canonical compatibility alias.

## PDR-031 Persist append-only plan revisions and atomically apply Markdown/DuckDB deltas

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: plan-storage
- Depends on: PDR-020, PDR-028, PDR-030
- Goal id: PDR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py, ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, test/api/test_agent_supervisor_plan_revision_store.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_revision_store.py test/api/test_agent_supervisor_task_source_parity.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/plan-storage
- Parallel lane: pdr-control
- Resource class: io-artifact
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py, ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, test/api/test_agent_supervisor_plan_revision_store.py
- Interfaces: PlanRevisionStore@1, MarkdownTaskSource, DuckDBTaskSource
- Conflict policy: One append-only canonical record with lossless projections; task-source layer does not redefine authority.
- Preconditions: Admitted create/steer previews and v2 records exist.
- Effects: Authorized apply survives restart and preserves immutable history across both backends.
- Evidence subset: intent journal, CAS/fence, revision/delta/supersession/event tables, projection parity, compensation and quarantine
- Acceptance: Apply re-observes all roots and authority, journals intent, appends revision/delta/records/events, never edits accepted/claimed specs, supports deferred successors, verifies round-trip and exact Markdown/DuckDB CIDs, atomically commits or restores prior active projection, recovers at every crash boundary, reloads continuation from CAS/store rather than process dictionaries, and quarantines split brain.

## PDR-032 Expose create/steer operations through shared Python, CLI, and lazy MCP

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: control-transports
- Depends on: PDR-028, PDR-030, PDR-031
- Goal id: PDR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py, ipfs_accelerate_py/agent_supervisor/control/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, ipfs_accelerate_py/agent_supervisor/control/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools.py, test/api/test_agent_supervisor_plan_control_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_control_conformance.py test/api/test_agent_supervisor_control_transport_parity.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/control-transports
- Parallel lane: pdr-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py, ipfs_accelerate_py/agent_supervisor/control/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, ipfs_accelerate_py/agent_supervisor/control/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools.py, test/api/test_agent_supervisor_plan_control_conformance.py
- Interfaces: PlanSupervisorService@1, plan_create_preview, plan_create_apply, plan_steer_preview, plan_steer_apply
- Conflict policy: All transports delegate to the shared service/catalog; no transport-local policy or provider construction.
- Preconditions: Preview and durable apply service exist.
- Effects: Operators and agents can create or steer the same canonical plan through any supported transport.
- Evidence subset: request/result schemas, authority/effects/idempotency, transport discovery and parity receipts
- Acceptance: Default control service binds live workflow handlers instead of reporting unavailable; Python/CLI/MCP produce identical canonical requests, results, roots, errors, cursor behavior and effects; workflow aliases preserve identity; preview is proposal-only and apply requires normal permit/lease/fence/idempotency/expected effects; help/import/discovery remain provider-free and no prompt/repository text can widen allowlists.

## PDR-033 Require active plan revision and compiled execution plan at runtime

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-adoption
- Depends on: PDR-026, PDR-031, PDR-032
- Goal id: PDR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/task_sources/task_source.py, test/api/test_agent_supervisor_parallel_plan_runtime.py
- Validation: python -m pytest test/api/test_agent_supervisor_parallel_plan_runtime.py test/api/test_agent_supervisor_worktree_lifecycle.py test/api/test_agent_supervisor_task_source_protocol.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/runtime-adoption
- Parallel lane: pdr-control
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/task_sources/task_source.py, test/api/test_agent_supervisor_parallel_plan_runtime.py
- Interfaces: ParallelExecutionPlan@1, TaskSource, WorkspaceLifecycleRecord
- Conflict policy: Preserve ASI-171 fenced worktree lifecycle and current resource/usage governance; serialize hot daemon edits.
- Preconditions: Revision store, controls and parallel plan compiler exist.
- Effects: Dispatch follows proved waves and fresh constraints instead of caller-authored lanes.
- Evidence subset: active revision, ready wave, fresh resource/provider/lease observation, worktree/fence/merge/validation receipts
- Acceptance: Daemon rejects partial/mixed/superseded revisions and tasks outside their execution slice; recomputes readiness/status CAS; acquires compiled lease/worktree/fence before claim; enforces conflicts, affinity/exclusive groups, resources/providers, fairness, critical path, merge train and post-merge validation; retains claimed tasks on immutable original revisions; capacity drift degrades/waits rather than overcommits; fake parallel labels cannot execute concurrently.

## PDR-040 Assemble a lazy production deterministic-Doctor runtime and CLI

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-runtime
- Depends on: PDR-010, PDR-011, PDR-012, PDR-014, PDR-015
- Goal id: PDR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py, ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_service.py, scripts/ops/agent_supervisor/deterministic_doctor.py, test/api/test_agent_supervisor_deterministic_doctor_runtime.py
- Validation: python -m pytest test/api/test_agent_supervisor_deterministic_doctor_runtime.py test/api/test_agent_supervisor_deterministic_doctor_service.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-runtime
- Parallel lane: pdr-doctor
- Resource class: cpu-large
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py, ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_service.py, scripts/ops/agent_supervisor/deterministic_doctor.py, test/api/test_agent_supervisor_deterministic_doctor_runtime.py
- Interfaces: DeterministicDoctorBackendFactory@1, DeterministicDoctorRuntime@1
- Conflict policy: `control` retains provider-free contracts; high-layer runtime lazily composes analysis/planning/proof/validation/effect backends without cycles.
- Preconditions: Canonical snapshot, production analysis factory, strategy registry, caches and cold discovery exist.
- Effects: Ordinary service/CLI can inspect and plan a real checkout without caller-supplied snapshots/findings/plans.
- Evidence subset: checkout/root resolution, backend capability graph, stage receipts, no-model/no-network guards
- Acceptance: Runtime consumes `--checkout-root`, enumerates allowlisted source/submodules, builds exact evidence, and wires diagnose, retrieve, tactician, proof, synthesis preview, impact, transaction and fixed-point stages lazily. Report-only works with optional providers absent. Discovery starts no providers/processes/databases. Deterministic mode hard-fails model/network routes. Stage unavailability is actionable. Service uses its control dependency for permits/effects rather than exposing a status-only flag.

## PDR-041 Require semantic theorem bodies, sealed typed proof receipts, and kernel replay

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-proof
- Depends on: PDR-002, PDR-012, PDR-014
- Goal id: PDR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py, ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, test/api/test_agent_supervisor_deterministic_doctor_proof_authority.py
- Validation: python -m pytest test/api/test_agent_supervisor_deterministic_doctor_proof_authority.py test/api/test_agent_supervisor_deterministic_doctor_hammer.py test/api/test_agent_supervisor_doctor_proof_cache.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-proof
- Parallel lane: pdr-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py, ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, test/api/test_agent_supervisor_deterministic_doctor_proof_authority.py
- Interfaces: DoctorAuthoritativeProofReceipt@1, DoctorSynthesisReceipt@2
- Conflict policy: Harden existing Hammer/cache/synthesis contracts; preserve candidate evidence for shadow diagnostics but deny it mutation authority.
- Preconditions: Authority policy and exact semantic cache identities exist.
- Effects: Caller flags and prebuilt mappings cannot manufacture reconstructed/kernel proof.
- Evidence subset: exact theorem body/lowering, property/premise/tree/toolchain/policy roots, native solver transcript, independent kernel reconstruction, sealed receipt-store lineage
- Acceptance: Mutation-capable proof requires a nonempty reviewed theorem body, exact lowering, pinned executable/toolchain, native execution and independent kernel replay; verifier reloads typed receipts from sealed store and recomputes CIDs/preimages/current roots; synthesis receipt is typed and binds uniqueness/consequence/property/toolchain. Raw mappings, duck-typed booleans, identity-only fallback, round-trip flags, wrong theorem/tool/tree/policy, forged/stale/cache/provider-local receipts and test injection cannot admit; negative tests attempt every forgery.

## PDR-042 Add causal localization, complete frontier accounting, and minimal mismatch slices

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-diagnosis
- Depends on: PDR-011, PDR-012, PDR-013, PDR-014, PDR-040
- Goal id: PDR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/doctor_causal_localization.py, ipfs_accelerate_py/agent_supervisor/analysis/doctor_repository_diagnostics.py, ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_impact.py, test/api/test_agent_supervisor_doctor_causal_localization.py
- Validation: python -m pytest test/api/test_agent_supervisor_doctor_causal_localization.py test/api/test_agent_supervisor_doctor_repository_diagnostics.py test/api/test_agent_supervisor_deterministic_doctor_impact.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-diagnosis
- Parallel lane: pdr-doctor
- Resource class: cpu-large
- Resource stage: analysis
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/doctor_causal_localization.py, ipfs_accelerate_py/agent_supervisor/analysis/doctor_repository_diagnostics.py, ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_impact.py, test/api/test_agent_supervisor_doctor_causal_localization.py
- Interfaces: DoctorCausalLocalization@1, DeterministicDoctorFinding
- Conflict policy: Extend diagnostics/impact with live inputs and minimal slices; retrieval signals nominate but do not decide causes.
- Preconditions: Production Doctor runtime and evidence/query adapters exist.
- Effects: Findings state what broke, why, which consumers/values/contracts are affected, and what remains unknown.
- Evidence subset: contract delta, static/dynamic slice, call/value/dependency graph, failing trace, delta-debug result, unsat core, counterexample and open frontier
- Acceptance: Real checkout diagnostics derive findings rather than consume expected outcomes; causal localization fuses exact graph/dataflow/contract/runtime facts, delta debugging and solver cores; issue CID is stable; every resolved mandatory caller/consumer is included; dynamic dispatch, reflection, generated/native/FFI/concurrency/type gaps remain explicit frontiers; poisoned/stale/vector-nearest evidence cannot choose a cause; diagnosis precision/correct-abstention fixtures pass.

## PDR-043 Translate Doctor findings into the shared obligation and planning kernel

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-planner-bridge
- Depends on: PDR-022, PDR-041, PDR-042
- Goal id: PDR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/diagnosis_obligation_adapter.py, test/api/test_agent_supervisor_diagnosis_obligation_adapter.py
- Validation: python -m pytest test/api/test_agent_supervisor_diagnosis_obligation_adapter.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-planner-bridge
- Parallel lane: pdr-doctor
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/diagnosis_obligation_adapter.py, test/api/test_agent_supervisor_diagnosis_obligation_adapter.py
- Interfaces: DiagnosisObligationBridge@1, ObligationGraph@1
- Conflict policy: Adapter only; do not create a Doctor-specific planner or duplicate formal contracts.
- Preconditions: Checked finding bridge, causal localization, proof-authority contract and obligation compiler exist.
- Effects: Doctor repair is a constrained instance of the same Planner used for create/steer.
- Evidence subset: expected/observed predicates, issue/causal/frontier/evidence IDs, repair postconditions and invalidators
- Acceptance: Every supported finding becomes typed desired/observed predicates, assumptions, prohibitions, impact obligations, proof/security/validation requirements and alternative repair subgoals. Root/schema/evidence IDs round-trip. Incomplete or contradictory diagnosis yields review/abstention obligations. No free-form finding text becomes a theorem or authorized effect. Planner and Doctor produce equivalent formal obligations for the same contract mismatch.

## PDR-050 Build a reviewed repair-operator and semantic-patch registry

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repair-operators
- Depends on: PDR-012, PDR-022
- Goal id: PDR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/repair_operator_registry.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/test_agent_supervisor_repair_operator_registry.py
- Validation: python -m pytest test/api/test_agent_supervisor_repair_operator_registry.py test/api/test_agent_supervisor_deterministic_doctor_transforms.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/repair-operators
- Parallel lane: pdr-repair
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/repair_operator_registry.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/test_agent_supervisor_repair_operator_registry.py
- Interfaces: RepairOperatorRegistry@1, DoctorRepairOperatorSpec@2
- Conflict policy: Register reviewed transforms over existing analytical-change machinery; no natural-language or runtime code injection.
- Preconditions: Diagnosis is expressed as typed obligations.
- Effects: Eligible repairs can be generated without an LLM using closed semantic operators.
- Evidence subset: operator kind/version, AST shape, preconditions, consequence, target/value/placement, supported languages, proof/validation requirements
- Acceptance: Registry covers exact rename/move/import/export/registration, missing argument/value threading, constructor/factory/adapter, schema/serializer/manifest/artifact and other already-supported analytical transforms plus reviewed semantic-patch/equality-rewrite hooks. Operators are canonical, idempotent, scope-bounded and capability-declared; unknown/dynamic/stateful/native/public-API/dependency-changing behavior requires approval or abstains; target/value/placement ambiguity rejects; operator lookup never grants proof/write authority.

## PDR-051 Add bounded deterministic synthesis/CEGIS and residual-only hybrid repair

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repair-synthesis
- Depends on: PDR-014, PDR-024, PDR-041, PDR-050
- Goal id: PDR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, test/api/test_agent_supervisor_program_repair_synthesis.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_repair_synthesis.py test/api/test_agent_supervisor_deterministic_doctor_synthesis.py test/api/test_agent_supervisor_counterexample_guided_tactician.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/repair-synthesis
- Parallel lane: pdr-repair
- Resource class: cpu-proof-solver
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, ipfs_accelerate_py/agent_supervisor/proof/counterexample_guided_tactician.py, test/api/test_agent_supervisor_program_repair_synthesis.py
- Interfaces: ProgramRepairSynthesizer@1, DeterministicDoctorSynthesizer@2
- Conflict policy: Deterministic Doctor remains no-model; any LLM path is a separately named hybrid proposal service under independent policy.
- Preconditions: Reviewed repair operators and authoritative proof receipts exist.
- Effects: Constraint/e-graph/enumerative/CEGIS search closes more repairs before bounded LLM syntax generation.
- Evidence subset: grammar/operator search, SMT/CHC/MaxSAT constraints, counterexamples, candidate overlay CIDs, residual packet and usage receipts
- Acceptance: Synthesis searches only reviewed operators/grammars under exact obligations, bounds and roots; CEGIS independently validates counterexamples and terminates on fixed budgets; e-graph/equality rewrites prove equivalence under declared theory; every candidate is proposal-only. If deterministic search leaves behavior-fixed syntax debt, hybrid service receives exact target/path/semantics/postconditions/tests and may not change authority/dependencies/meaning; deterministic mode proves zero model calls; malformed, extra-file/import/dependency, non-idempotent and scope-widening outputs fail.

## PDR-052 Implement real isolated worktree/VFS mutation, ref CAS, and rollback

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-transaction
- Depends on: PDR-002, PDR-010, PDR-041, PDR-050
- Goal id: PDR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transaction.py, test/api/test_agent_supervisor_doctor_worktree_adapter.py, test/integration/test_agent_supervisor_doctor_transaction_live.py
- Validation: python -m pytest test/api/test_agent_supervisor_doctor_worktree_adapter.py test/integration/test_agent_supervisor_doctor_transaction_live.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-transaction
- Parallel lane: pdr-repair
- Resource class: cpu-large
- Resource stage: implementation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transaction.py, test/api/test_agent_supervisor_doctor_worktree_adapter.py, test/integration/test_agent_supervisor_doctor_transaction_live.py
- Interfaces: DoctorWorktreeAdapter@1, DeterministicDoctorTransaction@2
- Conflict policy: Extend fenced worktree/lease/checkpoint/merge authority; default applicator/restore may not simulate effects.
- Preconditions: Runtime, authoritative proof and reviewed operator are available; plan is admitted.
- Effects: Candidate patches change real bytes in a disposable isolated overlay and can be atomically committed or exactly restored.
- Evidence subset: OS sandbox capability, clean base, before/after blob/tree/forest roots, writer lease/fence, checkpoint/fsync, ref CAS, SCC group transaction and rollback
- Acceptance: Adapter creates an allowlisted disposable git worktree or VFS overlay, verifies no symlink/path/network/process/secret escape, applies exact before-hash edits, rereads bytes, computes changed blob/tree/forest CIDs, and requires a nonempty expected change. Transaction owns writer lease/checkpoint/ref CAS, applies complete impact/SCC groups atomically, fsyncs durable intent/effects, and cannot report COMMITTED on a no-op/default fake applicator. Crash/tamper at every boundary restores exact bytes/ref/gitlinks or quarantines with proof; default restore independently compares roots.

## PDR-053 Run live reparse/static/security/replan/reprove fixed-point stages

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-fixed-point
- Depends on: PDR-027, PDR-041, PDR-042, PDR-052
- Goal id: PDR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_live_fixed_point.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_fixed_point.py, ipfs_accelerate_py/agent_supervisor/security_contract_analysis.py, test/api/test_agent_supervisor_deterministic_doctor_live_fixed_point.py
- Validation: python -m pytest test/api/test_agent_supervisor_deterministic_doctor_live_fixed_point.py test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py test/api/test_agent_supervisor_security_contract_analysis.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-fixed-point
- Parallel lane: pdr-repair
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_live_fixed_point.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_fixed_point.py, ipfs_accelerate_py/agent_supervisor/security_contract_analysis.py, test/api/test_agent_supervisor_deterministic_doctor_live_fixed_point.py
- Interfaces: DeterministicDoctorLiveFixedPoint@1, DoctorFixedPointReceipt
- Conflict policy: Keep pure validators; add a live producer that invokes actual stages and then supplies their sealed receipts.
- Preconditions: A real isolated candidate tree exists.
- Effects: Completion is based on independently renewed current-tree evidence rather than caller-supplied flags.
- Evidence subset: reparse/rebuild, index/cache invalidation, redelta, impact closure, static/type/effect/taint/supply-chain, IntentIR/SecurityIR/hyperproperty, tests, replan, reprove and root verification
- Acceptance: Runner independently reparses and rebuilds, re-indexes changed scope, invalidates dependency-local caches, re-diffs contracts, recloses all consumers/SCCs, extracts code security facts, checks intent/code forbidden logic and required security/hyperproperties, runs impact-selected tests/static/model checks, replans and replays kernel proofs. Second-order findings trigger another bounded iteration; oscillation/unchanged residual/budget/capability loss aborts and rolls back. Prebuilt fixed-point mappings or booleans cannot complete.

## PDR-054 Select repair candidates with independent multi-method validation

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repair-selection
- Depends on: PDR-012, PDR-013, PDR-014, PDR-050, PDR-051, PDR-052, PDR-053
- Goal id: PDR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/repair_candidate_portfolio.py, test/api/test_agent_supervisor_repair_candidate_portfolio.py
- Validation: python -m pytest test/api/test_agent_supervisor_repair_candidate_portfolio.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/repair-selection
- Parallel lane: pdr-repair
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/repair_candidate_portfolio.py, test/api/test_agent_supervisor_repair_candidate_portfolio.py
- Interfaces: RepairCandidatePortfolio@1, RepairCandidateDecision@1
- Conflict policy: Validation outcomes are evidence, not weighted authority; hard failures cannot be averaged away.
- Preconditions: Deterministic/hybrid candidates can be rendered in isolated overlays and live fixed-point checks exist.
- Effects: Multiple plausible patches are compared by independent correctness/security/minimality evidence.
- Evidence subset: property-based, fuzz, concolic, mutation, differential, metamorphic, sanitizers, static/model/proof/security checks, changed scope and cost
- Acceptance: Portfolio runs property-based/fuzz/concolic when supported, mutation tests against independent oracle, differential/metamorphic checks, sanitizers/static/model checks and proof/security gates under fixed seeds/budgets. It records flaky/unavailable lanes, rejects self-authored tests and candidate-as-oracle, requires all hard obligations, ranks only hard-admissible candidates by minimal blast radius/resource cost, preserves correct abstention, and proves selection/replay identity.

## PDR-055 Feed Doctor residuals into plan steering and bounded derived refill

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: doctor-refill
- Depends on: PDR-028, PDR-030, PDR-033, PDR-043, PDR-053, PDR-054
- Goal id: PDR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py, ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py, test/api/test_agent_supervisor_doctor_plan_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_doctor_plan_refill.py test/api/test_agent_supervisor_plan_supervisor_service.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor-refill
- Parallel lane: pdr-self-improvement
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py, ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py, test/api/test_agent_supervisor_doctor_plan_refill.py
- Interfaces: DoctorPlanResidual@1, PlanSteerRequest@1, ObjectiveWorkProposal
- Conflict policy: Residuals propose plan deltas or derived work; they never edit active/accepted task specs or seed boards.
- Preconditions: Live Doctor fixed point and create/steer service exist.
- Effects: Unsupported/open/failed Doctor obligations become small targeted successor tasks rather than repeated broad LLM prompts.
- Evidence subset: residual obligation/counterexample/frontier IDs, changed scope, attempted strategies, cache hits, required capability and proposed task delta
- Acceptance: Successful fixed point emits no work. New residuals are deduplicated by exact issue/obligation/root/attempt identities, mapped to the existing plan as append-only successors where possible, and otherwise emitted as bounded `ObjectiveWorkProposal` records. Unchanged failures back off; capability gaps name the exact provider/conformance work; no completion/mutation authority is granted; generated tasks target minimal files/context and go only to the derived runtime source after PDR-081.

## PDR-060 Bind receipt lineage and gate optional ZKP to an approved threat model

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: attestation
- Depends on: PDR-002, PDR-010, PDR-014, PDR-027, PDR-053
- Goal id: PDR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/planner_doctor_attestation.py, docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md, test/api/test_agent_supervisor_planner_doctor_attestation.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_attestation.py test/api/test_agent_supervisor_program_analysis_zkp.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/attestation
- Parallel lane: pdr-proof
- Resource class: cpu-proof-solver
- Resource stage: validation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/planner_doctor_attestation.py, docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md, test/api/test_agent_supervisor_planner_doctor_attestation.py
- Interfaces: ReasoningRunManifest@1, PlannerDoctorAttestation@1
- Conflict policy: Reuse proof attestation and `program_analysis_zkp`; no new assurance levels or generic correctness claims.
- Preconditions: Exact snapshot/cache/admission/mutation/fixed-point lineage exists.
- Effects: Consumers can verify committed inputs, receipt ancestry and narrowly approved private computations without receiving source bodies or witnesses.
- Evidence subset: CID/Merkle/signature chain, circuit/program/version/public input/private witness policy and independent verification
- Acceptance: Manifest links Planner/Doctor/cache/plan/permit/mutation/fixed-point/benchmark/promotion CIDs without collapsing evidence types. Wrong preimage/order/root/run replay fails. Threat model names the exact privacy/computation claim and why ordinary signatures/Merkle proofs are insufficient. Optional ZKP binds fixed code/circuit and inputs, protects witness, and verifies independently; unavailable/failed/simulated backends remain candidate/unavailable and never emit production `ATTESTED`, semantic correctness, inventory completeness or translator-soundness claims.

## PDR-070 Execute live paired Planner/Doctor benchmarks on hermetic repositories

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: live-benchmark
- Depends on: PDR-003, PDR-030, PDR-033, PDR-040, PDR-053, PDR-054, PDR-071, PDR-072
- Goal id: PDR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_live_benchmark.py, test/api/test_agent_supervisor_planner_doctor_live_benchmark.py, test/fixtures/agent_supervisor/planner_doctor_live/manifest.json
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_live_benchmark.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/live-benchmark
- Parallel lane: pdr-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_live_benchmark.py, test/api/test_agent_supervisor_planner_doctor_live_benchmark.py, test/fixtures/agent_supervisor/planner_doctor_live/manifest.json
- Interfaces: PlannerDoctorLiveBenchmark@1, LiveBenchmarkPairReceipt@1
- Conflict policy: Extend current benchmark contracts with live producer receipts; retain synthetic populations as conformance-only.
- Preconditions: Real Planner/Doctor services, runtime execution, transaction, fixed point and benchmark manifest exist.
- Effects: Benchmark results come from actual code/service execution rather than fixture expectations or hardcoded counters.
- Evidence subset: paired baseline/challenger repository runs, service/plan/repair/proof/validation/merge receipts, cache and concurrency strata
- Acceptance: Runner creates hermetic mini-repositories with seeded prompt/planning/contract/security/repair cases, invokes real create/steer and Doctor services, executes admitted tasks/repairs in isolated worktrees, runs real solvers/tools when capability-certified, and obtains results from independent validations. It never reads fixture `expected` fields to choose diagnosis/disposition/repair/completion. Current deterministic Doctor and V2 synthetic benchmarks are labeled model/conformance evidence only. Paired inputs match exactly; cold/exact/delta/restart and 1/2/4/configured-maximum runs are replayable; skips cannot qualify promotion.

## PDR-071 Attribute wall time, tokens, process-tree resources, GPU, I/O, and cost

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark-telemetry
- Depends on: PDR-003, PDR-033
- Goal id: PDR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, ipfs_accelerate_py/agent_supervisor/runtime/scheduler_metrics.py, ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_token_ledger.py, test/api/test_agent_supervisor_benchmark_telemetry.py
- Validation: python -m pytest test/api/test_agent_supervisor_benchmark_telemetry.py test/api/test_agent_supervisor_supervisor_token_ledger.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/telemetry
- Parallel lane: pdr-telemetry
- Resource class: cpu-medium
- Resource stage: validation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, ipfs_accelerate_py/agent_supervisor/runtime/scheduler_metrics.py, ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_token_ledger.py, test/api/test_agent_supervisor_benchmark_telemetry.py
- Interfaces: BenchmarkCausalSpan@1, BenchmarkResourceMeasurement@1
- Conflict policy: Join existing scheduler/resource/provider/token metrics by causal span; do not replace capacity admission or invent zero observations.
- Preconditions: Live benchmark produces task/run/attempt/process/provider identities.
- Effects: Every paired case has attributable clock, token, resource and provider-use evidence.
- Evidence subset: monotonic spans, process tree, provider usage settlement, host/hardware profile and missing-sensor status
- Acceptance: Measure end-to-end/makespan/critical path/queue/merge waits, provider-native input/output/reused/retry/cancelled tokens and calls, process-tree user/system CPU, peak RSS/GiB-seconds, I/O bytes, disk/artifact growth, child count, GPU utilization/VRAM/GPU-seconds, network bytes, provider quota/cost and optional energy estimate. Bind tokenizer/model/endpoint/hardware and span ancestry. Missing/permission-denied sensors are `unavailable`, never zero. Kill/cancel/retry and daemon children remain attributed exactly once; serialized counters cannot self-certify without source span replay.

## PDR-072 Add protected hidden quality oracles, adversarial cases, and ablations

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: quality-oracle
- Depends on: PDR-002, PDR-003, PDR-052, PDR-053, PDR-054, PDR-071
- Goal id: PDR-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_quality_oracle.py, test/fixtures/agent_supervisor/planner_doctor_holdout/oracle.manifest.json, test/api/test_agent_supervisor_planner_doctor_quality_oracle.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_quality_oracle.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/quality-oracle
- Parallel lane: pdr-benchmark
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_quality_oracle.py, test/fixtures/agent_supervisor/planner_doctor_holdout/oracle.manifest.json, test/api/test_agent_supervisor_planner_doctor_quality_oracle.py
- Interfaces: PlannerDoctorQualityOracle@1, PlannerDoctorAblation@1
- Conflict policy: Oracle bodies and holdout answers are operator-owned, read-only and outside candidate context/worktrees.
- Preconditions: Live paired runner and telemetry exist.
- Effects: Solution quality and anti-gaming are independently measured rather than inferred from task status or candidate tests.
- Evidence subset: seeded-defect truth, independent tests/properties/mutations/proofs/security cases, API/schema compatibility, patch minimality, recurrence/rollback and subsystem ablations
- Acceptance: Oracle measures defect/localization precision/recall, repair success/correct abstention, acceptance coverage, hidden tests, mutation score, property/fuzz/differential/metamorphic outcomes, proof coverage/kernel reconstruction, counterexample validity, SecurityIR/IntentIR conformance, API/schema compatibility, blast radius/minimality, flake/post-merge recurrence and exact rollback. Candidate-generated tests/proofs cannot define truth. Adversarial cases cover injection, poisoned indexes/caches, forged receipts, missing callers, dynamic/native/concurrency frontiers, sandbox/transaction/rollback/fixed-point faults, resource/telemetry loss and reward hacking. Ablations isolate AST/KG/vector/proof/cache/LLM/parallel components.

## PDR-080 Invoke bounded live self-improvement epochs from supervisor lifecycle

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: epoch-runtime
- Depends on: PDR-055, PDR-070, PDR-071, PDR-072
- Goal id: PDR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py, ipfs_accelerate_py/agent_supervisor/self_improvement/self_improvement_v2.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py, test/api/test_agent_supervisor_planner_doctor_epoch.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_epoch.py test/api/test_agent_supervisor_self_improvement_v2.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/epoch-runtime
- Parallel lane: pdr-self-improvement
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py, ipfs_accelerate_py/agent_supervisor/self_improvement/self_improvement_v2.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py, test/api/test_agent_supervisor_planner_doctor_epoch.py
- Interfaces: PlannerDoctorEpoch@1, V2CausalReceipt
- Conflict policy: Compose existing v2 epoch/rollout contracts with live producers; no unbounded daemon loop or implicit mutation authority.
- Preconditions: Live benchmark, telemetry, quality oracle and Doctor residual bridge exist.
- Effects: The running supervisor can execute finite baseline/propose/shadow/evaluate cycles with durable restart state.
- Evidence subset: epoch manifest, baseline/challenger roots, budgets, stage spans, live paired reports, stop/rollback reasons
- Acceptance: Lifecycle invokes the epoch controller under explicit mode/policy; freezes anchors and budgets; uses one isolated challenger; persists every state transition; resumes idempotently after crash; limits epochs/wall/CPU/memory/GPU/disk/tokens/cost/processes/storage/model calls/repairs; stops on safety or quality regression, unchanged residual, no admitted improvement, oracle/telemetry loss, rollback failure or budget exhaustion. No test-only `run_self_improvement_epoch` path may masquerade as daemon integration.

## PDR-081 Compile benchmark/Doctor residuals into bounded derived goals and tasks

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: derived-refill
- Depends on: PDR-033, PDR-055, PDR-080
- Goal id: PDR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, test/api/test_agent_supervisor_planner_doctor_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_refill.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/derived-refill
- Parallel lane: pdr-self-improvement
- Resource class: cpu-medium
- Resource stage: analysis
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, test/api/test_agent_supervisor_planner_doctor_refill.py
- Interfaces: PlannerDoctorRefill@1, ObjectiveWorkProposal, DuckDBTaskSource
- Conflict policy: Write only the configured derived runtime source/CAS; seed plan/objectives/taskboard/config/holdout/oracle remain protected.
- Preconditions: Live bounded epoch emits verified residuals.
- Effects: The supervisor gains small, evidence-targeted successor work without consuming the seed board or unlimited context.
- Evidence subset: benchmark gap, Doctor residual, proof/security/capability/resource regression, novelty/dedup/dependency/acceptance/context packet and admission receipt
- Acceptance: At most 8 goals/24 tasks per epoch and 48 open tasks; proposals carry exact source roots, goal/subgoal/task hierarchy, minimal files/context, acceptance/validation, resource/conflict/dependencies and stop policy; duplicates and semantically unchanged failures back off; replay is no-op; no candidate can edit anchors, authorize itself, lower thresholds or mark complete; generated work enters the separate DuckDB source only after independent plan/admission/parallel compilation.

## PDR-082 Gate baseline/challenger rollout with quality-safe Pareto and anti-gaming checks

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rollout
- Depends on: PDR-002, PDR-003, PDR-080, PDR-081
- Goal id: PDR-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_rollout.py, test/api/test_agent_supervisor_planner_doctor_rollout.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_rollout.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/rollout
- Parallel lane: pdr-self-improvement
- Resource class: cpu-medium
- Resource stage: validation
- Implementation timeout seconds: 9000
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_rollout.py, test/api/test_agent_supervisor_planner_doctor_rollout.py
- Interfaces: PlannerDoctorRolloutPolicy@1, PlannerDoctorPromotionReceipt@1
- Conflict policy: Extend v2 rollout/current-tree/rollback patterns; benchmark or candidate cannot edit policy or choose its comparator.
- Preconditions: Live epoch and derived refill exist with protected benchmark anchors.
- Effects: A challenger can advance through observe/shadow/assist/canary only when paired evidence supports it, and rolls back immediately on regression.
- Evidence subset: qualification/current-tree/holdout paired reports, confidence/non-inferiority, Pareto metrics, safety floors, anchor integrity, canary and rollback
- Acceptance: Quality/safety/authority are non-compensable; denominators and paired inputs match; synthetic/skipped/unavailable required evidence rejects; preregistered statistical/non-inferiority method prevents cherry-picking; at least one material Pareto improvement is required without resource ceiling regression; current-tree re-evaluation and independent holdout follow qualification; anti-gaming detects oracle/manifest/metric/task-status/context leakage and work shifting; automatic remains disabled until separate operator-approved fresh-root evidence; kill switch and exact rollback override all scores.

## PDR-090 Prove live E2E, transport/projection parity, adversarial safety, and chaos recovery

- Status: pending
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-qualification
- Depends on: PDR-032, PDR-033, PDR-053, PDR-060, PDR-072, PDR-082
- Goal id: PDR-G100
- Outputs: test/integration/test_agent_supervisor_planner_doctor_e2e.py, test/integration/test_agent_supervisor_planner_doctor_chaos.py, test/integration/test_agent_supervisor_planner_doctor_security.py
- Validation: python -m pytest test/integration/test_agent_supervisor_planner_doctor_e2e.py test/integration/test_agent_supervisor_planner_doctor_chaos.py test/integration/test_agent_supervisor_planner_doctor_security.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/release-qualification
- Parallel lane: pdr-release
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 18000
- Predicted files: test/integration/test_agent_supervisor_planner_doctor_e2e.py, test/integration/test_agent_supervisor_planner_doctor_chaos.py, test/integration/test_agent_supervisor_planner_doctor_security.py
- Interfaces: PlannerDoctorQualification@1
- Conflict policy: Test only against protected fixtures/policies; never weaken a floor or mark a skipped/unavailable case pass.
- Preconditions: Planner, Doctor, repair, attestation, live benchmark, telemetry/oracle and rollout are integrated.
- Effects: Produces independent qualification receipts for release and operator rollout.
- Evidence subset: create/steer/diagnose/repair/benchmark/refill across Python/CLI/MCP, Markdown/DuckDB, cold/warm/delta/restart and 1/2/4/configured-maximum lanes
- Acceptance: Exact transport/projection/restart/replay identities hold; adversarial injection/secret/path/policy/IR/security/provider/cache/proof/ZKP/task/oracle/authority attacks fail; missing callers, poisoned indexes, forged receipts, fake transactions/fixed points and model calls in deterministic mode are caught; chaos covers provider/tool/telemetry loss, process crash/PID reuse, worktree/lease/ref-CAS/merge/task-source split brain, rollback and repository drift; safety floors are zero, resource bounds hold, all required live checks run with no skip, and rollback restores exact roots.

## PDR-091 Deliver protected launch profiles, lifecycle controls, kill switch, and runbook

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operations
- Depends on: PDR-032, PDR-033, PDR-040, PDR-080, PDR-082, PDR-090
- Goal id: PDR-G100
- Outputs: scripts/ops/agent_supervisor/proof_directed_planner_doctor.py, docs/guides/PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md, test/api/test_agent_supervisor_planner_doctor_operations.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_operations.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/operations
- Parallel lane: pdr-release
- Resource class: cpu-medium
- Resource stage: implementation
- Implementation timeout seconds: 10800
- Predicted files: scripts/ops/agent_supervisor/proof_directed_planner_doctor.py, docs/guides/PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md, test/api/test_agent_supervisor_planner_doctor_operations.py
- Interfaces: PlannerDoctorOperations@1
- Conflict policy: Launcher reads protected anchors and writes only isolated state/worktrees/logs/derived task source; no silent policy elevation.
- Preconditions: Qualification, live epoch and rollout gates pass in shadow.
- Effects: Operators can validate, plan, start, status, stop, restart, pause, drain, benchmark, promote one stage, roll back and engage kill switch.
- Evidence subset: config validation, initial ready set, dependency/conflict/resource lanes, PID/process tree, health/events, budget, protected paths, promotion/rollback receipts
- Acceptance: Report-only/shadow defaults; `automatic`, Doctor mutation and refill are off until their prerequisite task receipts exist. Launcher validates clean target, exact gitlinks/capabilities, board/objective DAG, protected anchors, isolated state/worktree/merge queue, provider/resource telemetry and maximum six seed lanes. Operations are idempotent/fenced/restartable; kill switch forces report-only, cancels future dispatch safely and blocks promotion; runbook covers capability degradation, stale state, rollback/quarantine, held-out evaluation and recovery without editing anchors.

## PDR-092 Issue the independently replayed terminal PDR release receipt

- Status: pending
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release
- Depends on: PDR-027, PDR-033, PDR-053, PDR-060, PDR-070, PDR-071, PDR-072, PDR-080, PDR-081, PDR-082, PDR-090, PDR-091
- Goal id: PDR-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_release.py, docs/architecture/PROOF_DIRECTED_PLANNER_DOCTOR_RELEASE.md, test/api/test_agent_supervisor_planner_doctor_release.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_release.py test/integration/test_agent_supervisor_planner_doctor_e2e.py -q
- Board namespace: agent-supervisor-proof-directed-planner-doctor-v1
- Bundle: agent-supervisor/proof-directed-planner-doctor/release
- Parallel lane: pdr-release
- Resource class: cpu-large
- Resource stage: validation
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_release.py, docs/architecture/PROOF_DIRECTED_PLANNER_DOCTOR_RELEASE.md, test/api/test_agent_supervisor_planner_doctor_release.py
- Interfaces: PlannerDoctorReleaseReceipt@1
- Conflict policy: Release module reads current source evidence and protected anchors; it cannot edit task/objective status, policies or benchmark results.
- Preconditions: Every producing task is terminal and every child goal has current independently replayable evidence.
- Effects: Provides the only PDR terminal release evidence; it grants no automatic mode by itself.
- Evidence subset: current forest/gitlinks, contracts/capabilities, live Planner/Doctor/repair/fixed-point, transport/projection/runtime, attestation, benchmark/telemetry/oracle, epoch/refill/rollout/chaos/operations receipts
- Acceptance: Validator reloads every required source receipt/artifact, recomputes CIDs/preimages/current roots and child-goal coverage, rejects stale/synthetic/skipped/forged/self-authored/incomplete evidence, proves zero safety floors and exact rollback, and distinguishes task completion from objective completion. Replaying the same current inputs is identity-equivalent. Receipt documents unavailable optional capabilities without converting them to pass and keeps automatic promotion subject to a separate later held-out current-tree operator decision.
