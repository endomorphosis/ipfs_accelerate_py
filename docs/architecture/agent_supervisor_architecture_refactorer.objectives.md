# Proof-Carrying Architecture Refactorer objective heap

Machine-ingestible objective state for board namespace
`agent-supervisor-proof-carrying-architecture-refactorer-v1`. Goals and
subgoals are ordinary DuckDB goal records connected by `parent_goal_cid` and
goal edges. Markdown is a sealed bootstrap projection, never completion
authority.

## Goal tree

```text
PCAR-G000  Reduce recurring architectural reasoning with proof-carrying refactors
|-- PCAR-G010  Tranche 1: establish architecture truth
|   |-- PCAR-G011  Seal the baseline and inventory
|   |-- PCAR-G012  Build IR, graph, entropy, cone, and context analyses
|   `-- PCAR-G013  Resolve canonical ownership and duplicate authorities
|-- PCAR-G020  Tranche 2: construct bounded refactor candidates
|   |-- PCAR-G021  Discover duplicates, contracts, boundaries, and operators
|   `-- PCAR-G022  Model public surface, state, legacy, and simulation
|-- PCAR-G030  Tranche 3: execute and validate bounded refactors
|   |-- PCAR-G031  Isolate execution and compare behavior, effects, and translation
|   `-- PCAR-G032  Plan autonomy, monitor drift, and audit sibling contracts
`-- PCAR-G040  Tranche 4: qualify the current tree
    |-- PCAR-G041  Generate projections and typed public controls
    `-- PCAR-G042  Benchmark, attack, gate, and report
```

## PCAR-G000 Reduce recurring architectural reasoning with proof-carrying refactors

- Status: active
- Parent:
- Depends on:
- Priority: P0
- Track: pcar-root
- Goal: Deliver one canonical ArchitectureIR and a bounded proof-preserving refactor system that reduces semantic entropy, dependency cones, context cost, repeated architectural inference, validation amplification, and merge risk without weakening behavior, authority, effects, evidence, public contracts, or rollback.
- Completion contract: All four tranche goals are accepted against one exact merged tree and architecture root; every non-compensable invariant passes; zero mandatory task, proof, validation, mutation, merge, or release obligation remains unresolved; final reports distinguish verified claims, unrun checks, residual gaps, blockers, and rollback target.
- Evidence: pcar/architecture-root@1, pcar/refactor-receipt@1, pcar/qualification-report@1, pcar/release-manifest@1
- Acceptance criteria: exact-current-tree; one-architecture-ir; invariant-conjunction; accepted-child-goals; settled-control-state; verified-rollback; truthful-final-report
- Outputs: docs/architecture/architecture_refactorer_inventory/final_qualification_report.json, docs/architecture/architecture_refactorer_inventory/architecture_root_manifest.json
- Validation: python3 scripts/validate_agent_supervisor_architecture_refactorer_board.py --check-all
- Acceptance: No task-board state, model claim, entropy score, generated document, or DuckLake projection is completion evidence; only exact current-tree receipts from declared producers may satisfy the root.
- Gap task: PCAR-000 through PCAR-031

## PCAR-G010 Tranche 1: establish architecture truth

- Status: active
- Parent: PCAR-G000
- Depends on:
- Priority: P0
- Track: architecture-truth
- Goal: Seal the current source and prerequisites, inventory the relevant implementation and state forest, build canonical architecture facts and independent entropy metrics, and resolve authority ownership before any broad refactoring.
- Completion contract: PCAR-000 through PCAR-007 are accepted; every graph fact is source-bound with confidence/freshness/tree/content identity; initial concern ownership and duplicate-authority findings are explicit; optional missing prerequisites are typed blockers rather than simulated implementations.
- Evidence: pcar/baseline-seal@1, pcar/architecture-ir@1, pcar/entropy-report@1, pcar/authority-ownership-graph@1
- Acceptance criteria: sealed-baseline; classified-prerequisites; complete-inventory-scope; closed-graph-contract; measured-entropy; canonical-owner-findings
- Outputs: docs/architecture/architecture_refactorer_inventory/sealed_current_tree_baseline.json, docs/architecture/architecture_refactorer_inventory/authority_findings.json
- Validation: focused architecture_refactorer tranche-1 tests and prerequisite qualification receipts
- Acceptance: Tranche 2 may use the accepted architecture root; broad autonomous refactoring remains prohibited.
- Gap task: PCAR-000 through PCAR-007

## PCAR-G011 Seal the baseline and inventory

- Status: active
- Parent: PCAR-G010
- Depends on:
- Priority: P0
- Track: baseline-inventory
- Goal: Bind source, tree, branch protection, package/tool versions, gitlinks, operation catalog, schemas, tests, prerequisites, packages, entrypoints, authorities, and state stores to exact current-tree evidence.
- Completion contract: The revision forest and required inspection roots are complete; source and test identities justify every prerequisite status; unknowns remain typed; no sibling is written; qualification results distinguish run, pass, fail, skip, and not-run.
- Evidence: pcar/baseline-seal@1, pcar/prerequisite-matrix@1, pcar/repository-inventory@1
- Acceptance criteria: source-tree-seal; gitlink-pins; operation-catalog-revision; proof-receipt-schemas; qualified-test-ledger; scoped-inventory
- Outputs: docs/architecture/architecture_refactorer_inventory/sealed_current_tree_baseline.json, docs/architecture/architecture_refactorer_inventory/repository_inventory.json
- Validation: baseline and repository inventory schema/content tests
- Acceptance: Planning prose and historical receipts are never substituted for source and test identity.
- Gap task: PCAR-000, PCAR-001

## PCAR-G012 Build IR, graph, entropy, cone, and context analyses

- Status: active
- Parent: PCAR-G010
- Depends on: PCAR-G011
- Priority: P0
- Track: architecture-analysis
- Goal: Define and extract closed ArchitectureIR facts, then calculate independently auditable semantic-entropy, dependency-cone, context-burden, and validation-amplification dimensions.
- Completion contract: Contracts reject unknown fields and noncanonical identities; graph extraction binds source provenance and conservative uncertainty; metrics retain raw measures and frozen evidence rather than treating a score as proof.
- Evidence: pcar/architecture-ir@1, pcar/architecture-graph@1, pcar/semantic-entropy-report@1, pcar/context-benchmark-measure@1
- Acceptance criteria: deterministic-round-trip; closed-node-edge-confidence; exact-provenance; conservative-dynamic-analysis; independent-metric-dimensions
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/architecture_ir.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/graph_builder.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/entropy.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/metrics.py
- Validation: focused contracts, IR, graph, entropy, dependency-cone, and context-burden tests
- Acceptance: Heuristic and opaque facts widen analysis only and never establish ownership, equivalence, dead code, or safe removal.
- Gap task: PCAR-002 through PCAR-005

## PCAR-G013 Resolve canonical ownership and duplicate authorities

- Status: active
- Parent: PCAR-G010
- Depends on: PCAR-G011, PCAR-G012
- Priority: P0
- Track: authority
- Goal: Resolve the initial concern set to one canonical authority with explicit adapters/projections/quarantine, and detect competing production authorities or missing arbitration.
- Completion contract: All targeted concerns have evidence-bound ownership dispositions; unknown ownership and multiple production authorities produce hard blockers; bypass, simulation flow, surface mismatch, re-export authority, obsolete test, receipt-producer, and state-owner findings are represented.
- Evidence: pcar/authority-ownership-graph@1, pcar/duplicate-authority-finding@1
- Acceptance criteria: concern-coverage; one-canonical-owner; explicit-adapters; formal-arbitration-or-blocker; duplicate-authority-detections
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/authority_graph.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/duplicate_authority.py
- Validation: authority ownership and duplicate-authority positive/negative tests
- Acceptance: Content identity is not inferred to be authority and no finding alone authorizes a change.
- Gap task: PCAR-006, PCAR-007

## PCAR-G020 Tranche 2: construct bounded refactor candidates

- Status: active
- Parent: PCAR-G000
- Depends on: PCAR-G010
- Priority: P0
- Track: candidates
- Goal: Discover semantic duplication, mine candidate contracts, synthesize coherent interfaces, define the closed operator grammar, and model public surface, state, legacy, compatibility, fixtures, and simulations.
- Completion contract: PCAR-008 through PCAR-016 are accepted; every proposal separates candidate evidence from authority; state and public contracts are explicit; quarantine checks prove noncanonical outcomes cannot satisfy production predicates.
- Evidence: pcar/duplicate-candidate@1, pcar/contract-candidate@1, pcar/boundary-proposal@1, pcar/refactor-operator@1, pcar/quarantine-flow-proof@1
- Acceptance criteria: multi-signal-duplicates; ambiguity-preserved; bounded-operators; public-surface-manifest; unique-state-owner; quarantine-proof
- Outputs: docs/architecture/architecture_refactorer_inventory/public_surface_manifest.json, docs/architecture/architecture_refactorer_inventory/legacy_simulation_inventory.json
- Validation: focused architecture_refactorer tranche-2 tests
- Acceptance: Text similarity, repeated code behavior, tests, implementation checks, and entropy improvements remain non-authoritative evidence.
- Gap task: PCAR-008 through PCAR-016

## PCAR-G021 Discover duplicates, contracts, boundaries, and operators

- Status: active
- Parent: PCAR-G020
- Depends on: PCAR-G013
- Priority: P0
- Track: candidate-discovery
- Goal: Combine bounded semantic duplicate signals and validated rewrite domains with conservative contract extraction and interface synthesis, then express changes only through the closed declarative grammar.
- Completion contract: False duplicates and heuristic critical rewrites are rejected; overlapping behavior is preserved for interface extraction; conflicts emit ContractAmbiguity; each boundary/operator declares authority/effect/API/state impact, scope, validation, proof, and rollback.
- Evidence: pcar/duplicate-candidate@1, pcar/egraph-rule-set@1, pcar/contract-ambiguity@1, pcar/boundary-proposal@1, pcar/refactor-operator@1
- Acceptance criteria: multi-signal-classification; validated-rewrites; ambiguity-not-silenced; stable-boundary-contract; closed-operator-vocabulary
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/duplicate_detector.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/egraph_normalizer.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/contract_extractor.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/boundary_synthesizer.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/refactor_operators.py
- Validation: duplicate, e-graph, contract ambiguity, boundary, and operator bound tests
- Acceptance: No arbitrary executable refactor policy is admitted.
- Gap task: PCAR-008 through PCAR-012

## PCAR-G022 Model public surface, state, legacy, and simulation

- Status: active
- Parent: PCAR-G020
- Depends on: PCAR-G013
- Priority: P0
- Track: ownership-quarantine
- Goal: Classify every exported symbol and mutable semantic fact, inventory legacy/compatibility/fixture/simulation reachability, and enforce explicit non-production namespaces and data-flow barriers.
- Completion contract: Stable public contracts name owners and consumers; each mutable fact has one authoritative store; migration cannot create indefinite dual authority; production success/proof/completion/release cannot consume a quarantined origin.
- Evidence: pcar/public-surface-manifest@1, pcar/state-ownership-model@1, pcar/legacy-simulation-inventory@1, pcar/quarantine-flow-proof@1
- Acceptance criteria: export-classification; exact-stable-contracts; one-authoritative-store; consumer-aware-deprecation; static-and-dynamic-quarantine
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/public_surface.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/state_ownership.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/legacy_paths.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/quarantine.py
- Validation: public surface, state conflict, reachability, compatibility and simulation quarantine tests
- Acceptance: Dynamic-loading uncertainty prevents dead classification, and compatibility is not deleted before usage/migration/rollback evidence.
- Gap task: PCAR-013 through PCAR-016

## PCAR-G030 Tranche 3: execute and validate bounded refactors

- Status: active
- Parent: PCAR-G000
- Depends on: PCAR-G020
- Priority: P0
- Track: verified-refactoring
- Goal: Execute declarative candidates in isolated worktrees; compare behavior, effects, authority, and generated translations; plan within the autonomy ceiling; monitor drift; and audit sibling contracts read-only.
- Completion contract: PCAR-017 through PCAR-025 are accepted; every admitted execution has independent validation and rollback; hard gates cannot be optimized away; missing procedure/meta-controller capabilities remain narrow blockers; sibling writes are impossible.
- Evidence: pcar/refactor-execution@1, pcar/differential-report@1, pcar/translation-validation@1, pcar/planner-decision@1, pcar/drift-delta@1, pcar/cross-repository-audit@1
- Acceptance criteria: isolated-worktree; differential-and-effect-conjunction; independent-translation-check; autonomy-ceiling; drift-deduplication; read-only-audit
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/executor.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/planner.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/drift_monitor.py
- Validation: focused architecture_refactorer tranche-3 tests and rollback exercises
- Acceptance: No proposal, procedure, task status, or candidate may authorize or promote itself.
- Gap task: PCAR-017 through PCAR-025

## PCAR-G031 Isolate execution and compare behavior, effects, and translation

- Status: active
- Parent: PCAR-G030
- Depends on: PCAR-G020
- Priority: P0
- Track: execution-validation
- Goal: Build isolated declarative candidate execution with exact scope and rollback, then compare behavior, errors, receipts, effects, authority, and generated translations through independent evidence.
- Completion contract: Valid/invalid/boundary inputs, exceptions, state, receipts, performance, cancellation, timeout, and restart cases are covered; only contract-admitted differences pass; inconclusive required validators reject; the procedure compiler boundary remains a narrow adapter.
- Evidence: pcar/refactor-execution@1, pcar/differential-report@1, pcar/effect-authority-comparison@1, pcar/translation-validation@1, pcar/procedure-adapter@1
- Acceptance criteria: bounded-isolation; exact-rollback; complete-differential-matrix; no-effect-expansion; no-authority-weakening; independent-refinement
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/executor.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/differential.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/effect_comparison.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/translation_validation.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/procedure_adapter.py
- Validation: executor, rollback, differential, effect/authority, translation, and procedure adapter tests
- Acceptance: Unavailable or inconclusive required validation cannot be reclassified as success.
- Gap task: PCAR-017 through PCAR-021

## PCAR-G032 Plan autonomy, monitor drift, and audit sibling contracts

- Status: active
- Parent: PCAR-G030
- Depends on: PCAR-G031
- Priority: P0
- Track: autonomy-drift-audit
- Goal: Rank bounded recurring value under hard gates, execute only permitted low-risk classes, detect minimal deduplicated current-root drift, and compare sibling published contracts without writes.
- Completion contract: Hard rejects run before scoring; affected suffix replanning preserves valid prefixes; high-risk classes always require humans; unchanged trees remain idle; cross-repository results use the closed compatibility disposition vocabulary.
- Evidence: pcar/planner-decision@1, pcar/autonomy-ceiling@1, pcar/drift-delta@1, pcar/cross-repository-audit@1
- Acceptance criteria: smallest-beneficial-change; hard-gate-precedence; bounded-autonomy; self-promotion-rejected; minimal-drift-delta; no-repeated-finding; no-sibling-write
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/planner.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/autonomous_executor.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/drift_monitor.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/cross_repository_audit.py
- Validation: planner, autonomy, drift/deduplication/idle, scope escape, and cross-repository write rejection tests
- Acceptance: Optional sibling unavailability emits `unavailable`; it never causes local contract invention.
- Gap task: PCAR-022 through PCAR-025

## PCAR-G040 Tranche 4: qualify the current tree

- Status: active
- Parent: PCAR-G000
- Depends on: PCAR-G030
- Priority: P0
- Track: qualification
- Goal: Generate compact architecture projections, publish typed control projections, run frozen context/architecture and adversarial benchmarks, enforce promotion and rollback, and qualify the exact final merged tree.
- Completion contract: PCAR-026 through PCAR-031 are accepted; all safety gates pass non-compensably; efficiency claims use identical frozen criteria/evidence/provider/tokenizer; unrun checks and residual gaps remain explicit; promotion and rollback decisions are signed.
- Evidence: pcar/generated-projection-manifest@1, pcar/control-parity-report@1, pcar/benchmark-report@1, pcar/adversarial-report@1, pcar/promotion-decision@1, pcar/qualification-report@1
- Acceptance criteria: current-tree-projections; typed-control-parity; frozen-benchmark; adversarial-zero-escape; noncompensable-promotion; exact-final-tree
- Outputs: docs/architecture/architecture_refactorer_inventory/architecture_root_manifest.json, docs/architecture/architecture_refactorer_inventory/final_qualification_report.json
- Validation: focused tranche-4 tests, selected current-tree proofs, and full required promotion checks
- Acceptance: The final report makes no unsupported readiness, preservation, consolidation, simplification, or efficiency claim.
- Gap task: PCAR-026 through PCAR-031

## PCAR-G041 Generate projections and typed public controls

- Status: active
- Parent: PCAR-G040
- Depends on: PCAR-G032
- Priority: P0
- Track: projection-control
- Goal: Render compact architecture maps from ArchitectureIR and extend the canonical typed control service, CLI, and MCP category with equivalent architecture operations.
- Completion contract: Machine-readable ArchitectureIR/root remain authoritative; generated documents are reproducible projections; all reads and authorized/idempotent/dry-run/exact-tree/scoped/leased/fenced/audited mutations have Python/CLI/MCP transport parity; MCP never shells out.
- Evidence: pcar/generated-projection-manifest@1, control-operation-catalog successor, pcar/control-parity-report@1
- Acceptance criteria: compact-generated-maps; projection-non-authority; typed-service; cli-mcp-parity; no-direct-dispatch
- Outputs: ipfs_accelerate_py/agent_supervisor/architecture_refactorer/documentation.py, ipfs_accelerate_py/agent_supervisor/architecture_refactorer/cli.py, ipfs_accelerate_py/agent_supervisor/control/architecture_operations.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/architecture_tools.py
- Validation: generated documentation, operation catalog, service, CLI and MCP conformance tests
- Acceptance: A new operation is not live until the closed catalog and all typed projections agree.
- Gap task: PCAR-026, PCAR-027

## PCAR-G042 Benchmark, attack, gate, and report

- Status: active
- Parent: PCAR-G040
- Depends on: PCAR-G041
- Priority: P0
- Track: release
- Goal: Freeze and run representative architecture tasks, seed adversarial failures, enforce behavioral/authority/effect/evidence/context/validation/autonomy gates, exercise rollback, and emit exact-tree machine/human reports.
- Completion contract: Frozen inputs and fault schedules are content-addressed; coverage never falls; all seeded escapes remain zero; every threshold retains raw samples and uncertainty; final report contains every required comparison, accepted/rejected candidate, counterexample, intervention, unrun check, blocker, eligibility decision, and rollback target.
- Evidence: pcar/frozen-corpus@1, pcar/benchmark-report@1, pcar/adversarial-report@1, pcar/promotion-decision@1, pcar/qualification-report@1
- Acceptance criteria: frozen-task-corpus; same-evidence-policy; no-seeded-escape; rollback-exercised; threshold-conjunction; complete-report
- Outputs: benchmarks/agent_supervisor/architecture_refactorer/manifest.json, docs/architecture/architecture_refactorer_inventory/final_qualification_report.json
- Validation: benchmark determinism, assurance, promotion, rollback, receipt/root forgery, stale evidence, and release report tests
- Acceptance: A failed safety gate is final regardless of efficiency; a missed efficiency target is reported honestly without weakening evidence.
- Gap task: PCAR-028 through PCAR-031
