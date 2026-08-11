# Semantic compression harness objective heap

Machine-ingestible goal hierarchy for `ipfs_accelerate_py.agent_supervisor`.
The executable projection is
`docs/architecture/semantic_compression_harness.todo.md` with task prefix
`## SCH-`. The reviewed design is
`docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md`.

Implementation is launch-gated: `SCH-000` must pin and validate the unresolved
final repaired `ipfs_datasets_py` semantic-state/Merkle/capsule and
`ipfs_kit_py` generation-bearing durable-root commits before any other task is
eligible.

## Goal tree

```text
SCH-G000  Complete local Python semantic-compression loop
|-- SCH-G010  Pin MCP++ and repository adapters
|-- SCH-G020  Scheduling, routing, context, and execution
|-- SCH-G030  Isolated patch acceptance, receipts, and root commit
|-- SCH-G040  CLI, incremental sessions, and end-to-end acceptance
`-- SCH-G050  Exactly-40-task benchmark and release evidence
```

## SCH-G000 Complete local Python semantic-compression loop

- Status: active
- Parent:
- Depends on:
- Fib priority: 1
- Priority: P0
- Track: semantic-state-harness
- Bundle: sch/root
- Goal: Deliver a Python 3.12 and pytest local coding-agent loop that consumes deterministic symbol state, compresses unchanged dependencies without hiding uncertainty, verifies an isolated patch, emits content-addressed receipts, and atomically advances the semantic-state root.
- Evidence: sch/contract-pins@1, sch/context-pack@1, sch/patch-loop@1, sch/acceptance@1, sch/benchmark@1
- Acceptance criteria: sch/contract-pins@1; sch/context-pack@1; sch/patch-loop@1; sch/acceptance@1; sch/benchmark@1
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state, test/api/semantic_state, benchmarks/semantic_state, docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md
- Validation: python3.12 -m pytest -q test/api/semantic_state
- Acceptance: The controlled Python repository runs end to end with a datasets-owned symbol Merkle DAG, deterministic root manifests, bounded invalidation, existing-compiler assurance-aware context, selected/full test evidence, strict production provider gates, durable receipts, and generation-bearing expected-old root CAS; no UI, server, agent framework, or broad portfolio refactor is introduced.
- Gap task: SCH-000 through SCH-018
- Refinement: Compose the final datasets semantic-state contracts and existing accelerate/kit authorities through narrow ports.

## SCH-G010 Pin MCP++ and repository adapters

- Status: active
- Parent: SCH-G000
- Depends on:
- Fib priority: 2
- Priority: P0
- Track: contracts
- Bundle: sch/contracts
- Goal: Seal exact dependency commits and expose closed MCP++, semantic-state, and durable-state boundaries without creating competing identity, graph, or storage authorities.
- Evidence: sch/dependency-seal@1, sch/mcplusplus-wire@1, sch/datasets-adapter@1, sch/kit-adapter@1
- Acceptance criteria: sch/dependency-seal@1; sch/mcplusplus-wire@1; sch/datasets-adapter@1; sch/kit-adapter@1
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/contracts.py, ipfs_accelerate_py/agent_supervisor/semantic_state/wire.py, ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py, ipfs_accelerate_py/agent_supervisor/semantic_state/durable_state.py
- Validation: python3.12 -m pytest -q test/api/semantic_state/test_wire.py test/api/semantic_state/test_datasets_adapter.py test/api/semantic_state/test_durable_state.py
- Acceptance: Accelerate and MCP++ commits plus final repaired datasets/kit commits are exact and validated by fingerprints and producer tests; real CIDv1 wire artifacts conform; the datasets Merkle/capsule/source APIs remain authoritative; missing or mismatched capabilities fail closed; local durability is hermetic, ABA-safe, and single-writer CAS safe.
- Gap task: SCH-000, SCH-001, SCH-002, SCH-003
- Refinement: Preserve semantic-index CIDs, use MCP++ canonical bytes plus real Kubo-compatible CIDv1 for harness artifacts, and lazily import the pinned kit seam.

## SCH-G020 Scheduling, routing, context, and execution

- Status: active
- Parent: SCH-G000
- Depends on: SCH-G010
- Fib priority: 3
- Priority: P0
- Track: execution
- Bundle: sch/execution
- Goal: Admit datasets-owned confidence-preserving capsules, select sufficient tests/proofs, pack minimum-sufficient context through the existing ContextCompiler/ProductionContextSlice, route work by assurance, and execute through existing resource/provider/lease mechanisms.
- Evidence: sch/scheduling-contracts@1, sch/scheduling-adapter@1, sch/context-pack@1, sch/model-routing@1, sch/verification-runner@1
- Acceptance criteria: sch/scheduling-contracts@1; sch/scheduling-adapter@1; sch/context-pack@1; sch/model-routing@1; sch/verification-runner@1
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/scheduling.py, ipfs_accelerate_py/agent_supervisor/semantic_state/capsules.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/test_selection.py, ipfs_accelerate_py/agent_supervisor/semantic_state/verification.py
- Validation: python3.12 -m pytest -q test/api/semantic_state/test_scheduling.py test/api/semantic_state/test_context_pack.py test/api/semantic_state/test_routing.py test/api/semantic_state/test_verification.py
- Acceptance: Exact/conservative capsules substitute only when valid, heuristic/opaque/stale facts retrieve raw source, selected tests have auditable paths and full-suite fallback, unavailable providers/provers are typed, and simulation never becomes production evidence.
- Gap task: SCH-004, SCH-005, SCH-006, SCH-007, SCH-008
- Refinement: Reuse ContextCompiler, ProductionContextSlice, ResourceScheduler, ProviderExecutionGateway, WorktreeLifecycleStore, LeaseCoordinator, ValidationScheduler.run_staged, ProofScheduler, and event log; do not make PersistentTaskQueue authoritative.

## SCH-G030 Isolated patch acceptance, receipts, and root commit

- Status: active
- Parent: SCH-G000
- Depends on: SCH-G010, SCH-G020
- Fib priority: 5
- Priority: P0
- Track: patch-acceptance
- Bundle: sch/patch-acceptance
- Goal: Apply only an admitted patch in a fenced disposable worktree, rescan and verify it, issue freshness-bound MCP++ receipts, and publish the next state root only after every gate succeeds.
- Evidence: sch/worktree@1, sch/receipt@1, sch/freshness@1, sch/harness-loop@1
- Acceptance criteria: sch/worktree@1; sch/receipt@1; sch/freshness@1; sch/harness-loop@1
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/worktree.py, ipfs_accelerate_py/agent_supervisor/semantic_state/receipts.py, ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py
- Validation: python3.12 -m pytest -q test/api/semantic_state/test_worktree.py test/api/semantic_state/test_receipts.py test/api/semantic_state/test_harness.py
- Acceptance: The 14-step workflow reuses strict proposal/preimage validation, rejects malformed/out-of-scope patches and stale/simulated evidence, stores a complete SemanticStateRootManifest, binds static/test/prover/oracle results to exact inputs, leaves rejected candidates unreachable from the current root, and wins generation-bearing CAS exactly once.
- Gap task: SCH-009, SCH-010, SCH-011
- Refinement: Emit obligations and receipts, not arbitrary dependent-code rewrites or proof claims from unavailable tools.

## SCH-G040 CLI, incremental sessions, and end-to-end acceptance

- Status: active
- Parent: SCH-G000
- Depends on: SCH-G030
- Fib priority: 8
- Priority: P0
- Track: acceptance
- Bundle: sch/acceptance
- Goal: Make the local loop operable through a narrow CLI and restartable watcher/session coordinator, then prove all required controlled mutations and safety failures end to end.
- Evidence: sch/session@1, sch/cli@1, sch/fixture@1, sch/e2e@1
- Acceptance criteria: sch/session@1; sch/cli@1; sch/fixture@1; sch/e2e@1
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/session.py, ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py, test/fixtures/semantic_state_harness/controlled_repo, test/api/semantic_state/test_acceptance.py, pyproject.toml, setup.py
- Validation: python3.12 -m pytest -q test/api/semantic_state/test_session.py test/api/semantic_state/test_cli.py test/api/semantic_state/test_acceptance.py
- Acceptance: All requested CLI capabilities return deterministic JSON; notifications trigger canonical rescans; restart/replay and concurrent watchers are fenced; the full fixture matrix proves bounded invalidation, raw fallback, receipt freshness, CAS safety, no selection false negatives, and strict simulation rejection.
- Gap task: SCH-012, SCH-013, SCH-014, SCH-015
- Refinement: A dedicated local console command is sufficient; no network service, dashboard, MCP server, or target-code import.

## SCH-G050 Exactly-40-task benchmark and release evidence

- Status: active
- Parent: SCH-G000
- Depends on: SCH-G040
- Fib priority: 13
- Priority: P0
- Track: release
- Bundle: sch/release
- Goal: Measure an honest exactly-40-task Python maintenance corpus and publish reproducible release evidence, limitations, bottlenecks, and the remaining production/ZK work.
- Evidence: sch/benchmark-corpus@1, sch/benchmark-results@1, sch/release-docs@1, sch/import-safety@1
- Acceptance criteria: sch/benchmark-corpus@1; sch/benchmark-results@1; sch/release-docs@1; sch/import-safety@1
- Outputs: benchmarks/semantic_state/tasks, benchmarks/semantic_state/run_benchmark.py, docs/benchmarks/semantic_compression_harness_results.json, docs/benchmarks/semantic_compression_harness_results.md, docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md
- Validation: python3.12 -m pytest -q test/api/semantic_state && python3.12 benchmarks/semantic_state/run_benchmark.py --check
- Acceptance: Exactly 40 tasks report all required context/invalidation/test/proof/route/outcome fields; replay/oracle candidates are production-ineligible and separated from production acceptance; deterministic fields reproduce while timing remains observational; median reduction is at least 30 percent without omitting required raw code; controlled recall is 100 percent; stale/simulated evidence admission is zero; import and provider regressions pass; the final report is traceable to committed receipts.
- Gap task: SCH-016, SCH-017, SCH-018
- Refinement: Keep failed and escalated tasks in metrics and state Python unsoundness honestly; do not optimize the benchmark by weakening coverage.
