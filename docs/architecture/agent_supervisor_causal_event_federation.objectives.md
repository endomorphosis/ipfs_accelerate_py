# Causal Event-Driven Multi-Supervisor Federation objective heap

Machine-ingestible intent for program and board namespace
`agent-supervisor-causal-event-federation-v1`. Goals and subgoals become
ordinary DuckDB goal records with explicit parent and dependency edges.
Markdown is a sealed bootstrap/human projection, never completion authority.

## Goal tree

```text
CASF-G000  Deliver the CausalAbstractionSupervisorFederation
|-- CASF-G010  Tranche 1: establish the authoritative federation core
|   |-- CASF-G011  Seal and inventory current authorities
|   |-- CASF-G012  Define contracts, schema, registries, and trigger admission
|   `-- CASF-G013  Establish atomic events, waiting, routing, and storm control
|-- CASF-G020  Tranche 2: establish causal and semantic coordination
|   |-- CASF-G021  Build and validate the causal abstraction network
|   |-- CASF-G022  Share semantic, proof, and retrieval projections
|   `-- CASF-G023  Wake and advance only affected supervisors
|-- CASF-G030  Tranche 3: establish safe parallel federation
|   |-- CASF-G031  Deduplicate and budget a conflict-free frontier
|   |-- CASF-G032  Shard, steal, and rebalance with fencing
|   `-- CASF-G033  Merge, recover, and detect federation fixed point
`-- CASF-G040  Tranche 4: project history and qualify the product
    |-- CASF-G041  Project DuckLake history and monitor drift
    |-- CASF-G042  Publish controls and prove adversarial safety
    `-- CASF-G043  Benchmark, gate, and report the final tree
```

## CASF-G000 Deliver the CausalAbstractionSupervisorFederation

- Status: active
- Parent:
- Depends on:
- Priority: P0
- Track: federation-root
- Goal: Deliver one authenticated, transactional, event-driven and causally coordinated federation that shares exact DuckDB/Quack state, wakes only affected bounded supervisors, reuses semantic/proof/context evidence, safely coordinates approximately twelve supervisors and hundreds of logical subagents, and projects non-authoritative history to DuckLake.
- Completion contract: All four tranche goals are accepted against one exact merged tree and control-plane generation; every non-compensable safety, causal, event, parallel, token, and applicable DuckLake gate passes; no required task/effect/validation/proof/merge/event/recovery/human obligation remains; rollback is verified.
- Evidence: casf/federation-completion-receipt@1, casf/fixed-point@1, casf/qualification-report@1, casf/release-manifest@1
- Acceptance criteria: exact-current-tree; one-state-owner; authenticated-trigger; no-lost-events; causal-notification; conflict-free-parallelism; fixed-point; truthful-qualification
- Outputs: docs/architecture/causal_event_federation_inventory/final_qualification_report.json, docs/architecture/causal_event_federation_inventory/federation_root_manifest.json
- Validation: python3 scripts/validate_agent_supervisor_causal_event_federation_board.py && python3 -m pytest -q test/api/causal_federation
- Acceptance: No board state, process exit, quiet queue, model claim, historical receipt, metric, retrieval result, or DuckLake projection is completion evidence; only current-tree receipts from declared producers may satisfy the root.
- Gap task: CASF-000 through CASF-043

## CASF-G010 Tranche 1: establish the authoritative federation core

- Status: active
- Parent: CASF-G000
- Depends on:
- Priority: P0
- Track: authoritative-core
- Goal: Seal the current authority forest and implement strict federation/event contracts, normalized schema extensions, bounded registries, authenticated trigger admission, atomic outbox writes, owner-side event waiting, subscriptions, retries, backpressure, and dead letters.
- Completion contract: CASF-000 through CASF-012 are accepted; all authority classifications are source-bound; Quack is the exclusive live owner; arbitrary SQL/direct file writes/fallback are rejected; atomicity, replay, no-lost-wakeup, and idle behavior have exact evidence.
- Evidence: casf/authority-baseline@1, casf/schema-migration@1, casf/trigger-receipt@1, casf/outbox-atomicity@1, casf/event-wait-qualification@1
- Acceptance criteria: classified-authorities; strict-contracts; one-owner; authenticated-idempotent-create; atomic-event-outbox; no-lost-wakeup; bounded-storm-control
- Outputs: docs/architecture/causal_event_federation_inventory/authority_inventory.json, docs/architecture/causal_event_federation_inventory/tranche_1_qualification.json
- Validation: focused CASF tranche-1 contract, schema, trigger, owner, outbox, wait, subscription, and backpressure tests
- Acceptance: Event-driven and multi-supervisor claims remain prohibited until the exact wait/idle and exclusive-owner gates pass.
- Gap task: CASF-000 through CASF-012

## CASF-G011 Seal and inventory current authorities

- Status: active
- Parent: CASF-G010
- Depends on:
- Priority: P0
- Track: baseline-inventory
- Goal: Bind the exact commit/tree, migrations, operation catalog, DuckDB/Quack/DuckLake capabilities, state owner, runners, events, causal/semantic/proof/retrieval/control surfaces, documents, tests, entrypoints, and sibling published contracts to one classified evidence inventory.
- Completion contract: CASF-000 and CASF-001 inspect every mandated root; each capability is available, available_with_caveats, stale, incompatible, or missing with current source/test identity; unknowns remain typed and siblings remain read-only.
- Evidence: casf/baseline-seal@1, casf/authority-inventory@1, casf/prerequisite-matrix@1
- Acceptance criteria: source-tree-seal; mandated-root-coverage; explicit-authority-status; exact-capability-probe; no-sibling-write; typed-gaps
- Outputs: docs/architecture/causal_event_federation_inventory/sealed_current_tree_baseline.json, docs/architecture/causal_event_federation_inventory/authority_inventory.json
- Validation: board validator plus baseline/inventory schema and current-source identity tests
- Acceptance: A name, import, fixture, projection, embedded test, or historical receipt never implies live authority.
- Gap task: CASF-000, CASF-001

## CASF-G012 Define contracts, schema, registries, and trigger admission

- Status: active
- Parent: CASF-G010
- Depends on: CASF-G011
- Priority: P0
- Track: contracts-admission
- Goal: Define closed federation/supervisor/subagent/shard/budget/causal contracts, extend the canonical schema, register bounded populations, and admit authenticated external-agent federation requests transactionally.
- Completion contract: CASF-002 and CASF-004 through CASF-008 are accepted; unknown fields, raw credentials, arbitrary paths/SQL, self-promotion, invalid lifecycle/capacity, and unauthorized triggers fail closed; normalized join identities and idempotent create receipts verify.
- Evidence: casf/contracts-root@1, casf/schema-receipt@1, casf/registry-root@1, casf/federation-admission-receipt@1
- Acceptance criteria: closed-round-trip; normalized-schema; bounded-populations; strict-lifecycle; authenticated-delegation; idempotent-create
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/contracts.py, ipfs_accelerate_py/agent_supervisor/federation/trigger.py
- Validation: focused contract, schema, registry, lifecycle, capacity, authentication, delegation, expiry, and idempotency tests
- Acceptance: No model or triggering agent may manufacture authority, policy outcome, provider qualification, lease/fence, or completion.
- Gap task: CASF-002, CASF-004 through CASF-008

## CASF-G013 Establish atomic events, waiting, routing, and storm control

- Status: active
- Parent: CASF-G010
- Depends on: CASF-G012
- Priority: P0
- Track: event-core
- Goal: Define and implement atomic domain events/outbox rows, bounded subscriptions/cursors, owner-side lost-wakeup-free waiting, coalescing, retry, backpressure, dead-letter, quarantine, and recovery semantics.
- Completion contract: CASF-003 and CASF-009 through CASF-012 are accepted; state/event/outbox/generation commit atomically; replay is idempotent; required intermediate evidence is not coalesced; idle wait performs no repeated board scans or model calls.
- Evidence: casf/event-contract-root@1, casf/outbox-atomicity@1, casf/event-wait-receipt@1, casf/storm-control-report@1
- Acceptance criteria: closed-events; transaction-atomicity; durable-cursors; no-lost-wakeup; bounded-fanout; safe-coalescing; dead-letter-evidence
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/events.py, ipfs_accelerate_py/agent_supervisor/federation/event_wait.py
- Validation: outbox rollback, concurrent waiter, replay, duplicate/out-of-order, cursor recovery, coalescing, retry/dead-letter, storm, and idle tests
- Acceptance: At-least-once delivery is stated exactly; exactly-once applies only to fenced idempotent authoritative effects.
- Gap task: CASF-003, CASF-009 through CASF-012

## CASF-G020 Tranche 2: establish causal and semantic coordination

- Status: active
- Parent: CASF-G000
- Depends on: CASF-G010
- Priority: P0
- Track: causal-coordination
- Goal: Build the multilevel causal graph, evidence rules, abstraction/intervention validation, minimal frontier, world snapshots, semantic/proof/retrieval projections, and transactionally advanced event-driven supervisor wake path.
- Completion contract: CASF-013 through CASF-021 are accepted; exact descendants always wake; nomination-only evidence never grants authority; stale/unknown maps cannot suppress work; shared semantic/proof/index roots remain tree-bound; unchanged supervisors remain asleep.
- Evidence: casf/causal-graph-root@1, casf/abstraction-validation@1, casf/frontier-receipt@1, casf/world-snapshot@1, casf/wake-qualification@1
- Acceptance criteria: multilevel-network; evidence-authority-separation; intervention-consistency; complete-exact-frontier; shared-tree-root; affected-only-wake
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/causal_graph.py, ipfs_accelerate_py/agent_supervisor/federation/causal_frontier.py
- Validation: focused causal construction/evidence/map/intervention/frontier/snapshot/projection/wakeup and unnecessary-wakeup tests
- Acceptance: Retrieval proposes; exact analysis disposes. Unknown dependency widens rather than suppresses the frontier.
- Gap task: CASF-013 through CASF-021

## CASF-G021 Build and validate the causal abstraction network

- Status: active
- Parent: CASF-G020
- Depends on: CASF-G013
- Priority: P0
- Track: causal-network
- Goal: Persist L0-L4 causal nodes, closed edges and evidence; separate exact/admitted-conservative authority from nomination; validate abstraction maps by representative interventions; compile must/may/do-not-wake frontiers and world snapshots.
- Completion contract: CASF-013 through CASF-017 are accepted; cycles require explicit fixed-point groups; every authoritative edge has admitted evidence; observational similarity cannot prove cause/independence; intervention mismatches/exclusions are durable.
- Evidence: casf/causal-graph-root@1, casf/causal-evidence-receipt@1, casf/abstraction-map@1, casf/intervention-test@1, casf/frontier@1
- Acceptance criteria: closed-levels-edges; exact-evidence; nomination-isolation; intervention-faithfulness; stale-map-rejection; frontier-completeness
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/causal_graph.py, ipfs_accelerate_py/agent_supervisor/federation/world_snapshot.py
- Validation: causal graph, cycle, exact/nomination, map status, intervention mismatch, frontier widening, required/unnecessary wake tests
- Acceptance: `do_not_wake` requires proved or policy-admitted independence; unknown is never treated as independent.
- Gap task: CASF-013 through CASF-017

## CASF-G022 Share semantic, proof, and retrieval projections

- Status: active
- Parent: CASF-G020
- Depends on: CASF-G021
- Priority: P0
- Track: shared-state
- Goal: Project immutable `ipfs_datasets_py` semantic contracts, proof/test/cache/seal state, and BM25/vector/KG nomination indexes into the one tree-bound federation state without reinterpreting meaning or duplicating authority.
- Completion contract: CASF-018 through CASF-020 are accepted; incremental tree updates invalidate exactly affected capsules/proofs/index records; all retrieval results bind revision/tree/source/method/partition; no projection establishes authority or completion.
- Evidence: casf/semantic-projection-root@1, casf/proof-projection-root@1, casf/retrieval-index-root@1, casf/invalidation-receipt@1
- Acceptance criteria: canonical-semantic-owner; incremental-AST-symbol; affected-capsules; proof-cache-invalidation; bounded-retrieval-provenance; nomination-only
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/semantic_projection.py, ipfs_accelerate_py/agent_supervisor/federation/retrieval_projection.py
- Validation: AST/symbol update, capsule/proof cache invalidation, BM25/vector/KG nomination, tree mismatch, privacy, and source identity tests
- Acceptance: Accelerator records and queries sibling semantic contracts but never changes their meaning or writes the sibling repository.
- Gap task: CASF-018 through CASF-020

## CASF-G023 Wake and advance only affected supervisors

- Status: active
- Parent: CASF-G020
- Depends on: CASF-G021, CASF-G022
- Priority: P0
- Track: affected-wakeup
- Goal: Integrate event batches, durable cursors, minimal causal/context slices, affected capsule/task/proof recomputation, receipt reuse, work reservation, result events, and transactional cursor advancement into supervisor execution.
- Completion contract: CASF-021 is accepted; relevant events/ready tasks/leases/dependencies/proofs/merge/human/capability/health timers wake exactly eligible supervisors; irrelevant unchanged state produces no scan/model/context/write activity.
- Evidence: casf/supervisor-wake-receipt@1, casf/cursor-advance@1, casf/idle-stability@1
- Acceptance criteria: affected-only-wake; minimal-slice; unchanged-receipt-reuse; transactional-cursor; no-full-board-scan; bounded-idle
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/federated_supervisor_runner.py
- Validation: required/unnecessary wake, cursor crash/replay, slice bounds, no-event idle, lease deadline, and capability change tests
- Acceptance: The program may claim event-driven operation only after no-lost-wakeup and zero-idle-scan/model/write gates pass.
- Gap task: CASF-021

## CASF-G030 Tranche 3: establish safe parallel federation

- Status: active
- Parent: CASF-G000
- Depends on: CASF-G020
- Priority: P0
- Track: parallel-federation
- Goal: Deduplicate task intent, compile conflict-free parallel frontiers, enforce hierarchical budgets, specialize and shard supervisors, transfer eligible work, rebalance with fencing, integrate merge, recover failures, and prove fixed point.
- Completion contract: CASF-022 through CASF-030 are accepted; one task/effect owner exists per epoch; child budgets conserve parent reservations; transfers lose/duplicate no work; irreversible effects do not move; recovery preserves identity/evidence; fixed point rejects residual obligations.
- Evidence: casf/task-intent-root@1, casf/parallel-frontier@1, casf/budget-ledger@1, casf/rebalance-receipt@1, casf/recovery-receipt@1, casf/fixed-point@1
- Acceptance criteria: duplicate-suppression; conservative-independence; budget-conservation; exact-shards; fenced-transfer; coordinated-merge; autonomous-recovery; true-fixed-point
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/scheduler.py, ipfs_accelerate_py/agent_supervisor/federation/fixed_point.py
- Validation: focused dedup/frontier/budget/shard/steal/rebalance/merge/recovery/fixed-point tests
- Acceptance: High concurrency remains disabled until Quack, outbox, claims/fences, budgets, causal frontier, and recovery all qualify.
- Gap task: CASF-022 through CASF-030

## CASF-G031 Deduplicate and budget a conflict-free frontier

- Status: active
- Parent: CASF-G030
- Depends on: CASF-G023
- Priority: P0
- Track: parallel-frontier
- Goal: Compute task-intent identities and exact/subsumed/overlap/conflict/independence dispositions, then allocate conflict-free tasks under nested resource/token/proof/merge/provider reservations.
- Completion contract: CASF-022 through CASF-024 are accepted; duplicates share one result; unknown conflict reduces concurrency; each task has exact supervisor/agent/worktree/lease/fence/merge/validation ownership; no child overspends or consumes validation reserve speculatively.
- Evidence: casf/dedup-receipt@1, casf/parallel-frontier@1, casf/budget-ledger@1
- Acceptance criteria: canonical-task-intent; duplicate-subsumption; overlap-boundary; conflict-serialization; proven-independence; hierarchical-budget-CAS
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/deduplication.py, ipfs_accelerate_py/agent_supervisor/federation/budgets.py
- Validation: duplicate/subsumption/overlap/conflict, frontier, concurrent claim, budget reserve/return/transfer, and merge-pressure tests
- Acceptance: A model statement of independence is nomination only and cannot admit parallel effects.
- Gap task: CASF-022 through CASF-024

## CASF-G032 Shard, steal, and rebalance with fencing

- Status: active
- Parent: CASF-G030
- Depends on: CASF-G031
- Priority: P0
- Track: sharding
- Goal: Create conflict-free specialized supervisor shards, transfer eligible virgin work to idle capable supervisors, and rebalance on load/graph/resource/provider/hotspot/merge/failure changes without identity, budget, cursor, or effect corruption.
- Completion contract: CASF-025 through CASF-027 are accepted; assignment revisions freeze/drain/transfer/activate atomically; fencing increments; no task is double-owned or disappears; active irreversible effects stay put; policy/privacy/proof/merge ceilings remain enforced.
- Evidence: casf/shard-revision@1, casf/work-steal-receipt@1, casf/rebalance-receipt@1
- Acceptance criteria: bounded-specialization; exact-shard-boundaries; virgin-transfer-only; atomic-budget-transfer; no-double-ownership; irreversible-effect-pin
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/sharding.py, ipfs_accelerate_py/agent_supervisor/federation/rebalancing.py
- Validation: shard partition, capability/effect ceilings, work steal, concurrent transfer, fence rollover, crash/rebalance, and budget conservation tests
- Acceptance: Work stealing and rebalancing never bypass repository ownership, policy, proof, merge, privacy, or human review.
- Gap task: CASF-025 through CASF-027

## CASF-G033 Merge, recover, and detect federation fixed point

- Status: active
- Parent: CASF-G030
- Depends on: CASF-G032
- Priority: P0
- Track: merge-recovery-fixed-point
- Goal: Integrate existing worktree/merge authorities, recover every required owner/worker/event/provider/projection/effect failure, and compute fixed point over work, effects, validation/proof, merge, human, recovery, events, and semantic freshness.
- Completion contract: CASF-028 through CASF-030 are accepted; merge order is explicit; unknown effects reconcile before retry; failed attempts remain; stale cursors/leases/fences reject; projections rebuild; fixed point binds an exact world snapshot/watermark and rejects every false-quiet case.
- Evidence: casf/merge-receipt@1, casf/recovery-receipt@1, casf/effect-reconciliation@1, casf/fixed-point@1
- Acceptance criteria: isolated-merge; stale-fence-reject; durable-cursor-recovery; unknown-effect-reconcile; projection-rebuild; false-fixed-point-reject
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/recovery.py, ipfs_accelerate_py/agent_supervisor/federation/fixed_point.py
- Validation: merge conflict, owner/supervisor/subagent/consumer crash, disconnect/partition, lease/fence, event storm, provider/proof timeout, unknown effect, true/false fixed-point tests
- Acceptance: Neither process exit nor a quiet queue/completed board establishes completion.
- Gap task: CASF-028 through CASF-030

## CASF-G040 Tranche 4: project history and qualify the product

- Status: active
- Parent: CASF-G000
- Depends on: CASF-G030
- Priority: P0
- Track: product-qualification
- Goal: Implement recoverable DuckLake history, monitor architecture/event drift, publish typed control/CLI/MCP surfaces, formally model and attack safety, benchmark idle/parallel/load/token behavior, enforce promotion/rollback/quarantine, and qualify the exact final tree.
- Completion contract: CASF-031 through CASF-043 are accepted; product and transport parity holds; formal/chaos and frozen benchmarks meet every non-compensable gate; DuckLake failure is isolated; the final report distinguishes verified, failed, skipped, and not-run claims with rollback.
- Evidence: casf/ducklake-projection-receipt@1, casf/control-parity@1, casf/formal-model-report@1, casf/chaos-report@1, casf/benchmark-suite@1, casf/promotion-decision@1, casf/qualification-report@1
- Acceptance criteria: replayable-history; drift-detection; typed-public-controls; model-checked-safety; adversarial-zero-escape; frozen-benchmarks; conjunctive-promotion; truthful-report
- Outputs: docs/architecture/causal_event_federation_inventory/final_qualification_report.json, benchmarks/agent_supervisor/causal_event_federation/manifest.json
- Validation: focused tranche-4 tests plus live-marked capability, multiprocess, scale, idle, parallel, token, chaos, and model-check suites
- Acceptance: Failure of DuckLake-specific gates cannot block core control-plane qualification; it blocks only DuckLake promotion.
- Gap task: CASF-031 through CASF-043

## CASF-G041 Project DuckLake history and monitor drift

- Status: active
- Parent: CASF-G040
- Depends on: CASF-G033
- Priority: P0
- Track: history-drift
- Goal: Replace the placeholder with idempotent event-range-bound DuckLake projection and recovery/security receipts, then detect schema/operation/event/causal drift against exact roots without promoting analytics to authority.
- Completion contract: CASF-031 through CASF-033 are accepted; bounded immutable partitions/checksums/cursors/schema/redaction/tenant rules survive interruption; outage is typed and nonblocking; drift emits precise current-root findings without changing production state.
- Evidence: casf/ducklake-projection-receipt@1, casf/projection-recovery@1, casf/drift-report@1
- Acceptance criteria: event-range-binding; idempotent-projection; bounded-files; recoverable-cursor; privacy-redaction; non-authority; exact-drift
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/ducklake_projection.py, ipfs_accelerate_py/agent_supervisor/federation/drift_monitor.py
- Validation: DuckLake capability/outage/partial/replay/schema/redaction/tenant and architecture/event drift tests
- Acceptance: DuckLake is eventually consistent, rebuildable, and never a control-plane prerequisite.
- Gap task: CASF-031 through CASF-033

## CASF-G042 Publish controls and prove adversarial safety

- Status: active
- Parent: CASF-G040
- Depends on: CASF-G041
- Priority: P0
- Track: product-formal-chaos
- Goal: Extend the existing typed control service and direct CLI/MCP adapters, create formal models with existing tooling, and run adversarial/chaos suites across authorization, tenancy, events, leases/fences, lifecycle, rebalance, causal propagation, recovery, and secrets.
- Completion contract: CASF-034 through CASF-037 are accepted; all read/mutation operations have contract parity; mutations require authorization/roots/idempotency/generation/lease/fence/dry-run/effects/audit; model properties hold; attacks create no unauthorized effect or false completion.
- Evidence: casf/control-catalog@1, casf/control-parity@1, casf/model-check-report@1, casf/adversarial-report@1
- Acceptance criteria: typed-service; direct-adapters; parity; closed-mutations; model-checked-invariants; chaos-recovery; tenant-secret-isolation
- Outputs: ipfs_accelerate_py/agent_supervisor/federation/control_service.py, test/api/causal_federation/test_chaos.py
- Validation: Python/CLI/MCP parity, authorization/idempotency/fence, lifecycle/rebalance/event/causal model, crash/storm/tenant/secret chaos tests
- Acceptance: CLI/MCP never shells to command strings, and analytics/models/agents cannot promote authority or completion.
- Gap task: CASF-034 through CASF-037

## CASF-G043 Benchmark, gate, and report the final tree

- Status: active
- Parent: CASF-G040
- Depends on: CASF-G042
- Priority: P0
- Track: benchmark-release
- Goal: Freeze and run idle, 12-supervisor, 256-agent, event replay, parallel-throughput, and cross-supervisor token/context benchmarks, then enforce conjunctive promotion/rollback/quarantine and produce the exact final qualification report.
- Completion contract: CASF-038 through CASF-043 are accepted; idle work is bounded with zero forbidden activity; scale and throughput targets pass without assurance loss/duplicates; token/context targets pass; every safety and causal gate is zero-tolerance; residual gaps and rollback are explicit.
- Evidence: casf/idle-benchmark@1, casf/parallel-benchmark@1, casf/load-benchmark@1, casf/token-benchmark@1, casf/promotion-decision@1, casf/qualification-report@1
- Acceptance criteria: frozen-comparison; zero-idle-work; real-process-scale; throughput-three-x; token-context-targets; noncompensable-gates; exact-final-tree-report
- Outputs: benchmarks/agent_supervisor/causal_event_federation/manifest.json, docs/architecture/causal_event_federation_inventory/final_qualification_report.json
- Validation: benchmark manifest/corpus tests, live-marked benchmark runs, promotion negative/positive/rollback/quarantine tests, final report schema/content verification
- Acceptance: Unsupported event-driven, causal, multi-supervisor, parallel, token-efficient, or production-ready claims are release blockers.
- Gap task: CASF-038 through CASF-043
