# Autonomous Meta-Controller objective heap

Machine-ingestible objective bootstrap for
`agent-supervisor-autonomous-meta-controller-v1`. After materialization the
DuckDB control plane reached through Quack is authoritative; this document is
a sealed deterministic export. DuckLake is a non-authoritative projection.

## Goal tree

```text
APMC-G000  Evidence-driven autonomous control with lower model cost
|-- APMC-G010  A: seal current authorities and paired baseline
|-- APMC-G020  B: define the closed autonomy contract model
|-- APMC-G030  C: resolve named questions under objective budgets
|-- APMC-G040  D: learn routes from compact causally attributed episodes
|-- APMC-G050  E: distill repeated reasoning into bounded rules and skills
|-- APMC-G060  F: preserve plan prefixes and repair bounded suffixes
|-- APMC-G070  G: minimize human questions and compact semantic memory
|-- APMC-G080  H: integrate event-driven receding-horizon runtime
|-- APMC-G090  I: expose governed Python, CLI, and MCP controls
|-- APMC-G100  J: qualify on a frozen paired adversarial benchmark
`-- APMC-G110  K: promote or terminate honestly with a residual-gap report
```

## APMC-G000 Evidence-driven autonomous control with lower model cost

- Status: active
- Review only: true
- Parent:
- Depends on:
- Priority: P0
- Track: apmc-root
- Bundle: agent-supervisor/autonomy/root
- Parallel lane: control
- Resource class: cpu-large
- Goal: Move the supervisor from stage-triggered prompting toward evidence-driven autonomous control where deterministic software resolves routine named questions, models handle residual ambiguity, repeated reasoning becomes bounded rules/skills, and human authority is reserved for consequential choices.
- Subgoals: APMC-G010, APMC-G020, APMC-G030, APMC-G040, APMC-G050, APMC-G060, APMC-G070, APMC-G080, APMC-G090, APMC-G100, APMC-G110
- Evidence: accepted child-goal receipts, autonomy run receipts, paired benchmark receipt, promotion or non-promotion receipt
- Evidence criteria: Every required task has current-tree validation/proof evidence; all non-compensable safety gates are zero; cost/autonomy/quality/runtime gates are measured against exact baseline; database replay and restart are deterministic.
- Evidence source policy: Current Git identities, canonical contracts, DuckDB/Quack transactional records, fresh tests/proofs, content-addressed receipts, and externally authorized policy CAS are authoritative; Markdown, DuckLake projections, caches, model confidence, and simulated fixtures are not.
- Outputs: docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md, docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md, docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md, ipfs_accelerate_py/agent_supervisor/autonomy, benchmarks/agent_supervisor/autonomous_meta_controller
- Interfaces: AutonomousMetaController@1, AutonomyRunReceipt@1, AutonomyPromotionReceipt@1
- Validation: python3 scripts/validate_agent_supervisor_autonomous_meta_controller_board.py --check-all
- Acceptance: APMC-000 through APMC-020 are accepted from current evidence; no false completion, authority escalation, stale evidence, scope escape, validation reduction, simulated-as-live claim, or self-promotion exists.
- Gap task: APMC-020 aggregates child evidence and produces the release or exact non-promotion result.
- Refinement: Lower token cost is never allowed to compensate for safety, correctness, freshness, privacy, proof, evidence, or quality regression.
- Conflict policy: Child goals own disjoint implementation surfaces; APMC-019 owns promotion state and APMC-020 alone owns the joined release report.

## APMC-G010 A: seal current authorities and paired baseline

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on:
- Priority: P0
- Track: baseline
- Bundle: agent-supervisor/autonomy/baseline
- Parallel lane: inventory
- Resource class: cpu-medium
- Goal: Record exact source/pins/runtime/dependencies/tests/capabilities and classify existing supervisor authorities before adding code.
- Evidence: APMC-000
- Evidence criteria: Fresh origin/main, clean branch/worktree, commit/tree and Gitlinks, dependency-lock state, Python version, requested file inventory, DuckDB/Quack/DuckLake capability observations, and exact baseline commands are content-addressed.
- Evidence source policy: Git objects, local package metadata, read-only capability probes, source inspection, and command receipts are primary; old plan prose is historical evidence only.
- Outputs: docs/architecture/agent_supervisor_autonomous_meta_controller_inventory, benchmarks/agent_supervisor/autonomous_meta_controller/baseline_manifest.json, config/agent_supervisor_autonomous_meta_controller_scheduler.json, scripts/validate_agent_supervisor_autonomous_meta_controller_board.py, scripts/materialize_agent_supervisor_autonomous_meta_controller_board.py, ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py, ipfs_accelerate_py/agent_supervisor/analysis/mcp_invocation_trace.py, test/api/test_agent_supervisor_objective_daemon_import.py
- Interfaces: CurrentAuthorityInventory@1, APMCBaselineManifest@1, APMCBoardMaterializationReceipt@1, DatabaseProgramConfig@1, ObjectiveDaemonCompatibility@1, QuackOwnerMutationEnvelope@1
- Validation: python3 scripts/validate_agent_supervisor_autonomous_meta_controller_board.py --check-inventory --check-baseline-seal; python3 -m pytest -q test/api/test_agent_supervisor_intent_repository.py test/api/test_agent_supervisor_objective_daemon_import.py test/api/test_agent_supervisor_quack_state_server.py -k 'goal_edges or owner_mutation or objective_daemon'
- Acceptance: Every requested concern is classified as canonical authority, adapter, projection, compatibility surface, gap, or non-goal; the real objective daemon imports; the exact APMC board can be safely materialized; authenticated Quack owner mutations work and forged envelopes fail closed; and no sibling mutation or silent Gitlink change occurs.
- Gap task: APMC-000
- Refinement: DuckDB owns transactional state, Quack owns the single-owner transport boundary, and DuckLake is projection-only.
- Conflict policy: One inventory publisher; all later tasks consume the sealed result read-only.

## APMC-G020 B: define the closed autonomy contract model

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G010
- Priority: P0
- Track: contracts
- Bundle: agent-supervisor/autonomy/contracts
- Parallel lane: contracts
- Resource class: cpu-medium
- Goal: Define immutable, versioned, bounded autonomy/question/action/budget/episode/policy/skill/escalation/repair/receipt contracts by reusing canonical serialization and content identity.
- Evidence: APMC-001
- Evidence criteria: Closed enums, strict field sets, bounded payloads, safe paths, exact round trips, stable CIDs, and malicious/unknown input rejection pass hermetically.
- Evidence source policy: Existing CanonicalContract and content identity are authoritative; model-generated schemas and sibling-local duplicate contracts are prohibited.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py, test/api/autonomy/test_contracts.py
- Interfaces: AutonomyPolicy@1, AutonomyEnvelope@1, DecisionQuestion@1, CognitiveBudget@1
- Validation: python3 -m pytest -q test/api/autonomy/test_contracts.py
- Acceptance: Every required named contract exists, is immutable and canonical, rejects unknown/unbounded input, and cannot express unrestricted autonomy.
- Gap task: APMC-001
- Refinement: Contracts describe decisions and bounds but do not authorize effects.
- Conflict policy: APMC-001 exclusively owns autonomy contract definitions and exports.

## APMC-G030 C: resolve named questions under objective budgets

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G020
- Priority: P0
- Track: cognitive-control
- Bundle: agent-supervisor/autonomy/cognitive-control
- Parallel lane: cognitive
- Resource class: cpu-large
- Goal: Compile unresolved questions, preserve dependency-local state, reserve objective budgets, and deterministically choose software-first resolution actions with explicit model abstention.
- Evidence: APMC-002, APMC-003, APMC-004, APMC-005
- Evidence criteria: Question dedup/invalidation/restart, reservation/reconciliation/reserves, deterministic ordering, all model-call prohibition rules, and DecisionRuntime integration pass.
- Evidence source policy: Current evidence identities and existing context/planning/runtime authorities are primary; stage entry and model confidence do not trigger calls.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/decision_graph.py, ipfs_accelerate_py/agent_supervisor/autonomy/cognitive_budget.py, ipfs_accelerate_py/agent_supervisor/autonomy/cognitive_scheduler.py, ipfs_accelerate_py/agent_supervisor/autonomy/runtime.py
- Interfaces: DecisionGraph@1, CognitiveScheduler@1, ObjectiveBudgetLedger@1, AutonomousMetaController@1
- Validation: python3 -m pytest -q test/api/autonomy/test_decision_graph.py test/api/autonomy/test_cognitive_budget.py test/api/autonomy/test_cognitive_scheduler.py test/api/autonomy/test_runtime.py
- Acceptance: At least 80 percent of deterministic benchmark questions are resolved without a model and no action can consume protected validation reserves or bypass DecisionRuntime.
- Gap task: APMC-002, APMC-003, APMC-004, APMC-005
- Refinement: Existing context compiler, resource/proof schedulers, provider ledgers, and effect admission remain authoritative.
- Conflict policy: Contract, graph, budget, and scheduler files are disjoint; APMC-005 is their sole runtime integration owner.

## APMC-G040 D: learn routes from compact causally attributed episodes

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G020, APMC-G030
- Priority: P1
- Track: experience-policy
- Bundle: agent-supervisor/autonomy/experience-policy
- Parallel lane: policy-shadow
- Resource class: cpu-medium
- Goal: Persist privacy-safe content-addressed episodes, attribute failures from evidence, and evaluate a constrained shadow-only route learner without authority expansion.
- Evidence: APMC-006, APMC-007, APMC-008, APMC-009
- Evidence criteria: No prompt/source/private-reasoning retention; causal confounders remain explicit; shadow learner chooses only admitted actions; held-out/counterfactual evaluation returns insufficiency when propensity evidence is absent.
- Evidence source policy: Frozen identities and accepted validation/proof/merge results are authoritative; observational correlation alone cannot support causal or promotion claims.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/experience_ledger.py, ipfs_accelerate_py/agent_supervisor/autonomy/causal_attribution.py, ipfs_accelerate_py/agent_supervisor/autonomy/route_policy.py
- Interfaces: ExperienceLedger@1, CausalAttribution@1, ShadowRoutePolicy@1
- Validation: python3 -m pytest -q test/api/autonomy -k 'experience or attribution or route_policy or counterfactual'
- Acceptance: Episodes are bounded and replayable; controlled ablations remain shadow-only; rollback is exact; no learned choice changes authority, proof, validation, confirmation, provider, privacy, or simulation policy.
- Gap task: APMC-006, APMC-007, APMC-008, APMC-009
- Refinement: No online production exploration is allowed.
- Conflict policy: Ledger schema precedes attribution and learning; APMC-009 owns evaluation claims.

## APMC-G050 E: distill repeated reasoning into bounded rules and skills

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G040
- Priority: P1
- Track: distillation
- Bundle: agent-supervisor/autonomy/distillation
- Parallel lane: distillation
- Resource class: cpu-medium
- Goal: Convert independently validated repeated bounded decisions and workflows into declarative rules and allowlisted supervisor skills through a CEGIS-style process.
- Evidence: APMC-010, APMC-011
- Evidence criteria: Narrow DSL, stable typed features, bounded output vocabulary, explicit exceptions, development/counterexample/held-out/adversarial gates, shadow deployment, rollback, and out-of-domain fallback pass.
- Evidence source policy: Validated independent episodes and current assurance results are evidence; frequency and model-proposed executable code are not authority.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/policy_distillation.py, ipfs_accelerate_py/agent_supervisor/autonomy/supervisor_skills.py
- Interfaces: PolicyDistiller@1, DistilledDecisionRule@1, SupervisorSkill@1
- Validation: python3 -m pytest -q test/api/autonomy -k 'distill or skill'
- Acceptance: At least 20 percent of repeated qualifying decision classes are handled by promoted shadow rules with zero safety regression and exact fallback.
- Gap task: APMC-010, APMC-011
- Refinement: Skills are typed allowlisted programs, never arbitrary shell scripts.
- Conflict policy: Rule DSL lands before skills consume it; promotion remains owned by APMC-019.

## APMC-G060 F: preserve plan prefixes and repair bounded suffixes

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G030, APMC-G050
- Priority: P1
- Track: receding-repair
- Bundle: agent-supervisor/autonomy/receding-repair
- Parallel lane: repair
- Resource class: cpu-large
- Goal: Execute the nearest safe plan segment, preserve unaffected history, adapt existing delta-replan receipts, and compose the existing autonomous-repair engine under exact envelopes.
- Evidence: APMC-012, APMC-013
- Evidence criteria: Changed evidence invalidates only the dependency cone; provider outage reroutes eligible questions only; repair tiers enforce exact scope, isolation, tests/proofs, backoff, rollback, and merge policy.
- Evidence source policy: FormalDeltaReplanner, DecisionRuntime, existing autonomous-repair, worktree, lease, proof, validation, and merge authorities are canonical.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/receding_horizon.py, ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py
- Interfaces: PlanSuffixInvalidationReceipt@1, AutonomousRepairController@1
- Validation: python3 -m pytest -q test/api/autonomy -k 'suffix or repair or repeated_failure'
- Acceptance: No full replan occurs from a local change without dependency evidence and no repair can edit outside its admitted envelope or self-protecting authority paths.
- Gap task: APMC-012, APMC-013
- Refinement: APMC supplies facades/adapters, not replacement planning or repair engines.
- Conflict policy: Suffix adapter and repair facade are file-disjoint; repair consumes the accepted suffix contract.

## APMC-G070 G: minimize human questions and compact semantic memory

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G030, APMC-G040
- Priority: P1
- Track: human-memory
- Bundle: agent-supervisor/autonomy/human-memory
- Parallel lane: human-memory
- Resource class: cpu-medium
- Goal: Emit minimal irreducible human packets and retain only bounded, scoped, freshness-aware semantic memory over existing artifacts.
- Evidence: APMC-014, APMC-015
- Evidence criteria: One bounded question/options/consequences packet, mandatory-authority preservation, TTL/invalidation, no chain-of-thought/raw prompt/source copies, and frequency-without-authority-upgrade tests pass.
- Evidence source policy: Scoped human answers and current validated receipts retain their declared authority only; retrieval rank is never evidence class.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/human_escalation.py, ipfs_accelerate_py/agent_supervisor/autonomy/semantic_memory.py
- Interfaces: HumanEscalationCompiler@1, SemanticMemory@1
- Validation: python3 -m pytest -q test/api/autonomy -k 'human or escalation or memory or compaction'
- Acceptance: Human interventions fall on eligible benchmarks without bypassing any required human decision; memory growth and invalidation remain bounded.
- Gap task: APMC-014, APMC-015
- Refinement: Waiting time and unnecessary questions are first-class costs.
- Conflict policy: Escalation and memory implementations are disjoint; runtime joins them in APMC-016.

## APMC-G080 H: integrate event-driven receding-horizon runtime

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G030, APMC-G060, APMC-G070
- Priority: P0
- Track: runtime
- Bundle: agent-supervisor/autonomy/runtime
- Parallel lane: runtime
- Resource class: cpu-large
- Goal: Integrate the controller with existing event/runtime authorities so only meaningful changes wake the nearest safe segment and unchanged completed boards remain stable.
- Evidence: APMC-016
- Evidence criteria: Event dedup, dependency-local wake, cancellation, durable restart, idempotent replay, bounded logs/artifacts, healthy exhaustion, and zero unchanged writes/model calls pass.
- Evidence source policy: Existing event cursors, decision runtime, resource/proof schedulers, database repositories, and lease/fence observations are authoritative.
- Outputs: ipfs_accelerate_py/agent_supervisor/autonomy/runtime.py, test/api/autonomy/test_runtime.py
- Interfaces: AutonomousMetaController@1, AutonomyRuntime@1
- Validation: python3 -m pytest -q test/api/autonomy/test_runtime.py
- Acceptance: The controller progresses on current evidence, stops safely on insufficient authority/evidence/budget, and consumes near-zero resources while unchanged and idle.
- Gap task: APMC-016
- Refinement: Polling may exist only as a bounded safety timer.
- Conflict policy: APMC-016 is the sole runtime integration owner and may not absorb lower-layer authorities.

## APMC-G090 I: expose governed Python, CLI, and MCP controls

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G080
- Priority: P1
- Track: control-surface
- Bundle: agent-supervisor/autonomy/control
- Parallel lane: control-api
- Resource class: cpu-medium
- Goal: Add the specified autonomy read/mutation operations to SupervisorControlService and expose thin CLI/MCP adapters with exact parity.
- Evidence: APMC-017
- Evidence criteria: Python/CLI/MCP parity, side-effect-free discovery, direct typed MCP calls, authorization/idempotency/lease/fence/audit/dry-run/confirmation binding, and cancel/replay tests pass.
- Evidence source policy: SupervisorControlService is canonical; adapters cannot infer permission, call shell strings, or mint authority.
- Outputs: ipfs_accelerate_py/agent_supervisor/control, ipfs_accelerate_py/agent_supervisor/autonomy/cli.py, canonical MCP adapter
- Interfaces: SupervisorControlService autonomy operations, ipfs-accelerate agent autonomy
- Validation: python3 -m pytest -q test/api/autonomy -k 'control or cli or mcp or confirmation'
- Acceptance: All specified operations are present, typed, audited, and authority-equivalent across surfaces with no confirmation replay.
- Gap task: APMC-017
- Refinement: Read operations never start providers or mutate databases.
- Conflict policy: One service catalog owner; CLI and MCP adapters land after service contracts.

## APMC-G100 J: qualify on a frozen paired adversarial benchmark

- Status: active
- Review only: false
- Parent: APMC-G000
- Depends on: APMC-G010
- Priority: P0
- Track: benchmark
- Bundle: agent-supervisor/autonomy/benchmark
- Parallel lane: benchmark
- Resource class: cpu-io-large
- Goal: Build the fixed-seed paired corpus, measure tokens/calls/humans/quality/safety/runtime on identical fixtures, and preserve unavailable/not-run dimensions honestly.
- Evidence: APMC-018
- Evidence criteria: Corpus manifests cover all required case families; baseline/candidate inputs are identical; token attribution is complete; seeded defects and adversarial cases are sealed; generated board completion is excluded.
- Evidence source policy: Frozen manifests, fixture bytes, exact tree/policy/provider/fault/human bindings, and measured receipts are authoritative; inferred or simulated live quality is prohibited.
- Outputs: benchmarks/agent_supervisor/autonomous_meta_controller, test/fixtures/agent_supervisor/autonomy
- Interfaces: APMCBenchmarkManifest@1, APMCBenchmarkResult@1
- Validation: python3 benchmarks/agent_supervisor/autonomous_meta_controller/validate.py
- Acceptance: The benchmark can issue exact per-gate pass/fail/unavailable results and attributes every token/model/human/validation cost.
- Gap task: APMC-018
- Refinement: DuckLake is an optional analytics projection of benchmark receipts, not the result authority.
- Conflict policy: Benchmark manifests are frozen by APMC-018; later tasks append results through content-addressed receipts only.

## APMC-G110 K: promote or terminate honestly with a residual-gap report

- Status: active
- Review only: true
- Parent: APMC-G000
- Depends on: APMC-G090, APMC-G100
- Priority: P0
- Track: release
- Bundle: agent-supervisor/autonomy/release
- Parallel lane: release
- Resource class: cpu-large
- Goal: Evaluate every non-compensable gate, perform externally authorized expected-old policy CAS or exact rollback, and publish the current-tree release/residual-gap report.
- Evidence: APMC-019, APMC-020
- Evidence criteria: Held-out improvement, zero safety/quality regression, exact version binding, external authorization, rollback proof, final tests/benchmarks, process/claim/merge settlement, and current commit/tree are recorded.
- Evidence source policy: Current-tree validation, proof, benchmark, database, merge, and independent authorization receipts are primary; candidate policy cannot authorize itself.
- Outputs: autonomy promotion receipt, release report, residual-gap report
- Interfaces: AutonomyPromotionReceipt@1, APMCReleaseReport@1
- Validation: python3 scripts/validate_agent_supervisor_autonomous_meta_controller_board.py --check-all && python3 -m pytest -q test/api/autonomy
- Acceptance: Eligible promotion meets every requested gate; otherwise a non-promotion terminal names exact blockers without lowering thresholds.
- Gap task: APMC-019, APMC-020
- Refinement: A healthy non-promotion or exhausted terminal is preferable to a false success.
- Conflict policy: APMC-019 alone owns policy pointer CAS/rollback; APMC-020 is report-only and cannot alter policy.
