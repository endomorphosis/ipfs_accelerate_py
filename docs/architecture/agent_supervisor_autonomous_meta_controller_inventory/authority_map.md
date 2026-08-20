# Autonomous Meta-Controller authority map

This inventory was produced before APMC implementation against fetched
`origin/main` commit `bbf7f68799072c2b81f7d96eac91f2df3c4b3952`.

| Concern | Existing canonical authority | APMC composition rule |
| --- | --- | --- |
| Canonical serialization and identity | `proof.formal_verification_contracts.CanonicalContract`, `canonical_json_bytes`, `content_identity`; `core.multiformats_identity`; `task_sources.task_identity` | Reuse directly. Do not add a hash/CID or task identity system. |
| Context and value of information | `context.context_compiler.ContextCompiler`, `DecisionContextCompiler`, evidence selection, prefix capsule/reuse, retry delta and expansion contracts | Scheduler requests these actions; it never compiles a second context. |
| Final effect admission | `context.decision_runtime.DecisionRuntime`, `DecisionRuntimeInput`, `ExecutionPermit`, exact effect receipts | `AutonomousMetaController` selects resolution actions above this boundary and never grants effect authority. |
| Deterministic-first and replay suppression | `prompt.delta_task_packet` deterministic decisions, decision cache, replay circuit and unchanged-failure recording | Adapt to the cognitive action vocabulary and reuse its backoff/cache evidence. |
| Objective/task/obligation semantics | Objective graph, `planning.obligation_graph_compiler`, task identity, task-quality and completion contracts | Questions reference frozen criteria/obligations; they do not redefine them. |
| Planning | `planning.adaptive_planner`, `plan_evaluator`, `task_quality` | Reuse AND/OR candidates, hard constraints and quality admission. |
| Minimal suffix replanning | `planning.formal_replanner.FormalDeltaReplanner`, `DeltaReplanDecision` | `PlanSuffixInvalidationReceipt` is an adapter over this authority. |
| Verification | `verification.planner`, `selection`, `contracts`, `executor`, `receipt_cache`, `model_route` | Selected test/type/static actions delegate here and reuse freshness rules. |
| Proof | `proof.incremental_sealing`, `proof.proof_scheduler` and existing proof receipts | Reuse incremental proof and scheduling; do not add a prover/cache. |
| Compression policy | `semantic_governor` | Reuse privacy, context sufficiency, shadow comparison, omission diagnosis, held-out evaluation, CAS promotion and rollback. |
| Adversarial assurance | `adversarial_assurance` | Reuse mutation, survivor, remediation, seeded-defect, held-out and promotion gates. |
| Repair | Existing `agent_supervisor.autonomous_repair` package and engine | APMC adds a facade/controller only; it does not create a second repair engine. |
| Resource admission | `runtime.resource_scheduler`, proof scheduler and backpressure/cancellation services | Cognitive reservations are objective overlays; actual resources remain admitted here. |
| Provider/token accounting | `runtime.provider_call_ledger`, `runtime.provider_usage`, `self_improvement.supervisor_token_ledger`, efficiency metrics | Reuse actual-call/cost data and suppression; add only objective/epoch protected reserves. |
| Artifact/cache persistence | `runtime.runtime_cas`, artifact store, verification receipt store | Store content-addressed question/decision dependencies here; no vector DB or second cache. |
| Events and idle stability | `runtime.event_log`, `RuntimeWakeCoordinator`, projection checkpoint store | Use cursor replay, two-phase acknowledgement and no-unchanged-write checkpoints. |
| Control | `control.SupervisorControlService`, operation catalog, authorization, permits, leases, fencing, audit, confirmation and dry-run | Extend the typed service; CLI/MCP are thin direct adapters. |
| Operational database | `control_plane_schema`, `StateRepository`, `EmbeddedStateRepository`, `QuackStateRepository`, transactions, `IntentRepository`, `DatabaseTaskSource`, database event/artifact stores | DuckDB owns records/schema/CAS/fencing; Quack is mandatory for parallel writers and cannot fall back silently. |
| History/analytics projection | `integrations.ducklake_history_projection` and the released typed `ipfs_datasets_py` DuckLake API at the recorded pin | Optional, receipt-backed and non-authoritative. The API is present, but production activation is held behind DQK-088/DQK-094/DQK-102; APMC records that disabled disposition and never performs direct raw SQL/ATTACH. A genuinely absent required capability returns typed `unavailable`. |

## Genuinely new bounded scope

- closed autonomy policy/envelope/level and risk contracts;
- decision-question/belief lifecycle and dependency-local graph persistence;
- objective-and-epoch cognitive budget with validation reserves;
- cross-authority cognitive action scheduler and explicit model abstention;
- compact experience and causal attribution;
- constrained shadow route policy and honest counterfactual insufficiency;
- declarative decision-rule distillation and bounded skills;
- minimal human escalation and semantic-memory retention classes;
- one composing `AutonomousMetaController` runtime and governed public surface.

## Bootstrap compatibility findings

Two current-main defects prevented a truthful live launch even though they do
not change the authority map. First, the production objective daemon imported
reviewed MCP catalog/trace modules and multi-prover schema identities that were
absent from the source tree. The bootstrap restores the exact reviewed modules
and identifiers and tests the real import; no mock or permissive fallback is
used. Second, the generic Quack owner exposed the beta transport but did not
service the already-defined owner-side mutation inbox required for remote
UPDATE/DELETE/CAS. The bootstrap composes that inbox into the exclusive owner
with bounded authenticated envelopes and keeps the raw state credential out of
provider subprocesses. These repairs make existing authorities operable; they
do not give APMC a new database, mutation, or completion authority.

The APMC launch uses a dedicated program-bound DuckDB file and Quack store. It
does not reuse the live legal-board database or any archived DQK database,
whose repository/program/plan identities belong to different authorities.
The DuckLake projection remains disabled and non-authoritative at launch.

## Collision rules

APMC work must stop and record a typed design blocker if it would duplicate
`DecisionRuntime`, `FormalDeltaReplanner`, `ContextCompiler`, proof or
verification caches, provider/token ledgers, autonomous repair, content
identity, task/objective identities, or DuckDB/Quack repositories. A facade
must preserve the underlying receipt identity and cannot reinterpret authority.
