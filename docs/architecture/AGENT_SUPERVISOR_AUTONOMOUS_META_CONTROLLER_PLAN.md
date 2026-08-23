# Agent Supervisor Autonomous Meta-Controller plan

Program: `agent-supervisor-autonomous-meta-controller-v1`
Root objective: `APMC-G000`
Task namespace: `APMC-`
Target package: `ipfs_accelerate_py.agent_supervisor.autonomy`

## 1. Outcome and limits

`AutonomousMetaController` is a bounded question-and-action controller above the
existing supervisor layers. It identifies the exact unresolved decision that
prevents progress, selects the cheapest admissible evidence-producing action,
and abstains when evidence, authority, privacy, provider health, or reserved
validation budget is insufficient. It never becomes a second executor,
context compiler, planner, proof system, identity system, receipt envelope,
worktree manager, lease service, or policy authority.

The optimization target is accepted criteria and verified progress per unit of
model, token, provider, validation, human, and wall-time cost. Safety,
authority, freshness, correctness, privacy, proof coverage, and validation
coverage are hard constraints and are never exchanged for lower cost.

The controller does not implement an unrestricted autonomy level. It does not
generate executable policy code or arbitrary shell commands, store private
chain-of-thought, explore production routes, weaken validation, mutate trusted
keys or promotion rules, modify sibling repositories, or release high-risk
changes without external authority.

## 2. Sealed baseline

Planning began from fetched `origin/main` commit
`bbf7f68799072c2b81f7d96eac91f2df3c4b3952` on Python 3.12.3 in clean branch
`codex/agent-supervisor-autonomous-meta-controller-v1`. The exact Gitlink pins,
dependency state, capability probes, and authority inventory are recorded in
`agent_supervisor_autonomous_meta_controller_inventory/`.

The dependency state is requirements/optional-extra based: `pyproject.toml`,
`requirements.txt`, and specialist requirements files exist; no repository
lockfile was present at baseline. Gitlinks are recorded without changing their
pins. Optional sibling interfaces are imported lazily and absence produces a
typed `unavailable` result.

Current main also contained a bounded packaging regression: the real
objective daemon referenced reviewed MCP contract-catalog/trace modules and
multi-prover schema identities that were absent from the tree. The bootstrap
tranche restores those exact reviewed authorities and tests the production
import; it does not add a mock, fallback, or weaker gate.

The host capability observation is not a policy grant: DuckDB 1.5.5 is present;
the repository probe reports the installed Quack extension compatible and
healthy without network installation; DuckLake and httpfs 1.5.5 extensions are
installed. Quack remains beta. DuckLake remains a non-authoritative bounded
history/benchmark projection and cannot grant scheduling, mutation,
completion, proof, lease, CAS, or promotion authority.

## 3. Authorities that remain canonical

The implementation composes these authorities:

- `context.context_compiler.ContextCompiler` and
  `DecisionContextCompiler` remain the only context assembly, value-of-
  information, prefix reuse, expansion, and delta-retry authorities.
- `context.decision_runtime.DecisionRuntime` remains the final provider-free
  decision/context/plan-admission/permit boundary before effects. The meta-
  controller selects a question-resolution action and later supplies a bound
  `DecisionRuntimeInput`; it does not authorize mutation.
- `planning.adaptive_planner`, `plan_evaluator`, and `task_quality` remain the
  planning and quality authorities. `planning.formal_replanner` remains the
  dependency-minimal suffix-replan and repeated-failure authority.
- `semantic_governor` remains the compression, shadow comparison, omission,
  privacy, calibration, and promotion authority.
- `adversarial_assurance` remains the mutation, survivor, remediation, held-
  out, and assurance-promotion authority.
- `proof`, `verification`, their schedulers, receipt stores, and current
  incremental sealing contracts remain proof and validation authorities.
- `runtime.resource_scheduler`, provider usage/call ledgers, artifact/event
  stores, and current cancellation/backpressure machinery remain execution
  accounting and resource authorities.
- `control.SupervisorControlService` and existing authorization,
  idempotency, lease, fencing, audit, dry-run, and confirmation binding remain
  the mutation-control authority.
- `task_sources.task_identity` and
  `proof.formal_verification_contracts` remain content-identity and canonical
  serialization authorities.
- The existing `autonomous_repair` package remains the repair engine. A later
  `AutonomousRepairController` is a narrow facade/composer, not a replacement.
- Existing worktree, checkout-lock, lease, merge queue, intent repository,
  database task source, Quack state-owner/client, and DuckLake projection
  boundaries remain canonical.

Any inventory conflict is resolved by adapting to one existing authority or
recording a typed gap. A convenience facade never becomes authority merely
because it is easier to call.

## 4. Closed autonomy model

The P0 contract package defines immutable, versioned, bounded records for the
policy, envelope, risk, questions, graph, beliefs, actions, decisions, budgets,
episodes, attribution, route candidates, distillation, skills, escalation,
repair, run, and promotion receipts listed in the program requirements.
Unknown fields, floats, unknown enum members, duplicate identifiers, unsafe
paths, oversized text, oversized collections, inconsistent claimed content
identities, and self-authorizing records fail closed.

Autonomy levels are exactly:

```text
observe_only
recommend
dry_run
execute_reversible
execute_bounded_mutation
self_repair_isolated
```

Risk classes are exactly `R0_PURE`, `R1_READ_ONLY`,
`R2_REVERSIBLE_LOCAL`, `R3_BOUNDED_REPOSITORY_MUTATION`,
`R4_SECURITY_OR_PROTOCOL_SENSITIVE`, and
`R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL`. A record may constrain authority but may
not raise it. R4 defaults to proposal/dry-run and R5 always requires a human.

Every `AutonomyEnvelope` binds repository/tree identity, objective revision,
task and acceptance subset, risk and reversibility, blast radius, exact paths
and symbols, tests/proofs, authority/policy identities, resource and token
budgets, escalation policy, and expiry.

## 5. Decision and belief graph

Plans and tasks compile into named `DecisionQuestion` nodes. Semantic
deduplication keys exclude display aliases but include objective revision,
criterion subset, question type, alternatives, required evidence, dependency
questions, risk, and terminal rule. Evidence dependencies are explicit.

Question state changes are append-only observations. A changed evidence
identity invalidates only direct dependants and their question dependants.
Independent and already terminal questions remain stable. The durable graph is
canonical and restartable. A task can be proposed as complete only when every
mandatory question has a current admissible terminal decision; actual task
completion still goes through existing validation, proof, merge, and control
authorities.

`BeliefFact` records provenance, authority class, freshness, contradiction,
and scope. Frequency affects retrieval rank only. It never upgrades evidence.
Contradiction produces an unresolved question or typed escalation, never a
last-write-wins answer.

## 6. Cognitive scheduling and budgets

The scheduler has the closed action vocabulary in the requirements. Every
candidate declares preconditions, expected evidence, expected uncertainty
reduction, integer token/latency/provider/resource/privacy costs, risk,
cancellation, cacheability, and authority class.

Hard filters run before utility. The deterministic default route is:

1. current authoritative evidence;
2. deterministic software analysis;
3. current immutable cache;
4. incremental test/proof;
5. local small model;
6. remote standard model;
7. remote strong model;
8. human decision.

Within an equally authoritative route class, selection maximizes expected
decision value divided by the sum of bounded costs, using integer cross-
multiplication and stable content-ID tie breaking. A model call is forbidden
without a named unresolved question, when its answer cannot change an
admissible decision, when an authoritative deterministic route exists, on a
current identical result, on repeated failure without new evidence, when
privacy forbids disclosure, when the result cannot carry required authority,
or when validation reserves would be consumed.

`CognitiveBudget` is objective-and-epoch scoped. Reservation precedes action;
reconciliation attributes actual cost to the original question and decision.
Validation/proof reserves are protected, speculative and retry costs retain
causal ownership, cache/prefix savings are recorded conservatively, and any
unattributed token is a benchmark failure. Exhaustion returns a typed terminal
or escalation state and never expands itself.

## 7. Experience, learning, and distillation

The experience ledger stores only frozen input identities, typed features,
selected action and policy, provider/model identity, context/token metrics,
evidence references, terminal/acceptance/validation/proof/merge results,
intervention, signatures, counterexamples, cost, and latency. It excludes raw
prompts, source copies, private reasoning, decoded bodies, and unrestricted
transcripts.

Causal attribution uses the closed failure vocabulary from the requirements.
Attribution requires discriminating evidence; for example, a model is not
blamed when required source was omitted, and compression is not credited from
a passing compressed run alone. Controlled ablations execute only in bounded
shadow mode.

Route learning is dependency-light, shadow-only, held-out, rollbackable, and
constrained to already admitted actions. Initial linear scores use integer or
rational features. No production exploration occurs. Missing propensity or
comparison evidence yields `insufficient_counterfactual_evidence`.

The `PolicyDistiller` emits only the bounded declarative rule DSL. Candidate
rules are narrower than observed model behavior, keep an out-of-domain model
fallback, pass development/counterexample/held-out/adversarial gates, and are
promoted only by externally authorized expected-old CAS. `SupervisorSkill`
steps are allowlisted typed operations with explicit pre/postconditions,
validation, rollback, fallback, and scope; they are not arbitrary scripts.

## 8. Receding horizon, repair, and escalation

The controller freezes objective and accepted-plan revision, executes the
nearest safe segment, observes evidence, preserves the valid prefix, and asks
the existing `FormalDeltaReplanner` to replace only the dependency-affected
suffix. `PlanSuffixInvalidationReceipt` is an autonomy-facing adapter over the
existing delta-replan receipt, preserving its identities and proof of minimal
invalidation.

The repair facade selects among deterministic, template-constrained, and
model-assisted tiers but delegates admission and execution to the existing
autonomous-repair engine and `DecisionRuntime`. Model-assisted repair requires
exact files/symbols, a bounded patch envelope, sufficient context, an isolated
worktree, predetermined tests/proofs, and protected authority paths. Repeated
identical failures reuse diagnosis and back off.

Human escalation emits one precise bounded question with 2–4 options, safest
recommendation, consequences, cost/risk, evidence references, continuation
for each answer, and expiry. It is used only for irreducible ambiguity,
operator-only authority, irreversible/legal/financial effects, irresolvable
contradiction, or policy-required budget/privacy choices. Learning may reduce
unnecessary questions but cannot bypass mandatory human authority.

## 9. Semantic memory and event-driven runtime

Memory is a bounded index over existing content-addressed artifacts and
receipts. Classes are `ephemeral_attempt`, `short_lived_negative`,
`task_episode`, `repository_pattern`, `cross_repository_rule`,
`authoritative_current`, and `withdrawn`. Each entry has TTL, invalidation
dependencies, evidence class, and scope. Compaction retains stable contracts,
validated signatures/patterns/rules, current capabilities, proof/test
dependencies, accepted outcomes, scoped human answers, and counterexamples.

Runtime wakes only on meaningful repository/objective/task/validation/proof/
provider/lease/human/budget/counterexample/freshness/window events. An
unchanged complete or healthily exhausted board performs no model calls, no
unchanged writes, no refill, and no repeated full scans; only a bounded safety
timer remains.

The P0 `AutonomousMetaController` is provider-free. It owns question/action
selection and compact decision receipts. Effect execution continues through
`DecisionRuntime`, resource/proof schedulers, control service, and current
lease/worktree/merge authorities.

## 10. DuckDB + Quack + DuckLake topology

The executable board is materialized into a dedicated APMC database using the
existing normalized DuckDB control-plane schema and a clean-tree-bound,
fail-closed materializer. Markdown is a sealed bootstrap/export only after
materialization. One Quack state-owner exclusively opens DuckDB; parallel
supervisors use typed `QuackStateRepository` commands and cannot fall back to
direct multi-process file access. Store, schema, generation, server,
repository, process-birth, lease, fence, plan-root, and tree identities are
checked at every mutation boundary.

DuckLake receives only bounded, idempotent, receipt-backed history and
benchmark projections from a DuckDB outbox cursor through the reviewed public
typed integration. It is never queried to decide readiness or acceptance.
Projection failure creates targeted backpressure and typed `unavailable`
evidence while operational work continues if DuckDB/Quack remain healthy.
Direct accelerator ATTACH, raw DuckLake SQL, credentials, catalog-file access,
or sibling-source mutation are prohibited.

## 11. Public control surface

Read operations expose capabilities, status, metrics, graph, unresolved
questions, budget, experience summary, route policy, distillation candidates,
repair history, escalations, and shadow results. Mutation operations pause,
resume, set a bounded level, approve/reject/rollback an externally authorized
policy candidate, approve a repair, or cancel an action.

Python service methods are canonical. CLI and MCP are thin typed adapters; MCP
does not shell out. Every mutation retains existing authorization,
confirmation, idempotency, lease, fencing, expected-effect, audit, and dry-run
semantics. Discovery and imports are side-effect free.

## 12. Verification and benchmark strategy

Hermetic focused tests live under `test/api/autonomy/`. They cover strict
contract round trips/identity/bounds, graph deduplication/invalidation/restart,
budget reservation/reconciliation/reserves, deterministic ordering and all
abstention cases, unavailable providers/privacy/cache freshness, learning and
distillation insufficiency/rollback, escalation minimization, confirmation and
repair scope, repeated-failure backoff, suffix preservation, idle stability,
prompt injection, forged receipts/model authority/self-promotion, and seeded
defect escape. Live provider/Quack/DuckLake cases are separately marked.

The frozen paired corpus uses identical trees, objective revisions, provider
fixtures, model configuration, capabilities, fault schedule, policy, token
accounting, human fixtures, and fixed seeds for baseline and candidate.
Completion of a generated board is not a quality metric. Every result records
not-run/unavailable dimensions rather than substituting simulated values.

Safety gates are non-compensable: zero false completions, unauthorized
mutations, simulated-as-live results, stale authoritative cache hits,
confirmation replays, path or scope escapes, hidden validation reductions,
escaped critical seeded defects, or self-authorized policy promotions.

The exact token-efficiency gates against the sealed current-main baseline are
at least 30% lower median model input tokens per accepted criterion, at least
25% fewer total model calls per accepted task, at least 40% lower retry input
tokens, at least 20% of repeated decision classes handled by distilled rules,
and at least 50% of eligible low-risk tasks completed without a large remote
model. The accounting rejects a claimed saving that moves cost into retries,
validation, human questions, wall time, missing evidence, or worse accepted
patches.

The autonomy gates are at least 30% fewer human interventions on eligible
tasks, at least 80% of deterministic questions resolved without a model, and
at least 90% correct decision-action selection on held-out typed fixtures,
with no increase in unsafe actions or unnecessary repository changes. Quality
requires no loss of acceptance/evidence coverage or required mutation-kill
rate and no increase in post-merge regressions or inconclusive proof and
validation outcomes. Runtime requires deterministic restart, idempotent
replay, bounded artifact/event growth, and completed-board idle operation with
no unchanged writes.

A missed gate creates a content-addressed non-promotion receipt with its exact
blocker. Gates cannot be lowered by the candidate policy.

## 13. Delivery graph

The machine-ingestible goal tree is in
`agent_supervisor_autonomous_meta_controller.objectives.md`; executable tasks
are in `agent_supervisor_autonomous_meta_controller.todo.md`; the sealed
four-lane DuckDB/Quack launch profile is in
`config/agent_supervisor_autonomous_meta_controller_scheduler.json`. The
principal waves are:

```text
baseline
  -> contracts
     -> {decision graph, budget, experience, frozen benchmark}
        -> scheduler -> software-first integration
        -> attribution -> shadow route policy -> offline evaluation
        -> distillation -> bounded skills
        -> suffix controller -> repair facade
        -> escalation + memory
        -> event runtime
        -> control/CLI/MCP
        -> promotion/rollback
        -> current-tree release
```

File-disjoint tasks may run in parallel. Shared exports, schemas, migrations,
control catalogs, and release receipts have explicit integration owners.
Workers may propose task results; only current-tree validation/proof evidence
and the existing supervisor/control authorities may accept them.

## 14. P0 tranche and terminal definition

P0 delivers all closed contracts, a deterministic restartable question graph,
an objective budget ledger with protected validation reserves, a software-first
cognitive scheduler with explicit model abstention, and a provider-free
`AutonomousMetaController` composition shell. Focused tests run after each
bounded task.

The program is complete only when APMC-000 through APMC-020 have current-tree
accepted evidence, the frozen paired benchmark and all safety/quality gates
pass, DuckDB/Quack replay and restart are deterministic, DuckLake projection
is either qualified or truthfully unavailable without affecting authority,
the board is settled, and an externally authorized promotion CAS succeeds.
Otherwise the terminal receipt is `non_promoted`, `human_decision_required`,
`budget_exhausted`, `capability_unavailable`, or another closed non-success
state with exact residual blockers.
