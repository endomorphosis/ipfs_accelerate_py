# Semantic Compression Governor implementation plan

Status: reviewed supervisor input

Program: `semantic-compression-governor-v1`

Task prefix: `SCG-`
Root goal: `SCG-G000`

## 1. Outcome and boundary

Implement one bounded subsystem, `SemanticCompressionGovernor`, that audits
whether compressed coding-agent context was sufficient, compares selected
compressed executions with controlled expanded executions, diagnoses likely
omissions, performs bounded counterexample-guided expansion, calibrates future
context and model-route decisions, and evaluates declarative policy candidates
on held-out tasks before an authorized compare-and-swap promotion.

The optimization objective is:

```text
minimize context and inference cost
subject to accepted-patch quality, dependency coverage, verification,
risk, privacy, and authorization constraints
```

Token reduction is never an acceptance criterion by itself. The governor is an
auditor and proposal system, not an autonomous production self-modifier.

The governor may collect evidence, schedule bounded shadow evaluations, build
expanded contexts, rerun isolated tasks, update empirical calibration, propose
typed rules, evaluate candidates, and recommend promotion. It may not rewrite
production safety rules, lower assurance, directly change a production route,
mark a heuristic capsule exact, disable a full-suite fallback, suppress a
verification failure, treat model agreement as proof, or modify production
code outside an isolated evaluation worktree.

Explicit non-goals are another semantic index or capsule compiler, another
proof cache, content-identity or receipt format, a new ZK system or MCP++
profile, a new model provider, a GUI or default public server, general
reinforcement learning, autonomous production self-rewriting, automatic
trusted-source modification, or automatic lowering of assurance.

## 2. Exact authority baseline inspected on 2026-08-13

The clean supervisor controller is based on these exact revisions:

| Repository / authority | Commit | Status at planning |
| --- | --- | --- |
| `endomorphosis/ipfs_accelerate_py` | `dfd92b554e662d4312411f2e8e63a52368806f2a` | `origin/main`; completed SemanticCompressionHarness and incremental verification planner |
| `endomorphosis/ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` | completed incremental semantic index and semantic-state/capsule contracts |
| `endomorphosis/ipfs_kit_py` | `df2f9cc092456329de9724c45a50c54b410875d1` | completed durable semantic roots plus hermetic Profile-G vectors |
| `endomorphosis/Mcp-Plus-Plus` | `dc3164653a48d059ae9812078359daeafb451c07` | shared wire/conformance authority; no new execution profile permitted |
| Incremental verification freeze in accelerate history | `8c7800cedc5e1b848367db9952f912428466f8cc` | completed; API available under `agent_supervisor.verification` |
| Incremental proof-sealer development branch | `7dc8f1422cb7e80757077948dc0785c1aaa4fd25` | live upstream program; only early contracts landed when inspected |

The active outer checkouts were dirty or divergent and are not launch
authority. The SCG controller uses a separate clean branch and worktree and
pins its three nested repositories by gitlink. The contemporaneous proof
sealer is an external release dependency for final proof-seal qualification;
until its public API lands, SCG must return typed `unavailable`/`inconclusive`
and must not substitute the incremental-verification Merkle commitment for a
ZK or execution proof.

### Focused baseline evidence

Run from the pinned clean controller before authoring implementation tasks:

- accelerate semantic-state/context/routing plus incremental-verification
  contracts, planner, cache, route, and counterexample tests: **234 passed**;
- datasets index/capsule/invalidation/selection/content-identity tests:
  **67 passed**;
- kit coordination/root contract, CAS, recovery, and adapter tests:
  **72 passed** after pinning the hermetic-vector repair;
- MCP++ CID, conformance-vector, and event-DAG tests: **34 passed**.

The first kit run exposed seven collection errors caused by the older
`05ba9375...` test fixture's hard-coded MCP++ worktree path. Pinning
`df2f9cc0...`, whose sole purpose is hermetic vector vendoring, resolved them.
This is recorded as baseline provenance rather than hidden.

### Existing public interfaces to consume

Datasets:

- `IncrementalSemanticIndex.scan_repository`, `diff_repository_states`,
  `calculate_invalidation`, `explain_symbol`, `explain_impact`, and
  `watch_repository`;
- closed `RepositoryState`, `RepositoryStateDelta`, `InvalidationPlan`,
  symbol/artifact/edge/span models, and exact/conservative/heuristic/opaque
  confidence taxonomy;
- `build_semantic_state`, `verify_semantic_state_bundle`,
  `open_semantic_state`, `compile_semantic_capsule`,
  `assess_capsule_freshness`, `read_required_source`,
  `extend_semantic_invalidation`, `select_tests_and_proofs`, and
  `compare_test_selection_oracle`;
- canonical bytes and CID authority only from
  `logic.software_contracts.content`.

`SemanticCapsuleCompiler@1` is an interface identifier implemented by the
functional capsule compiler; it is not a public class and must not be
re-created.

Accelerate:

- `semantic_state.ContextPacker`, `pack_context`, `ContextCoveragePolicy`,
  `SemanticCompressionHarness`, `run_semantic_patch_loop`, isolated worktree
  and patch validation, scheduling contracts, provider execution, and
  `ResourceScheduler`;
- `verification.create_verification_plan`,
  `IncrementalVerificationPlanner`, `VerificationReceiptCache`,
  `VerificationExecutor`, `CounterexampleMinimizer`,
  `ModelRoutePlanner`, `choose_model_route`, bundle/summary builders, and
  selected-versus-full-suite evaluation;
- canonical `VerificationBundle`, `TestReceipt`, `ProofReceipt`,
  `CounterexampleReceipt`, route decisions, and terminal status vocabulary.

Kit:

- `DurableCoordinationStore` immutable blocks, recovery, operation-id
  idempotency, and one-writer expected-old compare-and-swap;
- `DurableStateRootAdapter`, `DurableStateRoots`, `StateRootSnapshot`,
  `StateRootCASResult`, and recovery reports.

MCP++:

- existing Profile A/B/F envelope, request/result, event-DAG, and CID vectors,
  plus the released Profile G scheduling/artifact codecs and conformance
  vectors consumed by kit; Profile G is existing authority, not a new SCG
  execution profile;
- shared schema/vector changes only when a true interoperability requirement
  exists; never a new SCG execution profile.

Release-time proof inventory must locate both canonical full-checkpoint and
delta/incremental seal surfaces (including `FullCheckpointSeal`/
`create_full_checkpoint`/`publish_full_checkpoint` and `DeltaSeal`/
`build_delta_seal`/`publish_delta_seal`, if those exact released names exist).
Each surface is recorded as available with its exact commit and public API, or
typed unavailable; the plan does not infer either from a top-level class name.

## 3. Current status and known limitations

Datasets confidence is closed to exact, conservative, heuristic, and opaque.
Freshness is fresh, stale, or unknown. Heuristic, opaque, invalid, or stale
capsules already require raw source. Unknown dynamic reachability can force
full pytest/proof fallback. Known weak areas include reflection, descriptors,
import hooks, metaclasses, monkey patching, plugins, generated bindings, native
extensions, runtime generation, uncontrolled I/O, and dynamic pytest behavior.

Incremental verification has a closed terminal vocabulary including passed,
failed, proved, disproved, unknown, timeout, unavailable, not-modeled, stale,
invalid, cancelled, and simulated. Its checked-in benchmark is explicitly RED
and non-authoritative: 20 cases, one seeded false negative, seven false
positives, real provers unavailable, and 40 percent frontier/human escalation.
That evidence is a baseline and known-failure input, not a success claim.

The completed semantic-compression harness benchmark is an offline controlled
oracle/replay workload. It reports 40 tasks, 58.90 percent median reduction,
100 percent recall, 36.22 percent precision, no controlled false negatives,
34 pass / 4 reject / 2 escalate outcomes, and no production-eligible result.
SCG must not relabel those simulations as live model quality.

## 4. Ownership and package boundaries

### `ipfs_datasets_py`

Owns canonical semantic-governor evidence models, context coverage,
pre-execution sufficiency analysis, omission attribution, bounded expansion
planning, empirical capsule/analyzer/task calibration, and typed declarative
rule proposals. It consumes the semantic index/state view and never rescans or
reimplements identity.

### `ipfs_accelerate_py`

Owns governor orchestration, shadow sampling, resource admission, isolated
compressed and expanded executions, semantic differential comparison,
verification/counterexample integration, bounded retry and model escalation,
active audit scheduling, route calibration, policy evaluation, CLI, metrics,
and reports. It wraps the existing harness rather than rebuilding its loop.

### `ipfs_kit_py`

Owns thin typed storage manifests over `DurableCoordinationStore`: immutable
audit cases, calibration and benchmark history, policy versions, promotion
state, receipts, recovery, and compare-and-swap publication. It does not
create another object store, WAL, or CID implementation.

### `Mcp-Plus-Plus`

Owns only genuinely shared schemas or cross-language conformance vectors. SCG
payloads otherwise remain application payloads inside existing profiles.

## 5. Required canonical artifacts

The implementation must expose closed, versioned models for:

```text
CompressionAuditCase
ContextSufficiencyClaim
ContextCoverageManifest
ExcludedArtifactRecord
OmissionHypothesis
OmissionEvidence
ContextExpansionPlan
ContextExpansionStep
ShadowExecutionPlan
ShadowExecutionResult
DifferentialPatchReport
SemanticOutcomeComparison
CapsuleCalibrationRecord
AnalyzerCalibrationProfile
TaskClassCalibrationProfile
ModelRouteCalibrationProfile
RuleProposal
RuleEvaluationReport
CompressionPolicy
CompressionPolicyCandidate
CompressionPolicyPromotionReceipt
GovernorDecision
GovernorRunReceipt
```

`CompressionPolicy` includes a closed task-class acceptance matrix declaring
required selected tests, full-suite fallback, static/type checks, proofs, and
human review for each admitted task/risk class. An absent or unknown mapping
fails closed; a verification pass outside that matrix cannot establish
acceptance or sufficiency.

Every durable artifact carries a schema version, canonical identity,
repository-state identity, relevant ContextPack identity, relevant
verification-bundle identity, generator version, provenance, assumptions, and
one closed terminal status. The common header is a typed application payload
that uses existing canonical serialization and receipt envelopes; it is not a
new generic receipt format.

Datasets owns the neutral `GovernorRunReceipt` and
`CompressionPolicyPromotionReceipt` payload schemas because they cross
execution and storage boundaries. Kit alone owns durable issuance, immutable
storage/history, expected-version CAS transitions, and their canonical
existing envelope bindings. This split introduces no second receipt hierarchy.

Sufficiency states are exactly:

```text
sufficient
sufficient_with_caveats
expansion_required
frontier_escalation_required
human_review_required
inconclusive
invalid
stale
evaluation_failed
```

Comparative outcomes are exactly:

```text
equivalent_success
compressed_better
expanded_better
both_valid_different
compressed_failed_expanded_succeeded
compressed_succeeded_expanded_failed
both_failed_same_reason
both_failed_different_reason
verification_inconclusive
human_review_required
```

A verification pass alone cannot establish `sufficient`.

## 6. Context coverage and pre-execution admission

For every ContextPack, build a `ContextCoverageManifest` with target symbols,
raw and capsule inclusions, raw exclusions, dependency paths, represented
proofs/tests/state reads/writes/schemas/configurations/fixtures, dynamic or
opaque dependencies, assumptions, per-exclusion confidence and token cost,
context budget, minimum-safe estimate, and known gaps.

Every exclusion has one closed reason:

```text
exact_capsule_substituted
conservative_capsule_substituted
proven_unrelated_by_dependency_graph
outside_affected_invalidation_cone
generated_from_included_authoritative_schema
verified_immutable_dependency
duplicate_representation
budget_exceeded_escalation_required
```

Heuristic irrelevance cannot exclude a critical dependency. The preliminary
decision joins confidence/freshness, opaque dependencies, unresolved
invalidation obligations, risk class, cone/cut size, proof/test coverage,
historical omission rates, budget, and route tier. Coverage uncertainty
broadens context; reasoning complexity with complete coverage may instead
raise the model tier; authorization, opaque high risk, conflicting evidence,
or disclosure/budget overflow requires human review.

## 7. Shadow evaluation and differential comparison

Shadowing is selected by a versioned audit policy using risk, capsule
uncertainty, new analyzers/task classes/routes, token savings, proof-cache
reuse, uncertainty, recent omissions, random quality-control sampling, and
promotion evaluation. Rates are configurable, including 100 percent during
development, high-risk mandatory shadowing, mature low-risk sampling, and
zero external shadow calls where disclosure policy forbids them.

The compressed run uses the normal ContextPack and route. The expanded run
uses the smallest complete affected raw dependency cone plus relevant schemas,
configuration, fixtures, tests, and optionally a stronger route. It runs in a
separate disposable evaluation worktree and is an oracle/candidate only; it
never silently replaces the accepted patch.

`DifferentialPatchReport` compares files/symbols, AST edit classes,
interfaces, side effects, exceptions, schemas, tests, proofs,
counterexamples, static analysis, performance, acceptance/review, tokens,
cost, and time. Textual difference is not semantic failure.

## 8. Omission diagnosis and bounded repair

When compressed work is inferior, rank omitted symbols/files/schemas/
fixtures/configuration/tests/proof obligations using graph paths, exclusion
reasons, capsule class, counterexamples/failures, source spans, expected
relevance, inclusion cost, confidence, expansion action, and proposed
long-term rule change.

The diagnosis must consider both omission and model-capability causes. The
bounded loop reads minimized failures, ranks hypotheses, adds the smallest
raw source or stronger capsule, retries the same model where appropriate, and
escalates the model only after context expansion is insufficient. Every step
records artifacts, reason, token increase, changed assumptions, prior/new
result, and hypothesis support. Hard limits cover steps, token growth,
retries, escalations, wall time, and spend.

## 9. Calibration, declarative rules, and promotion

Calibration tracks uses, task/risk classes, compressed/expanded success,
omission and stale failures, false exact classifications, unnecessary raw
fallbacks, token savings, verification cost, and review disagreement by
language, symbol kind, framework, analyzer/dynamic feature, capsule class,
task class, route tier, and repository family. Empirical success changes
routing and audit frequency only; it never upgrades formal exactness.

Rule proposals use a bounded typed DSL, never executable model text. Its
allowlisted categories cover dependency extraction, invalidation, capsule
completeness, raw-source inclusion, context ranking and packing, context
budget thresholds, model-route thresholds, shadow-sampling rates, and safe
full-suite-fallback policy. Each proposal names current version, proposed
rule, supporting audits, benefit, safety impact, scope, benchmark, and
rollback. Candidates are evaluated on a held-out partition disjoint from
calibration/development evidence; no proposal may disable the required
full-suite fallback.

Promotion requires schema/integrity validation, no critical-omission or stale
receipt regression, no high-risk assurance reduction, no hidden accepted
regressions, declared cost/context benefit, required authorization, release
qualification, and expected-version CAS. A stale candidate or a candidate
trying to authorize itself is rejected. Rollback is another authorized CAS,
not history deletion.

## 10. Security and privacy invariants

Comments, docstrings, issue text, tests, logs, and retrieved documentation are
untrusted data. Instruction-like content is recorded as audit evidence but
cannot change routing, verification, capsule classification, trusted keys,
proof systems, sampling, source inclusion, or promotion.

Provider policy controls disclosure. Expanded private source is local-only
unless an explicit provider authorization permits the exact disclosure.
Secrets are redacted before invocation; public reports store CIDs and managed
references, not raw private source. Models cannot change trusted keys,
authorities, proof systems, or promotion state. No arbitrary filesystem path
is exposed through a network interface, and no public server is added.

## 11. Required public APIs and CLI

The release surface provides equivalents of:

```python
evaluate_context_sufficiency(context_pack, repository_state,
                             verification_policy, calibration_profile)
create_shadow_plan(task, compressed_context, repository_state, audit_policy)
compare_shadow_results(compressed_result, expanded_result,
                       verification_evidence)
diagnose_omission(audit_case, repository_state, dependency_graph)
plan_context_expansion(audit_case, omission_hypotheses, token_budget)
execute_expansion_loop(plan, model_policy, verification_policy)
update_calibration(audit_case, current_profile)
propose_rule_change(calibration_profile, audit_cases)
evaluate_rule_candidate(candidate, held_out_benchmark)
promote_compression_policy(candidate, evaluation_report, authorization)
```

The `semantic-governor` console entry exposes `audit`, `shadow`, `diagnose`,
`expand`, `calibrate`, `propose-rules`, `evaluate-policy`, `promote-policy`,
`report`, and `dashboard-data`. It emits bounded deterministic JSON by default
and starts no public service or GUI.

## 12. Metrics and evidence

Metrics cover raw/retrieval/compressed/expanded tokens, percentiles and
expansion rate; accepted patches, regressions, selected-test false negatives,
proof failures, review disagreement, and outcome distribution; intentional
omission detection before/after execution, critical omissions accepted,
false alarms, expansion precision/recall; route share, escalation, and retries;
input/output tokens, model spend, verification/shadow compute, cost per
accepted patch, gross/net savings; and calibration uses, empirical omission
rate with confidence interval, last revision, and task coverage.

Initial targets are targets, never fabricated results: at least 95 percent
critical intentional-omission detection before automatic acceptance, zero
critical controlled omissions accepted, at least 50 percent median final
context reduction, no meaningful accepted-patch degradation, no stale capsule
or receipt admission, no heuristic capsule treated as exact, useful bounded
expansion before frontier escalation, reproducible rollback-safe promotion,
and audit spend below protected inference savings at the configured rate.

## 13. Benchmark and adversarial matrix

Use immutable calibration, development, and held-out partitions. Cover local
bugs, exception/API/schema migrations, state and configuration changes,
fixtures, dynamic imports, monkey patching, generated code, plugins,
cross-module refactors, documentation, and proof changes.

Controlled adversarial cases include hidden callee side effects, caller
exception contracts, configuration flags, pytest fixtures, serializers,
generated interfaces, stale capsules, confidence misclassification, opaque
dynamic imports, behavior-only dependency changes, security invariants,
migration paths, misleading comments, prompt injection, selected-pass/
full-fail, test-pass/formal-fail, raw-correct/compressed-wrong, and both-context
model failure. Simulated model outputs are excluded from live-quality claims.

## 14. Incremental sealing claim boundary

The governor seals benchmark definitions, compressed/expanded ContextPack
CIDs, verification bundles, differential reports, calibration profiles,
candidate rules, and promotion decisions through the released incremental
proof-sealer adapter. Until that capability is present, these artifacts can be
content-addressed and committed but the seal status is `unavailable`; the IVP
Merkle commitment remains explicitly non-ZK.

The initial proof scope may establish only that exact artifacts were evaluated,
required evaluations completed, declared thresholds were applied, no blocking
status was omitted, and the promoted policy equals the evaluated candidate.
It does not prove semantic sufficiency unless that property is actually encoded
and proven.

## 15. Parallel delivery graph

```text
W0  SCG-000
W1  SCG-001 | SCG-002 | SCG-003 | SCG-004
W2  SCG-005
W3  SCG-006 | SCG-040
W4  SCG-007 | SCG-008 | SCG-009 | SCG-010
W5  SCG-011 | SCG-013 | SCG-016 | SCG-019
W6  SCG-012 | SCG-014 | SCG-020 | SCG-021
W7  SCG-015 | SCG-022
W8  SCG-017
W9  SCG-018
W10 SCG-023 | SCG-041
W11 SCG-024 | SCG-028
W12 SCG-025
W13 SCG-026
W14 SCG-027
W15 SCG-029
W16 SCG-030
W17 SCG-031
W18 SCG-032
W19 SCG-033 | SCG-038 | SCG-042 | SCG-043
W20 SCG-035
W21 SCG-034
W22 SCG-036
W23 SCG-039
W24 SCG-037
W25 SCG-044
W26 SCG-045
W27 SCG-046 | SCG-047
W28 SCG-048
```

The taskboard is the executable dependency authority. Waves summarize
available parallelism but never override task dependencies, predicted-file
conflicts, strict sharding, resource admission, or privacy policy.

## 16. Completion and final claim

The terminal report records exact inspected and implemented commits, consumed
interfaces, audit population, differential outcomes, omission detection and
critical acceptance, expansion success/size, final context reduction, route
distribution/escalation, quality/regression comparison, overhead and cost,
rules proposed/rejected/promoted, rollback, seal scope, heuristics, and
remaining production risks. Missing or unavailable measurements are reported
as such.

The maximum justified claim is:

> The system empirically and structurally audits semantic compression,
> detects and diagnoses omitted-context failures, expands context using
> counterexamples and dependency evidence, calibrates future compression
> decisions, and promotes rule changes only after held-out evaluation and
> authorized, reproducible qualification.

It never claims to prove that every compressed context is semantically
complete.
