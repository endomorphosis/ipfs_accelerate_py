# Adversarial Assurance Engine implementation plan

Status: reviewed supervisor input

Program: `adversarial-assurance-engine-v1`

Task prefix: `AAE-`

Root goal: `AAE-G000`

## 1. Outcome and governing claim

Implement one focused subsystem, `AdversarialAssuranceEngine`, that uses
bounded counterfactual semantic mutations to discover important incorrect
behaviors which the current tests, proofs, policies, semantic summaries,
state-machine constraints, and incremental seals would mistakenly accept. It
then diagnoses the smallest assurance boundary at fault, evaluates candidate
strengthening against held-out mutations, and promotes only authorized,
content-addressed policy revisions.

The optimization objective is:

```text
discover important incorrect behaviors that current verification accepts,
then strengthen the smallest relevant assurance boundary without weakening
soundness or increasing context unnecessarily
```

An ordinary mutation score is not the objective and never proves correctness.
A successful test or proof establishes only the property actually specified.
It does not establish specification completeness, realistic assumptions,
nonvacuity, exercised code, deployment connection, required-set completeness,
direct execution, selected-test sufficiency, or semantic-capsule completeness.
Every report and receipt must preserve those distinctions.

## 2. Exact authority baseline inspected on 2026-08-13

The clean controller is
`/home/barberb/lift_coding/.worktrees/ipfs-accelerate-adversarial-assurance-engine`
on `agent/adversarial-assurance-engine-v1`, with this scoped repository
forest. The controller and the three owned top-level gitlinks are recursively
recorded; unrelated or uninitialized nested dependencies are not implied to be
release evidence:

| Authority | Exact commit | Planning disposition |
| --- | --- | --- |
| `endomorphosis/ipfs_accelerate_py` | `7c9f3fa3d2ac14c7b5bfa5036e2fe6fb59f0afda` | clean AAE controller base; a live, incomplete SCG branch snapshot |
| `endomorphosis/ipfs_datasets_py` | `fbd1ba9f70803de157622bb20e22595ef09d606f` | initialized gitlink; semantic governor and index/state contracts present |
| `endomorphosis/ipfs_kit_py` | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` | initialized gitlink; durable roots and governor store present |
| `endomorphosis/Mcp-Plus-Plus` | `dc3164653a48d059ae9812078359daeafb451c07` | initialized gitlink; shared schemas/vectors only |
| live SemanticCompressionGovernor controller | `2a2db39cfdc0b5c0408d3a43d7f66fc6897a5e3d` at inspection, datasets `90ca31b729a4b911703f99eb01f208159c34b98e` | clean but moving; program still running |
| live IncrementalProofSealer branch | `65ef6ed4cb9fa51d7c2c6c1a4f9036678f8cb893` at inspection | external moving dependency; program incomplete |
| proof-sealer datasets / kit gitlinks | `321fe10c191b9dab84206b5f9bd598aa2e46bcc8` / `866ae2cd0a1a94a794ff9316a9a4f67a10245957` | partial evidence, `ProofUnit`, admission, and store surfaces only |

The similarly named outer `lift_coding` worktree is structurally broken and is
not authority. No task may use it.

Focused pre-change evidence from the clean controller:

- datasets semantic index, capsule, invalidation, selection, and governor
  contracts: **152 passed** in 3.12 seconds;
- accelerate verification and semantic-governor selection: **283 passed, 1
  failed** out of 284; the failure is the frozen SCG authority matrix expecting
  datasets `1330038f...` while the live gitlink is `fbd1ba9f...`;
- kit durable semantic roots and governor store: **196 passed**;
- MCP++ CID envelopes, event DAG, conformance vectors, and Profile G codec:
  **58 passed** in 0.15 seconds.

The stale authority-matrix failure is a known baseline blind spot, not an AAE
success and not permission to weaken the check. Task `AAE-005` must reconcile
the exact dependency release or preserve a typed blocker.

## 3. Existing authorities to reuse

Datasets provides `IncrementalSemanticIndex`, repository-state/delta and
invalidation models, dependency graphs, `SemanticCapsuleCompiler@1` through
the functional capsule compiler, capsule freshness and raw-source fallback,
test/proof selection, canonical bytes, and CID authority. The compiler name is
an interface identifier, not a class to recreate.

Accelerate provides `ContextPacker`/`pack_context` (the current-tree equivalent
of the requested `ContextPackBuilder`), `IncrementalVerificationPlanner`,
`VerificationReceiptCache`, `VerificationExecutor`, `ModelRoutePlanner`,
selected/full-suite comparison, process isolation, resource scheduling,
timeouts, cancellation, counterexample minimization, and provider routing.

Kit provides `DurableCoordinationStore`, immutable blocks, operation-id
idempotency, recovery, `DurableStateRootAdapter`, expected-old compare-and-swap,
and semantic-governor history patterns.

The live proof-sealer branch currently exposes datasets evidence/`ProofUnit`
and kit proof-seal store contracts/transports, but does **not** yet expose the
required public Python `IncrementalProofSealer`, `FullCheckpointSeal`,
`DeltaSeal`, `create_full_checkpoint`, `publish_full_checkpoint`,
`build_delta_seal`, or `publish_delta_seal` surfaces. Runtime and seal tasks are
release-gated. Missing capability is `typed_unavailable`; an AAE-local proof
system or commitment masquerading as a seal is forbidden.

Current taxonomies that adapters must preserve are:

- semantic confidence `exact`, `conservative`, `heuristic`, `opaque`; capsule
  freshness `fresh`, `stale`, `unknown`; and capsule admission as exact
  substitute, conservative with caveats, or raw source required;
- verification receipt kinds static analysis, type check, test, and proof, with
  terminal statuses `passed`, `failed`, `proved`, `disproved`, `unknown`,
  `timeout`, `unavailable`, `not_modeled`, `stale`, `invalid`, `cancelled`, and
  `simulated`; cache dispositions also distinguish reused, missing, corrupt,
  mismatched, non-authoritative, and policy/terminal rejection;
- existing `MutationLedger@1` file-lineage states accepted, rejected,
  quarantined, no-op, partial, and rolled back; it is reusable lineage evidence,
  not an assurance campaign engine;
- proof modes direct execution, theorem certificate, signed receipt, receipt
  aggregation, integrity only, and simulated; proof/seal statuses separately
  preserve disproved, unknown, timeout, unavailable, cancelled, invalid,
  simulated, stale parent, incomplete manifest, invalid cache, proof failure,
  and full-reproof-required conditions. A verification Merkle commitment is
  structural evidence, not a ZK execution proof.

Existing test assets include the semantic-state mutation manifest (body,
signature, schema, exception, fixture, configuration, plugin/lock/policy,
generated/dynamic/native, deletion, and rename cases); the SCG 60-case corpus
partitioned 14 calibration, 14 development, and 32 held-out cases; proof-reuse
invalidation mutation fixtures; and broad Hypothesis property tests. There is
no integrated semantic mutation campaign, general equivalent-mutant analyzer,
expected-detection engine, assurance-gap taxonomy, or authorized AAE promotion
flow. The only located formal-vacuity support is a narrow heuristic
`is_vacuous_statement`/`_check_non_vacuity` path for proof candidates.

Known semantic blind spots include dynamic dispatch/imports, reflection,
descriptors/decorators, `eval`/`exec`, metaclasses, monkey patching, pytest
plugin collection, generated/native bindings, runtime generation,
configuration/fixture discovery, renames, and uncontrolled I/O. The existing
incremental-verification benchmark is explicitly RED (20 cases, one seeded
false negative, seven false positives, real provers unavailable, 40 percent
frontier/human escalation). Existing SCG numbers are offline replay only (40
tasks, 58.9 percent median reduction, 100 percent recall, 36.22 percent
precision, 34 pass / 4 reject / 2 escalate, zero production-eligible) and must
not be relabeled live assurance.

Do not introduce another semantic index, dependency graph, capsule compiler,
context packer, CID implementation, proof cache, receipt envelope, scheduler,
proof/ZK system, provider, or MCP++ profile. A new MCP++ profile is forbidden.

The explicit non-goals are: no production deployment; no general autonomous
code repair; no unrestricted mutation of arbitrary external repositories; no
GUI implementation; no legal advice; no payment processing; and no automatic
lowering of assurance requirements. GUI/action-binding fixture analysis is
permitted only where an existing canonical optimizer artifact is already
available, and it does not create a GUI.

## 4. Repository ownership

### `ipfs_datasets_py`

Owns closed mutation and assurance models; deterministic semantic mutation
operators and generation; claim/specification analysis; risk selection;
expected detection construction; bounded equivalence and vacuity analysis;
gap classification; adequacy profiles; and candidate test, proof, policy, and
analyzer specifications. Canonical package:
`ipfs_datasets_py.logic.software_contracts.adversarial_assurance`.

### `ipfs_accelerate_py`

Owns assurance manifests, campaign composition, isolated disposable worktrees,
resource admission, mutation workers, test/prover/policy execution, incremental
selection, timeouts/cancellation, temporary proof forests, minimization,
diagnosis orchestration, remediation evaluation, promotion orchestration, CLI,
metrics, benchmark, and report. Canonical package:
`ipfs_accelerate_py.agent_supervisor.adversarial_assurance`.

### `ipfs_kit_py`

Owns thin typed persistence over existing durable primitives: immutable mutant
artifacts, campaign state, execution receipts, assurance gaps, benchmark
artifacts, campaign Merkle roots, policy revisions, CAS promotion state,
idempotent replay, and crash recovery. Canonical package:
`ipfs_kit_py.adversarial_assurance_store`.

### `Mcp-Plus-Plus`

Owns only a demonstrably shared mutation-campaign schema, assurance-receipt
schema, canonical vector, or cross-language conformance datum. Application
payloads remain inside existing profiles. A new AAE or MCP++ profile is
forbidden.

## 5. Closed, versioned artifact vocabulary

Implement at least these models, rejecting unknown fields and noncanonical,
nonfinite, executable, path-escaping, oversized, or versionless inputs:

```text
MutationOperatorDefinition       MutationTarget
MutationCandidate                MutationCampaignPolicy
MutationCampaignPlan             ExpectedDetectionSet
MutationExecutionPlan            MutationExecutionReceipt
MutationOutcome                  MutationEquivalenceAssessment
SurvivingMutantReport            AssuranceGap
VacuityFinding                   DetectionFailure
TestAdequacyProfile              ProofAdequacyProfile
PolicyAdequacyProfile            CapsuleAdequacyProfile
CandidateTestSpecification       CandidateProofObligation
CandidatePolicyConstraint        CandidateAnalyzerRule
GapRemediationPlan               RemediationEvaluationReport
AssuranceCampaignReceipt         AssurancePolicyPromotionReceipt
AssuranceManifest
```

Every persisted artifact binds repository identity and repository-state root;
target symbol/artifact identities; relevant semantic-capsule and proof-unit
identities; environment and dependency-lock identities; operator, campaign
policy, and generator versions; terminal status; provenance; canonical
identity; and referenced receipt/proof identities where applicable.

Signed assurance evidence also binds signer/key identity, audience, action,
signature bytes, and signature verification status through the existing
receipt/signature authority. Content addressing or Merkle inclusion never
substitutes for authenticity or direct-execution evidence.

Mutation outcomes use only:

```text
killed_by_static_analysis        killed_by_type_check
killed_by_test                   killed_by_formal_proof
killed_by_policy                 killed_by_runtime_invariant
killed_by_full_suite             survived_selected_verification
survived_full_verification       equivalent
probably_equivalent              invalid_mutant
uncompilable                     infrastructure_failure
timeout                          inconclusive
human_review_required
```

Equivalence assessment uses only `equivalent`, `probably_equivalent`,
`not_equivalent`, and `unknown`. Invalid, uncompilable, infrastructure-failed,
timeout, inconclusive, or equivalent mutants are not counted as killed.
Survival is evidence of an assurance gap candidate, not automatically a
product defect.

Assurance gaps use a closed set including `missing_test`, `weak_assertion`,
`missing_proof_obligation`, `vacuous_proof`, `missing_policy_constraint`,
`stale_or_incomplete_dependency_edge`, `capsule_completeness_failure`,
`test_selection_failure`, `unmodeled_side_effect`,
`missing_state_transition_constraint`, `missing_environment_binding`,
`receipt_authenticity_gap`, `specification_ambiguity`,
`intentionally_unconstrained`, `probably_equivalent`, and `unknown`.

## 6. Deterministic mutation operators

The registry implements bounded, semantically meaningful operators for all
eleven required classes:

1. control flow: conditional inversion, branch removal/unconditional behavior,
   comparison boundaries, recovery/obligation/cancellation/termination changes;
2. data/schema: required/nullable/default/version/order/bounds/float/unknown
   field, truncation, swapping, and Unicode normalization changes;
3. interface contract: parameter/pre/postcondition/error/exception/version/
   handler and structurally-valid-but-wrong result changes;
4. side effect: missing/wrong/early/double/reordered writes, audit omission,
   success-before-observation, and missing compensation;
5. error/retry: swallowed or misclassified failures, unavailable/unknown to
   success/allow, missing retry budget, cancellation, and integrity bypass;
6. authorization/policy: authentication, tenant, attenuation, audience,
   expiry, revocation, stale/default policy, confirmation replay, and payment;
7. state/distributed: illegal/skipped transitions, CAS/fencing/lease/ownership,
   idempotency, partial mutation, convergence, proof-forest, and parent seals;
8. storage/durability: pre-commit acknowledgement, directory sync/checksum,
   stale reads, corruption replacement, queued-as-committed, and provider-ack;
9. test/proof: deletion/skip/weak assertions/fixtures, impossible assumptions,
   unreachable proofs, stale/omitted units, unchecked signatures, and
   integrity-only evidence;
10. semantic compression: omitted dependencies/config/fixture/exception/effect,
    stale/wrong-root/heuristic capsules, opaque-as-exact, and selection misses;
11. GUI/action binding, only when canonical GUI optimizer artifacts exist:
    dispatchability, confirmation, handler, stale policy, recovery, early
    success, and critical keyboard accessibility. Broad visual mutation is out.

Each `MutationOperatorDefinition` declares ID/version, supported language or
artifact types, target prerequisites, semantic intent, expected violated
property classes, risk class, likely-equivalent conditions, syntactic
transformation, scope limits, rollback, required sandbox, and maximum mutants
per target. Given identical source state, target, operator, seed, and config,
generation and identities must be byte-for-byte deterministic.

## 7. Admission and expected detection

For each candidate, the executor creates a disposable Git worktree, applies
the bounded edit, rescans the artifact, proves only declared files/symbols
changed, prevents verifier/policy/key/oracle mutation unless explicitly
targeted in a separate verifier fixture, parses syntax, performs basic type or
structural checks, rejects trivial invalidity, estimates equivalence, constructs
the expected detection set, and commits the immutable mutant identity. No
production worktree or branch may be a mutation target.

`ExpectedDetectionSet` may name static rules, type checks, selected unit and
integration tests, property tests, formal obligations, policy rules, runtime
invariants, full-suite fallback, incremental seal verification, and human
review. Every prediction binds the violated claim, why the detector should
observe it, the connecting source/proof dependency, required versus optional
strength, and expected terminal status. Classification records predicted,
selected, executed, observed, missed, and unexpected detectors separately.

## 8. Campaign and incremental execution

The campaign pipeline is: establish an unmutated green baseline; risk-select
targets; generate bounded candidates; admit valid mutants; run predicted
incremental checks first; broaden survivors per policy; classify; minimize
important survivors; diagnose gaps; propose remediations; evaluate on held-out
mutations; and seal the complete campaign.

For every mutant, use the existing invalidation plan, test/proof selection,
receipt cache, and verifier. Reuse unaffected proof units only after complete
cache-key validation, update a temporary proof forest, and never replace the
canonical repository seal. Full-suite fallback occurs only when policy,
uncertainty, or high risk requires it. Measure actual incremental cost and the
counterfactual cost of full verification for every mutant.

Risk weighting prioritizes security/authorization, durability, financial or
legal consequence, distributed transitions, proof/receipt trust, execution
frequency, defect history, capsule uncertainty, missing tests, recent changes,
fan-out, failure cost, and critical MCP++ invariants. Formatting, proven
generated code, immutable dependencies, and low-risk boilerplate may be
sampled at lower rates. Generation is always bounded.

## 9. Equivalence, vacuity, and survivor diagnosis

Bounded equivalence combines AST and normalized-IR comparison, constant
propagation, available symbolic execution, restricted SMT, reachability, and
identical public behavior over bounded domains. Unresolved high-value cases go
to human review. Difficulty to kill never implies equivalence.

Vacuity analysis must cover:

- formal proofs: unsatisfiable antecedents, unreachable modeled state,
  impossible discharge, unconstrained results, omitted behavior, and behavior
  assumed rather than proven;
- policy: unreachable rules/confirmations, shadowed prohibitions, impossible
  obligations, dominating defaults, and obsolete interface references;
- tests: tautologies, type-only/non-null assertions, behavior-independent
  mocks, uncalled targets, permanent skips, bypassing fixtures, and success
  before effect observation;
- ZK/receipts: unbound required fields/source/environment, inclusion without
  completeness, caller-selected verification key, signed aggregation presented
  as execution, and changed units missing from delta seals.

Every vacuity finding states exactly what remains proven. For a survivor,
diagnosis checks equivalence; detector selection/execution/path observation;
assertion strength; dependency/capsule omissions; unspecified or intentionally
unconstrained behavior; and need for human judgment.

Minimized reports contain the smallest changed source region, symbol IDs,
violated/missing property, detectors run and omitted, smallest reproducing
input, expected/observed behavior, source spans, dependency path, proof/receipt
IDs, reproduction command, and risk. Full logs are excluded from model context
unless bounded minimization fails.

## 10. Remediation, held-out evaluation, and promotion

Allowed candidates are additional/stronger tests, property tests, proof
obligations, pre/postconditions, policy constraints, state-machine invariants,
dependency edges, invalidation rules, capsule fields, manifest requirements,
and full-suite fallback rules. Model drafts start as `heuristic_candidate` and
cannot self-promote.

Candidate tests bind intended behavior, source/symbol identities, setup/input/
observation/fixtures, killed mutation classes, and requirement provenance; they
must not merely freeze the current implementation. Proof candidates bind the
proposition, assumptions, modeled/excluded state, source/interface connection,
prover, expected counterexample, practical nonvacuity condition, and risk.

Partition mutants deterministically into diagnosis, development, and held-out
sets. Evaluate the unmutated suite, diagnosis mutant, related development and
held-out mutants, unrelated behavior, performance/cost, false positives, and
overconstraint. Reject regressions, overfit implementation assertions, flakes,
mock bypass, safety weakening, impossible correct behavior, or unapproved
material cost increases.

Promotion requires canonical candidate identity, held-out evaluation, no
critical regression or new vacuity, declared cost and coverage impacts,
authorization, expected-old policy revision CAS, and a new incremental seal.
Stale writers lose. Generated tests/proofs cannot authorize themselves, and
production policy never changes during a fixture campaign. Promotion and
campaign receipts are signed by the released existing signer authority and
must pass signature verification before storage, sealing, or authorization.

## 11. Required fixture campaigns

### Security campaign

The controlled corpus includes all twenty cases: authentication bypass;
caller-selected tenant; missing attenuation; accepted expired and revoked
delegations; missing confirmation; cross-action confirmation replay; missing
policy default-to-allow; payment-as-authority; stale fencing token; retry
double execution; uncompensated partial mutation; provider acknowledgement as
verified storage; receipt before observed effect; invalid signature;
pseudo-CID; stale proof receipt; omitted proof unit; unknown prover as passed;
and simulated production evidence. Each fixture declares which existing
mechanism should kill it.

### Semantic-compression campaign

Fixtures omit a required side effect, exception, and result-changing fixture;
conceal a schema mutation with a stale capsule; substitute heuristic for raw
source; treat an opaque plugin as exact; miss a relevant selected test; and
show expanded context succeeding where compressed context fails. Results add
calibration evidence to the existing `SemanticCompressionGovernor`, but never
automatically change production policy.

### ZK and incremental-seal campaign

Fixtures remove a receipt leaf/required unit; change source root/environment
CID/parent seal/proof-forest order; use an old key; attach proof to the wrong
statement; delete a test without authorization; substitute simulated for
direct proof; ignore a blocking child; and replay a proof across branches. All
critical controlled cases must be rejected before qualification.

### Distributed, durability, crash, vacuity, and GUI campaigns

Exercise transitions, ownership, leases, fencing, CAS, idempotency,
compensation, durability observation, test/proof/policy vacuity, and conditional
canonical GUI action bindings. GUI scope excludes visual rendering mutations.

## 12. Required public surfaces

Expose semantically equivalent Python APIs:

```text
create_assurance_manifest(repository_state, verification_policy)
generate_mutation_candidates(manifest, mutation_policy)
predict_detection_set(mutation, assurance_manifest)
execute_mutation(mutation, verification_policy)
classify_mutation_outcome(mutation, expected_detection, observed_verification)
diagnose_surviving_mutant(mutation, outcome, repository_state)
analyze_vacuity(assurance_manifest, repository_state)
propose_gap_remediation(surviving_mutant, assurance_gap)
evaluate_remediation(remediation, held_out_campaign)
promote_assurance_policy(remediation, evaluation, authorization)
plan_mutation_campaign(repository_state, assurance_manifest,
                       mutation_policy, resource_budget)
execute_mutation_campaign(plan, verification_policy)
```

Expose CLI equivalents for `assurance mutate plan|run|target|explain`,
`assurance gaps`, `assurance vacuity`, `assurance remediate`,
`assurance evaluate-remediation`, `assurance promote`, `assurance report`, and
`assurance benchmark`. Exact spelling may follow the current CLI host.

## 13. Persistence, crash, and concurrency

Inject failures after mutant creation, during worktree setup/test/proof
execution, after receipt persistence, during diagnosis/evaluation/root update,
before policy CAS, and after CAS before cleanup. Restart must replay
deterministically, reject ambiguous execution claims, preserve immutable
completed artifacts, avoid partial promotion, prevent stale concurrent policy
writers, and safely clean only owned disposable worktrees.

Mutants run network-disabled by default without production credentials.
External effects use controlled fakes/local sandboxes. They cannot modify
trusted keys, policy, verifier, benchmark oracle, or promotion authority except
in explicit separate verifier fixtures. Source comments/test messages are
untrusted input, no arbitrary filesystem path is exposed through an API, and
no public network service is required.

## 14. Sealing scope and limits

Use the released existing `IncrementalProofSealer` to seal operator versions,
campaign policy, admitted set, expected detection sets, outcomes, survivor
reports, vacuity findings, held-out evaluations, and final policy revision.
The seal may establish exact committed campaign artifacts, declared-result
completeness, evaluation-to-promotion binding, and status-policy satisfaction.
It cannot establish repository correctness, mutation-set completeness,
specification completeness, or direct execution unless the underlying proof
does. Until released APIs are pinned, sealing is typed unavailable and
qualification remains blocked.

The campaign receipt remains separately signed assurance evidence: the seal
commits its exact bytes and completeness scope, while the verified signature
binds its signer/key identity and authorization provenance. Neither mechanism
is allowed to overclaim the other.

## 15. Metrics and success criteria

Track separately:

- mutation coverage: generated/admitted/invalid/equivalent/killed/selected
  survivors/full survivors, class kill rates, and risk-weighted score;
- detection quality: predicted/observed/missed/unexpected detectors, selected
  test/proof/policy rates, and full-suite-only detections;
- gaps: missing/weak tests, proof/vacuity, policy, dependency, capsule, seal,
  state, environment, authenticity, and ambiguity categories;
- remediation: candidates, held-out kills, regressions, overconstraint, cost,
  and accepted/rejected promotions;
- economics: full versus incremental cost per mutant, proof-cache reuse, model
  calls/tokens, wall time, cost per critical gap, and cost per promotion.

Initial targets are goals, never fabricated results: zero controlled critical
security survivors after accepted remediation; zero accepted stale-proof/seal
integrity mutants; at least 90 percent high-risk semantic detection; an
explicit gap for every high-risk survivor; held-out evaluation for every
promotion; no meaningful claim for a vacuous proof; at least 50 percent compute
savings; deterministic campaign IDs; no worktree escape; and no unauthorized
production policy change. Actual results, including misses, must be reported.

## 16. Parallel execution and release gates

The board has 64 bounded tasks and ten goals (one root plus nine subgoals).
Two strict resource-admitted lanes process disjoint task-ID shards across datasets analysis, accelerate
runtime, and kit/conformance/qualification. The initial inventory wave and all
pre-runtime contracts, operators, analysis, and storage can advance
independently. `AAE-006` is an operator-controlled,
unschedulable release gate requiring a final SCG receipt, released proof-sealer
surfaces, fresh exact pins, and focused baselines. Runtime/seal tasks depend on
that gate; workers may not emulate missing dependencies.

The scheduler is a thin source specialization of the existing fail-closed IVP
multi-supervisor. It uses strict sharding, isolated worktrees, bounded attempts,
quota-only provider fallback, resource claims, exact lifecycle fencing, and no
objective/codebase refill. The sealed launch is limited to two lanes because
the host was saturated at planning time; changing that limit requires a fresh
operator review and control seal. `preflight` is launch-only;
live monitoring uses `status` and exact stop fencing.

## 17. Deliverables and terminal qualification

### Requirement-to-task coverage

| Required capability | Owning tasks |
| --- | --- |
| exact inspection, manifests, baselines, blind spots, release gates | `AAE-001`–`AAE-006` |
| all closed/versioned models, identities, statuses, shared conformance | `AAE-007`–`AAE-013` |
| registry and all eleven deterministic bounded operator classes | `AAE-014`–`AAE-020` |
| claim/target/risk selection, generation, expected detectors, admission | `AAE-021`–`AAE-024` |
| equivalence, four vacuity families, gaps, adequacy, survivor diagnosis | `AAE-025`–`AAE-031` |
| candidate specifications and diagnosis/development/held-out policy | `AAE-032`–`AAE-033` |
| immutable artifacts, campaign history, roots, policy CAS, recovery | `AAE-034`–`AAE-038` |
| manifests, campaign planning, isolation/workers, incremental execution | `AAE-039`–`AAE-044` |
| minimization, remediation evaluation, controlled promotion, all APIs | `AAE-045`–`AAE-048` |
| partitioned corpus and security/compression/ZK/distributed/vacuity/GUI campaigns | `AAE-049`–`AAE-055` |
| every required CLI surface and disjoint metrics/economics | `AAE-056`–`AAE-058` |
| sandbox/security and all crash/concurrency qualification | `AAE-059`–`AAE-060` |
| held-out promotion, success-target evaluation, sealing, benchmark, SCG evidence | `AAE-061`–`AAE-062` |
| trust model, limitations, actual results, and current-tree final report | `AAE-063` |

Each task declares repository-scoped predicted files, dependency edges,
resource class, strict lane shard, time and context bounds, canonical
interfaces, focused validation, conflict policy, evidence subset, and
acceptance criteria in the supervisor taskboard. The objective heap groups the
same tasks into authority, contracts, mutation, analysis, durability, runtime,
campaign, qualification, and closeout subgoals.

The terminal fan-in covers the operator registry, schemas, detection model,
isolated incremental executor, equivalence and vacuity analysis, survivor/gap
diagnosis, remediation generation and held-out evaluation, controlled
promotion, security/compression/distributed/ZK campaigns, cache and sealer
integration, campaign seal, benchmark/cost report, and trust/limitations docs.

The final report records exact commits; reused modules; operators; generated,
admitted, killed, survived, equivalent, invalid, and inconclusive counts;
risk-weighted score; high-risk survivors; vacuity and gap categories;
selected/full-suite and proof/policy detection rates; incremental savings and
cache reuse; proposed/evaluated/promoted/rejected remediations; regressions and
overconstraint; seal results; improved assurance; remaining unspecified or
unverified behavior; and exact enterprise/production next steps.

The only appropriate concluding claim is:

```text
The system used semantically targeted counterfactual mutations to test whether
declared tests, proofs, policies, semantic summaries, and incremental seals
reject important incorrect behavior. Surviving mutants were classified as
assurance gaps, candidate remediations were evaluated against held-out
mutations, and accepted assurance-policy changes were promoted through a
reproducible, content-addressed qualification process.
```

No high mutation score is presented as proof of correctness.
