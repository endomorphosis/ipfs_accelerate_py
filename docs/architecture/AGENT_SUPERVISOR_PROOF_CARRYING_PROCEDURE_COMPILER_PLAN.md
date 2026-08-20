# Proof-Carrying Procedure Compiler v1

Program: `agent-supervisor-proof-carrying-procedure-compiler-v1`
Principal subsystem: `ProofCarryingProcedureCompiler`
Task namespace: `PCPC-`
Root objective: `PCPC-G000`
Implementation branch: `codex/proof-carrying-procedure-compiler-v1`

## 1. Outcome and scope

This program compiles repeated, independently validated supervisor work into
bounded declarative procedures. A procedure is a reusable planning operator;
it is never an authority, capability grant, proof, confirmation, promotion
decision, or completion receipt. The target steady-state path is:

```text
admitted execution evidence
  -> normalized trajectory
  -> task-family boundary
  -> structural anti-unification and candidate specifications
  -> bounded CEGIS synthesis
  -> independent verification and adversarial assurance
  -> shadow registry and held-out evaluation
  -> authorized promotion
  -> deterministic interpretation with typed residual holes
```

This plan extends existing supervisor authorities. It does not replace the
context compiler, semantic indexes, proof cache, adaptive/formal planners,
provider router, worktree/lease/fencing/merge controls, task identity, receipt
hierarchy, or control service.

## 2. Non-compensable invariants

All stages fail closed on these floors:

- Zero unauthorized mutation or effects.
- No mutation without existing admission, authority, scope, lease, and fence.
- No procedure execution outside declared repository-relative scope.
- No arbitrary shell, Python, network request, or filesystem-path synthesis.
- No model-generated executable policy or direct promotion of model text.
- No procedure self-authorization, self-certification, self-promotion, or
  self-completion.
- No procedure assertion may establish a precondition, observed effect,
  postcondition, rollback, proof, or completion.
- No stale procedure, certificate, proof, validation, policy, environment, or
  repository-tree binding may be reused.
- No simulated, mocked, task-board, prose, or receipt-shaped data may be
  represented as live evidence.
- No validation, proof, assurance, review, or release requirement may be
  removed, weakened, hidden by composition, or traded against a soft metric.
- No secret, credential, private prompt, private chain-of-thought, unbounded
  source body, or model transcript may enter a public artifact.
- A CID establishes canonical identity only. Authority remains an independent
  admitted decision.

## 3. Qualified starting baseline

The initial implementation baseline is repository commit
`bbf7f68799072c2b81f7d96eac91f2df3c4b3952`, tree
`a698da9e4b54e2929adacb613bc61ba3e72eed58`, package version `0.0.45`.
The detailed evidence-bound inventory is under
`docs/architecture/procedure_compiler_inventory/`.

The gate currently classifies the required authorities as follows:

| Authority | Classification | Planned treatment |
|---|---|---|
| SemanticCompressionHarness | available | Reuse its admitted semantic-state receipts. |
| SemanticCompressionGovernor | available | Reuse its bounded governance decisions. |
| Adversarial assurance | available_with_caveats | Adapt `AssuranceCampaignApi`; do not invent a second engine. |
| IncrementalVerificationPlanner | available | Reuse current-tree proof/test planning. |
| IncrementalProofSealer | available_with_caveats | Use leaf APIs through a narrow import adapter until the facade exports them. |
| AdaptivePlanner | incompatible | Typed blocker: committed import references a missing committed MCP catalog module. |
| SupervisorControlService | available | Extend its typed service and operation catalog in tranche 4. |
| ContextCompiler | available | Reuse content references, prefix reuse, budget evidence, and compilation receipts. |
| Value-of-information selection | available | Reuse `ContextCompiler` policy and evidence types. |
| Delta retry contexts | available_with_caveats | Reuse core API; retain the known daemon-integration failure as a blocker. |
| Provider capacity and model-route policy | available_with_caveats | Use typed capacity snapshots; do not depend on the stale default daemon route. |
| Worktree, lease, fencing, merge | available_with_caveats | Reuse canonical owners; qualify board-scoped concurrency and lease horizon before promotion. |
| AutonomousMetaController/autonomy/cognitive scheduler/experience ledger/policy distillation | missing | Do not recreate. Record comparison-baseline and later integration blockers. |

Missing successor-generation authorities do not block P0 contracts, parsing,
interpretation, world-state projection, delta extraction, or transition
calibration. They do block claims against an autonomous-meta-controller
baseline and any feature that specifically depends on them.

## 4. Authority and storage boundaries

### 4.1 Existing owners retained

- Canonical DAG-JSON and CIDv1 identity:
  `proof.formal_verification_contracts`.
- Public control authorization and effect lattice: `control.control_contracts`
  and `SupervisorControlService`.
- Semantic world snapshot:
  `semantic_state.world_snapshot_contracts` and its admission builder.
- Repository bytes/tree: Git plus `analysis.repository_snapshot`.
- AST, dependency, and semantic queries: existing analysis providers.
- Context selection and expansion: `ContextCompiler`.
- Test/proof/adversarial evidence: existing validation, verification, proof,
  and assurance producers.
- Worktree, lease, fence, merge queue, and merge-train mutation:
  existing merge and coordination packages.
- Artifact and event persistence: existing database artifact/event stores.
- Operational intent and task state: existing DuckDB control-plane schema and
  `DatabaseTaskSource`/`IntentRepository`.

### 4.2 DuckDB + Quack + DuckLake profile

The shared DuckDB control file is the authoritative transactional store for
goals, subgoals, tasks, dependencies, evidence references, events, status CAS,
and eventually current procedure-registry metadata. Quack is the mandatory
loopback multi-process transport to its one exclusive state owner. It does not
grant authorization or define schema. Quack failure is fail-closed; no worker
silently opens the shared control file directly.

The installed Quack transport authenticates a connection but does not restrict
the SQL that a token holder may submit. The token is therefore confined to the
trusted scheduler/control clients and an isolated owner process; it is never
passed to an implementation provider. Normal remote writes use a closed,
canonical request grammar, exact server-generation binding, request and result
authentication, replay/expiry bounds, and one owner-side transaction containing
the status CAS, task revision, completion evidence when applicable, and domain
event. A conflict is a typed result and the client never decomposes this bundle
into independently committed statements.

The network-facing Quack catalog is a bounded, non-authoritative read replica,
never the writer connection. Before readiness and after every admitted write
bundle, the owner checkpoints the authoritative file, copies it through a
bounded fsynced temporary file, atomically replaces the replica, opens the
replica `read_only`, disables external access, seals configuration, and proves
the source identity and requested effects before acknowledging the write. The
endpoint is unavailable during refresh. Raw authenticated SQL therefore cannot
mutate the control database, write or read host files, or issue network reads;
all authoritative writes continue through the authenticated owner grammar.

The Quack owner, Codex implementation providers, and every execution of
model-authored validation run behind externally enforced container boundaries.
The owner receives the repository read-only and exactly the dedicated
`state/agent_supervisor_proof_carrying_procedure_compiler/control/` directory
read-write. That directory alone contains `control.duckdb`, its DuckDB WAL,
the owner lease, status, mutation inbox, and handle-bound token. The owner also
receives the exact prequalified Quack and HTTPFS extension files read-only in a
private home; it receives no provider credential, host PID namespace, other
program state, or container-control socket. The store path is therefore
`state/agent_supervisor_proof_carrying_procedure_compiler/control/control.duckdb`
and the owner state directory is its sibling `quack-owner/`.

A provider receives only its isolated task worktree read-write, a private
temporary home, and a read-only provider credential; it receives no program
state, Quack token, host PID namespace, or container-control socket. Candidate
validation uses the same content-bound runtime with networking disabled, no
credential mount, a masked program-state subtree, and only its exact workspace
plus read-only linked Git metadata. There is no host validation fallback under
the sealed profile. All three roles use a read-only root, private PID/IPC,
dropped capabilities, `no-new-privileges`, bounded processes/memory/CPU, and
exact image/executable identity admission. The current runtime identity is
`sha256:ca52183d6e3f6d472b36092fc07a76fde0b7962da92b84dad2dc1038d93009ad`
(`2026-08-20-v3`, Linux arm64, Python 3.12.3, DuckDB 1.5.5, pytest 9.1.1,
Codex executable SHA-256
`7515d0b61e723374c68d4acdcb8815e378f84d088b0c50638f27d1094bffe536`).
The launch fails closed when any configured or live observation differs. An
in-container bypass of the provider's own sandbox does not expand the outer
container's filesystem or process authority.

The owner uses a receipt-bound Docker bridge, not rootless host networking.
Quack binds `0.0.0.0:45671` only inside that container; Docker publishes the
single TCP mapping `127.0.0.1:45671:45671`, and the advertised identity remains
the loopback endpoint. The isolation receipt and live container inspection bind
the bridge mode, both addresses, both ports, and the absence of any additional
publication. Rootless host-network mode is rejected because it is not reachable
from the host supervisor on the qualified runtime.

The existing database implementation daemon uses a different coordination
schema from the shared control-plane schema. Until a verified adapter unifies
those schemas, each strict home-shard lane owns private, one-writer DuckDB
execution and coordination sidecars for bounded attempt, claim, lease, fence,
and recovery state. The launcher binds those sidecars to the lane state path,
filters tasks before registration/claim, and uses the shared Quack status CAS
as a second race gate. Work stealing is disabled in this compatibility mode.
These sidecars never replace the shared goal/task/evidence authority and are
never opened by multiple processes.

DuckLake is a non-authoritative, append-oriented history projection for
bounded event, artifact, benchmark, calibration, and metrics rows. Its cursor
is owned by DuckDB. DuckLake loss or lag cannot establish or erase task state,
authority, evidence, completion, registry promotion, or rollback. Projection
replay is idempotent and bounded. The launch profile records DuckDB, Quack,
DuckLake, and extension fingerprints separately.

Markdown plan/objective/task documents are sealed bootstrap inputs and human
exports. After materialization, they are not task-status authority.

## 5. Closed artifact vocabulary

`procedure_compiler.contracts` supplies immutable, versioned, content-addressed
models for this complete closed vocabulary:

```text
RepositoryWorldState
AbstractRepositoryState
WorldStateDelta
TransitionObservation
TransitionModel
TransitionPrediction
PredictionCalibration
ExecutionTrajectory
TrajectoryStep
TrajectoryOutcome
TrajectoryNormalizationReceipt
TaskFamily
TaskFamilyMembership
TaskFamilyBoundary
TaskFamilyCounterexample
ProcedureSpec
ProcedureVersion
ProcedureParameter
ProcedureLocal
ProcedureStep
ProcedureBranch
ProcedureLoop
ProcedureHole
ProcedureEffect
ProcedureObservation
ProcedurePrecondition
ProcedureInvariant
ProcedurePostcondition
ProcedureRollback
ProcedureFallback
ProcedureResourceEnvelope
ProcedureAuthorityEnvelope
ProcedureValidationPlan
ProcedureCandidate
ProcedureSynthesisPlan
ProcedureSynthesisCounterexample
ProcedureVerificationResult
ProcedureCertificate
ProcedureInvocation
ProcedureInvocationReceipt
ProcedureExecutionTrace
ProcedureOutcome
ProcedureFailure
ProcedureRecoveryPlan
SpecificationCandidate
SpecificationEvidence
SpecificationCounterexample
SpecificationMiningReceipt
InvariantCandidate
InvariantValidationReceipt
NonVacuityReceipt
AntiUnificationPattern
GeneralizationBoundary
GeneralizationCounterexample
ProcedureRegistry
ProcedureRegistryRevision
ProcedurePromotionReceipt
ProcedureRollbackReceipt
ProcedureDeprecationReceipt
ProcedureDriftReport
HoleRequest
HoleCandidate
HoleResolution
HoleValidationReceipt
DistillationCorpus
DistillationExample
DistillationEvaluation
LocalDecisionModelArtifact
GeneratedToolSpec
GeneratedToolCandidate
GeneratedToolCertificate
GeneratedToolInvocationReceipt
ExperimentPlan
ExperimentObservation
ExperimentEvaluation
ProcedureCompilerRunReceipt
ProcedureCompilerReleaseReceipt
```

Shared bounds constrain text, references, arrays, maps, serialized bytes, and
nesting. Decoders reject unknown normative fields, floats/nonfinite numbers,
callbacks, executable objects, bytes, recursion, and unsafe paths. Large
payloads are retained only as checked content references.

Every execution-relevant artifact binds repository, tree, objective, task,
operation-contract revision, policy revision, verification-policy revision,
environment identity, and—where applicable—registry revision and certificate.
Lifecycle values use the closed states `candidate`, `verified`, `promoted`,
`stale`, `superseded`, `revoked`, and `rejected`, with additional closed
runtime outcomes where required.

## 6. ProcedureIR and operation catalog

ProcedureIR is a flat bounded control-flow graph rather than recursive code.
It contains identity/version, task family, parameters, locals, preconditions,
declared reads/effects, ordered steps, bounded branches/loops, typed holes,
invariants, postconditions, observations, validation, rollback, fallback,
authority/resource/token/time envelopes, scope, and provenance.

Each step binds an existing or separately reviewed operation contract, typed
inputs/outputs, declared effects, required authority, timeout, retry policy,
idempotency class, failure transition, and produced evidence. Inputs come only
from literals, parameters, locals, prior outputs, or verified content
references. There is no expression evaluation or string-to-command bridge.

The v1 operation vocabulary is exactly:

```text
READ_STATE
QUERY_AST_INDEX
QUERY_DEPENDENCY_GRAPH
QUERY_SEMANTIC_INDEX
QUERY_RECEIPT_CACHE
SELECT_EVIDENCE
EXPAND_CONTEXT_REFERENCE
CHECK_CAPABILITY
CHECK_POLICY
CHECK_AUTHORITY
CREATE_ISOLATED_WORKTREE
APPLY_APPROVED_PATCH_TEMPLATE
REQUEST_TYPED_MODEL_HOLE
RUN_STATIC_ANALYSIS
RUN_TYPE_CHECK
RUN_SELECTED_TESTS
RUN_FULL_TEST_FALLBACK
RUN_PROOF
RUN_ADVERSARIAL_ASSURANCE
CHECK_DIFF
CHECK_SCOPE
CHECK_POSTCONDITION
PREPARE_MERGE
MERGE_IN_ISOLATED_TRAIN
VERIFY_MERGED_TREE
PERSIST_ARTIFACT
EMIT_RECEIPT
ROLLBACK
ESCALATE
```

The following operation categories are unconditionally forbidden:

```text
ARBITRARY_SHELL
ARBITRARY_PYTHON
ARBITRARY_NETWORK_REQUEST
ARBITRARY_FILESYSTEM_PATH
DISABLE_VALIDATION
MODIFY_AUTHORITY_POLICY
MODIFY_TRUSTED_KEYS
CLAIM_COMPLETION
```

They are rejected even if an implementation adapter happens to be installed.
Trusted runtime adapters are held outside serialized IR and are selected by
exact catalog revision and contract ID.

Parser checks include uniqueness, initialized dataflow, reachability, required
ordering, branch convergence, validation retention, postcondition coverage,
scope/effect containment, maximum graph sizes, no undeclared cycles, no nested
loops or recursion in v1, and bounded retry/loop/time/token/resource values.

## 7. Deterministic interpreter

The interpreter executes this state machine:

1. Parse and structurally verify the procedure and catalog revision.
2. Obtain independent certificate admission; absence fails closed.
3. Match exact repository/tree/policy/environment/registry bindings.
4. Obtain independent precondition and task-family boundary observations.
5. Reserve resource and token envelopes.
6. Acquire existing task/resource/worktree leases and current fencing tokens.
7. Write a step-start checkpoint before any effect.
8. Invoke only the exact trusted adapter for the declared contract.
9. Independently admit observations and actual effects, enforcing that actual
   effects are subsets of step, procedure, scope, authority, and policy.
10. Persist the observed checkpoint, update deterministic bindings, and take
    only the declared branch/loop transition.
11. Verify invariants after each step and postconditions on the actual tree.
12. Execute declared compensation/rollback on qualifying failures and verify
    rollback externally.
13. Emit a bounded trace and outcome receipt without claiming task completion.

The idempotency key binds invocation, procedure/version, step/attempt,
resolved-input CID, repository/tree/policy, and fence. A started but unobserved
effect becomes `unknown_external_outcome`; it is never blindly repeated. A
restarted interpreter resumes only from admitted checkpoints and current
leases. Repeated identical failure without new evidence cannot trigger another
model call.

Full certificate synthesis/verification lands in PCPC-017. P0 therefore
defines a fail-closed certificate-admission port: effectful execution is
unavailable without externally admitted certificate evidence. Explicit test
and read-only shadow adapters remain labeled and cannot be admitted as live.

## 8. Repository world and transition models

The repository world model adapter, `RepositoryWorldModel`, builds the projection;
`RepositoryWorldState` is a bounded planning projection of existing admitted
world snapshots and repository observations—not a competing state authority.
It references repository/tree, changed paths/symbols, graphs, objective and
acceptance state, task dependencies, test/proof status, capabilities/provider
capacity, worktrees/leases, merge queue, caches, artifact pressure, budgets,
failure signatures, and registry revision.

`WorldStateDelta` compares two states deterministically. `TransitionModel`
contains immutable reviewed rules keyed by transition class and compatible
state fingerprints. Missing rules predict `unknown`.

Confidence is closed: `exact`, `conservative`, `empirical`, `heuristic`, or
`unknown`. Only exact predictions—or conservative predictions backed by a
separate admitted obligation—may discharge deterministic planning duties.
Empirical/heuristic values may affect cost and priority only.

Calibration compares predicted and observed changed files/symbols/effects,
tests, proofs, duration, tokens, provider cost, merge conflicts, and terminal
status using integer ratios. Drift beyond policy demotes or invalidates the
rule and creates a typed drift report.

## 9. Validated trajectory ingestion

PCPC-009 accepts only independently validated, current episodes: admitted task
and post-merge receipts, verified proof/test receipts, successful rollback,
authorized human decisions, typed rejections, and failed-then-recovered
records. Model confidence, board status, prose, unsigned/stale receipts,
simulated production, and pre-merge-only validation cannot be positive
demonstrations.

Normalization retains abstract initial/terminal states, accepted criteria,
operation contracts/order, observations, holes/model calls, effects,
validation, cost/tokens/latency, and human intervention. It removes prompts,
chain-of-thought, secrets, credentials, redundant bodies, and unbounded logs.

## 10. Task-family discovery and boundaries

The deterministic baseline uses goal semantics, precondition shape, artifact
and effect classes, required tools, validation, failure signatures,
postconditions, and rollback—not titles or embeddings alone. Initial families
are the closed set listed in the program prompt.

Every family records positives, negatives, boundary cases, unknowns, risk
ceiling, repositories, languages/frameworks, effects, authority, validation,
rollback, and proof obligations. A family is rejected when any merge would
cross materially different authority, effects, validation, rollback, legal or
security treatment, ownership, or proof obligations. Overgeneralization is a
critical safety failure.

## 11. Anti-unification, specifications, and CEGIS

Structural anti-unification preserves semantic ordering, authority/effect
classes, every validation and postcondition, failure transitions, and lost
detail. Differences become parameters, bounded optional branches, or typed
holes. Paths, credentials, omitted tests, missing postconditions, and uncertain
authority are never generalized.

Specification mining draws candidates from types, operation contracts, tests,
proof obligations, runtime checks, admitted/rejected traces, failure
signatures, mutants, and authoritative documentation. All mined properties
remain candidate-tier until independent validation. Non-vacuity campaigns
attack impossible preconditions, unreachable branches, empty output domains,
non-invoking tests, mock substitutions, fixture shortcuts, and constant
restatements through the existing adversarial assurance API.

The bounded CEGIS order is: existing verified procedure, built-in template,
anti-unified pattern, enumerative/constraint synthesis over ProcedureIR,
model-proposed declarative sketch, then human candidate. Every synthesis plan
fixes candidate/step/branch/hole/loop/model-call/token/validation/proof/wall
bounds. Counterexamples are immutable and deduplicated by candidate and set
identity. Exhaustion returns a typed incomplete result.

## 12. Verification, certificates, and registry

PCPC-017 checks structural, authority, effect, dataflow, temporal, semantic,
and validation obligations. Every `ProcedureCertificate` binds all of:

```text
procedure CID
procedure version
task-family CID
source episode CIDs
specification CIDs
counterexample-set CID
operation-catalog revision
effect-policy revision
authority-policy revision
verification-policy revision
repository families
supported language and framework classes
risk ceiling
proof and test receipts
adversarial-assurance results
held-out evaluation
shadow evaluation
known limitations
issuer
signature
expiry or review horizon
```

Certificate verification is independent of procedure content. Identity alone
does not establish authority or usability.

The versioned registry supports exact/family lookup, capability/risk/
environment filters, version choice, rollback, revocation, drift demotion, and
the closed states `candidate`, `development`, `shadow`, `promoted`, `degraded`,
`stale`, `revoked`, `superseded`, and `rejected`. Promotion uses authorized
expected-old CAS and an exact rollback target. Contract, schema, authority,
effect, dependency, evidence, environment, or boundary drift demotes or
revokes automatically.

## 13. Planner integration and composition

After the AdaptivePlanner compatibility blocker is resolved, planner order is:

```text
exact verified procedure
-> composable verified procedures
-> deterministic baseline
-> bounded local synthesis
-> small local model
-> standard remote model
-> strong remote model
-> human escalation
```

Composition requires exact post(A)-to-pre(B) entailment evidence, compatible
effects/authority/environment, additive bounded resources, a defined composed
rollback, and complete validation. Cycles and hidden effect escalation are
rejected. Procedures may satisfy a task, criterion, subgoal, repair suffix, or
validation stage without claiming more.

## 14. Typed holes and distillation

Holes use only the approved closed types and declare schemas, provider classes,
context budget, authority/effect class, validation, fallback, and attempts.
Hole outputs are candidates until independently validated. Authority, policy,
confirmation, key choice, test/proof omission or acceptance, promotion,
completion, and unbounded commands cannot be holes.

Validated hole resolutions feed the hierarchy exact cache -> declarative rule
-> deterministic classifier -> small local model -> remote model. Corpus rows
bind typed features, content references, candidate, validation/proof outcome,
counterexamples, repository family, and language/framework. Local models remain
proposal producers.

## 15. Deterministic tool synthesis and experiments

Repeated pure/bounded transformations may become interpreted DSL tools from a
reviewed grammar/template library. Generated tools require closed schemas,
effects, path limits, resources, tests, translation validation, adversarial
fixtures, and certificate. Optimized Python is candidate-only until
differentially equivalent; arbitrary scripts are forbidden.

Shadow experiments declare question, hypothesis, counterfactual, required data,
risk/privacy/cost, decision rule, and execution bound. They run only in
disposable worktrees or fixtures, never mutate production or policy, and are
skipped when they cannot change a decision.

## 16. Transfer, repair, recovery, and review

Cross-repository transfer requires explicit compatibility of operation,
effects, authority, language/framework, validation, family boundary, and path
assumptions plus held-out repositories. Similar names, embeddings, descriptions,
language, or maintainer do not establish portability.

Autonomous repair remains inside the existing isolated repair path. Promoted
procedures may identify scoped files/symbols, apply approved templates, request
typed patch holes, validate, perform bounded retry, and produce a patch/PR.
They cannot edit outside scope, weaken/remove tests, skip proofs, alter trusted
policy without high-risk review, self-complete, or high-risk merge.

Recovery order is existing diagnostic, policy-authorized retry, deterministic
evidence, named context expansion, typed hole, stronger/composed procedure,
affected-suffix replan, escalation, then quarantine. Human review receives a
compact packet containing the candidate, boundary, preconditions/effects,
counterexamples, held-out results, complete cost comparison, limitations,
disposition, and rollback—not raw trajectories by default.

## 17. Control surfaces and memory

PCPC-028 adds these exact read operations to the existing typed
`SupervisorControlService`:

```text
procedures.capabilities
procedures.list
procedures.get
procedures.explain
procedures.match
procedures.registry_status
procedures.task_families
procedures.counterexamples
procedures.drift
procedures.metrics
procedures.shadow_results
procedures.synthesis_status
procedures.world_model_status
```

It adds these exact mutation operations:

```text
procedures.synthesize
procedures.evaluate
procedures.promote
procedures.rollback
procedures.revoke
procedures.quarantine
procedures.run_shadow
procedures.cancel
procedures.request_review
```

The same typed operations are exposed through direct Python,
`ipfs-accelerate agent procedures ...`, and canonical MCP adapters. MCP calls
the service directly. Every mutation requires authorization, idempotency,
exact targets, dry-run, lease/fence as applicable, and an audit receipt.

Semantic memory stores promoted procedures, boundaries, counterexamples,
candidate specifications, failure signatures, calibration, distilled
resolvers, and promotion/rollback history by content reference. It excludes
chain-of-thought, model transcripts, source bodies, duplicated contexts,
expired policy, and credentials. Compaction factors shared subprocedures only
after differential execution proves exact preservation of effects, authority,
validation, and acyclicity.

## 18. Metrics and promotion gates

Metrics preserve denominators and include synthesis, failed matches, shadow
evaluation, hole filling, validation, rollback, and human review. The complete
metric vocabulary is:

```text
procedure coverage by task family
procedure match precision
procedure match recall
unsafe generalization count
remote-model calls avoided
large-model calls avoided
tokens avoided
planning tokens avoided
hole-filling tokens
validation cost
procedure synthesis cost
amortization break-even
human interventions avoided
procedure failure rate
procedure rollback rate
procedure drift rate
world-model prediction accuracy
post-merge regression rate
```

Safety gates require exactly zero unauthorized effects, path/scope escapes,
hidden validation reductions, simulated-as-live results, stale procedure
executions, stale proof reuse, procedure self-promotion, authority escalation,
confirmation replay, high-risk autonomous merge, and escaped critical seeded
defects.

Correctness gates require 100% required-postcondition coverage on admitted
benchmark tasks, 100% validation retention, 100% correct rejection of known
boundary counterexamples, no increase in post-merge regression, and no lower
proof or test coverage.

Against the qualified autonomous-meta-controller baseline, token-efficiency
gates require at least 50% lower median planning tokens, 40% lower total model
input tokens, 60% fewer remote-model calls, and 70% lower retry tokens on their
declared eligible/covered populations. Autonomy gates require at least 60% of
eligible recurring tasks without a remote model, 80% of deterministic
repair-family tasks without any model, 30% of accepted benchmark work through verified
procedures, and 25% fewer human interventions. Transfer requires zero unsafe
cross-repository transfer, explicit held-out results, and typed refusal when
assumptions differ. Safety and correctness floors are non-compensable. The
missing qualified autonomous-meta-controller baseline is reported as a typed
comparison blocker; it is never replaced with an unqualified estimate.

Every promoted procedure reports qualification cost, per-use savings,
observed use count, break-even count, and whether break-even is observed.
No one-demonstration token reduction can promote.

## 19. Frozen benchmark and testing

The P0 benchmark manifest freezes the requested family and partition vocabulary
but is explicitly `scaffold_only`: its case counts are zero and it cannot
support benchmark, transfer, savings, or promotion claims. PCPC-029 must give
every family disjoint synthesis/training, development, held-out, negative,
boundary, and adversarial cases before the corpus becomes qualified. Git stores
compact manifests and content references, not large trajectories or AST dumps.

Tests live under `test/api/procedure_compiler/`. Default tests are hermetic,
provider-free, network-free, clock-injected, and use disposable repositories,
worktrees, and databases. Live-provider tests are explicitly marked and never
required for import or collection. Each task runs its focused declared test;
each tranche runs the package directory and affected existing authority tests.

The required matrix includes contract round trips/canonical identity/unknown
fields; parser and interpreter determinism; effect, authority, confirmation,
loop/recursion, checkpoint/restart, rollback, concurrent invocation, and
idempotent replay; task-family boundaries and anti-unification; vacuity,
invariants, CEGIS convergence/exhaustion, and replay; stale procedure and
registry rollback/corruption; contract and transition drift/calibration; hole
bounds/validation/distillation; generated-tool translation safety; injection,
forged certificate, and self-promotion; transfer denial; fencing and unknown
external outcomes; no-op/test deletion/validation weakening/scope/symlink/
submodule escape; idle stability; and large-artifact rejection.

## 20. Execution DAG and tranches

The executable board carries exact dependencies, goals, predicted files and
symbols, effects, resources, token budgets, validation, proofs, conflict
policies, and acceptance. Its high-level critical path is:

```text
PCPC-000 -> 001 -> 002 -> {003,006}
003 -> 004; 004 + 006 -> 005
006 -> 007 -> 008
{002,003,008} -> {009,011,013} -> remaining mining/synthesis tasks 010..017
017 -> registry/planner/hole/tool/transfer tasks 018..027
{017,018,027,029} -> surfaces/gates/release 028..031
```

- Tranche 1 / `PCPC-G010`: PCPC-000..008, contracts and deterministic runtime.
- Tranche 2 / `PCPC-G020`: PCPC-009..017, mining, synthesis, and certificates.
- Tranche 3 / `PCPC-G030`: PCPC-018..027, planning and execution integration.
- Tranche 4 / `PCPC-G040`: PCPC-028..031, product surface and qualification.

Broad autonomous repair cannot begin before interpreter/effect checks, family
boundaries, and certificate verification are admitted. Parallel workers use
the strict home shard `int(sha256(task_id)[:8], 16) mod 4`, and every board
`Parallel lane` value must match it. An `Allow concurrent with` declaration may
never name a direct or transitive dependency in either direction. File/symbol
ownership remains declared on each task; shared contract and public-surface
changes serialize. Conflicts are classified and merged in isolated trains.

The executable bootstrap is deliberately ordered: exact-tree qualification;
single-writer DuckDB materialization; bounded DuckLake projection; Docker owner
creation; actual-container-ID isolation receipt; owner start; authenticated
handle-only read query and owner-bundle CAS probe; then the detached configured
board scheduler with `--implement`. The program-specific launcher refuses an
unknown existing container, a dirty or drifted tree, a missing materialization,
an unqualified extension/image, a host-mode owner, or a scheduler start before
remote readiness. Status admission combines Docker process identity, the
canonical owner status, authenticated database queries, coordinator/lane PID
birth identities, and task-attempt/event evidence; a status JSON file alone is
never liveness or completion evidence.

## 21. Completion evidence and release report

Markdown/database status is coordination state only. Task completion requires
fresh evidence from every declared producer bound to the actual post-merge
tree. P0 completion is admitted only after contract/parser/interpreter/world/
transition tests, cold-import checks, board validation, exact source identity,
and current control-plane materialization receipts pass. Stored command-shaped
JSON and its CID remain provenance only: launch verification reruns the exact
qualification producers on the unchanged tree instead of admitting that file
as evidence. The final release task requires post-merge requalification and an
authorized release decision.

The final machine/human report records starting/ending commits and trees, files
and symbols, contracts, synthesized/promoted/rejected procedures, boundaries,
counterexamples, tests/proofs run and omitted, model/token/cost/intervention
metrics, unsafe matches and transfer results, safety gates, blockers,
promotion eligibility, and rollback target. Terms such as “production ready”
are prohibited without corresponding exact evidence.

## 22. Rollback and current limitations

The initial rollback target is the exact starting commit above. Runtime state
has separate fenced shutdown/checkpoint and DuckLake projection replay paths.
Source rollback uses a reviewed Git operation and never rewrites unrelated
operator work.

Current known limitations are explicit:

- Adaptive planner import is incompatible on the clean committed tree.
- The autonomous-meta-controller/cognitive/experience/policy-distillation
  authorities and qualified comparison baseline are absent.
- Default ordered provider-route integration has stale failing tests; launch
  uses the existing direct Codex route (whose current daemon defaults are
  gpt-5.6-terra with medium reasoning) and records observed provider results.
- Several lease/worktree concurrency improvements exist only as unrelated
  uncommitted overlays and are not inherited.
- Incremental sealer imports require a narrow facade adapter and pinned sibling
  packages for full qualification.
- The shared control schema and database coordinator schema are incompatible;
  P0 therefore uses strict, disjoint, one-writer lane sidecars and disables
  work stealing. A schema-compatible Quack coordination adapter remains a
  typed integration task before dynamic cross-lane transfer.
- Shared Quack task-status CAS and lane-local attempt insertion cannot be one
  transaction across the two stores. Normal conflicts and portal failures
  have exact release/requeue reconciliation, but a process death in that
  cross-store window still relies on the existing stale-claim reconciler and
  must be exercised during the later recovery qualification.
- Quack 1.5.5 does not provide a server-side SQL allowlist, and DuckDB cannot
  hold writer and read-only handles to the same persistent file concurrently.
  P0 therefore serves only the synchronously refreshed read-only replica
  described above and keeps external access disabled after the pinned
  extensions load. Live launch remains forbidden unless the owner, provider,
  and candidate-validation boundaries are admitted exactly; direct serving of
  the writer and host-mode Quack service are not allowed fallbacks.
- P0 dispatch callbacks are trusted adapters. The interpreter enforces their
  declared budget before dispatch and checks observed elapsed time, but a
  callback that never returns cannot be preempted safely in-process. Live
  operation adapters must therefore enforce their own subprocess/service
  timeout; this must be independently qualified before promotion.
- Repository-relative scope checks do not themselves resolve live symlinks or
  submodule ownership. Effectful operation adapters and the isolated-worktree
  admission path remain responsible for canonical filesystem resolution and
  must produce the corresponding scope evidence.
- DuckLake is historical projection only and never operational authority.

These limitations block only dependent promotion or integration work. They do
not justify recreating authorities, lowering assurance, or stopping independent
tasks.
