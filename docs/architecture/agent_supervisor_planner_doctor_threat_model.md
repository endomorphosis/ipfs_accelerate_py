# Planner/Doctor authority and threat model

Interface: `PlannerDoctorThreatModel@1`
Policy interface: `PlannerDoctorAuthorityPolicy@1`
Program: `agent-supervisor-proof-directed-planner-doctor-v1`
Task: `PDR-002`
Status: externally sealed by
`config/agent_supervisor_planner_doctor_authority_policy.seal.json`; runtime
activation requires fresh verification

## 1. Purpose and status

This document defines the non-compensable trust boundary for the
proof-directed Planner, deterministic Doctor, optional hybrid repair path,
unattended benchmark controller, and derived goal/task refill loop. Its
machine-readable companion is
`config/agent_supervisor_planner_doctor_authority_policy.json`.

The policy is deliberately fail-closed:

- the Planner starts in `shadow`;
- the deterministic Doctor starts in `report_only`;
- mutation, automatic promotion, and derived refill start disabled;
- unknown, ambiguous, stale, incomplete, unavailable, or conflicting required
  evidence causes rejection or typed abstention, never a pass; and
- only a human operator can manually seal this policy or grant a scoped
  mutation capability.

The policy body cannot seal itself. PDR-002 has manual completion. The separate
operator receipt binds the exact final authority-policy, threat-model, and
authority-test SHA-256 roots; policy revision; reviewed base commit/tree; and
operator identity and authority basis. The operator-approved receipt identity
is independently pinned in the operator-owned, protected scheduler
configuration; the receipt's self-hash is integrity evidence, not its own
authorization. Neither candidate work nor the self-improvement controller may
change the scheduler pin, receipt, or a bound artifact. A policy change
requires a new revision, roots, receipt, scheduler pin, re-baseline, and manual
review.

Taskboard status is not a seal. The only accepted basis for this receipt is
`interactive_user_delegation`; an execution permit cannot issue it. The
receipt grants exactly `activate_policy_revision`. It grants no mutation,
completion, promotion, task-status, or protected-anchor-write authority. Until
a long-running supervisor freshly loads and verifies both policy and receipt,
it must retain the pre-seal shadow/report-only state.

This policy composes existing contracts. It does not create another assurance
lattice, control permit, proof cache, transaction authority, or completion
authority. If this document conflicts with a stricter shipped control, proof,
Doctor, security, or completion contract, the stricter shipped contract wins.

## 2. Scope and security objectives

### In scope

- prompt-to-control-plane planning and control-plane mutation;
- repository, recursive gitlink, dirty-overlay, task-source, policy, IR,
  capability, toolchain, provider, and benchmark identity;
- AST/static/dynamic analysis, knowledge-graph, BM25/vector/GraphRAG, theorem
  prover, solver, proof-cache, SecurityIR, IntentIR, and ZKP evidence;
- proposal, repair, validation, merge, completion, promotion, rollback, and
  refill;
- parallel worktrees, leases, fencing epochs, merge trains, and fixed-point
  evaluation;
- provider/network/credential/private-witness boundaries; and
- unattended benchmarking and self-improvement.

### Security objectives

1. No read, retrieval, model, candidate, test, cache, solver, proof, or
   attestation result silently acquires write or completion authority.
2. Every authoritative claim remains bound to its exact statement, scope,
   assumptions, bounds, roots, toolchain, and independent verifier.
3. Every mutation is explicitly permitted, bounded, fenced, transactional,
   independently validated, reversible, and revalidated against the current
   tree immediately before dispatch.
4. Intent and generated code both fail closed against current IntentIR and
   SecurityIR. Unknown security is not safe.
5. Candidate work cannot change its task source, judge, hidden oracle,
   denominator, safety floor, promotion policy, or evidence requirements.
6. Parallel execution cannot turn stale authority, partial effects, missing
   consumers, or a false fixed point into success.
7. Secrets and private witnesses never enter prompts, taskboards, command
   lines, public CIDs, cache keys, logs, or receipts.
8. Completion derives from fresh acceptance evidence under the existing goal
   completion contract. Completed task counts are neither proof nor authority.

### Non-goals

This threat model does not claim that:

- an index is complete merely because it was built successfully;
- a parser or IR translator is sound merely because output was produced;
- a solver candidate is a kernel proof;
- an ordinary or generated test proves arbitrary program semantics;
- a CID proves semantic correctness;
- a ZKP proves any property not encoded in its verified program;
- a model is trustworthy because it supplied a plausible explanation; or
- this policy alone enables unattended repository mutation.

## 3. Protected assets and anchors

The following asset classes require protection:

| Asset | Required property |
| --- | --- |
| Repository forest | Exact superproject commit/tree, recursive gitlinks, and dirty overlay are current and non-forgeable. |
| Task source | Seed plan, objectives, taskboard, and scheduler remain operator-owned; derived refill uses a separate runtime source. |
| Authority policy and external seal | Exact canonical policy/receipt bytes, revision, reviewed Git base, three artifact roots, operator basis, and activation-only grant are immutable after manual seal. |
| IR and capability roots | IntentIR, SecurityIR, provider policy, capability catalog, parsers, indexes, translators, and toolchains are bound to each decision. |
| Proof and validation receipts | Preimages, references, freshness, verifier identity, bounds, and assurance reconstruct under current roots. |
| Mutation state | Permit, expected effects, idempotency key, lease, fence, checkpoint, transaction phase, and rollback lineage remain consistent. |
| Benchmark judge | Population, paired denominators, holdout manifest, hidden oracle, non-inferiority margins, safety floors, and promotion policy cannot be changed by a candidate. |
| Secrets/private witnesses | Bodies stay inside their owning backend or independent evaluator and are not exposed through control-plane artifacts. |

The concrete protected paths and logical fields are closed in the policy JSON.
The in-repository holdout root is protected as a whole, but path protection is
not secrecy. It may contain only opaque handles, root CIDs, denominators, and
population metadata. Hidden bodies live in an independent evaluator-owned
read-only repository or CAS outside candidate worktrees.

## 4. Actors, assumptions, and trust boundaries

### Actors

- **Operator:** owns policy sealing, protected-anchor changes, scope grants,
  exceptional approvals, promotion, and kill-switch clearing. The PDR-002 seal
  is issued under the interactive user's explicit delegation.
- **Control service:** validates typed requests and exact authorization
  bindings immediately before dispatch.
- **Planner/Doctor kernel:** builds or diagnoses obligation graphs and emits
  read-only proposals by default.
- **Candidate agent/model:** may nominate plans, premises, tests, or edits. It
  is untrusted for admission, validation, mutation, completion, and promotion.
- **Analysis/retrieval provider:** may produce structural observations or
  nominations within recorded capabilities. Provider availability is not
  evidence of correctness.
- **Solver/prover:** may produce a bounded result or proof candidate.
- **Independent kernel/attestation verifier:** checks exact proof objects or
  attestations against committed inputs and pinned verification code.
- **Validation runner:** independently rebuilds, analyzes, tests, checks IR,
  and evaluates fixed point in an isolated environment.
- **Benchmark evaluator:** owns holdout/oracle access and emits body-free
  results only after candidate submission.
- **Unattended controller:** may schedule shadow evaluation within policy. It
  does not own the policy, judge, mutation permit, completion, or promotion.

### Trust boundaries

1. **Prompt → normalized request.** Prompt text is untrusted input. Fixed rules
   derive mandatory analyses, scope, non-goals, budgets, and required roots.
   Model suggestions cannot suppress those rules or select credentials or
   endpoints.
2. **Index/retrieval → evidence view.** BM25, vectors, embeddings, GraphRAG,
   history, and graph neighborhoods nominate relevant records. Exact facts
   require root-bound independent reconstruction.
3. **Candidate → admission.** A candidate plan, patch, proof flag, serialized
   receipt, generated test, or expected benchmark value cannot validate itself.
4. **Solver → kernel.** SAT/SMT/ATP output is solver-checked at most until an
   independent native kernel reconstructs the exact theorem and premises.
5. **Cache → consumer.** A hit is memoization, not trust. The consumer
   recomputes the preimage/CID, follows source receipts, checks current roots,
   and re-derives assurance.
6. **Preview → apply.** Preview is read/proposal authority only. Apply is a new
   request with a fresh mutation decision, scoped permit, expected effects,
   lease, fence, checkpoint, and rollback.
7. **Candidate worktree → protected tree.** Candidate output remains isolated
   until independent admission and transactional merge. Worktree presence or a
   green local check does not imply merge.
8. **Candidate → benchmark judge.** Candidates cannot read hidden oracle
   bodies/results before submission and cannot write judge anchors. The
   evaluator mounts hidden bodies read-only only after candidate output is
   finalized; they are otherwise unmounted and denied at the OS boundary.
9. **Evidence → completion/promotion.** Existing completion and promotion gates
   independently replay current evidence. Task status and aggregate scores do
   not bypass hard gates.

## 5. Closed authority ladder

The ladder distinguishes evidence quality from operation authority. Even the
highest semantic evidence does not itself grant mutation or completion.

| Stage | Maximum assurance | Permitted use | Explicitly not permitted |
| --- | --- | --- | --- |
| Nomination | `candidate` | Discover, retrieve, rank, suggest candidates or premises | Admit, prove, write, promote, or complete |
| Observation | `candidate` | Record a root-bound static fact, runtime trace, test result, or counterexample candidate | Claim unbounded semantics, kernel proof, write, or completion |
| Bounded check | `solver_checked` | Establish or refute a property under exact recorded bounds and assumptions | Claim outside bounds, claim kernel verification, or authorize a write alone |
| Kernel proof | `kernel_verified` | Satisfy the exact reconstructed theorem under exact premises and current roots | Prove an unencoded property, bypass a permit, or complete a goal alone |
| Attestation | `attested` | Verify committed bounded computation, membership, private-witness possession, receipt lineage, or counter aggregation | Expand the underlying semantic claim, prove translator soundness/inventory completeness, write, or complete |

`OperationAuthority` remains the closed control vocabulary:
`read`, `proposal`, and `mutation`. A proof-assurance rank is never converted
into an operation authority by comparison.

## 6. Threats and required mitigations

The transition IDs below match the machine policy.

| Threat | Attack | Required defense and disposition |
| --- | --- | --- |
| T01 Self-certification | Candidate code supplies its own test, proof flag, expected value, or receipt and claims authority. | `candidate_or_self_certification_to_authority` is rejected. Independent reconstruction and validation are required. |
| T02 Retrieval authority laundering | A high similarity score, graph path, learned rank, or provider response is treated as truth. | `provider_or_retrieval_result_to_semantic_authority` is rejected; retrieval remains nomination-only. |
| T03 Model write | Prompt/model output directly edits source, state, task source, or judge. | `model_output_to_write` is rejected. A proposal overlay must pass every independent gate and a separate operator-authorized apply. |
| T04 Stale replay/TOCTOU | A valid old receipt, permit, lease, proof, or plan is replayed after tree/policy/IR drift. | Revalidate all authority roots at each listed boundary; `stale_replay_to_current_evidence` is rejected. |
| T05 Cache trust upgrade | Cached metadata, a matching key, or a CID upgrades assurance. | Recompute canonical preimage, traverse source receipt, verify freshness, and re-derive assurance; `cache_hit_to_assurance_upgrade` is rejected. |
| T06 Unproved security | Missing, unknown, conflicting, stale, or detached IntentIR/SecurityIR evidence is treated as safe. | Check intent and generated code at every security stage; `unproved_unknown_or_conflicting_security_to_admission` is rejected. |
| T07 Partial transaction | Some edits/effects land and the transaction reports success. | Expected-effect equality, atomic SCC transaction, CAS ref update, compensation/repair, and quarantine; `partial_transaction_to_commit` is forbidden. |
| T08 False fixed point | A supplied boolean/mapping, unchanged loop, missed consumer, or stale re-index reports convergence. | Independently rebuild/re-index, invalidate dependent caches, close all consumers/SCCs, rerun proof/security/validation, and reject `prebuilt_or_boolean_fixed_point_to_completion`. |
| T09 Judge/anchor mutation or exposure | Candidate changes policy, seed tasks, benchmark population, denominator, holdout, oracle, floor, margin, or promotion policy, or reads hidden bodies from its repository/provider context. | Metadata-only in-repository manifests, external evaluator-owned body storage, two-phase read-only mount, OS denial, protected anchors, and independent evaluator; `candidate_work_to_protected_anchor_mutation` and `benchmark_or_oracle_mutation_to_valid_evaluation` are quarantined and escalated. |
| T10 False completion | Task count/status, aggregate score, or candidate claim completes a goal. | Existing completion contract replays fresh criterion evidence and child coverage; `task_count_or_task_status_to_completion` is rejected. |
| T11 Synthetic promotion | Synthetic fixtures or skipped checks are presented as live evidence. | Paired live receipts and independent holdout are mandatory; `synthetic_or_skipped_evidence_to_live_promotion` is rejected. |
| T12 ZKP overclaim | Simulated ZK or a valid proof of a narrow circuit is presented as arbitrary semantic correctness. | Simulated ZK never emits `attested`; `verified_attestation_to_unencoded_semantic_claim` and `simulated_zk_to_attested` are rejected. |
| T13 Capability/scope escape | A permit is widened, delegated, reused cross-repository, or applied to undeclared paths/effects. | Exact operation/path/effect/bounds/expiry/root bindings, no delegation, path containment, lease/fence, and immediate live authorization. |
| T14 Secret/provider exfiltration | Credentials or private witnesses enter prompts, argv, logs, taskboards, receipts, public cache keys, or CIDs. | Backend/environment-only credentials, body-free control artifacts, network deny by default, provider policy binding, and a zero secret-exposure floor. |
| T15 Parallel race | Two lanes use the same stale fence, collide on an SCC, validate different trees, or overstate parallel completion. | Conflict/resource compilation, isolated worktrees, single writer lease, fencing epoch, merge-train serialization, and merged-tree revalidation. |
| T16 Kill/rollback failure | Work continues after a hard-gate breach, or rollback silently fails. | Kill switch blocks new dispatch/promotion/refill, active transactions recover or quarantine, and only an operator may clear after audit. |
| T17 Toolchain/submodule substitution | Parser, verifier, dependency, recursive gitlink, or provider changes without invalidating evidence. | Bind recursive forest and toolchain/provider roots; drift causes abstention and replan. |
| T18 Resource exhaustion | Candidate forces unbounded analysis, proof search, model use, or iterations and degrades the control plane. | Explicit item/byte/time/process/memory/provider bounds; exhaustion aborts, rolls back if needed, and escalates. |

Every listed safety event has an exact integer floor of zero. Speed, token
savings, cost, throughput, or a favorable aggregate score cannot compensate
for a nonzero floor.

## 7. Preview and mutation protocol

The only legal path to a repository or authoritative-state mutation is:

```text
untrusted request
  -> normalize scope, roots, non-goals, and bounds
  -> build current evidence coverage
  -> emit read-only preview/proposal
  -> independent plan/proof/security/impact admission
  -> explicit operator scope decision
  -> new apply request
  -> fresh permit + authorization decision
  -> validate exact expected effects + lease + fence + roots
  -> checkpoint isolated candidate worktree
  -> PREPARED -> DISPATCHING
  -> independently rebuild/re-index/reprove/revalidate
  -> current merged-tree fixed-point check
  -> COMMITTED
     or COMPENSATION_REQUIRED / REPAIR_REQUIRED -> recovered
     or quarantine
```

Preview may only observe or propose. It may not write repository bytes,
authoritative supervisor state, lifecycle state, the seed task source, or the
judge. A preview artifact is not an apply permit. Apply always uses a separate
request and fresh bindings.

An apply permit is exact and non-delegable. It binds caller, operation,
repository/state roots, repository/tree/objective/policy revisions, allowed
paths, expected effect IDs, resource bounds, idempotency key, lease, fencing
epoch, grants, and expiry. Missing or extra effects fail. A path or
cross-repository escape fails. Authority that was not explicitly granted is
not inferred.

Commit requires a real byte change, complete impact closure, one disposition
for every resolved consumer, an atomic SCC transaction, a compare-and-swap ref
update, current post-edit roots, and a live logic/program fixed point. Partial
success is never completion. If compensation or repair cannot be independently
verified, the result is quarantined and manually escalated.

## 8. Proof, security, cache, and ZKP boundary

### Proof and refutation

- Solver/ATP/SMT output is at most `solver_checked`.
- Authoritative proof requires the exact theorem, premises, assumptions,
  current roots, native proof object, pinned toolchain, and independent kernel
  reconstruction.
- A counterexample becomes a scoped refutation only after independent
  reproduction under the recorded bounds.
- Tests and runtime observations are valuable independent evidence but do not
  substitute for a required proof.
- Provider flags or prebuilt mappings cannot manufacture reconstruction.

### Security admission

Intent and generated code are checked against current IntentIR and SecurityIR
at plan admission, pre-execution, post-generation, merge admission, and
merged-tree revalidation. Forbidden logic, a deny, a conflict, unknown
security, a missing required gate, stale evidence, or a detached receipt
rejects admission. For an authoritative security theorem, minimum assurance is
`kernel_verified`; an attestation may bind that theorem but cannot broaden it.

### Content-addressed caches

Cache identity includes the operation/property, repository forest, exact
scope, premises/assumptions, parser/index/translator/toolchain/provider
capability, policy/IR/catalog roots, required assurance, and bounds. A hit:

1. reloads its source receipt;
2. recomputes the canonical preimage and content identity;
3. validates reference lineage and current roots;
4. reruns any required native reconstruction or validation; and
5. emits the same or lower re-derived assurance.

A cache miss is neither a refutation nor a pass. Secrets and private witnesses
are excluded from public keys and receipts.

### ZKP

After a separate approved ZKP threat model, genuine ZK may attest only a fixed
bounded verification program over committed inputs, policy membership,
receipt lineage, private-witness possession, or committed counter
aggregation. It does not establish:

- completeness of repository/index/query coverage;
- correctness of an AST or IR translator;
- arbitrary Python or cross-language semantics;
- a theorem absent from the verification program; or
- goal completion.

Simulated ZK is test evidence and never production `attested` evidence.

## 9. Deterministic Doctor and hybrid model boundary

The deterministic Doctor inherits the shipped hard-false boundary:

- no LLM router or LLM invocation;
- no remote model-provider call;
- no remote embedding;
- no network access;
- no target-code import as an analysis shortcut; and
- no semantic authority for knowledge graphs, vectors, embeddings, candidates,
  or proof-cache metadata.

Provider absence produces a typed abstention and must not trigger a model
fallback. Report-only startup remains available without optional providers.

A hybrid model path, if later approved, is a separately named system. It may
request bounded residual syntax only after deterministic analysis, and its
output remains nomination authority. It cannot suppress mandatory queries,
choose provider endpoints or credentials, write directly, approve itself,
claim completion, or inherit deterministic-Doctor authority. Its proposal
overlay must be analyzed exactly like any other untrusted candidate.

## 10. Completion, promotion, and refill

All tasks being marked complete is necessary operational bookkeeping at most;
it is not sufficient evidence that a goal is complete. The existing goal
completion contract remains authoritative and must independently verify:

- every acceptance criterion has fresh, non-contradictory evidence;
- each required proof assurance is satisfied;
- evidence binds the current repository/objective/policy roots;
- child-goal coverage is complete;
- no required uncertainty, contradiction, or impact frontier remains open;
- the terminal receipt replays from canonical evidence; and
- every safety floor remains exactly zero.

Promotion begins in `shadow`, is manual and monotonic, and requires paired live
current-tree evidence, an independent protected holdout, non-inferior solution
quality, and all non-compensable gates. Synthetic data and skipped live checks
may measure plumbing but may not promote.

Content-addressed development/holdout partitions prove frozen membership and
exact-CID disjointness, not hidden independence. The benchmark must additionally
reject overlap by normalized AST identity, contract identity, and provenance
family, plus symlink/hardlink escape. Provider context and transcripts must
contain neither hidden bodies nor gold answers. Only the independent oracle
runner receives a read-only mount after candidate output is final.

Derived refill remains disabled at bootstrap. When a later task enables it,
new goals/tasks are written to a separate derived runtime source. Refill may
not mutate the seed plan, objectives, board, scheduler, authority policy,
benchmark anchors, or promotion policy. New work also cannot manufacture
completion of its parent.

## 11. Kill switch, rollback, and manual escalation

The kill switch can be engaged by an operator or automatically after a
non-compensable gate failure. While engaged it:

- forces `report_only`;
- blocks new mutation dispatch, promotion, refill, and model-provider calls;
- cancels prepared transactions;
- compensates, repairs, or quarantines dispatching transactions; and
- independently revalidates committed transactions or rolls them back.

No candidate, model, or automatic controller may clear it. Only an operator
may clear it after an audit, recovery validation, fresh current-tree
revalidation, and a recorded decision.

Manual escalation keeps the system in report-only/abstain state. It is required
for protected anchors, new network/provider scope, cross-repository effects,
scope/effect expansion, approval-required change classes, unmet proof/security
requirements, repeated oscillation, rollback/recovery failure, and capability
loss. The operator decision must bind fresh roots and cannot be supplied by the
candidate under review.

## 12. Deterministic policy identity

The JSON policy contains no timestamp, float, non-finite number, duplicate key,
or stored self-hash. Its canonical preimage is UTF-8 JSON with recursively
sorted keys and compact separators. The policy root is SHA2-256 over those
canonical bytes and is stored by the existing artifact/CAS layer outside the
policy body. Excluding a self-hash avoids a recursive identity definition.

Every later Planner/Doctor admission, benchmark receipt, mutation permit, and
promotion decision must bind the exact sealed policy revision and root.
Changing a task status or reconciling a board cannot manufacture the operator
seal. Runtime activation requires a fresh policy load so a daemon cannot keep
using a pre-seal in-memory protection or authority snapshot.

### 12.1 External receipt identity

The separate seal receipt is canonical UTF-8 JSON with recursively sorted keys,
compact separators, unique keys, no floats/non-finite numbers, and no
insignificant whitespace in its identity preimage. `receipt_id` is:

```text
"sha256:" + hex(
  SHA-256(canonical_json(receipt with the top-level receipt_id removed))
)
```

The receipt binds a Git base rather than its own eventual commit. A commit
cannot contain its own not-yet-known commit ID without a recursive definition.
`reviewed_base.commit` therefore names the exact pre-artifact base commit and
`reviewed_base.tree` must equal that commit's Git tree. The final three artifact
byte roots form the reviewed overlay. At activation, the base must still exist
and be equal to or an ancestor of the activation HEAD. This ancestry check is
provenance, not current-tree authorization: every operational admission still
performs the current-tree revalidation defined by the policy.

## 13. Seal-receipt validation rules

A fresh scheduler load may activate policy revision `1` only when all of the
following checks pass:

1. Read the policy and receipt from their exact configured paths without
   following a path outside the repository root.
2. Parse both with duplicate-key rejection, UTF-8 decoding, no floats, and no
   `NaN`/infinity.
3. Require the exact policy schema/interface/revision and exact seal
   schema/interface/receipt version. Reject unknown receipt fields.
4. Require the claimed `receipt_id` to constant-time match the exact identity
   independently pinned in the protected scheduler configuration. Remove only
   top-level `receipt_id`, canonicalize the remaining receipt, recompute the
   exact `sha256:` identity, and constant-time compare that identity too.
5. Require task `PDR-002` and board namespace
   `agent-supervisor-proof-directed-planner-doctor-v1`.
6. Require exactly three artifact bindings with no duplicate role or path:
   `authority_policy`, `threat_model`, and `authority_test`, at the exact paths
   declared by the policy.
7. Read each final file as bytes, compute `sha256:` plus lowercase 64-hex
   digest and exact byte count, and match both receipt fields.
8. Require receipt `policy_revision` to equal the policy's revision and the
   grant's revision/scope.
9. Validate the reviewed Git base: lowercase 40-hex SHA-1 commit and tree,
   object format `sha1`, both objects exist locally, and
   `git rev-parse <commit>^{tree}` equals the receipt tree. Require the commit
   to equal or be an ancestor of activation HEAD.
10. Require operator identity `interactive_user` and authority basis
    `interactive_user_delegation`. These literal fields describe the grant but
    do not authenticate it; authentication comes from the independently
    protected receipt-identity pin. Neither may be supplied or overridden by a
    candidate, model, task row, environment variable, or execution permit.
11. Require the grant object to equal the closed activation-only grant:
    action `activate_policy_revision`, non-delegable, and all mutation,
    completion, promotion, task-status, and protected-anchor-write authorities
    false.
12. Require the policy body to keep self-sealing forbidden and to name the
    exact receipt path/schema/interface/identity algorithm and the same three
    artifact paths.
13. On any absence, parse error, root/size/revision/base/operator/grant/identity
    mismatch, or extra authority, do not partially activate. Emit a stable
    rejection receipt, engage no mutation, and remain `shadow`/`report_only`.
14. After successful verification, activate only this policy revision and
    record its policy root plus seal `receipt_id` in the runtime authority
    snapshot. Never infer task completion, promotion, or mutation authority.

The loader must perform these checks on every fresh process start and whenever
the policy/receipt watch identity changes. A pre-seal in-memory snapshot cannot
be upgraded by a task-status event.

## 14. Manual seal checklist

An operator may mark PDR-002 complete only after independently confirming:

1. policy and receipt parse without duplicate keys, floats, or non-finite
   values and have reproducible canonical identities;
2. its control and assurance vocabularies match the shipped enums;
3. all authority-ladder stages are distinct and none grants mutation or
   completion;
4. deterministic Doctor model/network/provider flags remain hard false;
5. preview/apply, permit, lease, fence, expected-effect, checkpoint, rollback,
   and current-root gates are present;
6. seed task sources, authority artifacts, holdout/oracle, denominators,
   safety floors, and promotion policy are protected;
7. every forbidden transition is rejected or quarantined;
8. every safety floor is an exact integer zero;
9. task counts/status cannot complete a goal;
10. the focused PDR-002 validation and its tamper cases pass on the reviewed
    tree;
11. the external content-addressed receipt binds all three artifact SHA-256
    roots, revision, reviewed base commit/tree, interactive user identity and
    delegation basis, and the activation-only grant;
12. the receipt identity exactly matches the separate pin in the protected
    scheduler configuration; and
13. every long-running supervisor freshly reloads and verifies the sealed policy
    instead of retaining a pre-seal snapshot.

If any validation later fails, the only safe operational state is
shadow/report-only with mutation, automatic promotion, and refill disabled.
