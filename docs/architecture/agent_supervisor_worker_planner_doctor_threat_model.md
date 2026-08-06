# Worker Planner–Doctor authority and threat model

Interface: `WorkerPlannerDoctorThreatModel@1`
Policy interface: `WorkerPlannerDoctorAuthorityPolicy@1`
Program: `agent-supervisor-worker-planner-doctor-v1`
Task: `WPD-003`
Status: published non-compensable safety floors; policy root is protected
after operator review

## 1. Purpose and status

This document defines the non-compensable trust boundary for wiring the live
agent supervisor and implementation workers to the existing Planner and
Deterministic Doctor as the primary reasoning path. Its machine-readable
companion is `config/supervisor_worker_planner_doctor_authority_policy.json`
at the monorepo root.

WPD is an integration program. It reuses Planner/Doctor contracts from PDR
and LPR; it does not invent a second assurance lattice, control permit
vocabulary, proof cache, or completion authority. If this document conflicts
with a stricter shipped control, proof, Doctor, security, or completion
contract, the stricter shipped contract wins.

The policy is deliberately fail-closed:

- Planner and Doctor start in `shadow` / `report_only`;
- mutation, automatic promotion, and derived refill start disabled;
- provider-first implementation is not the default path;
- free re-prompt after a typed failure is forbidden;
- unknown, ambiguous, stale, incomplete, unavailable, or conflicting required
  evidence causes rejection or typed abstention, never a pass; and
- candidate work cannot self-certify, seal policy, or mutate protected
  control anchors.

Every listed safety event has an exact integer floor of zero. Speed, token
savings, cost, throughput, or a favorable aggregate score cannot compensate
for a nonzero floor.

## 2. Scope and security objectives

### In scope

- task claim → pre-implementation kernel → plan/doctor disposition;
- residual LLM authorization under sealed packets only;
- validation/merge failure → Doctor inspect → formal replan → residual packet;
- supervisor selection, retry, rescue, and refill preferring typed
  dispositions over model re-entry;
- deterministic Doctor hard-off for network, remote models, and LLM routers;
- protected WPD control artifacts after review; and
- release gates that require zero safety floors.

### Security objectives

1. No LLM provider call occurs unless disposition is exactly
   `residual_llm_authorized` and a sealed residual packet is bound.
2. Deterministic Doctor paths never load network clients, LLM routers, or
   remote model/embedding providers.
3. Provider prose, synthetic Doctor success flags, task status, and candidate
   claims are never completion authority.
4. After a typed failure, the next provider attempt requires Doctor inspect,
   formal replan, and a residual packet—not free-form task prose re-injection.
5. Candidate code cannot certify itself, issue its own admission receipt, or
   edit protected WPD anchors.
6. Writes require mutation permit, writer lease, fencing epoch, expected
   effects, exact roots, checkpoint, and rollback; path/scope escape fails.
7. False fixed points (prebuilt mappings, boolean convergence without live
   rebuild) cannot complete work.

### Non-goals

This threat model does not claim that:

- hermetic analytical fixtures prove production provider reduction alone;
- a residual LLM packet proves semantic correctness of the resulting edit;
- this policy alone enables unattended repository mutation; or
- historical PDR/LPR completion rows are current-tree evidence without
  revalidation.

## 3. Protected assets and anchors

| Asset | Required property |
| --- | --- |
| WPD control plane | Plan, objectives, todo, scheduler, supervisor profile, board validator, and supervisor script remain operator-owned and non-writable by candidates after review. |
| Authority policy | Exact canonical policy bytes and revision; floors and forbidden transitions are immutable without a new policy revision and re-baseline. |
| Threat model and authority tests | Content roots bound to the policy artifact set; tests encode floors as machine checks. |
| Residual LLM packets | Exact write paths, obligations, counterexample capsules, validation commands, and authority roots; no secrets or unbounded dumps. |
| Implementation disposition | Closed enum; only `residual_llm_authorized` may invoke a provider. |
| Mutation state | Permit, lease, fence, expected effects, checkpoint, and rollback lineage remain consistent. |
| Completion contract | Existing goal completion remains authoritative; provider exit 0 is not completion. |

Concrete protected paths are closed in the policy JSON under
`protected_anchors.paths`, including the seven operator-protected control
artifacts and this program's authority policy, threat model, and test.

## 4. Actors, assumptions, and trust boundaries

### Actors

- **Operator:** owns policy review, protected-anchor changes, exceptional
  approvals, promotion, and kill-switch clearing.
- **Control service:** validates typed requests and exact authorization
  bindings immediately before dispatch.
- **Pre-implementation kernel:** evaluates task + forest + policy and emits a
  closed disposition before any provider call.
- **Planner/Doctor kernel:** builds or diagnoses obligation graphs; default
  outputs are read-only proposals.
- **Candidate agent/model:** may nominate plans, premises, tests, or edits. It
  is untrusted for admission, validation, mutation, completion, and promotion.
- **Validation runner:** independently rebuilds, analyzes, tests, and evaluates
  fixed point in an isolated environment.
- **Unattended controller:** may schedule shadow evaluation within policy. It
  does not own the policy, judge, mutation permit, completion, or promotion.

### Trust boundaries

1. **Claim → pre-implementation kernel.** Claiming a task and creating a
   worktree does not authorize a provider call. The kernel must run first.
2. **Disposition → provider.** Only `residual_llm_authorized` with a sealed
   residual packet may enter a provider. Other dispositions record receipts
   and must not free-form re-prompt.
3. **Deterministic Doctor → network/LLM.** Deterministic mode is hard-off for
   network, remote models, embeddings, and LLM routers. Provider absence is
   typed abstention, never automatic model fallback.
4. **Typed failure → retry.** Validation/merge failure must enter Doctor
   inspect and formal replan. A free re-prompt that omits the residual packet
   is a floor breach.
5. **Candidate → authority.** Self-supplied proof flags, expected oracles,
   admission receipts, or task-status writes cannot manufacture authority.
6. **Provider prose → completion.** Exit codes and narrative success claims
   are operational signals only; independent completion gates remain.
7. **Candidate worktree → protected anchors.** Candidate output cannot mutate
   WPD control artifacts or authority floors.

## 5. Closed authority ladder

The ladder distinguishes evidence quality from operation authority. Even the
highest semantic evidence does not itself grant mutation or completion.

| Stage | Maximum assurance | Permitted use | Explicitly not permitted |
| --- | --- | --- | --- |
| Nomination | `candidate` | Discover, retrieve, rank, suggest | Admit, prove, write, promote, complete, invoke unauthorized LLM |
| Observation | `candidate` | Record root-bound facts, traces, tests | Unbounded semantics, kernel proof, write, completion |
| Bounded check | `solver_checked` | Establish/refute under recorded bounds | Outside bounds, kernel claim, write alone |
| Kernel proof | `kernel_verified` | Satisfy exact reconstructed theorem | Unencoded property, bypass permit, complete alone |
| Attestation | `attested` | Bound computation / lineage under verified program | Expand semantic claim, write, complete |

`OperationAuthority` remains the closed control vocabulary: `read`,
`proposal`, and `mutation`. Proof-assurance rank is never converted into
operation authority by comparison.

## 6. Non-compensable safety floors

The following floors are exact integer zero in the machine policy. They are
non-compensable: no quality or efficiency metric may trade against them.

| Floor key | Meaning |
| --- | --- |
| `unauthorized_llm_calls` | Provider/LLM call when disposition ≠ `residual_llm_authorized`. |
| `deterministic_doctor_network_or_llm_load` | Deterministic Doctor path loads network, LLM router, or remote model/embedding clients. |
| `completion_from_provider_prose` | Completion claimed from provider narrative, exit code, or prose alone. |
| `free_reprompt_after_typed_failure` | Free-form re-prompt after typed failure without residual packet / Doctor replan. |
| `candidate_self_certification_count` | Candidate certifies its own proof, oracle, receipt, or completion. |
| `false_fixed_point_count` | Prebuilt/boolean fixed point treated as live convergence. |
| `policy_scope_escape_count` | Write or effect outside exact permit/path/root bindings. |

Additional zero floors in the policy (LLM router rates, path escape, secret
exposure, synthetic Doctor completion, and related counts) support the same
boundary and must also remain zero.

## 7. Threats and required mitigations

Transition IDs match `forbidden_transitions` in the machine policy.

| Threat | Attack | Required defense and disposition |
| --- | --- | --- |
| T01 Unauthorized LLM | Implementation path calls a provider without `residual_llm_authorized` or without a sealed residual packet. | `unauthorized_llm_invocation` rejected; floor `unauthorized_llm_calls` stays 0. |
| T02 Deterministic network/LLM | Deterministic Doctor mode imports or calls network/LLM/remote embedding clients. | `deterministic_mode_network_or_llm_load` rejected; cold import hygiene; floors stay 0. |
| T03 Completion from prose | Provider exit 0 or narrative “done” marks the goal complete. | `completion_from_provider_prose` rejected; existing completion contract required. |
| T04 Free re-prompt | After typed validation/merge failure, worker re-injects full task prose without Doctor replan + residual packet. | `free_reprompt_after_typed_failure` rejected. |
| T05 Self-certification | Candidate supplies its own test/proof flag/expected value/receipt and claims authority. | `candidate_or_self_certification_to_authority` rejected. |
| T06 Synthetic Doctor success | Synthetic Doctor success flags used as completion or promotion evidence. | `completion_from_synthetic_doctor_success` rejected. |
| T07 Model write | Prompt/model output directly edits source or control state. | `model_output_to_write` rejected. |
| T08 False fixed point | Prebuilt mapping or boolean reports convergence without live rebuild. | `prebuilt_or_boolean_fixed_point_to_completion` rejected. |
| T09 Scope/path escape | Permit widened, path escapes allowlist, or undeclared effects land. | `scope_escape_or_path_escape_to_write` quarantined and escalated. |
| T10 Write without permit | Mutation without permit, lease, fence, or exact roots. | `write_without_mutation_permit_or_lease_or_exact_roots` rejected. |
| T11 Protected anchor mutation | Candidate edits WPD control plane or authority floors. | `candidate_work_to_protected_anchor_mutation` quarantined and escalated. |
| T12 Retrieval authority laundering | Similarity/rank/provider response treated as semantic truth. | `provider_or_retrieval_result_to_semantic_authority` rejected. |
| T13 Task-status completion | Task count/status alone completes a goal. | `task_count_or_task_status_to_completion` rejected. |

## 8. Live worker loop (normative)

```text
claim task + worktree lease
         │
         ▼
PreImplementationKernel.evaluate(task, forest, policy)
         │
         ├─ Doctor inspect (if failure residual / contract gap)
         ├─ Evidence factory (AST/graph/cache snapshot)
         ├─ Formal/adaptive plan compile + validate
         ├─ Proof-carrying / analytical transform attempt
         │
         ▼
   disposition ∈ {
     closed_deterministic,      # no LLM
     residual_llm_authorized,   # sealed residual packet only
     abstain_review,            # typed residual for operator
     defer_capability           # missing optional backend
   }
         │
         ▼
if closed_deterministic → validate → merge admit → completion gates
if residual_llm_authorized → provider(packet only) → validate → …
if abstain/defer → record receipt; do not free-form re-prompt
```

On validation or merge failure:

```text
typed failure
  → Doctor inspect
  → formal replan
  → residual packet admission
  → disposition residual_llm_authorized (or abstain/defer)
  → provider only if residual authorized
```

Skipping Doctor/replan/packet and re-prompting the model is a floor breach
(`free_reprompt_after_typed_failure`).

## 9. Deterministic Doctor and residual LLM boundary

### Deterministic Doctor (hard-false)

- no LLM router or LLM invocation;
- no remote model-provider call;
- no remote embedding;
- no network access or network-client import;
- no target-code import as an analysis shortcut;
- no semantic authority for knowledge graphs, vectors, embeddings, candidates,
  or proof-cache metadata; and
- cold import of worker/doctor bridges must not load `requests` or remote
  provider clients.

Provider absence produces typed abstention and must not trigger model
fallback. Report-only startup remains available without optional providers.

### Residual LLM (named, sealed path)

Residual LLM is a separately named path distinct from deterministic mode. It
may request bounded residual syntax only after deterministic analysis, under
disposition `residual_llm_authorized`, with a sealed packet that:

- lists exact write paths and validation commands;
- binds obligations and counterexample capsules;
- excludes secrets and unbounded source dumps; and
- remains nomination authority only.

Model output cannot suppress mandatory queries, choose credentials or
endpoints, write directly, approve itself, claim completion, or inherit
deterministic-Doctor authority.

## 10. Completion, promotion, and kill switch

All tasks being marked complete is necessary operational bookkeeping at most;
it is not sufficient evidence that a goal is complete. The existing goal
completion contract remains authoritative and must independently verify fresh
acceptance evidence, current roots, required assurance, and zero safety
floors. Provider prose and synthetic Doctor success flags are never
completion authority.

Promotion begins in `shadow`, is manual and monotonic, and requires paired
live current-tree evidence. Synthetic data and skipped live checks may
measure plumbing but may not promote.

The kill switch can be engaged by an operator or automatically after a
non-compensable gate failure. While engaged it forces `report_only` and
blocks new mutation dispatch, promotion, refill, and model-provider calls.
Only an operator may clear it after audit and current-tree revalidation.

## 11. Deterministic policy identity

The JSON policy contains no timestamp, float, non-finite number, duplicate
key, or stored self-hash. Its canonical preimage is UTF-8 JSON with
recursively sorted keys and compact separators. The policy root is SHA2-256
over those canonical bytes and is stored by the existing artifact/CAS layer
outside the policy body.

Every later pre-implementation admission, residual LLM authorization,
benchmark receipt, mutation permit, and promotion decision must bind the exact
policy revision and root. Changing a task status or reconciling a board cannot
manufacture policy authority.

## 12. Machine-check contract

Tests under
`external/ipfs_accelerate/test/api/test_agent_supervisor_worker_planner_doctor_authority_policy.py`
encode the floors and forbidden transitions as machine checks:

1. declared outputs and interfaces exist;
2. policy canonicalization forbids floats and duplicate keys;
3. safety floors are exact integer zero and cover the non-compensable set;
4. forbidden transitions include unauthorized LLM, deterministic-mode network,
   completion-from-prose, free re-prompt after typed failure, and candidate
   self-certification;
5. deterministic Doctor hard-false flags remain false;
6. free-reprompt and residual-LLM policies fail closed; and
7. threat-model text names the policy interface and non-compensable floors.

A green pytest run does not grant completion authority; it only proves the
published floors remain machine-enforceable.
