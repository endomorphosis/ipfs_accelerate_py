# Proof-Backed Test Reuse: Runtime Activation Operator Handoff

Status: normative operator handoff for the runtime-activation repair closeout  
Interfaces: `ProofReuseActivationContract@1`, `RuntimeContextRevalidator@1`, `PytestProofReusePlugin@1`, `ProofTestReuseCurrentTreeGateDecision@1`, `RuntimeActivationE2E@1`  
Evidence: cross-repository cold/pass/deferred/warm receipts, zero-false-skip matrix, warm benchmark, refreshed 53-task gate  
Controllers:

* Outer objective closeout: `scripts/proof_backed_test_reuse_supervisor.py closeout`
* Board validation: `scripts/validate_proof_backed_test_reuse_board.py`

## Purpose

Document how the reviewed runtime-activation repair (`PTR-131` … `PTR-142`) is
proved, how the final current-tree gate expands from 41 to **exactly 53** tasks
with **fresh repair evidence**, and how an operator invokes the **existing**
outer closeout controller only after validation succeeds and a human reviews the
protected lifecycle update.

This document is the operator handoff produced by **PTR-142**. Completing
PTR-142 is a precondition for invoking live closeout on the expanded board.
**Task completion precedes, and does not itself constitute, the live operator
closeout.**

## Non-negotiable doctrine

1. **No test-file hardwiring or manual service injection** is required for the
   automatic runtime. Production repository bootstraps remain loader-only;
   session-scoped defaults compose identity, lookup, revalidation, local
   verification, terminal pass capture, deferred issuance, and xdist fencing.
2. **Authority sequence is sealed.** Locator → candidate descriptor → retained
   candidate rehash → current frontier rebuild → exact comparison + local
   verification of a real certificate → else execute once → post-pass retain →
   deferred public issuance → controller atomic publish.
3. **Simulated proof is never skip authority.** Missing or failing installers,
   packages, Groth16, ProveKit, cache, IPFS, network, keys, or circuits yield
   typed `RUN`/`DEFERRED` and never block pytest or the supervisor.
4. **Every admitted mutation forces RUN.** Source, AST, indirect dependency,
   fixture, hook, parameter, environment, lock, capability, policy, circuit,
   key, issuer, epoch, cache, and transport mutations never authorize
   `proof-cache-hit`.
5. **xdist workers never publish.** Workers return public intents only; one
   controller writes. No duplicate, partial, or private authority artifacts.
6. **Zero false skips before benchmark.** Sequential
   `IPFS_TEST_PROOF_REUSE_MODE=off` assurance must report zero authoritative
   false skips before warm benchmark evidence is admitted.
7. **The protected objective heap remains operator-owned.** Closeout writes a
   state-root candidate only. Promoting that candidate into
   `implementation_plan/docs/46-proof-backed-test-reuse.objectives.md` requires
   an explicit human commit after review.
8. **PTR-142 does not run live closeout.** It refreshes assurance and the gate,
   then hands the existing fenced outer controller to the operator.

## What PTR-142 proved

| Surface | Location | Result |
| --- | --- | --- |
| Automatic activation e2e | `test/api/test_proof_reuse_runtime_activation_e2e.py` | Per-repository cold miss → one complete pass + runtime trace → retained candidate → real local certificate → one warm `proof-cache-hit` skip; zero false skips under the mutation matrix |
| Cross-repository bootstrap | `test/api/test_proof_reuse_cross_repository_e2e.py` | Entry points + loader-only fallbacks; production bootstraps free of service injection/item hardwiring |
| Invalidation population | `test/api/test_proof_reuse_invalidation_mutations.py` | Historical mutation corpus still forces RUN; sequential zero-false-skip assurance |
| Security/concurrency | `test/api/test_proof_reuse_security_concurrency.py` | Hostile/partial/private publication never authorizes skip; controller intents public-only |
| Warm benchmark | `test/api/test_agent_supervisor_proof_reuse_benchmark.py` | Off-mode zero false skips first; warm verification cheaper than execution; configured warm-skip target met |
| Current-tree gate | `proof_test_reuse_current_tree_gate.py` | Sealed production population = **53** tasks; requires fresh `runtime-activation` repair evidence covering `PTR-131` … `PTR-142` |

### Validation command (implementation gate)

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \
  external/ipfs_accelerate/test/api/test_proof_reuse_runtime_activation_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_cross_repository_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_invalidation_mutations.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_security_concurrency.py \
  external/ipfs_accelerate/test/api/test_agent_supervisor_proof_reuse_benchmark.py \
  external/ipfs_accelerate/test/api/test_agent_supervisor_proof_test_reuse_current_tree_gate.py \
  -q
```

Pytest must remain green when optional stacks are absent. Never enable proof
reuse as deployment authority during this validation (`MODE=off`).

## Gate population and repair evidence

### Exact 53-task board

The production constant `REQUIRED_PTR_TASK_IDS` is the sealed set of **53**
implementation tasks: the historical 41-task board plus the runtime-activation
repair wave:

`PTR-131`, `PTR-132`, `PTR-133`, `PTR-134`, `PTR-135`, `PTR-136`, `PTR-137`,
`PTR-138`, `PTR-139`, `PTR-140`, `PTR-141`, `PTR-142`.

`SEALED_PRODUCTION_TASK_COUNT == 53`. The producing task for final-gate evidence
remains **`PTR-122`** (self-reference free). PTR-142 refreshes the population
and repair premises; it does not reintroduce `PTR-G110` as a child premise.

### Fresh repair evidence (required)

When the gate evaluates the production population (any required set that
intersects the repair wave), it demands a fresh, authoritative repair evidence
record:

| Field | Requirement |
| --- | --- |
| `authority` | `authoritative` |
| `repair_id` | `runtime-activation` |
| `repair_task_ids` | covers every id in `RUNTIME_ACTIVATION_REPAIR_TASK_IDS` |
| `passed` | `true` |
| `false_skips` | `0` |
| `zero_false_skip_assurance` | `true` |
| `activation_e2e_passed` | `true` |
| `requirement_id` (optional) | `ptr/runtime-activation-repair-evidence@1` |
| freshness + bindings | same tree/forest/policy/capability/key/circuit window as other premises |
| `evidence_cid` | present and rehashable when retained as dag-json |

Missing, stale, mismatched, failed, or false-skip repair evidence refuses the
gate. Unit tests that use a small task subset without the repair wave do not
require this premise.

## Preconditions for live closeout (expanded board)

Before an operator invokes live closeout on the integration checkout:

1. **Every implementation task is closed** on the validated board
   (`PTR-000` … `PTR-142`, **53** tasks). Open tasks refuse closeout.
2. **PTR-142 is complete** (this runbook, the activation e2e, refreshed gate,
   and validation command above pass with proof reuse **off**). Task completion
   is necessary and **not sufficient**.
3. **Checkout is clean** on the reviewed integration branch, with the expected
   tree identity.
4. **Supervisor lanes are healthy and work-complete**, so a fresh three-lane
   supervisor-health receipt can be captured.
5. **Final-gate premises are retained** (task provenance for all 53 tasks,
   G010–G100 evidence, adversarial populations, benchmark, rollout readiness,
   supervisor health, **and fresh runtime-activation repair evidence**) and are
   replayable by canonical CID.
6. **Sequential zero-false-skip assurance** has been recorded under
   `IPFS_TEST_PROOF_REUSE_MODE=off` before warm benchmark admission.
7. **No concurrent closeout writer** holds the fence.

### Genuine approvals (unchanged from PTR-130)

Historical planning/review tasks still require genuine operator or reviewer
provenance where the merge queue lacks managed-merge receipts:

| Task | Provenance kind |
| --- | --- |
| `PTR-000` | `operator_planning_seal` |
| `PTR-001`, `PTR-011` | `operator_reviewed_integration` |
| `PTR-041` | `retrospective_integration_verification` |

All other sealed tasks normally carry `managed_merge` provenance. Quarantine,
unsuccessful merges, and unknown provenance kinds never count as complete.

## Operator procedure (live current-tree closeout)

> **Reminder:** Finishing PTR-142 (or seeing green activation e2e) is **not**
> live closeout. Only the sequence below, ending in an explicit operator commit
> of the protected objective file, closes the live root.

### 1. Confirm the PTR-142 validation gate

From a clean integration worktree, with proof reuse **off**:

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \
  external/ipfs_accelerate/test/api/test_proof_reuse_runtime_activation_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_cross_repository_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_invalidation_mutations.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_security_concurrency.py \
  external/ipfs_accelerate/test/api/test_agent_supervisor_proof_reuse_benchmark.py \
  external/ipfs_accelerate/test/api/test_agent_supervisor_proof_test_reuse_current_tree_gate.py \
  -q
```

Do not proceed if any test fails, if optional-capability absence aborts the
suite, or if any authoritative false skip appears.

### 2. Report-only diagnosis (no writes)

```bash
IPFS_TEST_PROOF_REUSE_MODE=off \
  python3 scripts/proof_backed_test_reuse_supervisor.py closeout --report-only
```

Report-only must not write the repository or emit a candidate. Inspect the
diagnosed open tasks, missing premises, and repair-evidence gaps.

### 3. Fenced three-phase closeout (state-root candidate only)

```bash
IPFS_TEST_PROOF_REUSE_MODE=off \
  python3 scripts/proof_backed_test_reuse_supervisor.py closeout
```

Expected effects (state root only):

* writer fence acquired (compare-and-swap);
* phase-1 provisional transitions for drained goals;
* current validation rerun with proof reuse **off**;
* phase-2 verification of `PTR-G010` … `PTR-G100`;
* final-gate admission against the **53-task** population and **fresh repair
  evidence** (producing task `PTR-122`; no G110 self-premise);
* phase-3 verification of `PTR-G110` then `PTR-G000`;
* candidate objective + lifecycle projection + closeout status JSON under the
  configured state-root completion directory;
* `operator_commit_required: true` and `repository_written: false`.

### 4. Operator review of the protected lifecycle update

Review the candidate under the XDG state root (paths from
`config/proof_backed_test_reuse_supervisor.json`,
`objectiveProjection.*PathSuffix`):

* every required task id is present and closed with accepted provenance;
* repair evidence covers `PTR-131` … `PTR-142` and is fresh;
* zero false skips on adversarial and off-mode assurance premises;
* benchmark and rollout premises re-verify;
* supervisor-health shows three healthy lanes;
* `operator_commit_required` is true; the repository was not written.

### 5. Explicit protected commit (human only)

Only after review, promote the candidate into the protected objective heap with
an explicit human commit. Implementation agents and worker lanes must never
mutate:

* `implementation_plan/docs/46-proof-backed-test-reuse-plan-2026-07-31.md`
* `implementation_plan/docs/46-proof-backed-test-reuse.objectives.md`
* `implementation_plan/docs/46-proof-backed-test-reuse.todo.md`
* `config/proof_backed_test_reuse_supervisor.json`
* `scripts/validate_proof_backed_test_reuse_board.py`
* `scripts/proof_backed_test_reuse_supervisor.py`

## Fail-closed classes (never verify / never skip)

| Class | Examples |
| --- | --- |
| Missing | Absent task, goal, evidence CID, gate artifact, repair evidence, supervisor-health |
| Stale | Tree-mismatched or age-expired gate/evidence/health/benchmark/repair packets |
| Forged | Tampered retained bytes, forged completion decisions, multihash mismatch |
| Noncanonical | Legacy `sha256:` labels, CIDv0, wrong codec, uppercase base32, path escapes |
| Mismatched | Git tree ≠ forest ≠ objective-completion identity; policy/capability/key drift |
| Ordinary-skip | Pytest skip text or disposition `ordinary_skip` as task authority |
| Simulated-proof | `authority=simulated`, mock/demo certificates, non-real backends labeled verified |
| Unavailable optional stack | Groth16/ProveKit/snarkjs/IPFS/cache absent without a reviewed real-certificate fixture |
| Mutation | Any admitted source/AST/fixture/hook/parameter/env/lock/capability/policy/circuit/key/issuer/epoch/cache/transport change |
| Private/partial xdist | Worker-published authority, partial temp blobs, witness material in intents |
| Tree-mutated / restart | Dirty checkout or interrupted closeout leaving the live heap verified |

## Relationship to prior handoffs

| Document | Role |
| --- | --- |
| `TEST_PROOF_REUSE_RUNTIME_ACTIVATION.md` | Sealed activation contracts (PTR-131) |
| `TEST_PROOF_REUSE_OBJECTIVE_CLOSEOUT.md` | Three-phase outer closeout mechanism (PTR-130) |
| **This document** | Runtime-activation proof, 53-task gate refresh, operator invocation of the same outer controller after PTR-142 |

The outer controller API is unchanged. Only the population, repair evidence,
and operator preconditions expand.

## Explicit non-goals

* Autonomous gap or codebase refill remains disabled.
* PTR-142 does not rewrite completed task contracts for PTR-000 … PTR-141.
* PTR-142 does not promote state-root candidates into the protected objective
  file.
* Synthetic or simulated certificates are never deployment authority.
* The proof-reuse cache used during ordinary development is never the authority
  for validating this implementation (use disposable stores and MODE=off gates).
