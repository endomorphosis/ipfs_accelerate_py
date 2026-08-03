# Proof-Backed Test Reuse: Runtime Activation Operator Handoff

Status: PTR-149 deliverables complete; live closeout remains operator-owned and
unavailable until the board is closed, the refreshed 60-task gate admits fresh
production-runtime-activation evidence, and a human reviews the protected
lifecycle update
Interfaces: `ProofReuseActivationContract@1`, `RuntimeContextRevalidator@1`, `PytestProofReusePlugin@1`, `ProofTestReuseCurrentTreeGateDecision@1`, `ProofReuseRuntimeActivationReport@1`, `RuntimeActivationE2E@1`
Evidence: genuine three-repository two-process cold/warm receipts, real Groth16 certificate, zero-false-skip matrix, measured subprocess benchmark, live typed capability report, refreshed 60-task gate
Controllers:

* Outer objective closeout: `scripts/proof_backed_test_reuse_supervisor.py closeout`
* Board validation: `scripts/validate_proof_backed_test_reuse_board.py`
* Live capability report: `proof_reuse_runtime_activation_report` (`ProofReuseRuntimeActivationReport@1`)

## Purpose

Document why the historical runtime-activation evidence from `PTR-138`,
`PTR-140`, and `PTR-142` is not production authority, how the corrective wave
(`PTR-143` … `PTR-149`) expands the final current-tree gate to **exactly 60**
tasks with fresh production-activation evidence, how live capability reporting
derives readiness from typed services and bounded non-mutating probes, and how
an operator invokes the **existing** outer closeout controller only after
validation succeeds and a human reviews the protected lifecycle update.

**PTR-149** produces this handoff, the live activation report, and the refreshed
gate. Completing all seven corrective tasks is a precondition for invoking live
closeout. **Task completion precedes, and does not itself constitute, the live
operator closeout.**

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
8. **Historical PTR-142 evidence is inadmissible.** Only fresh PTR-149 evidence
   from the ordinary zero-injection, two-process, real-Groth16 path may satisfy
   the production-activation premise, and PTR-149 still does not run live
   closeout itself.

## Why the historical activation claim is superseded

| Surface | Audit result | Corrective owner |
| --- | --- | --- |
| Collection identity | Full assembly requires runtime evidence before producing the locator needed for runtime-candidate retrieval | PTR-143 |
| Warm lookup | Revalidator is built over the wrong store, has no production current-context provider, and is absent from lookup authority | PTR-145 |
| Cold execution | Lifecycle counters never start the runtime tracer or publish a final execution key and complete canonical candidate | PTR-146 |
| Real issuance | Default issuer has no real prover and its public path does not return publishable verified certificate material | PTR-144, PTR-147 |
| Activation e2e | Tests inject services/item identity or construct a deterministic pseudo-certificate instead of using two independent default pytest processes | PTR-148 |
| Benchmark/reporting/gate | Timing is synthetic, reporting is hard-coded, and the 53-task gate accepts the historical claim | PTR-148, PTR-149 |

### Live capability reporting (PTR-149)

`proof_reuse_runtime_activation_report` / `ProofReuseRuntimeActivationReport@1`
derives availability only from already-composed default services and bounded
non-mutating probes:

* native Groth16 installation/readiness is reported separately from
  test-certificate authority;
* the generic pre-PTR-144 `knowledge_of_axioms` backend can never satisfy
  test-certificate authority, even when native readiness is true;
* the report never installs packages, never starts a prove/setup/network
  process, and never imports optional stacks merely to claim readiness;
* cold `proof_reuse_dependency_plan` inventory remains static plan metadata and
  is not admission authority for live readiness.

PTR-138, PTR-140, and PTR-142 retain historical completion provenance. They do
not satisfy the new production-activation premise and are not reopened; the
bounded correction supplies new task and evidence identities.

### Validation command (implementation gate)

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \
  external/ipfs_accelerate/test/api/test_proof_reuse_runtime_activation_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_cross_repository_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_subprocess_benchmark.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_runtime_activation_report.py \
  external/ipfs_accelerate/test/api/test_agent_supervisor_proof_test_reuse_current_tree_gate.py \
  -q
```

Pytest must remain green when optional stacks are absent. Never enable proof
reuse as deployment authority during this validation (`MODE=off`).

## Gate population and repair evidence

### Exact 60-task board

The production constant `REQUIRED_PTR_TASK_IDS` is the sealed set of **60**
implementation tasks: the historical 53-task board plus the corrective wave:

`PTR-143`, `PTR-144`, `PTR-145`, `PTR-146`, `PTR-147`, `PTR-148`, `PTR-149`.

`SEALED_PRODUCTION_TASK_COUNT == 60`. The producing task for final-gate evidence
remains **`PTR-122`** (self-reference free). PTR-149 refreshes the population
and corrective premise; it does not reintroduce `PTR-G110` as a child premise.

### Fresh repair evidence (required)

When the gate evaluates the production population (any required set that
intersects the repair wave), it demands a fresh, authoritative repair evidence
record:

| Field | Requirement |
| --- | --- |
| `authority` | `authoritative` |
| `repair_id` | `production-runtime-activation` |
| `producer_task_id` | `PTR-149` |
| `repair_task_ids` | exactly every id in `PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS` (`PTR-143` … `PTR-149`) |
| `passed` | `true` |
| `false_skips` | `0` |
| `zero_false_skip_assurance` | `true` |
| `activation_e2e_passed` | `true` |
| `zero_injection_default_path` | `true` |
| `three_repository_cold_warm` | `true` |
| `real_groth16_certificate` | `true` |
| `measured_subprocess_benchmark` | `true` |
| `historical_activation_claims_superseded` | `true` |
| `sealed_task_count` | `60` |
| `requirement_id` | `ptr/production-runtime-activation-evidence@1` |
| freshness + bindings | same tree/forest/policy/capability/key/circuit window as other premises |
| `evidence_cid` | present and rehashable when retained as dag-json |

Missing, stale, mismatched, failed, injected, pseudo-certificate, synthetic
timing, 53-task, PTR-142, or false-skip evidence refuses the gate. Unit tests
that use a small task subset without the corrective wave do not require this
premise.

## Preconditions for live closeout (expanded board)

Before an operator invokes live closeout on the integration checkout:

1. **Every implementation task is closed** on the validated board
   (`PTR-000` … `PTR-149`, **60** tasks). Open tasks refuse closeout.
2. **PTR-149 is complete** (this runbook, the genuine activation e2e, refreshed gate,
   and validation command above pass with proof reuse **off**). Task completion
   is necessary and **not sufficient**.
3. **Checkout is clean** on the reviewed integration branch, with the expected
   tree identity.
4. **Supervisor lanes are healthy and work-complete**, so a fresh three-lane
   supervisor-health receipt can be captured.
5. **Final-gate premises are retained** (task provenance for all 60 tasks,
   G010–G100 evidence, adversarial populations, benchmark, rollout readiness,
   supervisor health, **and fresh production-runtime activation evidence**) and are
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

> **Reminder:** Finishing PTR-149 (or seeing green activation e2e) is **not**
> live closeout. Only the sequence below, ending in an explicit operator commit
> of the protected objective file, closes the live root.

### 1. Confirm the PTR-149 validation gate

From a clean integration worktree, with proof reuse **off**:

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \
  external/ipfs_accelerate/test/api/test_proof_reuse_runtime_activation_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_cross_repository_e2e.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_subprocess_benchmark.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_runtime_activation_report.py \
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
* final-gate admission against the **60-task** population and **fresh corrective
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
* production-activation evidence covers exactly `PTR-143` … `PTR-149`, is
  produced by PTR-149, and is fresh;
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
| **This document** | Corrective production-activation requirements, 60-task gate refresh, operator invocation of the same outer controller after PTR-149 |

The outer controller API is unchanged. Only the population, repair evidence,
and operator preconditions expand.

## Explicit non-goals

* Autonomous gap or codebase refill remains disabled.
* PTR-149 does not rewrite completed task contracts for PTR-000 … PTR-148.
* PTR-149 does not promote state-root candidates into the protected objective
  file; only the existing fenced outer closeout may emit a state-root candidate,
  and only after the board is closed, the refreshed gate passes, and an operator
  reviews the protected lifecycle update.
* Synthetic or simulated certificates are never deployment authority.
* The proof-reuse cache used during ordinary development is never the authority
  for validating this implementation (use disposable stores and MODE=off gates).
* Hard-coded cold dependency-plan activation booleans are never live readiness
  claims; use `ProofReuseRuntimeActivationReport@1`.
