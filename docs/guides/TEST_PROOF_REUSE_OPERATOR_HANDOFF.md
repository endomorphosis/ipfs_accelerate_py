# Proof-Backed Test Reuse: Authenticated Current-Tree Operator Handoff

Status: PTR-169 deliverable — seals the exact **78-task** authenticated
current-tree handoff. Live root completion remains operator-owned and requires
an outer-controller post-merge re-run of the gate after PTR-169 merges.
Interfaces: `AuthenticatedProofReuseCurrentTreeGateV5@1`,
`ProofTestReuseObjectiveReconciler@1`, `ObjectiveCloseoutReceipt@1`,
`ProofReuseBenchmarkReceipt@2` (measured subprocess join),
`PytestProofReuseE2E@2`
Evidence: exact 78-task inventory, reachable gitlinks, trusted signed receipts,
locally verified real proofs, genuine three-repository cold/warm/forced-replay,
zero false skips, measured subprocess savings, supervisor health, authenticated
repair evidence covering `PTR-160` … `PTR-171`
Controllers:

* Outer objective closeout: `scripts/proof_backed_test_reuse_supervisor.py closeout`
* Objective reconciler: `scripts/proof_backed_test_reuse_objective_reconciliation.py`
* Board validation: `scripts/validate_proof_backed_test_reuse_board.py`
* Authenticated gate: `AuthenticatedProofReuseCurrentTreeGateV5`

## Purpose

Document how the sealed **78-task** v9 board reaches a validated **candidate**
handoff under the authenticated current-tree gate (`PTR-169` /
`ptr/authenticated-current-tree-gate-v5@1` on `PTR-G140`), what historical
packets are rejected as stale, and how an operator promotes a state-root
candidate into the protected objective heap only after the outer controller
re-runs the exact gate on the merged PTR-169 commit/tree.

**PTR-169 may emit only a pre-merge candidate receipt for itself.** Authoritative
completion of `PTR-G140` and `PTR-G000` requires the outer controller to re-run
the exact 78-task gate after the PTR-169 merge commit is present and prove that
commit/tree. Completing the implementation task is necessary and **not
sufficient** for live root closeout.

## Non-negotiable doctrine

1. **Exact 78-task inventory.** Production `REQUIRED_PTR_TASK_IDS` /
   `SEALED_PRODUCTION_TASK_COUNT == 78`. Historical **66-task** (PTR-149),
   **76-task** (v7), **77-task** (v8), and older packets are provenance only and
   fail closed as stale.
2. **G120, G130, and G140 remain mandatory.** The reconciler refuses root
   completion while any of these goals is still active or unverified. A valid
   PTR-169 artifact cannot bypass unfinished repair goals.
3. **Warm hits require trusted signed receipts and locally verified real
   proofs.** Ordinary pytest skips, simulated proofs, and unsigned materials
   never authorize `proof-cache-hit` for closeout.
4. **Genuine three-repo e2e and forced replay must agree.** Cold, warm, and
   forced-replay ordinary pytest processes (no `-p`, no service injection, no
   tracer monkeypatches) with a body oracle establish zero false skips.
5. **Benchmark meets the reviewed threshold.** Measured subprocess receipts
   retain raw wall-clock samples; synthetic cost constants never populate
   savings fields. Corpus warm-skip thresholds remain fail-closed.
6. **Optional capability gaps stay truthful `RUN` / `DEFERRED`.** Missing
   Groth16, ProveKit, snarkjs, IPFS, cache, or network never block ordinary
   tests and never invent skip authority.
7. **Worker lanes never reconcile goals.** Only the outer controller's fenced
   closeout writer advances lifecycle state, and only into a state-root
   candidate. Promoting that candidate into
   `implementation_plan/docs/46-proof-backed-test-reuse.objectives.md` requires
   an explicit human commit after review.
8. **Pre-merge candidate ≠ authoritative completion.** A PTR-169 pre-merge
   candidate receipt proves the join shape and current evidence bindings. It
   does not complete `PTR-G140` or `PTR-G000`.

## What changed from the 66-task (PTR-149) handoff

| Surface | PTR-149 (stale for closeout) | PTR-169 (production) |
| --- | --- | --- |
| Sealed population | 66 tasks | **78** tasks (`PTR-160` … `PTR-171` added) |
| Final goal / task | Historical G110 / PTR-122 path for premises | **`PTR-G140` / `PTR-169`** |
| Acceptance criterion | production-runtime-activation evidence | **`ptr/authenticated-current-tree-gate-v5@1`** |
| Review revision | v4 / intermediate repair labels | **`authenticated-receipt-current-tree-repair-v9`** |
| Warm authority | Real current-v4 certificate path | Signed receipt **and** local real-proof verification |
| Self-evidence | N/A (PTR-149 was not final gate) | Pre-merge candidate only until outer post-merge re-run |

## Exact 78-task board

The production constant set is the historical 66-task board plus the
authenticated current-tree repair wave:

`PTR-160`, `PTR-161`, `PTR-162`, `PTR-163`, `PTR-164`, `PTR-165`, `PTR-166`,
`PTR-167`, `PTR-168`, `PTR-169`, `PTR-170`, `PTR-171`.

`SEALED_PRODUCTION_TASK_COUNT == 78`. Child premises for the authenticated gate
include `PTR-G010` … `PTR-G130` (including mandatory `PTR-G110`, `PTR-G120`,
`PTR-G130`). `PTR-G140` is never a child premise of itself.

### Stale packets (fail closed)

| Packet | Reason code family |
| --- | --- |
| 66-task / PTR-149 production-activation | `repair_evidence_historical_66_task_population`, `…_ptr149_inadmissible` |
| 76-task / v7 | `…_historical_76_task_population`, reconciler `stale_gate_task_count` |
| 77-task / v8 | `…_historical_77_task_population`, reconciler `stale_gate_review_revision` |
| Review revision ≠ `authenticated-receipt-current-tree-repair-v9` | `stale_gate_review_revision` |
| Producer ≠ PTR-169 for final artifact | `wrong_gate_producer` |

## Lifecycle (three staged reconciliations)

| Phase | Name | Legal transitions | Must not |
| --- | --- | --- | --- |
| 1 | Provisional | Drained goals → `provisionally_complete` | Claim `verified_complete` |
| 2 | Children | After current validation (proof reuse **off**), `PTR-G010`…`PTR-G130` → `verified_complete` | Skip active G120/G130; verify G140/G000 |
| 3 | Final root | Admit authenticated gate evidence, then `PTR-G140` → `verified_complete`, then `PTR-G000` → `verified_complete` | Verify G000 before G140; admit pre-merge candidate as root authority |

### Pre-merge vs post-merge authority

1. **Implementation worktree (PTR-169 candidate):**
   `AuthenticatedProofReuseCurrentTreeGateV5.evaluate` may accept a
   `pre_merge_candidate` provenance **only for PTR-169 itself** and emit a
   candidate receipt with `authority=pre_merge_candidate`. No authoritative
   G140/G000 completion evidence is emitted.
2. **Outer controller after merge:** Re-run the exact 78-task gate against the
   merge commit/tree with closed non-candidate provenance for every task
   (including PTR-169). Only then may the gate emit authoritative
   `ptr/authenticated-current-tree-gate-v5@1` and
   `ptr/cross-repository-current-tree-gate@1` evidence.
3. **Reconciler:** Requires `producing_task_id=PTR-169`, `task_count=78`,
   `review_revision=authenticated-receipt-current-tree-repair-v9`, and
   authoritative evidence for G140 then G000. Pre-merge candidates do not
   admit root completion.

## Operator procedure

### Preconditions

1. Board validation reports `task_count=78` and a sealed preflight.
2. Every implementation task is closed on the validated board (or the
   controller knowingly re-runs after PTR-169 merges).
3. Checkout is clean on the integration branch with the expected tree identity.
4. Supervisor lanes are healthy; a fresh three-lane supervisor-health receipt
   can be captured.
5. Retained premises exist: task provenance, G010–G130 evidence, adversarial
   zero-false-skip populations, benchmark, rollout readiness, authenticated
   repair evidence, supervisor health.
6. No concurrent closeout writer holds the fence.

### Validation (implementation gate)

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \
  test/test_proof_backed_test_reuse_objective_reconciliation.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_authenticated_current_tree_gate.py \
  external/ipfs_accelerate/test/api/test_proof_reuse_authenticated_subprocess_benchmark.py \
  -q
```

Pytest must remain green when optional stacks are absent. Never enable proof
reuse as deployment authority during this validation (`MODE=off`).

### Live closeout (outer controller only)

```bash
# Report-only diagnosis — never mutates the protected heap
python3 scripts/proof_backed_test_reuse_supervisor.py closeout --report-only

# Fenced three-phase reconciliation to a state-root candidate
python3 scripts/proof_backed_test_reuse_supervisor.py closeout
```

Review the candidate under the configured state root
(`projection/completion/objective_candidate.md` and gate/evidence artifacts).
Only an explicit human commit may promote that candidate into the protected
objective file.

## Required authenticated repair evidence (PTR-169)

When the gate evaluates any population that includes the authenticated wave, it
demands a fresh authoritative repair record produced by **PTR-169** covering
**exactly** `PTR-160` … `PTR-171`:

| Field | Requirement |
| --- | --- |
| `authority` | `authoritative` |
| `repair_id` | `authenticated-current-tree` |
| `producer_task_id` | `PTR-169` |
| `repair_task_ids` | exactly every id in `AUTHENTICATED_CURRENT_TREE_REPAIR_TASK_IDS` |
| `sealed_task_count` | `78` |
| `requirement_id` | `ptr/authenticated-current-tree-gate-v5@1` |
| `review_revision` | `authenticated-receipt-current-tree-repair-v9` |
| `trusted_signed_receipts` | `true` |
| `locally_verified_real_proofs` | `true` |
| `genuine_three_repository_e2e` | `true` |
| `forced_replay_agrees` | `true` |
| `zero_false_skips` | `true` |
| `benchmark_meets_threshold` | `true` |
| `optional_capability_gaps_truthful` | `true` |
| `false_skips` | `0` |
| `passed` | `true` |

## Fail-closed classes (never verify)

| Class | Examples |
| --- | --- |
| Missing | Absent task, goal, evidence CID, gate artifact, supervisor-health receipt |
| Stale | Tree-mismatched or age-expired packets; 66/76/77-task inventories; v7/v8 review revisions |
| Forged | Tampered retained bytes, forged completion decisions, multihash mismatch |
| Pre-merge as root | PTR-169 candidate receipt used as G140/G000 authority |
| Ordinary-skip | Pytest skip text as task authority without signed receipt + real proof |
| Simulated-proof | `authority=simulated`, mock certificates, non-real backends labeled verified |
| Active repair goals | G120/G130 still active while admitting final root |
| Concurrent writer / dirty checkout | Fence loss; dirty or changed source tree during closeout |

## Symbols and modules

| Symbol / path | Role |
| --- | --- |
| `AuthenticatedProofReuseCurrentTreeGateV5` | Production 78-task authenticated gate |
| `build_authenticated_current_tree_repair_evidence` | Fresh PTR-169 repair evidence builder |
| `ProofTestReuseObjectiveReconciler` | Fenced three-phase outer reconciler |
| `scripts/proof_backed_test_reuse_objective_reconciliation.py` | Reconciler CLI |
| `external/ipfs_accelerate/docs/guides/TEST_PROOF_REUSE_OPERATOR_HANDOFF.md` | This handoff |

## Related documents

* `docs/architecture/TEST_PROOF_REUSE_OBJECTIVE_CLOSEOUT.md` — closeout lifecycle doctrine
* `docs/architecture/TEST_PROOF_REUSE_RUNTIME_ACTIVATION_HANDOFF.md` — historical 66-task (PTR-149) handoff; **not** production authority for the 78-task board
* `docs/guides/TEST_PROOF_REUSE_RUNBOOK.md` — operator runbook for ordinary reuse operation
