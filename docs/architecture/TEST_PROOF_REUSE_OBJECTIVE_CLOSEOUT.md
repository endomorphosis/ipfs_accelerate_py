# Proof-Backed Test Reuse: Objective Closeout and Operator Handoff

Status: normative operator closeout contract  
Interfaces: `ProofTestReuseObjectiveReconciler@1`, `ObjectiveCloseoutReceipt@1`, `ProofTestReuseCurrentTreeGateDecision@1`, `ProofTestReuseObjectiveEvidenceBundle`  
Evidence: hermetic closeout receipts (PTR-130), live state-root candidate (operator-owned)  
Controller: `scripts/proof_backed_test_reuse_supervisor.py closeout`  
Reconciler: `scripts/proof_backed_test_reuse_objective_reconciliation.py`

## Purpose

Define how the sealed 41-task implementation board and the 12-goal objective
heap reach **verified** goal state through a fenced, three-phase reconciliation,
and how that work is handed to a human operator for the **only** step that may
mutate the protected objective heap.

This document is the operator handoff produced by **PTR-130**. Completing
PTR-130 (and the rest of the board) is a precondition for invoking live
closeout. **Task completion precedes, and does not itself constitute, the live
operator closeout.**

## Non-negotiable doctrine

1. **Worker lanes never reconcile goals.** Implementation lanes run with goal
   reconciliation disabled. Only the outer controller's fenced closeout writer
   may advance lifecycle state, and only into a **state-root candidate**.
2. **Three phases, never skipped.** Provisional → children verified → final
   root verified. No phase may invent a later state.
3. **Verification is evidence-bound, not label-bound.** `Status: completed` on
   a todo item is not goal authority. Ordinary skips, simulated proofs, stale
   or forged artifacts, and missing optional backends never verify a goal.
4. **The protected objective heap is operator-owned.** Closeout writes a
   candidate under the XDG state root. Promoting that candidate into
   `implementation_plan/docs/46-proof-backed-test-reuse.objectives.md` requires
   an explicit human commit after review.
5. **PTR-130 is proof of mechanism, not live root completion.** The hermetic
   e2e in `test_proof_test_reuse_objective_closeout_e2e.py` exercises a
   disposable exact population. It must not be confused with closing the live
   current-tree program.

## Lifecycle (three staged reconciliations)

| Phase | Name | Legal transitions | Must not |
| --- | --- | --- | --- |
| 1 | Provisional | Drained goals → `provisionally_complete` | Claim `verified_complete` |
| 2 | Children | After current validation (proof reuse **off**), `PTR-G010`…`PTR-G100` → `verified_complete` | Verify `PTR-G110` or `PTR-G000` |
| 3 | Final root | Admit final current-tree gate evidence, then `PTR-G110` → `verified_complete`, then `PTR-G000` → `verified_complete` | Verify G000 before G110; skip gate admission |

Every refresh recomputes per-goal bindings. Bounded replay converges.
Interruption resumes from durable phase checkpoints. Mutation or contradiction
reopens affected ancestors and dependents. Missing optional services produce
typed nonterminal gaps (`retain_typed_gap_and_continue_tests`) rather than
blocking ordinary tests or supervisors.

### Fail-closed classes (never verify)

The following inputs never produce verified goals or authoritative completion
evidence:

| Class | Examples |
| --- | --- |
| Missing | Absent task, goal, evidence CID, gate artifact, supervisor-health receipt |
| Stale | Tree-mismatched or age-expired gate/evidence/health/benchmark packets |
| Forged | Tampered retained bytes, forged completion decisions, multihash mismatch |
| Noncanonical | Legacy `sha256:` labels, CIDv0, wrong codec, uppercase base32, path escapes |
| Mismatched | Git tree ≠ forest ≠ objective-completion identity; policy/capability/key drift |
| Quorum-short | Fewer than two independent exhaustive/audit quorum members |
| Validation-failed | Declared validation with `IPFS_TEST_PROOF_REUSE_MODE=off` not passed |
| Ordinary-skip | Pytest skip text or disposition `ordinary_skip` as task authority |
| Simulated-proof | `authority=simulated`, mock/demo certificates, non-real backends labeled verified |
| Unavailable backend without real fixture | Groth16/ProveKit/snarkjs/IPFS/cache absent and no reviewed real-certificate fixture |
| Tree-mutated | Dirty or changed checkout during closeout; concurrent writer fence loss |
| Restart-interrupted | Partial checkpoint without successful phase completion leaving the live heap verified |

## Preconditions for live closeout

Before an operator invokes live closeout on the integration checkout:

1. **Every implementation task is closed** on the validated board
   (`PTR-000` … `PTR-130`, 41 tasks). Open tasks refuse closeout.
2. **PTR-130 is complete** (this runbook and the hermetic e2e exist and pass
   with proof reuse off). Task completion is necessary and **not sufficient**.
3. **Checkout is clean** on `agent/proof-backed-test-reuse` (or the reviewed
   integration branch), with the expected tree identity.
4. **Supervisor lanes are healthy and work-complete**, so a fresh
   current-tree/config-bound three-lane supervisor-health receipt can be
   captured.
5. **Final-gate premises are retained** (task provenance, G010–G100 evidence,
   adversarial/analyzer populations, benchmark, rollout readiness,
   supervisor health) and replayable by canonical CID.
6. **No concurrent closeout writer** holds the fence.

### Genuine approvals required for historical provenance

The merge queue may lack managed-merge receipts for early planning and review
tasks. The following task IDs require **genuine operator or reviewer
provenance** (not `Status: completed` alone, and not an ordinary queue row):

| Task | Accepted provenance kind | What the operator must supply or reconfirm |
| --- | --- | --- |
| `PTR-000` | `operator_planning_seal` | Planning seal CID + operator approval CID bound to the sealed objective revision |
| `PTR-001` | `operator_reviewed_integration` | Reviewed integration receipt + operator review CID targeting the integration commit |
| `PTR-011` | `operator_reviewed_integration` | Same reviewed-integration shape as PTR-001 |
| `PTR-041` | `retrospective_integration_verification` | Verified Git ancestry to the current commit **and** a current proof-reuse-off rerun **and** an immutable policy approval |

Retrospective provenance is fail-closed: failed ancestry, failed current-tree
rerun, policy mismatch, or missing approval never closes the task for the gate.

All other sealed tasks normally carry `managed_merge` provenance from the
serial merge queue (merge succeeded, receipt CID present). Quarantine,
unsuccessful merges, and unknown provenance kinds never count as complete.

## Operator procedure (live current-tree closeout)

> **Reminder:** Finishing PTR-130 (or seeing green hermetic e2e) is **not**
> live closeout. Only the sequence below, ending in an explicit operator
> commit of the protected objective file, closes the live root.

### 1. Report-only diagnosis (no writes)

From a clean integration worktree:

```bash
IPFS_TEST_PROOF_REUSE_MODE=off \
  python3 scripts/proof_backed_test_reuse_supervisor.py closeout --report-only
```

Or invoke the reconciler directly with `--report-only` and the state-root
paths declared in `config/proof_backed_test_reuse_supervisor.json`
(`objectiveProjection.*PathSuffix`). Report-only must not write the
repository or emit a candidate.

### 2. Fenced three-phase closeout

```bash
IPFS_TEST_PROOF_REUSE_MODE=off \
  python3 scripts/proof_backed_test_reuse_supervisor.py closeout
```

Expected effects (state root only):

- writer fence acquired (compare-and-swap);
- phase-1 provisional transitions for drained goals;
- current validation rerun with proof reuse **off**;
- phase-2 verification of `PTR-G010` … `PTR-G100`;
- final-gate admission (no G110 self-premise; producing task `PTR-122`);
- phase-3 verification of `PTR-G110` then `PTR-G000`;
- candidate objective + lifecycle projection + closeout status JSON under the
  configured state-root completion directory;
- `operator_commit_required: true` and `repository_written: false`.

If any refuse class fires, fix the underlying evidence and re-run. Do not
hand-edit goal statuses in the protected objectives file to force green.

### 3. Explicit operator commit (the actual closeout)

1. Diff the candidate against the live
   `implementation_plan/docs/46-proof-backed-test-reuse.objectives.md`.
2. Confirm every goal binding, optional capability gap, and receipt set.
3. Confirm genuine approvals for `PTR-000`, `PTR-001`, `PTR-011`, and
   `PTR-041` are still valid for the **current** tree and policy.
4. Copy or apply the candidate into the protected objectives file **only**
   via an operator-owned commit (no worker lane, no autonomous refill).
5. Record the commit id next to the closeout status artifact.
6. Restart ordinary lanes only after the protected heap commit is on the
   integration branch.

Until step 4–5 succeed, the live root remains unverified even if every
implementation task (including PTR-130) is marked completed.

### 4. After closeout

- Keep worker-lane goal reconciliation **disabled**.
- Treat optional missing Groth16 / ProveKit / snarkjs / IPFS / shared cache as
  typed gaps, not startup blockers.
- Any later tree mutation, contradiction, or false authoritative skip reopens
  affected goals; do not re-close from task labels alone.

## Hermetic proof (PTR-130)

Validation (no network, no test-file registry):

```bash
IPFS_TEST_PROOF_REUSE_MODE=off \
  python3 -m pytest \
  external/ipfs_accelerate/test/api/test_proof_test_reuse_objective_closeout_e2e.py -q
```

The e2e builds a disposable exact 41-task board and 12-goal heap under a
temporary git root, drives the reconciler through all three phases, evaluates
the final current-tree gate on the sealed population (with genuine provenance
shapes for the historical tasks above), and asserts that the fail-closed
classes listed in this runbook never verify. Optional capability absence is
retained as non-blocking typed gaps while closeout still converges on the
disposable candidate.

## Relationship to other artifacts

| Artifact | Role |
| --- | --- |
| `46-proof-backed-test-reuse.todo.md` | Machine board; protected from agents |
| `46-proof-backed-test-reuse.objectives.md` | Protected goal heap; operator commit only |
| `config/proof_backed_test_reuse_supervisor.json` | Lane/profile + closeout path suffixes |
| `proof_test_reuse_current_tree_gate.py` | Final gate authority (PTR-122) |
| `proof_test_reuse_objective_evidence.py` | Bound evidence assembly (PTR-120) |
| `proof_backed_test_reuse_objective_reconciliation.py` | Three-phase reconciler (PTR-121) |
| This runbook + hermetic e2e | Operator handoff proof (PTR-130) |

## Definition of done (operator)

Live closeout is done only when:

1. Hermetic PTR-130 validation is green under `IPFS_TEST_PROOF_REUSE_MODE=off`.
2. Live closeout produces a passed candidate with
   `operator_commit_required: true`.
3. The operator has committed the candidate into the protected objectives file
   on the clean integration tree.
4. G010–G100, G110, and G000 are `verified_complete` **in that committed
   heap**, not merely in a disposable test or a state-root draft.

Until (3) and (4), the program remains open at the root regardless of task
board status.
