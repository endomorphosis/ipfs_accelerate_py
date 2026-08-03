# Proof-Directed Planner Doctor Release (PDR-092)

**Status:** terminal release gate for the Proof-directed Planner and Doctor board  
**Task:** `PDR-092`  
**Goal:** `PDR-G100`  
**Board namespace:** `agent-supervisor-proof-directed-planner-doctor-v1`  
**Module:** `ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_release.py`  
**Interface:** `PlannerDoctorReleaseReceipt@1`

This document describes the **independently replayed terminal PDR release
receipt**. It is the unique board sink: it depends on the completed producing
tasks for planning admission, runtime adoption, live fixed point, attestation,
benchmarks, epochs/refill/rollout, E2E qualification, and operations, and proves
every child goal on the same current target tree.

Related operator surface: [Proof-Directed Planner Doctor Guide](../guides/PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md).  
Normative program plan: [Proof-Directed Planner Doctor Plan](AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md).

## Trust boundary

**No automatic promotion. No completion authority. No protected-anchor rewrite.**

| Surface | Role at release |
| --- | --- |
| Exact roots / CIDs / sealed task preimages | Authority |
| Child-goal evidence artifacts reloaded from the current tree | Objective coverage |
| Protected plan / objectives / todo / scheduler / seals / holdout | Read-only anchors |
| Rollout safety floors | Absolute zero (non-compensable) |
| Optional ZKP / GPU / remote model providers | Documented only — never auto-pass |
| Automatic mode | **Off** until a later held-out current-tree operator decision |

Release receipts are content-addressed. Metrics and task-status counts never
become admission or completion authority. Mutation, completion, and automatic
promotion remain unauthorized on this surface.

## What the terminal gate proves

1. **Board / DAG** — exactly **43** canonical tasks, **11** goals, and
   `PDR-092` as the unique terminal sink under `PDR-G100`. Terminal
   dependencies match the sealed board
   (`PDR-027`, `PDR-033`, `PDR-053`, `PDR-060`, `PDR-070`–`072`,
   `PDR-080`–`082`, `PDR-090`, `PDR-091`).
2. **Source artifact reload** — every required source receipt/artifact is
   reopened from the current tree; content digests and a forest root are
   recomputed. Seal receipts must verify; forged or missing required sources
   fail closed.
3. **Child-goal coverage** — every child of `PDR-G000`
   (`PDR-G010` … `PDR-G100`) has current independently reloadable evidence
   artifacts. Objective completion is **not** inferred from completed task
   counts.
4. **Task vs objective completion** — a well-formed terminal task board is
   necessary but not sufficient; objective completion requires independent
   evidence roots for each child goal.
5. **Bad evidence rejection** — stale, synthetic, skipped, forged,
   self-authored, incomplete, and unavailable-required evidence classes are
   rejected for required surfaces.
6. **Zero safety floors** — every floor aligned with
   `PlannerDoctorRolloutPolicy@1` is exactly zero (authority/path/secret escape,
   stale/forged CID or proof admission, missed mandatory consumer, false fixed
   point / false completion, partial transaction, rollback failure, synthetic
   or skipped promotion observations, and related non-compensable counts).
7. **Exact rollback** — rollback restores identity-equivalent roots; the
   worktree adapter, live fixed-point runner, and rollout contracts remain
   present.
8. **Optional capabilities** — ZKP provers, GPU telemetry, remote model
   providers, torch/transformers, and Lean may be absent. Absence is documented
   and **never** converted into a release pass for required gates.
9. **Automatic promotion gated** — authority policy and seed operations keep
   automatic off; promotion still requires a separate later held-out
   current-tree operator decision. This receipt **does not** enable automatic.
10. **Six-lane drain** — the healthy supervisor (`max_lanes=6`, protected
    anchors present) can drain the PDR DAG without dependency, provider,
    protected-path, merge, or lifecycle blockage.
11. **Cold imports / report-only** — release-critical modules import without
    optional provider side effects; default mode is report-only and validation
    writes nothing to the target tree.
12. **Identity-equivalent replay** — replaying the same current inputs reseals
    to the same `receipt_id`.

## Default policy

```text
mode                  = report_only
mutation_authorized   = false
completion_authoritative = false
automatic_promotion   = false
doctor_mutation       = false
refill_enabled        = false
llm / remote model    = false
network required      = false
max_lanes (seed)      = 6
```

Promotion of planner/doctor automation remains the operator ladder owned by
`PDR-082` / `PDR-091` (off → observe → shadow → assist → canary → automatic).
The terminal release **never** elevates those stages and **never** enables
automatic without the separate held-out current-tree decision.

## API surface

| Symbol | Role |
| --- | --- |
| `PlannerDoctorReleasePolicy` | Immutable fail-closed release policy |
| `PlannerDoctorReleaseReceipt` | Content-addressed terminal receipt |
| `validate_planner_doctor_release` | Full gate; returns sealed receipt |
| `replay_release_receipt` | Prove identity-equivalent reseal |
| `PlannerDoctorReleaseValidator` | Facade for doctor/run_all |
| `classify_evidence_disposition` | Reject bad evidence classes |

### Minimal usage

```python
from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_release import (
    validate_planner_doctor_release,
    replay_release_receipt,
)

receipt = validate_planner_doctor_release()
assert receipt.valid
assert receipt.board_terminal == "PDR-092"
assert receipt.automatic_promotion_enabled is False
assert receipt.completion_authoritative is False
assert replay_release_receipt(receipt)["identity_ok"]
```

### Validation commands

```bash
python -m pytest -q \
  test/api/test_agent_supervisor_planner_doctor_release.py \
  test/integration/test_agent_supervisor_planner_doctor_e2e.py
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py doctor
```

## Safety floors (absolute zero)

Aligned with `PlannerDoctorRolloutPolicy@1` / `SAFETY_FLOOR_METRICS`:

- Authority, policy, scope, secret, or path escape
- Stale cache / forged CID / forged proof admission
- Missed mandatory consumer or falsely closed impact frontier
- SecurityIR / IntentIR prohibition misses
- Hidden oracle access or benchmark/denominator mutation
- Partial transaction, false fixed point, rollback failure, false completion
- Synthetic or skipped observation used for promotion

Any nonzero floor fails the terminal release. Floors are never weakened to
obtain a green receipt.

## Rejected evidence classes

Required surfaces fail closed on:

| Class | Meaning |
| --- | --- |
| `stale` | Evidence not bound to current roots |
| `synthetic` | Fabricated / fixture-as-live evidence |
| `skipped` | Skipped observation treated as success |
| `forged` | Content identity mismatch or unsealed claim |
| `self_authored` | Candidate/model self-seal |
| `incomplete` | Partial required bundle |
| `unavailable_required` | Missing required artifact |

Optional capability absence uses `unavailable_optional` and is documented
without counting as a positive qualification.

## Operator notes

* **Report-only is the release default.** Automatic stays off until an
  explicit later held-out current-tree operator decision (outside this receipt).
* **Optional datasets / embedding / prover / GPU lanes** may be absent; the
  supervisor degrades or abstains. Absence does not block report-only release.
* **Protected control-plane files** (plan, objectives, todo, scheduler, authority
  and benchmark seals, holdout manifest, threat model) are read-only for this
  task and must remain present for six-lane drain proofs.
* **Task completion ≠ objective completion.** Completing every task heading is
  not sufficient; each child goal must still present current independent
  evidence artifacts.

## Definition of done

The PDR program is release-complete for the terminal gate only when this module
returns a sealed, dual-run-stable `PlannerDoctorReleaseReceipt` with
`valid=true`, every required check in `{pass, skip, warn}`, zero safety floors,
exact-root rollback, full child-goal coverage without task-count authority,
documented optional-capability gaps that do not convert to pass, and
`automatic_promotion_enabled=false` pending the separate held-out operator
decision.
