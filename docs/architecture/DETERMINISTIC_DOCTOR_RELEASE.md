# Deterministic Doctor Joined Release (LPR-042)

**Status:** terminal release gate for the Tactician-Hammer logic-repair board  
**Task:** `LPR-042`  
**Goal:** `LPR-G110`  
**Board namespace:** `agent-supervisor-tactician-hammer-logic-repair-v1`  
**Module:** `ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_release.py`

This document describes the **joined VFS + deterministic-doctor fixed-point
release**. It is the unique board sink: it depends on `LPR-028` (VFS
generalization cutover) and `LPR-041` (doctor rollout controls) and proves both
branches on the same current target tree.

Related operator surface: [Deterministic Doctor Guide](../guides/DETERMINISTIC_DOCTOR_GUIDE.md).  
Normative program plan: [Tactician-Hammer Logic Repair Plan](AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md).

## Trust boundary

**No LLM. No remote model provider. No advisory authority promotion.**

| Surface | Role at release |
| --- | --- |
| Exact roots / CIDs / sealed task identities | Authority |
| Deterministic analytical transforms under reviewed policy | Write only when gates hold |
| VFS generalization equivalence + two-profile conformance | Proof of cutover |
| Doctor benchmark dual-run receipts | Measurement + safety floors |
| KG / vector / embedding / Tactician ranking | Nominate only — never admit |
| LLM / remote model-provider routes | **Forbidden** in deterministic mode |

Release receipts are content-addressed. Metrics never become admission
authority. Mutation and completion remain unauthorized on this surface.

## What the joined gate proves

1. **Board / DAG** — exactly **43** canonical tasks, **12** goals, and
   `LPR-042` as the unique terminal sink. Semantic CIDs of `LPR-000` through
   `LPR-028` are preserved byte-for-byte against the sealed map.
2. **VFS + non-VFS profiles** — migrated IPFS Kit VFS assurance and a hermetic
   non-VFS profile both pass shared generic engines; dual-run content identities
   match (`VfsGeneralizationEquivalenceReceipt`,
   `AssuranceTwoProfileConformance`).
3. **Deterministic-doctor fixtures** — every positive and adversarial
   real-checkout fixture dual-runs with identity-equivalent CIDs/receipts
   (`DeterministicDoctorRunReceipt@1`, `DeterministicDoctorMetrics`).
4. **Cold imports** — release-critical modules import without loading optional
   providers (`openai`, `anthropic`, `transformers`, `torch`, …).
5. **Optional provider absence** — missing optional providers are actionable and
   never block **report-only** startup.
6. **Report-only no-write** — default mode is report-only; mutation is not
   authorized; a disposable probe tree is unchanged.
7. **Eligible no-model fixed point** — positive analytical cases repair all
   mandatory callers and reach a complete/reached joint fixed point with
   `PropagationCompletionReceipt@1` / `LogicFixedPointEvidenceAttachment@1`
   interface pins (no model path).
8. **Abstention + rollback** — every ambiguous/unsupported case abstains with a
   clean tree; crash/rollback restores exact authority roots.
9. **Zero safety floors** — LLM/model-provider invocation, KG/vector/embedding
   authority promotion, stale/forged cache/CID admission, missed
   caller/open-frontier mutation, sandbox/path escape, partial transaction,
   rollback failure, nondeterminism, and false completion are exactly zero.
10. **Four-lane drain** — the healthy supervisor (strict `max_lanes=4`, protected
    control-plane artifacts present) can drain the joined DAG without dependency,
    provider, protected-path, merge, or lifecycle blockage.

## Default policy

```text
mode                  = report_only
mutation_authorized   = false
completion_authoritative = false
narrow_auto           = false
llm / remote model    = false
remote embeddings     = false
network access        = false
KG/vector/embedding authority = false
dual_run_passes       = 2
```

Promotion of doctor automation remains the LPR-041 operator ladder
(report-only → plan → sandbox-auto → narrow-auto). The joined release **never**
elevates those stages and **never** enables model flags.

## API surface

| Symbol | Role |
| --- | --- |
| `DeterministicDoctorReleasePolicy` | Immutable fail-closed release policy |
| `DeterministicDoctorReleaseReceipt` | Content-addressed joined receipt |
| `validate_deterministic_doctor_release` | Full gate; returns sealed receipt |
| `replay_release_receipt` | Prove identity-equivalent reseal |
| `DeterministicDoctorReleaseValidator` | Facade for doctor/run_all |

### Minimal usage

```python
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_release import (
    validate_deterministic_doctor_release,
    replay_release_receipt,
)

receipt = validate_deterministic_doctor_release()
assert receipt.valid
assert receipt.board_terminal == "LPR-042"
assert replay_release_receipt(receipt)["identity_ok"]
```

### Validation commands

```bash
python -m pytest -q \
  test/api/test_agent_supervisor_deterministic_doctor_end_to_end.py \
  test/api/test_agent_supervisor_deterministic_doctor_replay.py
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py
python scripts/validate_tactician_hammer_logic_repair_board.py --check-all
```

## Safety floors (absolute zero)

- LLM or remote model-provider invocation
- KG / vector / embedding authority promotion
- Stale, poisoned, forged, or mismatched cache/CID admission
- Missed mandatory caller or open-frontier mutation
- Sandbox / path / trusted-base escape
- Partial transaction completion
- Rollback restoration failure
- Nondeterministic patch/receipt replay
- False fixed-point / false completion

Any nonzero floor fails the joined release. Floors are never weakened to obtain
a green receipt.

## Operator notes

* **Report-only is the release default.** Narrow auto stays off until an
  explicit, manual, monotonic LPR-041 promotion.
* **Optional datasets / embedding / prover lanes** may be absent; the doctor
  degrades or abstains. Absence does not block report-only or supervisor
  startup.
* **Protected control-plane files** (plan, objectives, todo, scheduler, board
  validator, launcher) are read-only for this task and must remain present for
  four-lane drain proofs.
* **Operational appendix tasks** beyond the sealed 43-task DAG (retry-budget or
  reconciliation repairs) do not change the canonical terminal identity of
  `LPR-042`.

## Definition of done

The program is release-complete for the joined VFS + deterministic-doctor branch
only when this gate returns a sealed, dual-run-stable
`DeterministicDoctorReleaseReceipt` with `valid=true`, every required check in
`pass`/`skip`/`warn`, zero safety floors, preserved `LPR-000`–`LPR-028` CIDs,
and no model-provider invocation.
