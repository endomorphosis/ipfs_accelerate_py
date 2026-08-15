# LPC-111 Enforce supervisor admission of receipts

**Task:** LPC-111 — Enforce supervisor admission of receipts  
**Goal:** LPC-G110  
**Depends on:** LPC-110 (`SupervisorLogicPlatformClient@1`)  
**Interface:** `SupervisorLogicPlatformReceiptAdmission@1`  
**Evidence module:** `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_admission.py`  
**Tests:** `test/api/test_logic_platform_admission.py`  
**Acceptance:** A result may affect completion or merge only after structural validity, content identity, source/tree/environment/policy binding, translation chain, evidence kind, authority ceiling, required reconstruction, freshness, non-simulation, and policy admission.  
**Conflict policy:** Own admission helper and tests.  
**Validation:**  
`python -m pytest test/api/test_logic_platform_admission.py -q`

## Purpose

`SupervisorLogicPlatformClient@1` (LPC-110) projects provider receipts as
**untrusted** envelopes (`admitted=False`, `trusted=False`). LPC-111 owns the
fail-closed **ten-point admission gate** that decides whether any such result
may influence **completion** or **merge**.

Provider / operation success is never sufficient (LPC-032). Partial check
passes never promote authority. `may_affect_completion` and `may_affect_merge`
are locked to full admission.

## Canonical call path

```text
Supervisor caller / merge or completion gate
  → SupervisorLogicPlatformClient@1.receipts(...)   # untrusted projection
  → SupervisorLogicPlatformReceiptAdmission@1
       admit_receipt(receipt, AdmissionContext)
         1..10 ten-point floor (all required)
  → only when admitted=True may the result affect completion or merge
```

Importing `logic_platform_admission` is quiet: no network, install, probe, or
`ipfs_datasets_py` load.

## Surface

| Symbol | Role |
| --- | --- |
| `SupervisorLogicPlatformReceiptAdmission` | Stable interface object |
| `SUPERVISOR_LOGIC_PLATFORM_RECEIPT_ADMISSION_INTERFACE` | Exact id `SupervisorLogicPlatformReceiptAdmission@1` |
| `AdmissionContext` | Immutable binding of task/tree/policy/operation/authority floors |
| `AdmissionCheck` | Closed ten-point check vocabulary |
| `AdmissionResult` | Admitted/rejected + ordered check trace |
| `admit_receipt(receipt, context)` | Evaluate one untrusted receipt |
| `admit_receipts(receipts, context)` | Evaluate a sequence |
| `may_affect_completion_or_merge(receipt, context)` | True only after full admission |
| `get_receipt_admission()` | Process-local factory |

Interface / schema constants:

* `SupervisorLogicPlatformReceiptAdmission@1`
* Schema: `ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-admission@1`
* Context schema: `ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-admission-context@1`
* Result schema: `ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-admission-result@1`
* Task / goal binding: `LPC-111` / `LPC-G110`

## Ten-point floor (normative)

A proof result may influence completion or merge **only when all hold**
(plan §8). Checks run in this fixed order; every check is recorded even after
an early failure so the rejection is auditable.

| # | Check id | Requirement |
| --- | --- | --- |
| 1 | `structural_validity` | Receipt is an object (or `ProofReceipt`) with identity fields |
| 2 | `content_identity` | Content / receipt id is well-formed and matches expected when pinned |
| 3 | `source_tree_environment_policy_binding` | Source / tree / environment / policy bindings match the context |
| 4 | `translation_chain` | Translation chain is valid (or explicitly bound / not required) |
| 5 | `evidence_kind` | Evidence kind supports the claimed semantic verdict |
| 6 | `authority_ceiling` | Authority ceiling meets the required floor |
| 7 | `required_reconstruction` | Required reconstruction / kernel checks passed |
| 8 | `freshness` | Receipt and evidence are current (not stale / unknown) |
| 9 | `non_simulation` | Receipt and evidence are not simulated |
| 10 | `policy_admission` | Supervisor policy admits that authority for the requested operation |

### Conjunction rule

```text
admitted ⇔ ∀ check ∈ TEN_POINT_CHECKS : check.passed
may_affect_completion ⇔ admitted
may_affect_merge      ⇔ admitted
```

There is no partial-admit disposition that can affect completion or merge.

## AdmissionContext bindings

| Binding | Required | Rule |
| --- | --- | --- |
| `task_id` | yes | Calling supervisor task identity |
| `repository_tree_id` | yes | Must match receipt tree |
| `policy_id` | yes | Must match receipt policy |
| `operation` | yes | Requested operation the policy is asked to admit |
| `required_authority` | yes (default `kernel_verified`) | Floor the receipt must meet |
| `repository_id` | when known | Mismatch fails closed |
| `environment_id` | when known | Mismatch fails closed |
| `source_id` | when known | Mismatch fails closed |
| `policy_revision` | when known | Mismatch fails closed |
| `plan_id` / `obligation_id` | when scoped | Mismatch fails closed |
| `expected_content_id` | when pinned | Mismatch fails closed |
| `require_reconstruction` | default true | Kernel-required policies cannot disable it |
| `require_kernel` | default true | Kernel acceptance required for kernel floors |
| `network_allowed` | default false | Receipt cannot claim network when policy denies |

## Receipt inputs accepted

1. **`ProofReceipt`** — preferred typed contract; identity, assurance, and
   verdict are re-derived from evidence (provider claims ignored for authority).
2. **Mapping envelopes** — including LPC-110 projections
   `{ "receipt": {...}, "admitted": false, ... }`; nested body is unwrapped.
3. **Malformed input** — non-objects fail `structural_validity` and reject.

## Non-inference / fail-closed rules

1. **Success ≠ admission.** `operation_status=succeeded` never satisfies any
   ten-point check by itself and cannot raise authority.
2. **Candidate / advisory / simulated evidence** cannot satisfy kernel-required
   policy admission (LPC-032).
3. **Stale or unknown freshness** always fails point 8.
4. **Heuristic translation classes** never support kernel/completion influence.
5. **Missing reconstruction or kernel checks** fail point 7 when required.
6. **Policy operation mismatch or network overclaim** fails point 10.
7. **Partial passes never set** `may_affect_completion` or `may_affect_merge`.

## Authority lattice (comparison only)

```
unverified < candidate < solver_checked < kernel_verified < attested
```

Aliases `unknown`/`none` → `unverified`; `advisory`/`simulated` → `candidate`.
Rank comparison is not inference: success still cannot promote a level.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-110 | Client returns untrusted receipts; LPC-111 admits them |
| LPC-032 | Success is not proof; admission never consults success as authority |
| LPC-052 | Provider response envelopes remain typed; admission is a separate gate |
| LPC-100 | Manifest handshake is orthogonal; compatibility ≠ proof authority |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/receipt_admission.md` | This contract note (declared output) |
| `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_admission.py` | Admission helper (evidence / predicted) |
| `test/api/test_logic_platform_admission.py` | Focused regression suite (implied validation) |

Task-owned proposal envelope for LPC-111 (fail closed):

* **Declared Outputs:** this note.
* **Predicted / validation files:** admission module + admission tests + this note.
* Paths outside that envelope (client module, daemon, protected plan files) are
  **out of scope** for LPC-111 admission.
* Do not absorb admission into the LPC-110 client proposal.

## What this task does **not** do

* Does not create another supervisor, daemon, merge train, or scheduler.
* Does not redefine family, property, profile, notation, or catalog identity.
* Does not treat provider success, handshake compatibility, or catalog presence
  as proof authority.
* Does not weaken LPC-032 non-inference rules.
* Does not edit protected board/plan/validator files.
* Does not require Git, sibling checkouts, or monorepo layout.

## Acceptance

- Interface id is exactly `SupervisorLogicPlatformReceiptAdmission@1`.
- Ten checks match plan §8 vocabulary and fixed order.
- A result may affect completion or merge **only** after all ten checks pass.
- Structural invalidity, content-identity failure, binding mismatch, invalid
  translation, unsupported evidence kind, inadequate authority, missing
  reconstruction/kernel, staleness, simulation, and policy denial each reject.
- Provider success alone never admits.
- Validation:  
  `python -m pytest test/api/test_logic_platform_admission.py -q`
