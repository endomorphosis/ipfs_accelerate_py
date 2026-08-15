# LPC-071 Separate Advisor Proposals from Proof Authority

**Task:** LPC-071 — Separate advisor proposals from proof authority  
**Goal:** LPC-G070  
**Depends on:** LPC-070 (`notes/tactician_plan_model.md`)  
**Plan model:** `CanonicalProofPlan@1` / `ipfs_datasets_py.logic.tactician@1`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py -q`

## Purpose

Advisors, learned models, nomination sources, and plan alternatives may
**propose** work: routes, subgoals, candidates, lemmas, tactics, repairs, and
ranked alternatives. They may **not** exercise proof, completion, admission,
or production authority.

This note freezes the advisor authority ceiling for the tactician track.
Executable adversarial coverage lives in
`ipfs_datasets_py/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py`.

## Authority split

| Actor | May do | Must not do |
| --- | --- | --- |
| **Advisor / model / nomination source** | Emit proposal-class candidates and plans | Claim proof, raise authority, choose verification keys, skip reconstruction, approve production, silently add assumptions, drop blocking obligations |
| **Domain-neutral tactician** (`LogicTactician`) | Order sources, exclude denied classes, decompose gaps, record stop/abstain | Set `semantic_authority=True`, prove, write, network, execute proofs |
| **Independent reconstruction / kernel** | Check reconstructed proof objects | Accept advisor prose or confidence as proof |
| **Receipt admission (LPC-111 ten-point floor)** | Admit evidence that meets every gate | Infer admission from advisor success alone |
| **`GoalCompletion`** | Record completion only with elevated authority + evidence/receipts | Be minted by a plan, candidate, or advisor result |

Default advisor / proposal ceiling is at most:

```text
none < advisory < candidate
```

Theorem, attestation, reconstruction, and production admission are **achieved**
by independent evidence and policy, never **declared** by a proposal.

## Forbidden advisor actions (acceptance)

LPC-071 acceptance requires that every of the following fail closed.

### 1. Mark proposals proved

Proposal-class records must keep proof/completion claims false (or absent).

| Forbidden claim | Live rejection |
| --- | --- |
| `proof_claimed=True` | Software-verification contracts (`EndGoalSpec`, `FormalGoal`, `ProofHole`, `CandidateProofStep`, `CandidateValidation`, plan steps); hard prune `PROOF_CLAIM` |
| `completion_claimed=True` | Same surfaces; hard prune `COMPLETION_CLAIM` |
| `proved` / `is_proved` / `kernel_verified` / `admitted` | `_PROPOSAL_FORBIDDEN_TRUE_CLAIMS`, proposal-advisor `_AUTHORITY_CLAIM_KEYS` |
| Plan self-status `proved` / `complete` | Plan lifecycle is proposal-only (LPC-070 status dimension) |
| `ProposalCandidate.is_proved` | Always `False`; confidence never yields proof |

Only `GoalCompletion` with elevated authority and evidence/receipts may record
a completion verdict. Advisors never mint that contract.

### 2. Raise authority

| Forbidden promotion | Live rejection |
| --- | --- |
| `semantic_authority=True` on plan/policy/receipt | `TacticianPlan` / `TacticianPolicy` / `TacticianReceipt` validation |
| Metadata keys `semantic_authority`, `proof_authority`, `expectation_authority`, `write_authority`, `authoritative` | `_AUTHORITY_PROMOTION_KEYS` / `_reject_authority_promotion` |
| `authority` above `candidate` on candidates | `CandidateProofStep` cap; proposal-only source cap; `ProposalCandidate.authority` fixed to `unverified_candidate_only` |
| Advisor role + any ceiling as certified authority | `role_can_satisfy_certified_authority(ADVISOR, …) is False` |
| Advisor/candidate tool ceiling above advisory/candidate | `FormalVerificationToolRole` role/ceiling consistency |

Provider or advisor **success** still never implies proof authority (LPC-032).

### 3. Choose verification keys

Verification keys, circuit identities, CRS digests, and attestation key
bindings are **policy/backend-owned**. Advisors cannot select or substitute
them.

| Rule | Enforcement |
| --- | --- |
| Proposal candidates reject unknown fields | `ProposalCandidate.from_dict` closed field set — `verification_key_id` / `vk_id` / similar are not admitted |
| Authority payload cannot smuggle verification claims | `_reject_authority_payload` on candidate metadata |
| Attestation binds keys from backend policy | `ProofReceiptAttestation` / backend policy; not from advisor text |
| Cache/corpus verification keys are reviewed bindings | Proof store / verification-key schema; advisor prose is not a key source |

An advisor may **nominate** that verification is needed. It may not pick the
key that will be trusted.

### 4. Skip reconstruction

When a goal or policy requires reconstruction or independent kernel check,
advisor output cannot bypass that stage.

| Rule | Enforcement |
| --- | --- |
| Ten-point admission requires reconstruction/kernel when mandated | Plan §8 point 7; LPC-111 |
| Advisor role never certifies without independent reconstruction | Toolchain role matrix; `independent_reconstruction_required` on advisor/candidate tools where declared |
| Plan reconstruction policy is skip-forbidden for production-affecting work | LPC-070 `reconstruction.skip_forbidden` |
| Solve-only / model-success ≠ reconstruction | LPC-032 non-inference; candidate authority remains candidate |

Setting `skip_reconstruction=True` (or equivalent) on a proposal is an
authority-elevation attempt and fails closed.

### 5. Approve production

| Rule | Enforcement |
| --- | --- |
| Catalog presence ≠ production admission | `presence_implies_production_admission()` hard zero |
| Family presence never admits production | `is_production_admitted(...)` always `False` from presence alone |
| Advisor / candidate results are not production certificates | Role matrix; proposal authority ceiling |
| Plans cannot self-approve production readiness | Completeness boundary `plan_may_claim_*=false` (LPC-070) |

Production approval is an operator/policy gate over admitted evidence, not an
advisor flag.

### 6. Silently add assumptions

| Rule | Enforcement |
| --- | --- |
| Goal assumptions are an explicit closed list | `TacticianGoal.assumptions` |
| Planner does not invent assumptions from sources | `LogicTactician.plan` copies goal gaps/roots; assumptions stay on the goal binding |
| New step assumptions must be named | `new_assumption_ids` on candidate/plan steps |
| Assumption-heavy plans pay cost / bound | Ranking `max_new_assumptions` hard prune |
| Hypothetical assumptions cannot claim elevated authority | `AssumptionBinding` class rules |

Ranking, scheduling, and advisor nomination may **reference** assumptions only
when they already appear in the closed set (or are newly listed as explicit
`new_assumption_ids`). Silent injection fails closed.

### 7. Drop blocking obligations

| Rule | Enforcement |
| --- | --- |
| Required obligations must remain covered | Ranking hard prune `MISSING_COVERAGE` against `required_obligation_ids` |
| Blocking proof gaps survive planning | `TacticianPlan.proof_gaps` equals goal gaps; subgoals address gaps rather than erase them |
| Fallbacks preserve blocking obligations | LPC-070 fallback rule `preserves_blocking_obligations` |
| Completeness lists blocking assumptions/obligations | LPC-070 completeness boundary |

An advisor alternative that omits a required/blocking obligation is hard-pruned,
not soft-scored away.

## Proposal envelope (normative shape)

For admission checks, treat every advisor/model/nomination output as a
**proposal envelope** with at least these fail-closed fields:

| Field | Advisor default | Forbidden advisor value |
| --- | --- | --- |
| `proof_claimed` | `false` | `true` |
| `completion_claimed` | `false` | `true` |
| `semantic_authority` / authority ceiling | `false` / ≤ `candidate` | `true` / theorem·attestation·reconstruction·… |
| `verification_key_id` (or equivalent) | absent / empty | any self-chosen key identity |
| `skip_reconstruction` | `false` | `true` when reconstruction is required |
| `production_approved` | `false` | `true` |
| `silent_assumptions` / undeclared assumption ids | empty | any undeclared addition |
| `dropped_blocking_obligations` | empty | any required/blocking id removed |

Executable admission helper (unit suite only; not a second planner):

* `AdvisorProposalEnvelope` — closed proposal shape for adversarial checks
* `advisor_proposal_rejection_reasons(...)` — ordered fail-closed reason codes
* `advisor_proposal_is_admissible(...)` — true only when the reason tuple is empty

Admissible envelopes may still be **wrong** as proofs. Admissibility only means
the proposal stayed within the advisor ceiling; independent validation remains
mandatory. Advisors cannot mark proposals proved, raise authority, choose
verification keys, skip reconstruction, approve production, silently add
assumptions, or drop blocking obligations.

## Live enforcement anchors

| Surface | Module | What it freezes |
| --- | --- | --- |
| Domain-neutral plan/policy/receipt | `logic/tactician/{models,policy,planner,receipts}.py` | `semantic_authority=False`; no proof/write/network; content-addressed plans |
| Authority-promotion metadata | `models._AUTHORITY_PROMOTION_KEYS` | Metadata cannot smuggle authority |
| Software-verification contracts | `logic/software_verification/tactician/contracts.py` | Proposal forbidden true claims; candidate authority cap; `GoalCompletion` only for completion |
| Plan ranking hard prune | `logic/software_verification/tactician/proof_plan.py` | Proof/completion claims, missing coverage, insufficient authority |
| Candidate synthesis | `logic/software_verification/tactician/candidate_synthesis.py` | Proposal-only sources stay ≤ candidate |
| Leanstral/SymAI proposal advisors | `logic/formalization/proposal_advisors.py` | `unverified_candidate_only`; confidence never proves; closed fields |
| Toolchain roles | `logic/backends/toolchain_roles.py` | Advisor/candidate never certify |
| Catalog production floor | `logic/families/canonical_catalog.py` | Presence never admits production |
| Success ≠ proof | LPC-032 / `ir_core.axes` | Operation success is not authority |

## Pipeline position

```text
Advisor / model / nomination
  → proposal envelope (≤ candidate, no proof claims)
  → CanonicalProofPlan@1 / TacticianPlan / ranked alternatives
  → supervisor scheduling (reorder valid lanes only; no rewrite)
  → provider ops (LPC-052 untrusted default)
  → reconstruction + kernel checks (not skippable by advisor)
  → ten-point receipt admission (LPC-111)
  → GoalCompletion / production policy (not advisor-owned)
```

## Relationship to LPC-070

LPC-070 freezes the plan field inventory and completeness boundary. LPC-071
freezes the **actor authority** of anything that only proposes into that model:

1. Plans describe reconstruction, assumptions, obligations, and completeness.
2. Advisors may fill proposal slots inside that model.
3. Advisors may not rewrite those slots into proof, skip, production approval,
   silent assumptions, or dropped blockers.
4. Completion remains outside the plan (`GoalCompletion` + admitted receipts).

## What this does **not** do

1. Does not ban advisors from existing — they remain useful nomination sources.
2. Does not make reconstruction optional when policy requires it.
3. Does not let high model confidence substitute for kernel checks (LPC-032).
4. Does not redefine cache keys (LPC-080) or repository APIs (LPC-081).
5. Does not authorize the supervisor to rewrite obligation identity when
   reordering lanes.

## Acceptance checklist (LPC-071)

| Criterion | Satisfied by |
| --- | --- |
| Advisors cannot mark proposals proved | Contract/plan/candidate proof-claim rejection; `is_proved` always false |
| Advisors cannot raise authority | `semantic_authority` fixed false; candidate authority cap; advisor role non-certifying |
| Advisors cannot choose verification keys | Closed proposal fields; attestation/policy-owned keys |
| Advisors cannot skip reconstruction | Reconstruction required when mandated; skip flag fails closed |
| Advisors cannot approve production | Production admission hard zero; no production_approved claim |
| Advisors cannot silently add assumptions | Explicit assumption sets; undeclared additions rejected |
| Advisors cannot drop blocking obligations | Required-coverage hard prune; plan gaps preserved |

## Validation

```bash
python -m pytest ipfs_datasets_py/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py -q
```

The focused suite adversarially attempts each forbidden action against live
tactician, contract, advisor, role, and catalog surfaces and asserts fail-closed
rejection.
