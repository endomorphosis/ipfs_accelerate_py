# LPC-071 Separate Advisor Proposals from Proof Authority

**Task:** LPC-071 — Separate advisor proposals from proof authority  
**Goal:** LPC-G070  
**Depends on:** LPC-070 (`notes/tactician_plan_model.md`)  
**Interface:** `CanonicalProofPlan@1` (proposal-class only)  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py -q`

## Purpose

Advisors, learned models, hammers, and plan rankers may **propose** routes,
candidates, repairs, and missing-proof plans. They may **not** mint proof or
completion authority. Proof, kernel verification, production admission, and
merge influence remain independent of proposal metadata.

This note freezes that boundary for LPC-G070. Executable adversarial coverage
lives in
`ipfs_datasets_py/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py`.

## Ownership

| Owner | Responsibility |
| --- | --- |
| **Datasets** (`logic/tactician`) | Domain-neutral plan/policy/receipt records fixed at `semantic_authority=False` |
| **Datasets** (`logic/software_verification/tactician`) | Proposal contracts, candidate portfolio, reconstruction graphs, ranking hard-prunes |
| **Datasets** (`logic/formalization`) | Bounded formalization / Leanstral / SymAI advisors; no silent assumption mutation |
| **Datasets** (`logic/backends`) | Advisor role ceilings; advisor execution gate; success ≠ proof |
| **Supervisor** | Scheduling and ten-point receipt admission (LPC-111); never promotes advisor text |

Conflict policy (LPC-071): own the authority test and the smallest policy
fix. Do not invent a second planner or a second authority lattice.

## Proposal vs proof (normative)

```text
Advisor / model / hammer / ranker
  → proposal-class artifacts only
       (plans, candidates, repairs, ranked alternatives)
  → independent reconstruction / kernel / validation
  → ten-point receipt admission (LPC-111)
  → GoalCompletion / merge influence
```

No arrow from a proposal field to theorem, attestation, production, or
completion authority is permitted. Provider or advisor **success** never
crosses the boundary (LPC-032, LPC-052).

## Closed authority ceilings for proposals

| Artifact class | Max self-authority | May claim proof? | May claim completion? |
| --- | --- | --- | --- |
| `TacticianPlan` / `TacticianPolicy` / `TacticianReceipt` | none (`semantic_authority=False`) | no | no |
| `EndGoalSpec@1` | advisory / candidate / declarative | no | no |
| `CandidateProofStep@1` | candidate | no | no |
| `GoalDirectedProofPlan@1` | below theorem / attestation | no | no |
| `ProofCandidatePortfolio` learned sources | candidate (proposal-only) | no | no |
| Formalization / Leanstral / SymAI candidates | `unverified_candidate_only` | no | no |
| `AdvisorProviderEvidence@2` | candidate result authority | no | no |
| Tool role `advisor` | advisory ceiling | cannot certify | n/a |

Completion is a **separate** contract (`GoalCompletion` / admitted receipts),
never a plan or candidate self-status.

## Acceptance matrix (fail closed)

LPC-071 acceptance requires that advisors **cannot** perform any of the
following. Each row names live enforcement anchors exercised by the unit test.

| Forbidden advisor action | What it would try to do | Enforcement anchors |
| --- | --- | --- |
| **Mark proposals proved** | Set `proof_claimed`, `proved`, `kernel_verified`, `is_proved`, or `semantic_authority=True` on proposal artifacts | `_PROPOSAL_FORBIDDEN_TRUE_CLAIMS`; `CandidateProofStep` / `GoalDirectedProofPlan` / `EndGoalSpec` validation; `TacticianPlan` / `TacticianReceipt` authority flags; `collect_hard_failures` `PROOF_CLAIM`; formalization `_reject_authority_claims` |
| **Raise authority** | Advertise theorem, attestation, reconstruction, or certified ceilings from advisor output | Authority caps on candidates and plans; `cap_experimental_authority`; `_cap_authority_for_source` / `proposal_only`; `role_can_satisfy_certified_authority(ADVISOR, …)=False`; `advisor_never_establishes_proof`; `confidence_never_yields_proof` |
| **Choose verification keys** | Smuggle `verification_key`, `verification_key_id`, or verifier-binding fields into proposal metadata so a certificate selects its own key | Closed wire fields (`_reject_unknown`); proposal/advisor metadata authority-key rejection (`verification_status` / `verification_result`); verifier-side `TestPassCircuitBinding` is never populated from certificate metadata |
| **Skip reconstruction** | Bypass required reconstruction / kernel stages via experimental methods or skip flags | Experimental reconstruction methods stay non-trusted (`is_experimental_reconstruction`, `cap_experimental_authority`); ten-point point 7 (`required_reconstruction`); plans describe reconstruction requirements but cannot complete without them (LPC-070 §9) |
| **Approve production** | Claim production readiness, production eligibility, write/network/proof execution, or merge authority from advisor text | `TacticianPolicy` capability flags fixed closed; plan metadata cannot smuggle `complete` / `proved`; Groth16 ceremony `production_eligible` requires independent contributor quorum and complete key artifacts — advisor claims never set it |
| **Silently add assumptions** | Mutate assumption sets, trust, provenance, or license during advice without an explicit closed listing | Formalization `PROTECTED_SEMANTIC_FIELDS` (assumptions, provenance, trust, license); proposal candidates cannot invent free-form authority metadata; `new_assumption_ids` on candidates are **explicit** and costed / bounded by ranking policy |
| **Drop blocking obligations** | Rank or schedule away required obligations, holes, or blocking assumptions | Ranking hard-prunes `MISSING_COVERAGE` when required obligations are uncovered; incomplete steps hard-pruned; fallbacks must preserve blocking obligations (LPC-070 §11); planner `order_sources` never drops candidates silently |

## Domain-neutral tactician invariants

From `ipfs_datasets_py.logic.tactician.models` / `policy` / `receipts`:

1. `semantic_authority` is always `False` on plans, policies, and receipts.
2. `network_allowed`, `write_allowed`, and `proof_execution_allowed` remain
   `False` on the planner policy (the planner never proves, writes, or
   networks).
3. Metadata cannot smuggle `_AUTHORITY_PROMOTION_KEYS`
   (`semantic_authority`, `proof_authority`, `write_authority`, …).
4. Default abstain conditions include `authority_promotion_attempt`.
5. Content-addressed `plan_id` / `receipt_id` bind the advisory body; forging
   an id does not mint authority.

## Software-verification tactician invariants

From `logic/software_verification/tactician/{contracts,candidate_synthesis,proof_graph,proof_plan}`:

1. `_PROPOSAL_FORBIDDEN_TRUE_CLAIMS` rejects true-ish values for
   `proof_claimed`, `proved`, `complete`, `completion_claimed`, `admitted`,
   `kernel_verified`, and related smuggle keys on proposal payloads.
2. `CandidateProofStep` authority is capped at `candidate`.
3. `GoalDirectedProofPlan` cannot claim theorem or attestation authority and
   cannot smuggle completion through metadata.
4. Learned portfolio sources (`advisor:*`, `provider:leanstral`, …) remain
   `proposal_only` with authority ≤ candidate.
5. Experimental reconstruction / inference paths are hard-capped at candidate
   and never discharge trusted leaves alone.
6. `collect_hard_failures` rejects proof/completion claims, incomplete steps,
   cycles, insufficient authority, and uncovered **required** obligations
   before soft ranking can compensate.

## Formalization and advisor-provider invariants

1. Formalization advisors return `authority="unverified_candidate_only"` and
   cannot alter protected assumption / provenance / trust / license paths.
2. Leanstral / SymAI `ProposalCandidate` objects reject non-unverified
   authority and authority-claim metadata keys; `is_proved` is always false
   via `confidence_never_yields_proof`.
3. `accept_candidate` requires deterministic compilation **and** independent
   solver/kernel validation; confidence is not a parameter.
4. `AdvisorProviderEvidence@2` cannot satisfy certified authority and cannot
   set `proof_established=True`.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-070 | Canonical plan model; this task freezes the proposal-only authority ceiling on that model |
| LPC-032 | Provider success is not proof; advisors inherit the same non-inference rule |
| LPC-052 | Provider responses default untrusted (`advisory` / `candidate`) |
| LPC-111 | Ten-point admission (incl. reconstruction) is the only path to completion/merge influence |
| LPC-080/081 | Cache keys and repository slots store plans; storage lifecycle ≠ proof authority |

## What this does **not** do

1. Does not forbid advisors from listing **explicit** assumptions, gaps, or
   candidate repairs — it forbids silent or authority-raising mutations.
2. Does not replace kernel checkers, reconstruction engines, or ceremony
   validation with advisor confidence.
3. Does not let ranking utility compensate for missing required obligations.
4. Does not authorize the supervisor to rewrite obligation identity or skip
   reconstruction by reordering lanes.

## Acceptance checklist (LPC-071)

| Criterion | Satisfied when |
| --- | --- |
| Cannot mark proposals proved | Proposal contracts and hard-prunes reject proof claims; tactician `semantic_authority` fixed false |
| Cannot raise authority | Proposal authority caps + advisor/toolchain certified-authority denial |
| Cannot choose verification keys | Closed fields + metadata rejection; verifier bindings not certificate-selected |
| Cannot skip reconstruction | Experimental methods non-trusted; reconstruction remains admission-gated |
| Cannot approve production | Capability flags closed; production eligibility independent of advisor claims |
| Cannot silently add assumptions | Protected formalization fields; explicit `new_assumption_ids` only |
| Cannot drop blocking obligations | Hard-prune on uncovered required obligations; incomplete steps rejected |

All seven criteria are required. Pytest green alone does not relax any row.
