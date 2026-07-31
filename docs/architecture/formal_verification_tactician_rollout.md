# Formal Verification Tactician Rollout Policy

**Interface:** `FormalVerificationTacticianRollout@1`  
**Goal:** `FVT-G080`  
**Task:** `FVT-035`  
**Program:** `formal-verification-tactician/release`  
**Schema:** `formal-verification-tactician-rollout/v1`

This document is the operator and implementation contract for promoting
goal-directed formalization, proof-gap proposals, validated proof plans, and
counterexample-guided repair through staged modes with automatic quarantine
and rollback. It is the rollout evidence artifact for objective `FVT-G080`.

Promotion is always **property-specific and provider-specific**. Aggregate
portfolio success never promotes a weaker pair, and there is no global
“tactician enabled” switch that overrides per-pair stages.

Companion artifacts:

| Artifact | Role |
| --- | --- |
| `ipfs_datasets_py/docs/logic/software_verification_rollout.md` | LFV property/provider stage baseline (`declared` / `shadow` / `canary` / `enforced`) |
| `docs/architecture/formal_verification_tactician_benchmark.json` | Receipt-derived quality / privacy / authority benchmark (`GoalTacticianBenchmark@1`) |
| `docs/architecture/formal_verification_toolchain_certificate.json` | Hermetic offline toolchain identities |
| `docs/architecture/formal_verification_readiness_baseline.json` | Status ladder; implementation ≠ deployment |
| `docs/formal_verification_tactician.md` | Product architecture and authority vocabulary |
| `docs/operations/formal_verification_tactician_runbook.md` | Operator failure / demotion matrix |
| `ipfs_accelerate_py/agent_supervisor/objectives/goal_development_contracts.py` | `GoalDevelopmentMode` (`off` / `shadow` / `assist` / `auto_safe`) |
| `ipfs_accelerate_py/agent_supervisor/proof/leanstral_goal_benchmark.py` | Adjacent promotion gate (`evaluate_goal_rollout_promotion`) |

Executable validation:

```text
python -m pytest test/api/test_formal_verification_tactician_rollout.py -q
```

---

## 1. Program invariants

1. **No side-effect discovery.** Imports, declaration probes, inventory, and
   rollout evaluation never install packages, download toolchains, open the
   network, spawn unbounded processes, or mutate a checkout.
2. **Typed, non-interchangeable authority.** Advisor, Leanstral, SymAI,
   autoencoder, cache hit, monitor, bounded model, test result, or ZKP
   attestation never silently becomes theorem or source-translation authority.
3. **Property/provider locality.** Promotion and enforcement are scoped to a
   `(property_kind, provider_id, authority_class)` triple (a *pair* when
   authority is fixed by the provider). Aggregate success across unrelated
   pairs never advances a lagging pair.
4. **Receipt-backed gates only.** Promotion and auto-safe admission consume
   actual conformance suites, benchmark reports, and toolchain certificates —
   never synthetic hardcoded success counters or documentation-only claims.
5. **Reversible stages.** Every stage can demote. Historical receipts remain
   immutable; demotion changes only live policy for new work.
6. **Explicit non-success.** Unsupported fragments and unavailable tools stay
   disclosed; they never become silent success or fabricated readiness.
7. **Counterexample closure is verifier-only.** A structural repair never
   closes a counterexample. Only a fresh matching verifier receipt can close it.
8. **Public envelopes are secret-safe.** Raw source, credentials, tokens,
   private witnesses, and hidden channels never enter public API responses or
   model context.

---

## 2. Stage ladder

Tactician-facing modes form a strictly ordered ladder. Unknown values fail
closed to `off`. Transitions must be **adjacent** unless an operator records an
explicit demotion (any higher stage may roll back to a lower stage in one step).

| Stage | Token | Meaning | May change live plans / admit work? | Default for new pairs |
| --- | --- | --- | --- | --- |
| Off | `off` | Feature disabled for the pair. No proposals, no plans, no CEGIS. | No | Yes |
| Shadow | `shadow` | Observational only. Records candidate quality, holes, and witness metrics without mutating plans. | No (advisory envelopes only) | After declaration + smoke |
| Assist | `assist` | Surfaces proposals and ranked plans for human review. Review required; never auto-admits. | Review only | After shadow gates |
| Auto-safe | `auto_safe` | May admit **only** allowlisted, independently validated, deterministic steps under explicit policy opt-in. | Yes, allowlist only | After assist gates + opt-in |
| Enforced | `enforced` | Property/provider-specific enforcement of the pair’s declared authority class for the reviewed fragment. | Yes, for closed authority only | After auto-safe maturity + hard-zero clearance |

Alignment with related ladders:

| This policy | LFV property/provider stages | Goal development modes |
| --- | --- | --- |
| `off` | (undeclared or disabled) | `off` |
| `shadow` | `shadow` | `shadow` |
| `assist` | `canary` (opt-in diagnostic; non-proof) | `assist` |
| `auto_safe` | (admission band before full enforcement) | `auto_safe` |
| `enforced` | `enforced` | (pair-local enforcement above auto_safe) |

`repair_only` (goal-development) is an operational restriction, not a promotion
target: it may run under `shadow` / `assist` / `auto_safe` policy without
granting broader admission.

Learned advisors (`provider.learned_proposals`, Leanstral draft formalization,
SymAI, autoencoders) remain permanently at most `assist` for **admission
authority**. They may appear in `shadow` quality reports. They never transition
to `enforced` as theorem authority.

### 2.1 Stage semantics for tactician capabilities

| Capability | `off` | `shadow` | `assist` | `auto_safe` | `enforced` |
| --- | --- | --- | --- | --- | --- |
| End-goal formalization | Disabled | Advisory proposals only | Reviewable proposals | Admit only independently validated formalizations on allowlist | Pair may require formalization path for the fragment |
| Proof-gap / hole discovery | Disabled | Observational holes | Reviewable hole set | Admit validated hole set bindings | Enforce hole inventory on plan start |
| Validated proof plans | Disabled | Ranked shadow plans | Human-selected plan | Admit allowlisted plan steps with independent validation receipts | Enforce plan gates for the pair |
| Counterexample-guided repair (CEGIS/CEGAR) | Disabled | Shadow synthesis; no closure | Review before apply | Admit only steps closed by fresh matching verifier receipt | Enforce CEGIS policy; quarantine on hard-zero |
| Public API / CLI / MCP | Surfaces report `off` / unavailable for the pair | Surfaces return advisory | Surfaces return review-required | Surfaces may admit allowlisted ops | Surfaces apply enforcement for the pair |

---

## 3. Promotion rules

### 3.1 Adjacent promotion only

Promotion evaluates one step at a time:

```text
off → shadow → assist → auto_safe → enforced
```

Skipping a stage is rejected with reason code `promotion_must_be_adjacent`.
Demotion may skip downward (for example `enforced → shadow`) when a quarantine
trigger fires.

### 3.2 Evidence inputs (mandatory)

Gates **must** consume actual current-tree identities, not aspirational docs:

| Gate input | Primary evidence path | What counts |
| --- | --- | --- |
| Conformance / corpus | `ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json` and related suite green | Executable fixtures for the property fragment |
| Benchmark quality | `docs/architecture/formal_verification_tactician_benchmark.json` plus suite `test/benchmarks/test_formal_verification_tactician_benchmark.py` | Hard gates at 100% for correctness, privacy, authority; timing observational unless calibrated |
| Toolchain certificate | `docs/architecture/formal_verification_toolchain_certificate.json` | Exact offline tool identities; unavailable tools disclosed |
| Readiness baseline | `docs/architecture/formal_verification_readiness_baseline.json` | Status ladder; no PATH-as-usable inference |
| Goal-development promotion report | `evaluate_goal_rollout_promotion` / paired goal benchmark | Adjacent mode decision with reason codes |
| Adversarial / security | `test/security/test_formal_verification_tactician_adversarial.py` (and datasets peer) | Zero hard-zero counters for the pair |

Documentation-only prover catalogs, synthetic counters, or LFV completion
receipts alone **never** authorize tactician `enforced` for a pair that lacks
current executable evidence.

### 3.3 Per-transition requirements

#### `off → shadow`

1. Pair is declared in the capability census or prover catalog with sorted
   evidence paths.
2. Smoke path exists and does not install tools or open the network.
3. Zero hard-zero counters for the pair on the promotion cohort.
4. Unsupported / unavailable lanes for the pair are disclosed (not fabricated).

#### `shadow → assist`

1. All `off → shadow` requirements still hold.
2. Shadow cohort meets minimum observation floor (default: 25 paired
   observations for assist; see `GoalRolloutGatePolicy`).
3. Schema and type acceptance floors met; fallback ceiling not exceeded.
4. Material quality / evidence-coverage deltas met when a paired baseline
   exists; no unsupported-semantics regression.
5. Assist output is labeled review-required; `authoritative_for` remains empty
   for advisory drafts.

#### `assist → auto_safe`

1. All prior requirements hold.
2. Explicit operator / policy opt-in: `allow_auto_safe_promotion=true` (default
   **false**). Reason code when missing:
   `auto_safe_promotion_not_explicitly_authorized`.
3. Higher observation floor (default: 100).
4. Stricter schema/type acceptance and fallback ceilings.
5. Auto-safe **allowlist** published for the pair (section 4). Empty allowlist
   blocks promotion.
6. Independent validation path exists for every allowlisted step kind
   (`validate_proof_candidate` or equivalent independent check).

#### `auto_safe → enforced`

1. All prior requirements hold for the exact property/provider/authority
   triple.
2. Mandatory contract and adversarial tests for the pair pass.
3. Hard-zero counters remain zero (section 5).
4. Documented assurance ceiling, supported semantic fragment, and resource
   class.
5. Toolchain certificate shows the pair’s tools as usable/production-certified
   **or** the pair remains disclosed `unavailable` and cannot enforce.
6. Explicit rollback instructions and quarantine owner recorded.
7. No authority-boundary violations; cache hits preserve authority and identity
   (never upgrade).

### 3.4 Non-timing correctness

Benchmarks may record wall-clock, CPU, memory, and latency distributions for
capacity planning. **Timing ratios are never correctness gates** unless a
calibrated timing gate is explicitly documented for the pair. Correctness is
decided only by:

- semantic agreement with golden fixtures and mutations;
- explicit unavailable / timeout / malformed / unsupported terminal states;
- reconstruction success under bound identities;
- zero hard-zero gate counts;
- resource-bound compliance (hard limits, not relative speed).

---

## 4. Auto-safe admission allowlist

`auto_safe` is the only mode that may produce an admission receipt
(`GoalDevelopmentAdmissionReceipt` with decision `admitted`). Admission is
fail-closed.

### 4.1 Admission preconditions

An auto-safe admission is allowed only when **all** of the following hold:

1. Live stage for the pair is `auto_safe` or `enforced`.
2. The step kind appears on the pair’s allowlist (section 4.2).
3. Independent validation produced a fresh authoritative receipt id bound to
   the same tree, property, assumptions, provider/version, policy, and bounds.
4. The proposal was previously accepted as a candidate (not a free-form draft
   injected past validation).
5. No open hard-zero quarantine flag exists for the pair.
6. Authority class of the receipt does not exceed the pair’s assurance ceiling.
7. For counterexample closure steps: witness is closed only by a **fresh
   matching verifier receipt**, not by structural edit alone.

Missing any precondition yields non-admission with explicit reason codes.
Assist mode may request review; it **cannot** admit.

### 4.2 Default allowlisted step kinds

Operators may tighten the list per pair. The default closed set of step kinds
that may be auto-admitted when independently validated:

| Step kind | Notes |
| --- | --- |
| `admit_validated_lemma` | Candidate lemma after independent `validate_proof_candidate` success |
| `admit_validated_invariant` | Same for invariants |
| `admit_validated_contract` | Same for contracts / VC assumptions that are obligations, not silent premises |
| `admit_plan_step_with_receipt` | Plan step whose required receipt class was obtained |
| `close_counterexample_with_verifier_receipt` | Only with fresh matching verifier receipt |
| `apply_deterministic_minimization` | Minimizer with declared guarantee level; no privacy leakage |
| `replay_confirmed_counterexample` | Exact-bounds replay success recorded as evidence, not as proof of unrelated goals |

**Never** allowlisted (in any stage):

- `promote_proof_authority` / authority escalation
- `admit_goal` without validation
- `force_complete` / `close_plan` without adequate receipts
- `lease_steal` or other forbidden supervisor controls
- learned-advisor drafts without independent validation
- cache hits that would raise authority above the cached receipt
- favorable assumptions inserted solely because they entail the goal

### 4.3 Allowlist binding

Each pair’s live allowlist must name:

- `property_kind`
- `provider_id`
- `authority_class` ceiling
- allowed step kinds (subset of section 4.2)
- evidence receipt ids for the promotion decision that enabled `auto_safe`
- resource class and bound profile

Changing the allowlist is itself a policy revision and requires re-validation;
it does not rewrite historical admissions.

---

## 5. Hard-zero quarantine and rollback

### 5.1 Hard-zero signals

Any positive count of the following **blocks promotion** and **forces
quarantine** for the affected property/provider pair:

| Signal | Typical reason code | Effect |
| --- | --- | --- |
| False proof | `false_proof_observed` | Immediate demotion; never auto-promote |
| False counterexample closure | `false_closure_observed` | Demote; keep CE open |
| Secret / private-witness leakage | `secret_or_witness_leakage` | Quarantine public path; purge channel |
| Source-binding mismatch | `binding_mismatch` | Invalidate plan epoch; demote |
| Authority escalation / mislabel | `authority_boundary_violation` | Demote to `declared`/`off` band; open defect |
| Unresolved cross-provider disagreement represented as success | `unresolved_disagreement` | Fail-closed quarantine (no majority vote) |
| Fabricated readiness / synthetic production claim | `fabricated_readiness` | Block deployment claims |

Aligned hard-zero names used in LFV and goal benchmarks:

- `authority_boundary_violations`
- `false_proof_count` / `false_completion_count`
- `secret_or_witness_leakage_count`
- `unresolved_cross_provider_disagreement_count`

### 5.2 Quarantine procedure

When a hard-zero signal fires for pair `P`:

1. **Freeze promotions** for `P` (and only `P` — do not globally disable
   unrelated pairs based on aggregate portfolio metrics).
2. Set live stage for `P` to `shadow` or `off` according to severity:
   - leakage, false proof, authority mislabel → prefer `off` or LFV `declared`
   - tool identity drift, resource breach → `shadow`
3. Quarantine related cache keys and in-flight admissions for `P`.
4. Preserve all receipts and artifacts for audit (immutable history).
5. Fall back to the last reviewed deterministic path for `P` (explicit
   non-success if none).
6. Record reason codes, timestamp, operator or automated detector id, and
   evidence receipt ids.
7. Re-promotion must restart from the adjacent ladder using fresh evidence
   (section 3); quarantine clearance is not automatic.

### 5.3 Other demotion triggers (non hard-zero)

| Trigger | Demote to | Notes |
| --- | --- | --- |
| Tool identity change | `shadow` | Invalidate warm cache keys |
| Missing optional tool | keep declaration; report `unavailable` | Never fabricate |
| Resource-bound breach | terminal non-success; hold stage | Do not promote partial results |
| Restart recovery unstable | hold or demote one stage | `restart_recovery_unstable` |
| Operator request | explicit target stage | Reason code + timestamp required |
| Publication / tree identity change | hold enforcement claims | Re-bind receipts to new tree |

---

## 6. Property- and provider-specific policy

### 6.1 No global enforcement from aggregates

Forbidden:

- “All SMT pairs green → enable every property”
- “Mean portfolio score above threshold → set default to enforced”
- Promoting provider `A` for property `contract` because provider `B` succeeded
  on `liveness`

Required:

- Named `property_kind` + `provider_id` (+ authority ceiling) on every
  promotion decision and enforcement flag
- Independent hard-zero and conformance evidence for that triple
- Disclosure of pairs that remain `off`, `unsupported`, or `unavailable`

### 6.2 Default maturity by property family

Defaults follow the LFV rollout table and readiness baseline. New tactician
pairs start at `off` or `shadow`. Illustrative ceilings (not automatic
enforcement):

| Property family | Typical providers | Max authority when enforced | Default tactician stage until evidence |
| --- | --- | --- | --- |
| Contracts / invariants / heap safety | SMT (Z3/CVC5) | `bounded_solver_outcome` | `shadow` |
| Liveness / state machines | TLA+/Apalache, monitors | `bounded_state_machine` / monitor | `shadow` / assist for monitors |
| Authorization | Datalog/SecPAL | `authorization_policy` | `shadow` |
| Protocol secrecy / auth | Tamarin/ProVerif | protocol authority classes | `shadow` |
| Hyperproperties | HyperLTL family | hyperproperty satisfaction | `off`/`shadow` until tool smoke |
| Theorems / kernels | Lean/Rocq/Isabelle | `kernel_checked_proof` | `shadow`; reconstruction required |
| Learned formalization | Leanstral / advisors | advisory only | never `enforced` as theorem |

### 6.3 Unsupported and unavailable disclosure

| State | Meaning | Rollout implication |
| --- | --- | --- |
| `unsupported` | Product boundary: fragment intentionally out of scope | Remain disclosed; stage stays `off` or declaration-only; no fabricate |
| `unavailable` | Tool/runtime missing or not probeable on this machine | Report unavailable; may keep shadow declaration; cannot reach `enforced` |
| `usable` / `production_certified` | Baseline + certificate ladders | Prerequisite for `enforced` claims on that host |

Implementation completeness (code present, fixture-tested) is **not**
deployment certification and is **not** `enforced`.

---

## 7. Operator decision record

Every promotion, demotion, or allowlist change should record:

```text
decision_id          content-addressed or UUID
property_kind        e.g. contract
provider_id          e.g. z3
authority_class      ceiling after decision
from_stage           off|shadow|assist|auto_safe|enforced
to_stage             adjacent (promotion) or any lower (demotion)
allowed              bool
reason_codes         empty iff allowed for promotion
evidence_ids         benchmark report, certificate, conformance suite ids
allowlist_revision   when auto_safe/enforced
operator_or_system   who authorized
timestamp            UTC ISO-8601
tree_identity        parent commit / content id
```

Executable companions may emit `GoalRolloutGateDecision` (goal-development
modes) or LFV pair stage receipts; both must remain reconcilable with this
ladder.

---

## 8. Validation checklist (pre-enforced)

For every property/provider pair before `enforced`:

1. **Declared** with evidence paths and explicit unsupported/unavailable notes.
2. **Smoke** path exists; no install/network.
3. **Shadow** cohort recorded; advisory only.
4. **Assist** review path exercised; no silent admission.
5. **Auto-safe** opt-in + non-empty allowlist + independent validation receipts.
6. **Hard-zero** counters at zero for the pair.
7. **Benchmark** hard gates (correctness, privacy, authority) pass for the
   cohort; timing not used as correctness unless calibrated.
8. **Toolchain** certificate or explicit unavailable disclosure for required
   tools.
9. **Rollback** owner, demotion target, and quarantine procedure documented.
10. **Public surfaces** (API/CLI/MCP) report the live stage without promising
    missing tools.

---

## 9. Keeping the policy current

1. Change implementation and executable evidence first.
2. Update this document together with baseline, benchmark, and certificate
   artifacts when stages or allowlists change.
3. Run:

   ```text
   python -m pytest test/api/test_formal_verification_tactician_rollout.py -q
   ```

4. Reconcile with `software_verification_rollout.md` when LFV pair stages move.
5. Refresh completion receipts only when hard-zero gates still hold on a clean
   current tree.

Never refresh documentation solely to silence a failing inventory, authority,
or hard-zero test.

---

## 10. Acceptance mapping (`FVT-G080`)

| Acceptance criterion | Where this policy satisfies it |
| --- | --- |
| Gates consume actual conformance/benchmark/toolchain receipts | §3.2, §8 |
| Auto-safe admits only allowlisted independently validated steps | §4 |
| False proof/closure, leakage, binding mismatch, authority escalation, or unresolved disagreement triggers quarantine and rollback | §5 |
| Unsupported/unavailable lanes remain disclosed | §1, §6.3 |
| Property/provider-specific; no global enforce from aggregate success | §1, §6.1 |
| Stages off / shadow / assist / auto_safe / enforced with automatic quarantine | §2, §5 |

Conflict policy: this document and
`test/api/test_formal_verification_tactician_rollout.py` own the tactician
rollout policy surface. Do not globally enforce a provider or property based on
aggregate success.
