# LPC-070 Canonical Proof-Plan Model

**Task:** LPC-070 — Define the canonical proof-plan model  
**Goal:** LPC-G070  
**Depends on:** LPC-044 (admitted slices only), LPC-052 (typed provider responses, untrusted default authority)  
**Interface:** `CanonicalProofPlan@1`  
**Module:** `ipfs_datasets_py.logic.tactician.models`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/tactician/models.py`  
**Planner evidence:** `ipfs_datasets_py/ipfs_datasets_py/logic/tactician/planner.py`  
**Schema:** `ipfs_datasets_py/canonical-proof-plan@1`  
**Schema version:** `canonical-proof-plan/v1`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/tactician/test_models.py -q`

## Purpose

Proof work needs **one** datasets-owned plan model so domain adapters,
software-verification tacticians, multi-prover portfolios, and the supervisor
do not invent parallel vocabularies for goals, obligations, lanes, and
completion.

This note freezes `CanonicalProofPlan@1`. A plan is an **execution proposal**:
it may order work, name dependencies, bind reconstructions, declare bounds,
and record fallbacks. It may **not** mark itself proved, raise authority,
skip required reconstruction, approve production, silently add assumptions,
or drop blocking obligations (LPC-071).

Executable coverage for the domain-neutral tactician surface lives in
`ipfs_datasets_py/tests/unit/logic/tactician/test_models.py` (and the sibling
planner / receipt tests under `tests/unit/logic/tactician`).

## Ownership

| Owner | Responsibility |
| --- | --- |
| **Datasets** (`logic/tactician`) | Semantic plan model, content identity, acyclicity, authority-closed validation, completeness boundary |
| **Datasets** (`logic/software_verification/tactician`) | Domain-specialized projections (`EndGoalSpec@1`, `GoalDirectedProofPlan@1`, holes/graphs) that **must** map onto this model |
| **Datasets** (`logic/common/proof_repository`) | Durable plan records keyed by `CanonicalProofCacheKey@1` (LPC-080/081) |
| **Supervisor** | Scheduling, isolation, resource placement, single-flight; may **reorder** semantically valid lanes only |
| **Supervisor** `ProofPlan` / `ProofPlanStep` | Compatibility facade until cutover; must not become a second semantic authority |

Conflict policy (LPC-G070 / LPC-070): own the tactician plan types and tests.
**Do not add a second planner or a second tactician.**

## Generation map

| Generation | Module / path | Classification | Role |
| --- | --- | --- | --- |
| **`CanonicalProofPlan@1`** | `logic/tactician/models.py` | **canonical** (this task) | Single semantic plan model |
| `TacticianPlan` / `LogicTactician` | `logic/tactician/{models,planner}.py` | **canonical** (domain-neutral planner) | Content-addressed route/subgoal plan; LPC-G070 evidence |
| `GoalDirectedProofPlan@1` | `logic/software_verification/tactician/contracts.py` | specialized projection | Software-verification missing-proof ranking |
| `ProofPlan` (supervisor) | `agent_supervisor/proof/formal_verification_contracts.py` | compatibility_facade | Operational DAG until cutover |
| `PortfolioPlan` | `agent_supervisor/proof/multi_prover_router.py` | compatibility_facade | Multi-prover routing only |
| `ProofPlanRecord` | `logic/common/proof_repository.py` | repository carrier | Durable plan slot under cache key |

Rules:

1. One plan model. Domain and supervisor surfaces **project** onto
   `CanonicalProofPlan@1`; they do not redefine goal/obligation meaning.
2. Advisors and models **propose**. Proof and completion authority come only
   from admitted receipts (LPC-032, LPC-052, LPC-111 ten-point floor).
3. Supervisor may reorder **semantically valid** lanes; it must not rewrite
   interpretation, obligation identity, reconstruction requirements, or
   completeness conditions.

## Interface identity

| Constant | Value |
| --- | --- |
| `interface` | `CanonicalProofPlan@1` |
| `schema` | `ipfs_datasets_py/canonical-proof-plan@1` |
| `schema_version` | `canonical-proof-plan/v1` |
| Live domain-neutral interface | `ipfs_datasets_py.logic.tactician@1` |
| Live domain-neutral schema version | `1.0.0` (`SCHEMA_VERSION` in `models.py`) |

`plan_id` is always a **content digest** over the canonical plan body
(`compute_content_digest` / body without `plan_id`). Self-assigned ids that
do not match the body fail closed.

## Acceptance field inventory

Every admitted `CanonicalProofPlan` body represents **all** of the following
dimensions (LPC-070 acceptance). Missing dimensions fail closed at
construction or admission.

| Dimension | Carrier | What it binds |
| --- | --- | --- |
| **goal** | `goal` | Root goal identity, statement ref, family, exact roots |
| **interpretations** | `interpretations[]` | Candidate semantic readings of the goal |
| **properties** | `properties[]` | Target properties / property classes under the selected reading |
| **assumptions** | `assumptions[]` | Closed, explicit assumption set (no silent adds) |
| **obligations** | `obligations[]` | Discrete proof obligations (from admitted slices only) |
| **dependency graph** | `dependency_graph` | Acyclic obligation/step edges |
| **translations** | `translations[]` | Translation steps / preservation claims in the plan |
| **lanes** | `lanes[]` | Semantically valid parallel execution groups |
| **reconstruction** | `reconstruction` | Required reconstruction / kernel-check policy |
| **bounds** | `bounds` | Finite resource and semantic bounds |
| **fallbacks** | `fallbacks[]` | Ordered recovery when a lane/step fails or is unsupported |
| **status** | `status` | Plan lifecycle (never self-complete / self-proved) |
| **completeness boundary** | `completeness` | What “done” means; who may assert it |

Closed constant intent: treat these thirteen dimensions as required identity
of a plan, analogous to the sixteen cache-key fields of LPC-080.

### 1. Goal

| Field | Type | Notes |
| --- | --- | --- |
| `goal.goal_id` | non-empty string | Stable opaque identity |
| `goal.statement_ref` | opaque ref | Never an unbounded statement body |
| `goal.goal_family` | domain-supplied string | Family label only; not authority |
| `goal.goal_root` | opaque root | Exact identity binding |
| `goal.corpus_root` | opaque root | Corpus the plan must bind |
| `goal.config_root` | opaque root | Planner/config identity |
| `goal.authority_roots` | map string→opaque | Additional exact roots (tree, policy, …) |
| `goal.proof_gaps` | finite string list | Gaps the plan should cover |
| `goal.selected_interpretation_id` | string | Must reference an entry in `interpretations` when non-empty |

Live carrier: `TacticianGoal` in `models.py` (exact opaque roots, bounded
gaps/assumptions/metadata; rejects authority-promotion metadata keys).

### 2. Interpretations

Interpretations are **candidate semantic readings**. Multiple interpretations
imply residual ambiguity until one is selected.

| Field | Type | Notes |
| --- | --- | --- |
| `interpretation_id` | non-empty string | Stable within the plan |
| `controlled_english` | bounded string | Human-readable reading |
| `property_class` | closed vocabulary | e.g. safety, liveness, theorem, contract |
| `quantifiers` | closed list | exists / forall / eventually / always / … |
| `current_state` / `target_state` | bounded maps | State framing for the reading |
| `environment` | bounded map | Environment assumptions of the reading |
| `unresolved_fields` | string list | Explicit residual holes |
| `selected` | bool | At most one selected when ambiguity is resolved |

Projection: `EndGoalInterpretation` / `EndGoalSpec.interpretations` in
`software_verification/tactician/contracts.py`. Ambiguity statuses
(`none | candidates_present | requires_selection | resolved | unsupported`)
must not be silently collapsed.

### 3. Properties

| Field | Type | Notes |
| --- | --- | --- |
| `property_id` | non-empty string | Stable property identity |
| `property_class` | closed vocabulary | Same closed set as interpretation classes |
| `statement_ref` | opaque ref | Formal/property statement identity |
| `interpretation_id` | string | Owning interpretation |
| `assurance_target` | authority ceiling | Desired ceiling; not a claim of achievement |
| `obligation_ids` | string list | Obligations that discharge this property |

Properties are **targets**, not proofs. A plan that lists a property with
`assurance_target=theorem` still has advisory/candidate plan authority until
reconstruction and admission succeed.

### 4. Assumptions

| Field | Type | Notes |
| --- | --- | --- |
| `assumption_id` | non-empty string | Stable identity |
| `class` | `trusted \| must_prove \| hypothetical` | How the assumption may be used |
| `statement_ref` | opaque ref | Assumption content identity |
| `introduced_by` | string | Step / interpretation / external source |
| `blocking` | bool | If true, dropping it is a completeness failure |

Rules:

* Plans may **list** assumptions; they may not **silently add** them during
  ranking or scheduling (LPC-071).
* `trusted` assumptions require external policy admission; they are not
  free authority upgrades.
* New assumptions introduced by steps must appear in this closed set.

### 5. Obligations

| Field | Type | Notes |
| --- | --- | --- |
| `obligation_id` | non-empty string | Content-addressed when possible |
| `slice_id` / `slice_digest` | identity | Must come from an **admitted** `DomainLogicSlice@2` (LPC-044) |
| `statement_ref` | opaque ref | Obligation surface |
| `kind` | string | Hole/obligation kind or template id |
| `required_assurance` | authority ceiling | Floor for discharge |
| `blocking` | bool | Blocking obligations cannot be dropped |
| `fallback_check_ids` | string list | Deterministic checks when proof is unsupported |
| `lane_id` | string | Optional owning lane |

Executable obligations are seeded only through admitted slices:

```text
DomainLogicSlice@2 (status=admitted)
  → LogicObligation@2.from_slice
  → BackendRequest@2.from_slice
  → plan.obligations[]
```

Unadmitted, rejected, or unsupported slices never appear as executable
obligations (LPC-044).

### 6. Dependency graph

| Field | Type | Notes |
| --- | --- | --- |
| `nodes` | map id→node kind | Obligation / step / assumption / evidence / join |
| `edges` | list of `{from, to, kind}` | `depends_on`, alternatives, repairs, evidence refs |
| `root_ids` | string list | Entry nodes |
| `acyclic` | derived | Must be true; cycles fail closed |

Live enforcement:

* `TacticianSubgoal.depends_on` + `detect_cycle` on `TacticianPlan`
* Supervisor `ProofPlan._validate_graph` (compatibility facade)
* Software-verification obligation graphs (`ProofObligationGraph@1`)

Self-dependencies and directed cycles raise validation errors
(`TacticianValidationError` / contract errors). Unknown external dependency
ids may be treated as leaves only when explicitly allowed by policy; unknown
internal step references fail closed.

Edge kinds (closed v1 intent): `depends_on`, `alternative`, `repair`,
`regression`, `evidence_ref`, `weakest_precondition`, `preimage`,
`rule_inversion`, `unification`, `subsumption`.

### 7. Translations

| Field | Type | Notes |
| --- | --- | --- |
| `translation_id` | non-empty string | Step identity |
| `from_artifact` / `to_artifact` | digests/ids | Ends of the translation |
| `kind` / `preservation` | closed axes | Exact, equisatisfiable, bounded, conservative, heuristic |
| `receipt_ref` | optional string | `LogicTranslationReceipt@1` when available |
| `required` | bool | Whether reconstruction/admission needs this step |

Translation claims on the plan are **proposed**. Provider success on a
translate op remains untrusted until receipt admission (LPC-052 / LPC-032).
Heuristic or conservative translations cannot silently claim kernel ceilings.

### 8. Lanes

A **lane** is a semantically valid parallel group of steps that share meaning
but may be scheduled independently.

| Field | Type | Notes |
| --- | --- | --- |
| `lane_id` | non-empty string | Stable lane identity |
| `obligation_ids` / `step_ids` | string lists | Members of the lane |
| `provider_ids` | string list | Allowed providers for the lane |
| `resource_class` | string | Scheduling class (supervisor placement) |
| `semantic_fingerprint` | digest | Meaning binding the lane must preserve |
| `reorderable` | bool | Supervisor may reorder only when true **and** fingerprint holds |
| `stage_order` | ordered stages | Local stage sequence within the lane |

Stage vocabulary (aligned with supervisor `ProofStage` for cutover):

`translate → model_draft → solve → reconstruct → kernel_verify → validate → attest → persist`

Supervisor rules:

1. May reorder **lanes** relative to each other when each lane’s
   `semantic_fingerprint` and dependency edges remain satisfied.
2. Must **not** rewrite interpretation, obligation identity, reconstruction
   requirements, fallback ordering meaning, or completeness conditions.
3. Must **not** promote a candidate lane into a kernel lane by scheduling alone.

Domain-neutral projection: `TacticianRoute` selected/excluded routes with
`stage_index` and gap coverage act as route-level lanes for source ordering.

### 9. Reconstruction

| Field | Type | Notes |
| --- | --- | --- |
| `required` | bool | Whether reconstruction is mandatory for completion |
| `methods` | closed list | Named reconstruction / kernel methods |
| `kernel_check_required` | bool | Independent kernel verify stage required |
| `skip_forbidden` | bool | Always true for production-affecting plans |
| `receipt_classes` | string list | Expected reconstruction receipt classes |

Rules:

* Plans that require reconstruction **cannot** be completed by solve-only
  success (LPC-032, ten-point admission point 7).
* Experimental / string-equality-only methods cannot satisfy kernel-required
  policies.
* Advisors cannot set `skip_forbidden=false` to bypass reconstruction
  (LPC-071).

### 10. Bounds

| Field | Type | Notes |
| --- | --- | --- |
| `wall_time_ms` | non-negative int | Finite wall budget |
| `memory_bytes` | non-negative int | Finite memory budget |
| `max_steps` / `max_depth` / `max_nodes` / `max_candidates` | non-negative int | Structural bounds |
| `max_routes` / `max_subgoals` / `max_sources` | positive int | Planner policy bounds (`TacticianPolicy`) |
| `network_allowed` | bool | Fixed `false` on advisory tactician policy |
| `proof_execution_allowed` | bool | Fixed `false` on the planner itself |
| `write_allowed` | bool | Fixed `false` on the planner itself |
| `extra` | map string→non-negative int | Additional integer budgets only |

Bounds are **finite and positive where required**. Executable provider ops
also carry operation bounds under `LogicProviderProtocol@2` (LPC-050/052).
Zero / missing executable bounds fail closed at request construction.

Live carriers: `TacticianPolicy` max_* fields; software-verification
`ResourceBounds`; supervisor `ResourceBudget`.

### 11. Fallbacks

| Field | Type | Notes |
| --- | --- | --- |
| `fallback_id` | non-empty string | Stable identity |
| `trigger` | closed string | e.g. `unsupported`, `timeout`, `provider_unavailable`, `lane_failed` |
| `action` | closed string | e.g. `deterministic_checks`, `alternate_lane`, `abstain`, `repair` |
| `check_ids` / `lane_ids` | string lists | Concrete recovery targets |
| `preserves_blocking_obligations` | bool | Must remain true |

Every complete step/lane names at least one fallback (or an explicit
`abstain` fallback). Fallbacks never drop blocking obligations and never
upgrade authority.

Live projections:

* `ProofPlanStepSpec.fallback` (software-verification completeness fields)
* `CodeProofObligation.fallback_checks` (supervisor facade)
* `TacticianPlan.abstain_conditions` / `stop_conditions` (domain-neutral)

### 12. Status

Closed plan lifecycle (proposal-class; **no** self-complete terminal):

| Status | Meaning |
| --- | --- |
| `draft` | Under construction |
| `ranked` | Ranked among alternatives |
| `selected` | Chosen for execution |
| `executing` | Steps in flight under supervisor |
| `blocked` | Waiting on dependency / capability |
| `failed` | Unrecoverable under declared fallbacks |
| `superseded` | Replaced by another plan |
| `abstained` | Planner abstained (no admissible work) |

Forbidden as plan self-status: `proved`, `complete`, `completed` as proof,
`admitted`, `kernel_verified`. Completion is a **separate** contract
(`GoalCompletion` / admitted receipts), never a plan field claim.

Live carriers:

* `TacticianPlan.stop_disposition` (`continue | budget_exhausted | gaps_closed | no_admissible_sources | cycle_detected | abstain | policy_denied`)
* `PlanStatus` on `GoalDirectedProofPlan` / repository `ProofPlanRecord`
* `semantic_authority` fixed `False` on `TacticianPlan` / `TacticianPolicy`

### 13. Completeness boundary

The completeness boundary is the closed set of conditions under which a goal
may be considered finished. **Plans describe the boundary; they do not cross
it.**

| Field | Type | Notes |
| --- | --- | --- |
| `required_obligation_ids` | string list | All must be discharged or explicitly unsupported with fallback |
| `required_receipt_classes` | string list | Receipt kinds required for completion |
| `required_reconstruction` | bool | Must match reconstruction.required |
| `authority_floor` | authority ceiling | Minimum admitted authority |
| `blocking_assumption_ids` | string list | Assumptions that remain open blockers |
| `completion_contract` | string | e.g. `GoalCompletion` only |
| `plan_may_claim_completion` | bool | Always `false` |
| `plan_may_claim_proof` | bool | Always `false` |

Completeness is asserted only by:

1. admitted receipts meeting the ten-point floor (LPC-111 / plan §8); and
2. a dedicated completion contract (`GoalCompletion` / merge gate), never by
   plan metadata (`proof_claimed`, `completion_claimed`, `complete`, `proved`).

Step-level completeness fields (software-verification acceptance) that every
complete step must name:

`dependencies`, `expected_receipts`, `validation`, `fallback`, `resources`,
`completion_conditions`.

## Authority and proposal rules (fail closed)

| Rule | Enforcement |
| --- | --- |
| Plans never set `semantic_authority=True` | `TacticianPlan` / `TacticianPolicy` validation |
| Plans never set `proof_claimed` / `completion_claimed` | software-verification contracts; LPC-071 |
| Metadata cannot smuggle authority keys | `_AUTHORITY_PROMOTION_KEYS` / `_PROPOSAL_FORBIDDEN_TRUE_CLAIMS` |
| Provider success ≠ proof / authority | LPC-032, LPC-052 defaults (`advisory` / `candidate` / `unknown`) |
| Advisors cannot raise authority or skip reconstruction | LPC-071 acceptance |
| Blocking obligations cannot be dropped | completeness + fallback rules |
| Content id must match body digest | `TacticianPlan.validate` / content-addressed contracts |

Default plan authority ceiling is at most `candidate` / `advisory`.
Theorem, attestation, and reconstruction authority are **achieved** by
evidence + admission, not declared by the plan.

## Content identity and validation helpers

| Symbol | Role |
| --- | --- |
| `TACTICIAN_INTERFACE` | `ipfs_datasets_py.logic.tactician@1` |
| `SCHEMA_VERSION` | `1.0.0` live domain-neutral schema |
| `TacticianGoal` | Goal binding with exact opaque roots |
| `TacticianSource` / `TacticianRoute` | Source candidates and ordered lanes/routes |
| `TacticianSubgoal` | Acyclic decomposition node |
| `TacticianPolicy` | Finite bounds; capability flags fixed closed |
| `TacticianPlan` | Content-addressed plan; `semantic_authority=False` |
| `TacticianPlan.build` | Derive `plan_id` from body digest |
| `compute_content_digest` | CIDv1 when available, else `sha256:<hex>` |
| `detect_cycle` | Directed cycle detection over dependency maps |
| `LogicTactician.plan` | Deterministic planner; no proof/write/network |

Hard bounds (live models): `MAX_ID_LENGTH`, `MAX_OPAQUE_ROOT_LENGTH`,
`MAX_STRING_FIELD_LENGTH`, `MAX_LIST_LENGTH`, `MAX_MAP_ENTRIES`,
`MAX_NESTING_DEPTH`, `MAX_METADATA_JSON_BYTES`.

## Relationship to live surfaces

### Domain-neutral tactician (`logic/tactician`)

`TacticianPlan` is the production-ready domain-neutral **core** of
`CanonicalProofPlan@1`:

* goal + roots + policy + planner id
* selected/excluded routes (lane/route ordering)
* proof gaps + subgoal dependency DAG
* stop/abstain conditions and disposition
* content-addressed `plan_id`
* fixed `semantic_authority=False`

Callers supply opaque source classes and gap ids; the generic models never
hard-code legal or program-repair vocabulary.

### Software-verification tactician

`EndGoalSpec@1`, `FormalGoal`, `ProofHole@1`, `ProofObligationGraph@1`, and
`GoalDirectedProofPlan@1` are **specialized projections** that fill
interpretation, property, hole/obligation, and ranking detail. They must
continue to forbid self-proof and self-completion. Ranking may reorder
complete alternatives; incomplete, cyclic, or authority-overclaiming
alternatives are hard-pruned.

### Supervisor compatibility facade

Supervisor `ProofPlan` + `ProofPlanStep` + `ProofStage` remain operational
for scheduling until full cutover. They project into this model as:

* obligations → `obligations[]`
* steps + depends_on → `dependency_graph` + lane membership
* resource_budget / max_parallel → `bounds` + lane scheduling
* stages → lane `stage_order`
* fallback_checks → `fallbacks[]`

The facade must not mint a competing semantic root goal or claim theorem
authority from provider text.

### Proof repository

`ProofPlanRecord` stores plan identity under `CanonicalProofCacheKey@1`.
Repository status (`draft | active | completed | failed | invalidated |
cancelled`) tracks **storage lifecycle**, not proof authority. A repository
`completed` plan slot still requires admitted receipts for goal completion.

## Pipeline position

```text
Admitted DomainLogicSlice@2 (LPC-044)
  → goal + interpretations + properties + assumptions
  → obligations + dependency graph
  → CanonicalProofPlan@1 (this model)
       lanes / translations / reconstruction / bounds / fallbacks
  → supervisor scheduling (reorder valid lanes only)
  → LogicProviderProtocol@2 ops
  → LogicProviderResponse@2 (untrusted default; LPC-052)
  → reconstruction + kernel checks
  → ten-point receipt admission (LPC-111)
  → GoalCompletion / merge influence
```

## What this does **not** do

1. Does not execute proofs, network fetches, or writes from the planner.
2. Does not treat provider or operation success as proof (LPC-032).
3. Does not admit unadmitted slices as obligations (LPC-044).
4. Does not let advisors mark plans proved or raise authority (LPC-071).
5. Does not redefine cache-key identity (LPC-080) or repository APIs (LPC-081).
6. Does not authorize the supervisor to rewrite lane meaning when reordering.
7. Does not replace domain adapters; it is the single plan vocabulary they
   project into.

## Acceptance checklist (LPC-070)

| Criterion | Satisfied by |
| --- | --- |
| Goal | `goal` / `TacticianGoal` |
| Interpretations | `interpretations[]` / `EndGoalInterpretation` projection |
| Properties | `properties[]` with assurance targets |
| Assumptions | `assumptions[]` closed set; no silent adds |
| Obligations | `obligations[]` from admitted slices only |
| Dependency graph | `dependency_graph` + `detect_cycle` / subgoal DAG |
| Translations | `translations[]` with preservation claims |
| Lanes | `lanes[]`; supervisor reorder without rewrite |
| Reconstruction | `reconstruction` required/skip-forbidden policy |
| Bounds | `bounds` / `TacticianPolicy` finite bounds |
| Fallbacks | `fallbacks[]` / step fallback completeness |
| Status | proposal lifecycle; no self-proved/complete |
| Completeness boundary | `completeness` with `plan_may_claim_*=false` |

## Validation

```bash
python -m pytest ipfs_datasets_py/tests/unit/logic/tactician/test_models.py -q
```

Focused suite covers interface constants, goal/source/policy validation,
cycle rejection, authority rejection, and content-stable `TacticianPlan`
round-trips — the live domain-neutral anchors of `CanonicalProofPlan@1`.
