# LPC-043 Intent and UI/UX Domain Adapter Conformance

**Task:** LPC-043 — Intent and UI/UX domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Same adapter contract as legal/security. No universal domain IR.  
**Conflict policy:** Own intent and UI/UX slice adapters only. Never invent a universal domain IR; never collapse intent ↔ ui_ux or either into legal/security/software/crypto.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`

## Purpose

Intent IR and UI/UX IR each own a sealed domain ontology. New formalization
writes lower those ontologies **through** `DomainLogicSlice@2` (LPC-040) under
the same adapter contract used by legal (LPC-041) and security/software/crypto
(LPC-042): every route declares source domain, view, family/profile, property,
notation, preserved/lost semantics, assumptions, unsupported constructs,
proof-safety, and counterexample-safety.

This note freezes adapter locations, domain identities, admitted (or
declaration-only) route tables, assumption axes, preservation/loss
declarations, authority ceilings, and non-collapse rules. It is the durable
LPC-043 evidence for the two domain adapters.

**Important production disposition:**

| Domain | DomainLogicSlice@2 generation today |
| --- | --- |
| Intent | Live: `IntentLogicSlice@2` emits admitted `DomainLogicSlice@2` per route |
| UI/UX | Gated: package absent from pinned revision; `UIUXLogicSlice@2` is declaration-only and **refuses** admission until exact source import + owner-scoped adapter gap close |

UI/UX therefore satisfies the adapter **contract surface** and fail-closed
admission rules without inventing a universal domain IR or fabricating a
`ui_ux_ir` package.

## Canonical lowering path

```text
Domain IR view / obligation
  → TypedExpression (family + profile from the domain route)
  → DomainLogicSlice@2   (DomainLogicSliceV2.from_typed_expression)
  → LogicObligation@2
  → BackendRequest@2
  → compiled / parsed / replay / authority lineage
```

Shared contract module: `ipfs_datasets_py.logic.formalization.artifacts_v3`  
(`DomainLogicSlice@2`, admission gates `require_admitted` / `validate_against`).

Every admitted slice binds (LPC-040 inventory):

| Binding | Required on admitted slice |
| --- | --- |
| Source identity | `document_id`, `source_digest` |
| Expression identity | `expression_id`, `expression_digest` |
| Namespace axes | `family`, `profile`, `property`, `view`, `notation` |
| Features / assumptions | `features`, `assumption_ids` |
| Unsupported extensions | empty when `status=admitted` |
| Status / content identity | `status=admitted`, `content_digest` |
| Domain | domain-specific id (`intent_ir` or `ui_ux_ir`) |

Construction pattern used by the live Intent adapter (and required of UI/UX
once source lands):

```text
DomainLogicSliceV2.from_typed_expression(
    expression,
    slice_id=...,
    domain=<domain_id>,
    document_id=...,
    source_digest=...,
    property=property_id(...),
    view=view_id(...),
    notation=notation_id(...),
    source_range=...,
    features=...,
    assumption_ids=...,
)
domain_slice.require_admitted()
domain_slice.validate_against(document=..., expression=...)
```

## Production adapter modules

Inventory LPC-004 predicted paths named `domain_slice.py`. Live accelerate
implementations for Intent live as `*LogicSlice@2` connectors that **emit**
`DomainLogicSlice@2` records. For UI/UX, the production surface is the
exact-source gate and declaration-only slice (never a fabricated package).

| Domain | Domain id | Adapter interface | Production module | Emits |
| --- | --- | --- | --- | --- |
| Intent | `intent_ir` | `IntentLogicSlice@2` | `ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted intent route |
| UI/UX | `ui_ux_ir` | `UIUXLogicSlice@2` | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | Declaration-only / adapter-gap slice; **no** admitted `DomainLogicSlice@2` while source missing |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Domain | Supporting modules | Role |
| --- | --- | --- |
| Intent | `intent_ir/formalize/typed_compiler.py` | `IntentFormalizationCompiler@2`, `resolve_intent_route`, never-family property/view roles |
| Intent | `intent_ir/formalize/compiler.py`, `advisor.py`, `obligations.py` | Structural formalization, advisor candidates, obligation helpers |
| UI/UX | `conformance/ui_ux_source_gate.py` | Legacy exact-source gate companion |
| UI/UX | `conformance/matrix.py` | Declaration-only matrix cells for UI views while package absent |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `intent_ir.graphrag.*` (LPR retrieval / skillcenter projections)
- `intent_ir.invocation.*`, `intent_ir.source_adapters.*`, evaluation splits
- UI matrix refill / baseline join bookkeeping that only records source-missing disposition

## Shared adapter contract (both domains)

Each route/obligation descriptor declares the LPC-G040 / LPC-041-class fields
(same contract as legal/security):

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` (or gate `domain_id`) | Exact domain id; must match parent formalization artifact |
| View | route `view_name` → `view_id(...)` | Typed view namespace; never free-form |
| Family / profile | expression + slice (`family`, `profile`) | From the domain route table only; no new families |
| Property | route `property_name` → `property_id(...)` | Property is never promoted to a family |
| Notation | route `notation_name` → `notation_id(...)` | Surface notation for the admitted view |
| Preserved semantics | translation edge `preservation` | From reviewed translation catalog edge |
| Lost semantics | `_loss_ids_for(route)` on compiled target | Explicit loss ids; never silent |
| Assumptions | domain-specific assumption axes | Declared even when empty / N/A |
| Unsupported constructs | deferred kind sets | Rejected fail-closed (not admitted) |
| Proof-safety | `authority_ceiling` + `result_authority` | Ceiling never upgrades along lineage |
| Counterexample-safety | sat/model/trace result kinds + replay digests | Counterexamples remain bound to exact request digests |

Lineage stages required on every admitted end-to-end connection:

```text
typed_origin → semantics → translation → request → result → replay → authority_lineage
```

Hermetic fixtures may supply provider execution and replay without live provers.
Tool absence is an availability result, never a mock proof (LPC-032).
Advisor / prompt confidence never establishes correctness or tool authority.

---

## 1. Intent IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `intent_ir` |
| Interface | `IntentLogicSlice@2` |
| Schema | `intent-logic-slice/v2` |
| Connector | `IntentLogicSlice` in `intent_ir/formalize/logic_slice_v2.py` |
| Formalization alignment | `IntentFormalizationCompiler@2` in `intent_ir/formalize/typed_compiler.py` |

### Ontology kept distinct

Intent keeps typed facts, skill effects, prompt candidates, skill goals,
guards/effects, workflow temporal control, tool authorization, deontic/modal
policy, and safety/liveness **property kinds** as distinct route classes.
Verification conditions remain a **view role** (never a family). Full
BDI/agency and prioritized normative overlays are deferred (LFP2-044 after
LFP2-037 / LFP2-040), not silently flattened into FOL.

Base executable routes (Wave-2):

| Route kind | Family | Profile | Property | View | Notation | Authority ceiling | Route namespace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `intent` | `first_order` | `default` | `validity` | `facts` | `intent_facts` | satisfiability | family |
| `skill` | `program` | `dynamic_hoare` | `partial_correctness` | `action_hoare` | `hoare_action_contract` | candidate | profile |
| `prompt` | `first_order` | `default` | `validity` | `facts` | `prompt_candidate` | candidate | family |
| `goal` | `intention_agency` | `skill_goals` | `goal_satisfaction` | `skill_goals` | `skill_goal` | candidate | profile |
| `guard` | `first_order` | `guards_effects` | `guard` | `guards_effects` | `guard_predicate` | satisfiability | profile |
| `workflow` | `temporal` | `workflow_temporal` | `ordering` | `workflows` | `workflow_temporal` | bounded / model_check | profile |
| `authorization` | `authorization` | `tool_permissions` | `authorization` | `tool_permissions` | `tool_permission` | authorization | profile |
| `policy` | `deontic` | `default` | `obligation` | `norms` | `deontic_norm` | candidate | family |
| `safety` | `temporal` | `safety` | `safety` | `safety` | `safety_invariant` | bounded / model_check | **property** |
| `liveness` | `temporal` | `liveness` | `liveness` | `liveness` | `liveness_progress` | finite_trace / monitor | **property** |
| `verification_condition` | `program` (underlying) | `dynamic_hoare` | `validity` | `verification_condition` | `vc_surface` | candidate | **view_role** |

Evidence subset required on the intent slice catalog:

`intent`, `skill`, `prompt`, `goal`, `guard`, `workflow`, `authorization`, `policy`.

Namespace discipline notes:

- Safety and liveness are **property kinds** under `temporal`, never families.
- Verification-condition is a **view role**, never a family (`family_id` on the
  formalization route for VC must stay empty; the connector uses `program`
  only as the underlying typed-expression family for identity).
- Prompt-derived formulas are **candidates** until deterministic parse,
  typecheck, and verification receipts exist.
- Tool authority never follows confidence alone — grounded permission evidence
  is required on authorization routes.
- Advisor confidence cannot establish intent correctness
  (`assumption:advisor_confidence_not_correctness` on every route).

### Assumption axes (intent)

Every admitted intent route declares all five axes (empty only when N/A):

| Axis | Role |
| --- | --- |
| `source_grounding` | Source spans, entity/predicate/action/goal identity, polarity |
| `tool_authority` | Grounded permission, delegation scope, action identity (authorization) |
| `bound` | Quantifier instantiations, program steps, trace length, VC depth |
| `policy_authority` | Policy authority bound, world policy, effect polarity |
| `advisor_scope` | Advisor/prompt candidate-only; confidence ≠ correctness |

Route-local examples:

| Route | Notable assumptions |
| --- | --- |
| `intent` | `assumption:source_grounding`, `assumption:entity_identity`, `bound:quantifier_instantiations` |
| `skill` | `assumption:action_identity`, `assumption:frame_conditions`, `bound:program_steps` |
| `prompt` | `assumption:prompt_derived`, `assumption:parse_typecheck_required`, `assumption:prompt_confidence_not_correctness` |
| `goal` | `assumption:goal_identity`, `assumption:agent_identity`, `assumption:intention_force` |
| `guard` | `assumption:guard_polarity`, `assumption:effect_polarity` |
| `workflow` | `bound:trace_length`, `assumption:fairness_constraint`, `assumption:edge_direction` |
| `authorization` | `assumption:grounded_permission_required`, `assumption:tool_authority_not_from_confidence`, `assumption:policy_authority_bound` |
| `policy` | `assumption:operator_force`, `assumption:norm_polarity`, `assumption:world_policy` |
| `safety` | `assumption:invariant_polarity`, `assumption:bad_state_exclusion`, `bound:trace_length` |
| `liveness` | `assumption:progress_condition`, `assumption:fairness_constraint`, `assumption:finite_trace` |
| `verification_condition` | `assumption:obligation_identity`, `assumption:assumption_set`, `bound:vc_depth` |

### Preserved / lost semantics (intent)

| Route | Explicit losses (`loss_ids`) |
| --- | --- |
| `workflow` | `loss.bounded_trace` |
| `safety` | `loss.bounded_trace` |
| `liveness` | `loss.finite_trace`, `loss.fairness_restriction` |
| `goal` | `loss.intention_reification` |
| `policy` | `loss.deontic_reification` |
| `skill` | `loss.frame_approximation` |
| `prompt` | `loss.prompt_candidate_only` |
| `verification_condition` | `loss.vc_view_role` |
| `intent` / `guard` / `authorization` | (empty; preservation via translation edge only) |

Translation families used by routes: `program`, `policy_modal`, `state_temporal`
(reviewed catalog edges only; no free-form translation).

### Unsupported / deferred (intent)

Rejected for executable `IntentLogicSlice@2` (deferred overlays / non-family
roles):

`bdi_overlay`, `agency_overlay`, `normative_overlay`, `argumentation`,
`description_logic`, `free_form`, `boolean_receipt`, `graph_projection`,
`proof_translation`, `structural_round_trip`.

Also rejected by the typed compiler when offered as families:

- Operation roles: `verification_condition`, `graph_projection`,
  `proof_translation`, `structural_round_trip`, `round_trip`, `decompiler`,
  `external_provers`, `prover_router`, `prover`
- Property kinds: `safety`, `liveness`, `safety_liveness`, `invariant`,
  `validity`, `reachability`

### Proof-safety and counterexample-safety (intent)

- Authority ceilings are route-local (`SATISFIABILITY`, `CANDIDATE`, `BOUNDED`,
  `AUTHORIZATION`, `FINITE_TRACE`); lineage records never upgrade authority.
- Prompt, skill, goal, policy, and VC routes remain **candidate-class** until
  stronger receipts exist; they do not mint theorem authority.
- Workflow/safety use bounded model-check authority with explicit trace bounds;
  liveness uses finite-trace / monitor authority, not unbounded liveness claims.
- Authorization uses authorization authority only; mock confidence cannot
  upgrade to allow.
- Sat/model/monitor results and replay digests stay bound to the exact request
  digest through the seven lineage stages.

---

## 2. UI/UX IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `ui_ux_ir` |
| Interface | `UIUXLogicSlice@2` |
| Schema | `ui-ux-logic-slice/v2` |
| Connector / gate | `UIUXLogicSlice` + `UIUXSourceGate@2` in `conformance/ui_ux_logic_gate_v2.py` |
| Formalization interface | `UIUXFormalizationAdapter@2` (declaration until source import) |
| Package path (exact) | `ipfs_datasets_py/logic/ui_ux_ir` (absent from pinned revision) |

### Production disposition (fail-closed)

The pinned datasets tree **does not invent, copy, or edit** `ui_ux_ir`.

| Source presence | Slice status | Backend admission |
| --- | --- | --- |
| Absent (current pinned tree) | `declaration_only` | `require_admitted()` always raises `UIUXSliceAdmissionError` |
| Present at exact reviewed path | `adapter_gap` (exactly one owner-scoped gap) | Still not admitted until gap closed |
| After gap closed (future) | `admitted` | May emit `DomainLogicSlice@2` under domain `ui_ux_ir` |

`UIUXLogicSlice@2` construction currently **forbids** `status=admitted`
(`UIUXLogicSlice@2 cannot be admitted until the adapter gap is closed`). That
is intentional: incomplete UI formalization never masquerades as an admitted
domain slice (aligns with LPC-044 unadmitted-slice rejection).

Gate side-effect policy: pure filesystem presence checks only. No network,
installation, model download, or subprocess at import or scan time. The gate
never blocks other domain work (`blocks_other_work=false`).

### Ontology kept distinct (declared requirement surfaces)

Even while source is missing, the slice records the fixed owner-scoped
requirement surfaces that any future admitted adapter must cover. These are
**not** collapsed into a universal domain IR:

| Surface id | Family hint | Description |
| --- | --- | --- |
| `accessibility` | `first_order` | Accessibility property obligations over UI structure and state |
| `authorization` | `authorization` | Authorization and permission constraints over UI actions |
| `interaction_event` | `event_calculus` | Interaction / event-calculus obligations for user/system events |
| `observable_state` | `transition_system` | Observable navigation and runtime state transitions |
| `ontology_frame` | `frame_logic` | Ontology/frame (F-logic) component and relation structure |
| `workflow` | `temporal` | Workflow temporal obligations over multi-step UI journeys |

Capability-matrix declaration-only views (source not in pinned revision):

| View id | Family | Profile notes |
| --- | --- | --- |
| `ui-ux-ir-view/ontology/v1` | `frame_logic` | UI ontology / F-logic |
| `ui-ux-ir-view/event-calculus/v1` | `event_calculus` | Interaction events |
| `ui-ux-ir-view/tdfol/v1` | `tdfol` | TDFOL/DCEC UI norms (not legal TDFOL domain id) |
| `ui-ux-ir-view/navigation-temporal/v1` | `temporal` | Navigation temporal |
| `ui-ux-ir-view/navigation-transition/v1` | `transition_system` | Navigation transitions |
| `ui-ux-ir-view/accessibility/v1` | `first_order` | profile `accessibility_property` |

Owner-scoped adapter gap scopes (exactly one gap when source present):

`accessibility`, `authorization`, `component_frame`, `event`,
`navigation_state`, `permission`, `privacy`, `runtime_journey`, `tdfol_dcec`,
`workflow`.

### Assumption axes (UI/UX — required when admitted)

Until source lands, axes are declared on the adapter acceptance contract as
requirements rather than per-route hermetic fixtures. Any future admitted UI
route must still bind the shared LPC-040 inventory and declare:

| Axis | Role |
| --- | --- |
| Source grounding | Exact package fingerprint + document/source digests |
| Interaction identity | Component, event, and navigation-state identity |
| Bound | Finite journey / trace / state-space bounds |
| Policy / permission authority | UI action permissions independent of presentation confidence |
| Frame / ontology | `frame_logic` dual-read aliases (`FLogic`, `F-logic` → `frame_logic`) |

Frame-logic alias canonicalization is part of adapter-gap acceptance
(`declared_syntax_parsing`, `frame_logic_alias_canonicalization`,
`typed_structural_round_trips`). Token-presence greps are explicitly
**rejected** acceptance (`token_presence`).

### Preserved / lost semantics (UI/UX)

While declaration-only, no executable translation losses are emitted (no
backend request). When the owner-scoped adapter lands, each route must attach
reviewed translation-edge preservation and explicit `loss_ids` (same contract
as Intent/Security). Free-form text as typed origin is rejected
(`UIUXFreeFormRejectedError`).

Expected loss classes once admitted (contract placeholders; not live
executable losses today):

| Surface | Expected loss class |
| --- | --- |
| `workflow` / navigation temporal | Bounded journey / finite-trace losses |
| `observable_state` | Finite navigation state-space losses |
| `ontology_frame` | Frame-logic reification / alias canonicalization losses |
| `interaction_event` | Event-calculus approximation losses |
| `accessibility` | Finite structure / state abstraction losses |
| `authorization` | None that upgrade permission authority from confidence |

### Unsupported / deferred (UI/UX)

| Construct | Disposition |
| --- | --- |
| Free-form / token-presence formalization | Rejected |
| Universal domain IR merging UI with intent/legal/security | Forbidden |
| Fabricated `ui_ux_ir` package by the gate | Forbidden (`writes_ui_ux_ir=false`) |
| `status=admitted` before adapter gap closed | Forbidden |
| Backend request without admitted `DomainLogicSlice@2` | Rejected (LPC-044) |

### Proof-safety and counterexample-safety (UI/UX)

- Declaration-only / source-missing cells use matrix availability
  `source_missing` and support `declaration_only`; they do not mint proof
  authority.
- `require_admitted()` always fails closed on the current gate projection.
- Future admitted routes must keep authority ceilings route-local and never
  upgrade along lineage; counterexamples stay digest-bound to exact requests.
- Tool/prover absence remains availability, never a mock proof (LPC-032).

---

## Non-collapse rules (intent ↔ ui_ux ↔ other domains)

| Rule | Enforcement |
| --- | --- |
| Distinct domain ids | `intent_ir` ≠ `ui_ux_ir` on every slice; neither equals `legal_ir`, `security_ir`, `software_verification`, or `crypto_ir` |
| Distinct connector interfaces | `IntentLogicSlice@2` / `UIUXLogicSlice@2` |
| Shared families, domain-local profiles | Catalog families may overlap (e.g. `temporal`, `authorization`, `first_order`); profiles and views remain domain-scoped |
| No universal domain IR | Free-form / universal routes are deferred or rejected; UI gate refuses package invention |
| Property ≠ family | Safety, liveness stay property kinds; VC stays a view role |
| Intent ≠ UI workflow | Intent `workflow_temporal` under `intent_ir` does not emit `ui_ux_ir` navigation slices |
| UI TDFOL ≠ legal TDFOL | Matrix view `ui-ux-ir-view/tdfol/v1` is UI-domain declaration; legal TDFOL/DCEC/frame remain legal-domain owners (LPC-041) |
| UI frame_logic ≠ free F-logic family rename | Dual-read aliases canonicalize to `frame_logic`; not a new family |
| Intent deontic policy ≠ legal deontic IR | Intent policy uses domain `intent_ir` with intent assumption axes |
| No new families | Adapters only select existing catalog families (LPC-G040) |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| Intent safety/liveness | Semantic families named `safety` / `liveness` |
| Intent verification_condition | Family named `verification_condition` |
| Intent tool authorization | Permission grant from advisor/prompt confidence |
| Intent goal / BDI base | Full agency overlay without LFP2-044 admission |
| UI declaration-only cell | Admitted `DomainLogicSlice@2` or proof receipt |
| UI ontology / accessibility | Generic FOL free-form text |
| UI TDFOL/DCEC surface | Legal-domain TDFOL/DCEC slices without domain rebinding |
| Any domain | Free-form text as typed origin / universal domain IR |

## End-to-end admission checklist

### Intent (live)

For each admitted route the connector must:

1. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
2. Cross-check `resolve_intent_route` admission and namespace invariants
   (property kinds / view roles).
3. Emit `DomainLogicSlice@2` via `from_typed_expression` with domain
   `intent_ir`.
4. Call `require_admitted()` and `validate_against(document, expression)`.
5. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
6. Attach translation-edge preservation and explicit `loss_ids`.
7. Record hermetic execution/replay without authority upgrade.
8. Cover all seven lineage stages with digest coherence source → request →
   execution → replay.

### UI/UX (current pinned tree)

For the gate projection:

1. Scan exact package path; record `source_missing` when absent.
2. Project `UIUXLogicSlice@2` with `declaration_only` status and all six
   requirement surfaces.
3. Refuse `require_admitted()` and refuse formalization of free-form payloads.
4. Never write under `ui_ux_ir`.
5. When source later appears: emit **exactly one** owner-scoped adapter gap
   whose acceptance requires declared-syntax parsing, frame_logic alias
   canonicalization, and typed structural round trips — then lower through the
   same `DomainLogicSlice@2` path as Intent under domain `ui_ux_ir`.

Incomplete slices fail closed before backend request construction (LPC-044
rejects executable requests without an admitted `DomainLogicSlice@2`).

## File ownership (LPC-043)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | Intent domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py` | Intent formalization routes / never-family guards |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | UI/UX exact-source gate + `UIUXLogicSlice@2` / formalization adapter declaration |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md` | This conformance note |

Inventory aliases (`intent_ir.domain_slice`, `ui_ux_ir.domain_slice`) refer to
the adapter **role** satisfied by the modules above. Intent’s production write
path is `formalize/logic_slice_v2.py`. UI/UX’s production surface is the
gate/declaration slice until exact source import closes the owner-scoped
adapter gap; the predicted package path `logic/ui_ux_ir/domain_slice.py` must
not be fabricated by this task.

Related tests (validation surfaces):

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/tests/unit/logic/intent_ir/` | Intent IR unit suite (schema, formalize, invocation, graphrag, …) |
| `ipfs_datasets_py/tests/unit/logic/ui_ux_ir/` | Predicted unit path for a future package-local suite |
| `ipfs_datasets_py/tests/conformance/logic/test_ui_ux_logic_gate_v2.py` | Live gate / declaration-only conformance for UI/UX |

## Acceptance

- **Intent** keeps facts, skill effects, prompt candidates, goals, guards,
  workflows, tool authorization, deontic policy, safety/liveness property
  kinds, and VC view roles distinct, and lowers each admitted route through
  `DomainLogicSlice@2` with domain `intent_ir`.
- **UI/UX** keeps accessibility, authorization, interaction/event, observable
  state, ontology/frame, and workflow surfaces distinct under domain
  `ui_ux_ir`, records them on `UIUXLogicSlice@2`, and **does not** admit
  backend slices or invent a package while source is missing.
- Both domains use the same adapter contract as legal/security (source domain,
  view, family/profile, property, notation, preserved/lost semantics,
  assumptions, unsupported constructs, proof-safety, counterexample-safety).
- No adapter invents a universal domain IR or collapses another domain’s
  ontology.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`
