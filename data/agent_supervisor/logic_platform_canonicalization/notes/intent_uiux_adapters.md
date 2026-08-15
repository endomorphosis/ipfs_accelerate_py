# LPC-043 Intent and UI/UX Domain Adapter Conformance

**Task:** LPC-043 — Intent and UI/UX domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Same adapter contract as legal/security. No universal domain IR.  
**Conflict policy:** Own intent and UI/UX slice adapters only.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`

## Purpose

Intent IR and UI/UX IR each own a sealed domain ontology. New formalization
writes lower those ontologies **through** `DomainLogicSlice@2` (LPC-040)
without inventing a universal domain IR and without silently remapping one
domain’s families into another.

This note freezes adapter locations, domain identities, admitted route tables
(or declaration-only requirement surfaces for UI/UX while source is absent),
namespace axes, assumption axes, preservation/loss declarations, authority
ceilings, and non-collapse rules. It is the durable LPC-043 evidence for the
two domain adapters that share the legal/security adapter contract.

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
| Domain | domain-specific id (see table below) |

Construction pattern used by admitted domain connectors (Intent fully; UI/UX
only after exact source import closes the adapter gap):

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

Inventory LPC-004 / LPC-043 predicted paths named `domain_slice.py`. Live
accelerate implementations live as `*LogicSlice@2` connectors that **emit**
`DomainLogicSlice@2` records (or, for UI/UX while source is missing, a typed
declaration-only slice that refuses admission). Those connectors are the
production domain adapters for LPC-043.

| Domain | Domain id | Adapter interface | Production module | Emits |
| --- | --- | --- | --- | --- |
| Intent | `intent_ir` | `IntentLogicSlice@2` | `ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted intent route |
| UI/UX | `ui_ux_ir` | `UIUXLogicSlice@2` | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | Declaration-only / adapter-gap `UIUXLogicSlice@2` until source lands; never invents package files |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Domain | Supporting modules | Role |
| --- | --- | --- |
| Intent | `intent_ir/formalize/typed_compiler.py` | `resolve_intent_route`, never-family property/view roles, formula candidate classification |
| UI/UX | `conformance/ui_ux_source_gate.py` | Exact-source gate companion; presence checks only |
| UI/UX | `UIUXFormalizationAdapter@2` in `ui_ux_logic_gate_v2.py` | Declaration-only formalization interface until owner-scoped adapter gap closes |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `intent_ir.graphrag.*` retrieval / LPR surfaces
- `intent_ir.invocation.*`, `intent_ir.source_adapters.*`, skillcenter snapshots
- UI/UX package invent/copy/edit under `logic/ui_ux_ir` (forbidden by the source gate)
- Token-presence greps as formalization acceptance (explicitly rejected)

## Shared adapter contract (legal / security / intent / UI/UX)

Each route/obligation (or UI/UX requirement surface) declares the LPC-G040 /
LPC-041-class fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` (or `domain_id` on `UIUXLogicSlice@2`) | Exact domain id; must match parent formalization artifact when admitted |
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

Lineage stages required on every admitted end-to-end connection (Intent; UI/UX
after admission is enabled):

```text
typed_origin → semantics → translation → request → result → replay → authority_lineage
```

Hermetic fixtures may supply provider execution and replay without live provers.
Tool absence is an availability result, never a mock proof (LPC-032).
Advisor confidence never establishes intent correctness. Free-form token
presence never establishes UI/UX formalization.

---

## 1. Intent IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `intent_ir` |
| Interface | `IntentLogicSlice@2` |
| Schema | `intent-logic-slice/v2` |
| Connector | `IntentLogicSlice` in `intent_ir/formalize/logic_slice_v2.py` |
| Formalization alignment | `resolve_intent_route` / never-family guards in `typed_compiler.py` |

### Ontology kept distinct

Intent routes use intent-scoped views, notations, and assumption axes. They
share **logic families** from the catalog (first-order, program, temporal,
authorization, deontic, intention_agency, …) but never collapse into a
universal domain IR, free-form text origin, or another domain’s id.

Evidence subset required on the intent slice catalog:

`intent`, `skill`, `prompt`, `goal`, `guard`, `workflow`, `authorization`,
`policy`.

Admitted Wave-2 routes (before LFP2-044 overlays):

| Route kind | Family | Profile | Property | View | Notation | Namespace | Authority ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `intent` | `first_order` | `default` | `validity` | `facts` | `intent_facts` | family | satisfiability |
| `skill` | `program` | `dynamic_hoare` | `partial_correctness` | `action_hoare` | `hoare_action_contract` | profile | candidate |
| `prompt` | `first_order` | `default` | `validity` | `facts` | `prompt_candidate` | family | candidate |
| `goal` | `intention_agency` | `skill_goals` | `goal_satisfaction` | `skill_goals` | `skill_goal` | profile | candidate |
| `guard` | `first_order` | `guards_effects` | `guard` | `guards_effects` | `guard_predicate` | profile | satisfiability |
| `workflow` | `temporal` | `workflow_temporal` | `ordering` | `workflows` | `workflow_temporal` | profile | bounded / model_check |
| `authorization` | `authorization` | `tool_permissions` | `authorization` | `tool_permissions` | `tool_permission` | profile | authorization |
| `policy` | `deontic` | `default` | `obligation` | `norms` | `deontic_norm` | family | candidate |
| `safety` | `temporal` | `safety` | `safety` | `safety` | `safety_invariant` | **property** | bounded / model_check |
| `liveness` | `temporal` | `liveness` | `liveness` | `liveness` | `liveness_progress` | **property** | finite_trace / monitor |
| `verification_condition` | `program` (typed only) | `dynamic_hoare` | `validity` | `verification_condition` | `vc_surface` | **view role** | candidate |

Namespace discipline notes:

- Safety and liveness are **property kinds** under `temporal`, never families.
- Verification-condition is a **view role**, never a family (`family_id` on the
  formalization route for VC must remain empty; the slice uses program only for
  typed-expression identity under the view role).
- Prompt- and advisor-derived formulas are candidates until deterministic
  parse, typecheck, and verification receipts exist.
- Tool authority never follows confidence alone.

### Assumption axes (intent)

Every admitted intent route declares all five axes (empty only when N/A):

| Axis | Role |
| --- | --- |
| `source_grounding` | Entity/predicate/action/goal/agent identity, polarity, signatures |
| `tool_authority` | Grounded permissions, delegation scope; never confidence-derived |
| `bound` | Quantifier instantiations, program steps, trace length, VC depth |
| `policy_authority` | Policy authority bound, world policy, effect polarity |
| `advisor_scope` | Advisor candidate-only; confidence is never correctness |

Representative route assumptions:

| Route | Notable assumptions |
| --- | --- |
| `intent` | `assumption:source_grounding`, `assumption:entity_identity`, `bound:quantifier_instantiations` |
| `skill` | `assumption:action_identity`, `assumption:frame_conditions`, `bound:program_steps` |
| `prompt` | `assumption:prompt_derived`, `assumption:parse_typecheck_required`, `assumption:prompt_confidence_not_correctness` |
| `goal` | `assumption:goal_identity`, `assumption:agent_identity`, `assumption:intention_force` |
| `guard` | `assumption:guard_polarity`, `assumption:effect_polarity` |
| `workflow` | `bound:trace_length`, `assumption:fairness_constraint` |
| `authorization` | `assumption:grounded_permission_required`, `assumption:tool_authority_not_from_confidence` |
| `policy` | `assumption:operator_force`, `assumption:policy_authority_bound` |
| `safety` | `assumption:invariant_polarity`, `bound:trace_length` |
| `liveness` | `assumption:progress_condition`, `assumption:finite_trace` |
| `verification_condition` | `assumption:obligation_identity`, `bound:vc_depth` |

All routes also carry advisor-scope candidate assumptions
(`assumption:advisor_candidate_only`,
`assumption:advisor_confidence_not_correctness`).

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
| `intent` / `guard` / `authorization` | (none beyond catalog edge preservation) |

Preservation comes from reviewed translation edges
(`vc_to_smt`, `program_to_smt`, `intention_to_fol_reified`,
`temporal_ltl_to_tla_plus`, `authorization_to_secpal`,
`deontic_to_fol_reified`, `temporal_mtl_to_runtime_mtl`).

### Unsupported / deferred (intent)

Rejected for executable `IntentLogicSlice@2` (LFP2-044 overlays after
LFP2-037 / LFP2-040):

`bdi_overlay`, `agency_overlay`, `normative_overlay`, `argumentation`,
`description_logic`, `free_form`, `boolean_receipt`, `graph_projection`,
`proof_translation`, `structural_round_trip`.

### Proof-safety and counterexample-safety (intent)

- Authority ceilings are route-local (`SATISFIABILITY`, `CANDIDATE`,
  `BOUNDED`, `AUTHORIZATION`, `FINITE_TRACE`); lineage records never upgrade
  authority along the chain.
- Prompt, skill, goal, policy, and VC results are **candidates** until stronger
  receipts exist; advisor confidence cannot mint proof.
- Authorization uses authorization authority only; confidence cannot grant
  tool authority.
- Workflow/safety model-check results are bound-depth results; liveness uses
  finite-trace / monitor authority, not unbounded progress claims.
- Hermetic sat/model/monitor outputs and replay digests stay bound to the exact
  request digest through all seven lineage stages.

---

## 2. UI/UX IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `ui_ux_ir` |
| Interface | `UIUXLogicSlice@2` |
| Schema | `ui-ux-logic-slice/v2` |
| Connector | `UIUXLogicSlice` / `UIUXLogicSliceConnector` in `conformance/ui_ux_logic_gate_v2.py` |
| Exact-source gate | `UIUXSourceGate@2` (same module; companion `ui_ux_source_gate.py`) |
| Formalization adapter (declaration) | `UIUXFormalizationAdapter@2` |
| Package path (exact source) | `ipfs_datasets_py/logic/ui_ux_ir` |

### Ontology kept distinct

UI/UX owns accessibility, interaction/event, workflow, ontology/frame,
authorization/permission, and observable-state surfaces. It reuses catalog
families as **hints** only while source is missing; it never collapses into a
universal domain IR, free-form token presence, or another domain’s adapter.

Owner-scoped requirement surfaces (`UIUXLogicSlice@2`):

| Surface id | Family hint | Role |
| --- | --- | --- |
| `accessibility` | `first_order` | Accessibility properties over UI structure and state |
| `authorization` | `authorization` | Permission constraints over UI actions |
| `interaction_event` | `event_calculus` | User/system interaction and event obligations |
| `observable_state` | `transition_system` | Navigation and runtime state transitions |
| `ontology_frame` | `frame_logic` | Component/relation ontology (F-logic dual-read) |
| `workflow` | `temporal` | Multi-step UI journey temporal obligations |

Owner-scoped adapter gap scopes (when exact source is present):

`accessibility`, `authorization`, `component_frame`, `event`,
`navigation_state`, `permission`, `privacy`, `runtime_journey`,
`tdfol_dcec`, `workflow`.

### Source-gate dispositions (current production)

The pinned datasets tree does **not** invent, copy, or edit `ui_ux_ir`.

| Source presence | Gate disposition | Slice status | Admits `DomainLogicSlice@2`? |
| --- | --- | --- | --- |
| Absent | `declaration_only` / `source_missing` | `declaration_only` | No — `require_admitted()` fails closed |
| Present | emit exactly one owner-scoped adapter gap | `adapter_gap` | No — until adapter gap closes |
| Present + gap closed | (future) admitted routes | `admitted` (not yet enabled) | Yes — only then |

Matrix disposition when source is missing:

- support: `declaration_only`
- availability: `source_missing`
- authority ceiling: `none`
- `blocks_other_work=false` (other domain work continues)

### Assumption axes (UI/UX)

While declaration-only / adapter-gap, the slice records requirement surfaces
and gate identity rather than admitting backend routes. When admission is
enabled after exact source import, each surface must still declare the shared
adapter contract fields (source domain, view, family/profile, property,
notation, preserved/lost semantics, assumptions, unsupported constructs,
proof-safety, counterexample-safety) under domain id `ui_ux_ir`.

Frame-logic dual-read is sealed:

| Alias label | Canonical family |
| --- | --- |
| `frame_logic` | `frame_logic` |
| `FLogic` | `frame_logic` |
| `F-logic` | `frame_logic` |

Unknown labels fail closed; non-`frame_logic` resolutions are rejected.

### Preserved / lost semantics (UI/UX)

Adapter-gap acceptance requirements (explicit, non-token):

| Required acceptance | Meaning |
| --- | --- |
| `declared_syntax_parsing` | Parse declared UI/UX syntax; not free-form greps |
| `frame_logic_alias_canonicalization` | Dual-read aliases resolve to `frame_logic` |
| `typed_structural_round_trips` | Typed structural round trips, not token presence |

Explicitly rejected acceptance:

| Rejected | Reason |
| --- | --- |
| `token_presence` | Token greps never establish formalization |

Preserve under adapter gap: `authority_flags`, `golden_vectors`,
`graph_schemas`, `source_maps`.

### Unsupported / deferred (UI/UX)

- Free-form text / token-presence formalization (`UIUXFreeFormRejectedError`)
- Package invent/copy/edit via the gate (`UIUXPackageWriteForbiddenError`)
- Backend route admission while source is missing or adapter gap is open
  (`UIUXSliceAdmissionError`)
- Universal domain IR / free-form typed origins

### Proof-safety and counterexample-safety (UI/UX)

- Declaration-only disposition cannot claim non-empty authority ceilings.
- Absent source yields authority ceiling `none` / `unknown` only.
- Formalization refuses until exact source import and the owner-scoped adapter
  implement formalization; present source alone does not mint proof.
- When admitted routes eventually lower through `DomainLogicSlice@2`, they use
  domain id `ui_ux_ir` and the shared proof/counterexample safety rules
  (ceiling never upgrades; sat/model/trace results digest-bound to requests).

---

## Non-collapse rules (intent ↔ UI/UX ↔ other domains)

| Rule | Enforcement |
| --- | --- |
| Distinct domain ids | `intent_ir` ≠ `ui_ux_ir` ≠ legal/security/software/crypto on every slice |
| Distinct connector interfaces | `IntentLogicSlice@2` / `UIUXLogicSlice@2` |
| Shared families, domain-local profiles/views | Catalog families may overlap (e.g. `authorization`, `temporal`); profiles, views, notations remain domain-scoped |
| No universal domain IR | Free-form / universal routes deferred or rejected on both adapters |
| Property ≠ family | Intent safety/liveness stay property kinds; VC stays a view role |
| Intent ≠ legal deontic | Intent policy uses intent norms/candidate authority; legal keeps TDFOL/DCEC/frame axes |
| Intent ≠ security authorization | Intent tool-permissions profile stays under `intent_ir`; security SecPAL/threat views stay under `security_ir` |
| UI/UX ≠ intent workflow | UI journey temporal surfaces use domain `ui_ux_ir`; intent workflow temporal uses `intent_ir` |
| UI/UX frame_logic ≠ legal frame collapse | UI/UX dual-read aliases canonicalize to `frame_logic` without inventing object-framing FOL |
| No new families | Adapters only select existing catalog families (LPC-G040) |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| Intent safety/liveness | Semantic families named `safety` / `liveness` |
| Intent verification_condition | Family id `verification_condition` |
| Intent prompt/advisor confidence | Proof or tool authority |
| Intent tool authorization | Security IR SecPAL/threat routes without domain rebinding |
| UI/UX token presence | Admitted typed formalization |
| UI/UX absent package | Invented `ui_ux_ir` package or admitted backend routes |
| Any domain | Free-form text as typed origin / universal domain IR |

## End-to-end admission checklist

### Intent (admitted routes)

For each admitted route the connector must:

1. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
2. Emit `DomainLogicSlice@2` via `from_typed_expression` with domain `intent_ir`.
3. Call `require_admitted()` and `validate_against(document, expression)`.
4. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
5. Attach translation-edge preservation and explicit `loss_ids`.
6. Record hermetic execution/replay without authority upgrade.
7. Cover all seven lineage stages with digest coherence source → request → execution → replay.
8. Reject advisor confidence as correctness and deferred overlays fail-closed.

### UI/UX (source-gated)

For each gate scan the connector must:

1. Observe exact package presence under `logic/ui_ux_ir` without writing it.
2. If absent: emit `declaration_only` / `source_missing` disposition; refuse admission.
3. If present: emit exactly one content-addressed owner-scoped adapter gap covering all requirement surfaces and adapter scopes.
4. Never block other domain work; never claim package writes.
5. Reject free-form formalization and token-presence acceptance.
6. Canonicalize F-logic dual-read aliases to `frame_logic` only.
7. Only after the adapter gap is closed: admit routes through `DomainLogicSlice@2` with domain `ui_ux_ir` under the shared legal/security contract.

Incomplete or unadmitted slices fail closed before backend request construction
(LPC-044 rejects executable requests without an admitted `DomainLogicSlice@2`).

## File ownership (LPC-043)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | Intent domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py` | Intent route resolution / never-family guards |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | UI/UX domain adapter + source gate → `UIUXLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md` | This conformance note |

Inventory aliases (`intent_ir.domain_slice`, `ui_ux_ir.domain_slice`) refer to
the adapter role satisfied by the `logic_slice_v2` / `ui_ux_logic_gate_v2`
modules above; those modules are the production write path for admitted domain
lowering (Intent) and source-gated declaration (UI/UX) under LPC-043.

## Acceptance

- **Intent** keeps its facts, skill effects, prompt candidates, goals, guards,
  workflows, tool authorization, policy/norms, safety, liveness, and VC-view
  ontology and lowers each admitted route through `DomainLogicSlice@2` with
  domain `intent_ir`.
- **UI/UX** keeps accessibility, interaction/event, workflow, ontology/frame,
  authorization, and observable-state surfaces under domain `ui_ux_ir`, fails
  closed while source is missing, never invents a universal domain IR or
  package files, and only admits routes after the owner-scoped adapter gap
  closes.
- Both adapters declare the same contract fields as legal/security: source
  domain, view, family/profile, property, notation, preserved/lost semantics,
  assumptions, unsupported constructs, proof-safety, and counterexample-safety.
- No adapter invents a universal domain IR or collapses another domain’s
  ontology.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`
