# LPC-043 Intent and UI/UX Domain Adapter Conformance

**Task:** LPC-043 — Intent and UI/UX domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Same adapter contract as legal/security. No universal domain IR.  
**Conflict policy:** Own intent and UI/UX slice adapters only. Never invent a universal domain IR or collapse intent ↔ ui_ux (or either into legal/security/software/crypto).  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`

## Purpose

Intent IR and UI/UX each own a sealed domain ontology. New formalization writes
lower those ontologies **through** `DomainLogicSlice@2` (LPC-040) under the
same LPC-G040 / LPC-041-class adapter contract used by legal and security
adapters: source domain, view, family/profile, property, notation,
preserved/lost semantics, assumptions, unsupported constructs, proof-safety,
and counterexample-safety.

This note freezes adapter locations, domain identities, admitted (or
declaration-only) route/surface tables, assumption axes, preservation/loss
declarations, authority ceilings, and non-collapse rules. It is the durable
LPC-043 evidence for the two domain adapters that feed backend requests via
admitted slices only — or fail closed when admission is not available.

## Canonical lowering path

```text
Domain IR view / obligation / requirement surface
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

Construction pattern used by the executable Intent adapter (identical contract
shape as legal/security/software/crypto connectors):

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

UI/UX currently **cannot** complete this path while source is missing: its
connector records a typed declaration-only / adapter-gap slice and
`require_admitted()` fails closed (see §2). That is conformance, not a
universal IR bypass.

## Production adapter modules

Inventory LPC-004 predicted paths named `domain_slice.py`. Live accelerate
implementations live as `*LogicSlice@2` connectors that **emit**
`DomainLogicSlice@2` records (or refuse admission under an exact-source gate).
Those connectors are the production domain adapters for LPC-043.

| Domain | Domain id | Adapter interface | Production module | Emits / disposition |
| --- | --- | --- | --- | --- |
| Intent | `intent_ir` | `IntentLogicSlice@2` | `ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted intent route |
| UI/UX | `ui_ux_ir` | `UIUXLogicSlice@2` | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | Declaration-only or adapter-gap record; **no** admitted `DomainLogicSlice@2` until exact source import + derived adapter close the gap |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Domain | Supporting modules | Role |
| --- | --- | --- |
| Intent | `intent_ir/formalize/typed_compiler.py` | `INTENT_IR_DOMAIN_ID`, `ADMITTED_INTENT_VIEW_NAMES`, `resolve_intent_route` admission cross-check |
| Intent | `intent_ir/formalize/compiler.py`, `obligations.py` | Formalization compiler and obligation projection |
| UI/UX | `conformance/ui_ux_source_gate.py` | `UIUXSourceGate@1` / `UIUXFormalizationAdapter@1` exact-source gate (v1) |
| UI/UX | capability matrix / baseline join | Matrix cells remain `declaration_only` + `source_missing` until import |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- Intent GraphRAG / SkillCenter retrieval (`intent_ir/graphrag/*`), evaluation splits, source-adapters for MCP/prompt/skillcenter projection without formalization
- Intent invocation envelopes (`intent_ir/invocation/*`) that project records without executing skills
- UI/UX matrix disposition and derived adapter-gap receipts (gate bookkeeping, not admitted lowering)

## Shared adapter contract (intent + UI/UX; same as legal/security)

Each route / obligation / requirement-surface descriptor declares the
LPC-G040 / LPC-041-class fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` (intent) or `domain_id` on `UIUXLogicSlice@2` | Exact domain id; must match parent formalization artifact when admitted |
| View | route `view_name` → `view_id(...)` (intent); formal view ids in matrix for UI/UX | Typed view namespace; never free-form |
| Family / profile | expression + slice (`family`, `profile`) | From the domain route / surface table only; no new families |
| Property | route `property_name` → `property_id(...)` | Property is never promoted to a family |
| Notation | route `notation_name` → `notation_id(...)` | Surface notation for the admitted view |
| Preserved semantics | translation edge `preservation` | From reviewed translation catalog edge |
| Lost semantics | `_loss_ids_for(route)` / explicit gap losses | Explicit loss ids; never silent |
| Assumptions | domain-specific assumption axes | Declared even when empty / N/A |
| Unsupported constructs | deferred kind sets / free-form rejection | Rejected fail-closed (not admitted) |
| Proof-safety | `authority_ceiling` + `result_authority` | Ceiling never upgrades along lineage |
| Counterexample-safety | sat/model/trace result kinds + replay digests | Counterexamples remain bound to exact request digests |

Lineage stages required on every admitted end-to-end connection (intent):

```text
typed_origin → semantics → translation → request → result → replay → authority_lineage
```

Hermetic fixtures may supply provider execution and replay without live provers.
Tool absence is an availability result, never a mock proof (LPC-032).

---

## 1. Intent IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `intent_ir` |
| Interface | `IntentLogicSlice@2` |
| Schema | `intent-logic-slice/v2` |
| Connector | `IntentLogicSlice` in `intent_ir/formalize/logic_slice_v2.py` |
| Formalization alignment | `ADMITTED_INTENT_VIEW_NAMES` / `resolve_intent_route` in `typed_compiler.py` |

### Ontology kept distinct

Intent routes use intent-scoped profiles, views, and notations. They share
**logic families** from the catalog (`first_order`, `program`, `temporal`,
`authorization`, `deontic`, `intention_agency`, …) but never collapse into
legal, security, software verification, crypto, or UI/UX domain ids, IR
packages, or obligation tables.

Base executable routes (Wave-2, before normative/BDI overlays):

| Route kind | Family | Profile | Property | View | Notation | Authority ceiling | Namespace |
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
| `verification_condition` | `program` (expression family only) | `dynamic_hoare` | `validity` | `verification_condition` | `vc_surface` | candidate | **view_role** |

Evidence subset required on the intent slice catalog: intent, skill, prompt,
goal, guard, workflow, authorization, policy.

Namespace discipline notes:

- Safety and liveness are **property kinds** under `temporal`, never families.
- Verification condition is a **view role**, never a family
  (`family_id` on the route is only the typed expression family for identity;
  the formalization compiler route must not set `family_id` for VC).
- Advisor / prompt confidence never establishes intent correctness or tool
  authority.
- Full BDI/agency and prioritized normative overlays remain deferred
  (LFP2-044 after LFP2-037 / LFP2-040).

### Assumption axes (intent)

Every admitted intent route declares all five axes (empty only when N/A):

| Axis | Role / examples |
| --- | --- |
| `source_grounding` | Source grounding, entity/predicate/action/goal identity, polarity |
| `tool_authority` | Grounded permission, delegation scope; never confidence-granted |
| `bound` | Quantifier instantiations, program steps, trace length, VC depth |
| `policy_authority` | Policy authority bound, world policy, effect polarity |
| `advisor_scope` | Advisor candidate-only; confidence is not correctness |

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
| `intent` / `guard` / `authorization` | none beyond route-local bounds/authority |

Preservation comes from the reviewed translation catalog edge selected by
each route (`vc_to_smt`, `program_to_smt`, `intention_to_fol_reified`,
`authorization_to_secpal`, `deontic_to_fol_reified`,
`temporal_ltl_to_tla_plus`, `temporal_mtl_to_runtime_mtl`, …).

### Unsupported / deferred (intent)

Rejected for executable `IntentLogicSlice@2`:

`bdi_overlay`, `agency_overlay`, `normative_overlay`, `argumentation`,
`description_logic`, `free_form`, `boolean_receipt`, `graph_projection`,
`proof_translation`, `structural_round_trip`.

### Proof-safety and counterexample-safety (intent)

- Authority ceilings are route-local (`SATISFIABILITY`, `CANDIDATE`,
  `BOUNDED`, `AUTHORIZATION`, `FINITE_TRACE`); lineage records
  `never_upgrades=true`.
- Prompt- and advisor-derived formulas stay **candidates** until parse,
  typecheck, and verification receipts exist.
- Skill and goal routes are candidate-class until stronger proof authority
  is admitted; they do not mint kernel proof authority.
- Tool authorization uses authorization authority only; confidence alone
  never grants tool authority.
- Safety/workflow model-check results and liveness monitor results stay
  digest-bound to the exact request; hermetic fixtures supply execution and
  replay without live provers.

---

## 2. UI/UX adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `ui_ux_ir` |
| Interface | `UIUXLogicSlice@2` |
| Schema | `ui-ux-logic-slice/v2` |
| Connector | `UIUXLogicSlice` / `UIUXLogicSliceConnector` in `conformance/ui_ux_logic_gate_v2.py` |
| Exact-source gate | `UIUXSourceGate@2` (same module); v1 gate in `ui_ux_source_gate.py` |
| Formalization adapter (declaration) | `UIUXFormalizationAdapter@2` |
| Package path (when present) | `ipfs_datasets_py/logic/ui_ux_ir` |

### Ontology kept distinct (declaration-only until source import)

The pinned datasets revision does **not** contain the `ui_ux_ir` package.
`UIUXLogicSlice@2` still records the owner-scoped requirement surfaces that
the derived adapter must cover once exact source lands. It never invents,
copies, or edits `ui_ux_ir`, never blocks other domain work, and never
admits backend routes while source is missing.

Requirement surfaces recorded by `UIUXLogicSlice@2` (fixed set):

| Surface | Family hint | Description |
| --- | --- | --- |
| `accessibility` | `first_order` | Accessibility property obligations over UI structure and state |
| `authorization` | `authorization` | Authorization and permission constraints over UI actions |
| `interaction_event` | `event_calculus` | Interaction and event-calculus obligations for user/system events |
| `observable_state` | `transition_system` | Observable navigation and runtime state transition obligations |
| `ontology_frame` | `frame_logic` | Ontology/frame (F-logic) component and relation structure |
| `workflow` | `temporal` | Workflow temporal obligations over multi-step UI journeys |

Matrix formal views (declaration-only / source_missing cells; not admitted
slices):

| Formal view id | Family | Profile |
| --- | --- | --- |
| `ui-ux-ir-view/accessibility/v1` | `first_order` | `accessibility_property` |
| `ui-ux-ir-view/event-calculus/v1` | `event_calculus` | `default` |
| `ui-ux-ir-view/navigation-temporal/v1` | `temporal` | `default` |
| `ui-ux-ir-view/navigation-transition/v1` | `transition_system` | `default` |
| `ui-ux-ir-view/ontology/v1` | `frame_logic` | `default` |
| `ui-ux-ir-view/tdfol/v1` | `tdfol` | `default` |

Owner-scoped adapter gap scopes (exactly one content-addressed gap when
source is present; zero when absent):

`accessibility`, `authorization`, `component_frame`, `event`,
`navigation_state`, `permission`, `privacy`, `runtime_journey`,
`tdfol_dcec`, `workflow`.

### Adapter contract declarations under source-missing (UI/UX)

While `ui_ux_ir` is absent, the LPC-041-class fields are declared as follows
without inventing a universal domain IR or minting proof authority:

| Declaration | Current disposition |
| --- | --- |
| Source domain | `ui_ux_ir` on every gate/slice/matrix record |
| View / family / profile / property / notation | Declared via requirement surfaces + matrix formal views; not lowered to admitted `DomainLogicSlice@2` |
| Preserved semantics | Required acceptance of the derived gap includes declared-syntax parsing, `frame_logic` alias canonicalization (`FLogic` / `F-logic` → `frame_logic`), and typed structural round trips |
| Lost semantics | Explicit: package absent ⇒ no admitted semantics; token-presence greps are rejected as acceptance |
| Assumptions | Gate preserves `authority_flags`, `golden_vectors`, `graph_schemas`, `source_maps` for the future adapter; no silent defaults that imply proof |
| Unsupported constructs | Free-form text rejected (`UIUXFreeFormRejectedError`); `token_presence` is an explicit rejected acceptance mode |
| Proof-safety | Matrix authority ceiling is `none` under declaration-only; `UIUXLogicSlice.require_admitted()` always raises until the adapter gap is closed |
| Counterexample-safety | No executable UI/UX requests or counterexample surfaces while not admitted |

Slice status machine:

| Status | Meaning |
| --- | --- |
| `declaration_only` | Source absent; matrix `source_missing` + `declaration_only`; no adapter gaps |
| `adapter_gap` | Exact source present; exactly one content-addressed owner-scoped gap |
| `admitted` | **Forbidden** until the adapter gap is closed (`UIUXSliceAdmissionError`) |

### Frame-logic alias canonicalization (UI/UX)

`frame_logic` dual-read aliases (`FLogic`, `F-logic`, and registry-declared
aliases) canonicalize to family id `frame_logic`. Unknown labels fail closed.
This is the same non-collapse rule legal adapters use: F-logic is not silently
mapped to generic object framing.

### Unsupported / deferred (UI/UX)

- Executable `DomainLogicSlice@2` emission for any UI/UX route while source is
  missing or the adapter gap remains open.
- Free-form / token-presence formalization.
- Creating, copying, or editing `ui_ux_ir` via the gate
  (`UIUXPackageWriteForbiddenError`).
- Blocking other domain work (`blocks_other_work` must remain false).

### Proof-safety and counterexample-safety (UI/UX)

- Declaration-only disposition cannot claim non-empty authority.
- Formalization adapter `formalize()` refuses until exact source import and
  the derived owner-scoped adapter implement the real lowering path.
- No sat/model/trace results or replay digests are minted for UI/UX while
  not admitted — there is no mock proof surface.

---

## Non-collapse rules (intent ↔ UI/UX; no universal domain IR)

| Rule | Enforcement |
| --- | --- |
| Distinct domain ids | `intent_ir` ≠ `ui_ux_ir` on every admitted or declaration-only slice |
| Distinct connector interfaces | `IntentLogicSlice@2` / `UIUXLogicSlice@2` |
| Shared families, domain-local profiles | Catalog families may overlap (e.g. `temporal`, `authorization`, `first_order`); profiles, views, and notations remain domain-scoped |
| No universal domain IR | Free-form / universal routes are deferred or rejected; UI/UX never invents package content to “fill” a universal IR |
| Property ≠ family | Safety, liveness, accessibility property remain roles/properties |
| View role ≠ family | Intent verification_condition stays a view role |
| Intent ≠ UI/UX workflow | Intent workflow temporal control does not emit `ui_ux_ir` slices; UI workflow journeys do not emit `intent_ir` |
| Intent authorization ≠ UI permission generic | Intent uses `tool_permissions` under domain `intent_ir`; UI authorization/permission surfaces stay under `ui_ux_ir` |
| Frame logic stays frame logic | UI ontology_frame / matrix ontology view keep `frame_logic` (aliases only); not object framing |
| TDFOL/DCEC stay legal-adjacent, not intent | UI matrix `tdfol` view and adapter-gap `tdfol_dcec` scope remain UI-owned declarations; intent policy uses `deontic` reification, not TDFOL collapse |
| No new families | Adapters only select existing catalog families (LPC-G040) |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| Intent safety/liveness | Free-standing safety/liveness families |
| Intent verification_condition | A semantic family |
| Intent tool authorization | Security SecPAL domain or UI permission without domain rebinding |
| Intent workflow temporal | UI navigation/workflow journey under `ui_ux_ir` without domain rebinding |
| UI frame_logic | Generic object framing / FOL |
| UI event_calculus / TDFOL cells | Legal IR TDFOL/DCEC claims without legal domain + assumptions |
| Any domain | Free-form text or token presence as typed origin |
| UI source_missing | Admitted `DomainLogicSlice@2` or non-empty proof authority |

## End-to-end admission checklist

### Intent (admitted routes)

For each admitted intent route the connector must:

1. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
2. Cross-check `resolve_intent_route` for the intent view (property/view-role
   invariants).
3. Emit `DomainLogicSlice@2` via `from_typed_expression` with domain
   `intent_ir`.
4. Call `require_admitted()` and `validate_against(document, expression)`.
5. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
6. Attach translation-edge preservation and explicit `loss_ids`.
7. Record hermetic execution/replay without authority upgrade.
8. Cover all seven lineage stages with digest coherence source → request →
   execution → replay.

### UI/UX (source-gated)

For each gate scan the connector must:

1. Observe exact package presence under the logic root (no network, no writes).
2. If absent: emit `UIUXLogicSlice@2` with status `declaration_only`, matrix
   `source_missing` / `declaration_only`, authority ceiling `none`.
3. If present: emit exactly one content-addressed adapter gap covering all
   requirement surfaces and scopes, still without admitting routes.
4. Keep `require_admitted()` fail-closed until the derived adapter closes the
   gap and can lower through `DomainLogicSlice@2` with domain `ui_ux_ir`.
5. Reject free-form / token-presence formalization at every stage.

Incomplete slices fail closed before backend request construction (LPC-044
rejects executable requests without an admitted `DomainLogicSlice@2`).

## File ownership (LPC-043)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | Intent domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py` | Intent formalization routes / admitted views |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | UI/UX domain adapter (`UIUXLogicSlice@2`) + source gate v2 |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_source_gate.py` | UI/UX source gate v1 / formalization adapter declaration |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md` | This conformance note |

Inventory aliases (`intent_ir.domain_slice`, `ui_ux_ir.domain_slice`) refer to
the adapter **role** satisfied by the modules above. Intent production write
path is `formalize/logic_slice_v2.py`. UI/UX production write path is the
source-gated `UIUXLogicSlice@2` connector; the physical package
`logic/ui_ux_ir` remains absent from the pinned revision and must not be
invented by this task.

## Acceptance

- **Intent** keeps its facts, skill effects, prompt candidates, goals, guards,
  workflows, tool authorization, deontic policy, safety, liveness, and
  verification-condition ontology and lowers each admitted route through
  `DomainLogicSlice@2` with domain `intent_ir`.
- **UI/UX** keeps accessibility, interaction/event, workflow, ontology/frame,
  authorization, and observable-state surfaces distinct under domain
  `ui_ux_ir`, records them on `UIUXLogicSlice@2`, and fails closed on admission
  until exact source import closes the adapter gap — without inventing the
  package or a universal domain IR.
- Both domains declare the same LPC-041-class adapter contract fields as
  legal/security (source domain, view, family/profile, property, notation,
  preserved/lost semantics, assumptions, unsupported constructs, proof-safety,
  counterexample-safety).
- No adapter invents a universal domain IR or collapses the other domain’s
  ontology (or legal/security/software/crypto ontologies).
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`
