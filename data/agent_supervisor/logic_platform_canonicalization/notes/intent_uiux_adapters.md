# LPC-043 Intent and UI/UX Domain Adapter Conformance

**Task:** LPC-043 — Intent and UI/UX domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Same adapter contract as legal/security. No universal domain IR.  
**Conflict policy:** Own intent and UI/UX slice adapters only. Never invent a universal domain IR or collapse intent ↔ ui_ux ↔ legal/security/software/crypto ontologies.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`

## Purpose

Intent IR and UI/UX IR each own a sealed domain ontology. New formalization
writes lower those ontologies **through** `DomainLogicSlice@2` (LPC-040)
without inventing a universal domain IR and without silently remapping one
domain’s families, views, or properties into another.

This note freezes adapter locations, domain identities, admitted route tables
(or declaration-only gate status for UI/UX), namespace axes, assumption axes,
preservation/loss declarations, authority ceilings, and non-collapse rules.
It is the durable LPC-043 evidence for the two domain adapters that feed
backend requests via admitted slices only.

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

Construction pattern used by the intent production adapter (and required of
any future admitted UI/UX adapter):

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
implementations live as `*LogicSlice@2` connectors that **emit** (or, for
UI/UX while source is missing, gate) `DomainLogicSlice@2` admission.

| Domain | Domain id | Adapter interface | Production module | Emits / status |
| --- | --- | --- | --- | --- |
| Intent | `intent_ir` | `IntentLogicSlice@2` | `ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted intent route |
| UI/UX | `ui_ux_ir` | `UIUXLogicSlice@2` | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | Declaration-only / adapter-gap until exact source import; **never** admits while `ui_ux_ir` package is absent |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Domain | Supporting modules | Role |
| --- | --- | --- |
| Intent | `intent_ir/formalize/typed_compiler.py` | `IntentFormalizationCompiler@2`, `resolve_intent_route`, never-family property/view rules |
| Intent | `intent_ir/formalize/compiler.py`, `advisor.py`, `obligations.py` | Formalization artifact / advisor / obligation surfaces |
| Intent | `intent_ir/source_adapters/*`, `intent_ir/invocation/*` | Source projection into Intent IR (not DomainLogicSlice generation) |
| UI/UX | `conformance/ui_ux_source_gate.py` | Legacy source-presence gate (v1) |
| UI/UX | `conformance/ui_ux_logic_gate_v2.py` | `UIUXSourceGate@2`, `UIUXFormalizationAdapter@2`, requirement surfaces, adapter gap |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `intent_ir.graphrag.*` retrieval / corpus projectors
- `intent_ir.evaluation.*` splits and benchmarks
- UI/UX matrix cells while `source_missing` (declaration-only; no package invent)
- Any free-form / token-presence acceptance of UI structure

Inventory aliases `intent_ir.domain_slice` and `ui_ux_ir.domain_slice` refer to
the adapter roles satisfied by the modules above; those modules are the
production write path (or fail-closed gate) for domain lowering under LPC-043.

## Shared adapter contract (same as legal/security)

Each route/obligation descriptor declares the LPC-G040 / LPC-041-class fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` | Exact domain id; must match parent formalization artifact |
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

---

## 1. Intent IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `intent_ir` |
| Interface | `IntentLogicSlice@2` |
| Schema | `intent-logic-slice/v2` |
| Version | `2.0.0` |
| Connector | `IntentLogicSlice` in `intent_ir/formalize/logic_slice_v2.py` |
| Formalization alignment | `IntentFormalizationCompiler@2` / `resolve_intent_route` in `typed_compiler.py` |

### Ontology kept distinct

Intent keeps typed facts, skill effects, prompt candidates, goals, guards,
workflows, tool authorization, deontic/modal policy, safety, liveness, and
verification-condition **view roles** as distinct route kinds. Catalog families
may be shared (`first_order`, `program`, `temporal`, `authorization`,
`deontic`, `intention_agency`), but domain id, profiles, views, and assumption
axes stay intent-scoped.

Safety and liveness are **property kinds** under `temporal`, never families.
Verification conditions are a **view role**, never a family.

Base admitted routes (Wave-2 evidence subset: intent, skill, prompt, goal,
guard, workflow, authorization, policy — plus property/view-role routes):

| Route kind | Family | Profile | Property | View | Notation | Authority ceiling | Result authority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `intent` | `first_order` | `default` | `validity` | `facts` | `intent_facts` | satisfiability | satisfiability |
| `skill` | `program` | `dynamic_hoare` | `partial_correctness` | `action_hoare` | `hoare_action_contract` | candidate | candidate |
| `prompt` | `first_order` | `default` | `validity` | `facts` | `prompt_candidate` | candidate | candidate |
| `goal` | `intention_agency` | `skill_goals` | `goal_satisfaction` | `skill_goals` | `skill_goal` | candidate | candidate |
| `guard` | `first_order` | `guards_effects` | `guard` | `guards_effects` | `guard_predicate` | satisfiability | satisfiability |
| `workflow` | `temporal` | `workflow_temporal` | `ordering` | `workflows` | `workflow_temporal` | bounded | model_check |
| `authorization` | `authorization` | `tool_permissions` | `authorization` | `tool_permissions` | `tool_permission` | authorization | authorization |
| `policy` | `deontic` | `default` | `obligation` | `norms` | `deontic_norm` | candidate | candidate |
| `safety` | `temporal` | `safety` | `safety` | `safety` | `safety_invariant` | bounded | model_check |
| `liveness` | `temporal` | `liveness` | `liveness` | `liveness` | `liveness_progress` | finite_trace | monitor |
| `verification_condition` | `program` (expression only) | `dynamic_hoare` | `validity` | `verification_condition` | `vc_surface` | candidate | candidate |

Route ids (formalization compiler alignment):

| Route kind | Intent route id |
| --- | --- |
| `intent` / `prompt` | `intent-route/facts/v1` |
| `skill` | `intent-route/action-hoare/v1` |
| `goal` | `intent-route/skill-goals/v1` |
| `guard` | `intent-route/guards-effects/v1` |
| `workflow` | `intent-route/workflow-temporal/v1` |
| `authorization` | `intent-route/tool-permissions/v1` |
| `policy` | `intent-route/norms/v1` |
| `safety` | `intent-route/safety/v1` |
| `liveness` | `intent-route/liveness/v1` |
| `verification_condition` | `intent-route/verification-condition-role/v1` |

Namespace discipline notes:

- `safety` / `liveness` use `route_namespace=property`; they must never become
  family ids (`NEVER_FAMILY_PROPERTY_KINDS`).
- `verification_condition` uses `route_namespace=view_role`; formalization
  route must not set `family_id` for the VC role itself.
- Prompt-derived and advisor-scored formulas remain **candidates** until
  deterministic parse, typecheck, and verification receipts exist.
- Tool authority never follows confidence alone.

### Assumption axes (intent)

Every admitted intent route declares all five axes (empty only when N/A; axis
still present):

| Axis | Role |
| --- | --- |
| `source_grounding` | Source span, entity/predicate/action/goal identity, polarity |
| `tool_authority` | Grounded permissions, delegation scope, action identity (authorization) |
| `bound` | Quantifier instantiations, program steps, trace length, VC depth |
| `policy_authority` | Policy authority bound, world policy, effect polarity |
| `advisor_scope` | Advisor/prompt confidence is candidate-only and not correctness |

Common advisor-scope ids on every route:

- `assumption:advisor_candidate_only`
- `assumption:advisor_confidence_not_correctness`

Route-local examples:

| Route | Additional assumptions |
| --- | --- |
| `intent` | `assumption:entity_identity`, `assumption:predicate_signature`, `bound:quantifier_instantiations` |
| `skill` | `assumption:action_identity`, `assumption:frame_conditions`, `bound:program_steps` |
| `prompt` | `assumption:prompt_derived`, `assumption:parse_typecheck_required`, `assumption:prompt_confidence_not_correctness` |
| `goal` | `assumption:goal_identity`, `assumption:agent_identity`, `assumption:intention_force` |
| `guard` | `assumption:guard_polarity`, `assumption:effect_polarity` |
| `workflow` | `assumption:edge_direction`, `assumption:temporal_operator`, `bound:trace_length`, `assumption:fairness_constraint` |
| `authorization` | `assumption:grounded_permission_required`, `assumption:tool_authority_not_from_confidence`, `assumption:policy_authority_bound` |
| `policy` | `assumption:operator_force`, `assumption:norm_polarity`, `assumption:world_policy` |
| `safety` | `assumption:invariant_polarity`, `assumption:bad_state_exclusion`, `bound:trace_length` |
| `liveness` | `assumption:progress_condition`, `assumption:fairness_constraint`, `assumption:finite_trace` |
| `verification_condition` | `assumption:obligation_identity`, `assumption:assumption_set`, `bound:vc_depth` |

### Preserved / lost semantics (intent)

Preserved semantics come from reviewed translation edges
(`program`, `state_temporal`, `policy_modal` families). Explicit losses attached
at compile time:

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
| `intent` / `guard` / `authorization` | (none beyond translation-edge preservation) |

### Unsupported / deferred (intent)

Rejected for executable `IntentLogicSlice@2` (`DEFERRED_ROUTE_KINDS`):

| Construct | Disposition |
| --- | --- |
| `bdi_overlay`, `agency_overlay`, `normative_overlay` | Deferred overlays (LFP2-044 after LFP2-037 / LFP2-040) |
| `argumentation`, `description_logic` | Declaration-only / deferred |
| `free_form`, `boolean_receipt` | Rejected as typed origin / proof |
| `graph_projection`, `proof_translation`, `structural_round_trip` | Operation / view roles — never families |
| Property labels as families | `safety`, `liveness`, `invariant`, `validity`, … rejected as family ids |
| Operation labels as families | `verification_condition`, `graph_projection`, `prover_router`, … rejected |
| Future unsupported family claims | `probabilistic`, `fuzzy*`, `finite_field*`, `zk*`, `situation_calculus`, … |

Connector wire form: `weakens_to_free_form=false`,
`advisor_confidence_establishes_correctness=false`.

### Proof-safety and counterexample-safety (intent)

#### Proof-safety

- Authority ceilings are route-local (`SATISFIABILITY`, `CANDIDATE`, `BOUNDED`,
  `AUTHORIZATION`, `FINITE_TRACE`); lineage records `never_upgrades=true`.
- Prompt and advisor confidence never mint proof or tool authority.
- Skill/goal/policy/VC routes remain **candidate** until stronger independent
  receipts exist; routes alone never mint theorem authority.
- Safety uses bounded model-check authority; liveness uses finite-trace /
  monitor authority, not unbounded claims.
- Authorization uses authorization authority only; confidence cannot grant
  permission.

#### Counterexample-safety

- Sat/model/trace/monitor counterexamples bind to exact request digests
  (`source_digest`, `expression_digest`, domain-slice content digest,
  obligation/request digests).
- Bounded workflow/safety traces do not authorize unbounded counterexamples.
- Finite-trace liveness witnesses stay under declared fairness and length.
- Hermetic fixture execution/replay re-bind the same digests; unbound models
  are rejected.
- Advisor “high confidence” is never a counterexample or a proof.

### Intent end-to-end admission checklist

For each admitted intent route the connector must:

1. Resolve route kind against `default_obligation_routes()` and
   `resolve_intent_route` (namespace invariants).
2. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
3. Emit `DomainLogicSlice@2` via `from_typed_expression` with domain
   `intent_ir`.
4. Call `require_admitted()` and `validate_against(document, expression)`.
5. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
6. Attach translation-edge preservation and explicit `loss_ids`.
7. Record hermetic execution/replay without authority upgrade.
8. Cover all seven lineage stages with digest coherence source → request →
   execution → replay.

Helpers: `connect_intent_route`, `connect_all_intent_routes`,
`validate_intent_logic_slice`.

---

## 2. UI/UX IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `ui_ux_ir` |
| Interface | `UIUXLogicSlice@2` |
| Schema | `ui-ux-logic-slice/v2` |
| Version | `2.0.0` |
| Source gate | `UIUXSourceGate@2` (`ui-ux-source-gate/v2`) |
| Formalization adapter (declaration) | `UIUXFormalizationAdapter@2` |
| Production module | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` |
| Package path (exact source) | `ipfs_datasets_py/logic/ui_ux_ir` (superproject: `ipfs_datasets_py/ipfs_datasets_py/logic/ui_ux_ir`) |
| Owner id | `domain:ui_ux_ir` |

### Ontology kept distinct (declaration-only until source import)

The pinned datasets revision currently **lacks** the `ui_ux_ir` package.
`UIUXSourceGate@2` records a typed `source_missing` /
`declaration_only` disposition and **never** creates, copies, or edits
`ui_ux_ir`. Missing UI source does not block other domain work.

`UIUXLogicSlice@2` still records the owner-scoped requirement surfaces the
derived adapter must cover once exact source lands. Those surfaces keep the
UI/UX ontology distinct from intent, legal, security, software, and crypto:

| Requirement surface | Family hint | Meaning |
| --- | --- | --- |
| `accessibility` | `first_order` | Accessibility property obligations over UI structure and state |
| `authorization` | `authorization` | Authorization / permission constraints over UI actions |
| `interaction_event` | `event_calculus` | Interaction and event-calculus obligations for user/system events |
| `observable_state` | `transition_system` | Observable navigation and runtime state transitions |
| `ontology_frame` | `frame_logic` | Ontology/frame (F-logic) component and relation structure |
| `workflow` | `temporal` | Workflow temporal obligations over multi-step UI journeys |

Declared formal views (capability matrix; source not in pinned revision):

| Formal view id | Family | Profile |
| --- | --- | --- |
| `ui-ux-ir-view/accessibility/v1` | `first_order` | `accessibility_property` |
| `ui-ux-ir-view/event-calculus/v1` | `event_calculus` | `default` |
| `ui-ux-ir-view/navigation-temporal/v1` | `temporal` | `default` |
| `ui-ux-ir-view/navigation-transition/v1` | `transition_system` | `default` |
| `ui-ux-ir-view/ontology/v1` | `frame_logic` | `default` |
| `ui-ux-ir-view/tdfol/v1` | `tdfol` | `default` |

Owner-scoped adapter gap scopes (exactly one gap when source is present):

`accessibility`, `authorization`, `component_frame`, `event`,
`navigation_state`, `permission`, `privacy`, `runtime_journey`, `tdfol_dcec`,
`workflow`.

### Current gate dispositions

| Source presence | Gate disposition | Slice status | Matrix support / availability | Authority ceiling | Admits `DomainLogicSlice@2`? |
| --- | --- | --- | --- | --- | --- |
| Absent (current pinned tree) | `declaration_only` | `declaration_only` | `declaration_only` / `source_missing` | `none` | **No** — `require_admitted()` fails closed |
| Present (exact reviewed path) | `emit_adapter_gap` | `adapter_gap` | `declaration_only` / `declared` | `none` until gap closed | **No** until adapter gap closed |
| Gap closed (future) | (derived adapter) | `admitted` | upgraded only after reviewed adapter | route-local | Yes — same LPC-040 construction as legal/security/intent |

`SliceStatus.ADMITTED` is rejected on the gate slice until the owner-scoped
adapter gap is closed. Formalization while source is missing raises
`UIUXSourceMissingError`. Free-form / token-presence payloads raise
`UIUXFreeFormRejectedError`.

### Assumption axes (UI/UX — required when admitted)

When a future admitted UI/UX `DomainLogicSlice@2` is emitted, every route must
declare the shared LPC-G040 fields plus UI-scoped axes aligned to requirement
surfaces (empty only when N/A, still explicit):

| Axis | Role |
| --- | --- |
| `source_grounding` | Exact UI source identity / fingerprint, component identity, source maps |
| `accessibility` | Accessibility property polarity, structure/state binding |
| `interaction_event` | Event identity, user/system polarity, event horizon |
| `navigation_state` | Observable state schema, transition direction |
| `workflow_bound` | Journey / trace bounds and fairness for multi-step UI workflows |
| `authorization` / `permission` | Principal, action, resource, policy authority |
| `frame_ontology` | Typed frame roles; `frame_logic` dual-read aliases `FLogic` / `F-logic` → `frame_logic` |
| `privacy` | Observation / information-flow assumptions when privacy scope applies |

### Preserved / lost semantics (UI/UX)

Until the package is imported and the gap is closed, no executable losses are
claimed as proofs. The **adapter gap acceptance contract** requires:

| Required acceptance | Meaning |
| --- | --- |
| `declared_syntax_parsing` | Declared-syntax parse, not free text |
| `frame_logic_alias_canonicalization` | Dual-read F-logic aliases canonicalize to `frame_logic` |
| `typed_structural_round_trips` | Typed structural round trips |

Explicitly **rejected** acceptance:

| Rejected acceptance | Meaning |
| --- | --- |
| `token_presence` | Grep/token presence never establishes adapter conformance |

Preserve-on-import fields: `authority_flags`, `golden_vectors`, `graph_schemas`,
`source_maps`.

When routes later admit through `DomainLogicSlice@2`, losses must be explicit
(examples, not silent remaps):

| Surface | Expected loss class (when discharged) |
| --- | --- |
| Workflow / navigation temporal | `loss.bounded_trace` / finite journey window |
| Interaction event | Finite event horizon; not generic FOL facts without EC axioms |
| Ontology frame | Not object framing; not graph-projection-as-family |
| Accessibility FOL | Finite structure/state domain when SAT-discharged |
| Authorization | Policy authority bound; confidence never grants UI permission |
| TDFOL/DCEC scopes | Retain `tdfol` / event composition identity; no silent FOL/deontic collapse |

### Unsupported / deferred (UI/UX)

| Construct | Disposition |
| --- | --- |
| Inventing / copying / editing `ui_ux_ir` via the gate | Forbidden (`UIUXPackageWriteForbiddenError`) |
| Free-form text / token presence as formalization | Rejected |
| Admitting routes while source missing | Fail closed (`UIUXSliceAdmissionError`) |
| Blocking other domain work because UI source is absent | Forbidden (`blocks_other_work=false`) |
| Universal domain IR / free-form family | Rejected |
| Graph projection / proof translation as families | Operation roles only |
| Collapsing UI frame logic into object framing | Forbidden (same rule as legal frame logic) |
| Collapsing UI accessibility into intent facts domain | Forbidden — domain remains `ui_ux_ir` |

### Proof-safety and counterexample-safety (UI/UX)

#### Proof-safety

- Declaration-only / source-missing cells claim **no** authority ceiling
  (`AuthorityCeiling.NONE`).
- Present-source adapter gap still claims no executable authority until the
  gap is closed with declared-syntax parsing and typed round trips.
- Future admitted UI routes use route-local ceilings only; lineage never
  upgrades without independent kernel/backend receipts.
- Gate receipts and slice digests never mint theorem or official authority.

#### Counterexample-safety

- No executable counterexamples are claimed under `source_missing`.
- After admission, UI counterexamples (accessibility violations, navigation
  traces, event timelines, permission denials) must bind to exact request and
  source-fingerprint digests.
- Bounded journey/event windows do not authorize unbounded UI counterexamples.
- Token absence/presence is never a semantic counterexample.

### UI/UX end-to-end admission checklist

While source is missing (current tree):

1. Scan with `UIUXSourceGate@2` / `scan_ui_ux_source_gate_v2`.
2. Project `UIUXLogicSlice@2` via `UIUXLogicSliceConnector` /
   `build_ui_ux_logic_slice_v2`.
3. Record all six requirement surfaces; emit zero adapter gaps.
4. Matrix disposition: `declaration_only` + `source_missing` +
   `blocks_other_work=false` + `writes_ui_ux_ir=false`.
5. Refuse `require_admitted()` and refuse formalization.

When exact source is present:

1. Scan yields exactly one content-addressed owner-scoped `UIUXAdapterGap`.
2. Slice status becomes `adapter_gap` (still not admitted).
3. Derived adapter must implement declared-syntax parsing, frame_logic alias
   canonicalization, and typed structural round trips across all scopes.
4. Only after gap closure may routes emit `DomainLogicSlice@2` with domain
   `ui_ux_ir` using the shared LPC-040 construction and the seven lineage
   stages.

Incomplete or unadmitted slices fail closed before backend request construction
(LPC-044 rejects executable requests without an admitted `DomainLogicSlice@2`).

---

## Non-collapse rules (intent ↔ UI/UX ↔ other domains)

| Rule | Enforcement |
| --- | --- |
| Distinct domain ids | `intent_ir` ≠ `ui_ux_ir` ≠ `legal_ir` ≠ `security_ir` ≠ `software_verification` ≠ `crypto_ir` |
| Distinct connector interfaces | `IntentLogicSlice@2` / `UIUXLogicSlice@2` |
| Shared families, domain-local profiles/views | Catalog families may overlap; profiles, views, notations, and assumption axes stay domain-scoped |
| No universal domain IR | Free-form / universal routes deferred or rejected on both domains |
| Property ≠ family | Safety, liveness, accessibility properties stay properties/roles |
| View role ≠ family | Intent VC and UI operation roles never become families |
| Intent ≠ UI/UX | Intent skill/prompt/goal routes do not emit `ui_ux_ir` slices; UI surfaces do not emit `intent_ir` |
| Intent ≠ legal policy generic | Intent deontic norms use intent policy/tool axes; legal keeps TDFOL/DCEC/frame foundations |
| UI frame ≠ object framing | `frame_logic` retained; dual-read aliases canonicalize; graph projection stays an operation role |
| UI event ≠ generic FOL | Event-calculus surface retains EC identity |
| No new families | Adapters only select existing catalog families (LPC-G040) |
| UI source missing ≠ invent package | Gate never writes `ui_ux_ir` |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| Intent safety/liveness | Semantic families (must remain property kinds under temporal) |
| Intent verification_condition | Family id (must remain view role) |
| Intent prompt confidence | Proof, tool authority, or correctness |
| Intent authorization | Security_ir policy route without domain rebinding |
| Intent workflow temporal | Software-verification monitor IR without domain rebinding |
| UI accessibility | Intent facts domain or free-form FOL dump |
| UI frame_logic / F-logic | Object framing or graph-projection family |
| UI event calculus | Generic FOL facts without EC axioms |
| UI workflow | Unbounded temporal claims without journey bounds |
| Any domain | Free-form text as typed origin / theorem authority |
| Any domain | Universal / cross-domain IR collapsing ontologies |

## End-to-end admission checklist (shared)

For each admitted route/obligation the connector must:

1. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
2. Emit `DomainLogicSlice@2` via `from_typed_expression` with the domain id.
3. Call `require_admitted()` and `validate_against(document, expression)`.
4. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
5. Attach translation-edge preservation and explicit `loss_ids`.
6. Record hermetic execution/replay without authority upgrade.
7. Cover all seven lineage stages with digest coherence source → request →
   execution → replay.

UI/UX additionally requires exact source presence and closed adapter gap before
steps 2–7 apply. Intent already implements the full path for every Wave-2
admitted route.

## File ownership (LPC-043)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | Intent domain adapter → `DomainLogicSlice@2` (`IntentLogicSlice@2`) |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py` | Intent route catalog, never-family rules, formalization compiler |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | UI/UX domain adapter gate → `UIUXLogicSlice@2` / `UIUXSourceGate@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_source_gate.py` | Legacy UI/UX source-presence gate |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `ipfs_datasets_py/tests/unit/logic/intent_ir/` | Intent unit regression surface (validation target) |
| `ipfs_datasets_py/tests/unit/logic/ui_ux_ir/` | UI/UX unit regression surface (validation target; may be empty while package is source_missing) |
| `ipfs_datasets_py/tests/conformance/logic/test_intent_ir_slice_v2.py` | Intent slice end-to-end conformance |
| `ipfs_datasets_py/tests/conformance/logic/test_ui_ux_logic_gate_v2.py` | UI/UX gate / slice v2 conformance |
| `data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md` | This conformance note |

Inventory aliases (`intent_ir.domain_slice`, `ui_ux_ir.domain_slice`) refer to
the adapter roles satisfied by the production modules above; those modules are
the production write path (or fail-closed gate) for admitted domain lowering
under LPC-043.

## Acceptance

- **Intent** keeps its facts, skill, prompt, goal, guard, workflow,
  authorization, policy, safety, liveness, and verification-condition ontology
  and lowers each admitted route through `DomainLogicSlice@2` with domain
  `intent_ir`.
- Every intent adapter declaration states source domain, view, family/profile,
  property, notation, preserved/lost semantics, assumptions, unsupported
  constructs, proof-safety, and counterexample-safety (same contract class as
  legal/security).
- Safety/liveness remain property kinds; VC remains a view role; advisor
  confidence never establishes correctness or tool authority.
- **UI/UX** keeps accessibility, interaction/event, workflow, ontology/frame,
  authorization, and observable-state surfaces distinct under domain
  `ui_ux_ir`. While source is missing, `UIUXLogicSlice@2` is declaration-only
  and never admits executable routes or invents the package.
- No adapter invents a universal domain IR or collapses another domain’s
  ontology.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`
