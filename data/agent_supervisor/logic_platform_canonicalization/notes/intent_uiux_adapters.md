# LPC-043 Intent and UI/UX Domain Adapter Conformance

**Task:** LPC-043 — Intent and UI/UX domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Same adapter contract as legal/security. No universal domain IR.  
**Conflict policy:** Own intent and UI/UX slice adapters only.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`

## Purpose

Intent IR and UI/UX each own a sealed domain ontology. New formalization writes
lower those ontologies **through** `DomainLogicSlice@2` (LPC-040) without
inventing a universal domain IR and without silently remapping either domain's
families, properties, or views into another domain's identity.

This note freezes adapter locations, domain identities, admitted (or
declaration-only) route tables, namespace axes, assumption axes,
preservation/loss declarations, authority ceilings, and non-collapse rules. It
is the durable LPC-043 evidence for the two domain adapters that feed backend
requests via admitted slices only — the same contract shape as LPC-041
(legal) and LPC-042 (security / software / crypto).

Inventory component ids (LPC-004 / syntax formalization inventory):

| Inventory id | Adapter role alias | Status under LPC-043 |
| --- | --- | --- |
| `dls:intent-domain-slice` | `intent_ir.domain_slice` | Satisfied by production `IntentLogicSlice@2` |
| `dls:ui-ux-domain-slice` | `ui_ux_ir.domain_slice` | Satisfied by exact-source-gated `UIUXLogicSlice@2` (declaration-only while package absent) |

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

Construction pattern used by the intent adapter (and required of any future
admitted UI/UX formalization path once exact source lands):

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

Lineage stages required on every admitted end-to-end connection:

```text
typed_origin → semantics → translation → request → result → replay → authority_lineage
```

Hermetic fixtures may supply provider execution and replay without live provers.
Tool absence is an availability result, never a mock proof (LPC-032).

## Production adapter modules

Inventory LPC-004 predicted paths named `domain_slice.py` under
`intent_ir` and `ui_ux_ir`. Live accelerate implementations satisfy the
DomainLogicSlice@2 **adapter role** without those predicted filenames:

| Domain | Domain id | Adapter interface | Production module | Emits / disposition |
| --- | --- | --- | --- | --- |
| Intent | `intent_ir` | `IntentLogicSlice@2` | `ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted intent route |
| UI/UX | `ui_ux_ir` | `UIUXLogicSlice@2` | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | Exact-source-gated slice; **not admitted** while package absent |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Domain | Supporting modules | Role |
| --- | --- | --- |
| Intent | `intent_ir/formalize/typed_compiler.py` | `IntentFormalizationCompiler@2`, `resolve_intent_route`, property/view-role non-collapse |
| Intent | `intent_ir/formalize/compiler.py`, `obligations.py`, `features.py` | Formalization compiler surfaces feeding route metadata |
| UI/UX | `UIUXSourceGate@2`, `UIUXFormalizationAdapter@2` (same gate module) | Exact-source scan, declaration-only formalization contract, frame_logic alias dual-read |
| UI/UX | `conformance/ui_ux_source_gate.py` | Legacy dual-read gate surface aligned with v2 dispositions |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `intent_ir.graphrag.*` (including retrieval) — GraphRAG / SkillCenter surfaces, not DomainLogicSlice generations
- `intent_ir.evaluation.*`, `intent_ir.invocation.*`, `intent_ir.source_adapters.*`
- Inventing, copying, or editing a live `ui_ux_ir` package via the gate (forbidden by `UIUXSourceGate@2`)

Inventory aliases `intent_ir.domain_slice` and `ui_ux_ir.domain_slice` refer to
the adapter **roles** satisfied by the production modules above. Those modules
are the production write path (or declaration-only gate) under LPC-043. Creating
stub `domain_slice.py` files is not required and is not admitted as a path to
satisfy inventory placeholders. Creating a live `ui_ux_ir` package via this
task is forbidden.

### Predicted path → production mapping

| Predicted inventory path | Production authority | Why no stub |
| --- | --- | --- |
| `ipfs_datasets_py/logic/intent_ir/domain_slice.py` | `intent_ir/formalize/logic_slice_v2.py` (`IntentLogicSlice@2`) | Live connector already emits admitted `DomainLogicSlice@2` |
| `ipfs_datasets_py/logic/ui_ux_ir/domain_slice.py` | `conformance/ui_ux_logic_gate_v2.py` (`UIUXLogicSlice@2`) | Package absent; gate is declaration-only and must not invent `ui_ux_ir` |

### Production conformance evidence (existing)

Executable coverage for the production adapters (beyond the board unit-path
command) already lives in:

| Adapter | Conformance surface |
| --- | --- |
| Intent | `ipfs_datasets_py/tests/conformance/logic/test_intent_ir_slice_v2.py` |
| UI/UX v2 gate | `ipfs_datasets_py/tests/conformance/logic/test_ui_ux_logic_gate_v2.py` |
| UI/UX source gate | `ipfs_datasets_py/tests/conformance/logic/test_ui_ux_source_gate.py` |
| Intent formalization routes | `ipfs_datasets_py/tests/unit/logic/intent_ir/formalize/*` |

## Shared adapter contract (intent + UI/UX)

Each route/obligation descriptor declares the LPC-G040 / LPC-041-class fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` (intent) or `domain_id` on `UIUXLogicSlice@2` | Exact domain id; never a universal IR id |
| View | route `view_name` → `view_id(...)` | Typed view namespace; never free-form |
| Family / profile | expression + slice (`family`, `profile`) | From the domain route table only; no new families |
| Property | route `property_name` → `property_id(...)` | Property is never promoted to a family |
| Notation | route `notation_name` → `notation_id(...)` | Surface notation for the admitted view |
| Preserved semantics | translation edge `preservation` | From reviewed translation catalog edge |
| Lost semantics | `_loss_ids_for(route)` / explicit deferred sets | Explicit loss ids; never silent |
| Assumptions | domain-specific assumption axes | Declared even when empty / N/A |
| Unsupported constructs | deferred kind sets | Rejected fail-closed (not admitted) |
| Proof-safety | `authority_ceiling` + `result_authority` | Ceiling never upgrades along lineage |
| Counterexample-safety | sat/model/trace result kinds + replay digests | Counterexamples remain bound to exact request digests |

---

## 1. Intent IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `intent_ir` (`INTENT_IR_DOMAIN_ID`) |
| Interface | `IntentLogicSlice@2` |
| Schema | `intent-logic-slice/v2` |
| Version | `2.0.0` |
| Connector | `IntentLogicSlice` in `intent_ir/formalize/logic_slice_v2.py` |
| Formalization alignment | `IntentFormalizationCompiler@2` / `resolve_intent_route` in `typed_compiler.py` |
| Obligation lineage schema | `intent-obligation-lineage/v2` |
| Inventory alias | `intent_ir.domain_slice` / `dls:intent-domain-slice` |

### Ontology kept distinct

Intent routes use intent-scoped views, profiles, and assumption axes. They may
select catalog families (`first_order`, `program`, `temporal`,
`intention_agency`, `authorization`, `deontic`) but never collapse into
`legal_ir`, `security_ir`, `software_verification`, `crypto_ir`, or
`ui_ux_ir` domain ids.

Evidence subset named by the slice (must appear in supported kinds):

`intent`, `skill`, `prompt`, `goal`, `guard`, `workflow`, `authorization`, `policy`

### Admitted route table (`default_obligation_routes`)

| Route kind | Family | Profile | Property | View | Notation | Authority ceiling | Result authority | Namespace | Route id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `intent` | `first_order` | `default` | `validity` | `facts` | `intent_facts` | satisfiability | satisfiability | family | `intent-route/facts/v1` |
| `skill` | `program` | `dynamic_hoare` | `partial_correctness` | `action_hoare` | `hoare_action_contract` | candidate | candidate | profile | `intent-route/action-hoare/v1` |
| `prompt` | `first_order` | `default` | `validity` | `facts` | `prompt_candidate` | candidate | candidate | family | `intent-route/facts/v1` |
| `goal` | `intention_agency` | `skill_goals` | `goal_satisfaction` | `skill_goals` | `skill_goal` | candidate | candidate | profile | `intent-route/skill-goals/v1` |
| `guard` | `first_order` | `guards_effects` | `guard` | `guards_effects` | `guard_predicate` | satisfiability | satisfiability | profile | `intent-route/guards-effects/v1` |
| `workflow` | `temporal` | `workflow_temporal` | `ordering` | `workflows` | `workflow_temporal` | bounded | model_check | profile | `intent-route/workflow-temporal/v1` |
| `authorization` | `authorization` | `tool_permissions` | `authorization` | `tool_permissions` | `tool_permission` | authorization | authorization | profile | `intent-route/tool-permissions/v1` |
| `policy` | `deontic` | `default` | `obligation` | `norms` | `deontic_norm` | candidate | candidate | family | `intent-route/norms/v1` |
| `safety` | `temporal` | `safety` | `safety` | `safety` | `safety_invariant` | bounded | model_check | **property** | `intent-route/safety/v1` |
| `liveness` | `temporal` | `liveness` | `liveness` | `liveness` | `liveness_progress` | finite_trace | monitor | **property** | `intent-route/liveness/v1` |
| `verification_condition` | `program` (expression only) | `dynamic_hoare` | `validity` | `verification_condition` | `vc_surface` | candidate | candidate | **view_role** | `intent-route/verification-condition-role/v1` |

Property kinds (`safety`, `liveness`) and the view role
(`verification_condition`) must never be admitted as semantic families
(`PROPERTY_KIND_ROUTE_KINDS`, `VIEW_ROLE_ROUTE_KINDS`,
`NEVER_FAMILY_PROPERTY_KINDS`, `NEVER_FAMILY_OPERATION_ROLES`).

Legacy aliases that preserve namespace discipline:

| Alias label | Resolves to | Namespace preserved |
| --- | --- | --- |
| `vc` | `verification_condition` | view role (never family) |
| safety / liveness legacy labels | same property kinds under `temporal` | property (never family) |

### Encoding / evidence / provider bindings (intent)

| Route kind | Encoding | Evidence | Provider |
| --- | --- | --- | --- |
| `intent` | `smtlib2` | `model` | `z3` |
| `skill` | `smtlib2` | `model` | `z3` |
| `prompt` | `smtlib2` | `candidate` | `z3` |
| `goal` | `smtlib2` | `candidate` | `z3` |
| `guard` | `smtlib2` | `model` | `z3` |
| `workflow` | `tla_plus` | `bounded` | `tla_tlc` |
| `authorization` | `datalog` | `authorization` | `datalog_secpal` |
| `policy` | `smtlib2` | `candidate` | `z3` |
| `safety` | `tla_plus` | `bounded` | `tla_tlc` |
| `liveness` | `runtime_mtl` | `trace` | `runtime_mtl` |
| `verification_condition` | `smtlib2` | `candidate` | `z3` |

### Translation edges (preserved semantics)

| Route kind | Translation edge | Translation family | Compiler id |
| --- | --- | --- | --- |
| `intent` | `vc_to_smt` | `program` | `intent.facts.smtlib2` |
| `skill` | `program_to_smt` | `program` | `intent.skill.program_smt` |
| `prompt` | `vc_to_smt` | `program` | `intent.prompt.candidate` |
| `goal` | `intention_to_fol_reified` | `policy_modal` | `intent.goal.intention_fol` |
| `guard` | `vc_to_smt` | `program` | `intent.guard.smtlib2` |
| `workflow` | `temporal_ltl_to_tla_plus` | `state_temporal` | `intent.workflow.tla_plus` |
| `authorization` | `authorization_to_secpal` | `policy_modal` | `intent.authorization.secpal` |
| `policy` | `deontic_to_fol_reified` | `policy_modal` | `intent.policy.deontic_fol` |
| `safety` | `temporal_ltl_to_tla_plus` | `state_temporal` | `intent.safety.tla_plus` |
| `liveness` | `temporal_mtl_to_runtime_mtl` | `state_temporal` | `intent.liveness.runtime_mtl` |
| `verification_condition` | `vc_to_smt` | `program` | `intent.vc.smtlib2` |

Preservation labels come from the reviewed translation-catalog edge for each
route (`TranslationEdgeLineage.preservation`). They are never invented on the
slice connector.

### Preserved / lost semantics (intent)

Explicit losses from `IntentLogicSlice._loss_ids_for(route)`:

| Route | Explicit losses (`loss_ids`) |
| --- | --- |
| `intent` | (none — full first-order fact discharge under stated assumptions) |
| `skill` | `loss.frame_approximation` |
| `prompt` | `loss.prompt_candidate_only` |
| `goal` | `loss.intention_reification` |
| `guard` | (none under stated polarity/signature assumptions) |
| `workflow` | `loss.bounded_trace` |
| `authorization` | (none under grounded permission assumptions) |
| `policy` | `loss.deontic_reification` |
| `safety` | `loss.bounded_trace` |
| `liveness` | `loss.finite_trace`, `loss.fairness_restriction` |
| `verification_condition` | `loss.vc_view_role` |

### Assumption axes (every admitted intent route)

| Axis | Rule |
| --- | --- |
| `source_grounding` | Explicit; source-span lineage required |
| `tool_authority` | Grounded permissions only; confidence never grants authority |
| `bound` | Declared even when empty; workflow/safety/liveness require trace bounds |
| `policy_authority` | Declared for policy/authorization routes |
| `advisor_scope` | Must include confidence-not-correctness / candidate-only assumptions |

Prompt-derived and advisor candidates stay at candidate authority until
deterministic parse, typecheck, and verification receipts exist. Advisor
confidence cannot establish intent correctness
(`AdvisorConfidenceAsCorrectnessError`, tool-authority-from-confidence rejection).

### Unsupported / deferred constructs (intent)

Rejected for executable `IntentLogicSlice@2` / admitted `DomainLogicSlice@2`
family routing (`DEFERRED_ROUTE_KINDS` and future-unsupported family claims):

| Construct | Disposition |
| --- | --- |
| `bdi_overlay`, `agency_overlay`, `normative_overlay` | Deferred overlays (LFP2-044 after LFP2-037 / LFP2-040) |
| `argumentation`, `description_logic` | Declaration-only / deferred |
| `graph_projection`, `proof_translation`, `structural_round_trip` | Operation / view roles — never families |
| `free_form`, `boolean_receipt` | Rejected fail-closed |
| Probabilistic / fuzzy / finite-field / ZKP / defeasible / nonmonotonic as implied intent families | Future-unsupported claims |

### Proof-safety and counterexample-safety (intent)

- Authority ceilings are route-local; lineage records never upgrade without
  independent kernel/backend receipts.
- Prompt/NL/advisor sources force candidate ceilings; they are never theorem
  authority (LPC-032).
- Safety/liveness remain property kinds under `temporal`; VC remains a view
  role under program expression identity — never family promotions.
  Formalization dual-read for VC keeps `family_id` empty on the route label
  itself; the connector uses `program` only for typed-expression identity.
- Counterexamples (models, bounded traces, authorization denials) remain bound
  to exact request digests: source, expression, domain-slice content, and
  obligation/request ids. Replay rebinds the same digests.

### Intent end-to-end admission checklist

For each admitted intent route the connector must:

1. Resolve the route via the sealed table and cross-check
   `resolve_intent_route` admission.
2. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
3. Emit `DomainLogicSlice@2` via `from_typed_expression` with domain
   `intent_ir`.
4. Call `require_admitted()` and `validate_against(document, expression)`.
5. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
6. Attach translation-edge preservation and explicit `loss_ids`.
7. Record hermetic execution/replay without authority upgrade.
8. Cover all seven lineage stages with digest coherence source → request →
   execution → replay, including the five assumption axes.

---

## 2. UI/UX adapter (exact-source-gated)

### Identity

| Field | Value |
| --- | --- |
| Domain id | `ui_ux_ir` (`UIUX_DOMAIN_ID`) |
| Package name | `ui_ux_ir` |
| Package (package-relative) | `ipfs_datasets_py/logic/ui_ux_ir` |
| Package (superproject-relative) | `ipfs_datasets_py/ipfs_datasets_py/logic/ui_ux_ir` |
| Interfaces | `UIUXLogicSlice@2`, `UIUXSourceGate@2`, `UIUXFormalizationAdapter@2` |
| Schemas | `ui-ux-logic-slice/v2`, `ui-ux-source-gate/v2`, `ui-ux-formalization-adapter/v2` |
| Module | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` |
| Owner | `domain:ui_ux_ir` |
| Task / goal lineage | LFP2-026 / LFP2-G050 |
| Inventory alias | `ui_ux_ir.domain_slice` / `dls:ui-ux-domain-slice` |

### Ontology kept distinct

UI/UX keeps accessibility, interaction/event, workflow, ontology/frame,
authorization, and observable-state surfaces distinct from intent skill/prompt
routes and from legal/security/software/crypto domains. Shared catalog families
may be **hints** only until exact source import closes the adapter gap.

### Exact-source gate (fail-closed, non-blocking)

| Source observation | Slice status | Matrix disposition | Backend admission |
| --- | --- | --- | --- |
| Package absent from pinned revision | `declaration_only` | `source_missing` / `declaration_only` | **Not admitted** (`require_admitted` raises) |
| Package present at reviewed path | `adapter_gap` | present disposition + exactly one owner-scoped adapter gap | **Not admitted** until gap closed |
| Attempted invent/copy/edit of `ui_ux_ir` via gate | forbidden | n/a | `UIUXPackageWriteForbiddenError` |
| Free-form / token-presence “formalization” | rejected | n/a | `UIUXFreeFormRejectedError` |

`UIUXLogicSlice@2` must never set `blocks_other_work=true`. Absent UI/UX source
does not block other domain work. Declaration-only dispositions use
`AuthorityCeiling.NONE` / `UNKNOWN` only — never a non-empty authority claim.

Package presence is a pure filesystem scan for package markers
(`__init__.py`, `py.typed`, or `README.md`) under the reviewed relative paths.
No network, install, model download, or subprocess is allowed at import or
scan time.

### Requirement surfaces (fixed set)

| Surface id | Family hint | Description |
| --- | --- | --- |
| `accessibility` | `first_order` | Accessibility property obligations over UI structure and state |
| `authorization` | `authorization` | Authorization and permission constraints over UI actions |
| `interaction_event` | `event_calculus` | Interaction / event-calculus obligations for user/system events |
| `observable_state` | `transition_system` | Observable navigation and runtime state transitions |
| `ontology_frame` | `frame_logic` | Ontology/frame (F-logic) component and relation structure |
| `workflow` | `temporal` | Workflow temporal obligations over multi-step UI journeys |

Family hints are **not** admitted family bindings. They record the expected
catalog family once the owner-scoped adapter lands; they do not emit
`DomainLogicSlice@2` rows while status is `declaration_only` or `adapter_gap`.

### Adapter-gap acceptance (when source present)

Owner-scoped adapter scopes include: accessibility, authorization,
component_frame, event, navigation_state, permission, privacy, runtime_journey,
tdfol_dcec, workflow.

Required acceptance of the derived adapter gap:

- `declared_syntax_parsing`
- `frame_logic_alias_canonicalization` (dual-read `FLogic` / `F-logic` → `frame_logic`)
- `typed_structural_round_trips`

Rejected acceptance: `token_presence` greps alone.

`UIUXFormalizationAdapter@2` is a **declaration-only** interface until exact
source import and the owner-scoped adapter land. It refuses free-form payloads
and refuses formalization while source is missing.

### Preserved / lost semantics (UI/UX)

While the package is absent or the adapter gap is open:

| Status | Preserved | Lost / deferred |
| --- | --- | --- |
| `declaration_only` | Fixed requirement-surface inventory and non-blocking matrix disposition | All executable formalization; no admitted slice, no backend request |
| `adapter_gap` | Source fingerprint + owner-scoped gap acceptance criteria | Admitted lowering until declared-syntax parse, frame_logic alias canon, and typed round trips land |

Future admitted UI/UX routes must declare route-local `loss_ids` on the same
contract fields as intent/legal/security; none are admitted in this revision.

### Proof-safety and counterexample-safety (UI/UX)

- No backend route may claim admitted `DomainLogicSlice@2` status while the
  slice is `declaration_only` or `adapter_gap`.
- Authority ceilings for future admitted UI/UX routes must remain route-local
  and must not upgrade from declaration-only matrix cells.
- Counterexamples (when admitted) must bind exact request digests; declaration-
  only cells produce no executable counterexample claims.

### UI/UX admission checklist

1. Scan with `UIUXSourceGate@2` (filesystem presence only; no network/install).
2. Record the fixed requirement surfaces on `UIUXLogicSlice@2`.
3. If source absent → declaration-only slice; do not invent `ui_ux_ir`.
4. If source present → emit exactly one content-addressed adapter gap; still
   refuse `status=admitted` until the gap is closed.
5. Only after the owner-scoped adapter implements declared-syntax parsing,
   frame_logic alias canonicalization, and typed structural round trips may
   routes emit admitted `DomainLogicSlice@2` with domain `ui_ux_ir` via the
   shared LPC-040 construction pattern.

---

## Non-collapse rules (intent ↔ UI/UX ↔ universal IR)

| Rule | Enforcement |
| --- | --- |
| Distinct domain ids | `intent_ir` ≠ `ui_ux_ir` on every slice / gate record |
| Distinct connector interfaces | `IntentLogicSlice@2` / `UIUXLogicSlice@2` |
| No universal domain IR | Free-form / universal routes deferred or rejected; domain id is never a generic bag |
| Intent ≠ UI/UX | Intent skill/prompt/goal routes do not emit `ui_ux_ir` slices; UI/UX surfaces do not emit `intent_ir` |
| Property ≠ family | Safety, liveness, invariants stay properties |
| View role ≠ family | Verification condition, graph projection, proof translation, structural round-trip stay roles |
| No new families | Adapters only select existing catalog families (LPC-G040) |
| UI/UX absence non-blocking | Missing package is declaration-only, not a global work blocker |
| No invented UI/UX package | Gate must never create/copy/edit `ui_ux_ir` |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| Intent facts / prompts | Universal free-form domain IR or theorem authority from confidence |
| Intent safety / liveness | Semantic families named `safety` / `liveness` |
| Intent verification condition | Family id `verification_condition` |
| Intent tool authorization | Authority granted by advisor/prompt confidence alone |
| Intent workflow temporal | Unbounded model claims without trace bounds |
| UI/UX declaration-only cells | Admitted backend `DomainLogicSlice@2` without source + adapter gap closure |
| UI/UX frame_logic surfaces | Free-form token presence or uncanonicalized `FLogic` family labels |
| Either domain | Legal / security / software / crypto domain ids without explicit rebinding |

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

Incomplete slices fail closed before backend request construction (LPC-044
rejects executable requests without an admitted `DomainLogicSlice@2`).
UI/UX declaration-only / adapter-gap slices correctly fail that gate until the
owner-scoped adapter lands.

## File ownership (LPC-043)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py` | Intent domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py` | Intent route catalog and non-collapse routing |
| `ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` | UI/UX exact-source gate + `UIUXLogicSlice@2` declaration path |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md` | This conformance note (LPC-043 declared output) |

Inventory aliases (`intent_ir.domain_slice`, `ui_ux_ir.domain_slice`) refer to
the adapter roles satisfied by the modules above. Predicted
`.../domain_slice.py` paths are inventory placeholders, not mandatory new files
for LPC-043 admission. Production policy is preserved: document the live
adapters; do not invent universal domain IR or stub packages to satisfy path
strings.

## Acceptance

- **Intent** keeps its skill, prompt, goal, guard, workflow, authorization,
  policy, safety, liveness, and verification-condition ontology and lowers each
  admitted route through `DomainLogicSlice@2` with domain `intent_ir`.
- **UI/UX** keeps accessibility, interaction/event, workflow, ontology/frame,
  authorization, and observable-state surfaces distinct; remains
  declaration-only / adapter-gap until exact source import; never invents a
  universal domain IR or blocks other work when source is missing.
- Same adapter contract fields as legal/security (source domain, view,
  family/profile, property, notation, preserved/lost semantics, assumptions,
  unsupported constructs, proof-safety, counterexample-safety).
- No adapter invents a universal domain IR or collapses the other domain's
  ontology.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`
