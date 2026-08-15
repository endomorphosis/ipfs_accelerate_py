# LPC-043 Intent and UI/UX Domain Adapter Conformance

**Task:** LPC-043 — Intent and UI/UX domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Same adapter contract as legal/security. No universal domain IR.  
**Conflict policy:** Own intent and UI/UX slice adapters only.  
**Repair context:** LPC-173 resolves the validation retry-budget blocker filed after
repeated LPC-043 `proposal_gate_failed` attempts that tried to invent predicted
`domain_slice.py` paths instead of documenting the production adapters that
already satisfy the DomainLogicSlice@2 role.  
**Validation (LPC-043):**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`  
**Validation (LPC-173 repair gate):**  
`test -f data/agent_supervisor/logic_platform_canonicalization/state/discovery/2026-08-15-lpc-173-lpc-043-retry-budget.md`

## Purpose

Intent IR and UI/UX each own a sealed domain ontology. New formalization writes
lower those ontologies **through** `DomainLogicSlice@2` (LPC-040) without
inventing a universal domain IR and without silently remapping either domain’s
families, properties, or views into another domain’s identity.

This note freezes adapter locations, domain identities, admitted (or
declaration-only) route tables, namespace axes, assumption axes,
preservation/loss declarations, authority ceilings, and non-collapse rules. It
is the durable LPC-043 evidence for the two domain adapters that feed backend
requests via admitted slices only — the same contract shape as LPC-041
(legal) and LPC-042 (security / software / crypto).

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

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `intent_ir.graphrag.*` (including retrieval) — GraphRAG / SkillCenter surfaces, not DomainLogicSlice generations
- `intent_ir.evaluation.*`, `intent_ir.invocation.*`, `intent_ir.source_adapters.*`
- Inventing, copying, or editing a live `ui_ux_ir` package via the gate (forbidden by `UIUXSourceGate@2`)

Inventory aliases `intent_ir.domain_slice` and `ui_ux_ir.domain_slice` refer to
the adapter **roles** satisfied by the production modules above. Those modules
are the production write path (or declaration-only gate) under LPC-043. Creating
stub `domain_slice.py` files is not required and was the recurring proposal-gate
failure mode that LPC-173 repairs by documenting the real adapters.

## Shared adapter contract (intent + UI/UX)

Each route/obligation descriptor declares the LPC-G040 / LPC-041-class fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` (intent) or `domain_id` on `UIUXLogicSlice@2` | Exact domain id; never a universal IR id |
| View | route `view_name` → `view_id(...)` | Typed view namespace; never free-form |
| Family / profile | expression + slice (`family`, `profile`) | From the domain route table only; no new families |
| Property | route `property_name` → `property_id(...)` | Property is never promoted to a family |
| Notation | route `notation_name` → `notation_id(...)` | Surface notation for the admitted view |
| Preserved semantics | translation edge / route notes | From reviewed translation catalog edge |
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

### Ontology kept distinct

Intent routes use intent-scoped views, profiles, and assumption axes. They may
select catalog families (`first_order`, `program`, `temporal`,
`intention_agency`, `authorization`, `deontic`) but never collapse into
`legal_ir`, `security_ir`, `software_verification`, `crypto_ir`, or
`ui_ux_ir` domain ids.

Evidence subset named by the slice (must appear in supported kinds):

`intent`, `skill`, `prompt`, `goal`, `guard`, `workflow`, `authorization`, `policy`

### Admitted route table (`default_obligation_routes`)

| Route kind | Family | Profile | Property | View | Notation | Authority ceiling | Namespace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `intent` | `first_order` | `default` | `validity` | `facts` | `intent_facts` | satisfiability | family |
| `skill` | `program` | `dynamic_hoare` | `partial_correctness` | `action_hoare` | `hoare_action_contract` | candidate | profile |
| `prompt` | `first_order` | `default` | `validity` | `facts` | `prompt_candidate` | candidate | family |
| `goal` | `intention_agency` | `skill_goals` | `goal_satisfaction` | `skill_goals` | `skill_goal` | candidate | profile |
| `guard` | `first_order` | `guards_effects` | `guard` | `guards_effects` | `guard_predicate` | satisfiability | profile |
| `workflow` | `temporal` | `workflow_temporal` | `ordering` | `workflows` | `workflow_temporal` | bounded | profile |
| `authorization` | `authorization` | `tool_permissions` | `authorization` | `tool_permissions` | `tool_permission` | authorization | profile |
| `policy` | `deontic` | `default` | `obligation` | `norms` | `deontic_norm` | candidate | family |
| `safety` | `temporal` | `safety` | `safety` | `safety` | `safety_invariant` | bounded | **property** |
| `liveness` | `temporal` | `liveness` | `liveness` | `liveness` | `liveness_progress` | finite_trace | **property** |
| `verification_condition` | `program` (expression only) | `dynamic_hoare` | `validity` | `verification_condition` | `vc_surface` | candidate | **view_role** |

Property kinds (`safety`, `liveness`) and the view role
(`verification_condition`) must never be admitted as semantic families
(`PROPERTY_KIND_ROUTE_KINDS`, `VIEW_ROLE_ROUTE_KINDS`,
`NEVER_FAMILY_PROPERTY_KINDS`, `NEVER_FAMILY_OPERATION_ROLES`).

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
| Domain id | `ui_ux_ir` |
| Package (when present) | `ipfs_datasets_py/logic/ui_ux_ir` |
| Interfaces | `UIUXLogicSlice@2`, `UIUXSourceGate@2`, `UIUXFormalizationAdapter@2` |
| Schemas | `ui-ux-logic-slice/v2`, `ui-ux-source-gate/v2`, `ui-ux-formalization-adapter/v2` |
| Module | `ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py` |
| Owner | `domain:ui_ux_ir` |

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
does not block other domain work.

### Requirement surfaces (fixed set)

| Surface id | Family hint | Description |
| --- | --- | --- |
| `accessibility` | `first_order` | Accessibility property obligations over UI structure and state |
| `authorization` | `authorization` | Authorization and permission constraints over UI actions |
| `interaction_event` | `event_calculus` | Interaction / event-calculus obligations for user/system events |
| `observable_state` | `transition_system` | Observable navigation and runtime state transitions |
| `ontology_frame` | `frame_logic` | Ontology/frame (F-logic) component and relation structure |
| `workflow` | `temporal` | Workflow temporal obligations over multi-step UI journeys |

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
| `data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md` | This conformance note (LPC-043 / LPC-173 declared output) |

Inventory aliases (`intent_ir.domain_slice`, `ui_ux_ir.domain_slice`) refer to
the adapter roles satisfied by the modules above. Predicted
`.../domain_slice.py` paths are inventory placeholders, not mandatory new files
for LPC-043 admission. Production policy is preserved: document the live
adapters; do not invent universal domain IR or stub packages to satisfy path
strings.

## LPC-173 repair notes

| Finding | Resolution |
| --- | --- |
| Failure kind | `proposal_validation_failed` / `proposal_gate_failed` (validation never ran; rc 78) |
| Observed attempts | 4 consecutive LPC-043 failures (retry budget 3) |
| Evidence | `data/agent_supervisor/logic_platform_canonicalization/state/discovery/2026-08-15-lpc-173-lpc-043-retry-budget.md` |
| Root cause | Proposal path envelope rejected inventing predicted `domain_slice.py` / `ui_ux_ir` package paths outside the declared note output, while production adapters already exist under `logic_slice_v2` / `ui_ux_logic_gate_v2` |
| Repair | Emit this declared note only; preserve production admission policy and tests |
| Release effect | Completing LPC-173 releases LPC-043 from strategy `blocked_tasks` so the supervisor can re-admit the source task against the documented contract |

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
- No adapter invents a universal domain IR or collapses the other domain’s
  ontology.
- Validation (source task LPC-043):
  `python -m pytest ipfs_datasets_py/tests/unit/logic/intent_ir ipfs_datasets_py/tests/unit/logic/ui_ux_ir -q`
- Validation (repair task LPC-173):
  evidence file present at the discovery path recorded above.
