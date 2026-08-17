# LPC-041 Legal Domain Adapter Conformance (TDFOL, DCEC, Frame Logic)

**Task:** LPC-041 — Legal domain adapter conformance (TDFOL, DCEC, frame logic)  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Adapter declares source domain, view, family/profile, property, notation, preserved/lost semantics, assumptions, unsupported constructs, proof-safety, and counterexample-safety.  
**Conflict policy:** Own legal slice adapter. Do not map TDFOL/DCEC/frame logic to generic FOL/deontic/object framing.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py -q`

## Purpose

Legal IR owns a sealed domain ontology for normative, temporal, event, and frame
reasoning. New formalization writes lower that ontology **through**
`DomainLogicSlice@2` (LPC-040) without inventing a universal domain IR and
without silently remapping TDFOL, DCEC, or frame logic into generic FOL,
generic deontic, or object framing.

This note freezes adapter locations, domain identity, admitted route tables,
namespace axes, assumption axes, preservation/loss declarations, authority
ceilings, and non-collapse rules. It is the durable LPC-041 evidence for the
legal domain adapter that feeds backend requests via admitted slices only.

## Canonical lowering path

```text
Legal IR claim / view / obligation
  → TypedExpression (family + profile from the legal route)
  → DomainLogicSlice@2   (DomainLogicSliceV2.from_typed_expression)
  → LegalLogicSlice@2    (explicit deontic/temporal/defeasibility/jurisdiction/priority/authority axes)
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
| Domain | `legal_ir` |

Construction pattern used by the legal adapter:

```text
DomainLogicSliceV2.from_typed_expression(
    expression,
    slice_id=...,
    domain="legal_ir",
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

Inventory LPC-004 predicted path named `legal_ir.domain_slice`
(`ipfs_datasets_py/logic/legal_ir/domain_slice.py`). Live accelerate
implementation lives as the `LegalLogicSlice@2` connector that **emits**
`DomainLogicSlice@2` records. That connector is the production domain adapter
for LPC-041.

| Domain | Domain id | Adapter interface | Production module | Emits |
| --- | --- | --- | --- | --- |
| Legal | `legal_ir` | `LegalLogicSlice@2` | `ipfs_datasets_py/logic/legal_ir/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted legal claim/route |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Module | Role |
| --- | --- |
| `legal_ir/typed_adapter.py` | `LegalFormalizationAdapter@2`, `LEGAL_LOGIC_ROUTE_CATALOG`, `resolve_legal_route` |
| `legal_ir/adapter.py` | Legacy/shared formalization view registry (multi-view sample adaptation) |
| `families/profiles.py` | Canonical `tdfol` / `dcec` compositions with mandatory metadata |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `legal_ir.canonical_*` round-trip compiler/decompiler surfaces
- `legal_ir.constraint_query`, `legal_ir.proof_cache`
- Multi-view training/optimizer hooks under modal Leanstral / legal_ir hammer paths

Inventory alias `legal_ir.domain_slice` refers to the adapter role satisfied by
`logic_slice_v2`; that module is the production write path for admitted legal
domain lowering under LPC-041.

## Shared adapter contract (legal domain)

Each route/claim descriptor declares the LPC-G040 / LPC-041 fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` | Exact id `legal_ir`; must match parent formalization artifact |
| View | route `view_name` → `view_id(...)` | Typed view namespace; never free-form |
| Family / profile | expression + slice (`family`, `profile`) | From the legal route table only; no new families |
| Property | route/kind `property_name` → `property_id(...)` | Property is never promoted to a family |
| Notation | route notation → `notation_id(...)` | Surface notation for the admitted view (default `canonical_text`) |
| Preserved semantics | route `preservation_rules` | Explicit preservation ids from the typed route catalog |
| Lost semantics | explicit `loss_ids` per route class | Explicit loss ids; never silent |
| Assumptions | legal axis assumption ids | Declared even when empty / N/A |
| Unsupported constructs | deferred overlays + operation roles | Rejected fail-closed (not admitted) |
| Proof-safety | authority ceiling + result authority | Ceiling never upgrades along lineage; NL is never proof |
| Counterexample-safety | sat/model/trace result kinds + request digests | Counterexamples remain bound to exact request digests |

Lineage stages required on every admitted end-to-end connection:

```text
typed_origin → semantics → translation → request → result → replay → authority_lineage
```

Hermetic fixtures may supply provider execution and replay without live provers.
Tool absence is an availability result, never a mock proof (LPC-032).

---

## Distinct foundation families (must not collapse)

TDFOL, DCEC, and frame logic are **pairwise distinct** legal foundation families.
They may co-exist in a multi-view legal document, but each admitted slice binds
exactly one primary family/profile pair.

| Foundation | Canonical family id | Profile | Must not silently become |
| --- | --- | --- | --- |
| TDFOL | `tdfol` | `temporal_first_order` | `first_order` / generic FOL / opaque `temporal_fol` string |
| DCEC | `dcec` (composition-required) with event surface `event_calculus` | composition `dcec_default`; event profile `event_calculus` | generic `deontic` alone, generic FOL, or opaque multi-family string |
| Frame logic | `frame_logic` | `frame_logic` (default profile when none specialized) | object framing / free-form graph triples as a family / `graph_projection` |

### Composition discipline

- Catalog compositions `build_tdfol_composition()` and `build_dcec_composition()`
  retain the canonical family ids `tdfol` and `dcec` with mandatory
  `CompositionMetadata` (`COMPOSITION_REQUIRED_FAMILY_IDS`).
- Opaque replacement strings such as `temporal_first_order`,
  `first_order_temporal`, `temporal-fol`, `temporal_fol`, and `tfol` are
  **profiles or compositions**, never replacement family ids
  (`OPAQUE_REPLACEMENT_FAMILY_STRINGS`).
- DCEC composition metadata links components `deontic` + `event_calculus` +
  `modal` under retained `dcec` identity. Component roles stay explicit
  (`norms` / `events_fluents` / `cognitive_attitudes`); they are not flattened
  into monadic deontic alone.

### Legal IR route table alignment

| Route label / alias | Family id | Profile | View | Property | Notation | Authority ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `tdfol`, `TDFOL`, `temporal_first_order` | `tdfol` | `temporal_first_order` | `tdfol` | `validity` | `canonical_text` | candidate |
| `cec`, `event_calculus`, `dcec` (event surface alias) | `event_calculus` | `event_calculus` | `cec` / `source` | `reachability` | `canonical_text` | candidate |
| `frame_logic`, `flogic`, `modal.frame_logic` | `frame_logic` | `frame_logic` | `frame_logic` | `frame` | `canonical_text` | candidate / advisory |
| `deontic`, `conditional_normative` | `deontic` | `conditional_normative` | `deontic` / `source` | `validity` | `canonical_text` | candidate |
| `defeasible`, `defeasible_normative` | `deontic` | `defeasible_normative` | `defeasible_normative` / `source` | `validity` | `canonical_text` | candidate |
| `first_order`, `fol` | `first_order` | `first_order` | `first_order` | `validity` | `canonical_text` | candidate |
| `authorization`, `secpal`, `policy` | `authorization` | `secpal` | `authorization` / `source` | `authorization` | `canonical_text` | authorization / bounded |

Notes on DCEC surface routing:

- Full cognitive DCEC remains the catalog composition family `dcec` with
  mandatory composition metadata.
- Legal event/fluent claims lower through the typed `event_calculus` route
  (aliases include `cec`, `CEC.native`, and historical `dcec` event-surface
  labels). That route is **not** generic deontic and **not** FOL.
- A pure deontic base-norm route (`family=deontic`) never substitutes for DCEC
  composition or for TDFOL temporal force.

---

## 1. TDFOL adapter declaration

### Identity

| Field | Value |
| --- | --- |
| Source domain | `legal_ir` |
| View | `tdfol` (`view:tdfol` / legal view id `legal-ir-view/tdfol/v1`) |
| Family / profile | `tdfol` / `temporal_first_order` |
| Property | `validity` |
| Notation | `canonical_text` |
| Route id | `legal-route/tdfol/v1` |
| Target component | `TDFOL.prover` |
| Disposition | typed / native |

### Preserved semantics (TDFOL)

| Preservation id | Meaning |
| --- | --- |
| `quantifier_scope` | First-order quantifier binding structure |
| `temporal_anchor` | Explicit time points / intervals on formulas |
| `event_order` | Ordered event/time succession when present |
| `deontic_force` | Normative force carried into temporal context when present |

### Lost semantics (TDFOL)

| Loss id | Meaning |
| --- | --- |
| `loss.bounded_trace` | Finite-trace / bounded time window when discharged to model-check or SAT |
| `loss.metric_time_optional` | Metric interval density not assumed unless the temporal axis declares it |
| `loss.unbounded_time_not_claimed` | Unbounded continuous time is not implied by discrete finite profiles |

### Assumptions (TDFOL)

| Axis | Assumption ids (examples) |
| --- | --- |
| Temporal model | `axis:temporal:legal_discrete_finite`, `assumption:discrete_time` |
| Trace bound | `bound:trace_length`, `assumption:finite_trace` |
| Composition | `assumption:tdfol_composition_metadata` (catalog composition required) |
| Jurisdiction / authority | `axis:jurisdiction:*`, `axis:authority:candidate` |

### Unsupported constructs (TDFOL)

- Opaque family strings `temporal_fol` / `tfol` as replacements for `tdfol`
- Free-form natural language as theorem authority
- Silent rewrite of `tdfol` → `first_order` without temporal composition metadata

### Proof-safety (TDFOL)

- Result authority ceiling is **candidate** unless an independent kernel/backend
  receipt upgrades authority (legal slice construction alone never mints
  `official` / `theorem`).
- Natural-language extraction forces candidate ceiling (LPC-032:
  success ≠ proof).
- Authority never upgrades along lineage without independent reconstruction.

### Counterexample-safety (TDFOL)

- Sat/model/trace counterexamples bind to the exact `BackendRequest@2` digests
  (`source_digest`, `expression_digest`, `slice_digest` / request content).
- Finite-trace witnesses do not imply unbounded counterexamples outside the
  declared temporal bounds.
- Replay packages must re-bind the same digests; unbound models are rejected.

---

## 2. DCEC / event-calculus adapter declaration

### Identity

| Field | Value |
| --- | --- |
| Source domain | `legal_ir` |
| View | `cec` / `source` for event claims (`legal-ir-view/cec/v1`) |
| Family / profile | Event surface: `event_calculus` / `event_calculus`; composition family: `dcec` / `dcec_default` |
| Property | `reachability` (event surface); composition may also carry normative validity under deontic component |
| Notation | `canonical_text` |
| Route id (event surface) | `legal-route/event-calculus/v1` |
| Target component | `CEC.native` |
| Disposition | typed / native |

### Preserved semantics (DCEC / event surface)

| Preservation id | Meaning |
| --- | --- |
| `event_identity` | Event term identity across Happens/Initiates/Terminates style atoms |
| `fluent_identity` | Fluent term identity across HoldsAt-style atoms |
| `transition_direction` | Initiates vs Terminates polarity |
| `time_anchor` | Explicit time argument on event/fluent atoms |

Composition-level preserved roles (catalog `dcec`):

| Component family | Role |
| --- | --- |
| `deontic` | `norms` |
| `event_calculus` | `events_fluents` |
| `modal` | `cognitive_attitudes` |

### Lost semantics (DCEC / event surface)

| Loss id | Meaning |
| --- | --- |
| `loss.finite_event_horizon` | Finite event/time horizon under bounded discharge |
| `loss.cognitive_attitude_not_in_event_surface` | Pure event_calculus surface does not claim full DCEC cognitive attitudes |
| `loss.not_generic_deontic` | Event surface must not be read as monadic deontic obligation alone |
| `loss.not_classical_fol` | Event calculus fluents/events are not classical FOL facts without EC axioms |

### Assumptions (DCEC / event surface)

| Axis | Assumption ids (examples) |
| --- | --- |
| Temporal model | `axis:temporal:event_calculus_discrete`, `assumption:event_order` |
| Event horizon | `bound:event_horizon`, `assumption:finite_or_infinite_trace_declared` |
| Composition (when `dcec` family used) | `assumption:dcec_composition_metadata`, component roles declared |
| Jurisdiction / authority | `axis:jurisdiction:*`, `axis:authority:candidate` |

### Unsupported constructs (DCEC)

- Mapping DCEC / event surface to generic monadic deontic without event/fluent
  axes
- Mapping DCEC to classical FOL without EC axioms and composition metadata
- Dropping composition metadata for catalog family `dcec`
- Free-form narrative as event calculus proof

### Proof-safety (DCEC)

- Event-surface and DCEC composition results remain **candidate** unless an
  independent backend establishes a stronger admitted authority class.
- Cognitive/modal components never silently inherit theorem authority from a
  deontic candidate.
- Legal slices cannot claim official/theorem authority without kernel/backend
  receipts.

### Counterexample-safety (DCEC)

- Reachability counterexamples (event traces, fluent timelines) bind to the
  exact request digests and declared event horizon.
- A finite event-horizon counterexample does not authorize claims outside that
  horizon.
- Norm conflicts detected across claims are explicit records (`NormConflict`),
  never dropped to force a single “proved” outcome.

---

## 3. Frame logic adapter declaration

### Identity

| Field | Value |
| --- | --- |
| Source domain | `legal_ir` |
| View | `frame_logic` (`legal-ir-view/frame-logic/v1`) |
| Family / profile | `frame_logic` / `frame_logic` |
| Property | `frame` |
| Notation | `canonical_text` |
| Route id | `legal-route/frame-logic/v1` |
| Target component | `modal.frame_logic` |
| Disposition | typed / native (advisory evidence authority) |

### Preserved semantics (frame logic)

| Preservation id | Meaning |
| --- | --- |
| `typed_role` | Frame role / slot typing |
| `relation_direction` | Subject–predicate–object directionality |
| `modal_operator` | Modal operators attached to frame relations when present |
| `exception_scope` | Exception/defeater scope linked to frame roles when present |

### Lost semantics (frame logic)

| Loss id | Meaning |
| --- | --- |
| `loss.not_object_framing` | Frame-logic F-logic style roles are not generic object-oriented framing |
| `loss.not_graph_projection_family` | Graph projection / Neo4j export is an operation role, not this family |
| `loss.advisory_evidence` | Frame evidence authority is advisory; not independently checkable proof |
| `loss.slot_inheritance_optional` | Inheritance/isa closure only when frame conditions declare it |

### Assumptions (frame logic)

| Axis | Assumption ids (examples) |
| --- | --- |
| Frame conditions | `assumption:typed_roles`, `frame:slot_attachment_well_typed` (when translated) |
| Authority | `axis:authority:candidate` / advisory |
| Jurisdiction | `axis:jurisdiction:*` when bound into a legal slice |

### Unsupported constructs (frame logic)

- Treating `graph_projection`, `knowledge_graphs`, or `neo4j_compat` as the
  `frame_logic` family
- Collapsing frame logic into free-form object framing or untyped triple dumps
- Promoting advisory frame evidence to theorem authority

### Proof-safety (frame logic)

- Evidence authority is **advisory**; proof authority role is advisory;
  result ceiling remains candidate.
- Frame-logic slices never mint official/theorem authority.
- Operation roles (graph projection, proof translation, structural round-trip)
  cannot become families on legal slices.

### Counterexample-safety (frame logic)

- Frame mismatch / relation failures used as counterexamples must cite the
  exact expression and request digests.
- Graph-export differences are operation residuals, not semantic disproof of a
  frame-logic obligation unless an admitted translation edge says otherwise.

---

## Base legal claim kinds (DomainLogicSlice@2 owners)

These claim kinds lower through `LegalLogicSlice@2` with explicit axes. They
reuse catalog families without collapsing TDFOL/DCEC/frame foundations.

| Claim kind | Family | Profile | Property | View | Notation | Authority ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `base_norm` | `deontic` | `conditional_normative` | `validity` | `source` | `canonical_text` | candidate |
| `exception` | `deontic` | `defeasible_normative` | `validity` | `source` | `canonical_text` | candidate |
| `event` | `event_calculus` | `event_calculus` | `reachability` | `source` | `canonical_text` | candidate |
| `jurisdiction` | `authorization` | `secpal` | `authorization` | `source` | `canonical_text` | authorization / bounded |
| `priority` | `deontic` | `defeasible_normative` | `validity` | `source` | `canonical_text` | candidate |
| `conflict` | `deontic` | `conditional_normative` | `validity` | `source` | `canonical_text` | candidate |
| `policy` | `authorization` | `secpal` | `authorization` | `source` | `canonical_text` | authorization / bounded |

### Explicit legal axes (every admitted LegalLogicSlice@2)

| Axis | Schema | Rule |
| --- | --- | --- |
| Deontic profile | `legal-deontic-profile/v1` | Form and permission never `not_applicable` |
| Temporal model | `legal-temporal-model/v1` | Density and trace model never `not_applicable` |
| Defeasibility | `legal-defeasibility/v1` | Exception/defeater structure explicit |
| Jurisdiction | `legal-jurisdiction/v1` | Required for core kinds that declare it |
| Priority | `legal-priority/v1` | Ordered norm identities when priorities apply |
| Authority | `legal-authority-binding/v1` | Fail-closed; NL never theorem |

### Assumption axes (legal domain, all routes)

Every admitted legal route declares these axes (empty only when N/A, still
explicit via assumption ids or axis records):

| Axis | Examples |
| --- | --- |
| `deontic_profile` | monadic / dyadic / conditional / defeasible form, permission strength |
| `temporal_model` | discrete/dense time, finite/infinite trace, event order anchors |
| `defeasibility` | exception ids, defeater scope, unresolved conflicts |
| `jurisdiction` | territory, subject-matter, authority scope |
| `priority` | ordered norm identities, strict total order |
| `authority` | candidate / bounded / advisory; NL extraction flag |

---

## Unsupported / deferred constructs (legal domain)

Rejected for executable `LegalLogicSlice@2` / admitted `DomainLogicSlice@2`
family routing:

| Construct | Disposition |
| --- | --- |
| `graph_projection`, `knowledge_graphs`, `neo4j_compat` | Operation / view role — never family |
| `proof_translation`, `external_provers`, `prover_router` | Operation / view role — never family |
| `structural_round_trip`, `decompiler`, `round_trip` | Operation / view role — never family |
| `argumentation` | Declaration-only overlay (deferred) |
| `description_logic` | Declaration-only overlay (deferred) |
| `defeasible_logic` (full family) | Declaration-only; defeasible **norms** use deontic profile |
| `nonmonotonic_logic` | Deferred overlay |
| Free-form / natural-language as theorem | Authority rejected / forced candidate |
| Opaque `temporal_fol` family replacement | Forbidden; use `tdfol` + composition |

---

## Proof-safety and counterexample-safety (legal domain)

### Proof-safety

- Authority ceilings are route-local (`candidate`, `bounded`/`authorization`,
  `advisory`); lineage records never upgrade without independent kernel/backend
  receipts.
- `LegalAuthorityRole.OFFICIAL` and `ResultAuthority.THEOREM` are rejected at
  legal slice construction alone.
- Natural-language extraction forces candidate role and candidate result
  ceiling; it is never proof (LPC-032).
- Graph projection and other operation roles never establish semantic family
  proof authority.
- TDFOL and DCEC catalog compositions retain canonical ids; they do not borrow
  FOL or monadic-deontic theorem authority by silent remapping.

### Counterexample-safety

- Counterexamples (models, finite traces, event timelines, authorization
  denials) remain bound to exact request digests:
  `source_digest`, `expression_digest`, domain-slice content digest, and
  obligation/request ids.
- Bounded traces/event horizons do not authorize unbounded counterexample
  claims.
- Norm conflicts and parse ambiguities are explicit records; they are never
  silently dropped to manufacture a unique “proved” or “disproved” outcome.
- Replay must re-bind the same digests; unbound or cross-request counterexamples
  are rejected.

---

## Non-collapse rules (TDFOL ↔ DCEC ↔ frame logic ↔ deontic/FOL)

| Rule | Enforcement |
| --- | --- |
| Distinct domain id | Every admitted slice uses `domain=legal_ir` |
| Distinct connector interface | `LegalLogicSlice@2` only for legal domain lowering |
| TDFOL ≠ FOL | `tdfol` family retained; not rewritten to `first_order` |
| DCEC ≠ generic deontic | Event surface is `event_calculus`; composition family is `dcec` with metadata; not monadic deontic alone |
| Frame logic ≠ object framing | `frame_logic` family retained; graph projection stays an operation role |
| TDFOL ≠ DCEC ≠ frame_logic | Pairwise distinct family ids on foundation routes |
| Property ≠ family | Validity, reachability, frame, authorization stay properties |
| No new families | Adapter only selects existing catalog families (LPC-G040) |
| Operation roles ≠ families | Graph projection / proof translation / round-trip rejected as families |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| TDFOL (`tdfol`) | Classical FOL (`first_order`) without temporal composition |
| DCEC / event calculus | Generic monadic deontic or FOL facts |
| Frame logic | Object framing, free-form triples, or `graph_projection` family |
| Deontic base norm | TDFOL or DCEC without temporal/event axes |
| Any legal route | Free-form text as typed origin / theorem authority |

---

## End-to-end admission checklist (legal)

For each admitted legal claim/route the connector must:

1. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
2. Emit `DomainLogicSlice@2` via `from_typed_expression` with domain `legal_ir`.
3. Bind explicit legal axes (deontic, temporal, defeasibility, jurisdiction,
   priority, authority).
4. Call `require_admitted()` and `validate_against(document, expression)`.
5. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_slice`
   (or the legal slice helpers).
6. Attach route `preservation_rules` and explicit `loss_ids` for the foundation
   family in use.
7. Record hermetic execution/replay without authority upgrade.
8. Cover lineage stages with digest coherence source → request → execution →
   replay.

Incomplete slices fail closed before backend request construction (LPC-044
rejects executable requests without an admitted `DomainLogicSlice@2`).

## File ownership (LPC-041)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/logic_slice_v2.py` | Legal domain adapter → `DomainLogicSlice@2` (`LegalLogicSlice@2`) |
| `ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/typed_adapter.py` | Legal route catalog and non-collapse routing |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py` | LPC-041 regression coverage |
| `data/agent_supervisor/logic_platform_canonicalization/notes/legal_domain_adapter.md` | This conformance note |

## Acceptance

- Legal IR keeps TDFOL, DCEC (composition + event surface), and frame logic
  distinct and lowers admitted claims through `DomainLogicSlice@2` with domain
  `legal_ir`.
- Every adapter declaration for those foundations states source domain, view,
  family/profile, property, notation, preserved/lost semantics, assumptions,
  unsupported constructs, proof-safety, and counterexample-safety.
- No silent mapping of TDFOL → FOL, DCEC → generic deontic, or frame logic →
  object framing / graph-projection family.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py -q`
