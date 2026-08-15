# LPC-041 Legal Domain Adapter Conformance (TDFOL, DCEC, Frame Logic)

**Task:** LPC-041 — Legal domain adapter conformance (TDFOL, DCEC, frame logic)  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (`notes/new_write_path.md`, `DomainLogicSlice@2`)  
**Interfaces:** `DomainLogicSlice@2`, `LegalFormalizationAdapter@2`, `LegalLogicSlice@2`  
**Domain:** `legal` / `legal_ir`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py -q`

## Purpose

The legal domain adapter lowers legal IR claims into admitted
**DomainLogicSlice@2** records without collapsing ontologies. TDFOL, DCEC, and
frame logic remain **pairwise distinct** semantic families. Silent remapping to
generic first-order logic, monadic deontic alone, or untyped object framing is
forbidden.

Every adapter declaration binds the LPC-G040 inventory:

| Declaration field | Role |
| --- | --- |
| source domain | Stable domain id (`legal` / `legal_ir`) |
| view | Typed view identity used on the slice |
| family / profile | Semantic family and fragment/profile |
| property | Obligation / property kind |
| notation | Source notation identity |
| preserved semantics | What the lowering keeps |
| lost semantics | What the lowering deliberately drops or weakens |
| assumptions | Explicit assumption ids required for admission |
| unsupported constructs | Constructs that force `unsupported` (never silent drop) |
| proof-safety | Whether a prove outcome may establish the legal claim without reconstruction |
| counterexample-safety | Whether a counterexample may stand as a legal counter-model without reconstruction |

Executable regression coverage lives in
`ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py`.

## Non-collapse rules (normative)

| Source surface | Retained family on `DomainLogicSlice@2` | Forbidden silent mapping |
| --- | --- | --- |
| TDFOL / temporal first-order legal formulas | `tdfol` | `first_order` / bare FOL without temporal+deontic force |
| DCEC / CEC / dynamic event-calculus legal events | `dcec` | generic `deontic` alone, or FOL without event/fluent identity |
| Frame logic / F-logic legal roles | `frame_logic` | untyped object framing / anonymous triple bags |

Related but **different** legal surfaces (deontic norms, authorization, pure
FOL facts, event_calculus-only CEC.native encoding targets) stay on their own
routes. They must not absorb TDFOL/DCEC/frame identities.

### Alias discipline

| Surface label | Adapter family | Notes |
| --- | --- | --- |
| `tdfol`, `TDFOL`, `temporal_first_order` | `tdfol` | Profile `temporal_first_order` is composition metadata over retained `tdfol` |
| `dcec`, `DCEC`, `cec`, `CEC.native`, `event_calculus` (DCEC path) | `dcec` | Retained composition family; deontic + event_calculus + modal components stay linked in composition metadata, not collapsed |
| `frame_logic`, `flogic`, `modal.frame_logic` | `frame_logic` | Typed roles/relations; advisory evidence ceiling |

Legal typed route catalog (`LegalFormalizationAdapter@2`) may resolve the
`dcec` **alias** onto the CEC.native / `event_calculus` **component** route for
backend targeting. The domain adapter declaration still stamps
`family=dcec` on new DomainLogicSlice@2 writes so the composition identity is
not erased. Component routing is not ontology collapse.

## Adapter catalog

### 1. TDFOL (`legal-adapter/tdfol/v1`)

| Field | Value |
| --- | --- |
| source domain | `legal` (`legal_ir` package domain) |
| view | `tdfol` (`legal-ir-view/tdfol/v1`) |
| family | `tdfol` |
| profile | `temporal_first_order` |
| property | `validity` |
| notation | `canonical_text` |
| target component | `TDFOL.prover` |
| route id | `legal-route/tdfol/v1` |
| proof-safety | **false** — result ceiling is candidate; NL extraction and provider success never mint theorem authority (LPC-032) |
| counterexample-safety | **true** — finite discrete temporal counter-models with explicit anchors may be treated as checkable counterexamples for the admitted fragment |

**Preserved semantics**

* quantifier scope over legal individuals and time points
* temporal anchors and event order relative to the formula
* deontic force carried inside the temporal formula (not stripped to pure FOL)
* predicate / atom identity from the typed expression
* source document id + digest and expression id + digest

**Lost semantics**

* continuous / dense time metrics not listed in the profile
* unreviewed natural-language glosses of the formula
* kernel theorem authority (never inferred from formalization `ok`)
* infinite-trace fairness constraints outside the finite profile

**Assumptions** (must appear as `assumption_ids` on admitted slices)

* `asm:legal-domain`
* `asm:tdfol-temporal-first-order`
* `asm:finite-discrete-time`
* `asm:candidate-authority-ceiling`

**Unsupported constructs** (force `status=unsupported`, never silent drop)

* unanchored temporal operators without a time sort or event order
* free-form family strings (`logic_family` metadata routing)
* graph_projection / proof_translation / structural_round_trip as families
* natural-language-only claims without a typed expression

### 2. DCEC (`legal-adapter/dcec/v1`)

| Field | Value |
| --- | --- |
| source domain | `legal` (`legal_ir` package domain) |
| view | `dcec` (`legal-ir-view/cec/v1` component view retained under DCEC identity) |
| family | `dcec` |
| profile | `dcec_default` |
| property | `reachability` |
| notation | `canonical_text` |
| target component | `CEC.native` (encoding/target; does not replace family id) |
| route id | `legal-route/event-calculus/v1` (component route; slice family remains `dcec`) |
| proof-safety | **false** — candidate ceiling; composition does not auto-promote to kernel |
| counterexample-safety | **true** — finite event/fluent traces that violate a fluent lifecycle may stand as checkable counterexamples for the admitted fragment |

**Preserved semantics**

* event identity and fluent identity
* transition direction (initiates / terminates / holds)
* discrete time anchors for event order
* composition link of deontic, event_calculus, and cognitive/modal components under retained `dcec` identity
* source and expression digests

**Lost semantics**

* continuous fluents / hybrid dynamics outside the discrete profile
* full cognitive-attitude nesting beyond the admitted modal fragment
* silent rewrite of `dcec` → monadic `deontic` or bare FOL
* theorem authority from formalization success alone

**Assumptions**

* `asm:legal-domain`
* `asm:dcec-composition`
* `asm:event-fluent-identity`
* `asm:candidate-authority-ceiling`

**Unsupported constructs**

* events without fluent or time anchors when the claim requires lifecycle force
* collapsing DCEC to a pure deontic operator set without event/fluent structure
* operation roles (`graph_projection`, …) as family ids
* unversioned multi-family string ids replacing `dcec`

### 3. Frame logic (`legal-adapter/frame-logic/v1`)

| Field | Value |
| --- | --- |
| source domain | `legal` (`legal_ir` package domain) |
| view | `frame_logic` (`legal-ir-view/frame-logic/v1`) |
| family | `frame_logic` |
| profile | `typed_frame` |
| property | `validity` |
| notation | `canonical_text` |
| target component | `modal.frame_logic` |
| route id | `legal-route/frame-logic/v1` |
| proof-safety | **false** — evidence authority is advisory; never proof |
| counterexample-safety | **false** — frame triples are structural/advisory; they are not independently checkable legal counter-models without a separate model theory reconstruction |

**Preserved semantics**

* typed role / slot identity on frame triples
* relation direction (subject → object)
* selected frame id when present
* modal operator attachment when co-present on the formula
* exception scope tags attached to frame slots
* source and expression digests

**Lost semantics**

* untyped object bags / anonymous property graphs without role types
* Neo4j-compatible graph projection (that is a **view/operation role**, not this family)
* proof or kernel authority
* full F-logic inheritance / path expressions outside the typed triple core

**Assumptions**

* `asm:legal-domain`
* `asm:frame-typed-roles`
* `asm:advisory-authority-ceiling`

**Unsupported constructs**

* object framing without typed roles (anonymous subject/predicate/object only)
* treating `graph_projection` / `knowledge_graphs` as the frame_logic family
* free-form `flogic` payloads without expression identity
* promoting advisory frame structure to theorem authority

## DomainLogicSlice@2 lowering contract

Each adapter produces one admitted slice through the LPC-040 path:

```text
Legal claim / typed formula
  → TypedExpression (family + profile bound)
  → DomainLogicSlice@2.from_typed_expression(...)
  → require_admitted()
```

Required slice bindings (inherited from LPC-040):

* `document_id`, `source_digest`
* `expression_id`, `expression_digest`
* `family`, `profile`, `property`, `view`, `notation`
* `features`, `assumption_ids`, `unsupported_extensions` (empty when admitted)
* `status=admitted`, `content_digest`
* `domain` ∈ {`legal`, `legal_ir`} for legal writes

### Features stamped per adapter

| Adapter | Features (sorted on normalize) |
| --- | --- |
| TDFOL | `legal_ir`, `tdfol`, `temporal`, `first_order` |
| DCEC | `dcec`, `event`, `fluent`, `legal_ir` |
| Frame logic | `frame_logic`, `legal_ir`, `typed_role` |

## Authority and safety summary

| Adapter | Evidence authority (route) | Proof authority role | proof-safe | counterexample-safe |
| --- | --- | --- | --- | --- |
| TDFOL | independently_checkable | candidate | false | true |
| DCEC | independently_checkable | candidate | false | true |
| Frame logic | advisory | advisory | false | false |

Rules:

1. **Formalization `ok` ≠ proof** (LPC-032 / LPC-040).
2. Natural-language extraction is never proof authority.
3. Operation/view roles never become families.
4. Unsupported constructs list on the slice or reject; silent drops are forbidden.
5. Counterexample-safe adapters still require admitted expression identity and
   finite-fragment assumptions; they do not invent models from empty digests.

## Ownership and related modules

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/legal_domain_adapter.md` | This durable declaration |
| `ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py` | LPC-041 regression gate (declarations + DomainLogicSlice@2 admission) |
| `ipfs_datasets_py/logic/legal_ir/typed_adapter.py` | `LegalFormalizationAdapter@2` route catalog |
| `ipfs_datasets_py/logic/legal_ir/logic_slice_v2.py` | `LegalLogicSlice@2` vertical connector |
| `ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Production `DomainLogicSlice@2` contract |

Inventory id `dls:legal-domain-slice` is satisfied by this declaration plus the
executable gate in `test_domain_slice.py`. Downstream LPC-044 rejects
executable requests that lack an admitted slice.

## Acceptance

* Adapter declares **source domain**, **view**, **family/profile**, **property**,
  **notation**, **preserved/lost semantics**, **assumptions**, **unsupported
  constructs**, **proof-safety**, and **counterexample-safety** for **TDFOL**,
  **DCEC**, and **frame logic**.
* The three families are pairwise distinct and never silently mapped to FOL /
  `first_order`, generic `deontic`, or untyped object framing.
* Each adapter admits a `DomainLogicSlice@2` with the declared axes and
  assumption inventory; unsupported constructs force `status=unsupported`.
* Proof-safety is **false** for all three adapters (candidate/advisory ceiling).
* Counterexample-safety is **true** for TDFOL and DCEC, **false** for frame logic.
* Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/legal_ir/test_domain_slice.py -q`
