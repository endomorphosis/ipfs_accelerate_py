# LPC-040 Typed New-Write Path: FormalizationArtifact@3 and DomainLogicSlice@2

**Task:** LPC-040 — Enforce FormalizationArtifact@3 and DomainLogicSlice@2 on new writes  
**Goal:** LPC-G040  
**Depends on:** LPC-021 (catalog drift), LPC-032 (success ≠ proof)  
**Interfaces:** `FormalizationArtifact@3`, `DomainLogicSlice@2`  
**Production owner:** `ipfs_datasets_py.logic.formalization.artifacts_v3`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/formalization/test_admission.py -q`

## Purpose

Only **FormalizationArtifact@3** and **DomainLogicSlice@2** may admit new domain
formalization writes. Every admitted write binds exact source identity, typed
expression identity, namespace axes, features, assumptions, unsupported
extensions, status, and content identity. Free-form routing payloads and
legacy compiler envelopes cannot seed backend requests.

This note freezes the new-write path and the binding inventory. Executable
admission regression coverage lives in
`ipfs_datasets_py/tests/unit/logic/formalization/test_admission.py`.

## Canonical pipeline

```text
SourceDocument
  → syntax_core parse artifact
  → TypedExpression
  → ElaborationArtifactV2
  → FormalizationArtifact@3
  → DomainLogicSlice@2
  → LogicObligation@2 / BackendRequest@2
```

| Stage | Interface / type | Role on new writes |
| --- | --- | --- |
| Source | `SourceDocument` | Exact document id + content digest + spans |
| Expression | `TypedExpression@1` | Expression id + content digest + family/profile |
| Elaboration | `ElaborationArtifactV2` | Backend-ready typed expression with lineage |
| Formalization | **`FormalizationArtifact@3`** | Domain-neutral envelope; only new-write generation |
| Domain slice | **`DomainLogicSlice@2`** | Admitted domain lowering; only backend-seed generation |
| Request | `BackendRequest@2` | Requires at least one admitted slice |

## Only admitted generations

| Generation | Classification | Module | New-write role |
| --- | --- | --- | --- |
| `FormalizationArtifact@3` | canonical | `logic.formalization.artifacts_v3` | Only admitted formalization write surface |
| `DomainLogicSlice@2` | canonical | `logic.formalization.artifacts_v3` | Only admitted domain-slice write surface |
| `FormalizationArtifact` (compiler v1) | legacy | `logic.formalization.compiler` | Dual-read / bridge only; not a new-write admit path |
| Advisors / autoencoder proposals | experimental | `proposal_advisors`, `autoencoder_advisor` | Candidates only; never proof authority |

## Required bindings on every new write

Acceptance (LPC-040): new writes bind **all** of the following. Missing or
mismatched fields fail closed.

### FormalizationArtifact@3 envelope

| Binding | Fields | Rule |
| --- | --- | --- |
| Source identity | `document_id`, `source_digest` | Digest is `sha256` hex; must match `SourceDocument` when cross-checked |
| Expression identity | `expression_id`, `expression_digest` | Digest is `sha256` hex; must match `TypedExpression` when cross-checked |
| Spans | `source_map` (optional but lineage-checked) | Map `document_id` must match artifact; ranges must fit document byte length |
| Namespace axes | `family`, `profile`, `view`, `notation` | Typed `LogicIdentity` namespaces; cross-namespace labels rejected |
| Assumptions | `assumption_ids` | Sorted unique record ids |
| Status | `status` | `ok` / `partial` / `failed` / `rejected` |
| Content identity | `content_digest`, `lineage_digest` | Computed from identity/lineage payloads; wrong digests rejected |
| Domain slices | `slices` | `status=ok` requires ≥1 admitted `DomainLogicSlice@2` |
| Metadata hygiene | `metadata` | Forbids free-form routing keys (`payload`, `raw_formula`, `logic_family`, …) |

### DomainLogicSlice@2 (admitted)

| Binding | Fields | Rule |
| --- | --- | --- |
| Source identity | `document_id`, `source_digest` | Required for `status=admitted`; must match parent artifact |
| Expression identity | `expression_id`, `expression_digest` | Required for `status=admitted`; must match typed expression when cross-checked |
| Spans | `source_range` | Optional; validated against document byte length when present |
| Namespace axes | `family`, `profile`, `property`, `view`, `notation` | All five required; typed namespaces only |
| Features | `features` | Feature-identity strings; unique; sorted on normalize |
| Assumptions | `assumption_ids` | Sorted unique record ids |
| Unsupported extensions | `unsupported_extensions` | **Must be empty** when `status=admitted`; required non-empty when `status=unsupported` |
| Status | `status` | `admitted` / `rejected` / `unsupported` |
| Content identity | `content_digest` | Matches slice identity payload |
| Domain | `domain` | Lowercase domain identifier; must match parent artifact domain |

## Admission gates (fail-closed)

Production enforcement anchors in `artifacts_v3` (preserved; not rewritten by
this task):

1. **Admitted slice completeness.** `DomainSliceStatus.ADMITTED` requires
   non-empty `document_id`, `source_digest`, `expression_id`, and
   `expression_digest`, and forbids unsupported extensions.
2. **OK formalization requires admitted slice.**
   `FormalizationArtifactStatus.OK` requires ≥1 slice and ≥1 admitted slice,
   and rejects error/fatal diagnostics.
3. **Lineage coherence.** Every slice must share the artifact `document_id`,
   `source_digest`, and `domain`. Expression identity on each slice must be
   bound (non-empty id + digest).
4. **`require_admitted` / `require_admitted_slices`.** Backend-facing callers
   must call these gates; rejected or unsupported slices raise
   `DomainSliceAdmissionError`.
5. **Content digests.** Provided digests must match recomputed content/lineage
   digests; silent rewrite of identity is rejected.
6. **No free-form routing.** Metadata keys that re-introduce opaque payloads or
   free-form family strings raise `ArtifactV3Error`.
7. **Namespace discipline.** Family/profile/property/view/notation accept only
   their own `NamespaceKind`; e.g. a provider id cannot stand in for a family.

### Construction helpers (preferred new-write entrypoints)

| Helper | Produces | Binding behaviour |
| --- | --- | --- |
| `DomainLogicSliceV2.from_typed_expression(...)` | `DomainLogicSlice@2` | Copies expression id/digest/family/profile; binds source digest + property/view/notation |
| `FormalizationArtifactV3.from_elaboration(...)` | `FormalizationArtifact@3` | Projects backend-ready elaboration into one admitted slice + envelope |

Direct constructors remain valid when callers supply the full binding set.
Incomplete constructors fail in `__post_init__`.

## Forbidden new-write patterns

| Pattern | Disposition |
| --- | --- |
| Write via compiler `FormalizationArtifact` (v1) as the only envelope | Not admitted for new domain formalization writes |
| Admit a slice with empty source or expression digests | `DomainSliceAdmissionError` |
| Admit a slice that lists unsupported extensions | `DomainSliceAdmissionError` |
| `status=ok` formalization with only rejected slices | `ArtifactV3Error` |
| Metadata carrying `payload` / `raw_formula` / `logic_family` | `ArtifactV3Error` |
| Cross-namespace identity (e.g. provider id as family) | `ArtifactV3Error` |
| Slice document/source/domain mismatch vs parent artifact | `ArtifactV3LineageError` |
| Infer proof authority from formalization `ok` alone | Forbidden (LPC-032); formalization status ≠ kernel authority |

## Domain adapter expectations (downstream LPC-041..043)

Adapters lower domain IRs **through** `DomainLogicSlice@2` without collapsing
ontologies:

| Domain | Must keep distinct | Forbidden silent mapping |
| --- | --- | --- |
| legal | TDFOL, DCEC, frame logic | FOL / generic deontic / object framing |
| security / software / crypto | respective family ontologies | cross-domain collapse |
| intent / ui_ux | domain ontologies | universal domain IR |

Each adapter must declare source domain, view, family/profile, property,
notation, preserved/lost semantics, assumptions, unsupported constructs,
proof-safety, and counterexample-safety (LPC-041..043). Executable requests
without an admitted `DomainLogicSlice@2` are rejected (LPC-044).

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Production `FormalizationArtifact@3` / `DomainLogicSlice@2` contracts (preserve) |
| `ipfs_datasets_py/tests/unit/logic/formalization/test_admission.py` | LPC-040 admission regression + new-write binding gate |
| `data/agent_supervisor/logic_platform_canonicalization/notes/new_write_path.md` | This durable path note |

The inventory entry `fa:formalization-admission` (`formalization.admission`) is
satisfied by the admission gate exercised in `test_admission.py` against
`artifacts_v3`. That helper does **not** replace `artifacts_v3`; it enforces
the binding inventory for new writes and fails closed on incomplete envelopes.

## Acceptance

- New writes construct only through `FormalizationArtifact@3` and
  `DomainLogicSlice@2`.
- Every admitted write binds source, digest, spans, expression identity,
  family/profile/property/view/notation, features, assumptions, unsupported
  extensions, status, and content identity.
- Incomplete, free-form, or lineage-broken writes raise admission/contract
  errors before any backend request can form.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/formalization/test_admission.py -q`
