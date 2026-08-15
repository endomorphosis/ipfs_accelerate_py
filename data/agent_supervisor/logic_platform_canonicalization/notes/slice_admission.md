# LPC-044 Reject unadmitted slices at executable request construction

**Task:** LPC-044 — Reject unadmitted slices at executable request construction  
**Goal:** LPC-G040  
**Depends on:** LPC-041 (legal), LPC-042 (security/software/crypto), LPC-043 (intent/ui_ux)  
**Interfaces:** `DomainLogicSlice@2`, `LogicObligation@2`, `BackendRequest@2`  
**Tests:** `ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py`  
**Acceptance:** Executable requests without an admitted `DomainLogicSlice@2` are rejected.  
**Conflict policy:** Own admission tests and the smallest production gate.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py -q`

## Purpose

Domain adapters (LPC-041..043) lower domain IRs through `DomainLogicSlice@2`.
That slice is the **only** admissible seed for executable backend work:

```text
SourceDocument + TypedExpression
  → DomainLogicSlice@2          # status must be admitted
  → LogicObligation@2.from_slice
  → BackendRequest@2.from_slice  # executable request construction
  → LogicProviderProtocol@2 executable ops
```

LPC-044 freezes the fail-closed rule that **no** executable request may be
built from a missing, rejected, unsupported, or otherwise unadmitted slice.
Provider selection and execution never see work that skipped admission.

## Canonical call path

```text
Domain adapter / formalization new-write path
  → DomainLogicSliceV2 (artifacts_v3)
       require_admitted()                 # gate: status + no unsupported ext
  → LogicObligationV2.from_slice(...)     # re-checks require_admitted()
  → BackendRequestV2.from_slice(...)      # obligation path only
  → Prove/Check/Translate/… request       # binds admitted BackendRequest@2
```

Any failure at `require_admitted` or `from_slice` raises
`DomainSliceAdmissionError` (or `RequestV2Error` when the value is not a
`DomainLogicSliceV2`) **before** provider selection.

## Admission statuses

| `DomainSliceStatus` | May seed `BackendRequest@2`? | Notes |
| --- | --- | --- |
| `admitted` | yes | Must bind source + expression identity; no unsupported extensions |
| `rejected` | no | Fail closed with `DomainSliceAdmissionError` |
| `unsupported` | no | Must list `unsupported_extensions`; never executable |

`is_admitted` is true only for status `admitted`. `require_admitted()` returns
the same instance when admitted; otherwise it raises.

## Production gate surfaces

| Surface | Module | Behaviour |
| --- | --- | --- |
| `DomainLogicSliceV2.require_admitted` | `logic/formalization/artifacts_v3.py` | Rejects non-admitted status and unsupported extensions |
| `LogicObligationV2.from_slice` | `logic/backends/requests_v2.py` | Calls `require_admitted()`; binds slice id/digest into the obligation |
| `BackendRequestV2.from_slice` | `logic/backends/requests_v2.py` | Builds only through `LogicObligationV2.from_slice` |
| Executable `LogicProviderProtocol@2` ops | `logic/backends/protocol_v2.py` | Require an admitted `BackendRequest@2` (translate/prove/check/reconstruct/verify/attest) |

There is no alternate constructor that promotes a rejected or unsupported
slice into an executable request. Free-form payloads, raw formulas, and bare
family strings remain forbidden on the v2 request path (see LFP2-007 /
`requests_v2` admission rules).

## Rejected inputs (normative)

| Input | Expected failure |
| --- | --- |
| Slice with `status=rejected` | `DomainSliceAdmissionError` ("not admitted") |
| Slice with `status=unsupported` | `DomainSliceAdmissionError` ("not admitted") |
| Value that is not `DomainLogicSliceV2` | `RequestV2Error` ("from_slice requires DomainLogicSliceV2") |
| Missing finite bounds on `from_slice` | `MissingBoundsError` (bounds are required even for admitted slices) |
| Admitted slice with unsupported extensions | Construction of the admitted slice itself fails (`DomainSliceAdmissionError`) |

## Positive path (admitted only)

An admitted slice binds, before any provider is selected:

* `document_id` + `source_digest`
* `expression_id` + `expression_digest`
* typed `family` / `profile` / `property` / `view` / `notation`
* `slice_id` + `content_digest` (carried as obligation/request `slice_digest`)
* features and assumption ids from the slice

`BackendRequestV2.from_slice` copies those bindings into the request so
downstream executable protocol ops cannot invent lineage.

## Relationship to nearby tasks

| Task | Boundary |
| --- | --- |
| LPC-040 | New formalization writes admit only `FormalizationArtifact@3` / `DomainLogicSlice@2` |
| LPC-041..043 | Domain adapters emit admitted slices without collapsing ontologies |
| **LPC-044** | Executable request construction rejects anything that is not an admitted slice |
| LPC-032 | Provider/operation success never mints proof authority (orthogonal axis rule) |
| LPC-050+ | `LogicProviderProtocol@2` executable ops consume admitted `BackendRequest@2` |

## Forbidden silent promotions

| Observation | Forbidden outcome |
| --- | --- |
| Rejected domain slice | `LogicObligation@2` / `BackendRequest@2` construction |
| Unsupported domain slice | Executable prove/check/translate elevation |
| Missing slice (no `DomainLogicSlice@2`) | Executable request constructed from free-form payload |
| Partial formalization `ok` alone | Implied slice admission or backend request |
| Domain adapter advisory retention | Silent elevation to executable `@2` work |

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | `DomainLogicSlice@2` + `require_admitted` (preserve) |
| `ipfs_datasets_py/ipfs_datasets_py/logic/backends/requests_v2.py` | `from_slice` gates for obligation and backend request (preserve) |
| `ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py` | LPC-044 regression coverage |
| `data/agent_supervisor/logic_platform_canonicalization/notes/slice_admission.md` | This durable admission note |

Inventory marker `dls:unadmitted-slice-gate` is satisfied by the production
`require_admitted` / `from_slice` gates and the regression test above.

## Acceptance

- `LogicObligationV2.from_slice` and `BackendRequestV2.from_slice` reject
  rejected and unsupported `DomainLogicSlice@2` instances.
- Non-`DomainLogicSliceV2` inputs cannot enter the slice admission path.
- Admitted slices alone produce executable requests that carry source,
  expression, and slice identity.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py -q`
