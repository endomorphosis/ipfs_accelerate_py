# LPC-052 Typed Provider Responses with Untrusted Default Authority

**Task:** LPC-052 — Typed provider responses with untrusted default authority  
**Goal:** LPC-G050  
**Depends on:** LPC-050 (`LogicProviderProtocol@2` requests)  
**Interface:** `LogicProviderResponse@2`  
**Module:** `ipfs_datasets_py.logic.backends.response_v2`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/backends/response_v2.py`  
**Protocol version:** `2`  
**Schema:** `ipfs_datasets_py/logic-provider-response@2`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_provider_response_v2.py -q`

## Purpose

`LogicProvider@1` responses (`LogicProviderResponse` in `provider.py`) carry only
correlation fields plus a free-form `result` object or a typed `error`. That is
insufficient for the orthogonal axis model (LPC-030) and for fail-closed
authority handling (LPC-032).

LPC-052 adds **LogicProviderResponse@2**: a typed response envelope that always
carries request correlation, provider identity, operation lifecycle status,
semantic verdict, evidence kind/authority, boundedness, lineage collections,
resource usage, cache provenance, and error. Provider-emitted authority
**defaults to untrusted** (`advisory`). Operation success never upgrades trust.

## Generation map

| Generation | Module | Role |
| --- | --- | --- |
| `LogicProviderResponse@1` | `logic/backends/provider.py` | Live portable wire leaf; `ok` + free-form `result` |
| `LogicProviderProtocol@2` requests | `logic/backends/protocol_v2.py` | Typed operation-specific requests (LPC-050) |
| **`LogicProviderResponse@2`** | **`logic/backends/response_v2.py`** | **Typed responses with untrusted default authority (this task)** |
| Typed backend results | `logic/backends/results.py` | Authority-scoped conclusion records (downstream) |

## Required field inventory

Every admitted `ProviderResponseV2` body exposes **all** of the following
fields (acceptance LPC-052). Missing fields fail closed at construction or
admission.

| Field | Type / carrier | Question answered |
| --- | --- | --- |
| `request_id` | trimmed string | Which request does this correlate to? |
| `operation` | `ProtocolOperationV2` | Which closed operation produced this? |
| `provider_id` | trimmed string | Which provider emitted this? |
| `provider_version` | trimmed string | Which provider version? |
| `operation_status` | `LogicOperationStatus` | Did the attempt finish, and how? |
| `verdict` | `LogicSemanticVerdict` | What semantic conclusion (if any)? |
| `evidence_kind` | `LogicEvidenceKind` | What format/category of evidence? |
| `evidence_authority` | `LogicEvidenceAuthority` | What trust ceiling does it carry? |
| `boundedness` | `LogicBoundedness` | What semantic scope bounds the conclusion? |
| `assumptions` | `tuple[str, …]` | Explicit assumption record ids |
| `translations` | `tuple[ResponseTranslationRef, …]` | Translation step identities + preservation |
| `sources` | `tuple[ResponseSourceRef, …]` | Source document identities + digests |
| `artifacts` | `tuple[ResponseArtifactRef, …]` | Artifact identities + digests + kind |
| `resources` | `ResourceUsage` | Observed wall/steps/memory/output usage |
| `cache_provenance` | `CacheProvenanceV2` | Cache hit/miss disposition + digests |
| `error` | `LogicProviderFailure \| null` | Typed failure when status is non-success |

Additional identity fields (always present, not free-form routing):

| Field | Value |
| --- | --- |
| `interface` | `LogicProviderResponse@2` |
| `protocol_version` | `2` |
| `schema_version` | `ipfs_datasets_py/logic-provider-response@2` |
| `translation_preservation` | response-level translation claim (axis) |
| `duration_ms` | non-negative wall time for the attempt |
| `metadata` | strict JSON object (no free-form payload routing) |

Closed constant: `REQUIRED_RESPONSE_FIELDS` in `response_v2.py` lists the
sixteen acceptance fields above.

## Untrusted default authority

| Constant | Default value | Meaning |
| --- | --- | --- |
| `DEFAULT_EVIDENCE_AUTHORITY` | `advisory` | Non-binding trust ceiling |
| `DEFAULT_EVIDENCE_KIND` | `candidate` | Format only; not trust |
| `DEFAULT_SEMANTIC_VERDICT` | `unknown` | No decisive semantic answer |
| `DEFAULT_BOUNDEDNESS` | `unknown` | Scope not established |
| `DEFAULT_TRANSLATION_PRESERVATION` | `not_applicable` | No translation claim |

Factory `ProviderResponseV2.succeeded(...)` applies these defaults unless the
caller supplies explicit axis values. **Succeeded + unknown + advisory** remains
a representable, valid coordinate (LPC-032 counterexample).

### Non-inference rules

1. `operation_status=succeeded` does **not** imply `verdict=proved`.
2. `operation_status=succeeded` does **not** imply `evidence_authority` above
   `advisory`.
3. Cache hits (`CacheProvenanceV2.hit_kind=hit`) never raise authority.
4. Raising authority above the untrusted default requires
   `with_authority(..., allow_upgrade=True)` after independent validation or
   reconstruction — silent upgrade fails closed (`ResponseAuthorityError`).
5. `is_success` answers lifecycle only; `is_trusted` answers authority only.

## Nested carriers

| Type | Schema | Role |
| --- | --- | --- |
| `ResponseTranslationRef` | `…/logic-provider-response-translation-ref@2` | `translation_id`, optional digest, `preservation` |
| `ResponseSourceRef` | `…/logic-provider-response-source-ref@2` | `document_id`, optional `source_digest` |
| `ResponseArtifactRef` | `…/logic-provider-response-artifact-ref@2` | `artifact_id`, optional digest, `kind` |
| `CacheProvenanceV2` | `…/logic-provider-cache-provenance@2` | `hit_kind`, key/entry digests, reason |

`CacheHitKind` closed set: `miss`, `hit`, `negative_hit`, `bypass`, `unknown`.
A `hit` without key or entry digest fails closed.

## Status / error coupling

| `operation_status` | `error` |
| --- | --- |
| `succeeded` | must be `null` |
| `failed` / `error` / `invalid` | required |
| other terminal statuses | optional (caller-supplied) |

Failed factory `ProviderResponseV2.failed(...)` always sets
`evidence_authority=advisory` (untrusted default) and a typed
`LogicProviderFailure`.

## Admission helpers

| Symbol | Role |
| --- | --- |
| `admit_provider_response_v2` | Admit typed body or mapping; revalidates all fields |
| `default_untrusted_authority` | Returns `LogicEvidenceAuthority.ADVISORY` |
| `response_carries_required_fields` | Inventory check for the sixteen acceptance fields |
| `ProviderResponseV2.from_dict` / `to_dict` | Strict JSON round-trip |
| `ProviderResponseV2.succeeded` / `failed` | Construction factories |

## What this does **not** do

1. **Does not** replace the @1 wire leaf for existing supervisor facades.
2. **Does not** implement LPC-051 v1 response adaptation.
3. **Does not** promote provider success into proof authority (LPC-032).
4. **Does not** make cache hits a trust root.
5. **Does not** invent a second axis model; reuses `ir_core.axes` enums.

## Relationship to requests

| Request (LPC-050) | Response (LPC-052) |
| --- | --- |
| `CapabilityRequestV2` | `ProviderResponseV2` with `operation=capability` |
| `TranslationRequestV2` | `ProviderResponseV2` with `operation=translate` |
| `ProveCheckRequestV2` | `ProviderResponseV2` with `operation=prove` or `check` |
| `ReconstructRequestV2` | `ProviderResponseV2` with `operation=reconstruct` |
| `VerifyRequestV2` | `ProviderResponseV2` with `operation=verify` |
| `AttestRequestV2` | `ProviderResponseV2` with `operation=attest` |

Correlation: response `request_id` and `operation` must match the admitted
request. Provider identity fields are required on every @2 response.

## Validation coverage

`tests/unit/logic/backends/test_provider_response_v2.py` asserts:

* interface identity `LogicProviderResponse@2` and protocol version `2`;
* full acceptance field inventory on every constructed response;
* untrusted default authority (`advisory`) on succeeded responses;
* succeeded + unknown + advisory is representable and not trusted;
* silent authority upgrade fails closed;
* failed responses require error and keep untrusted authority;
* cache provenance hit/miss rules;
* dict/JSON round-trip preserves all required fields;
* note documents the field inventory and default authority rule.

## Acceptance

- Responses carry request id, operation, provider id/version, operation status,
  verdict, evidence kind/authority, boundedness, assumptions, translations,
  sources, artifacts, resources, cache provenance, and error.
- Default evidence authority is untrusted (`advisory`).
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_provider_response_v2.py -q`
