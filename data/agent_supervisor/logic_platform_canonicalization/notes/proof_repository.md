# LPC-081 Unified Backend-Neutral Proof Repository Interface

**Task:** LPC-081 — Unified backend-neutral proof repository interface  
**Goal:** LPC-G080  
**Depends on:** LPC-080 (canonical semantic cache-key contract)  
**Interface:** `ProofRepository@1`  
**Module:** `ipfs_datasets_py.logic.common.proof_repository`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/common/proof_repository.py`  
**Schema:** `ipfs_datasets_py/proof-repository@1`  
**Schema version:** `proof-repository/v1`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/common/test_proof_repository.py -q`

## Purpose

Proof work produces more than cache entries. Plans, attempts, evidence,
receipts, counterexamples, attestations, freshness state, invalidations, and
lineage must share **one** datasets-owned interface so backends (in-memory,
DuckDB, remote) do not invent parallel vocabularies.

This note freezes `ProofRepository@1`. Executable coverage lives in
`ipfs_datasets_py/tests/unit/logic/common/test_proof_repository.py`.

## Ownership

| Owner | Responsibility |
| --- | --- |
| **Datasets** (`proof_repository.py`) | Public protocol, record types, capability inventory, fail-closed admission, reference in-memory backend |
| **Datasets** (`canonical_cache_key.py`) | Semantic cache-key identity used by every keyed record (LPC-080) |
| **DuckDB proof stack** (`duckdb_proof_*`) | Durable implementation detail; must project into this interface |
| **Supervisor** | Placement, single-flight, leases — not a second repository vocabulary |

Conflict policy (LPC-G080): own the canonical key **and** repository interface.
Do not redefine cache semantics in the supervisor. DuckDB may remain an
implementation.

## Capability inventory (acceptance)

One interface covers **all** of the following closed capabilities
(`PROOF_REPOSITORY_CAPABILITIES`):

| Capability | Protocol surface | Record type |
| --- | --- | --- |
| **plans** | `put_plan` / `get_plan` / `list_plans` | `ProofPlanRecord` |
| **attempts** | `put_attempt` / `get_attempt` / `list_attempts` | `ProofAttemptRecord` |
| **evidence** | `put_evidence` / `get_evidence` / `list_evidence` | `ProofEvidenceRecord` |
| **receipts** | `put_receipt` / `get_receipt` / `list_receipts` | `ProofReceiptRecord` |
| **counterexamples** | `put_counterexample` / `get_counterexample` / `list_counterexamples` | `ProofCounterexampleRecord` |
| **attestations** | `put_attestation` / `get_attestation` / `list_attestations` | `ProofAttestationRecord` |
| **lookup** | `lookup` | `ProofLookupResult` |
| **freshness** | `freshness` | `FreshnessReport` |
| **invalidation** | `invalidate` / `list_invalidations` | `ProofInvalidationRecord` |
| **lineage** | `put_lineage` / `lineage_of` / `list_lineage` | `ProofLineageEdge` |

Backends that omit any capability fail closed via
`require_full_capabilities` / `repository_covers_acceptance`.

## Interface identity

| Constant | Value |
| --- | --- |
| `interface` | `ProofRepository@1` |
| `schema` | `ipfs_datasets_py/proof-repository@1` |
| `schema_version` | `proof-repository/v1` |
| `module_version` | `1.0.0` |
| Reference backend id | `backend:in-memory` |

## Key binding (LPC-080)

Every keyed record admits a full `CanonicalProofCacheKey@1` body (not a bare
string). Lookup re-validates identity through `admit_canonical_cache_key` and
rejects **cross-environment hits** through `admit_cache_hit`.

A repository hit re-derives assurance from bound evidence and receipts; it does
**not** raise authority (LPC-032 / LPC-080 “cache is not a trust root”).

## Record semantics

### Plans

`ProofPlanRecord` — DAG of attempt slots under one cache key.

* Status: `draft | active | completed | failed | invalidated | cancelled`
* Carries `node_ids`, `depends_on`, owner, metadata
* Invalidation of the key marks non-terminal plans `invalidated`

### Attempts

`ProofAttemptRecord` — one execution against a key / plan node.

* Status: `pending | running | succeeded | failed | cancelled | invalidated`
* Optional `outcome_digest` (sha256), provider id, error text
* Provider success alone never mints kernel authority

### Evidence

`ProofEvidenceRecord` — content-addressed evidence blob.

* Binds `evidence_kind` + `authority_ceiling` (LPC-080 enums)
* Disposition: `candidate | draft | admitted | revoked`
* **Candidate-as-kernel** pairings fail closed at admission
* Content identity is `content_digest` (`sha256:…`)

### Receipts

`ProofReceiptRecord` — auditable binding of an action to a subject.

* Kinds: `evidence | translation | reconstruction | kernel_check | policy | attempt`
* Names subject / evidence / attempt ids; is **not** itself authority

### Counterexamples

`ProofCounterexampleRecord` — first-class negative result.

* Never collapsed into a positive proof evidence kind
* Stores a structured `model` plus content digest
* Lookup surfaces counterexamples separately from evidence

### Attestations

`ProofAttestationRecord` — claim over a subject with optional expiry.

* Kinds: `producer | independent_check | kernel | policy | corpus | external`
* Attestations do not silently raise authority ceilings
* Expired attestations remain stored; consumers check `is_expired`

### Lookup

`ProofLookupResult` aggregates, for one key:

* disposition: `hit | miss | stale | invalidated | environment_mismatch | rejected`
* optional plan, attempts, evidence, receipts, counterexamples, attestations
* invalidations and lineage edges for the key
* freshness report

### Freshness

`FreshnessReport` evaluates a key slot against:

1. presence of any stored record;
2. sticky invalidation;
3. TTL (`DEFAULT_FRESHNESS_TTL_SECONDS`, overridable per backend);
4. environment identity (via key admission on lookup).

`require_fresh=True` (default) converts TTL-exceeded slots into `stale`
lookups rather than hits.

### Invalidation

`ProofInvalidationRecord` is append-only.

* Reasons: `manual | stale | superseded | revoked | environment_drift | policy_change | lineage_break | tamper`
* Invalidation is **sticky** for the key slot: subsequent lookups return
  `invalidated` until a distinct key identity is used
* Active plans/attempts under the key are marked `invalidated`

### Lineage

`ProofLineageEdge` is a directed edge between record ids.

* Relations: `parent | derived_from | supersedes | attests | evidences | receipt_of | attempt_of | counterexample_of | invalidates`
* `lineage_of(id, direction=both|parents|children)` walks edges
* Optional `cache_key_id` scopes edges to a semantic key

## Protocol sketch

```text
ProofRepository@1
  interface / schema_version / backend_id / capabilities()
  put_plan / get_plan / list_plans
  put_attempt / get_attempt / list_attempts
  put_evidence / get_evidence / list_evidence
  put_receipt / get_receipt / list_receipts
  put_counterexample / get_counterexample / list_counterexamples
  put_attestation / get_attestation / list_attestations
  lookup(key, *, now=None, require_fresh=True) -> ProofLookupResult
  freshness(key, *, now=None) -> FreshnessReport
  invalidate(key, *, reason=..., ...) -> ProofInvalidationRecord
  list_invalidations(*)
  put_lineage / lineage_of / list_lineage
  stats()
```

Factory: `build_proof_repository(backend="memory", ...)`.

Reference implementation: `InMemoryProofRepository` (thread-safe, process-local,
no DuckDB/network/filesystem I/O on import).

## Rejection rules (fail-closed)

| Rejection | Trigger |
| --- | --- |
| Missing / empty ids | Empty `plan_id`, `attempt_id`, … |
| Missing cache key | Record without admissible `CanonicalProofCacheKey` |
| Empty digests | Blank or malformed `sha256:` fields |
| Candidate-as-kernel | Candidate evidence kind + kernel-grade ceiling |
| Cross-environment hit | Stored key environment ≠ request environment |
| Incomplete capabilities | Backend omits any of the ten acceptance surfaces |
| Self-lineage | `parent_id == child_id` |
| Schema drift | Unknown per-record schema version |
| Capacity | In-memory backend exceeds `max_records` |

## Relationship to other surfaces

| Surface | Role relative to this interface |
| --- | --- |
| `CanonicalProofCacheKey@1` (LPC-080) | Semantic identity for every keyed record |
| `VerificationCacheProtocol@1` | Backend-local exact verification cache; narrower |
| `DuckDBProofStore` / `DuckDBProofService` | Durable / coordinated implementations; must project here |
| `ProofCorpusDuckDBRepository` | Corpus envelopes / revocation; may feed attestations |
| Supervisor proof cache | Placement and single-flight only |

## What this does **not** do

1. **Does not** redefine LPC-080 cache-key fields in the supervisor.
2. **Does not** treat cache hits, provider success, or attestations as kernel
   trust roots.
3. **Does not** collapse counterexamples into positive proof evidence.
4. **Does not** require DuckDB at import time; DuckDB remains optional backend.
5. **Does not** clear sticky invalidation by writing more records under the
   same key_id.

## Validation coverage

`tests/unit/logic/common/test_proof_repository.py` asserts:

* interface identity `ProofRepository@1` and schema version;
* full ten-capability inventory on the reference backend;
* round-trip put/get/list for plans, attempts, evidence, receipts,
  counterexamples, and attestations;
* lookup hits aggregate the bound records;
* freshness TTL expiry yields `stale` when `require_fresh=True`;
* invalidation is sticky and marks plans/attempts;
* lineage edges are queryable by parent/child;
* candidate-as-kernel evidence is rejected;
* cross-environment lookup is rejected;
* note documents the capability inventory and protocol surface.

## Acceptance

- One interface covers plans, attempts, evidence, receipts, counterexamples,
  attestations, lookup, freshness, invalidation, and lineage.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/common/test_proof_repository.py -q`
