# LPC-080 Canonical Semantic Cache-Key Contract

**Task:** LPC-080 — Canonical semantic cache-key contract  
**Goal:** LPC-G080  
**Depends on:** LPC-032 (success is not proof), LPC-052 (typed provider responses)  
**Interface:** `CanonicalProofCacheKey@1`  
**Module:** `ipfs_datasets_py.logic.common.canonical_cache_key`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/common/canonical_cache_key.py`  
**Schema:** `ipfs_datasets_py/canonical-proof-cache-key@1`  
**Schema version:** `canonical-proof-cache-key/v1`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/common/test_canonical_cache_key.py -q`

## Purpose

Proof-cache identity must bind every semantic dimension that can change a
result. Datasets owns that contract. The supervisor may own **placement** and
**single-flight** only; it must not invent a second key vocabulary or silently
promote candidate evidence into kernel authority.

This note freezes `CanonicalProofCacheKey@1`. Executable coverage lives in
`ipfs_datasets_py/tests/unit/logic/common/test_canonical_cache_key.py`.

## Ownership

| Owner | Responsibility |
| --- | --- |
| **Datasets** (`canonical_cache_key.py`) | Semantic fields, digests, CID validity, candidate-as-kernel rejection, environment binding |
| **Supervisor** (`ProofCacheKey` / formal-verification cache) | Placement, TTL, single-flight, projection onto this contract |
| **Verification cache protocol** (`VerificationCacheProtocol@1`) | Backend-local exact-cache shape; must not redefine LPC-G080 fields |

Live surfaces that remain **compatibility / local** only:

* `logic/common/proof_cache.py` — unified prover result cache (CID hashing)
* `logic/backends/cache_protocol.py` — verification-cache key/entry protocol
* supervisor `ProofCacheKey` — projects onto datasets identity; does not redefine it

## Required identity fields

Every admitted `CanonicalProofCacheKey` body binds **all** of the following
fields (LPC-G080 / LPC-080 acceptance). Missing fields fail closed at
construction or admission.

| Field | Carrier | What it binds |
| --- | --- | --- |
| `source` | `sha256:<hex>` digest | Source document / snapshot identity |
| `expression` | digest | Typed expression / obligation surface form |
| `formalization` | digest | Formalization artifact identity |
| `slice` | digest | `DomainLogicSlice@2` (or equivalent) identity |
| `obligation` | digest | Proof obligation identity |
| `assumptions` | digest | Closed assumption set |
| `bounds` | digest | Semantic / resource bounds identity |
| `translation` | digest | Translation chain / receipt identity |
| `provider` | stable id (or valid CID) | Provider product identity |
| `environment` | digest | Toolchain / environment identity (cross-env hits fail) |
| `policy` | digest | Admission / proof policy identity |
| `schema` | digest | Schema / IR schema binding |
| `checker` | stable id (or valid CID) | Checker / kernel / solver product identity |
| `network_policy` | digest | Network / outbound policy binding |
| `evidence_kind` | `LogicEvidenceKind` | Format/category of cached evidence |
| `authority_ceiling` | `LogicEvidenceAuthority` | Trust ceiling claimed for the entry |

Closed constant: `REQUIRED_IDENTITY_FIELDS` in `canonical_cache_key.py` lists
these sixteen fields.

Additional identity metadata (always present on wire projection):

| Field | Value |
| --- | --- |
| `interface` | `CanonicalProofCacheKey@1` |
| `schema_version` | `canonical-proof-cache-key/v1` |
| `key_id` | `canonical-proof-cache-key:sha256:…` (derived) |
| `source_cid` | optional; when set must be a valid CIDv1 |

## Rejection rules (fail-closed)

| Rejection | Trigger | Error |
| --- | --- | --- |
| **Missing semantic fields** | Any required identity field absent or empty | `CanonicalCacheKeyError` |
| **Empty digests** | `""`, whitespace, or `sha256:` with no hex | `EmptyDigestError` |
| **Invalid / CID-looking non-CIDs** | Value looks like a CID (`b…` base32, synthetic `bafy…` hex keys, broken `Qm…`) but is not a structurally valid CIDv1 | `InvalidCidError` |
| **Default-string unknown objects** | Placeholders such as `unknown`, `<unknown>`, `null`, `unspecified`, `default` used as stable ids | `CanonicalCacheKeyError` |
| **Candidate-as-kernel** | `evidence_kind` ∈ candidate family **and** `authority_ceiling` ∈ `{authoritative, independently_checkable}` | `CandidateAsKernelError` |
| **Cross-environment hits** | Stored key environment digest ≠ request environment digest | `CrossEnvironmentHitError` |

### Candidate family (cannot claim kernel ceiling)

`candidate`, `atp_candidate`, `smt_candidate`, `llm_output`, `model_output`,
`declaration`, `review`.

### Kernel-grade authority ceilings

`authoritative`, `independently_checkable`.

Candidate evidence may still be cached under `advisory` / `bounded` / `none` /
`unknown` ceilings. Kernel-grade ceilings require non-candidate evidence kinds
(for example `kernel_checked_proof`, `checked_proof`, `proof_certificate`).

### CID validation (dependency-free)

Validation does **not** import optional multiformats packages. A CIDv1 string
must:

1. Match multibase base32 text form (`b` + RFC4648 base32 body);
2. Decode to a complete multiformat stream;
3. Carry CIDv1 (`version == 1`) and a well-formed multicodec + multihash.

Synthetic HuggingFace-style keys (`bafy` + truncated hex) and other CID-shaped
impostors fail this check and are rejected.

Digest slots require `sha256:` + 64 lowercase hex digits (bare 64-hex is
normalized). Empty digests never normalize to a valid identity.

## Cache is not a trust root

1. A cache hit re-derives assurance from bound evidence; it does not raise
   authority (LPC-032).
2. Provider success on a prior attempt does not mint kernel authority for a hit
   (LPC-032 / LPC-052 untrusted defaults).
3. Supervisor single-flight outcomes that are non-authoritative must not be
   stored as kernel-grade cache entries.
4. Cross-environment reuse is never a hit, even when obligation digests match.

## Admission helpers

| Symbol | Role |
| --- | --- |
| `CanonicalProofCacheKey` | Frozen key type; validates on construction |
| `CanonicalProofCacheKey.build` | Digest raw values into a key |
| `CanonicalProofCacheKey.from_dict` / `to_dict` | Strict JSON projection |
| `admit_canonical_cache_key` | Admit typed body or mapping; revalidates all fields |
| `admit_cache_hit` | Exact identity match + environment equality |
| `key_carries_required_identity_fields` | Inventory check for the sixteen fields |
| `reject_candidate_as_kernel` | Explicit candidate-as-kernel gate |
| `require_digest` / `require_valid_cid` | Primitive validators |
| `environments_compatible` | Environment digest comparison |

## Relationship to other key shapes

| Surface | Role relative to this contract |
| --- | --- |
| Supervisor `ProofCacheKey` | Compatibility facade; project onto / restore from datasets fields |
| `VerificationCacheKey` | Backend verification exact-cache; narrower dimension set |
| Family-local proof caches | Local optimization only; must not redefine LPC-G080 semantics |

Supervisor live fields (`obligation`, `premises`, `translator`, `solver`,
`kernel`, `toolchain`, `theorem_registry`, `policy`, `resource_budget`,
`candidate_tree`) must map into the datasets inventory above rather than
standing as a second authority.

## What this does **not** do

1. **Does not** implement the unified proof repository interface (LPC-081).
2. **Does not** own supervisor placement, TTL, or single-flight.
3. **Does not** promote provider success or cache hits into proof authority.
4. **Does not** accept synthetic `bafy…` cache keys as multiformats CIDs.
5. **Does not** allow candidate evidence to be stored under kernel-grade
   authority ceilings.

## Validation coverage

`tests/unit/logic/common/test_canonical_cache_key.py` asserts:

* interface identity `CanonicalProofCacheKey@1` and schema version;
* full required identity-field inventory on every constructed key;
* empty digests rejected;
* CID-looking non-CIDs (synthetic `bafy…` hex keys, truncated/broken strings)
  rejected;
* candidate-as-kernel pairings rejected;
* cross-environment hits rejected;
* valid keys bind all sixteen fields and round-trip through dict admission;
* note documents the field inventory and rejection rules.

## Acceptance

- Keys bind the required identity fields (source, expression, formalization,
  slice, obligation, assumptions, bounds, translation, provider, environment,
  policy, schema, checker, network policy, evidence kind, authority ceiling).
- Invalid CIDs, empty digests, and candidate-as-kernel entries are rejected.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/common/test_canonical_cache_key.py -q`
