# LPC-100 LogicPlatformManifest@1 and handshake

**Task:** LPC-100  
**Goal:** LPC-G100  
**Interface:** `LogicPlatformManifest@1`  
**Evidence module:** `ipfs_datasets_py/logic/platform/manifest.py`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py -q`

## Summary

LPC-100 replaces Git-layout / sibling-checkout checks with a **package-neutral
manifest** that supervisors can handshake against from installed wheels.
Semantic compatibility is decided from declared package identity, interface
versions, catalog content root, schema roots, operation versions, receipt/plan
versions, and compatible adapter versions. **Git remains optional provenance
only** and is never required for a successful default handshake.

## Surface

| Symbol | Role |
| --- | --- |
| `LogicPlatformManifest` | Immutable package-neutral identity envelope |
| `build_logic_platform_manifest(...)` | Construct from installed package + catalog snapshot |
| `handshake(requirements, *, manifest=...)` | Typed compatibility check |
| `HandshakeRequirements` | Caller-declared constraints |
| `HandshakeResult` | `compatible` flag + typed `ManifestIncompatibility` list |
| `DEFAULT_LOGIC_PLATFORM_MANIFEST` | Sealed default built at import |

Interface / schema constants:

* `LogicPlatformManifest@1`
* `logic-platform-manifest/v1`
* `logic-platform-handshake-result/v1`
* Task / goal binding: `LPC-100` / `LPC-G100`

## Manifest fields

| Field | Authority / source | Notes |
| --- | --- | --- |
| `package_name` / `package_version` | `importlib.metadata` → `ipfs_datasets_py.__version__` → fallback | No Git |
| `interface_versions` | Declared platform interfaces (catalog, formalization, obligation, request, plan, provider protocol, self) | Exact string map |
| `catalog_root` / `catalog_digest` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT` content identity | CIDv1 + `sha256:` |
| `schema_roots` | Schema ids for catalog, artifacts, obligations, requests, plans, provider wire | Exact string map |
| `operation_versions` | handshake, catalog, capability/translate/prove/reconstruct/verify/attest | Exact string map |
| `receipt_versions` / `plan_versions` | Receipt and plan schema ids | Exact string map |
| `compatible_adapter_versions` | `SupervisorCanonicalLogicAdapter@1`, `SupervisorLogicPlatformClient@1` | Closed list |
| `source_commit` | Optional env / package metadata / explicit override | **Never required by default** |

Hard safety floors on the envelope:

* `requires_git() → False`
* `requires_sibling_repos() → False`
* `requires_repository_layout() → False`

## Handshake rules

1. **Default success path.** `handshake()` with no requirements (or empty
   requirements) against the installed default manifest is compatible even when
   `source_commit` is `None`.
2. **Typed incompatibility.** Version / identity mismatches return
   `HandshakeResult(compatible=False, incompatibilities=(...))` with a closed
   `IncompatibilityCode` vocabulary. They do **not** raise.
3. **Structural errors still fail closed.** Malformed manifests or requirements
   raise `LogicPlatformManifestError` (empty ids, non-`sha256:` digest, schema
   drift, duplicate map keys).
4. **Git optional.** `require_source_commit=True` without provenance yields
   `source_commit_required`. A wrong commit yields `source_commit_mismatch`.
   Neither path consults `.git`, `git` CLI, or sibling checkouts.
5. **Catalog root is content identity.** Callers may pin
   `required_catalog_root` / `required_catalog_digest` to the catalog snapshot
   content identity; layout paths are never compared.
6. **Adapters are exact.** Required adapter versions must appear in
   `compatible_adapter_versions`; unknown adapters are typed failures.

### Incompatibility codes

| Code | Meaning |
| --- | --- |
| `manifest_interface` | Manifest interface id mismatch |
| `package_name` / `package_version` | Package identity / version constraint failed |
| `interface_version` | Required interface version missing or wrong |
| `catalog_root` / `catalog_digest` | Catalog content identity mismatch |
| `schema_root` | Schema root map mismatch |
| `operation_version` | Operation version map mismatch |
| `receipt_version` / `plan_version` | Receipt or plan version map mismatch |
| `adapter_version` | Required adapter not listed as compatible |
| `source_commit_required` / `source_commit_mismatch` | Optional provenance demanded or mismatched |

## Provenance resolution (no Git)

`optional_source_commit()` order (first hit wins):

1. Explicit caller value  
2. Environment: `LOGIC_PLATFORM_SOURCE_COMMIT`, `IPFS_DATASETS_SOURCE_COMMIT`,
   `IPFS_DATASETS_PY_SOURCE_COMMIT`  
3. Distribution metadata fields `Source-Commit` / `Git-Commit` / `Vcs-Commit`  

If none are present, provenance is `None`. The resolver never walks the
filesystem for `.git`, never shells out to `git`, and never inspects sibling
repositories.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-020 / LPC-023 catalog | Catalog root / digest come from `CanonicalLogicCatalogSnapshot@1` |
| LPC-G090 supervisor maps | Adapter ids listed as compatible; maps still owned by LPC-090 |
| LPC-110 supervisor client | Consumes `handshake()` as the first lazy client step |
| LPC-G150 packaging | Wheel / no-sibling / no-Git install lanes exercise this surface |

## What this task does **not** do

* Does not implement `SupervisorLogicPlatformClient` (LPC-110).  
* Does not rewrite supervisor maps (LPC-090).  
* Does not move `verification_api` internals (LPC-060).  
* Does not treat Git alignment, monorepo adjacency, or checkout layout as
  semantic compatibility authority.  
* Does not claim provider availability or proof authority from catalog presence.

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/manifest.py` | `LogicPlatformManifest@1` + handshake |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/__init__.py` | Package exports |
| `ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py` | Wheel / no-Git / typed-incompatibility regression suite |
| `data/agent_supervisor/logic_platform_canonicalization/notes/manifest_handshake.md` | This note |

## Acceptance matrix

| Check | Fail-closed behavior | Primary APIs |
| --- | --- | --- |
| Wheel / no-sibling / no-Git | Default handshake compatible without commit or layout | `build_logic_platform_manifest`, `handshake` |
| Optional provenance | Env/metadata only; absence is not failure | `optional_source_commit` |
| Interface / adapter / catalog pin | Typed `HandshakeResult` incompatibilities | `HandshakeRequirements`, `IncompatibilityCode` |
| Structural validity | `LogicPlatformManifestError` on bad digests/ids | `LogicPlatformManifest` |
| Catalog identity | Root/digest match sealed snapshot | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT` |
