# LPC-021 Catalog drift tests

**Task:** LPC-021  
**Goal:** LPC-G020  
**Depends on:** LPC-020 (`CanonicalLogicCatalogSnapshot@1`)  
**Evidence module:** `ipfs_datasets_py/tests/unit/logic/families/test_catalog_drift.py`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/families/test_catalog_drift.py -q`

## Summary

LPC-021 adds fail-closed regression coverage for catalog identity drift. Tests
import the sealed composition root from LPC-020 and assert that alias misuse,
namespace coercion, broken profile/family references, illegal provider
operations, executable-vs-declared feature inflation, authority-ceiling
overclaim, and content-root mutation all raise rather than silently admit.

The drift suite does **not** probe live toolchains, PATH, or installed provers.
Catalog presence remains declaration-only; production admission is never
inferred.

## Acceptance matrix

| Surface | Fail-closed behavior covered | Primary APIs |
| --- | --- | --- |
| Aliases | Unknown labels, wrong-namespace dual-read, frozen sealed registry, alias collisions, non-canonical write rejection | `snapshot.aliases`, `LogicAliasRegistry` |
| Namespace coercion | Cross-role `require` / `coerce` / `as_namespace`, same surface string across roles, unknown role-scoped resolve | `snapshot.namespaces`, `LogicIdentity` |
| Profile/family references | Profile family mismatch, missing registry profile, task mismatch, unpublished family host | `LogicProfileCatalogV3.validate_against_registry`, snapshot builder |
| Provider operations | Operations outside family operation set, unknown family, non-family profile promotion, declaration-only executable claim | `ProviderCapabilityEntry`, taxonomy `validate_provider_capability` |
| Executable vs declared | Declaration-only cannot list executable features; executable features ⊆ declared; controlled_executable requires explicit list; declared ≠ executable | `FamilyPublicationEntry`, `ProfileCatalogEntryV3`, snapshot claims |
| Authority ceilings | Advisory provider hard ceilings, authoritative evidence ban on advisory lanes, closed ceiling identifiers, unknown family stage queries | `ProviderCapabilityEntry`, publication/profile ceilings |
| Catalog-root reproducibility | Rebuild identity stability; envelope mutation changes CID/digest; notes drift diverges root without weakening integrity floors | `content_root`, `content_digest`, `canonical_identity` |

## Composition integrity

End-to-end checks through `build_canonical_logic_catalog_snapshot(validate=True)`:

1. Default sealed snapshot validates cleanly.  
2. Drifted profile layer (family rebinding) cannot seal.  
3. Missing profile referenced by publication cannot seal.  
4. `presence_implies_executability()` and `presence_implies_production_admission()` remain hard-zero.  
5. Provider catalog ↔ executable matrix join and taxonomy validation stay closed on the snapshot.

## Relationship to LPC-020

| LPC-020 | LPC-021 |
| --- | --- |
| Composes typed layers into `CanonicalLogicCatalogSnapshot@1` | Imports that snapshot as the drift oracle |
| Proves composition completeness and ladder semantics | Proves misuse and layer contradiction fail closed |
| Content root is reproducible for defaults | Content root changes when layer content drifts |

## What this task does **not** do

* Does not rewrite catalog production modules (aliases, namespaces, registries,
  providers, matrix, snapshot).  
* Does not introduce registry v4 or flatten typed layers.  
* Does not claim live provider availability or proof authority.  
* Does not implement supervisor map generation (LPC-G090) or package manifests
  (LPC-G100).

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/tests/unit/logic/families/test_catalog_drift.py` | Fail-closed drift regression suite |
| `data/agent_supervisor/logic_platform_canonicalization/notes/catalog_drift_tests.md` | This note |

## Downstream consumers

| Consumer | Expectation |
| --- | --- |
| LPC-022 registry role docs | May cite these fail-closed invariants for v2/v3 roles |
| LPC-023 generated catalog | Divergence from source layers remains a test failure |
| LPC-G090 supervisor maps | Unknown values must fail closed the same way |
| LPC-G100 package manifest | Catalog root binding can rely on reproducible `content_root` |
