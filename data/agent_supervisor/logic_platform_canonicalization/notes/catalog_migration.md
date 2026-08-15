# LPC-020 Catalog migration: CanonicalLogicCatalogSnapshot

**Task:** LPC-020  
**Goal:** LPC-G020  
**Interface:** `CanonicalLogicCatalogSnapshot@1`  
**Evidence module:** `ipfs_datasets_py/logic/families/canonical_catalog.py`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/families/test_canonical_catalog.py -q`

## Summary

LPC-020 composes the existing typed logic-family layers into one immutable
snapshot with a reproducible content root. Layers are **composed, not
flattened**. Registry v2 remains the descriptor taxonomy layer; registry v3
remains the lifecycle / publication layer. There is no registry v4 rename.

## Composition map

| Snapshot field / projection | Source layer | Interface | Role |
| --- | --- | --- | --- |
| `taxonomy` | `registry.py` | `LogicFamilyRegistry@2` | Descriptor taxonomy |
| `namespaces` | `namespaces.py` | `LogicIdentityNamespaces@1` | Typed identity roles |
| `aliases` | `aliases.py` | `LogicAliasRegistry@1` | Dual-read / one-write aliases |
| `publication` | `registry_v3.py` | `LogicFamilyRegistry@3` | Lifecycle / publication |
| `profiles` | `profile_catalog_v3.py` | `LogicProfileCatalog@3` | Profile catalog |
| `providers` | `providers.py` | `ProviderCapabilityCatalog@1` | Provider capability catalog |
| `matrix` | `provider_matrix_v2.py` | `ProviderCapabilityMatrix@2` | Provider × lane matrix |
| `generated` | `generated_catalog.py` | `GeneratedProviderTranslationCatalog@1` | Generated projection |
| `properties` | namespaces + taxonomy | — | Property/obligation ids |
| `views` | namespaces | — | View-role ids |
| `notations` | namespaces | — | Source notation ids |
| `encodings` | namespaces | — | Target encoding ids |
| `lanes` | namespaces + matrix | — | Execution lane ids |
| `evidence` | namespaces + taxonomy | — | Evidence-kind ids |
| `translations` | taxonomy + generated | — | Translation edge ids |
| `versions` | all layers | — | Version envelope |
| `content_root` / `content_digest` | `ir_core.identity` | `ir-canonical-identity-v1` | Reproducible content identity |

Builder entrypoint:

* `build_canonical_logic_catalog_snapshot(...)`
* sealed default: `DEFAULT_CANONICAL_CATALOG_SNAPSHOT`

## Publication ladder

The snapshot exposes an explicit ladder from **identity-exists** through
**production-admitted**:

1. `identity_exists` — present in the namespace vocabulary only  
2. `declared` — present in taxonomy and/or declaration-only publication  
3. `discoverable` — reserved (not auto-assigned by composition)  
4. `parse_print` — registry v3 `parse_print` disposition  
5. `controlled_executable` — registry v3 `controlled_executable` with explicit features  
6. `shadow` / `canary` — reserved operational stages  
7. `production_admitted` — **never inferred from catalog presence**

`is_production_admitted(family_id)` always returns `False` for composed
defaults. `presence_implies_executability()` and
`presence_implies_production_admission()` are hard-zero safety floors.

## Invariants preserved

1. Declaration never implies executability.  
2. Catalog presence never implies production admission.  
3. Provider / matrix presence never claims live availability or proof authority.  
4. v2 taxonomy and v3 lifecycle stay distinct typed objects.  
5. Namespace roles (family, profile, property, view, notation, encoding,
   provider, lane, evidence) remain non-interchangeable.  
6. Generated catalog is a projection, not a second hand-written authority.  
7. Content root is deterministic: recomposing the default layers yields the
   same CIDv1 / `sha256:` digest.

## What this task does **not** do

* Does not delete registry v2 or v3.  
* Does not flatten typed layers into one untyped dictionary.  
* Does not introduce registry v4.  
* Does not implement LPC-021 drift tests (aliases, authority ceilings, etc.
  beyond composition integrity).  
* Does not rewrite supervisor maps (LPC-G090) or package manifests (LPC-G100).

## Downstream consumers

| Consumer | Expectation |
| --- | --- |
| LPC-021 catalog drift tests | Import the snapshot; fail closed on layer drift and content-root changes |
| LPC-022 registry role docs | Cite v2 = taxonomy, v3 = lifecycle, snapshot = composition |
| LPC-023 generated catalog | Remains a projection under the snapshot, not a second authority |
| LPC-G090 / supervisor maps | Project onto snapshot identities; do not redefine them |
| LPC-G100 package manifest | May bind `catalog root` to `snapshot.content_root` |

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/families/canonical_catalog.py` | Composition root implementation |
| `ipfs_datasets_py/tests/unit/logic/families/test_canonical_catalog.py` | Regression contract tests |
| `data/agent_supervisor/logic_platform_canonicalization/notes/catalog_migration.md` | This migration note |
