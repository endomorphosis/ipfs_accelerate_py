# LPC-023 Generated catalog projections

**Task:** LPC-023  
**Goal:** LPC-G020  
**Interface:** `GeneratedProviderTranslationCatalog@1`  
**Evidence module:** `ipfs_datasets_py/logic/families/generated_catalog.py`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/families/test_generated_catalog.py -q`

## Summary

LPC-023 treats the provider/translation catalog as a **generated projection**
over sealed source declarations, not as a second hand-written inventory.
`build_generated_provider_translation_catalog()` projects:

1. **Providers** from `ProviderCapabilityCatalog@1` baseline entries
   (`providers.py` / `BASELINE_PROVIDER_CATALOG`), joined with the executable
   provider matrix ids and reviewed aliases from
   `ipfs_datasets_py.logic.backends.registry`.
2. **Translations** from every reviewed edge on
   `LogicFamilyRegistry@2` (`registry.py` / `DEFAULT_REGISTRY.translations`).

The sealed default is `DEFAULT_GENERATED_CATALOG`. Divergence from those source
layers is a **test failure**, not a silent second authority. Catalog presence
never means live tool availability or proof authority.

## Projection map

| Projection field | Source declaration | Authority |
| --- | --- | --- |
| `providers` / `provider_ids` | Baseline provider catalog entries (`BASELINE_PROVIDER_IDS` ∪ `EXECUTABLE_PROVIDER_IDS`) | `ProviderCapabilityCatalog@1` |
| `executable_matrix_provider_ids` | `EXECUTABLE_PROVIDER_MATRIX` / `EXECUTABLE_PROVIDER_IDS` | Backend executable matrix |
| `reviewed_aliases` | Per-entry aliases + `EXECUTABLE_PROVIDER_ALIASES` | Baseline + matrix join |
| `translations` / `translation_ids` | `DEFAULT_REGISTRY.translations` | `LogicFamilyRegistry@2` |
| Family referents on edges | `BASELINE_FAMILY_IDS` ∪ registry families | Taxonomy |
| Snapshot slot `generated` | Same sealed projection | `CanonicalLogicCatalogSnapshot@1` |

Builder entrypoint:

* `build_generated_provider_translation_catalog(registry=..., baseline=..., validate=True)`
* sealed default: `DEFAULT_GENERATED_CATALOG`

## Invariants

1. **No second hand inventory.** Provider and translation rows are projected
   from baseline + registry; the generated module does not maintain a parallel
   free-form id list that can drift independently.
2. **Exact provider closure.** Projected provider ids equal
   `BASELINE_PROVIDER_IDS ∪ EXECUTABLE_PROVIDER_IDS`. Missing or extra ids fail
   `validate_closure()`.
3. **Executable-matrix join.** `executable_matrix_provider_ids` equals the
   sealed matrix id set; alias resolution matches `EXECUTABLE_PROVIDER_ALIASES`.
4. **Translation coverage.** Every registry translation id appears in the
   projection with matching source/target family and kind. Dropping a registry
   edge fails closed.
5. **Baseline source preserved.** Baseline provider ids remain
   `ProviderCatalogSource.BASELINE`; re-sourcing them as generated-closure fails.
6. **Closed projection.** `generated_closure_open` must be `False` on the final
   catalog.
7. **Presence ≠ availability / proof.** `claims_availability` and
   `claims_proof` are hard-zero for every projected provider.
8. **Duplicates / unknown referents rejected.** Duplicate provider or
   translation ids, and translation edges that name unknown families, fail.

## Acceptance matrix

| Check | Fail-closed behavior | Primary APIs |
| --- | --- | --- |
| Provider parity | Generated provider set and baseline-sourced rows match source catalog | `build_generated_provider_translation_catalog`, `BASELINE_PROVIDER_CATALOG` |
| Matrix join | Executable ids and reviewed aliases match backend matrix | `executable_matrix_provider_ids`, `reviewed_aliases` |
| Translation parity | Registry translation edges project field-for-field | `GeneratedTranslationEdge`, `DEFAULT_REGISTRY.translations` |
| Hand-crafted drift | Missing/extra providers, dropped translations, open closure, baseline overwrite | `validate_closure`, `GeneratedCatalogError` family |
| Snapshot composition | Canonical snapshot holds the same sealed projection | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.generated` |
| Non-authority | Presence never upgrades to availability or proof | `claims_availability`, `claims_proof`, `is_eager` |

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-020 catalog migration | Composes `generated` as one typed layer under `CanonicalLogicCatalogSnapshot@1` |
| LPC-021 catalog drift tests | Broader layer misuse; LPC-023 owns projection-vs-source parity |
| LPC-022 registry roles | Documents v2 taxonomy / v3 lifecycle; generated stays a projection |
| LPC-G090 supervisor maps | May consume the projection; must not invent a second inventory |

## What this task does **not** do

* Does not rewrite baseline providers, registry translations, or the executable
  matrix as free-form hand lists inside the generated module.
* Does not claim live prover availability or production admission from catalog
  presence.
* Does not introduce registry v4 or flatten typed layers.
* Does not implement supervisor map generation (LPC-G090) or package manifests
  (LPC-G100).

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/families/generated_catalog.py` | Projection implementation (`GeneratedProviderTranslationCatalog@1`) |
| `ipfs_datasets_py/tests/unit/logic/families/test_generated_catalog.py` | Projection parity / drift regression suite |
| `data/agent_supervisor/logic_platform_canonicalization/notes/generated_catalogs.md` | This note |

## Downstream consumers

| Consumer | Expectation |
| --- | --- |
| `CanonicalLogicCatalogSnapshot@1` | Binds `generated` to the sealed projection |
| LPC-G090 supervisor maps | Project onto generated ids; do not re-list providers/translations by hand |
| Conformance / closure suites | May restate the same fail-closed invariants |
| Package / content-root binding | Treats generated content as derived, not authoritative source |
