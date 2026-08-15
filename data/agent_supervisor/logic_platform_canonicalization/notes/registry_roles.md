# LPC-022 Registry v2/v3 semantic roles

**Task:** LPC-022  
**Goal:** LPC-G020  
**Depends on:** LPC-020 (`CanonicalLogicCatalogSnapshot@1`)  
**Conflict policy:** Own this role note only.  
**Validation:** `test -f data/agent_supervisor/logic_platform_canonicalization/notes/registry_roles.md`

## Decision (authoritative)

| Layer | Module | Interface | Semantic role |
| --- | --- | --- | --- |
| Registry v2 | `ipfs_datasets_py/logic/families/registry.py` | `LogicFamilyRegistry@2` | **Descriptor taxonomy** |
| Registry v3 | `ipfs_datasets_py/logic/families/registry_v3.py` | `LogicFamilyRegistry@3` | **Lifecycle / publication** |
| Snapshot | `ipfs_datasets_py/logic/families/canonical_catalog.py` | `CanonicalLogicCatalogSnapshot@1` | **Composition root** |

**Arrangement retained:** v2 = taxonomy, v3 = lifecycle, snapshot = composition.  
**No registry v4 rename.** Generations stay distinct typed objects. A new
generation would require a genuine wire-format migration; renaming the current
pair for cosmetic uniformity is out of scope and rejected.

No tested better arrangement replaces this split. LPC-020 composition and
LPC-021 drift tests treat the two registries as separate sealed layers under
the snapshot. Flattening them into one untyped dictionary, deleting either
generation, or inventing a v4 alias for the same wire content is forbidden.

## Why two registries exist

The platform needs two orthogonal answers for every family identity:

1. **What is it?** — vocabulary, descriptors, and declared relationships
   (taxonomy / v2).
2. **How far is it admitted?** — lifecycle disposition, explicit executable
   features, and authority ceiling (publication / v3).

Collapsing those answers into one structure reintroduces the historical failure
mode where catalog presence is misread as executability or production
admission. Keeping them separate makes the safety floors mechanical:

* Declaration never implies executability.
* Registry presence never implies production admission.
* Provider success never implies semantic success.

## Registry v2 — descriptor taxonomy

| Axis | Value |
| --- | --- |
| Path | `ipfs_datasets_py/logic/families/registry.py` |
| Interface | `LogicFamilyRegistry@2` |
| Module version | `2.0.0` (`REGISTRY_VERSION`) |
| Schema | `TAXONOMY_SCHEMA_VERSION` from `models.py` |
| Snapshot field | `taxonomy: LogicFamilyRegistry` |
| Semantic role | `canonical` descriptor taxonomy |
| Side effects | None. Does not import providers, installers, solvers, or process runners. |

### Owns

Closed, validated descriptor sets and their internal references:

* families (`LogicFamilyDescriptor`)
* fragments
* properties / obligations
* operations
* runtimes
* evidence kinds
* boundedness
* translations
* provider capability declarations (taxonomy-level capability rows)

Also owns sealed baseline family partitions used by later layers:

* `FOUNDATION_FAMILY_IDS`
* `PLANNED_EXTENSION_FAMILY_IDS`
* `DECLARATION_ONLY_FAMILY_IDS`
* `NON_FAMILY_PROFILE_LABELS` (labels that must remain profiles/aliases, never families)

### Validates

* Unique descriptor identifiers (no silent reuse).
* Alias collision rejection (`AliasCollisionError`).
* Declared semantic equivalence only (no silent merge).
* Cross-descriptor references resolve at registration time.
* Provider capability rows do not contradict the taxonomy.

### Does **not** own

* Lifecycle disposition (`declaration_only` / `parse_print` / `controlled_executable`).
* Explicit executable feature authorization for publication.
* Authority ceilings on the publication ladder.
* Production admission.
* Live toolchain / PATH / prover availability.

v2 answers “this identity exists in the vocabulary and these descriptors hang
together.” It does not answer “this identity may execute” or “this identity is
production-admitted.”

## Registry v3 — lifecycle / publication

| Axis | Value |
| --- | --- |
| Path | `ipfs_datasets_py/logic/families/registry_v3.py` |
| Interface | `LogicFamilyRegistry@3` |
| Class | `LogicFamilyRegistryV3` |
| Schema | `logic-family-registry/v3` |
| Entry schema | `logic-family-publication-entry/v3` |
| Module version | `3.0.0` |
| Snapshot field | `publication: LogicFamilyRegistryV3` |
| Semantic role | `canonical` lifecycle / publication |
| Side effects | None. Does not import parsers, installers, solvers, or process runners. |

### Owns

Per-family publication entries (`FamilyPublicationEntry`):

* `family_id`, `task_id`, `name`
* `disposition` (`FamilyLifecycleDisposition`)
* `profile_ids`, `feature_ids`, `executable_features`
* optional `parser_module`, `notation_id`, `semantic_identity`
* `authority_ceiling` (default `advisory`)
* publication aliases and baseline cross-checks

### Lifecycle dispositions

| Disposition | Meaning | Executable features |
| --- | --- | --- |
| `declaration_only` | Identity published without execution claim | Must be empty |
| `parse_print` | Controlled parse/print (and closely related) surface | Optional; subset of parse/print/elaborate/source_map/evaluate; requires declared `parse` |
| `controlled_executable` | Explicit controlled execution | Must list non-empty `executable_features` ⊆ `feature_ids` |

`FamilyPublicationEntry.is_executable` is true only when disposition and
explicit features authorize execution. Registry presence alone is never enough.

### Validates against v2

`LogicFamilyRegistryV3.validate_against_baseline(taxonomy)` fail-closes when a
publication entry contradicts baseline taxonomy rules. Publication does not
redefine the vocabulary; it stages identities already (or jointly) present in
the taxonomy layer.

### Does **not** own

* The full descriptor taxonomy (fragments, properties, operations, etc.).
* Generated catalog projections.
* Supervisor adapter maps.
* Package-manifest binding of catalog root.
* Production admission inference (always false for composed defaults).

## Composition root (not a third registry generation)

`CanonicalLogicCatalogSnapshot@1` composes both registries with the other typed
layers. It is **not** registry v4 and must not be renamed into one.

```
models (typed descriptors)
  + namespaces + aliases
  + registry v2  → snapshot.taxonomy
  + registry v3  → snapshot.publication
  + profile_catalog_v3
  + provider_matrix_v2 / providers
  + generated_catalog (projection)
  → CanonicalLogicCatalogSnapshot@1
```

| Snapshot concern | Source |
| --- | --- |
| Descriptor vocabulary | v2 taxonomy |
| Publication ladder / stages | v3 lifecycle |
| Identity namespaces and aliases | `namespaces` / `aliases` |
| Profiles and provider matrix | `profile_catalog_v3` / `provider_matrix_v2` |
| Generated views | `generated_catalog` (projection only) |
| Content root / digest | composition envelope (`ir-canonical-identity-v1`) |

Safety floors on the snapshot:

* `presence_implies_executability()` → hard zero
* `presence_implies_production_admission()` → hard zero
* `is_production_admitted(family_id)` → always `False` for composed defaults

## Publication ladder (snapshot view of v3)

| Stage | How it is reached |
| --- | --- |
| `identity_exists` | Present in namespace vocabulary only |
| `declared` | Present in taxonomy and/or declaration-only publication |
| `discoverable` | Reserved (not auto-assigned by composition) |
| `parse_print` | v3 disposition `parse_print` |
| `controlled_executable` | v3 disposition `controlled_executable` with explicit features |
| `shadow` / `canary` | Reserved operational stages |
| `production_admitted` | **Never inferred from catalog presence** |

v2 can place a family on the lower rungs of identity/declaration. Only v3 (plus
explicit operational policy outside this note) can raise a family into
parse/print or controlled execution. Production admission remains outside both
registry generations.

## Explicit non-goals

| Non-goal | Rationale |
| --- | --- |
| Registry v4 rename | No wire-format migration; cosmetic renames hide the taxonomy/lifecycle split |
| Flatten v2 + v3 into one dict | Loses typed fail-closed boundaries proven by LPC-020/021 |
| Delete v2 or v3 | Both remain canonical components of the snapshot |
| Treat generated catalog as authority | Generated surfaces are projections (LPC-023) |
| Infer executability from presence | Contradicts module contracts and drift tests |
| Infer production admission from presence | Snapshot hard floor |

## Arrangement evaluation (no better alternative recorded)

| Candidate | Status | Why rejected / deferred |
| --- | --- | --- |
| Keep v2 taxonomy + v3 lifecycle under snapshot | **Accepted** | Matches sealed modules, LPC-020 composition, LPC-021 drift tests, and plan epic A |
| Merge into single “registry v4” | Rejected | No wire-format migration; would flatten typed layers and invite presence→executability bugs |
| Collapse lifecycle into taxonomy descriptors | Rejected | Mixes vocabulary with admission; breaks disposition/feature ceilings |
| Collapse taxonomy into publication entries | Rejected | Publication entries are not a full descriptor graph |
| Snapshot-only single structure | Rejected | Snapshot is composition, not a replacement generation |

If a future task proposes a different arrangement, it must land as a tested
change with explicit migration evidence and update this note. Until then, the
accepted arrangement is the production contract.

## Invariants (fail-closed)

1. v2 taxonomy and v3 lifecycle stay distinct typed objects.  
2. Snapshot composes layers; it does not replace either generation.  
3. No registry v4 unless a genuine wire-format migration is required.  
4. Declaration never implies executability.  
5. Catalog presence never implies production admission.  
6. `executable_features` ⊆ `feature_ids`; declaration-only forbids executables.  
7. Aliases cannot silently merge distinct identities.  
8. Generated catalogs project; they do not author semantics.  
9. Supervisor adapters remain facades over datasets meaning.  
10. Drift that violates the above fails closed (LPC-021).

## Cross-references

| Artifact | Relationship |
| --- | --- |
| LPC-003 `inventory/registries.md` | Classifies surfaces; defers role prose to this note |
| LPC-020 `notes/catalog_migration.md` | Composition map; cites v2 taxonomy / v3 lifecycle |
| LPC-021 `notes/catalog_drift_tests.md` | Fail-closed tests that protect these roles |
| LPC-023 generated catalogs | Projection under the snapshot, not a registry generation |
| Plan epic A | “Compose, do not flatten. v2 = taxonomy, v3 = lifecycle, snapshot = composition.” |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/registry_roles.md` | This LPC-022 role note (sole declared output) |

## Downstream consumers

| Consumer | Expectation |
| --- | --- |
| LPC-023 generated catalog notes | Cite v2/v3 roles; keep generated surfaces non-authoritative |
| LPC-G090 supervisor maps | Project onto snapshot identities; do not redefine taxonomy or lifecycle |
| LPC-G100 package manifest | Bind catalog root to snapshot content identity, not a renamed registry generation |
| Future registry work | Treat a v4 only as a real wire migration with tests and an update here |
