# LPC-003 Registry, namespace, alias, and catalog inventory

**Task:** LPC-003  
**Goal:** LPC-G010  
**Interface:** `LogicPlatformInventory@1`  
**Machine-readable companion:** `registries.json`  
**Inspected:** 2026-08-15  

This inventory classifies every required registry generation and related
catalog surface with a semantic role. It is read-only against production
`logic/families` sources and writes only inventory artifacts.

## Authority and source probe

| Axis | Value |
| --- | --- |
| Datasets reviewed baseline | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| Datasets current head (authority) | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| Datasets declared checkout | `/home/barberb/lift_coding/external/ipfs_datasets` |
| Nested submodule slot | `ipfs_datasets_py/` (empty / unpopulated in this worktree) |
| Accelerator reviewed baseline | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` |
| Accelerator current head (authority) | `ea11293bb996f052d620eae989f5377a956764b1` |

Current heads remain implementation authority. In this worktree the external
datasets checkout and nested submodule were not populated, so classifications
are grounded in sealed plan contracts, LFV completion evidence, and supervisor
bindings rather than silent omission of required surfaces.

## Semantic roles

Allowed roles from LPC-G010:

| Role | Meaning in this slice |
| --- | --- |
| `canonical` | Datasets-owned semantic authority for a generation or composition root |
| `canonical_component` | Required typed layer that composes into a canonical surface |
| `compatibility_facade` | Supervisor or public adapter that must not redefine meaning |
| `legacy` | Older surface retained only for compatibility |
| `experimental` | Non-admitted exploratory surface |
| `declaration_only` | Declared identity without executable claim |
| `generated` | Projection or fixture census derived from declarations |
| `duplicate` | Parallel surface that must not become a second authority |
| `obsolete` | Scheduled for removal after adapters exist |
| `unresolved` | Not dropped; remaining open confirmation |

## Required surfaces

| Surface | Generation | Semantic role | Layer role | Expected import / path |
| --- | --- | --- | --- | --- |
| `registry` | v2 | **canonical** | descriptor taxonomy | `ipfs_datasets_py.logic.families.registry` |
| `registry_v3` | v3 | **canonical** | lifecycle / publication | `ipfs_datasets_py.logic.families.registry_v3` |
| `profile_catalog_v3` | v3 | **canonical** | profile catalog | `ipfs_datasets_py.logic.families.profile_catalog_v3` |
| `provider_matrix_v2` | v2 | **canonical** | provider capability matrix | `ipfs_datasets_py.logic.families.provider_matrix_v2` |
| `namespaces` | v1 | **canonical_component** | namespace vocabulary | `ipfs_datasets_py.logic.families.namespaces` |
| `aliases` | v1 | **canonical_component** | identity alias table | `ipfs_datasets_py.logic.families.aliases` |
| `generated_catalog` | projection | **generated** | generated projection | `ipfs_datasets_py.logic.families.generated_catalog` |
| `conformance_inventories` | fixture corpus | **generated** | capability / conformance census | fixtures and docs under `tests/fixtures/logic/...` and `docs/logic/...` |

All eight required surfaces are classified.

## Layer composition (do not flatten)

```
models (typed descriptors)
  + namespaces + aliases
  + registry v2 (taxonomy)
  + registry v3 (lifecycle / publication)
  + profile_catalog_v3
  + provider_matrix_v2
  -> generated_catalog projections
  -> CanonicalLogicCatalogSnapshot (LPC-020)
```

Plan rule: compose, do not flatten. v2 remains taxonomy; v3 remains lifecycle.
No registry v4 rename unless a genuine wire-format migration is required.

## Surface detail

### 1. `registry` (v2) — canonical

- **Interface:** `LogicFamilyRegistry@1`
- **Category:** `registry_generation`
- **Paths:** `ipfs_datasets_py/logic/families/registry.py`, shared models in
  `.../models.py`, unit tests in `tests/unit/logic/families/test_registry.py`
- **Role:** Canonical descriptor taxonomy for family, fragment, property,
  operation, runtime, evidence, boundedness, translation, and provider
  capability descriptors.
- **Evidence:** Bound by LFV-G012 completion receipt; supervisor lazy-loads
  `ipfs_datasets_py.logic.families.registry` via
  `SupervisorCanonicalLogicAdapter@1`.
- **Does not imply:** provider executability, toolchain installation, or
  production admission.

### 2. `registry_v3` — canonical

- **Layer role:** lifecycle and publication
- **Path:** `ipfs_datasets_py/logic/families/registry_v3.py`
- **Role:** Canonical publication ladder from identity-exists through
  production-admitted. Distinct from v2 taxonomy.
- **Conflict policy (downstream):** Do not delete v2/v3; do not flatten into one
  untyped dictionary; document roles under LPC-022.

### 3. `profile_catalog_v3` — canonical

- **Path:** `ipfs_datasets_py/logic/families/profile_catalog_v3.py`
- **Role:** Canonical profile catalog generation. Profile-to-family references
  and authority ceilings are catalog identity, not runtime probes.
- **Drift:** Broken profile/family references must fail closed (LPC-021).

### 4. `provider_matrix_v2` — canonical

- **Path:** `ipfs_datasets_py/logic/families/provider_matrix_v2.py`
- **Role:** Canonical provider capability matrix. Declared vs executable
  features stay orthogonal.
- **Related non-authority / adjacent surfaces:**
  - `ipfs_datasets_py.logic.backends.registry` — canonical component for backend
    provider registration, not the families matrix generation.
  - `ipfs_accelerate_py.agent_supervisor.proof.prover_matrix_registry` —
    **duplicate** supervisor evidence-bound matrix; must project onto datasets
    identity rather than redefine it.
- **Does not imply:** installed toolchains, live availability, or semantic
  success from provider success.

### 5. `namespaces` — canonical_component

- **Path:** `ipfs_datasets_py/logic/families/namespaces.py`
- **Role:** Canonical namespace vocabulary and coercion rules used by catalog
  identity.
- **Supervisor note:** Residual supervisor-only families use reserved
  `supervisor.<name>` namespaces so reverse mapping remains exact
  (`canonical_logic_adapter`).

### 6. `aliases` — canonical_component

- **Path:** `ipfs_datasets_py/logic/families/aliases.py`
- **Role:** Canonical alias table for deterministic identity resolution.
  Collisions fail closed; aliases must not silently merge distinct semantics.
- **Supervisor maps** (`_ANALYSIS_FAMILY_TO_CANONICAL`, provider and property
  maps in `canonical_logic_adapter`) are **compatibility facades** until
  generated supervisor maps replace hand lists (LPC-G090).

### 7. `generated_catalog` — generated

- **Path:** `ipfs_datasets_py/logic/families/generated_catalog.py`
- **Planned by:** LPC-023
- **Role:** Generated projection over the layers above. Must not become a second
  hand-written inventory. Divergence from source declarations is a test failure.
- **Composition target:** `CanonicalLogicCatalogSnapshot@1` at
  `ipfs_datasets_py/logic/families/canonical_catalog.py` (LPC-020).

### 8. `conformance_inventories` — generated

Conformance and capability census artifacts (not a second semantic registry):

| Artifact | Role |
| --- | --- |
| `docs/logic/software_verification_capability_inventory.md` | Human capability census |
| `tests/fixtures/logic/software_verification/capability_matrix.json` | Machine maturity matrix |
| `tests/unit/logic/software_verification/test_capability_inventory.py` | Inventory integrity tests |
| `tests/fixtures/logic/software_verification/conformance/manifest.json` | Cross-family/provider conformance corpus |
| `tests/integration/logic/test_software_verification_conformance.py` | Conformance runner |
| `tests/integration/logic/test_verification_provider_matrix_api.py` | Provider matrix API surface |
| `test/api/test_root_mcp_formal_verification_parity.py` | Accelerator parity consumer |

Maturity states remain distinct: declared, discoverable, installed,
smoke-tested, translation-conformant, reconstruction-capable, shadow, canary,
and authoritative-for. Fixture success never upgrades to live or production
certification.

## Invariants preserved by this inventory

1. Registry presence never implies executability.
2. Declaration never implies production admission.
3. Provider success never implies semantic success.
4. v2 taxonomy and v3 lifecycle stay distinct.
5. Aliases cannot merge distinct identities.
6. Generated catalogs are projections, not authorities.
7. Supervisor adapters remain facades over datasets meaning.

## Unresolved (explicit, not dropped)

| Item | Semantic role | Reason |
| --- | --- | --- |
| Datasets source tree readability in this worktree | `unresolved` | External checkout and nested submodule were unpopulated; path-level AST confirmation is deferred to revision composition (LPC-001 / LPC-008) once the source tree is readable. |

No required surface was silently omitted.

## Coverage checklist

- [x] `registry` classified (`canonical`)
- [x] `registry_v3` classified (`canonical`)
- [x] `profile_catalog_v3` classified (`canonical`)
- [x] `provider_matrix_v2` classified (`canonical`)
- [x] `namespaces` classified (`canonical_component`)
- [x] `aliases` classified (`canonical_component`)
- [x] `generated_catalog` classified (`generated`)
- [x] `conformance_inventories` classified (`generated`)
