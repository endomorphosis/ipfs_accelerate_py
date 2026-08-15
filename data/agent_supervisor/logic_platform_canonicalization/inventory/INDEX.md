# Logic Platform Canonical Inventory Index

**Task:** LPC-008  
**Goal:** LPC-G010  
**Interface:** `LogicPlatformInventory@1`  
**Machine-readable companion:** [`inventory.json`](./inventory.json)  
**Composed:** 2026-08-15

This index composes LPC-001 through LPC-007 slice inventories into one
machine-readable and human-readable catalog. Every required inventory category
carries at least one classified item. Unresolved items are listed explicitly
and are not silently dropped. Full per-symbol detail remains in the owning
slice files; this index guarantees category coverage, revision pins, and an
unresolved ledger.

## Implementation authority

**Current heads are implementation authority.** Reviewed baselines are
comparison pins only. Record intervening changes (LPC-001) before any
production edit. No production contract rewrite is admitted until this
inventory goal is closed.

### Reviewed baselines

| Repository | Reviewed baseline |
| --- | --- |
| `endomorphosis/ipfs_datasets_py` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| `endomorphosis/ipfs_accelerate_py` | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` |

### Current heads (implementation authority)

| Repository | Branch | Current head | Ahead | Behind |
| --- | --- | --- | ---: | ---: |
| `ipfs_datasets_py` | `agent/logic-platform-canonicalization` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | 0 | 0 |
| `ipfs_accelerate_py` | `agent/logic-platform-canonicalization` | `ea11293bb996f052d620eae989f5377a956764b1` | 0 | 1245 |

Accelerate current head is **1,245 commits behind** the reviewed baseline;
merge-base is the current head. Datasets current head equals the reviewed
baseline. Aggregate production dirty paths at seal: **none**.

Source: [`revisions.json`](./revisions.json) / [`revisions.md`](./revisions.md).

## Classification vocabulary

`canonical`, `canonical_component`, `compatibility_facade`, `legacy`,
`experimental`, `declaration_only`, `generated`, `duplicate`, `obsolete`,
`unresolved`.

LPC-006 slice classifications `operational`, `compatibility`, `duplicate`, and
`canonical projection` are normalized into this vocabulary for the composed
index; the original slice classification is retained on those items as
`slice_classification` where present.

## Required categories

Every category below appears in `inventory.json` with at least one
classification. Slice files hold the full census.

| Category | Index items | Primary slice owner(s) | Full slice census |
| --- | ---: | --- | ---: |
| `public_logic_api` | 4 | LPC-002 | 34+ |
| `registry_generation` | 3 | LPC-003 | 3 |
| `family_profile_property_provider` | 3 | LPC-003 | 3+ |
| `ast_or_typed_expression` | 4 | LPC-004 | 6 |
| `formalization_artifact` | 5 | LPC-004 | 9 |
| `domain_logic_slice` | 4 | LPC-004 | 9 |
| `backend_request` | 3 | LPC-005 | 4 |
| `provider_protocol` | 3 | LPC-005 | 7 |
| `translation_contract` | 2 | LPC-005 | 4 |
| `proof_plan` | 2 | LPC-005 | 4 |
| `receipt_and_evidence` | 3 | LPC-005 | 5 |
| `cache_key` | 3 | LPC-005 | 7 |
| `provider_matrix` | 2 | LPC-003 | 1+ |
| `status_enum` | 3 | LPC-005 | (from axes) |
| `authority_enum` | 3 | LPC-005 | (from axes) |
| `boundedness_enum` | 2 | LPC-005 | (from axes) |
| `alias_table` | 1 | LPC-003 | 1 |
| `installer_mutation_boundary` | 2 | LPC-005 | 6 |
| `supervisor_import_into_datasets` | 4 | LPC-006 | 60 |
| `duplicate_supervisor_semantic_type` | 4 | LPC-006 | 38 |
| `mcp_cli_python_exposure` | 4 | LPC-002, LPC-007 | 15+ |
| `compatibility_shim` | 3 | LPC-002, LPC-007 | 14+ |
| `deprecated_module` | 3 | LPC-002, LPC-007 | 6+ |
| `test_and_conformance_corpus` | 4 | LPC-003, LPC-007 | 20+ |

**Composed index items:** 74 (category representatives + unresolved promotions)  
**Unresolved ledger entries:** 12  
**Slice items referenced (not silently dropped):** LPC-002=59, LPC-003=8 surfaces, LPC-004=24, LPC-005=44, LPC-006=98, LPC-007=49

## Slice sources

| Task | Artifact | Role |
| --- | --- | --- |
| LPC-001 | [`revisions.json`](./revisions.json) / [`revisions.md`](./revisions.md) | Exact revisions, ahead/behind, dirty paths, intervening changes |
| LPC-002 | [`datasets_public_api.json`](./datasets_public_api.json) / [`datasets_public_api.md`](./datasets_public_api.md) | Public logic APIs and compatibility shims |
| LPC-003 | [`registries.json`](./registries.json) / [`registries.md`](./registries.md) | Registries, namespaces, aliases, generated catalogs |
| LPC-004 | [`syntax_formalization.json`](./syntax_formalization.json) / [`syntax_formalization.md`](./syntax_formalization.md) | Syntax, formalization, domain slices |
| LPC-005 | [`providers_evidence.json`](./providers_evidence.json) / [`providers_evidence.md`](./providers_evidence.md) | Providers, receipts, cache keys, axes |
| LPC-006 | [`supervisor_semantics.json`](./supervisor_semantics.json) / [`supervisor_semantics.md`](./supervisor_semantics.md) | Supervisor imports and duplicate semantic types |
| LPC-007 | [`tests_and_surfaces.json`](./tests_and_surfaces.json) / [`tests_and_surfaces.md`](./tests_and_surfaces.md) | Tests, MCP/CLI, deprecated modules |

## Category index (composed representatives)

### `public_logic_api`

| ID | Name | Classification | Path |
| --- | --- | --- | --- |
| `LPC-002:mod:logic.verification_api` | LogicVerificationAPI@1 | **canonical** | `ipfs_datasets_py/logic/verification_api.py` |
| `LPC-002:mod:logic.__init__` | package root | **compatibility_facade** | `ipfs_datasets_py/logic/__init__.py` |
| `LPC-002:py:logic.verification_api.STABLE_OPERATIONS` | STABLE_OPERATIONS | **canonical** | same module |
| `LPC-002:py:gt.plan_proof` | plan_proof (GoalTactician) | **canonical** | same module |

Full public-symbol census: 59 items in `datasets_public_api.json`.

### `registry_generation`

| ID | Name | Classification | Path |
| --- | --- | --- | --- |
| `LPC-003:registry` | LogicFamilyRegistry@1 (v2 taxonomy) | **canonical** | `logic/families/registry.py` |
| `LPC-003:registry_v3` | LogicFamilyRegistryPublication@3 | **canonical** | `logic/families/registry_v3.py` |
| `LPC-003:generated_catalog` | GeneratedLogicCatalog@1 | **generated** | `logic/families/generated_catalog.py` |

Registry presence never implies executability. v2 = taxonomy; v3 = lifecycle/publication.

### `family_profile_property_provider`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-003:profile_catalog_v3` | LogicProfileCatalog@3 | **canonical** |
| `LPC-003:namespaces` | LogicNamespaces@1 | **canonical_component** |
| `LPC-003:families_models` | families.models | **canonical_component** |

### `ast_or_typed_expression`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-004:ast:syntax-core-package` | syntax_core | **canonical** |
| `LPC-004:ast:typed-expression` | TypedExpression | **canonical** |
| `LPC-004:ast:elaboration-artifact` | ElaborationArtifact | **canonical_component** |
| `LPC-004:ast:supervisor-program-source-document` | program_ast SourceDocument | **duplicate** |

Canonical pipeline front: SourceDocument → syntax_core parse → TypedExpression → ElaborationArtifact.

### `formalization_artifact`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-004:fa:formalization-artifact-v3` | FormalizationArtifact@3 | **canonical** |
| `LPC-004:fa:formalization-artifact-compiler` | compiler surface | **legacy** |
| `LPC-004:fa:formalization-artifact-v1` | FormalizationArtifact@1 | **unresolved** |
| `LPC-004:fa:formalization-artifact-v2` | FormalizationArtifact@2 | **unresolved** |
| `LPC-004:fa:proposal-advisors` | proposal advisors | **experimental** |

Only `@3` is admitted for new writes.

### `domain_logic_slice`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-004:dls:domain-logic-slice-v2` | DomainLogicSlice@2 | **canonical** |
| `LPC-004:dls:domain-logic-slice-v1` | DomainLogicSlice@1 | **unresolved** |
| `LPC-004:dls:legal-domain-slice` | legal_ir | **canonical_component** |
| `LPC-004:dls:security-domain-slice` | security_ir | **canonical_component** |

Domain adapters also include software_verification, crypto_ir, intent_ir, ui_ux_ir (full list in `syntax_formalization.json`).

### `backend_request`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:br:backend-request-v2` | BackendRequest@2 | **canonical** |
| `LPC-005:br:backend-request-ir-core` | ir_core.protocols.BackendRequest | **legacy** |
| `LPC-005:br:backend-request-v1-generic` | v1 generic payload | **legacy** |

v1 generic payloads cannot bypass BackendRequest@2.

### `provider_protocol`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:pp:logic-provider-v1` | LogicProvider@1 (live wire) | **canonical** |
| `LPC-005:pp:logic-provider-protocol-v2` | LogicProviderProtocol@2 (planned) | **declaration_only** |
| `LPC-005:pp:supervisor-proof-provider-protocol-v1` | Supervisor ProofProvider v1 | **compatibility_facade** |

### `translation_contract`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:tc:logic-translation-receipt-v1` | LogicTranslationReceipt@1 | **canonical** |
| `LPC-005:tc:supervisor-translation-contract` | supervisor translation validation | **compatibility_facade** |

### `proof_plan`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:plan:canonical-proof-plan-v1` | CanonicalProofPlan@1 | **declaration_only** |
| `LPC-005:plan:supervisor-proof-plan` | ProofPlan (supervisor) | **compatibility_facade** |

Advisors propose; they do not prove or raise authority.

### `receipt_and_evidence`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:rc:trusted-proof-receipt` | TrustedProofReceipt | **declaration_only** |
| `LPC-005:rc:supervisor-proof-receipt` | ProofReceipt (supervisor) | **compatibility_facade** |
| `LPC-005:rc:ten-point-receipt-admission` | ten-point admission policy | **canonical_component** |

A structurally valid receipt is not authenticated evidence.

### `cache_key`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:ck:canonical-proof-cache-key-v1` | CanonicalProofCacheKey@1 | **declaration_only** |
| `LPC-005:ck:datasets-proof-cache` | proof_cache | **canonical** |
| `LPC-005:ck:supervisor-proof-cache-key` | ProofCacheKey (supervisor) | **duplicate** |

Datasets owns cache-key semantics; supervisor owns placement and single-flight.

### `provider_matrix`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-003:provider_matrix_v2` | LogicProviderMatrix@2 | **canonical** |
| `LPC-003:provider_matrix_supervisor_prover_matrix` | prover_matrix_registry | **duplicate** |

### `status_enum`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:ax:logic-operation-status` | LogicOperationStatus@1 | **declaration_only** |
| `LPC-005:ax:supervisor-attempt-status` | AttemptStatus | **legacy** |
| `LPC-005:ax:verification-status-overlapping` | overlapping VerificationStatus set | **unresolved** |

Operation status ≠ semantic verdict. `succeeded` does not imply `proved`.

### `authority_enum`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:ax:supervisor-evidence-authority` | EvidenceAuthority (proof) | **legacy** |
| `LPC-005:ax:supervisor-assurance-level` | AssuranceLevel | **legacy** |
| `LPC-005:ax:duplicate-evidence-authority-goal-quality` | EvidenceAuthority (goal_quality) | **duplicate** |

### `boundedness_enum`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:ax:boundedness` | Boundedness axis | **declaration_only** |
| `LPC-005:ax:supervisor-resource-budget` | ResourceBudget | **compatibility_facade** |

### `alias_table`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-003:aliases` | LogicAliasTable@1 | **canonical_component** |

Aliases must not silently merge distinct semantic identities.

### `installer_mutation_boundary`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-005:inst:lazy-installer` | external_provers.lazy_installer | **canonical_component** |
| `LPC-005:inst:pure-data-import-boundary` | pure-data import boundary | **canonical** |

Installation is not verify. Imports perform no install/network/write side effects.

### `supervisor_import_into_datasets`

| ID | Name | Classification | Slice class |
| --- | --- | --- | --- |
| `LPC-006:imp-001` | software_contracts.cache | **canonical_component** | canonical projection |
| `LPC-006:imp-026` | families.registry | **canonical_component** | canonical projection |
| `LPC-006:imp-025` | backends.provider | **canonical_component** | canonical projection |
| `LPC-006:imp-014` | utils.symai_config | **canonical_component** | operational |

Full census: **60** direct imports in `supervisor_semantics.json`.

### `duplicate_supervisor_semantic_type`

| ID | Name | Classification | Slice class |
| --- | --- | --- | --- |
| `LPC-006:sem-001` | LogicFamily | **duplicate** | duplicate |
| `LPC-006:sem-002` | PropertyKind | **duplicate** | duplicate |
| `LPC-006:sem-005` | LogicForm | **duplicate** | duplicate |
| `LPC-006:sem-003` | PropertyType | **compatibility_facade** | compatibility |

Full census: **38** types in `supervisor_semantics.json`. LPC-090/091 own residual migration.

### `mcp_cli_python_exposure`

| ID | Name | Classification | Channel |
| --- | --- | --- | --- |
| `LPC-007:surface.python.logic_verification_api` | LogicVerificationAPI@1 | **canonical** | python |
| `LPC-007:surface.cli.logic_verification_cli` | LogicVerificationCLI@1 | **canonical** | cli |
| `LPC-007:surface.mcp.logic_verification` | LogicVerificationMCP@1 | **canonical** | mcp |
| `LPC-007:unresolved.full_logic_api_export_census` | full export census | **unresolved** | — |

Python = CLI = MCP parity is required (LPC-G130); partial channel tests already exist.

### `compatibility_shim`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-002:mod:logic.api` | logic.api | **compatibility_facade** |
| `LPC-007:shim.provider_protocol_v1_adapter` | protocol v1 adapter | **compatibility_facade** |
| `LPC-007:shim.accelerate_native_logic_tools_delegate` | native logic_tools delegate | **compatibility_facade** |

### `deprecated_module`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-002:mod:legacy_mcp_tools.temporal_deontic` | legacy_mcp_tools path | **obsolete** |
| `LPC-007:deprecated.legacy_cec_tdfol_mcp_tools` | legacy CEC/TDFOL MCP tools | **legacy** |
| `LPC-007:unresolved.deprecated_datasets_module_exhaustive_list` | exhaustive datasets deprecated list | **unresolved** |

### `test_and_conformance_corpus`

| ID | Name | Classification |
| --- | --- | --- |
| `LPC-007:corpus.accelerate.goal_tactician_cli_mcp_parity` | goal tactician CLI/MCP parity | **canonical** |
| `LPC-007:corpus.planned.lpc_mandatory_matrix` | LPC mandatory test matrix | **declaration_only** |
| `LPC-003:conformance_inventories` | capability/conformance inventories | **generated** |
| `LPC-007:corpus.datasets.unit_logic` | datasets unit logic corpus | **unresolved** |

Mocks cannot satisfy real-provider gates. Hermetic required vs installed-provider optional vs network/heavy opt-in (LPC-G140).

## Unresolved items

These remain open. They are listed so later goals do not treat them as closed by omission.

| ID | Source | Category | Summary |
| --- | --- | --- | --- |
| `LPC-008:datasets_source_tree_readability` | LPC-008 (cross-cutting) | — | Datasets checkout and nested submodule not populated in inventory worktrees; classifications use plan contracts and accelerate evidence |
| `LPC-003:datasets_source_tree_readability` | LPC-003 | — | Path-level AST confirmation of registry modules open |
| `LPC-004:fa:formalization-artifact-v1` | LPC-004 | `formalization_artifact` | FormalizationArtifact@1 module path not observed |
| `LPC-004:fa:formalization-artifact-v2` | LPC-004 | `formalization_artifact` | FormalizationArtifact@2 module path not observed |
| `LPC-004:dls:domain-logic-slice-v1` | LPC-004 | `domain_logic_slice` | DomainLogicSlice@1 module path not observed |
| `LPC-005:ax:verification-status-overlapping` | LPC-005 | `status_enum` | Full overlapping enum census deferred to LPC-030/031 |
| `LPC-005:datasets_source_tree_readability` | LPC-005 | — | requests_v2 / protocol_v2 / TrustedProofReceipt AST confirmation open |
| `LPC-007:unresolved.datasets_checkout_absent` | LPC-007 | `test_and_conformance_corpus` | Datasets corpora cannot be file-listed |
| `LPC-007:unresolved.full_logic_api_export_census` | LPC-007 | `mcp_cli_python_exposure` | Complete public symbol lists need datasets source |
| `LPC-007:unresolved.deprecated_datasets_module_exhaustive_list` | LPC-007 | `deprecated_module` | Exhaustive deprecated module list needs datasets source |
| `LPC-007:corpus.datasets.unit_logic` | LPC-007 | `test_and_conformance_corpus` | Unit logic corpus not file-listed without checkout |
| `LPC-007:corpus.datasets.integration_logic` | LPC-007 | `test_and_conformance_corpus` | Integration logic corpus not file-listed without checkout |

**Remediation for the cross-cutting source gap:** mount datasets at implementation-authority revision `ac82107e246b30e35a2bbdcf75e01370d22350c6` before production contract rewrites. Current heads remain authority even when the worktree probe could not read the tree.

## Invariants carried forward

- Current heads are implementation authority; reviewed baselines are comparison pins only.
- Registry presence never implies executability.
- Declaration never implies production admission.
- Provider success is not semantic success.
- Datasets owns logic semantics; supervisor owns scheduling, isolation, resources, and admission policy.
- Unresolved items are listed explicitly and are not silently dropped.
- No production contract rewrite is admitted until LPC-G010 closes.

## Downstream consumers

| Goal / task | Uses this inventory for |
| --- | --- |
| LPC-G020 / LPC-020 | CanonicalLogicCatalogSnapshot composition inputs |
| LPC-G030 / LPC-030 | Status, authority, and boundedness axis consolidation |
| LPC-G040+ | Formalization and domain-slice admission |
| LPC-G050+ | Provider protocol and BackendRequest@2 migration |
| LPC-G060+ | Public API facade split |
| LPC-G090 / LPC-091 | Supervisor map generation and residual type classification |
| LPC-G130 / LPC-G140 | Channel parity and test matrix |

## Acceptance

- [x] Machine-readable `inventory.json` exists with nonempty `items`
- [x] Human-readable `INDEX.md` exists
- [x] Every required category appears with a classification
- [x] Current heads and reviewed baselines are named
- [x] Unresolved items are listed, not silently dropped
- [x] Validation: `python scripts/validate_logic_platform_canonicalization_board.py --check-inventory`
