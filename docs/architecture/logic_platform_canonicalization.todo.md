# Logic Platform Canonicalization Task Board

Executable projection of
`docs/architecture/logic_platform_canonicalization.objectives.md`.
Task prefix: `## LPC-`. Live runs must use that prefix.

Operator-protected inputs:

- `docs/architecture/LOGIC_PLATFORM_CANONICALIZATION_PLAN.md`
- `docs/architecture/logic_platform_canonicalization.objectives.md`
- `docs/architecture/logic_platform_canonicalization.todo.md`
- `config/agent_supervisor_logic_platform_canonicalization_scheduler.json`
- `scripts/validate_logic_platform_canonicalization_board.py`

Implementation authority:

- datasets: `/home/barberb/lift_coding/external/ipfs_datasets` branch
  `agent/logic-platform-canonicalization` @
  `ac82107e246b30e35a2bbdcf75e01370d22350c6`
- accelerate: this worktree branch `agent/logic-platform-canonicalization` @
  `ea11293bb996f052d620eae989f5377a956764b1`

Do not edit overlapping canonical contracts. Inventory tasks are read-only
against production sources. Broad refactors wait until LPC-G010 is closed.

## LPC-000 Seal campaign control files

- Status: completed
- Completion: auto
- Priority: P0
- Track: control
- Depends on:
- Goal id: LPC-G000
- Bundle: logic-platform/root
- Parallel lane: lpc-control
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/lpc000_seal_receipt.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/lpc000_seal_receipt.md
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/notes/lpc000_seal_receipt.md && python scripts/validate_logic_platform_canonicalization_board.py --check-all
- Conflict policy: Own the seal receipt only. Protected plan/objectives/todo/scheduler/validator files are operator-sealed and must not be task Outputs.
- Resource class: cpu-small
- Is schedulable: false
- Review only: false
- Acceptance: Objectives, todo, human plan, scheduler, validator, and FormalWorkPlan exist; validator --check-all returns valid true.

## LPC-001 Record exact source revisions and intervening changes

- Status: completed
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-revisions
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/revisions.md, data/agent_supervisor/logic_platform_canonicalization/inventory/revisions.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/revisions.md, data/agent_supervisor/logic_platform_canonicalization/inventory/revisions.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/revisions.json
- Conflict policy: Own revision inventory files only. Do not fetch, checkout, or rewrite either repository.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-002, LPC-003, LPC-004, LPC-005, LPC-006, LPC-007
- Acceptance: JSON records reviewed baselines, current heads, ahead/behind counts, dirty paths, and the rule that current heads are implementation authority.

## LPC-002 Inventory datasets public logic APIs and compatibility shims

- Status: completed
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-api
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/datasets_public_api.md, data/agent_supervisor/logic_platform_canonicalization/inventory/datasets_public_api.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/datasets_public_api.md, data/agent_supervisor/logic_platform_canonicalization/inventory/datasets_public_api.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/datasets_public_api.json
- Conflict policy: Read-only against /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic. Write only inventory artifacts.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-001, LPC-003, LPC-004, LPC-005, LPC-006, LPC-007
- Acceptance: Every public import from logic.__init__, logic.api, logic.verification_api, CLI, and MCP is classified.

## LPC-003 Inventory registries, namespaces, aliases, and generated catalogs

- Status: completed
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-registry
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/registries.md, data/agent_supervisor/logic_platform_canonicalization/inventory/registries.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/registries.md, data/agent_supervisor/logic_platform_canonicalization/inventory/registries.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/registries.json
- Conflict policy: Read-only against logic/families. Write only inventory artifacts.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-001, LPC-002, LPC-004, LPC-005, LPC-006, LPC-007
- Acceptance: registry, registry_v3, profile_catalog_v3, provider_matrix_v2, namespaces, aliases, generated_catalog, and conformance inventories are classified with semantic roles.

## LPC-004 Inventory syntax, formalization, and domain-slice generations

- Status: completed
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-syntax
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/syntax_formalization.md, data/agent_supervisor/logic_platform_canonicalization/inventory/syntax_formalization.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/syntax_formalization.md, data/agent_supervisor/logic_platform_canonicalization/inventory/syntax_formalization.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/syntax_formalization.json
- Conflict policy: Read-only against syntax_core, formalization, legal_ir, security_ir, intent_ir, crypto_ir, software_verification, ui_ux_ir. Write only inventory artifacts.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-001, LPC-002, LPC-003, LPC-005, LPC-006, LPC-007
- Acceptance: Every AST/typed-expression, FormalizationArtifact generation, and DomainLogicSlice generation is listed with classification.

## LPC-005 Inventory provider protocols, requests, receipts, and cache keys

- Status: todo
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-provider
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/providers_evidence.md, data/agent_supervisor/logic_platform_canonicalization/inventory/providers_evidence.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/providers_evidence.md, data/agent_supervisor/logic_platform_canonicalization/inventory/providers_evidence.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/providers_evidence.json
- Conflict policy: Read-only against logic/backends, hammers, common/proof_cache, ir_core. Write only inventory artifacts.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-001, LPC-002, LPC-003, LPC-004, LPC-006, LPC-007
- Acceptance: BackendRequest generations, provider protocol generations, translation contracts, proof-plan contracts, receipts, cache-key types, installer mutation boundaries, and status/authority/boundedness enums are classified.

## LPC-006 Inventory supervisor semantic types and datasets imports

- Status: completed
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-supervisor
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/supervisor_semantics.md, data/agent_supervisor/logic_platform_canonicalization/inventory/supervisor_semantics.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/supervisor_semantics.md, data/agent_supervisor/logic_platform_canonicalization/inventory/supervisor_semantics.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/supervisor_semantics.json
- Conflict policy: Read-only against ipfs_accelerate_py/agent_supervisor/proof and integrations. Write only inventory artifacts.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-001, LPC-002, LPC-003, LPC-004, LPC-005, LPC-007
- Acceptance: Every direct supervisor-to-datasets import and every duplicate supervisor-side semantic type is classified as operational, compatibility, duplicate, or canonical projection.

## LPC-007 Inventory tests, MCP, CLI, and deprecated modules

- Status: todo
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-000
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-surface
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/tests_and_surfaces.md, data/agent_supervisor/logic_platform_canonicalization/inventory/tests_and_surfaces.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/tests_and_surfaces.md, data/agent_supervisor/logic_platform_canonicalization/inventory/tests_and_surfaces.json
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/inventory/tests_and_surfaces.json
- Conflict policy: Read-only scan of tests, CLI, MCP, and deprecated modules. Write only inventory artifacts.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-001, LPC-002, LPC-003, LPC-004, LPC-005, LPC-006
- Acceptance: Relevant test/conformance corpora, MCP/CLI/Python exposures, compatibility shims, and deprecated modules are classified.

## LPC-008 Compose the canonical inventory index

- Status: todo
- Completion: auto
- Priority: P0
- Track: inventory
- Depends on: LPC-001, LPC-002, LPC-003, LPC-004, LPC-005, LPC-006, LPC-007
- Goal id: LPC-G010
- Bundle: logic-platform/inventory
- Parallel lane: lpc-inventory-index
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/INDEX.md, data/agent_supervisor/logic_platform_canonicalization/inventory/inventory.json
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/inventory/INDEX.md, data/agent_supervisor/logic_platform_canonicalization/inventory/inventory.json
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-inventory
- Conflict policy: Own only the composed index. Do not rewrite slice inventories except to add cross-references.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: One machine-readable inventory.json and human INDEX.md cover every required category with a classification. Unresolved items are listed, not silently dropped.

## LPC-020 Compose CanonicalLogicCatalogSnapshot from existing layers

- Status: todo
- Completion: auto
- Priority: P0
- Track: catalog
- Depends on: LPC-008
- Goal id: LPC-G020
- Bundle: logic-platform/catalog
- Parallel lane: lpc-catalog
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/catalog_migration.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/families/canonical_catalog.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/families/test_canonical_catalog.py, data/agent_supervisor/logic_platform_canonicalization/notes/catalog_migration.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/families/test_canonical_catalog.py -q
- Conflict policy: Own new snapshot module and tests in the datasets campaign branch. Do not delete registry v2/v3 or flatten typed layers.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-030
- Acceptance: Snapshot composes taxonomy, namespaces, aliases, publication, profiles, properties, views, notations, encodings, providers, matrix, lanes, evidence, translations, versions, and content identity.

## LPC-021 Add catalog drift tests

- Status: todo
- Completion: auto
- Priority: P0
- Track: catalog
- Depends on: LPC-020
- Goal id: LPC-G020
- Bundle: logic-platform/catalog
- Parallel lane: lpc-catalog
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/catalog_drift_tests.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/families/test_catalog_drift.py, data/agent_supervisor/logic_platform_canonicalization/notes/catalog_drift_tests.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/families/test_catalog_drift.py -q
- Conflict policy: Own drift tests only.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: Aliases, namespace coercion, profile/family references, provider operations, executable-vs-declared features, authority ceilings, and catalog-root reproducibility fail closed.

## LPC-022 Document registry v2/v3 semantic roles

- Status: todo
- Completion: auto
- Priority: P1
- Track: catalog
- Depends on: LPC-020
- Goal id: LPC-G020
- Bundle: logic-platform/catalog
- Parallel lane: lpc-catalog-docs
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/registry_roles.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/registry_roles.md
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/notes/registry_roles.md
- Conflict policy: Own the role note only.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: v2 taxonomy vs v3 lifecycle is documented, or a tested better arrangement is recorded. No registry v4 rename.

## LPC-023 Generate catalog projections instead of hand-written inventories

- Status: todo
- Completion: auto
- Priority: P1
- Track: catalog
- Depends on: LPC-021
- Goal id: LPC-G020
- Bundle: logic-platform/catalog
- Parallel lane: lpc-catalog
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/generated_catalogs.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/families/generated_catalog.py, data/agent_supervisor/logic_platform_canonicalization/notes/generated_catalogs.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/families/test_generated_catalog.py -q
- Conflict policy: Own generated catalog projection. Do not keep a second hand inventory.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: Generated catalogs differ from source declarations only by failing tests.

## LPC-030 Define orthogonal canonical axes

- Status: todo
- Completion: auto
- Priority: P0
- Track: axes
- Depends on: LPC-008
- Goal id: LPC-G030
- Bundle: logic-platform/axes
- Parallel lane: lpc-axes
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/axis_migration.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/ir_core/axes.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/ir_core/test_axes.py, data/agent_supervisor/logic_platform_canonicalization/notes/axis_migration.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/ir_core/test_axes.py -q
- Conflict policy: Own new axes module. Do not reuse one enum as another.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-020
- Acceptance: Operation status, semantic verdict, availability, evidence kind, evidence authority, boundedness, and translation preservation are distinct.

## LPC-031 Add explicit legacy enum mappings

- Status: todo
- Completion: auto
- Priority: P0
- Track: axes
- Depends on: LPC-030
- Goal id: LPC-G030
- Bundle: logic-platform/axes
- Parallel lane: lpc-axes
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/legacy_enum_mappings.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/ir_core/legacy_axis_map.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/ir_core/test_legacy_axis_map.py, data/agent_supervisor/logic_platform_canonicalization/notes/legacy_enum_mappings.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/ir_core/test_legacy_axis_map.py -q
- Conflict policy: Own mapping module and tests.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Every inventoried legacy enum has an explicit mapping. Unknown labels fail closed.

## LPC-032 Forbid inferring proof authority from provider success

- Status: todo
- Completion: auto
- Priority: P0
- Track: axes
- Depends on: LPC-031
- Goal id: LPC-G030
- Bundle: logic-platform/axes
- Parallel lane: lpc-axes
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/no_success_implies_proof.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_success_is_not_proof.py, data/agent_supervisor/logic_platform_canonicalization/notes/no_success_implies_proof.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_success_is_not_proof.py -q
- Conflict policy: Own the adversarial test. Narrow production fixes only where a silent promotion exists.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: succeeded + unknown + advisory is representable and cannot pass a kernel-required policy.

## LPC-040 Enforce FormalizationArtifact@3 and DomainLogicSlice@2 on new writes

- Status: todo
- Completion: auto
- Priority: P0
- Track: formalization
- Depends on: LPC-021, LPC-032
- Goal id: LPC-G040
- Bundle: logic-platform/formalization
- Parallel lane: lpc-formalization
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/new_write_path.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/formalization/admission.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/formalization/test_admission.py, data/agent_supervisor/logic_platform_canonicalization/notes/new_write_path.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/formalization/test_admission.py -q
- Conflict policy: Own admission helper and tests. Preserve artifacts_v3.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: New writes bind source, digest, spans, expression identity, family/profile/property/view/notation, features, assumptions, unsupported extensions, status, and content identity.

## LPC-041 Legal domain adapter conformance (TDFOL, DCEC, frame logic)

- Status: todo
- Completion: auto
- Priority: P0
- Track: formalization
- Depends on: LPC-040
- Goal id: LPC-G040
- Bundle: logic-platform/formalization
- Parallel lane: lpc-domain-legal
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/legal_domain_adapter.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/legal_ir/domain_slice.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/legal_ir/test_domain_slice.py, data/agent_supervisor/logic_platform_canonicalization/notes/legal_domain_adapter.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/legal_ir/test_domain_slice.py -q
- Conflict policy: Own legal slice adapter. Do not map TDFOL/DCEC/frame logic to generic FOL/deontic/object framing.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-042, LPC-043
- Acceptance: Adapter declares source domain, view, family/profile, property, notation, preserved/lost semantics, assumptions, unsupported constructs, proof-safety, and counterexample-safety.

## LPC-042 Security, software, and crypto domain adapter conformance

- Status: todo
- Completion: auto
- Priority: P0
- Track: formalization
- Depends on: LPC-040
- Goal id: LPC-G040
- Bundle: logic-platform/formalization
- Parallel lane: lpc-domain-security
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/security_software_crypto_adapters.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/security_ir/domain_slice.py, /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/software_verification/domain_slice.py, /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/crypto_ir/domain_slice.py, data/agent_supervisor/logic_platform_canonicalization/notes/security_software_crypto_adapters.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/security_ir /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/software_verification /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/crypto_ir -q
- Conflict policy: Own those domain_slice modules and their tests. Keep contracts, STS, authorization, concurrency, separation, hyperproperties, protocols, and monitors distinct.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-041, LPC-043
- Acceptance: Each domain keeps its ontology and lowers through DomainLogicSlice@2.

## LPC-043 Intent and UI/UX domain adapter conformance

- Status: todo
- Completion: auto
- Priority: P1
- Track: formalization
- Depends on: LPC-040
- Goal id: LPC-G040
- Bundle: logic-platform/formalization
- Parallel lane: lpc-domain-intent
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/intent_ir/domain_slice.py, /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/ui_ux_ir/domain_slice.py, data/agent_supervisor/logic_platform_canonicalization/notes/intent_uiux_adapters.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/intent_ir /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/ui_ux_ir -q
- Conflict policy: Own intent and UI/UX slice adapters only.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-041, LPC-042
- Acceptance: Same adapter contract as legal/security. No universal domain IR.

## LPC-044 Reject unadmitted slices at executable request construction

- Status: todo
- Completion: auto
- Priority: P0
- Track: formalization
- Depends on: LPC-041, LPC-042, LPC-043
- Goal id: LPC-G040
- Bundle: logic-platform/formalization
- Parallel lane: lpc-formalization
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/slice_admission.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_unadmitted_slice_rejected.py, data/agent_supervisor/logic_platform_canonicalization/notes/slice_admission.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_unadmitted_slice_rejected.py -q
- Conflict policy: Own admission tests and the smallest production gate.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Executable requests without an admitted DomainLogicSlice@2 are rejected.

## LPC-050 Add LogicProviderProtocol@2 operation-specific requests

- Status: todo
- Completion: auto
- Priority: P0
- Track: provider
- Depends on: LPC-021, LPC-032
- Goal id: LPC-G050
- Bundle: logic-platform/provider
- Parallel lane: lpc-provider
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/provider_protocol_migration.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/backends/protocol_v2.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_provider_protocol_v2.py, data/agent_supervisor/logic_platform_canonicalization/notes/provider_protocol_migration.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_provider_protocol_v2.py -q
- Conflict policy: Own protocol_v2 and tests. Reuse existing successor if inventory proves one already exists.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-040
- Acceptance: Typed requests exist for capability, translation, prove/check, reconstruct, verify, and attest. Executable ops require positive finite bounds.

## LPC-051 Keep v1 generic payloads from bypassing BackendRequest@2

- Status: todo
- Completion: auto
- Priority: P0
- Track: provider
- Depends on: LPC-050
- Goal id: LPC-G050
- Bundle: logic-platform/provider
- Parallel lane: lpc-provider
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/provider_v1_adapter.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/backends/protocol_v1_adapter.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_protocol_v1_adapter.py, data/agent_supervisor/logic_platform_canonicalization/notes/provider_v1_adapter.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_protocol_v1_adapter.py -q
- Conflict policy: Own v1 adapter and tests.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: v1 payloads are parsed into an operation type, rejected, or retained as advisory. New writes use v2.

## LPC-052 Typed provider responses with untrusted default authority

- Status: todo
- Completion: auto
- Priority: P0
- Track: provider
- Depends on: LPC-050
- Goal id: LPC-G050
- Bundle: logic-platform/provider
- Parallel lane: lpc-provider
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/provider_responses.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_provider_response_v2.py, data/agent_supervisor/logic_platform_canonicalization/notes/provider_responses.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/backends/test_provider_response_v2.py -q
- Conflict policy: Own response types/tests.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Responses carry request id, operation, provider id/version, operation status, verdict, evidence kind/authority, boundedness, assumptions, translations, sources, artifacts, resources, cache provenance, and error.

## LPC-060 Decompose verification_api into internal platform services

- Status: todo
- Completion: auto
- Priority: P0
- Track: api
- Depends on: LPC-051
- Goal id: LPC-G060
- Bundle: logic-platform/api
- Parallel lane: lpc-api
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/api_decomposition.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/platform/service.py, data/agent_supervisor/logic_platform_canonicalization/notes/api_decomposition.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_verification_api.py -q
- Conflict policy: Own logic/platform/* internal modules. Keep verification_api.py as a facade.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: Internal layout matches contracts/catalog/discovery/formalization/obligations/translations/providers/planning/execution/evidence/receipts/counterexamples/installation/compatibility/service.

## LPC-061 Keep pure-data imports side-effect free

- Status: todo
- Completion: auto
- Priority: P0
- Track: api
- Depends on: LPC-060
- Goal id: LPC-G060
- Bundle: logic-platform/api
- Parallel lane: lpc-api
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/pure_data_imports.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_pure_data_import.py, data/agent_supervisor/logic_platform_canonicalization/notes/pure_data_imports.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_pure_data_import.py -q
- Conflict policy: Own the import test and the smallest import-graph fix.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Importing contracts, catalog, syntax, formalization, provider protocol, and supervisor adapter does not import solvers, install packages, open the network, start processes, mutate files, probe hardware, or change environment variables.

## LPC-062 Thin compatibility facades for logic.api and logic.__init__

- Status: todo
- Completion: auto
- Priority: P1
- Track: api
- Depends on: LPC-060
- Goal id: LPC-G060
- Bundle: logic-platform/api
- Parallel lane: lpc-api-compat
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/compat_facades.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/api.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_logic_api_v1_compatibility.py, data/agent_supervisor/logic_platform_canonicalization/notes/compat_facades.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_logic_api_v1_compatibility.py -q
- Conflict policy: Own facades and compatibility tests. No second implementation.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Existing public imports still resolve. Deprecation diagnostics emit where appropriate.

## LPC-070 Define the canonical proof-plan model

- Status: todo
- Completion: auto
- Priority: P0
- Track: tactician
- Depends on: LPC-044, LPC-052
- Goal id: LPC-G070
- Bundle: logic-platform/tactician
- Parallel lane: lpc-tactician
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/tactician_plan_model.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/tactician/models.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/tactician/test_models.py, data/agent_supervisor/logic_platform_canonicalization/notes/tactician_plan_model.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/tactician/test_models.py -q
- Conflict policy: Own tactician models. Do not add a second planner.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-080
- Acceptance: Plan represents goal, interpretations, properties, assumptions, obligations, dependency graph, translations, lanes, reconstruction, bounds, fallbacks, status, and completeness boundary.

## LPC-071 Separate advisor proposals from proof authority

- Status: todo
- Completion: auto
- Priority: P0
- Track: tactician
- Depends on: LPC-070
- Goal id: LPC-G070
- Bundle: logic-platform/tactician
- Parallel lane: lpc-tactician
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/advisor_authority.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py, data/agent_supervisor/logic_platform_canonicalization/notes/advisor_authority.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/tactician/test_advisor_cannot_raise_authority.py -q
- Conflict policy: Own the authority test and the smallest policy fix.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Advisors cannot mark proposals proved, raise authority, choose verification keys, skip reconstruction, approve production, silently add assumptions, or drop blocking obligations.

## LPC-080 Canonical semantic cache-key contract

- Status: todo
- Completion: auto
- Priority: P0
- Track: cache
- Depends on: LPC-032, LPC-052
- Goal id: LPC-G080
- Bundle: logic-platform/cache
- Parallel lane: lpc-cache
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/cache_key_contract.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/common/canonical_cache_key.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/common/test_canonical_cache_key.py, data/agent_supervisor/logic_platform_canonicalization/notes/cache_key_contract.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/common/test_canonical_cache_key.py -q
- Conflict policy: Own canonical key module. Supervisor cache files may only call it.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-070
- Acceptance: Keys bind the required identity fields and reject invalid CIDs, empty digests, and candidate-as-kernel entries.

## LPC-081 Unified backend-neutral proof repository interface

- Status: todo
- Completion: auto
- Priority: P1
- Track: cache
- Depends on: LPC-080
- Goal id: LPC-G080
- Bundle: logic-platform/cache
- Parallel lane: lpc-cache
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/proof_repository.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/common/proof_repository.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/common/test_proof_repository.py, data/agent_supervisor/logic_platform_canonicalization/notes/proof_repository.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/common/test_proof_repository.py -q
- Conflict policy: Own the public repository interface. DuckDB may remain an implementation.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: One interface covers plans, attempts, evidence, receipts, counterexamples, attestations, lookup, freshness, invalidation, and lineage.

## LPC-090 Generate supervisor compatibility maps from the catalog

- Status: todo
- Completion: auto
- Priority: P0
- Track: supervisor-maps
- Depends on: LPC-023, LPC-031
- Goal id: LPC-G090
- Bundle: logic-platform/supervisor-adapter
- Parallel lane: lpc-supervisor-maps
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/canonical_logic_adapter.py, test/api/test_canonical_logic_adapter.py, data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md
- Validation: python -m pytest test/api/test_canonical_logic_adapter.py -q
- Conflict policy: Own adapter generation. Keep lazy imports. Do not hand-maintain family lists.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-100
- Acceptance: Generated artifact maps supervisor legacy value to canonical identity, disposition, residual, deprecation, and catalog root. Unknown values fail closed.

## LPC-091 Classify leftover supervisor semantic types

- Status: todo
- Completion: auto
- Priority: P1
- Track: supervisor-maps
- Depends on: LPC-090
- Goal id: LPC-G090
- Bundle: logic-platform/supervisor-adapter
- Parallel lane: lpc-supervisor-maps
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_type_classification.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_type_classification.md
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_type_classification.md
- Conflict policy: Own the classification note. Public-type removals require a migration path.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: LogicFamily, PropertyKind, LogicForm, TranslationClass, supervisor capability/operation/matrix/cache types are classified.

## LPC-100 Add LogicPlatformManifest@1 and handshake

- Status: todo
- Completion: auto
- Priority: P0
- Track: manifest
- Depends on: LPC-023
- Goal id: LPC-G100
- Bundle: logic-platform/manifest
- Parallel lane: lpc-manifest
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/manifest_handshake.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/platform/manifest.py, /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/platform/test_manifest.py, data/agent_supervisor/logic_platform_canonicalization/notes/manifest_handshake.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/platform/test_manifest.py -q
- Conflict policy: Own manifest module and tests.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Allow concurrent with: LPC-090
- Acceptance: Handshake works from wheels without sibling repos or Git metadata. Git remains optional provenance.

## LPC-110 Implement SupervisorLogicPlatformClient

- Status: todo
- Completion: auto
- Priority: P0
- Track: supervisor-client
- Depends on: LPC-052, LPC-090, LPC-100
- Goal id: LPC-G110
- Bundle: logic-platform/supervisor-client
- Parallel lane: lpc-supervisor-client
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py, test/api/test_supervisor_logic_platform_client.py, data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md
- Validation: python -m pytest test/api/test_supervisor_logic_platform_client.py -q
- Conflict policy: Own the client module and tests. Do not create another supervisor.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: Client supports handshake, catalog, formalization, slice/obligation/plan, capability discovery, typed invocation, reconstruction, verification, receipts, counterexamples, and cache freshness.

## LPC-111 Enforce supervisor admission of receipts

- Status: todo
- Completion: auto
- Priority: P0
- Track: supervisor-client
- Depends on: LPC-110
- Goal id: LPC-G110
- Bundle: logic-platform/supervisor-client
- Parallel lane: lpc-supervisor-client
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/receipt_admission.md
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/logic_platform_admission.py, test/api/test_logic_platform_admission.py, data/agent_supervisor/logic_platform_canonicalization/notes/receipt_admission.md
- Validation: python -m pytest test/api/test_logic_platform_admission.py -q
- Conflict policy: Own admission helper and tests.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: A result may affect completion or merge only after structural validity, content identity, source/tree/environment/policy binding, translation chain, evidence kind, authority ceiling, required reconstruction, freshness, non-simulation, and policy admission.

## LPC-120 Derive Hammer adapter vocabularies from the catalog

- Status: todo
- Completion: auto
- Priority: P0
- Track: hammer
- Depends on: LPC-090, LPC-052
- Goal id: LPC-G120
- Bundle: logic-platform/hammer
- Parallel lane: lpc-hammer
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/hammer_adapter.md
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_ipfs_datasets_logic_provider.py, data/agent_supervisor/logic_platform_canonicalization/notes/hammer_adapter.md
- Validation: python -m pytest test/api/test_ipfs_datasets_logic_provider.py -q
- Conflict policy: Own adapter vocabulary derivation. Hammer stays candidate-producing.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: No hand-maintained family/provider/encoding/authority lists remain in the adapter. Semantic separations are preserved.

## LPC-130 Python, CLI, and MCP parity tests

- Status: todo
- Completion: auto
- Priority: P0
- Track: parity
- Depends on: LPC-062, LPC-110
- Goal id: LPC-G130
- Bundle: logic-platform/parity
- Parallel lane: lpc-parity
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_channel_parity.py, test/api/test_logic_channel_parity.py, data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md
- Validation: python -m pytest /home/barberb/lift_coding/external/ipfs_datasets/tests/unit/logic/test_channel_parity.py test/api/test_logic_channel_parity.py -q
- Conflict policy: Own parity tests and generated operation catalog.
- Resource class: cpu-medium
- Is schedulable: true
- Review only: false
- Acceptance: Channels agree on names, schemas, status, authority, failure codes, and opt-in. Installation is not an ordinary verify operation.

## LPC-140 Hermetic required tests and fail-closed cases

- Status: todo
- Completion: auto
- Priority: P0
- Track: tests
- Depends on: LPC-044, LPC-052, LPC-080, LPC-110
- Goal id: LPC-G140
- Bundle: logic-platform/tests
- Parallel lane: lpc-tests
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-test-matrix
- Conflict policy: Own the matrix note and missing hermetic tests.
- Resource class: cpu-validation
- Is schedulable: true
- Review only: false
- Acceptance: Pure import, catalog, syntax, admission, translation, protocol, evidence, and adversarial tests are listed as hermetic required and pass.

## LPC-141 Direct-versus-supervisor parity tests

- Status: todo
- Completion: auto
- Priority: P0
- Track: tests
- Depends on: LPC-111, LPC-140
- Goal id: LPC-G140
- Bundle: logic-platform/tests
- Parallel lane: lpc-tests-parity
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/direct_supervisor_parity.md
- Predicted files: test/api/test_direct_vs_supervisor_logic_parity.py, data/agent_supervisor/logic_platform_canonicalization/notes/direct_supervisor_parity.md
- Validation: python -m pytest test/api/test_direct_vs_supervisor_logic_parity.py -q
- Conflict policy: Own the parity test.
- Resource class: cpu-validation
- Is schedulable: true
- Review only: false
- Acceptance: Representative operations agree on request, obligation, provider request, verdict, evidence, authority, boundedness, and receipt identities.

## LPC-142 Real local provider smoke path

- Status: todo
- Completion: auto
- Priority: P1
- Track: tests
- Depends on: LPC-140
- Goal id: LPC-G140
- Bundle: logic-platform/tests
- Parallel lane: lpc-tests-smoke
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md
- Conflict policy: Own smoke notes and an existing local provider test. Do not add a new prover.
- Resource class: cpu-proof-solver
- Is schedulable: true
- Review only: false
- Acceptance: At least one already-supported local provider is exercised. Mocks are labeled and do not satisfy the real-provider gate.

## LPC-150 Clean-install and no-sibling packaging tests

- Status: todo
- Completion: auto
- Priority: P0
- Track: packaging
- Depends on: LPC-141
- Goal id: LPC-G150
- Bundle: logic-platform/packaging
- Parallel lane: lpc-packaging
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-ci
- Conflict policy: Own packaging tests and notes.
- Resource class: cpu-validation
- Is schedulable: true
- Review only: false
- Acceptance: Alone-datasets, alone-accelerate, compatible together, incompatible together, no sibling, no Git, no optional solver, and one local solver scenarios are specified and the hermetic subset passes.

## LPC-151 Make required CI lanes fail on failure

- Status: todo
- Completion: auto
- Priority: P0
- Track: packaging
- Depends on: LPC-150
- Goal id: LPC-G150
- Bundle: logic-platform/packaging
- Parallel lane: lpc-packaging
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/ci_lanes.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/notes/ci_lanes.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-ci
- Conflict policy: Own CI job definitions. Do not use continue-on-error or || true on required lanes.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Required lanes exist for contracts, unit, parser, domain-slice, provider, tactician, receipts, adapter, manifest, parity, wheel install, doc drift, and catalog drift.

## LPC-160 Generate verified documentation and migration guide

- Status: todo
- Completion: auto
- Priority: P1
- Track: docs
- Depends on: LPC-151
- Goal id: LPC-G160
- Bundle: logic-platform/docs
- Parallel lane: lpc-docs
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/documentation.md
- Predicted files: /home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/logic/README.md, data/agent_supervisor/logic_platform_canonicalization/notes/documentation.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-docs
- Conflict policy: Own docs and generated tables. No hardcoded counts or readiness claims.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: README, API, catalogs, matrix, translations, tactician, supervisor integration, migration, examples, and runbooks are updated or generated.

## LPC-170 Write the 21-section evidence-based final report

- Status: todo
- Completion: auto
- Priority: P0
- Track: report
- Depends on: LPC-160
- Goal id: LPC-G170
- Bundle: logic-platform/report
- Parallel lane: lpc-report
- Outputs: data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Predicted files: data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-final-report
- Conflict policy: Own the final report only.
- Resource class: cpu-small
- Is schedulable: true
- Review only: false
- Acceptance: Sections 1-21 are present. Closing claim is the required evidence-based paragraph. Unresolved gaps and the next work board are explicit.
