# Logic Platform Canonicalization Objective Heap

Program prefix: `LPC`. Root goal: `LPC-G000`. Companion board:
[`logic_platform_canonicalization.todo.md`](./logic_platform_canonicalization.todo.md).
Human plan:
[`LOGIC_PLATFORM_CANONICALIZATION_PLAN.md`](./LOGIC_PLATFORM_CANONICALIZATION_PLAN.md).
Formal work plan:
`data/agent_supervisor/logic_platform_canonicalization/formal_work_plan.json`.

This heap is the durable intent for making `ipfs_datasets_py.logic` the only
semantic, formalization, provider, evidence, and verification authority, and
for making `ipfs_accelerate_py.agent_supervisor` consume those contracts
through one lazy, version-negotiated boundary.

Reviewed baseline revisions (compare, then use current heads as authority):

- `ipfs_datasets_py`: `ac82107e246b30e35a2bbdcf75e01370d22350c6`
- `ipfs_accelerate_py`: `485edc0871c55b0e2ef21d83bece9fa12c2c8d84`

Current implementation authority at campaign start:

- datasets checkout: `/home/barberb/lift_coding/external/ipfs_datasets` @
  `ac82107e246b30e35a2bbdcf75e01370d22350c6` (equals reviewed baseline)
- accelerate checkout: `/home/barberb/lift_coding/external/ipfs_accelerate` @
  `ea11293bb996f052d620eae989f5377a956764b1` (1,245 commits *behind* the
  reviewed baseline; merge-base is current HEAD)

Program invariants:

- Do not create another theorem-prover framework, agent supervisor,
  logic-family registry, receipt format, or MCP++ profile.
- Do not add new logic families or new theorem provers.
- Supervisor may schedule, isolate, lease, cancel, and merge. It must not
  redefine family, property, profile, notation, evidence, translation,
  verdict, receipt, cache, or formalization identity.
- Registry presence never implies executability.
- Provider success never implies semantic success.
- Candidate evidence cannot satisfy kernel-required policy.
- Imports stay side-effect free. Installation is explicit. Network is denied
  by default.
- Do not begin broad refactoring until the inventory goal is closed.
- Concurrent agents must not edit overlapping canonical contracts.
- Compatibility APIs stay as thin adapters. New writes use the typed path.

## LPC-G000 ipfs_datasets_py.logic is the canonical semantic authority

- Status: active
- Parent:
- Depends on: LPC-G170
- Fib priority: 4181
- Priority: P0
- Track: integration
- Bundle: logic-platform/root
- Goal: Establish one typed fail-closed pipeline from SourceDocument through TrustedProofReceipt, with the supervisor owning only scheduling, isolation, resources, cancellation, leases, model routing, and workflow state.
- Evidence: data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Outputs: data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Validation: test -f data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Acceptance: The 21-section final report is present, narrowly evidence-based, and does not claim the whole logic platform is production-ready solely because refactor and tests complete.
- Conflict policy: Tracking-only root. Reconcile after LPC-G170.
- Interfaces: LogicPlatformProgram@1
- Resource class: cpu-small

## LPC-G010 Produce the mandatory current-state inventory

- Status: active
- Parent: LPC-G000
- Depends on:
- Fib priority: 1
- Priority: P0
- Track: inventory
- Bundle: logic-platform/inventory
- Goal: Classify every public logic API, registry generation, family/profile/property/provider vocabulary, AST, formalization artifact, domain slice, backend request, provider protocol, translation, proof-plan, receipt, cache-key, matrix, status/authority/boundedness enum, alias table, installer mutation, supervisor-to-datasets import, duplicate supervisor semantic type, MCP/CLI/Python exposure, compatibility shim, deprecated module, and relevant test corpus as canonical, canonical component, compatibility facade, legacy, experimental, declaration-only, generated, duplicate, obsolete, or unresolved.
- Evidence: data/agent_supervisor/logic_platform_canonicalization/inventory/INDEX.md, data/agent_supervisor/logic_platform_canonicalization/inventory/inventory.json
- Outputs: data/agent_supervisor/logic_platform_canonicalization/inventory/INDEX.md, data/agent_supervisor/logic_platform_canonicalization/inventory/inventory.json
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-inventory
- Acceptance: Machine-readable and human-readable inventories exist, name current heads and the two reviewed baselines, and classify every required category. No production contract rewrite is admitted until this goal is closed.
- Conflict policy: Own only inventory artifacts under data/agent_supervisor/logic_platform_canonicalization/inventory/. Read production sources; do not edit them.
- Interfaces: LogicPlatformInventory@1
- Resource class: cpu-small

## LPC-G020 Establish one immutable CanonicalLogicCatalogSnapshot

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G010
- Fib priority: 2
- Priority: P0
- Track: catalog
- Bundle: logic-platform/catalog
- Goal: Compose taxonomy, namespaces, aliases, registry v2, registry v3, profile catalog v3, provider matrix v2, and generated catalogs into one immutable snapshot with a reproducible content root. Registry v2 remains the descriptor taxonomy layer and registry v3 remains the lifecycle/publication layer unless a documented better arrangement is tested.
- Evidence: ipfs_datasets_py/logic/families/canonical_catalog.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/catalog_migration.md
- Validation: python -m pytest tests/unit/logic/families/test_canonical_catalog.py -q
- Acceptance: Snapshot distinguishes identity-exists through production-admitted. Declaration never implies executability. Drift tests cover aliases, namespaces, profiles, providers, authority ceilings, and catalog-root reproducibility. No registry v4 unless a genuine wire-format migration is required.
- Conflict policy: Own new snapshot/composition/tests. Do not flatten typed layers into one untyped dictionary. Do not delete registry v2 or v3.
- Interfaces: CanonicalLogicCatalogSnapshot@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G030 Consolidate orthogonal status, verdict, authority, and boundedness axes

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G010
- Fib priority: 2
- Priority: P0
- Track: axes
- Bundle: logic-platform/axes
- Goal: Replace overlapping VerificationStatus, proof/tactician statuses, EvidenceKind/Authority, availability, support, runtime, boundedness, and translation enums with orthogonal canonical axes plus explicit legacy mappings.
- Evidence: ipfs_datasets_py/logic/ir_core/axes.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/axis_migration.md
- Validation: python -m pytest tests/unit/logic/ir_core/test_axes.py -q
- Acceptance: Operation status, semantic verdict, availability, evidence kind, evidence authority, boundedness, and translation preservation are separate types. A succeeded provider response can still carry unknown/advisory. No code infers proof authority from operation success.
- Conflict policy: Own new axis module, mappings, and tests. Legacy enums remain through explicit adapters.
- Interfaces: LogicOperationStatus@1, LogicSemanticVerdict@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G040 Make typed syntax and FormalizationArtifact@3 the only new-write path

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G020, LPC-G030
- Fib priority: 3
- Priority: P0
- Track: formalization
- Bundle: logic-platform/formalization
- Goal: Bind every new domain formalization to source identity, typed-expression identity, family/profile/property/view/notation, assumptions, unsupported extensions, and content identity. Domain adapters lower legal, security, intent, crypto, software, and UI/UX IRs through DomainLogicSlice@2 without collapsing ontologies.
- Evidence: ipfs_datasets_py/logic/formalization/artifacts_v3.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/domain_slice_conformance.md
- Validation: python -m pytest tests/unit/logic/formalization tests/unit/logic/syntax_core -q
- Acceptance: TDFOL, DCEC, and frame logic stay distinct. Software/security families stay distinct. Every adapter declares preservation, loss, proof-safety, and counterexample-safety. No new families.
- Conflict policy: Own adapters, slice admission, and tests. Do not silently map TDFOL to FOL, DCEC to generic deontic, or F-logic to object framing.
- Interfaces: FormalizationArtifact@3, DomainLogicSlice@2
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G050 Replace generic provider payloads with LogicProviderProtocol@2

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G020, LPC-G030
- Fib priority: 3
- Priority: P0
- Track: provider
- Bundle: logic-platform/provider
- Goal: Add an operation-specific typed provider-protocol successor unless an equivalent already exists. Capability, translation, prove/check, reconstruct, verify, and attest requests replace unrestricted JSON payloads. Every executable operation carries positive finite bounds.
- Evidence: ipfs_datasets_py/logic/backends/provider.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/provider_protocol_migration.md
- Validation: python -m pytest tests/unit/logic/backends/test_provider_protocol_v2.py -q
- Acceptance: v1 generic payloads are parsed, rejected, or retained as advisory. They cannot bypass BackendRequest@2. Provider output stays untrusted until validation or reconstruction.
- Conflict policy: Own protocol types, v1 adapter, and tests. Keep lazy loading, JSON compatibility, cancellation, deadlines, and network policy.
- Interfaces: LogicProviderProtocol@2, BackendRequest@2
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G060 Split verification_api behind compatibility facades

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G050
- Fib priority: 5
- Priority: P0
- Track: api
- Bundle: logic-platform/api
- Goal: Decompose logic.verification_api into internal services equivalent to logic/platform/{contracts,catalog,discovery,formalization,obligations,translations,providers,planning,execution,evidence,receipts,counterexamples,installation,compatibility,service}.py while preserving public imports.
- Evidence: ipfs_datasets_py/logic/verification_api.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/api_decomposition.md
- Validation: python -m pytest tests/unit/logic/test_logic_api_v1_compatibility.py tests/unit/logic/test_verification_api.py -q
- Acceptance: Pure-data imports probe nothing. Runtime discovery, install, and execution stay explicit. logic.api, logic.verification_api, and logic.__init__ remain thin adapters. No second independent implementation.
- Conflict policy: Own internal platform modules and facades. Do not create another top-level namespace.
- Interfaces: LogicPlatformService@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G070 Consolidate the proof tactician into one plan model

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G040, LPC-G050
- Fib priority: 5
- Priority: P0
- Track: tactician
- Bundle: logic-platform/tactician
- Goal: Establish one canonical proof-plan model for interpretation, obligations, lanes, reconstruction, bounds, and completeness. Models may propose; they may not mark themselves proved or raise authority.
- Evidence: ipfs_datasets_py/logic/tactician/planner.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/tactician_plan_model.md
- Validation: python -m pytest tests/unit/logic/tactician -q
- Acceptance: Datasets logic owns semantic interpretation and receipt verification. Supervisor owns scheduling and isolation. Supervisor may reorder semantically valid lanes but must not rewrite their meaning.
- Conflict policy: Own tactician plan types and tests. Do not add a second tactician.
- Interfaces: CanonicalProofPlan@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G080 Canonical semantic cache-key and unified proof repository

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G030, LPC-G050
- Fib priority: 5
- Priority: P0
- Track: cache
- Bundle: logic-platform/cache
- Goal: Create one datasets-owned cache-key contract binding source, expression, formalization, slice, obligation, assumptions, bounds, translation, provider, environment, policy, schema, checker, network policy, evidence kind, and authority ceiling. Supervisor may own placement and single-flight only.
- Evidence: ipfs_datasets_py/logic/common/proof_cache.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/cache_key_contract.md
- Validation: python -m pytest tests/unit/logic/common/test_canonical_cache_key.py -q
- Acceptance: Reject CID-looking non-CIDs, empty digests, default-string unknown objects, missing semantic fields, cross-environment hits, and candidate-as-kernel cache entries.
- Conflict policy: Own canonical key + repository interface. Do not redefine cache semantics in the supervisor.
- Interfaces: CanonicalProofCacheKey@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LPC-G090 Replace supervisor hand-maintained semantic maps

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G020, LPC-G030
- Fib priority: 8
- Priority: P0
- Track: supervisor-maps
- Bundle: logic-platform/supervisor-adapter
- Goal: Keep SupervisorCanonicalLogicAdapter as the single lazy boundary. Replace manual family/property/profile/view/notation/encoding/evidence/provider/translation/authority maps with generated projections from the catalog snapshot. Unknown values fail closed.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/canonical_logic_adapter.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md
- Validation: python -m pytest test/api/test_canonical_logic_adapter.py -q
- Acceptance: New supervisor records write canonical identities. Legacy enums exist only through explicit adapters. No silent collapse of distinct supervisor identities except residual compatibility data used only by the adapter.
- Conflict policy: Own adapter generation and tests. Do not remove a public type without a migration path.
- Interfaces: SupervisorCanonicalLogicAdapter@1
- Resource class: cpu-medium

## LPC-G100 Replace Git-layout checks with LogicPlatformManifest@1

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G020
- Fib priority: 8
- Priority: P0
- Track: manifest
- Bundle: logic-platform/manifest
- Goal: Expose a package-neutral manifest for package name/version, interface versions, catalog root, schema roots, operation versions, receipt/plan versions, compatible adapter versions, and optional source commit. Supervisor handshake must work from wheels without sibling repos or Git metadata.
- Evidence: ipfs_datasets_py/logic/platform/manifest.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/manifest_handshake.md
- Validation: python -m pytest tests/unit/logic/platform/test_manifest.py test/api/test_logic_platform_manifest.py -q
- Acceptance: Git alignment is optional provenance only. Incompatible versions return a typed incompatibility result.
- Conflict policy: Own manifest types and handshake tests. Do not use local repository layout as semantic compatibility authority.
- Interfaces: LogicPlatformManifest@1
- Resource class: cpu-small

## LPC-G110 Create SupervisorLogicPlatformClient

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G050, LPC-G090, LPC-G100
- Fib priority: 13
- Priority: P0
- Track: supervisor-client
- Bundle: logic-platform/supervisor-client
- Goal: Provide one lazy supervisor-side client for handshake, catalog, formalization, slice/obligation/plan creation, typed provider invocation, reconstruction, verification, receipts, counterexamples, and cache freshness.
- Evidence: ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md
- Validation: python -m pytest test/api/test_supervisor_logic_platform_client.py -q
- Acceptance: Requests bind task/tree/policy/plan/budget/network/cancellation/deadline/correlation/evidence/authority. Caller cannot overclaim authority. Admission requires the ten receipt checks in the human plan.
- Conflict policy: Own the client and its tests. Do not let the supervisor redefine semantic identities.
- Interfaces: SupervisorLogicPlatformClient@1
- Resource class: cpu-medium

## LPC-G120 Clean Hammer and datasets-logic adapters

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G020, LPC-G050, LPC-G090
- Fib priority: 13
- Priority: P0
- Track: hammer
- Bundle: logic-platform/hammer
- Goal: Remove duplicate hand-maintained family, translation, provider, solver-alias, encoding, and authority lists from ipfs_datasets_logic_provider.py and related adapters. Preserve family vs encoding vs solver vs kernel separation.
- Evidence: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/hammer_adapter.md
- Validation: python -m pytest test/api/test_ipfs_datasets_logic_provider.py -q
- Acceptance: Hammer remains candidate-producing unless independent reconstruction establishes stronger authority. F-logic, DCEC, Lean source, ATP candidates, and SMT sat stay distinct from proof authority.
- Conflict policy: Own adapter lists and tests. Do not collapse semantic distinctions because a bridge exists.
- Interfaces: DatasetsLogicProvider@1
- Resource class: cpu-medium

## LPC-G130 Establish Python, CLI, and MCP operation parity

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G060, LPC-G110
- Fib priority: 21
- Priority: P0
- Track: parity
- Bundle: logic-platform/parity
- Goal: Derive one operation catalog from the canonical service. Every channel agrees on operation name, request/response schema, status, authority, failure codes, and opt-in requirements.
- Evidence: data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md
- Validation: python -m pytest tests/unit/logic/test_channel_parity.py test/api/test_logic_channel_parity.py -q
- Acceptance: Supervisor-only mutation controls are not exposed from datasets logic. Installation is not an ordinary verification operation.
- Conflict policy: Own catalog projection and parity tests. Do not add a new MCP++ profile.
- Interfaces: LogicOperationCatalog@1
- Resource class: cpu-medium

## LPC-G140 Mandatory conformance, parity, and fail-closed tests

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G040, LPC-G050, LPC-G080, LPC-G110
- Fib priority: 21
- Priority: P0
- Track: tests
- Bundle: logic-platform/tests
- Goal: Create the current mandatory test matrix: pure import, catalog, syntax/formalization, typed request admission, translation safety, provider protocol, evidence/receipts, supervisor parity, real-provider smoke, and bounded adversarial tests.
- Evidence: data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-test-matrix
- Acceptance: Hermetic required tests pass. Mocks cannot satisfy real-provider gates. Direct and supervisor-mediated verification agree on semantic identities for tested slices.
- Conflict policy: Own new tests and matrix notes. Do not skip required tests or treat unavailable providers as passed.
- Interfaces: LogicConformanceMatrix@1
- Resource class: cpu-validation

## LPC-G150 Independent packaging and blocking CI

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G140
- Fib priority: 34
- Priority: P0
- Track: packaging
- Bundle: logic-platform/packaging
- Goal: Make both packages testable as independently installed distributions, including clean wheel, no-sibling, no-Git, no-optional-solver, one local solver, and incompatible-version handshake cases. Required CI lanes fail on failure.
- Evidence: data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-ci
- Acceptance: No continue-on-error, || true, skipped required tests, or historical reports as current evidence.
- Conflict policy: Own packaging tests and CI job definitions. Do not rely on nested submodules or import-time repair.
- Interfaces: LogicPlatformPackaging@1
- Resource class: cpu-validation

## LPC-G160 Replace stale documentation with generated or verified docs

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G150
- Fib priority: 34
- Priority: P0
- Track: docs
- Bundle: logic-platform/docs
- Goal: Update logic README, API docs, catalogs, provider matrix, translation guarantees, tactician docs, supervisor integration, migration guide, examples, and runbooks. Generate capability tables from live declarations.
- Evidence: ipfs_datasets_py/logic/README.md
- Outputs: data/agent_supervisor/logic_platform_canonicalization/notes/documentation.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-docs
- Acceptance: No hardcoded test counts, coverage percentages, submillisecond proof claims, provider availability, or production-readiness claims. Features are labeled stable, beta, experimental, declaration-only, unavailable, or deprecated.
- Conflict policy: Own documentation and generated tables. Do not invent availability.
- Interfaces: LogicPlatformDocs@1
- Resource class: cpu-small

## LPC-G170 File the 21-section evidence-based final report

- Status: active
- Parent: LPC-G000
- Depends on: LPC-G160
- Fib priority: 55
- Priority: P0
- Track: report
- Bundle: logic-platform/report
- Goal: Produce the required final report and the next-board recommendation without claiming production-readiness beyond tested providers and slices.
- Evidence: data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Outputs: data/agent_supervisor/logic_platform_canonicalization/final_report.md
- Validation: python scripts/validate_logic_platform_canonicalization_board.py --check-final-report
- Acceptance: All 21 required sections are present. The closing claim matches the evidence-based paragraph in the human plan.
- Conflict policy: Own only the final report artifact.
- Interfaces: LogicPlatformFinalReport@1
- Resource class: cpu-small
