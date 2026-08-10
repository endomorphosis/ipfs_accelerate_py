# IPFS Datasets Logic-Family Parser Wave-2 Task Board

Canonical seed tasks are immutable except for `Status`. Once `LFP2-048`
publishes the validated scorer and admission contract, objective refill may
append content-addressed derived cards before the `LFP2-049` fixed point and
`LFP2-050` release; it may never rewrite seed cards or the objective heap. All
implementation outputs are owner-scoped within the pinned `ipfs_datasets_py`
worktree. The four initial tasks own disjoint files.

## LFP2-000 Seal the Wave-2 control plane and predecessor binding

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Priority: P0
- Track: control
- Depends on:
- Goal id: LFP2-G000
- Outputs: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json, scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py
- Validation: python scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py --check-all
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/control
- Parallel lane: lfp2-control
- Resource class: cpu-small
- Resource stage: control
- Estimated tokens: 8000
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json, scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py
- Interfaces: LogicParserWave2Program@1, ConfiguredBoardScheduler@1
- Allow concurrent with:
- Conflict policy: Control artifacts are protected and changed only through reviewed operator maintenance while all supervisors are quiescent.
- Preconditions: Wave-1 release commit and datasets receipt exist and validate.
- Effects: Creates a distinct v2 namespace, branch, runtime, DAG, provider route, refill policy, and predecessor seal.
- Evidence subset: plan objective task config validator predecessor provider route refill
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Validator and configured-board preflight pass; Wave-1 control/release bytes are unchanged; initial ready set is exactly LFP2-001 through LFP2-004.
- Embedding query: wave2 supervisor control predecessor immutable board grok terra quota

## LFP2-001 Audit declared claims against current executable runtime evidence

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-baseline
- Depends on: LFP2-000
- Goal id: LFP2-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/claim_runtime_audit.py, ipfs_datasets_py/tests/unit/logic/conformance/test_claim_runtime_audit.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/claim_runtime_audit.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_claim_runtime_audit.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/baseline/claims
- Parallel lane: lfp2-contracts
- Resource class: cpu-medium
- Resource stage: discovery
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/claim_runtime_audit.py, ipfs_datasets_py/tests/unit/logic/conformance/test_claim_runtime_audit.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/claim_runtime_audit.json
- Interfaces: LogicClaimRuntimeAudit@1
- Allow concurrent with: LFP2-002, LFP2-003, LFP2-004
- Conflict policy: Own only the claim audit module, test, and generated report.
- Preconditions: Wave-1 release matrix and provider catalog are readable.
- Effects: Classifies each claim as declared, parsed, elaborated, translatable, compilable, executable, replayed, or independently validated.
- Evidence subset: registry matrix parser translator runner decoder replay kernel
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Every executable or authority-bearing claim has exact current-tree evidence or an owner-scoped typed gap; mocks and metadata-only records cannot satisfy execution.
- Embedding query: claim runtime evidence lifecycle provider parser translation replay

## LFP2-002 Inventory shared-AST bypasses and raw logic boundaries

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-baseline
- Depends on: LFP2-000
- Goal id: LFP2-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/raw_boundary_inventory.py, ipfs_datasets_py/tests/unit/logic/conformance/test_raw_boundary_inventory.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/raw_boundary_inventory.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_raw_boundary_inventory.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/baseline/boundaries
- Parallel lane: lfp2-parsers
- Resource class: cpu-medium
- Resource stage: discovery
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/raw_boundary_inventory.py, ipfs_datasets_py/tests/unit/logic/conformance/test_raw_boundary_inventory.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/raw_boundary_inventory.json
- Interfaces: RawLogicBoundaryInventory@1
- Allow concurrent with: LFP2-001, LFP2-003, LFP2-004
- Conflict policy: Own only raw-boundary discovery and report paths.
- Preconditions: Parser, formalization, backend, legacy, advisor, and domain roots exist.
- Effects: Records every raw formula/source/payload path and whether it crosses ParseArtifact, TypedExpression, compiled-artifact, and parsed-target gates.
- Evidence subset: raw string frozen json extension payload parser bypass target source
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Inventory is exhaustive under sealed roots and fails on an unclassified executable raw ingress or silent parser bypass.
- Embedding query: raw formula boundary typed ast parse artifact backend ingress

## LFP2-003 Build the sparse reachable domain-to-provider capability graph

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-baseline
- Depends on: LFP2-000
- Goal id: LFP2-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/reachable_graph.py, ipfs_datasets_py/tests/unit/logic/conformance/test_reachable_graph.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/reachable_capability_graph.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_reachable_graph.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/baseline/reachability
- Parallel lane: lfp2-translations
- Resource class: cpu-medium
- Resource stage: discovery
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/reachable_graph.py, ipfs_datasets_py/tests/unit/logic/conformance/test_reachable_graph.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/reachable_capability_graph.json
- Interfaces: ReachableCapabilityGraph@1
- Allow concurrent with: LFP2-001, LFP2-002, LFP2-004
- Conflict policy: Own only sparse graph construction, tests, and report.
- Preconditions: Domain views, family/profile registry, translations, and providers are discoverable.
- Effects: Joins domain view, family/profile, translation path, provider feature, evidence kind, lifecycle, and authority ceiling without Cartesian expansion.
- Evidence subset: domain view family profile translation provider evidence reachability
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Every admitted route is explainable and every unreachable cell is excluded with a typed reason; full Cartesian unsupported cells do not become work.
- Embedding query: sparse reachable capability graph domain logic solver provider

## LFP2-004 Expand the content-addressed logic conformance corpus

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evidence-baseline
- Depends on: LFP2-000
- Goal id: LFP2-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/corpus_v2.py, ipfs_datasets_py/tests/unit/logic/conformance/test_corpus_v2.py, ipfs_datasets_py/tests/fixtures/logic_conformance_v2/manifest.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_corpus_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/baseline/corpus
- Parallel lane: lfp2-evidence
- Resource class: cpu-medium
- Resource stage: discovery
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/corpus_v2.py, ipfs_datasets_py/tests/unit/logic/conformance/test_corpus_v2.py, ipfs_datasets_py/tests/fixtures/logic_conformance_v2/manifest.json
- Interfaces: LogicConformanceCorpus@2
- Allow concurrent with: LFP2-001, LFP2-002, LFP2-003
- Conflict policy: Own only the v2 corpus loader, manifest, and tests.
- Preconditions: Wave-1 fixtures and release identities are readable.
- Effects: Adds positive, negative, ambiguous, adversarial, round-trip, witness, model, trace, attack, proof, and resource-limit fixture contracts.
- Evidence subset: corpus fixture digest expected parse translate execute replay
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Manifest is deterministic, schema-validated, source-licensed, profile-specific, and rejects missing expected evidence or unbounded inputs.
- Embedding query: logic conformance corpus roundtrip adversarial witness proof trace

## LFP2-005 Join the Wave-2 baseline and publish lifecycle maturity rules

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: shared-contracts
- Depends on: LFP2-001, LFP2-002, LFP2-003, LFP2-004
- Goal id: LFP2-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/baseline_v2.py, ipfs_datasets_py/tests/unit/logic/conformance/test_baseline_v2.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/baseline_join.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_baseline_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/contracts/baseline
- Parallel lane: lfp2-contracts
- Resource class: cpu-medium
- Resource stage: contract
- Estimated tokens: 16000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/baseline_v2.py, ipfs_datasets_py/tests/unit/logic/conformance/test_baseline_v2.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline/baseline_join.json
- Interfaces: LogicRuntimeBaseline@2, CapabilityLifecycle@1
- Allow concurrent with:
- Conflict policy: Join task consumes immutable baseline artifacts and owns only joined output.
- Preconditions: LFP2-001 through LFP2-004 pass and bind the same source identity.
- Effects: Publishes maturity transitions and prevents declaration, parse, compile, execute, replay, and authority states from being conflated.
- Evidence subset: baseline join lifecycle maturity source identity
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Conflicting claims fail closed; each reachable gap has one owner and evidence obligation.
- Embedding query: baseline join capability lifecycle declared executable validated

## LFP2-006 Add schema-governed extensions and versioned parse/elaboration artifacts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shared-contracts
- Depends on: LFP2-005
- Goal id: LFP2-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/extensions.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/artifacts_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/ast.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/elaboration.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/codec.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_artifacts_v2.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_extensions.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_artifacts_v2.py tests/unit/logic/syntax_core/test_extensions.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/contracts/extensions
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: contract
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/extensions.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/artifacts_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/ast.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/elaboration.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/codec.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_artifacts_v2.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_extensions.py
- Interfaces: ExtensionSchemaRegistry@1, ParseArtifact@2, ElaborationArtifact@2
- Allow concurrent with:
- Conflict policy: Own extension registry, v2 syntax artifacts, AST/elaboration/codec integration, and focused tests; legacy contracts remain readable through explicit adapters.
- Preconditions: Raw extension payload gaps are classified by LFP2-002/LFP2-005.
- Effects: Registers payload codecs, child/binder positions, sorts, traversal, substitution, normalization, and unsupported behavior, then binds source/CST/AST and typed elaboration identities in versioned artifacts.
- Evidence subset: extension schema binder scope sort codec substitution normalization
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Unknown or malformed extension payloads fail with stable diagnostics; registered nodes participate in algebra, elaboration, codecs, and semantic hashing; parse and elaboration artifacts preserve exact source and diagnostic lineage.
- Embedding query: logic extension schema binder type scope traversal codec

## LFP2-007 Introduce common formalization slices, LogicObligation, and BackendRequest v2

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shared-contracts
- Depends on: LFP2-005, LFP2-006
- Goal id: LFP2-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/requests_v2.py, ipfs_datasets_py/tests/unit/logic/formalization/test_artifacts_v3.py, ipfs_datasets_py/tests/unit/logic/backends/test_requests_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/formalization/test_artifacts_v3.py tests/unit/logic/backends/test_requests_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/contracts/requests
- Parallel lane: lfp2-contracts
- Resource class: cpu-large
- Resource stage: contract
- Estimated tokens: 22000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/requests_v2.py, ipfs_datasets_py/tests/unit/logic/formalization/test_artifacts_v3.py, ipfs_datasets_py/tests/unit/logic/backends/test_requests_v2.py
- Interfaces: FormalizationArtifact@3, DomainLogicSlice@2, LogicObligation@2, BackendRequest@2
- Allow concurrent with:
- Conflict policy: Own common formalization/domain-slice and successor request contracts; legacy artifact and request modules remain dual-read until migration completes.
- Preconditions: Canonical namespace and extension contracts are available.
- Effects: Publishes a source-mapped domain-neutral formalization/slice envelope and replaces free-form family/payload routing with typed family, profile, property, view, notation, encoding, expression, feature, evidence, and bound fields.
- Evidence subset: backend request obligation family profile encoding evidence bounds
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Every admitted domain slice binds source and typed-expression identity before BackendRequest@2; cross-namespace misuse, arbitrary payloads, unsupported extensions, missing bounds, and authority overclaims fail before provider selection.
- Embedding query: typed backend request logic obligation family profile evidence

## LFP2-008 Gate raw target and result paths with compiled, parsed, execution, and replay receipts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shared-contracts
- Depends on: LFP2-007
- Goal id: LFP2-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/artifacts_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/evidence_v2.py, ipfs_datasets_py/tests/unit/logic/backends/test_artifacts_v2.py, ipfs_datasets_py/tests/unit/logic/backends/test_evidence_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/backends/test_artifacts_v2.py tests/unit/logic/backends/test_evidence_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/contracts/artifacts
- Parallel lane: lfp2-contracts
- Resource class: cpu-large
- Resource stage: contract
- Estimated tokens: 22000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/artifacts_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/evidence_v2.py, ipfs_datasets_py/tests/unit/logic/backends/test_artifacts_v2.py, ipfs_datasets_py/tests/unit/logic/backends/test_evidence_v2.py
- Interfaces: CompiledLogicArtifact@1, ParsedTargetArtifact@1, ProviderExecutionReceipt@2, EvidenceReplayReceipt@1
- Allow concurrent with:
- Conflict policy: Own common target-artifact and provider-evidence envelopes and validation tests; backend-specific compilers/adapters migrate later.
- Preconditions: BackendRequest@2 exists.
- Effects: Binds target text/bytes to typed origin, source map, compiler, encoding, toolchain request, assumptions, losses, bounds, launch/tool/output/result identities, decoded evidence, and replay disposition.
- Evidence subset: compiled target source parsed output receipt source map identity
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: No admitted executable backend accepts or returns unidentifiable raw target/result content; metadata-only or mock records cannot claim execution or replay through the v2 route.
- Embedding query: compiled logic artifact parsed target source receipt backend

## LFP2-009 Generate ProviderCapabilityMatrix v2 and migrate canonical writes

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shared-contracts
- Depends on: LFP2-005, LFP2-007, LFP2-008
- Goal id: LFP2-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/provider_matrix_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/migration_v2.py, ipfs_datasets_py/tests/unit/logic/families/test_provider_matrix_v2.py, ipfs_datasets_py/tests/unit/logic/backends/test_migration_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_provider_matrix_v2.py tests/unit/logic/backends/test_migration_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/contracts/providers
- Parallel lane: lfp2-providers
- Resource class: cpu-large
- Resource stage: migration
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/provider_matrix_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/migration_v2.py, ipfs_datasets_py/tests/unit/logic/families/test_provider_matrix_v2.py, ipfs_datasets_py/tests/unit/logic/backends/test_migration_v2.py
- Interfaces: ProviderCapabilityMatrix@2, LogicContractMigration@1
- Allow concurrent with:
- Conflict policy: Own generated v2 matrix and migration adapter; do not hand-edit old duplicate provider lists.
- Preconditions: Baseline and successor request/artifact contracts pass.
- Effects: Generates evidence-specific provider capabilities from canonical descriptors and supplies dual-read/canonical-write migration receipts.
- Evidence subset: provider capability feature evidence authority generated migration
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Provider names, syntaxes, properties, and lanes cannot masquerade as families; legacy reads diagnose aliases; every new write is canonical.
- Embedding query: provider capability matrix canonical family migration dual read

## LFP2-010 Publish the common frontend and profile descriptor contract

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: frontend-convergence
- Depends on: LFP2-006, LFP2-008, LFP2-009
- Goal id: LFP2-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/frontend_contract.py, ipfs_datasets_py/tests/unit/logic/parsers/test_frontend_contract.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_frontend_contract.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/frontends/contract
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: parser
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/frontend_contract.py, ipfs_datasets_py/tests/unit/logic/parsers/test_frontend_contract.py
- Interfaces: SharedFrontendConformance@1, LogicFrontendDescriptor@1
- Allow concurrent with:
- Conflict policy: Own common descriptor and tests; notation modules consume it without redefining core artifacts.
- Preconditions: Extension and compiled-artifact contracts pass.
- Effects: Standardizes notation/profile/features, parse modes, limits, recovery, printers, typed artifact output, and unsupported behavior.
- Evidence subset: frontend descriptor notation profile parse elaborate printer limits
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: A frontend cannot register without shared artifact output, declared limits, stable diagnostics, and feature-scoped fixtures.
- Embedding query: common logic frontend profile descriptor parse artifact

## LFP2-011 Converge SMT-LIB2 on the shared artifact pipeline

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: frontend-convergence
- Depends on: LFP2-010
- Goal id: LFP2-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/smtlib_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_smtlib_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_smtlib_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/frontends/smtlib
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: parser
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/smtlib_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_smtlib_v2.py
- Interfaces: SMTLIBFrontend@2
- Allow concurrent with: LFP2-012, LFP2-013, LFP2-014
- Conflict policy: Own SMT-LIB v2 module/tests; shared descriptor changes remain with LFP2-010.
- Preconditions: Common frontend contract passes.
- Effects: Emits source-aware typed artifacts for declared theories, commands, binders, datatypes, and controlled proof/model forms.
- Evidence subset: smtlib parser cst elaborate theories command model proof
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Supported constructs round-trip semantically; unsupported vendor/theory features and duplicate declarations fail with exact spans.
- Embedding query: smtlib2 shared parse artifact z3 cvc5 theories

## LFP2-012 Converge TPTP and TSTP on shared typed artifacts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: frontend-convergence
- Depends on: LFP2-010
- Goal id: LFP2-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tptp_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_tptp_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_tptp_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/frontends/tptp
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: parser
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tptp_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_tptp_v2.py
- Interfaces: TPTPFrontend@2, TSTPFrontend@1
- Allow concurrent with: LFP2-011, LFP2-013, LFP2-014
- Conflict policy: Own TPTP/TSTP v2 module/tests and controlled CNF/FOF/TFF proof records.
- Preconditions: Common frontend contract passes.
- Effects: Parses typed problem statements and controlled proof/status records with include policies, roles, source maps, and feature limits.
- Evidence subset: tptp tff fof cnf tstp szs proof source
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Vampire/E inputs and controlled TSTP outputs are typed; THF remains profile-scoped or explicit unsupported until admitted.
- Embedding query: tptp tstp vampire eprover typed parser proof

## LFP2-013 Converge Datalog, SecPAL, and F-logic rule frontends

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: frontend-convergence
- Depends on: LFP2-010
- Goal id: LFP2-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/rules_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/flogic_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_rules_frame_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_rules_frame_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/frontends/rules
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: parser
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/rules_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/flogic_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_rules_frame_v2.py
- Interfaces: RuleFrameFrontend@2
- Allow concurrent with: LFP2-011, LFP2-012, LFP2-014
- Conflict policy: Own v2 rule/frame modules and joint conformance test.
- Preconditions: Common frontend contract passes.
- Effects: Types variables, safety, stratification, delegation, frame slots, rule priority, queries, and controlled ErgoAI source.
- Evidence subset: datalog horn chc secpal flogic ergoai rule frame
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Unsafe/ambiguous rules and raw query strings cannot reach execution without typed artifacts and exact diagnostics.
- Embedding query: datalog secpal horn chc frame logic ergoai parser

## LFP2-014 Converge protocol, Tamarin, and program frontends

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: frontend-convergence
- Depends on: LFP2-010
- Goal id: LFP2-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/protocol_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/program_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_protocol_program_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_protocol_program_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/frontends/protocol-program
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: parser
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/protocol_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/program_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_protocol_program_v2.py
- Interfaces: ProtocolProgramFrontend@2
- Allow concurrent with: LFP2-011, LFP2-012, LFP2-013
- Conflict policy: Own v2 protocol/program modules and shared tests; backend execution remains G060.
- Preconditions: Common frontend contract passes.
- Effects: Types terms/equations/roles/events/rules and contracts/commands/VCs/resources with source maps and profile limits.
- Evidence subset: protocol tamarin program hoare contract verification condition
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: No raw protocol rule, target source, or program assertion bypasses parse/elaboration artifacts.
- Embedding query: protocol tamarin program hoare shared frontend typed

## LFP2-015 Publish temporal/modal/resource profiles and migrate legacy importers

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: frontend-convergence
- Depends on: LFP2-010, LFP2-011, LFP2-012, LFP2-013, LFP2-014
- Goal id: LFP2-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/profile_catalog_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/legacy_import_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_profile_legacy_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_profile_legacy_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/frontends/join
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Resource stage: migration
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/profile_catalog_v2.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/legacy_import_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_profile_legacy_v2.py
- Interfaces: LogicProfileCatalog@2, LegacyLogicBoundary@2
- Allow concurrent with:
- Conflict policy: Join task owns generated profile catalog and compatibility importer; it does not rewrite legacy islands.
- Preconditions: All frontend-family tasks pass their feature fixtures.
- Effects: Publishes one profile catalog and imports modal, temporal, resource, TDFOL, CEC/DCEC formulas through explicit ambiguity/loss receipts.
- Evidence subset: temporal modal resource profile tdfol dcec legacy importer
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Every registered parser emits shared artifacts; overloaded operators and legacy approximations require a declared profile and loss receipt.
- Embedding query: parser profile catalog temporal modal resource dcec tdfol

## LFP2-016 Implement the compositional TranslationPath planner

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: translation-graph
- Depends on: LFP2-008, LFP2-009, LFP2-015
- Goal id: LFP2-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/translations/planner.py, ipfs_datasets_py/tests/unit/logic/translations/test_planner.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations/test_planner.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/translations/planner
- Parallel lane: lfp2-translations
- Resource class: cpu-large
- Resource stage: translation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/translations/planner.py, ipfs_datasets_py/tests/unit/logic/translations/test_planner.py
- Interfaces: TranslationPathPlanner@1, TranslationPathReceipt@1
- Allow concurrent with:
- Conflict policy: Own planner and receipt composition; semantic edge modules are separate tasks.
- Preconditions: Typed requests, artifacts, provider features, and frontend profiles are published.
- Effects: Selects feature-total paths while composing assumptions, losses, polarity, bounds, reconstruction, and authority ceilings.
- Evidence subset: translation path feature preservation loss authority composition
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Unsupported features and authority/approximation laundering fail before compilation; path identity is deterministic.
- Embedding query: compositional logic translation planner preservation authority

## LFP2-017 Add program, VC, and separation routes to FOL, CHC, and SMT

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: translation-graph
- Depends on: LFP2-016
- Goal id: LFP2-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/translations/program.py, ipfs_datasets_py/tests/unit/logic/translations/test_program.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations/test_program.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/translations/program
- Parallel lane: lfp2-translations
- Resource class: cpu-proof-solver
- Resource stage: translation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/translations/program.py, ipfs_datasets_py/tests/unit/logic/translations/test_program.py
- Interfaces: ProgramTranslationEdges@1
- Allow concurrent with: LFP2-018, LFP2-019, LFP2-020
- Conflict policy: Own program/VC/separation edges and fixtures.
- Preconditions: Translation planner accepts reviewed edge descriptors.
- Effects: Lowers supported contracts, commands, frames, and VCs with explicit heap/resource abstractions and soundness direction.
- Evidence subset: program vc separation fol chc smt frame abstraction
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Supported obligations preserve validity direction; heap/resource loss is explicit; metamorphic and negative fixtures pass.
- Embedding query: program verification condition separation translation fol chc smt

## LFP2-018 Add state, concurrency, refinement, and temporal routes

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: translation-graph
- Depends on: LFP2-016
- Goal id: LFP2-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/translations/state_temporal.py, ipfs_datasets_py/tests/unit/logic/translations/test_state_temporal.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations/test_state_temporal.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/translations/state
- Parallel lane: lfp2-translations
- Resource class: cpu-proof-solver
- Resource stage: translation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/translations/state_temporal.py, ipfs_datasets_py/tests/unit/logic/translations/test_state_temporal.py
- Interfaces: StateTemporalTranslationEdges@1
- Allow concurrent with: LFP2-017, LFP2-019, LFP2-020
- Conflict policy: Own transition/concurrency/refinement/temporal edges and fixtures.
- Preconditions: Translation planner accepts reviewed edge descriptors.
- Effects: Maps supported state and trace properties to TLA+, bounded SMT, runtime MTL, and HyperLTL with fairness/trace/bound receipts.
- Evidence subset: transition concurrency refinement temporal tla mtl hyperltl
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Finite/infinite trace, fairness, refinement direction, clocks, and bounds cannot be omitted.
- Embedding query: transition temporal refinement translation tla runtime mtl hyperltl

## LFP2-019 Add authorization, frame, event, modal, and cognitive routes

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: translation-graph
- Depends on: LFP2-016
- Goal id: LFP2-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/translations/policy_modal.py, ipfs_datasets_py/tests/unit/logic/translations/test_policy_modal.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations/test_policy_modal.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/translations/policy-modal
- Parallel lane: lfp2-translations
- Resource class: cpu-proof-solver
- Resource stage: translation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/translations/policy_modal.py, ipfs_datasets_py/tests/unit/logic/translations/test_policy_modal.py
- Interfaces: PolicyModalTranslationEdges@1
- Allow concurrent with: LFP2-017, LFP2-018, LFP2-020
- Conflict policy: Own authorization/frame/event/modal/cognitive edges and fixtures.
- Preconditions: Translation planner accepts reviewed edge descriptors.
- Effects: Adds typed Datalog/SecPAL, FOL/ATP, relational, and reified encodings for supported policy and modal profiles.
- Evidence subset: authorization frame event deontic epistemic intention dcec tdfol
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Frame conditions, norm semantics, event closure, agent indices, reification, and approximation direction are explicit.
- Embedding query: authorization event modal deontic dcec tdfol translation

## LFP2-020 Add bounded HyperLTL self-composition routes

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: translation-graph
- Depends on: LFP2-016
- Goal id: LFP2-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/translations/hyper.py, ipfs_datasets_py/tests/unit/logic/translations/test_hyper.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations/test_hyper.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/translations/hyper
- Parallel lane: lfp2-translations
- Resource class: cpu-proof-solver
- Resource stage: translation
- Estimated tokens: 22000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/translations/hyper.py, ipfs_datasets_py/tests/unit/logic/translations/test_hyper.py
- Interfaces: HyperpropertyTranslationEdges@1
- Allow concurrent with: LFP2-017, LFP2-018, LFP2-019
- Conflict policy: Own self-composition/hyperproperty edge module and fixtures.
- Preconditions: Translation planner and hyperproperty frontend profiles exist.
- Effects: Implements restricted trace-quantifier/self-composition paths with explicit alternation, system-copy, bound, and witness contracts.
- Evidence subset: hyperltl self composition trace quantifier noninterference bound
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Unsupported alternation or unbounded composition fails; accepted transformations have differential/witness fixtures.
- Embedding query: hyperltl self composition translation noninterference

## LFP2-021 Join translation edges and controlled protocol/kernel-target compilers

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: translation-graph
- Depends on: LFP2-011, LFP2-012, LFP2-013, LFP2-014, LFP2-015, LFP2-017, LFP2-018, LFP2-019, LFP2-020
- Goal id: LFP2-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/translations/catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations/protocol_targets.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations/kernel_targets.py, ipfs_datasets_py/tests/unit/logic/translations/test_protocol_targets.py, ipfs_datasets_py/tests/conformance/logic/test_translation_paths_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations/test_protocol_targets.py tests/conformance/logic/test_translation_paths_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/translations/join
- Parallel lane: lfp2-translations
- Resource class: cpu-proof-kernel
- Resource stage: translation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/translations/catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations/protocol_targets.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations/kernel_targets.py, ipfs_datasets_py/tests/unit/logic/translations/test_protocol_targets.py, ipfs_datasets_py/tests/conformance/logic/test_translation_paths_v2.py
- Interfaces: LogicTranslationGraph@3, ProtocolTargetTranslationEdges@1, KernelTargetCompiler@2
- Allow concurrent with:
- Conflict policy: Join task publishes catalogs plus controlled protocol and theory targets; other semantic edges remain owned by their tasks.
- Preconditions: All Wave-2 frontend and translation-edge fixtures pass.
- Effects: Publishes the composed graph, separate neutral-protocol translations to ProVerif applied-pi and Tamarin multiset rewriting, and typed Lean/Rocq/Isabelle target-theory artifacts with imports/axioms/source maps.
- Evidence subset: translation catalog composed path proverif applied pi tamarin multiset rewriting kernel target lean rocq isabelle
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: All registered paths are feature total and loss receipted; protocol equations, roles/rules, channels, attacker semantics, and query identities remain dialect-specific; target theories are compilation candidates until official kernels accept them.
- Embedding query: translation graph catalog lean rocq isabelle target compiler

## LFP2-022 Connect Security IR through typed executable logic slices

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: domain-vertical-slices
- Depends on: LFP2-017, LFP2-018, LFP2-019, LFP2-020, LFP2-021
- Goal id: LFP2-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_security_ir_slice_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_security_ir_slice_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/domains/security
- Parallel lane: lfp2-domains
- Resource class: cpu-proof-solver
- Resource stage: domain
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_security_ir_slice_v2.py
- Interfaces: SecurityLogicSlice@2
- Allow concurrent with: LFP2-023, LFP2-024, LFP2-025, LFP2-026, LFP2-027
- Conflict policy: Own the Security IR v2 adapter and conformance slice only.
- Preconditions: Shared frontend and translation catalogs are published.
- Effects: Joins security claims, policies, threats, temporal behavior, authorization, protocols, separation, and hyperproperties to typed requests and evidence.
- Evidence subset: security claim source map typed expression translation backend replay
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Each admitted route has source-span-to-result lineage; information-flow, attacker, bound, and policy authority assumptions are explicit.
- Embedding query: security ir typed logic authorization protocol hyperproperty solver

## LFP2-023 Connect Crypto IR base ledger, protocol, finality, and arithmetic slices

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: domain-vertical-slices
- Depends on: LFP2-017, LFP2-018, LFP2-020, LFP2-021
- Goal id: LFP2-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir/formalization/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_crypto_ir_slice_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_crypto_ir_slice_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/domains/crypto
- Parallel lane: lfp2-domains
- Resource class: cpu-proof-solver
- Resource stage: domain
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir/formalization/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_crypto_ir_slice_v2.py
- Interfaces: CryptoLogicSlice@2
- Allow concurrent with: LFP2-022, LFP2-024, LFP2-025, LFP2-026, LFP2-027
- Conflict policy: Own the Crypto IR v2 formalization slice and tests only.
- Preconditions: Program/state/hyper translation edges and typed artifacts exist.
- Effects: Connects ledger transitions, finality/reorg, consensus, authorization, symbolic protocol, privacy, and supported integer/bitvector obligations through base/common families; finite-field and ZK-constraint overlays attach in LFP2-044 after LFP2-042.
- Evidence subset: crypto ledger consensus finality protocol arithmetic hyperproperty
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: Network/chain model, arithmetic domain, adversary, trace, finality, and approximation assumptions are never implicit.
- Embedding query: crypto currency network ir consensus finality protocol zkp logic

## LFP2-024 Connect Intent IR base goals, guards, workflows, and policy slices

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: domain-vertical-slices
- Depends on: LFP2-017, LFP2-018, LFP2-019, LFP2-021
- Goal id: LFP2-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_intent_ir_slice_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_intent_ir_slice_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/domains/intent
- Parallel lane: lfp2-domains
- Resource class: cpu-proof-solver
- Resource stage: domain
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_intent_ir_slice_v2.py
- Interfaces: IntentLogicSlice@2
- Allow concurrent with: LFP2-022, LFP2-023, LFP2-025, LFP2-026, LFP2-027
- Conflict policy: Own the Intent IR v2 formalization slice and tests only.
- Preconditions: Program, state, policy/modal, and kernel target translation paths exist.
- Effects: Types goals, guards, skill effects, workflows, existing policy/modal views, authorization, and tool invocation constraints through base/common families; normative and BDI/agency overlays attach in LFP2-044 after LFP2-037 and LFP2-040.
- Evidence subset: intent skill prompt goal guard workflow authorization policy
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Safety/liveness remain properties and VC remains a view role; advisor confidence cannot establish intent correctness.
- Embedding query: intent ir skill prompt goal guard workflow policy typed logic

## LFP2-025 Connect Legal IR base norm, exception, event, and jurisdiction slices

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: domain-vertical-slices
- Depends on: LFP2-019, LFP2-021
- Goal id: LFP2-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_legal_ir_slice_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_legal_ir_slice_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/domains/legal
- Parallel lane: lfp2-domains
- Resource class: cpu-proof-solver
- Resource stage: domain
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_legal_ir_slice_v2.py
- Interfaces: LegalLogicSlice@2
- Allow concurrent with: LFP2-022, LFP2-023, LFP2-024, LFP2-026, LFP2-027
- Conflict policy: Own Legal IR v2 adapter/tests; legacy legal parsers remain compatibility inputs.
- Preconditions: Policy/modal/event translation and controlled kernel targets exist.
- Effects: Joins existing supported norm/policy views, temporal events, exceptions, priorities, conflicts, and jurisdictions through base/common typed evidence paths; new normative, argumentation, and description-logic overlays attach in LFP2-044 after LFP2-037 through LFP2-039.
- Evidence subset: legal norm policy exception priority event conflict jurisdiction
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: Deontic profile, temporal model, defeasibility, jurisdiction, priority, and authority are explicit; graph projection is not a family.
- Embedding query: legal ir norm policy exception priority event jurisdiction theorem prover

## LFP2-026 Maintain an exact-source-gated UI and accessibility logic adapter

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: domain-vertical-slices
- Depends on: LFP2-019, LFP2-021
- Goal id: LFP2-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py, ipfs_datasets_py/tests/conformance/logic/test_ui_ux_logic_gate_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_ui_ux_logic_gate_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/domains/ui
- Parallel lane: lfp2-domains
- Resource class: cpu-medium
- Resource stage: domain
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_logic_gate_v2.py, ipfs_datasets_py/tests/conformance/logic/test_ui_ux_logic_gate_v2.py
- Interfaces: UIUXLogicSlice@2, UIUXSourceGate@2
- Allow concurrent with: LFP2-022, LFP2-023, LFP2-024, LFP2-025, LFP2-027
- Conflict policy: Never create, copy, or edit ui_ux_ir; own only gate/alias/watch tests.
- Preconditions: Source identity resolver and policy/modal routes exist.
- Effects: Records accessibility, interaction/event, workflow, ontology/frame, authorization, and observable-state requirements; emits a derived task only for a pinned reviewed source revision.
- Evidence subset: ui ux accessibility interaction source gate alias declaration only
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Absent source yields typed source_missing/declaration_only without blocking other work; present source produces one content-addressed owner-scoped adapter gap.
- Embedding query: ui ux ir accessibility interaction formal logic source gate

## LFP2-027 Connect base software-verification and contract obligations end to end

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: domain-vertical-slices
- Depends on: LFP2-017, LFP2-018, LFP2-021
- Goal id: LFP2-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_software_verification_slice_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_software_verification_slice_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/domains/software
- Parallel lane: lfp2-domains
- Resource class: cpu-proof-solver
- Resource stage: domain
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/logic_slice_v2.py, ipfs_datasets_py/tests/conformance/logic/test_software_verification_slice_v2.py
- Interfaces: SoftwareVerificationLogicSlice@2
- Allow concurrent with: LFP2-022, LFP2-023, LFP2-024, LFP2-025, LFP2-026
- Conflict policy: Own successor bridge/tests; retain rich software-verification IRs unchanged where possible.
- Preconditions: Program/state/kernel translation paths are published.
- Effects: Links contracts, VCs, program/state obligations, separation, concurrency, refinement, temporal properties, counterexamples, and target theories through base/common families; session/process overlays attach in LFP2-044 after LFP2-043.
- Evidence subset: software verification contract vc separation concurrency refinement
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Every supported obligation has typed origin, semantics, translation, request, result, replay, and authority lineage.
- Embedding query: software verification contract vc logic solver kernel bridge

## LFP2-028 Execute and replay typed Z3 and cvc5 SMT/CHC evidence

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-011, LFP2-017
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_smt_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_smt_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/smt
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Resource stage: provider
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_smt_execution_v2.py
- Interfaces: SMTProviderEvidence@2
- Allow concurrent with: LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Conflict policy: Own shared SMT v2 execution/replay layer and integration test.
- Preconditions: Provider matrix, SMT frontend, and program/SMT translations exist.
- Effects: Runs typed obligations with pinned Z3/cvc5, validates models/cores/proofs where supported, and differentially compares matched fragments.
- Evidence subset: z3 cvc5 smt chc model unsat core proof replay
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: Solver disagreement and unsupported theory/proof features are typed outcomes; success is never promoted beyond the evidence receipt.
- Embedding query: z3 cvc5 typed smt execution model core proof replay

## LFP2-029 Execute TLC and Apalache with typed state/temporal semantics

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-018
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_tla_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_tla_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/state
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Resource stage: provider
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_tla_execution_v2.py
- Interfaces: StateProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Conflict policy: Own state-provider v2 execution and trace replay tests.
- Preconditions: State/temporal translation artifacts exist.
- Effects: Separates finite-state, step-bounded, safety, liveness, fairness, and approximation semantics; decodes and replays traces.
- Evidence subset: tla tlc apalache state temporal fairness trace replay
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: TLC and Apalache capabilities/results remain distinct and every counterexample binds bounds, config, module, property, and replay outcome.
- Embedding query: tla tlc apalache model checker trace replay typed

## LFP2-030 Execute SecPAL and Datalog authorization parity paths

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-013, LFP2-019
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/datalog/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_secpal_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_secpal_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/rules
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Resource stage: provider
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/datalog/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_secpal_execution_v2.py
- Interfaces: RuleProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Conflict policy: Own authorization execution/parity layer and test.
- Preconditions: Rule frontend and policy translation edges pass.
- Effects: Runs typed queries with delegation, stratification, provenance, closed/open-world, and engine/native-shadow parity receipts.
- Evidence subset: secpal datalog authorization delegation query provenance parity
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Authorization answers bind policy/query/provenance/semantics; fallback or mock output cannot establish policy authority.
- Embedding query: secpal datalog authorization provider execution parity

## LFP2-031 Execute ProVerif and Tamarin protocol evidence with attack replay

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-014, LFP2-021
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_protocol_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_protocol_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/protocol
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Resource stage: provider
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_protocol_execution_v2.py
- Interfaces: ProtocolProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-030, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Conflict policy: Own neutral protocol execution/attack replay layer and test.
- Preconditions: Protocol frontend, typed artifacts, and separate ProVerif/Tamarin protocol-target translation edges are published.
- Effects: Preserves equations, roles/rules, channels, attacker, secrecy, reachability, correspondence/authentication, and tool-specific result identities.
- Evidence subset: proverif tamarin protocol equational correspondence attack replay
- Symbolic first: true
- LLM context budget bytes: 44000
- Acceptance: Provider-specific assumptions remain distinct; reported attacks/witnesses are parsed and replayed or explicitly non-replayable.
- Embedding query: proverif tamarin symbolic protocol attack replay correspondence

## LFP2-032 Split and execute AutoHyper and MCHyper capability paths

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: provider-execution
- Depends on: LFP2-009, LFP2-020
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/hyperproperties/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_hyper_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_hyper_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/hyper
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Resource stage: provider
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/hyperproperties/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_hyper_execution_v2.py
- Interfaces: HyperProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Conflict policy: Own split hyperproperty provider descriptors/execution and witness tests.
- Preconditions: Hyperproperty translations and provider matrix exist.
- Effects: Separates engine identities, system models, quantifier-prefix ceilings, finite/bounded semantics, and witness replay.
- Evidence subset: autohyper mchyper hyperltl system witness quantifier
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: One engine's support cannot establish another's capability; every result identifies engine, system, formula, bounds, and witness status.
- Embedding query: hyperltl autohyper mchyper provider witness replay

## LFP2-033 Execute Vampire and E with typed TPTP/TSTP reconstruction

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-012, LFP2-019
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_atp_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_atp_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/atp
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Resource stage: provider
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_atp_execution_v2.py
- Interfaces: ATPProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-034, LFP2-035, LFP2-036
- Conflict policy: Own typed ATP execution/reconstruction and test.
- Preconditions: TPTP/TSTP frontend and relational/reified translations exist.
- Effects: Runs FOL/TFF obligations, parses SZS/TSTP, reconstructs proof/countermodel evidence, and labels DCEC/TDFOL as translated rather than native.
- Evidence subset: vampire eprover tptp tstp szs proof countermodel reconstruction
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: ATP success remains candidate evidence until checked/replayed; input profile and translation assumptions are exact.
- Embedding query: vampire eprover tptp tstp proof reconstruction

## LFP2-034 Separate Hammer, reconstruction, and official kernel phases

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-012, LFP2-021
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_kernel_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_kernel_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/kernel
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-kernel
- Resource stage: provider
- Estimated tokens: 32000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_kernel_execution_v2.py
- Interfaces: KernelProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-035, LFP2-036
- Conflict policy: Own phase orchestration and pinned kernel integration tests; generated theory model remains LFP2-021.
- Preconditions: Typed target theories and provider capabilities exist.
- Effects: Separates premise selection, ATP candidate, reconstruction, target compilation, elaboration, and Lean/Rocq/Isabelle kernel acceptance.
- Evidence subset: hammer premise proof reconstruction lean rocq isabelle kernel
- Symbolic first: true
- LLM context budget bytes: 46000
- Acceptance: Imports, axioms, admits, trust escapes, environment, source theorem, and official kernel result are bound; Hammer never becomes proof authority.
- Embedding query: hammer lean rocq isabelle proof reconstruction kernel

## LFP2-035 Gate ErgoAI and SymbolicAI proposals through deterministic parsing

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-013
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/advisor_execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_advisor_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_advisor_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/advisors
- Parallel lane: lfp2-providers
- Resource class: cpu-medium
- Resource stage: provider
- Estimated tokens: 22000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/advisor_execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_advisor_execution_v2.py
- Interfaces: AdvisorProviderEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-036
- Conflict policy: Own advisor schemas/gates/tests; never weaken deterministic frontend or authority contracts.
- Preconditions: Provider matrix and rule/frame frontend exist.
- Effects: Types ErgoAI/SymAI requests/results, reparses all proposed source, checks signatures/features, and emits only unverified candidates until independent validation.
- Evidence subset: ergoai symai symbolicai advisor proposal reparse authority
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Confidence, fluent text, availability, or mock output cannot establish parse correctness, satisfiability, policy, or proof.
- Embedding query: ergoai symbolicai symai advisor deterministic parse gate

## LFP2-036 Wire runtime MTL to real monitoring and verdict replay

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-execution
- Depends on: LFP2-009, LFP2-011, LFP2-018
- Goal id: LFP2-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/runtime_mtl_execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_runtime_mtl_execution_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_runtime_mtl_execution_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/providers/runtime-mtl
- Parallel lane: lfp2-providers
- Resource class: cpu-medium
- Resource stage: provider
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/runtime_mtl_execution_v2.py, ipfs_datasets_py/tests/integration/logic_providers/test_runtime_mtl_execution_v2.py
- Interfaces: RuntimeMTLEvidence@2
- Allow concurrent with: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035
- Conflict policy: Own registry-to-monitor wiring and replay tests.
- Preconditions: Temporal translation, typed request, and provider descriptors exist.
- Effects: Replaces deferred UNKNOWN routing with actual monitor invocation and binds clock/event-time, lateness, prefix, interval, monitorability, and three-valued verdict semantics.
- Evidence subset: runtime mtl monitor trace clock interval verdict replay
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Deferred/unknown remains only an explicit unsupported/unavailable outcome; evaluated verdicts replay against the same trace and semantics.
- Embedding query: runtime mtl monitoring provider trace verdict replay

## LFP2-037 Add dyadic, defeasible, prioritized, and contrary-to-duty norms

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-015
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/normative_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_normative_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_normative_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/normative
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/normative_v2.py, ipfs_datasets_py/tests/unit/logic/parsers/test_normative_v2.py
- Interfaces: NormativeLogicProfiles@2
- Allow concurrent with: LFP2-038, LFP2-039, LFP2-040, LFP2-041, LFP2-042, LFP2-043
- Conflict policy: Own normative profile module/tests; registry publication occurs in conformance join.
- Preconditions: Extension schema and modal/profile contracts exist.
- Effects: Types conditional norms, exceptions, priority, conflicts, violations, reparations, and contrary-to-duty structures.
- Evidence subset: deontic dyadic defeasible priority exception contrary duty
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Each profile has a semantic decision record, parser/printer fixtures, negative ambiguity cases, and no unearned equivalence between norm systems.
- Embedding query: dyadic defeasible deontic norm priority exception legal

## LFP2-038 Add controlled argumentation and nonmonotonic reasoning

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-013
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/argumentation.py, ipfs_datasets_py/tests/unit/logic/parsers/test_argumentation.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_argumentation.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/argumentation
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/argumentation.py, ipfs_datasets_py/tests/unit/logic/parsers/test_argumentation.py
- Interfaces: ArgumentationLogic@1
- Allow concurrent with: LFP2-037, LFP2-039, LFP2-040, LFP2-041, LFP2-042, LFP2-043
- Conflict policy: Own argumentation/nonmonotonic nodes, profiles, and tests.
- Preconditions: Rule frontend and extension schema exist.
- Effects: Adds arguments, attacks/support, priorities, defeasible rules, grounded/preferred-style profile identities, and explicit undecided results.
- Evidence subset: argument attack support nonmonotonic defeasible semantics
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Semantics/profile is always named; undecided and multiple-extension outcomes are preserved; no classical entailment promotion.
- Embedding query: argumentation nonmonotonic defeasible legal reasoning logic

## LFP2-039 Add description-logic and ontology profiles

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-013
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/description_logic.py, ipfs_datasets_py/tests/unit/logic/parsers/test_description_logic.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_description_logic.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/description
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/description_logic.py, ipfs_datasets_py/tests/unit/logic/parsers/test_description_logic.py
- Interfaces: DescriptionLogicProfiles@1
- Allow concurrent with: LFP2-037, LFP2-038, LFP2-040, LFP2-041, LFP2-042, LFP2-043
- Conflict policy: Own controlled concept/role/axiom profiles and tests; do not claim complete OWL.
- Preconditions: Typed signatures, extension schemas, and rule/frame frontend exist.
- Effects: Adds controlled concept, role, individual, inclusion, disjointness, cardinality, and ontology import identities for legal/UI/intent/KG uses.
- Evidence subset: description logic ontology concept role axiom legal ui kg
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Supported DL profile and open-world semantics are explicit; unsupported OWL constructs fail without silent FOL approximation.
- Embedding query: description logic ontology owl legal ui knowledge graph

## LFP2-040 Add BDI, epistemic-temporal, agency, and intention profiles

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-015
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/agency.py, ipfs_datasets_py/tests/unit/logic/parsers/test_agency.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_agency.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/agency
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/agency.py, ipfs_datasets_py/tests/unit/logic/parsers/test_agency.py
- Interfaces: AgencyLogicProfiles@1
- Allow concurrent with: LFP2-037, LFP2-038, LFP2-039, LFP2-041, LFP2-042, LFP2-043
- Conflict policy: Own agency/BDI profile module/tests and explicit DCEC importer hooks.
- Preconditions: Modal/temporal profile catalog and extension schema exist.
- Effects: Types belief, knowledge, desire, intention, goal, action, agent, time, and accessibility semantics for intent/security/legal use.
- Evidence subset: bdi epistemic temporal agency intention dcec agent
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Agent/time indices and frame/introspection assumptions are explicit; BDI and DCEC profiles are not conflated.
- Embedding query: bdi belief desire intention epistemic temporal agency

## LFP2-041 Add mu-calculus syntax and controlled CTL-star lowering

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-015
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/fixed_point.py, ipfs_datasets_py/tests/unit/logic/parsers/test_fixed_point.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_fixed_point.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/fixed-point
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/fixed_point.py, ipfs_datasets_py/tests/unit/logic/parsers/test_fixed_point.py
- Interfaces: FixedPointLogicProfiles@1
- Allow concurrent with: LFP2-037, LFP2-038, LFP2-039, LFP2-040, LFP2-042, LFP2-043
- Conflict policy: Own mu/fixed-point nodes, guardedness checks, CTL-star fragment lowering, and tests.
- Preconditions: Temporal profile catalog and extension schema exist.
- Effects: Resolves mu_calculus lifecycle inconsistency and adds least/greatest fixed points with positivity/guardedness and controlled state-model routes.
- Evidence subset: mu calculus fixed point ctl star guarded positivity model checking
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Binder positivity/guardedness is checked; unsupported CTL-star or alternation depth is explicit; declaration never implies executable support.
- Embedding query: mu calculus fixed point ctl star temporal parser lowering

## LFP2-042 Add finite-field, bitvector, and ZK constraint profiles

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-011
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/finite_field.py, ipfs_datasets_py/tests/unit/logic/parsers/test_finite_field.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_finite_field.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/finite-field
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/finite_field.py, ipfs_datasets_py/tests/unit/logic/parsers/test_finite_field.py
- Interfaces: FiniteFieldConstraintLogic@1
- Allow concurrent with: LFP2-037, LFP2-038, LFP2-039, LFP2-040, LFP2-041, LFP2-043
- Conflict policy: Own finite-field/constraint profile and tests; ZKP backend proof authority remains separate.
- Preconditions: Typed sorts, extension schema, and SMT frontend exist.
- Effects: Types moduli, field operations, bitvectors, ranges, circuits, R1CS/PLONK-style constraints, and explicit translation/provider ceilings.
- Evidence subset: finite field bitvector circuit r1cs plonk crypto zkp
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Modulus/range/bit-width/circuit identities are explicit; simulated or arithmetic-solver evidence cannot become ZK proof authority.
- Embedding query: finite field bitvector circuit r1cs plonk crypto logic

## LFP2-043 Add linear, session, process, and relational refinement profiles

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: family-expansion
- Depends on: LFP2-006, LFP2-010, LFP2-014
- Goal id: LFP2-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/session_process.py, ipfs_datasets_py/tests/unit/logic/parsers/test_session_process.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_session_process.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/families/session-process
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Resource stage: family
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/session_process.py, ipfs_datasets_py/tests/unit/logic/parsers/test_session_process.py
- Interfaces: SessionProcessLogic@1
- Allow concurrent with: LFP2-037, LFP2-038, LFP2-039, LFP2-040, LFP2-041, LFP2-042
- Conflict policy: Own linear/session/process profile and tests; reuse existing resource/concurrency/refinement IRs.
- Preconditions: Protocol/program frontend and extension schema exist.
- Effects: Adds linear resources, session actions, channel/process composition, duality, progress, and relational refinement obligations.
- Evidence subset: linear session process channel duality refinement concurrency
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Linearity, duality, process scope, progress model, and refinement direction are checked; no resource duplication is silently normalized.
- Embedding query: linear logic session type process calculus refinement

## LFP2-044 Publish Wave-2 family routes and expand per-profile adversarial, round-trip, and fuzz evidence

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance
- Depends on: LFP2-015, LFP2-021, LFP2-022, LFP2-023, LFP2-024, LFP2-025, LFP2-026, LFP2-027, LFP2-037, LFP2-038, LFP2-039, LFP2-040, LFP2-041, LFP2-042, LFP2-043
- Goal id: LFP2-G080
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/registry_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/profile_catalog_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations/family_extensions.py, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/domain_family_bindings_v2.py, ipfs_datasets_py/tests/conformance/logic/test_family_route_publication_v2.py, ipfs_datasets_py/tests/fixtures/logic_conformance_v2/profile_manifest.json, ipfs_datasets_py/tests/fuzz/logic/test_wave2_parser_properties.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_family_route_publication_v2.py tests/fuzz/logic/test_wave2_parser_properties.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/conformance/corpus
- Parallel lane: lfp2-evidence
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/registry_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/profile_catalog_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations/family_extensions.py, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/domain_family_bindings_v2.py, ipfs_datasets_py/tests/conformance/logic/test_family_route_publication_v2.py, ipfs_datasets_py/tests/fixtures/logic_conformance_v2/profile_manifest.json, ipfs_datasets_py/tests/fuzz/logic/test_wave2_parser_properties.py
- Interfaces: LogicConformanceCorpus@2, LogicFamilyRegistry@3, LogicProfileCatalog@3, FamilyRoutePublication@1
- Allow concurrent with: LFP2-045, LFP2-046
- Conflict policy: Own v3 family/profile publication, family-extension routes, domain overlay bindings, profile manifest, publication tests, and Wave-2 fuzz suite; family parser modules remain owned by LFP2-037 through LFP2-043.
- Preconditions: Base domain slices and all seven family profiles publish stable feature identities and explicit executable or declaration-only dispositions.
- Effects: Publishes reviewed family/profile descriptors, feature-compatible translation and domain-overlay routes, and positive/negative/ambiguous/adversarial/round-trip/metamorphic fixtures with bounded parser/resource attacks per profile.
- Evidence subset: family registry profile catalog domain overlay route conformance fuzz unicode recursion ambiguity roundtrip
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Every family task has an exact registry/profile entry; every admitted new family-to-domain/provider route is reviewed, feature-compatible, and loss/authority receipted; registry presence alone never implies executability; every executable profile has representative fixtures and deterministic resource limits.
- Embedding query: logic profile conformance adversarial roundtrip fuzz parser bomb

## LFP2-045 Add the pinned process-backed provider validation tier

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance
- Depends on: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Goal id: LFP2-G080
- Outputs: ipfs_datasets_py/tests/integration/logic_providers/manifest.json, ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/integration/logic_providers/test_scheduled_provider_tier.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/conformance/providers
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-kernel
- Resource stage: validation
- Estimated tokens: 28000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_datasets_py/tests/integration/logic_providers/manifest.json, ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py
- Interfaces: ScheduledProviderTier@1
- Allow concurrent with: LFP2-044
- Conflict policy: Own scheduled manifest/harness; provider-specific fixtures remain provider task outputs.
- Preconditions: All listed provider execution adapters publish toolchain contracts.
- Effects: Runs real pinned binaries when available and emits typed unavailable receipts otherwise; separates hermetic and scheduled evidence.
- Evidence subset: provider subprocess pinned binary environment availability receipt
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: No metadata-only or mock run satisfies executable capability; exact command/environment/tool digest/output identity is recorded without secrets.
- Embedding query: process backed solver test pinned toolchain provider validation

## LFP2-046 Join process-backed vertical slices, differential alignment, replay, and reconstruction

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance
- Depends on: LFP2-022, LFP2-023, LFP2-024, LFP2-025, LFP2-026, LFP2-027, LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036, LFP2-045
- Goal id: LFP2-G080
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/replay_v2.py, ipfs_datasets_py/tests/conformance/logic/test_replay_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_replay_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/conformance/replay
- Parallel lane: lfp2-evidence
- Resource class: cpu-proof-kernel
- Resource stage: validation
- Estimated tokens: 30000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/replay_v2.py, ipfs_datasets_py/tests/conformance/logic/test_replay_v2.py
- Interfaces: LogicEvidenceReplay@1, ExecutableVerticalSliceReceipt@1
- Allow concurrent with: LFP2-044
- Conflict policy: Own evidence replay orchestrator/tests; never overwrite provider-native artifacts.
- Preconditions: Domain slices and provider decoders expose stable evidence identities, and LFP2-045 supplies pinned process-backed provider receipts.
- Effects: Emits a content-bound domain-source to parse to elaborate to translate to compile to real pinned-process to decode to replay/reconstruction receipt, then matches semantic fragments for differential checks and replays models, cores, traces, attacks, witnesses, TSTP/proof certificates, and kernel candidates.
- Evidence subset: differential model core trace attack witness proof reconstruction replay
- Symbolic first: true
- LLM context budget bytes: 44000
- Acceptance: Static or hermetic metadata cannot satisfy ExecutableVerticalSliceReceipt@1; disagreement is preserved; every authority-bearing result has independent replay/reconstruction or a typed ceiling that forbids promotion.
- Embedding query: solver differential witness model trace attack proof replay

## LFP2-047 Seal the reachable conformance matrix and hard-zero floors

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: conformance
- Depends on: LFP2-044, LFP2-045, LFP2-046
- Goal id: LFP2-G080
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/matrix_v2.py, ipfs_datasets_py/tests/conformance/logic/test_reachable_matrix_v2.py, ipfs_datasets_py/data/logic/conformance/reachable_matrix_v2.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_reachable_matrix_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/conformance/join
- Parallel lane: lfp2-evidence
- Resource class: cpu-proof-kernel
- Resource stage: validation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/matrix_v2.py, ipfs_datasets_py/tests/conformance/logic/test_reachable_matrix_v2.py, ipfs_datasets_py/data/logic/conformance/reachable_matrix_v2.json
- Interfaces: ReachableConformanceMatrix@2, LogicConformanceReport@2
- Allow concurrent with:
- Conflict policy: Join task owns matrix/report; source artifacts remain immutable inputs.
- Preconditions: Corpus, scheduled execution, and replay evidence share current identities.
- Effects: Joins domain source, profile, translation path, provider feature, execution, replay, disposition, and authority into a sparse matrix.
- Evidence subset: reachable matrix domain translation provider replay hard zero
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: Zero unexplained reachable gap, silent node drop/loss, raw ingress, family drift, false capability, authority escalation, or kernel trust escape.
- Embedding query: reachable conformance matrix hard zero authority logic

## LFP2-048 Implement reachable-gap scoring and strict derived-task admission

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: objective-refill
- Depends on: LFP2-005, LFP2-009
- Goal id: LFP2-G090
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill_v2.py, ipfs_datasets_py/tests/unit/logic/conformance/test_refill_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_refill_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/refill/scorer
- Parallel lane: lfp2-refill
- Resource class: cpu-medium
- Resource stage: control
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill_v2.py, ipfs_datasets_py/tests/unit/logic/conformance/test_refill_v2.py
- Interfaces: ReachableGapScorer@1, DerivedTaskAdmission@2
- Allow concurrent with:
- Conflict policy: Own scorer/admission logic; objective heap and seed task definitions are immutable.
- Preconditions: Baseline lifecycle and provider contracts exist.
- Effects: Scores reproducible reachable gaps and requires content identity, evidence obligation, discovery, ownership, dependencies, scope, validation, dedupe, budget, and authority ceiling.
- Evidence subset: refill gap score content identity derived task admission dedupe
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Cartesian unsupported, advisor-only, vague cleanup, duplicate, unsafe, protected, or broad tasks are rejected before append.
- Embedding query: objective refill reachable gap scorer derived task admission

## LFP2-049 Establish two quiet reachable-gap epochs at current identity

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: true
- Provider role: deterministic-only
- Priority: P0
- Track: objective-refill
- Depends on: LFP2-047, LFP2-048
- Goal id: LFP2-G090
- Outputs: data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill/fixed_point_receipt.json, data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill/gap_ledger.jsonl
- Validation: PYTHONPATH=ipfs_datasets_py python -m ipfs_datasets_py.logic.conformance.fixed_point_v2 materialize --repo-root . --fixed-point-path data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill/fixed_point_receipt.json --ledger-path data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill/gap_ledger.jsonl && PYTHONPATH=ipfs_datasets_py python scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py --check-all
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/refill/fixed-point
- Parallel lane: lfp2-refill
- Resource class: cpu-medium
- Resource stage: control
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill/fixed_point_receipt.json, data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill/gap_ledger.jsonl
- Interfaces: ObjectiveRefillFixedPoint@2
- Allow concurrent with:
- Conflict policy: Fixed-point evidence is append-only/content-addressed and cannot mutate seed goals/tasks or semantic evidence.
- Preconditions: Reachable matrix hard-zero gates pass; every task other than LFP2-049 and LFP2-050 is terminal; and no admitted derived task remains open.
- Effects: Runs two bounded scans over identical source, registry, corpus, provider, objective, and reachable-matrix identities.
- Evidence subset: refill quiet epoch fixed point source identity zero open
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Both scans admit no task, no reachable P0/P1 gap remains, limits hold, and seed definitions are unchanged.
- Embedding query: objective refill fixed point quiet epochs reachable gaps

## LFP2-050 Seal the Wave-2 logic parser release receipt

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: true
- Provider role: deterministic-only
- Priority: P0
- Track: release
- Depends on: LFP2-049
- Goal id: LFP2-G100
- Outputs: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_V2_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_v2_release.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic tests/conformance/logic tests/fuzz/logic && python -m ipfs_datasets_py.logic.conformance.release_v2 materialize --repo-root .. --json-path data/logic/conformance/logic_family_parser_v2_release.json --markdown-path docs/architecture/logic/LOGIC_FAMILY_PARSER_V2_RELEASE.md && cd .. && PYTHONPATH=ipfs_datasets_py python scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py --check-all
- Board namespace: ipfs-datasets-logic-family-parser-v2
- Bundle: logic-family-parser-v2/release
- Parallel lane: lfp2-release
- Resource class: cpu-proof-kernel
- Resource stage: release
- Estimated tokens: 24000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_V2_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_v2_release.json
- Interfaces: LogicParserReleaseReceipt@2
- Allow concurrent with:
- Conflict policy: Review/evidence aggregation only; semantic repairs require a derived task under the owning goal.
- Preconditions: Every task other than LFP2-050 is terminal, the fixed point is current, hard-zero floors pass, and predecessor hashes match.
- Effects: Binds exact v1 predecessor, v2 source, schemas, parsers, translations, domains, providers, toolchains, execution/replay evidence, matrix, dispositions, and authority floors.
- Evidence subset: release predecessor identity schema translation provider replay fixed point authority
- Symbolic first: true
- LLM context budget bytes: 46000
- Acceptance: Release is reproducible and current; it grants no mutation or theorem authority; no open task, stale scan, altered predecessor, unexplained reachable gap, silent loss, false capability, or trust escape remains.
- Embedding query: logic parser wave2 release receipt predecessor replay authority
