# IPFS Datasets Logic-Family Parser Wave-2 Objective Heap

This is the immutable goal/subgoal hierarchy for the `LFP2-` program. The
normative design is `IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md`; executable
seed and derived work is projected into
`ipfs_datasets_logic_family_parser_v2.todo.md`.

Task IDs are never accepted as evidence. `Seed tasks` declares ownership;
`Evidence` declares content-bound interfaces, reports, tests, receipts, and
runtime outcomes that the gap scanner must find or generate owner-scoped work
for.

Program invariants:

- Wave-1 plans, goals, board, fixed-point receipt, and release receipt remain
  immutable predecessor evidence;
- semantic family, profile, property, view, notation, encoding, provider,
  execution lane, and evidence kind remain distinct typed namespaces;
- extension nodes, backend requests, compiled source, decoded target output,
  and provider capabilities are schema governed;
- domain IRs remain domain-rich and emit source-mapped typed formal views;
- translations record preservation, assumptions, loss, polarity, bounds,
  reconstruction, and authority;
- real provider execution never receives more authority than its evidence
  kind and pinned environment allow;
- UI/UX work remains exact-source gated;
- static goals are immutable and objective refill appends only
  content-addressed, owner-scoped tasks; and
- completion requires a reachable-path refill fixed point and release receipt.

## LFP2-G000 Deliver executable, replayable, extensible IR-to-logic-to-prover paths

- Status: active
- Review only: true
- Parent:
- Depends on:
- Fib priority: 1
- Track: wave2-program
- Priority: P0
- Bundle: logic-family-parser-v2/control
- Parallel lane: lfp2-control
- Resource class: cpu-large
- Goal: Convert the Wave-1 typed parsing foundation into executable and replayable domain-IR, translation, and theorem-prover paths, then add high-value logic families without silent loss or authority escalation.
- Subgoals: LFP2-G010, LFP2-G020, LFP2-G030, LFP2-G040, LFP2-G050, LFP2-G060, LFP2-G070, LFP2-G080, LFP2-G090, LFP2-G100
- Seed tasks: LFP2-000
- Evidence: LogicParserReleaseReceipt@2, reachable-capability-matrix-v2, provider-execution-receipts-v2, objective-refill-fixed-point-v2
- Evidence criteria: Every child has current-tree content-bound evidence and LFP2-050 binds exact source, schemas, translations, providers, replay validations, dispositions, and authority floors.
- Evidence source policy: Reviewed typed contracts define claims; real parser, translator, provider, replay, and official-kernel results supply bounded evidence only.
- Outputs: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json
- Predicted files: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json
- Interfaces: LogicParserWave2Program@1, LogicParserReleaseReceipt@2
- Validation: python scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py --check-all
- Acceptance: Every reachable admitted IR-to-provider path is typed, receipted, executable or explicitly dispositioned; no Wave-1 anchor changes; all seed and derived work is terminal.
- Gap task: Aggregate child evidence and create only owner-scoped tasks for missing current-tree obligations; never use task-ID presence as evidence.
- Refinement: Prefer explicit unsupported, unavailable, bounded, approximate, advisory, source-missing, or approval-required dispositions over an unearned proof or equivalence.
- Embedding query: wave2 typed logic parser domain ir translation prover replay authority refill
- AST query: TypedExpression BackendRequest CompiledLogicArtifact TranslationPathReceipt ProviderExecutionReceipt
- Conflict policy: Root is review/evidence aggregation only; child goals own semantic implementation and LFP2-050 owns the joined release receipt.

## LFP2-G010 Rebaseline claims, raw boundaries, reachability, and corpus evidence

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on:
- Fib priority: 1
- Track: evidence-baseline
- Priority: P0
- Bundle: logic-family-parser-v2/baseline
- Parallel lane: lfp2-baseline
- Resource class: cpu-medium
- Goal: Replace declaration-derived closure with a current-tree audit of actual typed stages, raw boundaries, reachable paths, and replay fixtures.
- Seed tasks: LFP2-001, LFP2-002, LFP2-003, LFP2-004
- Evidence: LogicClaimRuntimeAudit@1, RawLogicBoundaryInventory@1, ReachableCapabilityGraph@1, LogicConformanceCorpus@2
- Evidence criteria: Reports bind exact source revisions and distinguish declared, parsed, elaborated, translatable, compilable, executable, replayed, and independently validated states.
- Evidence source policy: Static source and registries establish inventory; only executable tests and receipts establish runtime lifecycle states.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline, ipfs_datasets_py/tests/fixtures/logic_conformance_v2
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance, ipfs_datasets_py/docs/architecture/logic/logic_parser_v2_baseline, ipfs_datasets_py/tests/fixtures/logic_conformance_v2
- Interfaces: LogicClaimRuntimeAudit@1, RawLogicBoundaryInventory@1, ReachableCapabilityGraph@1, LogicConformanceCorpus@2
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance
- Acceptance: Every Wave-1 unimplemented/declaration-only/executable claim and raw source boundary has an owner, lifecycle state, reachable-path disposition, and content identity.
- Gap task: Add one missing audit row or fixture with an exact source path and evidence obligation.
- Refinement: Do not turn the full Cartesian unsupported matrix into work; prioritize reachable domain requirements and declared executable claims.
- Embedding query: logic claim runtime audit raw boundary reachable capability corpus replay
- AST query: BackendRequest ParseArtifact TranslationReceipt ProviderCapabilityDescriptor
- Conflict policy: Four seed tasks own distinct report, inventory, graph, and corpus modules and may run concurrently.

## LFP2-G020 Enforce typed syntax, formalization, request, evidence, capability, and migration contracts

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G010
- Fib priority: 1
- Track: shared-contracts
- Priority: P0
- Bundle: logic-family-parser-v2/contracts
- Parallel lane: lfp2-contracts
- Resource class: cpu-large
- Goal: Eliminate arbitrary extension payloads, unversioned parse/elaboration/formalization/domain slices, free-form backend family requests, raw target or result ingress, duplicated capability claims, and flag-day migration risk.
- Seed tasks: LFP2-005, LFP2-006, LFP2-007, LFP2-008, LFP2-009
- Evidence: ExtensionSchemaRegistry@1, ParseArtifact@2, ElaborationArtifact@2, FormalizationArtifact@3, DomainLogicSlice@2, LogicObligation@2, BackendRequest@2, CompiledLogicArtifact@1, ParsedTargetArtifact@1, ProviderExecutionReceipt@2, EvidenceReplayReceipt@1, ProviderCapabilityMatrix@2, LogicContractMigration@1
- Evidence criteria: Type/binder/scope hooks and stable codecs cover extensions and syntax artifacts; formalization, domain slices, requests, execution, replay, and provider entries are canonical; legacy inputs emit explicit migration receipts.
- Evidence source policy: Contract tests and schema validation establish structure; they do not establish solver correctness.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core, ipfs_datasets_py/ipfs_datasets_py/logic/formalization, ipfs_datasets_py/ipfs_datasets_py/logic/backends, ipfs_datasets_py/ipfs_datasets_py/logic/families
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core, ipfs_datasets_py/ipfs_datasets_py/logic/formalization, ipfs_datasets_py/ipfs_datasets_py/logic/backends, ipfs_datasets_py/ipfs_datasets_py/logic/families
- Interfaces: ExtensionSchemaRegistry@1, ParseArtifact@2, ElaborationArtifact@2, FormalizationArtifact@3, DomainLogicSlice@2, BackendRequest@2, CompiledLogicArtifact@1, ProviderExecutionReceipt@2, EvidenceReplayReceipt@1, ProviderCapabilityMatrix@2
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core tests/unit/logic/formalization tests/unit/logic/backends tests/unit/logic/families
- Acceptance: No new arbitrary JSON extension, unversioned syntax/formalization slice, free-form family route, raw unreceipted target/result, or hand-duplicated provider capability enters an executable path.
- Gap task: Repair one exact contract/schema/migration hole and add negative evidence for the previously admitted unsafe input.
- Refinement: Preserve dual-read compatibility until consumer migration evidence is complete; canonical-write is immediate.
- Embedding query: extension schema backend request compiled artifact provider capability migration
- AST query: LogicExtensionNode BackendRequest CompiledLogicArtifact ProviderCapabilityDescriptor
- Conflict policy: Extension, request, artifact, provider, and migration modules have separate owners; shared exports join after their contracts land.

## LFP2-G030 Converge every controlled frontend on shared parse and elaboration artifacts

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G020
- Fib priority: 2
- Track: frontend-convergence
- Priority: P0
- Bundle: logic-family-parser-v2/frontends
- Parallel lane: lfp2-parsers
- Resource class: cpu-large
- Goal: Make classical, rule, frame, protocol, program, temporal, modal, resource, TDFOL, and DCEC frontends emit common source-aware typed artifacts with explicit feature limits.
- Seed tasks: LFP2-010, LFP2-011, LFP2-012, LFP2-013, LFP2-014, LFP2-015
- Evidence: SharedFrontendConformance@1, SMTLIBFrontend@2, TPTPFrontend@2, RuleFrameFrontend@2, ProtocolProgramFrontend@2, LogicProfileCatalog@2, LegacyLogicBoundary@2
- Evidence criteria: Every controlled frontend has positive/negative/round-trip/resource fixtures and no parser bypasses source/CST/elaboration diagnostics.
- Evidence source policy: Parse/elaboration tests prove only controlled syntax and static semantics.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers, ipfs_datasets_py/tests/unit/logic/parsers
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers, ipfs_datasets_py/tests/unit/logic/parsers
- Interfaces: ParseArtifact@2, ElaborationArtifact@2, SharedFrontendConformance@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers
- Acceptance: No admitted controlled parser silently skips source, loses a binder/sort/operator, or emits an untyped raw formula.
- Gap task: Convert one exact bypassing frontend or unsupported construct and add a bounded conformance fixture.
- Refinement: Do not claim complete SMT-LIB, TPTP, Tamarin, ErgoAI, TLA+, Lean, Rocq, or Isabelle language parsing.
- Embedding query: frontend convergence parse artifact elaboration smtlib tptp rules protocol program legacy
- AST query: ParseArtifact ElaborationArtifact SourceMap TypedExpression
- Conflict policy: Frontend tasks own disjoint modules; shared registry/export changes depend on the relevant frontend tasks.

## LFP2-G040 Build a compositional, loss-aware translation graph

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G020, LFP2-G030
- Fib priority: 2
- Track: translation-graph
- Priority: P0
- Bundle: logic-family-parser-v2/translations
- Parallel lane: lfp2-translations
- Resource class: cpu-proof-solver
- Goal: Add reviewed executable translations for program, state, authorization, symbolic protocol, modal/cognitive, hyperproperty, and composed routes.
- Seed tasks: LFP2-016, LFP2-017, LFP2-018, LFP2-019, LFP2-020, LFP2-021
- Evidence: LogicTranslationGraph@3, TranslationPathPlanner@1, TranslationPathReceipt@1, ProtocolTargetTranslationEdges@1
- Evidence criteria: Each edge and composed path binds feature preconditions, preservation, polarity, assumptions, losses, bounds, reconstruction, and authority ceiling.
- Evidence source policy: Metamorphic/differential/reconstruction tests validate declared properties; no test promotes an approximation to equivalence.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/translations_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations, ipfs_datasets_py/tests/unit/logic/translations
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/translations_v3.py, ipfs_datasets_py/ipfs_datasets_py/logic/translations, ipfs_datasets_py/tests/unit/logic/translations
- Interfaces: LogicTranslationGraph@3, TranslationPathReceipt@1, TranslationPathPlanner@1, ProtocolTargetTranslationEdges@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/translations tests/conformance/logic/test_translation_paths_v2.py
- Acceptance: Every selected path is feature-compatible and loss-receipted; unsupported compositions fail before backend dispatch.
- Gap task: Add one reachable missing edge/path with preservation and negative feature fixtures.
- Refinement: Prefer short reviewed paths; prevent authority or approximation laundering through composition.
- Embedding query: logic translation graph program state authorization modal hyperproperty planner
- AST query: TranslationDescriptor TranslationPathReceipt FeatureSet AuthorityCeiling
- Conflict policy: Each semantic edge family owns a module; the planner joins only registered edge descriptors.

## LFP2-G050 Replace domain Boolean receipts with typed executable vertical slices

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G030, LFP2-G040
- Fib priority: 2
- Track: domain-vertical-slices
- Priority: P0
- Bundle: logic-family-parser-v2/domains
- Parallel lane: lfp2-domains
- Resource class: cpu-large
- Goal: Connect security, crypto, intent, legal, UI, and software-verification claims to typed expressions, translation receipts, backend requests, results, and replay evidence.
- Seed tasks: LFP2-022, LFP2-023, LFP2-024, LFP2-025, LFP2-026, LFP2-027
- Evidence: SecurityLogicSlice@2, CryptoLogicSlice@2, IntentLogicSlice@2, LegalLogicSlice@2, UIUXLogicSlice@2, SoftwareVerificationLogicSlice@2
- Evidence criteria: Source ranges join domain claims to typed expressions, translations, requests, provider results, and bounded authority receipts.
- Evidence source policy: Domain adapters define obligation identity and source mapping; provider-specific validation defines only its declared evidence kind.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir, ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir, ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir, ipfs_datasets_py/ipfs_datasets_py/logic/conformance, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir, ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir, ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir, ipfs_datasets_py/ipfs_datasets_py/logic/conformance, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification
- Interfaces: DomainLogicSlice@2, FormalizationArtifact@3
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/security_ir tests/unit/logic/crypto_ir tests/unit/logic/intent_ir tests/unit/logic/legal_ir tests/unit/logic/software_verification tests/conformance/logic
- Acceptance: Every admitted domain route uses canonical identifiers and typed artifacts; UI remains source-gated when absent; no Boolean-only receipt claims semantic round-trip or proof.
- Gap task: Fill one reachable domain-view route or emit a typed explicit disposition tied to exact source identity.
- Refinement: Preserve domain richness; the common logic kernel is not a replacement security, ledger, intent, legal, UI, or program model.
- Embedding query: security crypto intent legal ui software verification typed logic vertical slice
- AST query: FormalizationArtifact TypedExpression TranslationPathReceipt BackendRequest
- Conflict policy: Each domain task owns its package adapter/tests; UI task cannot create or edit absent ui_ux_ir source.

## LFP2-G060 Execute, decode, and replay every admitted provider family

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G020, LFP2-G030, LFP2-G040
- Fib priority: 2
- Track: provider-execution
- Priority: P0
- Bundle: logic-family-parser-v2/providers
- Parallel lane: lfp2-providers
- Resource class: cpu-proof-solver
- Goal: Establish typed request, pinned launch, result decode, and replay/reconstruction paths for every provider listed in the program.
- Seed tasks: LFP2-028, LFP2-029, LFP2-030, LFP2-031, LFP2-032, LFP2-033, LFP2-034, LFP2-035, LFP2-036
- Evidence: SMTProviderEvidence@2, StateProviderEvidence@2, RuleProviderEvidence@2, ProtocolProviderEvidence@2, HyperProviderEvidence@2, ATPProviderEvidence@2, KernelProviderEvidence@2, AdvisorProviderEvidence@2, RuntimeMTLEvidence@2
- Evidence criteria: Each provider has a hermetic contract tier plus a real pinned optional-tool tier, evidence-specific decoder, and replay or explicit non-replay disposition.
- Evidence source policy: Tool identity and real subprocess output are required for execution claims; official kernels alone provide kernel authority.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends, ipfs_datasets_py/tests/integration/logic_providers, ipfs_datasets_py/tests/conformance/logic
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends, ipfs_datasets_py/tests/integration/logic_providers, ipfs_datasets_py/tests/conformance/logic
- Interfaces: ProviderExecutionReceipt@2, EvidenceReplayReceipt@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/backends tests/conformance/logic
- Acceptance: No provider is called through untyped raw ingress; no unavailable tool is replaced by authoritative mock evidence; all admitted results carry replay/reconstruction and authority metadata.
- Gap task: Add one missing provider feature/execution/replay fixture bound to a reachable domain and translation path.
- Refinement: Split aggregated providers and strategy phases; never infer one engine's capabilities from another engine in the same adapter.
- Embedding query: z3 cvc5 tlc apalache secpal proverif tamarin hyperltl vampire eprover lean rocq isabelle ergoai symai runtime mtl
- AST query: BackendRequest ProviderExecutionReceipt ParsedTargetArtifact EvidenceReplayReceipt
- Conflict policy: Provider-family tasks own separate backend/test modules; shared process and result contracts are owned by G020.

## LFP2-G070 Add high-value normative, ontology, agency, fixed-point, finite-field, and process families

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G020, LFP2-G030
- Fib priority: 3
- Track: family-expansion
- Priority: P1
- Bundle: logic-family-parser-v2/families
- Parallel lane: lfp2-families
- Resource class: cpu-large
- Goal: Add typed semantics and controlled routes for domain-driven families without declaring unsupported probabilistic, fuzzy, paraconsistent, or unrestricted higher-order coverage.
- Seed tasks: LFP2-037, LFP2-038, LFP2-039, LFP2-040, LFP2-041, LFP2-042, LFP2-043
- Evidence: NormativeLogicProfiles@2, ArgumentationLogic@1, DescriptionLogicProfiles@1, AgencyLogicProfiles@1, FixedPointLogicProfiles@1, FiniteFieldConstraintLogic@1, SessionProcessLogic@1
- Evidence criteria: Each family has typed nodes, profile semantics, parser/printer fixtures, at least one domain view, and a sound executable or explicit declaration-only route.
- Evidence source policy: Domain examples motivate profiles; formal contracts and provider validation establish only declared fragments.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers, ipfs_datasets_py/ipfs_datasets_py/logic/families, ipfs_datasets_py/ipfs_datasets_py/logic/translations
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers, ipfs_datasets_py/ipfs_datasets_py/logic/families, ipfs_datasets_py/ipfs_datasets_py/logic/translations
- Interfaces: NormativeLogicProfiles@2, ArgumentationLogic@1, DescriptionLogicProfiles@1, AgencyLogicProfiles@1, FixedPointLogicProfiles@1, FiniteFieldConstraintLogic@1, SessionProcessLogic@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families tests/unit/logic/parsers tests/unit/logic/translations
- Acceptance: New executable claims are profile-scoped and tested; remaining candidate families stay typed declaration-only with no provider claim.
- Gap task: Add one domain-required family profile or executable edge with exact semantics and negative unsupported fixtures.
- Refinement: Do not implement a family merely to fill a Cartesian matrix cell.
- Embedding query: defeasible argumentation description logic bdi mu calculus finite field session process
- AST query: LogicFamilyDescriptor LogicProfile TypedExpression TranslationDescriptor
- Conflict policy: Each family owns a distinct parser/profile/translation module; registry publication joins only after individual tests pass.

## LFP2-G080 Prove reachable parser, translation, provider, replay, and authority conformance

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G050, LFP2-G060, LFP2-G070
- Fib priority: 2
- Track: conformance
- Priority: P0
- Bundle: logic-family-parser-v2/conformance
- Parallel lane: lfp2-conformance
- Resource class: cpu-proof-solver
- Goal: Publish reviewed family/profile/domain routes and validate positive, negative, ambiguous, adversarial, process-backed, executable-vertical-slice, replay, reconstruction, and reachable-matrix evidence across the whole architecture.
- Seed tasks: LFP2-044, LFP2-045, LFP2-046, LFP2-047
- Evidence: LogicConformanceCorpus@2, LogicFamilyRegistry@3, LogicProfileCatalog@3, FamilyRoutePublication@1, ScheduledProviderTier@1, LogicEvidenceReplay@1, ExecutableVerticalSliceReceipt@1, ReachableConformanceMatrix@2
- Evidence criteria: Every reachable executable claim has corpus fixtures and pinned execution/replay evidence; hard-zero safety floors are machine checked.
- Evidence source policy: Hermetic tests prove contracts; scheduled pinned providers prove tool-specific execution; kernels prove only exact accepted theories.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families, ipfs_datasets_py/ipfs_datasets_py/logic/translations, ipfs_datasets_py/ipfs_datasets_py/logic/conformance, ipfs_datasets_py/tests/fixtures/logic_conformance_v2, ipfs_datasets_py/tests/integration/logic_providers, ipfs_datasets_py/tests/conformance/logic, ipfs_datasets_py/data/logic/conformance
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families, ipfs_datasets_py/ipfs_datasets_py/logic/translations, ipfs_datasets_py/ipfs_datasets_py/logic/conformance, ipfs_datasets_py/tests/fixtures/logic_conformance_v2, ipfs_datasets_py/tests/integration/logic_providers, ipfs_datasets_py/tests/conformance/logic, ipfs_datasets_py/data/logic/conformance
- Interfaces: LogicFamilyRegistry@3, LogicProfileCatalog@3, FamilyRoutePublication@1, ExecutableVerticalSliceReceipt@1, LogicConformanceReport@2, ReachableConformanceMatrix@2
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic tests/fuzz/logic
- Acceptance: Zero silent loss, raw ingress, family drift, false capability, authority escalation, trust escape, unexplained reachable gap, or unreplayed authoritative evidence remains.
- Gap task: Generate one content-addressed task for a failing reachable path, missing replay, or hard-zero violation.
- Refinement: A missing optional tool is an availability gap, not permission to weaken acceptance or substitute a mock proof.
- Embedding query: conformance corpus real provider process replay proof certificate reachable matrix authority
- AST query: ConformanceCase ProviderExecutionReceipt EvidenceReplayReceipt AuthorityCeiling
- Conflict policy: Corpus, scheduled execution, replay, and matrix tasks own distinct artifacts and join through content identities.

## LFP2-G090 Maintain bounded reachable-gap refill to a fixed point

- Status: active
- Review only: false
- Parent: LFP2-G000
- Depends on: LFP2-G080
- Fib priority: 3
- Track: objective-refill
- Priority: P0
- Bundle: logic-family-parser-v2/refill
- Parallel lane: lfp2-refill
- Resource class: cpu-medium
- Goal: Score and append only content-addressed reachable gaps, then establish two quiet epochs over identical source and evidence identities.
- Seed tasks: LFP2-048, LFP2-049
- Evidence: ReachableGapScorer@1, ObjectiveRefillFixedPoint@2, logic-parser-v2-gap-ledger
- Evidence criteria: Every admitted derived task has identity, discovery, owner, dependency lineage, bounded scope, validation, dedupe, and authority ceiling; two equal scans add none.
- Evidence source policy: Current source, registries, reachable graph, corpus, provider receipts, and objectives are discovery inputs; empty scans do not grant proof authority.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill_v2.py, data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill_v2.py, data/agent_supervisor/ipfs_datasets_logic_family_parser_v2/refill
- Interfaces: ReachableGapScorer@1, ObjectiveRefillFixedPoint@2
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_refill_v2.py
- Acceptance: Refill limits hold, seed goals/tasks remain immutable, duplicates and broad tasks are rejected, and two identical quiet epochs have no admissible reachable gap.
- Gap task: Repair only the exact scorer/admission/fixed-point evidence defect; never hand-author an appended card.
- Refinement: Static goals remain immutable; derived tasks attach to an existing goal and cannot rewrite the objective heap.
- Embedding query: objective refill reachable gap scorer content addressed fixed point dedupe
- AST query: ObjectiveFinding CanonicalTaskIdentity ReachableCapabilityGraph
- Conflict policy: Refill implementation and fixed-point evidence are serialized; generated tasks own only their declared domain paths.

## LFP2-G100 Seal the Wave-2 release and predecessor chain

- Status: active
- Review only: true
- Parent: LFP2-G000
- Depends on: LFP2-G090
- Fib priority: 5
- Track: release
- Priority: P0
- Bundle: logic-family-parser-v2/release
- Parallel lane: lfp2-release
- Resource class: cpu-proof-kernel
- Goal: Bind the immutable Wave-1 predecessor, Wave-2 source, schemas, translations, domains, providers, replay evidence, reachable matrix, fixed point, dispositions, and authority floors.
- Seed tasks: LFP2-050
- Evidence: LogicParserReleaseReceipt@2, immutable-v1-predecessor-binding, wave2-release-validation
- Evidence criteria: The release receipt transitively covers every seed task and open derived work is zero; predecessor hashes and all hard-zero floors validate.
- Evidence source policy: Release aggregates reviewed evidence and grants no mutation, theorem, policy, or kernel authority by itself.
- Outputs: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_V2_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_v2_release.json
- Predicted files: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_V2_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_v2_release.json
- Interfaces: LogicParserReleaseReceipt@2
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic tests/conformance/logic tests/fuzz/logic
- Acceptance: All seed and derived tasks are terminal, the fixed point is current, v1 anchors are unchanged, and exact current-tree evidence satisfies every child goal without authority escalation.
- Gap task: Aggregate or reject release; implementation changes belong to a reopened child obligation and a derived task.
- Refinement: Never edit evidence to make a red floor green; repair the owning semantic path.
- Embedding query: logic parser wave2 release receipt predecessor provider replay fixed point authority
- AST query: LogicParserReleaseReceipt ReachableConformanceMatrix ObjectiveRefillFixedPoint
- Conflict policy: Review/evidence aggregation only; no semantic implementation changes are allowed in the release task.
