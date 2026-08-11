# IPFS Datasets Logic-Family Parser Objective Heap

This is the durable goal/subgoal hierarchy for the `LFP-` program. The
normative design is `IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md`; executable
work is projected into `ipfs_datasets_logic_family_parser.todo.md`.

Program invariants:

- semantic family, profile, property, view, notation, encoding, provider,
  execution lane, and evidence kind are separate versioned namespaces;
- no free-form family identifier reaches routing or a proof claim;
- domain IRs remain domain-rich and lower through typed adapters;
- every translation declares preservation, assumptions, loss, bounds,
  source maps, and an authority ceiling;
- solver, advisor, model, and parser results never exceed their evidence
  contract;
- official proof-assistant kernels remain the authority for kernel proofs;
- ErgoAI and SymbolicAI remain proposal/advisor surfaces until independent
  deterministic verification;
- nested changes land as an exact datasets commit followed by a serialized
  accelerator gitlink update; and
- completion requires a current-tree vertical-slice and refill fixed point.

## LFP-G000 Deliver a typed, extensible, evidence-safe logic parsing and prover bridge architecture

- Status: active
- Review only: true
- Parent:
- Depends on:
- Fib priority: 1
- Track: logic-parser-program
- Priority: P0
- Bundle: logic-family-parser/control
- Parallel lane: lfp-control
- Resource class: cpu-large
- Goal: Make every admitted domain IR formal view parse, elaborate, translate, route, execute, decode, and validate through canonical logic-family/profile contracts without silent semantic loss or authority upgrades.
- Subgoals: LFP-G010, LFP-G020, LFP-G030, LFP-G040, LFP-G050, LFP-G060, LFP-G070, LFP-G080, LFP-G090, LFP-G100
- Evidence: LFP-G010, LFP-G020, LFP-G030, LFP-G040, LFP-G050, LFP-G060, LFP-G070, LFP-G080, LFP-G090, LFP-G100
- Evidence criteria: Every child goal has a current-tree evidence bundle and LFP-047 binds their exact source, translation, provider, validation, and authority identities.
- Evidence source policy: Reviewed syntax, taxonomy, translation, and provider contracts define claims; parsers, solvers, kernels, tests, models, graphs, and embeddings provide bounded evidence only.
- Outputs: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_scheduler.json
- Predicted files: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_scheduler.json
- Interfaces: LogicSyntaxCore@1, LogicFamilyRegistry@2, TranslationContract@2, LogicParserReleaseReceipt@1
- Validation: python scripts/validate_ipfs_datasets_logic_family_parser_board.py --check-all
- Acceptance: LFP-000 through LFP-047 are terminal with zero unregistered emitted family IDs, silent node drops, unsupported-semantics promotion, advisor authority escalation, kernel trust escape, false solver capability claim, or unexplained matrix gap.
- Gap task: Aggregate child evidence and decide release; do not implement family or backend semantics at the root.
- Refinement: Prefer explicit unsupported, bounded, approximate, inconclusive, declaration-only, or approval-required dispositions over an unearned equivalence or proof claim.
- Embedding query: formal logic parser typed ast family profile translation theorem prover domain ir security crypto intent legal ui
- AST query: LogicFamilyRegistry ParseArtifact TypedExpression TranslationContract ProviderCapabilityDescriptor
- Conflict policy: Root is review/evidence aggregation only; child goals own code and LFP-047 owns the joined immutable receipt.

## LFP-G010 Freeze the parser, family, provider, translation, and corpus baseline

- Status: active
- Parent: LFP-G000
- Depends on:
- Fib priority: 1
- Track: inventory
- Priority: P0
- Bundle: logic-family-parser/inventory
- Parallel lane: lfp-inventory
- Resource class: cpu-medium
- Goal: Inventory every parser, AST, family/profile label, domain view, backend capability, translation, result decoder, fixture, and public consumer before changing shared contracts.
- Evidence: LFP-001, LFP-002, LFP-003, LFP-004, LFP-005
- Evidence criteria: Static inventories are exhaustive under explicit roots and alias rules; the corpus and matrix are content addressed; the join records all unresolved drift.
- Evidence source policy: Git trees and reviewed overlays are inventory authority; import success, registry membership, documentation, mocks, and old receipts do not prove live semantics.
- Outputs: ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline, ipfs_datasets_py/tests/fixtures/logic_conformance
- Predicted files: ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline, ipfs_datasets_py/tests/fixtures/logic_conformance
- Interfaces: LogicSurfaceInventory@1, LogicConformanceCorpus@1, LogicCapabilityMatrix@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families tests/unit/logic/backends
- Acceptance: Existing 21 canonical families, all aliases, domain labels, providers, parser islands, translation edges, authority ceilings, missing readers, and UI/ErgoAI/SymbolicAI boundaries are recorded with exact paths and revisions.
- Gap task: Create only missing inventory/corpus/matrix evidence; do not refactor production parsers in this goal.
- Refinement: A name found in code is an observed label, not automatically a canonical semantic family.
- Embedding query: parser ast family provider translation inventory corpus matrix alias drift
- AST query: DEFAULT_REGISTRY EXECUTABLE_PROVIDER_MATRIX FormalFormula LogicFamily
- Conflict policy: Foundation tasks own new reports, fixtures, and audit tools in disjoint paths; production migration begins after LFP-005.

## LFP-G020 Converge canonical family, profile, provider, and translation vocabularies

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G010
- Fib priority: 1
- Track: taxonomy
- Priority: P0
- Bundle: logic-family-parser/taxonomy
- Parallel lane: lfp-taxonomy
- Resource class: cpu-medium
- Goal: Publish one versioned semantic vocabulary, provider/translation schemas, and exact baseline descriptors that reject or diagnose every legacy overload; the final generated projection is sealed after parser/domain integration in LFP-040.
- Evidence: LFP-006, LFP-007, LFP-008, LFP-009, LFP-010
- Evidence criteria: Baseline registry closure, alias migration, semantic profiles, preservation relations, every exact executable provider ID, and provider descriptor schemas are validated from one source and round-trip through canonical JSON/CIDs.
- Evidence source policy: Reviewed descriptors define advertised support; provider executables and tests establish availability/evidence, not undeclared semantics.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families, ipfs_datasets_py/tests/unit/logic/families
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families, ipfs_datasets_py/tests/unit/logic/families
- Interfaces: LogicFamilyRegistry@2, SemanticProfile@1, TranslationContract@2, ProviderCapabilityCatalog@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families tests/unit/logic/backends/test_registry.py
- Acceptance: No provider, view role, property, notation, or execution lane is emitted as a family; all legacy labels have a typed canonical/unsupported disposition; baseline providers and translation edges validate against registered families/fragments and expose extension points for LFP-040 final closure.
- Gap task: Add only vocabulary, migration, descriptor, and closure evidence necessary for canonical routing.
- Refinement: Do not add a new family merely to preserve a historical string; model semantic variation as composition or a profile when appropriate.
- Embedding query: canonical logic family semantic profile alias provider capability translation preservation
- AST query: LogicFamilyDescriptor FamilySupportDescriptor TranslationDescriptor ProviderCapabilityDescriptor
- Conflict policy: Taxonomy files are serialized through LFP-010; backend implementations remain read-only until generated descriptors stabilize.

## LFP-G030 Build the source-aware typed syntax and elaboration kernel

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G020
- Fib priority: 1
- Track: syntax-core
- Priority: P0
- Bundle: logic-family-parser/syntax-core
- Parallel lane: lfp-syntax-core
- Resource class: cpu-large
- Goal: Implement a bounded, immutable, source-aware common syntax core with binding, sorts, signatures, diagnostics, codecs, algebra, and modular extension protocols.
- Evidence: LFP-011, LFP-012, LFP-013, LFP-014, LFP-015, LFP-016
- Evidence criteria: Golden and property tests cover source mapping, parser registry, alpha-equivalence, substitution, normalization, typing, limits, deterministic identity, and lazy import behavior.
- Evidence source policy: Typed elaboration establishes syntax and static semantics only; it never proves truth, satisfiability, safety, or authorization.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/views.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/constraint_contracts.py, ipfs_datasets_py/tests/unit/logic/syntax_core
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/views.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/constraint_contracts.py, ipfs_datasets_py/tests/unit/logic/syntax_core
- Interfaces: SourceDocument@1, LogicCST@1, TypedExpression@1, ParseArtifact@1, ElaborationArtifact@1, LazyParserPublication@1, FormalizationArtifact@2, ConstraintContract@2
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core tests/unit/logic/formalization/test_typed_expression_bridge.py tests/unit/logic/formalization/test_constraint_contracts.py
- Acceptance: Core supports propositional and many-sorted FOL nodes plus versioned extension nodes; capture is impossible under tested substitution; every diagnostic has stable code/span; all limits fail closed; semantic hashes ignore irrelevant source trivia only where declared.
- Gap task: Add missing core algebra or bounded contracts, not family-specific parser behavior.
- Refinement: Keep the kernel small and extensible; do not collapse domain IRs or every proof-assistant construct into a monolithic enum.
- Embedding query: source span cst ast typed expression binder sort signature elaborator parser registry
- AST query: SourceDocument ParseRequest ParseArtifact TypedExpression Signature Diagnostic
- Conflict policy: Syntax-core modules are split by contracts, algebra, registry, and codec; shared public exports are joined only in LFP-016.

## LFP-G040 Add classical, rule, authorization, and frame-logic frontends

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G030
- Fib priority: 2
- Track: classical-rule-parsers
- Priority: P0
- Bundle: logic-family-parser/classical-rules
- Parallel lane: lfp-classical
- Resource class: cpu-large
- Goal: Provide typed parsers/printers/elaborators and loss-receipted lowerings for canonical FOL, SMT-LIB2, TPTP, Datalog/Horn/CHC/SecPAL, and F-logic/ErgoAI subsets.
- Evidence: LFP-017, LFP-018, LFP-019, LFP-020, LFP-021, LFP-022
- Evidence criteria: Each notation has golden, negative, round-trip, resource, and backend vertical-slice fixtures with exact unsupported boundaries.
- Evidence source policy: Source parsing does not inherit a target solver's authority; solver output remains typed candidate/model/proof evidence under the translation receipt.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers, ipfs_datasets_py/tests/unit/logic/parsers
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/fol.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/smtlib.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tptp.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/rules.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/flogic.py
- Interfaces: CanonicalFOLSyntax@1, SMTLIB2Frontend@1, TPTPFrontend@1, RuleFrontend@1, FLogicFrontend@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_fol.py tests/unit/logic/parsers/test_smtlib.py tests/unit/logic/parsers/test_tptp.py tests/unit/logic/parsers/test_rules.py tests/unit/logic/parsers/test_flogic.py
- Acceptance: Z3/cvc5, Vampire/E, Datalog/SecPAL, and ErgoAI routes consume typed expressions or explicit controlled target source; no unknown character, binder, sort, rule priority, or node is silently skipped.
- Gap task: Fill one missing notation/fragment/backend vertical slice with explicit preservation and authority contracts.
- Refinement: Full SMT-LIB, TPTP THF, or ErgoAI language coverage is not implied by a controlled subset.
- Embedding query: fol smtlib tptp datalog horn chc secpal frame logic ergoai parser
- AST query: SmtTerm TPTPFormula AuthorizationIR FLogicModel TranslationReceipt
- Conflict policy: Each notation owns distinct modules/fixtures; shared AST or registry changes return to G030 rather than being duplicated.

## LFP-G050 Unify modal, temporal, state, event, normative, and hyperproperty syntax

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G030
- Fib priority: 2
- Track: modal-temporal-parsers
- Priority: P0
- Bundle: logic-family-parser/modal-temporal
- Parallel lane: lfp-modal-temporal
- Resource class: cpu-large
- Goal: Define composable semantic profiles and typed syntax for modal/deontic/cognitive, temporal/state/monitor, event/TDFOL/DCEC, and hypertrace families.
- Evidence: LFP-023, LFP-024, LFP-025, LFP-026, LFP-027, LFP-028
- Evidence criteria: Controlled syntax and legacy importers preserve source maps and explicitly record finite/infinite trace, frame, norm, bound, fairness, and operator ambiguity choices.
- Evidence source policy: A monitor verdict, bounded state check, modal tableau, or legacy round trip has only its declared profile/bound authority.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/modal.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/temporal.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/state.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/hyper.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/event_calculus.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/legacy_modal.py
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/modal.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/temporal.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/state.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/hyper.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/event_calculus.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/legacy_modal.py
- Interfaces: ModalSyntax@1, TemporalSyntax@1, StatePropertySyntax@1, HyperpropertySyntax@1, EventCalculusSyntax@1, LegacyLogicImporter@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_modal.py tests/unit/logic/parsers/test_temporal.py tests/unit/logic/parsers/test_state.py tests/unit/logic/parsers/test_hyper.py tests/unit/logic/parsers/test_event_calculus.py
- Acceptance: TLC/Apalache, runtime MTL, HyperLTL tools, TDFOL, CEC/DCEC, and legal/modal routes share typed nodes or explicit adapters; overloaded O/P/F/box/diamond symbols require a declared syntax/profile; no bound or trace model is omitted from receipts.
- Gap task: Add the smallest missing semantic profile/adapter/fixture needed by an admitted domain or backend.
- Refinement: TLA+ full modules and complete legacy legal language parsing remain delegated/specialized; the common parser covers declared controlled fragments.
- Embedding query: modal deontic epistemic temporal ltl mtl tla event calculus dcec tdfol hyperltl
- AST query: TemporalFormula ModalOperator RuntimeMTLFormula TransitionSystem HyperpropertyIR
- Conflict policy: Temporal core, modal core, state, hyper, and legacy importer paths are disjoint; shared extension-node changes merge through G030 owners.

## LFP-G060 Add protocol, program, resource, refinement, and kernel target surfaces

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G030
- Fib priority: 3
- Track: protocol-program-kernel
- Priority: P1
- Bundle: logic-family-parser/protocol-program-kernel
- Parallel lane: lfp-protocol-program
- Resource class: cpu-proof-solver
- Goal: Add controlled typed surfaces for symbolic protocols, program/resource/refinement obligations, and proof-assistant target theories without pretending to reimplement official kernels.
- Evidence: LFP-029, LFP-030, LFP-031, LFP-032, LFP-033
- Evidence criteria: ProVerif/Tamarin, program VC/TLA/SMT, separation/concurrency/refinement, and Lean/Rocq/Isabelle/hammer routes have exact feature, assumption, import, axiom, and reconstruction receipts.
- Evidence source policy: Protocol tools and ATP/hammer outputs are candidates; Lean/Rocq/Isabelle become authoritative only after the exact generated theorem is accepted by the pinned kernel environment.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/protocol.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/program.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/kernel_targets.py
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/protocol.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/program.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/kernel_targets.py
- Interfaces: ProtocolSyntax@1, ProgramLogicSyntax@1, ResourceLogicSyntax@1, KernelTheoryModel@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_protocol.py tests/unit/logic/parsers/test_program.py tests/unit/logic/parsers/test_kernel_targets.py
- Acceptance: Protocol attacker/equational assumptions, program/frame/resource conditions, refinement direction, imports, axioms, and trust escapes are explicit; full proof-assistant language parsing is not claimed.
- Gap task: Add one controlled frontend/lowering/reconstruction route with exact official-kernel validation.
- Refinement: Hammer is a strategy/meta-provider, not a semantic family or proof authority.
- Embedding query: proverif tamarin protocol hoare dynamic separation concurrency refinement lean rocq isabelle hammer
- AST query: ProtocolIR VerificationCondition SeparationAssertion RefinementObligation KernelTheory
- Conflict policy: Protocol, program/resource, and kernel-target modules are file-disjoint; shared backend execution code changes are serialized by task dependencies.

## LFP-G070 Connect Security, Crypto, Intent, Legal, UI, and software-contract IRs

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G040, LFP-G050, LFP-G060
- Fib priority: 2
- Track: domain-adapters
- Priority: P0
- Bundle: logic-family-parser/domain-adapters
- Parallel lane: lfp-domain
- Resource class: cpu-large
- Goal: Make each domain IR emit canonical typed formal views, registered translation routes, complete source maps, and backend requests with no invented family identifiers.
- Evidence: LFP-034, LFP-035, LFP-036, LFP-037, LFP-038, LFP-039
- Evidence criteria: Every pinned domain has proof-safe and counterexample/monitor vertical slices where semantically appropriate plus explicit unsupported cells; absent UI/UX source has a typed exact-revision gate that deterministically triggers a derived adapter task when its reviewed commit enters the gitlink.
- Evidence source policy: Domain contracts establish claim intent and provenance; translated views inherit the translation authority ceiling and never reinterpret natural language at backend time.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir, ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir, ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_source_gate.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/formalization_adapter.py, ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir/formalization, ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize, ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_source_gate.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/syntax_bridge.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/syntax_bridge.py
- Interfaces: DomainFormalizationAdapter@2, DomainLogicCapabilityMatrix@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/security_ir tests/unit/logic/crypto_ir tests/unit/logic/intent_ir tests/unit/logic/legal_ir tests/unit/logic/software_verification tests/unit/logic/software_contracts tests/conformance/logic/test_ui_ux_source_gate.py
- Acceptance: `security_ir`, `crypto_ir`, `intent_ir`, `legal_ir`, and software contracts use canonical families/profiles/properties/views; `ui_ux_ir` is either bound to a reviewed pinned revision and a derived adapter task or explicitly declaration-only/source-missing; existing source/provenance/authority fields and golden fixtures remain compatible.
- Gap task: Generate one domain x family x backend task from an uncovered admitted matrix cell.
- Refinement: The pinned datasets tree currently lacks the user's untracked UI/UX package. LFP-038 may write only the source gate; never recreate or edit the package. Its exact reviewed commit/gitlink plus refreshed LFP-001/LFP-005 evidence content-triggers the derived narrow-adapter task.
- Embedding query: security ir crypto ledger intent skill prompt legal ui ux software contract formalization adapter
- AST query: SecurityIR CryptoIR IntentIR LegalIR UIUXIR FormalizationArtifact
- Conflict policy: One task owns each domain package; common formalization changes precede domain edits and cross-domain joins occur only in G080.

## LFP-G080 Prove parser, translation, backend, advisor, and authority conformance

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G040, LFP-G050, LFP-G060, LFP-G070
- Fib priority: 1
- Track: conformance
- Priority: P0
- Bundle: logic-family-parser/conformance
- Parallel lane: lfp-conformance
- Resource class: cpu-proof-solver
- Goal: Establish cross-family algebra, parser security, translation preservation, differential backend, reconstruction, and advisor-authority evidence on one pinned corpus.
- Evidence: LFP-040, LFP-041, LFP-042, LFP-043
- Evidence criteria: Golden/property/fuzz/differential/kernel/domain-matrix receipts are current, content addressed, resource bounded, and classify every disagreement or unsupported case.
- Evidence source policy: Differential agreement increases confidence but never votes a proof into existence; kernel reconstruction and declared model-check bounds determine final authority.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/generated_catalog.py, ipfs_datasets_py/tests/conformance/logic, ipfs_datasets_py/benchmarks/logic_parsers, ipfs_datasets_py/docs/architecture/logic/conformance
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/generated_catalog.py, ipfs_datasets_py/tests/conformance/logic, ipfs_datasets_py/benchmarks/logic_parsers, ipfs_datasets_py/docs/architecture/logic/conformance
- Interfaces: LogicParserCatalog@1, GeneratedProviderTranslationCatalog@1, LogicConformanceReceipt@1, LogicDifferentialReceipt@1, LogicAuthorityReceipt@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic
- Acceptance: Parser/AST/translation/domain/provider closure passes; fuzz/resource floors hold; advisor candidates cannot claim proof or policy authority; every authoritative proof route binds exact kernel/model/bound/axiom/import identities.
- Gap task: Reduce and project a failing fixture, disagreement, missing reconstruction, or authority mismatch into an owning goal.
- Refinement: Unknown, timeout, parser recovery, solver disagreement, or advisor confidence is inconclusive evidence, not success.
- Embedding query: parser conformance fuzz differential solver reconstruction authority matrix vertical slice
- AST query: ParseArtifact TranslationReceipt ProofReceipt CounterexampleReceipt AuthorityKind
- Conflict policy: Conformance tasks own fixtures/harness/reports; defects return to the owning parser/domain/backend task rather than being patched inside validators.

## LFP-G090 Migrate APIs safely and maintain a bounded refill fixed point

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G080
- Fib priority: 2
- Track: migration-refill
- Priority: P1
- Bundle: logic-family-parser/migration-refill
- Parallel lane: lfp-migration
- Resource class: cpu-medium
- Goal: Publish dual-read/one-write migration adapters, public API/docs, and bounded content-addressed refill from matrix and conformance gaps.
- Evidence: LFP-044, LFP-045, LFP-046
- Evidence criteria: Public compatibility, deprecation diagnostics, documentation, task identity, retry cooldown, open-task bounds, and unchanged-input idempotence are validated.
- Evidence source policy: Refill evidence nominates work only; the control plane may append derived records to its separately sealed derived section/runtime ledger but cannot rewrite seed task definitions, grant semantic authority, or widen task paths.
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/api.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/docs/architecture/logic, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill.py
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/api.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/docs/architecture/logic, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill.py
- Interfaces: LogicAPI@3, LegacyLogicAdapter@1, LogicGapRefill@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/test_api.py tests/unit/logic/test_verification_api.py tests/unit/logic/conformance/test_refill.py
- Acceptance: Canonical writes and reviewed legacy reads are compatible; docs match discovery; identical evidence emits no duplicate task; per-epoch derived caps/cooldowns/depth/retry rules exclude immutable seed goals and hold; seed definitions and protected artifacts are immutable.
- Gap task: Create only content-addressed migration or refill work from a typed, current evidence gap.
- Refinement: Do not make codebase-wide mechanical renames before consumer closure and compatibility evidence.
- Embedding query: api migration dual read canonical write deprecation refill task matrix gap fixed point
- AST query: LogicVerificationAPI LogicFamilyRegistry LegacyAdapter RefillCandidate
- Conflict policy: API migration, docs, and refill engine are separate tasks; public cutover follows all domain conformance.

## LFP-G100 Join release evidence and freeze the next-version baseline

- Status: active
- Parent: LFP-G000
- Depends on: LFP-G090
- Fib priority: 1
- Track: release
- Priority: P0
- Bundle: logic-family-parser/release
- Parallel lane: lfp-release
- Resource class: cpu-large
- Goal: Re-run the complete current-tree program, close or explicitly disposition every matrix gap, and emit the immutable release receipt and next refill baseline.
- Evidence: LFP-047
- Evidence criteria: The release receipt binds exact datasets/accelerator revisions, corpus, registries, parser schemas, translations, providers, domain views, backend environments, tests, fuzzing, benchmarks, and refill fixed point.
- Evidence source policy: Only fresh current-tree joined evidence may release; stale, synthetic, skipped, advisory, declaration-only, or unsupported evidence cannot be promoted.
- Outputs: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json
- Predicted files: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json
- Interfaces: LogicParserReleaseReceipt@1
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core tests/unit/logic/parsers tests/unit/logic/families tests/unit/logic/formalization tests/unit/logic/backends tests/unit/logic/security_ir tests/unit/logic/crypto_ir tests/unit/logic/intent_ir tests/unit/logic/legal_ir tests/unit/logic/software_verification tests/unit/logic/software_contracts tests/unit/logic/conformance/test_refill.py tests/conformance/logic tests/fuzz/logic
- Acceptance: All release floors pass with zero unexplained gaps and a bounded refill epoch emits no new task for unchanged evidence; the receipt is reproducible and rejects any source/config/environment drift.
- Gap task: Return a failing release premise to its owning goal; never patch production semantics in the release join.
- Refinement: A green subset, aggregate pass count, or stale solver receipt cannot substitute for the joined release contract.
- Embedding query: release receipt logic parser family provider translation domain conformance fixed point
- AST query: LogicParserReleaseReceipt LogicConformanceReceipt ProviderCapabilityCatalog
- Conflict policy: LFP-047 owns release reports only and cannot edit production parser, domain, backend, taxonomy, or syntax-core code.
