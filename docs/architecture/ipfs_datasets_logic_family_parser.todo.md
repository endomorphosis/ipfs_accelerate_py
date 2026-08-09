# IPFS Datasets Logic-Family Parser Taskboard

This is the drainable projection of
`ipfs_datasets_logic_family_parser.objectives.md`. The seed board contains 48
tasks. The supervisor control plane may append content-addressed operational
records to a derived section/runtime gap ledger that is merged for scheduling
but excluded from the sealed seed-definition digest; it may not rewrite seed
task definitions or other protected control documents.

Status values are `todo`, `in_progress`, `blocked`, and `completed`. A solver,
parser, advisor, or model response never completes a task without the declared
validation and evidence contract.

## LFP-000 Seal the plan, objective heap, taskboard, scheduler, and validator

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Priority: P0
- Track: control
- Depends on:
- Goal id: LFP-G000
- Outputs: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_scheduler.json, scripts/validate_ipfs_datasets_logic_family_parser_board.py
- Validation: python scripts/validate_ipfs_datasets_logic_family_parser_board.py --check-all
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/control
- Parallel lane: lfp-control
- Resource class: cpu-small
- Resource stage: control
- Estimated tokens: 12000
- Implementation timeout seconds: 3600
- Predicted files: docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md, docs/architecture/ipfs_datasets_logic_family_parser.objectives.md, docs/architecture/ipfs_datasets_logic_family_parser.todo.md, config/agent_supervisor_ipfs_datasets_logic_family_parser_scheduler.json, scripts/validate_ipfs_datasets_logic_family_parser_board.py
- Interfaces: LogicParserProgramControl@1
- Allow concurrent with:
- Conflict policy: Protected control artifact; implementation workers may not edit it.
- Preconditions: User scope includes security, crypto, intent, legal, UI, software verification, all named solvers, ErgoAI, and SymbolicAI.
- Effects: Four strict shards and bounded objective refill have a reviewed dependency graph and provider policy.
- Evidence subset: plan/objective/task/config/validator cross-references and provider route
- Symbolic first: true
- LLM context budget bytes: 16000
- Acceptance: Validator returns valid; task and goal graphs are acyclic; all outputs have owners; Grok 4.5 is primary and Terra high is quota-only fallback.
- Embedding query: seal logic parser supervisor plan taskboard objectives provider route

## LFP-001 Inventory parser, AST, type, printer, and decoder surfaces

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: parser-inventory
- Depends on: LFP-000
- Goal id: LFP-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/inventory.py, ipfs_datasets_py/tests/unit/logic/conformance/test_inventory.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/parser_inventory.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_inventory.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/inventory/parsers
- Parallel lane: lfp-inventory-parsers
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/inventory.py, ipfs_datasets_py/tests/unit/logic/conformance/test_inventory.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/parser_inventory.json
- Interfaces: LogicSurfaceInventory@1
- Allow concurrent with: LFP-002, LFP-003, LFP-004
- Conflict policy: Own parser inventory module/test/report only; production parser files are read-only evidence.
- Preconditions: Pinned datasets git tree and explicit logic roots are available.
- Effects: Every parser, AST, formula/term class, printer, compiler, result parser, and legacy duplicate has a stable identity and path.
- Evidence subset: TDFOL, CEC/DCEC, FOL, deontic, modal, F-logic, runtime MTL, solver adapters
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: Inventory is deterministic, bounded, side-effect-free, path-complete under policy, and explicitly identifies raw-string and arbitrary-JSON formula boundaries.
- Embedding query: inventory parser ast term formula printer decoder tdfol cec deontic modal

## LFP-002 Freeze the cross-family conformance corpus manifest

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: corpus
- Depends on: LFP-000
- Goal id: LFP-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/corpus.py, ipfs_datasets_py/tests/fixtures/logic_conformance/manifest.json, ipfs_datasets_py/tests/unit/logic/conformance/test_corpus.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_corpus.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/inventory/corpus
- Parallel lane: lfp-inventory-corpus
- Resource class: cpu-medium
- Resource stage: fixtures
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/corpus.py, ipfs_datasets_py/tests/fixtures/logic_conformance/manifest.json, ipfs_datasets_py/tests/unit/logic/conformance/test_corpus.py
- Interfaces: LogicConformanceCorpus@1
- Allow concurrent with: LFP-001, LFP-003, LFP-004
- Conflict policy: Own corpus manifest/loader/test only; copy no copyrighted or secret production input.
- Preconditions: Existing public and synthetic fixtures may be referenced by content identity.
- Effects: Positive, negative, ambiguous, adversarial, translation, model, proof, and trace fixtures share one versioned schema.
- Evidence subset: family/profile/notation/source/license/digest/expected diagnostics and authority
- Symbolic first: true
- LLM context budget bytes: 20000
- Acceptance: Manifest rejects missing digests, duplicate IDs, unsafe paths, unbounded payloads, and fixtures without expected disposition; labels not yet known to the baseline registry are preserved losslessly with an explicit unknown disposition for LFP-003/LFP-010 closure.
- Embedding query: conformance corpus golden negative ambiguity adversarial fixture manifest

## LFP-003 Audit canonical and free-form family identifiers

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: family-audit
- Depends on: LFP-000
- Goal id: LFP-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/audit.py, ipfs_datasets_py/tests/unit/logic/families/test_audit.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/family_label_audit.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_audit.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/inventory/families
- Parallel lane: lfp-inventory-families
- Resource class: cpu-small
- Resource stage: static-analysis
- Estimated tokens: 16000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/audit.py, ipfs_datasets_py/tests/unit/logic/families/test_audit.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/family_label_audit.json
- Interfaces: LogicFamilyAudit@1
- Allow concurrent with: LFP-001, LFP-002, LFP-004
- Conflict policy: Own audit module/test/report; do not rename production strings in this task.
- Preconditions: Default family registry and backend/domain roots are readable without imports.
- Effects: Every observed family-like string is classified as canonical family, alias, profile, property, view, notation, provider, lane, evidence kind, or unknown.
- Evidence subset: backends registry, formalization views, security/crypto/intent/legal/UI adapters
- Symbolic first: true
- LLM context budget bytes: 18000
- Acceptance: Deterministic audit covers the configured roots, reports all current known drift, and never treats tool names or safety/liveness/VC/view roles as semantic families.
- Embedding query: family string audit alias profile property view provider namespace drift

## LFP-004 Materialize the domain-family-provider capability matrix baseline

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: capability-matrix
- Depends on: LFP-000
- Goal id: LFP-G010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/matrix.py, ipfs_datasets_py/tests/unit/logic/conformance/test_matrix.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/capability_matrix.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_matrix.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/inventory/matrix
- Parallel lane: lfp-inventory-matrix
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/matrix.py, ipfs_datasets_py/tests/unit/logic/conformance/test_matrix.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/capability_matrix.json
- Interfaces: LogicCapabilityMatrix@1
- Allow concurrent with: LFP-001, LFP-002, LFP-003
- Conflict policy: Own matrix schema/loader/report only; provider registry remains read-only.
- Preconditions: Domain IRs and provider registries are statically inspectable.
- Effects: Every domain x formal view x family/profile x provider cell has native, translated, approximate, bounded, advisory, declaration-only, unsupported, or unknown status.
- Evidence subset: Z3 cvc5 TLC Apalache SecPAL ProVerif Tamarin HyperLTL Vampire E Hammer Lean Rocq Isabelle runtime MTL ErgoAI SymbolicAI
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: Matrix distinguishes support from availability and authority, records exact source evidence, and exposes all unknown/unimplemented cells for later refill.
- Embedding query: domain family provider solver capability matrix security crypto intent legal ui

## LFP-005 Join and seal the current-state baseline

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inventory-join
- Depends on: LFP-001, LFP-002, LFP-003, LFP-004
- Goal id: LFP-G010
- Outputs: ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/README.md, ipfs_datasets_py/tests/unit/logic/conformance/test_baseline_join.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_baseline_join.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/inventory/join
- Parallel lane: lfp-inventory-join
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 14000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_datasets_py/docs/architecture/logic/logic_parser_baseline/README.md, ipfs_datasets_py/tests/unit/logic/conformance/test_baseline_join.py
- Interfaces: LogicParserBaselineReceipt@1
- Allow concurrent with:
- Conflict policy: Own joined baseline documentation/test only; discovered defects become G020/G030 tasks.
- Preconditions: Four initial inventory artifacts pass independently.
- Effects: One content-addressed receipt binds current revisions, roots, inventories, corpus, matrix, gaps, and known active UI work.
- Evidence subset: exact reports and unresolved gap counts
- Symbolic first: true
- LLM context budget bytes: 16000
- Acceptance: Join rejects revision/digest/schema drift and explicitly lists zero hidden or silently normalized unknown labels.
- Embedding query: baseline join parser family provider corpus matrix receipt

## LFP-006 Separate the canonical logic identity namespaces

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: taxonomy-namespaces
- Depends on: LFP-005
- Goal id: LFP-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/namespaces.py, ipfs_datasets_py/tests/unit/logic/families/test_namespaces.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_namespaces.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/taxonomy/namespaces
- Parallel lane: lfp-taxonomy-namespaces
- Resource class: cpu-small
- Resource stage: contracts
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/namespaces.py, ipfs_datasets_py/tests/unit/logic/families/test_namespaces.py
- Interfaces: LogicIdentityNamespaces@1
- Allow concurrent with: LFP-007
- Conflict policy: Own new namespace contracts/tests; do not modify registry defaults until join tasks.
- Preconditions: Baseline label audit is sealed.
- Effects: Family, profile, property, view, notation, encoding, provider, lane, and evidence identifiers become non-interchangeable typed values.
- Evidence subset: canonical JSON, identifier validation, collision and wrong-namespace rejection
- Symbolic first: true
- LLM context budget bytes: 18000
- Acceptance: Cross-namespace coercion fails closed; aliases cannot collide; values serialize deterministically and preserve schema/version.
- Embedding query: typed identifiers family profile property view notation provider evidence namespace

## LFP-007 Define compositional semantic profiles and family extensions

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: semantic-profiles
- Depends on: LFP-005
- Goal id: LFP-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/profiles.py, ipfs_datasets_py/tests/unit/logic/families/test_profiles.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_profiles.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/taxonomy/profiles
- Parallel lane: lfp-taxonomy-profiles
- Resource class: cpu-medium
- Resource stage: contracts
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/profiles.py, ipfs_datasets_py/tests/unit/logic/families/test_profiles.py
- Interfaces: SemanticProfile@1, FamilyComposition@1
- Allow concurrent with: LFP-006
- Conflict policy: Own new profile/composition contracts; default registry mutation waits for LFP-008.
- Preconditions: Baseline family and backend labels are known.
- Effects: Consequence, world policy, bounds, traces, time, frames, norms, attacker, hypertrace, SMT theory, and kernel environment are explicit profile fields.
- Evidence subset: profile validation, composition conflicts, finite bounds, frame and norm semantics
- Symbolic first: true
- LLM context budget bytes: 20000
- Acceptance: Profiles reject contradictory or incomplete semantic choices; canonical tdfol and dcec IDs are retained with mandatory, versioned composition metadata; temporal-FOL is expressed as declared composition rather than an opaque replacement family string.
- Embedding query: semantic profile composition classical intuitionistic time frame norm attacker bound

## LFP-008 Add versioned aliases and dual-read one-write canonicalization

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: alias-migration
- Depends on: LFP-006, LFP-007
- Goal id: LFP-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/aliases.py, ipfs_datasets_py/tests/unit/logic/families/test_aliases.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_aliases.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/taxonomy/aliases
- Parallel lane: lfp-taxonomy-aliases
- Resource class: cpu-medium
- Resource stage: migration
- Estimated tokens: 18000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/aliases.py, ipfs_datasets_py/tests/unit/logic/families/test_aliases.py
- Interfaces: LogicAliasRegistry@1, LogicMigrationDiagnostic@1
- Allow concurrent with:
- Conflict policy: Own aliases/tests only; no domain-wide replacements in this task.
- Preconditions: Typed namespaces and semantic profiles pass.
- Effects: Reviewed legacy labels resolve with replacement diagnostics; canonical writers emit only canonical namespace values.
- Evidence subset: fol/smt/tla_plus/hyperltl/protocol/secpal/VC/safety/provider/view cases
- Symbolic first: true
- LLM context budget bytes: 18000
- Acceptance: Unknown and wrong-namespace labels fail closed; alias cycles/collisions are impossible; canonicalization is deterministic and idempotent.
- Embedding query: alias canonicalization migration diagnostic dual read canonical write

## LFP-009 Unify translation preservation, loss, and authority contracts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: translation-contracts
- Depends on: LFP-006, LFP-007
- Goal id: LFP-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/translations.py, ipfs_datasets_py/tests/unit/logic/families/test_translations_v2.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_translations_v2.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/taxonomy/translations
- Parallel lane: lfp-taxonomy-translations
- Resource class: cpu-medium
- Resource stage: contracts
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/translations.py, ipfs_datasets_py/tests/unit/logic/families/test_translations_v2.py
- Interfaces: TranslationContract@2, TranslationCompositionReceipt@1
- Allow concurrent with: LFP-008
- Conflict policy: Own new translation contracts/tests; existing compiler behavior remains unchanged until adapters.
- Preconditions: Typed namespaces and profiles pass.
- Effects: Equivalence/equisatisfiability/theorem/model/trace/bounded/approximate/heuristic relations, polarity, assumptions, node maps, and authority ceilings are explicit.
- Evidence subset: composition weakest-link, proof-safe/counterexample-safe, silent-drop rejection
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: Composed translations inherit the weakest guarantee and lowest authority; unknown nodes or assumptions cannot disappear; content identities bind compiler/profile/config.
- Embedding query: translation preservation equivalence equisatisfiable model trace approximation authority

## LFP-010 Establish provider-capability schemas and the canonical baseline catalog

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-catalog
- Depends on: LFP-008, LFP-009
- Goal id: LFP-G020
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/providers.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/registry.py, ipfs_datasets_py/tests/unit/logic/families/test_provider_catalog.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/families/test_provider_catalog.py tests/unit/logic/backends/test_registry.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/taxonomy/providers
- Parallel lane: lfp-taxonomy-providers
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/families/providers.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/registry.py, ipfs_datasets_py/tests/unit/logic/families/test_provider_catalog.py
- Interfaces: ProviderCapabilityCatalog@1, LogicFamilyRegistry@2
- Allow concurrent with:
- Conflict policy: Serialized join of taxonomy/provider registries; preserve backend factories and lazy imports.
- Preconditions: Alias and translation contracts pass.
- Effects: Baseline schemas and descriptors cover exact current provider IDs z3, cvc5, tla_tlc, apalache, datalog_secpal, proverif, tamarin, hyperltl_autohyper_mchyper, vampire, eprover, hammer, lean, rocq, isabelle, runtime_mtl, ergoai, and symbolicai; later parser/domain tasks contribute translation edges for LFP-040 projection.
- Evidence subset: family/fragment/property/operation/evidence/translation/availability/authority claims
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Baseline catalog contains no free-form family drift; explicitly enumerates planned extension families epistemic, doxastic, intention_agency, and session_process plus declaration-only dependent_type, description_logic, defeasible_logic, nonmonotonic_logic, argumentation, situation_calculus, probabilistic, fuzzy_weighted, relevance_paraconsistent, and finite_field_constraint; preserves dynamic_logic only as a program alias/profile and information_flow only as a hyperproperty profile; enumerates every exact executable-matrix provider ID and reviewed alias; distinguishes baseline descriptors from LFP-040 generated closure; never treats presence as availability/proof; and gives advisory providers hard authority ceilings.
- Embedding query: provider capability catalog solver z3 cvc5 tla proverif tamarin lean symai ergoai

## LFP-011 Define source, token, CST, AST, diagnostic, and parse contracts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: syntax-contracts
- Depends on: LFP-010
- Goal id: LFP-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/contracts.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_contracts.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_contracts.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/syntax-core/contracts
- Parallel lane: lfp-syntax-contracts
- Resource class: cpu-medium
- Resource stage: contracts
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/contracts.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_contracts.py
- Interfaces: SourceDocument@1, LogicToken@1, LogicCST@1, ParseRequest@1, ParseArtifact@1
- Allow concurrent with:
- Conflict policy: Own contracts/test only; public exports wait for LFP-016.
- Preconditions: Canonical namespaces and profiles pass.
- Effects: Bounded immutable source/parse envelopes and stable diagnostics become the common frontend boundary.
- Evidence subset: spans/encoding/trivia/recovery/limits/ambiguity/unsupported/provenance
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: Invalid ranges, unsafe encodings, missing source coverage, unbounded limits, duplicate diagnostics, and wrong namespace/profile IDs fail closed.
- Embedding query: source document token cst ast parse artifact diagnostic span bounded

## LFP-012 Implement typed core terms, formulas, binders, sorts, and signatures

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: syntax-ast
- Depends on: LFP-011
- Goal id: LFP-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/ast.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/signatures.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_ast.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_ast.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/syntax-core/ast
- Parallel lane: lfp-syntax-ast
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/ast.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/signatures.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_ast.py
- Interfaces: TypedExpression@1, LogicSignature@1, LogicExtensionNode@1
- Allow concurrent with: LFP-014
- Conflict policy: Own AST/signature modules/tests; no parser grammar code.
- Preconditions: Core source/parse contracts pass.
- Effects: Propositional and many-sorted FOL nodes plus versioned family extensions have stable immutable identities.
- Evidence subset: constants variables applications predicates equality connectives quantifiers lets extension nodes
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Sort/arity/signature invariants fail at construction or elaboration; extension nodes declare family/profile/features and cannot carry opaque unversioned payloads.
- Embedding query: typed ast term formula binder quantifier sort signature extension node

## LFP-013 Implement alpha-equivalence, free variables, and capture-avoiding substitution

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: syntax-algebra
- Depends on: LFP-012
- Goal id: LFP-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/algebra.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_algebra.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_algebra.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/syntax-core/algebra
- Parallel lane: lfp-syntax-algebra
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/algebra.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_algebra.py
- Interfaces: LogicExpressionAlgebra@1
- Allow concurrent with: LFP-014
- Conflict policy: Own algebra/test only; do not patch legacy TDFOL substitution here.
- Preconditions: Core AST protocol is stable enough for visitors.
- Effects: Shared safe binding operations replace ad-hoc substitution in new frontends and provide legacy migration tests.
- Evidence subset: alpha rename shadowing nested binders capture adversarial terms idempotence
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: Property tests prove substitution never captures free variables, alpha-equivalent expressions share semantic identity, and traversal is bounded.
- Embedding query: alpha equivalence capture avoiding substitution free bound variables binder algebra

## LFP-014 Implement bounded lexing, diagnostics, source maps, and resource guards

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: syntax-lexing
- Depends on: LFP-011
- Goal id: LFP-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/lexer.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/diagnostics.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_lexer_diagnostics.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_lexer_diagnostics.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/syntax-core/lexer
- Parallel lane: lfp-syntax-lexer
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/lexer.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/diagnostics.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_lexer_diagnostics.py
- Interfaces: BoundedLexer@1, LogicDiagnostic@1, LogicSourceMap@1
- Allow concurrent with: LFP-012, LFP-013
- Conflict policy: Own lexer/diagnostics/tests; family token tables remain in parser modules.
- Preconditions: Source and token contracts pass.
- Effects: Unknown characters, confusables, NULs, malformed strings/comments/numbers, and resource exhaustion become typed exact-span failures.
- Evidence subset: unicode ascii comments nesting token depth input bytes diagnostic caps
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: No unknown input is logged-and-skipped; strict mode rejects, recovery mode preserves an explicit error node, and all configured bounds terminate deterministically.
- Embedding query: bounded lexer diagnostic source map unicode confusable parser bomb

## LFP-015 Add parser registry, elaborator, typechecker, normalizer, and codec

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: syntax-elaboration
- Depends on: LFP-012, LFP-013, LFP-014
- Goal id: LFP-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/elaboration.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/codec.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_elaboration_codec.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_elaboration_codec.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/syntax-core/elaboration
- Parallel lane: lfp-syntax-elaboration
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/elaboration.py, ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/codec.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_elaboration_codec.py
- Interfaces: LogicParserRegistry@1, LogicElaborator@1, TypedLogicCodec@1
- Allow concurrent with:
- Conflict policy: Serialized syntax-core integration; parser families register later.
- Preconditions: AST, algebra, lexer, diagnostics, and profile contracts pass.
- Effects: Frontends resolve notation/profile explicitly and produce typed deterministic artifacts with signature/type errors and semantic identities.
- Evidence subset: binding overload resolution sort checking normalization codec migration parser selection
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Registry collision/implicit fallback is rejected; codecs round-trip; normalization is idempotent; unresolved overloads/unknown signatures do not reach backends.
- Embedding query: parser registry elaborator typechecker normalizer codec semantic hash

## LFP-016 Publish syntax core and bridge shared formalization contracts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: syntax-publication
- Depends on: LFP-015
- Goal id: LFP-G030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/views.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/constraint_contracts.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_publication.py, ipfs_datasets_py/tests/unit/logic/formalization/test_typed_expression_bridge.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core/test_publication.py tests/unit/logic/formalization/test_typed_expression_bridge.py tests/unit/logic/formalization/test_constraint_contracts.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/syntax-core/publication
- Parallel lane: lfp-syntax-publication
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/syntax_core/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/views.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/constraint_contracts.py, ipfs_datasets_py/tests/unit/logic/syntax_core/test_publication.py, ipfs_datasets_py/tests/unit/logic/formalization/test_typed_expression_bridge.py, ipfs_datasets_py/tests/unit/logic/formalization/test_constraint_contracts.py
- Interfaces: LogicSyntaxCore@1, LazyParserPublication@1, FormalizationArtifact@2, ConstraintContract@2
- Allow concurrent with:
- Conflict policy: Serialized public export/formalization bridge; retain legacy artifact decoding.
- Preconditions: Complete syntax-core suite passes.
- Effects: Formal views and constraints carry a versioned typed expression envelope; parser packages publish lazy inert local-descriptor contracts; source maps, coverage, withholding, provenance, and legacy reads remain intact.
- Evidence subset: cold import, canonical export list, lazy parser descriptors, typed/legacy dual read, typed canonical write, translation loss contract
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Cold import starts no network/process/model/installer; typed expressions validate family/profile/schema; constraint contracts dual-read legacy payloads but write TypedExpression and TranslationContract@2; arbitrary JSON/text and a boolean loss flag cannot masquerade as elaborated syntax or a preservation receipt.
- Embedding query: publish syntax core formalization artifact typed expression legacy bridge

## LFP-017 Implement canonical many-sorted FOL parser and printer

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: fol-parser
- Depends on: LFP-016
- Goal id: LFP-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/fol.py, ipfs_datasets_py/tests/unit/logic/parsers/test_fol.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_fol.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/classical/fol
- Parallel lane: lfp-fol
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/fol.py, ipfs_datasets_py/tests/unit/logic/parsers/test_fol.py
- Interfaces: CanonicalFOLSyntax@1
- Allow concurrent with: LFP-018, LFP-019, LFP-020, LFP-021
- Conflict policy: Own FOL parser/printer/tests only; shared core changes require separate follow-up.
- Preconditions: Syntax core public contracts pass.
- Effects: A human-readable canonical notation replaces formula-string-only exchange for admitted FOL fragments.
- Evidence subset: precedence quantifiers scopes sorts signatures unicode/ascii roundtrip diagnostics
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Parse/print/parse is alpha-equivalent; implication associativity and binder scope are explicit; undeclared symbols/sorts and trailing input fail with exact spans.
- Embedding query: first order logic parser printer quantifier scope sort signature

## LFP-018 Implement SMT-LIB2 reader, elaborator, printer, and SMT bridge

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: smtlib-parser
- Depends on: LFP-016
- Goal id: LFP-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/smtlib.py, ipfs_datasets_py/tests/unit/logic/parsers/test_smtlib.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_smtlib.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/classical/smtlib
- Parallel lane: lfp-smtlib
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/smtlib.py, ipfs_datasets_py/tests/unit/logic/parsers/test_smtlib.py
- Interfaces: SMTLIB2Frontend@1
- Allow concurrent with: LFP-017, LFP-019, LFP-020, LFP-021
- Conflict policy: Own SMT-LIB frontend/tests; reuse typed SMT compiler rather than duplicate theory semantics.
- Preconditions: Syntax core and provider taxonomy pass.
- Effects: Controlled SMT-LIB scripts and terms parse into typed expressions/obligations and print with declared theory/profile.
- Evidence subset: s-expressions declarations lets quantifiers bv arrays arithmetic strings models cores unsupported commands
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Reader is bounded and complete for the declared subset; unknown commands/theories are explicit; Z3/cvc5 common-fragment round trips preserve symbol/sort semantics.
- Embedding query: smtlib2 parser s expression z3 cvc5 theories model core

## LFP-019 Implement TPTP CNF/FOF/TFF and TSTP controlled frontends

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: tptp-parser
- Depends on: LFP-016
- Goal id: LFP-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tptp.py, ipfs_datasets_py/tests/unit/logic/parsers/test_tptp.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_tptp.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/classical/tptp
- Parallel lane: lfp-tptp
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tptp.py, ipfs_datasets_py/tests/unit/logic/parsers/test_tptp.py
- Interfaces: TPTPFrontend@1, TSTPCandidateFrontend@1
- Allow concurrent with: LFP-017, LFP-018, LFP-020, LFP-021
- Conflict policy: Own TPTP/TSTP frontend/tests; ATP process adapters remain unchanged.
- Preconditions: Syntax core passes.
- Effects: Vampire/E problem files and proof candidates have typed roles, symbols, formulas, includes, and source maps.
- Evidence subset: CNF FOF TFF roles annotations includes SZS TSTP unsupported THF
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Declared subset round-trips; unsafe includes/path traversal and malformed annotations fail; THF is explicit unsupported until separately implemented; candidate proofs remain untrusted.
- Embedding query: tptp cnf fof tff tstp vampire eprover parser proof candidate

## LFP-020 Implement Datalog, Horn/CHC, and SecPAL rule frontends

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rule-parser
- Depends on: LFP-016
- Goal id: LFP-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/rules.py, ipfs_datasets_py/tests/unit/logic/parsers/test_rules.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_rules.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/classical/rules
- Parallel lane: lfp-rules
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/rules.py, ipfs_datasets_py/tests/unit/logic/parsers/test_rules.py
- Interfaces: RuleFrontend@1, SecPALFrontend@1
- Allow concurrent with: LFP-017, LFP-018, LFP-019, LFP-021
- Conflict policy: Own rule parser/tests; backend authorization semantics remain in existing adapters.
- Preconditions: Syntax core and authorization profiles pass.
- Effects: Facts/rules/queries/delegation/speaks-for/constraints/negation/priorities parse with explicit closed-world and authorization profiles.
- Evidence subset: range restriction recursion stratification negation principals delegation CHC lowering
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Unsafe variables, unstratified negation, ambiguous principal/resource/action terms, and missing world/priority semantics fail or receive explicit unsupported disposition.
- Embedding query: datalog horn chc secpal authorization parser rules delegation

## LFP-021 Implement the controlled F-logic and ErgoAI frontend

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: flogic-parser
- Depends on: LFP-016
- Goal id: LFP-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/flogic.py, ipfs_datasets_py/tests/unit/logic/parsers/test_flogic.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_flogic.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/classical/flogic
- Parallel lane: lfp-flogic
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/flogic.py, ipfs_datasets_py/tests/unit/logic/parsers/test_flogic.py
- Interfaces: FLogicFrontend@1, ErgoAIControlledSource@1
- Allow concurrent with: LFP-017, LFP-018, LFP-019, LFP-020
- Conflict policy: Own controlled F-logic frontend/tests; do not import/install/execute ErgoAI during parsing or discovery.
- Preconditions: Frame-logic family/profile and syntax core pass.
- Effects: Frames/classes/methods/inheritance/rules/queries use typed nodes instead of raw rule/query strings for the declared subset.
- Evidence subset: frame terms inheritance signatures methods rules queries source output diagnostics
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Parse/print and deterministic normalization pass; unsupported ErgoAI constructs are retained/diagnosed; execution remains lazy and advisor/candidate authority is explicit.
- Embedding query: frame logic flogic ergoai parser class frame inheritance rule query

## LFP-022 Join classical/rule parsers with Z3, cvc5, Vampire, E, SecPAL, and ErgoAI routes

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: classical-backend-join
- Depends on: LFP-017, LFP-018, LFP-019, LFP-020, LFP-021
- Goal id: LFP-G040
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/classical_adapters.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/advisor_parser_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_classical_backend_routes.py, ipfs_datasets_py/tests/conformance/logic/test_advisor_parser_boundary.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_classical_backend_routes.py tests/conformance/logic/test_advisor_parser_boundary.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/classical/join
- Parallel lane: lfp-classical-join
- Resource class: cpu-proof-solver
- Resource stage: integration
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/classical_adapters.py, ipfs_datasets_py/ipfs_datasets_py/logic/formalization/advisor_parser_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_classical_backend_routes.py, ipfs_datasets_py/tests/conformance/logic/test_advisor_parser_boundary.py
- Interfaces: ClassicalBackendAdapter@1, AdvisorCandidateParser@1
- Allow concurrent with: LFP-023, LFP-024, LFP-029, LFP-030, LFP-031, LFP-032
- Conflict policy: Own join adapters/conformance test; do not change individual parser grammars in the join.
- Preconditions: Five parser suites and provider catalog pass.
- Effects: Typed source reaches shared backend requests and typed result decoders with preservation and authority receipts; SymbolicAI proposals pass through deterministic parse/elaboration before becoming typed candidates.
- Evidence subset: proof-safe validity, counterexample-safe model, ATP candidate, authorization decision, ErgoAI/SymbolicAI advisor-only result, deterministic parse failure
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Exact routes run hermetically or report unavailable; approximate/unsupported routes cannot promote authority; no backend reparses natural language or free-form family labels; SymbolicAI parse/type failure remains an unverified candidate under formalization/proposal_advisors.py.
- Embedding query: classical parser backend join z3 cvc5 vampire e secpal ergoai

## LFP-023 Implement modal, normative, epistemic, and intention profiles

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: modal-parser
- Depends on: LFP-016
- Goal id: LFP-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/modal.py, ipfs_datasets_py/tests/unit/logic/parsers/test_modal.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_modal.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/modal-temporal/modal
- Parallel lane: lfp-modal
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/modal.py, ipfs_datasets_py/tests/unit/logic/parsers/test_modal.py
- Interfaces: ModalSyntax@1, NormativeProfile@1, CognitiveProfile@1
- Allow concurrent with: LFP-024, LFP-029, LFP-031
- Conflict policy: Own modal frontend/tests; shared extension nodes stay in syntax_core.
- Preconditions: Typed syntax extensions and canonical modal/deontic profiles pass.
- Effects: Kripke K/D/T/S4/S5, deontic O/P/F, conditional norms, and epistemic/doxastic/intention modalities are explicit profile-bound nodes; program-indexed dynamic logic is owned solely by LFP-031.
- Evidence subset: operator precedence frame axioms agent index norm polarity exceptions ambiguity diagnostics
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Profile-free overloaded symbols fail; parse/print preserves binding and source maps; unsupported dyadic or defeasible constructs retain typed diagnostics and cannot masquerade as classical equivalence.
- Embedding query: modal deontic epistemic doxastic intention bdi kripke parser

## LFP-024 Implement unified LTL, LTLf, past-LTL, MTL, CTL, and CTL-star syntax

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: temporal-parser
- Depends on: LFP-016
- Goal id: LFP-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/temporal.py, ipfs_datasets_py/tests/unit/logic/parsers/test_temporal.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_temporal.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/modal-temporal/temporal
- Parallel lane: lfp-temporal
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/temporal.py, ipfs_datasets_py/tests/unit/logic/parsers/test_temporal.py
- Interfaces: TemporalSyntax@1, TraceSemanticsProfile@1
- Allow concurrent with: LFP-023, LFP-029, LFP-031
- Conflict policy: Own temporal frontend/tests; existing runtime monitor and TLA modules are read-only until adapter tasks.
- Preconditions: Typed extension registry, rational interval types, and temporal profiles pass.
- Effects: Linear/branching, finite/infinite, past/future, dense/discrete, point/interval, and path semantics are explicit rather than inferred from spelling.
- Evidence subset: precedence associativity intervals trace kind time domain path quantifiers monitorability
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Parse/print/parse is alpha-equivalent; invalid or unbounded intervals and ambiguous F/G/U/R syntax fail with stable spans; profile and time domain enter semantic identity.
- Embedding query: ltl ltlf past mtl ctl ctl star temporal parser interval trace

## LFP-025 Add controlled transition-system and TLA property adapters

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: state-parser
- Depends on: LFP-024
- Goal id: LFP-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/state.py, ipfs_datasets_py/tests/unit/logic/parsers/test_state.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_state.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/modal-temporal/state
- Parallel lane: lfp-state
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 24000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/state.py, ipfs_datasets_py/tests/unit/logic/parsers/test_state.py
- Interfaces: StatePropertySyntax@1, ControlledTLAProperty@1
- Allow concurrent with: LFP-026, LFP-027
- Conflict policy: Own controlled property/state adapter; never claim to parse complete TLA+ modules.
- Preconditions: Temporal syntax and existing software-verification transition IR are available.
- Effects: State predicates, next-state relations, invariants, fairness, and declared temporal properties lower to TLC/Apalache inputs with explicit finite/bounded contracts.
- Evidence subset: variables init next invariant fairness stuttering bound source map TLC Apalache
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Controlled expressions round-trip; full-module constructs are declaration-only or unsupported; TLC finite-state and Apalache bounded results cannot be promoted to unbounded proof.
- Embedding query: transition system tla property parser tlc apalache state invariant fairness

## LFP-026 Bridge runtime MTL onto the shared temporal syntax

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: runtime-mtl-adapter
- Depends on: LFP-024
- Goal id: LFP-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/runtime_mtl_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_runtime_mtl_syntax.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_runtime_mtl_syntax.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/modal-temporal/runtime-mtl
- Parallel lane: lfp-runtime-mtl
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/runtime_mtl_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_runtime_mtl_syntax.py
- Interfaces: RuntimeMTLSyntaxAdapter@1
- Allow concurrent with: LFP-025, LFP-027
- Conflict policy: Own adapter/conformance tests; preserve runtime monitor wire compatibility.
- Preconditions: Shared MTL syntax and existing runtime monitor model pass independently.
- Effects: Runtime MTL formulas use one source/parser identity while finite traces, exact-rational time, monitorability, and three-valued verdicts remain explicit evidence properties.
- Evidence subset: syntax mapping rational intervals finite trace inconclusive verdict monitorability source maps
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Existing golden traces remain stable; Python and declared cross-runtime fixtures agree; incomplete traces never produce theorem authority.
- Embedding query: runtime mtl monitor shared temporal syntax finite trace three valued

## LFP-027 Implement HyperLTL and hyperproperty syntax and lowerings

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: hyperproperty-parser
- Depends on: LFP-024
- Goal id: LFP-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/hyper.py, ipfs_datasets_py/tests/unit/logic/parsers/test_hyper.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_hyper.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/modal-temporal/hyper
- Parallel lane: lfp-hyper
- Resource class: cpu-proof-solver
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/hyper.py, ipfs_datasets_py/tests/unit/logic/parsers/test_hyper.py
- Interfaces: HyperpropertySyntax@1, HyperLTLAdapter@1
- Allow concurrent with: LFP-025, LFP-026
- Conflict policy: Own hyper syntax/tests; provider-specific fragment checks remain in backend adapters.
- Preconditions: Temporal AST supports trace-indexed propositions and scoped trace binders.
- Effects: Hypertrace prefixes, relational predicates, noninterference templates, and tool-fragment restrictions are typed for AutoHyper/MCHyper routes.
- Evidence subset: forall exists trace prefix alternation indexed proposition tool fragment bound noninterference
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Trace variables are scoped and capture-safe; unsupported alternation reports exact cause; bounded/model-check results retain their declared authority ceiling.
- Embedding query: hyperltl hyperproperty autohyper mchyper trace quantifier noninterference parser

## LFP-028 Import legacy TDFOL, CEC/DCEC, event-calculus, legal, and modal syntax

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: legacy-modal-import
- Depends on: LFP-023, LFP-025, LFP-026, LFP-027
- Goal id: LFP-G050
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/event_calculus.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/legacy_modal.py, ipfs_datasets_py/tests/unit/logic/parsers/test_event_calculus.py, ipfs_datasets_py/tests/conformance/logic/test_legacy_modal_import.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_event_calculus.py tests/conformance/logic/test_legacy_modal_import.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/modal-temporal/legacy-join
- Parallel lane: lfp-legacy-modal
- Resource class: cpu-large
- Resource stage: migration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/event_calculus.py, ipfs_datasets_py/ipfs_datasets_py/logic/parsers/legacy_modal.py, ipfs_datasets_py/tests/unit/logic/parsers/test_event_calculus.py, ipfs_datasets_py/tests/conformance/logic/test_legacy_modal_import.py
- Interfaces: EventCalculusSyntax@1, LegacyLogicImporter@1, TDFOLProfile@1, DCECProfile@1
- Allow concurrent with: LFP-022, LFP-033
- Conflict policy: Own compatibility importer and corpus; do not delete or wholesale rewrite legacy parsers.
- Preconditions: Modal, temporal, state, runtime, and hyper syntax contracts are stable.
- Effects: A controlled event-calculus frontend owns events, fluents, initiates/terminates/releases, happens/holds, and time points; legacy ASTs/text receive explicit profile, ambiguity, loss, and source-map receipts before entering the common kernel.
- Evidence subset: TDFOL DCEC CEC event calculus deontic modal optimizer legal ambiguity capture associativity
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Controlled event-calculus syntax round-trips; unknown characters/sorts no longer disappear, implication associativity and O/P/F ambiguity are explicit, substitutions are capture-safe, and legacy golden vectors remain traceable.
- Embedding query: legacy tdfol dcec cec event calculus legal modal importer migration

## LFP-029 Implement the target-neutral symbolic protocol DSL and ProVerif adapter

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: protocol-proverif
- Depends on: LFP-016
- Goal id: LFP-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/protocol.py, ipfs_datasets_py/tests/unit/logic/parsers/test_protocol.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_protocol.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/protocol-program/proverif
- Parallel lane: lfp-protocol
- Resource class: cpu-proof-solver
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/protocol.py, ipfs_datasets_py/tests/unit/logic/parsers/test_protocol.py
- Interfaces: SymbolicProtocolSyntax@1, ProVerifControlledSource@1
- Allow concurrent with: LFP-031, LFP-032
- Conflict policy: Own neutral protocol frontend and ProVerif lowering/tests; reuse existing ProtocolIR.
- Preconditions: Typed term/signature kernel and cryptographic-protocol profiles pass.
- Effects: Terms, equations, roles, channels, adversary, events, secrecy, authentication, and correspondence claims are source-aware typed protocol nodes.
- Evidence subset: Dolev-Yao equational theory channels roles events secrecy correspondence ProVerif approximation
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Equational theory and attacker model enter identity; unsupported process constructs fail explicitly; ProVerif results retain symbolic over-approximation and query-specific authority.
- Embedding query: protocol dsl proverif applied pi attacker secrecy authentication parser

## LFP-030 Add Tamarin multiset-rewriting protocol mappings

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: protocol-tamarin
- Depends on: LFP-029
- Goal id: LFP-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tamarin.py, ipfs_datasets_py/tests/unit/logic/parsers/test_tamarin.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_tamarin.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/protocol-program/tamarin
- Parallel lane: lfp-tamarin
- Resource class: cpu-proof-solver
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/tamarin.py, ipfs_datasets_py/tests/unit/logic/parsers/test_tamarin.py
- Interfaces: TamarinControlledSource@1, ProtocolRewritingAdapter@1
- Allow concurrent with: LFP-031, LFP-032
- Conflict policy: Own controlled Tamarin mapping/tests; common protocol nodes are coordinated with LFP-029 through interfaces.
- Preconditions: Typed terms, events, facts, equations, and protocol profile descriptors pass.
- Effects: Multiset rules, persistent/linear facts, restrictions, actions, state, equations, and trace lemmas map to the neutral protocol model with receipts.
- Evidence subset: multiset rewriting facts restrictions equational theory trace lemma state Tamarin
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Mapping identifies unsupported theory/rule features and preserves event/fact provenance; Tamarin status is decoded as a tool/version/profile-bound symbolic result and cannot become proof authority without an independently replayable route.
- Embedding query: tamarin multiset rewriting protocol facts restrictions trace lemma parser

## LFP-031 Implement Hoare, contract, dynamic-logic, and verification-condition syntax

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: program-logic
- Depends on: LFP-016
- Goal id: LFP-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/program.py, ipfs_datasets_py/tests/unit/logic/parsers/test_program.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_program.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/protocol-program/program
- Parallel lane: lfp-program
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/program.py, ipfs_datasets_py/tests/unit/logic/parsers/test_program.py
- Interfaces: ProgramLogicSyntax@1, VerificationConditionBridge@1
- Allow concurrent with: LFP-029, LFP-030, LFP-032
- Conflict policy: Own program frontend/tests; reuse software_verification contracts, program, and VC IRs.
- Preconditions: Typed FOL kernel and program/refinement profiles pass.
- Effects: Hoare triples, pre/postconditions, modifies clauses, invariants, dynamic modalities, and VC roles are distinct typed syntax/view concepts.
- Evidence subset: hoare contract dynamic logic wp sp invariant modifies vc source maps lowering
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Binding and state versions are explicit; unsupported effects/loops produce obligations rather than assumptions; VC is never emitted as a semantic family ID.
- Embedding query: hoare contract dynamic logic verification condition parser weakest precondition

## LFP-032 Implement separation, concurrency, session, relational, and refinement syntax

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: resource-refinement
- Depends on: LFP-016
- Goal id: LFP-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/resource.py, ipfs_datasets_py/tests/unit/logic/parsers/test_resource.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/parsers/test_resource.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/protocol-program/resource
- Parallel lane: lfp-resource
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/resource.py, ipfs_datasets_py/tests/unit/logic/parsers/test_resource.py
- Interfaces: ResourceLogicSyntax@1, SessionProcessSyntax@1, RefinementSyntax@1
- Allow concurrent with: LFP-029, LFP-030, LFP-031
- Conflict policy: Own resource/concurrency/refinement frontend/tests; existing typed software-verification IR remains the semantic lowering target.
- Preconditions: Typed extensions and separation/concurrency/refinement profiles pass.
- Effects: Separating conjunction/implication, heap predicates, rely-guarantee, happens-before, session/process actions and duality, relational states, simulations, and refinement obligations are typed.
- Evidence subset: separation heap resource algebra concurrency rely guarantee session process duality relational refinement simulation
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Resource ownership, session channels, protocol duality, and two-state binding are capture-safe; unsupported resource algebras, process operators, or concurrency assumptions lower only with explicit loss and bounds.
- Embedding query: separation logic concurrency rely guarantee relational refinement parser resource

## LFP-033 Join protocol, program, and proof-assistant target surfaces

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: kernel-targets
- Depends on: LFP-029, LFP-030, LFP-031, LFP-032
- Goal id: LFP-G060
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/kernel_targets.py, ipfs_datasets_py/tests/conformance/logic/test_protocol_program_kernel_routes.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_protocol_program_kernel_routes.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/protocol-program/join
- Parallel lane: lfp-kernel-join
- Resource class: cpu-proof-kernel
- Resource stage: integration
- Estimated tokens: 34000
- Implementation timeout seconds: 12600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/kernel_targets.py, ipfs_datasets_py/tests/conformance/logic/test_protocol_program_kernel_routes.py
- Interfaces: TargetTheoryModel@1, KernelTargetGenerator@1, HammerStrategyReceipt@1
- Allow concurrent with: LFP-022, LFP-028
- Conflict policy: Own target-neutral theory/declaration/generator model and join tests; do not implement complete Lean, Rocq, or Isabelle parsers.
- Preconditions: Protocol/program/resource frontends and provider capability contracts pass.
- Effects: ProVerif/Tamarin, SMT/CHC, and controlled Lean/Rocq/Isabelle theorem targets receive exact imports, declarations, axioms, source maps, and trust receipts; Hammer remains a strategy provider.
- Evidence subset: target theory declarations imports axioms theorem kernel elaboration hammer reconstruction trust
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Official kernels are sole proof authority; generated sources reject sorry/admit/trust escapes; Hammer/ATP suggestions remain candidates until reconstructed; exact theorem and environment identities are recorded.
- Embedding query: lean rocq isabelle kernel theorem target generator hammer reconstruction protocol program

## LFP-034 Migrate security_ir formal views to canonical typed logic

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: security-ir
- Depends on: LFP-022, LFP-028, LFP-033
- Goal id: LFP-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/formalization_adapter_v2.py, ipfs_datasets_py/tests/conformance/logic/test_security_ir_logic_routes.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_security_ir_logic_routes.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/domains/security
- Parallel lane: lfp-security
- Resource class: cpu-proof-solver
- Resource stage: migration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/formalization_adapter_v2.py, ipfs_datasets_py/tests/conformance/logic/test_security_ir_logic_routes.py
- Interfaces: SecurityFormalizationAdapter@2
- Allow concurrent with: LFP-035, LFP-036, LFP-037, LFP-038, LFP-039
- Conflict policy: Own new security adapter and route tests; preserve existing adapter during dual-read migration.
- Preconditions: Classical, modal/temporal, protocol/program, and canonical provider routes pass.
- Effects: Threat, authorization, VC, state, temporal, protocol, noninterference, separation, and concurrency views use family/profile/view-role namespaces correctly.
- Evidence subset: z3 cvc5 tlc apalache secpal proverif tamarin hyperltl atp kernel runtime mtl
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Every admitted security view parses and elaborates before lowering; unsupported cells are explicit; proof/model/monitor authority is bounded by translation and backend receipts.
- Embedding query: security ir formalization typed logic threat authorization protocol noninterference

## LFP-035 Migrate crypto_ir cryptocurrency-network views to canonical typed logic

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: crypto-ir
- Depends on: LFP-022, LFP-028, LFP-033
- Goal id: LFP-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir/formalization/typed_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_crypto_ir_logic_routes.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_crypto_ir_logic_routes.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/domains/crypto
- Parallel lane: lfp-crypto
- Resource class: cpu-proof-solver
- Resource stage: migration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir/formalization/typed_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_crypto_ir_logic_routes.py
- Interfaces: CryptoNetworkFormalizationAdapter@1
- Allow concurrent with: LFP-034, LFP-036, LFP-037, LFP-038, LFP-039
- Conflict policy: Own typed crypto adapter/tests; retain legacy crypto LogicFamily enum as a diagnosed input alias only.
- Preconditions: Arithmetic/state/protocol/hyper/refinement syntax and provider catalog pass.
- Effects: Transactions, balances, consensus, reorg/finality, bridges, wallets, permissions, symbolic protocols, arithmetic, and privacy views gain typed routes.
- Evidence subset: ledger transition temporal refinement authorization protocol smt hyperproperty finite field future
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Cryptocurrency-network semantics name attacker, consensus, finality, bound, arithmetic, and trace assumptions; legacy smt_lib/fol labels canonicalize or fail; no future probabilistic/ZK claim is implied.
- Embedding query: crypto ir cryptocurrency network ledger consensus finality bridge typed logic

## LFP-036 Migrate intent_ir skill-prompt views to canonical typed logic

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: intent-ir
- Depends on: LFP-022, LFP-028, LFP-033
- Goal id: LFP-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py, ipfs_datasets_py/tests/conformance/logic/test_intent_ir_logic_routes.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_intent_ir_logic_routes.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/domains/intent
- Parallel lane: lfp-intent
- Resource class: cpu-proof-solver
- Resource stage: migration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/intent_ir/formalize/typed_compiler.py, ipfs_datasets_py/tests/conformance/logic/test_intent_ir_logic_routes.py
- Interfaces: IntentFormalizationCompiler@2
- Allow concurrent with: LFP-034, LFP-035, LFP-037, LFP-038, LFP-039
- Conflict policy: Own typed intent compiler/tests; preserve prompt/source provenance and withhold unsafe inferred obligations.
- Preconditions: FOL, normative/BDI, program, workflow temporal, policy, and kernel routes pass.
- Effects: Skill goals, tool permissions, guards/effects, norms, intentions, workflows, safety/liveness properties, and VCs use separate canonical namespaces.
- Evidence subset: typed first order intention deontic dynamic hoare workflow temporal authorization tool flow
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Safety/liveness remain property kinds and VC remains a view role; prompt-derived formulas are candidates until deterministic parsing/typechecking/verification; tool authority never follows confidence alone.
- Embedding query: intent ir skill prompt formalization bdi deontic workflow hoare permissions

## LFP-037 Migrate legal_ir views to canonical normative, temporal, rule, and event logic

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: legal-ir
- Depends on: LFP-022, LFP-028, LFP-033
- Goal id: LFP-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/typed_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_legal_ir_logic_routes.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_legal_ir_logic_routes.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/domains/legal
- Parallel lane: lfp-legal
- Resource class: cpu-proof-solver
- Resource stage: migration
- Estimated tokens: 34000
- Implementation timeout seconds: 12600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/typed_adapter.py, ipfs_datasets_py/tests/conformance/logic/test_legal_ir_logic_routes.py
- Interfaces: LegalFormalizationAdapter@2
- Allow concurrent with: LFP-034, LFP-035, LFP-036, LFP-038, LFP-039
- Conflict policy: Own typed legal adapter/tests; preserve citations, spans, coverage, ambiguity, and withholding receipts.
- Preconditions: Normative/temporal/event, FOL/TPTP, rule/frame, state, and kernel routes pass.
- Effects: Conditional/defeasible norms, exceptions/priorities, temporal FOL, events, authorization/rules, and frames become typed views; argumentation is declaration-only until a refill task supplies a reviewed frontend; graph projection stays an operation role.
- Evidence subset: deontic conditional defeasible temporal first order event calculus frame rule argument ambiguity
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Norm conflicts and ambiguity are explicit; argumentation receives an explicit declaration-only/unsupported disposition; graph_projection/proof_translation/structural_round_trip never route as families; natural-language extraction is never proof authority.
- Embedding query: legal ir deontic defeasible temporal fol event calculus frame logic

## LFP-038 Register the exact-source UI/UX migration gate without touching user work

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ui-ir
- Depends on: LFP-022, LFP-028, LFP-033
- Goal id: LFP-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_source_gate.py, ipfs_datasets_py/tests/conformance/logic/test_ui_ux_source_gate.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_ui_ux_source_gate.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/domains/ui
- Parallel lane: lfp-ui
- Resource class: cpu-proof-solver
- Resource stage: migration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/ui_ux_source_gate.py, ipfs_datasets_py/tests/conformance/logic/test_ui_ux_source_gate.py
- Interfaces: UIUXSourceGate@1, UIUXFormalizationAdapter@1
- Allow concurrent with: LFP-034, LFP-035, LFP-036, LFP-037, LFP-039
- Conflict policy: This seed task owns only the source gate and test. Never recreate, copy, or edit ui_ux_ir. After the user's exact UI/UX tree is committed/imported and the accelerator gitlink plus LFP-001/LFP-005 baseline are refreshed, emit a content-addressed derived adapter task that preserves graph schemas, source maps, authority flags, and golden vectors.
- Preconditions: Pinned source identity and baseline inventory are available; no external/untracked checkout is imported implicitly.
- Effects: An absent UI/UX package receives an explicit source_not_in_pinned_revision/declaration-only matrix disposition and a revision-triggered refill rule; a future exact source commit generates the owner-scoped component/frame, event, TDFOL/DCEC, navigation/state, accessibility, permission, privacy, and runtime-journey adapter task.
- Evidence subset: ui ux exact source gate revision trigger declaration only refill adapter graph source map
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: The current absent UI/UX source causes no ui_ux_ir writes and yields a typed, content-addressed external-source gate rather than a blocked lane; when a new pinned revision contains it, identical scanning emits exactly one derived migration task whose acceptance requires declared-syntax parsing, frame_logic alias canonicalization, and typed structural round trips rather than token presence.
- Embedding query: ui ux ir typed formalization component event journey accessibility frame logic

## LFP-039 Bridge software_verification IRs through the shared syntax kernel

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: software-verification-ir
- Depends on: LFP-022, LFP-028, LFP-033
- Goal id: LFP-G070
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/syntax_bridge.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/syntax_bridge.py, ipfs_datasets_py/tests/conformance/logic/test_software_verification_syntax_bridge.py, ipfs_datasets_py/tests/conformance/logic/test_software_contracts_syntax_bridge.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_software_verification_syntax_bridge.py tests/conformance/logic/test_software_contracts_syntax_bridge.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/domains/software-verification
- Parallel lane: lfp-software-verification
- Resource class: cpu-proof-solver
- Resource stage: migration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/syntax_bridge.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/syntax_bridge.py, ipfs_datasets_py/tests/conformance/logic/test_software_verification_syntax_bridge.py, ipfs_datasets_py/tests/conformance/logic/test_software_contracts_syntax_bridge.py
- Interfaces: SoftwareVerificationSyntaxBridge@1, SoftwareContractsSyntaxBridge@1
- Allow concurrent with: LFP-034, LFP-035, LFP-036, LFP-037, LFP-038
- Conflict policy: Own bridge/conformance tests; retain rich typed software-verification IRs as semantic lowerings rather than replacing them.
- Preconditions: All parser/backend join contracts pass.
- Effects: State, transitions, programs, contracts, VCs, temporal, trace, authorization, protocol, hyperproperty, heap, concurrency, refinement, and the software_contracts AST IR can publish/consume typed expressions.
- Evidence subset: software verification typed ir transition contract vc trace authorization protocol heap concurrency refinement
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Round trips preserve domain invariants and source identities; no bridge weakens existing typed models to arbitrary JSON/text; loss and unsupported semantics are explicit.
- Embedding query: software verification ir syntax bridge contracts vc transition protocol refinement

## LFP-040 Build the domain-view-family-provider cross-product conformance suite

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: cross-product-conformance
- Depends on: LFP-034, LFP-035, LFP-036, LFP-037, LFP-038, LFP-039
- Goal id: LFP-G080
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/generated_catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/runner.py, ipfs_datasets_py/tests/conformance/logic/test_parser_catalog.py, ipfs_datasets_py/tests/conformance/logic/test_registry_closure.py, ipfs_datasets_py/tests/conformance/logic/test_domain_provider_matrix.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_parser_catalog.py tests/conformance/logic/test_registry_closure.py tests/conformance/logic/test_domain_provider_matrix.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/conformance/matrix
- Parallel lane: lfp-conformance-matrix
- Resource class: cpu-proof-solver
- Resource stage: validation
- Estimated tokens: 34000
- Implementation timeout seconds: 12600
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/parsers/catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/generated_catalog.py, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/runner.py, ipfs_datasets_py/tests/conformance/logic/test_parser_catalog.py, ipfs_datasets_py/tests/conformance/logic/test_registry_closure.py, ipfs_datasets_py/tests/conformance/logic/test_domain_provider_matrix.py
- Interfaces: LogicParserCatalog@1, GeneratedProviderTranslationCatalog@1, LogicConformanceRunner@1
- Allow concurrent with:
- Conflict policy: Serialized final parser/provider/translation catalog projection and generic matrix join; consume local inert parser descriptors and domain/provider fixtures without editing individual adapters.
- Preconditions: Five pinned domain migrations, software verification/contracts bridges, and the UI/UX exact-source gate pass against the same canonical registries.
- Effects: The complete lazy parser descriptor catalog and final generated provider/translation projection close against canonical registries; every domain x view x family/profile x translation x provider cell is native, lossless, approximate, bounded, declaration-only, advisor-only, unavailable, or unsupported with a reason.
- Evidence subset: security crypto intent legal ui software verification all providers preservation authority
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Every individual parser contributes an inert local descriptor without editing a shared registry; the final catalog has no duplicate/eager/unknown entry; the suite contains every exact provider ID and domain, rejects unexplained registry/matrix gaps, and executes hermetically or emits typed unavailable evidence without false skips.
- Embedding query: cross product conformance domain view family provider matrix authority

## LFP-041 Add fuzzing, parser-bomb, Unicode, and performance/resource gates

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: parser-hardening
- Depends on: LFP-040
- Goal id: LFP-G080
- Outputs: ipfs_datasets_py/tests/fuzz/logic/test_parser_properties.py, ipfs_datasets_py/tests/conformance/logic/test_parser_resource_limits.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/fuzz/logic/test_parser_properties.py tests/conformance/logic/test_parser_resource_limits.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/conformance/hardening
- Parallel lane: lfp-hardening
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/tests/fuzz/logic/test_parser_properties.py, ipfs_datasets_py/tests/conformance/logic/test_parser_resource_limits.py
- Interfaces: ParserResourcePolicy@1
- Allow concurrent with: LFP-042
- Conflict policy: Own hardening tests/fixtures; production fixes return to the owning parser task through derived gaps.
- Preconditions: Cross-product corpus and parser registry enumerate all admitted frontends.
- Effects: Depth, node, token, input, diagnostic, recovery, normalization, and wall-time budgets are exercised against random/adversarial inputs.
- Evidence subset: property tests alpha substitution roundtrip unicode confusable nul nesting ambiguity parser bomb
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: All parsers terminate within declared bounds, fail closed on exhaustion, preserve exact spans, reject silent drops, and expose stable reduced counterexamples.
- Embedding query: parser fuzz property unicode confusable resource limit parser bomb performance

## LFP-042 Audit advisor, solver, Hammer, and proof-kernel authority boundaries

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-audit
- Depends on: LFP-040
- Goal id: LFP-G080
- Outputs: ipfs_datasets_py/tests/conformance/logic/test_authority_boundaries.py, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/authority_audit.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_authority_boundaries.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/conformance/authority
- Parallel lane: lfp-authority
- Resource class: cpu-proof-kernel
- Resource stage: validation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/tests/conformance/logic/test_authority_boundaries.py, ipfs_datasets_py/ipfs_datasets_py/logic/conformance/authority_audit.py
- Interfaces: LogicAuthorityAudit@1
- Allow concurrent with: LFP-041
- Conflict policy: Own audit/tests; provider adapters may be patched only through newly derived owner-scoped tasks.
- Preconditions: Domain-provider matrix and translation authority vocabulary pass.
- Effects: SymAI/ErgoAI, Hammer, ATPs, SMT, model checkers, protocol tools, monitors, and kernels face adversarial promotion and trust-escape tests.
- Evidence subset: symai ergoai advisor candidate hammer premise solver model checker bounded monitor kernel axiom imports
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Confidence never proves parse correctness; generic success never becomes proof; quota/unavailability never becomes logic evidence; only official kernel success under pinned imports establishes kernel authority.
- Embedding query: authority audit symai ergoai hammer solver kernel trust evidence

## LFP-043 Join differential, metamorphic, reconstruction, and end-to-end evidence

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance-join
- Depends on: LFP-041, LFP-042
- Goal id: LFP-G080
- Outputs: ipfs_datasets_py/tests/conformance/logic/test_differential_and_reconstruction.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_conformance_report.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_differential_and_reconstruction.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/conformance/join
- Parallel lane: lfp-conformance-join
- Resource class: cpu-proof-kernel
- Resource stage: validation
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_datasets_py/tests/conformance/logic/test_differential_and_reconstruction.py, ipfs_datasets_py/docs/architecture/logic/logic_parser_conformance_report.json
- Interfaces: LogicConformanceReport@1
- Allow concurrent with:
- Conflict policy: Own joined tests/report; disagreements produce evidence records and derived gaps rather than majority-vote fixes.
- Preconditions: Cross-product, hardening, and authority suites pass.
- Effects: Z3/cvc5, Vampire/E, TLC/Apalache common fragments, aligned ProVerif/Tamarin and HyperLTL fragments, runtime monitors, and kernel reconstruction are compared under exact contracts.
- Evidence subset: differential metamorphic translation preservation reconstruction disagreement inconclusive end to end
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Disagreement is typed inconclusive; every translation has positive/negative preservation fixtures; high-assurance candidates reconstruct or retain a lower authority ceiling; report is deterministic and content-addressed.
- Embedding query: differential solver conformance metamorphic translation reconstruction end to end

## LFP-044 Implement public API dual-read and canonical-write migration

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: api-migration
- Depends on: LFP-043
- Goal id: LFP-G090
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/api.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/conformance/logic/test_public_api_migration.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/test_api.py tests/unit/logic/test_verification_api.py tests/conformance/logic/test_public_api_migration.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/migration/api
- Parallel lane: lfp-api-migration
- Resource class: cpu-medium
- Resource stage: migration
- Estimated tokens: 28000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/api.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py, ipfs_datasets_py/tests/conformance/logic/test_public_api_migration.py
- Interfaces: VerificationAPI@2, CanonicalLogicDiscovery@1
- Allow concurrent with:
- Conflict policy: Own versioned API and migration tests; preserve current public entry until explicit deprecation gate.
- Preconditions: Joined conformance report passes and canonical catalogs are stable.
- Effects: Discovery exposes separate family/profile/property/view/notation/provider/encoding/evidence namespaces; legacy aliases are read with diagnostics and never written.
- Evidence subset: public api discovery alias dual read canonical write deprecation serialization
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Existing accepted artifacts migrate deterministically; new artifacts contain no legacy/free-form family labels; callers can inspect translation loss and provider authority without backend-specific heuristics.
- Embedding query: public verification api migration canonical family provider discovery dual read

## LFP-045 Migrate documentation and consumer surfaces, then close legacy drift

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: consumer-migration
- Depends on: LFP-044
- Goal id: LFP-G090
- Outputs: ipfs_datasets_py/docs/architecture/logic/LOGIC_SYNTAX_AND_FAMILY_CONTRACTS.md, ipfs_datasets_py/tests/conformance/logic/test_consumer_family_closure.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/conformance/logic/test_consumer_family_closure.py
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/migration/consumers
- Parallel lane: lfp-consumer-migration
- Resource class: cpu-medium
- Resource stage: migration
- Estimated tokens: 26000
- Implementation timeout seconds: 9000
- Predicted files: ipfs_datasets_py/docs/architecture/logic/LOGIC_SYNTAX_AND_FAMILY_CONTRACTS.md, ipfs_datasets_py/tests/conformance/logic/test_consumer_family_closure.py
- Interfaces: LogicConsumerClosure@1
- Allow concurrent with:
- Conflict policy: Own normative syntax/family document and closure test; derived tasks own individual stale consumers.
- Preconditions: Conformance report identifies every public/internal consumer and alias.
- Effects: Normative architecture, examples, registries, import surfaces, docs, and declared generated catalogs converge; stale overclaims are diagnosed.
- Evidence subset: documentation examples import consumer alias closure generated catalog deprecation
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Static closure records every unregistered emitted ID, undocumented controlled syntax, stale consumer, and failing public example as an owner-scoped typed gap; historical plans are clearly nonnormative; LFP-046, not this discovery task, requires the drained zero-drift fixed point.
- Embedding query: logic documentation consumer migration family closure legacy drift

## LFP-046 Run bounded objective refill to a current-tree fixed point

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: objective-refill
- Depends on: LFP-044, LFP-045
- Goal id: LFP-G090
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill.py, ipfs_datasets_py/tests/unit/logic/conformance/test_refill.py, data/agent_supervisor/ipfs_datasets_logic_family_parser/refill/fixed_point_receipt.json, data/agent_supervisor/ipfs_datasets_logic_family_parser/refill/gap_ledger.jsonl
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/conformance/test_refill.py && cd .. && python scripts/validate_ipfs_datasets_logic_family_parser_board.py --check-all
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/migration/refill
- Parallel lane: lfp-refill
- Resource class: cpu-medium
- Resource stage: control
- Estimated tokens: 24000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/conformance/refill.py, ipfs_datasets_py/tests/unit/logic/conformance/test_refill.py, data/agent_supervisor/ipfs_datasets_logic_family_parser/refill/fixed_point_receipt.json, data/agent_supervisor/ipfs_datasets_logic_family_parser/refill/gap_ledger.jsonl
- Interfaces: LogicGapRefill@1, ObjectiveRefillFixedPoint@1
- Allow concurrent with:
- Conflict policy: Derived tasks live in a control-plane-owned append-only derived section/ledger merged with the sealed immutable seed projection and may edit only owner-scoped implementation/evidence paths; seed task definitions and protected control artifacts are immutable.
- Preconditions: API and consumer migration tests pass; objective scanner is bound to exact current trees.
- Effects: Bounded content-addressed tasks are generated from uncovered matrix cells, unregistered labels, unsupported nodes, missing fixtures, differential disagreements, and unreconstructed candidates.
- Evidence subset: refill epoch goal gap task digest dependency dedupe retry cooldown fixed point
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Two consecutive scans over identical source/config/corpus identities produce no new admissible tasks; per-epoch derived limits exclude the 11 immutable seed goals; open/attempt/depth limits hold; seed definitions never change; duplicates and broad unscoped codebase tasks are rejected.
- Embedding query: objective refill derived tasks fixed point gap ledger content addressed

## LFP-047 Seal the logic-parser release receipt

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: release
- Depends on: LFP-046
- Goal id: LFP-G100
- Outputs: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/logic/syntax_core tests/unit/logic/parsers tests/unit/logic/families tests/unit/logic/formalization tests/unit/logic/backends tests/unit/logic/security_ir tests/unit/logic/crypto_ir tests/unit/logic/intent_ir tests/unit/logic/legal_ir tests/unit/logic/software_verification tests/unit/logic/software_contracts tests/unit/logic/conformance/test_refill.py tests/conformance/logic tests/fuzz/logic && cd .. && python scripts/validate_ipfs_datasets_logic_family_parser_board.py --check-all
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Bundle: logic-family-parser/release
- Parallel lane: lfp-release
- Resource class: cpu-proof-kernel
- Resource stage: release
- Estimated tokens: 22000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_RELEASE.md, ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json
- Interfaces: LogicParserReleaseReceipt@1
- Allow concurrent with:
- Conflict policy: Review/evidence aggregation only; no semantic implementation changes are permitted in the release task.
- Preconditions: Refill fixed point is valid and every parent goal has current-tree evidence.
- Effects: Exact datasets and accelerator revisions, registries, schemas, corpus, matrix, translations, providers, tests, tools, bounds, assumptions, authority ceilings, and remaining declaration-only gaps are bound into one immutable receipt.
- Evidence subset: release git identity cid registry schema corpus matrix provider validation authority fixed point
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: All LFP seed and derived tasks are terminal; all goals are verified or explicitly blocked with user-approved disposition; zero silent semantic loss, false capability, authority escalation, trust escape, or unexplained matrix gap remains.
- Embedding query: logic parser release receipt proof authority fixed point current tree

## LFP-048 Resolve 2 preflight-conflicting backlogged worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: preflight_merge_conflict
- Reconciliation reason: preflight_merge_conflict
- Reconciliation fingerprint: 0215bd6d31c82e1fdfd6a7a29061095a33a4a4f5
- Reconciliation discovery: /home/barberb/lift_coding/.worktrees/logic-family-parser-supervisor-runtime/data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery/2026-08-09-lfp-048-reconciliation-0215bd6d31c8.md
- Resolution receipt digest: sha256:32538129c5f298371af503fe3cfed75e4f7f511c179a9307508b45564f0da889
- Canonical board task: false
- Fingerprint: 0215bd6d31c82e1fdfd6a7a29061095a33a4a4f5
- Dedupe key: reconciliation_guardrail:preflight_merge_conflict
- Depends on:
- Outputs: data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery, docs/architecture/ipfs_datasets_logic_family_parser.todo.md
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Goal id: LFP-G000
- Bundle: logic-family-parser/control
- Parallel lane: lfp-control
- Resource class: cpu-small
- Validation: test -f /home/barberb/lift_coding/.worktrees/logic-family-parser-supervisor-runtime/data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery/2026-08-09-lfp-048-reconciliation-0215bd6d31c8.md
- Acceptance: Reconciliation guardrail filed this because 2 branch or worktree cleanup candidates are blocked by preflight_merge_conflict. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.worktrees/logic-family-parser-supervisor-runtime/data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery/2026-08-09-lfp-048-reconciliation-0215bd6d31c8.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## LFP-049 Resolve dirty main checkout blocking 2 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1
- Reconciliation kind: main_checkout_dirty
- Reconciliation reason: main_checkout_dirty
- Reconciliation fingerprint: c3abe0c7a4e083a96ef866487d4feabd31116149
- Reconciliation discovery: /home/barberb/lift_coding/.worktrees/logic-family-parser-supervisor-runtime/data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery/2026-08-09-lfp-049-reconciliation-c3abe0c7a4e0.md
- Resolution receipt digest: sha256:0577b86dd16b6b775adef7bbe0017a76e0bc1b3254e48fa7ca8e9dfc9556b9ca
- Canonical board task: false
- Fingerprint: c3abe0c7a4e083a96ef866487d4feabd31116149
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery, docs/architecture/ipfs_datasets_logic_family_parser.todo.md
- Board namespace: ipfs-datasets-logic-family-parser-v1
- Goal id: LFP-G000
- Bundle: logic-family-parser/control
- Parallel lane: lfp-control
- Resource class: cpu-small
- Validation: test -f /home/barberb/lift_coding/.worktrees/logic-family-parser-supervisor-runtime/data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery/2026-08-09-lfp-049-reconciliation-c3abe0c7a4e0.md
- Acceptance: Reconciliation guardrail filed this because 2 branch or worktree cleanup candidates are blocked by main_checkout_dirty. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.worktrees/logic-family-parser-supervisor-runtime/data/agent_supervisor/ipfs_datasets_logic_family_parser/state/discovery/2026-08-09-lfp-049-reconciliation-c3abe0c7a4e0.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.
