# Logic Formal Verification Expansion Objective Heap

Ultimate goal: expose a sound, extensible, software-verification logic
platform through `ipfs_datasets_py.logic`, reuse the existing prover and
supervisor machinery, and let `ipfs_accelerate_py.agent_supervisor` execute
property-specific prover portfolios without confusing proposals, monitors,
bounded checks, policy decisions, caches, or attestations with theorem proof.

Program invariants:

- `ipfs_datasets_py.logic.ir_core` and `logic.backends` are the canonical
  semantic and provider contracts.
- `ipfs_accelerate_py.agent_supervisor` owns orchestration, resource leases,
  isolation, durable scheduling, and operational monitoring.
- Imports and declaration discovery perform no install, network, process,
  environment, or write side effects.
- Every translation declares preservation class, assumptions, bounds,
  unsupported constructs, and assurance ceiling.
- Result authorities are typed and non-interchangeable.
- Autoencoder, Leanstral, SymAI, embeddings, and other learned components are
  proposal/advisor sources only.
- Hammer search becomes authoritative only through the declared
  solver/kernel reconstruction policy.
- Exact cache hits inherit, but never increase, the authority of a validated
  current receipt.
- ZKP attests an existing receipt; it does not prove the source-to-logic
  translation or raise semantic assurance.
- Tool absence, unsupported input, timeout, malformed output, and semantic
  loss are explicit non-success states.
- Existing legal, DCEC, TDFOL, CEC, deontic, modal, and frame-logic public
  behavior remains available through compatibility adapters.

## LFV-G000 General-purpose formal verification logic platform

- Status: provisionally_complete
- Parent:
- Depends on: LFV-G083
- Fib priority: 4181
- Priority: P0
- Track: integration
- Bundle: logic-formal-verification/quality
- Goal: Complete and attest the software-verification logic-family, provider, utility, public-API, and supervisor integration program.
- Evidence: docs/architecture/logic_formal_verification_expansion_completion_receipt.json
- Outputs: docs/architecture/logic_formal_verification_expansion_completion_receipt.json
- Validation: python -m pytest test/api/test_logic_formal_verification_completion.py -q
- Acceptance: A current-tree receipt binds the parent and datasets commits, all child receipts, capability matrix, translations, fixtures, external-tool identities, conformance population, benchmark report, rollout policy, and zero proof-authority boundary violations.
- Conflict policy: Tracking-only root; reconcile only after LFV-G083 and every executable child goal are complete.
- Interfaces: LogicFormalVerificationProgram@1
- Resource class: cpu-validation

## LFV-G005 Align and continuously verify the datasets submodule

- Status: active
- Parent: LFV-G000
- Depends on:
- Fib priority: 1
- Priority: P0
- Track: foundation
- Bundle: logic-formal-verification/foundation
- Goal: Bind the implementation branch to the intended published `ipfs_datasets_py` revision and detect sibling, remote-main, and embedded-gitlink drift before workers or tests run.
- Evidence: tools/logic/verify_submodule_alignment.py, test/api/test_logic_submodule_alignment.py
- Outputs: tools/logic/verify_submodule_alignment.py, test/api/test_logic_submodule_alignment.py
- Validation: python -m pytest test/api/test_logic_submodule_alignment.py -q
- Acceptance: The check reports parent commit, gitlink, embedded HEAD, embedded origin/main, cleanliness, and availability of required logic modules; mismatches fail with actionable diagnostics and never rewrite a checkout.
- Conflict policy: Own the read-only alignment checker and its parent-repository tests; do not fetch, checkout, commit, publish, or edit datasets source during validation.
- Interfaces: LogicSubmoduleAlignment@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-small

## LFV-G010 Produce an executable capability census and maturity matrix

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G005
- Fib priority: 2
- Priority: P0
- Track: foundation
- Bundle: logic-formal-verification/foundation
- Goal: Inventory every logic family, compiler, provider, installer, probe, adapter, conformance suite, authority role, and public access path across datasets and the supervisor.
- Evidence: ipfs_datasets_py/docs/logic/software_verification_capability_inventory.md, ipfs_datasets_py/tests/fixtures/logic/software_verification/capability_matrix.json, ipfs_datasets_py/tests/unit/logic/software_verification/test_capability_inventory.py
- Outputs: ipfs_datasets_py/docs/logic/software_verification_capability_inventory.md, ipfs_datasets_py/tests/fixtures/logic/software_verification/capability_matrix.json, ipfs_datasets_py/tests/unit/logic/software_verification/test_capability_inventory.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_capability_inventory.py -q
- Acceptance: The matrix distinguishes declared, discoverable, installed, smoke-tested, translation-conformant, reconstruction-capable, shadow, canary, and authoritative-for states; stale paths and inconsistent metadata are test failures.
- Conflict policy: Own only the new inventory, fixture, and inventory test; inspect existing implementations without editing their behavior or public registries.
- Interfaces: LogicCapabilityMatrix@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-small

## LFV-G011 Freeze the existing public logic and proof-authority surface

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G005
- Fib priority: 2
- Priority: P0
- Track: compatibility
- Bundle: logic-formal-verification/foundation
- Goal: Freeze reviewed imports, API/CLI/MCP behavior, result meanings, and representative canonical payloads before adding general software-verification surfaces.
- Evidence: ipfs_datasets_py/docs/logic/logic_api_v1_compatibility.md, ipfs_datasets_py/tests/fixtures/logic/api_v1/manifest.json, ipfs_datasets_py/tests/unit/logic/test_logic_api_v1_compatibility.py
- Outputs: ipfs_datasets_py/docs/logic/logic_api_v1_compatibility.md, ipfs_datasets_py/tests/fixtures/logic/api_v1/manifest.json, ipfs_datasets_py/tests/unit/logic/test_logic_api_v1_compatibility.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/test_logic_api_v1_compatibility.py -q
- Acceptance: FOL, deontic, modal, CEC/DCEC, TDFOL, FLogic, bridges, caches, ZKP, API, CLI, MCP, and lazy import behavior have exact reviewed fixtures; optional-tool absence is recorded separately from success.
- Conflict policy: Add compatibility documentation, fixtures, and tests only; do not edit production logic modules, exports, registries, or generated artifacts.
- Interfaces: LogicAPICompatibility@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-small

## LFV-G012 Define the canonical logic-family and provider-capability taxonomy

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G010
- Fib priority: 3
- Priority: P0
- Track: family-taxonomy
- Bundle: logic-formal-verification/foundation
- Goal: Define versioned family, fragment, property, operation, runtime, evidence, boundedness, translation, and provider-capability descriptors shared by datasets and the supervisor, including Horn/CHC, PDR/IC3, and declaration-only families.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/families/models.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/registry.py, ipfs_datasets_py/tests/unit/logic/families/test_registry.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/families/models.py, ipfs_datasets_py/ipfs_datasets_py/logic/families/registry.py, ipfs_datasets_py/tests/unit/logic/families/test_registry.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/families/test_registry.py -q
- Acceptance: The registry covers existing and planned families including Horn/CHC and PDR/IC3 operations, rejects alias collisions and silent semantic equivalence, declares native, translated, declaration-only, and unsupported support, serializes deterministically, and imports no provider runtime.
- Conflict policy: Own the new family package and tests; do not edit shared package exports, backend registry, supervisor enums, or existing family implementations.
- Interfaces: LogicFamilyRegistry@1, ProviderCapabilityDescriptor@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G013 Generalize typed backend results and authority normalization

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G012
- Fib priority: 5
- Priority: P0
- Track: typed-results
- Bundle: logic-formal-verification/foundation
- Goal: Normalize theorem, satisfiability, model-check, monitor, authorization, protocol, hyperproperty, candidate, reconstruction, and attestation outcomes without collapsing their authority.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/results.py, ipfs_datasets_py/tests/unit/logic/backends/test_typed_results.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/results.py, ipfs_datasets_py/tests/unit/logic/backends/test_typed_results.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_typed_results.py -q
- Acceptance: Typed normalizers preserve unknown, timeout, unavailable, unsupported, malformed, witness, assumptions, bounds, translation ceiling, and resource usage; adversarial tests reject every cross-authority substitution.
- Conflict policy: Own the new result-normalization module and tests; adapt existing `ir_core.protocols` by composition and defer registry integration to LFV-G070.
- Interfaces: TypedBackendResult@1, ResultAuthorityNormalization@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-type-check

## LFV-G014 Build one bounded external-tool process lifecycle

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G012
- Fib priority: 5
- Priority: P0
- Track: runtime
- Bundle: logic-formal-verification/foundation
- Goal: Provide a reusable injected runner for native, JVM, OCaml/opam, and WASM-capable tools with strict process, path, time, output, memory, cancellation, and cleanup controls.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/process.py, ipfs_datasets_py/tests/unit/logic/backends/test_process_lifecycle.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/process.py, ipfs_datasets_py/tests/unit/logic/backends/test_process_lifecycle.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_process_lifecycle.py -q
- Acceptance: Argument arrays avoid shell interpretation; workspaces are isolated; timeouts terminate process trees; output and paths are bounded; secrets are redacted; fake runners make tests deterministic; imports and probes install nothing.
- Conflict policy: Adapt patterns from Hammer and supervisor runners into the new leaf module; do not change installers or any concrete prover adapter.
- Interfaces: BoundedToolRunner@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G015 Reconcile datasets and supervisor provider contracts

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G005, LFV-G012, LFV-G013
- Fib priority: 8
- Priority: P0
- Track: provider-contract
- Bundle: logic-formal-verification/foundation
- Goal: Define a stable wire/provider contract usable by datasets logic and thin supervisor adapters without creating a cyclic import or a fifth provider abstraction.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, test/api/test_logic_provider_contract.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/provider.py, ipfs_accelerate_py/agent_supervisor/logic_provider_contract.py, test/api/test_logic_provider_contract.py
- Validation: python -m pytest test/api/test_logic_provider_contract.py ipfs_datasets_py/tests/unit/logic/backends -q
- Acceptance: Requests/responses round trip canonically across the submodule boundary; provider discovery stays lazy; cancellations and resources are representable; supervisor compatibility is additive; dataset code never imports the parent package.
- Conflict policy: Own the new provider contract, supervisor facade, and contract test; do not register concrete providers or edit routing policy.
- Interfaces: LogicProvider@1, SupervisorLogicProviderFacade@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G020 Create the shared software-verification IR and property vocabulary

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G012
- Fib priority: 5
- Priority: P0
- Track: verification-ir
- Bundle: logic-formal-verification/semantics-core
- Goal: Define immutable source-grounded software-verification documents, declarations, properties, assumptions, bounds, diagnostics, artifacts, and canonical identities above provider syntax.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/ir.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/properties.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_ir.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/ir.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/properties.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_ir.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_ir.py -q
- Acceptance: Documents are immutable or defensively copied, canonical identity excludes observational output, every property and assumption is source mapped, extensions are namespaced, and unsupported constructs survive as diagnostics.
- Conflict policy: Own only the new IR/property modules and tests; reuse `ir_core` and do not edit domain IRs, package exports, or backend code.
- Interfaces: SoftwareVerificationIR@1, VerificationProperty@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G021 Implement loss-aware cross-logic translation receipts

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G020
- Fib priority: 8
- Priority: P0
- Track: translation-receipts
- Bundle: logic-formal-verification/semantics-core
- Goal: Make every exact, equisatisfiable, conservative, bounded, approximate, or heuristic translation explicit and content addressed.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/translations.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/receipts.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_translations.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/translations.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/receipts.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_translations.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_translations.py -q
- Acceptance: Receipts bind source/target family and versions, compilers, assumptions, bounds, unsupported constructs, preservation claim, witnesses, semantic mutations, and authority ceiling; missing or stale receipts fail closed.
- Conflict policy: Own translation and receipt leaf modules; adapt the supervisor translation vocabulary without editing its router until LFV-G072.
- Interfaces: LogicTranslationReceipt@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-translate

## LFV-G022 Add state, transition, action-system, and Kripke semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021
- Fib priority: 13
- Priority: P0
- Track: state-semantics
- Bundle: logic-formal-verification/semantics-state
- Goal: Represent typed state variables, initial states, actions, transition relations, fairness, invariants, variants, labels, and Kripke structures independently of TLA or SMT syntax.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/state.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/transitions.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_state.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/state.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/transitions.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_state.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_state.py -q
- Acceptance: State schemas are typed and deterministic; actions expose read/write frames; initial/next/invariant/fairness roles are distinct; finite bounds are explicit; invalid or ambiguous transitions fail closed.
- Conflict policy: Own state/transition leaf modules and tests; do not emit TLA, execute a model checker, or edit shared exports.
- Interfaces: StateTransitionIR@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G023 Add event, trace, LTL, LTLf, MTL, CTL, and CTL-star semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021
- Fib priority: 13
- Priority: P0
- Track: temporal-semantics
- Bundle: logic-formal-verification/semantics-state
- Goal: Define typed events, clocks, finite and infinite traces, path quantification, temporal formulas, intervals, observation policies, and monitorability.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py -q
- Acceptance: Finite-prefix, infinite-trace, and branching-time semantics are non-interchangeable; time units and interval boundaries are canonical; monitorable fragments are declared; CTL/CTL-star remain declaration/translation-only until a conformant semantics-preserving backend exists; clean prefixes never imply global proof.
- Conflict policy: Own trace/temporal leaf modules and tests; adapt existing event/temporal types by explicit conversion without modifying them.
- Interfaces: TraceIR@1, TemporalFormula@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G024 Add program, CFG, contract, Hoare, and dynamic-logic semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021
- Fib priority: 13
- Priority: P0
- Track: program-semantics
- Bundle: logic-formal-verification/semantics-program
- Goal: Define language-neutral functions, commands, expressions, CFGs, pre/postconditions, loop invariants, variants, exceptions, effects, and dynamic-logic modalities.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/program.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/contracts.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_program.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/program.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/contracts.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_program.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_program.py -q
- Acceptance: Source locations and evaluation order are retained; normal and exceptional exits are separate; purity, effects, frames, and undefined behavior are explicit; malformed CFGs and unbound symbols fail closed.
- Conflict policy: Own program/contract modules and tests; do not build language frontends or solver syntax in this task.
- Interfaces: ProgramIR@1, ProgramContract@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G025 Generate weakest preconditions and verification conditions

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G021, LFV-G024
- Fib priority: 21
- Priority: P0
- Track: vc-generation
- Bundle: logic-formal-verification/semantics-program
- Goal: Generate source-bound weakest-precondition and verification-condition obligations for contracts, branches, loops, exceptions, frames, and resource assertions.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py -q
- Acceptance: Each obligation binds its source construct, assumptions, generated symbols, rule, and parent contract; loop rules require invariant/variant policy; unsupported effects remain explicit; mutation tests detect dropped branches and frames.
- Conflict policy: Own the VC generator and test; consume program and translation contracts without editing their definitions or any provider.
- Interfaces: VerificationConditionGenerator@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-translate

## LFV-G026 Add separation, heap, ownership, and resource logic

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G024
- Fib priority: 21
- Priority: P1
- Track: separation-logic
- Bundle: logic-formal-verification/semantics-program
- Goal: Represent heaps, points-to assertions, separating conjunction, permissions, ownership transfer, disjointness, resource algebras, and frame obligations.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py -q
- Acceptance: Ownership and aliasing are typed; separating and ordinary conjunction differ; permissions are bounded and conserved; frame inference emits explicit obligations; unsupported heap theories cannot silently lower to plain FOL.
- Conflict policy: Own heap/separation modules and tests; defer provider encodings and exports.
- Interfaces: SeparationLogicIR@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G027 Add concurrency, rely-guarantee, session, and refinement semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G022, LFV-G024
- Fib priority: 34
- Priority: P1
- Track: concurrency-refinement
- Bundle: logic-formal-verification/semantics-program
- Goal: Represent threads/processes, interference, atomic regions, rely/guarantee contracts, channels, session protocols, linearizability points, and forward/backward simulation.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/concurrency.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/refinement.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_concurrency_refinement.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/concurrency.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/refinement.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_concurrency_refinement.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_concurrency_refinement.py -q
- Acceptance: Environment and component steps are distinct; interference and fairness assumptions are explicit; session duality and simulation relations validate; bounded schedules never claim unbounded refinement.
- Conflict policy: Own concurrency/refinement modules and tests; do not edit state/program contracts, TLA emitters, or kernels.
- Interfaces: ConcurrencyIR@1, RefinementIR@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G028 Add authorization, Datalog, and SecPAL-style semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021
- Fib priority: 13
- Priority: P1
- Track: semantics
- Bundle: logic-formal-verification/authorization
- Goal: Define finite facts, rules, principals, roles, speaks-for, delegation, constraints, deny/allow precedence, explanations, and policy-decision queries.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/authorization.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_authorization.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/authorization.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_authorization.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_authorization.py -q
- Acceptance: The IR is finite and stratification-aware; allow, deny, conflict, and unknown are distinct; delegation depth and trust roots are bounded; authorization decisions cannot masquerade as theorem proof.
- Conflict policy: Own authorization semantics and tests; do not execute an engine or change existing UCAN/legal policy modules.
- Interfaces: AuthorizationIR@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G029 Add symbolic cryptographic-protocol semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021
- Fib priority: 13
- Priority: P1
- Track: semantics
- Bundle: logic-formal-verification/protocols
- Goal: Define roles, fresh names, keys, messages, channels, adversary knowledge, rewrite facts, events, secrecy, authentication, correspondence, and equivalence claims.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/protocol.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_protocol.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/protocol.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_protocol.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_protocol.py -q
- Acceptance: Protocol models are source bound and typed; trust/adversary/channel assumptions are explicit; claims distinguish secrecy, reachability, correspondence, and equivalence; unsupported equational theories fail closed.
- Conflict policy: Own protocol semantics and tests; do not emit Tamarin/ProVerif syntax or edit domain crypto-exchange models.
- Interfaces: ProtocolIR@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G030 Add hyperproperty and information-flow semantics

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021, LFV-G023
- Fib priority: 21
- Priority: P1
- Track: semantics
- Bundle: logic-formal-verification/hyperproperties
- Goal: Represent trace variables, quantifier alternation, observations, low/high labels, declassification, relational pre/postconditions, noninterference, and witness trace bundles.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/hyperproperties.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_hyperproperties.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/hyperproperties.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_hyperproperties.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/software_verification/test_hyperproperties.py -q
- Acceptance: Trace cardinality and quantifier order are canonical; observations and declassification are explicit; self-composition declares finite bounds; a bounded witness or clean sample cannot become universal proof.
- Conflict policy: Own hyperproperty semantics and tests; do not execute HyperLTL tools or edit the supervisor verifier.
- Interfaces: HyperpropertyIR@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G031 Build source and domain adapters into the shared IR

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G022, LFV-G023, LFV-G025
- Fib priority: 55
- Priority: P0
- Track: integration
- Bundle: logic-formal-verification/semantics-program
- Goal: Convert supported Python and JavaScript/TypeScript source, existing Intent dynamic-Hoare/workflow/safety/VC views, and Security transition-system/VC views into the shared software-verification IR with explicit compatibility and unsupported-feature receipts.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/source_adapters.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/domain_adapters.py, ipfs_datasets_py/tests/integration/logic/test_software_verification_domain_lowering.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/source_adapters.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/domain_adapters.py, ipfs_datasets_py/tests/integration/logic/test_software_verification_domain_lowering.py, ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, ipfs_accelerate_py/agent_supervisor/code_security_facts.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_software_verification_domain_lowering.py -q
- Acceptance: Supported Python and JavaScript/TypeScript subsets plus Intent and Security fixtures generate source-bound shared artifacts and obligations; language/runtime/memory/undefined-behavior assumptions are explicit; existing domain identities remain stable; unsupported syntax and semantics are retained; fake-backend-only success is replaced by canonical backend requests.
- Conflict policy: Own the new source/domain adapters and integration test; reuse supervisor AST/security-fact extractors through narrow compatibility edits and do not edit provider implementations or exports.
- Interfaces: SourceSoftwareVerificationAdapter@1, IntentSoftwareVerificationAdapter@1, SecuritySoftwareVerificationAdapter@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-translate

## LFV-G040 Implement the shared semantic SMT compiler

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G021, LFV-G025, LFV-G026, LFV-G027, LFV-G031
- Fib priority: 89
- Priority: P0
- Track: smt-compiler
- Bundle: logic-formal-verification/smt
- Goal: Lower supported shared verification conditions and Horn/CHC reachability obligations into deterministic SMT-LIB with typed sorts, theories, declarations, assumptions, goals, model/unsat-core requests, and translation receipts.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/compiler.py, ipfs_datasets_py/tests/unit/logic/backends/smt/test_semantic_compiler.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/compiler.py, ipfs_datasets_py/tests/unit/logic/backends/smt/test_semantic_compiler.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/backends/smt/test_semantic_compiler.py -q
- Acceptance: Supported arithmetic, equality, arrays/maps, datatypes, quantifiers, Horn/CHC reachability, state transitions, VCs, heap/resource fragments, interference, and refinement obligations have golden SMT-LIB; theorem-by-negation, SAT, and fixed-point queries are explicit; PDR/IC3 claims are capability bound; unsupported temporal, heap, concurrency, or refinement features cannot become uninterpreted native claims.
- Conflict policy: Own the new shared SMT compiler and tests; do not edit Z3/CVC5 adapters or backend registry.
- Interfaces: SoftwareVerificationSMTCompiler@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-translate

## LFV-G041 Run shared verification conditions through Z3 and CVC5

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G014, LFV-G040
- Fib priority: 144
- Priority: P0
- Track: smt-execution
- Bundle: logic-formal-verification/smt
- Goal: Adapt the existing Z3 and CVC5 backends to the semantic compiler, typed results, models, unsat cores, exact receipts, and differential verification.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/z3/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/cvc5/compiler.py, ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py -q
- Acceptance: Both adapters run identical canonical VCs when available, expose explicit unavailability otherwise, agree on reviewed fixtures, preserve disagreement evidence, reject malformed outputs, and bind versions/resources/translations.
- Conflict policy: Own the differential module/test and the Z3/CVC5 adapter integration edits; do not edit other providers, public API, or routing.
- Interfaces: Z3SoftwareVerificationBackend@1, CVC5SoftwareVerificationBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G042 Normalize Vampire, E, DCEC, TDFOL, and legacy prover adapters

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G014, LFV-G021
- Fib priority: 89
- Priority: P1
- Track: prover
- Bundle: logic-formal-verification/atp
- Goal: Wrap native and legacy ATP/legal prover stacks behind canonical requests, capabilities, candidates, proof objects, countermodels, and compatibility receipts.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py -q
- Acceptance: Vampire/E/TPTP and native DCEC/TDFOL results are typed, bounded, and source bound; heuristic/duck-typed success is removed; unreconstructed ATP output remains candidate; reviewed legacy behavior remains compatible.
- Conflict policy: Own the new ATP adapter package/test and minimal compatibility shims; do not refactor native engines, Hammer, public exports, or router policy.
- Interfaces: ATPCompatibilityBackends@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G043 Build deterministic property-specific prover portfolios

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G041, LFV-G042
- Fib priority: 233
- Priority: P1
- Track: routing
- Bundle: logic-formal-verification/smt
- Goal: Plan staged solver, ATP, model-checker, monitor, policy, protocol, hyperproperty, and kernel attempts by property and required assurance.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/portfolio.py, ipfs_datasets_py/tests/unit/logic/backends/test_portfolio.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/portfolio.py, ipfs_datasets_py/tests/unit/logic/backends/test_portfolio.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_portfolio.py -q
- Acceptance: Routing is deterministic and side-effect free; capability gaps are explicit; order cannot change final authority; disagreement quarantines; candidates route to reconstruction; resource and assurance policy bound every plan.
- Conflict policy: Own portfolio planning and tests; do not register providers, launch tools, or edit supervisor routing until LFV-G072.
- Interfaces: VerificationPortfolio@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G044 Generalize TLA+, TLC, and Apalache state-model checking

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G014, LFV-G022, LFV-G023, LFV-G027
- Fib priority: 89
- Priority: P1
- Track: prover
- Bundle: logic-formal-verification/state-model-checking
- Goal: Extract the supervisor state-model implementation into reusable TLA translation plus distinct bounded TLC and Apalache backends.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py -q
- Acceptance: Generated modules/configs are deterministic and source mapped; state, concurrency, rely/guarantee, and refinement projections disclose losses; TLC and Apalache capabilities/bounds differ explicitly; counterexamples parse and replay; liveness/fairness limitations are disclosed; absent JVM/tools return unavailable.
- Conflict policy: Own the new TLA backend package/test; port generic behavior without deleting or breaking the supervisor-local facade.
- Interfaces: TLABackend@1, TLCBackend@1, ApalacheBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G045 Build generic runtime MTL with portable TypeScript parity

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G023
- Fib priority: 89
- Priority: P1
- Track: monitor
- Bundle: logic-formal-verification/runtime-monitoring
- Goal: Extract a generic Python finite-trace MTL/LTLf monitor, define portable formula/trace/result schemas, and provide a TypeScript reference implementation over the same golden fixtures.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/package-lock.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py -q && npm --prefix ipfs_datasets_py/typescript/logic-runtime-mtl test
- Acceptance: Python and TypeScript agree on interval boundaries, clocks, missing/late events, violations, inconclusive prefixes, and serialization; results always have monitor authority; no-violation-observed never becomes proof.
- Conflict policy: Own the new monitor packages and parity test; leave crypto-exchange/supervisor monitors as compatibility consumers.
- Interfaces: RuntimeMTLMonitor@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G046 Implement Datalog and SecPAL-style authorization backends

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G014, LFV-G028
- Fib priority: 89
- Priority: P1
- Track: prover
- Bundle: logic-formal-verification/authorization
- Goal: Generalize the deterministic supervisor authorization evaluator and add conformant Datalog/SecPAL-style adapters with bounded explanations.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/datalog/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_authorization_backends.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/datalog/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_authorization_backends.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_authorization_backends.py -q
- Acceptance: The reference evaluator and available external engine agree on allow/deny/conflict/unknown fixtures; recursion/delegation/resources are bounded; explanations bind rules; engine output cannot grant theorem authority.
- Conflict policy: Own the new Datalog backend/test; keep existing UCAN and supervisor authorization behavior through thin adapters and defer registry edits.
- Interfaces: DatalogAuthorizationBackend@1, SecPALAuthorizationBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G047 Implement Tamarin and ProVerif protocol backends

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G014, LFV-G029
- Fib priority: 89
- Priority: P1
- Track: prover
- Bundle: logic-formal-verification/protocols
- Goal: Generalize reviewed supervisor/domain protocol models into deterministic Tamarin and ProVerif compilers, runners, result parsers, and attack-trace receipts.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/tamarin.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol/proverif.py, ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py -q
- Acceptance: Compilers disclose the Dolev-Yao/symbolic-model ceiling, equational theory, and claim support; tool versions and Maude/opam dependencies bind receipts; attack traces normalize and replay; disagreement and inconclusive results quarantine; missing tools are explicit.
- Conflict policy: Own the new protocol backend modules/test; port generic logic without modifying reviewed domain fixtures, installers, public API, or supervisor routing.
- Interfaces: TamarinBackend@1, ProVerifBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G048 Implement HyperLTL, AutoHyper, and MCHyper backends

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G014, LFV-G030
- Fib priority: 89
- Priority: P1
- Track: prover
- Bundle: logic-formal-verification/hyperproperties
- Goal: Add real external execution paths and typed witness bundles for HyperLTL-family tools while retaining bounded self-composition as a non-authoritative fallback.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/hyperproperties/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_hyperproperty_backends.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/hyperproperties/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_hyperproperty_backends.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_hyperproperty_backends.py -q
- Acceptance: Each engine has separate discovery/capabilities; quantifier order and observation maps survive translation; counterexample trace tuples replay; fallback bounds are explicit; absent tools and unsupported alternation return non-success.
- Conflict policy: Own the new hyperproperty adapter/test; preserve the supervisor fallback and do not represent it as external-tool proof.
- Interfaces: HyperLTLBackend@1, AutoHyperBackend@1, MCHyperBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G049 Normalize Lean and Rocq/Coq kernel-checking backends

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G014, LFV-G021, LFV-G025
- Fib priority: 89
- Priority: P1
- Track: kernel-backends
- Bundle: logic-formal-verification/kernels
- Goal: Expose Lean and Rocq/Coq theorem generation, native and WASM-compatible capability probing, compilation, proof checking, diagnostics, and receipts through canonical kernel backends.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/lean.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/rocq.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/wasm.py, ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/lean.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/rocq.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/wasm.py, ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py -q
- Acceptance: Kernel receipts bind exact theorem, imports, generated proof, toolchain, source tree, and translation; Lean `sorry`/unsafe axioms and Rocq/Coq `Admitted` are rejected or explicitly downgrade authority; native and WASM/browser-compatible capability states are separate and real WASM absence is explicit; failure diagnostics are inert and bounded; unavailable kernels never pass.
- Conflict policy: Own the new kernel adapters/test; reuse existing bridges and Hammer reconstructors without editing public exports or advisor code.
- Interfaces: LeanKernelBackend@1, RocqKernelBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-kernel

## LFV-G050 Integrate Isabelle and Hammer reconstruction as canonical backends

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G042, LFV-G049
- Fib priority: 144
- Priority: P1
- Track: hammer-reconstruction
- Bundle: logic-formal-verification/kernels
- Goal: Make the mature Hammer portfolio and Isabelle/Lean/Rocq reconstructors registry-driven, source bound, and available as candidate-search plus independent reconstruction operations.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/hammers/backend.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/isabelle.py, ipfs_datasets_py/tests/integration/logic/hammers/test_canonical_backend.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/hammers/backend.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/kernel/isabelle.py, ipfs_datasets_py/tests/integration/logic/hammers/test_canonical_backend.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/hammers/test_canonical_backend.py -q
- Acceptance: Premise selection, SMT/ATP search, proof candidates, reconstruction, and kernel receipts are separate stages; provider sets are registry driven; unreconstructed success is candidate only; Isabelle `sorry` and unreviewed axiomatization reject or downgrade authority; Isabelle path metadata is corrected.
- Conflict policy: Own the new Hammer/kernel adapters and test; make focused edits inside Hammer but defer public API and global registry changes.
- Interfaces: HammerBackend@1, IsabelleKernelBackend@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-kernel

## LFV-G060 Generalize the autoencoder into a bounded formalization advisor

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G020, LFV-G021
- Fib priority: 89
- Priority: P1
- Track: autoencoder-advisor
- Bundle: logic-formal-verification/advisors
- Goal: Adapt modal-autoencoder introspection, ranking, compression, and repair guidance from legal-only samples to domain-neutral formalization samples and software-verification families.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/autoencoder_advisor.py, ipfs_datasets_py/tests/unit/logic/formalization/test_autoencoder_advisor.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/autoencoder_advisor.py, ipfs_datasets_py/tests/unit/logic/formalization/test_autoencoder_advisor.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/formalization/test_autoencoder_advisor.py -q
- Acceptance: Advisors rank premises/views and propose bounded repairs without changing sources, assumptions, modalities, or trust; checkpoints bind schemas/code/data; duplicate/source-family-safe splits pass; outputs are candidate only.
- Conflict policy: Own the new adapter/test; reuse the existing modal optimizer without broad edits to its legal training pipeline.
- Interfaces: FormalizationAutoencoderAdvisor@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G061 Normalize Leanstral and SymAI as untrusted proposal providers

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G049, LFV-G060
- Fib priority: 144
- Priority: P1
- Track: proposal-advisors
- Bundle: logic-formal-verification/advisors
- Goal: Provide strict advisor adapters for specification, lemma, tactic, premise, and repair proposals and remove legacy neural routes that infer proof from `is_valid` or confidence.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/prover_router.py, ipfs_datasets_py/ipfs_datasets_py/logic/integration/symbolic/neurosymbolic/reasoning_coordinator.py, ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py -q
- Acceptance: Inputs/outputs are bounded and sanitized; prompts/responses are inert and source bound; generic `is_valid`, similarity, or confidence never yields proof; accepted candidates require deterministic compilation and independent solver/kernel validation.
- Conflict policy: Own the new advisor/test and narrowly repair the two identified neural proof-authority defects; do not refactor model runtimes or kernel backends.
- Interfaces: LeanstralAdvisor@1, SymAIAdvisor@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G062 Unify exact caches with the immutable proof corpus

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G013, LFV-G015, LFV-G021
- Fib priority: 89
- Priority: P1
- Track: cache-corpus
- Bundle: logic-formal-verification/evidence
- Goal: Define a shared cache-key/protocol adapter and bridge validated attempt/proof/counterexample receipts into the proof corpus without forcing one storage implementation.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/cache_protocol.py, ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus/backend_store.py, ipfs_datasets_py/tests/integration/logic/test_backend_cache_corpus.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/cache_protocol.py, ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus/backend_store.py, ipfs_datasets_py/tests/integration/logic/test_backend_cache_corpus.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_backend_cache_corpus.py -q
- Acceptance: Keys bind IR/property/assumptions, translation, backend/binary/version/config, resources, tree, and policy; single-flight and negative TTL behavior are deterministic; stale/tampered entries reject; cache never raises authority.
- Conflict policy: Own the new protocol/store adapters and test; adapt Hammer/supervisor caches through wrappers and do not rewrite all legacy cache implementations.
- Interfaces: VerificationCacheProtocol@1, BackendProofCorpusStore@1
- Submodules: ipfs_datasets_py
- Resource class: io-artifact

## LFV-G063 Bind production ZKP attestations to trusted proof receipts

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G049, LFV-G062
- Fib priority: 144
- Priority: P1
- Track: zkp-attestation
- Bundle: logic-formal-verification/evidence
- Goal: Normalize Groth16/ProveKit attestation and verification over current trusted receipts while fencing simulated backends and private witnesses.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/proof_receipt_attestation.py, ipfs_datasets_py/tests/integration/logic/test_proof_receipt_attestation.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/proof_receipt_attestation.py, ipfs_datasets_py/tests/integration/logic/test_proof_receipt_attestation.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_proof_receipt_attestation.py -q
- Acceptance: Public inputs bind theorem/property, translation, receipt, tree, policy, circuit, CRS/setup ceremony, proving-key and verification-key identities, backend, revocation policy, and freshness; private witnesses never serialize; simulated/circuit-mismatched/stale/revoked attestations fail; attestation is orthogonal to and preserves underlying semantic authority.
- Conflict policy: Own the new receipt-attestation adapter/test; reuse ZKP backends and supervisor policy without changing circuit code or representing ZKP as a theorem prover.
- Interfaces: ProofReceiptAttestation@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-proof-solver

## LFV-G070 Expose the stable Python software-verification API

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G011, LFV-G026, LFV-G027, LFV-G043, LFV-G044, LFV-G045, LFV-G046, LFV-G047, LFV-G048, LFV-G050, LFV-G061, LFV-G062, LFV-G063
- Fib priority: 377
- Priority: P1
- Track: python-api
- Bundle: logic-formal-verification/api
- Goal: Add lightweight generic family/provider discovery, compilation, checking, monitoring, portfolio, counterexample, receipt, advisor, and attestation operations while preserving legacy imports.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/api.py, ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py, ipfs_datasets_py/ipfs_datasets_py/logic/submodule_registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api.py
- Validation: python -m pytest ipfs_datasets_py/tests/unit/logic/test_verification_api.py ipfs_datasets_py/tests/unit/logic/test_logic_api_v1_compatibility.py -q
- Acceptance: API imports quietly without optional tools; discovery is declarative; responses expose typed status/authority/assumptions/bounds/translations/witnesses/cache provenance; absent features are explicit; legacy API behavior stays green.
- Conflict policy: Single owner for `logic.api`, package exports, submodule registry, and the new verification facade; do not edit CLI/MCP or supervisor routers.
- Interfaces: LogicVerificationAPI@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G071 Expose equivalent CLI and MCP verification operations

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G070
- Fib priority: 610
- Priority: P1
- Track: cli-mcp
- Bundle: logic-formal-verification/api
- Goal: Add bounded machine-readable CLI and MCP operations for the stable verification API with capability inspection and receipt retrieval.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/cli.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_tools/__init__.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/__init__.py, ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py -q
- Acceptance: CLI/MCP cover list, capability, compile, check, monitor, portfolio, counterexample, receipt, advisor, and attestation operations; schemas match Python; inputs/outputs are bounded; errors and unavailable tools are stable and secret safe.
- Conflict policy: Single owner for CLI/MCP registration and tests; reuse the Python facade and preserve existing command/tool names and behavior.
- Interfaces: LogicVerificationCLI@1, LogicVerificationMCP@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G072 Adapt the agent supervisor to the canonical logic platform

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G015, LFV-G043, LFV-G044, LFV-G045, LFV-G046, LFV-G047, LFV-G048, LFV-G050, LFV-G062, LFV-G070
- Fib priority: 610
- Priority: P1
- Track: integration
- Bundle: logic-formal-verification/supervisor
- Goal: Replace overlapping supervisor family/capability/provider vocabularies with thin canonical adapters while retaining scheduling, resource, isolation, routing, cache, and evidence behavior.
- Evidence: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
- Outputs: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py, ipfs_accelerate_py/agent_supervisor/proof/logic_translation_validation.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_canonical_logic.py
- Validation: python -m pytest test/api/test_agent_supervisor_canonical_logic.py -q
- Acceptance: Analysis families, property kinds, translation forms, matrix entries, capability probes, providers, routes, resources, caches, and receipts map losslessly; supervisor-local facades remain compatible; datasets imports are lazy; cross-repo current-revision checks pass.
- Conflict policy: Single owner for supervisor registry/router compatibility edits; do not move orchestration/resource code into datasets or duplicate semantic contracts.
- Interfaces: SupervisorCanonicalLogicAdapter@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-medium

## LFV-G080 Build the cross-family and cross-provider conformance corpus

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G041, LFV-G043, LFV-G044, LFV-G045, LFV-G046, LFV-G047, LFV-G048, LFV-G050, LFV-G061, LFV-G063, LFV-G071, LFV-G072
- Fib priority: 987
- Priority: P0
- Track: conformance
- Bundle: logic-formal-verification/quality
- Goal: Provide reusable positive, negative, mutation, metamorphic, translation, disagreement, malformed-output, timeout, and authority-boundary fixtures across every family/provider.
- Evidence: ipfs_datasets_py/tests/fixtures/logic/software_verification/conformance/manifest.json, ipfs_datasets_py/tests/integration/logic/test_software_verification_conformance.py
- Outputs: ipfs_datasets_py/tests/fixtures/logic/software_verification/conformance/manifest.json, ipfs_datasets_py/tests/integration/logic/test_software_verification_conformance.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_software_verification_conformance.py -q
- Acceptance: Fake runners cover every adapter offline; opt-in real-tool lanes declare skips/unavailability; semantic mutations are detected; counterexamples replay; disagreements quarantine; no authority upgrade or stale-cache acceptance occurs.
- Conflict policy: Own conformance fixtures/runner only; consume provider APIs without editing implementations, registries, or public surfaces.
- Interfaces: LogicProviderConformance@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## LFV-G081 Deliver source-bound end-to-end software-verification exemplars

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G080
- Fib priority: 1597
- Priority: P1
- Track: examples
- Bundle: logic-formal-verification/quality
- Goal: Demonstrate contracts/resources, heap ownership, concurrent workflows, authorization, cryptographic protocols, noninterference, and runtime temporal monitoring through the public API.
- Evidence: ipfs_datasets_py/examples/logic/software_verification/manifest.json, ipfs_datasets_py/tests/integration/logic/test_software_verification_examples.py
- Outputs: ipfs_datasets_py/examples/logic/software_verification/manifest.json, ipfs_datasets_py/tests/integration/logic/test_software_verification_examples.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_software_verification_examples.py -q
- Acceptance: Seven deterministic examples bind sources to IR, translations, requests, results, witnesses, receipts, and declared assurance; unavailable optional tools degrade explicitly; at least one negative/counterexample case exists per lane.
- Conflict policy: Own the example tree and integration test; do not change provider behavior to fit fixtures or commit transient live outputs.
- Interfaces: SoftwareVerificationExamples@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## LFV-G082 Harden installers, resources, isolation, and adversarial behavior

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G014, LFV-G044, LFV-G045, LFV-G046, LFV-G047, LFV-G048, LFV-G049
- Fib priority: 144
- Priority: P0
- Track: security
- Bundle: logic-formal-verification/quality
- Goal: Complete explicit pinned tool discovery/installation metadata, resource classes, process isolation, secret/witness handling, and adversarial execution tests for every provider.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchains.py, ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchains.py, ipfs_datasets_py/ipfs_datasets_py/logic/integration/bridges/prover_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py -q
- Acceptance: TLC, Hyper tools, Datalog/SecPAL, and runtime-MTL gaps are declared; installs require explicit calls and pins/checksums; JVM/opam/Maude/circuit dependencies are bound; malicious paths/output/process trees and secret leakage are contained.
- Conflict policy: Own toolchain metadata/security tests and focused installer additions; never install during tests/imports and do not mutate system package managers.
- Interfaces: VerificationToolchainRegistry@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation

## LFV-G083 Benchmark, document, roll out, and issue the completion receipt

- Status: active
- Parent: LFV-G000
- Depends on: LFV-G011, LFV-G026, LFV-G027, LFV-G071, LFV-G072, LFV-G080, LFV-G081, LFV-G082
- Fib priority: 2584
- Priority: P0
- Track: release
- Bundle: logic-formal-verification/quality
- Goal: Reconcile documentation with executable capabilities, benchmark semantic quality/resources/cache behavior, define per-property shadow/canary/enforcement gates, and emit the final current-tree completion receipt.
- Evidence: ipfs_datasets_py/docs/logic/software_verification_rollout.md, ipfs_datasets_py/tests/integration/benchmarks/logic_pipeline/test_software_verification_matrix.py, test/api/test_logic_formal_verification_completion.py
- Outputs: ipfs_datasets_py/docs/logic/software_verification_rollout.md, ipfs_datasets_py/docs/security_verification/prover_matrix.md, ipfs_datasets_py/tests/integration/benchmarks/logic_pipeline/test_software_verification_matrix.py, test/api/test_logic_formal_verification_completion.py, docs/architecture/logic_formal_verification_expansion_completion_receipt.json
- Validation: python -m pytest ipfs_datasets_py/tests/integration/benchmarks/logic_pipeline/test_software_verification_matrix.py test/api/test_logic_formal_verification_completion.py -q
- Acceptance: The matrix is generated from current executable evidence; benchmarks report semantic and resource distributions without timing-ratio correctness gates; rollout is property specific and reversible; receipt binds all 41 child goals with zero authority-boundary violations.
- Conflict policy: Single owner for final docs, stale matrix reconciliation, benchmark, rollout, root test, and completion receipt; do not weaken provider tests or fabricate unavailable external-tool evidence.
- Interfaces: LogicFormalVerificationRelease@1
- Submodules: ipfs_datasets_py
- Resource class: cpu-validation
