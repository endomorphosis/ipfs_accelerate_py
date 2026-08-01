# Logic Formal Verification Expansion Plan

## Status

- Program: `logic_formal_verification_expansion`
- Supervisor namespace: `LFV`
- Implementation branch: `agent/software-verification-prover-matrix`
- Primary implementation repository: `ipfs_datasets_py`
- Supervisor integration repository: `ipfs_accelerate_py`
- Objective heap:
  `docs/architecture/logic_formal_verification_expansion.objectives.md`
- Executable board:
  `docs/architecture/logic_formal_verification_expansion.todo.md`
- Supervisor state:
  `data/agent_supervisor/logic_formal_verification_expansion/`

This plan expands the logic package from a predominantly legal-logic surface
into a general software-verification platform. It does not discard the legal,
deontic, modal, DCEC, TDFOL, or frame-logic work. It puts those families beside
program, state, trace, temporal, authorization, protocol, heap, concurrency,
and hyperproperty logics behind one typed API and one evidence model.

## Outcome

The completed program provides:

1. a versioned registry of logic families, properties, translations, prover
   capabilities, and assurance ceilings;
2. immutable program, state, transition, trace, temporal, authorization,
   protocol, heap, concurrency, refinement, and hyperproperty IRs;
3. bounded adapters for Z3, CVC5, TLC, Apalache, Datalog/SecPAL-style engines,
   Tamarin, ProVerif, HyperLTL-family tools, Lean, Rocq/Coq, Isabelle,
   Vampire, E, DCEC, TDFOL, and runtime MTL;
4. a stable Python API, CLI, and MCP surface that exposes capabilities,
   compilation, checking, monitoring, portfolios, counterexamples, receipts,
   and explicit unavailability;
5. shared integrations for Hammer, the modal autoencoder, Leanstral, SymAI,
   proof caches, the proof corpus, IPFS/content identities, and ZKP
   attestation without allowing an advisor, cache, monitor, or attestation to
   manufacture proof authority; and
6. cross-repository supervisor routing that consumes the same contracts
   instead of maintaining a second logic-family vocabulary.

## Why this is an integration program, not a greenfield rewrite

The repositories already contain substantial pieces of the target:

- `ipfs_datasets_py.logic.ir_core` defines immutable claims, proof
  obligations, backend requests, typed results, and non-interchangeable result
  authorities.
- `ipfs_datasets_py.logic.backends` has bounded Z3 and CVC5 SMT-LIB adapters.
- Intent and Security formalizers already name program-verification views such
  as verification conditions, dynamic Hoare logic, workflows, safety,
  liveness, and transition systems.
- `ipfs_datasets_py.logic.hammers` has premise selection, SMT-LIB and TPTP
  translation, portfolios, provenance, reconstruction, Lean/Rocq/Isabelle
  frontends, receipts, and caching.
- the modal autoencoder and Leanstral pipeline already provide bounded
  synthesis, diagnostics, repair proposals, and validation hooks;
- SymbolicAI/SymAI integrations already provide neural-symbolic proposal
  paths;
- the ZKP package contains simulated and production-capable backend surfaces;
- the proof corpus supplies content-addressed evidence storage; and
- `ipfs_accelerate_py.agent_supervisor` already contains richer, but
  supervisor-local, state-model, runtime-MTL, authorization, protocol,
  hyperproperty, kernel, translation, conformance, cache, prover-matrix, and
  routing components.

The work therefore standardizes on `ipfs_datasets_py.logic.ir_core` plus
`ipfs_datasets_py.logic.backends`, extracts reusable domain-neutral behavior
from the supervisor, and leaves compatibility facades at old import paths.
Legacy CEC, TDFOL, external-prover, and domain-specific runners become
adapters to this contract rather than new sources of backend semantics.

## Current gaps

### Public surface

`logic.api`, `logic.cli`, the MCP logic tools, and the lazy submodule registry
remain weighted toward FOL, deontic, modal, TDFOL, CEC, FLogic, and legal
conversion. The modern IR kernel, backend registry, Hammer, proof corpus,
Intent/Security formalization, and the supervisor's software-verification
families are not available through one coherent API.

### Semantic lowering

Z3 and CVC5 can execute deterministic SMT-LIB requests, but the existing
Intent and Security verification-condition views are not generally lowered
through those production adapters. Several legacy converters encode modal,
deontic, or temporal operators as uninterpreted predicates. That encoding can
be useful only when a translation receipt declares its limitations; it is not
native temporal or modal verification.

### Provider fragmentation

There are overlapping CEC, TDFOL, external-prover, Security-model, Hammer, and
supervisor provider abstractions. Runtime-monitor, policy-decision,
model-checker, protocol-attack, and hyperproperty results need typed
normalizers rather than being squeezed into theorem/satisfiability outcomes.

### Capability truth

The checked-in prover matrix says only Z3 is implemented, while current code
contains additional compilers, bridges, installers, probes, and
supervisor-local lanes. A declared provider, an installed binary, a passing
probe, a conformant adapter, and an enforcement-authorized result are
different states and must be reported separately.

### Trust and cache consistency

Some legacy neural routes can turn an `is_valid` or similarity threshold into
a proof-like result. Proof caches are fragmented and do not all bind the same
semantic inputs. Simulated ZKP behavior must not be confused with a
production proof or with proof of the translated program property.

## Architecture

```text
Python/source/Intent IR/Security IR/runtime events
                         |
                         v
             Software Verification IR
   program + state + trace + heap + protocol + hyperproperty
                         |
              typed, loss-aware translations
                         |
                         v
             Property and capability router
       /----------+----------+----------+----------\
      SMT       model      policy     protocol    kernel
   Z3/CVC5   TLC/Apalache  Datalog   Tamarin/    Lean/Rocq/
                                      ProVerif    Isabelle
       \----------+----------+----------+----------/
                         |
       typed result + witness + translation receipt
                         |
            reconstruction / cross-check policy
                         |
            exact-bound cache and proof corpus
                         |
          optional ZKP attestation of the receipt
                         |
               logic API / CLI / MCP
                         |
              agent supervisor adapters
```

### Canonical request flow

1. A frontend emits a canonical source-bound verification artifact.
2. The property is classified without selecting a provider.
3. A translation declares its source and target family, preservation claim,
   unsupported features, assumptions, bounds, and assurance ceiling.
4. Routing selects only providers whose declared and probed capabilities
   satisfy the request and resource policy.
5. Each attempt returns a typed result. `unknown`, timeout, unavailable,
   malformed output, or unsupported translation fail closed.
6. Proof candidates are independently reconstructed or checked when the
   assurance policy requires it.
7. Exact request, translation, provider, binary, version, tree, and resource
   identities bind the receipt and cache entry.
8. A ZKP backend may attest that receipt; it cannot strengthen the theorem or
   translation semantics.

## Logic-family expansion

| Family | Core artifacts | Software-verification role | Primary providers |
| --- | --- | --- | --- |
| Propositional/FOL/SMT | terms, sorts, quantified claims, theories | functional correctness, constraints, arithmetic, resources | Z3, CVC5, Vampire, E |
| Horn clauses/CHC/fixpoint | relations, constrained rules, reachability queries, invariants | recursive programs, safety invariants, abstract interpretation | Z3 Spacer/PDR and capability-declared CHC engines |
| State/transition/action | states, variables, initial relation, next relation, invariants | workflows, distributed systems, lifecycle safety | TLC, Apalache, SMT |
| Trace/event temporal | events, clocks, finite/infinite traces, LTL/MTL formulas | runtime monitoring, ordering, deadlines, liveness | runtime MTL, TLC, Apalache |
| Branching temporal | Kripke structures, CTL/CTL* path quantifiers | universal/existential state exploration | model checkers through declared translations |
| Hoare/dynamic/WP | commands, preconditions, postconditions, invariants, variants | code contracts and verification-condition generation | Z3, CVC5, Lean, Rocq |
| Separation/heap/resource | heaps, ownership, disjointness, permissions, resource algebras | memory safety, aliasing, linear resources | SMT fragments, Lean, Rocq |
| Concurrency/refinement | rely/guarantee, interference, simulation, session/process protocols | races, atomicity, implementation refinement | SMT, TLA+, kernels |
| Authorization | facts, rules, delegation, speaks-for, deny/allow/unknown | access control and capability delegation | Datalog/SecPAL-style |
| Protocol | roles, messages, facts, adversary, secrecy/authentication claims | cryptographic protocol verification | Tamarin, ProVerif |
| Hyperproperty | trace sets, quantifier alternation, observations, declassification | noninterference and information flow | HyperLTL, AutoHyper, MCHyper |
| Modal/deontic/event calculus | worlds, accessibility, obligations, permissions, events, fluents | compatibility with existing legal and agent-policy models | native DCEC/TDFOL, ATP, kernels |
| Refinement types/higher order | typed propositions, inductive relations, executable specifications | reusable proof libraries and kernel validation | Lean, Rocq, Isabelle |

Every family has a stable identifier, schema version, supported
property/query kinds, native versus translated status, serialization rules,
and compatibility aliases. String aliases never silently equate semantically
different families.

## Prover matrix

| Prover | Current reusable surface | Target access path | Primary fit | Required authority gate |
| --- | --- | --- | --- | --- |
| Z3/CVC5 | deterministic SMT-LIB backend compilers; legacy/domain runners | canonical Python backend | VCs, SMT checks, resource invariants | semantic lowering tests plus differential and mutation checks |
| TLC/Apalache | supervisor-local state model and bounded receipts; pinned Apalache installer | bounded external JVM backend | workflows, state machines, safety/liveness | source-bound TLA translation, bound disclosure, parsed counterexamples |
| Datalog/SecPAL style | supervisor-local reference and shadow adapters | in-process reference plus optional external engine | authorization and delegation | policy-result authority, deny/unknown behavior, differential corpus |
| Tamarin/ProVerif | installers, probes, supervisor-local adapters/models | bounded external processes | cryptographic protocols | model/claim binding, attack-trace parsing, reviewed golden models |
| HyperLTL/AutoHyper/MCHyper | supervisor-local adapters and bounded self-composition | bounded external processes | noninterference and hyperproperties | trace-quantifier preservation, witness bundle, no bounded-to-universal promotion |
| Lean/Rocq/Coq | interactive bridges and Hammer reconstruction | proof backend and kernel-check service | proof reconstruction and theorem checking | independent kernel acceptance bound to source and toolchain |
| Isabelle | Hammer frontend/reconstructor and supervisor kernel lane | external kernel backend | higher-order reconstruction | checked theory/session receipt |
| Vampire/E | CEC and integration adapters; Hammer TPTP | canonical ATP backend | FOL search and lemma discovery | proof object/reconstruction where required; candidate otherwise |
| DCEC/TDFOL/ShadowProver | mature native legal/modal stacks | compatibility backend | existing event/deontic/modal reasoning | explicit family semantics and typed authority |
| Runtime MTL | domain-specific and supervisor-local monitors | Python reference, portable schema, TypeScript parity | online finite-trace monitoring | monitor authority only; a finite prefix never proves global correctness |
| Hammer | mature local portfolio and reconstructors | canonical meta-backend | premise selection, ATP/SMT search, reconstruction | kernel/solver reconstruction decides final authority |
| Autoencoder | modal optimizer and introspection | advisor/premise-selector plugin | retrieval, ranking, counterexample clustering, repair hints | candidate only; deterministic bounds and held-out evaluation |
| Leanstral/SymAI | proposal and neural prover bridges | advisor plugin | formalization/proof/repair proposals | untrusted candidate only; sanitation and independent checking |
| ZKP | simulated plus Groth16/ProveKit surfaces | receipt-attestation backend | privacy-preserving proof-receipt disclosure | production verifier, public-input binding, current trusted receipt |

## Trust and assurance model

The result type, not the provider name, determines what a result means.

| Level | Meaning | Examples |
| --- | --- | --- |
| `candidate` | untrusted proposal or search output | autoencoder, Leanstral, SymAI, unreconstructed ATP proof |
| `syntax_checked` | parsed and well typed, but not semantically established | generated Lean/SMT/TLA source |
| `bounded_checked` | property held or failed under declared finite bounds | Apalache, bounded self-composition, bounded model checking |
| `solver_checked` | solver established a claim under the recorded theory and translation | Z3/CVC5 result with validated lowering |
| `kernel_verified` | a trusted proof kernel accepted the exact theorem and proof | Lean, Rocq, Isabelle reconstruction |

Attestation is an orthogonal status, not a higher semantic level. A receipt can
be unattested or production-attested while retaining its existing
`candidate`, `syntax_checked`, `bounded_checked`, `solver_checked`, or
`kernel_verified` semantic authority. Groth16/ProveKit therefore authenticate
the bound receipt and disclosure statement without moving it upward in this
table.

Rules:

- satisfiable is not theorem-proved; unsatisfiable proves only the encoded
  obligation under its assumptions and translation;
- a monitor result is not a theorem result;
- evidence readiness and policy approval are not theorem results;
- bounded checking never becomes unbounded proof;
- tool absence, timeout, malformed output, unsupported syntax, and translation
  loss return explicit non-success states;
- a model, embedding, autoencoder, or neural-symbolic result never grants proof
  authority;
- simulated ZKP verifies serialization/test behavior only and cannot produce
  `attested`; and
- an attestation authenticates a receipt but does not improve the receipt's
  semantic assurance.

## Shared integrations

### Hammer

Hammer becomes a canonical meta-backend. It may select premises, translate to
SMT-LIB/TPTP, run a bounded portfolio, and propose reconstruction. Solver and
kernel results remain distinguishable. Lean, Rocq, and Isabelle
reconstruction receipts bind the theorem, premises, generated proof, tool
version, and source tree.

### Autoencoder

The existing modal autoencoder is generalized through a narrow advisor
protocol. It may rank premises, choose views, cluster failed obligations,
compress contexts, and propose repairs. Training and evaluation use
source-family/duplicate-safe splits. Checkpoints bind feature schema, family
registry, code, data, and configuration. Its output is never proof evidence.

### Leanstral and SymAI

Both are bounded proposal providers. Inputs are size limited and source
grounded; outputs use strict schemas and sanitation; prompts and responses are
inert evidence. Suggested formulas, tactics, lemmas, or repairs must pass the
same deterministic compiler and independent solver/kernel checks as
hand-authored candidates.

### Caches and proof corpus

The program defines three layers:

1. an ephemeral compile/normalization cache;
2. an exact proof-attempt cache with single-flight coordination; and
3. an immutable proof-corpus/artifact store.

Keys bind canonical IR/property/assumptions, translation profile, backend and
binary identity, backend version/configuration, resource policy, proof policy,
source tree, and trust-policy version. Negative, timeout, counterexample, and
unavailable results may be cached with explicit TTL and scope. A cache entry
cannot claim more authority than the independently validated receipt it
contains.

### ZKP

ZKP is applied after trusted proof or solver validation. Public inputs bind
the receipt digest, theorem/property digest, translation digest, provider
identity, source tree, policy, and freshness epoch. Private witnesses must not
leak through serialization, logs, cache keys, or failure messages.

## Stable logic API

The new API is additive and keeps legacy imports operational. Its generic
operations are:

- `list_logic_families()`
- `list_providers()` and `provider_capabilities()`
- `compile_verification_artifact()`
- `check()` for typed proof/satisfiability/model-check requests
- `monitor()` for runtime traces
- `run_portfolio()`
- `explain_counterexample()`
- `verify_receipt()` and `attest_receipt()`
- explicit `probe_provider()` and opt-in `install_provider()`

Responses expose:

- request, property, source, translation, provider, binary, configuration,
  resource, and tree identities;
- typed status and authority;
- assumptions, bounds, unsupported features, and assurance ceiling;
- proof/counterexample/attack-trace/monitor witness references;
- cache provenance and freshness; and
- diagnostics safe for CLI, MCP, and supervisor consumers.

Importing the API, listing declarations, or inspecting capabilities performs
no network access, installation, environment mutation, process launch, or
disk write. Probe and install are separate explicit operations.

## Execution waves and parallelism

### Wave 0: alignment and capability truth

Pin the parent repository to the intended `ipfs_datasets_py` revision, freeze
the legacy API, inventory actual implementations, and replace stale binary
implemented/not-implemented claims with executable maturity states.

### Wave 1: contracts and software-verification IR

Build the family registry, result normalizers, process lifecycle, shared
provider contract, translation receipts, and independent family IRs. State,
trace, program, authorization, protocol, and hyperproperty work can proceed in
parallel after the shared contracts settle.

### Wave 2: immediate high-value software verification

Promote existing Intent/Security formal views, implement weakest-precondition
and VC generation, and run real semantic lowering through both Z3 and CVC5.
This is the first end-to-end milestone because it converts existing
repository intent into executable software checks without waiting for every
external tool.

### Wave 3: provider expansion

Run four conflict-separated lanes:

- SMT and ATP;
- TLA+/state/runtime temporal;
- authorization/protocol/hyperproperties; and
- kernel/Hammer/advisors/evidence.

External tools remain optional. Fake runners and golden parsers make unit
tests deterministic; real-tool lanes run only when a probe reports the pinned
capability.

### Wave 4: shared utilities and public surface

Unify Hammer, autoencoder, Leanstral, SymAI, caches, proof corpus, and ZKP
roles; then expose the generic Python, CLI, and MCP APIs. Adapt supervisor
registries and routers to the same family/provider vocabulary.

### Wave 5: conformance and rollout

Run cross-provider, mutation, metamorphic, differential, timeout, malformed
output, cache-staleness, resource, witness-privacy, and authority-boundary
tests. Roll out through `declared`, `shadow`, `canary`, and `enforced` states
per property/provider pair. No global “provider enabled” switch bypasses
property-specific policy.

## Supervisor decomposition

The objective heap contains one root goal and 41 executable subgoals. The
objective daemon creates one canonical `LFV-*` task for each uncovered
subgoal, translates goal dependencies into task dependencies, and shards
tasks by bundle. The principal bundles are:

| Bundle | Ownership |
| --- | --- |
| `logic-formal-verification/foundation` | alignment, census, compatibility, taxonomy, provider contracts |
| `logic-formal-verification/semantics-core` | shared software-verification IR and translation receipts |
| `logic-formal-verification/semantics-state` | state and temporal family IRs |
| `logic-formal-verification/semantics-program` | program, VC, heap, concurrency, and refinement IRs |
| `logic-formal-verification/smt` | SMT compilation, Z3/CVC5, and portfolios |
| `logic-formal-verification/atp` | Vampire, E, DCEC, TDFOL, and compatibility adapters |
| `logic-formal-verification/state-model-checking` | TLA+/TLC/Apalache |
| `logic-formal-verification/runtime-monitoring` | Python and TypeScript runtime MTL |
| `logic-formal-verification/authorization` | authorization semantics and Datalog/SecPAL |
| `logic-formal-verification/protocols` | protocol semantics and Tamarin/ProVerif |
| `logic-formal-verification/hyperproperties` | hyperproperty semantics and HyperLTL-family tools |
| `logic-formal-verification/kernels` | Lean/Rocq/Isabelle, Hammer, and WASM-compatible validation |
| `logic-formal-verification/advisors` | autoencoder, Leanstral, and SymAI |
| `logic-formal-verification/evidence` | exact caches, proof corpus, ZKP receipts |
| `logic-formal-verification/api` | Python API, CLI, MCP, compatibility exports |
| `logic-formal-verification/supervisor` | cross-repository provider/router integration |
| `logic-formal-verification/quality` | conformance, examples, security, resources, benchmarks, rollout |

The live scheduler is limited to four lanes. Dependency order and predicted
file ownership allow parallel work without letting multiple lanes edit shared
registries or exports. Registry, API, supervisor, and rollout edits are
deliberately late single-owner tasks.

## Conformance and test strategy

### Contract tests

- stable canonical identities and round trips;
- no side effects during import/discovery;
- typed result normalization for every query kind;
- explicit unsupported/unavailable/timeout/malformed states;
- deterministic fake-runner tests independent of installed tools; and
- pinned real-tool probes and parsers when available.

### Semantic tests

- positive and negative golden examples per logic family;
- semantic mutation must change the expected result;
- translation round-trip, equisatisfiability, boundedness, and loss receipts;
- Z3/CVC5 differential checks over shared VCs;
- TLA/Apalache counterexample replay;
- Datalog/SecPAL differential policy decisions;
- Tamarin/ProVerif attack-trace replay;
- HyperLTL witness replay; and
- Lean/Rocq/Isabelle reconstruction.

### Adversarial tests

- prompt/model output cannot become proof authority;
- uninterpreted-function lowering cannot claim native temporal/modal support;
- finite monitors and bounded runs cannot claim universal proof;
- stale cache, changed binary, changed translation, changed tree, and changed
  policy are rejected;
- malicious external output, path traversal, oversized output, process-tree
  leakage, and timeout cleanup are contained;
- hidden assumptions and unsupported constructs fail closed; and
- simulated or public-input-mismatched ZKP never attests.

### End-to-end exemplars

The program ships at least these source-bound examples:

1. function pre/postcondition and arithmetic/resource invariant;
2. heap ownership and aliasing safety;
3. concurrent workflow/state-machine invariant and liveness bound;
4. authorization/delegation decision;
5. cryptographic protocol secrecy/authentication property;
6. noninterference across paired traces; and
7. online MTL violation detection with Python/TypeScript fixture parity.

## Metrics and promotion gates

Every property/provider pair reports:

- declared, probed, adapter, conformance, shadow, canary, and enforcement
  state;
- success, unknown, timeout, unsupported, malformed, and unavailable counts;
- translation loss and assurance ceiling;
- proof/counterexample reconstruction rate;
- cross-provider agreement and disagreement disposition;
- cold/warm exact-cache behavior without timing-ratio correctness assertions;
- CPU, memory, wall-clock, process, and output bounds;
- advisor acceptance/rejection and false-proof counts; and
- authority-boundary violations.

Promotion requires:

- all mandatory contract and adversarial tests passing;
- no unresolved semantic disagreement or authority-boundary violation;
- supported features and assurance ceiling documented by property;
- reproducible current-tree receipts;
- explicit rollback/quarantine behavior; and
- a current executable prover matrix.

## Completion criteria

The program is complete only when:

- all 41 executable subgoals have fresh current-tree receipts;
- the public API discovers every planned family/provider and reports absent
  tools explicitly;
- the seven end-to-end exemplars pass with their declared assurance;
- existing legal/DCEC/TDFOL/CEC imports and reviewed behavior remain
  compatible;
- supervisor routing uses the canonical family, capability, result, cache, and
  receipt contracts;
- no advisor, cache, monitor, bounded check, policy result, or simulated ZKP
  is represented as stronger authority;
- the prover matrix is generated from executable capability evidence; and
- a final completion receipt binds the parent commit, `ipfs_datasets_py`
  commit, tests, fixtures, toolchain identities, matrix, benchmark report,
  rollout policy, and all child receipts.
