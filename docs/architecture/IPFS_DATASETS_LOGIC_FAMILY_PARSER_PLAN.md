# IPFS Datasets Logic-Family Parser and Prover Bridge Plan

Status: proposed implementation program

Program ID: `ipfs-datasets-logic-family-parser-v1`

Task prefix: `LFP-`

Root goal: `LFP-G000`

Target repository: `ipfs_datasets_py` at the pinned `ipfs_datasets_py` gitlink

Supervisor: `ipfs_accelerate_py.agent_supervisor`

Provider route: Grok `grok-4.5` first; Codex `gpt-5.6-terra` with `high`
reasoning only after independently verified Grok hard-quota exhaustion

## Executive decision

The logic package does not primarily need another solver registry or another
text-bearing `FormalFormula` wrapper. It needs a small, typed, extensible logic
syntax and elaboration kernel between domain IRs and the existing typed solver
lowerings.

The program will:

1. separate semantic family, fragment/profile, property, view role, notation,
   target encoding, provider, and execution lane;
2. converge every current family label on one versioned registry;
3. introduce source-aware CST/AST, binding, sort, signature, diagnostics, and
   parser/printer contracts;
4. add parsers in dependency-ordered family clusters rather than pretending
   that one grammar can cover every logic;
5. require an explicit semantic-preservation and authority contract on every
   IR-to-logic and logic-to-backend edge;
6. retain the rich domain IRs and solver-specific typed models already present;
7. keep ErgoAI and SymbolicAI advisory until deterministic parsing,
   elaboration, and independent proof/model checking validate their proposals;
8. generate refill tasks from uncovered capability-matrix cells, unsupported
   AST nodes, failing conformance fixtures, and unreconstructed proof
   candidates; and
9. release through vertical slices: source -> parse -> elaborate -> translate
   -> backend -> result decode -> independent validation -> authority receipt.

## Current-state evidence

### What is already strong

- `ipfs_datasets_py/logic/ir_core/` supplies canonical JSON, content identity,
  provenance, artifacts, claims, diagnostics, and schema registration.
- `ipfs_datasets_py/logic/families/` supplies good pure-data contracts for
  fragments, properties, operations, evidence kinds, boundedness, translation
  kinds, family support, and providers.
- `ipfs_datasets_py/logic/software_verification/` contains valuable typed IRs
  for transitions, programs, contracts, verification conditions, temporal
  properties, authorization, protocols, hyperproperties, separation logic,
  concurrency, and refinement.
- `ipfs_datasets_py/logic/backends/` has fail-closed provider execution,
  process isolation, typed results, portfolio planning, toolchain identities,
  and significant typed compiler models for SMT, TLA+, protocols, ATPs,
  kernels, and monitors.
- TDFOL, CEC/DCEC, modal, deontic, F-logic, hammers, proof corpus, and domain IR
  packages contain substantial domain knowledge and tests that must be
  migrated rather than discarded.

### The blocking architectural gaps

1. The frozen default family registry contains 21 canonical families but only
   two default translations and no default provider capability descriptors.
2. Backend capability metadata uses noncanonical labels including `fol`,
   `smt`, `smtlib2`, `state_transition`, `tla_plus`, `hyperltl`, provider names,
   property names, and execution-lane names as if each were a logic family.
3. Domain adapters introduce additional labels such as `typed_first_order`,
   `temporal_first_order`, `workflow_temporal`, `dynamic_hoare`,
   `threat_model`, `verification_condition`, `graph_projection`, and
   `structural_round_trip`.
4. `FormalFormula.expression` and related constraints accept arbitrary JSON or
   text. They do not establish binding, arity, sorts, alpha-equivalence,
   capture-avoiding substitution, or semantic identity.
5. Parser islands each define their own terms, formulas, operators, errors,
   source handling, and precedence. TDFOL, CEC/DCEC, legal deontic, FOL, modal,
   F-logic, runtime MTL, and UI formalization cannot safely exchange formulas.
6. There is no standalone SMT-LIB reader, no complete typed TPTP frontend, no
   shared Datalog/SecPAL grammar, and no typed F-logic/ErgoAI source parser.
7. Lean, Rocq, and Isabelle bridges are execution/scanning surfaces, not full
   language frontends. Their official elaborators and kernels must remain the
   authority.
8. Tests are extensive but siloed. There is no cross-product conformance suite
   covering domain IR x family/profile x translation x backend x authority.

### Existing code that this program must preserve

- `ui_ux_ir` is active uncommitted work in another checkout. Supervisor tasks
  must not recreate, copy, or overwrite it. The pinned datasets tree does not
  currently contain that package. UI integration is therefore an exact-source
  gate: LFP-038 records a typed declaration-only/source-missing disposition
  without touching `ui_ux_ir`. After the user's tree is committed/imported,
  the accelerator gitlink is updated, and the parser inventory/baseline join
  is rerun, objective refill emits one exact-revision derived task for a narrow
  adapter that preserves source maps, authority flags, graph projections, and
  golden fixtures.
- `security_ir`, `crypto_ir`, `intent_ir`, and `legal_ir` remain domain-rich
  models. The common logic kernel is not a replacement domain model.
- Typed backend request/result and evidence contracts remain the execution
  boundary. A parser never promotes a candidate result into a proof.
- Generated kernel source is checked by Lean/Rocq/Isabelle. Python-side
  parsing or a successful regex scan is not kernel authority.

## Target architecture

```text
Domain documents / solver files / controlled generated source
                          |
                    SourceDocument
             bytes, encoding, spans, provenance
                          |
                    Lossless CST
        tokens, trivia, recovery, source coverage
                          |
                 Modular surface AST
      notation-specific nodes and explicit ambiguity
                          |
                      Elaborator
 signature, scopes, sorts, arity, overloads, profiles
                          |
              Typed Logic Expression Kernel
 common nodes + versioned family extension nodes
                          |
                FormalizationArtifact views
 domain claim/obligation/source maps/coverage/withholding
                          |
               Reviewed Translation Graph
 preconditions, loss, bounds, polarity, authority ceiling
                          |
         Backend request / generated target theory
                          |
              solver / model checker / kernel
                          |
   typed result + trace/model/proof + validation receipt
```

### Required namespaces

These are distinct versioned identifiers. No field may silently carry more
than one role.

| Namespace | Examples |
| --- | --- |
| semantic family | `first_order`, `temporal`, `deontic`, `hyperproperty` |
| fragment/profile | QF_BV, Horn, finite-trace LTL, S4, SecPAL, Dolev-Yao |
| property/obligation kind | safety, liveness, validity, noninterference, VC |
| view role | source, normalized, proof translation, graph projection |
| notation/source syntax | canonical text, SMT-LIB2, TPTP FOF, TLA+, Tamarin |
| target encoding | SMT-LIB2, TPTP TFF, Lean 4, Rocq, Isabelle/HOL |
| provider | `z3`, `cvc5`, `tla_tlc`, `proverif`, `lean` |
| execution lane | SMT, state model, ATP, ITP kernel, runtime monitor, advisor |
| evidence kind | candidate proof, kernel proof, model, counterexample, monitor |

### Typed syntax kernel

Create `ipfs_datasets_py/logic/syntax_core/` with deliberately small,
side-effect-free contracts:

- `SourceDocument`, `SourceRange`, `SourceMap`, encoding and line index;
- token kinds, token values, trivia, bounded lexer contracts;
- CST node identity and complete source coverage;
- immutable AST node identity and versioned JSON codec;
- names, namespaces, symbols, sorts, type variables, signatures, arity;
- terms, applications, predicates, equality, Boolean connectives, binders,
  quantifiers, let/match where admitted;
- explicit extension nodes for modal, temporal, rule/fixed-point, protocol,
  heap/resource, trace, and program constructs;
- diagnostic severity/code/span/fix/related locations;
- strict and recovery parse modes with depth, token, input, ambiguity, and
  diagnostic limits;
- elaboration result containing symbol table, typed expression, unresolved
  nodes, assumptions, warnings, and content identity;
- capture-avoiding substitution, alpha-equivalence, free/bound-variable
  analysis, deterministic normalization, and canonical semantic hashing;
- parser, printer, elaborator, importer, and lowering protocols; and
- registry keys `(notation_id, notation_version, semantic_profile_id)`.

The core will not encode all semantics in one enum. Family extensions remain
modular, but every extension obeys the common identity, source, diagnostic,
visitor, codec, feature, and resource contracts.

## Canonical family and profile model

### Existing canonical foundations to retain

`propositional`, `first_order`, `higher_order`, `horn_chc`, `datalog`,
`frame_logic`, `modal`, `deontic`, `temporal`, `transition_system`,
`event_calculus`, `dcec`, `tdfol`, `mu_calculus`, `program`,
`separation_logic`, `concurrency`, `refinement`, `authorization`,
`cryptographic_protocol`, and `hyperproperty`.

### Version-2 additions and declaration-only candidates

- `epistemic`
- `doxastic`
- `intention_agency`
- `session_process`
- declaration-only initially: `dependent_type`, `description_logic`,
  `defeasible_logic`, `nonmonotonic_logic`, `argumentation`,
  `situation_calculus`, `probabilistic`, `fuzzy_weighted`,
  `relevance_paraconsistent`, and `finite_field_constraint`

`dynamic_logic` remains the current versioned alias/profile over canonical
`program`; it cannot also become a family without an alias-removal migration.
`information_flow` remains a property/profile concern under `hyperproperty`
until a distinct semantics and provider route are reviewed. Canonical `tdfol`
and `dcec` IDs are retained for compatibility and gain mandatory composition
metadata linking their temporal/first-order and deontic/event/cognitive
components; composition does not silently replace their identities.

### Semantic profiles, not new families

- classical, intuitionistic, paraconsistent consequence;
- open-world/closed-world/default-negation policy;
- finite/unbounded domains and bounded model-check depths;
- discrete/dense time, finite/infinite traces, stuttering, fairness;
- Kripke frame constraints K/D/T/S4/S5;
- strong/weak permission, dyadic norms, priorities, exceptions,
  contrary-to-duty semantics;
- Dolev-Yao adversary and equational theories;
- hypertrace quantifier prefix and supported alternation;
- SMT theory set, arithmetic semantics, and bit-vector widths; and
- proof-assistant universe/import/axiom environment.

### Required alias migration examples

| Legacy label | Canonical disposition |
| --- | --- |
| `fol` | family `first_order` |
| `smt`, `smtlib2`, `smt_lib` | notation/provider profile over typed families |
| `state_transition`, `tla_plus` | `transition_system` with TLA profile |
| `hyperltl` | `hyperproperty` with HyperLTL profile |
| `noninterference` | property/profile kind under `hyperproperty` |
| `protocol`, `proverif`, `tamarin` | `cryptographic_protocol`; tool names are providers |
| `secpal`, `policy` | `authorization` plus Datalog/SecPAL profile |
| `temporal_first_order` | composition of `temporal` and `first_order` |
| `safety`, `liveness` | property kinds |
| `verification_condition` | obligation/view role |
| `graph_projection` | view role |
| `lean`, `rocq`, `isabelle` | target/provider, not semantic family |
| `runtime` | execution/evidence lane |

Migration is dual-read/one-write: accept reviewed legacy aliases with typed
diagnostics, always emit canonical IDs, record the migration edge, and remove
an alias only after corpus and consumer-closure evidence.

## Translation and authority contract

Every translation edge declares:

- exact source/target family, profile, fragment, schema, and notation;
- feature preconditions and explicit unsupported constructs;
- added axioms, closure assumptions, fairness, bounds, attacker model, and
  arithmetic/domain changes;
- preservation relation: exact equivalence, equisatisfiable,
  theorem-preserving, model-preserving, trace-preserving, conservative over-
  or under-approximation, bounded, or heuristic;
- proof-safe and counterexample-safe polarity independently;
- total source-to-target node and symbol maps;
- unsupported, approximated, synthesized, and dropped nodes (silent drops are
  forbidden);
- compiler, profile, environment, and source/target content identities;
- maximum evidence authority that may emerge from the edge; and
- checker or reconstruction route.

Composed routes inherit the weakest guarantee and lowest authority ceiling.
Unknown or opaque semantics yield `unsupported`, `inconclusive`, or
`approval_required`, never an implicit success.

## Parser-family delivery waves

### Wave 0: vocabulary, inventory, and corpus

- Audit all parser/AST/type/family/provider strings and generate the initial
  matrix.
- Freeze conformance corpus schemas and representative positive, negative,
  ambiguous, adversarial, and translation fixtures.
- Establish provider/translation schemas and exact current baseline
  descriptors; generate the final parser/provider/translation projection only
  after family and domain adapters contribute their reviewed edges.

### Wave 1: common kernel and classical/rule syntaxes

- Canonical many-sorted FOL text.
- SMT-LIB2 S-expression reader/elaborator/printer.
- TPTP CNF/FOF/TFF reader/printer; THF later.
- Datalog, Horn/CHC, and SecPAL authorization syntax.
- F-logic/ErgoAI controlled subset.

This wave unlocks Z3, cvc5, Vampire, E, Datalog/SecPAL, ErgoAI, and most
bounded security/legal/crypto obligations.

### Wave 2: temporal, state, modal, normative, and cognitive syntax

- LTL, LTLf, past LTL, MTL intervals, CTL/CTL*, and trace/path quantifiers.
- Controlled TLA+ state/property expressions; full modules remain delegated to
  TLC/Apalache.
- K/D/T/S4/S5 modal profiles; deontic O/P/F and dyadic/defeasible norms;
  epistemic, doxastic, and intention/agency modalities.
- TDFOL, DCEC, event-calculus, and legacy legal/modal importers.
- Explicit syntax profiles remove overloaded single-letter ambiguity.

### Wave 3: protocols, programs, resources, and kernels

- Common symbolic-protocol DSL for terms, equations, roles, channels,
  adversaries, events, secrecy, authentication, and correspondence claims.
- ProVerif and Tamarin controlled source subsets and result trace mapping.
- Hoare/contracts/program-indexed dynamic logic, transition systems, separation/resources,
  concurrency/rely-guarantee, refinement, relational and session obligations.
- Target-neutral theory/declaration/theorem/import model and controlled
  generators/import manifests for Lean 4, Rocq, and Isabelle/HOL.
- Official elaborators/kernels remain authoritative; do not first attempt full
  proof-assistant parsers in Python.

### Wave 4: demand-driven advanced families

Description/ontology, defeasible/argumentation, situation calculus,
probabilistic/fuzzy, paraconsistent/relevance, richer fixed-point/mu-calculus,
and `finite_field_constraint`/ZK families remain declaration-only until a
domain use case, semantic profile, parser/semantics route, reference fixtures,
and validation route exist.

## Domain IR mapping

| Domain IR | Preferred families/views | Primary backends |
| --- | --- | --- |
| `security_ir` | SMT/FOL/CHC VCs; transition/temporal threat models; authorization; protocol; hyperproperty; separation/concurrency | Z3, cvc5, TLC, Apalache, SecPAL, ProVerif, Tamarin, HyperLTL, ATPs, kernels |
| `crypto_ir` | ledger/reorg/finality transition systems; arithmetic invariants; authorization/compliance; wallet/bridge/network protocols; anonymity/relational properties; refinement | Z3, cvc5, TLC, Apalache, SecPAL, ProVerif, Tamarin, HyperLTL, runtime MTL, kernels |
| `intent_ir` | typed facts/guards/effects; deontic/BDI intention; dynamic/Hoare skill effects; workflow temporal; tool/resource authorization | SMT, ATP, TLA, SecPAL, runtime MTL, kernels |
| `legal_ir` | deontic/conditional/defeasible; temporal FOL; event calculus; F-logic; description/argumentation declaration-only; authority/time profiles | Datalog/SecPAL, ATP, SMT bounded views, TLA processes, kernels, ErgoAI advisory |
| `ui_ux_ir` | F-logic/ontology; event calculus; TDFOL/DCEC; navigation transition/temporal; accessibility/privacy/security properties | SMT, TLA, runtime MTL, ATP, kernels, advisors |
| software verification IR | program/contract/VC; transition; temporal; separation; concurrency; refinement; protocol; hyperproperty | all exact compatible backend lanes |

The `ui_ux_ir` row is the target matrix, not a claim about the pinned tree.
Until its reviewed commit is present, every UI cell is declaration-only with
`source_not_in_pinned_revision`; LFP-038 records the gate and refill creates
the migration task when that source identity changes.

`safety`, `liveness`, `verification_condition`, `graph_projection`, and
`round_trip` remain property or view roles, never invented semantic families.

## Backend capability and evidence matrix

The executable provider registry keeps the existing provider IDs stable while
separating them from semantic family IDs:

| Provider ID | Provider lane |
| --- | --- |
| `z3` | SMT/CHC compiler and result decoder |
| `cvc5` | SMT compiler and result decoder; SyGuS declaration-only in v1 |
| `tla_tlc` | TLA+ finite-state TLC lane |
| `apalache` | bounded symbolic TLA+ lane |
| `datalog_secpal` | Datalog/Horn/SecPAL authorization lane |
| `proverif` | symbolic applied-pi protocol lane |
| `tamarin` | multiset-rewriting protocol lane |
| `hyperltl_autohyper_mchyper` | HyperLTL AutoHyper/MCHyper lane |
| `vampire` | classical TPTP ATP lane |
| `eprover` | E prover classical TPTP ATP lane |
| `hammer` | premise-selection/reconstruction strategy lane |
| `lean` | Lean kernel target lane |
| `rocq` | Rocq kernel target lane |
| `isabelle` | Isabelle/HOL kernel target lane |
| `runtime_mtl` | finite-trace metric-temporal monitor lane |
| `ergoai` (`ergo_ai` alias) | controlled F-logic/rule advisor lane |
| `symbolicai` (`symai` alias) | natural-language/symbolic proposal advisor lane |

Provider IDs select an executable adapter only. They never serve as family,
profile, property, view-role, syntax, or proof-authority identifiers.

| Provider | Native semantic lane | Required parsing/lowering work | Authority ceiling |
| --- | --- | --- | --- |
| Z3 | SMT theories, FOL, BV, arrays, CHC | typed SMT-LIB and AST lowering; model/core decode | exact query/profile only; validity via sound negation contract |
| cvc5 | SMT theories and FOL; SyGuS declaration-only in v1 | same shared SMT model; proof/model/core decode | exact implemented query/profile only |
| TLC | finite explicit TLA state models | controlled TLA expression/module generation | exhaustive only for configured finite state space |
| Apalache | symbolic bounded TLA analysis | typed transition/property lowering | bounded result only |
| Datalog/SecPAL | Horn/rule authorization | shared rule AST and explicit world/priority semantics | declared authorization profile only |
| ProVerif | applied-pi symbolic protocols | protocol AST -> source and attack trace parser | over-approximation-aware symbolic result |
| Tamarin | multiset rewriting/equational traces | protocol AST -> theory and trace parser | tool/version/profile-bound symbolic result unless independently replayed |
| AutoHyper/MCHyper | supported HyperLTL fragments | hypertrace AST and quantifier-profile checks | tool/bound-specific model-check result |
| Vampire | classical FOL/TPTP | TPTP AST and TSTP candidate parser | untrusted ATP candidate until checked/reconstructed |
| E prover | classical FOL/TPTP | shared TPTP path | untrusted ATP candidate until checked/reconstructed |
| Hammer | premise/tactic planning | typed goal snapshot adapters | advisory until reconstruction |
| Lean | dependent type theory/kernel | controlled generator/import manifest | kernel proof for exact pinned environment |
| Rocq | CIC/kernel | controlled generator/import manifest | kernel proof for exact pinned environment |
| Isabelle | HOL/LCF kernel | controlled generator/import manifest | kernel proof for exact pinned environment |
| runtime MTL | finite timed traces | shared temporal AST and exact interval codec | three-valued finite-trace monitor result |
| ErgoAI | F-logic/rules plus hybrid features | controlled F-logic parser, deterministic normalization | advisor/candidate unless independently verified |
| SymbolicAI | NL/symbolic proposal generation | candidate -> deterministic parser/elaborator | unverified candidate only |

Differential agreement is evidence, not voting authority. Disagreement among
Z3/cvc5, Vampire/E, TLC/Apalache, ProVerif/Tamarin, or HyperLTL tools becomes a
typed inconclusive case and a refill candidate.

## Validation program

### Parser conformance

- golden parse/print/parse semantic identity;
- complete source-span/trivia coverage;
- deterministic AST and semantic digests;
- exact negative diagnostic codes and ranges;
- explicit ambiguity and unsupported-node packets;
- input/token/depth/diagnostic/time/memory limits;
- Unicode/confusable/NUL/comment/string/numeric adversarial fixtures; and
- fuzzing with reduction and stable regression fixtures.

### AST algebra and elaboration

- alpha-renaming invariance;
- capture-avoiding substitution;
- free/bound-variable correctness;
- precedence and associativity;
- signature/arity/sort/universe errors;
- normalization and codec idempotence; and
- ASCII/Unicode notation equivalence only where explicitly promised.

### Translation validation

- feature-total positive and negative fixtures;
- no silent node or assumption loss;
- total node/symbol source maps;
- inverse round trips where meaningful;
- proof-safe/counterexample-safe polarity tests;
- explicit bounds/axioms/fairness/adversary receipts; and
- metamorphic models for preservation claims.

### Solver and kernel validation

- Z3/cvc5 differential checks on the common exact fragment;
- Vampire/E differential checks plus TSTP candidate normalization;
- TLC/Apalache aligned bounded state-model fixtures;
- ProVerif/Tamarin aligned protocol cases with documented semantic gaps;
- HyperLTL common-fragment cases;
- Python/TypeScript runtime-MTL golden traces where applicable; and
- Lean/Rocq/Isabelle reconstruction with pinned imports, axiom manifests, and
  rejection of `sorry`, `admit`, or equivalent trust escape hatches.

### Release floors

- zero emitted unregistered family IDs;
- every provider has a canonical capability descriptor;
- every emitted formal view has a registered translation route or explicit
  declaration-only/unsupported disposition;
- zero silent unsupported semantics, node drops, or authority upgrades;
- all domain IR vertical slices pass current-tree fixtures; and
- import/discovery remains lazy and free of network, installation, model, or
  subprocess side effects.

## Migration strategy

1. Add canonical contracts and audit tooling without changing legacy behavior.
2. Produce adapters from legacy ASTs/strings to typed expressions with typed
   diagnostics and loss receipts.
3. Dual-read canonical and legacy artifacts; write canonical only.
4. Migrate domain views and backend descriptors in file-disjoint lanes.
5. Gate each cutover on consumer-closure and conformance evidence.
6. Deprecate aliases with machine-readable replacements.
7. Remove legacy nodes/parsers only after corpus, public API, and downstream
   consumer evidence proves replacement coverage.

## Supervisor execution model

The program uses four hash-preferred lanes with non-strict fallback work
stealing, so an idle lane may claim any otherwise-unreserved ready task.
Control documents are protected. Domain and parser work occurs in initialized
`ipfs_datasets_py` nested worktrees; a nested commit must land before a
serialized accelerator gitlink update. Objective refill starts below eight
open tasks (two per lane), emits at most 24 findings per epoch, caps the open
set at 48, and observes a 3,600-second unchanged-evidence cooldown.

Initial ready width is four inventory/corpus/audit tasks. Shared contracts are
joined before parser implementation. After that gate, parser-family,
domain-adapter, backend-lowering, and conformance lanes run in parallel with
explicit file ownership.

The configured route is:

```text
grok_cli / grok-4.5
  -> only on independently verified hard quota exhaustion
codex / gpt-5.6-terra / high
```

Authentication failure, missing binary, timeout, generic rate limit, malformed
output, policy rejection, validation failure, or model preference does not
authorize Codex fallback.

## Refill policy

Objectives are durable; the seed task definitions are a sealed projection.
Refill is bounded and content-addressed. The supervisor control plane records
derived tasks in an append-only derived section and runtime gap ledger merged
with, but excluded from the immutable seed-definition seal. It may create
tasks from:

- a canonical family/profile/provider matrix cell without implementation or an
  explicit unsupported reason;
- an emitted unregistered family/profile/notation/provider ID;
- an unsupported AST node reached by an admitted domain corpus;
- a parser crash, fuzz reduction, ambiguous parse, or missing negative fixture;
- a translation preservation failure or silent-loss attempt;
- differential solver disagreement;
- a proof candidate not reconstructed by the declared authority route;
- a domain IR view without a parser/lowering/decoder vertical slice;
- provider capability drift; or
- a failed release floor.

Refill tasks must include the originating goal, family/profile, source/target
schema, preservation kind, authority ceiling, exact owned paths, fixture IDs,
validation command, and content identities. Default bounds are 8 goals, 24
derived tasks per epoch, 48 open derived tasks, depth 3, and two retries per
unchanged failure; the 11 immutable seed goals are excluded from derived-goal
limits. Refill never rewrites this plan, objective heap, seed task definitions,
scheduler config, or validator and never grants mutation or semantic authority.

## Completion definition

The program is complete only when the terminal release task joins current-tree
evidence for taxonomy closure, syntax-core algebra, parser conformance,
translation preservation, provider capability closure, domain vertical slices,
solver/kernel reconstruction, performance/resource bounds, lazy imports,
security/fuzzing, migration compatibility, and refill fixed point. A large test
count, successful model response, solver `sat`/`unsat`, or parser round trip by
itself is insufficient.
