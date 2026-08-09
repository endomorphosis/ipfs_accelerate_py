# IPFS Datasets Logic-Family Parser and Prover Bridge Plan — Wave 2

Status: implementation-ready successor program

Program ID: `ipfs-datasets-logic-family-parser-v2`

Task prefix: `LFP2-`

Root goal: `LFP2-G000`

Predecessor: completed `ipfs-datasets-logic-family-parser-v1`

Target repository: the `ipfs_datasets_py` gitlink in the
`agent/logic-family-parser-supervisor` accelerator branch

Supervisor: `ipfs_accelerate_py.agent_supervisor`

Provider route: Grok `grok-4.5` first; Codex `gpt-5.6-terra` with `high`
reasoning only after independently classified Grok hard-quota exhaustion

## Executive decision

Wave 1 established canonical family/profile/provider namespaces, a shared
source-aware `syntax_core` kernel, controlled parsers, typed domain adapters, a
capability matrix, authority ceilings, and a release receipt. Wave 2 does not
reopen or rewrite that evidence. It converts the remaining declared or
hermetic surfaces into typed, executable, replayable vertical slices and then
adds the next logic families on top of those stronger contracts.

The release matrix is administratively closed but still records 232
unimplemented cells, 558 explicit unsupported cells, 102 declaration-only
cells, only two registered translations, and a small conformance corpus. Some
frontends still bypass `ParseArtifact`/`TypedExpression`; some backends still
accept raw target source or a free-form family label; several end-to-end
records validate metadata without launching and replaying the real tool.

Wave 2 therefore follows this order:

1. rebaseline claims against executable current-tree behavior;
2. make extension nodes, backend requests, compiled artifacts, and provider
   capabilities typed and evidence-specific;
3. converge every controlled frontend on the shared parse/elaboration
   artifact pipeline;
4. grow a reviewed, compositional translation graph;
5. replace Boolean domain receipts with source-to-expression-to-backend
   evidence chains;
6. execute and replay every admitted solver family under pinned toolchain
   identities;
7. add high-value normative, ontology, agency, fixed-point, finite-field, and
   process/session families;
8. test reachable paths rather than claiming the full Cartesian product; and
9. refill only from content-addressed, owner-scoped, reachable gaps.

## Immutable predecessor binding

Wave 2 must preserve these Wave-1 anchors byte-for-byte:

- accelerator release commit
  `e162c19d087d4e6511f8eb97fd34ecb449777897`;
- datasets release commit
  `fc49cbb3e0e96bf07b367859da32123187d706c1`;
- Wave-1 seed-definition digest
  `sha256:f5d01bcc13c0b62d35b713cccb2e04abe49da454e9fa6f35cd28a5ad4b72eb44`;
- `docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md`;
- `docs/architecture/ipfs_datasets_logic_family_parser.objectives.md`;
- `docs/architecture/ipfs_datasets_logic_family_parser.todo.md`;
- `ipfs_datasets_py/docs/architecture/logic/LOGIC_FAMILY_PARSER_RELEASE.md`;
- `ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json`;
  and
- the Wave-1 fixed-point receipt and gap ledger.

Wave 2 uses a new namespace, task prefix, control files, runtime directory,
merge queue, worktree pool, evidence directory, and release receipt. It never
continues with hand-authored `LFP-052` tasks.

## Non-negotiable semantic contracts

### Distinct namespaces

The following remain separate typed identifiers:

| Namespace | Examples |
| --- | --- |
| semantic family | `first_order`, `temporal`, `deontic`, `mu_calculus` |
| profile/fragment | QF_BV, TFF, finite-trace LTL, S4, Dolev-Yao |
| property | validity, safety, liveness, noninterference, refinement |
| view role | source, normalized, VC, proof translation, graph projection |
| source notation | canonical text, SMT-LIB2, TPTP, TLA+, Tamarin |
| target encoding | SMT-LIB2, TSTP, Lean 4, Rocq, Isabelle/HOL |
| provider | `z3`, `cvc5`, `tla_tlc`, `vampire`, `lean` |
| execution lane | SMT, state model, ATP, kernel, monitor, advisor |
| evidence kind | parse, model, core, trace, attack, proof, kernel receipt |

No provider, syntax name, property, view, or strategy is admitted as a logic
family merely because an old string field used that label.

### Typed ingress and egress

Every admitted path has the following form:

```text
SourceDocument
  -> lossless/recovering ParseArtifact
  -> ElaborationArtifact + TypedExpression
  -> domain FormalizationArtifact and source maps
  -> reviewed TranslationPathReceipt
  -> BackendRequest@2 + CompiledLogicArtifact
  -> pinned provider execution
  -> ParsedTargetArtifact + evidence-specific result
  -> replay/reconstruction/independent validation
  -> bounded authority receipt
```

Raw target text may exist inside `CompiledLogicArtifact`, but cannot bypass a
typed family/profile/notation/encoding identity, source map, assumptions,
losses, resource bounds, and authority ceiling.

### Extension nodes

`LogicExtensionNode` becomes schema governed. Each extension kind declares:

- a versioned payload schema and canonical codec;
- child-node and binder positions;
- sort/kind checking hooks;
- free/bound-variable and substitution behavior;
- normalization and semantic-hash behavior;
- parser/printer registrations;
- translation feature requirements; and
- an explicit unsupported behavior when a consumer lacks the extension.

Opaque JSON payloads cannot silently cross an elaboration or translation
boundary.

### Authority

- Parsing proves syntax only.
- Elaboration proves static well-formedness only.
- Z3/cvc5/Vampire/E/TLC/Apalache/ProVerif/Tamarin/HyperLTL tools produce
  evidence under their declared semantics and bounds, not universal truth.
- Hammer proposes strategies and candidates; it is not a semantic family or
  proof authority.
- Lean, Rocq, and Isabelle claims become kernel evidence only after the exact
  generated theorem, imports, axioms, and toolchain identity are accepted by
  the official pinned environment.
- ErgoAI and SymbolicAI remain advisory until their output is reparsed,
  elaborated, translated, and independently validated.
- Runtime MTL verdicts carry clock, trace, finite-prefix, lateness, and
  monitorability assumptions.
- Differential agreement never votes a proof into existence.

## Workstream 1 — Evidence baseline and reachable matrix

Wave 2 first measures the current implementation rather than inheriting a
declaration as an executable claim.

- Audit every public claim against parser, elaborator, translator, compiler,
  runner, decoder, replay, and official-kernel evidence.
- Inventory all raw-source and arbitrary-payload ingress/egress boundaries.
- Replace the flat Cartesian interpretation with a sparse reachable graph:
  domain view -> typed family/profile -> translation path -> provider feature
  -> evidence kind.
- Track lifecycle states separately: declared, parsed, elaborated,
  translatable, compilable, executable, replayed, and independently validated.
- Expand the corpus with positive, negative, ambiguous, adversarial,
  round-trip, witness, proof, trace, attack, and resource-limit fixtures.

## Workstream 2 — Shared typed runtime contracts

Introduce these successor contracts without flag-day breakage:

- `ExtensionSchemaRegistry@1`;
- `LogicObligation@2` and `BackendRequest@2`;
- `CompiledLogicArtifact@1` and `ParsedTargetArtifact@1`;
- `ProviderCapabilityMatrix@2`, generated from one reviewed source; and
- dual-read/canonical-write migration receipts for legacy requests and
  provider descriptors.

Legacy fields remain readable only through explicit adapters that emit
deprecation and loss diagnostics. New writes use canonical IDs.

## Workstream 3 — Frontend convergence

Bring the controlled SMT-LIB2, TPTP/TSTP, Datalog/Horn/CHC/SecPAL, F-logic,
protocol, Tamarin, program, temporal, modal, resource, TDFOL, and DCEC
frontends onto the same source/CST/AST/elaboration contracts.

This is not a promise to parse every vendor extension or complete theorem
prover language. Each frontend publishes a feature profile, resource bounds,
recovery behavior, printer guarantees, and explicit unsupported nodes.

## Workstream 4 — Translation graph

Wave 2 grows the reviewed graph from two edges to useful domain routes:

- program, verification-condition, and separation obligations to FOL,
  Horn/CHC, and SMT;
- transition, concurrency, refinement, and temporal properties to TLA+,
  runtime monitors, bounded SMT, and HyperLTL where sound;
- authorization, frame, and event-calculus views to Datalog/SecPAL, FOL, and
  ATP encodings;
- modal, deontic, epistemic, intention, TDFOL, and DCEC formulas through
  explicit relational or reified encodings;
- restricted HyperLTL self-composition with quantified-trace limits; and
- a planner that composes preservation kind, polarity, assumptions, loss,
  boundedness, reconstruction requirements, and authority ceilings.

No translation is selected only because source and target family names are
present in the same registry.

## Workstream 5 — Domain IR vertical slices

| Domain | Required Wave-2 families and obligations |
| --- | --- |
| `security_ir` | typed FOL/SMT/CHC, authorization, transition/temporal, attack/protocol, information-flow/hyperproperty, separation/concurrency |
| `crypto_ir` | ledger transition/finality/refinement, authorization, symbolic protocol, arithmetic/bitvector/finite-field, consensus safety/liveness, hyperproperty |
| `intent_ir` | typed FOL, BDI/intention, deontic/prioritized policy, workflow temporal, dynamic/Hoare, authorization and tool guards |
| `legal_ir` | dyadic/defeasible norms, temporal event calculus, argumentation, description logic/ontology, jurisdiction and exception priority |
| `ui_ux_ir` | accessibility, interaction/event, workflow temporal, ontology/frame, authorization and observable state; exact-source gated |
| software verification/contracts | VC, separation/resource, concurrency, session/process, refinement, trace and kernel-theory routes |

Every domain slice must preserve a source-span chain from the originating
claim through `TypedExpression` and the exact backend request. Boolean
`parsed=True` or `roundtrip=True` fields are insufficient evidence.

The pinned Wave-1 tree does not contain `ui_ux_ir`. Wave 2 must not invent,
copy, or overwrite it. A source-gated adapter task may complete with a typed
`source_missing/declaration_only` receipt; a content-addressed refill task is
created only when a reviewed git revision containing that package is pinned.

## Workstream 6 — Provider execution and replay

| Provider | Wave-2 execution contract |
| --- | --- |
| `z3`, `cvc5` | typed advanced SMT theories, models, unsat cores/proofs where available, cross-solver differential checks and replay |
| `tla_tlc`, `apalache` | typed module/property subset, fairness/bounds, trace decoding and replay |
| `datalog_secpal` | typed authorization/rule import, query/result parity and provenance |
| `proverif`, `tamarin` | equations, roles/rules, correspondence/authentication, secrecy and attack replay |
| `hyperltl_autohyper_mchyper` | separate provider capabilities, quantifier limits, self-composition and counterexample validation |
| `vampire`, `eprover` | typed TPTP/TSTP input/output, SZS status, proof/countermodel reconstruction |
| `hammer` | candidate selection, ATP call, reconstruction and kernel-check phases kept distinct |
| `lean`, `rocq`, `isabelle` | controlled theory generators, pinned imports/axioms, official elaborator/kernel receipt |
| `ergo_ai` / ErgoAI | controlled F-logic/rule proposal, deterministic reparse and bounded external validation |
| `symai` / SymbolicAI | unverified candidate generation only; deterministic parse/elaborate gate |
| `runtime_mtl` | real monitor invocation, event-time/clock semantics and verdict replay |

Tool absence is an availability result, never permission to use a mock as
proof. Scheduled CI separates hermetic always-on tests from pinned optional
toolchain tiers.

## Workstream 7 — New logic families

Add executable semantics only in dependency order:

1. dyadic, conditional, defeasible, and prioritized normative logic;
2. argumentation and controlled nonmonotonic rule semantics;
3. description-logic/ontology profiles for legal, UI, intent, and knowledge
   graph use cases;
4. BDI, epistemic-temporal, agency, and intention profiles;
5. mu-calculus and controlled CTL-star/fixed-point lowering;
6. finite-field, bitvector, circuit, R1CS/PLONK-style constraint profiles for
   crypto/ZK use cases; and
7. linear, session, process, and relational refinement profiles.

Probabilistic, fuzzy, paraconsistent, full dependent type theory, unrestricted
TPTP THF, and complete OWL remain declaration-only until a domain requirement,
semantic profile, and validation provider are selected. Declaration does not
mean executable support.

## Workstream 8 — Validation and release

Acceptance is based on reachable vertical slices:

- parse/print/parse semantic identity and exact diagnostics;
- property tests for alpha-renaming, substitution, codec identity, extension
  traversal, and normalization;
- real subprocess-backed solver/model-checker/monitor tiers;
- independent replay of models, cores, traces, attacks, counterexamples, and
  proof certificates where supported;
- official kernel checks for reconstructed Lean/Rocq/Isabelle results;
- a reachable IR x translation path x provider x evidence matrix;
- zero silent node drops, family aliases at routing, raw unreceipted ingress,
  authority escalation, false capability, or unexplained reachable gaps; and
- bounded fuzzing and parser-bomb, Unicode, recursion, token, memory, process,
  timeout, and output limits.

## Supervisor DAG and parallelism

The seed board has 51 tasks (`LFP2-000` through `LFP2-050`) under ten child
goals. `LFP2-000` seals the control plane. Four independent P0 tasks are ready
at launch:

| Lane focus | Initial task | Owned surface |
| --- | --- | --- |
| contracts/catalog | `LFP2-001` | claim-vs-runtime audit |
| parsers/corpus | `LFP2-002` | raw-boundary inventory |
| translations/capability | `LFP2-003` | reachable capability graph |
| domains/evidence | `LFP2-004` | conformance corpus v2 |

Strict hash sharding is disabled. Task selection is dynamic; precise output
claims prevent unrelated work from serializing. The merge queue serializes
the outer gitlink update while nested task commits remain isolated and are
rebased onto the latest accepted datasets revision before validation.

Dependency waves are:

```text
G010 evidence baseline
  -> G020 shared typed contracts
     -> G030 frontend convergence + G070 family expansions
        -> G040 translation graph
           -> G050 domain vertical slices + G060 provider execution
              -> G080 validation
                 -> G090 refill fixed point
                    -> G100 release
```

Tasks within each goal are split by owned modules and may run concurrently
after their shared prerequisite lands.

## Objective refill contract

Static Wave-2 goals are immutable. The supervisor may append tasks, not mutate
or invent seed goals. Refill is enabled when fewer than eight open tasks
remain and is bounded to 24 findings per epoch, 48 open derived tasks, depth
three, two unchanged retries, and a 3600-second cooldown.

Admissible triggers include:

- a parser/profile counterexample;
- an extension schema/type/binder hole;
- a reachable translation hole;
- a domain formal view lacking a `TypedExpression` and source map;
- raw target ingress without a compiled/parsed artifact receipt;
- a capability claim lacking an execution or replay fixture;
- provider/toolchain drift;
- unvalidated model/core/trace/attack/proof evidence;
- a new reviewed `ui_ux_ir` source revision; or
- a failing reachable matrix cell.

Full Cartesian unsupported cells, advisor-only routes, vague codebase cleanup,
and duplicates are not refill tasks. Derived tasks require content identity,
an evidence obligation, discovery receipt, owner-scoped outputs, dependency
lineage, context budget, test command, and completion-authority ceiling.

Two identical quiet epochs over the same source, registry, corpus, provider,
and objective identities are necessary but not sufficient for release.

## Provider routing and failure policy

Implementation dispatch is sealed to:

```text
primary provider:  grok_cli
primary model:     grok-4.5
fallback provider: codex
fallback model:    gpt-5.6-terra
fallback effort:   high
fallback trigger:  primary_quota_exhausted
```

Authentication failure, malformed output, timeout, network failure, generic
provider error, worker crash, or validation failure does not authorize Codex
fallback. Only an independently classified hard Grok quota-exhaustion receipt
does. Provider availability is operational evidence only and never affects
logic truth or proof authority.

## Completion criteria

Wave 2 completes only when:

1. every seed and admitted derived task is terminal;
2. every child goal has content-bound evidence rather than task-ID presence;
3. every reachable domain view has a typed family/profile and either a
   validated route or an explicit bounded disposition;
4. every executable provider claim has pinned launch, parse, and replay
   evidence;
5. every translation records assumptions, loss, bounds, polarity,
   reconstruction, and authority;
6. UI remains source-gated unless an exact reviewed revision is pinned;
7. two current-tree refill scans admit no new reachable gaps;
8. the v1 anchors remain byte-identical and reachable; and
9. `LFP2-050` binds source revisions, schemas, registries, corpus, reachable
   matrix, provider/toolchain identities, validations, remaining explicit
   dispositions, and authority floors into `LogicParserReleaseReceipt@2`.
