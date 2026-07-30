# Agent Supervisor Proof-Gated Contract Repair Plan

Status: approved implementation plan
Program prefix: `RPR-`
Board namespace: `agent-supervisor-proof-gated-contract-repair-v1`
Merge target: `agent/proof-gated-contract-repair`

## 1. Outcome

Extend `ipfs_accelerate_py.agent_supervisor` so a broken call path can locate a
refactored receiver or a valid implementation site without letting semantic
similarity choose where an LLM writes code. Extend the same authority model to
intentional code changes: compute everything a changed contract can affect,
derive the repairs required at every dependent consumer, synthesize missing
values or support behavior analytically where possible, and admit one atomic
transitive migration plan or abstain.

The runtime order is normative:

```text
broken trace
  -> authoritative sender requirement
  -> snapshot-bound candidate nomination
  -> receiver/placement contracts
  -> ipfs_datasets_py logic obligations and reconstruction
  -> hard eligibility filter
  -> deterministic rerank
  -> admitted target decision
  -> bounded repair packet
  -> implementation
  -> re-index, re-resolve, re-prove, validate
```

For an intentional change, the corresponding normative order is:

```text
base/candidate semantic diff
  -> typed ProgramContract delta
  -> snapshot-bound program graph, vector, and history nomination
  -> dependency-complete reverse impact closure plus unknown frontier
  -> one migration obligation per affected consumer
  -> missing-value and behavior-source nomination
  -> ipfs_datasets_py.logic proof/refutation/reconstruction
  -> deterministic analytical transforms
  -> one exact atomic propagation plan or abstention
  -> bounded llm_router escalation only for admitted unresolved steps
  -> checkpointed multi-file implementation
  -> fixed-point re-index, re-resolve, re-diff, re-prove, and validate
```

Vector search is a recall mechanism. It may nominate candidates, but it is
never proof evidence, write authority, or a substitute for contract analysis.
No candidate path reaches the implementation agent unless its target decision
is admitted under the exact repository, graph, index, translator, toolchain,
policy, and proof roots.

## 2. Why this program is needed

The repository already has most of the safety substrate:

- `RepositoryIndexer`, `AnalysisASTIndex`, and `analysis_retrieval` provide
  snapshot-aware source, AST, lexical, graph, and optional vector evidence.
- `program_ast_adapters` records signatures, annotations, calls, errors, and
  effects for supported source shapes.
- `program_contracts.ProgramContract@1` models typed inputs and outputs,
  sync/async shape, errors, effects, capabilities, authorization, idempotence,
  ordering, atomicity, consistency, resource bounds, and degradation.
- `ipfs_datasets_logic_provider` capability-probes LogicIR, TDFOL, CEC, SMT,
  and Hammer and does not promote solver candidates without reconstruction.
- `ContractFinding`, `mcp_contract_edit_packet`, and
  `contract_mismatch_refinery` provide a proof-carrying repair handoff.

The missing join is target selection:

1. The compact repository index does not expose a code-symbol vector index
   bound to the rich signature/call/effect sidecar.
2. Unresolved calls are not classified as likely rename, moved symbol,
   adapter-required, missing implementation, external, or dynamic.
3. Retrieval has no contract-substitution, history, ownership, or
   reconstructed-proof signal.
4. Current edit packet materialization requires
   `selected_write_paths == finding.affected_paths`. A finding therefore fixes
   the write location before retrieval and proof can identify a moved receiver
   or a better implementation site.
5. There is no generic sender/receiver obligation compiler for code repair.
6. There is no exact before/after contract delta, whole-program call/data/value
   provenance graph, or dependency-complete fixed-point worklist for propagating
   an intentional API or data-shape change.
7. There is no analytical missing-argument source synthesis, complex support
   behavior contract, atomic multi-file/SCC migration plan, or proof that every
   affected consumer was updated.

This program inserts a versioned proof-gated decision between findings and
repair packets. It does not replace the active VFS symbolic-assurance or
datasets contract-analysis programs.

## 3. Composition with active programs

### 3.1 Reuse, do not fork

The active VFS program owns these shared foundations and predicted paths:

- program evidence graph and conservative call resolver;
- dependency-complete graph queries and datasets GraphRAG projection;
- contract extraction and symbolic comparison;
- code-contract LogicIR translation, proving, and minimal proof context;
- generic findings, finding task source, repair packet, and refill.

The datasets contract-analysis program owns the generic
`ipfs_datasets_py.logic.software_contracts` stack, including repository
snapshots, AST frontends, contract IR, resolver/call graph, evidence retrieval,
obligations, solvers, reconstruction, and receipts.

RPR tasks use uniquely named adapter and decision modules. They must not edit
the VFS-owned files while that program is active. Shared cutovers are isolated
in serialized tasks after the RPR adapters and receipts are tested.

### 3.2 Capability admission, not package-presence inference

At runtime, an interface admission report binds exact module paths, schema
versions, content identities, git revisions, toolchains, and supported
semantics. Missing or incompatible upstream features yield `unsupported` or
`unavailable`; they do not authorize an approximation and do not stop
file-disjoint RPR work from proceeding.

Current launch snapshot:

- exact accelerator `ipfs_datasets_py` gitlink is initialized and import-bound;
- cvc5 CLI/Python 1.3.3 and Z3 4.16.0 are available;
- mypy 1.20.2 and ruff 0.16.0 are available to validation workers;
- Node 18.19.1 and the managed `typescript@5.9.3` CLI/compiler API are
  available under a versioned user-local root;
- Python dependencies are declared in `requirements.txt`/`setup.py`, Z3 and
  cvc5 are registered in both datasets lazy solver surfaces, and TypeScript is
  provisioned only through the explicit pinned npm loader.

Capability probes and ordinary tasks never install a proof or analysis
toolchain. Provisioning requires an explicit operator `--install` request.

## 4. Non-negotiable invariants

1. **Exact state.** Every trace, query, candidate, contract, obligation, proof,
   decision, packet, and validation receipt binds the same repository forest
   and candidate tree, plus applicable graph/index/model/config roots.
2. **Expectation independence.** Expected behavior comes from reviewed
   IDL/schema, public signatures/stubs, conformance tests, normative specs, and
   manifests under explicit precedence. The candidate implementation cannot
   define the expectation used to validate itself.
3. **Logic before authority.** Candidate-specific compatibility or placement
   obligations are compiled through the admitted `ipfs_datasets_py.logic`
   interface before ranking can grant eligibility.
4. **Candidate non-authority.** Embeddings, GraphRAG, lexical matches, LLM
   opinions, solver candidates, tests, and observations cannot independently
   mint proof authority.
5. **Fail closed.** Ambiguous, incomplete, stale, unsupported, timed-out, or
   unreconstructed evidence produces abstention or review-only output.
6. **No score can erase a failed obligation.** A vector-nearest candidate with
   an incompatible type, error, effect, capability, lifetime, or resource
   contract is rejected.
7. **Write authority is downstream.** Only an admitted
   `RepairTargetDecision@1` may establish the repair packet's write allowlist.
8. **External roots remain read-only.** Retrieval across a repository forest
   does not expand mutation authority.
9. **Bounded context.** Repair packets carry compact contracts, counterexamples,
   identities, spans, and expansion handles, never unbounded repository bodies.
10. **Patch-bound completion.** After editing, the changed dependency closure
    is re-indexed, re-resolved, re-proved, and validated on the candidate tree.
11. **Closed impact accounting.** Every changed public or internally reachable
    contract yields one disposition for every resolved dependent plus an
    explicit bounded unknown frontier. “All consumers” is never claimed when
    reflection, dynamic dispatch, generated code, native/FFI, or excluded paths
    leave coverage incomplete.
12. **Per-consumer obligations.** A new required input creates a distinct
    migration obligation at each call site; a default or one successful caller
    cannot discharge the others.
13. **Proven value provenance.** A missing value is admitted only when its
    scope, reaching definition, path condition, type/schema, information
    content, effects, capabilities, authorization, ownership, lifetime, and
    dependency direction satisfy the consumer obligation.
14. **Analytical first.** Deterministic codemods and finite constructions are
    preferred when a unique reconstructed result exists. Ambiguous or
    unsupported synthesis does not become an LLM guess.
15. **Behavior before implementation.** A new class, method, or data structure
    receives an independently sourced contract for invariants, construction,
    state transitions, lifecycle, errors, effects, serialization, concurrency,
    compatibility, and ownership before any implementation is requested.
16. **Atomic propagation.** Strongly connected edit groups and their wiring,
    schema, generated, test, and migration changes are admitted as one
    checkpointed plan. Partial completion cannot close the originating change.
17. **LLM proposal only.** `llm_router` may implement only an admitted step with
    exact paths and postconditions. It cannot invent semantics, select a value
    source, expand the impact frontier, or weaken a proof obligation.

## 5. Typed data contracts

### 5.1 `BrokenContractTrace@1`

Required fields:

- repository/forest/tree and dirty-overlay identities;
- caller symbol, module, exact span, language/runtime, call form, and route;
- unresolved or mismatched receiver reference and resolver disposition;
- actual argument count/names and known value/type/range facts;
- awaitedness, return-value uses, caught/allowed errors, cancellation behavior;
- permitted effects/capabilities/auth context and resource budget;
- expected contract source references and precedence;
- graph frontier, completeness, exclusions, and runtime/static witness refs;
- analyzer/config/toolchain/policy identities and strict bounds.

Resolver dispositions are closed:

`resolved_mismatch`, `missing_local`, `likely_refactor`,
`adapter_required`, `external`, `dynamic`, `ambiguous`, `unsupported`.

### 5.2 `CallRequirementContract@1`

This is the precise sender/receiver join contract. It imports the supported
vocabulary from `ProgramContract@1` and separates:

- caller-provided pre-state and argument domain;
- receiver-required precondition;
- receiver-guaranteed post-state and result domain;
- caller-required result/postcondition;
- errors, effects, capabilities, authorization, temporal/state transitions,
  idempotence, ordering, atomicity, consistency, resource bounds, and fallback;
- evidence sources, unsupported clauses, assumptions, and invalidators.

For a candidate receiver `R` and sender/consumer requirement `S`:

```text
S.provided_inputs  => R.accepted_inputs
R.guaranteed_output => S.required_output
R.errors           subset-of S.allowed_or_handled_errors
R.effects          subset-of S.permitted_effects
R.capabilities     subset-of S.authorized_capabilities
R.resource_bounds  <= S.resource_budget
R.temporal/lifecycle refines S.required_lifecycle
```

Inputs are checked contravariantly and outputs covariantly. Required/optional,
positional/keyword-only, overload, nullability, union, generic, schema-version,
sync/async, context-manager, cancellation, and state-transition rules are
explicit. Unsupported dynamic behavior remains unknown.

### 5.3 `MemorySafetyFacet@1`

`ProgramContract@1.max_memory_bytes` is a resource bound, not a memory-safety
proof. A separate facet records:

- ownership and mutation regions;
- borrow/lifetime and aliasing constraints where the language supports them;
- nullability and bounds evidence;
- unsafe, native-extension, FFI, allocator/deallocator, and serialization
  boundaries;
- required compiler/static analyzer/model checker/sanitizer receipts;
- sandbox/cgroup enforcement receipts for process memory bounds;
- supported, unsupported, empirical, or proved disposition.

Python and TypeScript paths can establish modeled shape, effect, and resource
properties, but cannot claim general memory safety across reflection, native
extensions, FFI, monkey patching, or unmodeled services. Native paths require
policy-selected evidence such as borrow checking, Miri, ASan/UBSan, or an
equivalent verifier. Missing evidence is `unsupported`, never “safe”.

### 5.4 Candidate and decision contracts

`RepairCandidate@1` binds:

- exact target symbol/path/span and strategy nomination;
- retrieval snapshot, code-vector index/model/config roots, query identity;
- exact/history/AST/call/dependency/ownership/doc/test/vector evidence refs;
- extracted receiver or placement contract and completeness;
- proof obligations, attempts, reconstruction receipts, and rejection reasons.

`RepairTargetDecision@1` binds:

- the complete bounded candidate set and deterministic ordering;
- selected strategy and target, or explicit abstention;
- hard-gate outcomes and proof receipt identities;
- exact permitted read/write paths and spans;
- required post-edit validation/re-proof obligations;
- repository/tree/graph/index/model/translator/toolchain/policy roots;
- expiry/invalidation selectors and decision identity.

Allowed strategies:

`rename_substitution`, `adapter`, `implement_existing_declaration`,
`new_implementation`, `reject`, `ambiguous`.

## 6. Broken-trace classification

The trace classifier consumes the conservative program graph/resolver when
available and preserves unknown frontiers. It distinguishes:

- symbol removed but content/history identifies a move or rename;
- import/export/re-export/registration wiring drift;
- receiver exists but contract changed;
- receiver is compatible only through a bounded adapter;
- declaration/interface exists but implementation is missing;
- no implementation exists and a new site must be selected;
- external/dynamic/reflection/FFI edge not safely resolvable;
- multiple viable targets with insufficient discriminating evidence.

Same-name matches and filename rename reuse do not prove call resolution.

## 7. Code-symbol vector index and nomination

This is separate from the objective/todo vector index. It indexes code symbols,
not task prose.

Each content-addressed symbol row includes only bounded, provenance-bearing
features:

- qualified name, aliases, visibility, language/runtime, module/package;
- normalized signature, annotations/schema, async/errors/effects/capabilities;
- imports/exports/re-exports, callers/callees, registrations, interface links;
- normative doc/test/manifest references;
- Git rename/move/copy lineage and stable structural fingerprints;
- ownership/layer/dependency metadata;
- source and rich AST sidecar references, never an unbounded body.

The index identity binds repository tree/forest, chunking and normalization
rules, embedding model/revision/dimensions, distance metric, configuration,
included/excluded paths, tombstones, and producer/toolchain. Incremental update
must prove equivalence with a clean rebuild for affected fixtures.

Candidate nomination unions:

1. exact Git lineage/content/symbol fingerprints;
2. resolved aliases, re-exports, registrations, interfaces, and call distance;
3. AST signature/type/effect/error compatibility hints;
4. ownership and architecture anchors;
5. lexical/BM25 and vector similarity.

Poisoned, stale, cross-tree, unbounded, forged, or dimension/config-mismatched
results are rejected. Nomination results always have
`semantic_authority=false`.

## 8. Logic obligations

### 8.1 Substitution at the broken call site

The minimum useful claim is one-way refinement for the actual sender contract:

- caller facts imply receiver preconditions;
- receiver postconditions imply every modeled consumer use;
- all receiver errors are handled or allowed;
- receiver effects/capabilities fit policy;
- temporal, cancellation, state, resource, trust, and memory facets fit.

### 8.2 Pure rename/refactor equivalence

Labeling a candidate a pure rename requires more than substitution:

- bidirectional refinement over the supported contract domain;
- matching effects, errors, temporal/state behavior, and memory-safety facets;
- exact Git lineage, content/AST fingerprint, or other reviewed identity
  evidence connecting old and new symbols;
- closed reachability/wiring proof for the proposed call route.

If only one-way refinement is proved, classify it as compatible substitution,
not behavioral equivalence.

### 8.3 Adapter

An adapter may be admitted only when finite argument/result/error mappings are
explicit, total for the sender domain, effect/capability preserving, and
reconstructed under the same assumptions. Hidden coercions and LLM-invented
axioms are forbidden.

### 8.4 Existing or new implementation site

Placement obligations establish:

- target module/interface ownership and mutation authority;
- declaration or architecture anchor;
- no already-reachable compatible implementation was omitted;
- import/dependency layering remains acyclic and allowed;
- visibility/registration/export wiring is satisfiable;
- required capabilities/effects and memory facet are supportable there;
- generated stub contract exactly matches the sender requirement;
- read-only roots and generated/vendor/archive paths are excluded.

Failure to prove uniqueness or admissibility yields abstention, not an
arbitrary “best file”.

### 8.5 Proof authority

Supported obligations lower into immutable `ipfs_datasets_py.logic` claims with
exact premise/source/assumption identities. Backends may generate candidates.
Authority requires policy-approved deterministic checking or independent
reconstruction. `unknown`, `sat` without a reconstructable counterexample,
timeout, malformed output, toolchain drift, incomplete call slice, or
unsupported semantics remains non-conclusive.

## 9. Proof-aware eligibility and ranking

Ranking is lexicographic. Hard eligibility is evaluated before any soft score.

Hard gates:

1. exact and fresh tree/forest/graph/index/model/config roots;
2. target exists at the bound span, or the insertion anchor is proved;
3. write authority, ownership, language/runtime, visibility, and dependency
   policy permit the target;
4. expected contract has an independent authoritative source;
5. required semantics and call slice are complete for the claim;
6. every mandatory type/error/effect/capability/temporal/resource/memory
   obligation is conclusive under policy;
7. proof candidates are reconstructed and receipts are current;
8. no counterexample or hard rejection is present.

Eligible candidates are ordered by:

1. proof disposition and contract coverage;
2. exact Git lineage/structural identity;
3. resolved call/dependency distance and architecture ownership;
4. authoritative spec/test proximity;
5. AST signature compatibility;
6. lexical similarity;
7. vector similarity.

Weights, if used within a tier, are fixed, normalized only over explicitly
available signals, and recorded in the decision. A proof failure cannot be
compensated by score. A unique winner and policy-defined margin are required;
ties or low margin yield `ambiguous`.

## 10. Repair handoff

Add a versioned repair edit packet rather than silently weakening `@1`.
`ContractRepairEditPacket@2` consumes an admitted target decision and includes:

- broken trace and expected/observed contract identities;
- selected strategy, exact target path/span, and selection rationale;
- compact sender/receiver contract table and minimal counterexample;
- exact read/write allowlists derived from the decision;
- candidate/index/proof receipt references and bounded expansion handles;
- implementation postconditions and focused validation/re-proof commands.

The LLM receives only the admitted target as mutation authority. Rejected
alternatives may be summarized by identity and reason but cannot expand scope.
The task source and daemon validate the decision again immediately before
provider invocation. Any tree, target, index, translator, policy, or proof
drift invalidates the packet.

## 11. Post-edit completion gate

A repair is not complete until the candidate tree:

1. rebuilds affected source/AST/vector rows and tombstones;
2. re-resolves the original broken edge;
3. re-extracts sender/receiver contracts;
4. re-runs every original and patch-introduced obligation;
5. validates type/schema, errors, effects, capabilities, lifecycle, resources,
   and policy-selected memory-safety evidence;
6. runs focused contract tests plus dependency-complete impacted tests;
7. proves the original finding is closed without weakening the contract,
   deleting the test, suppressing the checker, or omitting a dependent;
8. emits patch/tree-bound validation and completion receipts.

Missing optional tools remain explicit. A skipped mypy, Z3, TypeScript, Miri,
or sanitizer check cannot satisfy a policy that requires it.

## 12. Benchmark and adversarial corpus

Seeded cases include:

- pure function rename and module move;
- barrel re-export, alias, registry, generated client, and interface move;
- same-name incompatible decoy and vector-nearest poisoned decoy;
- signature/default/keyword/nullable/schema version drift;
- sync/async, cancellation, error, effect, authorization, and capability drift;
- adapter-required migration;
- declared-but-unimplemented receiver;
- genuinely new implementation with one admissible owner;
- multiple admissible sites requiring abstention;
- dynamic dispatch/reflection/monkey patch/FFI frontier;
- ownership/lifetime/unsafe boundary with missing evidence;
- stale tree/index/model/translator/proof receipt;
- read-only external target and forbidden dependency-cycle target;
- incremental tombstone and clean-rebuild equivalence.

Release metrics:

- nomination recall@K and proof-eligible recall@K;
- admitted target precision and wrong-path mutation rate;
- false authoritative admission rate;
- abstention quality for ambiguous/unsupported cases;
- rename-equivalence precision versus one-way substitution;
- repair success and original-finding closure;
- stale/poison rejection rate;
- index/retrieval/proof latency, cache hit rate, tokens, and context bytes.

Required safety floors:

- wrong-path automated mutation rate: `0`;
- failed-obligation override rate: `0`;
- stale/forged/poisoned authoritative admission rate: `0`;
- unsupported memory-safety claim promotion rate: `0`.

## 13. Rollout

1. **Shadow:** generate candidates, proofs, rankings, and decisions; no prompt or
   write-path changes.
2. **Assist:** show admitted target and contract to a human; implementation
   remains approval-gated.
3. **Narrow auto:** allow only reconstructed, unique pure renames or closed
   substitutions in supported Python shapes with exact write authority.
4. **Expanded auto:** admit adapters or insertion only after benchmark floors,
   toolchain policy, rollback, and independent review are satisfied.

Feature flags are per repository/program/policy. Capability regression, metric
floor breach, stale root, reconstruction failure, or elevated abstention error
rolls back to the preceding stage.

## 14. Work waves

| Wave | Parallel work | Gate |
| --- | --- | --- |
| 0 | plan seal | committed clean target branch |
| 1 | contracts, code-vector index, adversarial fixtures, capability adapter | four file-disjoint ready lanes |
| 2 | broken-trace classifier, sender/receiver compiler | typed schemas and fixtures |
| 3 | candidate nomination, obligation compiler, memory facet evidence | trace/contract interfaces |
| 4 | proof reconstruction/admission, implementation-site admissibility | logic obligations |
| 5 | proof-aware reranker and target decision | complete candidate evidence |
| 6 | edit packet v2, refinery, pre-provider gate | serialized shared-file cutovers |
| 7 | patch-bound validation, benchmark, CLI/docs | admitted decision pipeline |
| 8 | shadow/assist/narrow-auto release | all safety floors |
| 9 | propagation contracts, capability adapter, fixtures, program graph | committed extension seal; four strict shards |
| 10 | semantic change delta, value vector index, impact closure, consumer inventory | exact base/candidate snapshots |
| 11 | schema/protocol analysis, dynamic frontier, value provenance | dependency-complete graph coverage |
| 12 | missing-value nomination and complex behavior contract | one obligation per affected consumer |
| 13 | LogicIR obligations, proof/refutation/reconstruction, placement | independent behavior authority |
| 14 | deterministic codemods and atomic plan admission | unique reconstructed analytical results |
| 15 | multi-edit packet, bounded llm_router fallback, task/provider gates | admitted exact paths and postconditions |
| 16 | checkpoint/rollback transaction and fixed-point validator primitives | all new-file adapters complete |
| 17 | serialized pipeline/daemon integration that requires those primitives | no mutation or completion bypass |
| 18 | adversarial benchmark and shadow/assist/narrow-auto rollout | candidate-tree closure and all safety floors |
| 19 | end-to-end operations validation | healthy restartable four-shard supervisor |

The companion objective heap and taskboard encode the exact local dependency
DAG and file ownership. Cross-program VFS/datasets requirements are capability
preconditions, not unknown task IDs, so they cannot deadlock this board.

## 15. Definition of done

The program is complete when a seeded broken call can:

- recover a real moved/refactored receiver or identify a proved insertion site;
- state the sender and receiver contract precisely with unsupported clauses;
- use code-vector search only for nomination;
- compile and reconstruct candidate-specific `ipfs_datasets_py.logic`
  obligations before eligibility;
- deterministically admit one exact target or abstain;
- give the implementation agent only that target and contract;
- re-prove and validate the patch on the candidate tree;
- demonstrate all safety floors on the adversarial corpus;
- operate through a healthy, restartable, isolated parallel supervisor.

The change-propagation extension is complete only when it can also:

- bind a base and candidate snapshot and explain the semantic contract delta
  independently of textual diff noise;
- enumerate every statically resolved direct and transitive consumer, group
  cycles, and preserve an explicit unknown frontier for unsupported behavior;
- detect each caller that still supplies two arguments after a callee begins
  requiring three and issue a separate migration obligation for it;
- prove a unique in-scope or constructible source for the new argument, thread
  the requirement upward to a fixed point, or abstain with a precise reason;
- define and prove the contract and admissible owner for any required new
  class, method, data structure, factory, schema, or state transition;
- analytically materialize supported transformations before considering an LLM;
- provide `llm_router` only an admitted behavior specification, counterexample,
  exact read/write scope, and patch-bound validation obligations;
- apply an admitted multi-file/SCC plan transactionally and demonstrate zero
  unresolved impacted consumers, zero uncovered required frontier, and no new
  contract delta on the candidate tree.

## 16. Change-propagation scope and change identity

The extension accepts a reviewed base/candidate repository pair or a proposed
typed contract change. `ProgramChangeSet@1` binds both forest/tree identities,
dirty overlays, submodule/gitlink roots, build and generated-code manifests,
language/toolchain/config/policy identities, changed spans, tombstones, and the
producer that normalized the diff. It separates source edits from derived
semantic deltas so formatting, moves, generated churn, and comments do not
manufacture migration obligations.

`ProgramContractDelta@1` compares independently extracted before/after
`ProgramContract@1` records. Its closed delta kinds include:

- parameter add/remove/rename/reorder/default/keyword/variance changes;
- result, generic, nullability, schema, serialization, and protocol changes;
- sync/async, cancellation, errors, effects, capability, authorization,
  lifecycle, temporal/state, consistency, resource, and memory-facet changes;
- symbol move/rename/re-export/registration and visibility changes;
- constructor, field, method, class, data structure, interface, and factory
  introduction or removal.

Each clause is `breaking`, `compatible`, `behavioral`, `unknown`, or
`unsupported` for a stated consumer domain. A compatible source declaration
does not imply compatibility for every runtime route.

## 17. Program knowledge graph and complete impact accounting

The program graph is an adapter over the existing repository/AST indexes,
`CodeImpactIndex`, and `SemanticDependencyGraph`; it does not fork their
identity or trust models. Concrete `program_graph.py` and
`program_call_resolver.py` façades satisfy the existing capability probes.
Every node and edge records exact source roots, extractor identity, confidence,
authority, and completeness.

The graph covers, where statically supported:

- declarations, definitions, calls, overrides, implementations, overloads,
  constructors, factories, builders, dependency injection, registries,
  callbacks, decorators, and context managers;
- imports, exports, re-exports, aliases, modules, packages, build targets,
  generated bindings, native boundaries, and dependency-layer ownership;
- parameters, return values, fields, aliases, reaching definitions, dominance,
  path conditions, data flow, state flow, effects, resources, and capabilities;
- schemas, serializers/deserializers, migrations, databases, messages, RPC,
  HTTP/CLI surfaces, configuration/default providers, feature flags, and IDLs;
- tests, mocks, fixtures, examples, benchmarks, documentation contracts, and
  operational validation that exercise or promise the changed behavior.

`ImpactClosureReceipt@1` starts at each semantic delta and computes reverse
dependencies to a fixed point. It records resolved consumers, SCCs, required
validation, exclusions, resource bounds, and an unknown frontier. Vector,
GraphRAG, history, and runtime witnesses may nominate missing edges, but an
edge becomes authoritative only through an admitted extractor, reviewed
manifest, resolver, or reconstructed logic result. Reflection, monkey
patching, string dispatch, plugin loading, FFI, remote services, generated
sources, and excluded roots remain explicit frontiers unless closed by such
evidence.

## 18. Per-consumer compatibility and missing-value synthesis

Each impacted node receives a `ConsumerMigrationObligation@1`, not merely a
search hit. For the representative change:

```python
def process(left: A, right: B) -> R
def process(left: A, right: B, context: C) -> R
```

the system inventories every resolved caller, including aliases, wrappers,
method dispatch, factories, tests, mocks, generated clients, and calls reached
after an argument is threaded through intermediate functions. A caller that
still supplies two arguments gets a `MissingInputRequirement@1` containing:

- the exact callee clause and consumer path condition;
- required type/schema/range/nullability and information content;
- construction preconditions and result postconditions;
- allowed errors, effects, capabilities, authorization, trust, and resources;
- ownership, lifetime, mutability, concurrency, caching, and disposal rules;
- required propagation depth, compatibility policy, and proof obligations.

Candidate sources are nominated from in-scope names, parameters, receiver
state, reaching definitions, dominated branches, config and environment
providers allowed by policy, request/session context, DI containers, factories,
schemas, history, the program graph, and the code vector index. Nomination is
body-bounded and non-authoritative.

The analytical synthesizer then proves or refutes, for each candidate:

1. it is available on every relevant control-flow path;
2. its type/schema and refinement facts satisfy the missing input;
3. it carries the required information rather than merely sharing a type;
4. any conversion or constructor is total for the caller domain;
5. errors, effects, capabilities, authorization, trust, and resources fit;
6. ownership, lifetime, aliasing, mutation, and concurrency constraints fit;
7. importing, constructing, or threading it creates no forbidden cycle;
8. the same reconstructed conclusion holds under exact current roots.

A unique proved source yields a deterministic mapping. No source yields
refutation or a precisely typed upstream requirement. Multiple proved sources
yield ambiguity. Unsupported analysis yields abstention. When an input must be
threaded through callers, the worklist repeats until it reaches an
authoritative source, a public boundary with a reviewed compatibility choice,
or an unknown frontier.

## 19. Complex behavior and new-type contract synthesis

When the missing input requires a new class, method, data structure, provider,
schema, or stateful service, the system defines `RequiredBehaviorContract@1`
before choosing placement or generating code. Evidence precedence remains
independent of the candidate implementation:

1. reviewed IDL/schema/public stub and compatibility policy;
2. normative specification and conformance tests;
3. caller postconditions and callee preconditions;
4. data invariants, migration manifests, architecture ownership, and history;
5. implementation observations only as non-authoritative hypotheses.

The contract states:

- fields, variants, generic parameters, invariants, validation, and defaults;
- constructors/factories, totality, initialization ordering, and DI wiring;
- methods, state machine, allowed transitions, temporal behavior, and
  idempotence;
- ownership, lifetime, mutability, aliasing, caching, disposal, and concurrency;
- serialization, persistence, versioning, migrations, equality, and hashing;
- errors, cancellation, effects, capability/auth/trust boundaries, logging,
  privacy, resource limits, and degradation;
- compatibility adapters, rollout behavior, tests, and observability.

Placement is separately proved against ownership, visibility, architecture
layering, registration/export wiring, read/write authority, and dependency
cycles. If evidence cannot uniquely determine behavior or placement, the plan
records a decision requirement and does not ask a model to invent it.

## 20. Proof, analytical transformation, and plan admission

RPR-specific obligations lower supported deltas, consumer constraints, value
provenance, constructors, state transitions, and placement facts through the
capability-admitted `ipfs_datasets_py.logic` stack. Premise selection, SMT/ATP
outputs, and LegalIR-style gap suggestions are candidates only. Kernel
reconstruction under the exact premise, translator, toolchain, and policy roots
is required for authority.

`AnalyticalTransform@1` supports only closed transformations with deterministic
rendering and replay, such as:

- add/rename/reorder an argument using a unique proved expression;
- thread a parameter through an acyclic or explicitly grouped call chain;
- add a proved import/export/registration or finite adapter;
- update a typed constructor, schema field, serializer, fixture, and generated
  manifest when their mappings are total and policy-authorized.

`AtomicPropagationPlan@1` contains the complete candidate set identity, all
impact-closure nodes, one disposition per obligation, exact edit/validation
steps, step dependencies, SCC transaction groups, read/write authority,
preconditions, postconditions, invalidators, checkpoint/rollback strategy, and
the final fixed-point proof obligation. Admission requires:

- a current complete impact closure or an explicitly review-only frontier;
- reconstructed proof for every automated value mapping and behavior clause;
- no unresolved mandatory consumer, forbidden dependency, or counterexample;
- one deterministic plan; competing mappings or placements force abstention;
- exact paths derived from evidence and authority, never a similarity score.

## 21. Bounded `llm_router` fallback

Analytical repair is attempted first. `llm_router` is used only when the
behavior and placement are admitted but syntax or a bounded implementation is
not available as a deterministic transform. Routing uses the existing
proposal/reviewer/writer-lease boundary, never a direct model call.

The packet includes the exact plan and step identity, behavior contract,
before/after contract table, minimal counterexamples, proved value sources,
unsupported limits, exact read/write paths, per-edit postconditions, focused
commands, and fixed-point re-proof requirements. It excludes secrets,
unbounded bodies, rejected alternatives as scope, and unresolved semantics.
The model may propose code within the lease; it cannot choose a different
source, create a new dependency, relax a contract, omit a consumer, suppress a
checker, or expand paths. A pre-provider gate revalidates all roots and proofs
immediately before invocation.

## 22. Transactional execution and fixed-point completion

Execution occurs in an isolated candidate worktree with a content-addressed
checkpoint. Steps run in dependency order; SCC members are treated as one
transaction group. A failure, drift, scope escape, timeout, or incomplete
group rolls back to the checkpoint and retains a diagnostic receipt. No
partial plan may be merged or marked complete.

After every group, and again for the final candidate tree, the validator:

1. rebuilds affected repository, AST, vector, and graph rows and tombstones;
2. re-extracts the base/candidate semantic delta;
3. re-resolves calls, data/value flow, constructors, schemas, and wiring;
4. recomputes the reverse impact closure and unknown frontier;
5. checks that each original consumer obligation is discharged exactly once;
6. detects new deltas/consumers and repeats to a policy-bounded fixed point;
7. reconstructs all original and introduced logic obligations;
8. runs type/schema/effect/capability/resource/memory and dependency-complete
   tests without accepting weakened tests, deleted contracts, or suppression;
9. emits a candidate-tree-bound `PropagationCompletionReceipt@1`.

Completion requires zero unresolved mandatory consumers, zero omitted resolved
dependents, zero uncovered required frontier, and no unplanned breaking delta.
Bound exhaustion is an explicit incomplete result, not success.

## 23. Adversarial evaluation and rollout

Fixtures cover parameter addition/removal/rename/reorder, argument threading,
same-typed but semantically wrong sources, branch-local values, nullable and
schema mismatch, constructor failure, dependency cycles, async/context
changes, stateful services, new classes and serializers, generated bindings,
registries/reflection/plugins/FFI, poisoned graph/vector/history evidence,
stale roots, SCC partial failure, LLM scope escape, and a second-order delta
introduced by the first repair.

In addition to the original floors, release requires:

- missed resolved impacted-consumer rate: `0`;
- unproved or wrong value-source admission rate: `0`;
- behavior invented without independent authority rate: `0`;
- partial propagation completion rate: `0`;
- stale graph/index plan-admission rate: `0`;
- fixed-point false-completion rate: `0`.

Shadow mode records deltas, closure, proofs, and proposed plans. Assist mode
shows exact obligations and admitted plans to an operator. Narrow auto initially
permits only unique reconstructed analytical transforms on supported Python
shapes with a complete frontier. LLM-authored, complex stateful, schema/public
API, generated/native, dynamic, and cross-repository changes remain
approval-gated until separate reviewed policy and benchmark evidence expands
their scope. Any safety-floor breach, capability regression, stale root,
reconstruction failure, or coverage loss rolls back one stage.
