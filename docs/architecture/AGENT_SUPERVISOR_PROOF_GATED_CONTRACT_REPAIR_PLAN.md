# Agent Supervisor Proof-Gated Contract Repair Plan

Status: approved implementation plan
Program prefix: `RPR-`
Board namespace: `agent-supervisor-proof-gated-contract-repair-v1`
Merge target: `agent/proof-gated-contract-repair`

## 1. Outcome

Extend `ipfs_accelerate_py.agent_supervisor` so a broken call path can locate a
refactored receiver or a valid implementation site without letting semantic
similarity choose where an LLM writes code.

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
