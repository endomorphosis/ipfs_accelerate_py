# Agent Supervisor Formal Verification Plan

## Objective

Augment `ipfs_accelerate_py.agent_supervisor` with proof-aware planning,
validation, merge, and goal-completion controls by consuming the logic
capabilities exposed by `ipfs_datasets_py`.

The integration has two equally important outcomes:

1. Reduce model context by turning repository state, AST changes, invariants,
   prior proofs, and counterexamples into queryable, bounded context capsules.
2. Increase assurance by requiring machine-checkable evidence for selected
   code invariants before a task can merge or a goal can become verified.

This is not a plan to claim full formal verification of arbitrary Python.
Tests, static analysis, and review remain necessary. Formal gates apply only
where the supervisor can generate a sound obligation with supported semantics.

## Repository Findings

### Reusable `ipfs_datasets_py` capabilities

| Capability | Existing implementation | Integration decision |
| --- | --- | --- |
| Hammer portfolio | `logic/hammers/models.py`, `portfolio.py`, `reconstruction.py`, `receipts.py` | Reuse through an optional proof-provider adapter. ATP and SMT output remains an untrusted candidate until target-kernel reconstruction succeeds. |
| Trust-aware proof cache | `logic/hammers/proof_cache.py` | Reuse its content-addressed key dimensions and trust checks. Add supervisor single-flight and query projections, not another independent trust model. |
| Lean checking | `logic/hammers/frontends/lean.py`, `logic/modal/lean_runtime.py` | Use local Lean kernel acceptance as a high-assurance result. Reject `sorry`, `admit`, unsafe declarations, unavailable toolchains, and unchecked generated text. |
| Leanstral | `logic/modal/leanstral.py`, `leanstral_verifier.py`, `leanstral_validation.py` | Use only as a proposal and decomposition lane through `llm_router`. It never grants verified status. |
| Incremental AST validation | `optimizers/logic_theorem_optimizer/incremental_validation.py` | Generalize its typed changed-scope and staged-validation pattern for supervisor code changes. |
| Formula dependency graph | `logic/TDFOL/formula_dependency_graph.py` | Reuse proof-chain concepts for obligation dependencies and context selection. Do not equate legal TDFOL semantics with Python semantics. |
| TDFOL and external provers | `logic/TDFOL`, `logic/external_provers` | Use only for obligations with an explicit translation contract. Unsupported translations fail closed or fall back to non-formal validation. |
| Frame logic and graph projection | `logic/bridge/modal_frame_logic.py`, `logic/modal/kg_bridge.py` | Reuse graph shapes and query concepts for evidence navigation. Frame logic is not itself a verifier for Python source. |
| ZKP | `logic/zkp`, `logic/bridge/zkp_attestation.py` | Initially attest already verified receipts. Simulated proofs are never authoritative. Production enforcement requires a healthy cryptographic backend, versioned circuit, verification key, and no-leak checks. |
| Artifact persistence | Supervisor `artifact_store.py` | Extend existing paired JSON and DuckDB artifacts with proof, obligation, graph, and metric tables. |
| CPU admission and validation | Supervisor `resource_scheduler.py`, `validation_scheduler.py` | Add proof resource classes and one top-level budget so nested solver, kernel, test, and model pools cannot oversubscribe the host. |

### Baseline verification

The local capability sample established the following baseline:

- Hammer model, cache, and portfolio tests: 156 passed, 1 skipped.
- ZKP backend selection, theorem semantics, and witness no-leak tests:
  35 passed.
- Incremental validation tests: 9 passed.
- Lean and other solver executables are present locally: Lean, Lake, Z3,
  cvc5, and Coq.
- The selected Leanstral suite produced 2 passes and 22 failures because the
  current environment lacks spaCy. This is an environment/capability failure,
  not proof evidence. The supervisor must report it as degraded and continue
  with independent proof lanes.

## Soundness Boundary

The supervisor must use a monotonic trust lattice:

1. `unverified`: model text, heuristic inference, or unchecked artifacts.
2. `checked`: deterministic schema, AST, lint, type, or test evidence.
3. `solver_candidate`: ATP/SMT found a candidate result or counterexample.
4. `kernel_verified`: a configured trusted kernel reconstructed and accepted
   the exact obligation under the exact premises.
5. `attested`: a cryptographic envelope binds a kernel-verified receipt to its
   public statement and provenance.

No adapter may map an LLM response, solver exit code, cached untrusted result,
or simulated ZKP directly to `kernel_verified` or `attested`.

Every receipt must bind:

- repository and candidate tree identity;
- changed-scope and obligation CIDs;
- normalized assumptions and conclusion;
- selected premises and their provenance;
- translator, theorem-template, solver, kernel, and toolchain versions;
- proof policy and resource-budget hashes;
- stdout/stderr digests and bounded diagnostics;
- cache provenance, timestamps, and expiration;
- kernel acceptance and reconstruction status;
- optional attestation backend, circuit, verification key, and public inputs.

Unknown, unsupported, timed-out, stale, malformed, or contradictory evidence
must never be promoted into a successful formal result.

## Target Architecture

```text
task + candidate tree
        |
        v
AST changed-scope compiler -----> deterministic code/evidence graph
        |                                      |
        v                                      v
obligation template registry          bounded context query
        |                                      |
        +----------> proof-plan DAG <----------+
                           |
               cache lookup / single-flight
                           |
        +------------------+------------------+
        |                  |                  |
 deterministic checks   solver portfolio   Leanstral proposal
        |                  |                  |
        |             proof candidate         |
        |                  |                  |
        +------------ kernel reconstruction <-+
                           |
                     proof receipt
                           |
              validation / merge / goal gate
                           |
          JSON + DuckDB evidence graph and metrics
                           |
                optional ZKP attestation
```

### Dependency boundary

`ipfs_datasets_py` contains the logic engines and also vendors
`ipfs_accelerate_py` as a submodule. A mandatory import from the supervisor
back into the parent package would make installation and testing brittle.

The supervisor therefore owns a versioned `ProofProvider` protocol with:

- lazy in-process discovery when `ipfs_datasets_py` is importable;
- a bounded subprocess JSON protocol for isolation and independent packaging;
- deterministic capability and toolchain probes;
- explicit unsupported/degraded responses;
- timeouts, cancellation, resource limits, and network policy;
- fixture providers for supervisor unit tests.

The supervisor remains usable with no proof provider installed. Policy decides
whether that means shadow-mode degradation or a fail-closed merge gate for a
specific protected surface.

## Core Data Contracts

### `CodeProofObligation`

A canonical obligation includes:

- stable obligation CID and schema version;
- task, goal, subgoal, branch, base tree, and candidate tree identities;
- typed AST scopes and source hashes;
- invariant template identifier and version;
- normalized assumptions, conclusion, and premise references;
- required assurance and supported backend families;
- risk, timeout, memory, CPU, and confidentiality policy;
- fallback validation commands for unsupported proof surfaces.

### `ProofPlan`

A plan is a DAG of obligations and deterministic validation steps. It records:

- dependency and conflict edges;
- selected provider/backend and selection rationale;
- cache eligibility and required freshness;
- expected CPU and memory cost;
- whether model assistance is permitted;
- promotion policy for each result;
- critical path and downstream unlock value.

### `ProofReceipt`

A receipt records the exact request, environment lock, attempts,
reconstruction, verdict, cache provenance, and diagnostics. It has separate
fields for:

- solver verdict;
- kernel acceptance;
- deterministic validation;
- ZKP verification;
- authoritative assurance level.

The authoritative level is derived by code from evidence fields. Providers
cannot set it directly.

### `ProofContextCapsule`

A bounded capsule supplied to Codex or Leanstral contains only:

- the task contract and impacted qualified symbols;
- applicable invariant templates;
- trusted prior receipts and dependency proofs;
- compact counterexamples or unsat cores;
- unresolved obligations and unsupported semantics;
- required tests and promotion policy;
- bounded source excerpts selected by exact graph edges.

It excludes raw repository-wide AST records, full graph dumps, hidden ZKP
witnesses, and unrelated proof transcripts.

## Obligation Strategy

Start with finite, explicit supervisor invariants that can be modeled
soundly:

- task, goal, lease, merge, and retry state transitions are legal;
- a canonical task has at most one accepted live lease;
- fencing tokens prevent stale workers from publishing;
- task and proof dependency graphs are acyclic;
- merge candidates are unique and merge processing is idempotent;
- completion cannot become verified without fresh mandatory evidence;
- cache keys include every semantic and trust dimension;
- stale or contradictory evidence reopens the affected goal;
- JSON and DuckDB projections represent the same canonical records;
- generated task identities are stable and active work is deduplicated;
- resource reservations never exceed configured capacity;
- unsupported proof semantics cannot silently pass a gate.

For each template, maintain:

- a Python reference predicate;
- a canonical logical statement;
- bounded positive, negative, and mutation fixtures;
- supported translator/backend list;
- reconstruction expectations;
- fallback tests for unsupported cases.

Do not synthesize free-form formulas from arbitrary Python and call them
verified. New templates enter enforcement only after semantic review,
mutation tests, and a trusted reconstruction fixture.

## Knowledge Graph and Context Reduction

Build a deterministic code/evidence graph from existing supervisor records.
Initial node types are:

- goal, subgoal, task, branch, tree, file, AST symbol;
- invariant template, obligation, premise, proof attempt, proof receipt;
- validation command, validation receipt, merge receipt, contradiction;
- provider, solver, kernel, toolchain, policy, cache entry, attestation.

Initial edge types are:

- `implements`, `changes`, `imports`, `calls`, `depends_on`;
- `requires_obligation`, `uses_premise`, `proved_by`, `invalidated_by`;
- `validated_by`, `merged_by`, `covers_criterion`, `reopens`;
- `cached_as`, `attested_by`, `derived_from`.

The authoritative graph is derived from AST and receipts. LLM or GraphRAG
enrichment may suggest aliases and retrieval terms, but cannot create proof or
completion edges.

Persist paired JSON and DuckDB projections through the existing artifact
store. Context assembly queries indexed columns by task CID, symbol, tree,
obligation, assurance, and freshness. It must enforce row, byte, token, and
hop limits before constructing a model prompt.

## Parallel Execution Policy

Proof execution must preserve CPU throughput without nested oversubscription.

Resource classes:

- `cpu-proof-translate`: AST and deterministic translation work;
- `cpu-proof-solver`: bounded ATP/SMT subprocesses;
- `cpu-proof-kernel`: Lean, Coq, or Isabelle checking;
- `cpu-validation`: tests and static checks;
- `llm-proof-draft`: Leanstral or other model proposals;
- `io-artifact`: cache, DuckDB, and optional IPFS operations.

Scheduling rules:

- one supervisor-level resource lease controls all child pools;
- model concurrency and CPU proof concurrency are accounted separately;
- independent proof-plan DAG nodes may execute in parallel;
- conflicting changed scopes are colored with the existing conflict graph;
- solver portfolios cancel remaining attempts after a conclusive result;
- kernel reconstruction is independently budgeted;
- cache hits bypass execution but still undergo freshness and trust checks;
- tasks with unsupported proof semantics may continue through configured test
  fallbacks in shadow mode;
- enforcement-mode protected tasks fail closed when required proof is absent;
- host CPU, memory, disk, provider quota, and latency backpressure remain
  authoritative.

## Leanstral Policy

Leanstral is a bounded, non-mutating proposal lane:

- invoke it through `llm_router`, never as an authority;
- provide a fixed theorem statement and bounded context capsule;
- accept only schema-valid proof text or decomposition suggestions;
- reject imports, axioms, `sorry`, `admit`, unsafe declarations, and theorem
  substitution;
- run generated proof text through the local trusted Lean checker;
- constrain patch proposals to task-declared paths and `git apply --check`;
- keep inference resource accounting separate from kernel checking;
- cache model artifacts as untrusted until independently checked;
- report missing spaCy/model/toolchain dependencies as degraded capability.

## ZKP Policy

ZKP is not a substitute for code correctness.

Phase 1 attaches an optional attestation envelope to an existing
kernel-verified receipt. Simulated ZKP may exercise serialization only and is
always labeled non-authoritative.

A cryptographic backend can enter enforcement only when:

- backend health is verified at runtime;
- circuit and public-input schema versions are pinned;
- verification key identity and lifecycle are recorded;
- golden vectors and negative vectors pass;
- witness no-leak tests pass;
- receipt, tree, obligation, policy, and kernel identities are public inputs;
- verification failure cannot fall back to simulated success.

## Rollout

### Phase 0: Contracts and capability truth

Implement schemas, capability probing, provider isolation, trust derivation,
and policy modes. No merge behavior changes.

### Phase 1: AST scopes, graph, and context capsules

Build obligations and queryable evidence from deterministic AST and existing
receipts. Measure prompt-size reduction in shadow mode.

### Phase 2: Hammer, cache, and CPU proof scheduler

Run proof plans in parallel under shared host budgets. Store all outcomes, but
do not block merges.

### Phase 3: Completion evidence and canary gates

Require proof receipts for a small set of supervisor invariants. Gate only
canary paths and preserve an explicit operator override receipt.

### Phase 4: Leanstral shadow assistance

Use `llm_router` to draft proofs and decompositions. Promotion still requires
kernel acceptance and deterministic validation.

### Phase 5: Optional cryptographic attestation

Attach production ZKP envelopes only after backend health and circuit tests
pass. Simulated backends remain test-only.

### Phase 6: Broader enforcement

Expand protected surfaces based on measured soundness, latency, cache hit
rate, mutation detection, and false-block rates.

## Metrics and Exit Criteria

Track by goal, task, tree, provider, obligation template, and resource class:

- obligations generated, supported, unsupported, proved, disproved, timed out;
- kernel reconstruction and rejection counts;
- cache hit, stale hit, and trust-rejection rates;
- solver, kernel, model, queue, and merge latency;
- CPU saturation, process count, memory, and cancellation savings;
- context capsule bytes/tokens versus unbounded baseline;
- model tokens and cost per accepted implementation;
- proof-gate blocks, overrides, reopen events, and later contradictions;
- ZKP backend health and verification failures.

Before enforcement expands, require:

- no path from LLM, solver candidate, stale cache, or simulated ZKP to verified;
- deterministic receipt reproduction for fixed trees and toolchains;
- adversarial cache, receipt, premise, and proof mutations are rejected;
- proof and test schedulers remain within shared host budgets;
- JSON and DuckDB proof projections are equivalent;
- context capsules remain within configured row, byte, token, and graph-hop
  limits;
- canary tasks demonstrate useful token reduction without unacceptable
  throughput regression.

## Implementation Backlog

The root refactor supervisor owns the executable backlog:

- Goal: `G11`
- Subgoals: `G11.S1` through `G11.S8`
- Tasks: `REF-244` through `REF-274`

The task graph deliberately exposes parallel branches after the contract
foundation:

- AST/knowledge-graph work;
- Hammer/cache/provider work;
- Leanstral shadow integration;
- ZKP attestation policy;
- resource and validation scheduling.

Merge and goal-completion enforcement remains downstream of those branches.
