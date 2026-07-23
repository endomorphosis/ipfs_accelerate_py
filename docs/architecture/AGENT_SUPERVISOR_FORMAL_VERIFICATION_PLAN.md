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

## REF-244 Capability Discovery Implementation

Status: capability discovery implemented

Capability schema: `ipfs_accelerate_py/agent-supervisor/formal-verification-capabilities@1`

Report version: `1`

### Purpose

The agent supervisor needs to route a proof obligation only to toolchains that
can plausibly service it. Runtime discovery must not confuse “installed” with
“proved,” silently activate simulated cryptography, download optional
dependencies, or make the supervisor package unimportable on a minimal host.

`agent_supervisor.formal_verification_capabilities` supplies that discovery
boundary. It produces a versioned, immutable snapshot for scheduling and
operator diagnostics. The report is routing evidence only. A later proof
attempt must produce its own obligation-bound receipt and independently
verified result.

### Safety invariants

1. A capability probe never imports a provider module, loads model weights,
   invokes a prover, compiles a circuit, downloads an artifact, or generates or
   verifies a proof.
2. `proof_attempted` and `proof_success` are always `false` in a capability
   report, every provider row, and every dependency check.
3. The simulated ZKP backend is always `degraded` and explicitly identified as
   non-cryptographic. It cannot satisfy a cryptographic or attested assurance
   requirement.
4. A real ZKP backend is available only when its executable and its circuit/key
   artifacts are both discoverable.
5. Leanstral output is a candidate. It needs a separate Lean kernel check
   before any later receipt can treat it as checked.
6. Missing optional dependencies are data, not import failures. Each missing
   package, executable, model, or circuit has an explicit reason.
7. Discovery is bounded by both a maximum check count and a wall-clock budget.
   A stalled third-party discovery hook is isolated on a daemon thread and
   cannot block supervisor startup.
8. Reports have a TTL cache. Operators can force a refresh after installing a
   toolchain or clear the cache explicitly.

### Capability matrix

The report always includes these provider families:

| Provider ID | Runtime surfaces | Readiness rule |
|---|---|---|
| `hammer` | Hammer package, built-in graph selector, E, Vampire, Z3, CVC5 | Package plus at least one ATP/SMT executable is available; package alone is degraded |
| `tdfol` | TDFOL package, spaCy, configured spaCy model | Symbolic core can operate without NLP extras, but missing NLP health is reported as degraded |
| `external_provers` | Bridge package, Z3/CVC5 Python bindings, ATP/SMT/Coq executables, SymbolicAI | Bridge plus at least one solver binding or executable is available |
| `lean` | Lean bridge and `lean`/`lake` executable | Both are required for a usable Lean route |
| `leanstral` | Leanstral integration, local weights, Transformers, PyTorch, Lean | Missing local inference or checker dependencies degrades proposal generation/checking; it never implies proof |
| `frame_logic` | F-logic package and ErgoAI executable | Structural in-memory mode is degraded; executable-backed ErgoAI is available |
| `knowledge_graphs` | Knowledge-graph package, spaCy/model, Transformers, NetworkX | Core remains usable with rule-based fallbacks; missing NLP enhancements are degraded |
| `zkp_backends` | Registry, circuit definitions, simulated/Groth16/ProveKit providers, executables, setup artifacts, py-ecc | A real backend needs executable plus artifacts; simulation alone is degraded |

The external-prover row intentionally reports Python bindings separately from
command-line executables. Installing `z3-solver` does not imply a `z3` command
is on `PATH`, and finding a command does not imply that a Python bridge can
import its bindings.

### Report contract

The top-level JSON representation has stable routing fields:

```json
{
  "schema_version": "ipfs_accelerate_py/agent-supervisor/formal-verification-capabilities@1",
  "report_version": 1,
  "generated_at": "2026-07-22T12:00:00Z",
  "duration_seconds": 0.012,
  "probe_count": 39,
  "bounded": true,
  "cache_ttl_seconds": 300.0,
  "overall_status": "degraded",
  "proof_attempted": false,
  "proof_success": false,
  "providers": {
    "lean": {
      "provider_id": "lean",
      "display_name": "Lean 4",
      "status": "unavailable",
      "reason": "Lean prover executable is not installed or configured; no Lean kernel check can run",
      "health": {
        "provider": [],
        "executable": [],
        "package": [],
        "model": [],
        "circuit": [],
        "optional_dependency": []
      },
      "proof_attempted": false,
      "proof_success": false
    }
  }
}
```

The arrays in the abbreviated example contain full
`CapabilityHealthCheck` records in a real report. All six dimensions are
always present even when a provider has no dependency in one dimension:

- `provider`: aggregate provider/backend readiness;
- `executable`: command discovery or an explicit executable override;
- `package`: importable provider modules and required Python bindings;
- `model`: local spaCy or Leanstral weight discovery;
- `circuit`: circuit definitions and configured setup/key artifacts;
- `optional_dependency`: enhancements such as spaCy, Transformers, PyTorch,
  NetworkX, SymbolicAI, and py-ecc.

Checks include status, reason, requirement flag, optional version and location,
metadata, and the invariant false proof fields. `available` means discoverable
for a later attempt; `degraded` means a safe fallback or partial surface
exists; `unavailable` means the named route cannot currently be used;
`disabled` is reserved for an explicitly disabled but installed integration.

### Import and probe boundary

The module depends only on the Python standard library. Package discovery
traverses `importlib.machinery.PathFinder` specs; unlike a dotted-name
`importlib.util.find_spec` call, it does not execute parent package
initializers or import the discovered module.
Distribution versions use `importlib.metadata`. Executable discovery uses
`shutil.which` or a validated explicit path. Artifact checks are bounded
existence checks and never parse secrets, model weights, proving keys, or
verification keys.

The default limits are:

- 2 seconds total wall-clock time;
- 96 dependency checks;
- 300 seconds cache TTL.

`FormalVerificationCapabilityProbe` accepts injected package, executable,
version, environment, and clock functions. This makes deployment matrices
deterministic in tests and lets an embedding application use its own metadata
service without weakening the report contract.

The process-wide entry point is:

```python
from ipfs_accelerate_py.agent_supervisor import (
    probe_formal_verification_capabilities,
)

report = probe_formal_verification_capabilities()
lean = report.provider("lean")
if lean.available:
    # Eligible to plan a proof attempt. No proof has happened yet.
    schedule_lean_attempt()
```

Use `force_refresh=True` after a known environment change, or call
`clear_formal_verification_capability_cache()`. A custom probe instance keeps
an independent cache and supports a deployment-specific
`FormalVerificationProbeConfig`.

### Degradation behavior

The following failures are deliberately non-fatal:

- spaCy absent: TDFOL and knowledge graphs report rule-based/degraded NLP;
- `en_core_web_sm` absent: model health names the missing model weights;
- Z3 or CVC5 bindings absent: binding package health is unavailable even if a
  similarly named executable exists;
- Lean, E, Vampire, Coq, CVC5, Z3, ErgoAI, Groth16, or ProveKit commands absent:
  each executable has its own reason;
- Leanstral weights absent: local model health is unavailable while the
  orchestration package can remain degraded for an injected managed callback;
- Groth16/ProveKit artifacts absent: real backend health remains unavailable;
- provider import hooks raise: the exception type and message become a safe
  unavailable reason;
- discovery exceeds its check/time budget: remaining checks say they were not
  inspected and carry `probe_limited: true`.

None of these cases escapes from capability discovery or prevents importing
`ipfs_accelerate_py.agent_supervisor`.

### Proof-aware scheduling boundary

Capability discovery answers only:

> Is there enough local runtime surface to schedule an attempt on this route?

It does not answer:

- whether a translation preserves the obligation;
- whether premises are complete or trusted;
- whether the solver returned valid/invalid/unknown;
- whether a candidate proof reconstructs in a trusted kernel;
- whether a ZKP binds the expected circuit, statement, witness commitment, and
  verification key;
- whether cached evidence matches the current repository tree and policy.

The scheduler must therefore apply two gates:

1. **Route gate:** use this capability report to reject unavailable routes and
   surface degraded routes to policy.
2. **Evidence gate:** after execution, accept only a versioned proof receipt
   whose obligation, source tree, translator, solver/kernel, policy, and
   resource limits are bound and independently validated.

Availability must never be copied into an evidence or assurance field.

### Operational evolution

Schema changes are additive within report version 1. Removing or changing the
meaning of a field, provider ID, health state, or safety invariant requires a
new schema/report version. Provider version commands or active canary proofs
belong in a separate, opt-in diagnostic lane because executing them has a
larger failure and resource surface than startup discovery.

Future proof contracts and receipts should consume `provider_id` plus the
specific dependency check identities used for routing. They should record a
digest of the capability snapshot as contextual evidence, never as proof
evidence. Scheduling integration should preserve a cached snapshot per worker
environment and refresh it after toolchain installation, worker replacement,
or TTL expiry.

### Validation

The focused API suite covers:

- the complete provider matrix and version fields;
- independent serialization of all health dimensions;
- explicit missing spaCy, model, binding, executable, and artifact reasons;
- safe failure of import hooks;
- partial-provider degradation;
- TTL caching, forced refresh, and cache clearing;
- maximum-check and hard wall-clock bounds;
- ZKP executable-plus-circuit readiness;
- the invariant that discovery never reports proof success.

Run it with:

```bash
PYTHONPATH=ipfs_datasets_py/ipfs_accelerate_py \
python -m pytest \
  ipfs_datasets_py/ipfs_accelerate_py/test/api/test_agent_supervisor_formal_verification_capabilities.py \
  -q
```
