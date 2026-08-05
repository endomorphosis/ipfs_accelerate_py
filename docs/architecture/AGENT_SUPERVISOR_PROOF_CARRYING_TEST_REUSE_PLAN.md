# Agent Supervisor Proof-Carrying Test Reuse Plan

**Program:** `PTR`
**Status:** proposed; implementation disabled by default
**Objective heap:** `agent_supervisor_proof_carrying_test_reuse.objectives.md`
**Taskboard:** `agent_supervisor_proof_carrying_test_reuse.todo.md`
**Task prefix:** `PTR-`
**Namespace:** `agent-supervisor-proof-carrying-test-reuse-v1`

## Outcome

Extend `ipfs_accelerate_py.agent_supervisor` so it can reuse a prior successful
validation without executing the command again when the supervisor can prove
that every input capable of affecting that validation is unchanged.

The user-visible result is **not a fabricated new pass**. It is a typed
`proved_reuse_pass` disposition that:

1. references a prior authoritative, successful test receipt;
2. binds the current repository forest, test selection, semantic dependency
   slice, fixtures, configuration, runtime, toolchain, policy, and capabilities;
3. proves equality of the prior and current validation-input roots;
4. re-verifies all receipts and proofs at the point of use; and
5. is accepted as pass-equivalent only by an explicit, versioned reuse policy.

An incomplete graph, ambiguous dynamic behavior, stale receipt, changed input,
unsupported language, unavailable verifier, or capability loss causes normal
test execution. It never causes an optimistic pass.

## Why this is an extension, not a new subsystem

The repository already contains most required primitives:

| Existing surface | Reuse |
| --- | --- |
| `analysis.analysis_ast_index` | Incremental, current-snapshot AST index |
| `program_ast_adapters` | Content-bound Python/JS/TS/JSON/Markdown evidence |
| `program_graph` / `program_call_resolver` | Conservative typed call edges and unknown frontiers |
| `analysis.code_evidence_graph.CodeImpactIndex` | Direct and transitive impact selection |
| `repository_forest` | Commit, tree, gitlink, dirty-overlay, and policy identity |
| `multiformats_identity` | Strict CIDv1, DAG-JSON/raw, sha2-256 identity |
| `program_analysis_cache` | Dependency-aware cache and authority namespaces |
| `validation.validation_scheduler` | Exact successful-result cache, impact DAG, hermetic runtime |
| `program_analysis_zkp` | Public commitments, private witness policy, verifier conformance |
| `runtime_cas` / cache coordinator | Immutable artifacts, invalidation, quotas, single-flight |

The current validation cache is intentionally exact: it binds the whole target
commit and candidate dependency state. That safely reuses an identical
candidate, but a commit change invalidates the result even when the test's
complete semantic slice is unchanged. PTR adds a narrower, independently
verified **validation-input root** and a proof-carrying reuse decision. It does
not weaken or replace the exact cache.

## Non-negotiable correctness rules

1. **No hash-only shortcut.** Equality of one source file or function is not
   enough. The complete admitted test-input closure must match.
2. **No AST-completeness assumption.** A CID proves identity of the encoded
   object, not that the parser or resolver found every relevant dependency.
3. **No ZK semantic overclaim.** A ZK proof may prove commitment openings,
   membership, equality, and supported trace transitions. It does not prove
   Python semantics, call-graph completeness, or that a historical test truly
   passed.
4. **Prior execution remains the root observation.** Reuse starts from a
   trusted successful execution receipt produced by the existing hermetic
   validation path.
5. **Fail closed.** Unknown, partial, truncated, ambiguous, stale, corrupt,
   untrusted, unsupported, or unavailable means execute.
6. **No trust upgrade on cache hit.** Lookups re-derive authority from typed
   evidence. Serialized `passed`, `verified`, or `authoritative` flags are not
   trusted.
7. **No silent result conflation.** Executed, exact-cache, and proved-reuse
   passes remain distinguishable in receipts, metrics, CLI output, and audits.
8. **Broad and release tests stay policy-controlled.** A project may require
   periodic execution even when reuse is sound, particularly for flaky,
   timing-sensitive, hardware, network, security, release, and soak tests.
9. **TOCTOU is closed.** The observed tree and input root must be rechecked
   immediately before admitting reuse and again before merge/completion.
10. **Rollback is instant.** One policy switch disables reuse without disabling
    ordinary validation or deleting evidence.

## Claim and outcome taxonomy

Add a typed validation disposition rather than overloading `cache_hit`:

| Disposition | Meaning | Command executed now |
| --- | --- | --- |
| `executed_pass` | Current hermetic command passed | yes |
| `exact_cache_pass` | Existing exact validation key matched | no |
| `proved_reuse_pass` | Cross-tree validation-input equality was independently verified | no |
| `reuse_shadow_match` | Reuse predicted pass and shadow execution passed | yes |
| `reuse_shadow_mismatch` | Reuse predicted pass but execution did not pass | yes; blocks promotion |
| `reuse_ineligible` | Policy or evidence requires execution | yes |
| existing failure outcomes | Deterministic/flaky/timeout/infrastructure/inconclusive/cancelled | yes |

`proved_reuse_pass` may satisfy a selected validation node only when the node's
reuse policy allows it and the receipt contains the full current-tree binding.
It must not be rewritten to look like a freshly executed process.

## Identity hierarchy

All portable identities use the existing frozen multiformats profile:
CIDv1, lowercase base32, `raw` or canonical `dag-json`, `sha2-256`, 32-byte
digest. Existing local IDs remain linked through `IdentityLink`.

### 1. Blob identity

`SourceBlobIdentity@1` binds exact bytes, executable mode, repository identity,
canonical path policy, and generated/vendored classification. Raw-byte CID is
the leaf identity. Path is a binding, not part of the reusable byte identity.

### 2. AST identity

The existing `ASTBlobRecord` remains canonical for file-level AST evidence.
Add typed sidecar identities:

- `ASTModuleIdentity@1`: parser/version + complete canonical module AST;
- `ASTSymbolIdentity@1`: function, async function, method, class, module
  initializer, schema declaration, or test node;
- `ASTContextIdentity@1`: defaults, annotations, decorators, closure/free
  variables, class bases/MRO/metaclass, relevant module globals, imports,
  registrations, and module-initialization effects;
- `ASTEdgeIdentity@1`: source symbol, target/frontier, edge kind, resolver rule,
  confidence/status, and source span.

Source coordinates may be excluded from semantic equality only when a separate
exact source CID and parser-version binding remains present. Docstrings,
constants, decorators, defaults, annotations, class construction, and module
initializers are semantic inputs unless a reviewed language policy proves
otherwise.

### 3. Semantic call-slice identity

`SemanticSliceManifest@1` is a canonical IPLD DAG containing:

- the selected test nodes;
- all transitively reachable admitted code symbols;
- import/re-export edges;
- fixtures, autouse fixtures, `conftest.py`, test plugins, parametrization,
  marks, collection hooks, and setup/teardown;
- schemas, configuration, generated code, data fixtures, resources, and
  subprocess/MCP/HTTP/IPFS boundaries declared by policy;
- environment, platform, interpreter, dependency lock, installed distribution,
  native-extension, and tool identities;
- explicit unknown and dynamic frontiers;
- completeness/truncation/coverage receipts; and
- analyzer, resolver, schema, and policy versions.

The manifest root CID is the semantic-slice root. A Merkle proof can show that
the same leaves and edges occur in two current-tree projections. It cannot
establish that omitted leaves never mattered; that is the role of the
completeness and policy gates.

### 4. Test-input identity

`ValidationInputManifest@1` binds:

- normalized validation command and selected test collection;
- semantic-slice root;
- test/fixture/plugin collection root;
- repository-forest and current dirty-overlay observation;
- hermetic runtime ID and launcher/interpreter receipts;
- environment allowlist, dependency manifests, lockfiles, gitlinks, and
  installed-package inventory;
- filesystem/network/time/randomness policy;
- validation, reuse, parser, resolver, and analyzer policy revisions;
- acceptance criteria and selected impact-DAG node;
- capabilities and hardware/backend profile; and
- a nonce only when the policy intentionally forbids reuse.

Its CID is the **validation-input root**. Cross-tree reuse requires exact prior
and current root equality; comparing hand-selected fields is insufficient.

### 5. Prior pass and reuse proof

`ReusableValidationReceipt@1` wraps an existing successful validation receipt
and binds its validation-input root, result digest, execution time, stability
runs, producer, authority, expiration, and revocation domain.

`ValidationReuseProof@1` binds:

- prior and current repository/tree observations;
- equal prior/current validation-input root;
- prior pass receipt CID and successful-result digest;
- AST/call/test manifest roots and completeness receipts;
- proof method (`direct`, `merkle`, or `zk`);
- verifier, circuit/key/ceremony identifiers when applicable;
- policy and capability revisions;
- freshness, revocation, and TOCTOU checks; and
- a deterministic reason code for every accepted or rejected condition.

`ValidationReuseDecision@1` is the only object the scheduler may consume.

## What belongs in the dependency closure

The closure must include more than callable AST nodes.

### Always include

- selected tests and their collection metadata;
- imports, re-exports, module initializers, global reads/writes, decorators,
  defaults, annotations, closures, class construction, inheritance, and
  metaclasses;
- direct and transitive calls with resolver provenance;
- `conftest.py`, fixtures, plugins, hooks, parametrization, marks, and test
  configuration;
- referenced files, schemas, templates, snapshots, golden data, migrations,
  subprocess commands, and generated artifacts;
- dependency manifests, lockfiles, gitlinks, interpreter, launcher, installed
  wheels/distributions, native libraries, and relevant environment;
- feature flags, backend/capability selection, locale/timezone, and hardware
  profile where observable; and
- validation selection, acceptance criteria, and policy.

### Force execution unless a reviewed adapter closes the frontier

- `eval`, `exec`, dynamic source generation, unrestricted reflection, unknown
  `getattr`, monkey patching, import hooks, namespace-package ambiguity;
- unpinned network, clock, randomness, process state, external service, GPU,
  device, kernel, filesystem, or database dependencies;
- unresolved callback/DI targets, RPC/MCP dispatch, subprocesses, FFI/native
  extensions, or generated code;
- test-order dependence, shared mutable state, flaky/quarantined tests, soak,
  performance, fuzz, mutation, security, release, and deployment gates;
- parser failures, unsupported languages, inventory gaps, truncated graphs,
  missing submodules, dirty paths outside the allowed overlay, or a changed
  symlink/case/Unicode policy.

Adapters may later convert a frontier into a closed, content-bound dependency.
They may not simply mark it safe.

## Reuse decision algorithm

```text
selected validation node
  │
  ├─ exact validation cache hit?
  │      └─ verify existing exact receipt → exact_cache_pass
  │
  └─ build current ValidationInputManifest
         │
         ├─ incomplete/ambiguous/ineligible? → execute
         ├─ no prior reusable pass?          → execute
         ├─ prior receipt stale/revoked?     → execute
         ├─ input root differs?              → execute
         └─ verify equality/reuse proof
                ├─ rejected/unavailable      → execute
                └─ accepted
                     ├─ shadow policy        → predict, then execute
                     └─ enforce policy       → proved_reuse_pass
```

Before merge/completion, rebuild or replay the current manifest from the
candidate tree and require the same input root. A changed tree may still reuse
a result, but only because its admitted validation-input root is unchanged.

## Direct, Merkle, and zero-knowledge proof modes

### Direct equality

Inside one trusted supervisor process, compare canonical manifest CIDs and
re-verify the prior receipt. This is the simplest and preferred mode.

### Merkle/IPLD proof

For distributed caches, transfer the compact manifests, prior receipt, and
Merkle inclusion/equality proof. Verify content locally against the current
forest. IPFS availability is transport, not authority.

### Zero knowledge

Use ZK only when a reviewed use case needs to hide source, dependency names,
test inventory, or other witness material from a verifier across a trust
boundary. Extend the existing `program_analysis_zkp` public-input contract; do
not create a second ZK authority.

The circuit may prove:

1. prior/current manifest commitments open correctly;
2. both use the admitted schema, policy, analyzer, resolver, and toolchain
   commitments;
3. required Merkle memberships and closure transitions are satisfied;
4. prior and current validation-input roots are equal;
5. the prior receipt commitment is the one referenced publicly; and
6. the supported trace terminates in `reuse_eligible`.

The circuit does **not** prove:

- that the original test runner honestly executed or passed;
- that the AST/call graph is complete or semantically sound;
- arbitrary Python/runtime equivalence;
- absence of undeclared external state; or
- correctness beyond the prior test observation.

Those claims remain bound to the original trusted execution receipt,
independent inventory/completeness policy, and verifier. Simulated ZK remains
shadow-only. Production ZK requires the existing capability, ceremony, key,
codec, independent-verifier, and approved-use-case gates.

## Cache and storage design

Keep two explicit tiers:

1. **Exact execution cache** — existing `ValidationResultCache`, keyed by full
   target/candidate/runtime inputs.
2. **Proof-carrying reuse index** — maps a validation-input root to immutable
   prior execution and reuse-proof receipts.

The reuse index uses the existing namespace cache coordinator, authority
namespaces, quotas, atomic writes, corruption repair, negative TTLs,
single-flight, and RuntimeCAS/artifact-store boundaries. Large ASTs, manifests,
proofs, witnesses, and outputs stay in bounded CID-addressed artifacts.

Rules:

- only successful, stable, non-timeout, non-flaky executions are reusable;
- negative/inconclusive entries never satisfy validation;
- revocation and policy changes invalidate transitive dependents;
- cache records contain public receipts and references, never private witness;
- IPFS/P2P replicas are untrusted until local verification;
- optional UCAN scope controls who may publish/read receipts, but possession of
  a UCAN does not make a receipt correct;
- GC retains any artifact reachable from an unexpired authoritative receipt;
  and
- single-flight covers manifest construction, proof generation, and execution.

## Scheduler and daemon integration

Extend `ValidationScheduler` with a `ValidationReusePolicy`:

- `off` — current behavior;
- `shadow` — compute reuse decision, execute every command, compare;
- `advisory` — expose eligibility but always execute;
- `enforce_low_risk` — reuse only deterministic targeted/unit/contract nodes
  explicitly allowed by repository policy;
- `enforce_reviewed` — reuse all individually approved classes, preserving
  periodic-execution requirements.

Add CLI/config controls for mode, receipt age, mandatory rerun interval,
allowed validation kinds, maximum closure size, proof mode, verifier policy,
and per-test opt-out. No environment variable alone may enable production
reuse; a reviewed policy file and digest are required.

The daemon records:

- eligibility and rejection reason codes;
- prior/current roots and receipt CIDs;
- proof/verifier mode and latency;
- execution saved or shadow execution duration;
- TOCTOU recheck result;
- policy/capability identity;
- exact-cache versus proof-reuse counts; and
- merge/completion authority disposition.

## Security and failure model

Required adversarial coverage:

| Threat | Required response |
| --- | --- |
| Poisoned prior receipt | Re-verification rejects; execute |
| Hash/CID profile confusion or double hashing | Existing strict CID bridge rejects |
| Forged manifest omits a dependency | Completeness/coverage mismatch; execute |
| Changed `conftest.py`, plugin, parametrization, or data | Input root changes |
| Changed lockfile, wheel, native library, interpreter, launcher | Input root changes |
| Dynamic import/monkey patch/reflection | Unknown frontier; execute |
| Renamed/moved identical function | Reuse only if path/name bindings are irrelevant under reviewed policy |
| Changed module initializer/global/decorator/default | Context root changes |
| Test added or collection changed | Test collection root changes |
| Flaky/time/network/hardware test | Policy forces execution |
| Stale/expired/revoked receipt or key | Execute |
| Simulated ZK or verifier fallback | Non-authoritative; execute |
| IPFS/P2P cache poisoning | Local CID, schema, authority, and receipt verification |
| Concurrent tree mutation | Snapshot/TOCTOU mismatch; execute or abort |
| Partial inventory/parser truncation | Completeness false; execute |
| Test command broadens selection | Command/collection root changes |

The critical release invariant is:

> There must be zero cases where PTR reuses a pass and executing the same
> validation under the bound current runtime would produce a non-pass.

Shadow mismatches are severity-one correctness defects and immediately disable
enforcement for the affected policy/repository/test class.

## Test strategy

### Contract and unit tests

- canonical serialization and CID reproducibility;
- symbol/context/edge identities;
- semantic-slice construction and minimality;
- test/fixture/plugin/data/runtime manifest construction;
- proof and receipt round trips;
- stale, corrupt, foreign-authority, and capability-loss rejection;
- direct/Merkle verification;
- private-witness no-leak and simulated-ZK non-authority;
- scheduler outcomes, DAG barriers, completion gates, and rollback.

### Adversarial matrix

Seed one change at a time across function body, signature, decorator, default,
global, import, caller, transitive callee, fixture, autouse fixture,
`conftest.py`, plugin, parameter, data file, lockfile, submodule, environment,
interpreter, native extension, generated code, dynamic import, monkey patch,
clock, random seed, network response, filesystem, test collection, policy, and
verifying key. Each seed must either change the input root or force execution.

### Differential shadow oracle

For every predicted reuse in shadow mode, execute the validation and compare
the real outcome. Store no output secrets. Promotion requires:

- zero false reuse predictions;
- zero stale authoritative hits;
- deterministic roots across clean machines;
- identical decisions across process restart and cache replication;
- explicit handling of every unknown frontier; and
- no reduction in mandatory impact-DAG or acceptance coverage.

### Mutation testing

Mutate call-graph edges, manifests, receipts, cache entries, proof bytes,
environment records, collection manifests, and completeness flags. At least
one independent verifier must reject every authority-relevant mutation.

## Performance and load plan

Benchmark profiles:

1. exact unchanged candidate;
2. documentation-only unrelated change;
3. unrelated source-module change;
4. unchanged leaf test closure across a large merge;
5. changed leaf implementation;
6. changed public interface or shared dependency;
7. fixture/plugin/config/lockfile/environment drift;
8. dynamic/unsupported frontier fallback;
9. cold cache, warm local cache, warm IPFS/P2P replica;
10. 1/4/16/64 parallel supervisors contending on the same roots.

Measure:

- index and manifest build p50/p95/p99;
- direct/Merkle/ZK proof and verification latency;
- cache hit/reject/fallback rates by reason;
- tests and wall-clock seconds saved;
- CPU, RSS, disk, network, artifact bytes, and GC time;
- single-flight collapse ratio;
- shadow mismatch count;
- reuse precision (must be 1.0 for promotion);
- reuse coverage and mandatory-rerun frequency; and
- merge/completion latency.

Initial performance targets, subject to baseline ratification:

- zero false reuse;
- at least 50% wall-clock reduction for admitted unrelated-change profiles;
- p95 direct/Merkle decision under 250 ms for warm bounded slices;
- incremental index overhead below 10% of the avoided test time;
- no more than one producer for concurrent identical manifest/proof work; and
- bounded cache growth with successful quota/GC recovery.

Performance never overrides correctness. Missing a reuse opportunity is safe;
reusing an invalid pass is not.

## Rollout and rollback

1. **Contracts only:** schemas, identities, reason codes, and policy; no reuse.
2. **Offline fixtures:** direct and Merkle proofs over synthetic repositories.
3. **Shadow:** predict and execute every selected validation.
4. **Canary:** low-risk deterministic tests in an allowlisted repository.
5. **Bounded enforcement:** targeted/unit/contract tests with periodic reruns.
6. **Expanded reviewed classes:** only after class-specific shadow evidence.
7. **Optional ZK:** separate privacy/trust-boundary approval; never required
   for ordinary local reuse.

Rollback is one policy change to `off`, followed by ordinary validation. Cached
receipts remain audit evidence but cannot satisfy new decisions while disabled.
Any mismatch, authority violation, verifier disagreement, unexplained
incompleteness, or stale hit automatically returns the affected scope to
shadow/off.

## Definition of done

PTR is production-ready only when:

- the objective and task boards are complete and machine-ingestible;
- all identity, manifest, proof, receipt, and policy schemas are versioned;
- every admitted input class has a mutation/invalidation test;
- unknown/dynamic behavior demonstrably fails closed;
- prior passes are independently re-verified and current-tree rebound;
- direct and Merkle modes work without ZK;
- ZK remains optional and cannot exceed its reviewed claim;
- scheduler, daemon, status, CLI, and audit surfaces distinguish reuse;
- shadow testing shows zero false reuse across the ratified corpus and load;
- rollback and cache-corruption recovery are exercised; and
- merge/completion authority rejects stale, incomplete, foreign, simulated, or
  tampered evidence.
