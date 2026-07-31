# IPFS Kit VFS Symbolic Assurance and SwissKnife Contract Plan

## Outcome

Build a software-first assurance pipeline in
`ipfs_accelerate_py.agent_supervisor` that can:

1. freeze and exhaustively inventory the standalone SwissKnife checkout and
   the `ipfs_accelerate_py`, `ipfs_kit_py`, and `ipfs_datasets_py`
   repositories without treating sibling repositories as one Git tree;
2. incrementally index source, schemas, tests, documentation, MCP/MCP++
   registrations, and generated manifests into a content-addressed program
   graph;
3. extract expected contracts and observed call behavior, then issue bounded
   proof or counterexample receipts for contract matches, mismatches, and
   unsupported cases;
4. attest selected deterministic analysis traces with production-qualified
   zero-knowledge backends without claiming that a ZK receipt proves arbitrary
   program semantics;
5. convert fresh, deduplicated findings into a second, machine-readable repair
   taskboard; and
6. send Grok Build and Codex only small, CID-addressed repair packets instead
   of repository-scale prompts.

The native objective heap is
`docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md`. The
executable board is
`docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md`.

## Frozen initial scope

The launch configuration records, but does not mistake for immutable evidence,
the following observed checkouts:

| Repository | Initial checkout | Observed commit on 2026-07-29 |
| --- | --- | --- |
| SwissKnife | `/home/barberb/swissknife` | `df11f08fae17d35153e420fdcdc5b38d9f6b9a7f` |
| ipfs_accelerate_py | supervisor repository root | `ff401f83b7e722e58af1696243b3aff9679a7002` before this plan |
| ipfs_kit_py | `ipfs_kit_py` submodule | `f6a574375febbcf9a46fcd24bbc7bc5cfb551de5` |
| ipfs_datasets_py | `ipfs_datasets_py` submodule | `6672d69242731f53b49f4f793ed3023b7ba36a0d` |

Every real scan must derive a fresh repository descriptor containing the
resolved root, remote identity, commit, tree, recursive gitlink closure, dirty
state, dirty-diff digest when policy permits it, ignore policy, analyzer
versions, and filesystem case/Unicode policy. A descriptor forest, not a
common parent directory, is the unit of authority.

The external SwissKnife checkout is read-only for the initial tranche.
Code writes produced by the assurance pipeline target the accelerator checkout and its explicitly
configured submodule worktrees. Expanding write authority to another checkout
requires a separate repository descriptor, path allowlist, lease, and merge
policy.

## Existing components to reuse

The implementation must extend the established supervisor contracts rather
than create parallel parsers, caches, or proof authority:

- `analysis_ast_index.py`, `analysis_retrieval.py`,
  `analysis_contracts.py`, `analysis_pipeline.py`, and `analysis_cache.py`;
- `code_evidence_graph.py`, `semantic_dependency_graph.py`,
  `code_proof_obligations.py`, and `proof_scope_index.py`;
- `runtime_cas.py`, `artifact_store.py`, `cache_coordinator.py`, and
  `formal_verification_cache.py`;
- `ipfs_datasets_analysis_provider.py` and
  `ipfs_datasets_logic_provider.py`;
- `formal_logic_vocabulary.py`, `formal_verification_contracts.py`,
  `multi_prover_router.py`, `kernel_verification.py`, and
  `proof_attestation.py`;
- `adaptive_goal_refiner.py`, `backlog_refinery.py`,
  `task_proposal_router.py`, and `formal_replanner.py`;
- `ipfs_datasets_py.utils.cid_utils` for strict DAG-JSON, CIDv1,
  multihash, and CID validation;
- `ipfs_datasets_py` knowledge-graph/GraphRAG and logic provider surfaces
  after an explicit capability and conformance probe; and
- production-qualified implementations under
  `ipfs_datasets_py.logic.zkp`, with simulated or placeholder backends
  retained as test evidence only.

Import success, documentation, a class name, or a historical receipt is not a
capability proof. Optional providers remain lazy and advisory until a current
probe establishes their exact operations, versions, bounds, and authority.

## Claim taxonomy

Every finding and receipt carries exactly one claim level:

1. `observed_syntax`: a content-bound parser observed a definition, import,
   call expression, registration, schema, or test assertion.
2. `resolved_static`: a closed resolver linked a call site to one or more
   candidate targets and recorded ambiguity/unknown edges.
3. `model_proved` or `model_disproved`: a proof kernel established a theorem or
   counterexample inside a declared abstraction, assumptions, and scope.
4. `runtime_witnessed`: a hermetic contract test observed the declared
   behavior on a bound fixture and runtime.
5. `zk_trace_attested`: a qualified ZK verifier established that committed
   public inputs correspond to a valid supported analysis/proof trace.

No level implies a later level. In particular, `zk_trace_attested` proves the
configured circuit statement, not completeness of the source index, soundness
of an unverified translator, or all runtime behavior. A report may say
`contract_broken` only when a typed expected contract and a conclusive
counterexample or contradiction share the same repository, symbol, interface,
policy, and freshness binding. Otherwise it says `suspected`,
`ambiguous`, `unsupported`, `inconclusive`, or `stale`.

## Deterministic pipeline

```text
repository descriptor forest
  -> tracked-file inventory + explicit generated-source policy
  -> blob CIDs and language-neutral AST/symbol records
  -> import/call/type/effect/MCP registration edges
  -> GraphRAG/IPLD projection with provenance edges
  -> expected-contract IR + observed-behavior IR
  -> proof obligations and counterexample search
  -> proof/runtime/ZK receipts
  -> contract finding ledger
  -> compact repair taskboard and agent packet
```

Each stage accepts only bounded artifact references and emits a compact stage
receipt. Large source, AST, graph, proof, and witness bodies live in immutable
artifacts. Every result key includes:

- repository-forest identity and dirty-state policy;
- source blob or dependency CIDs;
- objective and policy revisions;
- analyzer, parser, schema, resolver, prover, circuit, and toolchain versions;
- configuration and query digests;
- capability snapshot;
- assumptions and contract-precedence policy; and
- bounds and redaction policy.

Success, partial, failed, timed-out, unsupported, inconclusive, and negative
entries remain distinct. Negative and inconclusive entries have bounded TTLs
and never satisfy completion. Changed leaves invalidate only transitive
dependents; policy, parser, resolver, contract, proof, or circuit changes
invalidate every affected authority-bearing projection.

## Source and program indexing

The scanner must be exhaustive relative to a published inventory policy, not
relative to whatever a recursive glob happened to see. It must:

- use Git object/tree enumeration for committed files and an explicit overlay
  for allowed dirty files;
- include TypeScript, TSX, JavaScript, Python, JSON/JSON Schema, Markdown
  contracts, package manifests, MCP IDL, and generated tool manifests;
- distinguish source, generated source, fixtures, vendored code, build output,
  archives, symlinks, submodules, and ignored paths;
- record skipped paths and reasons and fail completion on unexplained gaps;
- normalize paths without following an escaping symlink;
- reuse unchanged blob/AST records and expose scan coverage and truncation;
- represent dynamic import, re-export, monkey patching, dependency injection,
  decorators, RPC, subprocess, HTTP, libp2p, and MCP dispatch as typed
  ambiguous or external edges instead of fabricated direct calls; and
- retain symbol and edge provenance down to repository, blob CID, span,
  parser, and resolver rule.

GraphRAG is a bounded retrieval/indexing layer over this evidence. It may rank
relevant neighborhoods but may not create definitions, edges, contracts, or
proofs that were not emitted by an admitted producer.

## Contract model

Expected contracts are extracted under an explicit precedence policy from:

1. reviewed MCP++/MCP IDL, JSON Schema, typed interfaces, and protocol specs;
2. public signatures, type annotations, and stable exports;
3. executable contract and conformance tests;
4. normative documentation;
5. compatibility manifests and generated SDKs; and
6. implementation behavior as an observation, never as its own expectation.

Conflicting expectations are reported rather than silently resolved. Contract
IR covers inputs, outputs, errors, sync/async behavior, side effects,
capabilities, authorization, idempotence, ordering, atomicity, consistency,
resource bounds, and fallback/degradation behavior.

For MCP++, the evidence graph must trace:

```text
SwissKnife UI/service caller
  -> connector method
  -> transport and negotiated profile
  -> tools/list or declared interface
  -> tools/call name and schema
  -> registered server adapter
  -> real package implementation
  -> result/error mapping back to the caller
```

A same-named local helper, mock, fallback, static dashboard payload, or copied
manifest does not prove that MCP++ invokes the real implementation.

## VFS drift invariants

The first domain pack targets the competing VFS/fsspec/bucket/MCP facades in
`ipfs_kit_py`. It defines and checks at least:

- canonical path, Unicode, root, traversal, and mount resolution;
- read/write/open/close/seek/stat/list/mkdir/remove/rename/copy semantics;
- bytes versus text behavior and sync/async parity;
- stable exception and MCP error mappings;
- CID and size integrity of reads and writes;
- atomic mutation, journal/WAL replay, version/snapshot behavior, and crash
  recovery;
- cache invalidation, negative caching, pin metadata, and backend coherence;
- backend capability negotiation and explicit degraded behavior;
- authorization and path-scope preservation across Python, CLI, MCP, MCP++,
  HTTP, and libp2p surfaces;
- no silent mocks, placeholder success, swallowed errors, or shadow
  implementations; and
- generated manifest/SDK parity with the implementation actually registered
  by the live server.

The baseline inventory explicitly investigates `ipfs_fsspec.py`,
`enhanced_fsspec.py`, `vfs_manager.py`, `bucket_vfs_manager.py`,
`vfs_bucket_manager.py`, `mcp/ipfs_kit/vfs.py`, VFS handlers/endpoints/tools,
filesystem journals, version trackers, and files carrying `.fixed`, `.full`,
`.new`, `.clean`, `.optimized`, or `.broken` variants. Variant presence is
evidence of drift risk, not proof that a variant is wrong.

## Formal and zero-knowledge assurance

Contract IR is translated into a small, versioned logic vocabulary. Each
translation emits explicit unsupported semantics and a round-trip/conformance
receipt. Proof obligations are scoped to finite call paths and named
preconditions/postconditions/effects. Candidate solvers and premise selectors
may propose proofs; an admitted kernel or independently reconstructed runtime
witness establishes authority.

The ZK tranche begins in shadow mode. Its public inputs commit to the
repository forest, contract, normalized call slice, assumptions, analyzer and
resolver versions, proof result, and circuit/verifying-key identities. Private
witness material may include selected source/AST/proof-trace nodes when policy
requires redaction. The circuit proves only:

- commitments open to the supplied witness;
- the supported deterministic trace transition rules were followed; and
- the trace terminates in the committed result.

Simulated backends, placeholder field encodings, unversioned circuits,
unverified setup artifacts, missing ceremonies, or incompatible public-input
codecs cannot emit an authoritative ZK receipt.

## Finding and repair flow

The finding ledger is append-only and content identified. A repair candidate
contains:

- finding CID, claim level, severity, confidence, and freshness;
- exact repository/symbol/interface bindings;
- expected and observed contract CIDs;
- the shortest relevant call slice;
- counterexample/proof/runtime receipt references;
- precise allowed output paths and semantic effects;
- validation/proof commands;
- related and superseded finding IDs; and
- an explicit context budget.

Deterministic deduplication groups findings only when contract, root cause,
affected symbols, and merge fate agree. Vulnerability labels require a
security property, reachable or declared threat path, impact, and evidence;
ordinary correctness drift is not inflated into a vulnerability.

The default agent packet contains no repository dump and targets at most 16
KiB of canonical JSON plus bounded source spans. Grok Build and Codex may
request a content-addressed expansion one handle at a time. A retry receives
only changed evidence and the prior decision ID. Model output is a proposal;
the normal path, proof, validation, lease, and merge gates remain authoritative.

## Autonomous refill

The implementation supervisor owns one durable objective heap and taskboard.
It refills only when open work falls below the configured threshold:

1. ingest fresh analyzer receipts;
2. reject stale, duplicate, ambiguous, out-of-scope, or unbound findings;
3. append bounded child goals when a new failure family is not covered by an
   existing goal;
4. materialize bounded tasks with exact goal lineage and dependencies;
5. preserve a stable semantic key across repeated observations;
6. back off unchanged failures and escalate exhausted retries to a distinct
   repair/unblock task; and
7. stop creating work when the inventory is conclusively exhausted.

Initial ceilings are three child goals per refinement, depth four, eight
objective findings per pass, two surplus candidates per goal, and a refill
threshold of four open tasks. These are admission ceilings, not throughput
targets.

## Rollout gates

1. **Foundation:** repository-forest identity, claim contracts, capability
   matrix, and cache/CID conformance.
2. **Index:** exhaustive SwissKnife inventory and incremental AST/program graph
   with zero unexplained skips.
3. **Contracts:** deterministic expected/observed IR, MCP++ path resolution,
   and seeded mismatch precision.
4. **Proof:** supported logic translations, independent proof validation, and
   explicit unknown outcomes.
5. **ZK shadow:** production-backend capability, circuit conformance, verifier
   replay, and no authority from simulated proofs.
6. **VFS pilot:** frozen VFS corpus, differential/runtime witnesses, and a
   reviewed baseline finding ledger.
7. **Autonomous repair:** compact packets, provider sharding, validation,
   serialized merge, bounded refill, and rollback.

Promotion requires reproducible CIDs across clean machines, complete inventory
receipts, zero stale authoritative cache hits, no false `proved` claims in
adversarial fixtures, deterministic task identities, exact Python/CLI/MCP
report parity, and a measured reduction in model input without reduced finding
coverage. Any binding drift, capability loss, contradictory proof, index
coverage regression, or resource-policy violation returns the affected stage
to shadow and prevents new automatic repair admission.

## Assurance generalization cutover

Seven root-level VFS assurance modules were extracted into profile-driven
semantic packages so the same engines serve the IPFS Kit VFS job and hermetic
non-VFS programs. Source blobs remain pinned by
`config/agent_supervisor_vfs_generalization_sources.lock.json` (revision
`0cc04ebb640c4c981cf4650016e096a73ab0e8c0`); workers may read those blobs but
must not merge or cherry-pick the broad source snapshot.

| Historical root module | Generic destination |
| --- | --- |
| `vfs_surface_inventory.py` | `analysis/repository_surface_inventory.py` |
| `vfs_contract_pack.py` | `analysis/program_contract_profile.py` |
| `vfs_differential_harness.py` | `validation/differential_contract_harness.py` |
| `vfs_mcp_contract_checker.py` | `analysis/interface_contract_parity.py` |
| `vfs_symbolic_benchmark.py` | `validation/symbolic_efficiency_benchmark.py` |
| `vfs_symbolic_pilot.py` | `runtime/symbolic_assurance_pilot.py` |
| `vfs_symbolic_rollout.py` | `control/symbolic_assurance_rollout.py` |

The IPFS Kit job is assembled only by
`integrations/ipfs_kit_vfs_assurance.py` from
`config/ipfs_kit_vfs_symbolic_assurance.json`. The sole executable facade is
`scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py` with
subcommands `inventory`, `contracts`, `differential`, `parity`, `benchmark`,
`pilot`, `rollout`, and `verify`. It resolves the checkout, validates the
profile, lazy-loads the integration, and delegates; it owns no scan, proof,
comparison, gate, repair, or mutation logic.

After cutover:

- `ipfs_accelerate_py/agent_supervisor/` must contain **no** root `vfs_*.py`
  implementation or compatibility shim;
- no import may reference `agent_supervisor.vfs_*`;
- generic engines contain no VFS/IPFS/fsspec/SwissKnife constants, board IDs,
  or fixed-checkout branches;
- open board outputs and documentation links point at the package destinations
  above (plus the thin ops facade and locked config);
- equivalence is proved by profile-driven public contract parity, caller
  impact closure, and a second non-VFS profile traversing the same engines—not
  by renaming alone.

Placement and equivalence guards:

- `test/api/test_agent_supervisor_vfs_generalization_equivalence.py`
- `test/api/test_agent_supervisor_vfs_root_layout_guard.py`
- `test/api/test_agent_supervisor_assurance_two_profile_end_to_end.py`

See also
`docs/architecture/agent_supervisor/VFS_ASSURANCE_GENERALIZATION_MAP.md`.

## Operating profile

The checked-in control script launches two deterministic task shards:

- shard 0 uses the authenticated Grok Build CLI and owns bounded objective and
  static-refill control; and
- shard 1 uses Codex and consumes the same protected taskboard without
  independently mutating the objective heap.

Both use isolated worktrees and state, a shared serialized merge queue,
explicit submodule paths, protected plan/objective/taskboard files, bounded
timeouts, retry guardrails, and current supervisor event/status records.
Use:

```bash
scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh start
scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh status
```

The launch is deliberately implementation-authorized only inside the
accelerator checkout. External repository mutation remains disabled until the
multi-repository authority task is implemented and separately reviewed.
