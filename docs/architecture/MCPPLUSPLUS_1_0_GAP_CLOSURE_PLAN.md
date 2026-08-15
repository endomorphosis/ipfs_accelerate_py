# MCP++ 1.0 Gap-Closure Plan

**Status:** Sealed first projection  
**Program prefix:** `MCPP-` / `MCPP-G`  
**Board namespace:** `mcplusplus-1-0-gap-closure-v1`  
**Merge target:** `codex/mcplusplus-1.0-gap-closure`  
**Companion artifacts:**

- objective heap: `docs/architecture/mcplusplus_1_0_gap_closure.objectives.md`
- taskboard: `docs/architecture/mcplusplus_1_0_gap_closure.todo.md`
- scheduler: `config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json`
- validator: `scripts/validate_mcplusplus_1_0_gap_closure_board.py`

This plan is the durable human design. The objective heap is durable intent.
The taskboard is a drainable projection the `ipfs_accelerate_py` agent
supervisor can execute in parallel. Completing a board row is not by itself
goal completion. Goal completion requires the stated evidence and validation
command on the current tree.

## 1. Outcome

Turn MCP++ from a documentation-first protocol plus scattered reference
validators into a coherent, installable, current-spec, cryptographically
enforceable, state-aware, durable, and independently verifiable agent-mesh
framework.

Agents in the resulting release-candidate architecture must be able to:

- discover each other and compare capabilities
- exchange portable task and execution envelopes
- delegate attenuated authority
- evaluate permissions, prohibitions, temporal constraints, and obligations
- refer to immutable inputs, outputs, policies, proofs, code, schemas, state
  roots, and receipts by CID
- share state under an explicitly declared consistency model
- resume after crashes without duplicating completed side effects
- coordinate exclusive work with claims, leases, epochs, and fencing
- communicate over HTTP and optional P2P transports
- produce signed, independently verifiable execution receipts
- reconstruct causal history through a bounded Event DAG
- interoperate with current MCP and A2A implementations
- demonstrate these properties in a repeatable three-peer environment

Do not claim cryptographic, zero-knowledge, consensus, durability, or
production conformance unless the corresponding mechanism is implemented and
tested. Fail closed for authorization, policy, signature, proof, lease,
fencing, payment, and trust-boundary failures.

## 2. Repository ownership

Treat **Mcp-Plus-Plus** as the canonical home for:

- normative protocol specifications
- versioned schemas
- cross-language conformance vectors
- validators
- compatibility matrices
- interoperability tests
- release bundles

Put runtime-specific implementation in the repository that owns that runtime.
Do not duplicate large implementations inside the specification repository.

| Role | Checkout observed 2026-08-15 | Canonical remote |
| --- | --- | --- |
| Protocol / conformance | `lift_coding/Mcp-Plus-Plus` at `dc3164653a48d059ae9812078359daeafb451c07` (main, behind origin by 1, clean) | https://github.com/endomorphosis/Mcp-Plus-Plus |
| Accelerate nested spec submodule | `ipfs_accelerate_py/mcplusplus` at `15c1816d6c63a2b11edd505704f6a04a9abc6167` | same remote; **older than the lift_coding checkout** |
| Runtime: accelerate | `external/ipfs_accelerate` at `ea11293bb996f052d620eae989f5377a956764b1` on `fix/implementation-auto-rescue-20260809`, **dirty (~140 paths)** | https://github.com/endomorphosis/ipfs_accelerate_py |
| Runtime: datasets | `external/ipfs_datasets` detached `a2f5400b7cb89c8481819379a1b7b9959fe81d45`, dirty overlay | https://github.com/endomorphosis/ipfs_datasets_py |
| Runtime: kit | `external/ipfs_kit` at `5a7a2df8181cfdc33bc19be09989df7ff83f2d4e` on `main` (behind origin 45, clean) | https://github.com/endomorphosis/ipfs_kit_py |
| Runtime: SwissKnife | `lift_coding/swissknife` at `26f06277888b09a3e7c9b4a3b844001f1dbc0841` on `main` (behind origin 8, 2 dirty paths) | https://github.com/endomorphosis/swissknife |
| Superproject | `lift_coding` at `b6f40c05e0884867eb8557f8882cd25cb760ca2f` on `chore/fmt-check-main` (ahead 742), dirty gitlinks | https://github.com/endomorphosis/lift_coding.git |

SwissKnife’s remotes are `origin https://github.com/endomorphosis/swissknife`
and `upstream https://github.com/dnakov/anon-kode.git`. Do not invent a
repository URL. Discover the checkout with `git remote -v`.

Supervisor worktrees launched from `ipfs_accelerate_py` can isolate these
gitlinks:

- `ipfs_accelerate_py/mcplusplus`
- `ipfs_datasets_py`
- `ipfs_kit_py`

SwissKnife is a sibling in the lift_coding forest, not an accelerate
submodule. Tasks that mutate SwissKnife must bind the discovered checkout in
MCPP-001 evidence and use an extra worktree; they must not rewrite the
operator’s dirty `main` checkout.

## 3. Frozen planning baseline

The board was authored against the revisions in §2. A changed revision is not
automatically a failure, but it invalidates historical findings and requires
a fresh inventory.

### 3.1 Existing user work that must be preserved

Do not reset, discard, rewrite history, or force-push.

| Checkout | Preserve |
| --- | --- |
| accelerate | Uncommitted supervisor/MCP/runtime edits and many untracked DCR/SCA modules |
| datasets | Uncommitted logic/UI-IR and MCP++ P2P files |
| SwissKnife | Dirty `test-results/.../svd-132.json` and untracked `test/mcp-plus-plus/dcr090-hermetic-fixtures.test.ts` |
| superproject | Dirty gitlinks and untracked backup/worktree directories |

MCPP-001 records a dirty-overlay inventory and creates
`codex/mcplusplus-1.0-gap-closure` from the current HEAD of each repo that
needs modification, without touching uncommitted files in the operator
checkout. Implementation lanes work in isolated worktrees.

### 3.2 Observed MCP++ shape (not yet a conformance claim)

Mcp-Plus-Plus is documentation-first. Profiles A–H exist as draft specs.
Validators exist in Python, TypeScript, Go, and Rust. Almost all Profile C
and D validation is **structural field presence**, not cryptographic or
policy enforcement.

Confirmed facts from the current trees:

- Profiles A–H are registered in `docs/spec/mcp++-profiles-draft.md`.
- Capability negotiation is still specified as MCP `initialize` /
  `initialized` with `protocolVersion` `2024-11-05`.
- Conformance vector `initialize_result.json` pins `2024-11-05`.
- Versioned JSON Schema exists only for Profile H (`schemas/profile-h/1.0/`).
- Other profiles are validated by hand-written per-language models.
- Python UCAN validator checks that `iss`, `aud`, `att`, and `exp` exist. It
  does not verify signatures, attenuation, audience continuity, or revocation.
- Coverage documents claim “100 percent” and “validation complete”. Those
  claims are not accepted as evidence. Tests must be recomputed.
- Official MCP **2026-07-28** is the current specification
  (https://modelcontextprotocol.io/specification/2026-07-28). It is a
  stateless-core revision: the legacy initialize exchange is not current
  protocol behavior.
- Official A2A extensions are identified by **URI**, advertised on the Agent
  Card, and requested via `A2A-Extensions`. The assignment’s
  `io.mcplusplus.execution@1` is a working name. MCPP-010 must verify the
  current A2A extension identifier convention and record the stable URI
  (default candidate: `https://mcplusplus.io/extensions/execution/v1` with
  alias `io.mcplusplus.execution@1` if the official convention still allows
  reverse-DNS tokens).

Runtime code already exists and must be adapted, not rewritten:

- `ipfs_accelerate_py/mcp_server/mcplusplus/` and leftover
  `mcplusplus_module` shims
- `ipfs_datasets_py/mcp_server/mcplusplus/` plus P2P/workflow/task-queue
  engines
- `ipfs_kit_py` MCP++ UCAN / Profile D admission work (KITA-G080)
- SwissKnife Profile A–H adapters, UCAN manager, Helia artifact store, and
  `swissknife mcp-plus-plus` CLI

## 4. Operating rules

1. Begin every implementation lane from the bound worktree. Do not “fix” the
   operator’s dirty checkout.
2. Preserve all existing user work. Never discard uncommitted changes, reset
   branches, rewrite history, or force-push.
3. Create `codex/mcplusplus-1.0-gap-closure` in every repository that needs
   modification.
4. Use small, intentional commits organized by workstream.
5. Verify official MCP and A2A specs from primary sources before changing
   bindings.
6. Preserve backward compatibility through versioned adapters. Do not
   silently change the bytes or CIDs of existing artifact formats.
7. Where canonicalization changes are necessary, introduce an explicit
   canonicalization version and migration path.
8. Do not claim cryptographic, ZK, consensus, durability, or production
   conformance unless implemented and tested.
9. Fail closed for authorization, policy, signature, proof, lease, fencing,
   payment, and trust-boundary failures.
10. Do not substitute mocks for security-critical validation.
11. Prefer maintained, permissively licensed dependencies over inventing
    cryptographic, CRDT, workflow, or consensus primitives.
12. Make the most defensible engineering default, record it in an ADR, and
    proceed.
13. Avoid a wholesale rewrite. Preserve working Profiles A–H incrementally.
14. Recompute tests and coverage. Do not trust static documentation counts.
15. If an external service is unavailable, implement the stable interface,
    complete locally testable work, document the exact blocker, and continue.
16. Do not finish with only documentation, TODOs, skeletons, or unimplemented
    interfaces.

## 5. Key decisions

These defaults are binding unless a later ADR supersedes them with evidence.
MCPP-G020 writes the ADRs; later tasks implement them.

| ID | Decision | Rationale |
| --- | --- | --- |
| KD-1 | Spec repo owns schemas, vectors, validators, matrices. Runtimes own adapters. | Prevents a second unofficial protocol and keeps Mcp-Plus-Plus installable as a conformance package. |
| KD-2 | Profiles A–H are transport- and MCP-version-independent. MCP bindings are versioned documents/modules. | Current MCP 2026-07-28 removed the legacy initialize exchange; the draft still treats it as normative. |
| KD-3 | Dual bindings: `mcp-binding/legacy-2024-11-05` and `mcp-binding/2026-07-28`. A peer may support both. Downgrade or version forgery is rejected. | Installed implementations still speak 2024-11-05. Current clients must not be forced through initialize. |
| KD-4 | Mandatory crypto suite: Ed25519, explicit key ids, DID-compatible issuers/audiences, signatures over canonical bytes, CIDv1 (`raw` + `sha2-256` unless an existing artifact already uses another declared multicodec). | Matches existing CID-native direction and is widely implemented in all four languages. |
| KD-5 | Canonicalization identifier `mcpp-jcs-v1` = RFC 8785 JCS. Existing artifacts remain readable under their recorded historical algorithm. | Do not silently change CIDs. Version the algorithm. |
| KD-6 | Conformance levels are distinct: `structural`, `canonical`, `cryptographic`, `policy-enforced`, `receipt-signed`, `proof-verified`. Schema acceptance is never “implemented”. | Current validators are structural-only for C/D. |
| KD-7 | One canonical carrier: `ExecutionEnvelope@1`, `ExecutionResult@1`, `ExecutionReceipt@1`, `PortableError@1`. Profile B and G objects adapt without CID breakage. | Profiles currently overlap execution semantics. |
| KD-8 | `StateRef@1` must declare exactly one of `immutable`, `single_authority`, `causal`, `crdt`, `consensus`. | Observing two Event DAG branches must not silently merge mutable values. |
| KD-9 | Single-authority backend: SQLite with WAL and compare-and-swap, because it is already in-tree and restart-testable. DuckDB is an optional second adapter, not the mandatory one. | Repository-native, transactional, no new service. |
| KD-10 | CRDT backend: Automerge (permissive license, real CRDT, Python/JS bindings). Do not invent last-write-wins and call it a CRDT. | Assignment forbids informal LWW. |
| KD-11 | Consensus plugin documents four distinct guarantees: coordination, majority approval, crash consensus, BFT. Profile G neighborhood agreement is **coordination / majority approval**, not BFT. | Existing Profile G language already says neighborhood agreement is not global consensus. |
| KD-12 | Durable executor: define `DurableExecutor` in the spec repo. First production-capable adapter is a **SQLite journaled executor** that satisfies crash-recovery locally. Evaluate Restate and Dapr in the ADR; adopt one as a second adapter only if a repeatable local compose environment works without unpaid cloud. | Fail closed if the external engine cannot be tested locally. |
| KD-13 | A2A extension URI is verified in MCPP-010. Do not create a competing public task lifecycle. | A2A already provides Agent Card, Task, Message, Artifact, status, cancel, streaming. |
| KD-14 | Transport identity ≠ execution authority. PeerID / TLS client cert never grants UCAN capabilities. Payment never grants authorization. | Fail-closed trust boundary. |
| KD-15 | Encrypted artifact references carry ciphertext CID, algorithm, key-envelope, recipients/capability, plaintext schema CID, optional protected digest, disclosure and retention policy. Plaintext must not appear in logs, Event DAG metadata, or local fallback caches. | Content addressing is not publication. |
| KD-16 | Installable CLI command is `mcpp`. Three-peer demo is Docker Compose. A separate verifier process validates the evidence bundle. | Assignment-required demonstration path. |
| KD-17 | Profile bundles: Evidence Core (A,B,F), Secure Delegation (C,D), Federated Mesh (E,G), Commerce (H), Verified Execution (signed receipts / attestations / verified proofs only). | Matches assignment packaging. |

## 6. Goal tree

```text
MCPP-G000  MCP++ 1.0 release-candidate architecture
|-- MCPP-G010  Baseline, inventory, and traceability
|-- MCPP-G020  Architecture decision records
|-- MCPP-G030  Abstract profiles vs MCP version bindings
|-- MCPP-G040  Canonicalization and cross-language schemas
|-- MCPP-G050  Canonical portable execution envelope
|-- MCPP-G060  Interoperable shared-state semantics
|-- MCPP-G070  Cryptographic delegation and receipts
|-- MCPP-G080  Temporal deontic policy execution
|-- MCPP-G090  Durable execution adapter
|-- MCPP-G100  A2A interoperability extension
|-- MCPP-G110  Discovery, advertisement, and routing
|-- MCPP-G120  P2P transport hardening
|-- MCPP-G130  Profile G coordination specification
|-- MCPP-G140  Profile H payments remain non-authorizing
|-- MCPP-G150  Confidential CID-native artifacts
|-- MCPP-G160  Installable CLI and three-peer demonstration
`-- MCPP-G170  Documentation, CI, and release candidate
```

Root `MCPP-G000` is review-only. It aggregates child evidence and must not
implement subsystem behavior.

## 7. Parallelism and lane assignment

`max_lanes` is 6. `strict_task_sharding` assigns `MCPP-NNN` to
`NNN % 6`. Tasks in the same wave that must run together are numbered onto
different remainders and list each other in `Allow concurrent with`.

| Lane | Remainder | Name | Owns |
| --- | --- | --- | --- |
| 0 | 0 | mcpp-lane-spec | Spec text, bindings, Profile G/H normative docs, CI, report |
| 1 | 1 | mcpp-lane-schema | Schemas, canonicalization, envelope, StateRef, confidential refs |
| 2 | 2 | mcpp-lane-crypto | Crypto, policy, payments, revocation, receipts |
| 3 | 3 | mcpp-lane-state | State providers, durable executor, Profile G runtime |
| 4 | 4 | mcpp-lane-transport | A2A adapter, discovery, P2P framing |
| 5 | 5 | mcpp-lane-runtime | Accelerate / datasets / kit / SwissKnife adapters, CLI, demo |

File-disjointness rules:

- Spec, schema, and vector files live only under `ipfs_accelerate_py/mcplusplus/`.
- Accelerate runtime adapters live only under `ipfs_accelerate_py/mcp_server/`
  (and tests under `ipfs_accelerate_py/mcp/tests/` or `test/api/`).
- Datasets runtime adapters live only under `ipfs_datasets_py/`.
- Kit runtime adapters live only under `ipfs_kit_py/`.
- SwissKnife adapters live only in the bound SwissKnife checkout.
- Evidence bundles live under
  `docs/reports/mcplusplus-1.0-gap-closure/` in accelerate.
- Protected control artifacts are never rewritten after MCPP-000.

## 8. Dependency waves

Use this order unless repository evidence requires a documented change.
After each workstream: run focused tests, run affected cross-language
vectors, commit, update the traceability matrix, record unresolved risks.

```text
Wave 0  MCPP-000          seal (completed by this plan)
Wave 1  MCPP-001          preserve dirty work; create branches
Wave 2  MCPP-002..012     baseline per language/repo + official-spec verify
                          + inventory + traceability scaffold   [parallel]
Wave 3  MCPP-013..018     ADRs                                  [parallel]
Wave 4  MCPP-019..029     MCP bindings + canonicalization       [parallel after ADRs]
Wave 5  MCPP-030..045     envelope, state, crypto               [parallel]
Wave 6  MCPP-046..065     policy, durable, A2A, discovery, P2P  [parallel where file-disjoint]
Wave 7  MCPP-066..074     Profile G, Profile H, confidential
Wave 8  MCPP-075..077     CLI + three-peer demo
Wave 9  MCPP-078..083     docs, CI, report, draft PRs, terminal receipt
```

A later wave may start a file-disjoint task as soon as its `Depends on` set
is terminal. The graph is the authority, not this prose summary.

## 9. Workstream mapping

| Assignment workstream | Goal | First tasks | Runtime owners |
| --- | --- | --- | --- |
| Phase 0 baseline | G010 | 001–012 | all |
| ADRs | G020 | 013–018 | accelerate docs |
| WS1 MCP bindings | G030 | 019–023 | spec + accelerate + datasets |
| WS13 canonicalization | G040 | 024–029 | spec four languages |
| WS3 envelope | G050 | 030–034 | spec + accelerate |
| WS4 state | G060 | 035–040 | spec + accelerate + datasets |
| WS5 crypto | G070 | 041–045 | spec + SwissKnife + kit |
| WS6 policy | G080 | 046–049 | spec + datasets |
| WS7 durable | G090 | 050–053 | spec + accelerate |
| WS2 A2A | G100 | 054–057 | spec + SwissKnife |
| WS8 discovery | G110 | 058–061 | spec + accelerate |
| WS9 P2P | G120 | 062–065 | spec + datasets + kit |
| WS11 Profile G | G130 | 066–069 | spec + kit + accelerate |
| WS12 Profile H | G140 | 070–072 | spec + SwissKnife + accelerate |
| WS10 confidential | G150 | 073–074 | spec + kit |
| WS14 CLI/demo | G160 | 075–077 | spec CLI + compose |
| WS15–16 docs/CI/RC | G170 | 078–083 | all |

## 10. Acceptance gates

Do not call the program complete until each satisfied gate has a command and
an evidence artifact under
`docs/reports/mcplusplus-1.0-gap-closure/`.

1. Existing tests have been baselined and no user changes were lost.
2. Abstract profile semantics are separated from MCP-version bindings.
3. The current MCP binding follows the verified 2026-07-28 specification.
4. The current MCP binding does not depend on the removed legacy
   initialization exchange.
5. The legacy MCP binding still passes its compatibility tests.
6. An A2A extension and end-to-end handoff test exist.
7. `ExecutionEnvelope@1`, `ExecutionResult@1`, `ExecutionReceipt@1`, and
   `PortableError@1` exist.
8. `StateRef@1` exists with explicit consistency modes.
9. Existing Profile B and G artifacts can be adapted without silent CID
   breakage.
10. One real durable single-authority state backend passes restart tests.
11. One real CRDT backend passes concurrent-update and convergence tests.
12. Consensus and neighborhood coordination are labeled accurately.
13. Real cryptographic delegation verification exists.
14. Capability attenuation, expiration, revocation, audience, and replay are
    tested negatively.
15. Cross-trust-domain receipts are signed and independently verifiable.
16. Temporal obligations have lifecycle events and deadline tests.
17. At least one durable-execution adapter passes crash recovery without
    duplicate effects.
18. Profile G rejects stale fenced completion.
19. Three peers converge after a partition heals.
20. There is exactly one authoritative completion for an exclusive task.
21. P2P abuse and framing tests pass.
22. Confidential artifacts do not leak plaintext through tested persistence
    paths.
23. Profile H never treats payment as authorization.
24. Canonical bytes, CIDs, and signatures match across Python, TypeScript,
    Go, and Rust.
25. A one-command three-peer demonstration completes successfully.
26. A separate verifier process validates the resulting evidence bundle.
27. Static documentation claims match generated CI evidence.
28. Required CI workflows are green.

## 11. What this plan does **not** claim

- The current trees are not production-ready.
- Existing “100 percent coverage” documents are not evidence.
- Profile G is not Byzantine-fault-tolerant consensus.
- Profile F Groth16 / ZK material is not Verified Execution until a real
  verifier succeeds on current vectors.
- Payment, PeerID, or registry presence is not authorization.
- This sealed plan is not implementation. MCPP-000 only admits the program.

## 12. Final deliverables

Implementation lanes produce:

- `ipfs_accelerate_py/mcplusplus/docs/roadmap/mcplusplus-1.0-gap-closure.md`
  (requirement-to-evidence matrix; status is never “implemented” from schema
  acceptance alone)
- `ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md`
- a draft pull-request description per modified repository
- draft PRs only if GitHub authentication and permissions exist; do not merge

The implementation report must include: executive summary; repositories and
commit SHAs; architecture decisions; files changed by workstream; protocol
compatibility matrix; conformance levels achieved; exact test commands and
results; coverage; security tests; performance; three-peer demo evidence;
crash-recovery evidence; cryptographic evidence; remaining limitations;
items explicitly not claimed; migration instructions; and a NO-GO / RC / GO
recommendation with evidence.

## 13. How to launch the supervisor

From the `ipfs_accelerate_py` checkout, after MCPP-000 validation is green:

```bash
python scripts/validate_mcplusplus_1_0_gap_closure_board.py --check-all
python -m pytest -q test/api/test_mcplusplus_1_0_gap_closure_board.py
```

Launch the configured board scheduler with the sealed JSON. Do not enable
objective refill or codebase refill. Protected paths listed in the scheduler
are operator-owned after MCPP-000.

Provider contract is the ordered supervisor pair: Grok Build `grok-4.5`
primary, Codex `gpt-5.6-terra` only after independently signed quota
authority. Secrets come from the environment only.

## 14. PR Plan

Each PR is independently reviewable. Later PRs adapt earlier artifacts; they
do not rewrite them in place.

| PR | Title | Depends on | Repos |
| --- | --- | --- | --- |
| PR-00 | Seal supervisor program (plan, heap, board, scheduler, validator) | — | accelerate |
| PR-01 | Preserve overlays; create `codex/mcplusplus-1.0-gap-closure` | PR-00 | all in-scope |
| PR-02 | Baseline matrix and official MCP/A2A verification | PR-01 | all in-scope |
| PR-03 | Traceability matrix scaffold and Profile A–H inventory | PR-02 | Mcp-Plus-Plus |
| PR-04 | Architecture decision records | PR-03 | accelerate + Mcp-Plus-Plus |
| PR-05 | Abstract profiles; legacy and 2026-07-28 MCP bindings | PR-04 | Mcp-Plus-Plus, accelerate, datasets |
| PR-06 | `mcpp-jcs-v1` and four-language golden vectors | PR-04 | Mcp-Plus-Plus |
| PR-07 | ExecutionEnvelope@1 family + B/G adapters | PR-06 | Mcp-Plus-Plus, accelerate |
| PR-08 | StateRef@1 + four providers | PR-06 | Mcp-Plus-Plus, accelerate, datasets |
| PR-09 | Cryptographic delegation, revocation, signed receipts | PR-06 | Mcp-Plus-Plus, kit, SwissKnife |
| PR-10 | Policy evaluator and obligation lifecycle | PR-09 | Mcp-Plus-Plus, datasets |
| PR-11 | DurableExecutor + crash-recovery adapter | PR-07, PR-08 | Mcp-Plus-Plus, accelerate |
| PR-12 | A2A extension, adapter, handoff tests | PR-04, PR-07 | Mcp-Plus-Plus, SwissKnife |
| PR-13 | Advertisement schema and registries | PR-12 | Mcp-Plus-Plus, accelerate |
| PR-14 | Profile E framing and abuse tests | PR-05 | Mcp-Plus-Plus, datasets, kit |
| PR-15 | Normative Profile G + three-peer fencing | PR-07, PR-08 | Mcp-Plus-Plus, kit, accelerate |
| PR-16 | Profile H payment≠authorization | PR-09 | Mcp-Plus-Plus, SwissKnife, accelerate |
| PR-17 | Encrypted artifact references | PR-07 | Mcp-Plus-Plus, kit |
| PR-18 | `mcpp` CLI and three-peer demo | PR-11–PR-17 | Mcp-Plus-Plus |
| PR-19 | Docs, CI, implementation report, draft PRs | PR-18 | all |

## 15. Open questions

None that block planning. Official A2A extension URI spelling is verified in
MCPP-010; the default in KD-13 is used until that task records the verified
identifier. Restate versus Dapr versus the SQLite journaled executor is
decided in MCPP-017 with the default in KD-12.

## 16. Most important remaining technical risk

The accelerate checkout is dirty and its nested `ipfs_accelerate_py/mcplusplus`
submodule is behind the lift_coding Mcp-Plus-Plus checkout. If lanes treat
the older submodule as authority, they will rebase current Profile H and
cross-language work away. MCPP-001 must bind the newer spec tree into the
program worktrees without discarding either dirty overlay.
